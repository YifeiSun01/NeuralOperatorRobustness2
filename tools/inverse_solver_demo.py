#!/usr/bin/env python3
"""Run small inverse-problem demos through PyTorch and Exponax/JAX solvers.

The optimization variable is the initial condition. A fixed, intentionally
unphysical-looking reference final condition is chosen, then each solver is
differentiated to update the initial condition so the solver final state moves
toward that reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def global_gpu_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [float(line.strip()) for line in out.splitlines() if line.strip()]
        return values[0] if values else float("nan")
    except Exception:
        return float("nan")


def sync_torch(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def smooth_burgers_initial(nx: int, seed: int, domain_extent: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, domain_extent, nx, endpoint=False, dtype=np.float32)
    u = np.zeros_like(x)
    for k in range(1, 6):
        amp = rng.normal(0.0, 0.35 / k)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u += amp * np.sin(2.0 * np.pi * k * x / domain_extent + phase)
    return u.astype(np.float32)


def burgers_reference_final(nx: int, domain_extent: float = 2.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    saw = 2.0 * (4.0 * x - np.floor(4.0 * x + 0.5))
    target = 0.55 * saw
    target -= target.mean()
    return target.astype(np.float32)


def smooth_ns_initial(nx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    field = np.zeros((nx, nx), dtype=np.float32)
    for kx in range(1, 5):
        for ky in range(1, 5):
            amp = rng.normal(0.0, 0.18 / (kx + ky))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            field += amp * np.sin(2.0 * np.pi * (kx * x + ky * y) + phase)
            field += 0.5 * amp * np.cos(2.0 * np.pi * (kx * x - ky * y) + phase)
    return field.astype(np.float32)


def ns_reference_final(nx: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    width = 0.035
    soft_edge = 0.008
    positive_slash = 0.5 * (1.0 + np.tanh((width - np.abs(y - x)) / soft_edge))
    negative_slash = 0.5 * (1.0 + np.tanh((width - np.abs(y - (1.0 - x))) / soft_edge))
    target = positive_slash - negative_slash
    return target.astype(np.float32)


@dataclass
class RunResult:
    metrics_rows: list[dict[str, Any]]
    initial_history: list[np.ndarray]
    final_history: list[np.ndarray]
    target: np.ndarray
    summary: dict[str, Any]


def run_torch_burgers(args: argparse.Namespace, target_np: np.ndarray, initial_np: np.ndarray) -> RunResult:
    import torch
    from solvers import solve_burgers_final_batch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    target = torch.from_numpy(target_np[None, :]).to(device)
    u = torch.nn.Parameter(torch.from_numpy(initial_np[None, :]).to(device))
    opt = torch.optim.Adam([u], lr=args.burgers_lr)

    rows: list[dict[str, Any]] = []
    initial_history: list[np.ndarray] = []
    final_history: list[np.ndarray] = []
    total_start = time.perf_counter()
    global_start = global_gpu_used_mib()

    for step in range(args.opt_steps + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mem_before = global_gpu_used_mib()
        forward_start = time.perf_counter()
        pred = solve_burgers_final_batch(
            u,
            t_final=args.burgers_t_final,
            dt=args.burgers_dt,
            domain_extent=args.burgers_domain,
            diffusivity=args.burgers_nu,
            device=device,
            dtype=torch.float32,
        )
        loss = torch.mean((pred - target) ** 2)
        sync_torch(torch)
        forward_seconds = time.perf_counter() - forward_start
        forward_peak = (
            float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else float("nan")
        )

        initial_history.append(u.detach().cpu().numpy()[0].copy())
        final_history.append(pred.detach().cpu().numpy()[0].copy())

        backward_seconds = 0.0
        backward_peak = float("nan")
        update_seconds = 0.0
        grad_norm = float("nan")
        if step < args.opt_steps:
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            backward_start = time.perf_counter()
            loss.backward()
            sync_torch(torch)
            backward_seconds = time.perf_counter() - backward_start
            backward_peak = (
                float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else float("nan")
            )
            grad_norm = float(torch.linalg.vector_norm(u.grad.detach()).item())
            update_start = time.perf_counter()
            opt.step()
            sync_torch(torch)
            update_seconds = time.perf_counter() - update_start

        mem_after = global_gpu_used_mib()
        rows.append(
            {
                "case": "burgers_1d",
                "framework": "pytorch",
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "update_seconds": update_seconds,
                "grad_norm": grad_norm,
                "torch_forward_peak_allocated_mib": forward_peak,
                "torch_backward_peak_allocated_mib": backward_peak,
                "global_gpu_before_mib": mem_before,
                "global_gpu_after_mib": mem_after,
                "global_gpu_delta_mib": mem_after - mem_before if math.isfinite(mem_before) and math.isfinite(mem_after) else float("nan"),
            }
        )

    summary = summarize_rows(rows, total_start, global_start)
    return RunResult(rows, initial_history, final_history, target_np, summary)


def run_torch_ns(args: argparse.Namespace, target_np: np.ndarray, initial_np: np.ndarray) -> RunResult:
    import torch
    from solvers import solve_ns_zongyi_rollout_batch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    target = torch.from_numpy(target_np[None, :, :]).to(device)
    u = torch.nn.Parameter(torch.from_numpy(initial_np[None, :, :]).to(device))
    opt = torch.optim.Adam([u], lr=args.ns_lr)

    rows: list[dict[str, Any]] = []
    initial_history: list[np.ndarray] = []
    final_history: list[np.ndarray] = []
    total_start = time.perf_counter()
    global_start = global_gpu_used_mib()

    for step in range(args.opt_steps + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mem_before = global_gpu_used_mib()
        forward_start = time.perf_counter()
        seq = solve_ns_zongyi_rollout_batch(
            u,
            t_final=args.ns_t_final,
            fixed_step=args.ns_dt,
            domain_extent=args.ns_domain,
            diffusivity=args.ns_nu,
            return_time_last=True,
            device=device,
            dtype=torch.float32,
        )
        pred = seq[..., -1]
        loss = torch.mean((pred - target) ** 2)
        sync_torch(torch)
        forward_seconds = time.perf_counter() - forward_start
        forward_peak = (
            float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else float("nan")
        )

        initial_history.append(u.detach().cpu().numpy()[0].copy())
        final_history.append(pred.detach().cpu().numpy()[0].copy())

        backward_seconds = 0.0
        backward_peak = float("nan")
        update_seconds = 0.0
        grad_norm = float("nan")
        if step < args.opt_steps:
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            backward_start = time.perf_counter()
            loss.backward()
            sync_torch(torch)
            backward_seconds = time.perf_counter() - backward_start
            backward_peak = (
                float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else float("nan")
            )
            grad_norm = float(torch.linalg.vector_norm(u.grad.detach()).item())
            update_start = time.perf_counter()
            opt.step()
            sync_torch(torch)
            update_seconds = time.perf_counter() - update_start

        mem_after = global_gpu_used_mib()
        rows.append(
            {
                "case": "ns_2d",
                "framework": "pytorch",
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "update_seconds": update_seconds,
                "grad_norm": grad_norm,
                "torch_forward_peak_allocated_mib": forward_peak,
                "torch_backward_peak_allocated_mib": backward_peak,
                "global_gpu_before_mib": mem_before,
                "global_gpu_after_mib": mem_after,
                "global_gpu_delta_mib": mem_after - mem_before if math.isfinite(mem_before) and math.isfinite(mem_after) else float("nan"),
            }
        )

    summary = summarize_rows(rows, total_start, global_start)
    return RunResult(rows, initial_history, final_history, target_np, summary)


def make_jax_burgers_forward(args: argparse.Namespace):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import exponax as ex

    steps = int(round(args.burgers_t_final / args.burgers_dt))
    stepper = ex.stepper.Burgers(
        1,
        args.burgers_domain,
        args.burgers_nx,
        args.burgers_dt,
        diffusivity=args.burgers_nu,
        convection_scale=1.0,
        order=4,
        conservative=False,
        dealiasing_fraction=2 / 3,
        num_circle_points=16,
    )
    repeat_fn = ex.repeat(stepper, steps)

    def forward(u0):
        return repeat_fn(jnp.expand_dims(u0, axis=0))[0]

    return jax.jit(forward)


def make_jax_ns_forward(args: argparse.Namespace):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import exponax as ex

    steps_per_second = int(round(1.0 / args.ns_dt))
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2,
        args.ns_domain,
        args.ns_nx,
        args.ns_dt,
        diffusivity=args.ns_nu,
        order=4,
        num_circle_points=16,
        dealiasing_fraction=2 / 3,
    )

    def forward(u0):
        u = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))

        def micro(carry, _):
            return stepper(carry[None, ...])[0], None

        def one_second(carry, _):
            u_next, _ = jax.lax.scan(micro, carry, None, length=steps_per_second)
            return u_next, u_next

        _, seconds = jax.lax.scan(one_second, u, None, length=args.ns_t_final)
        final = seconds[-1]
        return jnp.swapaxes(final, -1, -2)

    return jax.jit(forward)


def run_jax_case(
    args: argparse.Namespace,
    case: str,
    target_np: np.ndarray,
    initial_np: np.ndarray,
) -> RunResult:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    if case == "burgers_1d":
        forward = make_jax_burgers_forward(args)
        lr = args.burgers_lr
    elif case == "ns_2d":
        forward = make_jax_ns_forward(args)
        lr = args.ns_lr
    else:
        raise ValueError(case)

    target = jnp.asarray(target_np, dtype=jnp.float32)

    @jax.jit
    def loss_only(u0):
        pred = forward(u0)
        return jnp.mean((pred - target) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_only))

    @jax.jit
    def adam_update(u0, grad, m, v, step_index):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        t = step_index + 1
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        u0 = u0 - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return u0, m, v

    u = jnp.asarray(initial_np, dtype=jnp.float32)
    m = jnp.zeros_like(u)
    v = jnp.zeros_like(u)
    rows: list[dict[str, Any]] = []
    initial_history: list[np.ndarray] = []
    final_history: list[np.ndarray] = []
    total_start = time.perf_counter()
    global_start = global_gpu_used_mib()

    for step in range(args.opt_steps + 1):
        mem_before = global_gpu_used_mib()

        forward_start = time.perf_counter()
        pred = forward(u)
        pred.block_until_ready()
        forward_seconds = time.perf_counter() - forward_start
        loss_forward = float(jnp.mean((pred - target) ** 2))

        initial_history.append(np.asarray(u).copy())
        final_history.append(np.asarray(pred).copy())

        value_grad_seconds = 0.0
        backward_estimate_seconds = 0.0
        update_seconds = 0.0
        grad_norm = float("nan")
        if step < args.opt_steps:
            value_grad_start = time.perf_counter()
            loss_vg, grad = value_and_grad(u)
            grad.block_until_ready()
            value_grad_seconds = time.perf_counter() - value_grad_start
            backward_estimate_seconds = max(0.0, value_grad_seconds - forward_seconds)
            grad_norm = float(jnp.linalg.norm(grad))

            update_start = time.perf_counter()
            u, m, v = adam_update(u, grad, m, v, step)
            u.block_until_ready()
            update_seconds = time.perf_counter() - update_start
            loss_value = float(loss_vg)
        else:
            loss_value = loss_forward

        mem_after = global_gpu_used_mib()
        rows.append(
            {
                "case": case,
                "framework": "jax_exponax",
                "step": step,
                "loss": loss_value,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_estimate_seconds,
                "jax_value_and_grad_seconds": value_grad_seconds,
                "update_seconds": update_seconds,
                "grad_norm": grad_norm,
                "torch_forward_peak_allocated_mib": float("nan"),
                "torch_backward_peak_allocated_mib": float("nan"),
                "global_gpu_before_mib": mem_before,
                "global_gpu_after_mib": mem_after,
                "global_gpu_delta_mib": mem_after - mem_before if math.isfinite(mem_before) and math.isfinite(mem_after) else float("nan"),
            }
        )

    summary = summarize_rows(rows, total_start, global_start)
    return RunResult(rows, initial_history, final_history, target_np, summary)


def summarize_values(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_rows(rows: list[dict[str, Any]], total_start: float, global_start: float) -> dict[str, Any]:
    final_loss = rows[-1]["loss"]
    initial_loss = rows[0]["loss"]
    global_end = global_gpu_used_mib()
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_factor": float(initial_loss / final_loss) if final_loss > 0 else float("inf"),
        "loss_reduction_percent": float(100.0 * (initial_loss - final_loss) / initial_loss) if initial_loss > 0 else float("nan"),
        "total_seconds": time.perf_counter() - total_start,
        "global_gpu_start_mib": global_start,
        "global_gpu_end_mib": global_end,
        "global_gpu_delta_total_mib": global_end - global_start if math.isfinite(global_start) and math.isfinite(global_end) else float("nan"),
        "forward_seconds": summarize_values([row["forward_seconds"] for row in rows[:-1]]),
        "backward_seconds": summarize_values([row["backward_seconds"] for row in rows[:-1]]),
        "torch_forward_peak_allocated_mib": summarize_values(
            [row["torch_forward_peak_allocated_mib"] for row in rows[:-1]]
        ),
        "torch_backward_peak_allocated_mib": summarize_values(
            [row["torch_backward_peak_allocated_mib"] for row in rows[:-1]]
        ),
        "global_gpu_delta_mib": summarize_values([row["global_gpu_delta_mib"] for row in rows[:-1]]),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_histories(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        initial_history=np.asarray(result.initial_history, dtype=np.float32),
        final_history=np.asarray(result.final_history, dtype=np.float32),
        target=np.asarray(result.target, dtype=np.float32),
    )


def make_gif_from_figures(frame_paths: list[Path], gif_path: Path, duration_ms: int = 250) -> None:
    from PIL import Image

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    for image in images:
        image.close()


def remove_intermediate_frames(frame_paths: list[Path]) -> None:
    if len(frame_paths) <= 2:
        return
    keep = {frame_paths[0].resolve(), frame_paths[-1].resolve()}
    for path in frame_paths:
        if path.resolve() in keep:
            continue
        path.unlink(missing_ok=True)


def plot_burgers_frames(
    out_dir: Path,
    torch_result: RunResult,
    jax_result: RunResult,
    max_frames: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = len(torch_result.initial_history)
    frame_ids = np.unique(np.linspace(0, steps - 1, num=min(max_frames, steps), dtype=int))
    x = np.arange(torch_result.target.shape[-1])
    frame_paths: list[Path] = []

    torch_losses = [row["loss"] for row in torch_result.metrics_rows]
    jax_losses = [row["loss"] for row in jax_result.metrics_rows]

    for frame_id in frame_ids:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
        axes[0].plot(x, torch_result.initial_history[frame_id], label="PyTorch init", linewidth=1.2)
        axes[0].plot(x, jax_result.initial_history[frame_id], label="JAX init", linewidth=1.0, linestyle="--")
        axes[0].set_title(f"1D Burgers inverse optimization, step={frame_id}: current initial")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].plot(x, torch_result.target, label="reference final", color="black", linewidth=1.3)
        axes[1].plot(x, torch_result.final_history[frame_id], label="PyTorch final", linewidth=1.2)
        axes[1].plot(x, jax_result.final_history[frame_id], label="JAX final", linewidth=1.0, linestyle="--")
        axes[1].set_title("Current final vs reference final")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)

        axes[2].plot(x, torch_result.initial_history[frame_id] - jax_result.initial_history[frame_id], label="init diff")
        axes[2].plot(x, torch_result.final_history[frame_id] - jax_result.final_history[frame_id], label="final diff")
        axes[2].set_title("PyTorch - JAX differences")
        axes[2].legend(loc="best")
        axes[2].grid(alpha=0.25)

        axes[3].plot(torch_losses[: frame_id + 1], label="PyTorch loss")
        axes[3].plot(jax_losses[: frame_id + 1], label="JAX loss")
        axes[3].set_yscale("log")
        axes[3].set_title("Loss curve")
        axes[3].legend(loc="best")
        axes[3].grid(alpha=0.25)
        fig.tight_layout()

        frame_path = out_dir / f"burgers_inverse_step_{frame_id:04d}.png"
        fig.savefig(frame_path, dpi=140)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = out_dir / "burgers_inverse_optimization.gif"
    make_gif_from_figures(frame_paths, gif_path)
    remove_intermediate_frames(frame_paths)
    return gif_path


def plot_ns_frames(
    out_dir: Path,
    torch_result: RunResult,
    jax_result: RunResult,
    max_frames: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = len(torch_result.initial_history)
    frame_ids = np.unique(np.linspace(0, steps - 1, num=min(max_frames, steps), dtype=int))
    frame_paths: list[Path] = []
    torch_losses = [row["loss"] for row in torch_result.metrics_rows]
    jax_losses = [row["loss"] for row in jax_result.metrics_rows]

    for frame_id in frame_ids:
        ti = torch_result.initial_history[frame_id]
        ji = jax_result.initial_history[frame_id]
        tf = torch_result.final_history[frame_id]
        jf = jax_result.final_history[frame_id]
        target = torch_result.target
        value_abs = max(
            float(np.max(np.abs(ti))),
            float(np.max(np.abs(ji))),
            float(np.max(np.abs(tf))),
            float(np.max(np.abs(jf))),
            float(np.max(np.abs(target))),
            1e-12,
        )
        diff_abs = max(
            float(np.max(np.abs(ti - ji))),
            float(np.max(np.abs(tf - jf))),
            float(np.max(np.abs(tf - target))),
            float(np.max(np.abs(jf - target))),
            1e-12,
        )

        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        panels = [
            (axes[0, 0], ti, "PyTorch initial", "field"),
            (axes[0, 1], tf, "PyTorch final", "field"),
            (axes[0, 2], tf - target, "PyTorch final - reference", "diff"),
            (axes[1, 0], ji, "JAX initial", "field"),
            (axes[1, 1], jf, "JAX final", "field"),
            (axes[1, 2], jf - target, "JAX final - reference", "diff"),
            (axes[2, 0], ti - ji, "Initial PyTorch - JAX", "diff"),
            (axes[2, 1], tf - jf, "Final PyTorch - JAX", "diff"),
            (axes[2, 2], target, "Reference final", "field"),
        ]
        for ax, arr, title, kind in panels:
            if kind == "field":
                im = ax.imshow(arr, cmap="viridis", origin="lower", vmin=-value_abs, vmax=value_abs)
            else:
                im = ax.imshow(arr, cmap="coolwarm", origin="lower", vmin=-diff_abs, vmax=diff_abs)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        fig.suptitle(
            f"2D NS inverse optimization, step={frame_id}; "
            f"loss torch={torch_losses[frame_id]:.4g}, jax={jax_losses[frame_id]:.4g}"
        )
        frame_path = out_dir / f"ns_inverse_step_{frame_id:04d}.png"
        fig.savefig(frame_path, dpi=130)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = out_dir / "ns_inverse_optimization.gif"
    make_gif_from_figures(frame_paths, gif_path)
    remove_intermediate_frames(frame_paths)
    return gif_path


def plot_loss_png(out_dir: Path, case: str, torch_result: RunResult, jax_result: RunResult) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case}_loss_curve.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([row["loss"] for row in torch_result.metrics_rows], label="PyTorch")
    ax.plot([row["loss"] for row in jax_result.metrics_rows], label="JAX/Exponax")
    ax.set_yscale("log")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"{case} inverse optimization loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "benchmark_results" / "inverse_solver_demo")
    parser.add_argument("--cases", default="burgers,ns")
    parser.add_argument("--opt-steps", type=int, default=30)
    parser.add_argument("--max-gif-frames", type=int, default=31)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--burgers-nx", type=int, default=512)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-nu", type=float, default=1e-3)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-lr", type=float, default=0.08)

    parser.add_argument("--ns-nx", type=int, default=64)
    parser.add_argument("--ns-t-final", type=int, default=2)
    parser.add_argument("--ns-dt", type=float, default=0.005)
    parser.add_argument("--ns-nu", type=float, default=1e-5)
    parser.add_argument("--ns-domain", type=float, default=1.0)
    parser.add_argument("--ns-lr", type=float, default=0.03)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = {case.strip().lower() for case in args.cases.split(",") if case.strip()}

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "settings": {
            "output_root": relpath(output_root),
            "cases": sorted(cases),
            "opt_steps": args.opt_steps,
            "seed": args.seed,
            "burgers": {
                "nx": args.burgers_nx,
                "t_final": args.burgers_t_final,
                "dt": args.burgers_dt,
                "nu": args.burgers_nu,
                "lr": args.burgers_lr,
            },
            "ns": {
                "nx": args.ns_nx,
                "t_final": args.ns_t_final,
                "dt": args.ns_dt,
                "nu": args.ns_nu,
                "lr": args.ns_lr,
            },
        },
        "cases": {},
    }

    if "burgers" in cases or "burgers_1d" in cases:
        print("[inverse] Burgers PyTorch", flush=True)
        b_init = smooth_burgers_initial(args.burgers_nx, args.seed, args.burgers_domain)
        b_target = burgers_reference_final(args.burgers_nx, args.burgers_domain)
        torch_b = run_torch_burgers(args, b_target, b_init)
        print("[inverse] Burgers JAX/Exponax", flush=True)
        jax_b = run_jax_case(args, "burgers_1d", b_target, b_init)
        case_dir = output_root / "burgers_1d"
        save_csv(case_dir / "metrics_pytorch.csv", torch_b.metrics_rows)
        save_csv(case_dir / "metrics_jax_exponax.csv", jax_b.metrics_rows)
        save_histories(case_dir / "histories_pytorch.npz", torch_b)
        save_histories(case_dir / "histories_jax_exponax.npz", jax_b)
        gif = plot_burgers_frames(case_dir / "visualizations", torch_b, jax_b, args.max_gif_frames)
        loss_png = plot_loss_png(case_dir / "visualizations", "burgers_1d", torch_b, jax_b)
        all_rows.extend(torch_b.metrics_rows)
        all_rows.extend(jax_b.metrics_rows)
        summary["cases"]["burgers_1d"] = {
            "pytorch": torch_b.summary,
            "jax_exponax": jax_b.summary,
            "gif": relpath(gif),
            "loss_png": relpath(loss_png),
        }

    if "ns" in cases or "ns_2d" in cases:
        print("[inverse] 2D NS PyTorch", flush=True)
        n_init = smooth_ns_initial(args.ns_nx, args.seed + 1000)
        n_target = ns_reference_final(args.ns_nx)
        torch_n = run_torch_ns(args, n_target, n_init)
        print("[inverse] 2D NS JAX/Exponax", flush=True)
        jax_n = run_jax_case(args, "ns_2d", n_target, n_init)
        case_dir = output_root / "ns_2d"
        save_csv(case_dir / "metrics_pytorch.csv", torch_n.metrics_rows)
        save_csv(case_dir / "metrics_jax_exponax.csv", jax_n.metrics_rows)
        save_histories(case_dir / "histories_pytorch.npz", torch_n)
        save_histories(case_dir / "histories_jax_exponax.npz", jax_n)
        gif = plot_ns_frames(case_dir / "visualizations", torch_n, jax_n, args.max_gif_frames)
        loss_png = plot_loss_png(case_dir / "visualizations", "ns_2d", torch_n, jax_n)
        all_rows.extend(torch_n.metrics_rows)
        all_rows.extend(jax_n.metrics_rows)
        summary["cases"]["ns_2d"] = {
            "pytorch": torch_n.summary,
            "jax_exponax": jax_n.summary,
            "gif": relpath(gif),
            "loss_png": relpath(loss_png),
        }

    save_csv(output_root / "metrics_all.csv", all_rows)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(json.dumps(jsonable(summary), indent=2), flush=True)
    print(f"[saved] {summary_path}", flush=True)
    print(f"[saved] {output_root / 'metrics_all.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
