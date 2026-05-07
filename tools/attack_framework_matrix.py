#!/usr/bin/env python3
"""Run PGD attacks for all PyTorch/JAX solver-model combinations.

The attack maximizes ||model(input) - solver(input)||_2^2 with projected
gradient ascent on the input.  Mixed-framework paths use DLPack autograd
bridges; same-framework paths stay native.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COMBOS = {
    "torch_solver_torch_model": ("torch", "torch"),
    "torch_solver_jax_model": ("torch", "jax"),
    "jax_solver_torch_model": ("jax", "torch"),
    "jax_solver_jax_model": ("jax", "jax"),
}


DEFAULT_BURGERS_MODEL_DIR = (
    PROJECT_ROOT / "1D_Burgers" / "trained_models" / "attack_ready" / "burgers_nu0.001_fno1d_500"
)
DEFAULT_BURGERS_TEST = (
    PROJECT_ROOT
    / "1D_Burgers"
    / "datasets"
    / "1D"
    / "Burgers"
    / "batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
)
DEFAULT_NS_RUN = PROJECT_ROOT / "fno_training_runs" / "ns_m12_w20_profile_500" / "ns_real_initial_laxmap" / "ns_2d"
DEFAULT_NS_TEST = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent"
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "real_initial_laxmap_single"
    / "test"
    / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
)


@dataclass
class AttackResult:
    combo: str
    rows: list[dict[str, Any]]
    input_history: list[np.ndarray]
    model_history: list[np.ndarray]
    solver_history: list[np.ndarray]
    summary: dict[str, Any]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def finite_or_none(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_or_none(payload), indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def global_gpu_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        values = [float(line.strip()) for line in out.splitlines() if line.strip()]
        return values[0] if values else float("nan")
    except Exception:
        return float("nan")


class PhaseGpuSampler:
    """Coarse per-phase GPU memory peak sampler using nvidia-smi."""

    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.samples.append(global_gpu_used_mib())

        def loop():
            while not self._stop.wait(self.interval_seconds):
                self.samples.append(global_gpu_used_mib())

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.samples.append(global_gpu_used_mib())
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.samples.append(global_gpu_used_mib())

    @property
    def peak_mib(self) -> float:
        arr = np.asarray(self.samples, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(arr.max()) if arr.size else float("nan")


def sync_torch(torch_mod, device=None) -> None:
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize(device)


def torch_peak_mib(torch_mod, device=None) -> float:
    if not torch_mod.cuda.is_available():
        return float("nan")
    return float(torch_mod.cuda.max_memory_allocated(device) / 1024**2)


class JaxToTorch(torch.autograd.Function if "torch" in sys.modules else object):
    pass


def make_jax_torch_apply():
    import jax
    import torch

    class _JaxToTorch(torch.autograd.Function):
        _vjp_cache: dict[int, Any] = {}

        @staticmethod
        def forward(ctx, x_torch, fn):
            x_dlpack = torch.utils.dlpack.to_dlpack(x_torch.contiguous())
            x_jax = jax.dlpack.from_dlpack(x_dlpack)
            y_jax = fn(x_jax)
            y_jax.block_until_ready()
            y_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(y_jax))
            ctx.save_for_backward(x_torch)
            ctx.fn = fn
            return y_torch

        @staticmethod
        def backward(ctx, grad_output):
            (x_torch,) = ctx.saved_tensors
            fn = ctx.fn
            cache_key = id(fn)
            if cache_key not in _JaxToTorch._vjp_cache:
                def vjp_apply(x_jax, grad_jax):
                    _, vjp_fn = jax.vjp(fn, x_jax)
                    return vjp_fn(grad_jax)[0]

                _JaxToTorch._vjp_cache[cache_key] = jax.jit(vjp_apply)
            grad_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(grad_output.contiguous()))
            x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch.contiguous()))
            grad_x = _JaxToTorch._vjp_cache[cache_key](x_jax, grad_jax)
            grad_x.block_until_ready()
            return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(grad_x)), None

    return _JaxToTorch.apply


def summarize(values: list[float]) -> dict[str, Any]:
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


def summarize_attack(rows: list[dict[str, Any]], total_seconds: float) -> dict[str, Any]:
    active = [row for row in rows if row["step"] < rows[-1]["step"]]
    return {
        "initial_loss": rows[0]["loss"],
        "final_loss": rows[-1]["loss"],
        "loss_growth_factor": rows[-1]["loss"] / rows[0]["loss"] if rows[0]["loss"] else float("inf"),
        "total_seconds": total_seconds,
        "forward_total_seconds": summarize([r["forward_total_seconds"] for r in active]),
        "model_forward_seconds": summarize([r["model_forward_seconds"] for r in active]),
        "solver_forward_seconds": summarize([r["solver_forward_seconds"] for r in active]),
        "backward_seconds": summarize([r["backward_seconds"] for r in active]),
        "update_seconds": summarize([r["update_seconds"] for r in active]),
        "forward_peak_gpu_mib": summarize([r["forward_peak_gpu_mib"] for r in active]),
        "backward_peak_gpu_mib": summarize([r["backward_peak_gpu_mib"] for r in active]),
        "model_forward_peak_global_gpu_mib": summarize([r["model_forward_peak_global_gpu_mib"] for r in active]),
        "solver_forward_peak_global_gpu_mib": summarize([r["solver_forward_peak_global_gpu_mib"] for r in active]),
        "loss_peak_global_gpu_mib": summarize([r["loss_peak_global_gpu_mib"] for r in active]),
        "backward_peak_global_gpu_mib": summarize([r["backward_peak_global_gpu_mib"] for r in active]),
        "update_peak_global_gpu_mib": summarize([r["update_peak_global_gpu_mib"] for r in active]),
        "global_gpu_after_forward_mib": summarize([r["global_gpu_after_forward_mib"] for r in active]),
        "global_gpu_after_backward_mib": summarize([r["global_gpu_after_backward_mib"] for r in active]),
    }


def load_burgers_sample(path: Path, sample_index: int) -> np.ndarray:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"][sample_index].float().numpy()
    return x[..., None].astype(np.float32)


def load_ns_sample(path: Path, sample_index: int, t_in: int = 10) -> np.ndarray:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    seq = data["y"][sample_index].float().numpy()
    return seq[..., :t_in].astype(np.float32)


def load_burgers_torch_model(path: Path, device):
    import torch

    mod = load_module("attack_fno1d_torch", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d.py")
    model = mod.FNO1d(modes=16, width=64, num_layers=4, dtype=torch.float32).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_ns_torch_model(path: Path, device):
    import torch

    mod = load_module("attack_fno2d_torch", PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d.py")
    fno = mod.FNO2d(modes1=12, modes2=12, width=20, num_layers=4, in_channels=10).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    fno.load_state_dict(ckpt.get("model_state_dict", ckpt))
    fno.eval()
    for param in fno.parameters():
        param.requires_grad_(False)
    return mod.RecurrentPredictor(fno, T_out=10, step=1).to(device).eval()


def load_jax_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["params"] if isinstance(payload, dict) and "params" in payload else payload


def make_burgers_jax_model(path: Path):
    import jax
    import jax.numpy as jnp

    mod = load_module("attack_fno1d_jax_ri", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d_jax_real_imag.py")
    params = jax.device_put(load_jax_checkpoint(path))

    @jax.jit
    def fn(x):
        return mod.fno1d_apply(params, x, modes=16, width=64, num_layers=4, dtype=jnp.float32)

    return fn


def make_ns_jax_model(path: Path):
    import jax
    import jax.numpy as jnp

    mod = load_module("attack_fno2d_jax_ri", PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d_jax_real_imag.py")
    params = jax.device_put(load_jax_checkpoint(path))

    @jax.jit
    def fn(x):
        return mod.recurrent_predictor_apply(
            params, x, modes1=12, modes2=12, width=20, num_layers=4, T_out=10, step=1, dtype=jnp.float32
        )

    return fn


def make_burgers_torch_solver(args, device):
    import torch
    from solvers import solve_burgers_final_batch

    def fn(x):
        y = solve_burgers_final_batch(
            x[..., 0],
            t_final=args.burgers_t_final,
            dt=args.burgers_dt,
            domain_extent=args.burgers_domain,
            diffusivity=args.burgers_nu,
            device=device,
            dtype=torch.float32,
        )
        return y[..., None]

    return fn


def make_ns_torch_solver(args, device):
    from solvers import solve_ns_zongyi_rollout_batch

    def fn(x):
        seq = solve_ns_zongyi_rollout_batch(
            x[..., -1],
            t_final=args.ns_t_out,
            fixed_step=args.ns_dt,
            domain_extent=args.ns_domain,
            diffusivity=args.ns_nu,
            return_time_last=True,
            device=device,
        )
        return seq[..., 1:]

    return fn


def make_burgers_jax_solver(args):
    import exponax as ex
    import jax
    import jax.numpy as jnp

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

    @jax.jit
    def fn(x):
        y = jax.vmap(lambda u: repeat_fn(jnp.expand_dims(u[:, 0], axis=0))[0])(x)
        return y[..., None]

    return fn


def make_ns_jax_solver(args):
    import exponax as ex
    import jax
    import jax.numpy as jnp

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

    def one_sample(x):
        u = jnp.rot90(jnp.flip(x[..., -1], axis=-2), 3, axes=(-2, -1))

        def micro(carry, _):
            return stepper(carry[None, ...])[0], None

        def one_second(carry, _):
            u_next, _ = jax.lax.scan(micro, carry, None, length=steps_per_second)
            return u_next, u_next

        _, seconds = jax.lax.scan(one_second, u, None, length=args.ns_t_out)
        return jnp.swapaxes(seconds, -1, -2).transpose(1, 2, 0)

    return jax.jit(jax.vmap(one_sample))


def run_torch_or_mixed_attack(
    *,
    args,
    case: str,
    combo: str,
    x0_np: np.ndarray,
    model_framework: str,
    solver_framework: str,
    torch_model,
    jax_model,
    torch_solver,
    jax_solver,
    device,
) -> AttackResult:
    import torch

    jax_apply = make_jax_torch_apply()
    x0 = torch.as_tensor(x0_np[None, ...], device=device, dtype=torch.float32)
    x_adv = x0.clone().detach().requires_grad_(True)
    lower = x0 - (args.ns_epsilon if case == "ns_2d" else args.burgers_epsilon)
    upper = x0 + (args.ns_epsilon if case == "ns_2d" else args.burgers_epsilon)
    alpha = args.ns_alpha if case == "ns_2d" else args.burgers_alpha

    def model_fn(x):
        return torch_model(x) if model_framework == "torch" else jax_apply(x, jax_model)

    def solver_fn(x):
        return torch_solver(x) if solver_framework == "torch" else jax_apply(x, jax_solver)

    rows: list[dict[str, Any]] = []
    input_history: list[np.ndarray] = []
    model_history: list[np.ndarray] = []
    solver_history: list[np.ndarray] = []
    total_start = time.perf_counter()

    for step in range(args.steps + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        global_before = global_gpu_used_mib()

        sync_torch(torch, device)
        with PhaseGpuSampler(args.gpu_sample_interval) as model_mem:
            model_start = time.perf_counter()
            model_out = model_fn(x_adv)
            sync_torch(torch, device)
            model_seconds = time.perf_counter() - model_start
        model_peak_global = model_mem.peak_mib

        with PhaseGpuSampler(args.gpu_sample_interval) as solver_mem:
            solver_start = time.perf_counter()
            solver_out = solver_fn(x_adv)
            sync_torch(torch, device)
            solver_seconds = time.perf_counter() - solver_start
        solver_peak_global = solver_mem.peak_mib

        with PhaseGpuSampler(args.gpu_sample_interval) as loss_mem:
            loss_start = time.perf_counter()
            loss = torch.mean((model_out - solver_out) ** 2)
            sync_torch(torch, device)
            loss_seconds = time.perf_counter() - loss_start
        loss_peak_global = loss_mem.peak_mib
        sync_torch(torch, device)
        forward_peak = torch_peak_mib(torch, device)
        global_after_forward = global_gpu_used_mib()

        input_history.append(x_adv.detach().cpu().numpy()[0].copy())
        model_history.append(model_out.detach().cpu().numpy()[0].copy())
        solver_history.append(solver_out.detach().cpu().numpy()[0].copy())

        backward_seconds = 0.0
        backward_peak = float("nan")
        update_seconds = 0.0
        grad_norm = float("nan")
        global_after_backward = float("nan")
        backward_peak_global = float("nan")
        update_peak_global = float("nan")
        if step < args.steps:
            if x_adv.grad is not None:
                x_adv.grad = None
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            with PhaseGpuSampler(args.gpu_sample_interval) as backward_mem:
                backward_start = time.perf_counter()
                loss.backward()
                sync_torch(torch, device)
                backward_seconds = time.perf_counter() - backward_start
            backward_peak_global = backward_mem.peak_mib
            backward_peak = torch_peak_mib(torch, device)
            global_after_backward = global_gpu_used_mib()
            grad = x_adv.grad.detach()
            grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu())
            with PhaseGpuSampler(args.gpu_sample_interval) as update_mem:
                update_start = time.perf_counter()
                with torch.no_grad():
                    x_adv = torch.maximum(torch.minimum(x_adv + alpha * torch.sign(grad), upper), lower)
                x_adv = x_adv.detach().requires_grad_(True)
                sync_torch(torch, device)
                update_seconds = time.perf_counter() - update_start
            update_peak_global = update_mem.peak_mib

        rows.append(
            {
                "case": case,
                "combo": combo,
                "solver_framework": solver_framework,
                "model_framework": model_framework,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "forward_total_seconds": model_seconds + solver_seconds + loss_seconds,
                "model_forward_seconds": model_seconds,
                "solver_forward_seconds": solver_seconds,
                "loss_seconds": loss_seconds,
                "backward_seconds": backward_seconds,
                "update_seconds": update_seconds,
                "grad_norm": grad_norm,
                "forward_peak_gpu_mib": forward_peak,
                "backward_peak_gpu_mib": backward_peak,
                "model_forward_peak_global_gpu_mib": model_peak_global,
                "solver_forward_peak_global_gpu_mib": solver_peak_global,
                "loss_peak_global_gpu_mib": loss_peak_global,
                "backward_peak_global_gpu_mib": backward_peak_global,
                "update_peak_global_gpu_mib": update_peak_global,
                "global_gpu_before_mib": global_before,
                "global_gpu_after_forward_mib": global_after_forward,
                "global_gpu_after_backward_mib": global_after_backward,
            }
        )

    return AttackResult(combo, rows, input_history, model_history, solver_history, summarize_attack(rows, time.perf_counter() - total_start))


def run_pure_jax_attack(
    *,
    args,
    case: str,
    combo: str,
    x0_np: np.ndarray,
    jax_model,
    jax_solver,
) -> AttackResult:
    import jax
    import jax.numpy as jnp

    alpha = args.ns_alpha if case == "ns_2d" else args.burgers_alpha
    epsilon = args.ns_epsilon if case == "ns_2d" else args.burgers_epsilon
    x0 = jnp.asarray(x0_np[None, ...], dtype=jnp.float32)
    lower = x0 - epsilon
    upper = x0 + epsilon

    @jax.jit
    def model_loss_solver(x):
        model_out = jax_model(x)
        solver_out = jax_solver(x)
        loss = jnp.mean((model_out - solver_out) ** 2)
        return loss, (model_out, solver_out)

    value_and_grad = jax.jit(jax.value_and_grad(lambda z: model_loss_solver(z)[0]))

    @jax.jit
    def update(x, grad):
        return jnp.clip(x + alpha * jnp.sign(grad), lower, upper)

    x = x0
    rows: list[dict[str, Any]] = []
    input_history: list[np.ndarray] = []
    model_history: list[np.ndarray] = []
    solver_history: list[np.ndarray] = []
    total_start = time.perf_counter()

    for step in range(args.steps + 1):
        global_before = global_gpu_used_mib()

        with PhaseGpuSampler(args.gpu_sample_interval) as model_mem:
            model_start = time.perf_counter()
            model_out = jax_model(x)
            model_out.block_until_ready()
            model_seconds = time.perf_counter() - model_start
        model_peak_global = model_mem.peak_mib

        with PhaseGpuSampler(args.gpu_sample_interval) as solver_mem:
            solver_start = time.perf_counter()
            solver_out = jax_solver(x)
            solver_out.block_until_ready()
            solver_seconds = time.perf_counter() - solver_start
        solver_peak_global = solver_mem.peak_mib

        with PhaseGpuSampler(args.gpu_sample_interval) as loss_mem:
            loss_start = time.perf_counter()
            loss_forward = jnp.mean((model_out - solver_out) ** 2)
            loss_forward.block_until_ready()
            loss_seconds = time.perf_counter() - loss_start
        loss_peak_global = loss_mem.peak_mib
        forward_seconds = model_seconds + solver_seconds + loss_seconds
        global_after_forward = global_gpu_used_mib()

        input_history.append(np.asarray(x[0]).copy())
        model_history.append(np.asarray(model_out[0]).copy())
        solver_history.append(np.asarray(solver_out[0]).copy())

        value_grad_seconds = 0.0
        backward_seconds = 0.0
        update_seconds = 0.0
        grad_norm = float("nan")
        global_after_backward = float("nan")
        backward_peak_global = float("nan")
        update_peak_global = float("nan")
        loss_value = float(loss_forward)
        if step < args.steps:
            with PhaseGpuSampler(args.gpu_sample_interval) as backward_mem:
                vg_start = time.perf_counter()
                loss_vg, grad = value_and_grad(x)
                grad.block_until_ready()
                value_grad_seconds = time.perf_counter() - vg_start
            backward_peak_global = backward_mem.peak_mib
            backward_seconds = max(0.0, value_grad_seconds - forward_seconds)
            global_after_backward = global_gpu_used_mib()
            grad_norm = float(jnp.linalg.norm(grad))
            with PhaseGpuSampler(args.gpu_sample_interval) as update_mem:
                update_start = time.perf_counter()
                x = update(x, grad)
                x.block_until_ready()
                update_seconds = time.perf_counter() - update_start
            update_peak_global = update_mem.peak_mib
            loss_value = float(loss_vg)

        rows.append(
            {
                "case": case,
                "combo": combo,
                "solver_framework": "jax",
                "model_framework": "jax",
                "step": step,
                "loss": loss_value,
                "forward_total_seconds": forward_seconds,
                "model_forward_seconds": model_seconds,
                "solver_forward_seconds": solver_seconds,
                "loss_seconds": loss_seconds,
                "backward_seconds": backward_seconds,
                "jax_value_and_grad_seconds": value_grad_seconds,
                "update_seconds": update_seconds,
                "grad_norm": grad_norm,
                "forward_peak_gpu_mib": max(model_peak_global, solver_peak_global, loss_peak_global),
                "backward_peak_gpu_mib": backward_peak_global,
                "model_forward_peak_global_gpu_mib": model_peak_global,
                "solver_forward_peak_global_gpu_mib": solver_peak_global,
                "loss_peak_global_gpu_mib": loss_peak_global,
                "backward_peak_global_gpu_mib": backward_peak_global,
                "update_peak_global_gpu_mib": update_peak_global,
                "global_gpu_before_mib": global_before,
                "global_gpu_after_forward_mib": global_after_forward,
                "global_gpu_after_backward_mib": global_after_backward,
            }
        )

    return AttackResult(combo, rows, input_history, model_history, solver_history, summarize_attack(rows, time.perf_counter() - total_start))


def plot_loss_curves(case_dir: Path, case: str, results: dict[str, AttackResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = case_dir / "loss_progression_all_combos.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for combo, result in results.items():
        ax.plot([r["step"] for r in result.rows], [r["loss"] for r in result.rows], label=combo)
    ax.set_xlabel("PGD step")
    ax.set_ylabel("attack loss")
    ax.set_yscale("log")
    ax.set_title(f"{case} attack loss progression")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_gif(case_dir: Path, case: str, results: dict[str, AttackResult], max_frames: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    vis_dir = case_dir / "frames"
    vis_dir.mkdir(parents=True, exist_ok=True)
    steps = len(next(iter(results.values())).rows)
    frame_ids = np.unique(np.linspace(0, steps - 1, min(max_frames, steps), dtype=int))
    paths: list[Path] = []

    for frame_id in frame_ids:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes_flat = axes.ravel()
        for ax, (combo, result) in zip(axes_flat, results.items()):
            model = result.model_history[frame_id]
            solver = result.solver_history[frame_id]
            loss = result.rows[frame_id]["loss"]
            if case == "burgers_1d":
                x = np.arange(model.shape[0])
                ax.plot(x, model[..., 0], label="model", linewidth=1.0)
                ax.plot(x, solver[..., 0], label="solver", linewidth=1.0, linestyle="--")
                ax.set_ylim(
                    min(float(model.min()), float(solver.min())) - 0.1,
                    max(float(model.max()), float(solver.max())) + 0.1,
                )
            else:
                diff = model[..., -1] - solver[..., -1]
                im = ax.imshow(diff, cmap="coolwarm")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{combo}\nstep={frame_id}, loss={loss:.3e}", fontsize=9)
            ax.grid(alpha=0.15)
            if case == "burgers_1d":
                ax.legend(fontsize=7)
        fig.tight_layout()
        path = vis_dir / f"{case}_attack_step_{frame_id:04d}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)

    gif_path = case_dir / f"{case}_attack_all_combos.gif"
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in paths]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=250, loop=0)
    for image in images:
        image.close()
    return gif_path


def run_case(args, case: str) -> dict[str, AttackResult]:
    import torch

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"))
    if case == "burgers_1d":
        x0_np = load_burgers_sample(args.burgers_test_path, args.sample_index)
        torch_model = load_burgers_torch_model(args.burgers_torch_checkpoint, device)
        jax_model = make_burgers_jax_model(args.burgers_jax_checkpoint)
        torch_solver = make_burgers_torch_solver(args, device)
        jax_solver = make_burgers_jax_solver(args)
    elif case == "ns_2d":
        x0_np = load_ns_sample(args.ns_test_path, args.sample_index, t_in=args.ns_t_in)
        torch_model = load_ns_torch_model(args.ns_torch_checkpoint, device)
        jax_model = make_ns_jax_model(args.ns_jax_checkpoint)
        torch_solver = make_ns_torch_solver(args, device)
        jax_solver = make_ns_jax_solver(args)
    else:
        raise ValueError(case)

    selected = list(COMBOS) if args.combos == "all" else [c.strip() for c in args.combos.split(",") if c.strip()]
    results: dict[str, AttackResult] = {}
    for combo in selected:
        solver_framework, model_framework = COMBOS[combo]
        print(f"[{case}] running {combo}: solver={solver_framework}, model={model_framework}", flush=True)
        if solver_framework == "jax" and model_framework == "jax":
            result = run_pure_jax_attack(
                args=args,
                case=case,
                combo=combo,
                x0_np=x0_np,
                jax_model=jax_model,
                jax_solver=jax_solver,
            )
        else:
            result = run_torch_or_mixed_attack(
                args=args,
                case=case,
                combo=combo,
                x0_np=x0_np,
                model_framework=model_framework,
                solver_framework=solver_framework,
                torch_model=torch_model,
                jax_model=jax_model,
                torch_solver=torch_solver,
                jax_solver=jax_solver,
                device=device,
            )
        results[combo] = result
    return results


def write_case_outputs(args, case: str, results: dict[str, AttackResult]) -> dict[str, Any]:
    case_dir = args.output_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for combo, result in results.items():
        combo_dir = case_dir / combo
        combo_dir.mkdir(parents=True, exist_ok=True)
        save_csv(combo_dir / "metrics.csv", result.rows)
        save_json(combo_dir / "summary.json", result.summary)
        np.savez_compressed(
            combo_dir / "histories.npz",
            input_history=np.asarray(result.input_history, dtype=np.float32),
            model_history=np.asarray(result.model_history, dtype=np.float32),
            solver_history=np.asarray(result.solver_history, dtype=np.float32),
        )
        all_rows.extend(result.rows)
        summary[combo] = result.summary

    save_csv(case_dir / "metrics_all_combos.csv", all_rows)
    loss_png = plot_loss_curves(case_dir, case, results)
    gif_path = make_gif(case_dir, case, results, args.max_gif_frames)
    summary["outputs"] = {
        "metrics_csv": rel(case_dir / "metrics_all_combos.csv"),
        "loss_png": rel(loss_png),
        "gif": rel(gif_path),
    }
    save_json(case_dir / "summary_all_combos.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="burgers_1d,ns_2d", help="Comma list: burgers_1d,ns_2d")
    parser.add_argument("--combos", default="all", help="all or comma list of combo names")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "benchmark_results" / "attack_framework_matrix_100step")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-gif-frames", type=int, default=30)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.02)

    parser.add_argument("--burgers-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--burgers-torch-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--burgers-jax-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "jax_real_imag_fno1d_500.pkl")
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-epsilon", type=float, default=0.05)
    parser.add_argument("--burgers-alpha", type=float, default=0.002)

    parser.add_argument("--ns-test-path", type=Path, default=DEFAULT_NS_TEST)
    parser.add_argument("--ns-torch-checkpoint", type=Path, default=DEFAULT_NS_RUN / "checkpoints" / "fno2d_pytorch.pt")
    parser.add_argument("--ns-jax-checkpoint", type=Path, default=DEFAULT_NS_RUN / "checkpoints" / "fno2d_jax_real_imag.pkl")
    parser.add_argument("--ns-nx", type=int, default=256)
    parser.add_argument("--ns-t-in", type=int, default=10)
    parser.add_argument("--ns-t-out", type=int, default=10)
    parser.add_argument("--ns-nu", type=float, default=1e-5)
    parser.add_argument("--ns-dt", type=float, default=0.005)
    parser.add_argument("--ns-domain", type=float, default=1.0)
    parser.add_argument("--ns-epsilon", type=float, default=0.05)
    parser.add_argument("--ns-alpha", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, Any] = {"config": vars(args), "cases": {}}
    for raw_case in args.cases.split(","):
        case = raw_case.strip()
        if not case:
            continue
        results = run_case(args, case)
        run_summary["cases"][case] = write_case_outputs(args, case, results)
    save_json(args.output_root / "summary.json", run_summary)
    print(f"[done] wrote {rel(args.output_root)}", flush=True)


if __name__ == "__main__":
    main()
