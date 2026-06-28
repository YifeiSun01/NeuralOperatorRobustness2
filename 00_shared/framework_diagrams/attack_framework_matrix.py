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
import shutil
import subprocess
import sys
import threading
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    grad_history: list[np.ndarray]
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
    if torch_mod.cuda.is_available() and (device is None or getattr(device, "type", None) == "cuda"):
        torch_mod.cuda.synchronize(device)


def torch_peak_mib(torch_mod, device=None) -> float:
    if not torch_mod.cuda.is_available() or (device is not None and getattr(device, "type", None) != "cuda"):
        return float("nan")
    return float(torch_mod.cuda.max_memory_allocated(device) / 1024**2)


def torch_memory_snapshot_mib(torch_mod, device=None) -> dict[str, float]:
    if not torch_mod.cuda.is_available() or (device is not None and getattr(device, "type", None) != "cuda"):
        return {
            "torch_cuda_allocated_mib": float("nan"),
            "torch_cuda_reserved_mib": float("nan"),
            "torch_cuda_peak_allocated_mib": float("nan"),
        }
    return {
        "torch_cuda_allocated_mib": float(torch_mod.cuda.memory_allocated(device) / 1024**2),
        "torch_cuda_reserved_mib": float(torch_mod.cuda.memory_reserved(device) / 1024**2),
        "torch_cuda_peak_allocated_mib": float(torch_mod.cuda.max_memory_allocated(device) / 1024**2),
    }


def cleanup_runtime_memory(torch_mod=None) -> None:
    gc.collect()
    if torch_mod is not None and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
        torch_mod.cuda.empty_cache()
        torch_mod.cuda.reset_peak_memory_stats()


def torch_dtype_from_name(name: str):
    import torch

    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype {name!r}")


def make_step_progress(args, *, case: str, combo: str, norm: str, epsilon: float, alpha: float):
    if args.no_progress:
        return range(args.steps + 1)
    try:
        from tqdm.auto import tqdm
    except Exception:
        print(
            f"[progress] tqdm is not installed; running plain loop for "
            f"case={case} combo={combo} steps={args.steps} norm={norm} eps={epsilon} alpha={alpha}",
            flush=True,
        )
        return range(args.steps + 1)

    return tqdm(
        range(args.steps + 1),
        total=args.steps + 1,
        desc=f"{case} | {combo} | {norm} eps={epsilon:g} alpha={alpha:g}",
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )


def progress_write(message: str) -> None:
    try:
        from tqdm.auto import tqdm

        tqdm.write(message)
    except Exception:
        print(message, flush=True)


def update_step_progress(progress, row: dict[str, Any]) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if set_postfix is None:
        return
    set_postfix(
        {
            "loss": f"{row['loss']:.3e}",
            "step_s": f"{row['total_pgd_step_seconds']:.3f}",
            "fwd": f"{row['forward_total_seconds']:.3f}",
            "bwd": f"{row['backward_seconds']:.3f}",
            "proj": f"{row['projection_seconds']:.3f}",
            "gpu_mib": f"{row.get('phase_peak_global_gpu_mib', float('nan')):.0f}",
        },
        refresh=True,
    )


def make_jax_torch_bridge():
    import jax
    import torch

    def torch_to_jax(tensor):
        return jax.dlpack.from_dlpack(tensor.detach().contiguous())

    def jax_to_torch(array):
        return torch.utils.dlpack.from_dlpack(array)

    class Bridge:
        records: list[dict[str, Any]] = []

        @classmethod
        def reset(cls) -> None:
            cls.records.clear()

        @classmethod
        def totals(cls) -> dict[str, float]:
            keys = [
                "forward_torch_to_jax_seconds",
                "forward_jax_compute_seconds",
                "forward_jax_to_torch_seconds",
                "backward_grad_torch_to_jax_seconds",
                "backward_input_torch_to_jax_seconds",
                "backward_jax_vjp_seconds",
                "backward_jax_to_torch_seconds",
            ]
            out = {key: 0.0 for key in keys}
            out["bridge_calls"] = float(len(cls.records))
            for record in cls.records:
                for key in keys:
                    out[key] += float(record.get(key, 0.0))
            return out

        def __call__(self, x_torch, fn, label: str):
            return _JaxToTorch.apply(x_torch, fn, label)

    class _JaxToTorch(torch.autograd.Function):
        _vjp_cache: dict[int, Any] = {}

        @staticmethod
        def forward(ctx, x_torch, fn, label):
            record: dict[str, Any] = {"label": str(label), "direction": "forward"}
            start = time.perf_counter()
            x_jax = torch_to_jax(x_torch)
            x_jax.block_until_ready()
            record["forward_torch_to_jax_seconds"] = time.perf_counter() - start

            start = time.perf_counter()
            y_jax = fn(x_jax)
            y_jax.block_until_ready()
            record["forward_jax_compute_seconds"] = time.perf_counter() - start

            start = time.perf_counter()
            y_torch = jax_to_torch(y_jax)
            sync_torch(torch, x_torch.device if x_torch.is_cuda else None)
            record["forward_jax_to_torch_seconds"] = time.perf_counter() - start
            Bridge.records.append(record)

            ctx.save_for_backward(x_torch)
            ctx.fn = fn
            ctx.label = str(label)
            return y_torch

        @staticmethod
        def backward(ctx, grad_output):
            (x_torch,) = ctx.saved_tensors
            fn = ctx.fn
            record: dict[str, Any] = {"label": ctx.label, "direction": "backward"}
            cache_key = id(fn)
            if cache_key not in _JaxToTorch._vjp_cache:
                def vjp_apply(x_jax, grad_jax):
                    _, vjp_fn = jax.vjp(fn, x_jax)
                    return vjp_fn(grad_jax)[0]

                _JaxToTorch._vjp_cache[cache_key] = jax.jit(vjp_apply)

            start = time.perf_counter()
            grad_jax = torch_to_jax(grad_output)
            grad_jax.block_until_ready()
            record["backward_grad_torch_to_jax_seconds"] = time.perf_counter() - start

            start = time.perf_counter()
            x_jax = torch_to_jax(x_torch)
            x_jax.block_until_ready()
            record["backward_input_torch_to_jax_seconds"] = time.perf_counter() - start

            start = time.perf_counter()
            grad_x = _JaxToTorch._vjp_cache[cache_key](x_jax, grad_jax)
            grad_x.block_until_ready()
            record["backward_jax_vjp_seconds"] = time.perf_counter() - start

            start = time.perf_counter()
            grad_x_torch = jax_to_torch(grad_x)
            sync_torch(torch, x_torch.device if x_torch.is_cuda else None)
            record["backward_jax_to_torch_seconds"] = time.perf_counter() - start
            Bridge.records.append(record)
            return grad_x_torch, None, None

    return Bridge()


def project_delta_torch(delta, epsilon: float, norm: str):
    import torch

    if norm in {"inf", "linf", "infinity"}:
        return torch.clamp(delta, -epsilon, epsilon)
    if norm in {"2", "l2"}:
        flat = delta.reshape(delta.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(float(epsilon) / norms, max=1.0)
        return (flat * scale).reshape_as(delta)
    raise ValueError(f"Unsupported norm {norm!r}")


def normalized_update_torch(grad, norm: str):
    import torch

    if norm in {"inf", "linf", "infinity"}:
        return torch.sign(grad)
    if norm in {"2", "l2"}:
        flat = grad.reshape(grad.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
        return (flat / norms).reshape_as(grad)
    raise ValueError(f"Unsupported norm {norm!r}")


def project_delta_jax(delta, epsilon: float, norm: str):
    import jax.numpy as jnp

    if norm in {"inf", "linf", "infinity"}:
        return jnp.clip(delta, -epsilon, epsilon)
    if norm in {"2", "l2"}:
        flat = delta.reshape((delta.shape[0], -1))
        norms = jnp.linalg.norm(flat, axis=1, keepdims=True)
        scale = jnp.minimum(1.0, epsilon / jnp.maximum(norms, 1e-12))
        return (flat * scale).reshape(delta.shape)
    raise ValueError(f"Unsupported norm {norm!r}")


def normalized_update_jax(grad, norm: str):
    import jax.numpy as jnp

    if norm in {"inf", "linf", "infinity"}:
        return jnp.sign(grad)
    if norm in {"2", "l2"}:
        flat = grad.reshape((grad.shape[0], -1))
        norms = jnp.linalg.norm(flat, axis=1, keepdims=True)
        return (flat / jnp.maximum(norms, 1e-12)).reshape(grad.shape)
    raise ValueError(f"Unsupported norm {norm!r}")


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
        "final_true_loss": rows[-1].get("true_loss", rows[-1]["loss"]),
        "loss_growth_factor": rows[-1]["loss"] / rows[0]["loss"] if rows[0]["loss"] else float("inf"),
        "total_seconds": total_seconds,
        "avg_step_time_seconds": summarize([r["total_pgd_step_seconds"] for r in active]),
        "forward_total_seconds": summarize([r["forward_total_seconds"] for r in active]),
        "model_forward_seconds": summarize([r["model_forward_seconds"] for r in active]),
        "solver_forward_seconds": summarize([r["solver_forward_seconds"] for r in active]),
        "loss_seconds": summarize([r["loss_seconds"] for r in active]),
        "backward_seconds": summarize([r["backward_seconds"] for r in active]),
        "projection_seconds": summarize([r["projection_seconds"] for r in active]),
        "forward_peak_gpu_mib": summarize([r["forward_peak_gpu_mib"] for r in active]),
        "backward_peak_gpu_mib": summarize([r["backward_peak_gpu_mib"] for r in active]),
        "peak_memory_mib": max(
            [
                value
                for row in rows
                for value in [
                    row.get("model_forward_peak_global_gpu_mib", float("nan")),
                    row.get("solver_forward_peak_global_gpu_mib", float("nan")),
                    row.get("loss_peak_global_gpu_mib", float("nan")),
                    row.get("backward_peak_global_gpu_mib", float("nan")),
                    row.get("projection_peak_global_gpu_mib", float("nan")),
                ]
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ],
            default=float("nan"),
        ),
        "model_forward_peak_global_gpu_mib": summarize([r["model_forward_peak_global_gpu_mib"] for r in active]),
        "solver_forward_peak_global_gpu_mib": summarize([r["solver_forward_peak_global_gpu_mib"] for r in active]),
        "loss_peak_global_gpu_mib": summarize([r["loss_peak_global_gpu_mib"] for r in active]),
        "backward_peak_global_gpu_mib": summarize([r["backward_peak_global_gpu_mib"] for r in active]),
        "projection_peak_global_gpu_mib": summarize([r["projection_peak_global_gpu_mib"] for r in active]),
        "global_gpu_after_forward_mib": summarize([r["global_gpu_after_forward_mib"] for r in active]),
        "global_gpu_after_backward_mib": summarize([r["global_gpu_after_backward_mib"] for r in active]),
        "conversion_forward_torch_to_jax_seconds": summarize(
            [r.get("conversion_forward_torch_to_jax_seconds", float("nan")) for r in active]
        ),
        "conversion_forward_jax_to_torch_seconds": summarize(
            [r.get("conversion_forward_jax_to_torch_seconds", float("nan")) for r in active]
        ),
        "conversion_backward_torch_to_jax_seconds": summarize(
            [r.get("conversion_backward_torch_to_jax_seconds", float("nan")) for r in active]
        ),
        "conversion_backward_jax_to_torch_seconds": summarize(
            [r.get("conversion_backward_jax_to_torch_seconds", float("nan")) for r in active]
        ),
        "conversion_backward_jax_vjp_seconds": summarize(
            [r.get("conversion_backward_jax_vjp_seconds", float("nan")) for r in active]
        ),
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
    from solvers import Burgers1DETDRK4, solve_burgers_final_batch, solve_burgers_final_batch_with_solver

    solver_dtype = torch_dtype_from_name(args.burgers_torch_solver_dtype)
    steps = int(round(args.burgers_t_final / args.burgers_dt))

    if args.torch_compile_solver:
        solver = Burgers1DETDRK4(
            num_points=args.burgers_nx,
            domain_extent=args.burgers_domain,
            dt=args.burgers_dt,
            diffusivity=args.burgers_nu,
            device=device,
            dtype=solver_dtype,
        )
        step_fn = maybe_compile_torch_solver(args, solver.step, "burgers_torch_solver_step")

        def fn(x):
            y = solve_burgers_final_batch_with_solver(x[..., 0], solver=solver, steps=steps, step_fn=step_fn)
            return y[..., None]

        return fn

    def fn(x):
        y = solve_burgers_final_batch(
            x[..., 0],
            t_final=args.burgers_t_final,
            dt=args.burgers_dt,
            domain_extent=args.burgers_domain,
            diffusivity=args.burgers_nu,
            device=device,
            dtype=solver_dtype,
        )
        return y[..., None]

    return fn


def make_ns_torch_solver(args, device):
    from solvers import (
        NavierStokesVorticity2DZongyiETDRK4,
        solve_ns_zongyi_rollout_batch,
        solve_ns_zongyi_rollout_batch_with_solver,
    )

    solver_dtype = torch_dtype_from_name(args.ns_torch_solver_dtype)

    if args.torch_compile_solver:
        solver = NavierStokesVorticity2DZongyiETDRK4(
            num_points=args.ns_nx,
            domain_extent=args.ns_domain,
            dt=args.ns_dt,
            diffusivity=args.ns_nu,
            num_circle_points=16,
            device=device,
            dtype=solver_dtype,
        )
        step_fn = maybe_compile_torch_solver(args, solver.step, "ns_torch_solver_step")

        def fn(x):
            seq = solve_ns_zongyi_rollout_batch_with_solver(
                x[..., -1],
                solver=solver,
                t_final=args.ns_t_out,
                fixed_step=args.ns_dt,
                return_time_last=True,
                step_fn=step_fn,
            )
            return seq[..., 1:]

        return fn

    def fn(x):
        seq = solve_ns_zongyi_rollout_batch(
            x[..., -1],
            t_final=args.ns_t_out,
            fixed_step=args.ns_dt,
            domain_extent=args.ns_domain,
            diffusivity=args.ns_nu,
            return_time_last=True,
            device=device,
            dtype=solver_dtype,
        )
        return seq[..., 1:]

    return fn


def maybe_compile_torch_solver(args, fn, name: str):
    if not args.torch_compile_solver:
        return fn
    import torch

    compile_kwargs: dict[str, Any] = {
        "backend": args.torch_compile_backend,
        "mode": args.torch_compile_mode,
        "fullgraph": args.torch_compile_fullgraph,
    }
    if args.torch_compile_dynamic is not None:
        compile_kwargs["dynamic"] = args.torch_compile_dynamic
    print(f"[torch-compile] compiling {name} with {compile_kwargs}", flush=True)
    return torch.compile(fn, **compile_kwargs)


def make_burgers_jax_solver(args):
    import exponax as ex
    import jax
    import jax.numpy as jnp

    solver_dtype = jnp.float64 if args.burgers_jax_solver_dtype == "float64" else jnp.float32
    jax.config.update("jax_enable_x64", args.burgers_jax_solver_dtype == "float64")
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
        y = jax.vmap(lambda u: repeat_fn(jnp.expand_dims(u[:, 0].astype(solver_dtype), axis=0))[0])(x)
        return y[..., None]

    return fn


def make_ns_jax_solver(args):
    import exponax as ex
    import jax
    import jax.numpy as jnp

    solver_dtype = jnp.float64 if args.ns_jax_solver_dtype == "float64" else jnp.float32
    jax.config.update("jax_enable_x64", args.ns_jax_solver_dtype == "float64")
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
        u = jnp.rot90(jnp.flip(x[..., -1].astype(solver_dtype), axis=-2), 3, axes=(-2, -1))

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

    cleanup_runtime_memory(torch)
    bridge = make_jax_torch_bridge()
    x0 = torch.as_tensor(x0_np[None, ...], device=device, dtype=torch.float32)
    x_adv = x0.clone().detach().requires_grad_(True)
    epsilon = args.ns_epsilon if case == "ns_2d" else args.burgers_epsilon
    alpha = args.ns_alpha if case == "ns_2d" else args.burgers_alpha
    norm = args.ns_norm if case == "ns_2d" else args.burgers_norm
    torch_solver_dtype = args.ns_torch_solver_dtype if case == "ns_2d" else args.burgers_torch_solver_dtype
    jax_solver_dtype = args.ns_jax_solver_dtype if case == "ns_2d" else args.burgers_jax_solver_dtype

    def model_fn(x):
        return torch_model(x.to(torch.float32)) if model_framework == "torch" else bridge(x, jax_model, "model")

    def solver_fn(x):
        return torch_solver(x) if solver_framework == "torch" else bridge(x, jax_solver, "solver")

    rows: list[dict[str, Any]] = []
    input_history: list[np.ndarray] = []
    model_history: list[np.ndarray] = []
    solver_history: list[np.ndarray] = []
    grad_history: list[np.ndarray] = []
    total_start = time.perf_counter()
    progress_write(
        "[attack-start] "
        f"case={case} combo={combo} solver={solver_framework} model={model_framework} "
        f"steps={args.steps} norm={norm} epsilon={epsilon} alpha={alpha} device={device} "
        f"torch_solver_dtype={torch_solver_dtype} jax_solver_dtype={jax_solver_dtype}"
    )

    progress = make_step_progress(args, case=case, combo=combo, norm=norm, epsilon=epsilon, alpha=alpha)
    for step in progress:
        cleanup_runtime_memory(torch)
        step_start = time.perf_counter()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        global_before = global_gpu_used_mib()

        sync_torch(torch, device)
        bridge.reset()
        with PhaseGpuSampler(args.gpu_sample_interval) as model_mem:
            model_start = time.perf_counter()
            model_out = model_fn(x_adv)
            sync_torch(torch, device)
            model_seconds = time.perf_counter() - model_start
        model_peak_global = model_mem.peak_mib
        model_bridge = bridge.totals()

        bridge.reset()
        with PhaseGpuSampler(args.gpu_sample_interval) as solver_mem:
            solver_start = time.perf_counter()
            solver_out = solver_fn(x_adv)
            sync_torch(torch, device)
            solver_seconds = time.perf_counter() - solver_start
        solver_peak_global = solver_mem.peak_mib
        solver_bridge = bridge.totals()

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
        current_delta = (x_adv.detach() - x0).detach()
        delta_norm_l2 = float(torch.linalg.vector_norm(current_delta).detach().cpu())
        delta_norm_linf = float(current_delta.abs().max().detach().cpu())

        backward_seconds = 0.0
        backward_peak = float("nan")
        projection_seconds = 0.0
        grad_norm = float("nan")
        global_after_backward = float("nan")
        backward_peak_global = float("nan")
        projection_peak_global = float("nan")
        backward_bridge: dict[str, float] = {}
        if step < args.steps:
            if x_adv.grad is not None:
                x_adv.grad = None
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            bridge.reset()
            with PhaseGpuSampler(args.gpu_sample_interval) as backward_mem:
                backward_start = time.perf_counter()
                loss.backward()
                sync_torch(torch, device)
                backward_seconds = time.perf_counter() - backward_start
            backward_peak_global = backward_mem.peak_mib
            backward_bridge = bridge.totals()
            backward_peak = torch_peak_mib(torch, device)
            global_after_backward = global_gpu_used_mib()
            grad = x_adv.grad.detach()
            grad_history.append(grad.detach().cpu().numpy()[0].copy())
            grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu())
            with PhaseGpuSampler(args.gpu_sample_interval) as projection_mem:
                projection_start = time.perf_counter()
                with torch.no_grad():
                    delta = x_adv - x0
                    direction = normalized_update_torch(grad, norm)
                    delta = project_delta_torch(delta + alpha * direction, epsilon, norm)
                    x_adv = x0 + delta
                x_adv = x_adv.detach().requires_grad_(True)
                sync_torch(torch, device)
                projection_seconds = time.perf_counter() - projection_start
            projection_peak_global = projection_mem.peak_mib

        step_seconds = time.perf_counter() - step_start
        conversion_forward_torch_to_jax = (
            model_bridge.get("forward_torch_to_jax_seconds", 0.0)
            + solver_bridge.get("forward_torch_to_jax_seconds", 0.0)
        )
        conversion_forward_jax_to_torch = (
            model_bridge.get("forward_jax_to_torch_seconds", 0.0)
            + solver_bridge.get("forward_jax_to_torch_seconds", 0.0)
        )
        conversion_backward_torch_to_jax = (
            backward_bridge.get("backward_grad_torch_to_jax_seconds", 0.0)
            + backward_bridge.get("backward_input_torch_to_jax_seconds", 0.0)
        )
        torch_mem = torch_memory_snapshot_mib(torch, device)
        phase_peak_global = max(
            [
                value
                for value in [
                    model_peak_global,
                    solver_peak_global,
                    loss_peak_global,
                    backward_peak_global,
                    projection_peak_global,
                ]
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ],
            default=float("nan"),
        )

        row = {
                "case": case,
                "combo": combo,
                "solver_framework": solver_framework,
                "model_framework": model_framework,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "attack_loss": float(loss.detach().cpu()),
                "true_loss": float(loss.detach().cpu()),
                "norm": norm,
                "epsilon": epsilon,
                "alpha": alpha,
                "torch_solver_dtype": torch_solver_dtype,
                "jax_solver_dtype": jax_solver_dtype,
                "total_pgd_step_seconds": step_seconds,
                "forward_total_seconds": model_seconds + solver_seconds + loss_seconds,
                "model_forward_seconds": model_seconds,
                "solver_forward_seconds": solver_seconds,
                "loss_seconds": loss_seconds,
                "backward_seconds": backward_seconds,
                "backward_includes_forward": False,
                "projection_seconds": projection_seconds,
                "delta_l2": delta_norm_l2,
                "delta_linf": delta_norm_linf,
                "grad_norm": grad_norm,
                "forward_peak_gpu_mib": forward_peak,
                "backward_peak_gpu_mib": backward_peak,
                "phase_peak_global_gpu_mib": phase_peak_global,
                **torch_mem,
                "model_forward_peak_global_gpu_mib": model_peak_global,
                "solver_forward_peak_global_gpu_mib": solver_peak_global,
                "loss_peak_global_gpu_mib": loss_peak_global,
                "backward_peak_global_gpu_mib": backward_peak_global,
                "projection_peak_global_gpu_mib": projection_peak_global,
                "global_gpu_before_mib": global_before,
                "global_gpu_after_forward_mib": global_after_forward,
                "global_gpu_after_backward_mib": global_after_backward,
                "model_bridge_calls": model_bridge.get("bridge_calls", 0.0),
                "solver_bridge_calls": solver_bridge.get("bridge_calls", 0.0),
                "backward_bridge_calls": backward_bridge.get("bridge_calls", 0.0),
                "model_conversion_torch_to_jax_seconds": model_bridge.get("forward_torch_to_jax_seconds", 0.0),
                "model_conversion_jax_compute_seconds": model_bridge.get("forward_jax_compute_seconds", 0.0),
                "model_conversion_jax_to_torch_seconds": model_bridge.get("forward_jax_to_torch_seconds", 0.0),
                "solver_conversion_torch_to_jax_seconds": solver_bridge.get("forward_torch_to_jax_seconds", 0.0),
                "solver_conversion_jax_compute_seconds": solver_bridge.get("forward_jax_compute_seconds", 0.0),
                "solver_conversion_jax_to_torch_seconds": solver_bridge.get("forward_jax_to_torch_seconds", 0.0),
                "conversion_forward_torch_to_jax_seconds": conversion_forward_torch_to_jax,
                "conversion_forward_jax_to_torch_seconds": conversion_forward_jax_to_torch,
                "conversion_backward_torch_to_jax_seconds": conversion_backward_torch_to_jax,
                "conversion_backward_jax_vjp_seconds": backward_bridge.get("backward_jax_vjp_seconds", 0.0),
                "conversion_backward_jax_to_torch_seconds": backward_bridge.get("backward_jax_to_torch_seconds", 0.0),
        }
        rows.append(row)
        update_step_progress(progress, row)
        del model_out, solver_out, loss
        if step < args.steps:
            del grad
        cleanup_runtime_memory(torch)

    close = getattr(progress, "close", None)
    if close is not None:
        close()
    progress_write(
        "[attack-done] "
        f"case={case} combo={combo} total_seconds={time.perf_counter() - total_start:.3f} "
        f"final_loss={rows[-1]['loss']:.6e}"
    )
    return AttackResult(combo, rows, input_history, model_history, solver_history, grad_history, summarize_attack(rows, time.perf_counter() - total_start))


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
    norm = args.ns_norm if case == "ns_2d" else args.burgers_norm
    jax_solver_dtype = args.ns_jax_solver_dtype if case == "ns_2d" else args.burgers_jax_solver_dtype
    x0 = jnp.asarray(x0_np[None, ...], dtype=jnp.float32)

    @jax.jit
    def model_loss_solver(x):
        model_out = jax_model(x)
        solver_out = jax_solver(x)
        loss = jnp.mean((model_out - solver_out) ** 2)
        return loss, (model_out, solver_out)

    value_and_grad = jax.jit(jax.value_and_grad(lambda z: model_loss_solver(z)[0]))

    @jax.jit
    def update(x, grad):
        delta = x - x0
        direction = normalized_update_jax(grad, norm)
        return x0 + project_delta_jax(delta + alpha * direction, epsilon, norm)

    x = x0
    rows: list[dict[str, Any]] = []
    input_history: list[np.ndarray] = []
    model_history: list[np.ndarray] = []
    solver_history: list[np.ndarray] = []
    grad_history: list[np.ndarray] = []
    total_start = time.perf_counter()
    progress_write(
        "[attack-start] "
        f"case={case} combo={combo} solver=jax model=jax steps={args.steps} "
        f"norm={norm} epsilon={epsilon} alpha={alpha} jax_solver_dtype={jax_solver_dtype}"
    )

    progress = make_step_progress(args, case=case, combo=combo, norm=norm, epsilon=epsilon, alpha=alpha)
    for step in progress:
        cleanup_runtime_memory()
        step_start = time.perf_counter()
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
        current_delta = x - x0
        delta_norm_l2 = float(jnp.linalg.norm(current_delta))
        delta_norm_linf = float(jnp.max(jnp.abs(current_delta)))

        value_grad_seconds = 0.0
        backward_seconds = 0.0
        projection_seconds = 0.0
        grad_norm = float("nan")
        global_after_backward = float("nan")
        backward_peak_global = float("nan")
        projection_peak_global = float("nan")
        loss_value = float(loss_forward)
        if step < args.steps:
            with PhaseGpuSampler(args.gpu_sample_interval) as backward_mem:
                vg_start = time.perf_counter()
                loss_vg, grad = value_and_grad(x)
                grad.block_until_ready()
                value_grad_seconds = time.perf_counter() - vg_start
            backward_peak_global = backward_mem.peak_mib
            # JAX value_and_grad evaluates the loss and gradient together, so this
            # is a combined loss+backward timing. Keep it explicit instead of
            # subtracting a separate forward call, which would mix two executions.
            backward_seconds = value_grad_seconds
            global_after_backward = global_gpu_used_mib()
            grad_history.append(np.asarray(grad[0]).copy())
            grad_norm = float(jnp.linalg.norm(grad))
            with PhaseGpuSampler(args.gpu_sample_interval) as projection_mem:
                projection_start = time.perf_counter()
                x = update(x, grad)
                x.block_until_ready()
                projection_seconds = time.perf_counter() - projection_start
            projection_peak_global = projection_mem.peak_mib
            loss_value = float(loss_vg)

        step_seconds = time.perf_counter() - step_start
        phase_peak_global = max(
            [
                value
                for value in [
                    model_peak_global,
                    solver_peak_global,
                    loss_peak_global,
                    backward_peak_global,
                    projection_peak_global,
                ]
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ],
            default=float("nan"),
        )

        row = {
                "case": case,
                "combo": combo,
                "solver_framework": "jax",
                "model_framework": "jax",
                "step": step,
                "loss": loss_value,
                "attack_loss": loss_value,
                "true_loss": loss_value,
                "norm": norm,
                "epsilon": epsilon,
                "alpha": alpha,
                "torch_solver_dtype": "",
                "jax_solver_dtype": jax_solver_dtype,
                "total_pgd_step_seconds": step_seconds,
                "forward_total_seconds": forward_seconds,
                "model_forward_seconds": model_seconds,
                "solver_forward_seconds": solver_seconds,
                "loss_seconds": loss_seconds,
                "backward_seconds": backward_seconds,
                "backward_includes_forward": True,
                "jax_value_and_grad_seconds": value_grad_seconds,
                "projection_seconds": projection_seconds,
                "delta_l2": delta_norm_l2,
                "delta_linf": delta_norm_linf,
                "grad_norm": grad_norm,
                "forward_peak_gpu_mib": max(model_peak_global, solver_peak_global, loss_peak_global),
                "backward_peak_gpu_mib": backward_peak_global,
                "phase_peak_global_gpu_mib": phase_peak_global,
                "torch_cuda_allocated_mib": float("nan"),
                "torch_cuda_reserved_mib": float("nan"),
                "torch_cuda_peak_allocated_mib": float("nan"),
                "model_forward_peak_global_gpu_mib": model_peak_global,
                "solver_forward_peak_global_gpu_mib": solver_peak_global,
                "loss_peak_global_gpu_mib": loss_peak_global,
                "backward_peak_global_gpu_mib": backward_peak_global,
                "projection_peak_global_gpu_mib": projection_peak_global,
                "global_gpu_before_mib": global_before,
                "global_gpu_after_forward_mib": global_after_forward,
                "global_gpu_after_backward_mib": global_after_backward,
                "conversion_forward_torch_to_jax_seconds": 0.0,
                "conversion_forward_jax_to_torch_seconds": 0.0,
                "conversion_backward_torch_to_jax_seconds": 0.0,
                "conversion_backward_jax_vjp_seconds": 0.0,
                "conversion_backward_jax_to_torch_seconds": 0.0,
        }
        rows.append(row)
        update_step_progress(progress, row)
        del model_out, solver_out, loss_forward
        if step < args.steps:
            del grad, loss_vg
        cleanup_runtime_memory()

    close = getattr(progress, "close", None)
    if close is not None:
        close()
    progress_write(
        "[attack-done] "
        f"case={case} combo={combo} total_seconds={time.perf_counter() - total_start:.3f} "
        f"final_loss={rows[-1]['loss']:.6e}"
    )
    return AttackResult(combo, rows, input_history, model_history, solver_history, grad_history, summarize_attack(rows, time.perf_counter() - total_start))


def plot_loss_curves(case_dir: Path, case: str, results: dict[str, AttackResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = case_dir / "loss_progression_all_combos.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "torch_solver_torch_model": ("tab:blue", "-"),
        "torch_solver_jax_model": ("tab:orange", "--"),
        "jax_solver_torch_model": ("tab:green", "-."),
        "jax_solver_jax_model": ("tab:red", ":"),
    }
    for combo in _combo_order(results):
        result = results[combo]
        color, linestyle = styles.get(combo, (None, "-"))
        ax.plot(
            [r["step"] for r in result.rows],
            [r["loss"] for r in result.rows],
            label=combo,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker="o",
            markersize=2.2,
            markevery=max(1, len(result.rows) // 12),
            alpha=0.9,
        )
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


def plot_loss_small_multiples(case_dir: Path, case: str, results: dict[str, AttackResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = case_dir / "loss_progression_small_multiples.png"
    combos = _combo_order(results)
    ncols = 2 if len(combos) > 1 else 1
    nrows = int(math.ceil(len(combos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows), squeeze=False, sharex=True)
    axes_flat = axes.ravel()
    for ax in axes_flat:
        ax.set_visible(False)
    for ax, combo in zip(axes_flat, combos):
        ax.set_visible(True)
        rows = results[combo].rows
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_title(combo, fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_ylabel("attack loss")
    for ax in axes[-1, :]:
        ax.set_xlabel("PGD step")
    fig.suptitle(f"{case} attack loss per combo", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _combo_order(results: dict[str, AttackResult]) -> list[str]:
    ordered = [combo for combo in COMBOS if combo in results]
    ordered.extend(combo for combo in results if combo not in ordered)
    return ordered


def _reference_combo(results: dict[str, AttackResult]) -> str:
    return "torch_solver_torch_model" if "torch_solver_torch_model" in results else _combo_order(results)[0]


def _flat_norms(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(diff)), float(np.max(np.abs(diff))) if diff.size else 0.0


def add_combo_difference_metrics(results: dict[str, AttackResult]) -> None:
    if not results:
        return
    ref_combo = _reference_combo(results)
    ref = results[ref_combo]
    for combo, result in results.items():
        steps = min(len(result.rows), len(ref.rows))
        for step in range(steps):
            input_l2, input_linf = _flat_norms(result.input_history[step], ref.input_history[step])
            delta = np.asarray(result.input_history[step], dtype=np.float64) - np.asarray(result.input_history[0], dtype=np.float64)
            ref_delta = np.asarray(ref.input_history[step], dtype=np.float64) - np.asarray(ref.input_history[0], dtype=np.float64)
            delta_l2, delta_linf = _flat_norms(delta, ref_delta)
            own_delta_l2 = float(np.linalg.norm(delta.reshape(-1)))
            own_delta_linf = float(np.max(np.abs(delta))) if delta.size else 0.0
            model_l2, model_linf = _flat_norms(result.model_history[step], ref.model_history[step])
            solver_l2, solver_linf = _flat_norms(result.solver_history[step], ref.solver_history[step])
            model_solver_l2, model_solver_linf = _flat_norms(result.model_history[step], result.solver_history[step])
            grad_l2 = grad_linf = float("nan")
            if step < len(result.grad_history) and step < len(ref.grad_history):
                grad_l2, grad_linf = _flat_norms(result.grad_history[step], ref.grad_history[step])
            result.rows[step].update(
                {
                    "reference_combo": ref_combo,
                    "input_l2_vs_reference": input_l2,
                    "input_linf_vs_reference": input_linf,
                    "delta_l2": own_delta_l2,
                    "delta_linf": own_delta_linf,
                    "delta_l2_vs_reference": delta_l2,
                    "delta_linf_vs_reference": delta_linf,
                    "grad_l2_vs_reference": grad_l2,
                    "grad_linf_vs_reference": grad_linf,
                    "model_l2_vs_reference": model_l2,
                    "model_linf_vs_reference": model_linf,
                    "solver_l2_vs_reference": solver_l2,
                    "solver_linf_vs_reference": solver_linf,
                    "model_solver_l2": model_solver_l2,
                    "model_solver_linf": model_solver_linf,
                }
            )


def plot_difference_curves(case_dir: Path, case: str, results: dict[str, AttackResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = case_dir / "combo_divergence_vs_reference.png"
    ref_combo = _reference_combo(results)
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    metrics = [
        ("input_l2_vs_reference", "adv input L2 vs reference"),
        ("delta_l2_vs_reference", "delta L2 vs reference"),
        ("grad_l2_vs_reference", "gradient L2 vs reference"),
        ("model_l2_vs_reference", "model output L2 vs reference"),
        ("solver_l2_vs_reference", "solver output L2 vs reference"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        for combo in _combo_order(results):
            rows = results[combo].rows
            ax.plot([r["step"] for r in rows], [r.get(key, 0.0) for r in rows], label=combo)
        ax.set_ylabel(ylabel)
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("PGD step")
    axes[0].set_title(f"{case} combo divergence; reference={ref_combo}")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_runtime_memory_curves(case_dir: Path, case: str, results: dict[str, AttackResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = case_dir / "runtime_memory_all_combos.png"
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    metrics = [
        ("total_pgd_step_seconds", "total step time (s)"),
        ("backward_seconds", "backward / value_and_grad time (s)"),
        ("phase_peak_global_gpu_mib", "phase peak global GPU MiB"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        for combo in _combo_order(results):
            rows = results[combo].rows
            ax.plot([r["step"] for r in rows], [r.get(key, float("nan")) for r in rows], label=combo)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("PGD step")
    axes[0].set_title(f"{case} runtime and memory by PGD step")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_conversion_curves(case_dir: Path, case: str, results: dict[str, AttackResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = case_dir / "conversion_overhead_all_combos.png"
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    metrics = [
        ("conversion_forward_torch_to_jax_seconds", "forward torch->jax (s)"),
        ("conversion_forward_jax_to_torch_seconds", "forward jax->torch (s)"),
        ("conversion_backward_torch_to_jax_seconds", "backward torch->jax (s)"),
        ("conversion_backward_jax_to_torch_seconds", "backward jax->torch (s)"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        for combo in _combo_order(results):
            rows = results[combo].rows
            ax.plot([r["step"] for r in rows], [r.get(key, 0.0) for r in rows], label=combo)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("PGD step")
    axes[0].set_title(f"{case} tensor conversion overhead")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _clean_temp_frame_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _save_gif_from_frames(paths: list[Path], gif_path: Path, duration_ms: int = 250) -> Path:
    from PIL import Image

    if not paths:
        raise ValueError(f"No frames were generated for {gif_path}")
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in paths]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    for image in images:
        image.close()
    return gif_path


def make_gif(case_dir: Path, case: str, results: dict[str, AttackResult], max_frames: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir = case_dir / ".attack_gif_frames_tmp"
    _clean_temp_frame_dir(vis_dir)
    combos = _combo_order(results)
    steps = min(len(results[combo].rows) for combo in combos)
    frame_ids = np.unique(np.linspace(0, steps - 1, min(max_frames, steps), dtype=int))
    paths: list[Path] = []

    for frame_id in frame_ids:
        nrows = len(combos) + 2
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.7 * nrows), squeeze=False)
        ref_combo = _reference_combo(results)
        ref_result = results[ref_combo]
        colors = {
            "torch_solver_torch_model": "tab:blue",
            "torch_solver_jax_model": "tab:orange",
            "jax_solver_torch_model": "tab:green",
            "jax_solver_jax_model": "tab:red",
        }

        if case == "burgers_1d":
            all_initial = [np.asarray(results[c].input_history[0])[..., 0] for c in combos]
            all_adv = [np.asarray(results[c].input_history[frame_id])[..., 0] for c in combos]
            all_delta = [adv - init for init, adv in zip(all_initial, all_adv)]
            all_final_base = [np.asarray(results[c].solver_history[0])[..., 0] for c in combos]
            all_final_adv = [np.asarray(results[c].solver_history[frame_id])[..., 0] for c in combos]
            y_min = min(float(np.min(v)) for v in all_initial + all_adv) - 0.08
            y_max = max(float(np.max(v)) for v in all_initial + all_adv) + 0.08
            f_min = min(float(np.min(v)) for v in all_final_base + all_final_adv) - 0.08
            f_max = max(float(np.max(v)) for v in all_final_base + all_final_adv) + 0.08
            d_abs = max(max(float(np.max(np.abs(v))) for v in all_delta), 1e-12)
            x = np.arange(all_initial[0].shape[0])

            for row_idx, combo in enumerate(combos):
                result = results[combo]
                row = result.rows[frame_id]
                initial = np.asarray(result.input_history[0])[..., 0]
                adv = np.asarray(result.input_history[frame_id])[..., 0]
                delta = adv - initial
                final_base = np.asarray(result.solver_history[0])[..., 0]
                final_adv = np.asarray(result.solver_history[frame_id])[..., 0]

                ax = axes[row_idx, 0]
                ax.plot(x, initial, color="0.55", linewidth=0.9, linestyle=":", label="before perturb")
                ax.plot(x, adv, color=colors.get(combo), linewidth=1.0, label="after perturb")
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel("input")
                ax.set_title(f"{combo}\ninitial condition before/after perturb\nstep={frame_id}", fontsize=8)
                ax.grid(alpha=0.18)
                ax.legend(fontsize=6, loc="best")

                ax = axes[row_idx, 1]
                ax.plot(x, final_base, color="0.55", linewidth=0.9, linestyle=":", label="solver final before perturb")
                ax.plot(x, final_adv, color=colors.get(combo), linewidth=1.0, label="solver final after perturb")
                ax.set_ylim(f_min, f_max)
                ax.set_ylabel("final")
                ax.set_title(f"{combo}\nfinal condition before/after perturb\nloss={row['loss']:.3e}", fontsize=8)
                ax.grid(alpha=0.18)
                ax.legend(fontsize=6, loc="best")

                ax = axes[row_idx, 2]
                ax.plot(x, delta, color=colors.get(combo), linewidth=1.0)
                ax.set_ylim(-d_abs, d_abs)
                ax.set_ylabel("delta")
                ax.set_title(f"{combo}\ndelta = x_adv - x0", fontsize=8)
                ax.grid(alpha=0.18)

            overlay_row = len(combos)
            for combo in combos:
                color = colors.get(combo)
                initial = np.asarray(results[combo].input_history[0])[..., 0]
                adv = np.asarray(results[combo].input_history[frame_id])[..., 0]
                delta = adv - initial
                final_adv = np.asarray(results[combo].solver_history[frame_id])[..., 0]
                axes[overlay_row, 0].plot(x, adv, label=combo, color=color, linewidth=0.9)
                axes[overlay_row, 1].plot(x, final_adv, label=combo, color=color, linewidth=0.9)
                axes[overlay_row, 2].plot(x, delta, label=combo, color=color, linewidth=0.9)
            axes[overlay_row, 0].plot(x, all_initial[0], label="shared before perturb", color="0.35", linestyle=":", linewidth=1.0)
            axes[overlay_row, 1].plot(x, all_final_base[0], label="reference final before perturb", color="0.35", linestyle=":", linewidth=1.0)
            for col_idx, title in enumerate(["all perturbed initial conditions", "all perturbed final conditions", "all deltas"]):
                axes[overlay_row, col_idx].set_title(title, fontsize=8)
                axes[overlay_row, col_idx].grid(alpha=0.18)
            axes[overlay_row, 0].set_ylim(y_min, y_max)
            axes[overlay_row, 1].set_ylim(f_min, f_max)
            axes[overlay_row, 2].set_ylim(-d_abs, d_abs)
            axes[overlay_row, 2].legend(fontsize=6, loc="best")

            diff_row = len(combos) + 1
            ref_initial = np.asarray(ref_result.input_history[0])[..., 0]
            ref_adv = np.asarray(ref_result.input_history[frame_id])[..., 0]
            ref_delta = ref_adv - ref_initial
            ref_final_adv = np.asarray(ref_result.solver_history[frame_id])[..., 0]
            diff_abs = 1e-12
            diffs_by_combo = {}
            for combo in combos:
                initial = np.asarray(results[combo].input_history[0])[..., 0]
                adv = np.asarray(results[combo].input_history[frame_id])[..., 0]
                delta = adv - initial
                final_adv = np.asarray(results[combo].solver_history[frame_id])[..., 0]
                diffs = (adv - ref_adv, final_adv - ref_final_adv, delta - ref_delta)
                diffs_by_combo[combo] = diffs
                diff_abs = max(diff_abs, *(float(np.max(np.abs(v))) for v in diffs))
            for combo, diffs in diffs_by_combo.items():
                color = colors.get(combo)
                for col_idx, values in enumerate(diffs):
                    axes[diff_row, col_idx].plot(x, values, label=combo, color=color, linewidth=0.9)
            for col_idx, title in enumerate(
                [
                    f"perturbed initial diff vs {ref_combo}",
                    f"perturbed final diff vs {ref_combo}",
                    f"delta diff vs {ref_combo}",
                ]
            ):
                axes[diff_row, col_idx].set_title(title, fontsize=8)
                axes[diff_row, col_idx].set_ylim(-diff_abs, diff_abs)
                axes[diff_row, col_idx].grid(alpha=0.18)
                axes[diff_row, col_idx].set_xlabel("x index")
            axes[diff_row, 2].legend(fontsize=6, loc="best")
        else:
            nrows = len(combos)
            ncols = 8
            plt.close(fig)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.75 * nrows), squeeze=False)
            base_inputs = [np.asarray(results[c].input_history[0])[..., -1] for c in combos]
            adv_inputs = [np.asarray(results[c].input_history[frame_id])[..., -1] for c in combos]
            input_deltas = [adv - base for base, adv in zip(base_inputs, adv_inputs)]
            base_solver = [np.asarray(results[c].solver_history[0])[..., -1] for c in combos]
            adv_solver = [np.asarray(results[c].solver_history[frame_id])[..., -1] for c in combos]
            solver_deltas = [adv - base for base, adv in zip(base_solver, adv_solver)]
            base_model = [np.asarray(results[c].model_history[0])[..., -1] for c in combos]
            adv_model = [np.asarray(results[c].model_history[frame_id])[..., -1] for c in combos]
            model_deltas = [adv - base for base, adv in zip(base_model, adv_model)]
            model_solver_diffs = [
                np.asarray(results[c].model_history[frame_id])[..., -1]
                - np.asarray(results[c].solver_history[frame_id])[..., -1]
                for c in combos
            ]

            input_vmin = min(float(np.min(v)) for v in base_inputs + adv_inputs)
            input_vmax = max(float(np.max(v)) for v in base_inputs + adv_inputs)
            solver_vmin = min(float(np.min(v)) for v in base_solver + adv_solver)
            solver_vmax = max(float(np.max(v)) for v in base_solver + adv_solver)
            input_delta_abs = max(1e-12, *(float(np.max(np.abs(v))) for v in input_deltas))
            solver_delta_abs = max(1e-12, *(float(np.max(np.abs(v))) for v in solver_deltas))
            model_delta_abs = max(1e-12, *(float(np.max(np.abs(v))) for v in model_deltas))
            model_solver_abs = max(1e-12, *(float(np.max(np.abs(v))) for v in model_solver_diffs))
            for row_idx, combo in enumerate(combos):
                result = results[combo]
                row = result.rows[frame_id]
                panels = [
                    (base_inputs[row_idx], "input x0 last frame", "viridis", input_vmin, input_vmax),
                    (adv_inputs[row_idx], "input x_adv last frame", "viridis", input_vmin, input_vmax),
                    (input_deltas[row_idx], "input perturbation", "coolwarm", -input_delta_abs, input_delta_abs),
                    (base_solver[row_idx], "solver final from x0", "viridis", solver_vmin, solver_vmax),
                    (adv_solver[row_idx], "solver final from x_adv", "viridis", solver_vmin, solver_vmax),
                    (solver_deltas[row_idx], "solver output change", "coolwarm", -solver_delta_abs, solver_delta_abs),
                    (model_deltas[row_idx], "model output change", "coolwarm", -model_delta_abs, model_delta_abs),
                    (
                        model_solver_diffs[row_idx],
                        "model - solver at x_adv",
                        "coolwarm",
                        -model_solver_abs,
                        model_solver_abs,
                    ),
                ]
                for col_idx, (values, title, cmap, vmin, vmax) in enumerate(panels):
                    ax = axes[row_idx, col_idx]
                    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f"{combo}\n{title}\nstep={frame_id}, loss={row['loss']:.3e}", fontsize=7)
                    ax.set_xticks([])
                    ax.set_yticks([])

        fig.suptitle(f"{case}: PGD attack comparison, step {frame_id}", fontsize=12)
        fig.tight_layout()
        path = vis_dir / f"{case}_attack_step_{frame_id:04d}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)

    gif_path = case_dir / f"{case}_attack_all_combos.gif"
    _save_gif_from_frames(paths, gif_path)
    shutil.copyfile(paths[0], case_dir / f"{case}_attack_first_frame.png")
    shutil.copyfile(paths[-1], case_dir / f"{case}_attack_final_frame.png")
    shutil.rmtree(vis_dir, ignore_errors=True)
    return gif_path


def run_case(args, case: str) -> dict[str, AttackResult]:
    import torch

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"))
    if case == "burgers_1d":
        x0_np = load_burgers_sample(args.burgers_test_path, args.sample_index)
        burgers_torch_checkpoint = args.burgers_torch_checkpoint
        if args.burgers_model_weight_source == "jax_real_imag":
            burgers_torch_checkpoint = DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "jax_real_imag_as_pytorch_fno1d_500.pt"
            if not burgers_torch_checkpoint.exists():
                raise FileNotFoundError(
                    f"Requested --burgers-model-weight-source jax_real_imag, but converted checkpoint is missing: "
                    f"{burgers_torch_checkpoint}"
                )
            print(
                "[burgers_1d] using same JAX real/imag weights for model backends: "
                f"torch_model_checkpoint={rel(burgers_torch_checkpoint)}, "
                f"jax_model_checkpoint={rel(args.burgers_jax_checkpoint)}",
                flush=True,
            )
        torch_model = load_burgers_torch_model(burgers_torch_checkpoint, device)
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
    shutil.rmtree(case_dir / "frames", ignore_errors=True)
    shutil.rmtree(case_dir / "difference_frames", ignore_errors=True)
    old_difference_gif = case_dir / f"{case}_combo_difference_vs_reference.gif"
    if old_difference_gif.exists():
        old_difference_gif.unlink()
    add_combo_difference_metrics(results)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    for combo, result in results.items():
        combo_dir = case_dir / combo
        combo_dir.mkdir(parents=True, exist_ok=True)
        save_csv(combo_dir / "metrics.csv", result.rows)
        save_json(combo_dir / "summary.json", result.summary)
        np.savez_compressed(
            combo_dir / "histories.npz",
            input_history=np.asarray(result.input_history),
            model_history=np.asarray(result.model_history),
            solver_history=np.asarray(result.solver_history),
            grad_history=np.asarray(result.grad_history),
        )
        all_rows.extend(result.rows)
        summary[combo] = result.summary
        solver_framework, model_framework = COMBOS[combo]
        input_l2_curve = [float(row.get("input_l2_vs_reference", 0.0)) for row in result.rows]
        table_rows.append(
            {
                "PDE Case": "Burgers" if case == "burgers_1d" else "NS / Stokes",
                "Solver": solver_framework.upper(),
                "Model": model_framework.upper(),
                "Avg Step Time": result.summary["avg_step_time_seconds"]["mean"],
                "Peak Memory": result.summary["peak_memory_mib"],
                "Peak Memory GiB": result.summary["peak_memory_mib"] / 1024.0,
                "Final True Loss": result.summary["final_true_loss"],
                "Final Input L2 vs Reference": input_l2_curve[-1] if input_l2_curve else float("nan"),
                "Max Input L2 vs Reference": max(input_l2_curve) if input_l2_curve else float("nan"),
                "Combo": combo,
            }
        )

    save_csv(case_dir / "metrics_all_combos.csv", all_rows)
    save_csv(case_dir / "comparison_table.csv", table_rows)
    loss_png = plot_loss_curves(case_dir, case, results)
    loss_small_multiples_png = plot_loss_small_multiples(case_dir, case, results)
    divergence_png = plot_difference_curves(case_dir, case, results)
    runtime_memory_png = plot_runtime_memory_curves(case_dir, case, results)
    conversion_png = plot_conversion_curves(case_dir, case, results)
    gif_path = make_gif(case_dir, case, results, args.max_gif_frames)
    first_frame_png = case_dir / f"{case}_attack_first_frame.png"
    final_frame_png = case_dir / f"{case}_attack_final_frame.png"
    summary["outputs"] = {
        "metrics_csv": rel(case_dir / "metrics_all_combos.csv"),
        "comparison_table_csv": rel(case_dir / "comparison_table.csv"),
        "loss_png": rel(loss_png),
        "loss_small_multiples_png": rel(loss_small_multiples_png),
        "divergence_png": rel(divergence_png),
        "runtime_memory_png": rel(runtime_memory_png),
        "conversion_png": rel(conversion_png),
        "gif": rel(gif_path),
        "first_frame_png": rel(first_frame_png),
        "final_frame_png": rel(final_frame_png),
    }
    summary["comparison_table"] = table_rows
    save_json(case_dir / "summary_all_combos.json", summary)
    return summary


def load_existing_case_results(args, case: str, combos: list[str]) -> dict[str, AttackResult]:
    case_dir = args.output_root / case
    results: dict[str, AttackResult] = {}
    for combo in combos:
        combo_dir = case_dir / combo
        metrics_path = combo_dir / "metrics.csv"
        histories_path = combo_dir / "histories.npz"
        summary_path = combo_dir / "summary.json"
        if not metrics_path.exists() or not histories_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"Missing existing outputs for {case}/{combo}. Expected metrics.csv, histories.npz, summary.json."
            )
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key, value in list(row.items()):
                if value == "":
                    continue
                try:
                    row[key] = float(value)
                    if key == "step":
                        row[key] = int(row[key])
                except ValueError:
                    pass
        histories = np.load(histories_path)
        grad_history = [arr for arr in histories["grad_history"]] if "grad_history" in histories.files else []
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        results[combo] = AttackResult(
            combo=combo,
            rows=rows,
            input_history=[arr for arr in histories["input_history"]],
            model_history=[arr for arr in histories["model_history"]],
            solver_history=[arr for arr in histories["solver_history"]],
            grad_history=grad_history,
            summary=summary,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="burgers_1d,ns_2d", help="Comma list: burgers_1d,ns_2d")
    parser.add_argument("--combos", default="all", help="all or comma list of combo names")
    parser.add_argument("--aggregate-only", action="store_true", help="Read existing per-combo outputs and only rebuild plots/tables.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "benchmark_results" / "attack_framework_matrix_100step")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--max-gif-frames", type=int, default=30)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.02)
    parser.add_argument("--torch-compile-solver", action="store_true", help="Wrap the PyTorch solver function with torch.compile.")
    parser.add_argument("--torch-compile-backend", default="inductor", help="Backend passed to torch.compile for the solver.")
    parser.add_argument(
        "--torch-compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="Mode passed to torch.compile for the solver.",
    )
    parser.add_argument("--torch-compile-fullgraph", action="store_true", help="Require a single full graph for torch.compile.")
    parser.add_argument(
        "--torch-compile-dynamic",
        choices=["true", "false"],
        default=None,
        help="Optional dynamic argument passed to torch.compile.",
    )

    parser.add_argument("--burgers-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--burgers-torch-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--burgers-jax-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "jax_real_imag_fno1d_500.pkl")
    parser.add_argument(
        "--burgers-model-weight-source",
        choices=["native", "jax_real_imag"],
        default="native",
        help=(
            "native uses independently trained PyTorch and JAX checkpoints. "
            "jax_real_imag uses the JAX real/imag checkpoint for native JAX and its converted PyTorch state_dict "
            "for the PyTorch model backend."
        ),
    )
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-torch-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--burgers-norm", choices=["inf", "linf", "infinity", "2", "l2"], default="inf")
    parser.add_argument("--burgers-epsilon", type=float, default=0.5)
    parser.add_argument("--burgers-alpha", type=float, default=0.05)

    parser.add_argument("--ns-test-path", type=Path, default=DEFAULT_NS_TEST)
    parser.add_argument("--ns-torch-checkpoint", type=Path, default=DEFAULT_NS_RUN / "checkpoints" / "fno2d_pytorch.pt")
    parser.add_argument("--ns-jax-checkpoint", type=Path, default=DEFAULT_NS_RUN / "checkpoints" / "fno2d_jax_real_imag.pkl")
    parser.add_argument("--ns-nx", type=int, default=256)
    parser.add_argument("--ns-t-in", type=int, default=10)
    parser.add_argument("--ns-t-out", type=int, default=10)
    parser.add_argument("--ns-nu", type=float, default=1e-5)
    parser.add_argument("--ns-dt", type=float, default=0.005)
    parser.add_argument("--ns-domain", type=float, default=1.0)
    parser.add_argument("--ns-torch-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--ns-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--ns-norm", choices=["inf", "linf", "infinity", "2", "l2"], default="inf")
    parser.add_argument("--ns-epsilon", type=float, default=1.0)
    parser.add_argument("--ns-alpha", type=float, default=0.01)
    args = parser.parse_args()
    if args.torch_compile_dynamic is not None:
        args.torch_compile_dynamic = args.torch_compile_dynamic == "true"
    return args


def selected_combos(args: argparse.Namespace) -> list[str]:
    return list(COMBOS) if args.combos == "all" else [c.strip() for c in args.combos.split(",") if c.strip()]


def run_combos_in_clean_subprocesses(args: argparse.Namespace, cases: list[str], combos: list[str]) -> None:
    base_argv = sys.argv[1:]
    filtered: list[str] = []
    skip_next = False
    for item in base_argv:
        if skip_next:
            skip_next = False
            continue
        if item in {"--combos", "--cases"}:
            skip_next = True
            continue
        if item.startswith("--combos=") or item.startswith("--cases="):
            continue
        if item == "--aggregate-only":
            continue
        filtered.append(item)

    for case in cases:
        for combo in combos:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *filtered,
                "--cases",
                case,
                "--combos",
                combo,
            ]
            print(f"[clean-run] launching isolated process: case={case} combo={combo}", flush=True)
            subprocess.run(cmd, check=True)

    aggregate_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        *filtered,
        "--cases",
        ",".join(cases),
        "--combos",
        ",".join(combos),
        "--aggregate-only",
    ]
    print("[clean-run] rebuilding combined tables/plots from isolated outputs", flush=True)
    subprocess.run(aggregate_cmd, check=True)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    combos = selected_combos(args)
    if not args.aggregate_only and (len(cases) > 1 or len(combos) > 1):
        run_combos_in_clean_subprocesses(args, cases, combos)
        return

    run_summary: dict[str, Any] = {"config": vars(args), "cases": {}}
    all_table_rows: list[dict[str, Any]] = []
    for case in cases:
        results = load_existing_case_results(args, case, combos) if args.aggregate_only else run_case(args, case)
        run_summary["cases"][case] = write_case_outputs(args, case, results)
        all_table_rows.extend(run_summary["cases"][case].get("comparison_table", []))
    save_csv(args.output_root / "comparison_table_all_cases.csv", all_table_rows)
    run_summary["comparison_table_csv"] = rel(args.output_root / "comparison_table_all_cases.csv")
    save_json(args.output_root / "summary.json", run_summary)
    print(f"[done] wrote {rel(args.output_root)}", flush=True)


if __name__ == "__main__":
    main()
