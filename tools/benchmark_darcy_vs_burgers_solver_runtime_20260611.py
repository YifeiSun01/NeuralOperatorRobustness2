#!/usr/bin/env python3
"""Benchmark Darcy and Burgers solver forward/backward runtimes.

This is a solver-only benchmark: no model forward, optimizer step, evaluation,
or checkpoint I/O.  It separates no-grad forward, graph-building forward, and
backward-only time so we can directly compare the cost of differentiating
through each solver.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")


def _configure_jax_cuda_toolchain() -> None:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return
    cuda_nvcc = Path(purelib) / "nvidia" / "cuda_nvcc"
    if not cuda_nvcc.exists():
        return
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_nvcc}")
    bin_dir = str(cuda_nvcc / "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


_configure_jax_cuda_toolchain()

import jax
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DARCY_ROOT = PROJECT_ROOT / "2D_Darcy_FNO2d"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_burgers_module():
    return load_module("benchmark_project_solvers", PROJECT_ROOT / "solvers.py")


def load_darcy_bridge():
    existing = sys.modules.get("solvers")
    if existing is not None and not hasattr(existing, "__path__"):
        sys.modules.pop("solvers", None)
    if str(DARCY_ROOT) not in sys.path:
        sys.path.insert(0, str(DARCY_ROOT))
    return load_module(
        "benchmark_darcy_binary_bridge",
        DARCY_ROOT / "perturbation_methods" / "attack_darcy_binary.py",
    )


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn: Callable[[], Any]) -> float:
    sync()
    start = time.perf_counter()
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    sync()
    return time.perf_counter() - start


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean_s": float(statistics.mean(values)),
        "median_s": float(statistics.median(values)),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
        "std_s": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def load_darcy_coefficients(path: Path, max_batch: int, device: torch.device) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"].float()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"Expected Darcy coefficient tensor (N,H,W), got {tuple(x.shape)}")
    x = x[:max_batch].contiguous()
    unique = torch.unique(x)
    if unique.numel() > 8:
        raise ValueError(f"Darcy benchmark expects binary-ish coefficients, got {unique.numel()} values")
    return x.to(device)


def make_smooth_burgers_input(
    batch: int,
    n: int,
    device: torch.device,
    *,
    seed: int,
    amplitude: float = 1.0,
    max_mode: int = 8,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + batch * 1009 + n)
    grid = torch.linspace(0.0, 2.0 * math.pi, n + 1, dtype=torch.float32, device=device)[:-1]
    u = torch.zeros((batch, n), dtype=torch.float32, device=device)
    for k in range(1, max_mode + 1):
        scale = amplitude / float(k)
        sin_coeff = torch.randn((batch, 1), generator=generator, dtype=torch.float32).to(device) * scale
        cos_coeff = torch.randn((batch, 1), generator=generator, dtype=torch.float32).to(device) * scale
        u = u + sin_coeff * torch.sin(k * grid)[None, :] + cos_coeff * torch.cos(k * grid)[None, :]
    u = u - u.mean(dim=-1, keepdim=True)
    u = u / (u.std(dim=-1, keepdim=True) + 1e-6)
    return u.contiguous()


def benchmark_darcy(
    *,
    bridge,
    solver,
    coeffs: torch.Tensor,
    batch: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "solver": "darcy",
        "batch": batch,
        "input_shape": list(coeffs[:batch].shape),
        "ok": True,
    }

    def forward_no_grad_once() -> torch.Tensor:
        with torch.no_grad():
            return bridge.JaxDarcySolver.apply(coeffs[:batch].contiguous(), solver)

    def forward_with_grad_once() -> torch.Tensor:
        a = coeffs[:batch].detach().clone().requires_grad_(True)
        return bridge.JaxDarcySolver.apply(a, solver)

    try:
        row["first_forward_no_grad_s"] = timed(forward_no_grad_once)
        a = coeffs[:batch].detach().clone().requires_grad_(True)
        out = bridge.JaxDarcySolver.apply(a, solver)
        sync()
        start = time.perf_counter()
        (out.square().mean()).backward()
        sync()
        row["first_backward_only_s"] = time.perf_counter() - start
        del a, out
        gc.collect()
        torch.cuda.empty_cache()

        for _ in range(warmups):
            _ = forward_no_grad_once()
            a = coeffs[:batch].detach().clone().requires_grad_(True)
            out = bridge.JaxDarcySolver.apply(a, solver)
            (out.square().mean()).backward()
            del a, out
            sync()

        fwd_no_grad: list[float] = []
        fwd_with_grad: list[float] = []
        bwd_only: list[float] = []
        total_fwd_bwd: list[float] = []

        for _ in range(repeats):
            fwd_no_grad.append(timed(forward_no_grad_once))

            a = coeffs[:batch].detach().clone().requires_grad_(True)
            sync()
            start = time.perf_counter()
            out = bridge.JaxDarcySolver.apply(a, solver)
            sync()
            fwd_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            (out.square().mean()).backward()
            sync()
            bwd_elapsed = time.perf_counter() - start

            fwd_with_grad.append(fwd_elapsed)
            bwd_only.append(bwd_elapsed)
            total_fwd_bwd.append(fwd_elapsed + bwd_elapsed)
            del a, out
            gc.collect()

        row.update({f"forward_no_grad_{k}": v for k, v in stats(fwd_no_grad).items()})
        row.update({f"forward_with_grad_{k}": v for k, v in stats(fwd_with_grad).items()})
        row.update({f"backward_only_{k}": v for k, v in stats(bwd_only).items()})
        row.update({f"forward_plus_backward_{k}": v for k, v in stats(total_fwd_bwd).items()})
        row["backward_over_forward_mean"] = row["backward_only_mean_s"] / max(
            row["forward_with_grad_mean_s"], 1e-12
        )
        row["fwd_bwd_over_fwd_no_grad_mean"] = row["forward_plus_backward_mean_s"] / max(
            row["forward_no_grad_mean_s"], 1e-12
        )
    except Exception as exc:
        row["ok"] = False
        row["error"] = repr(exc)
    return row


def benchmark_burgers(
    *,
    burgers_mod,
    batch: int,
    n: int,
    steps: int,
    warmups: int,
    repeats: int,
    device: torch.device,
    seed: int,
    remat_mode: str,
    remat_chunk_steps: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "solver": "burgers",
        "batch": batch,
        "input_shape": [batch, n],
        "steps": steps,
        "dt": 0.001,
        "ok": True,
    }
    solver = burgers_mod.Burgers1DETDRK4(
        num_points=n,
        domain_extent=2.0,
        dt=0.001,
        diffusivity=1e-3,
        convection_scale=1.0,
        conservative=False,
        dealiasing_fraction=2.0 / 3.0,
        device=device,
        dtype=torch.float32,
    )
    base = make_smooth_burgers_input(batch, n, device, seed=seed)

    def solve(u: torch.Tensor) -> torch.Tensor:
        return burgers_mod.solve_burgers_final_batch_with_solver(
            u,
            solver,
            steps,
            remat_mode=remat_mode,
            remat_chunk_steps=remat_chunk_steps,
        )

    def forward_no_grad_once() -> torch.Tensor:
        with torch.no_grad():
            return solve(base)

    try:
        row["first_forward_no_grad_s"] = timed(forward_no_grad_once)
        a = base.detach().clone().requires_grad_(True)
        out = solve(a)
        sync()
        start = time.perf_counter()
        (out.square().mean()).backward()
        sync()
        row["first_backward_only_s"] = time.perf_counter() - start
        del a, out
        gc.collect()
        torch.cuda.empty_cache()

        for _ in range(warmups):
            _ = forward_no_grad_once()
            a = base.detach().clone().requires_grad_(True)
            out = solve(a)
            (out.square().mean()).backward()
            del a, out
            sync()

        fwd_no_grad: list[float] = []
        fwd_with_grad: list[float] = []
        bwd_only: list[float] = []
        total_fwd_bwd: list[float] = []

        for _ in range(repeats):
            fwd_no_grad.append(timed(forward_no_grad_once))

            a = base.detach().clone().requires_grad_(True)
            sync()
            start = time.perf_counter()
            out = solve(a)
            sync()
            fwd_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            (out.square().mean()).backward()
            sync()
            bwd_elapsed = time.perf_counter() - start

            if not torch.isfinite(out).all():
                raise RuntimeError("Burgers solver produced non-finite output")

            fwd_with_grad.append(fwd_elapsed)
            bwd_only.append(bwd_elapsed)
            total_fwd_bwd.append(fwd_elapsed + bwd_elapsed)
            del a, out
            gc.collect()

        row.update({f"forward_no_grad_{k}": v for k, v in stats(fwd_no_grad).items()})
        row.update({f"forward_with_grad_{k}": v for k, v in stats(fwd_with_grad).items()})
        row.update({f"backward_only_{k}": v for k, v in stats(bwd_only).items()})
        row.update({f"forward_plus_backward_{k}": v for k, v in stats(total_fwd_bwd).items()})
        row["backward_over_forward_mean"] = row["backward_only_mean_s"] / max(
            row["forward_with_grad_mean_s"], 1e-12
        )
        row["fwd_bwd_over_fwd_no_grad_mean"] = row["forward_plus_backward_mean_s"] / max(
            row["forward_no_grad_mean_s"], 1e-12
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        row["ok"] = False
        row["error"] = repr(exc)
    except Exception as exc:
        row["ok"] = False
        row["error"] = repr(exc)
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--darcy-dataset",
        type=Path,
        default=PROJECT_ROOT
        / "2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/train/"
        / "dim2d_darcy_nx85_N1200_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/darcy_vs_burgers_solver_forward_backward_benchmark_20260611")
    parser.add_argument("--darcy-batches", type=str, default="1,4,8,16,32,64")
    parser.add_argument("--burgers-batches", type=str, default="1,4,8")
    parser.add_argument("--burgers-n", type=int, default=1024)
    parser.add_argument("--burgers-steps", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--burgers-remat-mode", type=str, default="none")
    parser.add_argument("--burgers-remat-chunk-steps", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend must be gpu, got {jax.default_backend()!r}")

    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    darcy_batches = parse_ints(args.darcy_batches)
    burgers_batches = parse_ints(args.burgers_batches)
    max_darcy_batch = max(darcy_batches)

    burgers_mod = load_burgers_module()
    bridge = load_darcy_bridge()
    darcy_solver = bridge.make_darcy_solve_fn(tol=1e-5, atol=0.0, maxiter=None)
    darcy_coeffs = load_darcy_coefficients(args.darcy_dataset, max_darcy_batch, device)

    rows: list[dict[str, Any]] = []

    for batch in darcy_batches:
        print(f"[darcy] batch={batch}", flush=True)
        rows.append(
            benchmark_darcy(
                bridge=bridge,
                solver=darcy_solver,
                coeffs=darcy_coeffs,
                batch=batch,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        )
        write_csv(rows, args.output_dir / "solver_runtime_summary.csv")
        (args.output_dir / "solver_runtime_summary.json").write_text(json.dumps(rows, indent=2))

    for batch in burgers_batches:
        print(f"[burgers] batch={batch}", flush=True)
        rows.append(
            benchmark_burgers(
                burgers_mod=burgers_mod,
                batch=batch,
                n=args.burgers_n,
                steps=args.burgers_steps,
                warmups=args.warmups,
                repeats=args.repeats,
                device=device,
                seed=args.seed,
                remat_mode=args.burgers_remat_mode,
                remat_chunk_steps=args.burgers_remat_chunk_steps,
            )
        )
        write_csv(rows, args.output_dir / "solver_runtime_summary.csv")
        (args.output_dir / "solver_runtime_summary.json").write_text(json.dumps(rows, indent=2))

    meta = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_device": torch.cuda.get_device_name(0),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "darcy_dataset": str(args.darcy_dataset),
        "darcy_coeff_unique": [float(x) for x in torch.unique(darcy_coeffs.detach().cpu())],
        "burgers_n": args.burgers_n,
        "burgers_steps": args.burgers_steps,
        "burgers_remat_mode": args.burgers_remat_mode,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"output_dir": str(args.output_dir), "rows": rows, "metadata": meta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
