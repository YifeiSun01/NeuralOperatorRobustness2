#!/usr/bin/env python3
"""Generate Darcy Flow datasets from two-phase GRF coefficients.

This follows the original FNO Darcy setup:

* latent field z ~ N(0, (-Delta + tau^2 I)^(-alpha)), with Neumann-Laplacian
  cosine modes;
* coefficient A(x) is thresholded pointwise, high on z >= 0 and low otherwise;
* U solves -div(A grad U) = 1 with zero Dirichlet boundary.

The solver is the differentiable JAX CG implementation in
``2D_Darcy_FNO2d/solvers/darcy_jax_solver.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
import time
from pathlib import Path
from typing import Iterable


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


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_configure_jax_cuda_toolchain()

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
DARCY_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from solvers.darcy_jax_solver import make_darcy_solve_fn, make_grf_batch_fn, residual_norm


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def split_counts(value: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, count = item.split(":", 1)
        out[name.strip()] = int(count)
    return out


def output_filename(
    *,
    split: str,
    n: int,
    resolution: int,
    solve_resolution: int,
    alpha: float,
    tau: float,
    low: float,
    high: float,
    seed: int,
) -> str:
    return (
        f"dim2d_darcy_nx{resolution}_N{n}_solver=jaxcg_"
        f"solve{solve_resolution}_alpha{alpha:g}_tau{tau:g}_"
        f"binary{low:g}-{high:g}_f1_seed{seed}_{split}.pt"
    )


def downsample_grid(x: torch.Tensor, target_resolution: int) -> torch.Tensor:
    source_resolution = x.shape[-1]
    if target_resolution == source_resolution:
        return x
    numerator = source_resolution - 1
    denominator = target_resolution - 1
    if numerator % denominator != 0:
        raise ValueError(
            f"resolution {target_resolution} is not an integer-grid downsample of {source_resolution}"
        )
    stride = numerator // denominator
    y = x[..., ::stride, ::stride]
    if y.shape[-2:] != (target_resolution, target_resolution):
        raise RuntimeError(f"downsample produced {tuple(y.shape[-2:])}, expected {target_resolution}")
    return y.contiguous()


def solve_batch(
    *,
    sampler,
    solver,
    base_key,
    start_index: int,
    batch_n: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    sample_indices = jnp.arange(start_index, start_index + batch_n, dtype=jnp.uint32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    latent_jax, a_jax = sampler(base_key, sample_indices)
    latent_jax.block_until_ready()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sample_seconds = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    u_jax = solver(a_jax)
    u_jax.block_until_ready()
    res_jax = residual_norm(a_jax, u_jax, forcing_value=1.0)
    res_jax.block_until_ready()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    solve_seconds = time.perf_counter() - t1

    latent = torch.from_numpy(np.asarray(latent_jax, dtype=np.float32).copy()).cpu()
    a = torch.from_numpy(np.asarray(a_jax, dtype=np.float32).copy()).cpu()
    u = torch.from_numpy(np.asarray(u_jax, dtype=np.float32).copy()).cpu()
    residual = np.asarray(res_jax)

    record = {
        "start_index": start_index,
        "end_index": start_index + batch_n,
        "batch_n": batch_n,
        "sample_seconds": sample_seconds,
        "solve_seconds": solve_seconds,
        "solve_seconds_per_sample": solve_seconds / max(1, batch_n),
        "residual_rel_mean": float(np.mean(residual)),
        "residual_rel_max": float(np.max(residual)),
    }
    return latent, a, u, record


def generate_split(
    *,
    split: str,
    count: int,
    global_start_index: int,
    args: argparse.Namespace,
) -> tuple[int, list[dict]]:
    sampler = make_grf_batch_fn(
        nx=args.solve_resolution,
        alpha=args.alpha,
        tau=args.tau,
        low=args.low,
        high=args.high,
        binary=not args.soft_coefficients,
        beta=args.soft_beta,
    )
    solver = make_darcy_solve_fn(tol=args.solver_tol, atol=args.solver_atol, maxiter=args.solver_maxiter)
    base_key = jax.random.PRNGKey(args.seed)

    high_latents = torch.empty((count, args.solve_resolution, args.solve_resolution), dtype=torch.float32)
    high_a = torch.empty_like(high_latents)
    high_u = torch.empty_like(high_latents)
    records: list[dict] = []

    local_write = 0
    for start in tqdm(range(0, count, args.batch_size), desc=f"{split} Darcy batches", unit="batch"):
        batch_n = min(args.batch_size, count - start)
        latent, a, u, record = solve_batch(
            sampler=sampler,
            solver=solver,
            base_key=base_key,
            start_index=global_start_index + start,
            batch_n=batch_n,
            args=args,
        )
        high_latents[local_write : local_write + batch_n].copy_(latent)
        high_a[local_write : local_write + batch_n].copy_(a)
        high_u[local_write : local_write + batch_n].copy_(u)
        record["split"] = split
        records.append(record)
        local_write += batch_n

    split_dir = args.output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for resolution in args.output_resolutions:
        a = downsample_grid(high_a, resolution)
        u = downsample_grid(high_u, resolution)
        latent = downsample_grid(high_latents, resolution) if args.save_latent else None
        path = split_dir / output_filename(
            split=split,
            n=count,
            resolution=resolution,
            solve_resolution=args.solve_resolution,
            alpha=args.alpha,
            tau=args.tau,
            low=args.low,
            high=args.high,
            seed=args.seed,
        )
        if path.exists() and not args.overwrite:
            print(f"[skip-existing] {path}")
            continue
        metadata = {
            "split": split,
            "nsamples": count,
            "resolution": resolution,
            "solve_resolution": args.solve_resolution,
            "source": "FNO Darcy GRF threshold benchmark",
            "pde": "-div(A grad U)=1, U|boundary=0",
            "coefficient": {
                "latent": "GRF(alpha,tau) with Neumann-Laplacian cosine modes",
                "threshold": "A=high if latent>=0 else low",
                "low": args.low,
                "high": args.high,
                "soft_coefficients": args.soft_coefficients,
                "soft_beta": args.soft_beta,
            },
            "alpha": args.alpha,
            "tau": args.tau,
            "seed": args.seed,
            "solver": {
                "name": "jax_matrix_free_cg_second_order_fd",
                "tol": args.solver_tol,
                "atol": args.solver_atol,
                "maxiter": args.solver_maxiter,
            },
            "timing_records": records,
        }
        payload = {"x": a, "y": u, "metadata": metadata}
        if latent is not None:
            payload["latent"] = latent
        torch.save(payload, path)
        print(f"[saved] {path}")

    return global_start_index + count, records


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-counts", default="train:1000,test:100")
    parser.add_argument("--solve-resolution", type=int, default=421)
    parser.add_argument("--output-resolutions", default="85,141,211,421")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--low", type=float, default=3.0)
    parser.add_argument("--high", type=float, default=12.0)
    parser.add_argument("--soft-coefficients", action="store_true")
    parser.add_argument("--soft-beta", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--solver-maxiter", type=int, default=None)
    parser.add_argument("--save-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DARCY_ROOT / "datasets" / "grf_darcy",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.output_dir = args.output_dir.resolve()
    args.output_resolutions = parse_int_list(args.output_resolutions)

    print(
        json.dumps(
            {
                "project_root": str(PROJECT_ROOT),
                "darcy_root": str(DARCY_ROOT),
                "output_dir": str(args.output_dir),
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
                "torch_cuda_available": torch.cuda.is_available(),
                "solve_resolution": args.solve_resolution,
                "output_resolutions": args.output_resolutions,
                "batch_size": args.batch_size,
            },
            indent=2,
        )
    )

    all_records: list[dict] = []
    next_index = 0
    for split, count in split_counts(args.split_counts).items():
        next_index, records = generate_split(
            split=split,
            count=count,
            global_start_index=next_index,
            args=args,
        )
        all_records.extend(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
