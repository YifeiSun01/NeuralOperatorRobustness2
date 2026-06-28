#!/usr/bin/env python3
"""Generate 1D Burgers datasets with batched Exponax rollouts.

The original 1D Burgers generator samples GRF initial conditions and solves one
sample at a time. This script keeps the same dataset semantics but uses
`jax.vmap` over chunks of initial conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

import exponax as ex

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
BURGERS_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(BURGERS_ROOT))

from GRFs.generateGRFs import GRFGenerator  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def make_filename(
    *,
    nx: int,
    num_records: int,
    solver_name: str,
    kernel: str,
    correlation_length: float,
    bc: str,
    nu: float,
    t_final: float,
    seed: int,
) -> str:
    return (
        f"dim1d_nx{nx}_N{num_records}_solver={solver_name}_"
        f"kernel={kernel}_correlation_length{correlation_length}_"
        f"bc{bc}_nu{nu}_t{t_final}_seed{seed}.pt"
    )


def sample_grfs(
    *,
    num_records: int,
    nx: int,
    kernel: str,
    correlation_length: float,
    bc: str,
    seed: int,
    zero_mean: bool,
) -> np.ndarray:
    rows = []
    for i in tqdm(range(num_records), desc="Sampling GRFs", unit="sample"):
        rows.append(
            GRFGenerator.generate_grf(
                shape=(nx,),
                kernel=kernel,
                kernel_params={"correlation_length": correlation_length},
                bc=bc,
                seed=seed + i,
                zero_mean=zero_mean,
            ).astype(np.float32, copy=False)
        )
    return np.stack(rows, axis=0)


def make_burgers_final_fn(
    *,
    nx: int,
    nu: float,
    dt: float,
    t_final: float,
    domain_extent: float,
    conservative: bool,
):
    steps = int(round(t_final / dt))
    stepper = ex.stepper.Burgers(
        1,
        domain_extent,
        nx,
        dt,
        diffusivity=nu,
        convection_scale=1.0,
        order=4,
        conservative=conservative,
    )
    repeat_fn = ex.repeat(stepper, steps)

    def one(u0):
        full_ic = jnp.expand_dims(u0.squeeze(), axis=0).astype(jnp.float32)
        return repeat_fn(full_ic)[0]

    return jax.jit(jax.vmap(one, in_axes=0, out_axes=0))


def solve_burgers_batched(
    x_np: np.ndarray,
    *,
    nu: float,
    dt: float,
    t_final: float,
    batch_size: int,
    domain_extent: float,
    conservative: bool,
) -> tuple[np.ndarray, list[dict]]:
    batch_fn = make_burgers_final_fn(
        nx=x_np.shape[1],
        nu=nu,
        dt=dt,
        t_final=t_final,
        domain_extent=domain_extent,
        conservative=conservative,
    )
    outputs = []
    records: list[dict] = []

    for start in tqdm(range(0, x_np.shape[0], batch_size), desc=f"Solving nu={nu}", unit="batch"):
        end = min(start + batch_size, x_np.shape[0])
        chunk = jnp.asarray(x_np[start:end])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_chunk = batch_fn(chunk)
        y_chunk.block_until_ready()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        outputs.append(np.asarray(y_chunk, dtype=np.float32))
        records.append(
            {
                "start": start,
                "end": end,
                "batch_n": end - start,
                "seconds": seconds,
                "seconds_per_sample": seconds / max(1, end - start),
            }
        )

    return np.concatenate(outputs, axis=0), records


def benchmark(x_np: np.ndarray, args: argparse.Namespace) -> None:
    compare_n = min(args.benchmark_samples, x_np.shape[0])
    x_small = x_np[:compare_n]
    results = {}
    reference = None
    reference_batch_size = None

    for batch_size in parse_int_list(args.benchmark_batch_sizes):
        y, records = solve_burgers_batched(
            x_small,
            nu=args.nu_values[0],
            dt=args.step,
            t_final=args.t_final,
            batch_size=batch_size,
            domain_extent=args.domain_extent,
            conservative=args.conservative,
        )
        total_seconds = sum(r["seconds"] for r in records)
        if reference is None:
            reference = y
            reference_batch_size = batch_size
        max_abs_diff = float(np.max(np.abs(y - reference)))
        results[str(batch_size)] = {
            "total_seconds": round(total_seconds, 4),
            "seconds_per_sample": round(total_seconds / compare_n, 4),
            "reference_batch_size": reference_batch_size,
            "max_abs_diff_vs_reference": max_abs_diff,
        }

    print("[benchmark]", json.dumps(results, indent=2, sort_keys=True))


def save_dataset(path: Path, x_np: np.ndarray, y_np: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": torch.from_numpy(x_np.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y_np.astype(np.float32, copy=False)),
            "t_final": metadata["t_final"],
            "metadata": metadata,
        },
        path,
    )
    print(f"[saved] {path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-records", type=int, default=1500)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--nu", dest="nu_raw", default="0.01,0.0005")
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--kernel", default="gaussian")
    parser.add_argument("--correlation-length", type=float, default=0.03)
    parser.add_argument("--bc", default="periodic")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--zero-mean", action="store_true")
    parser.add_argument("--domain-extent", type=float, default=2.0)
    parser.add_argument("--conservative", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=BURGERS_ROOT / "datasets" / "1D" / "Burgers" / "batched_exponax",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-samples", type=int, default=32)
    parser.add_argument("--benchmark-batch-sizes", default="1,8,32,128,256")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.nu_values = parse_float_list(args.nu_raw)

    print(
        json.dumps(
            {
                "project_root": str(PROJECT_ROOT),
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "num_records": args.num_records,
                "batch_size": args.batch_size,
                "nu_values": args.nu_values,
            },
            indent=2,
        )
    )

    x_np = sample_grfs(
        num_records=args.num_records,
        nx=args.nx,
        kernel=args.kernel,
        correlation_length=args.correlation_length,
        bc=args.bc,
        seed=args.seed,
        zero_mean=args.zero_mean,
    )

    if args.benchmark:
        benchmark(x_np, args)

    for nu in args.nu_values:
        filename = make_filename(
            nx=args.nx,
            num_records=args.num_records,
            solver_name="exponax_batched",
            kernel=args.kernel,
            correlation_length=args.correlation_length,
            bc=args.bc,
            nu=nu,
            t_final=args.t_final,
            seed=args.seed,
        )
        save_path = args.save_dir / filename
        if save_path.exists() and not args.overwrite:
            print(f"[skip] {save_path} already exists")
            continue

        y_np, records = solve_burgers_batched(
            x_np,
            nu=nu,
            dt=args.step,
            t_final=args.t_final,
            batch_size=args.batch_size,
            domain_extent=args.domain_extent,
            conservative=args.conservative,
        )
        metadata = {
            "solver": "exponax_batched",
            "num_records": args.num_records,
            "nx": args.nx,
            "nu": nu,
            "t_final": args.t_final,
            "step": args.step,
            "batch_size": args.batch_size,
            "kernel": args.kernel,
            "correlation_length": args.correlation_length,
            "bc": args.bc,
            "seed": args.seed,
            "zero_mean": args.zero_mean,
            "timing_records": records,
        }
        save_dataset(save_path, x_np, y_np, metadata)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
