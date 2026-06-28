#!/usr/bin/env python3
"""Generate 2D Navier-Stokes datasets from real initial conditions in batches.

Input initial conditions come from
`datasets/source_zongyi_real_initial/NS_data_zongyi_{split}_all_frame_real_initial.pt`.
Only the `x` tensor is used. It is spectrally upsampled to 256x256 and then
rolled out with the modified Exponax Navier-Stokes stepper.

Each split is saved as one complete `.pt` file. `--batch-size` controls how many
source samples are processed before copying results back to CPU, and
`--solver-batch-size` controls how many samples enter one JAX/Exponax rollout.
"""

from __future__ import annotations

import argparse
import json
import math
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
NS_ROOT = THIS_FILE.parents[1]
DEFAULT_SOURCE_DIR = NS_ROOT / "datasets" / "source_zongyi_real_initial"
DEFAULT_OUTPUT_DIR = NS_ROOT / "datasets" / "exponax_datasets" / "t20" / "real_initial_laxmap_single"


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_split_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def torch_to_jax(tensor: torch.Tensor):
    return jax.dlpack.from_dlpack(tensor.contiguous())


def jax_to_torch(array) -> torch.Tensor:
    return torch.from_dlpack(array)


def spectral_upsample(field: torch.Tensor, target_size: int = 256) -> torch.Tensor:
    *batch_dims, height, width = field.shape
    if target_size < height or target_size < width:
        raise ValueError("target_size must be no smaller than the input size")

    freq = torch.fft.fft2(field, norm="ortho")
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

    pad_h = (target_size - height) // 2
    pad_w = (target_size - width) // 2
    expanded = torch.zeros(
        *batch_dims,
        target_size,
        target_size,
        dtype=freq_shifted.dtype,
        device=freq_shifted.device,
    )
    expanded[..., pad_h : pad_h + height, pad_w : pad_w + width] = freq_shifted
    upsampled = torch.fft.ifft2(torch.fft.ifftshift(expanded, dim=(-2, -1)), norm="ortho")
    return (target_size / height) * upsampled.real


def make_ns_rollout_fn(
    *,
    nx: int,
    nu: float,
    t_final: int,
    fixed_step: float,
    domain_extent: float,
    solver_mode: str = "lax-map",
):
    steps_per_second = int(round(1.0 / fixed_step))
    if not math.isclose(steps_per_second * fixed_step, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("fixed_step must divide one second exactly for sparse integer-second saves")

    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2,
        domain_extent,
        nx,
        fixed_step,
        diffusivity=nu,
        order=4,
        num_circle_points=16,
        dealiasing_fraction=2 / 3,
    )

    def one(u0):
        u = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))

        def micro(carry, _):
            return stepper(carry[None, ...])[0], None

        def one_second(carry, _):
            u_next, _ = jax.lax.scan(micro, carry, None, length=steps_per_second)
            return u_next, u_next

        _, seconds = jax.lax.scan(one_second, u, None, length=t_final)
        seq = jnp.concatenate([u[None, ...], seconds], axis=0)
        return jnp.swapaxes(seq, -1, -2)

    def batch(u0_batch):
        if solver_mode == "vmap":
            seq = jax.vmap(one, in_axes=0, out_axes=0)(u0_batch)
        elif solver_mode == "lax-map":
            seq = jax.lax.map(one, u0_batch)
        else:
            raise ValueError(f"unknown solver_mode={solver_mode!r}")
        return jnp.moveaxis(seq, 1, -1)

    return jax.jit(batch)


def source_path(source_dir: Path, split: str, suffix: str) -> Path:
    return source_dir / f"NS_data_zongyi_{split}_all_frame{suffix}.pt"


def output_filename(
    *,
    split: str,
    n: int,
    batch_index: int,
    nx: int,
    nu: float,
    t_final: int,
    ntimepoints: int,
) -> str:
    return (
        f"dim2d_nx{nx}_N{n}_solver=exponax_nu{nu:.3f}_"
        f"t{float(t_final):.1f}_{split}_ntimepoints{ntimepoints}_"
        f"batch{batch_index}_all_frames.pt"
    )


def single_output_filename(
    *,
    split: str,
    n: int,
    nx: int,
    nu: float,
    t_final: int,
    ntimepoints: int,
) -> str:
    return (
        f"dim2d_nx{nx}_N{n}_solver=exponax_nu{nu:.3f}_"
        f"t{float(t_final):.1f}_{split}_ntimepoints{ntimepoints}_"
        "all_frames.pt"
    )


def load_source_x(source_dir: Path, split: str, suffix: str) -> torch.Tensor:
    path = source_path(source_dir, split, suffix)
    print(f"[load] {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "x" not in data:
        raise KeyError(f"{path} does not contain key 'x'")
    x = data["x"].float().contiguous()
    print({"split": split, "source_x_shape": tuple(x.shape), "source_x_dtype": str(x.dtype)})
    return x


def generate_split(args: argparse.Namespace, split: str) -> list[dict]:
    source_x = load_source_x(args.source_dir, split, args.filename_suffix)
    total_available = source_x.shape[0]
    default_n = total_available
    requested_n = args.nsamples if args.nsamples is not None else default_n
    start = args.start
    end = min(start + requested_n, total_available)
    if start >= end:
        raise ValueError(f"empty range for split={split}: start={start}, requested_n={requested_n}")

    output_dir = args.output_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_n = end - start
    out_path = output_dir / single_output_filename(
        split=split,
        n=selected_n,
        nx=args.target_size,
        nu=args.nu,
        t_final=args.t_final,
        ntimepoints=args.t_final + 1,
    )
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path}")
        return [
            {
                "split": split,
                "source_start": start,
                "source_end": end,
                "nsamples": selected_n,
                "skipped": True,
                "output_path": str(out_path),
            }
        ]

    records: list[dict] = []
    rollout_cache = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver_batch_size = args.solver_batch_size or args.batch_size
    x_out = torch.empty((selected_n, args.target_size, args.target_size), dtype=torch.float32)
    y_out = torch.empty(
        (selected_n, args.target_size, args.target_size, args.t_final + 1),
        dtype=torch.float32,
    )

    for local_batch_index, batch_start in enumerate(
        tqdm(range(start, end, args.batch_size), desc=f"{split} batches", unit="batch")
    ):
        batch_end = min(batch_start + args.batch_size, end)
        chunk_n = batch_end - batch_start
        local_start = batch_start - start
        local_end = batch_end - start

        chunk_cpu = source_x[batch_start:batch_end]
        chunk_gpu = chunk_cpu.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        upsampled = spectral_upsample(chunk_gpu, args.target_size).contiguous()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        upsample_seconds = time.perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        y_parts = []
        for sub_start in range(0, upsampled.shape[0], solver_batch_size):
            sub_end = min(sub_start + solver_batch_size, upsampled.shape[0])
            sub_n = sub_end - sub_start
            if sub_n not in rollout_cache:
                rollout_cache[sub_n] = make_ns_rollout_fn(
                    nx=args.target_size,
                    nu=args.nu,
                    t_final=args.t_final,
                    fixed_step=args.fixed_step,
                    domain_extent=args.domain_extent,
                    solver_mode=args.solver_mode,
                )
            y_sub = rollout_cache[sub_n](torch_to_jax(upsampled[sub_start:sub_end]))
            y_sub.block_until_ready()
            y_parts.append(jax_to_torch(y_sub).contiguous())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - t1

        y_gpu = torch.cat(y_parts, dim=0).contiguous()
        x_gpu = y_gpu[..., 0].contiguous()
        x_out[local_start:local_end].copy_(x_gpu.cpu())
        y_out[local_start:local_end].copy_(y_gpu.cpu())

        records.append(
            {
                "split": split,
                "batch_index": local_batch_index,
                "source_start": batch_start,
                "source_end": batch_end,
                "batch_n": chunk_n,
                "processing_batch_size": args.batch_size,
                "solver_batch_size": solver_batch_size,
                "solver_mode": args.solver_mode,
                "upsample_seconds": upsample_seconds,
                "rollout_seconds": rollout_seconds,
                "rollout_seconds_per_sample": rollout_seconds / max(1, chunk_n),
                "output_path": str(out_path),
            }
        )

        del y_parts, y_gpu, x_gpu, upsampled, chunk_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "split": split,
        "source_path": str(source_path(args.source_dir, split, args.filename_suffix)),
        "source_key": "x",
        "source_start": start,
        "source_end": end,
        "nu_ns": args.nu,
        "tfinal": args.t_final,
        "fixed_step": args.fixed_step,
        "nsamples": selected_n,
        "save_format": "single_file_per_split",
        "processing_batch_size": args.batch_size,
        "solver_batch_size": solver_batch_size,
        "solver_mode": args.solver_mode,
        "times": np.arange(args.t_final + 1, dtype=np.float32),
        "target_size": args.target_size,
        "timing_records": records,
    }
    torch.save({"x": x_out, "y": y_out, "metadata": metadata}, out_path)
    print(f"[saved] {out_path}")

    return records


def benchmark(args: argparse.Namespace, split: str) -> None:
    source_x = load_source_x(args.source_dir, split, args.filename_suffix)
    n = min(args.benchmark_samples, source_x.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upsampled = spectral_upsample(source_x[:n].to(device), args.target_size).contiguous()
    batch_sizes = parse_int_list(args.benchmark_batch_sizes)
    reference = None
    reference_batch_size = None
    results = {}

    for batch_size in batch_sizes:
        chunks = []
        rollout_cache = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            actual = end - start
            if actual not in rollout_cache:
                rollout_cache[actual] = make_ns_rollout_fn(
                    nx=args.target_size,
                    nu=args.nu,
                    t_final=args.t_final,
                    fixed_step=args.fixed_step,
                    domain_extent=args.domain_extent,
                    solver_mode=args.solver_mode,
                )
            out = rollout_cache[actual](torch_to_jax(upsampled[start:end]))
            out.block_until_ready()
            chunks.append(np.asarray(out))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        y = np.concatenate(chunks, axis=0)
        if reference is None:
            reference = y
            reference_batch_size = batch_size
        max_abs_diff = float(np.max(np.abs(y - reference)))
        results[str(batch_size)] = {
            "total_seconds": round(seconds, 4),
            "seconds_per_sample": round(seconds / n, 4),
            "reference_batch_size": reference_batch_size,
            "max_abs_diff_vs_reference": max_abs_diff,
        }

    print("[benchmark]", json.dumps(results, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="test,train")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--filename-suffix", default="_real_initial")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of source samples processed before copying results back to CPU. The split is still saved as one .pt file.",
    )
    parser.add_argument(
        "--solver-batch-size",
        type=int,
        default=32,
        help="Maximum number of samples passed to one JAX/Exponax rollout.",
    )
    parser.add_argument(
        "--solver-mode",
        choices=["vmap", "lax-map"],
        default="lax-map",
        help="Use lax-map for sequentially reproducible batches, or vmap for fastest batched execution.",
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--t-final", type=int, default=20)
    parser.add_argument("--fixed-step", type=float, default=0.005)
    parser.add_argument("--domain-extent", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-samples", type=int, default=4)
    parser.add_argument("--benchmark-batch-sizes", default="1,2,4")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    print(
        json.dumps(
            {
                "project_root": str(PROJECT_ROOT),
                "ns_root": str(NS_ROOT),
                "source_dir": str(args.source_dir),
                "output_dir": str(args.output_dir),
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "batch_size": args.batch_size,
                "solver_batch_size": args.solver_batch_size,
                "solver_mode": args.solver_mode,
                "fixed_step": args.fixed_step,
                "t_final": args.t_final,
            },
            indent=2,
        )
    )

    splits = parse_split_list(args.splits)
    if args.benchmark:
        benchmark(args, splits[0])

    all_records = []
    for split in splits:
        all_records.extend(generate_split(args, split))

    if all_records:
        summary_path = args.output_dir / "generation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2)
        print(f"[summary] {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
