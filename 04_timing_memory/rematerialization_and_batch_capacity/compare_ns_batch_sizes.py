#!/usr/bin/env python3
"""Compare 2D NS batched Exponax rollouts against true batch_size=1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NS_SCRIPT_DIR = PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "data_generation"
sys.path.insert(0, str(NS_SCRIPT_DIR))

from generate_ns_real_initial_batched import (  # noqa: E402
    DEFAULT_SOURCE_DIR,
    load_source_x,
    make_ns_rollout_fn,
    spectral_upsample,
    torch_to_jax,
)


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def run_rollout(
    upsampled: torch.Tensor,
    *,
    batch_size: int,
    target_size: int,
    nu: float,
    t_final: int,
    fixed_step: float,
    domain_extent: float,
    solver_mode: str,
) -> tuple[np.ndarray, float]:
    cache = {}
    chunks = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, upsampled.shape[0], batch_size):
        end = min(start + batch_size, upsampled.shape[0])
        actual = end - start
        if actual not in cache:
            cache[actual] = make_ns_rollout_fn(
                nx=target_size,
                nu=nu,
                t_final=t_final,
                fixed_step=fixed_step,
                domain_extent=domain_extent,
                solver_mode=solver_mode,
            )
        y = cache[actual](torch_to_jax(upsampled[start:end]))
        y.block_until_ready()
        chunks.append(np.asarray(y, dtype=np.float32))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return np.concatenate(chunks, axis=0), time.perf_counter() - t0


def compare_to_reference(y: np.ndarray, reference: np.ndarray) -> dict:
    diff = y - reference
    per_sample_max_abs = np.max(np.abs(diff), axis=(1, 2, 3))
    per_sample_rmse = np.sqrt(np.mean(diff * diff, axis=(1, 2, 3)))
    ref_rms = np.sqrt(np.mean(reference * reference, axis=(1, 2, 3)))
    per_sample_relative_rmse = per_sample_rmse / np.maximum(ref_rms, 1e-12)
    return {
        "global_max_abs": float(np.max(per_sample_max_abs)),
        "mean_sample_max_abs": float(np.mean(per_sample_max_abs)),
        "p95_sample_max_abs": float(np.percentile(per_sample_max_abs, 95)),
        "global_rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_sample_rmse": float(np.mean(per_sample_rmse)),
        "mean_sample_relative_rmse": float(np.mean(per_sample_relative_rmse)),
        "max_sample_relative_rmse": float(np.max(per_sample_relative_rmse)),
        "per_sample": [
            {
                "sample": int(i),
                "max_abs": float(per_sample_max_abs[i]),
                "rmse": float(per_sample_rmse[i]),
                "relative_rmse": float(per_sample_relative_rmse[i]),
            }
            for i in range(y.shape[0])
        ],
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# NS Batch Size Comparison",
        "",
        "Reference is true `batch_size=1` on the same samples.",
        "",
        "## Setup",
        "",
        f"- split: `{payload['setup']['split']}`",
        f"- samples: `{payload['setup']['num_samples']}`",
        f"- fixed_step: `{payload['setup']['fixed_step']}`",
        f"- t_final: `{payload['setup']['t_final']}`",
        f"- target_size: `{payload['setup']['target_size']}`",
        f"- solver_mode: `{payload['setup']['solver_mode']}`",
        f"- source: `{payload['setup']['source_dir']}`",
        f"- device: `{payload['setup']['torch_device']}`",
        "",
        "## Summary",
        "",
        "| batch_size | seconds | sec/sample | global max abs | global RMSE | mean rel RMSE | max rel RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch_size, row in payload["results"].items():
        cmp = row["comparison"]
        lines.append(
            f"| {batch_size} | {row['seconds']:.4f} | {row['seconds_per_sample']:.4f} "
            f"| {cmp['global_max_abs']:.8g} | {cmp['global_rmse']:.8g} "
            f"| {cmp['mean_sample_relative_rmse']:.8g} | {cmp['max_sample_relative_rmse']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `batch_size=1` is the reference, so all error columns are zero for it.",
        ]
    )
    if payload["setup"]["solver_mode"] == "lax-map":
        lines.extend(
            [
                "- `solver_mode=lax-map` maps samples sequentially inside JIT. It is slower than `vmap`, but it is designed to preserve the sequential numerical path.",
                "- If all error columns are zero for larger batches, this mode matched the sequential reference exactly in this run.",
            ]
        )
    else:
        lines.extend(
            [
                "- Small nonzero differences at larger batch sizes are floating-point differences from XLA/JAX vectorized execution, not different input data or paths.",
                "- If exact agreement with the sequential solver matters, use `--solver-mode lax-map` or `batch_size=1`.",
            ]
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--filename-suffix", default="_real_initial")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--t-final", type=int, default=20)
    parser.add_argument("--fixed-step", type=float, default=0.005)
    parser.add_argument("--domain-extent", type=float, default=1.0)
    parser.add_argument(
        "--solver-mode",
        choices=["vmap", "lax-map"],
        default="vmap",
        help="Use vmap for fastest batched execution, or lax-map for sequential map inside JIT.",
    )
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "run_logs" / "ns_batch_comparison.json")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "NS_BATCH_SIZE_COMPARISON.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_x = load_source_x(args.source_dir.resolve(), args.split, args.filename_suffix)
    source_x = source_x[: args.num_samples]
    upsampled = spectral_upsample(source_x.to(device), args.target_size).contiguous()

    print(
        json.dumps(
            {
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "split": args.split,
                "num_samples": args.num_samples,
                "batch_sizes": parse_int_list(args.batch_sizes),
                "solver_mode": args.solver_mode,
            },
            indent=2,
        )
    )

    reference = None
    payload = {
        "setup": {
            "split": args.split,
            "num_samples": args.num_samples,
            "batch_sizes": parse_int_list(args.batch_sizes),
            "source_dir": str(args.source_dir.resolve()),
            "target_size": args.target_size,
            "nu": args.nu,
            "t_final": args.t_final,
            "fixed_step": args.fixed_step,
            "domain_extent": args.domain_extent,
            "solver_mode": args.solver_mode,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "results": {},
    }

    for batch_size in parse_int_list(args.batch_sizes):
        y, seconds = run_rollout(
            upsampled,
            batch_size=batch_size,
            target_size=args.target_size,
            nu=args.nu,
            t_final=args.t_final,
            fixed_step=args.fixed_step,
            domain_extent=args.domain_extent,
            solver_mode=args.solver_mode,
        )
        if reference is None:
            reference = y
        comparison = compare_to_reference(y, reference)
        payload["results"][str(batch_size)] = {
            "seconds": seconds,
            "seconds_per_sample": seconds / args.num_samples,
            "comparison": comparison,
        }
        print(
            f"[batch={batch_size}] seconds={seconds:.4f} "
            f"sec/sample={seconds / args.num_samples:.4f} "
            f"max_abs_vs_batch1={comparison['global_max_abs']:.8g} "
            f"global_rmse={comparison['global_rmse']:.8g} "
            f"mean_rel_rmse={comparison['mean_sample_relative_rmse']:.8g}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[json] {args.output_json}")
    print(f"[markdown] {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
