#!/usr/bin/env python3
"""Small benchmark for real-initial 2D Navier-Stokes Exponax generation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--nsamples", type=int, default=3)
    parser.add_argument("--tfinal", type=int, default=20)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_generation_dir = project_root / "2D_NS_FNO2d_recurrent" / "data_generation"
    sys.path.insert(0, str(data_generation_dir))

    from VT_NS_gen_all_frame import DatasetGenerator  # noqa: PLC0415

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = (
            project_root
            / "2D_NS_FNO2d_recurrent"
            / "datasets"
            / "exponax_datasets"
            / "t20"
            / "real_initial_benchmark"
            / f"{args.split}_N{args.nsamples}"
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        {
            "split": args.split,
            "nsamples": args.nsamples,
            "tfinal": args.tfinal,
            "save_dir": str(save_dir),
        }
    )

    t0 = time.perf_counter()
    generator = DatasetGenerator(
        base_seed=0,
        train_test=args.split,
        filename_suffix="_real_initial",
        source_limit=args.nsamples,
    )
    t_after_load = time.perf_counter()

    generator.generate_dataset(
        nu_ns=1e-5,
        tfinal=args.tfinal,
        nsamples=args.nsamples,
        batch_size=args.nsamples,
        ntimepoints=args.tfinal + 1,
        save_dir=str(save_dir),
    )
    t1 = time.perf_counter()

    load_seconds = t_after_load - t0
    generate_seconds = t1 - t_after_load
    total_seconds = t1 - t0
    print(
        {
            "load_and_upsample_seconds": round(load_seconds, 3),
            "generate_seconds": round(generate_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "seconds_per_sample_generate_only": round(generate_seconds / args.nsamples, 3),
            "seconds_per_sample_total": round(total_seconds / args.nsamples, 3),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
