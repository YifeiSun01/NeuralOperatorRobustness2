#!/usr/bin/env python3
"""Run the Darcy metric/attack correlation study on 25 binary generalization samples.

This wrapper reuses the 10-sample implementation and replaces the hard-coded
sample list with the first N generated binary Darcy datasets. It keeps the same
metrics and plotting pipeline so the 25-sample run is directly comparable with
the earlier 10-sample pilot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tools.run_darcy_metric_correlation_10samples_20260612 as base

DEFAULT_TAG = "20260612_25samples_loss3attack20_corr_chunk64"


def select_samples(generalization_root: Path, max_samples: int) -> list[tuple[str, str, int]]:
    darcy_dir = generalization_root / "darcy"
    files = sorted(darcy_dir.glob("darcy_binary_loss3targeted_20260611_*.pt"))
    if len(files) < max_samples:
        raise FileNotFoundError(f"requested {max_samples} samples but only found {len(files)} in {darcy_dir}")
    samples: list[tuple[str, str, int]] = []
    for path in files[:max_samples]:
        dataset_id = path.stem
        seq = dataset_id.split("20260611_", 1)[-1].split("_", 1)[0]
        samples.append((f"gen{seq}", dataset_id, 0))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=base.PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-dir", type=Path, default=base.PROJECT_ROOT / "analysis_outputs" / f"darcy_metric_correlation_25samples_{DEFAULT_TAG}")
    parser.add_argument("--viz-dir", type=Path, default=base.PROJECT_ROOT / "visualizations" / f"darcy_metric_correlation_25samples_{DEFAULT_TAG}")
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--models", default=None, help="comma-separated subset of loss1,loss2,loss3,physics")
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--block-row-chunk", type=int, default=64)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    base.SAMPLES = select_samples(args.generalization_root.resolve(), args.max_samples)
    base.run(args)


if __name__ == "__main__":
    main()
