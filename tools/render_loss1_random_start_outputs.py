#!/usr/bin/env python3
"""Render visual outputs for the loss1 random-start round."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/three_loss_objective_round1_l2_eps8_alpha0p3_loss1_random_start"),
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--viz-dpi", type=int, default=110)
    parser.add_argument("--curve-dpi", type=int, default=180)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    commands = [
        [
            args.python,
            "tools/regenerate_three_loss_visualizations.py",
            "--root",
            str(args.root),
            "--fps",
            str(args.fps),
            "--stride",
            str(args.stride),
            "--dpi",
            str(args.viz_dpi),
        ],
        [
            args.python,
            "tools/plot_three_loss_curve_summary.py",
            "--root",
            str(args.root),
            "--losses",
            "loss1",
            "--dpi",
            str(args.curve_dpi),
        ],
    ]

    for cmd in commands:
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
