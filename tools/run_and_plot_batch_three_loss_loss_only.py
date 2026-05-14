#!/usr/bin/env python3
"""Run batch loss-only attacks, then plot mean/std loss curves."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/three_loss_batch100_loss_only"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--losses", nargs="+", choices=["loss1", "loss2", "loss3"], default=["loss1", "loss2", "loss3"])
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--p", default="2")
    parser.add_argument("--q", default="2")
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss1-initial-delta", choices=["zero", "random", "both"], default="random")
    parser.add_argument("--random-start-scale", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-kind", choices=["fno", "deeponet"], default="fno")
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--burgers-test-path", type=Path, default=None)
    parser.add_argument("--burgers-torch-checkpoint", type=Path, default=None)
    parser.add_argument("--deeponet-checkpoint", type=Path, default=None)
    parser.add_argument("--deeponet-output-transform-stats", type=Path, default=None)
    parser.add_argument("--burgers-nu", type=float, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-attack", action="store_true")
    args = parser.parse_args()

    if not args.skip_attack:
        attack_cmd = [
            args.python,
            "tools/run_batch_three_loss_loss_only.py",
            "--out-root",
            str(args.out_root),
            "--batch-size",
            str(args.batch_size),
            "--start-index",
            str(args.start_index),
            "--losses",
            *args.losses,
            "--epsilon",
            str(args.epsilon),
            "--alpha",
            str(args.alpha),
            "--steps",
            str(args.steps),
            "--p",
            str(args.p),
            "--q",
            str(args.q),
            "--eta",
            str(args.eta),
            "--regularization-c",
            str(args.regularization_c),
            "--seed",
            str(args.seed),
            "--loss1-initial-delta",
            args.loss1_initial_delta,
            "--loss1-random-start-scale",
            str(args.random_start_scale),
            "--device",
            args.device,
            "--model-kind",
            args.model_kind,
        ]
        optional_args = [
            ("--model-label", args.model_label),
            ("--burgers-test-path", args.burgers_test_path),
            ("--burgers-torch-checkpoint", args.burgers_torch_checkpoint),
            ("--deeponet-checkpoint", args.deeponet_checkpoint),
            ("--deeponet-output-transform-stats", args.deeponet_output_transform_stats),
            ("--burgers-nu", args.burgers_nu),
        ]
        for flag, value in optional_args:
            if value is not None:
                attack_cmd.extend([flag, str(value)])
        print("[run]", " ".join(attack_cmd), flush=True)
        subprocess.run(attack_cmd, check=True)

    plot_cmd = [
        args.python,
        "tools/plot_batch_three_loss_loss_only.py",
        "--root",
        str(args.out_root),
        "--losses",
        *args.losses,
    ]
    print("[run]", " ".join(plot_cmd), flush=True)
    subprocess.run(plot_cmd, check=True)

    index_plot_cmd = [
        args.python,
        "tools/plot_batch_single_index_loss_curves.py",
        "--root",
        str(args.out_root),
        "--dataset-index",
        str(args.start_index),
        "--losses",
        *args.losses,
    ]
    print("[run]", " ".join(index_plot_cmd), flush=True)
    subprocess.run(index_plot_cmd, check=True)

    delta_plot_cmd = [
        args.python,
        "tools/plot_batch_final_delta_comparison.py",
        "--root",
        str(args.out_root),
        "--losses",
        *args.losses,
    ]
    print("[run]", " ".join(delta_plot_cmd), flush=True)
    subprocess.run(delta_plot_cmd, check=True)


if __name__ == "__main__":
    main()
