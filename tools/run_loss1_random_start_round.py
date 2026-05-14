#!/usr/bin/env python3
"""Run the loss1 objective/method grid with a tiny nonzero random start."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = ("original", "increment_ratio", "regularized")
METHODS = (
    ("pgd", "pgd", "none"),
    ("lp_steepest_pgd", "lp_steepest_pgd", "none"),
    ("power_iteration", "generalized_power", "gradient"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/three_loss_objective_round1_l2_eps8_alpha0p3_loss1_random_start"))
    parser.add_argument("--case", default="burgers")
    parser.add_argument("--norm", default="2")
    parser.add_argument("--input-p", default="2")
    parser.add_argument("--output-q", default="2")
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-start-scale", type=float, default=1e-6)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    for variant in VARIANTS:
        for method_label, attack_method, power_variant in METHODS:
            run_name = (
                f"burgers_loss1_{variant}_{method_label}_p{args.input_p}_q{args.output_q}"
                f"_eta{args.eta:g}_C{args.regularization_c:g}_eps{args.epsilon:g}"
                f"_alpha{args.alpha:g}_steps{args.steps}_idx{args.index}_seed{args.seed}_random_start"
            )
            out_dir = args.out_root / run_name
            cmd = [
                args.python,
                "run_loss_comparison_attack.py",
                "--case",
                args.case,
                "--solver_backend",
                "jax",
                "--model_backend",
                "torch",
                "--loss_type",
                "loss1",
                "--objective_variant",
                variant,
                "--attack_method",
                attack_method,
                "--power_variant",
                power_variant,
                "--norm",
                args.norm,
                "--input_p",
                args.input_p,
                "--output_q",
                args.output_q,
                "--epsilon",
                str(args.epsilon),
                "--alpha",
                str(args.alpha),
                "--steps",
                str(args.steps),
                "--index",
                str(args.index),
                "--seed",
                str(args.seed),
                "--random_start",
                "--random_start_scale",
                str(args.random_start_scale),
                "--ratio_denominator_epsilon",
                str(args.eta),
                "--regularization_c",
                str(args.regularization_c),
                "--device",
                args.device,
                "--output_dir",
                str(out_dir),
            ]
            if args.no_progress:
                cmd.append("--no-progress")
            print("[run]", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
