#!/usr/bin/env python3
"""Fill the missing Darcy 7-model robustness artifacts.

This runner creates the missing pieces requested for the final 52-dataset,
7-model Darcy comparison:

- unified 52-dataset attack20 table for all seven models with delta metrics,
- baseline-only 25-sample metric/Jacobian correlation rows,
- baseline-only 5-sample Jacobian probe rows,
- final merged 7-model statistics/report.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.run_darcy_attack20_52datasets_50samples_20260612 as attack20
import tools.run_darcy_jacobian_probe_5gen_20260612 as jac5
import tools.run_darcy_metric_correlation_10samples_20260612 as metric_base
import tools.run_darcy_metric_correlation_25samples_20260612 as metric25
import tools.build_darcy_complete_7model_statistics_20260613 as build_complete


RUN_TAG = "20260612_full50_timematched_1000c"
RANDOM_TAG = "20260613_random_binary_source_1100"
DEFAULT_TAG = "20260613_7model_complete_missing_metrics"


@dataclass(frozen=True)
class CompleteModel:
    name: str
    checkpoint: Path
    trained_epochs: int
    objective: str


def complete_models() -> list[CompleteModel]:
    run_root = PROJECT_ROOT / "adversarial_training_runs"
    return [
        CompleteModel(
            "baseline",
            PROJECT_ROOT / "2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt",
            500,
            "baseline",
        ),
        CompleteModel(
            "loss1",
            run_root / f"darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_{RUN_TAG}/darcy/checkpoints/darcy_epoch1000_step001000.pt",
            1000,
            "loss1",
        ),
        CompleteModel(
            "loss2",
            run_root / f"darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_{RUN_TAG}/darcy/checkpoints/darcy_epoch1026_step001026.pt",
            1026,
            "loss2",
        ),
        CompleteModel(
            "loss3",
            run_root / f"darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_{RUN_TAG}/darcy/checkpoints/darcy_epoch1011_step001011.pt",
            1011,
            "loss3",
        ),
        CompleteModel(
            "physics_loss",
            run_root / f"darcy_binary_loss3targeted_physics_1040ep_full50_timematched_{RUN_TAG}/darcy/checkpoints/darcy_epoch1040_step001040.pt",
            1040,
            "physics",
        ),
        CompleteModel(
            "random_clean_y",
            run_root / f"darcy_binary_random_binary_fixed_y_1100ep_full50_{RANDOM_TAG}/darcy/checkpoints/darcy_epoch1100_step001100.pt",
            1100,
            "random-binary-fixed-y",
        ),
        CompleteModel(
            "random_solver_y",
            run_root / f"darcy_binary_random_binary_solver_y_1100ep_full50_{RANDOM_TAG}/darcy/checkpoints/darcy_epoch1100_step001100.pt",
            1100,
            "random-binary-solver-y",
        ),
    ]


def assert_checkpoints(models: list[CompleteModel]) -> None:
    missing = [m for m in models if not m.checkpoint.exists()]
    if missing:
        text = "\n".join(f"{m.name}: {m.checkpoint}" for m in missing)
        raise FileNotFoundError("Missing Darcy checkpoints:\n" + text)


def run_attack7(args: argparse.Namespace, models: list[CompleteModel]) -> None:
    attack20.MODEL_SPECS = [
        attack20.ModelSpec(m.name, m.checkpoint, m.trained_epochs, m.objective)
        for m in models
    ]
    old_argv = sys.argv[:]
    sys.argv = [
        "run_darcy_attack20_52datasets_50samples_20260612.py",
        "--tag",
        args.tag,
        "--generalization-root",
        str(args.generalization_root),
        "--out-dir",
        str(args.attack_out_dir),
        "--viz-dir",
        str(args.attack_viz_dir),
        "--samples-per-dataset",
        str(args.samples_per_dataset),
        "--sample-policy",
        args.sample_policy,
        "--seed",
        str(args.seed),
        "--attack-steps",
        str(args.attack_steps),
        "--epsilon-fraction",
        str(args.epsilon_fraction),
        "--batch-size",
        str(args.attack_batch_size),
        "--overwrite",
    ]
    try:
        attack20.main()
    finally:
        sys.argv = old_argv


def run_baseline_metric25(args: argparse.Namespace, baseline: CompleteModel) -> None:
    metric_base.MODELS = [
        metric_base.ModelSpec(baseline.name, baseline.checkpoint, baseline.trained_epochs, baseline.objective)
    ]
    metric_base.SAMPLES = metric25.select_samples(args.generalization_root.resolve(), args.metric_samples)
    metric_args = argparse.Namespace(
        tag=args.tag,
        generalization_root=args.generalization_root,
        out_dir=args.baseline_metric_out_dir,
        viz_dir=args.baseline_metric_viz_dir,
        max_samples=args.metric_samples,
        models=None,
        attack_steps=args.attack_steps,
        epsilon_fraction=args.epsilon_fraction,
        block_row_chunk=args.block_row_chunk,
        resume=True,
        cpu=False,
    )
    metric_base.run(metric_args)


def run_baseline_jacobian5(args: argparse.Namespace, baseline: CompleteModel) -> None:
    jac5.MODELS = [jac5.ModelSpec(baseline.name, baseline.checkpoint, baseline.objective)]
    jac_args = argparse.Namespace(
        tag=args.tag,
        generalization_root=args.generalization_root,
        out_dir=args.baseline_jac_out_dir,
        viz_dir=args.baseline_jac_viz_dir,
        sample_index=0,
        power_iterations=args.power_iterations,
        seed=args.seed,
        models=None,
        cpu=False,
    )
    jac5.run_probe(jac_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--attack-out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/darcy_attack20_52datasets_50samples_20260613_7models_delta_complete")
    parser.add_argument("--attack-viz-dir", type=Path, default=PROJECT_ROOT / "visualizations/darcy_attack20_52datasets_50samples_20260613_7models_delta_complete")
    parser.add_argument("--baseline-metric-out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/darcy_baseline_metric_correlation_25samples_20260613")
    parser.add_argument("--baseline-metric-viz-dir", type=Path, default=PROJECT_ROOT / "visualizations/darcy_baseline_metric_correlation_25samples_20260613")
    parser.add_argument("--baseline-jac-out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/darcy_baseline_jacobian_probe_5gen_20260613")
    parser.add_argument("--baseline-jac-viz-dir", type=Path, default=PROJECT_ROOT / "visualizations/darcy_baseline_jacobian_probe_5gen_20260613")
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--sample-policy", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--attack-batch-size", type=int, default=25)
    parser.add_argument("--metric-samples", type=int, default=25)
    parser.add_argument("--block-row-chunk", type=int, default=64)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument(
        "--stages",
        default="attack7,baseline_metric25,baseline_jacobian5,aggregate",
        help="Comma-separated subset: attack7,baseline_metric25,baseline_jacobian5,aggregate.",
    )
    args = parser.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    models = complete_models()
    assert_checkpoints(models)
    baseline = models[0]

    if "attack7" in stages:
        run_attack7(args, models)
    if "baseline_metric25" in stages:
        run_baseline_metric25(args, baseline)
    if "baseline_jacobian5" in stages:
        run_baseline_jacobian5(args, baseline)
    if "aggregate" in stages:
        build_complete.main()


if __name__ == "__main__":
    main()
