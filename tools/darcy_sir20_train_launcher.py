#!/usr/bin/env python3
"""Launch Darcy/SIR20 time-matched training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from darcy_sir20_common import (
    BASELINE_CHECKPOINT,
    METHOD_ORDER,
    METHODS,
    TRAINING_METHODS,
    base_training_command,
    default_bundle_root,
    ensure_bundle_dirs,
    read_summary,
    rel,
    run_command,
    validate_inputs,
    write_csv,
    write_json,
)


def load_plan(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def plan_rows_by_method(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["method"]): row for row in plan.get("methods", [])}


def checkpoint_has_optimizer(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return isinstance(payload, dict) and "model_state_dict" in payload and "optimizer_state_dict" in payload


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def training_epochs_for(method: str, args: argparse.Namespace, plan: dict[str, Any]) -> tuple[int, int, float | None]:
    if args.mode == "smoke":
        epochs = int(args.smoke_epochs)
        return epochs, epochs, None
    rows = plan_rows_by_method(plan)
    if method not in rows:
        raise KeyError(f"method {method} not found in calibration plan")
    row = rows[method]
    target_work = float(plan["loss3_reference_work_seconds"])
    if method == "loss3":
        original_epochs = int(plan.get("loss3_reference_epochs", args.loss3_reference_epochs))
        return original_epochs, int(math.ceil(original_epochs * float(args.epoch_multiplier))), None
    epochs = int(row["time_matched_epochs"])
    if args.use_max_work_seconds:
        stable = float(row["stable_work_sec_per_epoch_mean"])
        ceiling = int(math.ceil((target_work / max(stable, 1e-12)) * float(args.max_work_epoch_headroom)))
        epochs = max(epochs, ceiling)
        return epochs, int(math.ceil(epochs * float(args.epoch_multiplier))), target_work * float(args.epoch_multiplier)
    return epochs, int(math.ceil(epochs * float(args.epoch_multiplier))), None


def read_checkpoint_for_epoch(run_dir: Path, epoch: int) -> Path:
    checkpoints_csv = run_dir / "darcy" / "checkpoints.csv"
    if not checkpoints_csv.exists():
        raise FileNotFoundError(checkpoints_csv)
    rows: list[dict[str, str]] = []
    with checkpoints_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [row for row in rows if int(float(row.get("epoch", -1))) == int(epoch) and row.get("checkpoint_path")]
    if not matches:
        raise FileNotFoundError(f"no checkpoint row for epoch {epoch} in {checkpoints_csv}")
    return resolve_checkpoint(matches[-1]["checkpoint_path"])


def saved_variant_row(
    *,
    method: str,
    stage: str,
    epoch: int,
    run_dir: Path,
    summary_path: Path,
    checkpoint: Path,
    has_optimizer: bool,
    work_clock_seconds: float,
    stop_reason: str,
) -> dict[str, Any]:
    display = METHODS[method].display_name
    method_token = f"{method}_epoch{int(epoch)}"
    return {
        "method": method_token,
        "base_method": method,
        "display_name": f"{display} @ {int(epoch)}",
        "run_dir": rel(run_dir),
        "summary_json": rel(summary_path),
        "checkpoint": rel(checkpoint),
        "epochs_completed": int(epoch),
        "checkpoint_stage": stage,
        "stop_reason": stop_reason,
        "work_clock_seconds": float(work_clock_seconds),
        "optimizer_state_in_checkpoint": int(has_optimizer),
        "role": f"trained_{stage}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--methods", default=",".join(TRAINING_METHODS))
    parser.add_argument("--loss3-reference-epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--optimizer-batch-size", type=int, default=24)
    parser.add_argument("--darcy-train-max", type=int, default=64)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-eval-max-samples", type=int, default=1)
    parser.add_argument("--full-eval-max-samples", type=int, default=0)
    parser.add_argument("--max-generalization-eval", type=int, default=50)
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=100)
    parser.add_argument("--epoch-multiplier", type=float, default=2.0)
    parser.add_argument("--eps-jitter-low", type=float, default=0.25)
    parser.add_argument("--eps-jitter-high", type=float, default=1.75)
    parser.add_argument("--attack-probe-samples", type=int, default=8)
    parser.add_argument("--attack-probe-every-n-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-at-original-and-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-max-work-seconds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-work-epoch-headroom", type=float, default=1.08)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    if args.epoch_multiplier < 1.0:
        raise ValueError(f"--epoch-multiplier must be >= 1.0, got {args.epoch_multiplier}")
    if args.eps_jitter_high < args.eps_jitter_low:
        raise ValueError(
            f"--eps-jitter-high must be >= --eps-jitter-low, got {args.eps_jitter_low}..{args.eps_jitter_high}"
        )

    validate_inputs()
    bundle = (args.bundle or default_bundle_root()).resolve()
    dirs = ensure_bundle_dirs(bundle)
    plan = load_plan(args.plan_json)
    if args.mode == "full" and not plan:
        raise ValueError("--mode full requires --plan-json from darcy_sir20_calibrate.py")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in TRAINING_METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; valid={TRAINING_METHODS}")

    training_root = dirs["data"] / "training_runs"
    training_root.mkdir(parents=True, exist_ok=True)
    log_root = dirs["logs"] / f"training_{args.mode}"
    checkpoint_rows: list[dict[str, Any]] = [
        {
            "method": "baseline",
            "base_method": "baseline",
            "display_name": METHODS["baseline"].display_name,
            "run_dir": "",
            "summary_json": "",
            "checkpoint": rel(BASELINE_CHECKPOINT),
            "epochs_completed": 0,
            "checkpoint_stage": "baseline",
            "work_clock_seconds": 0.0,
            "optimizer_state_in_checkpoint": "",
            "role": "baseline",
        }
    ]
    all_saved_rows: list[dict[str, Any]] = list(checkpoint_rows)

    eval_max_samples = args.smoke_eval_max_samples if args.mode == "smoke" else args.full_eval_max_samples
    checkpoint_every = 1 if args.mode == "smoke" else args.checkpoint_every_epochs

    for method in methods:
        original_epochs, epochs, max_work_seconds = training_epochs_for(method, args, plan)
        per_method_checkpoint_every = checkpoint_every
        if args.mode == "full" and args.checkpoint_at_original_and_final:
            per_method_checkpoint_every = max(1, int(original_epochs))
        run_name = (
            f"darcy_sir20_{args.mode}_{METHODS[method].run_token}_"
            f"{original_epochs}to{epochs}ep_epsj{args.eps_jitter_low:g}to{args.eps_jitter_high:g}"
        )
        if max_work_seconds is not None:
            run_name += "_workmatched"
        run_dir = training_root / run_name
        summary_path = run_dir / "darcy" / "summary.json"
        if not (args.reuse and summary_path.exists()):
            cmd = base_training_command(
                run_name=run_name,
                output_root=training_root,
                method=method,
                epochs=epochs,
                eval_max_samples=eval_max_samples,
                max_generalization_eval=args.max_generalization_eval,
                batch_size=args.batch_size,
                optimizer_batch_size=args.optimizer_batch_size,
                checkpoint_every_epochs=per_method_checkpoint_every,
                eval_every_epochs=args.eval_every_epochs,
                attack_probe_samples=args.attack_probe_samples,
                attack_probe_every=args.attack_probe_every_n_epochs,
                max_work_seconds=max_work_seconds,
                epsilon_bucket_count=5,
                eps_jitter_low=args.eps_jitter_low,
                eps_jitter_high=args.eps_jitter_high,
                train_max_samples=args.darcy_train_max,
            )
            run_command(cmd, log_root / f"{run_name}.log")
        summary = read_summary(run_dir)
        ckpt = resolve_checkpoint(str(summary["final_checkpoint"]))
        has_opt = checkpoint_has_optimizer(ckpt)
        completed_epochs = int(summary.get("epochs", 0))
        stop_reason = summary.get("stop_reason", "")
        work_clock_seconds = float(summary.get("work_clock_seconds", float("nan")))
        checkpoint_rows.append(
            {
                "method": method,
                "base_method": method,
                "display_name": METHODS[method].display_name,
                "run_dir": rel(run_dir),
                "summary_json": rel(summary_path),
                "checkpoint": rel(ckpt),
                "epochs_original": int(original_epochs),
                "epochs_requested": epochs,
                "epochs_completed": completed_epochs,
                "checkpoint_stage": "final",
                "epoch_multiplier": float(args.epoch_multiplier),
                "epsilon_jitter_low": float(args.eps_jitter_low),
                "epsilon_jitter_high": float(args.eps_jitter_high),
                "darcy_train_max": "" if args.darcy_train_max is None else int(args.darcy_train_max),
                "attack_probe_samples": int(args.attack_probe_samples),
                "attack_probe_every_n_epochs": int(args.attack_probe_every_n_epochs),
                "stop_reason": stop_reason,
                "work_clock_seconds": work_clock_seconds,
                "local_work_clock_seconds": float(summary.get("local_work_clock_seconds", float("nan"))),
                "max_work_seconds": "" if max_work_seconds is None else float(max_work_seconds),
                "optimizer_state_in_checkpoint": int(has_opt),
                "role": "trained_final",
            }
        )
        if not has_opt:
            raise RuntimeError(f"final checkpoint lacks optimizer_state_dict: {ckpt}")
        if args.mode == "full" and args.checkpoint_at_original_and_final and completed_epochs >= int(original_epochs):
            original_ckpt = read_checkpoint_for_epoch(run_dir, int(original_epochs))
            original_has_opt = checkpoint_has_optimizer(original_ckpt)
            if not original_has_opt:
                raise RuntimeError(f"original checkpoint lacks optimizer_state_dict: {original_ckpt}")
            all_saved_rows.append(
                saved_variant_row(
                    method=method,
                    stage="original",
                    epoch=int(original_epochs),
                    run_dir=run_dir,
                    summary_path=summary_path,
                    checkpoint=original_ckpt,
                    has_optimizer=original_has_opt,
                    work_clock_seconds=work_clock_seconds,
                    stop_reason=stop_reason,
                )
            )
        all_saved_rows.append(
            saved_variant_row(
                method=method,
                stage="final",
                epoch=completed_epochs,
                run_dir=run_dir,
                summary_path=summary_path,
                checkpoint=ckpt,
                has_optimizer=has_opt,
                work_clock_seconds=work_clock_seconds,
                stop_reason=stop_reason,
            )
        )

    csv_path = dirs["checkpoints_manifest"] / f"training_checkpoints_{args.mode}.csv"
    json_path = dirs["checkpoints_manifest"] / f"training_checkpoints_{args.mode}.json"
    all_csv_path = dirs["checkpoints_manifest"] / f"training_checkpoints_{args.mode}_all_saved.csv"
    all_json_path = dirs["checkpoints_manifest"] / f"training_checkpoints_{args.mode}_all_saved.json"
    write_csv(csv_path, checkpoint_rows)
    write_csv(all_csv_path, all_saved_rows)
    write_json(
        json_path,
        {
            "bundle": rel(bundle),
            "mode": args.mode,
            "method_order": METHOD_ORDER,
            "checkpoints": checkpoint_rows,
            "plan_json": "" if args.plan_json is None else rel(args.plan_json.resolve()),
            "epoch_multiplier": float(args.epoch_multiplier),
            "epsilon_jitter_low": float(args.eps_jitter_low),
            "epsilon_jitter_high": float(args.eps_jitter_high),
            "darcy_train_max": "" if args.darcy_train_max is None else int(args.darcy_train_max),
            "attack_probe_samples": int(args.attack_probe_samples),
            "attack_probe_every_n_epochs": int(args.attack_probe_every_n_epochs),
            "eval_every_epochs": int(args.eval_every_epochs),
        },
    )
    write_json(
        all_json_path,
        {
            "bundle": rel(bundle),
            "mode": args.mode,
            "method_order": [row["method"] for row in all_saved_rows],
            "base_method_order": METHOD_ORDER,
            "checkpoints": all_saved_rows,
            "plan_json": "" if args.plan_json is None else rel(args.plan_json.resolve()),
            "epoch_multiplier": float(args.epoch_multiplier),
            "epsilon_jitter_low": float(args.eps_jitter_low),
            "epsilon_jitter_high": float(args.eps_jitter_high),
            "darcy_train_max": "" if args.darcy_train_max is None else int(args.darcy_train_max),
            "attack_probe_samples": int(args.attack_probe_samples),
            "attack_probe_every_n_epochs": int(args.attack_probe_every_n_epochs),
            "eval_every_epochs": int(args.eval_every_epochs),
        },
    )
    print(json_path)
    print(all_json_path)


if __name__ == "__main__":
    main()
