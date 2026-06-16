#!/usr/bin/env python3
"""Calibrate Darcy/SIR20 work-clock seconds per epoch and time-matched epochs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from darcy_sir20_common import (
    METHOD_ORDER,
    METHODS,
    TRAINING_METHODS,
    base_training_command,
    default_bundle_root,
    ensure_bundle_dirs,
    finite_mean,
    finite_median,
    method_run_name,
    now_tag,
    read_summary,
    read_work_clock_epochs,
    rel,
    run_command,
    validate_inputs,
    write_csv,
    write_json,
)


def stable_stats(df: pd.DataFrame, warmup_epochs: int) -> dict[str, Any]:
    stable = df[df["local_epoch"].astype(int) > int(warmup_epochs)].copy()
    if stable.empty:
        stable = df.copy()
    values = stable["work_clock_epoch_seconds"].astype(float)
    return {
        "stable_epoch_count": int(len(stable)),
        "stable_epoch_first": int(stable["epoch"].min()),
        "stable_epoch_last": int(stable["epoch"].max()),
        "stable_work_sec_per_epoch_mean": finite_mean(values),
        "stable_work_sec_per_epoch_median": finite_median(values),
        "all_work_sec_per_epoch_mean": finite_mean(df["work_clock_epoch_seconds"].astype(float)),
        "all_work_sec_total": float(df["work_clock_epoch_seconds"].astype(float).sum()),
    }


def write_report(path: Path, rows: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    lines = [
        "# Darcy/SIR20 Timing Calibration",
        "",
        "Observed from short calibration runs. Work-clock seconds include attack/random/solver-target generation plus forward/backward/optimizer work, and exclude evaluation/checkpoint/plot/upload.",
        "",
        f"- Loss3 reference epochs: `{plan['loss3_reference_epochs']}`",
        f"- Loss3 stable sec/epoch: `{plan['loss3_stable_work_sec_per_epoch']:.6f}`",
        f"- Loss3 3000-epoch work-clock budget: `{plan['loss3_reference_work_seconds']:.3f}` seconds (`{plan['loss3_reference_work_hours']:.3f}` hours)",
        "",
        "| method | calibration epochs | stable sec/epoch | time-matched epochs | projected work seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['calibration_epochs_completed']} | "
            f"{row['stable_work_sec_per_epoch_mean']:.6f} | {row['time_matched_epochs']} | "
            f"{row['projected_work_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- CSV: `{rel(path.parent.parent / 'data' / 'timing_calibration.csv')}`",
            f"- JSON: `{rel(path.parent.parent / 'data' / 'timing_calibration.json')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--loss3-reference-epochs", type=int, default=3000)
    parser.add_argument("--methods", default=",".join(TRAINING_METHODS))
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--optimizer-batch-size", type=int, default=24)
    parser.add_argument("--eval-max-samples", type=int, default=1)
    parser.add_argument("--max-generalization-eval", type=int, default=3)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    validate_inputs()
    bundle = (args.bundle or default_bundle_root(args.tag)).resolve()
    dirs = ensure_bundle_dirs(bundle)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in TRAINING_METHODS]
    if bad:
        raise ValueError(f"unknown calibration methods {bad}; valid={TRAINING_METHODS}")

    calibration_root = dirs["data"] / "calibration_runs"
    calibration_root.mkdir(parents=True, exist_ok=True)
    log_root = dirs["logs"] / "calibration"
    rows: list[dict[str, Any]] = []
    tag = args.tag or now_tag()

    for method in methods:
        run_name = method_run_name("darcy_sir20_calib", method, f"{args.epochs}ep_{tag}")
        run_dir = calibration_root / run_name
        summary_path = run_dir / "darcy" / "summary.json"
        if not (args.reuse and summary_path.exists()):
            cmd = base_training_command(
                run_name=run_name,
                output_root=calibration_root,
                method=method,
                epochs=args.epochs,
                eval_max_samples=args.eval_max_samples,
                max_generalization_eval=args.max_generalization_eval,
                batch_size=args.batch_size,
                optimizer_batch_size=args.optimizer_batch_size,
                checkpoint_every_epochs=args.checkpoint_every_epochs,
                attack_probe_samples=0,
                attack_probe_every=1,
                epsilon_bucket_count=0,
            )
            run_command(cmd, log_root / f"{run_name}.log")
        df = read_work_clock_epochs(run_dir)
        stats = stable_stats(df, args.warmup_epochs)
        summary = read_summary(run_dir)
        rows.append(
            {
                "method": method,
                "display_name": METHODS[method].display_name,
                "run_dir": rel(run_dir),
                "summary_json": rel(summary_path),
                "calibration_epochs_requested": int(args.epochs),
                "calibration_epochs_completed": int(summary.get("epochs", len(df))),
                "warmup_epochs_ignored": int(args.warmup_epochs),
                **stats,
            }
        )

    by_method = {row["method"]: row for row in rows}
    if "loss3" not in by_method:
        raise RuntimeError("loss3 must be included in calibration to define the reference work-clock budget")
    loss3_sec = float(by_method["loss3"]["stable_work_sec_per_epoch_mean"])
    if not math.isfinite(loss3_sec) or loss3_sec <= 0:
        raise RuntimeError(f"invalid loss3 stable sec/epoch: {loss3_sec}")
    reference_work = loss3_sec * int(args.loss3_reference_epochs)
    for row in rows:
        sec = float(row["stable_work_sec_per_epoch_mean"])
        matched_epochs = max(1, int(round(reference_work / sec))) if math.isfinite(sec) and sec > 0 else 0
        row["loss3_reference_epochs"] = int(args.loss3_reference_epochs)
        row["loss3_reference_work_seconds"] = reference_work
        row["time_matched_epochs"] = matched_epochs
        row["projected_work_seconds"] = matched_epochs * sec if matched_epochs else float("nan")
        row["projected_work_hours"] = row["projected_work_seconds"] / 3600.0 if matched_epochs else float("nan")
    for method in TRAINING_METHODS:
        if method not in by_method and method != "loss3":
            continue
    plan = {
        "bundle": rel(bundle),
        "methods": rows,
        "method_order": METHOD_ORDER,
        "training_methods": TRAINING_METHODS,
        "loss3_reference_epochs": int(args.loss3_reference_epochs),
        "loss3_stable_work_sec_per_epoch": loss3_sec,
        "loss3_reference_work_seconds": reference_work,
        "loss3_reference_work_hours": reference_work / 3600.0,
        "warmup_epochs_ignored": int(args.warmup_epochs),
    }

    csv_path = dirs["data"] / "timing_calibration.csv"
    json_path = dirs["data"] / "timing_calibration.json"
    md_path = dirs["reports"] / "timing_calibration.md"
    write_csv(csv_path, rows)
    write_json(json_path, plan)
    write_report(md_path, rows, plan)
    print(json_path)


if __name__ == "__main__":
    main()
