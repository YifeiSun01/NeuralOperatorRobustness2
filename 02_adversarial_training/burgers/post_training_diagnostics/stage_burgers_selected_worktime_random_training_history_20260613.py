#!/usr/bin/env python3
"""Stage stitched random-field training histories for selected-worktime plots.

The selected comparison uses:
- random_clean_y at epoch 8000
- random_solver_y at epoch 6000

Their training logs are split across continuation directories. This script
concatenates those existing CSV logs into two synthetic run directories that the
existing Burgers training-curve plotter can read. It does not train, evaluate, or
modify checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO / "adversarial_training_runs"
DEFAULT_OUT_ROOT = RUN_ROOT / "burgers_wideparam_selected_worktime_random_history_20260613"

SEGMENTS = {
    "random_clean_y": [
        RUN_ROOT / "burgers_wideparam_random_field_clean_y_2000ep_20260612",
        RUN_ROOT / "burgers_wideparam_random_field_clean_y_4000ep_continue_20260613",
        RUN_ROOT / "burgers_wideparam_random_field_clean_y_6000ep_continue_20260613",
        RUN_ROOT / "burgers_wideparam_random_field_clean_y_8000ep_continue_20260613",
    ],
    "random_solver_y": [
        RUN_ROOT / "burgers_wideparam_random_field_solver_y_2000ep_20260612",
        RUN_ROOT / "burgers_wideparam_random_field_solver_y_4000ep_continue_20260613",
        RUN_ROOT / "burgers_wideparam_random_field_solver_y_6000ep_continue_20260613",
    ],
}

CSV_FILES = [
    "train_steps.csv",
    "optimizer_steps.csv",
    "memory.csv",
    "evaluation_passes.csv",
    "eval_split_summary.csv",
    "eval_metrics.csv",
    "attack_probe_samples.csv",
    "attack_probe_epochs.csv",
    "attack_epoch_summary.csv",
    "attack_epsilon_bucket_summary.csv",
    "attack_batches.csv",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def read_csv(path: Path, segment_index: int, segment_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "source_segment_index", segment_index)
    df.insert(1, "source_segment_dir", rel(segment_dir))
    return df


def dedupe_and_sort(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    sort_cols = [col for col in ["epoch", "global_step", "split", "sample_index", "epsilon", "local_batch_idx"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    candidates = {
        "eval_split_summary.csv": ["phase", "epoch", "global_step", "split"],
        "evaluation_passes.csv": ["phase", "epoch", "global_step"],
        "train_steps.csv": ["epoch", "global_step", "local_batch_idx"],
        "optimizer_steps.csv": ["epoch", "global_step", "optimizer_global_step"],
        "memory.csv": ["phase", "epoch", "global_step"],
        "eval_metrics.csv": ["phase", "epoch", "global_step", "split", "dataset_index"],
        "attack_probe_samples.csv": ["epoch", "global_step", "sample_index"],
        "attack_probe_epochs.csv": ["epoch", "global_step"],
        "attack_epoch_summary.csv": ["epoch", "global_step"],
        "attack_epsilon_bucket_summary.csv": ["epoch", "global_step", "epsilon_bucket"],
        "attack_batches.csv": ["epoch", "global_step", "local_batch_idx"],
    }
    cols = [col for col in candidates.get(filename, []) if col in df.columns]
    if cols:
        df = df.drop_duplicates(subset=cols, keep="last")
    return df


def stitch_csv(filename: str, segments: list[Path], out_task_dir: Path) -> dict[str, Any] | None:
    frames = []
    for idx, segment in enumerate(segments):
        path = segment / "burgers" / filename
        if path.exists() and path.stat().st_size > 0:
            frames.append(read_csv(path, idx, segment))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = dedupe_and_sort(df, filename)
    out_path = out_task_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {"file": filename, "rows": int(len(df)), "path": rel(out_path)}


def build_summary(label: str, segments: list[Path], out_task_dir: Path) -> dict[str, Any]:
    summaries = [read_json(segment / "burgers" / "summary.json") for segment in segments]
    summaries = [summary for summary in summaries if summary]
    merged = dict(summaries[-1]) if summaries else {"task": "burgers"}
    elapsed = sum(finite_float(summary.get("elapsed_seconds")) for summary in summaries)
    final_epoch = max(int(summary.get("total_steps", summary.get("resume_global_step_offset", 0) + summary.get("epochs", 0))) for summary in summaries)
    merged.update(
        {
            "task": "burgers",
            "run_dir": rel(out_task_dir),
            "stitched_history_label": label,
            "stitched_history": True,
            "segments": [rel(segment) for segment in segments],
            "epochs": final_epoch,
            "total_steps": final_epoch,
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0,
            "elapsed_hours": elapsed / 3600.0,
        }
    )
    out_task_dir.mkdir(parents=True, exist_ok=True)
    (out_task_dir / "summary.json").write_text(json.dumps(merged, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return merged


def stage_label(label: str, segments: list[Path], out_root: Path) -> dict[str, Any]:
    out_task_dir = out_root / label / "burgers"
    if out_task_dir.exists():
        shutil.rmtree(out_task_dir)
    out_task_dir.mkdir(parents=True, exist_ok=True)
    csv_results = []
    for filename in CSV_FILES:
        result = stitch_csv(filename, segments, out_task_dir)
        if result is not None:
            csv_results.append(result)
    summary = build_summary(label, segments, out_task_dir)
    return {
        "label": label,
        "run_dir": rel(out_task_dir.parent),
        "task_dir": rel(out_task_dir),
        "segments": [rel(segment) for segment in segments],
        "csv_files": csv_results,
        "epochs": summary.get("epochs"),
        "elapsed_hours": summary.get("elapsed_hours"),
        "final_checkpoint": summary.get("final_checkpoint"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"out_root": rel(out_root), "models": []}
    for label, segments in SEGMENTS.items():
        missing = [segment for segment in segments if not (segment / "burgers" / "summary.json").exists()]
        if missing:
            raise FileNotFoundError(f"{label} missing segment summaries: " + ", ".join(rel(path) for path in missing))
        manifest["models"].append(stage_label(label, segments, out_root))
    (out_root / "stitch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
