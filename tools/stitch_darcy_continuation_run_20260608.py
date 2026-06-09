#!/usr/bin/env python3
"""Stitch a Darcy source self-training run and a continuation run.

The adversarial-training continuation records only the resume checkpoint and the
new local epochs. This helper builds a lightweight run directory whose CSVs cover
source epoch 0 through the continuation final epoch, so plotting and summary
scripts can report baseline-to-final metrics without rereading chat history.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_NAMES = [
    "eval_split_summary.csv",
    "eval_metrics.csv",
    "evaluation_passes.csv",
    "train_steps.csv",
    "attack_batches.csv",
    "attack_epoch_summary.csv",
    "attack_epsilon_bucket_summary.csv",
    "optimizer_steps.csv",
    "memory.csv",
    "checkpoints.csv",
    "attack_probe_epochs.csv",
    "attack_probe_samples.csv",
]
JSON_NAMES = [
    "config.json",
    "data_range_summary.json",
    "attack_probe_config.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def task_dir(run_dir: Path) -> Path:
    candidate = run_dir / "darcy"
    return candidate if candidate.exists() else run_dir


def concat_csv(source_csv: Path, continuation_csv: Path, output_csv: Path, source_epoch: int) -> None:
    frames: list[pd.DataFrame] = []
    if source_csv.exists():
        frames.append(pd.read_csv(source_csv))
    if continuation_csv.exists():
        cont = pd.read_csv(continuation_csv)
        if "epoch" in cont.columns:
            cont = cont[pd.to_numeric(cont["epoch"], errors="coerce") > source_epoch]
        frames.append(cont)
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    if "epoch" in out.columns:
        sort_cols = [col for col in ["epoch", "global_step", "global_step_last", "split", "phase"] if col in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, kind="stable")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--continuation-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--target-epoch", type=int, default=650)
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    continuation_run = args.continuation_run.resolve()
    output_run = args.output_run.resolve()
    source_task = task_dir(source_run)
    continuation_task = task_dir(continuation_run)
    output_task = output_run / "darcy"
    output_task.mkdir(parents=True, exist_ok=True)

    source_summary = read_json(source_task / "summary.json")
    continuation_summary = read_json(continuation_task / "summary.json")
    source_epoch = int(source_summary.get("epochs", 0) or 0)

    for name in CSV_NAMES:
        concat_csv(source_task / name, continuation_task / name, output_task / name, source_epoch)

    for name in JSON_NAMES:
        src = continuation_task / name
        if not src.exists():
            src = source_task / name
        if src.exists():
            shutil.copy2(src, output_task / name)

    stitched_task_summary = dict(continuation_summary)
    stitched_task_summary.update(
        {
            "stitched": True,
            "stitch_source_run": rel(source_run),
            "stitch_continuation_run": rel(continuation_run),
            "stitch_source_epoch": source_epoch,
            "stitch_target_epoch": int(args.target_epoch),
            "attack_loss_objective": args.objective,
        }
    )
    (output_task / "summary.json").write_text(json.dumps(stitched_task_summary, indent=2), encoding="utf-8")

    source_root_summary = read_json(source_run / "summary.json") if (source_run / "summary.json").exists() else {}
    continuation_root_summary = read_json(continuation_run / "summary.json") if (continuation_run / "summary.json").exists() else {}
    stitched_root_summary = dict(continuation_root_summary)
    stitched_root_summary.update(
        {
            "run_dir": rel(output_run),
            "stitched": True,
            "stitch_source_run": rel(source_run),
            "stitch_continuation_run": rel(continuation_run),
            "stitch_source_summary": rel(source_run / "summary.json") if source_root_summary else "",
            "stitch_continuation_summary": rel(continuation_run / "summary.json") if continuation_root_summary else "",
            "tasks": [stitched_task_summary],
        }
    )
    (output_run / "summary.json").write_text(json.dumps(stitched_root_summary, indent=2), encoding="utf-8")

    readme = output_run / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Darcy {args.objective} stitched continuation to epoch {args.target_epoch}",
                "",
                f"Source run: `{rel(source_run)}`",
                f"Continuation run: `{rel(continuation_run)}`",
                f"Source epoch cutoff: `{source_epoch}`",
                "",
                "This directory contains lightweight stitched CSV/JSON records for plotting and summary.",
                "Large checkpoint and attack-probe NPZ artifacts remain in the source and continuation run directories.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"output_run": rel(output_run), "source_epoch": source_epoch}, indent=2))


if __name__ == "__main__":
    main()
