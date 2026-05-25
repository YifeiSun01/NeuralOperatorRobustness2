#!/usr/bin/env python3
"""Summarize NS2D loss3 smoke-test runtimes from attack summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        run_root = str(payload.get("run_root", summary_path.parent))
        for item in payload.get("summaries", []):
            metric = str(item.get("loss3_metric", "unknown"))
            steps = int(item.get("steps", 0) or 0)
            runtime = _float_or_nan(item.get("runtime_seconds"))
            dataset_indices = item.get("dataset_indices", []) or []
            samples = len(dataset_indices)
            final_metrics = item.get("final_step_metrics", {}) or {}
            row = {
                "loss3_metric": metric,
                "runtime_seconds": runtime,
                "steps": steps,
                "samples": samples,
                "seconds_per_step": runtime / steps if steps > 0 and math.isfinite(runtime) else float("nan"),
                "seconds_per_step_per_sample": runtime / (steps * samples) if steps > 0 and samples > 0 and math.isfinite(runtime) else float("nan"),
                "epsilon": item.get("epsilon"),
                "alpha": item.get("alpha"),
                "method": item.get("method"),
                "mode_spec": item.get("mode_spec"),
                "active_loss_mean_final": final_metrics.get("active_loss_mean"),
                "true_loss_mean_final": final_metrics.get("true_loss_mean"),
                "summary_json": str(summary_path),
                "run_root": run_root,
            }
            rows.append(row)
    baseline = next((row for row in rows if row.get("loss3_metric") == "qnorm" and math.isfinite(row.get("runtime_seconds", float("nan")))), None)
    baseline_runtime = baseline["runtime_seconds"] if baseline else float("nan")
    baseline_step = baseline["seconds_per_step"] if baseline else float("nan")
    for row in rows:
        runtime = row["runtime_seconds"]
        per_step = row["seconds_per_step"]
        row["runtime_ratio_vs_qnorm"] = runtime / baseline_runtime if math.isfinite(runtime) and math.isfinite(baseline_runtime) and baseline_runtime > 0 else float("nan")
        row["per_step_ratio_vs_qnorm"] = per_step / baseline_step if math.isfinite(per_step) and math.isfinite(baseline_step) and baseline_step > 0 else float("nan")
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "loss3_metric",
        "runtime_seconds",
        "steps",
        "samples",
        "seconds_per_step",
        "seconds_per_step_per_sample",
        "runtime_ratio_vs_qnorm",
        "per_step_ratio_vs_qnorm",
        "epsilon",
        "alpha",
        "method",
        "mode_spec",
        "active_loss_mean_final",
        "true_loss_mean_final",
        "summary_json",
        "run_root",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No summary.json files found.")
        return
    headers = ["metric", "seconds", "sec/step", "ratio", "samples", "steps"]
    print("	".join(headers))
    for row in sorted(rows, key=lambda item: str(item.get("loss3_metric", ""))):
        print(
            "	".join(
                [
                    str(row.get("loss3_metric", "")),
                    f"{_float_or_nan(row.get('runtime_seconds')):.3f}",
                    f"{_float_or_nan(row.get('seconds_per_step')):.3f}",
                    f"{_float_or_nan(row.get('runtime_ratio_vs_qnorm')):.3f}",
                    str(row.get("samples", "")),
                    str(row.get("steps", "")),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Smoke-test output root or any parent containing summary.json files.")
    parser.add_argument("--out", type=Path, default=None, help="CSV output path. Defaults to <root>/smoke_timing_summary.csv.")
    args = parser.parse_args()
    rows = collect_rows(args.root)
    out_path = args.out or (args.root / "smoke_timing_summary.csv")
    write_csv(rows, out_path)
    print_table(rows)
    print(f"wrote_csv={out_path}")


if __name__ == "__main__":
    main()
