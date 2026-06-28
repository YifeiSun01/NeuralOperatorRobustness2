#!/usr/bin/env python3
"""Replot existing seven-model Darcy heatmaps from saved arrays.

This only redraws PNGs/README files. It does not rerun attacks or solvers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.plot_darcy_five_model_attack_heatmaps_20260612 as heat
import tools.plot_darcy_seven_model_attack_heatmaps_20260613 as seven


PREFIX = "darcy_seven_model_attack_heatmaps_"
DEFAULT_TAG_PREFIX = "20260613_loss3attack50_seven_models_random_inclusive"
ARRAY_KEYS = [
    "x0",
    "delta",
    "x_adv",
    "model_output",
    "solver_output",
    "model_minus_solver",
    "attack_loss_history",
    "attack_loss_gain_history",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_specs(path: Path) -> list[heat.SampleSpec]:
    samples = []
    for row in read_rows(path):
        samples.append(
            heat.SampleSpec(
                row["sample_id"],
                row.get("split", "generalization"),
                row["dataset_id"],
                int(float(row["sample_index"])),
            )
        )
    return samples


def load_records(summary_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in read_rows(summary_path):
        npz_path = PROJECT_ROOT / row["npz_path"]
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        arrays = np.load(npz_path)
        loaded = {key: np.asarray(arrays[key]) for key in ARRAY_KEYS}
        rec = {
            **loaded,
            "sample_id": row["sample_id"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "sample_index": int(float(row["sample_index"])),
            "model": row["model"],
            "attack_loss_gain": float(row["attack_loss_gain"]),
            "adv_relative_l2_model_vs_solver": float(row["adv_relative_l2_model_vs_solver"]),
        }
        records.append(rec)
        out_row: dict[str, Any] = dict(row)
        for key in [
            "attack_loss_gain",
            "adv_relative_l2_model_vs_solver",
            "clean_loss_before_attack",
            "adv_loss_after_attack",
        ]:
            if key in out_row and out_row[key] != "":
                out_row[key] = float(out_row[key])
        summary_rows.append(out_row)
    return records, summary_rows


def replot_one(analysis_dir: Path, viz_root: Path) -> dict[str, Any]:
    tag = analysis_dir.name.removeprefix(PREFIX)
    viz_dir = viz_root / analysis_dir.name
    viz_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_specs(analysis_dir / "selected_samples.csv")
    records, summary_rows = load_records(analysis_dir / "summary.csv")
    ranges = json.loads((analysis_dir / "shared_color_ranges.json").read_text(encoding="utf-8"))

    heat.MODELS = seven.SEVEN_MODELS
    heat.MODEL_COLORS = seven.MODEL_COLORS
    fig_paths = [seven.plot_sample_seven(sample, records, ranges, viz_dir) for sample in samples]

    attack_steps = int(float(summary_rows[0].get("attack_steps", 50))) if summary_rows else 50
    epsilon_fraction = float(summary_rows[0].get("epsilon_fraction", 0.025)) if summary_rows else 0.025
    args = SimpleNamespace(tag=tag, attack_steps=attack_steps, epsilon_fraction=epsilon_fraction)
    seven.write_report_seven(analysis_dir, viz_dir, summary_rows, fig_paths, samples, args)
    return {"analysis_dir": heat.rel(analysis_dir), "viz_dir": heat.rel(viz_dir), "figures": len(fig_paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-prefix", default=DEFAULT_TAG_PREFIX)
    parser.add_argument("--analysis-root", type=Path, default=PROJECT_ROOT / "analysis_outputs")
    parser.add_argument("--viz-root", type=Path, default=PROJECT_ROOT / "visualizations")
    args = parser.parse_args()

    pattern = f"{PREFIX}{args.tag_prefix}*"
    dirs = sorted(p for p in args.analysis_root.glob(pattern) if (p / "summary.csv").exists() and (p / "arrays").is_dir())
    if not dirs:
        raise FileNotFoundError(f"no existing analysis dirs matched {args.analysis_root / pattern}")

    results = [replot_one(path, args.viz_root) for path in dirs]
    print(json.dumps({"replotted_dirs": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
