#!/usr/bin/env python3
"""Evaluate Burgers final models on one explicit generalization root."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import build_specs, evaluate_dataset  # noqa: E402
from tools.evaluate_burgers_neutral_generalization_final_models_20260608 import (  # noqa: E402
    METRICS,
    MODEL_SPECS,
    assert_neutral_burgers_specs,
    gpu_preflight,
    json_ready,
    load_model_for_spec,
    markdown_table,
    rel,
    relative_split_summary,
    relative_to_baseline,
    split_summaries,
    write_csv,
    write_json,
)


def best_model_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    gen = [r for r in rows if r.get("split") == "generalization"]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gen:
        by_dataset[str(row["dataset_id"])].append(row)
    for metric in ("rmse", "relative_l2", "mae"):
        counts: dict[str, int] = defaultdict(int)
        for _, items in by_dataset.items():
            finite = [r for r in items if math.isfinite(float(r[metric]))]
            if finite:
                winner = min(finite, key=lambda r: float(r[metric]))
                counts[str(winner["model"])] += 1
        out[metric] = dict(sorted(counts.items()))
    return out


def write_report(
    report: Path,
    root_label: str,
    neutral_root: Path,
    out_root: Path,
    provenance: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    rel_summary: list[dict[str, Any]],
    best_counts: dict[str, dict[str, int]],
    elapsed_sec: float,
) -> None:
    has_target = any(
        str(row.get("dataset_id", "")).startswith("burgers_target_")
        or str(row.get("metadata_similarity_tier", "")).startswith("target_")
        or str(row.get("manual_tier", "")).startswith("target_")
        for row in provenance["provenance_rows"]
    )
    caveat = (
        "- Caveat: this root is non-attack semantic generalization data. Some `target_*` datasets are target-band semantic candidates, so it is neutral/non-attack but not a uniform random sample from all possible distributions."
        if has_target
        else "- Caveat: this root is the master semantic generalization suite recorded in `GENERALIZATION_DATASETS.md`; it is deterministic/curated, not an attack-generated root."
    )
    lines: list[str] = [
        f"# Burgers final-model evaluation on {root_label} - 2026-06-08",
        "",
        "Status: completed GPU inference evaluation on train/test plus the requested Burgers generalization root.",
        "",
        "Observed source data:",
        f"- Generalization root: `{rel(neutral_root / 'burgers')}`.",
        "- This root contains exactly 50 Burgers `.pt` files.",
        "- Provenance check found no `attack_objective`, `attack_steps`, or `epsilon_fraction` fields in the Burgers dataset metadata.",
        f"- Tier counts from metadata: `{json.dumps(provenance['tier_counts'], sort_keys=True)}`.",
        caveat,
        "",
        "Models evaluated:",
    ]
    for spec in MODEL_SPECS:
        lines.append(f"- `{spec['model']}`: epoch `{spec['epoch']}`, checkpoint `{rel(Path(spec['checkpoint']))}`")
    lines.extend([
        "",
        "Output files:",
        f"- Per-dataset metrics: `{rel(out_root / 'per_dataset_metrics.csv')}`",
        f"- Split summary: `{rel(out_root / 'split_summary.csv')}`",
        f"- Relative-to-baseline by dataset: `{rel(out_root / 'relative_to_baseline_by_dataset.csv')}`",
        f"- Relative-to-baseline split summary: `{rel(out_root / 'relative_to_baseline_split_summary.csv')}`",
        f"- Dataset provenance audit: `{rel(out_root / 'neutral_dataset_provenance.csv')}`",
        f"- GPU preflight: `{rel(out_root / 'gpu_preflight.json')}`",
        f"- Manifest: `{rel(out_root / 'manifest.json')}`",
        "",
        f"Evaluation wall time: `{elapsed_sec:.2f}` seconds.",
        "",
        "Split Mean Raw Metrics:",
    ])
    lines.extend(markdown_table(summary_rows, ["model", "split", "dataset_count", "rmse_dataset_mean", "relative_l2_dataset_mean", "mae_dataset_mean"]))
    lines.extend(["", "Percent Change vs Baseline by Split (negative means lower loss/error):"])
    lines.extend(markdown_table(rel_summary, ["model", "split", "pct_change_rmse_dataset_mean", "pct_change_relative_l2_dataset_mean", "pct_change_mae_dataset_mean"]))
    lines.extend(["", "Per-generalization best-model counts across the 50 datasets:", "| metric | lowest-loss model counts |", "|---|---|"])
    for metric in ("rmse", "relative_l2", "mae"):
        count_text = ", ".join(f"`{model}`: {count}" for model, count in best_counts.get(metric, {}).items())
        lines.append(f"| {metric} | {count_text} |")
    lines.extend([
        "",
        "Inference:",
        "- This evaluation does not use the round03 loss3-selective attack-generated stress root.",
        "- Use the relative-to-baseline CSVs for per-dataset increase/decrease percentages and the split table above for aggregate train/test/generalization changes.",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalization-root", type=Path, required=True)
    parser.add_argument("--root-label", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-max-samples", type=int, default=0, help="0 means all samples")
    args = parser.parse_args()

    neutral_root = args.generalization_root.resolve()
    out_root = args.out_root.resolve()
    report = args.report.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    preflight = gpu_preflight()
    write_json(out_root / "gpu_preflight.json", preflight)
    device = torch.device("cuda")

    for spec in MODEL_SPECS:
        if not Path(spec["checkpoint"]).exists():
            raise FileNotFoundError(spec["checkpoint"])

    specs = [s for s in build_specs(neutral_root) if s.task == "burgers"]
    provenance = assert_neutral_burgers_specs(specs)
    write_csv(out_root / "neutral_dataset_provenance.csv", provenance["provenance_rows"])

    max_samples = None if int(args.eval_max_samples) <= 0 else int(args.eval_max_samples)
    all_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for model_spec in MODEL_SPECS:
        model = load_model_for_spec(model_spec, device)
        for spec in specs:
            result = evaluate_dataset(model, spec, device, int(args.eval_batch_size), max_samples)
            all_rows.append({
                "model": model_spec["model"],
                "model_epoch": model_spec["epoch"],
                "checkpoint_path": rel(Path(model_spec["checkpoint"])),
                **result,
            })
        del model
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t0

    summary_rows = split_summaries(all_rows)
    rel_rows = relative_to_baseline(all_rows)
    rel_summary = relative_split_summary(summary_rows)
    best_counts = best_model_counts(all_rows)

    write_csv(out_root / "per_dataset_metrics.csv", all_rows)
    write_csv(out_root / "split_summary.csv", summary_rows)
    write_csv(out_root / "relative_to_baseline_by_dataset.csv", rel_rows)
    write_csv(out_root / "relative_to_baseline_split_summary.csv", rel_summary)

    manifest = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root_label": args.root_label,
        "neutral_root": neutral_root / "burgers",
        "dataset_count": len([s for s in specs if s.split == "generalization"]),
        "total_eval_specs": len(specs),
        "eval_batch_size": int(args.eval_batch_size),
        "eval_max_samples": max_samples,
        "models": MODEL_SPECS,
        "outputs": {
            "per_dataset_metrics": out_root / "per_dataset_metrics.csv",
            "split_summary": out_root / "split_summary.csv",
            "relative_to_baseline_by_dataset": out_root / "relative_to_baseline_by_dataset.csv",
            "relative_to_baseline_split_summary": out_root / "relative_to_baseline_split_summary.csv",
            "neutral_dataset_provenance": out_root / "neutral_dataset_provenance.csv",
            "gpu_preflight": out_root / "gpu_preflight.json",
            "report": report,
        },
        "elapsed_sec": elapsed,
        "provenance_tier_counts": provenance["tier_counts"],
        "best_model_counts": best_counts,
    }
    write_json(out_root / "manifest.json", manifest)
    write_report(report, args.root_label, neutral_root, out_root, provenance, summary_rows, rel_summary, best_counts, elapsed)
    print(json.dumps(json_ready(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
