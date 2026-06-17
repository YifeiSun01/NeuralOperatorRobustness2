#!/usr/bin/env python3
"""Evaluate Darcy/SIR20 baseline and trained checkpoints on all 52 datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from darcy_sir20_common import (
    GENERALIZATION_ROOT,
    METHOD_ORDER,
    SCREEN_TEST_PATH,
    SCREEN_TRAIN_PATH,
    default_bundle_root,
    ensure_bundle_dirs,
    rel,
    validate_inputs,
    write_csv,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import (
    DatasetSpec,
    build_combined_eval_cache,
    build_specs,
    evaluate_combined_eval_cache,
    evaluate_dataset,
)  # noqa: E402
import tools.adversarial_training as adv  # noqa: E402


def load_checkpoint_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints", payload if isinstance(payload, list) else [])
    out = []
    for row in rows:
        if row.get("method") and row.get("checkpoint"):
            out.append(row)
    return out


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def darcy_specs() -> list[DatasetSpec]:
    specs = [s for s in build_specs(GENERALIZATION_ROOT.resolve()) if s.task == "darcy"]
    out: list[DatasetSpec] = []
    for spec in specs:
        if spec.split == "train":
            out.append(DatasetSpec(spec.task, "train_screen_binary_grf_alpha2_tau3_n384", spec.split, SCREEN_TRAIN_PATH, "screen_20260607", spec.manual_tier, spec.manual_rank))
        elif spec.split == "test":
            out.append(DatasetSpec(spec.task, "test_screen_binary_grf_alpha2_tau3_n96", spec.split, SCREEN_TEST_PATH, "screen_20260607", spec.manual_tier, spec.manual_rank))
        else:
            out.append(spec)
    out.sort(key=lambda s: (0 if s.split == "train" else 1 if s.split == "test" else 2, float(s.manual_rank), s.dataset_id))
    if len(out) != 52:
        raise RuntimeError(f"expected 52 Darcy datasets, found {len(out)}")
    return out


def load_model(checkpoint: Path, device: torch.device):
    model = adv.load_model("darcy", device, model_checkpoint_override=checkpoint)
    model.eval()
    return model


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    out = []
    for (method, split), sub in df.groupby(["method", "split"], sort=False):
        out.append(
            {
                "method": method,
                "split": split,
                "dataset_count": int(sub["dataset_id"].nunique()),
                "sample_count": int(sub["num_samples_evaluated"].sum()),
                "rmse_mean": float(sub["rmse"].astype(float).mean()),
                "rmse_var": float(sub["rmse"].astype(float).var(ddof=0)),
                "rmse_std": float(sub["rmse"].astype(float).std(ddof=0)),
                "relative_l2_mean": float(sub["relative_l2"].astype(float).mean()),
                "relative_l2_var": float(sub["relative_l2"].astype(float).var(ddof=0)),
                "relative_l2_std": float(sub["relative_l2"].astype(float).std(ddof=0)),
                "mae_mean": float(sub["mae"].astype(float).mean()),
                "mae_var": float(sub["mae"].astype(float).var(ddof=0)),
                "mae_std": float(sub["mae"].astype(float).std(ddof=0)),
                "accuracy_score_mean": float(sub["accuracy_score"].astype(float).mean()),
                "accuracy_score_var": float(sub["accuracy_score"].astype(float).var(ddof=0)),
                "accuracy_score_std": float(sub["accuracy_score"].astype(float).std(ddof=0)),
            }
        )
    for method, sub in df.groupby("method", sort=False):
        out.append(
            {
                "method": method,
                "split": "ALL",
                "dataset_count": int(sub["dataset_id"].nunique()),
                "sample_count": int(sub["num_samples_evaluated"].sum()),
                "rmse_mean": float(sub["rmse"].astype(float).mean()),
                "rmse_var": float(sub["rmse"].astype(float).var(ddof=0)),
                "rmse_std": float(sub["rmse"].astype(float).std(ddof=0)),
                "relative_l2_mean": float(sub["relative_l2"].astype(float).mean()),
                "relative_l2_var": float(sub["relative_l2"].astype(float).var(ddof=0)),
                "relative_l2_std": float(sub["relative_l2"].astype(float).std(ddof=0)),
                "mae_mean": float(sub["mae"].astype(float).mean()),
                "mae_var": float(sub["mae"].astype(float).var(ddof=0)),
                "mae_std": float(sub["mae"].astype(float).std(ddof=0)),
                "accuracy_score_mean": float(sub["accuracy_score"].astype(float).mean()),
                "accuracy_score_var": float(sub["accuracy_score"].astype(float).var(ddof=0)),
                "accuracy_score_std": float(sub["accuracy_score"].astype(float).std(ddof=0)),
            }
        )
    return out


def write_report(path: Path, summary_rows: list[dict[str, Any]], metrics_csv: Path, map_csv: Path | None = None) -> None:
    lines = [
        "# Darcy/SIR20 Final 52-Dataset Evaluation",
        "",
        "Observed from baseline plus final trained checkpoints on screen train/test and the 50 binary loss3-targeted 20260611 generalization datasets.",
        "",
        "| method | split | datasets | RMSE mean | Relative L2 mean | accuracy score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["split"] not in {"train", "test", "generalization", "ALL"}:
            continue
        lines.append(
            f"| {row['method']} | {row['split']} | {row['dataset_count']} | "
            f"{row['rmse_mean']:.8g} | {row['relative_l2_mean']:.8g} | {row['accuracy_score_mean']:.6g} |"
        )
    lines.extend(["", "## Files", "", f"- Metrics CSV: `{rel(metrics_csv)}`"])
    if map_csv is not None:
        lines.append(f"- Combined dataset index map: `{rel(map_csv)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--combined-eval", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    validate_inputs()
    bundle = (args.bundle or default_bundle_root()).resolve()
    dirs = ensure_bundle_dirs(bundle)
    rows = load_checkpoint_manifest(args.checkpoint_manifest.resolve())
    specs = darcy_specs()
    max_samples = None if int(args.eval_max_samples) <= 0 else int(args.eval_max_samples)
    device = torch.device(args.device)

    metric_rows: list[dict[str, Any]] = []
    combined_map_rows: list[dict[str, Any]] = []
    combined_eval_cache = build_combined_eval_cache(specs, max_samples) if bool(args.combined_eval) else None
    if combined_eval_cache is not None:
        combined_map_rows = list(combined_eval_cache.get("map_rows", []))
    t0 = time.perf_counter()
    for manifest_row in rows:
        method = str(manifest_row["method"])
        checkpoint = resolve(str(manifest_row["checkpoint"]))
        model = load_model(checkpoint, device)
        if combined_eval_cache is not None:
            eval_results, _ = evaluate_combined_eval_cache(model, combined_eval_cache, device, int(args.batch_size))
        else:
            eval_results = [evaluate_dataset(model, spec, device, int(args.batch_size), max_samples) for spec in specs]
        for result in eval_results:
            result.update(
                {
                    "method": method,
                    "display_name": manifest_row.get("display_name", method),
                    "checkpoint": rel(checkpoint),
                    "eval_elapsed_since_start_sec": time.perf_counter() - t0,
                }
            )
            metric_rows.append(result)
            print(f"[eval] {method:13s} {result['split']:14s} {result['dataset_id']} rel_l2={result['relative_l2']:.6g}", flush=True)
        del model
        torch.cuda.empty_cache()

    summary_rows = summarize(metric_rows)
    metrics_csv = dirs["data"] / "final_eval_metrics.csv"
    summary_csv = dirs["data"] / "final_eval_summary_by_model_split.csv"
    map_csv = dirs["data"] / "final_eval_combined_dataset_index_map.csv"
    write_csv(metrics_csv, metric_rows)
    write_csv(summary_csv, summary_rows)
    if combined_map_rows:
        write_csv(map_csv, combined_map_rows)
    write_json(dirs["data"] / "final_eval_metrics.json", metric_rows)
    write_report(dirs["reports"] / "final_evaluation.md", summary_rows, metrics_csv, map_csv if combined_map_rows else None)
    print(metrics_csv)


if __name__ == "__main__":
    main()
