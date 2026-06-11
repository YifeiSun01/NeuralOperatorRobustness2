#!/usr/bin/env python3
"""Per-sample clean generalization loss significance for Burgers smooth round00."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_burgers_neutral_generalization_final_models_20260608 import (  # noqa: E402
    MODEL_SPECS,
    load_model_for_spec,
)

MODEL_ORDER = ["loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"]
MODEL_SHORT = {
    "loss1_epoch8000": "loss1",
    "loss2_epoch2000": "loss2",
    "loss3_epoch1500": "loss3",
}
DEFAULT_DATA_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611/round_00/burgers"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics/burgers_semantic_smooth_loss3_screen_round00_clean_loss_significance_20260611"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataset_id_from_file(path: Path) -> tuple[str, dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    return str(meta.get("dataset_id", path.stem)), meta


def finite_sample_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, np.ndarray]:
    # pred/y: [B, 1024, 1]. Metrics are per sample, over space/channel.
    finite = torch.isfinite(pred) & torch.isfinite(y)
    diff = pred - y
    safe_diff = torch.where(finite, diff, torch.zeros_like(diff))
    safe_y = torch.where(finite, y, torch.zeros_like(y))
    count = finite.flatten(1).sum(dim=1).clamp_min(1)
    sse = safe_diff.square().flatten(1).sum(dim=1)
    sae = safe_diff.abs().flatten(1).sum(dim=1)
    target_sse = safe_y.square().flatten(1).sum(dim=1).clamp_min(1e-20)
    invalid = 1.0 - finite.flatten(1).float().mean(dim=1)
    mse = sse / count
    rmse = torch.sqrt(mse)
    mae = sae / count
    rel = torch.sqrt(sse / target_sse)
    return {
        "mse": mse.detach().cpu().numpy().astype(np.float64),
        "rmse": rmse.detach().cpu().numpy().astype(np.float64),
        "mae": mae.detach().cpu().numpy().astype(np.float64),
        "relative_l2": rel.detach().cpu().numpy().astype(np.float64),
        "invalid_value_fraction": invalid.detach().cpu().numpy().astype(np.float64),
    }


def eval_dataset_per_sample(path: Path, models: dict[str, Any], batch_size: int, device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    x = data["x"].float().unsqueeze(-1)
    y = data["y"].float().unsqueeze(-1)
    out: dict[str, dict[str, list[np.ndarray]]] = {m: {"mse": [], "rmse": [], "mae": [], "relative_l2": [], "invalid_value_fraction": []} for m in MODEL_ORDER}
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            xb = x[start : start + batch_size].to(device, non_blocking=True)
            yb = y[start : start + batch_size].to(device, non_blocking=True)
            for name, model in models.items():
                pred = model(xb)
                metrics = finite_sample_metrics(pred, yb)
                for key, arr in metrics.items():
                    out[name][key].append(arr)
    return {name: {key: np.concatenate(parts) for key, parts in metrics.items()} for name, metrics in out.items()}


def mean_std_var(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)),
        "var": float(np.nanvar(arr, ddof=1)),
        "sem": float(stats.sem(arr, nan_policy="omit")),
        "median": float(np.nanmedian(arr)),
    }


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    # a is loss3, b is comparison. Negative diff means loss3 is lower/better.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    diff = a - b
    n = int(diff.size)
    diff_std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    diff_mean = float(np.mean(diff)) if n else float("nan")
    diff_sem = float(diff_std / math.sqrt(n)) if n > 1 else float("nan")
    dz = float(diff_mean / diff_std) if diff_std and np.isfinite(diff_std) and diff_std > 0 else float("nan")
    t_less = stats.ttest_rel(a, b, alternative="less", nan_policy="omit")
    t_two = stats.ttest_rel(a, b, alternative="two-sided", nan_policy="omit")
    try:
        w_less = stats.wilcoxon(a, b, alternative="less", zero_method="wilcox", method="auto")
        w_p = float(w_less.pvalue)
        w_stat = float(w_less.statistic)
    except Exception:
        w_p = float("nan")
        w_stat = float("nan")
    b_mean = float(np.mean(b)) if n else float("nan")
    a_mean = float(np.mean(a)) if n else float("nan")
    return {
        "n": n,
        "loss3_mean": a_mean,
        "comparison_mean": b_mean,
        "diff_mean_loss3_minus_comparison": diff_mean,
        "diff_std": diff_std,
        "diff_sem": diff_sem,
        "mean_reduction_pct": float((b_mean - a_mean) / b_mean * 100.0) if b_mean else float("nan"),
        "loss3_win_count": int(np.sum(a < b)),
        "loss3_win_fraction": float(np.mean(a < b)) if n else float("nan"),
        "paired_t_stat_less": float(t_less.statistic),
        "paired_t_p_less": float(t_less.pvalue),
        "paired_t_p_two_sided": float(t_two.pvalue),
        "wilcoxon_stat_less": w_stat,
        "wilcoxon_p_less": w_p,
        "cohens_dz_loss3_minus_comparison": dz,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    paths = sorted(args.data_root.glob("burgers_semantic_smooth_loss3screen_d*.pt"))
    if len(paths) != 50:
        raise RuntimeError(f"expected 50 smooth selected datasets under {args.data_root}, found {len(paths)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    wanted_specs = [s for s in MODEL_SPECS if s["model"] in MODEL_ORDER]
    models = {spec["model"]: load_model_for_spec(spec, device) for spec in wanted_specs}
    for model in models.values():
        model.eval()

    per_sample_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    dataset_mean_rows: list[dict[str, Any]] = []

    for idx, path in enumerate(paths, 1):
        dataset_id, meta = dataset_id_from_file(path)
        params = meta.get("params", {}) if isinstance(meta, dict) else {}
        family = params.get("base_family", meta.get("family", "")) if isinstance(params, dict) else meta.get("family", "")
        target_min = params.get("target_min", "") if isinstance(params, dict) else ""
        target_max = params.get("target_max", "") if isinstance(params, dict) else ""
        print(f"[dataset {idx:02d}/50] {dataset_id}", flush=True)
        metrics = eval_dataset_per_sample(path, models, int(args.batch_size), device)
        n = int(next(iter(metrics.values()))["rmse"].shape[0])
        for sample_idx in range(n):
            row = {"dataset_id": dataset_id, "sample_index": sample_idx, "family": family, "target_min": target_min, "target_max": target_max}
            for model_name in MODEL_ORDER:
                short = MODEL_SHORT[model_name]
                for metric in ["mse", "rmse", "relative_l2", "mae", "invalid_value_fraction"]:
                    row[f"{short}_{metric}"] = float(metrics[model_name][metric][sample_idx])
            row["loss3_rmse_best"] = bool(row["loss3_rmse"] < row["loss1_rmse"] and row["loss3_rmse"] < row["loss2_rmse"])
            row["loss3_relative_l2_best"] = bool(row["loss3_relative_l2"] < row["loss1_relative_l2"] and row["loss3_relative_l2"] < row["loss2_relative_l2"])
            per_sample_rows.append(row)

        drow: dict[str, Any] = {"dataset_id": dataset_id, "family": family, "target_min": target_min, "target_max": target_max, "n_samples": n}
        for model_name in MODEL_ORDER:
            short = MODEL_SHORT[model_name]
            for metric in ["mse", "rmse", "relative_l2", "mae"]:
                s = mean_std_var(metrics[model_name][metric])
                for key, value in s.items():
                    drow[f"{short}_{metric}_{key}"] = value
        for metric in ["rmse", "relative_l2", "mse"]:
            for comp_name in ["loss1_epoch8000", "loss2_epoch2000"]:
                comp_short = MODEL_SHORT[comp_name]
                ps = paired_stats(metrics["loss3_epoch1500"][metric], metrics[comp_name][metric])
                for key, value in ps.items():
                    drow[f"{metric}_loss3_vs_{comp_short}_{key}"] = value
        drow["rmse_loss3_mean_best"] = bool(drow["loss3_rmse_mean"] < drow["loss1_rmse_mean"] and drow["loss3_rmse_mean"] < drow["loss2_rmse_mean"])
        drow["relative_l2_loss3_mean_best"] = bool(drow["loss3_relative_l2_mean"] < drow["loss1_relative_l2_mean"] and drow["loss3_relative_l2_mean"] < drow["loss2_relative_l2_mean"])
        dataset_rows.append(drow)
        dataset_mean_rows.append({
            "dataset_id": dataset_id,
            "family": family,
            "loss1_rmse_mean": drow["loss1_rmse_mean"],
            "loss2_rmse_mean": drow["loss2_rmse_mean"],
            "loss3_rmse_mean": drow["loss3_rmse_mean"],
            "loss1_relative_l2_mean": drow["loss1_relative_l2_mean"],
            "loss2_relative_l2_mean": drow["loss2_relative_l2_mean"],
            "loss3_relative_l2_mean": drow["loss3_relative_l2_mean"],
        })

    # Overall paired tests across all samples pooled and across dataset means.
    arrays = {m: {metric: np.array([r[f"{MODEL_SHORT[m]}_{metric}"] for r in per_sample_rows], dtype=np.float64) for metric in ["rmse", "relative_l2", "mse"]} for m in MODEL_ORDER}
    mean_arrays = {m: {metric: np.array([r[f"{MODEL_SHORT[m]}_{metric}_mean"] for r in dataset_rows], dtype=np.float64) for metric in ["rmse", "relative_l2", "mse"]} for m in MODEL_ORDER}
    overall: dict[str, Any] = {
        "data_root": str(args.data_root),
        "out_root": str(args.out_root),
        "n_datasets": len(paths),
        "n_samples_total": len(per_sample_rows),
        "dataset_level_loss3_rmse_mean_best_count": int(sum(bool(r["rmse_loss3_mean_best"]) for r in dataset_rows)),
        "dataset_level_loss3_relative_l2_mean_best_count": int(sum(bool(r["relative_l2_loss3_mean_best"]) for r in dataset_rows)),
        "sample_level_loss3_rmse_best_count": int(sum(bool(r["loss3_rmse_best"]) for r in per_sample_rows)),
        "sample_level_loss3_relative_l2_best_count": int(sum(bool(r["loss3_relative_l2_best"]) for r in per_sample_rows)),
        "per_dataset_significant_counts_rmse_p_less_0p05": {
            "loss3_vs_loss1": int(sum(float(r["rmse_loss3_vs_loss1_paired_t_p_less"]) < 0.05 and float(r["rmse_loss3_vs_loss1_diff_mean_loss3_minus_comparison"]) < 0 for r in dataset_rows)),
            "loss3_vs_loss2": int(sum(float(r["rmse_loss3_vs_loss2_paired_t_p_less"]) < 0.05 and float(r["rmse_loss3_vs_loss2_diff_mean_loss3_minus_comparison"]) < 0 for r in dataset_rows)),
        },
        "per_dataset_significant_counts_rmse_bonferroni_100_tests": {
            "threshold": 0.05 / 100.0,
            "loss3_vs_loss1": int(sum(float(r["rmse_loss3_vs_loss1_paired_t_p_less"]) < 0.05 / 100.0 and float(r["rmse_loss3_vs_loss1_diff_mean_loss3_minus_comparison"]) < 0 for r in dataset_rows)),
            "loss3_vs_loss2": int(sum(float(r["rmse_loss3_vs_loss2_paired_t_p_less"]) < 0.05 / 100.0 and float(r["rmse_loss3_vs_loss2_diff_mean_loss3_minus_comparison"]) < 0 for r in dataset_rows)),
        },
        "overall_pooled_sample_tests": {},
        "overall_dataset_mean_tests": {},
        "model_distribution_over_dataset_means": {},
    }
    for metric in ["rmse", "relative_l2", "mse"]:
        overall["overall_pooled_sample_tests"][metric] = {
            "loss3_vs_loss1": paired_stats(arrays["loss3_epoch1500"][metric], arrays["loss1_epoch8000"][metric]),
            "loss3_vs_loss2": paired_stats(arrays["loss3_epoch1500"][metric], arrays["loss2_epoch2000"][metric]),
        }
        overall["overall_dataset_mean_tests"][metric] = {
            "loss3_vs_loss1": paired_stats(mean_arrays["loss3_epoch1500"][metric], mean_arrays["loss1_epoch8000"][metric]),
            "loss3_vs_loss2": paired_stats(mean_arrays["loss3_epoch1500"][metric], mean_arrays["loss2_epoch2000"][metric]),
        }
        overall["model_distribution_over_dataset_means"][metric] = {
            MODEL_SHORT[m]: mean_std_var(mean_arrays[m][metric]) for m in MODEL_ORDER
        }

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_root / "clean_loss_per_sample.csv", per_sample_rows)
    write_csv(args.out_root / "clean_loss_per_dataset_significance.csv", dataset_rows)
    write_csv(args.out_root / "clean_loss_dataset_means_compact.csv", dataset_mean_rows)
    write_json(args.out_root / "summary.json", overall)

    lines = [
        "# Burgers Smooth Round00 Clean Loss Significance - 2026-06-11",
        "",
        f"Data root: `{args.data_root}`",
        f"Datasets: {len(paths)}, samples total: {len(per_sample_rows)}",
        "",
        "## Main Result",
        "",
        f"- Dataset mean RMSE: loss3 is best on {overall['dataset_level_loss3_rmse_mean_best_count']} / {len(paths)} datasets.",
        f"- Dataset mean relative L2: loss3 is best on {overall['dataset_level_loss3_relative_l2_mean_best_count']} / {len(paths)} datasets.",
        f"- Sample-level RMSE strict best: {overall['sample_level_loss3_rmse_best_count']} / {len(per_sample_rows)} samples.",
        f"- Sample-level relative L2 strict best: {overall['sample_level_loss3_relative_l2_best_count']} / {len(per_sample_rows)} samples.",
        "",
        "## Per-Dataset Paired RMSE Significance",
        "",
        f"- p<0.05 one-sided paired t-test, loss3 < loss1: {overall['per_dataset_significant_counts_rmse_p_less_0p05']['loss3_vs_loss1']} / {len(paths)}",
        f"- p<0.05 one-sided paired t-test, loss3 < loss2: {overall['per_dataset_significant_counts_rmse_p_less_0p05']['loss3_vs_loss2']} / {len(paths)}",
        f"- Bonferroni p<0.0005, loss3 < loss1: {overall['per_dataset_significant_counts_rmse_bonferroni_100_tests']['loss3_vs_loss1']} / {len(paths)}",
        f"- Bonferroni p<0.0005, loss3 < loss2: {overall['per_dataset_significant_counts_rmse_bonferroni_100_tests']['loss3_vs_loss2']} / {len(paths)}",
        "",
        "## Dataset-Mean RMSE Distribution",
        "",
        "```json",
        json.dumps(overall["model_distribution_over_dataset_means"]["rmse"], indent=2),
        "```",
        "",
        "## Overall Dataset-Mean Paired Tests, RMSE",
        "",
        "```json",
        json.dumps(overall["overall_dataset_mean_tests"]["rmse"], indent=2),
        "```",
        "",
        "## Files",
        "",
        "- `clean_loss_per_sample.csv`",
        "- `clean_loss_per_dataset_significance.csv`",
        "- `clean_loss_dataset_means_compact.csv`",
        "- `summary.json`",
    ]
    (args.out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
