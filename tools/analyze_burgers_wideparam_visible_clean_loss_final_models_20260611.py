#!/usr/bin/env python3
"""Clean inference loss of baseline/loss1/loss2/loss3 on wide-parameter Burgers data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

MODEL_ORDER = ["baseline", "loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"]
SHORT = {"baseline": "baseline", "loss1_epoch8000": "loss1", "loss2_epoch2000": "loss2", "loss3_epoch1500": "loss3"}
DEFAULT_DATA_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_semantic_wideparam_visible_20260611/round_00/burgers"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_round00_clean_loss_final_models_20260611"


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


def mean_std(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)),
        "sem": float(stats.sem(arr, nan_policy="omit")),
        "median": float(np.nanmedian(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    # a=model, b=baseline/other. Negative diff means a lower/better.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    diff = a - b
    n = int(diff.size)
    diff_mean = float(np.mean(diff)) if n else float("nan")
    diff_std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    t_less = stats.ttest_rel(a, b, alternative="less", nan_policy="omit")
    base_mean = float(np.mean(b)) if n else float("nan")
    model_mean = float(np.mean(a)) if n else float("nan")
    return {
        "n": n,
        "model_mean": model_mean,
        "comparison_mean": base_mean,
        "diff_mean_model_minus_comparison": diff_mean,
        "reduction_pct": float((base_mean - model_mean) / base_mean * 100.0) if base_mean else float("nan"),
        "win_count": int(np.sum(a < b)),
        "win_fraction": float(np.mean(a < b)) if n else float("nan"),
        "paired_t_less_stat": float(t_less.statistic),
        "paired_t_less_p": float(t_less.pvalue),
        "cohens_dz": float(diff_mean / diff_std) if diff_std and np.isfinite(diff_std) and diff_std > 0 else float("nan"),
    }


def sample_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, np.ndarray]:
    finite = torch.isfinite(pred) & torch.isfinite(y)
    diff = torch.where(finite, pred - y, torch.zeros_like(pred))
    yy = torch.where(finite, y, torch.zeros_like(y))
    count = finite.flatten(1).sum(dim=1).clamp_min(1)
    sse = diff.square().flatten(1).sum(dim=1)
    target_sse = yy.square().flatten(1).sum(dim=1).clamp_min(1e-20)
    mse = sse / count
    rmse = torch.sqrt(mse)
    rel = torch.sqrt(sse / target_sse)
    return {"mse": mse.cpu().numpy().astype(np.float64), "rmse": rmse.cpu().numpy().astype(np.float64), "relative_l2": rel.cpu().numpy().astype(np.float64)}


def eval_dataset(path: Path, models: dict[str, Any], batch_size: int, device: torch.device) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    meta = data.get("metadata", {})
    params = meta.get("params", {}) if isinstance(meta, dict) else {}
    x = data["x"].float().unsqueeze(-1)
    y = data["y"].float().unsqueeze(-1)
    out = {m: {"mse": [], "rmse": [], "relative_l2": []} for m in MODEL_ORDER}
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            xb = x[start : start + batch_size].to(device, non_blocking=True)
            yb = y[start : start + batch_size].to(device, non_blocking=True)
            for name, model in models.items():
                pred = model(xb)
                met = sample_metrics(pred, yb)
                for k, arr in met.items():
                    out[name][k].append(arr)
    return {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in out.items()}, {
        "dataset_id": str(meta.get("dataset_id", path.stem)),
        "family": params.get("base_family", meta.get("family", "")),
        "display_label": params.get("display_label", params.get("descriptive_name", path.stem)),
        "path": str(path),
    }


def make_plot(out_root: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: r["dataset_id"])
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    for short, color in [("baseline", "#555555"), ("loss1", "#1f77b4"), ("loss2", "#ff7f0e"), ("loss3", "#2ca02c")]:
        y = np.array([float(r[f"{short}_rmse_mean"]) for r in rows])
        axes[0].plot(x, y, "o-", ms=3, lw=1, label=short, color=color)
    axes[0].set_ylabel("clean RMSE mean")
    axes[0].set_title("Wide-parameter Burgers clean inference loss by dataset")
    axes[0].grid(alpha=.25)
    axes[0].legend(ncol=4)
    base = np.array([float(r["baseline_rmse_mean"]) for r in rows])
    for short, color in [("loss1", "#1f77b4"), ("loss2", "#ff7f0e"), ("loss3", "#2ca02c")]:
        y = np.array([float(r[f"rmse_{short}_vs_baseline_reduction_pct"]) for r in rows])
        axes[1].plot(x, y, "o-", ms=3, lw=1, label=f"{short} vs baseline", color=color)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_ylabel("RMSE reduction vs baseline %")
    axes[1].set_xlabel("dataset index")
    axes[1].grid(alpha=.25)
    axes[1].legend(ncol=3)
    fig.savefig(out_root / "wideparam_clean_rmse_and_reduction_by_dataset.png", dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    paths = sorted(args.data_root.glob("burgers_widevis_*.pt"))
    if len(paths) != 50:
        raise RuntimeError(f"expected 50 widevis datasets, found {len(paths)} under {args.data_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    specs = [s for s in MODEL_SPECS if s["model"] in MODEL_ORDER]
    models = {s["model"]: load_model_for_spec(s, device) for s in specs}
    for m in models.values():
        m.eval()

    per_sample_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_arrays_by_model = {m: {metric: [] for metric in ["rmse", "mse", "relative_l2"]} for m in MODEL_ORDER}
    dataset_mean_arrays_by_model = {m: {metric: [] for metric in ["rmse", "mse", "relative_l2"]} for m in MODEL_ORDER}

    for idx, path in enumerate(paths, 1):
        print(f"[eval {idx:02d}/50] {path.stem}", flush=True)
        metrics, info = eval_dataset(path, models, int(args.batch_size), device)
        n = len(metrics["baseline"]["rmse"])
        drow = {**info, "n_samples": n}
        for model_name in MODEL_ORDER:
            short = SHORT[model_name]
            for metric in ["rmse", "mse", "relative_l2"]:
                arr = metrics[model_name][metric]
                sample_arrays_by_model[model_name][metric].append(arr)
                dataset_mean_arrays_by_model[model_name][metric].append(float(np.mean(arr)))
                stats_dict = mean_std(arr)
                for k, v in stats_dict.items():
                    drow[f"{short}_{metric}_{k}"] = v
        for model_name in ["loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"]:
            short = SHORT[model_name]
            for metric in ["rmse", "mse", "relative_l2"]:
                ps = paired(metrics[model_name][metric], metrics["baseline"][metric])
                for k, v in ps.items():
                    drow[f"{metric}_{short}_vs_baseline_{k}"] = v
        for comp in ["loss1_epoch8000", "loss2_epoch2000"]:
            comp_short = SHORT[comp]
            for metric in ["rmse", "mse", "relative_l2"]:
                ps = paired(metrics["loss3_epoch1500"][metric], metrics[comp][metric])
                for k, v in ps.items():
                    drow[f"{metric}_loss3_vs_{comp_short}_{k}"] = v
        drow["rmse_loss3_best_among_adv"] = bool(drow["loss3_rmse_mean"] < drow["loss1_rmse_mean"] and drow["loss3_rmse_mean"] < drow["loss2_rmse_mean"])
        drow["rmse_loss3_best_all4"] = bool(drow["loss3_rmse_mean"] < drow["baseline_rmse_mean"] and drow["loss3_rmse_mean"] < drow["loss1_rmse_mean"] and drow["loss3_rmse_mean"] < drow["loss2_rmse_mean"])
        dataset_rows.append(drow)
        family_groups[str(info["family"])].append(drow)
        for si in range(n):
            row = {"dataset_id": info["dataset_id"], "family": info["family"], "sample_index": si}
            for model_name in MODEL_ORDER:
                short = SHORT[model_name]
                for metric in ["rmse", "mse", "relative_l2"]:
                    row[f"{short}_{metric}"] = float(metrics[model_name][metric][si])
            per_sample_rows.append(row)

    for model_name in MODEL_ORDER:
        for metric in ["rmse", "mse", "relative_l2"]:
            sample_arrays_by_model[model_name][metric] = np.concatenate(sample_arrays_by_model[model_name][metric])
            dataset_mean_arrays_by_model[model_name][metric] = np.array(dataset_mean_arrays_by_model[model_name][metric], dtype=np.float64)

    summary: dict[str, Any] = {
        "data_root": str(args.data_root),
        "out_root": str(args.out_root),
        "n_datasets": len(paths),
        "n_samples_total": len(per_sample_rows),
        "model_distribution_over_dataset_means": {},
        "overall_dataset_mean_vs_baseline": {},
        "overall_pooled_sample_vs_baseline": {},
        "loss3_vs_loss1_loss2": {},
        "family_summary": {},
        "counts": {
            "loss3_best_among_adv_rmse_dataset_count": int(sum(r["rmse_loss3_best_among_adv"] for r in dataset_rows)),
            "loss3_best_all4_rmse_dataset_count": int(sum(r["rmse_loss3_best_all4"] for r in dataset_rows)),
            "loss1_improves_vs_baseline_dataset_count": int(sum(float(r["rmse_loss1_vs_baseline_reduction_pct"]) > 0 for r in dataset_rows)),
            "loss2_improves_vs_baseline_dataset_count": int(sum(float(r["rmse_loss2_vs_baseline_reduction_pct"]) > 0 for r in dataset_rows)),
            "loss3_improves_vs_baseline_dataset_count": int(sum(float(r["rmse_loss3_vs_baseline_reduction_pct"]) > 0 for r in dataset_rows)),
        },
    }
    for metric in ["rmse", "mse", "relative_l2"]:
        summary["model_distribution_over_dataset_means"][metric] = {
            SHORT[m]: mean_std(dataset_mean_arrays_by_model[m][metric]) for m in MODEL_ORDER
        }
        summary["overall_dataset_mean_vs_baseline"][metric] = {
            SHORT[m]: paired(dataset_mean_arrays_by_model[m][metric], dataset_mean_arrays_by_model["baseline"][metric])
            for m in ["loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"]
        }
        summary["overall_pooled_sample_vs_baseline"][metric] = {
            SHORT[m]: paired(sample_arrays_by_model[m][metric], sample_arrays_by_model["baseline"][metric])
            for m in ["loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"]
        }
        summary["loss3_vs_loss1_loss2"][metric] = {
            "loss3_vs_loss1": paired(dataset_mean_arrays_by_model["loss3_epoch1500"][metric], dataset_mean_arrays_by_model["loss1_epoch8000"][metric]),
            "loss3_vs_loss2": paired(dataset_mean_arrays_by_model["loss3_epoch1500"][metric], dataset_mean_arrays_by_model["loss2_epoch2000"][metric]),
        }
    for fam, fam_rows in sorted(family_groups.items()):
        item = {"n_datasets": len(fam_rows)}
        for short in ["baseline", "loss1", "loss2", "loss3"]:
            vals = np.array([float(r[f"{short}_rmse_mean"]) for r in fam_rows], dtype=np.float64)
            item[f"{short}_rmse_mean"] = float(vals.mean())
        for short in ["loss1", "loss2", "loss3"]:
            vals = np.array([float(r[f"rmse_{short}_vs_baseline_reduction_pct"]) for r in fam_rows], dtype=np.float64)
            item[f"{short}_rmse_reduction_vs_baseline_pct_mean"] = float(vals.mean())
        item["loss3_best_among_adv_count"] = int(sum(r["rmse_loss3_best_among_adv"] for r in fam_rows))
        summary["family_summary"][fam] = item

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_root / "per_dataset_clean_metrics.csv", dataset_rows)
    write_csv(args.out_root / "per_sample_clean_metrics.csv", per_sample_rows)
    write_json(args.out_root / "summary.json", summary)
    make_plot(args.out_root, dataset_rows)
    report = [
        "# Burgers Wide-Parameter Visible Clean Final-Model Evaluation - 2026-06-11",
        "",
        f"Data root: `{args.data_root}`",
        f"Datasets: {len(paths)}, samples total: {len(per_sample_rows)}",
        "",
        "## Overall Dataset-Mean RMSE",
        "",
        "```json",
        json.dumps(summary["model_distribution_over_dataset_means"]["rmse"], indent=2),
        "```",
        "",
        "## RMSE Reduction Vs Baseline, Dataset Means",
        "",
        "```json",
        json.dumps(summary["overall_dataset_mean_vs_baseline"]["rmse"], indent=2),
        "```",
        "",
        "## Loss3 Vs Loss1/Loss2, Dataset Means",
        "",
        "```json",
        json.dumps(summary["loss3_vs_loss1_loss2"]["rmse"], indent=2),
        "```",
        "",
        "## Family Summary",
        "",
        "```json",
        json.dumps(summary["family_summary"], indent=2),
        "```",
    ]
    (args.out_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
