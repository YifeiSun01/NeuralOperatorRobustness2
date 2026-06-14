#!/usr/bin/env python3
"""Combine old four-model Burgers results with selected-worktime random models.

Selected random checkpoints:
- random_clean_y: epoch8000
- random_solver_y: epoch6000

The script intentionally does not recompute baseline/loss1/loss2/loss3. It reads
their existing summary tables and replaces only the two random-model rows/columns
with the newly evaluated selected-worktime suite.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]

OLD_SUMMARY = REPO / "forensics/burgers_six_model_latest_wideparam_summary_20260613"
OLD_CLEAN = OLD_SUMMARY / "clean_52dataset_six_models_detailed.csv"
OLD_ROBUST = OLD_SUMMARY / "robustness_25sample_six_models_detailed_with_jerror_transpose_error.csv"
OLD_ATTACK4 = REPO / "forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608/summary_by_model_dataset.csv"
DEFAULT_RANDOM_ROOT = REPO / "forensics/burgers_random_field_selected_worktime_full_suite_20260613"
DEFAULT_OUT = REPO / "forensics/burgers_six_model_selected_worktime_summary_20260613"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
OLD4 = set(MODEL_ORDER[:4])
RANDOM = set(MODEL_ORDER[4:])
MODEL_LABELS = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "random_clean_y": "random clean Y",
    "random_solver_y": "random solver Y",
}
MODEL_COLORS = {
    "baseline": "#2f6f9f",
    "loss1": "#2f9b75",
    "loss2": "#d98a2b",
    "loss3": "#c35b5b",
    "random_clean_y": "#7b5fb3",
    "random_solver_y": "#4f9a9a",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def safe_corr(xs: list[float], ys: list[float], method: str) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic)


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def model_bar(
    ax: plt.Axes,
    values: dict[str, float],
    title: str,
    ylabel: str,
    *,
    log_y: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    xs = np.arange(len(MODEL_ORDER))
    ys = [values.get(model, math.nan) for model in MODEL_ORDER]
    colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    ax.bar(xs, ys, color=colors, width=0.72)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=24, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    if log_y:
        positives = [v for v in ys if math.isfinite(v) and v > 0]
        if positives:
            ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)


def long_model_means(rows: list[dict[str, Any]], metric: str, split: str | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for model in MODEL_ORDER:
        vals = []
        for row in rows:
            if row.get("model") != model:
                continue
            if split is not None and row.get("split") != split and row.get("source_split") != split:
                continue
            vals.append(finite(row.get(metric)))
        out[model] = stats(vals)["mean"]
    return out


def write_clean_plots(clean_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    gen_rows = [row for row in clean_rows if row.get("split") == "generalization"]
    for metric, title, ylabel in [
        ("rmse", "Generalization clean RMSE", "RMSE"),
        ("relative_l2", "Generalization clean relative L2", "Relative L2"),
    ]:
        means = {}
        medians = {}
        for model in MODEL_ORDER:
            vals = [finite(row.get(f"{model}_{metric}_mean")) for row in gen_rows]
            st = stats(vals)
            means[model] = st["mean"]
            medians[model] = st["median"]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
        model_bar(axes[0], means, f"{title} mean over generalization datasets", ylabel)
        model_bar(axes[1], medians, f"{title} median over generalization datasets", ylabel)
        fig.suptitle("Burgers selected-worktime six-model clean generalization", fontsize=12, weight="bold")
        out = plot_dir / f"clean_generalization_{metric}_six_models.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_attack_plots(attack_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for split in ("generalization", None):
        suffix = "generalization" if split else "all52"
        title_suffix = "generalization datasets" if split else "all 52 datasets"
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
        model_bar(
            axes[0],
            long_model_means(attack_rows, "attack_loss_increase_mean", split),
            f"P2Q2 attack loss increase, {title_suffix}",
            "loss increase",
            log_y=True,
        )
        model_bar(
            axes[1],
            long_model_means(attack_rows, "final_delta_rms_mean", split),
            f"P2Q2 final delta RMS, {title_suffix}",
            "delta RMS",
            log_y=False,
        )
        fig.suptitle("Burgers selected-worktime six-model adversarial attack summary", fontsize=12, weight="bold")
        out = plot_dir / f"attack_{suffix}_six_models.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_robustness_plots(robust_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    norm_metrics = [
        ("attack_loss_increase", "attack loss increase", "loss increase"),
        ("error_spectral_norm", "J_error spectral norm", "spectral norm"),
        ("error_fro_norm_comparable", "J_error Frobenius norm", "Frobenius norm"),
        ("bias_gradient_norm", "||J_error^T error||", "L2 norm"),
        ("j_error_delta_l2", "||J_error delta||", "L2 norm"),
        ("delta_l2", "||delta||", "L2 norm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.ravel(), norm_metrics, strict=False):
        model_bar(ax, long_model_means(robust_rows, metric), title, ylabel, log_y=True)
    fig.suptitle("Burgers selected-worktime 25-sample robustness norms", fontsize=13, weight="bold")
    out = plot_dir / "robustness_norm_metrics_25sample_six_models.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    direction_metrics = [
        ("delta_top_error_sv_abs_cos", "cos(delta, top error right singular vector)"),
        ("model_solver_top20_right_subspace_mean_cos", "top-20 right subspace mean cos"),
        ("model_solver_top20_left_subspace_mean_cos", "top-20 left subspace mean cos"),
        ("model_solver_top1_right_abs_cos", "top-1 right singular vector abs cos"),
        ("model_solver_top1_left_abs_cos", "top-1 left singular vector abs cos"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), direction_metrics, strict=False):
        model_bar(ax, long_model_means(robust_rows, metric), title, "cosine", ylim=(0, 1.02))
    for ax in axes.ravel()[len(direction_metrics) :]:
        ax.axis("off")
    fig.suptitle("Burgers selected-worktime 25-sample direction/subspace similarity", fontsize=13, weight="bold")
    out = plot_dir / "robustness_direction_metrics_25sample_six_models.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)
    return outputs


def write_correlation_plots(out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    corr_path = out_dir / "metric_correlations_with_attack_selected_worktime_25sample.csv"
    rank_path = out_dir / "per_sample_model_rank_similarity_selected_worktime_25sample.csv"
    outputs: list[Path] = []
    if corr_path.exists():
        rows = read_csv(corr_path)
        metrics = [row["metric"] for row in rows]
        y = np.arange(len(metrics))
        pearson = [finite(row.get("pearson_with_attack_loss_increase")) for row in rows]
        spearman = [finite(row.get("spearman_with_attack_loss_increase")) for row in rows]
        fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(metrics))), constrained_layout=True)
        ax.barh(y - 0.18, pearson, height=0.34, label="Pearson", color="#4f7cac")
        ax.barh(y + 0.18, spearman, height=0.34, label="Spearman", color="#c97944")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics, fontsize=8)
        ax.set_xlim(-1.0, 1.0)
        ax.axvline(0, color="#222222", lw=0.8)
        ax.grid(axis="x", alpha=0.25, lw=0.7)
        ax.legend(fontsize=9)
        ax.set_title("Correlation with attack loss increase across six models x 25 samples", fontsize=12, weight="bold")
        out = plot_dir / "metric_correlations_with_attack_25sample_six_models.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)
    if rank_path.exists():
        rows = read_csv(rank_path)
        metrics = sorted({row.get("metric", "") for row in rows if row.get("metric", "")})
        means = {}
        for metric in metrics:
            vals = [finite(row.get("spearman_across_models_with_attack_loss_increase")) for row in rows if row.get("metric") == metric]
            means[metric] = stats(vals)["mean"]
        y = np.arange(len(metrics))
        fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(metrics))), constrained_layout=True)
        ax.barh(y, [means[m] for m in metrics], color="#5c8d70")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics, fontsize=8)
        ax.set_xlim(-1.0, 1.0)
        ax.axvline(0, color="#222222", lw=0.8)
        ax.grid(axis="x", alpha=0.25, lw=0.7)
        ax.set_title("Mean per-sample model-rank similarity with attack loss increase", fontsize=12, weight="bold")
        out = plot_dir / "per_sample_model_rank_similarity_with_attack_25sample_six_models.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_summary_plots(
    out_dir: Path,
    clean_rows: list[dict[str, Any]],
    attack_rows: list[dict[str, Any]],
    robust_rows: list[dict[str, Any]],
) -> list[Path]:
    outputs: list[Path] = []
    if clean_rows:
        outputs.extend(write_clean_plots(clean_rows, out_dir))
    if attack_rows:
        outputs.extend(write_attack_plots(attack_rows, out_dir))
    if robust_rows:
        outputs.extend(write_robustness_plots(robust_rows, out_dir))
        outputs.extend(write_correlation_plots(out_dir))
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "plot_count": len(outputs),
        "plots": [str(path.relative_to(out_dir)) for path in outputs],
    }
    (plot_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return outputs


def build_clean_table(random_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    old_rows = read_csv(OLD_CLEAN)
    random_rows = read_csv(random_root / "clean_loss/per_dataset_clean_metrics.csv")
    random_by_dataset = {row["dataset_id"]: row for row in random_rows}
    out: list[dict[str, Any]] = []
    for old in old_rows:
        row: dict[str, Any] = dict(old)
        rnd = random_by_dataset.get(row["dataset_id"])
        if rnd is None:
            out.append(row)
            continue
        row["random_n_samples"] = rnd.get("num_samples", row.get("random_n_samples", ""))
        for model in RANDOM:
            for metric in ("rmse", "relative_l2", "mse"):
                row[f"{model}_{metric}_mean"] = rnd.get(f"{model}_{metric}_mean", "")
        for metric in ("rmse", "relative_l2", "mse"):
            vals = [(model, finite(row.get(f"{model}_{metric}_mean"))) for model in MODEL_ORDER]
            vals = [(m, v) for m, v in vals if math.isfinite(v)]
            row[f"best_{metric}_model"] = min(vals, key=lambda x: x[1])[0] if vals else ""
        out.append(row)
    write_csv(out_dir / "clean_52dataset_six_models_selected_worktime.csv", out)
    gen = [r for r in out if r.get("split") == "generalization"]
    gen_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        item: dict[str, Any] = {"model": model, "generalization_dataset_count": len(gen)}
        for metric in ("rmse", "relative_l2", "mse"):
            st = stats([finite(r.get(f"{model}_{metric}_mean")) for r in gen])
            for key, value in st.items():
                item[f"{metric}_{key}"] = value
        gen_rows.append(item)
    write_csv(out_dir / "clean_generalization_model_summary_selected_worktime.csv", gen_rows)
    return out


def infer_split(dataset_id: str) -> str:
    if dataset_id.startswith("train"):
        return "train"
    if dataset_id.startswith("test"):
        return "test"
    return "generalization"


def build_attack_table(random_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(OLD_ATTACK4):
        model = row.get("model", "")
        if model not in OLD4:
            continue
        rows.append(
            {
                "model": model,
                "dataset_index": row.get("dataset_index", ""),
                "split": infer_split(row.get("dataset_id", "")),
                "dataset_id": row.get("dataset_id", ""),
                "sample_count": row.get("sample_count", ""),
                "initial_loss_mean": row.get("initial_loss_mean", ""),
                "final_loss_mean": row.get("final_loss_mean", ""),
                "attack_loss_increase_mean": finite(row.get("final_loss_mean")) - finite(row.get("initial_loss_mean")),
                "final_delta_rms_mean": row.get("final_delta_rms_mean", ""),
                "source": "old4_existing",
            }
        )
    random_attack = random_root / "p2q2_attack/summary_by_model_dataset.csv"
    if random_attack.exists():
        for row in read_csv(random_attack):
            model = row.get("model", "")
            if model not in RANDOM:
                continue
            rows.append(
                {
                    "model": model,
                    "dataset_index": row.get("dataset_index", ""),
                    "split": row.get("split", infer_split(row.get("dataset_id", ""))),
                    "dataset_id": row.get("dataset_id", ""),
                    "sample_count": row.get("sample_count", ""),
                    "initial_loss_mean": row.get("initial_loss_mean", ""),
                    "final_loss_mean": row.get("final_loss_mean", ""),
                    "attack_loss_increase_mean": row.get("attack_increase_mean", ""),
                    "final_delta_rms_mean": row.get("final_delta_rms_mean", ""),
                    "source": "selected_random_worktime",
                }
            )
    write_csv(out_dir / "attack_52dataset_six_models_selected_worktime_long.csv", rows)
    return rows


def random_postprocess_to_old_columns(row: dict[str, str], old_fieldnames: list[str]) -> dict[str, Any]:
    out = {key: "" for key in old_fieldnames}
    out.update(
        {
            "sample_id": row.get("sample_id", ""),
            "source_split": row.get("source_split", ""),
            "dataset_id": row.get("dataset_id", ""),
            "local_index": row.get("local_index", ""),
            "model": row.get("model", ""),
            "attack_initial_mse": row.get("attack_initial_loss", ""),
            "attack_final_mse": row.get("attack_final_loss", ""),
            "attack_loss_increase": row.get("attack_loss_increase", ""),
            "attack_final_delta_rms": row.get("attack_delta_rms", ""),
            "delta_l2": row.get("attack_delta_l2", ""),
            "j_error_delta_l2": row.get("j_error_delta_l2", ""),
            "j_error_delta_rms": row.get("j_error_delta_rms", ""),
            "error_spectral_norm": row.get("error_spectral_norm", ""),
            "error_fro_norm": row.get("error_fro_norm_full_matrix", ""),
            "error_fro_norm_comparable": row.get("error_fro_norm_full_matrix", ""),
            "error_fro_norm_true_from_full_jacobian": row.get("error_fro_norm_full_matrix", ""),
            "error_fro_norm_original_reported": row.get("error_fro_norm_topk_svd_reported", ""),
            "error_effective_rank": row.get("error_effective_rank_topk_only", ""),
            "effective_rank_basis": "topk_svd_energy_distribution",
            "model_solver_top20_left_subspace_mean_cos": row.get("model_solver_top20_left_mean_principal_cos", ""),
            "model_solver_top20_right_subspace_mean_cos": row.get("model_solver_top20_right_mean_principal_cos", ""),
            "model_solver_top10_left_subspace_mean_cos": row.get("model_solver_top10_left_mean_principal_cos", ""),
            "model_solver_top10_right_subspace_mean_cos": row.get("model_solver_top10_right_mean_principal_cos", ""),
            "model_solver_top5_left_subspace_mean_cos": row.get("model_solver_top5_left_mean_principal_cos", ""),
            "model_solver_top5_right_subspace_mean_cos": row.get("model_solver_top5_right_mean_principal_cos", ""),
            "model_solver_top1_left_abs_cos": row.get("model_solver_top1_left_mean_principal_cos", ""),
            "model_solver_top1_right_abs_cos": row.get("model_solver_top1_right_mean_principal_cos", ""),
            "delta_top_error_sv_abs_cos": row.get("attack_delta_abs_cos_top_error_right", ""),
            "error_top_right_delta_abs_cos": row.get("attack_delta_abs_cos_top_error_right", ""),
            "attack_delta_svd_abs_cos": row.get("attack_delta_abs_cos_top_error_right", ""),
            "top_error_sv_value": row.get("error_sv_01", ""),
            "top_error_singular_value": row.get("error_sv_01", ""),
            "bias_gradient_norm": row.get("bias_gradient_norm", ""),
            "bias_gradient_rms": row.get("bias_gradient_rms", ""),
            "j_error_transpose_error_l2": row.get("bias_gradient_norm", ""),
            "j_error_transpose_error_rms": row.get("bias_gradient_rms", ""),
            "clean_residual_mse_recomputed": row.get("clean_residual_mse", ""),
            "clean_residual_norm_l2": row.get("clean_residual_l2", ""),
            "random_clean_residual_mse_recomputed": row.get("clean_residual_mse", ""),
            "random_clean_residual_norm_l2": row.get("clean_residual_l2", ""),
            "random_clean_residual_rms": row.get("clean_residual_rmse", ""),
            "random_bias_gradient_norm": row.get("bias_gradient_norm", ""),
            "random_bias_gradient_rms": row.get("bias_gradient_rms", ""),
            "random_j_error_transpose_error_l2": row.get("bias_gradient_norm", ""),
            "random_j_error_transpose_error_rms": row.get("bias_gradient_rms", ""),
            "svd_method": row.get("svd_method", ""),
            "stored_singular_value_count": row.get("svd_top_k", ""),
            "svd_vector_source": "selected_worktime_random_npz",
            "atb_or_jerr_delta_quantity": "J_error.T @ clean_error and separate J_error @ delta",
            "atb_norm": row.get("bias_gradient_norm", ""),
            "bias_gradient_definition": "J_error.T @ clean_error, clean_error = model(x)-solver(x)",
            "checkpoint_path": row.get("checkpoint_path", ""),
        }
    )
    return out


def build_robust_table(random_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    old_rows = read_csv(OLD_ROBUST)
    old_fieldnames = list(old_rows[0].keys()) if old_rows else []
    selected = [dict(row) for row in old_rows if row.get("model") in OLD4]
    random_path = random_root / "postprocess/svd_attack_bias_gradient_by_sample.csv"
    if random_path.exists():
        for row in read_csv(random_path):
            if row.get("model") in RANDOM:
                selected.append(random_postprocess_to_old_columns(row, old_fieldnames))
    extras = ["checkpoint_path"]
    fieldnames = old_fieldnames + [key for key in extras if key not in old_fieldnames]
    write_csv(out_dir / "robustness_25sample_six_models_selected_worktime.csv", selected, fieldnames)
    return selected


def build_metric_summaries(rows: list[dict[str, Any]], out_dir: Path) -> None:
    metrics = [
        "attack_loss_increase",
        "clean_residual_mse_recomputed",
        "error_spectral_norm",
        "error_fro_norm_comparable",
        "bias_gradient_norm",
        "j_error_delta_l2",
        "delta_l2",
        "delta_top_error_sv_abs_cos",
        "model_solver_top20_right_subspace_mean_cos",
        "model_solver_top20_left_subspace_mean_cos",
    ]
    model_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = [row for row in rows if row.get("model") == model]
        item: dict[str, Any] = {"model": model, "sample_count": len(subset)}
        for metric in metrics:
            st = stats([finite(row.get(metric)) for row in subset])
            item[metric] = st["mean"]
            item[f"{metric}_median"] = st["median"]
        model_rows.append(item)
    write_csv(out_dir / "model_level_metric_means_selected_worktime_25sample.csv", model_rows)

    y = [finite(row.get("attack_loss_increase")) for row in rows]
    corr_rows = []
    for metric in metrics:
        x = [finite(row.get(metric)) for row in rows]
        corr_rows.append(
            {
                "metric": metric,
                "n": int(np.isfinite(np.asarray(x)).sum()),
                "pearson_with_attack_loss_increase": safe_corr(x, y, "pearson"),
                "spearman_with_attack_loss_increase": safe_corr(x, y, "spearman"),
            }
        )
    write_csv(out_dir / "metric_correlations_with_attack_selected_worktime_25sample.csv", corr_rows)

    rank_rows = []
    for sample_id in sorted({row.get("sample_id", "") for row in rows}):
        subset = [row for row in rows if row.get("sample_id", "") == sample_id]
        yy = [finite(row.get("attack_loss_increase")) for row in subset]
        for metric in metrics:
            xx = [finite(row.get(metric)) for row in subset]
            rank_rows.append(
                {
                    "sample_id": sample_id,
                    "metric": metric,
                    "model_count": len(subset),
                    "spearman_across_models_with_attack_loss_increase": safe_corr(xx, yy, "spearman"),
                }
            )
    write_csv(out_dir / "per_sample_model_rank_similarity_selected_worktime_25sample.csv", rank_rows)


def write_readme(
    out_dir: Path,
    random_root: Path,
    clean_rows: list[dict[str, Any]],
    robust_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    gen = [row for row in clean_rows if row.get("split") == "generalization"]
    lines = [
        "# Burgers Six-Model Selected-Worktime Summary",
        "",
        "Model versions:",
        "",
        "- baseline/loss1/loss2/loss3: existing results, not recomputed here.",
        "- random_clean_y: `burgers_epoch8000_step008000.pt`.",
        "- random_solver_y: `burgers_epoch6000_step006000.pt`.",
        "",
        "Random-model source full suite:",
        f"`{random_root.relative_to(REPO)}`",
        "",
        f"Clean table datasets: {len(clean_rows)} total rows, {len(gen)} generalization rows.",
        f"Robustness/SVD rows: {len(robust_rows)} rows over 25-sample manifest.",
        "",
        "Key files:",
        "",
        "- `clean_52dataset_six_models_selected_worktime.csv`",
        "- `clean_generalization_model_summary_selected_worktime.csv`",
        "- `attack_52dataset_six_models_selected_worktime_long.csv`",
        "- `robustness_25sample_six_models_selected_worktime.csv`",
        "- `model_level_metric_means_selected_worktime_25sample.csv`",
        "- `metric_correlations_with_attack_selected_worktime_25sample.csv`",
        "- `per_sample_model_rank_similarity_selected_worktime_25sample.csv`",
        "",
        "Summary plots:",
        "",
    ]
    lines.extend(f"- `{path.relative_to(out_dir)}`" for path in plot_paths)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--random-root", type=Path, default=DEFAULT_RANDOM_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--random-clean-checkpoint", default="adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt")
    ap.add_argument("--random-solver-checkpoint", default="adversarial_training_runs/burgers_wideparam_random_field_solver_y_6000ep_continue_20260613/burgers/checkpoints/burgers_epoch6000_step006000.pt")
    args = ap.parse_args()
    random_root = args.random_root.resolve()
    out_dir = args.out_dir.resolve()
    required = [
        random_root / "clean_loss/per_dataset_clean_metrics.csv",
        random_root / "p2q2_attack/summary_by_model_dataset.csv",
        random_root / "postprocess/svd_attack_bias_gradient_by_sample.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing and not args.allow_incomplete:
        raise FileNotFoundError("missing selected random outputs: " + ", ".join(str(p) for p in missing))
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "random_root": str(random_root),
        "old_summary": str(OLD_SUMMARY),
        "model_order": MODEL_ORDER,
        "selected_random_checkpoints": {
            "random_clean_y": args.random_clean_checkpoint,
            "random_solver_y": args.random_solver_checkpoint,
        },
    }
    (out_dir / "selected_model_versions.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    clean_rows = build_clean_table(random_root, out_dir) if required[0].exists() else []
    attack_rows = build_attack_table(random_root, out_dir)
    robust_rows = build_robust_table(random_root, out_dir) if required[2].exists() else []
    if robust_rows:
        build_metric_summaries(robust_rows, out_dir)
    plot_paths = write_summary_plots(out_dir, clean_rows, attack_rows, robust_rows)
    write_readme(out_dir, random_root, clean_rows, robust_rows, plot_paths)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "clean_rows": len(clean_rows),
                "attack_rows": len(attack_rows),
                "robust_rows": len(robust_rows),
                "plot_count": len(plot_paths),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
