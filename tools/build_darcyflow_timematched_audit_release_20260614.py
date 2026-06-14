#!/usr/bin/env python3
"""Build a DarcyFlow time-matched audit/release from existing artifacts.

This is intentionally an audit/organization script. It does not rerun training,
robustness attacks, SVD, Jacobian, or dense image generation. Missing expensive
coverage is recorded as coverage notes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260614"
AUDIT = ROOT / "outputs" / f"darcyflow_timematched_full_or_audit_{DATE}"
RELEASE = ROOT / "outputs" / f"darcyflow_timematched_organized_release_{DATE}"

REQUIRED = ROOT / "outputs" / "darcy_sir20_required_figures_only_20260614"
CURATED = ROOT / "outputs" / "darcy_sir20_existing_curated_bundle_20260614"
RAW_AUDIT = ROOT / "outputs" / "darcy_raw_vs_corrected_audit_20260614"
SMOKE = ROOT / "outputs" / "darcy_sir20_timematched_full_20260614_smoke_initial"
RANDOM_STATUS = ROOT / "outputs" / "darcy_random_3000_20260614" / "replot_status.json"
R2_RECORD = ROOT / "outputs" / "r2_upload_darcy_20260614"

EVAL_METRICS = REQUIRED / "data" / "six_method_common_range_eval_metrics.csv"
EVAL_SPLIT = REQUIRED / "data" / "six_method_common_range_eval_split_summary.csv"
ROBUSTNESS = SMOKE / "data" / "robustness_attack_52datasets_samples.csv"
SVD = SMOKE / "data" / "svd_jacobian_metrics.csv"

METHOD_ORDER = ["baseline", "loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]
METHOD_LABEL = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "physics loss",
    "physics_loss": "physics loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}
METHOD_COLOR = {
    "baseline": "#4b5563",
    "loss1": "#1f77b4",
    "loss2": "#ff7f0e",
    "loss3": "#2ca02c",
    "physics": "#9467bd",
    "physics_loss": "#9467bd",
    "random_clean": "#e377c2",
    "random_solver": "#8c564b",
}
VARIANTS = {
    "all_six_models": ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"],
    "no_random_clean": ["loss1", "loss2", "loss3", "physics", "random_solver"],
    "loss123_only": ["loss1", "loss2", "loss3"],
}
LOWER_IS_BETTER = {
    "rmse": True,
    "relative_l2": True,
    "mae": True,
    "data_mse": True,
    "invalid_value_fraction": True,
    "clean_loss": True,
    "adv_loss": True,
    "loss_increase": True,
    "relative_increase": True,
    "delta_l2_rms": True,
    "delta_linf": True,
    "attack_loss_increase": True,
    "attack_relative_increase": True,
    "error_l2_norm": True,
    "jt_error_l2_norm": True,
    "sigma_input_right": True,
    "angle_singular_jt_error_deg": True,
    "angle_singular_attack_delta_deg": True,
    "angle_jt_error_attack_delta_deg": True,
    "accuracy_score": False,
    "cos_singular_jt_error": False,
    "cos_singular_attack_delta": False,
    "cos_jt_error_attack_delta": False,
    "corr_singular_jt_error": False,
    "corr_singular_attack_delta": False,
    "corr_jt_error_attack_delta": False,
}


@dataclass(frozen=True)
class SourceCopy:
    src: Path
    dst_rel: Path
    hardlink: bool = False


def ensure_dirs() -> None:
    for base in [AUDIT, RELEASE]:
        for sub in ["figures", "data", "reports", "logs", "manifests"]:
            (base / sub).mkdir(parents=True, exist_ok=True)
    for sub in [
        "figures/main_curves",
        "figures/existing_heatmaps",
        "figures/raw_vs_corrected_audit",
        "figures/dense_existing/group05_loss3_advantage",
        "figures/diagnostic_existing",
        "data/source_tables",
        "data/robustness_npz",
        "data/svd_jacobian_vectors",
    ]:
        (AUDIT / sub).mkdir(parents=True, exist_ok=True)
        (RELEASE / sub).mkdir(parents=True, exist_ok=True)


def clean_existing() -> None:
    for p in [AUDIT, RELEASE]:
        if p.exists():
            shutil.rmtree(p)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def metric_value_column(df: pd.DataFrame, metric: str) -> str:
    for candidate in [f"{metric}_plot", f"{metric}_corrected", metric]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"No column for metric {metric}")


def dataset_label(dataset_id: str, max_len: int = 42) -> str:
    text = str(dataset_id)
    replacements = [
        ("darcy_lossdrop_pool_", ""),
        ("train_screen_binary_grf_", "train "),
        ("test_screen_binary_grf_", "test "),
        ("original_binary_grf_", "original "),
        ("binary_", "binary "),
        ("soft_", "soft "),
        ("rectangles", "rectangles"),
        ("maternfine", "matern fine"),
        ("highpass", "high pass"),
        ("bandpass", "band pass"),
        ("_", " "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "."
    return text


def order_methods(methods: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(str(m) for m in methods))
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered.extend([m for m in present if m not in ordered])
    return ordered


def better_is_lower(metric: str) -> bool:
    return LOWER_IS_BETTER.get(metric, True)


def rank_group(df: pd.DataFrame, value_col: str, metric_col: str = "metric") -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby([c for c in ["source_table", "metric", "unit_id"] if c in df.columns], dropna=False):
        metric = g[metric_col].iloc[0] if metric_col in g.columns else str(keys)
        ascending = better_is_lower(str(metric))
        gg = g.copy()
        gg = gg.sort_values(value_col, ascending=ascending, kind="mergesort")
        gg["rank"] = np.arange(1, len(gg) + 1)
        gg["is_best"] = gg["rank"] == 1
        best_model = gg.iloc[0]["method"] if len(gg) else ""
        runner = gg.iloc[1]["method"] if len(gg) > 1 else ""
        best_value = gg.iloc[0][value_col] if len(gg) else np.nan
        runner_value = gg.iloc[1][value_col] if len(gg) > 1 else np.nan
        if len(gg) > 1:
            advantage = (runner_value - best_value) if ascending else (best_value - runner_value)
        else:
            advantage = np.nan
        gg["best_model"] = best_model
        gg["runner_up_model"] = runner
        gg["advantage_vs_runner_up"] = advantage
        rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else df.copy()


def final_eval_long(eval_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse", "relative_l2", "mae", "accuracy_score", "invalid_value_fraction"]
    available = [m for m in metrics if m in eval_df.columns or f"{m}_plot" in eval_df.columns or f"{m}_corrected" in eval_df.columns]
    df = eval_df.copy()
    if "phase" in df.columns:
        baseline = df[df["phase"].eq("baseline_before_adversarial_training")].copy()
    else:
        baseline = df.iloc[0:0].copy()
    baseline_rows = []
    if not baseline.empty:
        baseline = baseline.sort_values(["dataset_id", "method"]).groupby("dataset_id", as_index=False).first()
        baseline["method"] = "baseline"
        baseline_rows.append(baseline)
    train = df[df.get("phase", "").eq("during_adversarial_training")].copy()
    if train.empty:
        train = df.copy()
    train = train.sort_values(["method", "dataset_id", "epoch"]).groupby(["method", "dataset_id"], as_index=False).tail(1)
    base = pd.concat(baseline_rows + [train], ignore_index=True)
    rows = []
    for metric in available:
        col = metric_value_column(base, metric)
        out = base[["method", "dataset_id", "split", "source", "manual_tier", "manual_rank", "epoch", "path"]].copy()
        out["source_table"] = "clean_52dataset"
        out["metric"] = metric
        out["value"] = pd.to_numeric(base[col], errors="coerce")
        out["unit_id"] = out["dataset_id"].astype(str)
        out["coverage_status"] = np.where(out["method"].eq("baseline"), "baseline_reference", "available")
        rows.append(out)
    if "rmse" in available:
        col = metric_value_column(base, "rmse")
        out = base[["method", "dataset_id", "split", "source", "manual_tier", "manual_rank", "epoch", "path"]].copy()
        out["source_table"] = "clean_52dataset"
        out["metric"] = "data_mse"
        out["value"] = pd.to_numeric(base[col], errors="coerce") ** 2
        out["unit_id"] = out["dataset_id"].astype(str)
        out["coverage_status"] = np.where(out["method"].eq("baseline"), "baseline_reference", "available")
        rows.append(out)
    long = pd.concat(rows, ignore_index=True)
    epoch_max = train.groupby("method")["epoch"].max().to_dict() if "epoch" in train.columns else {}
    long["model_max_epoch"] = long["method"].map(epoch_max).fillna(0).astype(float)
    long.loc[long["method"].isin(["random_clean", "random_solver"]) & (long["model_max_epoch"] < 3000), "coverage_status"] = "partial_random_epoch"
    return rank_group(long.dropna(subset=["value"]), "value")


def attack_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = read_csv(path)
    if "method" in df.columns:
        df["method"] = df["method"].replace({"physics_loss": "physics"})
    metrics = ["clean_loss", "adv_loss", "loss_increase", "relative_increase", "delta_l2_rms", "delta_linf"]
    rows = []
    id_cols = ["method", "dataset_id", "split", "source", "manual_tier", "manual_rank", "sample_ordinal", "source_sample_index", "is_svd_sample"]
    for metric in metrics:
        if metric not in df.columns:
            continue
        out = df[id_cols].copy()
        out["source_table"] = "attack_52dataset"
        out["metric"] = metric
        out["value"] = pd.to_numeric(df[metric], errors="coerce")
        out["unit_id"] = out["dataset_id"].astype(str) + "::sample" + out["source_sample_index"].astype(str)
        out["coverage_status"] = "partial_smoke_2sample_per_dataset"
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return rank_group(pd.concat(rows, ignore_index=True).dropna(subset=["value"]), "value")


def svd_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = read_csv(path)
    if "method" in df.columns:
        df["method"] = df["method"].replace({"physics_loss": "physics"})
    metrics = [
        "clean_loss",
        "attack_loss_increase",
        "attack_relative_increase",
        "error_l2_norm",
        "jt_error_l2_norm",
        "sigma_input_right",
        "cos_singular_jt_error",
        "angle_singular_jt_error_deg",
        "corr_singular_jt_error",
        "cos_singular_attack_delta",
        "angle_singular_attack_delta_deg",
        "corr_singular_attack_delta",
        "cos_jt_error_attack_delta",
        "angle_jt_error_attack_delta_deg",
        "corr_jt_error_attack_delta",
    ]
    rows = []
    id_cols = ["method", "dataset_id", "split", "sample_index", "svd_role"]
    for metric in metrics:
        if metric not in df.columns:
            continue
        out = df[id_cols].copy()
        out["source_table"] = "svd_jacobian_25sample"
        out["metric"] = metric
        out["value"] = pd.to_numeric(df[metric], errors="coerce")
        out["unit_id"] = out["dataset_id"].astype(str) + "::sample" + out["sample_index"].astype(str)
        out["coverage_status"] = "partial_smoke_3samples_per_model"
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return rank_group(pd.concat(rows, ignore_index=True).dropna(subset=["value"]), "value")


def summarize_long(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    grouped = (
        long.groupby(["source_table", "metric", "method", "coverage_status"], dropna=False)
        .agg(mean=("value", "mean"), std=("value", "std"), median=("value", "median"), n=("value", "count"))
        .reset_index()
    )
    rows = []
    for (source_table, metric), g in grouped.groupby(["source_table", "metric"], dropna=False):
        ascending = better_is_lower(str(metric))
        gg = g.sort_values("mean", ascending=ascending, kind="mergesort").copy()
        gg["rank"] = np.arange(1, len(gg) + 1)
        gg["is_best"] = gg["rank"] == 1
        best = gg.iloc[0]
        runner = gg.iloc[1] if len(gg) > 1 else None
        gg["best_model"] = best["method"]
        gg["runner_up_model"] = runner["method"] if runner is not None else ""
        if runner is not None:
            gap = (runner["mean"] - best["mean"]) if ascending else (best["mean"] - runner["mean"])
        else:
            gap = np.nan
        gg["advantage_vs_runner_up"] = gap
        rows.append(gg)
    return pd.concat(rows, ignore_index=True)


def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return pd.Series(q, index=pvals.index)
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    vals = ranked * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.clip(vals, 0, 1)
    return pd.Series(q, index=pvals.index)


def paired_tests(long: pd.DataFrame, mode: str) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    rows = []
    for (source_table, metric), g in long.groupby(["source_table", "metric"], dropna=False):
        pivot = g.pivot_table(index="unit_id", columns="method", values="value", aggfunc="mean")
        if pivot.empty:
            continue
        ascending = better_is_lower(str(metric))
        if mode == "best_vs_other":
            means = pivot.mean(skipna=True).sort_values(ascending=ascending)
            if means.empty:
                continue
            reference = means.index[0]
            candidates = [m for m in pivot.columns if m != reference]
        else:
            reference = "loss3"
            if reference not in pivot.columns:
                continue
            candidates = [m for m in pivot.columns if m != reference]
        for other in candidates:
            pair = pivot[[reference, other]].dropna()
            n = len(pair)
            row: dict[str, Any] = {
                "source_table": source_table,
                "metric": metric,
                "reference_model": reference,
                "other_model": other,
                "n_pairs": n,
                "reference_mean": pair[reference].mean() if n else np.nan,
                "other_mean": pair[other].mean() if n else np.nan,
                "mean_difference_reference_minus_other": (pair[reference] - pair[other]).mean() if n else np.nan,
                "lower_is_better": ascending,
                "coverage_status": "available" if n >= 2 else "insufficient_pairs",
            }
            if n >= 2:
                diff = pair[reference] - pair[other]
                try:
                    t_res = stats.ttest_rel(pair[reference], pair[other], nan_policy="omit")
                    t_stat = float(t_res.statistic)
                    p_two = float(t_res.pvalue)
                except Exception:
                    t_stat, p_two = np.nan, np.nan
                try:
                    if np.allclose(diff.to_numpy(), 0):
                        w_stat, p_w = 0.0, 1.0
                    else:
                        w_res = stats.wilcoxon(pair[reference], pair[other], zero_method="wilcox", alternative="two-sided")
                        w_stat, p_w = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    w_stat, p_w = np.nan, np.nan
                if np.isfinite(p_two) and np.isfinite(t_stat):
                    if ascending:
                        p_one = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
                    else:
                        p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
                else:
                    p_one = np.nan
                row.update(
                    {
                        "paired_t_stat": t_stat,
                        "paired_t_p_two_sided": p_two,
                        "paired_t_p_one_sided_reference_better": p_one,
                        "wilcoxon_stat": w_stat,
                        "wilcoxon_p_two_sided": p_w,
                    }
                )
            else:
                row.update(
                    {
                        "paired_t_stat": np.nan,
                        "paired_t_p_two_sided": np.nan,
                        "paired_t_p_one_sided_reference_better": np.nan,
                        "wilcoxon_stat": np.nan,
                        "wilcoxon_p_two_sided": np.nan,
                    }
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_q_two_sided_bh_fdr"] = bh_fdr(out["paired_t_p_two_sided"])
        out["wilcoxon_q_two_sided_bh_fdr"] = bh_fdr(out["wilcoxon_p_two_sided"])
    return out


def correlation_tables(svd_df: pd.DataFrame) -> pd.DataFrame:
    if svd_df.empty:
        return pd.DataFrame()
    cols = [
        "clean_loss",
        "attack_loss_increase",
        "attack_relative_increase",
        "error_l2_norm",
        "jt_error_l2_norm",
        "sigma_input_right",
        "cos_singular_jt_error",
        "cos_singular_attack_delta",
        "cos_jt_error_attack_delta",
    ]
    cols = [c for c in cols if c in svd_df.columns]
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("all_models", svd_df)]
    groups.extend([(str(m), g) for m, g in svd_df.groupby("method")])
    for group_name, g in groups:
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                pair = g[[a, b]].dropna()
                if len(pair) < 3:
                    pearson = pearson_p = spearman = spearman_p = np.nan
                    status = "insufficient_pairs"
                else:
                    try:
                        pearson, pearson_p = stats.pearsonr(pair[a], pair[b])
                    except Exception:
                        pearson = pearson_p = np.nan
                    try:
                        spearman, spearman_p = stats.spearmanr(pair[a], pair[b])
                    except Exception:
                        spearman = spearman_p = np.nan
                    status = "partial_smoke_3samples_per_model" if group_name != "all_models" else "partial_smoke_21rows"
                rows.append(
                    {
                        "group": group_name,
                        "metric_a": a,
                        "metric_b": b,
                        "n_pairs": len(pair),
                        "pearson_r": pearson,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman,
                        "spearman_p": spearman_p,
                        "abs_pearson_r": abs(pearson) if np.isfinite(pearson) else np.nan,
                        "coverage_status": status,
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(["abs_pearson_r", "n_pairs"], ascending=[False, False], na_position="last")


def coverage_notes(eval_df: pd.DataFrame, attack_df: pd.DataFrame, svd_df: pd.DataFrame) -> pd.DataFrame:
    notes: list[dict[str, Any]] = []
    max_epochs = eval_df.groupby("method")["epoch"].max().to_dict() if not eval_df.empty else {}
    for method in ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]:
        epoch = int(max_epochs.get(method, 0))
        status = "complete_or_near_target" if epoch >= 3000 else "partial"
        notes.append(
            {
                "component": "training_curves_52dataset_eval",
                "method": method,
                "coverage_status": status,
                "observed_n": epoch,
                "target_n": 3000,
                "note": f"Max epoch observed in required figure CSV is {epoch}.",
            }
        )
    notes.extend(
        [
            {
                "component": "baseline_clean_eval",
                "method": "baseline",
                "coverage_status": "available",
                "observed_n": int(eval_df[eval_df.get("phase", "").eq("baseline_before_adversarial_training")]["dataset_id"].nunique()) if not eval_df.empty else 0,
                "target_n": 52,
                "note": "Baseline references are available from epoch-0 evaluation rows.",
            },
            {
                "component": "physics_loss_pde_residual_eval",
                "method": "all",
                "coverage_status": "missing_or_partial",
                "observed_n": 0,
                "target_n": 52,
                "note": "Main evaluation CSVs contain RMSE/Relative-L2/MAE/accuracy but no per-dataset PDE residual columns. Physics training rows contain darcy_physics_metric for physics runs only.",
            },
            {
                "component": "robustness_attack_52datasets",
                "method": "all",
                "coverage_status": "partial_smoke",
                "observed_n": int(attack_df.shape[0]) if not attack_df.empty else 0,
                "target_n": 52 * 50 * 7,
                "note": "Found smoke robustness covering 52 datasets x 2 samples x 7 models, not the requested 50 samples per dataset. Not recomputed by audit script.",
            },
            {
                "component": "svd_jacobian_25sample",
                "method": "all",
                "coverage_status": "partial_smoke",
                "observed_n": int(svd_df.shape[0]) if not svd_df.empty else 0,
                "target_n": 25 * 7,
                "note": "Found 3 samples per model in smoke SVD/Jacobian metrics, not full fixed 25 samples. Not recomputed by audit script.",
            },
            {
                "component": "model_solver_subspace_top50_top100",
                "method": "all",
                "coverage_status": "missing",
                "observed_n": 0,
                "target_n": 7,
                "note": "No local top50/top100 model-solver subspace artifacts found; coverage note recorded instead of running expensive sweeps.",
            },
            {
                "component": "dense_group00_to_group05_variants",
                "method": "all",
                "coverage_status": "partial_existing_loss3_advantage",
                "observed_n": len(list((CURATED / "figures" / "attack_heatmaps_loss3_advantage").glob("*.png"))) if (CURATED / "figures" / "attack_heatmaps_loss3_advantage").exists() else 0,
                "target_n": 6,
                "note": "Existing loss3-advantage heatmaps were copied as dense_existing/group05; all required group00..group05 all-model/no-random/loss123 linear/log variants were not found and not regenerated.",
            },
            {
                "component": "moving_average",
                "method": "all",
                "coverage_status": "excluded_from_main",
                "observed_n": len(list((REQUIRED / "figures_moving_average_ma51").glob("*.png"))) if (REQUIRED / "figures_moving_average_ma51").exists() else 0,
                "target_n": 0,
                "note": "Existing moving-average figures are kept out of the main release figures because this audit prompt requests raw evaluation points only.",
            },
            {
                "component": "r2_live_scan",
                "method": "all",
                "coverage_status": "not_live_scanned",
                "observed_n": 0,
                "target_n": 1,
                "note": "This builder uses local artifacts and previous upload records. Live R2 scan/upload is handled separately with environment credentials.",
            },
        ]
    )
    if RANDOM_STATUS.exists():
        try:
            status = json.loads(RANDOM_STATUS.read_text(encoding="utf-8"))
            for row in status.get("runs", []):
                notes.append(
                    {
                        "component": "active_random_3500_supervisor",
                        "method": row.get("method"),
                        "coverage_status": "running" if not row.get("done") else "complete",
                        "observed_n": row.get("train_max_epoch"),
                        "target_n": row.get("target_epoch"),
                        "note": f"Watcher state {status.get('state')} at {status.get('updated_at')}.",
                    }
                )
        except Exception as exc:
            notes.append(
                {
                    "component": "active_random_3500_supervisor",
                    "method": "all",
                    "coverage_status": "status_parse_failed",
                    "observed_n": 0,
                    "target_n": 1,
                    "note": str(exc),
                }
            )
    return pd.DataFrame(notes)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        cells = []
        for c in cols:
            val = str(row[c])
            if c == "best_model" and val and val != "nan":
                val = f"**{val}**"
            if c == "method" and "is_best" in cols and bool(row.get("is_best", False)):
                val = f"**{val}**"
            cells.append(val)
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def write_reports(
    summary: pd.DataFrame,
    best: pd.DataFrame,
    best_tests: pd.DataFrame,
    loss3_tests: pd.DataFrame,
    coverage: pd.DataFrame,
    corr: pd.DataFrame,
    generated_figures: list[Path],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    reports = AUDIT / "reports"
    rel_reports = RELEASE / "reports"
    for out_dir in [reports, rel_reports]:
        out_dir.mkdir(parents=True, exist_ok=True)
        main_lines = [
            "# DarcyFlow Time-Matched Existing-Artifact Audit",
            "",
            f"Generated: `{now}`",
            "",
            "This audit organizes existing DarcyFlow/data-flow artifacts. It does not rerun training, robustness attacks, SVD/Jacobian, or dense sweeps.",
            "",
            "## Main Findings",
            "",
            "- Clean 52-dataset evaluation and raw-point training curves are available for loss1/loss2/loss3/physics, with random clean/solver currently partial in the archived figure CSVs.",
            "- Physics/PDE residual metrics are not available as full 52-dataset evaluation columns; physics-specific training metrics exist only in physics-run training logs.",
            "- Robustness and SVD/Jacobian artifacts found locally are smoke/partial coverage, so all ranked robustness/SVD tables carry partial coverage notes.",
            "- Existing loss3-advantage Darcy heatmaps were copied into the release, but the full requested group00..group05 dense variant matrix was not found locally and was not recomputed.",
            "",
            "## Best Model Summary",
            "",
            markdown_table(best[["source_table", "metric", "best_model", "runner_up_model", "advantage_vs_runner_up", "coverage_status"]], max_rows=40),
            "",
            "## Coverage Notes",
            "",
            markdown_table(coverage, max_rows=80),
            "",
            "## Correlation/Similarity Notes",
            "",
            markdown_table(corr.head(30), max_rows=30),
            "",
            "## Figure Count",
            "",
            f"Generated raw-point main figures: `{len(generated_figures)}`.",
            "",
            "## Integrity Note",
            "",
            "Artifact-corrected columns are treated as derived visualization/audit data. Raw experiment logs are preserved separately.",
            "",
        ]
        (out_dir / "AUDIT_REPORT.md").write_text("\n".join(main_lines), encoding="utf-8")
        appendix_lines = [
            "# DarcyFlow Statistical Appendix",
            "",
            "## Metric Model Summary Ranked",
            "",
            markdown_table(summary, max_rows=80),
            "",
            "## Best-vs-Other Paired Tests",
            "",
            markdown_table(best_tests, max_rows=80),
            "",
            "## Loss3-vs-Other Paired Tests",
            "",
            markdown_table(loss3_tests, max_rows=80),
            "",
        ]
        (out_dir / "STATISTICAL_APPENDIX.md").write_text("\n".join(appendix_lines), encoding="utf-8")
        (out_dir / "COVERAGE_NOTES.md").write_text("# Coverage Notes\n\n" + markdown_table(coverage, max_rows=120) + "\n", encoding="utf-8")


def plot_split_means(split_df: pd.DataFrame, metric: str, x_axis: str, variant: str, yscale: str, out: Path) -> None:
    methods = VARIANTS[variant]
    value_col = metric_value_column(split_df, metric)
    if x_axis == "epoch":
        x_col = "epoch"
        xlabel = "epoch"
        x_factor = 1.0
        xlim = None
    elif x_axis == "work_hours":
        x_col = "work_seconds"
        xlabel = "work-clock time (hours)"
        x_factor = 1.0 / 3600.0
        xlim = (0, 8.0)
    else:
        x_col = "wall_seconds"
        xlabel = "wall-clock time (hours)"
        x_factor = 1.0 / 3600.0
        xlim = (0, 8.0)
    if x_col not in split_df.columns:
        return
    data = split_df[split_df["phase"].eq("during_adversarial_training") & split_df["method"].isin(methods) & split_df["split"].isin(["train", "test", "generalization"])].copy()
    if data.empty:
        return
    baseline = split_df[split_df["phase"].eq("baseline_before_adversarial_training") & split_df["split"].isin(["train", "test", "generalization"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), sharey=False)
    for ax, split in zip(axes, ["train", "test", "generalization"]):
        b = baseline[baseline["split"].eq(split)]
        if not b.empty:
            yb = pd.to_numeric(b[value_col], errors="coerce").dropna()
            if len(yb):
                ax.axhline(float(yb.iloc[0]), color=METHOD_COLOR["baseline"], linestyle="--", linewidth=1.25, alpha=0.85, label="baseline")
        for method in methods:
            m = data[data["method"].eq(method) & data["split"].eq(split)].sort_values(x_col)
            if m.empty:
                continue
            x = pd.to_numeric(m[x_col], errors="coerce") * x_factor
            y = pd.to_numeric(m[value_col], errors="coerce")
            ax.plot(x, y, color=METHOD_COLOR.get(method), linewidth=1.15, alpha=0.82, label=METHOD_LABEL.get(method, method))
        ax.set_title(split, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.9)
        if yscale == "log":
            ax.set_yscale("log")
        if xlim:
            ax.set_xlim(*xlim)
    handles, labels = axes[-1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle(f"DarcyFlow {metric.replace('_', ' ')} split means, {variant}, {yscale}", fontsize=14, fontweight="semibold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_generalization_grid(metrics_df: pd.DataFrame, metric: str, x_axis: str, variant: str, yscale: str, part: int, out: Path) -> None:
    methods = VARIANTS[variant]
    value_col = metric_value_column(metrics_df, metric)
    if x_axis == "epoch":
        x_col = "epoch"
        xlabel = "epoch"
        x_factor = 1.0
        xlim = None
    elif x_axis == "work_hours":
        x_col = "work_seconds"
        xlabel = "work-clock hours"
        x_factor = 1.0 / 3600.0
        xlim = (0, 8.0)
    else:
        x_col = "wall_seconds"
        xlabel = "wall-clock hours"
        x_factor = 1.0 / 3600.0
        xlim = (0, 8.0)
    if x_col not in metrics_df.columns:
        return
    gdf = metrics_df[metrics_df["split"].eq("generalization")].copy()
    order = (
        gdf[["dataset_id", "manual_rank"]]
        .drop_duplicates()
        .sort_values(["manual_rank", "dataset_id"], na_position="last")["dataset_id"]
        .tolist()
    )
    subset = order[(part - 1) * 25 : part * 25]
    if not subset:
        return
    data = gdf[gdf["phase"].eq("during_adversarial_training") & gdf["method"].isin(methods) & gdf["dataset_id"].isin(subset)].copy()
    baseline = gdf[gdf["phase"].eq("baseline_before_adversarial_training") & gdf["dataset_id"].isin(subset)].copy()
    fig, axes = plt.subplots(5, 5, figsize=(24, 15.2), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, dataset_id in zip(axes, subset):
        b = baseline[baseline["dataset_id"].eq(dataset_id)]
        if not b.empty:
            yb = pd.to_numeric(b[value_col], errors="coerce").dropna()
            if len(yb):
                ax.axhline(float(yb.iloc[0]), color=METHOD_COLOR["baseline"], linestyle="--", linewidth=0.95, alpha=0.85)
        for method in methods:
            m = data[data["method"].eq(method) & data["dataset_id"].eq(dataset_id)].sort_values(x_col)
            if m.empty:
                continue
            x = pd.to_numeric(m[x_col], errors="coerce") * x_factor
            y = pd.to_numeric(m[value_col], errors="coerce")
            ax.plot(x, y, color=METHOD_COLOR.get(method), linewidth=0.95, alpha=0.72)
        ax.set_title(dataset_label(dataset_id), fontsize=8.8, pad=3)
        ax.grid(True, color="#e5e7eb", linewidth=0.55, alpha=0.85)
        if yscale == "log":
            ax.set_yscale("log")
        if xlim:
            ax.set_xlim(*xlim)
        ax.tick_params(labelsize=7, length=2)
    for ax in axes[len(subset) :]:
        ax.axis("off")
    legend_lines = [plt.Line2D([0], [0], color=METHOD_COLOR[m], lw=2, alpha=0.8, label=METHOD_LABEL.get(m, m)) for m in methods]
    legend_lines.insert(0, plt.Line2D([0], [0], color=METHOD_COLOR["baseline"], lw=1.6, ls="--", label="baseline"))
    fig.legend(handles=legend_lines, loc="upper center", ncol=7, frameon=False, fontsize=10)
    fig.supxlabel(xlabel, y=0.02, fontsize=11)
    fig.suptitle(f"DarcyFlow generalization {metric.replace('_', ' ')} part {part}, {variant}, {yscale}", fontsize=15, fontweight="semibold", y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), h_pad=1.0, w_pad=0.65)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def generate_figures(metrics_df: pd.DataFrame, split_df: pd.DataFrame) -> list[Path]:
    out_paths: list[Path] = []
    metrics = ["rmse", "relative_l2"]
    axes = ["epoch", "work_hours", "wall_hours"]
    for variant in VARIANTS:
        for yscale in ["linear", "log"]:
            for metric in metrics:
                for x_axis in axes:
                    out = AUDIT / "figures" / "main_curves" / variant / yscale / f"{metric}_split_means_vs_{x_axis}.png"
                    plot_split_means(split_df, metric, x_axis, variant, yscale, out)
                    if out.exists():
                        out_paths.append(out)
                    for part in [1, 2]:
                        out = AUDIT / "figures" / "main_curves" / variant / yscale / f"{metric}_generalization_part{part:02d}_vs_{x_axis}.png"
                        plot_generalization_grid(metrics_df, metric, x_axis, variant, yscale, part, out)
                        if out.exists():
                            out_paths.append(out)
    return out_paths


def link_or_copy(src: Path, dst: Path, hardlink: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_tree_files(src: Path, dst: Path, patterns: tuple[str, ...] = ("*",), hardlink: bool = False) -> list[Path]:
    copied: list[Path] = []
    if not src.exists():
        return copied
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if not any(p.match(pattern) for pattern in patterns):
            continue
        relp = p.relative_to(src)
        target = dst / relp
        link_or_copy(p, target, hardlink=hardlink)
        copied.append(target)
    return copied


def copy_release_assets(generated_figures: list[Path]) -> None:
    copies = [
        SourceCopy(EVAL_METRICS, Path("data/source_tables/six_method_common_range_eval_metrics.csv")),
        SourceCopy(EVAL_SPLIT, Path("data/source_tables/six_method_common_range_eval_split_summary.csv")),
        SourceCopy(CURATED / "data" / "timing_calibration.csv", Path("data/source_tables/timing_calibration.csv")),
        SourceCopy(CURATED / "data" / "run_epoch_audit.csv", Path("data/source_tables/run_epoch_audit.csv")),
        SourceCopy(RAW_AUDIT / "data" / "source_eval_metrics_artifact_corrected.csv", Path("data/source_tables/source_eval_metrics_artifact_corrected.csv")),
        SourceCopy(RAW_AUDIT / "data" / "source_eval_split_summary_artifact_corrected.csv", Path("data/source_tables/source_eval_split_summary_artifact_corrected.csv")),
        SourceCopy(SMOKE / "data" / "robustness_attack_52datasets_samples.csv", Path("data/source_tables/robustness_attack_52datasets_samples.csv")),
        SourceCopy(SMOKE / "data" / "svd_jacobian_metrics.csv", Path("data/source_tables/svd_jacobian_metrics.csv")),
        SourceCopy(SMOKE / "data" / "attack_50sample_manifest.csv", Path("manifests/attack_50sample_manifest_partial_smoke.csv")),
        SourceCopy(SMOKE / "data" / "svd_jacobian_25sample_manifest.csv", Path("manifests/svd_jacobian_25sample_manifest_partial_smoke.csv")),
        SourceCopy(R2_RECORD / "remote_size_check.tsv", Path("manifests/previous_r2_remote_size_check.tsv")),
        SourceCopy(R2_RECORD / "paired_audit_remote_figures.txt", Path("manifests/previous_r2_paired_audit_remote_figures.txt")),
    ]
    for base in [AUDIT, RELEASE]:
        for item in copies:
            if item.src.exists():
                link_or_copy(item.src, base / item.dst_rel, hardlink=item.hardlink)
        for fig in generated_figures:
            target = base / fig.relative_to(AUDIT)
            if fig.exists() and target != fig:
                link_or_copy(fig, target)
        copy_tree_files(CURATED / "figures" / "attack_heatmaps_loss3_advantage", base / "figures" / "dense_existing" / "group05_loss3_advantage", ("*.png",))
        copy_tree_files(RAW_AUDIT / "figures", base / "figures" / "raw_vs_corrected_audit", ("*.png",))
        copy_tree_files(REQUIRED / "figures", base / "figures" / "diagnostic_existing" / "required_raw_figures_previous", ("*.png",))
        copy_tree_files(SMOKE / "data" / "robustness_deltas", base / "data" / "robustness_npz", ("*.npz",), hardlink=True)
        copy_tree_files(SMOKE / "data" / "svd_jacobian_vectors", base / "data" / "svd_jacobian_vectors", ("*.npz",), hardlink=True)


def write_manifest(base: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rows.append(
            {
                "path": rel(p),
                "relative_to_bundle": str(p.relative_to(base)),
                "bytes": p.stat().st_size,
                "suffix": p.suffix,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(base / "manifests" / "file_manifest.csv", index=False)
    (base / "manifests" / "file_manifest.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove existing audit/release directories before rebuilding.")
    parser.add_argument("--skip-figures", action="store_true", help="Reuse existing generated main figures and rebuild tables/reports/manifests only.")
    args = parser.parse_args()

    if args.clean:
        clean_existing()
    ensure_dirs()

    metrics_df = read_csv(EVAL_METRICS)
    split_df = read_csv(EVAL_SPLIT)
    attack_df = read_csv(ROBUSTNESS) if ROBUSTNESS.exists() else pd.DataFrame()
    svd_df = read_csv(SVD) if SVD.exists() else pd.DataFrame()
    for df in [attack_df, svd_df]:
        if not df.empty and "method" in df.columns:
            df["method"] = df["method"].replace({"physics_loss": "physics"})

    if args.skip_figures:
        generated_figures = sorted((AUDIT / "figures" / "main_curves").rglob("*.png"))
    else:
        generated_figures = generate_figures(metrics_df, split_df)

    clean_long = final_eval_long(metrics_df)
    attack_ranked = attack_long(ROBUSTNESS)
    svd_ranked = svd_long(SVD)
    all_long = pd.concat([clean_long, attack_ranked, svd_ranked], ignore_index=True)

    summary = summarize_long(all_long)
    best = summary[summary["is_best"]].copy() if not summary.empty else pd.DataFrame()
    best_tests = paired_tests(all_long, "best_vs_other")
    loss3_tests = paired_tests(all_long, "loss3_vs_other")
    corr = correlation_tables(svd_df)
    coverage = coverage_notes(metrics_df, attack_df, svd_df)

    data_dir = AUDIT / "data"
    clean_long.to_csv(data_dir / "clean_52dataset_metric_long_ranked.csv", index=False)
    attack_ranked.to_csv(data_dir / "attack_52dataset_metric_long_ranked.csv", index=False)
    svd_ranked.to_csv(data_dir / "robustness_25sample_metric_long_ranked.csv", index=False)
    svd_ranked[svd_ranked["metric"].isin(["sigma_input_right", "jt_error_l2_norm", "error_l2_norm", "attack_loss_increase"])].to_csv(
        data_dir / "svd_error_topk_ranked_tables.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "metric": "model_solver_subspace_top20",
                "coverage_status": "missing",
                "note": "No local top20/top50/top100 model-solver subspace tables found; not recomputed.",
            }
        ]
    ).to_csv(data_dir / "svd_error_top20_ranked_tables.csv", index=False)
    summary.to_csv(data_dir / "metric_model_summary_ranked.csv", index=False)
    best.to_csv(data_dir / "metric_best_summary_ranked.csv", index=False)
    best_tests.to_csv(data_dir / "metric_best_vs_other_significance_tests.csv", index=False)
    loss3_tests.to_csv(data_dir / "metric_loss3_vs_other_significance_tests.csv", index=False)
    summary.to_csv(data_dir / "model_level_scalar_ranked_tables.csv", index=False)
    corr.to_csv(data_dir / "correlations_sorted_tables.csv", index=False)
    coverage.to_csv(data_dir / "random_coverage_partial_metric_notes.csv", index=False)
    coverage.to_csv(data_dir / "coverage_notes.csv", index=False)
    generated_df = pd.DataFrame({"figure_path": [rel(p) for p in generated_figures], "bytes": [p.stat().st_size for p in generated_figures]})
    generated_df.to_csv(AUDIT / "manifests" / "generated_figure_manifest.csv", index=False)

    write_json(
        AUDIT / "manifests" / "audit_manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "audit_dir": rel(AUDIT),
            "release_dir": rel(RELEASE),
            "source_files": {
                "eval_metrics": rel(EVAL_METRICS),
                "eval_split": rel(EVAL_SPLIT),
                "robustness": rel(ROBUSTNESS),
                "svd": rel(SVD),
            },
            "generated_figures": len(generated_figures),
            "did_rerun_training": False,
            "did_rerun_robustness": False,
            "did_rerun_svd_jacobian": False,
        },
    )

    write_reports(summary, best, best_tests, loss3_tests, coverage, corr, generated_figures)
    copy_release_assets(generated_figures)

    # Copy generated data/reports/manifests into organized release after source assets.
    for sub in ["data", "reports", "manifests"]:
        for p in (AUDIT / sub).rglob("*"):
            if p.is_file():
                target = RELEASE / p.relative_to(AUDIT)
                if target != p:
                    link_or_copy(p, target)

    audit_manifest = write_manifest(AUDIT)
    release_manifest = write_manifest(RELEASE)
    summary_payload = {
        "audit_files": int(len(audit_manifest)),
        "audit_bytes": int(audit_manifest["bytes"].sum()) if not audit_manifest.empty else 0,
        "release_files": int(len(release_manifest)),
        "release_bytes": int(release_manifest["bytes"].sum()) if not release_manifest.empty else 0,
        "generated_figures": len(generated_figures),
    }
    write_json(AUDIT / "manifests" / "bundle_size_summary.json", summary_payload)
    write_json(RELEASE / "manifests" / "bundle_size_summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
