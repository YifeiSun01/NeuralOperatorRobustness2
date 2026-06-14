#!/usr/bin/env python3
"""Build a Darcy cflow time-matched audit/release from existing artifacts.

The script is intentionally conservative: it reuses local DarcyFlow artifacts and
does not rerun training, attacks, SVD/Jacobian, or dense image generation.
Missing/partial expensive coverage is recorded explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260614"

SRC_AUDIT = ROOT / "outputs" / f"darcyflow_timematched_full_or_audit_{DATE}"
SRC_RELEASE = ROOT / "outputs" / f"darcyflow_timematched_organized_release_{DATE}"

AUDIT = ROOT / "outputs" / f"darcy_cflow_timematched_full_or_audit_{DATE}"
RELEASE = ROOT / "outputs" / f"darcy_cflow_timematched_organized_release_{DATE}"

EVAL_METRICS = SRC_AUDIT / "data" / "source_tables" / "six_method_common_range_eval_metrics.csv"
EVAL_SPLIT = SRC_AUDIT / "data" / "source_tables" / "six_method_common_range_eval_split_summary.csv"
ATTACK = SRC_AUDIT / "data" / "source_tables" / "robustness_attack_52datasets_samples.csv"
SVD = SRC_AUDIT / "data" / "source_tables" / "svd_jacobian_metrics.csv"
RANDOM_STATUS = ROOT / "outputs" / "darcy_random_3000_20260614" / "replot_status.json"

METHOD_ORDER = ["baseline", "loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]
METHOD_LABEL = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "physics loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}
LOWER_IS_BETTER = {
    "rmse": True,
    "relative_l2": True,
    "mae": True,
    "data_mse": True,
    "invalid_value_fraction": True,
    "accuracy_score": False,
    "clean_loss": True,
    "adv_loss": True,
    "loss_increase": True,
    "relative_increase": True,
    "attack_loss_increase": True,
    "attack_relative_increase": True,
    "error_l2_norm": True,
    "jt_error_l2_norm": True,
    "sigma_input_right": True,
    "angle_singular_jt_error_deg": True,
    "angle_singular_attack_delta_deg": True,
    "angle_jt_error_attack_delta_deg": True,
    "cos_singular_jt_error": False,
    "cos_singular_attack_delta": False,
    "cos_jt_error_attack_delta": False,
    "corr_singular_jt_error": False,
    "corr_singular_attack_delta": False,
    "corr_jt_error_attack_delta": False,
}
QUALITY_METRICS = {
    "rmse",
    "relative_l2",
    "mae",
    "data_mse",
    "clean_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "attack_loss_increase",
    "attack_relative_increase",
    "error_l2_norm",
    "jt_error_l2_norm",
    "sigma_input_right",
}
DIAGNOSTIC_METRICS = {
    "delta_l2_rms",
    "delta_linf",
    "cos_singular_jt_error",
    "cos_singular_attack_delta",
    "cos_jt_error_attack_delta",
    "angle_singular_jt_error_deg",
    "angle_singular_attack_delta_deg",
    "angle_jt_error_attack_delta_deg",
    "corr_singular_jt_error",
    "corr_singular_attack_delta",
    "corr_jt_error_attack_delta",
}


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def mirror_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for p in src.rglob("*"):
        if p.is_file():
            link_or_copy(p, dst / p.relative_to(src))


def clean_dirs() -> None:
    for d in [AUDIT, RELEASE]:
        if d.exists():
            shutil.rmtree(d)


def ensure_dirs() -> None:
    for base in [AUDIT, RELEASE]:
        for sub in ["figures", "data", "reports", "logs", "manifests"]:
            (base / sub).mkdir(parents=True, exist_ok=True)


def better_is_lower(metric: str) -> bool:
    return LOWER_IS_BETTER.get(metric, True)


def qvals_bh(values: Iterable[float]) -> list[float]:
    p = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return q.tolist()
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    vals = ranked * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.clip(vals, 0, 1)
    return q.tolist()


def metric_column(df: pd.DataFrame, metric: str) -> str:
    for c in [f"{metric}_plot", f"{metric}_corrected", metric]:
        if c in df.columns:
            return c
    raise KeyError(metric)


def scope_for_split(split: str) -> str:
    split = str(split)
    if split == "generalization":
        return "generalization"
    if split in {"train", "test"}:
        return split
    return "all"


def final_clean_long(eval_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse", "relative_l2", "mae", "accuracy_score", "invalid_value_fraction"]
    metrics = [m for m in metrics if any(c in eval_df.columns for c in [m, f"{m}_plot", f"{m}_corrected"])]
    baseline = eval_df[eval_df["phase"].eq("baseline_before_adversarial_training")].copy()
    baseline_rows: list[pd.DataFrame] = []
    if not baseline.empty:
        baseline = baseline.sort_values(["dataset_id", "method"]).groupby("dataset_id", as_index=False).first()
        baseline["method"] = "baseline"
        baseline_rows.append(baseline)
    train = eval_df[eval_df["phase"].eq("during_adversarial_training")].copy()
    train = train.sort_values(["method", "dataset_id", "epoch"]).groupby(["method", "dataset_id"], as_index=False).tail(1)
    base = pd.concat(baseline_rows + [train], ignore_index=True)
    rows = []
    id_cols = ["method", "dataset_id", "split", "source", "manual_tier", "manual_rank", "epoch", "path"]
    for metric in metrics:
        col = metric_column(base, metric)
        out = base[id_cols].copy()
        out["source_table"] = "clean_52dataset"
        out["metric"] = metric
        out["value"] = pd.to_numeric(base[col], errors="coerce")
        out["unit_id"] = out["dataset_id"].astype(str)
        out["scope"] = out["split"].map(scope_for_split)
        out["coverage_status"] = np.where(out["method"].eq("baseline"), "baseline_reference", "available")
        rows.append(out)
    if "rmse" in metrics:
        col = metric_column(base, "rmse")
        out = base[id_cols].copy()
        out["source_table"] = "clean_52dataset"
        out["metric"] = "data_mse"
        out["value"] = pd.to_numeric(base[col], errors="coerce") ** 2
        out["unit_id"] = out["dataset_id"].astype(str)
        out["scope"] = out["split"].map(scope_for_split)
        out["coverage_status"] = np.where(out["method"].eq("baseline"), "baseline_reference", "available")
        rows.append(out)
    clean = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    return add_ranks(clean)


def attack_long(attack_df: pd.DataFrame) -> pd.DataFrame:
    if attack_df.empty:
        return pd.DataFrame()
    df = attack_df.copy()
    df["method"] = df["method"].replace({"physics_loss": "physics"})
    metrics = ["clean_loss", "adv_loss", "loss_increase", "relative_increase", "delta_l2_rms", "delta_linf"]
    id_cols = ["method", "dataset_id", "split", "source", "manual_tier", "manual_rank", "sample_ordinal", "source_sample_index", "is_svd_sample"]
    rows = []
    for metric in [m for m in metrics if m in df.columns]:
        out = df[id_cols].copy()
        out["source_table"] = "attack_52dataset"
        out["metric"] = metric
        out["value"] = pd.to_numeric(df[metric], errors="coerce")
        out["unit_id"] = out["dataset_id"].astype(str) + "::sample" + out["source_sample_index"].astype(str)
        out["scope"] = out["split"].map(scope_for_split)
        out["coverage_status"] = "partial_smoke_2sample_per_dataset"
        rows.append(out)
    return add_ranks(pd.concat(rows, ignore_index=True).dropna(subset=["value"])) if rows else pd.DataFrame()


def svd_long(svd_df: pd.DataFrame) -> pd.DataFrame:
    if svd_df.empty:
        return pd.DataFrame()
    df = svd_df.copy()
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
    id_cols = ["method", "dataset_id", "split", "sample_index", "svd_role"]
    rows = []
    for metric in [m for m in metrics if m in df.columns]:
        out = df[id_cols].copy()
        out["source_table"] = "svd_jacobian_25sample"
        out["metric"] = metric
        out["value"] = pd.to_numeric(df[metric], errors="coerce")
        out["unit_id"] = out["dataset_id"].astype(str) + "::sample" + out["sample_index"].astype(str)
        out["scope"] = out["split"].map(scope_for_split)
        out["coverage_status"] = "partial_smoke_3samples_per_model"
        rows.append(out)
    return add_ranks(pd.concat(rows, ignore_index=True).dropna(subset=["value"])) if rows else pd.DataFrame()


def add_ranks(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["source_table", "metric", "unit_id"]
    for _, g in long.groupby(group_cols, dropna=False):
        metric = str(g["metric"].iloc[0])
        ascending = better_is_lower(metric)
        gg = g.sort_values("value", ascending=ascending, kind="mergesort").copy()
        gg["rank"] = np.arange(1, len(gg) + 1)
        gg["is_best"] = gg["rank"].eq(1)
        best = gg.iloc[0]
        runner = gg.iloc[1] if len(gg) > 1 else None
        gg["best_model"] = best["method"]
        gg["runner_up_model"] = runner["method"] if runner is not None else ""
        if runner is not None:
            adv = (runner["value"] - best["value"]) if ascending else (best["value"] - runner["value"])
        else:
            adv = np.nan
        gg["advantage_vs_runner_up"] = adv
        rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else long


def model_summary(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    parts = []
    for scope_name, gscope in [("all", long), ("generalization", long[long["scope"].eq("generalization")])]:
        if gscope.empty:
            continue
        grouped = (
            gscope.groupby(["source_table", "metric", "method", "coverage_status"], dropna=False)
            .agg(
                mean=("value", "mean"),
                std=("value", "std"),
                median=("value", "median"),
                min=("value", "min"),
                max=("value", "max"),
                n=("value", "count"),
            )
            .reset_index()
        )
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
        grouped["scope"] = scope_name
        ranked = []
        for _, gg in grouped.groupby(["scope", "source_table", "metric"], dropna=False):
            metric = str(gg["metric"].iloc[0])
            ascending = better_is_lower(metric)
            rr = gg.sort_values("mean", ascending=ascending, kind="mergesort").copy()
            rr["rank"] = np.arange(1, len(rr) + 1)
            rr["is_best"] = rr["rank"].eq(1)
            best = rr.iloc[0]
            runner = rr.iloc[1] if len(rr) > 1 else None
            rr["best_model"] = best["method"]
            rr["runner_up_model"] = runner["method"] if runner is not None else ""
            rr["advantage_vs_runner_up"] = ((runner["mean"] - best["mean"]) if ascending else (best["mean"] - runner["mean"])) if runner is not None else np.nan
            rr["metric_interpretation"] = np.where(rr["metric"].isin(QUALITY_METRICS), "quality", "diagnostic")
            ranked.append(rr)
        parts.append(pd.concat(ranked, ignore_index=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(20260614)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def paired_tests(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if long.empty:
        return pd.DataFrame()
    for scope_name, scoped in [("all", long), ("generalization", long[long["scope"].eq("generalization")])]:
        for (source_table, metric), g in scoped.groupby(["source_table", "metric"], dropna=False):
            pivot = g.pivot_table(index="unit_id", columns="method", values="value", aggfunc="mean")
            if pivot.empty:
                continue
            means = pivot.mean(skipna=True)
            ascending = better_is_lower(str(metric))
            ordered = means.sort_values(ascending=ascending)
            if ordered.empty:
                continue
            candidates: list[tuple[str, str, str]] = []
            if len(ordered) > 1:
                candidates.append(("best_vs_runner_up", str(ordered.index[0]), str(ordered.index[1])))
            if "loss3" in pivot.columns:
                non_loss3 = ordered[[m for m in ordered.index if m != "loss3"]]
                if len(non_loss3):
                    candidates.append(("loss3_vs_best_non_loss3", "loss3", str(non_loss3.index[0])))
            if "physics" in pivot.columns:
                candidates.append(("physics_vs_loss3", "physics", "loss3"))
                if len(ordered):
                    candidates.append(("physics_vs_best_overall", "physics", str(ordered.index[0])))
            seen = set()
            for comparison, a, b in candidates:
                key = (comparison, a, b)
                if key in seen or a == b or a not in pivot.columns or b not in pivot.columns:
                    continue
                seen.add(key)
                pair = pivot[[a, b]].dropna()
                n = len(pair)
                diff = (pair[a] - pair[b]).to_numpy(dtype=float) if n else np.array([])
                row = {
                    "scope": scope_name,
                    "source_table": source_table,
                    "metric": metric,
                    "comparison": comparison,
                    "model_a": a,
                    "model_b": b,
                    "n_pairs": n,
                    "model_a_mean": pair[a].mean() if n else np.nan,
                    "model_b_mean": pair[b].mean() if n else np.nan,
                    "mean_difference_a_minus_b": diff.mean() if n else np.nan,
                    "lower_is_better": ascending,
                    "test_type": "paired" if n >= 2 else "insufficient_pairs",
                }
                if n >= 2:
                    try:
                        tres = stats.ttest_rel(pair[a], pair[b], nan_policy="omit")
                        row["paired_t_stat"] = float(tres.statistic)
                        row["paired_t_p_two_sided"] = float(tres.pvalue)
                    except Exception:
                        row["paired_t_stat"] = np.nan
                        row["paired_t_p_two_sided"] = np.nan
                    try:
                        if np.allclose(diff, 0):
                            row["wilcoxon_stat"] = 0.0
                            row["wilcoxon_p_two_sided"] = 1.0
                        else:
                            wres = stats.wilcoxon(pair[a], pair[b], zero_method="wilcox", alternative="two-sided")
                            row["wilcoxon_stat"] = float(wres.statistic)
                            row["wilcoxon_p_two_sided"] = float(wres.pvalue)
                    except Exception:
                        row["wilcoxon_stat"] = np.nan
                        row["wilcoxon_p_two_sided"] = np.nan
                    lo, hi = bootstrap_ci(diff)
                    row["bootstrap_mean_diff_ci95_low"] = lo
                    row["bootstrap_mean_diff_ci95_high"] = hi
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_q_bh_fdr"] = qvals_bh(out.get("paired_t_p_two_sided", pd.Series(dtype=float)))
        out["wilcoxon_q_bh_fdr"] = qvals_bh(out.get("wilcoxon_p_two_sided", pd.Series(dtype=float)))
    return out


def first_place_counts(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    best = long[long["is_best"]].copy()
    rows = []
    for scope_name, scoped in [("all", best), ("generalization", best[best["scope"].eq("generalization")])]:
        counts = (
            scoped.groupby(["source_table", "metric", "method"], dropna=False)
            .size()
            .reset_index(name="first_place_count")
        )
        totals = scoped.groupby(["source_table", "metric"], dropna=False).size().reset_index(name="total_units")
        counts = counts.merge(totals, on=["source_table", "metric"], how="left")
        counts["scope"] = scope_name
        counts["first_place_fraction"] = counts["first_place_count"] / counts["total_units"].replace(0, np.nan)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def scalar_correlations(svd_df: pd.DataFrame) -> pd.DataFrame:
    if svd_df.empty:
        return pd.DataFrame()
    df = svd_df.copy()
    df["method"] = df["method"].replace({"physics_loss": "physics"})
    pairs = [
        ("attack_loss_increase", "jt_error_l2_norm", "J_error^T_error_norm"),
        ("attack_loss_increase", "sigma_input_right", "spectral_norm_J_error"),
        ("attack_loss_increase", "error_l2_norm", "clean_residual_l2_proxy"),
        ("attack_loss_increase", "clean_loss", "clean_loss"),
        ("clean_loss", "jt_error_l2_norm", "J_error^T_error_norm"),
        ("clean_loss", "sigma_input_right", "spectral_norm_J_error"),
    ]
    groups = [
        ("all_models_all_samples", df),
        ("generalization_only", df[df["split"].eq("generalization")]),
        ("old_model_subset", df[df["method"].isin(["baseline", "loss1", "loss2", "loss3", "physics"])]),
        ("random_subset", df[df["method"].isin(["random_clean", "random_solver"])]),
        ("physics_subset", df[df["method"].eq("physics")]),
    ]
    rows = []
    for group_name, g in groups:
        for a, b, hypothesis in pairs:
            if a not in g.columns or b not in g.columns:
                rows.append({"group": group_name, "metric_a": a, "metric_b": b, "hypothesis": hypothesis, "n": 0, "coverage_status": "missing_column"})
                continue
            pair = g[[a, b]].dropna()
            row: dict[str, Any] = {
                "group": group_name,
                "metric_a": a,
                "metric_b": b,
                "hypothesis": hypothesis,
                "n": len(pair),
                "coverage_status": "available" if len(pair) >= 3 else "insufficient_pairs",
            }
            if len(pair) >= 3:
                try:
                    pr = stats.pearsonr(pair[a], pair[b])
                    row["pearson_r"] = float(pr.statistic)
                    row["pearson_p"] = float(pr.pvalue)
                except Exception:
                    row["pearson_r"] = row["pearson_p"] = np.nan
                try:
                    sr = stats.spearmanr(pair[a], pair[b])
                    row["spearman_r"] = float(sr.statistic)
                    row["spearman_p"] = float(sr.pvalue)
                except Exception:
                    row["spearman_r"] = row["spearman_p"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pearson_q_bh_fdr"] = qvals_bh(out.get("pearson_p", pd.Series(dtype=float)))
        out["spearman_q_bh_fdr"] = qvals_bh(out.get("spearman_p", pd.Series(dtype=float)))
        out["abs_pearson_r"] = pd.to_numeric(out.get("pearson_r"), errors="coerce").abs()
        out = out.sort_values(["group", "abs_pearson_r"], ascending=[True, False], na_position="last")
    return out


def coverage_notes(eval_df: pd.DataFrame, attack_df: pd.DataFrame, svd_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_epoch = eval_df.groupby("method")["epoch"].max().to_dict()
    for method in ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]:
        rows.append({
            "component": "training_curves_52dataset_eval",
            "method": method,
            "coverage_status": "complete_or_near_target" if int(max_epoch.get(method, 0)) >= 3000 else "partial",
            "observed_n": int(max_epoch.get(method, 0)),
            "target_n": 3000,
            "note": f"Max epoch observed in source evaluation CSV is {int(max_epoch.get(method, 0))}.",
        })
    rows.extend([
        {
            "component": "physics_residual_52dataset_eval",
            "method": "all",
            "coverage_status": "missing_or_partial",
            "observed_n": 0,
            "target_n": 52,
            "note": "Clean evaluation tables have RMSE/Relative-L2/MAE/accuracy but no per-dataset PDE residual columns; physics training logs include darcy_physics_metric only for physics runs.",
        },
        {
            "component": "attack_robustness_52dataset_50sample",
            "method": "all",
            "coverage_status": "partial_smoke",
            "observed_n": int(len(attack_df)),
            "target_n": 52 * 50 * 7,
            "note": "Available attack table is smoke coverage with 2 samples per dataset/model; not recomputed here.",
        },
        {
            "component": "svd_jacobian_25sample",
            "method": "all",
            "coverage_status": "partial_smoke",
            "observed_n": int(len(svd_df)),
            "target_n": 25 * 7,
            "note": "Available SVD/Jacobian table has 3 samples per model; top50/top100 and full fixed 25-sample coverage were not found locally.",
        },
        {
            "component": "dense_group00_to_group05",
            "method": "all",
            "coverage_status": "partial_existing_heatmaps",
            "observed_n": len(list((AUDIT / "figures" / "dense_existing" / "group05_loss3_advantage").glob("*.png"))) if AUDIT.exists() else 0,
            "target_n": 6,
            "note": "Existing 2D Darcy heatmaps for loss3-advantage group05 were reused; full group00..group05 all variants were not found and not regenerated.",
        },
        {
            "component": "quality_vs_diagnostic_metric_policy",
            "method": "all",
            "coverage_status": "documented",
            "observed_n": 1,
            "target_n": 1,
            "note": "Delta norm, cosine/angle, and model spectral norm are reported as diagnostics, not as standalone model-quality evidence.",
        },
    ])
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        cells = []
        for c in cols:
            val = str(row[c])
            if c in {"method", "best_model"} and bool(row.get("is_best", False)):
                val = f"**{val}**"
            if c == "best_model" and val not in {"", "nan"}:
                val = f"**{val.strip('*')}**"
            cells.append(val)
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def write_reports(summary: pd.DataFrame, first_counts: pd.DataFrame, tests: pd.DataFrame, corrs: pd.DataFrame, coverage: pd.DataFrame) -> None:
    reports = {
        AUDIT / "reports" / "DARCY_CFLOW_AUDIT_REPORT.md",
        RELEASE / "reports" / "DARCY_CFLOW_AUDIT_REPORT.md",
    }
    best = summary[summary["is_best"]].copy() if not summary.empty else pd.DataFrame()
    clean_best = best[best["source_table"].eq("clean_52dataset")][["scope", "metric", "best_model", "runner_up_model", "advantage_vs_runner_up", "n"]]
    attack_best = best[best["source_table"].eq("attack_52dataset")][["scope", "metric", "best_model", "runner_up_model", "advantage_vs_runner_up", "n"]]
    svd_best = best[best["source_table"].eq("svd_jacobian_25sample")][["scope", "metric", "best_model", "runner_up_model", "advantage_vs_runner_up", "n"]]
    mechanism = corrs[corrs["group"].eq("all_models_all_samples")].copy()
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Darcy cflow Time-Matched Audit",
        "",
        f"Generated: `{now}`",
        "",
        "This package audits existing 2D DarcyFlow / Darcy cflow artifacts. It reuses available models, clean evaluation tables, attack smoke traces, SVD/Jacobian smoke metrics, and 2D heatmaps. It does not rerun training, attacks, SVD/Jacobian, or dense image generation.",
        "",
        "## Clean / Generalization Conclusions",
        "",
        markdown_table(clean_best, max_rows=30),
        "",
        "## Physics Residual Conclusions",
        "",
        "Full per-dataset physics/PDE residual columns were not found in the clean 52-dataset evaluation tables. Physics-run training logs include `darcy_physics_metric`, so physics residual is documented as partial coverage and is not merged with RMSE/Relative-L2 conclusions.",
        "",
        "## Attack Robustness Conclusions",
        "",
        markdown_table(attack_best, max_rows=30),
        "",
        "Attack delta RMS/L2/Linf are included as process diagnostics only; they are controlled by attack budget and are not treated as direct model-quality evidence.",
        "",
        "## 25-Sample Jacobian / SVD Conclusions",
        "",
        markdown_table(svd_best, max_rows=30),
        "",
        "The available SVD/Jacobian evidence is smoke coverage: 3 samples per model, not the full requested fixed 25 samples. Top50/top100 model-solver subspace artifacts were not found locally and were not recomputed.",
        "",
        "## Mechanism Correlations",
        "",
        markdown_table(mechanism[["metric_a", "metric_b", "hypothesis", "n", "pearson_r", "pearson_p", "pearson_q_bh_fdr", "spearman_r", "spearman_p", "spearman_q_bh_fdr", "coverage_status"]], max_rows=20),
        "",
        "Mechanism interpretation is limited by smoke sample size. The report explicitly records `n` for every correlation. `J_error^T error` correlations are compared against spectral-norm correlations where both columns exist.",
        "",
        "## First-Place Counts",
        "",
        markdown_table(first_counts.sort_values(["source_table", "metric", "scope", "first_place_count"], ascending=[True, True, True, False]), max_rows=60),
        "",
        "## Significance / Bootstrap",
        "",
        markdown_table(tests[["scope", "source_table", "metric", "comparison", "model_a", "model_b", "n_pairs", "mean_difference_a_minus_b", "paired_t_p_two_sided", "paired_t_q_bh_fdr", "wilcoxon_p_two_sided", "wilcoxon_q_bh_fdr", "bootstrap_mean_diff_ci95_low", "bootstrap_mean_diff_ci95_high", "test_type"]], max_rows=80),
        "",
        "## Coverage / Reuse / Recompute",
        "",
        markdown_table(coverage, max_rows=80),
        "",
        "## What Not To Over-Claim",
        "",
        "- Do not claim loss3 is best on every metric; each metric has its own ranked table.",
        "- Do not treat delta norm as model quality; it is an attack-budget/process diagnostic.",
        "- Do not treat model spectral norm alone as model-solver closeness; error-Jacobian quantities are the relevant local robustness evidence.",
        "- Physics residual can disagree with RMSE/Relative-L2 and attack robustness; it must be reported separately.",
        "",
    ]
    for report in reports:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(base: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p.relative_to(ROOT)),
                "relative_to_bundle": str(p.relative_to(base)),
                "bytes": p.stat().st_size,
                "suffix": p.suffix,
            })
    df = pd.DataFrame(rows)
    df.to_csv(base / "manifests" / "file_manifest.csv", index=False)
    (base / "manifests" / "file_manifest.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df


def sync_generated_to_release() -> None:
    for sub in ["data", "reports", "manifests"]:
        for p in (AUDIT / sub).rglob("*"):
            if p.is_file():
                link_or_copy(p, RELEASE / p.relative_to(AUDIT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean:
        clean_dirs()
    ensure_dirs()
    mirror_tree(SRC_AUDIT, AUDIT)
    mirror_tree(SRC_RELEASE if SRC_RELEASE.exists() else SRC_AUDIT, RELEASE)

    eval_df = read_csv(EVAL_METRICS)
    attack_df = read_csv(ATTACK) if ATTACK.exists() else pd.DataFrame()
    svd_df = read_csv(SVD) if SVD.exists() else pd.DataFrame()
    if not attack_df.empty:
        attack_df["method"] = attack_df["method"].replace({"physics_loss": "physics"})
    if not svd_df.empty:
        svd_df["method"] = svd_df["method"].replace({"physics_loss": "physics"})

    clean = final_clean_long(eval_df)
    attack = attack_long(attack_df)
    svd = svd_long(svd_df)
    all_long = pd.concat([clean, attack, svd], ignore_index=True)
    summary = model_summary(all_long)
    tests = paired_tests(all_long)
    first_counts = first_place_counts(all_long)
    corrs = scalar_correlations(svd_df)
    coverage = coverage_notes(eval_df, attack_df, svd_df)

    data_dir = AUDIT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(data_dir / "cflow_clean_52dataset_metric_long_ranked.csv", index=False)
    attack.to_csv(data_dir / "cflow_attack_metric_long_ranked.csv", index=False)
    svd.to_csv(data_dir / "cflow_svd_jacobian_metric_long_ranked.csv", index=False)
    all_long.to_csv(data_dir / "cflow_all_scalar_metric_long_ranked.csv", index=False)
    summary.to_csv(data_dir / "cflow_metric_model_summary_ranked.csv", index=False)
    first_counts.to_csv(data_dir / "cflow_per_sample_first_place_counts.csv", index=False)
    tests.to_csv(data_dir / "cflow_paired_significance_bootstrap_tests.csv", index=False)
    corrs.to_csv(data_dir / "cflow_mechanism_scalar_correlations.csv", index=False)
    coverage.to_csv(data_dir / "cflow_coverage_reuse_recompute_notes.csv", index=False)
    pd.DataFrame(
        [
            {"metric_family": "quality", "metric": m, "note": "May be used as model-quality evidence when coverage is adequate."}
            for m in sorted(QUALITY_METRICS)
        ]
        + [
            {"metric_family": "diagnostic", "metric": m, "note": "Diagnostic/process quantity; do not use alone as model-quality evidence."}
            for m in sorted(DIAGNOSTIC_METRICS)
        ]
    ).to_csv(data_dir / "cflow_quality_vs_diagnostic_metric_policy.csv", index=False)
    dense_files = sorted((AUDIT / "figures" / "dense_existing").rglob("*.png"))
    pd.DataFrame(
        [{"figure_path": str(p.relative_to(AUDIT)), "bytes": p.stat().st_size, "coverage_status": "reused_existing_2d_heatmap"} for p in dense_files]
    ).to_csv(AUDIT / "manifests" / "cflow_dense_heatmap_manifest.csv", index=False)
    if RANDOM_STATUS.exists():
        link_or_copy(RANDOM_STATUS, AUDIT / "manifests" / "random_3500_replot_status.json")

    write_reports(summary, first_counts, tests, corrs, coverage)
    sync_generated_to_release()

    for _ in range(3):
        audit_manifest = write_manifest(AUDIT)
        release_manifest = write_manifest(RELEASE)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "audit_dir": str(AUDIT.relative_to(ROOT)),
            "release_dir": str(RELEASE.relative_to(ROOT)),
            "audit_files": int(len([p for p in AUDIT.rglob('*') if p.is_file()])),
            "audit_bytes": int(sum(p.stat().st_size for p in AUDIT.rglob('*') if p.is_file())),
            "release_files": int(len([p for p in RELEASE.rglob('*') if p.is_file()])),
            "release_bytes": int(sum(p.stat().st_size for p in RELEASE.rglob('*') if p.is_file())),
            "source_audit": str(SRC_AUDIT.relative_to(ROOT)),
            "did_rerun_training": False,
            "did_rerun_attack": False,
            "did_rerun_svd_jacobian": False,
            "did_generate_new_dense_heatmaps": False,
        }
        write_json(AUDIT / "manifests" / "cflow_bundle_summary.json", payload)
        write_json(RELEASE / "manifests" / "cflow_bundle_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
