#!/usr/bin/env python3
"""Per-dataset Welch tests for Burgers clean 52-dataset metrics.

This uses the existing per-dataset clean mean/std/n summaries for all six
models. It does not rerun model evaluation. Because a unified six-model
per-sample clean table is not present, these are Welch tests from summary
statistics, not paired tests.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
OLD4 = REPO / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_clean_metrics.csv"
RANDOM = REPO / "forensics/burgers_random_solver7860_clean8000_full_suite_20260614/clean_loss/per_dataset_clean_metrics.csv"
FINAL_CLEAN = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/clean_52dataset_six_models_selected_worktime.csv"
AUDIT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
OUT_DATA = AUDIT / "data/clean52_per_dataset_welch_tests_20260614"
OUT_REPORT = AUDIT / "reports/burgers_clean52_per_dataset_welch_tests_20260614.md"
DOC = REPO / "docs/burgers_clean52_per_dataset_welch_tests_20260614.md"

MODELS = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
METRICS = ["rmse", "relative_l2", "mse"]


def bh_fdr(pvals: list[float]) -> list[float]:
    arr = np.asarray([1.0 if not np.isfinite(p) else max(0.0, min(1.0, p)) for p in pvals], dtype=float)
    n = arr.size
    order = np.argsort(arr)
    ranked = arr[order]
    q_sorted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        q_sorted[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = q_sorted
    return out.tolist()


def welch_from_stats(mean_a: float, std_a: float, n_a: int, mean_b: float, std_b: float, n_b: int) -> dict[str, float]:
    """Return Welch t-test stats for a-b where smaller a is better."""
    if n_a <= 1 or n_b <= 1:
        return {"t": math.nan, "df": math.nan, "p_less": math.nan, "p_two_sided": math.nan}
    va = float(std_a) ** 2 / float(n_a)
    vb = float(std_b) ** 2 / float(n_b)
    denom = math.sqrt(max(va + vb, 0.0))
    if denom == 0.0:
        diff = mean_a - mean_b
        if diff < 0:
            return {"t": -math.inf, "df": math.inf, "p_less": 0.0, "p_two_sided": 0.0}
        if diff > 0:
            return {"t": math.inf, "df": math.inf, "p_less": 1.0, "p_two_sided": 0.0}
        return {"t": 0.0, "df": math.inf, "p_less": 1.0, "p_two_sided": 1.0}
    t = (float(mean_a) - float(mean_b)) / denom
    df_num = (va + vb) ** 2
    df_den = (va**2 / (n_a - 1)) + (vb**2 / (n_b - 1))
    df = df_num / df_den if df_den > 0 else math.inf
    p_less = float(stats.t.cdf(t, df))
    p_two = float(2.0 * min(p_less, 1.0 - p_less))
    return {"t": float(t), "df": float(df), "p_less": p_less, "p_two_sided": min(1.0, p_two)}


def fmt(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v) >= 1e4 or (abs(v) < 1e-4 and v != 0):
        return f"{v:.3e}"
    return f"{v:.6g}"


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    work = df.loc[:, cols].head(max_rows).copy()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def get_stats(row: pd.Series, model: str, metric: str) -> tuple[float, float, int]:
    mean = float(row[f"{model}_{metric}_mean"])
    std = float(row[f"{model}_{metric}_std"])
    if model.startswith("random_"):
        n = int(row["random_n_samples"])
    else:
        n = int(row["n_samples"])
    return mean, std, n


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    old4 = pd.read_csv(OLD4)
    random = pd.read_csv(RANDOM)
    final = pd.read_csv(FINAL_CLEAN)

    old_cols = ["dataset_id"]
    random_cols = ["dataset_id"]
    for model in ["baseline", "loss1", "loss2", "loss3"]:
        for metric in METRICS:
            old_cols.extend([f"{model}_{metric}_mean", f"{model}_{metric}_std"])
    for model in ["random_clean_y", "random_solver_y"]:
        for metric in METRICS:
            random_cols.extend([f"{model}_{metric}_mean", f"{model}_{metric}_std"])

    merged = final[
        [
            "dataset_order",
            "split",
            "dataset_id",
            "old4_dataset_id",
            "random_dataset_id",
            "n_samples",
            "random_n_samples",
            "family",
            "display_label",
            "description",
        ]
    ].merge(
        old4[old_cols],
        left_on="old4_dataset_id",
        right_on="dataset_id",
        how="left",
        suffixes=("", "_old4_join"),
    )
    merged = merged.drop(columns=["dataset_id_old4_join"])
    merged = merged.merge(
        random[random_cols],
        left_on="random_dataset_id",
        right_on="dataset_id",
        how="left",
        suffixes=("", "_random_join"),
    ).drop(columns=["dataset_id_random_join"])

    missing_cols = [c for c in merged.columns if c.endswith("_mean") or c.endswith("_std")]
    if merged[missing_cols].isna().any().any():
        bad = merged.loc[merged[missing_cols].isna().any(axis=1), ["dataset_order", "dataset_id", "old4_dataset_id", "random_dataset_id"]]
        raise SystemExit(f"missing joined clean stats:\n{bad.to_string(index=False)}")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        for metric in METRICS:
            stats_by_model = {model: get_stats(row, model, metric) for model in MODELS}
            means = {model: vals[0] for model, vals in stats_by_model.items()}
            stds = {model: vals[1] for model, vals in stats_by_model.items()}
            ns = {model: vals[2] for model, vals in stats_by_model.items()}
            ordered = sorted(MODELS, key=lambda m: means[m])
            best_model = ordered[0]
            second_model = ordered[1]
            comparison = min((m for m in MODELS if m != "loss3"), key=lambda m: means[m])
            loss3_mean, loss3_std, loss3_n = stats_by_model["loss3"]
            comp_mean, comp_std, comp_n = stats_by_model[comparison]
            test = welch_from_stats(loss3_mean, loss3_std, loss3_n, comp_mean, comp_std, comp_n)
            rows.append(
                {
                    "dataset_order": int(row["dataset_order"]),
                    "split": row["split"],
                    "dataset_id": row["dataset_id"],
                    "metric": metric,
                    "n_loss3": loss3_n,
                    "n_comparison": comp_n,
                    "best_model": best_model,
                    "second_model": second_model,
                    "loss3_rank": ordered.index("loss3") + 1,
                    "comparison_model": comparison,
                    "comparison_is_runner_up_when_loss3_best": bool(best_model == "loss3" and comparison == second_model),
                    "loss3_mean": loss3_mean,
                    "loss3_std": loss3_std,
                    "comparison_mean": comp_mean,
                    "comparison_std": comp_std,
                    "mean_advantage_comparison_minus_loss3": comp_mean - loss3_mean,
                    "relative_advantage_vs_comparison": (comp_mean - loss3_mean) / comp_mean if comp_mean != 0 else math.nan,
                    "welch_t_loss3_minus_comparison": test["t"],
                    "welch_df": test["df"],
                    "welch_p_loss3_lower_one_sided": test["p_less"],
                    "welch_p_two_sided": test["p_two_sided"],
                    **{f"{model}_mean": means[model] for model in MODELS},
                    **{f"{model}_std": stds[model] for model in MODELS},
                    **{f"{model}_n": ns[model] for model in MODELS},
                }
            )

    out = pd.DataFrame(rows)
    out["welch_q_loss3_lower_bh_fdr_all156"] = bh_fdr(out["welch_p_loss3_lower_one_sided"].tolist())
    out["loss3_significantly_lower_q05"] = (
        (out["mean_advantage_comparison_minus_loss3"] > 0)
        & (out["welch_q_loss3_lower_bh_fdr_all156"] < 0.05)
    )
    out["loss3_is_best"] = out["best_model"].eq("loss3")

    metric_summary = (
        out.groupby("metric", sort=False)
        .agg(
            dataset_rows=("dataset_id", "count"),
            loss3_best_rows=("loss3_is_best", "sum"),
            loss3_significantly_lower_rows=("loss3_significantly_lower_q05", "sum"),
            mean_advantage=("mean_advantage_comparison_minus_loss3", "mean"),
            min_advantage=("mean_advantage_comparison_minus_loss3", "min"),
            max_advantage=("mean_advantage_comparison_minus_loss3", "max"),
        )
        .reset_index()
    )
    split_summary = (
        out.groupby(["metric", "split"], sort=False)
        .agg(
            dataset_rows=("dataset_id", "count"),
            loss3_best_rows=("loss3_is_best", "sum"),
            loss3_significantly_lower_rows=("loss3_significantly_lower_q05", "sum"),
        )
        .reset_index()
    )
    best_counts = (
        out.groupby(["metric", "best_model"], sort=False)
        .size()
        .reset_index(name="dataset_rows")
        .sort_values(["metric", "dataset_rows"], ascending=[True, False])
    )

    out_path = OUT_DATA / "clean52_per_dataset_loss3_vs_best_other_welch_tests.csv"
    compact_path = OUT_DATA / "clean52_per_dataset_loss3_vs_best_other_welch_tests_compact.csv"
    metric_summary_path = OUT_DATA / "clean52_per_dataset_welch_metric_summary.csv"
    split_summary_path = OUT_DATA / "clean52_per_dataset_welch_split_summary.csv"
    best_counts_path = OUT_DATA / "clean52_per_dataset_best_model_counts.csv"

    out.to_csv(out_path, index=False)
    compact_cols = [
        "dataset_order",
        "split",
        "dataset_id",
        "metric",
        "n_loss3",
        "n_comparison",
        "best_model",
        "second_model",
        "loss3_rank",
        "comparison_model",
        "loss3_mean",
        "loss3_std",
        "comparison_mean",
        "comparison_std",
        "mean_advantage_comparison_minus_loss3",
        "welch_p_loss3_lower_one_sided",
        "welch_q_loss3_lower_bh_fdr_all156",
        "loss3_significantly_lower_q05",
    ]
    out[compact_cols].to_csv(compact_path, index=False)
    metric_summary.to_csv(metric_summary_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)
    best_counts.to_csv(best_counts_path, index=False)

    worst = out.sort_values(["metric", "mean_advantage_comparison_minus_loss3"]).groupby("metric", sort=False).head(6)

    report = "\n".join(
        [
            "# Burgers Clean 52 Per-Dataset Welch Tests, 20260614",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "This report answers the per-dataset clean-metric question for the 52-dataset six-model clean table. It uses existing per-dataset mean/std/n summaries and does not rerun evaluation.",
            "",
            "## Important Method Boundary",
            "",
            "These are Welch tests from summary statistics, not paired per-sample tests. A unified six-model per-sample clean table is not present in the final bundle. The old4 clean source has per-dataset mean/std/sem and old4 internal paired tests; the random-model clean source has per-sample and per-dataset summaries. The six-model strict common clean comparison can therefore be tested per dataset only with summary-stat Welch tests unless clean per-sample values are recomputed or recovered for all six models.",
            "",
            "## What Was Tested",
            "",
            "- Dataset rows: 52.",
            "- Metrics: RMSE, Relative L2, MSE.",
            "- Total per-dataset tests: 156.",
            "- Direction: lower is better.",
            "- Comparison: `loss3` versus the best non-loss3 model for the same dataset and metric. When `loss3` is best, that comparison is the runner-up.",
            "- Multiple-testing correction: BH/FDR over all 156 clean Welch tests.",
            "",
            "## Metric Summary",
            "",
            md_table(metric_summary, list(metric_summary.columns), max_rows=10),
            "",
            "## Best-Model Counts",
            "",
            md_table(best_counts, list(best_counts.columns), max_rows=20),
            "",
            "## Split Summary",
            "",
            md_table(split_summary, list(split_summary.columns), max_rows=20),
            "",
            "## Rows With Smallest Loss3 Advantage",
            "",
            md_table(
                worst,
                [
                    "metric",
                    "dataset_order",
                    "split",
                    "dataset_id",
                    "best_model",
                    "second_model",
                    "loss3_rank",
                    "comparison_model",
                    "loss3_mean",
                    "comparison_mean",
                    "mean_advantage_comparison_minus_loss3",
                    "welch_q_loss3_lower_bh_fdr_all156",
                    "loss3_significantly_lower_q05",
                ],
                max_rows=18,
            ),
            "",
            "## Output CSVs",
            "",
            f"- `{out_path.relative_to(REPO)}`",
            f"- `{compact_path.relative_to(REPO)}`",
            f"- `{metric_summary_path.relative_to(REPO)}`",
            f"- `{split_summary_path.relative_to(REPO)}`",
            f"- `{best_counts_path.relative_to(REPO)}`",
        ]
    )
    OUT_REPORT.write_text(report + "\n")
    DOC.write_text(report + "\n")

    print(
        json.dumps(
            {
                "rows": int(len(out)),
                "datasets": int(out["dataset_id"].nunique()),
                "metrics": METRICS,
                "metric_summary": metric_summary.to_dict(orient="records"),
                "report": str(OUT_REPORT.relative_to(REPO)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
