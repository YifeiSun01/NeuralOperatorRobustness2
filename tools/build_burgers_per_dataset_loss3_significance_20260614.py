#!/usr/bin/env python3
"""Per-dataset loss3-vs-best-other significance for strict Burgers attack losses.

This reads the already-computed strict latest P2Q2 attack NPZ files. It does not
rerun attacks. The paired unit is one manifest sample within one dataset.
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
OLD4 = REPO / "forensics/burgers_latest_old4_widevis_full52_p2q2_20step_20260614/p2q2_attack"
RANDOM = REPO / "forensics/burgers_random_solver7860_clean8000_full_suite_20260614/p2q2_attack"
AUDIT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
OUT_DATA = AUDIT / "data/per_dataset_loss3_significance_20260614"
OUT_REPORT = AUDIT / "reports/burgers_per_dataset_loss3_significance_20260614.md"
DOC = REPO / "docs/burgers_per_dataset_loss3_significance_20260614.md"

MODELS = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
MODEL_ROOTS = {
    "baseline": OLD4,
    "loss1": OLD4,
    "loss2": OLD4,
    "loss3": OLD4,
    "random_clean_y": RANDOM,
    "random_solver_y": RANDOM,
}
METRICS = [
    ("initial_loss", "Clean initial MSE on attack manifest"),
    ("final_loss", "Attack final MSE"),
    ("attack_loss_increase", "Attack loss increase"),
]


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


def load_manifest(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    df = pd.DataFrame(rows)
    df["manifest_index"] = np.arange(len(df), dtype=int)
    return df


def load_losses(model: str) -> dict[str, np.ndarray]:
    path = MODEL_ROOTS[model] / model / "losses_and_delta_rms_by_sample.npz"
    z = np.load(path, allow_pickle=False)
    return {
        "initial_loss": z["initial_loss"].astype(float),
        "final_loss": z["final_loss"].astype(float),
        "attack_loss_increase": (z["final_loss"] - z["initial_loss"]).astype(float),
        "final_delta_rms": z["final_delta_rms"].astype(float),
    }


def safe_t(diff: np.ndarray, alternative: str) -> float:
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return math.nan
    if float(np.nanstd(diff, ddof=1)) == 0.0:
        mean = float(np.nanmean(diff))
        if alternative == "greater":
            return 0.0 if mean > 0 else (1.0 if mean < 0 else 1.0)
        return 0.0 if mean < 0 else (1.0 if mean > 0 else 1.0)
    return float(stats.ttest_1samp(diff, 0.0, alternative=alternative).pvalue)


def safe_wilcoxon(diff: np.ndarray, alternative: str) -> float:
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return math.nan
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox").pvalue)
    except ValueError:
        return math.nan


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 20_000) -> tuple[float, float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return math.nan, math.nan, math.nan
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    p_le_0 = (float(np.sum(means <= 0.0)) + 1.0) / (n_boot + 1.0)
    return float(low), float(high), p_le_0


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


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    old_manifest = load_manifest(OLD4 / "manifest.json")
    random_manifest = load_manifest(RANDOM / "manifest.json")
    compare_cols = ["global_sample_id", "split", "dataset_id", "source_index", "dataset_sample_offset"]
    if not old_manifest[compare_cols].equals(random_manifest[compare_cols]):
        raise SystemExit("old4 and random attack manifests do not match")

    manifest = old_manifest
    losses = {model: load_losses(model) for model in MODELS}
    n_manifest = len(manifest)
    for model, vals in losses.items():
        for metric, arr in vals.items():
            if len(arr) != n_manifest:
                raise SystemExit(f"{model}:{metric} length {len(arr)} != manifest length {n_manifest}")

    rng = np.random.default_rng(20260614)
    rows: list[dict[str, object]] = []

    dataset_meta = (
        manifest.groupby(["dataset_id", "split"], sort=False)
        .agg(
            sample_count=("manifest_index", "size"),
            first_manifest_index=("manifest_index", "min"),
            last_manifest_index=("manifest_index", "max"),
        )
        .reset_index()
    )
    dataset_order = {d: i for i, d in enumerate(dataset_meta["dataset_id"].tolist())}

    for _, meta in dataset_meta.iterrows():
        dataset_id = str(meta["dataset_id"])
        split = str(meta["split"])
        idx = manifest.index[manifest["dataset_id"] == dataset_id].to_numpy(dtype=int)
        for metric, label in METRICS:
            means = {model: float(np.mean(losses[model][metric][idx])) for model in MODELS}
            stds = {model: float(np.std(losses[model][metric][idx], ddof=1)) for model in MODELS}
            ordered = sorted(MODELS, key=lambda m: means[m])
            best_model = ordered[0]
            second_model = ordered[1]
            best_other = min((m for m in MODELS if m != "loss3"), key=lambda m: means[m])
            loss3 = losses["loss3"][metric][idx]
            other = losses[best_other][metric][idx]
            diff = other - loss3
            ci_low, ci_high, boot_p = bootstrap_ci(diff, rng)
            rows.append(
                {
                    "dataset_order": dataset_order[dataset_id] + 1,
                    "split": split,
                    "dataset_id": dataset_id,
                    "metric": metric,
                    "metric_label": label,
                    "n_pairs": int(idx.size),
                    "best_model": best_model,
                    "second_model": second_model,
                    "loss3_rank": ordered.index("loss3") + 1,
                    "comparison_model": best_other,
                    "comparison_is_runner_up_when_loss3_best": bool(best_model == "loss3" and best_other == second_model),
                    "loss3_mean": means["loss3"],
                    "loss3_std": stds["loss3"],
                    "comparison_mean": means[best_other],
                    "comparison_std": stds[best_other],
                    "mean_advantage_comparison_minus_loss3": float(np.mean(diff)),
                    "std_advantage": float(np.std(diff, ddof=1)),
                    "loss3_sample_wins_vs_comparison": int(np.sum(loss3 < other)),
                    "loss3_sample_ties_vs_comparison": int(np.sum(np.isclose(loss3, other))),
                    "loss3_sample_losses_vs_comparison": int(np.sum(loss3 > other)),
                    "paired_t_p_loss3_better_one_sided": safe_t(diff, "greater"),
                    "paired_t_p_loss3_worse_one_sided": safe_t(diff, "less"),
                    "wilcoxon_p_loss3_better_one_sided": safe_wilcoxon(diff, "greater"),
                    "wilcoxon_p_loss3_worse_one_sided": safe_wilcoxon(diff, "less"),
                    "bootstrap_mean_advantage_ci95_low": ci_low,
                    "bootstrap_mean_advantage_ci95_high": ci_high,
                    "bootstrap_p_mean_advantage_le_0": boot_p,
                    **{f"{model}_mean": means[model] for model in MODELS},
                    **{f"{model}_std": stds[model] for model in MODELS},
                }
            )

    out = pd.DataFrame(rows)
    out["paired_t_q_loss3_better_bh_fdr_all156"] = bh_fdr(out["paired_t_p_loss3_better_one_sided"].tolist())
    out["paired_t_q_loss3_worse_bh_fdr_all156"] = bh_fdr(out["paired_t_p_loss3_worse_one_sided"].tolist())
    out["wilcoxon_q_loss3_better_bh_fdr_all156"] = bh_fdr(out["wilcoxon_p_loss3_better_one_sided"].tolist())
    out["wilcoxon_q_loss3_worse_bh_fdr_all156"] = bh_fdr(out["wilcoxon_p_loss3_worse_one_sided"].tolist())
    out["loss3_significantly_better_q05"] = (
        (out["mean_advantage_comparison_minus_loss3"] > 0)
        & (out["paired_t_q_loss3_better_bh_fdr_all156"] < 0.05)
        & (out["bootstrap_mean_advantage_ci95_low"] > 0)
    )
    out["loss3_significantly_worse_q05"] = (
        (out["mean_advantage_comparison_minus_loss3"] < 0)
        & (out["paired_t_q_loss3_worse_bh_fdr_all156"] < 0.05)
        & (out["bootstrap_mean_advantage_ci95_high"] < 0)
    )

    metric_summary = (
        out.groupby("metric", sort=False)
        .agg(
            dataset_rows=("dataset_id", "count"),
            loss3_best_rows=("best_model", lambda s: int((s == "loss3").sum())),
            loss3_rank1_rows=("loss3_rank", lambda s: int((s == 1).sum())),
            loss3_significantly_better_rows=("loss3_significantly_better_q05", "sum"),
            loss3_significantly_worse_rows=("loss3_significantly_worse_q05", "sum"),
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
            loss3_best_rows=("best_model", lambda s: int((s == "loss3").sum())),
            loss3_significantly_better_rows=("loss3_significantly_better_q05", "sum"),
            loss3_significantly_worse_rows=("loss3_significantly_worse_q05", "sum"),
        )
        .reset_index()
    )

    out_path = OUT_DATA / "per_dataset_loss3_vs_best_other_significance.csv"
    compact_path = OUT_DATA / "per_dataset_loss3_vs_best_other_significance_compact.csv"
    metric_summary_path = OUT_DATA / "per_dataset_loss3_significance_metric_summary.csv"
    split_summary_path = OUT_DATA / "per_dataset_loss3_significance_split_summary.csv"
    out.to_csv(out_path, index=False)
    compact_cols = [
        "dataset_order",
        "split",
        "dataset_id",
        "metric",
        "n_pairs",
        "best_model",
        "second_model",
        "loss3_rank",
        "comparison_model",
        "loss3_mean",
        "comparison_mean",
        "mean_advantage_comparison_minus_loss3",
        "paired_t_p_loss3_better_one_sided",
        "paired_t_q_loss3_better_bh_fdr_all156",
        "wilcoxon_p_loss3_better_one_sided",
        "wilcoxon_q_loss3_better_bh_fdr_all156",
        "bootstrap_mean_advantage_ci95_low",
        "bootstrap_mean_advantage_ci95_high",
        "loss3_significantly_better_q05",
        "loss3_significantly_worse_q05",
    ]
    out[compact_cols].to_csv(compact_path, index=False)
    metric_summary.to_csv(metric_summary_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)

    attack_increase = out[out["metric"] == "attack_loss_increase"].copy()
    initial = out[out["metric"] == "initial_loss"].copy()
    final = out[out["metric"] == "final_loss"].copy()
    worst = attack_increase.sort_values("mean_advantage_comparison_minus_loss3").head(10)

    report = "\n".join(
        [
            "# Burgers Per-Dataset Loss3 Significance, 20260614",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "This report answers the per-dataset version of the significance question. It uses the strict latest same-manifest P2Q2 attack NPZ files and runs paired tests inside each dataset row.",
            "",
            "## What Was Tested",
            "",
            "- Paired unit: one attack-manifest sample inside one dataset.",
            "- Dataset rows: 52.",
            "- Metrics per dataset: `initial_loss`, `final_loss`, and `attack_loss_increase`.",
            "- Total tests: 156 rows.",
            "- Comparison: `loss3` versus the best non-loss3 model for that same dataset and metric. When `loss3` is best, that comparison model is the runner-up.",
            "- Tests: one-sided paired t-test, one-sided Wilcoxon signed-rank test, BH/FDR q-values over all 156 rows, and 20000 paired bootstrap resamples for the mean advantage.",
            "",
            "Positive advantage means the comparison model has larger loss than `loss3`, so `loss3` is better.",
            "",
            "## Metric Summary",
            "",
            md_table(metric_summary, list(metric_summary.columns), max_rows=10),
            "",
            "## Split Summary",
            "",
            md_table(split_summary, list(split_summary.columns), max_rows=20),
            "",
            "## Attack Loss Increase Rows With Smallest Loss3 Advantage",
            "",
            md_table(
                worst,
                [
                    "dataset_order",
                    "split",
                    "dataset_id",
                    "n_pairs",
                    "best_model",
                    "second_model",
                    "comparison_model",
                    "loss3_mean",
                    "comparison_mean",
                    "mean_advantage_comparison_minus_loss3",
                    "paired_t_q_loss3_better_bh_fdr_all156",
                    "bootstrap_mean_advantage_ci95_low",
                    "bootstrap_mean_advantage_ci95_high",
                    "loss3_significantly_better_q05",
                ],
                max_rows=10,
            ),
            "",
            "## Interpretation",
            "",
            f"- Attack loss increase: `loss3` is rank 1 on {int((attack_increase['loss3_rank'] == 1).sum())}/52 dataset rows and significantly better after all-156 BH/FDR plus bootstrap on {int(attack_increase['loss3_significantly_better_q05'].sum())}/52 rows.",
            f"- Attack final loss: `loss3` is rank 1 on {int((final['loss3_rank'] == 1).sum())}/52 dataset rows and significantly better on {int(final['loss3_significantly_better_q05'].sum())}/52 rows.",
            f"- Initial clean loss on the attack manifest: `loss3` is rank 1 on {int((initial['loss3_rank'] == 1).sum())}/52 dataset rows and significantly better on {int(initial['loss3_significantly_better_q05'].sum())}/52 rows. This is clean MSE on the attack manifest, not the separate full clean RMSE/Relative L2 table.",
            "",
            "## Output CSVs",
            "",
            f"- `{out_path.relative_to(REPO)}`",
            f"- `{compact_path.relative_to(REPO)}`",
            f"- `{metric_summary_path.relative_to(REPO)}`",
            f"- `{split_summary_path.relative_to(REPO)}`",
        ]
    )
    OUT_REPORT.write_text(report + "\n")
    DOC.write_text(report + "\n")

    print(
        json.dumps(
            {
                "rows": int(len(out)),
                "datasets": int(out["dataset_id"].nunique()),
                "metrics": list(out["metric"].unique()),
                "attack_increase_loss3_rank1": int((attack_increase["loss3_rank"] == 1).sum()),
                "attack_increase_loss3_significant": int(attack_increase["loss3_significantly_better_q05"].sum()),
                "report": str(OUT_REPORT.relative_to(REPO)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
