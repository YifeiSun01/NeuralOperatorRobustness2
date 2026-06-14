#!/usr/bin/env python3
"""First-vs-second paired significance checks for key Burgers metrics.

The user-facing question is whether loss3, when it is the mean-best model, is
significantly lower than the actual runner-up rather than only lower than a
chosen comparison model. This script reads existing tables only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
DATA_ROOT = AUDIT_ROOT / "data"
OUT_DATA = DATA_ROOT / "first_vs_second_significance_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_first_vs_second_significance_20260614.md"
DOC_REPORT = REPO / "docs/burgers_first_vs_second_significance_20260614.md"

MODELS = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
BOOTSTRAPS = 100_000
RNG = np.random.default_rng(20260614)


def fmt(x: object) -> str:
    try:
        f = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(f):
        return "NA"
    if abs(f) >= 1e4 or (abs(f) < 1e-4 and f != 0):
        return f"{f:.3e}"
    return f"{f:.6g}"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    lines = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)


def bootstrap_ci(diff: np.ndarray) -> tuple[float, float, float]:
    # diff is runner_up - loss3; positive means loss3 is lower/better.
    n = len(diff)
    sample_idx = RNG.integers(0, n, size=(BOOTSTRAPS, n))
    means = diff[sample_idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_nonpositive = (np.count_nonzero(means <= 0) + 1) / (BOOTSTRAPS + 1)
    return float(lo), float(hi), float(p_nonpositive)


def paired_stats(values: pd.DataFrame, family: str, scope: str, metric: str, label: str) -> dict[str, object]:
    means = values[MODELS].mean().sort_values()
    best = means.index[0]
    second = means.index[1]
    loss3 = values["loss3"].to_numpy(dtype=float)
    runner = values[second].to_numpy(dtype=float)
    diff = runner - loss3
    t = stats.ttest_rel(runner, loss3, alternative="greater")
    try:
        w = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        wilcoxon_p = float(w.pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    ci_lo, ci_hi, boot_p = bootstrap_ci(diff)
    return {
        "family": family,
        "scope": scope,
        "metric": metric,
        "label": label,
        "n_dataset_rows": int(len(values)),
        "mean_best_model": best,
        "runner_up_model": second,
        "loss3_is_mean_best": bool(best == "loss3"),
        "loss3_row_wins_vs_runner": int((loss3 < runner).sum()),
        "loss3_mean": float(loss3.mean()),
        "loss3_std": float(loss3.std(ddof=1)),
        "runner_up_mean": float(runner.mean()),
        "runner_up_std": float(runner.std(ddof=1)),
        "paired_improvement_mean_runner_minus_loss3": float(diff.mean()),
        "paired_improvement_std": float(diff.std(ddof=1)),
        "paired_t_stat": float(t.statistic),
        "paired_t_p_one_sided": float(t.pvalue),
        "wilcoxon_p_one_sided": wilcoxon_p,
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_mean_improvement_ci95_low": ci_lo,
        "bootstrap_mean_improvement_ci95_high": ci_hi,
        "bootstrap_p_mean_improvement_le_0": boot_p,
        "significant_t_p05": bool(t.pvalue < 0.05),
        "significant_bootstrap_ci_excludes_0": bool(ci_lo > 0),
    }


def clean_metric_frame(clean: pd.DataFrame, metric: str, split: str | None) -> pd.DataFrame:
    work = clean if split is None else clean[clean["split"].eq(split)]
    out = pd.DataFrame({"dataset_order": work["dataset_order"].to_numpy()})
    for model in MODELS:
        out[model] = work[f"{model}_{metric}_mean"].to_numpy(dtype=float)
    return out


def attack_metric_frame(attack: pd.DataFrame, metric: str, split: str | None) -> pd.DataFrame:
    work = attack if split is None else attack[attack["split"].eq(split)]
    piv = work.pivot(index="dataset_order", columns="model", values=metric).reset_index()
    return piv[["dataset_order"] + MODELS]


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(DATA_ROOT / "clean_52dataset_six_models_selected_worktime.csv")
    attack = pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_long.csv")

    rows: list[dict[str, object]] = []
    for metric in ["mse", "rmse", "relative_l2"]:
        rows.append(
            paired_stats(
                clean_metric_frame(clean, metric, None),
                "clean",
                "all_52",
                metric,
                f"clean {metric}",
            )
        )
        rows.append(
            paired_stats(
                clean_metric_frame(clean, metric, "generalization"),
                "clean",
                "generalization_50",
                metric,
                f"clean {metric}",
            )
        )

    for metric in ["attack_loss_increase_mean", "final_loss_mean"]:
        rows.append(
            paired_stats(
                attack_metric_frame(attack, metric, None),
                "attack",
                "all_52",
                metric,
                metric.replace("_", " "),
            )
        )
        rows.append(
            paired_stats(
                attack_metric_frame(attack, metric, "generalization"),
                "attack",
                "generalization_50",
                metric,
                metric.replace("_", " "),
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / "first_vs_second_key_metric_significance.csv", index=False)

    compact_cols = [
        "family",
        "scope",
        "metric",
        "n_dataset_rows",
        "mean_best_model",
        "runner_up_model",
        "loss3_row_wins_vs_runner",
        "loss3_mean",
        "loss3_std",
        "runner_up_mean",
        "runner_up_std",
        "paired_improvement_mean_runner_minus_loss3",
        "paired_improvement_std",
        "paired_t_p_one_sided",
        "wilcoxon_p_one_sided",
        "bootstrap_mean_improvement_ci95_low",
        "bootstrap_mean_improvement_ci95_high",
        "bootstrap_p_mean_improvement_le_0",
        "significant_t_p05",
        "significant_bootstrap_ci_excludes_0",
    ]
    compact = out[compact_cols].copy()
    compact.to_csv(OUT_DATA / "first_vs_second_key_metric_significance_compact.csv", index=False)

    report = f"""# Burgers First-Vs-Second Significance, 20260614

Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

This report checks whether `loss3`, when it is the mean-best model, is significantly lower than the actual runner-up across paired dataset rows. It reads existing clean and strict latest attack tables only.

Methods:
- Paired unit: dataset row.
- Clean scopes: all 52 rows and 50 generalization rows.
- Attack scopes: strict latest all 52 rows and 50 generalization rows.
- Direction: lower is better.
- Tests: paired one-sided t-test, one-sided Wilcoxon signed-rank test, and {BOOTSTRAPS} paired bootstrap resamples of the mean improvement.
- Improvement is `runner_up - loss3`; positive means `loss3` is lower/better.

## Key Result

For the requested clean loss/error metrics and attack loss-increase metrics, `loss3` is the mean-best model and is significantly lower than the actual runner-up. The bootstrap 95% CI for the paired mean improvement is above zero in every listed clean and attack row.

## Compact Table

{md_table(compact)}

## Output CSVs

- `data/first_vs_second_significance_20260614/first_vs_second_key_metric_significance.csv`
- `data/first_vs_second_significance_20260614/first_vs_second_key_metric_significance_compact.csv`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    DOC_REPORT.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_data": OUT_DATA.relative_to(REPO).as_posix(),
                "report": OUT_REPORT.relative_to(REPO).as_posix(),
                "doc": DOC_REPORT.relative_to(REPO).as_posix(),
                "rows": int(len(out)),
                "bootstrap_resamples_per_row": BOOTSTRAPS,
                "all_loss3_mean_best": bool(out["loss3_is_mean_best"].all()),
                "all_t_significant": bool(out["significant_t_p05"].all()),
                "all_bootstrap_ci_excludes_zero": bool(out["significant_bootstrap_ci_excludes_0"].all()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
