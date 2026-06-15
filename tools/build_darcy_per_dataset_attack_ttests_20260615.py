#!/usr/bin/env python3
"""Per-dataset paired t-tests for Darcy final attack50 sample table.

For each dataset and metric, rank the seven models by the 50-sample mean, then
test best vs second-best using paired samples from the same source indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / "outputs/darcy_cflow_timematched_organized_release_20260614"
ROBUST = PROJECT / "outputs/darcy_cflow_final_robustness_20260615"
OUT = RELEASE / "data/per_dataset_ttests_20260615"
REPORT = RELEASE / "reports/per_dataset_attack50_ttests_20260615.md"

DISPLAY = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics_loss": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}

METRICS = [
    "clean_loss",
    "clean_rmse_from_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "delta_l2_rms",
    "delta_linf",
]
LABEL = {
    "clean_loss": "clean MSE loss",
    "clean_rmse_from_loss": "clean RMSE from per-sample MSE",
    "adv_loss": "adversarial MSE loss",
    "loss_increase": "absolute loss increase",
    "relative_increase": "relative loss increase",
    "delta_l2_rms": "delta L2 RMS",
    "delta_linf": "delta Linf",
}


def bh_adjust(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=float)
    out = np.full_like(arr, np.nan)
    finite = np.isfinite(arr)
    if not finite.any():
        return out.tolist()
    idx = np.where(finite)[0]
    p = arr[idx]
    order = np.argsort(p)
    ranked = p[order]
    m = float(len(ranked))
    adj = ranked * m / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[idx[order]] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def paired_ttest(best: np.ndarray, second: np.ndarray) -> dict[str, float | int | str]:
    diff = np.asarray(second, dtype=float) - np.asarray(best, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n < 2:
        return {"n_pairs": n, "status": "n_lt_2"}
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    df = n - 1
    if std == 0:
        t_stat = float("inf") if mean > 0 else (0.0 if mean == 0 else float("-inf"))
        p_one = 0.0 if mean > 0 else (0.5 if mean == 0 else 1.0)
        p_two = 0.0 if mean != 0 else 1.0
        sem = 0.0
        ci_low = ci_high = mean
        dz = float("inf") if mean > 0 else (0.0 if mean == 0 else float("-inf"))
    else:
        sem = std / np.sqrt(n)
        t_stat = float(mean / sem)
        p_one = float(stats.t.sf(t_stat, df))
        p_two = float(stats.t.sf(abs(t_stat), df) * 2.0)
        tcrit = float(stats.t.ppf(0.975, df))
        ci_low = float(mean - tcrit * sem)
        ci_high = float(mean + tcrit * sem)
        dz = float(mean / std)
    return {
        "n_pairs": n,
        "status": "ok",
        "mean_diff_second_minus_best": mean,
        "std_diff": std,
        "sem_diff": sem,
        "ci95_diff_low": ci_low,
        "ci95_diff_high": ci_high,
        "t_stat": t_stat,
        "df": df,
        "p_one_sided_best_lower": p_one,
        "p_two_sided": p_two,
        "cohens_dz": dz,
    }


def short_dataset(dataset_id: str) -> str:
    return (
        dataset_id.replace("darcy_binary_loss3targeted_20260611_", "")
        .replace("train_screen_binary_grf_alpha2_tau3_n384", "train")
        .replace("test_screen_binary_grf_alpha2_tau3_n96", "test")
    )


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in show.astype(str).values.tolist()]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    src = ROBUST / "data/robustness_attack_52datasets_samples.csv"
    df = pd.read_csv(src)
    df["clean_rmse_from_loss"] = np.sqrt(pd.to_numeric(df["clean_loss"], errors="coerce").clip(lower=0))

    rows = []
    for (split, dataset_id, metric), g in df.melt(
        id_vars=["method", "method_display", "dataset_id", "split", "source_sample_index", "sample_ordinal"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).groupby(["split", "dataset_id", "metric"], sort=False):
        pivot = g.pivot_table(index="source_sample_index", columns="method", values="value", aggfunc="first")
        means = pivot.mean().sort_values()
        best_model = str(means.index[0])
        second_model = str(means.index[1])
        res = paired_ttest(pivot[best_model].to_numpy(), pivot[second_model].to_numpy())
        rows.append(
            {
                "split": split,
                "dataset_id": dataset_id,
                "dataset_short": short_dataset(dataset_id),
                "metric": metric,
                "metric_label": LABEL[metric],
                "best_model": best_model,
                "best_model_display": DISPLAY.get(best_model, best_model),
                "second_model": second_model,
                "second_model_display": DISPLAY.get(second_model, second_model),
                "best_mean": float(means.iloc[0]),
                "second_mean": float(means.iloc[1]),
                "loss3_rank": int(list(means.index).index("loss3") + 1) if "loss3" in means.index else -1,
                "loss3_mean": float(means.get("loss3", np.nan)),
                "ranked_models": ",".join(means.index.tolist()),
                **res,
            }
        )
    out = pd.DataFrame(rows)
    # BH correction separately for each metric over 52 datasets and over 50 generalization datasets.
    out["p_bh_by_metric_all52"] = np.nan
    out["p_bh_by_metric_generalization50"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_bh_by_metric_all52"] = bh_adjust(out.loc[idx, "p_one_sided_best_lower"].tolist())
        gen_idx = out.index[(out["metric"] == metric) & (out["split"] == "generalization")]
        out.loc[gen_idx, "p_bh_by_metric_generalization50"] = bh_adjust(
            out.loc[gen_idx, "p_one_sided_best_lower"].tolist()
        )
    out["significant_p05"] = out["p_one_sided_best_lower"] < 0.05
    out["significant_bh_all52_p05"] = out["p_bh_by_metric_all52"] < 0.05
    out["significant_bh_gen50_p05"] = out["p_bh_by_metric_generalization50"] < 0.05
    out.to_csv(OUT / "attack50_per_dataset_first_vs_second_ttests.csv", index=False)

    summary_rows = []
    for split_name, split_df in [("generalization50", out[out["split"] == "generalization"]), ("all52", out)]:
        for metric, g in split_df.groupby("metric", sort=False):
            summary_rows.append(
                {
                    "scope": split_name,
                    "metric": metric,
                    "metric_label": LABEL[metric],
                    "tests": int(len(g)),
                    "best_model_counts": "; ".join(f"{k}:{v}" for k, v in g["best_model"].value_counts().items()),
                    "loss3_best_count": int(g["best_model"].eq("loss3").sum()),
                    "p05_significant_count": int(g["significant_p05"].sum()),
                    "bh_significant_count": int(
                        g["significant_bh_gen50_p05"].sum()
                        if split_name == "generalization50"
                        else g["significant_bh_all52_p05"].sum()
                    ),
                    "median_p": float(g["p_one_sided_best_lower"].median()),
                    "max_p": float(g["p_one_sided_best_lower"].max()),
                    "non_sig_dataset_short": "; ".join(
                        g.loc[~g["significant_p05"], "dataset_short"].astype(str).tolist()[:20]
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "attack50_per_dataset_ttest_summary.csv", index=False)

    focus_metrics = ["clean_loss", "clean_rmse_from_loss", "adv_loss", "loss_increase", "relative_increase"]
    report = []
    report.append("# Darcy Attack50 Per-Dataset First-vs-Second T-Tests")
    report.append("")
    report.append("Each row is one paired t-test inside one dataset using 50 matched attack samples. The test is `second - best > 0`, so a small p-value means the first-ranked model is significantly lower than the second-ranked model for that dataset.")
    report.append("")
    report.append("Important limitation: the clean 52-dataset RMSE/Relative L2 release table is aggregate-only, so per-dataset t-tests for full clean RMSE/Relative L2 require rerunning clean evaluation with per-sample errors saved. This report uses the sample-level final attack50 table, including `clean_loss` and `sqrt(clean_loss)` for the same 50 samples per dataset.")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(md_table(summary[summary["metric"].isin(focus_metrics)]))
    report.append("")
    report.append("## Generalization50 Rows for Main Metrics")
    report.append("")
    cols = [
        "dataset_short",
        "metric_label",
        "best_model_display",
        "second_model_display",
        "best_mean",
        "second_mean",
        "mean_diff_second_minus_best",
        "t_stat",
        "p_one_sided_best_lower",
        "p_bh_by_metric_generalization50",
        "significant_p05",
        "significant_bh_gen50_p05",
        "loss3_rank",
    ]
    main = out[(out["split"] == "generalization") & (out["metric"].isin(focus_metrics))]
    report.append(md_table(main[cols]))
    report.append("")
    report.append("## Files")
    report.append("")
    report.append(f"- `{(OUT / 'attack50_per_dataset_first_vs_second_ttests.csv').relative_to(PROJECT)}`")
    report.append(f"- `{(OUT / 'attack50_per_dataset_ttest_summary.csv').relative_to(PROJECT)}`")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(REPORT.relative_to(PROJECT))
    print((OUT / "attack50_per_dataset_first_vs_second_ttests.csv").relative_to(PROJECT))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
