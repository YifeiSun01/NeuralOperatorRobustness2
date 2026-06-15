#!/usr/bin/env python3
"""Paired t-tests for Darcy binary-20260611 Loss3 advantage.

The statistical unit is a dataset. For each metric we compare paired values on
the same 50 generalization datasets: competitor - loss3. A positive mean
difference means Loss3 is lower/better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / "outputs/darcy_cflow_timematched_organized_release_20260614"
ROBUST = PROJECT / "outputs/darcy_cflow_final_robustness_20260615"
OUT = RELEASE / "data/loss3_ttests_20260615"
REPORT = RELEASE / "reports/loss3_paired_ttests_20260615.md"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "physics_loss", "random_clean", "random_solver"]
DISPLAY = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "Physics Loss",
    "physics_loss": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}
NORMALIZE_MODEL = {"physics": "physics_loss"}

CLEAN_METRICS = ["data_mse", "rmse", "relative_l2"]
ATTACK_METRICS = [
    "clean_loss_mean",
    "adv_loss_mean",
    "loss_increase_mean",
    "relative_increase_mean",
    "delta_l2_rms_mean",
    "delta_linf_mean",
]
LABEL = {
    "data_mse": "generalization loss / data MSE",
    "rmse": "RMSE",
    "relative_l2": "Relative L2",
    "clean_loss_mean": "attack clean loss",
    "adv_loss_mean": "attack adv loss",
    "loss_increase_mean": "attack loss increase",
    "relative_increase_mean": "attack relative increase",
    "delta_l2_rms_mean": "delta L2 RMS",
    "delta_linf_mean": "delta Linf",
}


def bh_adjust(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
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
    adj = np.clip(adj, 0.0, 1.0)
    out[idx[order]] = adj
    return out.tolist()


def paired_ttest(diff: np.ndarray) -> dict[str, float | int | str]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n < 2:
        return {"n": n, "status": "n_lt_2"}
    mean = float(np.mean(diff))
    std = float(np.std(diff, ddof=1))
    sem = std / np.sqrt(n) if std > 0 else 0.0
    df = n - 1
    if std == 0:
        if mean > 0:
            t_stat = float("inf")
            p_one = 0.0
            p_two = 0.0
        elif mean == 0:
            t_stat = 0.0
            p_one = 0.5
            p_two = 1.0
        else:
            t_stat = float("-inf")
            p_one = 1.0
            p_two = 0.0
    else:
        t_stat = float(mean / sem)
        p_one = float(stats.t.sf(t_stat, df))
        p_two = float(stats.t.sf(abs(t_stat), df) * 2.0)
    tcrit = float(stats.t.ppf(0.975, df))
    ci_low = mean - tcrit * sem
    ci_high = mean + tcrit * sem
    dz = mean / std if std > 0 else (float("inf") if mean > 0 else 0.0)
    return {
        "n": n,
        "status": "ok",
        "mean_diff_competitor_minus_loss3": mean,
        "std_diff": std,
        "sem_diff": sem,
        "t_stat": t_stat,
        "df": df,
        "p_one_sided_loss3_lower": p_one,
        "p_two_sided": p_two,
        "ci95_diff_low": ci_low,
        "ci95_diff_high": ci_high,
        "cohens_dz": dz,
    }


def load_clean() -> pd.DataFrame:
    clean = pd.read_csv(RELEASE / "data/clean_52dataset_metric_long_ranked.csv")
    clean = clean[(clean["split"] == "generalization") & clean["metric"].isin(CLEAN_METRICS)].copy()
    clean["model"] = clean["method"].astype(str).replace(NORMALIZE_MODEL)
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    return clean[["dataset_id", "model", "metric", "value"]]


def load_attack() -> pd.DataFrame:
    attack = pd.read_csv(ROBUST / "data/attack50_summary_by_dataset_model.csv")
    attack = attack[attack["split"] == "generalization"].copy()
    attack["model"] = attack["method"].astype(str).replace(NORMALIZE_MODEL)
    rows = []
    for metric in ATTACK_METRICS:
        part = attack[["dataset_id", "model", metric]].rename(columns={metric: "value"}).copy()
        part["metric"] = metric
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def run_family(long: pd.DataFrame, system: str, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    second_rows = []
    for metric in metrics:
        pivot = long[long["metric"] == metric].pivot_table(index="dataset_id", columns="model", values="value", aggfunc="first")
        pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
        if "loss3" not in pivot.columns:
            continue
        means = pivot.mean().sort_values()
        best_mean_models = means[np.isclose(means, means.iloc[0], rtol=1e-10, atol=1e-14)].index.tolist()
        second_model = means.drop(index="loss3", errors="ignore").index[0]
        for competitor in [m for m in MODEL_ORDER if m in pivot.columns and m != "loss3"]:
            diff = pivot[competitor].to_numpy(dtype=float) - pivot["loss3"].to_numpy(dtype=float)
            res = paired_ttest(diff)
            row = {
                "system": system,
                "metric": metric,
                "metric_label": LABEL.get(metric, metric),
                "comparison": f"{DISPLAY.get(competitor, competitor)} - loss3",
                "competitor": competitor,
                "competitor_display": DISPLAY.get(competitor, competitor),
                "loss3_mean": float(pivot["loss3"].mean()),
                "competitor_mean": float(pivot[competitor].mean()),
                "competitor_minus_loss3_mean": float(pivot[competitor].mean() - pivot["loss3"].mean()),
                "loss3_lower_dataset_count": int((pivot["loss3"] < pivot[competitor]).sum()),
                "competitor_lower_dataset_count": int((pivot[competitor] < pivot["loss3"]).sum()),
                "ties": int(np.isclose(pivot["loss3"], pivot[competitor], rtol=1e-12, atol=1e-18).sum()),
                "loss3_is_best_by_mean": "loss3" in best_mean_models,
                "best_mean_models": ",".join(best_mean_models),
                "second_mean_model_if_loss3_first": second_model if "loss3" in best_mean_models else "",
            }
            row.update(res)
            pair_rows.append(row)
            if competitor == second_model:
                second_rows.append(row.copy())
    pair = pd.DataFrame(pair_rows)
    pair["p_one_sided_bh_all_family"] = bh_adjust(pair["p_one_sided_loss3_lower"].tolist())
    second = pd.DataFrame(second_rows)
    second["p_one_sided_bh_second_only"] = bh_adjust(second["p_one_sided_loss3_lower"].tolist())
    return pair, second


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
    clean = load_clean()
    attack = load_attack()
    clean_pair, clean_second = run_family(clean, "clean_generalization", CLEAN_METRICS)
    attack_pair, attack_second = run_family(attack, "attack50_generalization", ATTACK_METRICS)
    pair = pd.concat([clean_pair, attack_pair], ignore_index=True)
    second = pd.concat([clean_second, attack_second], ignore_index=True)
    pair["p_one_sided_bh_all_tests"] = bh_adjust(pair["p_one_sided_loss3_lower"].tolist())
    second["p_one_sided_bh_all_second_tests"] = bh_adjust(second["p_one_sided_loss3_lower"].tolist())
    pair.to_csv(OUT / "loss3_vs_all_models_paired_ttests.csv", index=False)
    second.to_csv(OUT / "loss3_vs_second_best_paired_ttests.csv", index=False)

    main_cols = [
        "system",
        "metric_label",
        "competitor_display",
        "n",
        "loss3_mean",
        "competitor_mean",
        "mean_diff_competitor_minus_loss3",
        "ci95_diff_low",
        "ci95_diff_high",
        "t_stat",
        "df",
        "p_one_sided_loss3_lower",
        "p_one_sided_bh_all_second_tests",
        "cohens_dz",
        "loss3_lower_dataset_count",
        "competitor_lower_dataset_count",
        "loss3_is_best_by_mean",
    ]
    pair_cols = [
        "system",
        "metric_label",
        "competitor_display",
        "n",
        "loss3_mean",
        "competitor_mean",
        "mean_diff_competitor_minus_loss3",
        "t_stat",
        "p_one_sided_loss3_lower",
        "p_one_sided_bh_all_family",
        "cohens_dz",
        "loss3_lower_dataset_count",
        "competitor_lower_dataset_count",
    ]

    report = []
    report.append("# Darcy Binary 20260611 Paired T-Tests for Loss3")
    report.append("")
    report.append("Statistical unit: one generalization dataset. `n=50` for each test.")
    report.append("")
    report.append("Test definition: paired t-test on `competitor - loss3`. The one-sided alternative is that this mean difference is greater than zero, i.e. Loss3 has lower error/loss than the competitor on the same datasets.")
    report.append("")
    report.append("The p-values below are paired-dataset tests. They are not per-sample tests inside each dataset.")
    report.append("")
    report.append("## Loss3 vs Second-Best Mean Model")
    report.append("")
    report.append(md_table(second[main_cols]))
    report.append("")
    report.append("## Loss3 vs All Other Models")
    report.append("")
    report.append(md_table(pair[pair_cols]))
    report.append("")
    report.append("## Files")
    report.append("")
    report.append(f"- `{(OUT / 'loss3_vs_second_best_paired_ttests.csv').relative_to(PROJECT)}`")
    report.append(f"- `{(OUT / 'loss3_vs_all_models_paired_ttests.csv').relative_to(PROJECT)}`")
    report.append("")
    report.append("## Interpretation Notes")
    report.append("")
    report.append("- For clean data MSE, RMSE, and Relative L2, Loss3 is significantly lower than the second-best mean model.")
    report.append("- For attack50 adversarial loss and attack loss increase, Loss3 is also significantly lower than the second-best mean model.")
    report.append("- `delta_linf_mean` is fixed by the epsilon box and all models tie; it is not a meaningful robustness-quality t-test.")
    report.append("- `delta_l2_rms_mean` is a perturbation-size diagnostic, not a direct model-quality metric; Loss3 is not the lowest on that auxiliary quantity.")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(REPORT.relative_to(PROJECT))
    print((OUT / "loss3_vs_second_best_paired_ttests.csv").relative_to(PROJECT))
    print(second[main_cols].to_string(index=False))


if __name__ == "__main__":
    main()
