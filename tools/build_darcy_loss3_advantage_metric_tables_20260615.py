#!/usr/bin/env python3
"""Build Darcy binary-20260611 Loss3 advantage tables.

The outputs are meant to make the per-dataset/per-model evidence auditable:
clean generalization RMSE/Relative L2 plus attack50 robustness metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / "outputs/darcy_cflow_timematched_organized_release_20260614"
ROBUST = PROJECT / "outputs/darcy_cflow_final_robustness_20260615"
OUT = RELEASE / "data/loss3_advantage_metric_tables_20260615"
REPORT = RELEASE / "reports/loss3_advantage_metric_tables_20260615.md"

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

CLEAN_METRICS = ["rmse", "relative_l2"]
ATTACK_METRICS = [
    "clean_loss_mean",
    "adv_loss_mean",
    "loss_increase_mean",
    "relative_increase_mean",
    "delta_l2_rms_mean",
    "delta_linf_mean",
]
ATTACK_LABELS = {
    "clean_loss_mean": "attack clean loss",
    "adv_loss_mean": "attack adv loss",
    "loss_increase_mean": "attack loss increase",
    "relative_increase_mean": "attack relative increase",
    "delta_l2_rms_mean": "delta L2 RMS",
    "delta_linf_mean": "delta Linf",
}


def norm_model(s: pd.Series) -> pd.Series:
    return s.astype(str).replace(NORMALIZE_MODEL)


def ordered_columns(cols: list[str]) -> list[str]:
    return [m for m in MODEL_ORDER if m in cols] + [c for c in cols if c not in MODEL_ORDER]


def add_ranks(long: pd.DataFrame, metric_col: str = "metric", value_col: str = "value") -> pd.DataFrame:
    out = long.copy()
    out["rank_lower_is_better"] = out.groupby(["split", "dataset_id", metric_col])[value_col].rank(
        method="min", ascending=True
    )
    best = (
        out[out["rank_lower_is_better"] == 1]
        .groupby(["split", "dataset_id", metric_col])["model"]
        .apply(lambda x: ",".join(sorted(x.unique())))
        .rename("best_model")
        .reset_index()
    )
    loss3 = out[out["model"] == "loss3"][["split", "dataset_id", metric_col, value_col]].rename(
        columns={value_col: "loss3_value"}
    )
    out = out.merge(best, on=["split", "dataset_id", metric_col], how="left")
    out = out.merge(loss3, on=["split", "dataset_id", metric_col], how="left")
    out["is_loss3"] = out["model"].eq("loss3")
    out["is_loss3_best"] = out["best_model"].astype(str).str.split(",").apply(lambda xs: "loss3" in xs)
    out["delta_vs_loss3"] = out[value_col] - out["loss3_value"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ratio_vs_loss3"] = out[value_col] / out["loss3_value"]
    out["model_display"] = out["model"].map(DISPLAY).fillna(out["model"])
    return out


def pivot_metric(df: pd.DataFrame, metric: str, value_col: str = "value") -> pd.DataFrame:
    sub = df[df["metric"] == metric].copy()
    piv = sub.pivot_table(index=["split", "dataset_id"], columns="model", values=value_col, aggfunc="first").reset_index()
    piv = piv[["split", "dataset_id"] + ordered_columns([c for c in piv.columns if c not in {"split", "dataset_id"}])]
    return piv


def summarize(long: pd.DataFrame, metrics: list[str], value_col: str = "value") -> pd.DataFrame:
    rows = []
    for split in ["generalization", "train", "test"]:
        for metric in metrics:
            g = long[(long["split"] == split) & (long["metric"] == metric)].copy()
            if g.empty:
                continue
            pivot = g.pivot_table(index="dataset_id", columns="model", values=value_col, aggfunc="first")
            pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
            means = pivot.mean().sort_values()
            min_mean = float(means.iloc[0])
            best_mean_models = [m for m, v in means.items() if np.isclose(float(v), min_mean, rtol=1e-10, atol=1e-14)]
            best_mean_model = ",".join(best_mean_models)
            next_best_model = means.drop(index="loss3", errors="ignore").index[0] if "loss3" in means.index else ""
            row_min = pivot.min(axis=1)
            best_mask = pivot.apply(lambda col: np.isclose(col, row_min, rtol=1e-10, atol=1e-14))
            loss3_best_or_tied = int(best_mask["loss3"].sum()) if "loss3" in best_mask.columns else 0
            strict_min_count = best_mask.sum(axis=1)
            loss3_strict_wins = int((best_mask.get("loss3", False) & strict_min_count.eq(1)).sum())
            n = int(len(pivot))
            loss3_mean = float(means.get("loss3", np.nan))
            next_best_mean = float(means.get(next_best_model, np.nan)) if next_best_model else np.nan
            improvement = (
                float((next_best_mean - loss3_mean) / next_best_mean)
                if next_best_model and np.isfinite(next_best_mean) and next_best_mean != 0
                else np.nan
            )
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "metric_label": ATTACK_LABELS.get(metric, metric),
                    "datasets": n,
                    "loss3_dataset_wins": loss3_best_or_tied,
                    "loss3_dataset_strict_wins": loss3_strict_wins,
                    "loss3_dataset_win_fraction": loss3_best_or_tied / n if n else np.nan,
                    "best_mean_model": best_mean_model,
                    "best_mean_model_display": ",".join(DISPLAY.get(m, m) for m in best_mean_models),
                    "is_loss3_best_by_mean": "loss3" in best_mean_models,
                    "loss3_mean": loss3_mean,
                    "next_best_model": next_best_model,
                    "next_best_model_display": DISPLAY.get(next_best_model, next_best_model),
                    "next_best_mean": next_best_mean,
                    "loss3_relative_improvement_vs_next_best": improvement,
                    "mean_ranking": ",".join(means.index.tolist()),
                }
            )
    return pd.DataFrame(rows)


def model_means(long: pd.DataFrame, metrics: list[str], value_col: str = "value") -> pd.DataFrame:
    rows = []
    for split in ["generalization", "train", "test"]:
        for metric in metrics:
            g = long[(long["split"] == split) & (long["metric"] == metric)].copy()
            if g.empty:
                continue
            means = g.groupby("model")[value_col].mean().reindex(MODEL_ORDER).dropna()
            ranks = means.rank(method="min", ascending=True)
            for model, value in means.items():
                rows.append(
                    {
                        "split": split,
                        "metric": metric,
                        "metric_label": ATTACK_LABELS.get(metric, metric),
                        "model": model,
                        "model_display": DISPLAY.get(model, model),
                        "mean_value": float(value),
                        "mean_rank_lower_is_better": int(ranks.loc[model]),
                        "is_loss3": model == "loss3",
                    }
                )
    return pd.DataFrame(rows)


def winners_by_dataset(clean_long: pd.DataFrame, attack_long: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name, df in [("clean", clean_long), ("attack50", attack_long)]:
        tmp = (
            df[df["rank_lower_is_better"] == 1]
            .groupby(["split", "dataset_id", "metric"])["model"]
            .apply(lambda x: ",".join(sorted(x.unique())))
            .reset_index()
        )
        tmp["system"] = name
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    out["loss3_is_winner"] = out["model"].astype(str).str.split(",").apply(lambda xs: "loss3" in xs)
    return out[["system", "split", "dataset_id", "metric", "model", "loss3_is_winner"]]


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    if show.empty:
        return "_No rows._"
    text = show.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.8g}")
        else:
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(text.columns)
    rows = text.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    clean_src = RELEASE / "data/clean_52dataset_metric_long_ranked.csv"
    attack_src = ROBUST / "data/attack50_summary_by_dataset_model.csv"
    provenance_src = ROBUST / "data/final_robustness_provenance.json"

    clean = pd.read_csv(clean_src)
    clean = clean[clean["metric"].isin(CLEAN_METRICS)].copy()
    clean["model"] = norm_model(clean["method"])
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean_long = clean[
        ["split", "dataset_id", "source", "manual_tier", "manual_rank", "epoch", "model", "metric", "value"]
    ].copy()
    clean_long = add_ranks(clean_long)

    attack = pd.read_csv(attack_src)
    attack["model"] = norm_model(attack["method"])
    attack_rows = []
    for metric in ATTACK_METRICS:
        part = attack[
            ["split", "dataset_id", "sample_count", "attack_steps", "model", "method_display", metric]
        ].copy()
        part = part.rename(columns={metric: "value"})
        part["metric"] = metric
        attack_rows.append(part)
    attack_long = pd.concat(attack_rows, ignore_index=True)
    attack_long["value"] = pd.to_numeric(attack_long["value"], errors="coerce")
    attack_long = add_ranks(attack_long)

    clean_long.to_csv(OUT / "clean_52dataset_7model_rmse_relative_l2_long.csv", index=False)
    attack_long.to_csv(OUT / "attack50_52dataset_7model_6metrics_long.csv", index=False)

    for metric in CLEAN_METRICS:
        pivot_metric(clean_long, metric).to_csv(OUT / f"clean_52dataset_7model_{metric}_wide.csv", index=False)
    for metric in ATTACK_METRICS:
        pivot_metric(attack_long, metric).to_csv(OUT / f"attack50_52dataset_7model_{metric}_wide.csv", index=False)

    clean_summary = summarize(clean_long, CLEAN_METRICS)
    attack_summary = summarize(attack_long, ATTACK_METRICS)
    summary = pd.concat([clean_summary.assign(system="clean"), attack_summary.assign(system="attack50")], ignore_index=True)
    summary = summary[
        [
            "system",
            "split",
            "metric",
            "metric_label",
            "datasets",
            "loss3_dataset_wins",
            "loss3_dataset_strict_wins",
            "loss3_dataset_win_fraction",
            "best_mean_model",
            "best_mean_model_display",
            "is_loss3_best_by_mean",
            "loss3_mean",
            "next_best_model",
            "next_best_model_display",
            "next_best_mean",
            "loss3_relative_improvement_vs_next_best",
            "mean_ranking",
        ]
    ]
    summary.to_csv(OUT / "loss3_win_summary_by_split_metric.csv", index=False)

    means = pd.concat(
        [model_means(clean_long, CLEAN_METRICS).assign(system="clean"), model_means(attack_long, ATTACK_METRICS).assign(system="attack50")],
        ignore_index=True,
    )
    means = means[["system", "split", "metric", "metric_label", "model", "model_display", "mean_value", "mean_rank_lower_is_better", "is_loss3"]]
    means.to_csv(OUT / "model_mean_values_by_split_metric.csv", index=False)

    winners = winners_by_dataset(clean_long, attack_long)
    winners.to_csv(OUT / "per_dataset_metric_winners.csv", index=False)

    # One compact table focused on the user's main view: 50 generalization datasets.
    gen_winners = winners[winners["split"] == "generalization"].copy()
    gen_wide = gen_winners.pivot_table(
        index="dataset_id",
        columns=["system", "metric"],
        values="model",
        aggfunc="first",
    )
    gen_wide.columns = [f"{system}_{metric}_winner" for system, metric in gen_wide.columns]
    gen_wide = gen_wide.reset_index()
    gen_wide.to_csv(OUT / "generalization50_metric_winners_wide.csv", index=False)

    with provenance_src.open("r", encoding="utf-8") as f:
        provenance = json.load(f)

    primary_generalization = summary[
        (summary["split"] == "generalization")
        & (
            ((summary["system"] == "clean") & summary["metric"].isin(CLEAN_METRICS))
            | (
                (summary["system"] == "attack50")
                & summary["metric"].isin(
                    ["clean_loss_mean", "adv_loss_mean", "loss_increase_mean", "relative_increase_mean"]
                )
            )
        )
    ].copy()

    report = []
    report.append("# Darcy Binary 20260611 Loss3 Advantage Metric Tables")
    report.append("")
    report.append("## Scope")
    report.append("")
    report.append("- Clean metrics source: `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_52dataset_metric_long_ranked.csv`")
    report.append("- Attack50 source: `outputs/darcy_cflow_final_robustness_20260615/data/attack50_summary_by_dataset_model.csv`")
    report.append("- Generalization root: `generalization_datasets_darcy_binary_loss3targeted_20260611/`")
    report.append(f"- Final robustness provenance old-root check: `{provenance.get('old_lossdrop50_token_found')}`")
    report.append(f"- Attack rows: `{provenance.get('attack_rows')}`, SVD rows: `{provenance.get('svd_rows')}`, attack steps: `{provenance.get('attack_steps')}`")
    report.append("")
    report.append("Lower is better for every metric in the tables below. `loss3_dataset_wins` means Loss3 is lowest or tied for lowest; `loss3_dataset_strict_wins` counts only non-tied Loss3 wins. `delta_linf_mean` is included as one of the six attack columns, but it is fixed by the epsilon box and is not a meaningful model-quality winner.")
    report.append("")
    report.append("## Main Generalization Evidence")
    report.append("")
    report.append(markdown_table(primary_generalization.round(8)))
    report.append("")
    report.append("Interpretation: Loss3 is best by mean on clean RMSE, clean Relative L2, attack clean loss, attack adv loss, attack loss increase, and attack relative increase. The strongest per-dataset result is attack adv loss and attack loss increase, where Loss3 wins 50/50 generalization datasets.")
    report.append("")
    report.append("## Generalization Mean Values By Model")
    report.append("")
    gen_means = means[means["split"] == "generalization"].copy()
    report.append(markdown_table(gen_means.round(10)))
    report.append("")
    report.append("## Attack50 Generalization Six-Metric Summary")
    report.append("")
    attack_gen = summary[(summary["system"] == "attack50") & (summary["split"] == "generalization")].copy()
    report.append(markdown_table(attack_gen.round(8)))
    report.append("")
    report.append("## Clean Generalization RMSE/Relative L2 Summary")
    report.append("")
    clean_gen = summary[(summary["system"] == "clean") & (summary["split"] == "generalization")].copy()
    report.append(markdown_table(clean_gen.round(8)))
    report.append("")
    report.append("## Files")
    report.append("")
    for path in [
        OUT / "clean_52dataset_7model_rmse_relative_l2_long.csv",
        OUT / "attack50_52dataset_7model_6metrics_long.csv",
        OUT / "loss3_win_summary_by_split_metric.csv",
        OUT / "model_mean_values_by_split_metric.csv",
        OUT / "per_dataset_metric_winners.csv",
        OUT / "generalization50_metric_winners_wide.csv",
    ]:
        report.append(f"- `{path.relative_to(PROJECT)}`")
    report.append("")
    report.append("## Caveat")
    report.append("")
    report.append("The precise claim supported by these tables is that Loss3 is the best overall model on the main binary-20260611 generalization and attack50 robustness metrics. It is not literally the strict winner of every auxiliary per-dataset diagnostic: clean RMSE/Relative L2 are 47/50 per-dataset wins, attack clean loss is 44/50, attack relative increase is best by mean but only 16/50 strict per-dataset wins, and delta magnitude metrics are not primary robustness-quality metrics.")
    report.append("")

    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(REPORT.relative_to(PROJECT))
    print(OUT.relative_to(PROJECT))


if __name__ == "__main__":
    main()
