#!/usr/bin/env python3
"""Audit Burgers solver7860/clean8000 metric tables for protocol consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATE = "20260614"
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
DATA_ROOT = AUDIT_ROOT / "data"
RANKED = DATA_ROOT / "ranked_metric_tables_20260614"
OUT_DATA = DATA_ROOT / "integrity_audit_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_metric_integrity_audit_20260614.md"
DOC_REPORT = REPO / "docs/burgers_metric_integrity_audit_20260614.md"

MODELS = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]


def finite_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def almost_equal(a: float, b: float, tol: float = 5e-12) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def classify_valid_models(counts: pd.Series) -> str:
    valid = {model for model, count in counts.items() if int(count) > 0}
    if valid == set(MODELS):
        return "six_model_common"
    if valid == set(MODELS[:4]):
        return "old4_only"
    if valid == set(MODELS[4:]):
        return "random_only"
    return "partial_mixed"


def validate_model_summary_means(model_summary: pd.DataFrame, long_tables: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, row in model_summary.iterrows():
        key = (row["family"], row["scope"])
        long = long_tables.get(key)
        if long is None:
            continue
        metric = row["metric"]
        model = row["model"]
        source = long[(long["metric"] == metric) & (long["model"] == model)]
        recomputed = finite_mean(source["value"]) if "value" in source else float("nan")
        recorded = float(row["mean"]) if pd.notna(row["mean"]) else float("nan")
        rows.append(
            {
                "family": row["family"],
                "scope": row["scope"],
                "metric": metric,
                "model": model,
                "recorded_mean": recorded,
                "recomputed_mean": recomputed,
                "abs_diff": abs(recorded - recomputed) if pd.notna(recorded) and pd.notna(recomputed) else np.nan,
                "ok": almost_equal(recorded, recomputed),
                "source_rows": int(source["value"].notna().sum()) if "value" in source else 0,
            }
        )
    return pd.DataFrame(rows)


def validate_best_summary(best: pd.DataFrame, model_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index_cols = ["family", "scope", "metric"]
    for _, brow in best.iterrows():
        group = model_summary[
            (model_summary["family"] == brow["family"])
            & (model_summary["scope"] == brow["scope"])
            & (model_summary["metric"] == brow["metric"])
        ].copy()
        group = group[pd.to_numeric(group["mean"], errors="coerce").notna()]
        if group.empty:
            continue
        direction = brow["direction"]
        if direction == "higher":
            computed_row = group.loc[group["mean"].astype(float).idxmax()]
        else:
            computed_row = group.loc[group["mean"].astype(float).idxmin()]
        rows.append(
            {
                **{col: brow[col] for col in index_cols},
                "direction": direction,
                "recorded_best_model": brow["best_model"],
                "computed_best_model": computed_row["model"],
                "recorded_best_mean": float(brow["mean"]),
                "computed_best_mean": float(computed_row["mean"]),
                "ok_model": brow["best_model"] == computed_row["model"],
                "ok_mean": almost_equal(float(brow["mean"]), float(computed_row["mean"])),
            }
        )
    return pd.DataFrame(rows)


def build_robustness_comparability(robust_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in robust_long.groupby("metric"):
        counts = group.groupby("model")["value"].apply(lambda s: s.notna().sum()).reindex(MODELS, fill_value=0)
        rows.append(
            {
                "metric": metric,
                "comparability_class": classify_valid_models(counts),
                **{f"n_{model}": int(counts[model]) for model in MODELS},
                "valid_models": ",".join(model for model in MODELS if int(counts[model]) > 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["comparability_class", "metric"])


def audit_attack52_sources() -> pd.DataFrame:
    selected = pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_selected_worktime_long.csv")
    recovered = pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_recovered_full_long.csv")
    strict_path = DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_long.csv"
    strict = pd.read_csv(strict_path) if strict_path.exists() else pd.DataFrame()
    rows = []
    table_sources = [
        ("selected_worktime_original", selected),
        ("recovered_full_historical_reference", recovered),
    ]
    if not strict.empty:
        table_sources.append(("strict_latest_widevis_full52_ranked_source", strict))
    for name, df in table_sources:
        for (source, model), group in df.groupby(["source", "model"], dropna=False):
            rows.append(
                {
                    "table": name,
                    "source": source,
                    "model": model,
                    "rows": int(len(group)),
                    "dataset_count": int(group["dataset_index"].nunique()) if "dataset_index" in group else np.nan,
                    "sample_count_sum": int(pd.to_numeric(group.get("sample_count", pd.Series(dtype=float)), errors="coerce").sum()),
                    "attack_increase_mean": finite_mean(group["attack_loss_increase_mean"])
                    if "attack_loss_increase_mean" in group
                    else np.nan,
                    "final_loss_mean": finite_mean(group["final_loss_mean"]) if "final_loss_mean" in group else np.nan,
                }
            )
    return pd.DataFrame(rows)


def audit_dense_trace() -> pd.DataFrame:
    trace_root = REPO / "forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614"
    rows = []
    for csv_path in sorted(trace_root.glob("group*/attack_loss_curves_all_six_models.csv")):
        df = pd.read_csv(csv_path)
        first = (
            df[df["step"] == df["step"].min()]
            .rename(columns={"loss_mse": "initial_loss"})
            [["model", "sample_id", "initial_loss"]]
        )
        last = (
            df[df["step"] == df["step"].max()]
            .rename(columns={"loss_mse": "final_loss"})
            [["model", "sample_id", "final_loss"]]
        )
        merged = first.merge(last, on=["model", "sample_id"], how="inner")
        merged["attack_increase"] = merged["final_loss"] - merged["initial_loss"]
        for model, group in merged.groupby("model"):
            rows.append(
                {
                    "group": csv_path.parent.name,
                    "model": model,
                    "n_rows": int(len(group)),
                    "initial_loss_mean": finite_mean(group["initial_loss"]),
                    "final_loss_mean": finite_mean(group["final_loss"]),
                    "attack_increase_mean": finite_mean(group["attack_increase"]),
                }
            )
    return pd.DataFrame(rows)


def write_markdown(
    checks: dict[str, object],
    attack_sources: pd.DataFrame,
    comparability: pd.DataFrame,
    dense: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        cols = [str(col) for col in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in df.columns:
                value = row[col]
                if isinstance(value, float):
                    if pd.isna(value):
                        vals.append("")
                    elif abs(value) >= 100:
                        vals.append(f"{value:.3f}")
                    elif abs(value) >= 1:
                        vals.append(f"{value:.6g}")
                    elif abs(value) >= 1e-3:
                        vals.append(f"{value:.6f}")
                    else:
                        vals.append(f"{value:.3e}")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    class_counts = comparability["comparability_class"].value_counts().to_dict()
    dense_pooled = dense.groupby("model")[["initial_loss_mean", "final_loss_mean", "attack_increase_mean"]].mean()
    dense_pooled = dense_pooled.reindex(MODELS)
    attack_lines = md_table(attack_sources)
    class_lines = md_table(comparability.groupby("comparability_class").size().reset_index(name="metric_count"))
    dense_lines = md_table(dense_pooled.reset_index().rename(columns={"index": "model"}))
    checks_lines = pd.DataFrame(
        [{"check": key, "value": json.dumps(value, sort_keys=True)} for key, value in checks.items()]
    )
    checks_lines = md_table(checks_lines)

    text = f"""# Burgers Metric Integrity Audit, {DATE}

This audit rechecks the generated Burgers solver7860/clean8000 metric tables
without rerunning training, attack, Jacobian, SVD, or plotting jobs.

## Check Summary

{checks_lines}

## Attack-52 Source Audit

The selected-worktime attack table has only baseline and the two random models.
The recovered full table is a mixed source historical/current table:
old-four rows come from the old full-52 20-step run, while random rows come from
the current solver7860/clean8000 random full suite. The current ranked attack
table now uses `attack_52dataset_six_models_strict_latest_widevis_long.csv`
when that strict table is available.

{attack_lines}

## Robustness Metric Comparability

The 54 robustness metrics are not all six-model-common metrics.

{class_lines}

Interpretation:

- `six_model_common`: all six models have at least one valid value, although
  random attack trace fields may have 24 rather than 25 valid samples.
- `old4_only`: only baseline/loss1/loss2/loss3 have values.
- `random_only`: only random_clean_y/random_solver_y have values.

## Dense Trace Pooled Means

These are the latest dense visual trace means, pooled by group means over
group00 through group05.

{dense_lines}

## Main Corrections

- Use the strict latest full-52 widevis attack table for current attack claims:
  `loss3 e1000` has lower attack increase than `random_solver_y e7860`
  on the same 52 dataset rows.
- The recovered attack-52 ranked rows are historical mixed evidence; they remain
  useful only with that caveat.
- Do not interpret all 54 robustness metrics as equal model-quality rankings.
  Split them by comparability class first.
- The previous random top50/top100 SVD export gap is now resolved by the
  20260614 supplement derived from stored random-model Jacobian matrices.
- The random-model affine/local-gain fields are now recorded as a supplement
  derived from existing checkpoints and stored Jacobians; attack-delta cosine
  has n=24 per random model because sample_id=4 is outside the saved attack
  manifest.
- For latest direct evidence, use the strict full-52 attack table, the
  25-sample robustness/SVD table, the dense visual trace set for `loss3 e1000`,
  and the clean 52-dataset table for clean generalization.
"""
    OUT_REPORT.write_text(text, encoding="utf-8")
    DOC_REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    clean_long = pd.read_csv(RANKED / "clean_52dataset_metric_long_ranked.csv")
    attack_long = pd.read_csv(RANKED / "attack_52dataset_metric_long_ranked.csv")
    robust_long = pd.read_csv(RANKED / "robustness_25sample_metric_long_ranked.csv")
    svd_topk = pd.read_csv(RANKED / "svd_error_topk_ranked_long.csv")
    svd_rank = pd.read_csv(RANKED / "svd_error_top20_ranked_long.csv")
    svd100_supp = pd.read_csv(RANKED / "svd_error_top100_supplement_ranked_long.csv")
    svd100_topk_supp = pd.read_csv(RANKED / "svd_error_topk_top100_supplement_ranked_long.csv")
    subspace100_supp = pd.read_csv(RANKED / "model_solver_subspace_top100_supplement_ranked_long.csv")
    affine_supp = pd.read_csv(RANKED / "random_affine_direction_supplement_metric_long_ranked.csv")
    model_summary = pd.read_csv(RANKED / "metric_model_summary_ranked.csv")
    best = pd.read_csv(RANKED / "metric_best_summary_ranked.csv")

    long_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for scope, group in clean_long.groupby(
        clean_long["split"].map(
            {"train": "clean_train_1dataset", "test": "clean_test_1dataset", "generalization": "clean_generalization_50dataset"}
        ).fillna("")
    ):
        if scope:
            long_tables[("clean_generalization", scope)] = group
    long_tables[("clean_generalization", "clean_all_52dataset")] = clean_long

    for scope, group in attack_long.groupby(
        attack_long["split"].map(
            {"train": "attack_train_1dataset", "test": "attack_test_1dataset", "generalization": "attack_generalization_50dataset"}
        ).fillna("")
    ):
        if scope:
            long_tables[("attack_robustness_52dataset", scope)] = group
    long_tables[("attack_robustness_52dataset", "attack_all_52dataset")] = attack_long

    for scope, group in robust_long.groupby(
        robust_long["source_split"].map(
            {
                "train": "robustness_train_2sample",
                "test": "robustness_test_2sample",
                "generalization": "robustness_generalization_21sample",
            }
        ).fillna("")
    ):
        if scope:
            long_tables[("robustness_svd_jacobian_25sample", scope)] = group
    long_tables[("robustness_svd_jacobian_25sample", "robustness_all_25sample")] = robust_long
    long_tables[("svd_error_spectrum", "svd_error_topk_25sample")] = svd_topk
    svd_rank_named = svd_rank.copy()
    svd_rank_named["metric"] = svd_rank_named["singular_rank"].astype(int).map(
        lambda singular_rank: f"error_singular_value_rank{singular_rank:02d}"
    )
    long_tables[("svd_error_spectrum", "svd_error_rank_by_rank_25sample")] = svd_rank_named
    long_tables[("svd_error_spectrum", "svd_error_top20_all_values_25sample")] = svd_rank.assign(
        metric="error_singular_value_top20_all",
        value=svd_rank["value"],
    )
    long_tables[("svd_error_spectrum_top100_supplement", "svd_error_top100_all_values_25sample")] = svd100_supp.assign(
        metric="error_singular_value_top100_all",
        value=svd100_supp["value"],
    )
    long_tables[("svd_error_spectrum_top100_supplement", "svd_error_topk_25sample")] = svd100_topk_supp
    long_tables[("model_solver_subspace_top100_supplement", "model_solver_subspace_25sample")] = subspace100_supp
    long_tables[("random_affine_direction_supplement", "random_affine_25sample")] = affine_supp

    mean_checks = validate_model_summary_means(model_summary, long_tables)
    best_checks = validate_best_summary(best, model_summary)
    comparability = build_robustness_comparability(robust_long)
    attack_sources = audit_attack52_sources()
    dense = audit_dense_trace()

    mean_checks.to_csv(OUT_DATA / "model_summary_mean_recompute_checks.csv", index=False)
    best_checks.to_csv(OUT_DATA / "best_summary_recompute_checks.csv", index=False)
    comparability.to_csv(OUT_DATA / "robustness_metric_comparability_audit.csv", index=False)
    attack_sources.to_csv(OUT_DATA / "attack52_protocol_source_audit.csv", index=False)
    dense.to_csv(OUT_DATA / "dense_trace_group_metric_audit.csv", index=False)

    checks = {
        "mean_recompute_rows": int(len(mean_checks)),
        "mean_recompute_failures": int((~mean_checks["ok"]).sum()),
        "best_recompute_rows": int(len(best_checks)),
        "best_model_failures": int((~best_checks["ok_model"]).sum()),
        "best_mean_failures": int((~best_checks["ok_mean"]).sum()),
        "robustness_metric_class_counts": comparability["comparability_class"].value_counts().to_dict(),
        "selected_attack_models": sorted(
            pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_selected_worktime_long.csv")["model"].unique().tolist()
        ),
        "recovered_attack_models": sorted(
            pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_recovered_full_long.csv")["model"].unique().tolist()
        ),
        "strict_latest_attack_models": sorted(
            pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_long.csv")["model"].unique().tolist()
        )
        if (DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_long.csv").exists()
        else [],
    }
    (OUT_DATA / "integrity_audit_summary.json").write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(checks, attack_sources, comparability, dense)
    print(json.dumps(checks, indent=2, sort_keys=True))
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
