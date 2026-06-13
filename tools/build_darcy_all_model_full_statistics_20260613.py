#!/usr/bin/env python3
"""Build all-model Darcy statistics tables for 52 datasets and 7 models.

This report is intentionally explicit about metric coverage. Clean generalization
is available for all seven models. Existing robustness artifacts cover six models
(loss1/loss2/loss3/physics_loss/random_clean_y/random_solver_y); baseline
robustness was not computed in the prior attack/Jacobian runs and is represented
as missing rather than fabricated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs/darcy_all_model_full_statistics_20260613"
DOC_PATH = ROOT / "docs/darcy_all_model_full_statistics_20260613.md"

SIX_ROOT = ROOT / "analysis_outputs/darcy_six_model_random_source_report_20260613"
CLEAN_DATASET = SIX_ROOT / "six_model_clean_by_dataset.csv"
CLEAN_SPLIT = SIX_ROOT / "six_model_clean_by_split.csv"
ATTACK_DATASET = SIX_ROOT / "six_model_attack20_by_dataset_model.csv"
ATTACK_SPLIT = SIX_ROOT / "six_model_attack20_by_split.csv"
METRIC25_MEANS = SIX_ROOT / "six_model_metric25_model_means.csv"
METRIC25_SAMPLES = SIX_ROOT / "six_model_metric25_samples.csv"
METRIC25_CORR = SIX_ROOT / "six_model_metric25_correlations.csv"
JAC5_MEANS = SIX_ROOT / "six_model_jacobian5_model_means.csv"

MODEL_ORDER = [
    "baseline",
    "loss1",
    "loss2",
    "loss3",
    "physics_loss",
    "random_clean_y",
    "random_solver_y",
]
MODEL_RANK_ORDER = {m: i for i, m in enumerate(MODEL_ORDER)}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write(df: pd.DataFrame, name: str) -> Path:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    return path


def sort_models(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    out["_model_order"] = out[model_col].map(MODEL_RANK_ORDER).fillna(999).astype(int)
    sort_cols = ["_model_order"]
    for c in ["split", "dataset_id", "metric_system", "metric"]:
        if c in out.columns:
            sort_cols.append(c)
    out = out.sort_values(sort_cols).drop(columns=["_model_order"])
    return out


def complete_model_frame(df: pd.DataFrame, *, template_cols: list[str] | None = None) -> pd.DataFrame:
    present = set(df["model"].astype(str)) if "model" in df.columns else set()
    rows = []
    for model in MODEL_ORDER:
        if model not in present:
            row = {c: np.nan for c in (template_cols or df.columns.tolist())}
            row["model"] = model
            row["metric_available"] = False
            row["missing_reason"] = "not_computed_in_existing_artifacts"
            rows.append(row)
    out = df.copy()
    if "metric_available" not in out.columns:
        out["metric_available"] = True
    if "missing_reason" not in out.columns:
        out["missing_reason"] = ""
    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return sort_models(out)


def build_52x7_dataset_table(clean: pd.DataFrame, attack: pd.DataFrame) -> pd.DataFrame:
    clean_keep = clean.rename(
        columns={
            "relative_l2": "clean_relative_l2",
            "rmse": "clean_rmse",
            "mae": "clean_mae",
            "accuracy_score": "clean_accuracy_score",
            "num_samples_evaluated": "clean_num_samples_evaluated",
        }
    )[
        [
            "model",
            "split",
            "dataset_id",
            "clean_num_samples_evaluated",
            "clean_relative_l2",
            "clean_rmse",
            "clean_mae",
            "clean_accuracy_score",
        ]
    ].copy()

    attack_keep = attack[
        [
            "model",
            "dataset_id",
            "split",
            "sample_count",
            "attack_steps",
            "epsilon_fraction",
            "mean_clean_loss",
            "mean_attack_loss_gain",
            "mean_adv_loss",
            "mean_attack_loss_gain_relative",
            "relative_gain_from_means",
            "median_attack_loss_gain",
            "median_attack_loss_gain_relative",
        ]
    ].copy()
    attack_keep = attack_keep.rename(columns={"sample_count": "attack20_sample_count"})
    merged = clean_keep.merge(attack_keep, on=["model", "split", "dataset_id"], how="left")
    merged["attack20_available"] = merged["mean_attack_loss_gain"].notna()
    merged["attack20_missing_reason"] = np.where(
        merged["attack20_available"], "", "not_computed_in_existing_artifacts"
    )
    return sort_models(merged)


def split_summary(clean_split: pd.DataFrame, attack_split: pd.DataFrame) -> pd.DataFrame:
    clean = clean_split.rename(
        columns={
            "dataset_count": "clean_dataset_count",
            "mean_relative_l2": "clean_mean_relative_l2",
            "median_relative_l2": "clean_median_relative_l2",
            "mean_rmse": "clean_mean_rmse",
        }
    ).copy()
    atk_cols = [
        "model",
        "split",
        "trained_epochs",
        "dataset_count",
        "sample_count",
        "attack_steps",
        "epsilon_fraction",
        "mean_clean_loss",
        "mean_attack_loss_gain",
        "mean_adv_loss",
        "mean_attack_loss_gain_relative",
        "relative_gain_from_means",
        "median_attack_loss_gain",
        "median_attack_loss_gain_relative",
    ]
    atk = attack_split[atk_cols].rename(
        columns={"dataset_count": "attack20_dataset_count", "sample_count": "attack20_sample_count"}
    )
    out = clean.merge(atk, on=["model", "split"], how="outer")
    out["attack20_available"] = out["mean_attack_loss_gain"].notna()
    out["attack20_missing_reason"] = np.where(out["attack20_available"], "", "not_computed_in_existing_artifacts")
    return sort_models(out)


def metric25_summary(metric25: pd.DataFrame) -> pd.DataFrame:
    d = metric25.copy()
    d["metric_system"] = "metric25_attack_jacobian_proxy"
    d = complete_model_frame(d)
    return d


def jac5_summary(jac5: pd.DataFrame) -> pd.DataFrame:
    d = jac5.copy()
    d["metric_system"] = "jacobian5_probe"
    d = complete_model_frame(d)
    return d


def winners_from_summary(split_df: pd.DataFrame, metric25_df: pd.DataFrame, jac5_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(system: str, scope: str, metric: str, df: pd.DataFrame, lower_is_better: bool = True):
        if metric not in df.columns:
            return
        vals = df[["model", metric]].dropna().copy()
        if vals.empty:
            rows.append(
                {
                    "metric_system": system,
                    "scope": scope,
                    "metric": metric,
                    "best_model": "not_available",
                    "best_value": np.nan,
                    "model_count_available": 0,
                    "lower_is_better": lower_is_better,
                }
            )
            return
        vals = vals.sort_values(metric, ascending=lower_is_better)
        rows.append(
            {
                "metric_system": system,
                "scope": scope,
                "metric": metric,
                "best_model": vals.iloc[0]["model"],
                "best_value": vals.iloc[0][metric],
                "model_count_available": len(vals),
                "lower_is_better": lower_is_better,
                "ranking": ",".join(vals["model"].astype(str).tolist()),
            }
        )

    for split in ["all", "train", "test", "generalization"]:
        sdf = split_df[split_df["split"].eq(split)]
        if split == "generalization":
            add("generalization_clean", split, "clean_mean_relative_l2", sdf)
            add("generalization_clean", split, "clean_mean_rmse", sdf)
        add("attack20_52datasets", split, "mean_attack_loss_gain", sdf)
        add("attack20_52datasets", split, "mean_adv_loss", sdf)
        add("attack20_52datasets", split, "mean_attack_loss_gain_relative", sdf)

    add("metric25_attack_jacobian_proxy", "25_generalization_samples", "mean_attack_gain", metric25_df)
    add("metric25_attack_jacobian_proxy", "25_generalization_samples", "mean_jt_error", metric25_df)
    add("metric25_attack_jacobian_proxy", "25_generalization_samples", "mean_binary_first_order", metric25_df)
    add("metric25_attack_jacobian_proxy", "25_generalization_samples", "mean_sigma", metric25_df)

    add("jacobian5_probe", "5_generalization_samples", "mean_relative_l2", jac5_df)
    add("jacobian5_probe", "5_generalization_samples", "mean_jt_error_l2_norm", jac5_df)
    add("jacobian5_probe", "5_generalization_samples", "mean_jt_error_l2_norm_sq", jac5_df)
    add("jacobian5_probe", "5_generalization_samples", "mean_j_error_l2_norm", jac5_df)
    add("jacobian5_probe", "5_generalization_samples", "mean_spectral_norm_top_sigma", jac5_df)
    return pd.DataFrame(rows)


def coverage_table(clean: pd.DataFrame, attack: pd.DataFrame, metric25: pd.DataFrame, jac5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "model": model,
                "clean_52_dataset_metrics": int((clean["model"] == model).sum()),
                "attack20_52_dataset_metrics": int((attack["model"] == model).sum()),
                "metric25_sample_rows": int((metric25["model"] == model).sum()) if "sample_id" in metric25.columns else int(model in set(metric25["model"])),
                "jacobian5_model_mean_available": bool(model in set(jac5["model"])),
            }
        )
    out = pd.DataFrame(rows)
    out["clean_expected_rows"] = 52
    out["attack20_expected_rows"] = 52
    out["metric25_expected_rows"] = 25
    return out


def format_float(x: object) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6g}"
    return str(x)


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    d = df[cols].copy()
    if max_rows is not None:
        d = d.head(max_rows)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(format_float(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_dataset = read(CLEAN_DATASET)
    clean_split = read(CLEAN_SPLIT)
    attack_dataset = read(ATTACK_DATASET)
    attack_split = read(ATTACK_SPLIT)
    metric25_means_raw = read(METRIC25_MEANS)
    metric25_samples = read(METRIC25_SAMPLES)
    metric25_corr = read(METRIC25_CORR)
    jac5_raw = read(JAC5_MEANS)

    dataset52 = build_52x7_dataset_table(clean_dataset, attack_dataset)
    split = split_summary(clean_split, attack_split)
    metric25 = metric25_summary(metric25_means_raw)
    jac5 = jac5_summary(jac5_raw)
    winners = winners_from_summary(split, metric25, jac5)
    coverage = coverage_table(clean_dataset, attack_dataset, metric25_samples, jac5_raw)

    outputs = {
        "dataset52_model7_clean_attack20": write(dataset52, "dataset52_model7_clean_attack20.csv"),
        "split_model7_clean_attack20_summary": write(split, "split_model7_clean_attack20_summary.csv"),
        "metric25_model7_summary": write(metric25, "robustness_metric25_model7_summary.csv"),
        "metric25_model6_sample_rows": write(metric25_samples, "robustness_metric25_model6_sample_rows.csv"),
        "metric25_correlations": write(metric25_corr, "robustness_metric25_correlations.csv"),
        "jacobian5_model7_summary": write(jac5, "robustness_jacobian5_model7_summary.csv"),
        "winner_summary": write(winners, "winner_summary_by_metric.csv"),
        "coverage": write(coverage, "metric_coverage_by_model.csv"),
    }

    gen = split[split["split"].eq("generalization")].copy()
    gen_clean = gen.sort_values("clean_mean_relative_l2")
    gen_attack = gen.sort_values("mean_attack_loss_gain", na_position="last")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models_expected": MODEL_ORDER,
        "dataset_count_expected": 52,
        "outputs": {k: rel(v) for k, v in outputs.items()},
        "source_tables": {
            "clean_dataset": rel(CLEAN_DATASET),
            "clean_split": rel(CLEAN_SPLIT),
            "attack_dataset": rel(ATTACK_DATASET),
            "attack_split": rel(ATTACK_SPLIT),
            "metric25_means": rel(METRIC25_MEANS),
            "metric25_samples": rel(METRIC25_SAMPLES),
            "metric25_correlations": rel(METRIC25_CORR),
            "jacobian5_means": rel(JAC5_MEANS),
        },
        "important_note": "Clean metrics cover 52 datasets x 7 models. Existing robustness artifacts cover 6 models; baseline robustness is marked not_computed_in_existing_artifacts.",
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Darcy All-Model Full Statistics - 2026-06-13",
        "",
        "This report aggregates the requested Darcy statistics across 52 datasets and 7 models.",
        "",
        "Important: clean metrics are complete for all 7 models. Existing robustness artifacts cover 6 models; baseline robustness was not computed in the previous attack/Jacobian runs, so those baseline robustness fields are marked missing instead of being fabricated.",
        "",
        "## Output Tables",
        "",
    ]
    for key, path in outputs.items():
        lines.append(f"- `{key}`: `{rel(path)}`")
    lines.append(f"- `manifest`: `{rel(manifest_path)}`")
    lines += [
        "",
        "## Coverage",
        "",
        md_table(coverage, ["model", "clean_52_dataset_metrics", "attack20_52_dataset_metrics", "metric25_sample_rows", "jacobian5_model_mean_available"]),
        "",
        "## Generalization Clean Metrics, 7 Models",
        "",
        md_table(gen_clean, ["model", "clean_dataset_count", "clean_mean_relative_l2", "clean_mean_rmse", "attack20_available"]),
        "",
        "## Attack20 Generalization Robustness",
        "",
        md_table(gen_attack, ["model", "attack20_dataset_count", "attack20_sample_count", "mean_clean_loss", "mean_attack_loss_gain", "mean_adv_loss", "mean_attack_loss_gain_relative", "attack20_available", "attack20_missing_reason"]),
        "",
        "## Metric25 Robustness Proxy Summary",
        "",
        md_table(metric25.sort_values("mean_attack_gain", na_position="last"), ["model", "mean_attack_gain", "mean_jt_error", "mean_binary_first_order", "mean_sigma", "metric_available", "missing_reason"]),
        "",
        "## Jacobian5 Probe Summary",
        "",
        md_table(jac5.sort_values("mean_jt_error_l2_norm", na_position="last"), ["model", "mean_relative_l2", "mean_jt_error_l2_norm", "mean_jt_error_l2_norm_sq", "mean_j_error_l2_norm", "mean_spectral_norm_top_sigma", "metric_available", "missing_reason"]),
        "",
        "## Winner Summary",
        "",
        md_table(winners, ["metric_system", "scope", "metric", "best_model", "best_value", "model_count_available", "ranking"]),
        "",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": rel(OUT_DIR), "doc": rel(DOC_PATH), "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
