#!/usr/bin/env python3
"""Build the final Darcy 52-dataset x 7-model x 8-metric matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_DISPLAY = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "Physics Loss",
    "physics_loss": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}
METHOD_ORDER = ["baseline", "loss1", "loss2", "loss3", "physics_loss", "random_clean", "random_solver"]

EIGHT_METRICS = [
    "clean_rmse",
    "clean_relative_l2",
    "attack_clean_loss_mean",
    "attack_adv_loss_mean",
    "attack_loss_increase_mean",
    "attack_relative_increase_mean",
    "attack_delta_l2_rms_mean",
    "attack_delta_linf_mean",
]

SVD_METRICS = [
    "error_l2_norm",
    "jt_error_l2_norm",
    "sigma_input_right",
    "block2_sigma1",
    "attack_loss_increase",
    "attack_relative_increase",
    "cos_singular_jt_error",
    "cos_singular_attack_delta",
    "cos_jt_error_attack_delta",
    "topk_subspace_cos_jt_error",
    "topk_subspace_cos_attack_delta",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_method(method: object) -> str:
    key = str(method)
    if key == "physics":
        return "physics_loss"
    return key


def display_method(method: object) -> str:
    return METHOD_DISPLAY.get(str(method), str(method))


def rank_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    for metric in EIGHT_METRICS:
        rank_col = f"{metric}_rank"
        out[rank_col] = out.groupby("dataset_id")[metric].rank(method="min", ascending=True)
        out[f"{metric}_is_best"] = out[rank_col].eq(1)
    return out


def best_count_table(ranked: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = ranked if split == "ALL" else ranked[ranked["split"] == split]
    rows = []
    for method in METHOD_ORDER:
        sm = sub[sub["method_key"] == method]
        row = {
            "method_key": method,
            "method": display_method(method),
            "split": split,
            "datasets": int(sm["dataset_id"].nunique()),
        }
        for metric in EIGHT_METRICS:
            row[f"{metric}_best_count"] = int(sm[f"{metric}_is_best"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def model_mean_table(matrix: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = matrix if split == "ALL" else matrix[matrix["split"] == split]
    grouped = sub.groupby(["method_key", "method"], sort=False)
    rows = []
    for (method_key, method), sm in grouped:
        row = {
            "method_key": method_key,
            "method": method,
            "split": split,
            "datasets": int(sm["dataset_id"].nunique()),
        }
        for metric in EIGHT_METRICS:
            row[f"{metric}_mean"] = float(pd.to_numeric(sm[metric], errors="coerce").mean())
            row[f"{metric}_median"] = float(pd.to_numeric(sm[metric], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 1e-3:
        return f"{value:.4e}"
    return f"{value:.6g}"


def write_report(
    path: Path,
    matrix: pd.DataFrame,
    means: pd.DataFrame,
    best_counts: pd.DataFrame,
    svd_means: pd.DataFrame,
) -> None:
    gen = means[means["split"] == "generalization"].copy()
    all_split = means[means["split"] == "ALL"].copy()
    gen_best = best_counts[best_counts["split"] == "generalization"].copy()
    all_best = best_counts[best_counts["split"] == "ALL"].copy()

    lines = [
        "# Darcy CFlow Final 8-Metric Matrix Summary",
        "",
        "Observed from the final seven-model clean evaluation, attack20, and SVD/Jacobian outputs on the same lossdrop50 52-dataset set.",
        "",
        "## Matrix Definition",
        "",
        "- Rows: `52 datasets x 7 models = 364`.",
        "- Metrics per row: `8`.",
        "- Clean metrics: `clean_rmse`, `clean_relative_l2`.",
        "- Attack20 metrics: `attack_clean_loss_mean`, `attack_adv_loss_mean`, `attack_loss_increase_mean`, `attack_relative_increase_mean`, `attack_delta_l2_rms_mean`, `attack_delta_linf_mean`.",
        "- Ranking convention in this summary: lower is better for all eight scalar metrics; delta magnitudes are diagnostics, not standalone robustness claims.",
        "- SVD/Jacobian diagnostics are a separate fixed-sample table: `25 samples x 7 models = 175`, as requested for compute cost.",
        "",
        "## Files",
        "",
        f"- 8-metric matrix: `{rel(path.parent.parent / 'data' / 'final_52dataset_7model_8metric_matrix.csv')}`",
        f"- Ranked matrix: `{rel(path.parent.parent / 'data' / 'final_52dataset_7model_8metric_ranked.csv')}`",
        f"- Model means: `{rel(path.parent.parent / 'data' / 'final_8metric_model_means.csv')}`",
        f"- Best counts: `{rel(path.parent.parent / 'data' / 'final_8metric_best_counts.csv')}`",
        f"- SVD/Jacobian means: `{rel(path.parent.parent / 'data' / 'final_svd25_model_means.csv')}`",
        "",
        "## Generalization Means",
        "",
        "| method | datasets | clean RMSE | clean RelL2 | adv loss | loss increase | relative increase | delta L2 | delta Linf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in gen.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['datasets'])} | "
            f"{format_float(row['clean_rmse_mean'])} | {format_float(row['clean_relative_l2_mean'])} | "
            f"{format_float(row['attack_adv_loss_mean_mean'])} | {format_float(row['attack_loss_increase_mean_mean'])} | "
            f"{format_float(row['attack_relative_increase_mean_mean'])} | {format_float(row['attack_delta_l2_rms_mean_mean'])} | "
            f"{format_float(row['attack_delta_linf_mean_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Generalization Best Counts Out Of 50",
            "",
            "| method | clean RMSE | clean RelL2 | attack clean | adv loss | loss increase | relative increase | delta L2 | delta Linf |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in gen_best.iterrows():
        lines.append(
            f"| {row['method']} | "
            f"{row['clean_rmse_best_count']} | {row['clean_relative_l2_best_count']} | "
            f"{row['attack_clean_loss_mean_best_count']} | {row['attack_adv_loss_mean_best_count']} | "
            f"{row['attack_loss_increase_mean_best_count']} | {row['attack_relative_increase_mean_best_count']} | "
            f"{row['attack_delta_l2_rms_mean_best_count']} | {row['attack_delta_linf_mean_best_count']} |"
        )
    lines.extend(
        [
            "",
            "## All 52-Dataset Best Counts",
            "",
            "| method | clean RMSE | clean RelL2 | attack clean | adv loss | loss increase | relative increase | delta L2 | delta Linf |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in all_best.iterrows():
        lines.append(
            f"| {row['method']} | "
            f"{row['clean_rmse_best_count']} | {row['clean_relative_l2_best_count']} | "
            f"{row['attack_clean_loss_mean_best_count']} | {row['attack_adv_loss_mean_best_count']} | "
            f"{row['attack_loss_increase_mean_best_count']} | {row['attack_relative_increase_mean_best_count']} | "
            f"{row['attack_delta_l2_rms_mean_best_count']} | {row['attack_delta_linf_mean_best_count']} |"
        )
    lines.extend(
        [
            "",
            "## SVD/Jacobian 25-Sample Means",
            "",
            "| method | samples | sigma | JT-error norm | SVD attack cos | JT-error attack cos | top-k attack subspace cos |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in svd_means[svd_means["split"] == "ALL"].iterrows():
        lines.append(
            f"| {row['method']} | {int(row['samples'])} | "
            f"{format_float(row['sigma_input_right_mean'])} | {format_float(row['jt_error_l2_norm_mean'])} | "
            f"{format_float(row['cos_singular_attack_delta_mean'])} | {format_float(row['cos_jt_error_attack_delta_mean'])} | "
            f"{format_float(row['topk_subspace_cos_attack_delta_mean'])} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    data = bundle / "data"
    clean = pd.read_csv(data / "final_eval_metrics.csv")
    attack = pd.read_csv(data / "robustness_attack_52datasets_samples.csv")
    svd = pd.read_csv(data / "svd_jacobian_metrics.csv")

    clean["method_key"] = clean["method"].map(normalize_method)
    clean["method"] = clean["method_key"].map(display_method)
    clean_wide = clean[
        [
            "method_key",
            "method",
            "dataset_id",
            "split",
            "source",
            "manual_tier",
            "manual_rank",
            "num_samples_evaluated",
            "rmse",
            "relative_l2",
        ]
    ].rename(
        columns={
            "num_samples_evaluated": "clean_samples",
            "rmse": "clean_rmse",
            "relative_l2": "clean_relative_l2",
        }
    )

    attack["method_key"] = attack["method"].map(normalize_method)
    attack_agg = (
        attack.groupby(["method_key", "dataset_id", "split"], sort=False)
        .agg(
            attack_samples=("sample_ordinal", "count"),
            attack_clean_loss_mean=("clean_loss", "mean"),
            attack_adv_loss_mean=("adv_loss", "mean"),
            attack_loss_increase_mean=("loss_increase", "mean"),
            attack_relative_increase_mean=("relative_increase", "mean"),
            attack_delta_l2_rms_mean=("delta_l2_rms", "mean"),
            attack_delta_linf_mean=("delta_linf", "mean"),
        )
        .reset_index()
    )

    matrix = clean_wide.merge(attack_agg, on=["method_key", "dataset_id", "split"], how="inner")
    expected_rows = 52 * 7
    if len(matrix) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} merged rows, got {len(matrix)}")
    missing = matrix[EIGHT_METRICS].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(f"missing values in matrix metrics:\n{missing}")

    method_order = {m: i for i, m in enumerate(METHOD_ORDER)}
    matrix["_method_order"] = matrix["method_key"].map(method_order)
    matrix = matrix.sort_values(["split", "manual_rank", "dataset_id", "_method_order"]).drop(columns=["_method_order"])
    ranked = rank_matrix(matrix)

    mean_tables = [model_mean_table(matrix, split) for split in ["ALL", "train", "test", "generalization"]]
    means = pd.concat(mean_tables, ignore_index=True)
    best_tables = [best_count_table(ranked, split) for split in ["ALL", "train", "test", "generalization"]]
    best_counts = pd.concat(best_tables, ignore_index=True)

    svd["method_key"] = svd["method"].map(normalize_method)
    svd["method"] = svd["method_key"].map(display_method)
    svd_rows = []
    for split in ["ALL", "train", "test", "generalization"]:
        sub_all = svd if split == "ALL" else svd[svd["split"] == split]
        for method in METHOD_ORDER:
            sub = sub_all[sub_all["method_key"] == method]
            if sub.empty:
                continue
            row = {
                "method_key": method,
                "method": display_method(method),
                "split": split,
                "samples": int(len(sub)),
                "datasets": int(sub["dataset_id"].nunique()),
            }
            for metric in SVD_METRICS:
                row[f"{metric}_mean"] = float(pd.to_numeric(sub[metric], errors="coerce").mean())
                row[f"{metric}_median"] = float(pd.to_numeric(sub[metric], errors="coerce").median())
            svd_rows.append(row)
    svd_means = pd.DataFrame(svd_rows)

    matrix.to_csv(data / "final_52dataset_7model_8metric_matrix.csv", index=False)
    ranked.to_csv(data / "final_52dataset_7model_8metric_ranked.csv", index=False)
    means.to_csv(data / "final_8metric_model_means.csv", index=False)
    best_counts.to_csv(data / "final_8metric_best_counts.csv", index=False)
    svd_means.to_csv(data / "final_svd25_model_means.csv", index=False)
    write_report(bundle / "reports" / "final_8metric_matrix_summary.md", matrix, means, best_counts, svd_means)
    print(bundle / "reports" / "final_8metric_matrix_summary.md")


if __name__ == "__main__":
    main()
