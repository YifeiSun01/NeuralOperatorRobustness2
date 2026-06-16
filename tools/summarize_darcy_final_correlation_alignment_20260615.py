#!/usr/bin/env python3
"""Summarize Darcy final scalar correlations and vector alignments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DISPLAY = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "Physics Loss",
    "physics_loss": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
    "ALL_METHODS_POOLED": "ALL_METHODS_POOLED",
}
METHOD_ORDER = ["baseline", "loss1", "loss2", "loss3", "physics_loss", "random_clean", "random_solver"]

SCALAR_PAIRS = [
    ("sigma_input_right", "attack_loss_increase"),
    ("sigma_input_right", "attack_relative_increase"),
    ("jt_error_l2_norm", "attack_loss_increase"),
    ("jt_error_l2_norm", "attack_relative_increase"),
    ("sigma_input_right", "jt_error_l2_norm"),
    ("block2_sigma1", "attack_loss_increase"),
]

VECTOR_COLUMNS = [
    "cos_singular_attack_delta",
    "angle_singular_attack_delta_deg",
    "cos_jt_error_attack_delta",
    "angle_jt_error_attack_delta_deg",
    "cos_singular_jt_error",
    "angle_singular_jt_error_deg",
    "topk_subspace_cos_attack_delta",
    "topk_subspace_angle_attack_delta_deg",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 1e-3:
        return f"{value:.4e}"
    return f"{value:.4f}"


def method_label(method_key: str) -> str:
    return DISPLAY.get(str(method_key), str(method_key))


def scalar_subset(corr: pd.DataFrame, split: str, corr_type: str) -> pd.DataFrame:
    sub = corr[(corr["split"] == split) & (corr["correlation"] == corr_type)].copy()
    wanted = pd.MultiIndex.from_tuples(SCALAR_PAIRS, names=["x", "y"])
    sub = sub.set_index(["x", "y"]).loc[wanted.intersection(sub.set_index(["x", "y"]).index)].reset_index()
    sub["pair"] = sub["x"] + " vs " + sub["y"]
    return sub


def build_scalar_wide(corr: pd.DataFrame, split: str, corr_type: str) -> pd.DataFrame:
    sub = scalar_subset(corr, split, corr_type)
    wide = sub.pivot_table(index=["method_key", "method"], columns="pair", values="value", aggfunc="first").reset_index()
    order = {m: i for i, m in enumerate(METHOD_ORDER + ["ALL_METHODS_POOLED"])}
    wide["_order"] = wide["method_key"].map(order).fillna(999)
    return wide.sort_values("_order").drop(columns="_order")


def build_vector_digest(align: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = align[align["split"] == split].copy()
    keep = ["method_key", "method", "samples"]
    for col in VECTOR_COLUMNS:
        keep.extend([f"{col}_mean", f"{col}_median"])
    keep = [c for c in keep if c in sub.columns]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    sub["_order"] = sub["method_key"].map(order).fillna(999)
    return sub.sort_values("_order")[keep]


def strongest_abs_correlations(corr: pd.DataFrame, split: str, corr_type: str, n: int = 12) -> pd.DataFrame:
    sub = corr[(corr["split"] == split) & (corr["correlation"] == corr_type)].copy()
    sub["abs_value"] = pd.to_numeric(sub["value"], errors="coerce").abs()
    sub = sub[np.isfinite(sub["abs_value"])]
    return sub.sort_values("abs_value", ascending=False).head(n)


def write_report(bundle: Path, scalar_wide: pd.DataFrame, vector_digest: pd.DataFrame, top_corr: pd.DataFrame) -> None:
    report = bundle / "reports" / "final_correlation_alignment_summary.md"
    data = bundle / "data"
    lines = [
        "# Darcy CFlow Final Correlation And Alignment Summary",
        "",
        "Observed from the final attack20/SVD25 run.",
        "",
        "## Files",
        "",
        f"- Scalar correlations: `{rel(data / 'svd_attack_scalar_correlations.csv')}`",
        f"- Vector alignment summary: `{rel(data / 'vector_alignment_summary_by_model.csv')}`",
        f"- Per-sample SVD/Jacobian metrics: `{rel(data / 'svd_jacobian_metrics.csv')}`",
        f"- Scalar digest CSV: `{rel(data / 'final_scalar_correlation_digest.csv')}`",
        f"- Vector digest CSV: `{rel(data / 'final_vector_alignment_digest.csv')}`",
        "",
        "## Counts",
        "",
        "- Scalar correlation rows: `522`.",
        "- Vector alignment summary rows: `28`.",
        "- Per-sample SVD/Jacobian rows: `175`.",
        "",
        "## Generalization Pearson Scalar Correlations",
        "",
        "| method | sigma vs increase | sigma vs relative | JT norm vs increase | JT norm vs relative | sigma vs JT norm | block2 sigma vs increase |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cols = [
        "sigma_input_right vs attack_loss_increase",
        "sigma_input_right vs attack_relative_increase",
        "jt_error_l2_norm vs attack_loss_increase",
        "jt_error_l2_norm vs attack_relative_increase",
        "sigma_input_right vs jt_error_l2_norm",
        "block2_sigma1 vs attack_loss_increase",
    ]
    for _, row in scalar_wide.iterrows():
        lines.append(
            f"| {row['method']} | "
            + " | ".join(fmt(float(row.get(col, np.nan))) for col in cols)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Generalization Vector Alignment Means",
            "",
            "| method | samples | singular-delta cos | singular-delta angle | JT-delta cos | JT-delta angle | singular-JT cos | singular-JT angle | top-k delta cos | top-k delta angle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in vector_digest.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['samples'])} | "
            f"{fmt(float(row['cos_singular_attack_delta_mean']))} | "
            f"{fmt(float(row['angle_singular_attack_delta_deg_mean']))} | "
            f"{fmt(float(row['cos_jt_error_attack_delta_mean']))} | "
            f"{fmt(float(row['angle_jt_error_attack_delta_deg_mean']))} | "
            f"{fmt(float(row['cos_singular_jt_error_mean']))} | "
            f"{fmt(float(row['angle_singular_jt_error_deg_mean']))} | "
            f"{fmt(float(row['topk_subspace_cos_attack_delta_mean']))} | "
            f"{fmt(float(row['topk_subspace_angle_attack_delta_deg_mean']))} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Generalization Pearson Correlations",
            "",
            "| method | x | y | value | n |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in top_corr.iterrows():
        lines.append(f"| {row['method']} | {row['x']} | {row['y']} | {fmt(float(row['value']))} | {int(row['n'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Single-vector alignments are generally weak: singular-vector vs attack-delta and `J^T error` vs attack-delta cosines are small in magnitude for most models.",
            "- Top-k singular subspace overlap with attack delta is moderate, but not close to one, so attack deltas are not explained by the leading singular vector/subspace alone.",
            "- Scalar correlations vary strongly by model and should be read as 25-sample diagnostics, not as full 52-dataset statistics.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    data = bundle / "data"
    corr = pd.read_csv(data / "svd_attack_scalar_correlations.csv")
    align = pd.read_csv(data / "vector_alignment_summary_by_model.csv")

    scalar_wide = build_scalar_wide(corr, "generalization", "pearson")
    vector_digest = build_vector_digest(align, "generalization")
    top_corr = strongest_abs_correlations(corr, "generalization", "pearson", n=12)

    scalar_wide.to_csv(data / "final_scalar_correlation_digest.csv", index=False)
    vector_digest.to_csv(data / "final_vector_alignment_digest.csv", index=False)
    top_corr.to_csv(data / "final_top_scalar_correlations.csv", index=False)
    write_report(bundle, scalar_wide, vector_digest, top_corr)
    print(bundle / "reports" / "final_correlation_alignment_summary.md")


if __name__ == "__main__":
    main()
