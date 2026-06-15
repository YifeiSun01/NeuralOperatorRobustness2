#!/usr/bin/env python3
"""Summarize final Darcy CFlow robustness, SVD/Jacobian, and correlations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DISPLAY_NAMES = {
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def finite_float_series(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    return out[np.isfinite(out)]


def mean_or_nan(values: pd.Series) -> float:
    vals = finite_float_series(values)
    return float(vals.mean()) if len(vals) else float("nan")


def method_display(method: object) -> str:
    key = str(method)
    return DISPLAY_NAMES.get(key, key)


def corr_value(df: pd.DataFrame, x_col: str, y_col: str, method: str) -> tuple[float, int]:
    sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]
    if len(sub) < 2:
        return float("nan"), int(len(sub))
    x = sub[x_col].to_numpy(dtype=np.float64)
    y = sub[y_col].to_numpy(dtype=np.float64)
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-30:
        return float("nan"), int(len(sub))
    return float(np.dot(x, y) / den), int(len(sub))


def grouped_with_all(df: pd.DataFrame, keys: list[str]) -> Iterable[tuple[tuple[str, ...], pd.DataFrame]]:
    for group_key, sub in df.groupby(keys, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        yield tuple(str(x) for x in group_key), sub


def attack_summary(attack: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "clean_loss",
        "adv_loss",
        "loss_increase",
        "relative_increase",
        "delta_l2_rms",
        "delta_linf",
        "delta_std",
    ]
    for method, sub_m in attack.groupby("method", sort=False):
        groups = [("ALL", sub_m)]
        groups.extend((str(split), sub) for split, sub in sub_m.groupby("split", sort=False))
        for split, sub in groups:
            row: dict[str, object] = {
                "method_key": str(method),
                "method": method_display(method),
                "split": split,
                "datasets": int(sub["dataset_id"].nunique()),
                "samples": int(len(sub)),
                "attack_steps": int(pd.to_numeric(sub["attack_steps"], errors="coerce").dropna().iloc[0]) if len(sub) else "",
            }
            for metric in metrics:
                row[f"{metric}_mean"] = mean_or_nan(sub[metric])
                row[f"{metric}_median"] = float(finite_float_series(sub[metric]).median()) if len(finite_float_series(sub[metric])) else float("nan")
            clean = float(row["clean_loss_mean"])
            adv = float(row["adv_loss_mean"])
            row["relative_increase_from_means"] = (adv - clean) / clean if math.isfinite(clean) and abs(clean) > 1e-30 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def scalar_correlations(svd: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("sigma_input_right", "jt_error_l2_norm"),
        ("sigma_input_right", "attack_loss_increase"),
        ("sigma_input_right", "attack_relative_increase"),
        ("block2_sigma1", "jt_error_l2_norm"),
        ("block2_sigma1", "attack_loss_increase"),
        ("block2_sigma1", "attack_relative_increase"),
        ("jt_error_l2_norm", "attack_loss_increase"),
        ("jt_error_l2_norm", "attack_relative_increase"),
        ("error_l2_norm", "attack_loss_increase"),
    ]
    rows: list[dict[str, object]] = []
    for method, sub_m in svd.groupby("method", sort=False):
        groups = [("ALL", sub_m)]
        groups.extend((str(split), sub) for split, sub in sub_m.groupby("split", sort=False))
        for split, sub in groups:
            for x_col, y_col in pairs:
                for corr_method in ("pearson", "spearman"):
                    value, n = corr_value(sub, x_col, y_col, corr_method)
                    rows.append(
                        {
                            "method_key": str(method),
                            "method": method_display(method),
                            "split": split,
                            "x": x_col,
                            "y": y_col,
                            "correlation": corr_method,
                            "value": value,
                            "n": n,
                        }
                    )
    pooled = svd.copy()
    for x_col, y_col in pairs:
        for corr_method in ("pearson", "spearman"):
            value, n = corr_value(pooled, x_col, y_col, corr_method)
            rows.append(
                {
                    "method_key": "ALL_METHODS_POOLED",
                    "method": method_display("ALL_METHODS_POOLED"),
                    "split": "ALL",
                    "x": x_col,
                    "y": y_col,
                    "correlation": corr_method,
                    "value": value,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def vector_alignment_summary(svd: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "cos_singular_jt_error",
        "angle_singular_jt_error_deg",
        "corr_singular_jt_error",
        "cos_singular_attack_delta",
        "angle_singular_attack_delta_deg",
        "corr_singular_attack_delta",
        "cos_jt_error_attack_delta",
        "angle_jt_error_attack_delta_deg",
        "corr_jt_error_attack_delta",
        "topk_subspace_cos_jt_error",
        "topk_subspace_angle_jt_error_deg",
        "topk_subspace_cos_attack_delta",
        "topk_subspace_angle_attack_delta_deg",
    ]
    rows: list[dict[str, object]] = []
    for method, sub_m in svd.groupby("method", sort=False):
        groups = [("ALL", sub_m)]
        groups.extend((str(split), sub) for split, sub in sub_m.groupby("split", sort=False))
        for split, sub in groups:
            row: dict[str, object] = {
                "method_key": str(method),
                "method": method_display(method),
                "split": split,
                "samples": int(len(sub)),
            }
            for metric in metrics:
                if metric in sub.columns:
                    row[f"{metric}_mean"] = mean_or_nan(sub[metric])
                    row[f"{metric}_median"] = float(finite_float_series(sub[metric]).median()) if len(finite_float_series(sub[metric])) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def best_rows(summary: pd.DataFrame, split: str) -> list[str]:
    sub = summary[summary["split"] == split].copy()
    lines: list[str] = []
    metric_labels = [
        ("clean_loss_mean", "clean loss"),
        ("adv_loss_mean", "attack final loss"),
        ("loss_increase_mean", "attack loss increase"),
        ("relative_increase_from_means", "relative increase from means"),
        ("delta_l2_rms_mean", "delta L2 RMS"),
    ]
    for metric, label in metric_labels:
        vals = pd.to_numeric(sub[metric], errors="coerce")
        if vals.notna().any():
            idx = vals.idxmin()
            lines.append(f"- Lowest {label}: `{sub.loc[idx, 'method']}` ({vals.loc[idx]:.6g}).")
    return lines


def write_report(bundle: Path, attack_summary_df: pd.DataFrame, svd: pd.DataFrame, corr_df: pd.DataFrame, align_df: pd.DataFrame) -> None:
    report = bundle / "reports" / "final_robustness_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Darcy CFlow Final Robustness Summary",
        "",
        "Observed from the final seven-model attack20/SVD25 outputs in this bundle.",
        "",
        "## Files",
        "",
        f"- Attack summary: `{rel(bundle / 'data' / 'attack20_summary_by_model_split.csv')}`",
        f"- SVD/Jacobian metrics: `{rel(bundle / 'data' / 'svd_jacobian_metrics.csv')}`",
        f"- Scalar correlations: `{rel(bundle / 'data' / 'svd_attack_scalar_correlations.csv')}`",
        f"- Vector alignment summary: `{rel(bundle / 'data' / 'vector_alignment_summary_by_model.csv')}`",
        "",
        "## Counts",
        "",
        f"- Attack summary rows: `{len(attack_summary_df)}`",
        f"- SVD/Jacobian rows: `{len(svd)}`",
        f"- Scalar correlation rows: `{len(corr_df)}`",
        f"- Vector alignment summary rows: `{len(align_df)}`",
        "",
        "## Best Models By Mean Attack Metric",
        "",
        "Generalization split:",
        "",
        *best_rows(attack_summary_df, "generalization"),
        "",
        "All splits pooled:",
        "",
        *best_rows(attack_summary_df, "ALL"),
        "",
        "## Method Summary",
        "",
        "| method | split | samples | clean | adv | increase | relative from means |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in attack_summary_df.iterrows():
        if row["split"] not in {"ALL", "generalization", "train", "test"}:
            continue
        lines.append(
            f"| {row['method']} | {row['split']} | {int(row['samples'])} | "
            f"{float(row['clean_loss_mean']):.6g} | {float(row['adv_loss_mean']):.6g} | "
            f"{float(row['loss_increase_mean']):.6g} | {float(row['relative_increase_from_means']):.6g} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    data = bundle / "data"
    attack_path = data / "robustness_attack_52datasets_samples.csv"
    svd_path = data / "svd_jacobian_metrics.csv"
    if not attack_path.exists():
        raise FileNotFoundError(attack_path)
    if not svd_path.exists():
        raise FileNotFoundError(svd_path)

    attack = pd.read_csv(attack_path)
    svd = pd.read_csv(svd_path)
    attack_summary_df = attack_summary(attack)
    corr_df = scalar_correlations(svd)
    align_df = vector_alignment_summary(svd)

    attack_summary_df.to_csv(data / "attack20_summary_by_model_split.csv", index=False)
    corr_df.to_csv(data / "svd_attack_scalar_correlations.csv", index=False)
    align_df.to_csv(data / "vector_alignment_summary_by_model.csv", index=False)
    write_report(bundle, attack_summary_df, svd, corr_df, align_df)
    print(bundle / "reports" / "final_robustness_summary.md")


if __name__ == "__main__":
    main()
