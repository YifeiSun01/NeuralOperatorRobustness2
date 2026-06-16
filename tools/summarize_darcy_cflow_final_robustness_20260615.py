#!/usr/bin/env python3
"""Summarize final Darcy CFlow robustness outputs.

This script is intentionally post-hoc: it reads the final 2026-06-15
robustness/SVD raw tables and writes compact CSV/Markdown summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "outputs/darcy_cflow_final_robustness_20260615"
CURRENT_GEN_TOKEN = "darcy_binary_loss3targeted_20260611"
OLD_GEN_TOKEN = "lossdrop50_selected_20260607"

METHOD_ORDER = [
    "baseline",
    "loss1",
    "loss2",
    "loss3",
    "physics_loss",
    "random_clean",
    "random_solver",
]

DISPLAY = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics_loss": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}

LOWER_IS_BETTER = {
    "clean_loss": True,
    "adv_loss": True,
    "loss_increase": True,
    "relative_increase": True,
    "delta_l2_rms": True,
    "delta_linf": True,
    "error_l2_norm": True,
    "jt_error_l2_norm": True,
    "sigma_input_right": True,
    "block2_sigma1": True,
    "topk_subspace_angle_jt_error_deg": True,
    "topk_subspace_angle_attack_delta_deg": True,
    "angle_singular_jt_error_deg": True,
    "angle_singular_attack_delta_deg": True,
    "angle_jt_error_attack_delta_deg": True,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def safe_corr(x: Iterable[float], y: Iterable[float], kind: str) -> float:
    xs = pd.Series(x, dtype="float64")
    ys = pd.Series(y, dtype="float64")
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(xs[mask].corr(ys[mask], method=kind))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def check_provenance(attack: pd.DataFrame, svd: pd.DataFrame) -> dict[str, object]:
    all_text = "\n".join(
        [
            "\n".join(attack.astype(str).agg(" ".join, axis=1).head(5_000).tolist()),
            "\n".join(svd.astype(str).agg(" ".join, axis=1).tolist()),
        ]
    )
    gen = attack[attack["split"].eq("generalization")]
    return {
        "old_lossdrop50_token_found": OLD_GEN_TOKEN in all_text,
        "generalization_dataset_count": int(gen["dataset_id"].nunique()),
        "generalization_dataset_prefix_all_current_20260611": bool(
            gen["dataset_id"].astype(str).str.startswith(CURRENT_GEN_TOKEN).all()
        ),
        "attack_rows": int(len(attack)),
        "svd_rows": int(len(svd)),
        "attack_steps": sorted(int(v) for v in attack["attack_steps"].dropna().unique()),
        "methods": sorted(attack["method"].unique(), key=method_sort_key),
    }


def summarize_attack(attack: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "clean_loss",
        "adv_loss",
        "loss_increase",
        "relative_increase",
        "delta_l2_rms",
        "delta_linf",
    ]
    grouped = (
        attack.groupby(["method", "method_display", "split"], sort=False)
        .agg(
            sample_count=("dataset_id", "size"),
            dataset_count=("dataset_id", "nunique"),
            svd_sample_count=("is_svd_sample", "sum"),
            attack_steps=("attack_steps", "max"),
            clean_loss_mean=("clean_loss", "mean"),
            clean_loss_median=("clean_loss", "median"),
            adv_loss_mean=("adv_loss", "mean"),
            adv_loss_median=("adv_loss", "median"),
            loss_increase_mean=("loss_increase", "mean"),
            loss_increase_median=("loss_increase", "median"),
            relative_increase_mean=("relative_increase", "mean"),
            relative_increase_median=("relative_increase", "median"),
            delta_l2_rms_mean=("delta_l2_rms", "mean"),
            delta_linf_mean=("delta_linf", "mean"),
        )
        .reset_index()
    )
    grouped["method_rank_key"] = grouped["method"].map(method_sort_key)
    grouped = grouped.sort_values(["method_rank_key", "split"]).drop(columns=["method_rank_key"])

    dataset = (
        attack.groupby(["method", "method_display", "split", "dataset_id"], sort=False)
        .agg(
            sample_count=("dataset_id", "size"),
            attack_steps=("attack_steps", "max"),
            clean_loss_mean=("clean_loss", "mean"),
            adv_loss_mean=("adv_loss", "mean"),
            loss_increase_mean=("loss_increase", "mean"),
            relative_increase_mean=("relative_increase", "mean"),
            delta_l2_rms_mean=("delta_l2_rms", "mean"),
            delta_linf_mean=("delta_linf", "mean"),
        )
        .reset_index()
    )

    wins: list[dict[str, object]] = []
    for split, sub in grouped.groupby("split", sort=False):
        for metric in metrics:
            col = f"{metric}_mean"
            if col not in sub:
                continue
            ranked = sub.sort_values(col, ascending=LOWER_IS_BETTER.get(metric, True))
            best = ranked.iloc[0]
            wins.append(
                {
                    "metric_system": "attack50_52datasets_50samples",
                    "scope": split,
                    "metric": col,
                    "best_model": best["method"],
                    "best_display": best["method_display"],
                    "best_value": float(best[col]),
                    "ranking": ",".join(ranked["method"].tolist()),
                    "model_count": int(len(ranked)),
                    "lower_is_better": LOWER_IS_BETTER.get(metric, True),
                }
            )
    return grouped, dataset, pd.DataFrame(wins)


def summarize_svd(svd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agg_cols = [
        "clean_loss",
        "attack_loss_increase",
        "attack_relative_increase",
        "error_l2_norm",
        "jt_error_l2_norm",
        "sigma_input_right",
        "block2_sigma1",
        "topk_subspace_cos_jt_error",
        "topk_subspace_angle_jt_error_deg",
        "topk_subspace_cos_attack_delta",
        "topk_subspace_angle_attack_delta_deg",
        "cos_singular_jt_error",
        "angle_singular_jt_error_deg",
        "corr_singular_jt_error",
        "cos_singular_attack_delta",
        "angle_singular_attack_delta_deg",
        "corr_singular_attack_delta",
        "cos_jt_error_attack_delta",
        "angle_jt_error_attack_delta_deg",
        "corr_jt_error_attack_delta",
    ]
    summary = (
        svd.groupby(["method", "method_display"], sort=False)
        .agg(sample_count=("dataset_id", "size"), dataset_count=("dataset_id", "nunique"))
        .reset_index()
    )
    for col in agg_cols:
        vals = svd.groupby(["method", "method_display"], sort=False)[col].agg(["mean", "median"]).reset_index()
        vals = vals.rename(columns={"mean": f"{col}_mean", "median": f"{col}_median"})
        summary = summary.merge(vals, on=["method", "method_display"], how="left")
    summary["method_rank_key"] = summary["method"].map(method_sort_key)
    summary = summary.sort_values("method_rank_key").drop(columns=["method_rank_key"])

    wins: list[dict[str, object]] = []
    for metric in [
        "attack_loss_increase",
        "attack_relative_increase",
        "error_l2_norm",
        "jt_error_l2_norm",
        "sigma_input_right",
        "block2_sigma1",
        "angle_singular_attack_delta_deg",
        "angle_jt_error_attack_delta_deg",
    ]:
        col = f"{metric}_mean"
        ranked = summary.sort_values(col, ascending=LOWER_IS_BETTER.get(metric, True))
        best = ranked.iloc[0]
        wins.append(
            {
                "metric_system": "svd_jacobian_25samples",
                "scope": "fixed_25_samples",
                "metric": col,
                "best_model": best["method"],
                "best_display": best["method_display"],
                "best_value": float(best[col]),
                "ranking": ",".join(ranked["method"].tolist()),
                "model_count": int(len(ranked)),
                "lower_is_better": LOWER_IS_BETTER.get(metric, True),
            }
        )

    corr_rows: list[dict[str, object]] = []
    pairs = [
        ("sigma_input_right", "attack_loss_increase"),
        ("block2_sigma1", "attack_loss_increase"),
        ("jt_error_l2_norm", "attack_loss_increase"),
        ("sigma_input_right", "jt_error_l2_norm"),
        ("block2_sigma1", "jt_error_l2_norm"),
        ("error_l2_norm", "attack_loss_increase"),
        ("clean_loss", "attack_loss_increase"),
    ]
    for scope, sub in [("overall", svd)] + [(f"model:{m}", g) for m, g in svd.groupby("method", sort=False)]:
        for x, y in pairs:
            corr_rows.append(
                {
                    "scope": scope,
                    "x": x,
                    "y": y,
                    "n": int(sub[[x, y]].dropna().shape[0]),
                    "pearson_r": safe_corr(sub[x], sub[y], "pearson"),
                    "spearman_rho": safe_corr(sub[x], sub[y], "spearman"),
                }
            )

    return summary, pd.DataFrame(wins), pd.DataFrame(corr_rows)


def write_report(
    bundle: Path,
    provenance: dict[str, object],
    attack_summary: pd.DataFrame,
    svd_summary: pd.DataFrame,
    winners: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    report = bundle / "reports" / "final_robustness_summary_20260615.md"
    lines: list[str] = [
        "# Darcy CFlow Final Robustness Summary - 2026-06-15",
        "",
        "## Scope",
        "",
        "Observed from the final post-hoc robustness run in this bundle.",
        "",
        f"- Generalization root: `generalization_datasets_darcy_binary_loss3targeted_20260611/`",
        f"- Generalization datasets: `{provenance['generalization_dataset_count']}`",
        f"- Generalization prefix check: `{provenance['generalization_dataset_prefix_all_current_20260611']}`",
        f"- Old lossdrop50 token found in raw summary inputs: `{provenance['old_lossdrop50_token_found']}`",
        f"- Attack rows: `{provenance['attack_rows']}`",
        f"- SVD/Jacobian rows: `{provenance['svd_rows']}`",
        f"- Attack steps: `{provenance['attack_steps']}`",
        "",
        "The 52-dataset attack set is train + test + 50 binary 20260611 generalization datasets.",
        "",
        "## Attack50 Generalization Means",
        "",
        "| model | samples | clean loss | adv loss | loss increase | relative increase | delta L2 RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    gen = attack_summary[attack_summary["split"].eq("generalization")].copy()
    gen = gen.sort_values("loss_increase_mean")
    for _, row in gen.iterrows():
        lines.append(
            f"| {row['method_display']} | {int(row['sample_count'])} | "
            f"{row['clean_loss_mean']:.6g} | {row['adv_loss_mean']:.6g} | "
            f"{row['loss_increase_mean']:.6g} | {row['relative_increase_mean']:.6g} | "
            f"{row['delta_l2_rms_mean']:.6g} |"
        )
    lines += [
        "",
        "## SVD/Jacobian Fixed-25 Means",
        "",
        "| model | samples | attack increase | J^T error norm | sigma input right | angle singular-delta | angle J^T error-delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    svd_order = svd_summary.sort_values("attack_loss_increase_mean")
    for _, row in svd_order.iterrows():
        lines.append(
            f"| {row['method_display']} | {int(row['sample_count'])} | "
            f"{row['attack_loss_increase_mean']:.6g} | {row['jt_error_l2_norm_mean']:.6g} | "
            f"{row['sigma_input_right_mean']:.6g} | {row['angle_singular_attack_delta_deg_mean']:.3f} | "
            f"{row['angle_jt_error_attack_delta_deg_mean']:.3f} |"
        )
    lines += [
        "",
        "## Winner Summary",
        "",
        "| metric system | scope | metric | best model | best value | ranking |",
        "|---|---|---|---|---:|---|",
    ]
    for _, row in winners.iterrows():
        lines.append(
            f"| {row['metric_system']} | {row['scope']} | {row['metric']} | "
            f"{row['best_display']} | {row['best_value']:.6g} | `{row['ranking']}` |"
        )
    lines += [
        "",
        "## Correlation Files",
        "",
        f"- `data/svd_scalar_correlations.csv` contains scalar correlations for same-sample SVD/Jacobian/attack metrics.",
        f"- `data/svd_jacobian_metrics.csv` contains the per-model, per-sample vector angle/cosine/correlation diagnostics.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    bundle = DEFAULT_BUNDLE
    data = bundle / "data"
    attack_csv = data / "robustness_attack_52datasets_samples.csv"
    svd_csv = data / "svd_jacobian_metrics.csv"
    attack = pd.read_csv(attack_csv)
    svd = pd.read_csv(svd_csv)

    provenance = check_provenance(attack, svd)
    if provenance["old_lossdrop50_token_found"]:
        raise SystemExit("Refusing to summarize: old 20260607 lossdrop50 token was found.")
    if not provenance["generalization_dataset_prefix_all_current_20260611"]:
        raise SystemExit("Refusing to summarize: not all generalization rows use 20260611 binary dataset ids.")

    attack_summary, dataset_summary, attack_winners = summarize_attack(attack)
    svd_summary, svd_winners, scalar_corr = summarize_svd(svd)
    winners = pd.concat([attack_winners, svd_winners], ignore_index=True)

    write_csv(data / "attack50_summary_by_model_split.csv", attack_summary)
    write_csv(data / "attack50_summary_by_dataset_model.csv", dataset_summary)
    write_csv(data / "svd_jacobian_summary_by_model.csv", svd_summary)
    write_csv(data / "svd_scalar_correlations.csv", scalar_corr)
    write_csv(data / "winner_summary_by_metric.csv", winners)
    (data / "final_robustness_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_report(bundle, provenance, attack_summary, svd_summary, winners, scalar_corr)

    print(bundle / "reports" / "final_robustness_summary_20260615.md")


if __name__ == "__main__":
    main()
