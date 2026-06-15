#!/usr/bin/env python3
"""Build explicit mean/std tables for final Darcy robustness diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROBUST = PROJECT / "outputs/darcy_cflow_final_robustness_20260615"
OUT = ROBUST / "data/final_metric_mean_std_20260615"
REPORT = ROBUST / "reports/final_metric_mean_std_20260615.md"
DOC = PROJECT / "docs/darcy_cflow_final_metric_mean_std_20260615.md"

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

ATTACK_METRICS = [
    "clean_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "delta_l2_rms",
    "delta_linf",
    "delta_mean",
    "delta_std",
]

SVD_METRICS = [
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


def method_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def add_mean_std(grouped, metrics: list[str]) -> pd.DataFrame:
    group_names = list(grouped._grouper.names)
    out = grouped.size().rename("sample_count").reset_index()
    for metric in metrics:
        agg = grouped[metric].agg(["mean", "std", "median", "min", "max"]).reset_index()
        agg = agg.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "median": f"{metric}_median",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
            }
        )
        out = out.merge(agg, on=group_names, how="left")
    return out


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
    lines.extend("| " + " | ".join(row) + " |" for row in show.astype(str).values.tolist())
    return "\n".join(lines)


def parse_top_singular_values(svd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in svd.iterrows():
        vals = json.loads(row["block2_top_singular_values_json"])
        item = {
            "method": row["method"],
            "method_display": row["method_display"],
            "dataset_id": row["dataset_id"],
            "split": row["split"],
            "sample_index": row["sample_index"],
            "vector_npz": row["vector_npz"],
        }
        for idx, value in enumerate(vals, start=1):
            item[f"block2_sigma_top{idx}"] = float(value)
        rows.append(item)
    return pd.DataFrame(rows)


def check_provenance(attack: pd.DataFrame, svd: pd.DataFrame) -> dict[str, object]:
    attack_text = "\n".join(attack[["dataset_id", "delta_npz", "checkpoint"]].astype(str).agg(" ".join, axis=1).tolist())
    svd_text = "\n".join(svd[["dataset_id", "vector_npz", "checkpoint"]].astype(str).agg(" ".join, axis=1).tolist())
    return {
        "attack_rows": int(len(attack)),
        "attack_methods": sorted(attack["method"].unique(), key=method_key),
        "attack_datasets": int(attack["dataset_id"].nunique()),
        "attack_steps": sorted(int(v) for v in attack["attack_steps"].dropna().unique()),
        "svd_rows": int(len(svd)),
        "svd_methods": sorted(svd["method"].unique(), key=method_key),
        "svd_datasets": int(svd["dataset_id"].nunique()),
        "svd_samples_per_model": {k: int(v) for k, v in svd.groupby("method").size().to_dict().items()},
        "old_lossdrop50_token_found": "lossdrop50_selected_20260607" in attack_text or "lossdrop50_selected_20260607" in svd_text,
        "smoke_token_found": "smoke" in attack_text.lower() or "smoke" in svd_text.lower(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    attack = pd.read_csv(ROBUST / "data/robustness_attack_52datasets_samples.csv")
    svd = pd.read_csv(ROBUST / "data/svd_jacobian_metrics.csv")

    provenance = check_provenance(attack, svd)
    if provenance["old_lossdrop50_token_found"]:
        raise RuntimeError("obsolete lossdrop50 token appeared in final robustness tables")
    if provenance["attack_steps"] != [50]:
        raise RuntimeError(f"expected attack_steps [50], got {provenance['attack_steps']}")

    attack_dataset = add_mean_std(
        attack.groupby(["method", "method_display", "split", "dataset_id"], sort=False),
        ATTACK_METRICS,
    )
    attack_dataset["method_rank"] = attack_dataset["method"].map(method_key)
    attack_dataset = attack_dataset.sort_values(["method_rank", "split", "dataset_id"]).drop(columns=["method_rank"])
    attack_dataset.to_csv(OUT / "attack50_52dataset_7model_mean_std.csv", index=False)

    attack_split = add_mean_std(
        attack.groupby(["method", "method_display", "split"], sort=False),
        ATTACK_METRICS,
    )
    attack_split["dataset_count"] = attack_split.apply(
        lambda r: int(
            attack[
                (attack["method"] == r["method"])
                & (attack["method_display"] == r["method_display"])
                & (attack["split"] == r["split"])
            ]["dataset_id"].nunique()
        ),
        axis=1,
    )
    attack_split["method_rank"] = attack_split["method"].map(method_key)
    attack_split = attack_split.sort_values(["method_rank", "split"]).drop(columns=["method_rank"])
    attack_split.to_csv(OUT / "attack50_by_model_split_mean_std.csv", index=False)

    svd_summary = add_mean_std(svd.groupby(["method", "method_display"], sort=False), SVD_METRICS)
    svd_summary["dataset_count"] = svd_summary.apply(
        lambda r: int(svd[(svd["method"] == r["method"]) & (svd["method_display"] == r["method_display"])]["dataset_id"].nunique()),
        axis=1,
    )
    svd_summary["method_rank"] = svd_summary["method"].map(method_key)
    svd_summary = svd_summary.sort_values("method_rank").drop(columns=["method_rank"])
    svd_summary.to_csv(OUT / "svd_jacobian_25sample_7model_mean_std.csv", index=False)

    svd_split = add_mean_std(svd.groupby(["method", "method_display", "split"], sort=False), SVD_METRICS)
    svd_split["method_rank"] = svd_split["method"].map(method_key)
    svd_split = svd_split.sort_values(["method_rank", "split"]).drop(columns=["method_rank"])
    svd_split.to_csv(OUT / "svd_jacobian_25sample_by_model_split_mean_std.csv", index=False)

    top_sv_raw = parse_top_singular_values(svd)
    top_sv_raw.to_csv(OUT / "svd_block2_top10_singular_values_raw.csv", index=False)
    top_cols = [c for c in top_sv_raw.columns if c.startswith("block2_sigma_top")]
    top_sv_summary = add_mean_std(top_sv_raw.groupby(["method", "method_display"], sort=False), top_cols)
    top_sv_summary["method_rank"] = top_sv_summary["method"].map(method_key)
    top_sv_summary = top_sv_summary.sort_values("method_rank").drop(columns=["method_rank"])
    top_sv_summary.to_csv(OUT / "svd_block2_top10_singular_values_by_model_mean_std.csv", index=False)

    vector_manifest = svd[
        [
            "method",
            "method_display",
            "dataset_id",
            "split",
            "sample_index",
            "vector_npz",
            "jt_error_l2_norm",
            "sigma_input_right",
            "block2_sigma1",
            "topk_subspace_cos_jt_error",
            "topk_subspace_cos_attack_delta",
            "cos_singular_jt_error",
            "cos_singular_attack_delta",
            "cos_jt_error_attack_delta",
            "angle_singular_jt_error_deg",
            "angle_singular_attack_delta_deg",
            "angle_jt_error_attack_delta_deg",
        ]
    ].copy()
    vector_manifest.to_csv(OUT / "svd_jacobian_vector_manifest_with_metrics.csv", index=False)

    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    attack_focus = attack_split[attack_split["split"].eq("generalization")][
        [
            "method_display",
            "sample_count",
            "dataset_count",
            "clean_loss_mean",
            "clean_loss_std",
            "adv_loss_mean",
            "adv_loss_std",
            "loss_increase_mean",
            "loss_increase_std",
            "relative_increase_mean",
            "relative_increase_std",
        ]
    ]
    svd_focus = svd_summary[
        [
            "method_display",
            "sample_count",
            "dataset_count",
            "sigma_input_right_mean",
            "sigma_input_right_std",
            "jt_error_l2_norm_mean",
            "jt_error_l2_norm_std",
            "topk_subspace_cos_jt_error_mean",
            "topk_subspace_cos_jt_error_std",
            "topk_subspace_cos_attack_delta_mean",
            "topk_subspace_cos_attack_delta_std",
            "angle_jt_error_attack_delta_deg_mean",
            "angle_jt_error_attack_delta_deg_std",
        ]
    ]
    lines = [
        "# Darcy Final Robustness Mean/Std Tables",
        "",
        "This report records explicit mean/std tables derived from the final attack50 and SVD/Jacobian raw outputs.",
        "",
        "## Provenance",
        "",
        f"- Attack rows: `{provenance['attack_rows']}` = 7 models x 52 datasets x 50 samples.",
        f"- Attack steps: `{provenance['attack_steps']}`.",
        f"- SVD/Jacobian rows: `{provenance['svd_rows']}` = 7 models x 25 fixed samples.",
        f"- Old lossdrop50 token found: `{provenance['old_lossdrop50_token_found']}`.",
        f"- Smoke token found: `{provenance['smoke_token_found']}`.",
        "",
        "## Attack50 Generalization Mean/Std",
        "",
        md_table(attack_focus),
        "",
        "## SVD/Jacobian Mean/Std",
        "",
        md_table(svd_focus),
        "",
        "## Files",
        "",
        f"- `{rel(OUT / 'attack50_52dataset_7model_mean_std.csv')}`",
        f"- `{rel(OUT / 'attack50_by_model_split_mean_std.csv')}`",
        f"- `{rel(OUT / 'svd_jacobian_25sample_7model_mean_std.csv')}`",
        f"- `{rel(OUT / 'svd_jacobian_25sample_by_model_split_mean_std.csv')}`",
        f"- `{rel(OUT / 'svd_block2_top10_singular_values_raw.csv')}`",
        f"- `{rel(OUT / 'svd_block2_top10_singular_values_by_model_mean_std.csv')}`",
        f"- `{rel(OUT / 'svd_jacobian_vector_manifest_with_metrics.csv')}`",
        f"- `{rel(OUT / 'provenance.json')}`",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    DOC.write_text(text, encoding="utf-8")
    print(REPORT.relative_to(PROJECT))
    print((OUT / "attack50_52dataset_7model_mean_std.csv").relative_to(PROJECT))
    print((OUT / "svd_jacobian_25sample_7model_mean_std.csv").relative_to(PROJECT))


if __name__ == "__main__":
    main()
