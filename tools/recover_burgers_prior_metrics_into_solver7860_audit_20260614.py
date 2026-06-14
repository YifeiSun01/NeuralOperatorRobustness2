#!/usr/bin/env python3
"""Recover prior Burgers metric artifacts into the solver7860 audit output.

This script does not rerun attacks or SVD. It copies already-produced local/R2
artifacts into the final audit folder and builds a few join tables that make the
coverage explicit.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATE = "20260614"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def copy_tree(src: Path, dst: Path, *, skip_hidden: bool = False) -> tuple[int, int]:
    if not src.exists():
        raise FileNotFoundError(src)
    files = 0
    bytes_total = 0
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        if skip_hidden and any(part.startswith(".") for part in path.relative_to(src).parts):
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == path.stat().st_size:
            files += 1
            bytes_total += target.stat().st_size
            continue
        shutil.copy2(path, target)
        files += 1
        bytes_total += target.stat().st_size
    return files, bytes_total


def split_from_dataset_index(index: int) -> str:
    if index == 0:
        return "train"
    if index == 1:
        return "test"
    return "generalization"


def build_six_model_attack52(root: Path, audit_data: Path) -> dict[str, object]:
    old4 = pd.read_csv(
        root
        / "forensics"
        / "burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608"
        / "summary_by_model_dataset.csv"
    )
    current = pd.read_csv(audit_data / "attack_52dataset_six_models_selected_worktime_long.csv")
    clean = pd.read_csv(audit_data / "clean_52dataset_six_models_selected_worktime.csv")

    model_map = {
        "baseline": "baseline",
        "loss1_epoch8000": "loss1",
        "loss2_epoch2000": "loss2",
        "loss3_epoch1500": "loss3",
    }
    old_rows = old4.copy()
    old_rows["model"] = old_rows["model"].map(model_map)
    old_rows["split"] = old_rows["dataset_index"].astype(int).map(split_from_dataset_index)
    old_rows["attack_loss_increase_mean"] = old_rows["final_loss_mean"] - old_rows["initial_loss_mean"]
    old_rows = old_rows[
        [
            "model",
            "dataset_index",
            "split",
            "dataset_id",
            "sample_count",
            "initial_loss_mean",
            "final_loss_mean",
            "attack_loss_increase_mean",
            "final_delta_rms_mean",
        ]
    ].copy()
    old_rows["source"] = "r2_first_master_old4_52dataset_20step"

    random_rows = current[current["model"].isin(["random_clean_y", "random_solver_y"])].copy()
    random_rows = random_rows[
        [
            "model",
            "dataset_index",
            "split",
            "dataset_id",
            "sample_count",
            "initial_loss_mean",
            "final_loss_mean",
            "attack_loss_increase_mean",
            "final_delta_rms_mean",
            "source",
        ]
    ]
    random_rows["source"] = "solver7860_clean8000_random_52dataset_current"

    long = pd.concat([old_rows, random_rows], ignore_index=True)
    long["dataset_index"] = long["dataset_index"].astype(int)
    long = long.sort_values(["dataset_index", "model"]).reset_index(drop=True)

    clean_key = clean[
        [
            "dataset_order",
            "split",
            "dataset_id",
            "old4_dataset_id",
            "random_dataset_id",
            "family",
            "display_label",
            "description",
        ]
    ].copy()
    clean_key["dataset_index"] = clean_key["dataset_order"].astype(int) - 1
    long = long.merge(
        clean_key.drop(columns=["split"]),
        on="dataset_index",
        how="left",
        suffixes=("", "_clean52"),
    )
    long = long.rename(columns={"dataset_id": "attack_dataset_id", "dataset_id_clean52": "clean_dataset_id"})

    wide_index_cols = [
        "dataset_index",
        "split",
        "clean_dataset_id",
        "old4_dataset_id",
        "random_dataset_id",
        "family",
        "display_label",
        "description",
    ]
    wide_parts = []
    for model, model_df in long.groupby("model", sort=True):
        cols = wide_index_cols + [
            "attack_dataset_id",
            "sample_count",
            "initial_loss_mean",
            "final_loss_mean",
            "attack_loss_increase_mean",
            "final_delta_rms_mean",
            "source",
        ]
        part = model_df[cols].copy()
        rename = {
            c: f"{model}_{c}"
            for c in [
                "attack_dataset_id",
                "sample_count",
                "initial_loss_mean",
                "final_loss_mean",
                "attack_loss_increase_mean",
                "final_delta_rms_mean",
                "source",
            ]
        }
        part = part.rename(columns=rename)
        wide_parts.append(part)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on=wide_index_cols, how="outer")
    wide = wide.sort_values("dataset_index").reset_index(drop=True)

    long_path = audit_data / "attack_52dataset_six_models_recovered_full_long.csv"
    wide_path = audit_data / "attack_52dataset_six_models_recovered_full_wide.csv"
    long.to_csv(long_path, index=False)
    wide.to_csv(wide_path, index=False)

    return {
        "attack52_recovered_long": str(long_path.relative_to(root)),
        "attack52_recovered_wide": str(wide_path.relative_to(root)),
        "attack52_recovered_rows": int(len(long)),
        "attack52_recovered_models": sorted(long["model"].unique().tolist()),
        "attack52_recovered_dataset_indices": int(long["dataset_index"].nunique()),
    }


def build_six_model_singular_top20(root: Path, audit_data: Path) -> dict[str, object]:
    old_top100 = pd.read_csv(
        root
        / "forensics"
        / "burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611"
        / "singular_values_top100_long.csv"
    )
    random_top20 = pd.read_csv(audit_data / "jacobian_svd__top_singular_values_long.csv")

    old = old_top100[
        (old_top100["model_key"].isin(["solver", "baseline", "loss1", "loss2", "loss3"]))
        & (old_top100["rank"].astype(int) <= 20)
    ].copy()
    old = old.rename(columns={"model_key": "model"})
    old["source"] = "r2_old4_svd_top100_recovered"
    old = old[
        [
            "sample_id",
            "source_split",
            "dataset_id",
            "model",
            "jacobian_kind",
            "rank",
            "singular_value",
            "source",
        ]
    ]

    rnd = random_top20.copy().rename(columns={"model_name": "model"})
    rnd["source"] = "solver7860_random_svd_top20_current"
    rnd = rnd[
        [
            "sample_id",
            "source_split",
            "dataset_id",
            "model",
            "jacobian_kind",
            "rank",
            "singular_value",
            "source",
        ]
    ]

    combined = pd.concat([old, rnd], ignore_index=True)
    combined["rank"] = combined["rank"].astype(int)
    combined = combined.drop_duplicates(
        ["sample_id", "source_split", "dataset_id", "model", "jacobian_kind", "rank"],
        keep="last",
    )
    combined = combined.sort_values(["sample_id", "model", "jacobian_kind", "rank"]).reset_index(drop=True)
    out = audit_data / "singular_values_top20_six_models_recovered_long.csv"
    combined.to_csv(out, index=False)

    coverage = (
        combined.groupby(["model", "jacobian_kind"])
        .agg(rows=("singular_value", "size"), samples=("sample_id", "nunique"), max_rank=("rank", "max"))
        .reset_index()
    )
    coverage_path = audit_data / "singular_values_top20_six_models_recovered_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    return {
        "singular_values_top20_six_models": str(out.relative_to(root)),
        "singular_values_top20_coverage": str(coverage_path.relative_to(root)),
        "singular_values_top20_rows": int(len(combined)),
    }


def bh_fdr(p_values: list[float]) -> list[float]:
    p = np.asarray([np.nan if v is None else v for v in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q.tolist()
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[finite])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        original_rank = m - rank + 1
        val = min(prev, p[i] * m / original_rank)
        q[i] = val
        prev = val
    return q.tolist()


def paired_test_rows(values: pd.DataFrame, metrics: list[str], scope: str, index_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    models = sorted(m for m in values["model"].dropna().unique().tolist() if m != "loss3")
    loss3 = values[values["model"] == "loss3"]
    for metric in metrics:
        if metric not in values.columns:
            continue
        base = loss3[[index_col, metric]].rename(columns={metric: "loss3_value"})
        for model in models:
            other = values[values["model"] == model][[index_col, metric]].rename(columns={metric: "other_value"})
            merged = base.merge(other, on=index_col, how="inner").dropna(subset=["loss3_value", "other_value"])
            n = len(merged)
            if n < 3:
                continue
            diff = merged["loss3_value"].to_numpy(dtype=float) - merged["other_value"].to_numpy(dtype=float)
            t_stat, p_two = stats.ttest_rel(merged["loss3_value"], merged["other_value"], nan_policy="omit")
            if not math.isfinite(float(t_stat)):
                p_less = math.nan
                p_greater = math.nan
            else:
                p_less = p_two / 2 if t_stat < 0 else 1 - p_two / 2
                p_greater = p_two / 2 if t_stat > 0 else 1 - p_two / 2
            try:
                _wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            except Exception:
                wilcoxon_p = math.nan
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "comparison": f"loss3_vs_{model}",
                    "other_model": model,
                    "n_pairs": int(n),
                    "loss3_mean": float(merged["loss3_value"].mean()),
                    "loss3_std": float(merged["loss3_value"].std(ddof=1)),
                    "other_mean": float(merged["other_value"].mean()),
                    "other_std": float(merged["other_value"].std(ddof=1)),
                    "mean_diff_loss3_minus_other": float(diff.mean()),
                    "std_diff": float(diff.std(ddof=1)),
                    "cohen_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else math.nan,
                    "paired_t_stat": float(t_stat),
                    "paired_t_p_two_sided": float(p_two),
                    "paired_t_p_loss3_less": float(p_less),
                    "paired_t_p_loss3_greater": float(p_greater),
                    "wilcoxon_p_two_sided": float(wilcoxon_p),
                }
            )
    return rows


def build_paired_tests(audit_data: Path) -> dict[str, object]:
    clean = pd.read_csv(audit_data / "clean_52dataset_six_models_selected_worktime.csv")
    gen = clean[clean["split"] == "generalization"].copy()
    long_rows = []
    for _, row in gen.iterrows():
        for model in ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]:
            long_rows.append(
                {
                    "dataset_order": int(row["dataset_order"]),
                    "model": model,
                    "rmse_mean": row[f"{model}_rmse_mean"],
                    "relative_l2_mean": row[f"{model}_relative_l2_mean"],
                    "mse_mean": row[f"{model}_mse_mean"],
                }
            )
    clean_long = pd.DataFrame(long_rows)
    rows = paired_test_rows(
        clean_long,
        ["rmse_mean", "relative_l2_mean", "mse_mean"],
        "clean_generalization_50dataset",
        "dataset_order",
    )

    robust = pd.read_csv(audit_data / "robustness_25sample_six_models_selected_worktime.csv")
    robust_metrics = [
        "attack_initial_mse",
        "attack_final_mse",
        "attack_loss_increase",
        "attack_final_delta_rms",
        "clean_residual_mse_recomputed",
        "clean_residual_norm_l2",
        "error_spectral_norm",
        "error_fro_norm_comparable",
        "error_effective_rank",
        "bias_gradient_norm",
        "bias_gradient_rms",
        "j_error_transpose_error_l2",
        "j_error_transpose_error_rms",
        "model_solver_top1_right_abs_cos",
        "model_solver_top1_left_abs_cos",
        "model_solver_top5_right_subspace_mean_cos",
        "model_solver_top5_left_subspace_mean_cos",
        "model_solver_top10_right_subspace_mean_cos",
        "model_solver_top10_left_subspace_mean_cos",
        "model_solver_top20_right_subspace_mean_cos",
        "model_solver_top20_left_subspace_mean_cos",
        "delta_top_error_sv_abs_cos",
        "attack_delta_svd_abs_cos",
    ]
    rows.extend(paired_test_rows(robust, robust_metrics, "robustness_25sample", "sample_id"))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_q_two_sided_bh_fdr"] = bh_fdr(out["paired_t_p_two_sided"].tolist())
        out["wilcoxon_q_two_sided_bh_fdr"] = bh_fdr(out["wilcoxon_p_two_sided"].tolist())
    path = audit_data / "paired_tests_loss3_vs_other_models.csv"
    out.to_csv(path, index=False)
    return {"paired_tests_loss3_vs_other_models": str(path), "paired_tests_rows": int(len(out))}


def write_coverage_audit(root: Path, audit_data: Path, copy_stats: dict[str, dict[str, int]]) -> dict[str, object]:
    rows = [
        {
            "item": "clean_52dataset_6models_rmse_relative_l2_mse",
            "status": "complete",
            "coverage": "52 datasets x 6 models; RMSE, Relative L2, MSE",
            "path": "data/clean_52dataset_six_models_selected_worktime.csv",
            "note": "Includes 50 generalization datasets plus train/test.",
        },
        {
            "item": "attack_52dataset_6models_summary",
            "status": "recovered_and_joined",
            "coverage": "52 datasets x 6 models",
            "path": "data/attack_52dataset_six_models_recovered_full_long.csv",
            "note": "Old4 loss1/loss2/loss3/baseline recovered from R2 first-master 20-step run; random clean/solver taken from solver7860/clean8000 current summary.",
        },
        {
            "item": "attack_52dataset_old4_raw_npz",
            "status": "recovered_from_r2",
            "coverage": "baseline/loss1/loss2/loss3 52-dataset P2Q2 raw losses, deltas, FFT summaries",
            "path": "data/recovered_prior/first_master_full_p2q2_52datasets_4models_finalonly_20step/",
            "note": "R2 source had 21 objects.",
        },
        {
            "item": "robustness_25sample_6models_core_metrics",
            "status": "complete_for_core_columns",
            "coverage": "25 samples x 6 models for spectral/Fro/effective-rank, bias-gradient, residual, top1/top5/top10/top20 subspace, J^T error metrics",
            "path": "data/robustness_25sample_six_models_selected_worktime.csv",
            "note": "Some attack-delta columns have 24 random-model pairs because one random sample attack trace is missing in the existing source.",
        },
        {
            "item": "old4_svd25_raw_npz_and_top100_tables",
            "status": "recovered_from_r2",
            "coverage": "25 samples; solver plus baseline/loss1/loss2/loss3 model/error SVD, top100 singular values, top-k pair similarities, correlations",
            "path": "data/historical_svd_attack25_reuse3/",
            "note": "R2 source had 501 objects and includes raw NPZ payloads.",
        },
        {
            "item": "random_solver7860_clean8000_raw_svd_and_attack",
            "status": "copied_from_local_completed_run",
            "coverage": "random_clean_y/random_solver_y 25-sample Jacobian/SVD and 52-dataset P2Q2 attack raw outputs",
            "path": "data/recovered_prior/random_solver7860_clean8000_full_suite_20260614/",
            "note": "This was already in the organized release; now also copied into final audit data.",
        },
        {
            "item": "singular_values_top20_6models",
            "status": "recovered_and_joined",
            "coverage": "25 samples; solver plus six models for model/error top20 singular values",
            "path": "data/singular_values_top20_six_models_recovered_long.csv",
            "note": "Old4 source has top100; random source has top20, so the six-model common-rank table is top20.",
        },
        {
            "item": "singular_values_top50_top100_6models",
            "status": "partial_not_found_for_random_models",
            "coverage": "old4 has top100; random_clean_y/random_solver_y have top20 only",
            "path": "data/historical_svd_attack25_reuse3/singular_values_top100_long.csv",
            "note": "No existing R2/local random top50/top100 SVD payload was found in the checked solver7860/clean8000 or latest-wideparam summaries.",
        },
        {
            "item": "biased_local_direction_full_old4",
            "status": "complete_for_old4",
            "coverage": "baseline/loss1/loss2/loss3 25 samples",
            "path": "data/biased_local_direction/",
            "note": "Full affine/outward/SVD local-gain direction suite is old4 only.",
        },
        {
            "item": "biased_jerror_direction_random_models",
            "status": "partial_for_random_models",
            "coverage": "random_clean_y/random_solver_y J^T error, SVD/outward, attack-delta direction summaries",
            "path": "data/recovered_prior/six_model_latest_wideparam_summary_20260613/random_models_jerror_transpose_error_25sample.csv",
            "note": "Random has corrected J^T-error direction rows, but not the full old4 affine local-gain sweep.",
        },
        {
            "item": "metric_correlations_pairwise_rank_similarity",
            "status": "recovered_and_copied",
            "coverage": "six-model corrected JerrT metric correlations, pairwise correlations, model/sample rank similarity summaries",
            "path": "data/recovered_prior/six_model_latest_wideparam_summary_20260613/",
            "note": "R2 source had 42 files; copied into final audit.",
        },
        {
            "item": "loss3_vs_other_paired_tests",
            "status": "computed_from_existing_tables",
            "coverage": "clean 50-dataset and robustness 25-sample paired t-test/Wilcoxon rows",
            "path": "data/paired_tests_loss3_vs_other_models.csv",
            "note": "Derived only from existing metrics; no model rerun.",
        },
    ]
    audit = pd.DataFrame(rows)
    csv_path = audit_data / "missing_metric_coverage_audit.csv"
    json_path = audit_data / "missing_metric_coverage_audit.json"
    audit.to_csv(csv_path, index=False)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "copy_stats": copy_stats,
        "items": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    readme = audit_data / "RECOVERED_PRIOR_METRICS_README.md"
    readme.write_text(
        "\n".join(
            [
                "# Recovered Prior Burgers Metrics",
                "",
                "This audit update copied previously completed Burgers results from local/R2-backed source roots into the final solver7860/clean8000 audit output. No model training, attack, or SVD was rerun.",
                "",
                "Key recovered items:",
                "",
                "- Six-model 52-dataset attack summary: `attack_52dataset_six_models_recovered_full_long.csv` and `_wide.csv`.",
                "- Old four-model 52-dataset P2Q2 raw attack payloads: `recovered_prior/first_master_full_p2q2_52datasets_4models_finalonly_20step/`.",
                "- Full old four-model SVD25 raw payload and top100 tables: `historical_svd_attack25_reuse3/`.",
                "- Six-model latest/corrected metric/correlation/rank-similarity tables: `recovered_prior/six_model_latest_wideparam_summary_20260613/`.",
                "- Random solver7860/clean8000 raw attack/SVD suite: `recovered_prior/random_solver7860_clean8000_full_suite_20260614/`.",
                "- Paired loss3-vs-other statistical test table: `paired_tests_loss3_vs_other_models.csv`.",
                "",
                "Remaining genuine gap: old4 has top100 SVD, but the random solver7860/clean8000 SVD source has top20 only; therefore the common six-model singular-value table is top20.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "coverage_audit_csv": str(csv_path.relative_to(root)),
        "coverage_audit_json": str(json_path.relative_to(root)),
        "recovered_prior_readme": str(readme.relative_to(root)),
    }


def main() -> None:
    root = repo_root()
    audit = root / "outputs" / "burgers_timematched_solver7860_clean8000_audit_20260614"
    audit_data = audit / "data"
    recovered = audit_data / "recovered_prior"
    recovered.mkdir(parents=True, exist_ok=True)

    source_latest = root / "forensics" / "burgers_six_model_latest_wideparam_summary_20260613"
    source_first_master = (
        root / "forensics" / "burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608"
    )
    source_old4_svd = root / "forensics" / "burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611"
    source_random_suite = root / "forensics" / "burgers_random_solver7860_clean8000_full_suite_20260614"

    copy_stats: dict[str, dict[str, int]] = {}
    for label, src, dst, skip_hidden in [
        (
            "six_model_latest_wideparam_summary_20260613",
            source_latest,
            recovered / "six_model_latest_wideparam_summary_20260613",
            False,
        ),
        (
            "first_master_full_p2q2_52datasets_4models_finalonly_20step",
            source_first_master,
            recovered / "first_master_full_p2q2_52datasets_4models_finalonly_20step",
            False,
        ),
        (
            "historical_svd_attack25_reuse3_full",
            source_old4_svd,
            audit_data / "historical_svd_attack25_reuse3",
            False,
        ),
        (
            "random_solver7860_clean8000_full_suite_20260614",
            source_random_suite,
            recovered / "random_solver7860_clean8000_full_suite_20260614",
            False,
        ),
    ]:
        files, bytes_total = copy_tree(src, dst, skip_hidden=skip_hidden)
        copy_stats[label] = {"files": files, "bytes": bytes_total}

    results: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "copy_stats": copy_stats,
    }
    results.update(build_six_model_attack52(root, audit_data))
    results.update(build_six_model_singular_top20(root, audit_data))
    results.update(build_paired_tests(audit_data))
    results.update(write_coverage_audit(root, audit_data, copy_stats))

    done_path = audit_data / "recovered_prior_metrics_done.json"
    done_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = audit / "manifests" / "recovered_prior_metrics_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
