#!/usr/bin/env python3
"""Build a compact inventory of Burgers metric coverage.

This script does not rerun experiments. It records where the scalar metrics,
significance tests, correlations, vector cosine/angle summaries, and top-k
subspace/SVD summaries live in the final Burgers audit bundle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
DATA_ROOT = AUDIT_ROOT / "data"
RANKED = DATA_ROOT / "ranked_metric_tables_20260614"
RECOVERED = DATA_ROOT / "recovered_prior/six_model_latest_wideparam_summary_20260613"
OUT_DATA = DATA_ROOT / "metric_coverage_inventory_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_metric_coverage_inventory_20260614.md"
DOC_REPORT = REPO / "docs/burgers_metric_coverage_inventory_20260614.md"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_shape(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        df = pd.read_csv(path)
        return f"{df.shape[0]}x{df.shape[1]}"
    except Exception as exc:  # noqa: BLE001
        return f"unreadable:{type(exc).__name__}"


def fmt(x: object) -> str:
    try:
        f = float(x)
    except Exception:
        return str(x)
    if abs(f) >= 1e4 or (abs(f) < 1e-4 and f != 0):
        return f"{f:.3e}"
    return f"{f:.6g}"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    lines = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    artifacts = {
        "clean_52": DATA_ROOT / "clean_52dataset_six_models_selected_worktime.csv",
        "attack_strict_52": DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_long.csv",
        "robustness_25": DATA_ROOT / "robustness_25sample_six_models_selected_worktime.csv",
        "metric_model_summary": RANKED / "metric_model_summary_ranked.csv",
        "best_summary": RANKED / "metric_best_summary_ranked.csv",
        "best_vs_other_tests": RANKED / "metric_best_vs_other_significance_tests.csv",
        "six_model_best_summary": RANKED / "metric_best_summary_six_model_evidence_ranked.csv",
        "six_model_tests": RANKED / "metric_best_vs_other_six_model_evidence_significance_tests.csv",
        "pairwise_correlations": RANKED / "metric_pairwise_correlations_sorted.csv",
        "correlations_with_attack": RANKED / "correlations_with_attack_sorted.csv",
        "top20_svd_ranked": RANKED / "svd_error_top20_ranked_long.csv",
        "topk_svd_ranked": RANKED / "svd_error_topk_ranked_long.csv",
        "top100_svd_ranked": RANKED / "svd_error_top100_supplement_ranked_long.csv",
        "top100_subspace_ranked": RANKED / "model_solver_subspace_top100_supplement_ranked_long.csv",
        "direction_angles": RECOVERED / "direction_angle_similarity_summary_corrected_jerrT_25sample.csv",
        "subspace_summary": RECOVERED / "subspace_similarity_summary_corrected_jerrT_25sample.csv",
        "top100_svd_tests": DATA_ROOT / "random_top100_svd_supplement_20260614/loss3_vs_other_top100_svd_tests.csv",
        "top50_top100_subspace_tests": DATA_ROOT / "random_top100_svd_supplement_20260614/loss3_vs_other_top50_top100_subspace_tests.csv",
        "first_vs_second_key": DATA_ROOT / "first_vs_second_significance_20260614/first_vs_second_key_metric_significance_compact.csv",
    }

    artifact_inventory = pd.DataFrame(
        [
            {
                "artifact": name,
                "path": path.relative_to(REPO).as_posix(),
                "exists": path.exists(),
                "shape": safe_shape(path),
            }
            for name, path in artifacts.items()
        ]
    )

    coverage = pd.DataFrame(
        [
            {
                "category": "52x6 clean RMSE/RelativeL2/MSE",
                "covered": True,
                "primary_tables": "clean_52; clean_52dataset_metric_long_ranked; metric_model_summary",
                "best_and_runner_up": True,
                "significance": True,
                "correlations": "not central; scalar correlations are in metric_pairwise_correlations_sorted",
                "vector_similarity_or_angle": False,
                "notes": "Clean all-52 and gen-50 first-vs-second significance is in first_vs_second_key.",
            },
            {
                "category": "52x6 strict attack final loss / loss increase / delta",
                "covered": True,
                "primary_tables": "attack_strict_52; attack_52dataset_metric_long_ranked; first_vs_second_key",
                "best_and_runner_up": True,
                "significance": True,
                "correlations": "25-sample attack correlations in correlations_with_attack_sorted; random full10200 delta/loss correlations are recovered separately",
                "vector_similarity_or_angle": "attack delta angle/cosine in direction_angles",
                "notes": "Attack loss increase and final loss: loss3 wins every 52/52 row; delta RMS is process/constraint, not quality evidence.",
            },
            {
                "category": "25-sample robustness scalar norms",
                "covered": True,
                "primary_tables": "robustness_25; robustness_25sample_metric_long_ranked; metric_model_summary",
                "best_and_runner_up": True,
                "significance": True,
                "correlations": "metric_pairwise_correlations_sorted; correlations_with_attack_sorted",
                "vector_similarity_or_angle": "some scalar rows are derived from vector comparisons; raw vector angles summarized separately",
                "notes": "Includes clean residual, Frobenius, spectral norm, J_error^T error norm/RMS, J_error delta diagnostics.",
            },
            {
                "category": "top-k singular values / SVD spectrum",
                "covered": True,
                "primary_tables": "top20_svd_ranked; topk_svd_ranked; top100_svd_ranked; top100_svd_tests",
                "best_and_runner_up": True,
                "significance": True,
                "correlations": "correlations_with_attack_sorted and svd_attack_correlations in recovered historical bundle",
                "vector_similarity_or_angle": "top singular vector cosine with attack delta in direction_angles",
                "notes": "Top1 mean-best is loss3 but can be nonsignificant in all-25; top5+ and top100 supplement are significant.",
            },
            {
                "category": "top-k model/solver subspace similarity",
                "covered": True,
                "primary_tables": "subspace_summary; top100_subspace_ranked; top50_top100_subspace_tests",
                "best_and_runner_up": True,
                "significance": "partial",
                "correlations": "model_solver_subspace_similarity_correlations and correlations_with_attack_sorted",
                "vector_similarity_or_angle": True,
                "notes": "Higher is better. Some top50/top100 rows are random_solver_y best; this is diagnostic, not clean/attack quality proof.",
            },
            {
                "category": "vector direction cosine/angle",
                "covered": True,
                "primary_tables": "direction_angles; old4/random direction angle summaries",
                "best_and_runner_up": "not treated as quality best",
                "significance": "not generally used as best-model evidence",
                "correlations": "random_affine_direction_correlations and metric correlations",
                "vector_similarity_or_angle": True,
                "notes": "Includes attack delta vs top error SV, attack delta vs outward, SVD outward; mean/median cosine and mean/median angle in degrees.",
            },
            {
                "category": "scalar-scalar correlations",
                "covered": True,
                "primary_tables": "metric_pairwise_correlations_sorted; correlations_with_attack_sorted",
                "best_and_runner_up": False,
                "significance": "correlation p/q present in ranked correlation tables",
                "correlations": True,
                "vector_similarity_or_angle": False,
                "notes": "Includes Pearson/Spearman and BH/FDR q for ranked correlation tables.",
            },
        ]
    )

    model_summary = read_csv(artifacts["metric_model_summary"])
    best_summary = read_csv(artifacts["best_summary"])
    tests = read_csv(artifacts["best_vs_other_tests"])
    corr_pair = read_csv(artifacts["pairwise_correlations"])
    corr_attack = read_csv(artifacts["correlations_with_attack"])
    direction = read_csv(artifacts["direction_angles"])
    subspace = read_csv(artifacts["subspace_summary"])
    first_vs_second = read_csv(artifacts["first_vs_second_key"])

    summary_counts = pd.DataFrame(
        [
            {"item": "metric_model_summary_rows", "value": len(model_summary)},
            {"item": "metric_best_summary_rows", "value": len(best_summary)},
            {"item": "best_vs_other_significance_rows", "value": len(tests)},
            {"item": "pairwise_scalar_correlation_rows", "value": len(corr_pair)},
            {"item": "correlations_with_attack_rows", "value": len(corr_attack)},
            {"item": "direction_angle_model_rows", "value": len(direction)},
            {"item": "subspace_summary_model_rows", "value": len(subspace)},
            {"item": "first_vs_second_key_rows", "value": len(first_vs_second)},
            {
                "item": "six_model_evidence_best_counts",
                "value": json.dumps(
                    read_csv(artifacts["six_model_best_summary"])["best_model"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            },
        ]
    )

    top_attack_corr = corr_attack.head(12).copy()
    top_pair_corr = corr_pair.head(12).copy()
    direction_compact = direction[
        [
            "model",
            "n",
            "delta_top_error_sv_abs_cos_mean",
            "delta_top_error_sv_abs_cos_angle_deg_mean",
            "attack_delta_outward_abs_cos_mean",
            "attack_delta_outward_abs_cos_angle_deg_mean",
            "svd_outward_abs_cos_mean",
            "svd_outward_abs_cos_angle_deg_mean",
        ]
    ].copy()
    subspace_compact = subspace.copy()

    artifact_inventory.to_csv(OUT_DATA / "artifact_inventory.csv", index=False)
    coverage.to_csv(OUT_DATA / "metric_coverage_summary.csv", index=False)
    summary_counts.to_csv(OUT_DATA / "metric_coverage_counts.csv", index=False)
    top_attack_corr.to_csv(OUT_DATA / "top_correlations_with_attack.csv", index=False)
    top_pair_corr.to_csv(OUT_DATA / "top_scalar_pairwise_correlations.csv", index=False)
    direction_compact.to_csv(OUT_DATA / "direction_angle_compact.csv", index=False)
    subspace_compact.to_csv(OUT_DATA / "subspace_similarity_compact.csv", index=False)

    report = f"""# Burgers Metric Coverage, Correlation, And Similarity Inventory, 20260614

Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

This report answers whether the requested Burgers clean/attack/robustness/SVD/Jacobian scalar metrics, scalar significance tests, scalar correlations, and vector cosine/angle/subspace similarities are present in the final audit bundle.

## Short Answer

Yes: the requested quantities are present locally, but they are split across several tables. Scalar quantities have best-model, runner-up, mean/std/median, paired t/Wilcoxon/FDR significance where paired model coverage is available. Scalar correlations have Pearson/Spearman and q-values in the ranked correlation tables. Vector quantities are summarized as cosine similarity and angle in degrees, and top-k subspace similarities are ranked; these vector quantities are diagnostic and are not always valid as a single 'best model quality' claim.

## Coverage Summary

{md_table(coverage)}

## Counts

{md_table(summary_counts)}

## Direction / Angle Summary

{md_table(direction_compact)}

## Model-Solver Subspace Summary

{md_table(subspace_compact)}

## Top Correlations With Attack

{md_table(top_attack_corr, max_rows=12)}

## Top Scalar Pairwise Correlations

{md_table(top_pair_corr, max_rows=12)}

## Output CSVs

- `data/metric_coverage_inventory_20260614/artifact_inventory.csv`
- `data/metric_coverage_inventory_20260614/metric_coverage_summary.csv`
- `data/metric_coverage_inventory_20260614/metric_coverage_counts.csv`
- `data/metric_coverage_inventory_20260614/top_correlations_with_attack.csv`
- `data/metric_coverage_inventory_20260614/top_scalar_pairwise_correlations.csv`
- `data/metric_coverage_inventory_20260614/direction_angle_compact.csv`
- `data/metric_coverage_inventory_20260614/subspace_similarity_compact.csv`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    DOC_REPORT.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_data": OUT_DATA.relative_to(REPO).as_posix(),
                "report": OUT_REPORT.relative_to(REPO).as_posix(),
                "doc": DOC_REPORT.relative_to(REPO).as_posix(),
                "coverage_rows": len(coverage),
                "artifact_rows": len(artifact_inventory),
                "missing_artifacts": int((~artifact_inventory["exists"]).sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
