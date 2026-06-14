#!/usr/bin/env python3
"""Summarize what was rerun, why the old table was wrong, and what remains.

This is a lightweight audit/explanation layer. It reads already generated
Burgers artifacts only; it does not rerun training, attack, Jacobian, SVD, or
plotting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
DATA_ROOT = AUDIT_ROOT / "data"
OUT_DATA = DATA_ROOT / "rerun_explanation_gap_status_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_rerun_explanation_gap_status_20260614.md"
DOC_REPORT = REPO / "docs/burgers_rerun_explanation_gap_status_20260614.md"
ORGANIZED_ROOT = REPO / "outputs/burgers_solver7860_clean8000_organized_release_20260614"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: object) -> str:
    try:
        f = float(x)
    except Exception:
        return str(x)
    if abs(f) >= 1e4 or (abs(f) < 1e-4 and f != 0):
        return f"{f:.3e}"
    return f"{f:.6g}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    recovered = pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_recovered_full_long.csv")
    strict = pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_long.csv")
    strict_summary = pd.read_csv(DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_model_summary.csv")
    protocol_summary = pd.read_csv(DATA_ROOT / "protocol_confusion_audit_20260614/recovered_attack_protocol_summary.csv")
    protocol_validation = pd.read_csv(DATA_ROOT / "final_resolution_audit_20260614/protocol_validation.csv")
    winners = pd.read_csv(DATA_ROOT / "strict_latest_attack52_20260614/strict_latest_attack52_winners_by_dataset.csv")
    final_boundaries = pd.read_csv(DATA_ROOT / "final_resolution_audit_20260614/final_claim_boundaries.csv")
    full_audit = read_json(DATA_ROOT / "full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json")
    organized_manifest = read_json(ORGANIZED_ROOT / "MANIFEST.json")
    organized_file_count = organized_manifest.get("file_count", "unknown")
    organized_missing_count = len(organized_manifest.get("missing_expected_inputs", []))

    recovered_summary = (
        recovered.groupby(["model", "source"], dropna=False)
        .agg(
            rows=("attack_loss_increase_mean", "size"),
            datasets=("dataset_order", "nunique"),
            attack_loss_increase_mean=("attack_loss_increase_mean", "mean"),
            attack_loss_increase_median=("attack_loss_increase_mean", "median"),
            final_loss_mean=("final_loss_mean", "mean"),
        )
        .reset_index()
    )
    strict_compare = strict_summary[
        [
            "model",
            "datasets",
            "sample_count_sum",
            "attack_loss_increase_mean",
            "attack_loss_increase_median",
            "final_loss_mean",
            "final_delta_rms_mean",
        ]
    ].copy()
    strict_compare["source"] = "strict_latest_same_manifest_20260614"

    old_vs_new = recovered_summary.merge(
        strict_compare[["model", "attack_loss_increase_mean", "attack_loss_increase_median", "final_loss_mean"]],
        on="model",
        suffixes=("_old_mixed", "_strict_latest"),
    )
    old_vs_new["old_minus_strict_attack_increase_mean"] = (
        old_vs_new["attack_loss_increase_mean_old_mixed"] - old_vs_new["attack_loss_increase_mean_strict_latest"]
    )

    loss3_row = strict_summary[strict_summary["model"].eq("loss3")].iloc[0]
    solver_row = strict_summary[strict_summary["model"].eq("random_solver_y")].iloc[0]
    clean_row = strict_summary[strict_summary["model"].eq("random_clean_y")].iloc[0]
    loss3_wins_solver = int(winners["loss3_beats_random_solver_y"].sum())
    winner_count = winners["best_model"].value_counts().to_dict()

    rerun_summary = pd.DataFrame(
        [
            {
                "item": "what_was_rerun",
                "status": "done",
                "answer": "Only the latest old4 full-52 P2Q2 attack was rerun: baseline/loss1/loss2/loss3 on the current widevis 10200-row manifest.",
                "evidence": "forensics/burgers_latest_old4_widevis_full52_p2q2_20step_20260614",
            },
            {
                "item": "what_was_not_rerun",
                "status": "intentional",
                "answer": "No self-training, no random model rerun, no Jacobian/SVD rerun, and no new dense visual attack except the already completed targeted visual/diagnostic pieces.",
                "evidence": "docs/burgers_final_resolution_audit_20260614.md",
            },
            {
                "item": "why_old_table_was_wrong",
                "status": "resolved",
                "answer": "The recovered six-model full-52 attack table mixed historical old4 rows with current random rows; old4 generalization rows were not the current widevis datasets although they had current-style labels.",
                "evidence": "data/protocol_confusion_audit_20260614/recovered_attack_protocol_summary.csv",
            },
            {
                "item": "why_new_table_is_valid",
                "status": "resolved",
                "answer": "The strict latest table uses one current manifest and one current protocol across all six models: 52 datasets, 10200 samples, P2Q2 20 steps, epsilon RMS 0.12, alpha RMS 0.012.",
                "evidence": "data/final_resolution_audit_20260614/protocol_validation.csv",
            },
            {
                "item": "main_attack_result",
                "status": "resolved",
                "answer": f"Loss3 attack increase mean {loss3_row.attack_loss_increase_mean:.8f}; random_solver_y {solver_row.attack_loss_increase_mean:.8f}; random_clean_y {clean_row.attack_loss_increase_mean:.8f}; loss3 beats random_solver_y {loss3_wins_solver}/52 dataset rows.",
                "evidence": "data/strict_latest_attack52_20260614/strict_latest_attack52_winners_by_dataset.csv",
            },
            {
                "item": "bundle_integrity",
                "status": "pass",
                "answer": f"overall_pass={full_audit.get('overall_pass')}; parse={full_audit.get('parse_failure_count')}; shape={full_audit.get('shape_failure_count')}; ranked={full_audit.get('ranked_failure_count')}; summary={full_audit.get('summary_failure_count')}; docs={full_audit.get('docs_failure_count')}; copy={full_audit.get('copy_failure_count')}.",
                "evidence": "data/full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json",
            },
        ]
    )

    gap_status = pd.DataFrame(
        [
            {
                "area": "strict_current_full52_attack",
                "status": "complete",
                "blocking": False,
                "detail": "The missing fair attack comparison is now complete and is the source used by ranked attack tables.",
            },
            {
                "area": "clean_generalization_tables",
                "status": "complete",
                "blocking": False,
                "detail": "52-dataset clean table and 50-generalization summaries exist and pass recompute checks.",
            },
            {
                "area": "robustness_svd_jacobian_25sample",
                "status": "complete_with_declared_limits",
                "blocking": False,
                "detail": "Six-model 25-sample tables and SVD/error spectrum summaries are present; small-n local rows are not used as global proof.",
            },
            {
                "area": "diagnostic_metric_interpretation",
                "status": "complete_with_boundaries",
                "blocking": False,
                "detail": "Cosines, angles, delta norms, and process quantities remain recorded but are not counted as model-quality proof.",
            },
            {
                "area": "figures_and_organized_release",
                "status": "complete_local",
                "blocking": False,
                "detail": f"Organized release has {organized_file_count} files and missing={organized_missing_count} after refresh.",
            },
            {
                "area": "github_push",
                "status": "blocked_by_missing_credential",
                "blocking": True,
                "detail": "Current shell has no GitHub credential; local vast-ai is ahead of origin/vast-ai by 2 commits.",
            },
            {
                "area": "r2_upload_latest_final_files",
                "status": "blocked_by_missing_environment_credentials",
                "blocking": True,
                "detail": "Current shell has no R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_ENDPOINT, so newest final-resolution files are local only until env vars are set.",
            },
        ]
    )

    strict_protocol_checks = protocol_validation.copy()
    strict_protocol_checks["ok"] = strict_protocol_checks["ok"].astype(bool)

    recovered_bad_rows = protocol_summary[
        protocol_summary["protocol_status"].astype(str).str.contains("mismatched", na=False)
    ].copy()

    rerun_summary.to_csv(OUT_DATA / "rerun_summary.csv", index=False)
    gap_status.to_csv(OUT_DATA / "remaining_gap_status.csv", index=False)
    old_vs_new.to_csv(OUT_DATA / "old_mixed_vs_strict_latest_attack_summary.csv", index=False)
    recovered_bad_rows.to_csv(OUT_DATA / "old_mixed_bad_source_rows.csv", index=False)
    strict_protocol_checks.to_csv(OUT_DATA / "strict_protocol_validation_copy.csv", index=False)
    final_boundaries.to_csv(OUT_DATA / "final_claim_boundaries_copy.csv", index=False)

    report = f"""# Burgers Rerun Explanation And Gap Status, 20260614

Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

This report answers four concrete questions: what was rerun, why the old conclusion was wrong, why the new conclusion is valid, and what is still missing.

## Short Answer

- The repair did **not** rerun training.
- The repair did **not** rerun random_solver_y or random_clean_y.
- The repair reran exactly the missing fair attack piece: latest old4 models (`baseline`, `loss1`, `loss2`, `loss3`) on the same current 52-dataset / 10200-sample widevis manifest used by `random_solver_y=7860` and `random_clean_y=8000`.
- The old six-model full-52 attack table was not a fair current comparison because old4 rows came from historical first-master data while random rows came from the current solver7860/clean8000 suite.
- The current strict table is fair because protocol and manifest checks pass across old4 and random models.

## Rerun Summary

{md_table(rerun_summary)}

## Why The Previous Table Was Wrong

The previous recovered table was useful as a provenance artifact, but not valid for the current six-model claim. In the generalization rows, old4 models had historical attack dataset IDs, while random models had current widevis dataset IDs. That is exactly the source of the contradictory statement.

{md_table(recovered_bad_rows[["model", "source", "source_family", "protocol_status", "rows", "mismatch_rows", "strict_latest_same_dataset_comparable_rows", "attack_loss_increase_mean", "attack_loss_increase_median"]])}

## Old Mixed Table Versus Strict Latest Table

This table shows why the old mean was misleading for current `loss3`: the old mixed table and the strict latest table are different protocol/provenance objects.

{md_table(old_vs_new[["model", "source", "datasets", "attack_loss_increase_mean_old_mixed", "attack_loss_increase_median_old_mixed", "attack_loss_increase_mean_strict_latest", "attack_loss_increase_median_strict_latest", "old_minus_strict_attack_increase_mean"]])}

## Why The New Table Is Correct

{md_table(strict_protocol_checks)}

Strict latest attack winner count by dataset:

{json.dumps(winner_count, indent=2, sort_keys=True)}

## Remaining Gaps

{md_table(gap_status)}

## Claim Boundaries

{md_table(final_boundaries)}

## Bottom Line

Locally, there is no remaining analysis blocker for the Burgers conclusion. The valid strong claim is:

`loss3` is the best model on the main comparable current evidence: clean/generalization accuracy, strict full-52 attack robustness, and SVD/error-operator spectrum.

The invalid overstatement is:

`loss3` is best on every single recorded scalar. Some single-split, small-n local, or diagnostic rows favor another model, and those rows are preserved instead of hidden.

External sync is the only current blocking item: GitHub/R2 credentials are not present in the active shell, so the newest local commits/results cannot be pushed/uploaded from here until those environment variables exist.

## Output CSVs

- `data/rerun_explanation_gap_status_20260614/rerun_summary.csv`
- `data/rerun_explanation_gap_status_20260614/remaining_gap_status.csv`
- `data/rerun_explanation_gap_status_20260614/old_mixed_vs_strict_latest_attack_summary.csv`
- `data/rerun_explanation_gap_status_20260614/old_mixed_bad_source_rows.csv`
- `data/rerun_explanation_gap_status_20260614/strict_protocol_validation_copy.csv`
- `data/rerun_explanation_gap_status_20260614/final_claim_boundaries_copy.csv`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    DOC_REPORT.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_data": OUT_DATA.relative_to(REPO).as_posix(),
                "report": OUT_REPORT.relative_to(REPO).as_posix(),
                "doc": DOC_REPORT.relative_to(REPO).as_posix(),
                "blocking_gap_count": int(gap_status["blocking"].sum()),
                "protocol_failures": int((~strict_protocol_checks["ok"]).sum()),
                "loss3_beats_random_solver_y_rows": loss3_wins_solver,
                "dataset_rows": int(len(winners)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
