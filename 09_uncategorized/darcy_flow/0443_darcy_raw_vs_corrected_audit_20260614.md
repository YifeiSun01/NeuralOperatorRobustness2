# Darcy Raw vs Artifact-Corrected Audit Figures - 2026-06-14

## Status

Complete.

## Purpose

This is a transparency/audit bundle comparing raw old logs with derived
artifact-corrected visualization columns. The corrected panels are not raw
experiment measurements.

## Source Files

- `outputs/darcy_eval_artifact_corrected_20260614/data/eval_split_summary_artifact_corrected.csv`
- `outputs/darcy_generalization50_artifact_corrected_20260614/data/eval_metrics_artifact_corrected.csv`

## Output

- `outputs/darcy_raw_vs_corrected_audit_20260614/figures/`
- `outputs/darcy_raw_vs_corrected_audit_20260614/reports/raw_vs_corrected_audit.md`

## Figures

- `raw_vs_corrected_epoch_rmse_train_test_generalization.png`
- `raw_vs_corrected_epoch_relative_l2_train_test_generalization.png`
- `raw_vs_corrected_rmse_generalization_part01_epoch.png`
- `raw_vs_corrected_rmse_generalization_part02_epoch.png`
- `raw_vs_corrected_relative_l2_generalization_part01_epoch.png`
- `raw_vs_corrected_relative_l2_generalization_part02_epoch.png`
- `paired_raw_vs_corrected_rmse_generalization_part01_epoch.png`
- `paired_raw_vs_corrected_rmse_generalization_part02_epoch.png`
- `paired_raw_vs_corrected_relative_l2_generalization_part01_epoch.png`
- `paired_raw_vs_corrected_relative_l2_generalization_part02_epoch.png`

## Notes

- Yellow spans mark rows where corrected columns were imputed.
- Raw experiment CSVs were not overwritten.
- These figures are for inspection and audit; the derived panels should not be
  presented as raw experiment measurements.
- The `paired_raw_vs_corrected_*` figures place each dataset's raw and corrected
  panels directly adjacent to each other for pairwise visual comparison.
