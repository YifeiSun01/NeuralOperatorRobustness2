# DarcyFlow Time-Matched Existing-Artifact Audit Release, 2026-06-14

Status: complete for existing local artifacts. This did not rerun training,
robustness attacks, SVD/Jacobian, dense groups, or expensive sweeps.

## Output Directories

- Final audit:
  `outputs/darcyflow_timematched_full_or_audit_20260614/`
- Organized release:
  `outputs/darcyflow_timematched_organized_release_20260614/`

Both directories contain:

- `figures/`
- `data/`
- `reports/`
- `logs/`
- `manifests/`

## Generated Tables

The audit data folder includes the requested ranked/statistical tables:

- `metric_model_summary_ranked.csv`
- `metric_best_summary_ranked.csv`
- `metric_best_vs_other_significance_tests.csv`
- `metric_loss3_vs_other_significance_tests.csv`
- `clean_52dataset_metric_long_ranked.csv`
- `attack_52dataset_metric_long_ranked.csv`
- `robustness_25sample_metric_long_ranked.csv`
- `svd_error_topk_ranked_tables.csv`
- `svd_error_top20_ranked_tables.csv`
- `model_level_scalar_ranked_tables.csv`
- `correlations_sorted_tables.csv`
- `random_coverage_partial_metric_notes.csv`
- `coverage_notes.csv`

## Figures

Generated `108` raw-point main training/evaluation figures:

- variants: all six models, no random clean, loss123 only
- y-scales: linear, log
- x-axes: epoch, work-clock hours, wall-clock hours
- metrics: RMSE, Relative L2
- scopes: train/test/generalization mean, generalization part01, generalization part02

Existing diagnostic figures were copied separately:

- paired raw-vs-artifact-corrected audit figures
- previous required raw figures
- five existing loss3-advantage Darcy attack heatmaps under
  `figures/dense_existing/group05_loss3_advantage/`

## Coverage

Observed coverage from local files:

- Clean 52-dataset evaluation is available for baseline plus loss1/loss2/loss3/physics and partial random clean/solver.
- Archived required-figure CSV max epochs are loss1 `3000`, loss2 `3079`, loss3 `3033`, physics `3121`, random clean `1100`, random solver `1100`.
- Robustness found locally is smoke coverage: 52 datasets x 2 samples x 7 models, not 50 samples per dataset.
- SVD/Jacobian found locally is smoke coverage: 3 samples per model, not 25 samples.
- Full physics/PDE residual evaluation columns were not found in the clean 52-dataset evaluation CSVs.
- Full dense group00..group05 variant matrix was not found; existing loss3-advantage heatmaps were organized but no dense recomputation was launched.

The detailed coverage notes are in:

- `outputs/darcyflow_timematched_full_or_audit_20260614/data/coverage_notes.csv`
- `outputs/darcyflow_timematched_full_or_audit_20260614/reports/COVERAGE_NOTES.md`

## Reports

- `outputs/darcyflow_timematched_full_or_audit_20260614/reports/AUDIT_REPORT.md`
- `outputs/darcyflow_timematched_full_or_audit_20260614/reports/STATISTICAL_APPENDIX.md`
- `outputs/darcyflow_timematched_full_or_audit_20260614/reports/COVERAGE_NOTES.md`

The report bolds the best model for each metric and records coverage status
next to ranked and statistical results.

## R2 Upload

Uploaded under:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/`

Verified paths:

- `darcyflow_timematched_full_or_audit_20260614`: 554 objects,
  1,418,427,564 bytes, `rclone check --size-only --one-way` found
  0 differences.
- `darcyflow_timematched_organized_release_20260614`: 554 objects,
  1,418,431,996 bytes, `rclone check --size-only --one-way` found
  0 differences.

The first R2 multipart upload attempt reported transient `501 Not Implemented`
responses; rclone retried, then both final remote checks matched the local
directories.

## Reproduction

Generated with:

```bash
adv_robust/bin/python tools/build_darcyflow_timematched_audit_release_20260614.py --clean
```

After small report-format fixes, tables/reports/manifests were refreshed without
redrawing figures:

```bash
adv_robust/bin/python tools/build_darcyflow_timematched_audit_release_20260614.py --skip-figures
```
