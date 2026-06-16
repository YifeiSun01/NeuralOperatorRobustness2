# Darcy cflow time-matched audit release - 2026-06-14

Status: complete existing-artifact audit and R2 upload.

This release packages the 2D DarcyFlow / Darcy cflow time-matched artifacts that
were available locally after the random clean and random solver extensions. It
does not rerun training, attacks, SVD/Jacobian, or dense heatmap generation. The
builder is conservative: it reuses the existing DarcyFlow audit/release payloads,
adds cflow-specific ranked tables and reports, and records missing expensive
coverage explicitly.

## Local payloads

- `outputs/darcy_cflow_timematched_full_or_audit_20260614/`
- `outputs/darcy_cflow_timematched_organized_release_20260614/`

Each final payload directory contains `569` files and `1,560,307,897` logical
bytes after adding `reports/R2_GITHUB_SYNC_RECORD.md`.

## R2 upload

Bucket prefix:

```text
neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected
```

Uploaded and verified remote paths:

- `outputs/darcy_cflow_timematched_full_or_audit_20260614`
- `outputs/darcy_cflow_timematched_organized_release_20260614`

Final R2 verification:

| payload | remote object count | remote bytes | check |
| --- | ---: | ---: | --- |
| full_or_audit | 569 | 1,560,307,897 | `rclone check --size-only`: 0 differences |
| organized_release | 569 | 1,560,307,897 | `rclone check --size-only`: 0 differences |

R2 returned transient `501 Not Implemented` responses on the first multipart
pass, then rclone retried successfully. The final size and size-only checks
matched the local payloads.

## GitHub branch

- Branch updated: `vast-ai-darcy-flow`
- Code path: `tools/build_darcy_cflow_timematched_audit_20260614.py`
- This Git update intentionally excludes large output payloads.

## Main generated cflow files

- `reports/DARCY_CFLOW_AUDIT_REPORT.md`
- `reports/R2_GITHUB_SYNC_RECORD.md`
- `data/cflow_clean_52dataset_metric_long_ranked.csv`
- `data/cflow_attack_metric_long_ranked.csv`
- `data/cflow_svd_jacobian_metric_long_ranked.csv`
- `data/cflow_all_scalar_metric_long_ranked.csv`
- `data/cflow_metric_model_summary_ranked.csv`
- `data/cflow_per_sample_first_place_counts.csv`
- `data/cflow_paired_significance_bootstrap_tests.csv`
- `data/cflow_mechanism_scalar_correlations.csv`
- `data/cflow_coverage_reuse_recompute_notes.csv`
- `data/cflow_quality_vs_diagnostic_metric_policy.csv`
- `manifests/cflow_bundle_summary.json`
- `manifests/cflow_dense_heatmap_manifest.csv`

## Key conclusions

Clean 52-dataset evaluation:

- `loss3` is best for RMSE, Relative L2, MSE, MAE, and accuracy-score summaries
  in both all-dataset and generalization-only scopes.
- Baseline only wins the invalid-value-fraction diagnostic because all methods
  have zero invalid fraction in the available table.

Attack robustness smoke artifacts:

- For all datasets, `physics` has the smallest final adversarial loss and clean
  loss in the available attack table.
- `loss3` has the smallest all-dataset mean loss increase and relative increase.
- For generalization-only attack rows, `physics` has the smallest mean loss
  increase, while `loss3` has the smallest relative increase.
- Delta L2/RMS/Linf are recorded as process diagnostics only, not as direct
  model-quality evidence.

SVD/Jacobian smoke artifacts:

- The available SVD/Jacobian evidence is smoke coverage: 3 samples per model, not
  the requested full fixed 25 samples.
- Available scalar correlations support the mechanism that `||J_error^T e||`
  explains attack loss increase better than the top error singular value in this
  smoke set.
- The report keeps all correlation `n` values visible so this limited coverage
  is not overread.

Physics residual:

- Full per-dataset physics/PDE residual columns were not found in the clean
  52-dataset evaluation tables.
- Physics-run training logs include a partial `darcy_physics_metric`, so physics
  residual is documented separately and is not merged into RMSE/Relative L2
  conclusions.

## Coverage notes

- Random clean and random solver are included at the extended 3500-epoch level in
  the source DarcyFlow audit tables.
- Robustness coverage found locally is 52 datasets x 2 samples x 7 models, not
  the requested 50 samples per dataset.
- SVD/Jacobian coverage found locally is 3 samples per model, not the requested
  fixed 25-sample manifest.
- Full top50/top100 model-solver subspace artifacts were not found locally.
- Existing loss3-advantage 2D heatmaps are reused; the full dense group00-group05
  variant matrix was not regenerated.

## Credential handling

R2 and GitHub credentials were used only as ephemeral command environment values.
No token or secret was written to code, Markdown, Git config, or payload reports.
