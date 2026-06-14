# Darcy SIR20 Artifact Upload, 2026-06-14

Status: uploaded selected Darcy SIR20 deliverable artifacts to Cloudflare R2 and
prepared code/docs for the `vast-ai-darcy-flow` GitHub branch.

## R2 Destination

Bucket/prefix:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Uploaded under the matching repository-style paths below the prefix.

## Uploaded Output Directories

Remote size checks after upload:

| Local/remote path | Objects | Bytes |
| --- | ---: | ---: |
| `outputs/darcy_corrected_loss_figures_review_20260614` | 16 | 23,386,780 |
| `outputs/darcy_eval_artifact_corrected_20260614` | 12 | 72,262,123 |
| `outputs/darcy_generalization50_artifact_corrected_20260614` | 15 | 1,042,705,371 |
| `outputs/darcy_optimizer_artifact_corrected_20260614` | 18 | 6,830,367 |
| `outputs/darcy_raw_vs_corrected_audit_20260614` | 14 | 581,974,888 |
| `outputs/darcy_sir20_existing_curated_bundle_20260614` | 66 | 1,155,057,170 |
| `outputs/darcy_sir20_required_figures_only_20260614` | 30 | 716,219,908 |
| `outputs/darcy_random_3000_20260614` | 9 | 2,311 |
| `outputs/experiment_prompts_darcy_burgers_20260614` | 2 | 9,355 |

## Paired Audit Figures Verified On R2

The paired raw/corrected audit directory contains:

- `paired_raw_vs_corrected_rmse_generalization_part01_epoch.png`
- `paired_raw_vs_corrected_rmse_generalization_part02_epoch.png`
- `paired_raw_vs_corrected_relative_l2_generalization_part01_epoch.png`
- `paired_raw_vs_corrected_relative_l2_generalization_part02_epoch.png`

These are diagnostic raw-vs-artifact-corrected plots. The corrected panels are
derived visualization columns; the raw logs are preserved separately and were not
overwritten.

## Local Upload Logs

Local upload records are in:

- `outputs/r2_upload_darcy_20260614/rclone_selected_outputs.log`
- `outputs/r2_upload_darcy_20260614/remote_size_check.tsv`
- `outputs/r2_upload_darcy_20260614/paired_audit_remote_figures.txt`
