# Burgers Full New Generated Data R2 Backup, 20260614

## Summary

This record documents the follow-up R2 backup requested after the initial sync of
only the main `outputs/burgers_timematched_solver7860_clean8000_audit_20260614`
folder. The follow-up backup covered the broader set of newly generated or
untracked Burgers artifacts, including training runs, forensics, outputs,
visualizations, run logs, and generated generalization datasets.

No tokens or secrets are stored in this file or in the repository.

## R2 Destination

Bucket/prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Each local root was copied to the same relative path under that prefix.

## Backed-Up Roots

| local root | local files | local bytes | R2 one-way check |
| --- | ---: | ---: | --- |
| `adversarial_training_runs` | 25514 | 3895658265 | 0 differences, 25514 matching files |
| `forensics` | 3099 | 5835482614 | 0 differences, 3099 matching files |
| `generalization_datasets_burgers_semantic_wideparam_visible_20260611` | 4 | 101760 | 0 differences, 4 matching files |
| `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611` | 61 | 83901713 | 0 differences, 61 matching files |
| `outputs` | 4353 | 11410707855 | 0 differences, 4353 matching files |
| `run_logs` | 76 | 3707447 | 0 differences, 76 matching files |
| `visualizations` | 407 | 841675944 | 0 differences, 407 matching files |

## Upload Result

The upload ran as an R2-compatible rclone copy with:

- `--s3-disable-checksum`
- `--s3-no-head`
- `--s3-no-system-metadata`
- `--s3-no-check-bucket`
- `--size-only`

The copy completed with return code 0 for every root:

- `adversarial_training_runs`
- `forensics`
- `generalization_datasets_burgers_semantic_wideparam_visible_20260611`
- `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611`
- `outputs`
- `run_logs`
- `visualizations`

The follow-up one-way check also completed with return code 0 for every root.
This means every local file in those roots has a same-size object at the
corresponding R2 path. The R2 roots may contain additional historical objects;
therefore the correct validation is one-way local-to-remote equality, not equal
total remote object count.

## Symlink Note

During `forensics` checking, rclone reported one symlink:

- `forensics/burgers_wideparam_loss123_randomfield_round00_p2q2_six_model_visuals_20260613/comparison_dense`

That symlink points to:

- `visualizations/burgers_wideparam_loss123_randomfield_round00_p2q2_six_model_visuals_20260613/comparison_dense`

R2 does not preserve the symlink object itself in this copy mode, but the target
directory is under `visualizations`, and the `visualizations` root passed the
one-way check with 407 matching files. Therefore the actual image data behind
the symlink is backed up.

## Logs

Local logs:

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/logs/r2_upload_burgers_full_new_series_20260614.log`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/logs/r2_check_burgers_full_new_series_20260614.log`

The logs were also copied to R2 under the same `outputs/.../logs/` paths.

## Scope Boundary

This backup intentionally covered broad newly generated result/data roots. It is
separate from GitHub source-control sync: code and Markdown records are committed
to GitHub branch `vast-ai`, while heavy generated data, checkpoints, NPZ, CSV,
PNG, and logs are backed up to R2.
