# Burgers Timematched Solver7860 Clean8000 GitHub/R2 Sync, 20260614

## Summary

The Burgers time-matched solver7860/clean8000 audit records and result bundle
were synced on 2026-06-14.

## GitHub

Branch:

- `vast-ai`

Pushed commits included the Burgers audit/report updates through:

- `1f93be5 Record Burgers mechanism summary output placement`

The push used temporary Git credential handling. No token was written to repo
files, git config, or committed artifacts.

## R2

Bucket/prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Uploaded result root:

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/`

Remote location:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/burgers_timematched_solver7860_clean8000_audit_20260614/`

Verification after upload:

| check | value |
| --- | ---: |
| local file count | 2066 |
| remote object count | 2066 |
| local file bytes | 5359656110 |
| remote bytes | 5359656110 |

Key newly confirmed files on R2:

- `reports/burgers_jte_spectral_attack_mechanism_summary_20260614.md`
- `data/jte_spectral_attack_mechanism_summary_20260614/README.md`
- `data/jte_spectral_attack_mechanism_summary_20260614/correlations_with_attack_sorted.csv`
- `data/jte_spectral_attack_mechanism_summary_20260614/robustness_25sample_metric_long_ranked.csv`
- `data/jte_spectral_attack_mechanism_summary_20260614/random_solver7860_clean8000_svd_attack_bias_gradient_by_sample.csv`

Upload logs:

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/logs/r2_upload_burgers_timematched_solver7860_clean8000_audit_20260614.log`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/logs/r2_upload_burgers_timematched_solver7860_clean8000_audit_20260614_success.log`

## Notes

The first R2 attempt used default rclone S3 behavior and hit Cloudflare R2
`501 NotImplemented` responses. The successful upload used R2-compatible rclone
options:

- `--s3-disable-checksum`
- `--s3-no-head`
- `--s3-no-system-metadata`
- `--s3-no-check-bucket`
- `--size-only`

No credentials are stored in this repository or in the uploaded reports.
