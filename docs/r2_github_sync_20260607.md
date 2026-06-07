# R2 And GitHub Sync - 2026-06-07

Status: R2 upload completed for the Burgers round03 final-extension artifacts. GitHub commit/push is handled after this record is written.

Timestamp: `2026-06-07T13:50:00+00:00`

## R2 Upload

Destination:

- Bucket: `neural-operator-robustness`
- Prefix: `machine-sync/NeuralOperatorRobustness2-selected`
- Endpoint: Cloudflare R2 account endpoint supplied by the user.

Observed upload:

- Tool: `rclone`.
- Mode: `copy`, not `sync`; remote objects were not deleted or pruned.
- R2 compatibility flags used: `--s3-no-check-bucket --s3-no-head --s3-disable-checksum`.
- Temporary rclone config was written under `/tmp` and deleted after use.

Uploaded/updated selected paths:

- `generalization_datasets_burgers_loss3_selective_search/round_03`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss123_final_extension_20260606_logs`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_gpu_preflight_20260606`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`
- `visualizations/burgers_loss3_selective_round03_loss1_5000ep_long_20260605_plots`
- `visualizations/burgers_loss3_selective_round03_loss2_2000ep_long_20260605_plots`
- `visualizations/burgers_loss3_selective_round03_loss3_1500ep_long_20260605_plots`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`
- final-extension Markdown reports under `docs/`
- final-extension helper scripts under `tools/`

Observed remote size checks:

- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606`: `2046` objects, `308.409 MiB`.
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`: `395` objects, `809.670 MiB`.

Observed remote existence checks passed for:

- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/checkpoints/burgers_epoch5000_step015000.pt`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_jacobian_svd_summary.md`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_final_generated50_loss3_advantage_hist.png`
- `docs/burgers_loss3_selective_round03_final_extension_completion_20260607.md`

## GitHub Upload Scope

Planned GitHub commit scope:

- `EXPERIMENT_LEDGER.md`
- final-extension Markdown reports under `docs/`
- final-extension/post-continuation automation scripts under `tools/`
- the final-extension SVD Markdown summary under `forensics/.../round03_loss123_final_extension_jacobian_svd_summary.md`

Excluded from GitHub:

- model checkpoints (`*.pt`)
- arrays (`*.npz`, `*.npy`)
- generated CSV tables
- PNG visualizations
- generated datasets and training run directories

Security note: credentials were not written into repository files. The user should rotate the pasted R2/GitHub credentials after confirming the sync.
