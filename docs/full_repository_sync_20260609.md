# Full Repository GitHub/R2 Sync - 2026-06-09

## Scope

Requested sync target:

- GitHub branch: `YifeiSun01/NeuralOperatorRobustness2`, branch `vast-ai`
- R2 prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

## GitHub

Code and Markdown files are recorded in GitHub. Large generated arrays, model checkpoints, datasets, images, and experiment outputs are not force-added to Git; they are synced to R2.

## R2

The full R2 sync is launched with:

- Script: `tools/start_full_r2_sync_20260609.sh`
- Manifest: `forensics/full_r2_sync_20260609/full_sync_manifest.txt`
- Log: `forensics/full_r2_sync_20260609/rclone_full_sync.log`

The sync covers these local roots when present:

- `1D_Burgers`
- `1D_Burgers_FNO_generalization`
- `1D_Burgers_deeponet`
- `2D_Darcy_FNO2d`
- `2D_NS_FNO2d_recurrent`
- `adversarial_training_runs`
- `forensics`
- `visualizations`
- `run_logs`
- `generalization_datasets*`
- `generalization_eval*`

Credentials are intentionally not written to this file or to the script. The script expects R2 authentication to be provided through environment variables or an existing rclone config.
