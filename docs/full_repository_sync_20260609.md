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

## 2026-06-09 Broad Sync Stop And R2 Inventory

The broad R2 job was stopped after review because the command type was correct
but the scope was too wide. The job used `rclone copy`, not `rclone sync`, so it
does not delete remote files and should skip unchanged same-size objects. The
problem was that the manifest included old top-level roots that were already
present on R2 and did not need to be scanned again.

Observed R2 target inventory for
`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`:

- Total: `143764` objects, `389357585709` bytes, about `389.4 GB` decimal or `362.6 GiB`.
- `adversarial_training_runs`: `129.739 GiB`, `25030` files.
- `2D_NS_FNO2d_recurrent`: `96.840 GiB`, `6080` files.
- `2D_Darcy_FNO2d`: `35.827 GiB`, `859` files.
- `forensics`: `23.550 GiB`, `18538` files.
- Generalization dataset roots around `14.3 GiB` each.
- `Model`: `8.880 GiB`, `2493` files.
- `results`: `1.910 GiB`, `47120` files.
- `visualizations`: `0.488 GiB`, `536` files.
- Old 1D roots were already small on R2: `1D_Burgers` `0.179 GiB`,
  `1D_Burgers_deeponet` `0.028 GiB`, and
  `1D_Burgers_FNO_generalization` `0.028 GiB`.

Observed local broad-copy log:

- The stopped job had scanned or started copying `1D_Burgers`,
  `1D_Burgers_FNO_generalization`, `1D_Burgers_deeponet`,
  `2D_Darcy_FNO2d`, `2D_NS_FNO2d_recurrent`,
  `adversarial_training_runs`, and then `forensics`.
- No broad R2 upload session remains active after stopping
  `full_r2_sync_20260609`.

Corrected sync policy:

- Use `rclone copy` only for newly generated or recently modified experiment
  roots, visualizations, selected datasets, and forensics.
- Do not rescan historical roots such as old Burgers/DeepONet directories unless
  they were regenerated or explicitly requested.
- Keep Python, shell, and Markdown records in GitHub; keep checkpoints, arrays,
  datasets, images, logs, and generated experiment products in R2.
