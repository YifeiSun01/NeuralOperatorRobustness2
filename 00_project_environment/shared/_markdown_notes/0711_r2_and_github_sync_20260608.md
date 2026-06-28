# R2 And GitHub Sync - 2026-06-08

## Status

The local `NeuralOperatorRobustness2` workspace was synchronized according to the requested storage split:

- raw data, generated model/results artifacts, and visualizations were copied to Cloudflare R2;
- Python, shell, and Markdown files were staged for GitHub on the `vast-ai` branch.

No credentials are recorded in this file or in the repository.

## R2 Target

Cloudflare R2 S3 target:

- bucket: `neural-operator-robustness`
- prefix: `machine-sync/NeuralOperatorRobustness2-selected`

The upload used `rclone` with the Cloudflare S3 provider and the repository `.r2exclude` file. The excluded paths include `.git`, `adv_robust`, virtual environments, Python caches, and other cache directories.

## Local Snapshot Scope

Observed before upload with:

`rclone size . --exclude-from .r2exclude`

Local snapshot size:

- objects: `26,755`
- size: `24.240 GiB`

The snapshot includes repository data/model/result/visualization folders such as:

- `1D_Burgers`, `1D_Burgers_FNO_generalization`, `1D_Burgers_deeponet`
- `2D_Darcy_FNO2d`
- `2D_NS_FNO2d_recurrent`, `2D_NS_FNO2d_recurrent_real_initial`, `2D_NS_FNO3d`, `2D_NS_compare`, `2D_NS_old`
- `adversarial_training_runs`
- `forensics`
- `visualizations`
- `results`
- `generalization_datasets_*`
- `generalization_eval*`
- `analysis_outputs`, `benchmark_results`, `fno_training_runs`, `gradient_audit`, `path_audit`

## Upload Notes

The first R2 attempt used an S3 ACL setting and Cloudflare R2 returned `501 NotImplemented` on object upload. The run was stopped and rerun without ACL settings.

The successful pass used a no-ACL `rclone copy`. A final clean pass excluded `forensics/r2_upload_20260608/**` and wrote logs outside the source tree to avoid copying a log while it was still being written.

Final clean-pass log:

- local saved log: `forensics/r2_upload_20260608/rclone_copy_noacl_final_20260608.log`
- observed final message: `There was nothing to transfer`

## Remote Verification

Observed with remote `rclone size` after upload:

- remote prefix objects: `136,532`
- remote prefix total size: `280.973 GiB`

This remote total is larger than the local snapshot because it includes content already present under the same R2 prefix before this sync.

## GitHub Scope

For GitHub, only lightweight source and record files are intended to be committed:

- `*.py`
- `*.sh`
- `*.md`

Large generated arrays, checkpoints, images, datasets, and logs remain on R2 rather than being force-added to git.
