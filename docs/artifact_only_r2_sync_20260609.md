# Artifact-Only R2 Sync - 2026-06-09

## Scope

Requested target:

- R2 prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`
- Local repository: `/workspace/NeuralOperatorRobustness2`

This upload is intentionally artifact-only. Source code, shell scripts, and
Markdown records are handled through GitHub. R2 is used for model checkpoints,
datasets, generated numeric results, logs, images, visualizations, and archives.

## Command

Launcher:

- `tools/start_artifact_r2_sync_20260609.sh`

Runtime records:

- Manifest: `forensics/artifact_r2_sync_20260609/artifact_sync_manifest.txt`
- Log: `forensics/artifact_r2_sync_20260609/rclone_artifact_sync.log`

The launcher uses `rclone copy` with `--size-only` so existing same-size remote
objects are skipped and missing local artifacts are uploaded incrementally. It
does not delete any remote R2 object.

## Included Artifacts

The sync includes generated experiment roots such as:

- `adversarial_training_runs/`
- `forensics/`
- `visualizations/`
- `visualization_outputs/`
- `analysis_outputs/`
- `figures/`
- `results/`
- `run_logs/`
- `data/`
- `Model/`
- `generalization_datasets*/`
- `generalization_eval*/`

It also includes nested model/data/result directories and common artifact file
types, including checkpoints (`*.pt`, `*.pth`, `*.ckpt`), NumPy/JAX/PyTorch data
(`*.npz`, `*.npy`), tables/configs/logs (`*.csv`, `*.json`, `*.txt`, `*.log`),
figures (`*.png`, `*.pdf`, `*.svg`, `*.jpg`), videos, and archives.

Excluded from R2 artifact upload:

- `.git/`
- `adv_robust/`
- `__pycache__/`
- Python bytecode
- Files not matching artifact directories or artifact file extensions

Credentials are intentionally not written to this file or to the launcher.
