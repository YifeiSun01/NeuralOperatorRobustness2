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

The launcher uses `rclone copy` with `--size-only` and ordered `--filter` rules
so existing same-size remote objects are skipped and missing local artifacts are
uploaded incrementally. It does not delete any remote R2 object.

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
- Source and record files intended for GitHub: `*.py`, `*.sh`, `*.md`
- Files not matching artifact directories or artifact file extensions

Credentials are intentionally not written to this file or to the launcher.

## Launch Record

Observed launch status:

- A first attempt started at `2026-06-09T06:12:06+00:00` was stopped after
  rclone warned that mixed `--include` and `--exclude` rules have indeterminate
  parsing order.
- The launcher was revised to ordered `--filter` rules.
- A short `nohup` restart at `2026-06-09T06:14:06+00:00` did not remain active
  long enough to be used as the official background run.
- The artifact-only upload was then launched in tmux at
  `2026-06-09T06:15:59+00:00`.
- Tmux session: `artifact_r2_sync_20260609`
- Observed launcher process: `bash tools/start_artifact_r2_sync_20260609.sh`
- Observed rclone process: `rclone copy . R2:neural-operator-robustness/...`
- Local PID file: `forensics/artifact_r2_sync_20260609/artifact_sync.pid`
- Current log: `forensics/artifact_r2_sync_20260609/rclone_artifact_sync.log`

The restarted job is intended to keep running in the background. Completion
should be checked from the log rather than from chat history.

## Failure And Retry

Observed after launch:

- The tmux-launched job exited at `2026-06-09T06:25:41+00:00` with
  `exit_status=1`.
- The log recorded `Attempt 1/5 failed with 163 errors` on Darcy selected and
  stockgeneralization `.pt` files with R2 `NotImplemented` responses.
- Later retries reduced the repeated failure to the live upload log itself:
  `forensics/artifact_r2_sync_20260609/rclone_artifact_sync.log`.
- Because the upload job was writing that log while rclone was also trying to
  copy it, the live log remained a repeated failing object and caused the
  artifact copy to exit nonzero.

Correction:

- `tools/start_artifact_r2_sync_20260609.sh` now excludes
  `forensics/artifact_r2_sync_20260609/` from the artifact copy.
- The retry should continue as an incremental `rclone copy --size-only`, so
  files already present on R2 with the same size are skipped.
- The corrected retry was launched in tmux session `artifact_r2_sync_20260609`
  at `2026-06-09T06:41:23+00:00`.
- Observed active retry processes after launch: launcher PID `1367058` and
  rclone PID `1367074`.

## Completion Record

Observed completion:

- The corrected retry completed at `2026-06-09T06:43:04+00:00`.
- The completion evidence is the line
  `Completed: 2026-06-09T06:43:04+00:00` in
  `forensics/artifact_r2_sync_20260609/rclone_artifact_sync.log`.
- After completion, no `artifact_r2_sync_20260609` tmux session and no active
  `rclone copy` process were observed.

Interpretation:

- The artifact-only incremental R2 copy finished successfully after excluding
  the runtime log directory.
- Earlier `NotImplemented` errors in the same log belong to the failed
  pre-correction attempt and should not be read as the final retry status.
