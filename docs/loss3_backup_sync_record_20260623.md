# Loss3 Backup Sync Record - 2026-06-23

This note records the backup actions for the current Loss3 optimizer and
mechanism-analysis artifacts.

## R2 Snapshot

- Snapshot prefix:
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/snapshots/20260623T014210Z/NeuralOperatorRobustness2`
- Local selection manifest:
  `analysis_outputs/backup_manifests/backup_selection_20260623T014210Z.txt`
- Local upload log:
  `analysis_outputs/backup_manifests/r2_upload_20260623T014210Z.log`

Selected directories:

- `docs`
- `tools`
- `analysis_outputs/optimizer_ablation_20260622`
- `analysis_outputs/mechanism_20260622`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622`
- `analysis_outputs/backup_manifests`

## GitHub Snapshot

Code and Markdown are backed up through the `merge-vast-ai-darcy-flow` branch.
Large analysis outputs are intentionally backed up to R2 rather than committed to
git.

## Notes

- Secrets were not written into this repository.
- The NS2D exact mechanism job and the extended mechanism queue were still
  running when this backup was started, so their later outputs need a follow-up
  R2 snapshot after the queue finishes.
