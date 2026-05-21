# Loss1/Loss2/Loss3 R2 and GitHub Sync - 2026-05-21

Status: completed code/documentation push and R2 artifact upload/update.

## GitHub

Observed output:

- Branch: `vast-ai`.
- Remote: `origin` / `https://github.com/YifeiSun01/NeuralOperatorRobustness2.git`.
- Substantive pushed commit: `7c1ebd4` (`Add loss1 loss2 combined optimizer findings`).
- Files committed in that commit: `EXPERIMENT_LEDGER.md`, the new loss1/loss2/loss3 Markdown records under `docs/`, and the new plotting/running scripts under `tools/`.
- Generated large artifacts under `forensics/`, `results/`, and `all_requested_figures_20260521/` were intentionally not committed to Git.

## R2

Destination prefix:

`s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/`

Upload/update mode:

- Used `rclone copy` rather than destructive sync, so remote-only historical objects were not deleted.
- Uploaded/updated: `docs/`, `tools/`, `EXPERIMENT_LEDGER.md`, `all_requested_figures_20260521/`, `results/`, and `forensics/`.
- The temporary local rclone config was used only for this sync and should be removed after verification.

Observed R2 verification:

- Key docs present on R2: `three_loss_burgers_optimizer_findings_summary_20260521.md`, `gpi_fast_optimizer_three_losses_interpretation_20260521.md`, `loss1_loss2_loss3_core4_combined_0to100_20260521.md`, and `all_requested_figures_single_folder_20260521.md`.
- Key tools present on R2: `plot_loss1_loss2_loss3_combined_core4_0to100.py` and `run_loss1_loss2_core4_baseline_visuals.py`.
- `EXPERIMENT_LEDGER.md` is present at the R2 root prefix.
- `all_requested_figures_20260521/`: `62` objects, `11,019,609` bytes.
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`: `36` objects, `6,221,148` bytes.
- Total `forensics/` prefix: `14,407` objects, `18,104,140,782` bytes.
- Total `results/` prefix: `41,449` objects, `1,992,387,343` bytes.
- No stale unlabeled `combined_*_by_*_0to100.png` files were observed in the remote combined artifact directory.

## Notes

Credentials were not written into repository files and were not committed. Since the credentials were shared in chat, they should be rotated after the sync if long-term security matters.
