# R2 and GitHub Sync 20260615 - NS External Forcing and Darcy Results

Status: completed on 2026-06-15.

## GitHub

- Repository: `YifeiSun01/NeuralOperatorRobustness2`.
- Branch used: `vast-ai-darcy-flow`.
- Commit pushed before this sync note: `1a4210a`
  (`Add NS external forcing crop-stitch figures`).
- GitHub payload scope: code and Markdown only.
- Files included in `1a4210a`:
  - `EXPERIMENT_LEDGER.md`
  - `docs/ns_external_forcing_mean_perturbation_crops_20260615.md`
  - `tools/make_ns_external_forcing_crop_stitch.py`
  - `tools/make_ns_external_forcing_from_uploaded_screenshots.py`
  - `tools/replot_ns_fno2d_r6_remove_range.py`

## R2

- Bucket/prefix:
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`.
- R2 endpoint was used through temporary rclone environment variables; no
  credentials were written to repository files.
- R2 payload scope: generated results, data tables, visualizations, and run
  artifacts.

Synced local directories:

- `analysis_outputs/` ->
  `.../NeuralOperatorRobustness2-selected/analysis_outputs/`
- `outputs/` ->
  `.../NeuralOperatorRobustness2-selected/outputs/`
- `adversarial_training_runs/` ->
  `.../NeuralOperatorRobustness2-selected/adversarial_training_runs/`

Verification checks:

- Confirmed on R2:
  `analysis_outputs/ns_external_forcing_summary_20260615/ns_external_forcing_attack_perturbation_uploaded_8row.png`.
- Confirmed on R2:
  `analysis_outputs/ns_external_forcing_summary_20260615/ns_external_forcing_attack_perturbation_uploaded_available_rows.png`.
- Confirmed on R2:
  `outputs/ns_fno2d_r6_m96x96_w80_remove_range_20260615/GRF/FNO2d_r6_m96x96_w80_Tin10_T10_remove_range_m1p5_m1_drop.png`.
- Confirmed on R2:
  `outputs/darcy_cflow_final_robustness_20260615/data/attack50_summary_by_dataset_model.csv`.
- Confirmed on R2 that the two 3000-epoch Darcy supervised run directories each
  expose the expected `attack_epoch_summary.csv`, `eval_metrics.csv`, and
  `train_steps.csv` files under their `darcy/` subdirectories.

Operational notes:

- `analysis_outputs/` was copied once normally and once with `--copy-links`.
  The copy-links pass reported nothing additional to transfer, indicating the
  linked targets were already covered remotely.
- `outputs/` needed `--no-update-modtime` because Cloudflare R2 returned
  `501 NotImplemented` for rclone attempts to update object modification times
  on already-existing objects.
- The live rclone log files under `outputs/r2_upload_darcy_20260615/*.log` were
  excluded from the final clean `outputs/` pass to avoid copying a log file
  while rclone was still writing it.
