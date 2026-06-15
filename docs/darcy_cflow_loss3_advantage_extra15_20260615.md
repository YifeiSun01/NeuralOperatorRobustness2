# Darcy cflow loss3 advantage extra-15 heatmaps (2026-06-15)

## Important Scope Correction

These 15 extra heatmaps are curated seven-model visual diagnostics selected
from final clean Relative-L2 loss3-advantage cases. They are not the complete
50-sample-per-dataset robustness sweep and are not final robustness evidence.

## Status

Complete.

## Selection

Observed from
`outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`.

Fifteen additional generalization samples were selected from the final clean
Relative-L2 table. Each selected dataset has `loss3` lower than the best
non-loss3 method. The selected set expands beyond the original five-data-set
heatmap pool and covers these families:

- `matern_smooth`: 2 samples
- `rectangles`: 1 sample
- `wave_mix`: 2 samples
- `blocky_tiles`: 2 samples
- `cellular_blobs`: 2 samples
- `bandpass_grf`: 2 samples
- `matern_fine`: 2 samples
- `highpass_grf`: 2 samples

The manifest is:

- `outputs/darcy_cflow_timematched_organized_release_20260614/manifests/loss3_advantage_extra15_manifest_20260615.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/manifests/loss3_advantage_extra15_manifest_20260615.json`

## Attack Heatmaps

Important caveat: these heatmaps are curated seven-model visual diagnostics.
The samples were selected from the final clean Relative-L2 table, but the
heatmap attack comparison itself was generated with the existing seven-model
heatmap script/checkpoints, not a complete 3000-epoch robustness sweep.
Therefore, the 15/15 attack advantage statement below applies only to this
curated heatmap run and must not be generalized as the complete robustness
ranking.

The seven-model binary loss3 attack heatmap script was run with 50 attack steps
and epsilon fraction 0.025. The attack run produced:

- source analysis output:
  `analysis_outputs/darcy_seven_model_attack_heatmaps_20260615_loss3_advantage_extra15/`
- source visualizations:
  `visualizations/darcy_seven_model_attack_heatmaps_20260615_loss3_advantage_extra15/`
- organized release folder:
  `outputs/darcy_cflow_timematched_organized_release_20260614/figures/dense_existing/group15_loss3_advantage_extra/`
- combined old-5 plus new-15 folder:
  `outputs/darcy_cflow_timematched_organized_release_20260614/figures/dense_existing/group20_loss3_advantage_combined/`

## Verification

Observed from
`outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_extra15_attack_ranks_20260615.csv`:

- `attack_loss_gain`: loss3 is best on 15 of 15 samples.
- `adv_relative_l2_model_vs_solver`: loss3 is best on 15 of 15 samples.
- `adv_rmse_model_vs_solver`: loss3 is best on 15 of 15 samples.

Image checks:

- `group15_loss3_advantage_extra`: 15 PNG files.
- `group20_loss3_advantage_combined`: 20 PNG files.
- Generated image dimensions are `5428 x 5991`.
- Pixel variance checks were nonzero for all checked PNGs, so the copied figures
  are not blank.

## Interpretation

Observed evidence supports using these fifteen extra heatmaps as a stronger and
more diverse curated `loss3 advantage` visual panel. The selection is not
limited to the previous matern-fine/highpass/rectangles/bandpass pool, and the
generated attack metrics confirm loss3 as the best model on the chosen samples
within this specific seven-model heatmap run.
