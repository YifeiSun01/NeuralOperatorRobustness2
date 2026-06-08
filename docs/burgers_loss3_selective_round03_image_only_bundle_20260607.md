# Burgers Round03 Image-Only Bundle - 2026-06-07

## Status

Created an image-only bundle for Burgers round03 loss1/loss2/loss3 and dense comparison plots at `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607`.

## Observed Evidence

- Bundle timestamp: `2026-06-07T20:13:50Z`.
- Top-level folders: `loss1`, `loss2`, `loss3`, and `comparison_dense`.
- PNG count: `53`.
- Non-PNG file count inside the bundle: `0`.
- Loss folders source images:
  - `loss1`: copied from `visualizations/burgers_loss3_selective_round03_loss1_8000ep_long_20260605_plots`.
  - `loss2`: copied from `visualizations/burgers_loss3_selective_round03_loss2_2000ep_long_20260605_plots`.
  - `loss3`: copied from `visualizations/burgers_loss3_selective_round03_loss3_1500ep_long_20260605_plots`.
- Comparison source images were copied from `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605` and compact comparison PNGs from `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`.
- Added round03 per-loss p2q2 attack verification PNGs to each loss polished report:
  - `loss1/polished_report/round03_loss1_baseline_vs_epoch8000_p2q2_attack_step000_before_perturbation.png`
  - `loss1/polished_report/round03_loss1_baseline_vs_epoch8000_p2q2_attack_step100_after_perturbation.png`
  - `loss2/polished_report/round03_loss2_baseline_vs_epoch2000_p2q2_attack_step000_before_perturbation.png`
  - `loss2/polished_report/round03_loss2_baseline_vs_epoch2000_p2q2_attack_step100_after_perturbation.png`
  - `loss3/polished_report/round03_loss3_baseline_vs_epoch1500_p2q2_attack_step000_before_perturbation.png`
  - `loss3/polished_report/round03_loss3_baseline_vs_epoch1500_p2q2_attack_step100_after_perturbation.png`
- Added clipped wall-clock every-epoch comparison plots to `comparison_dense`:
  - `round03_dense_wall_clock_rmse_every1_train_test_generalization_xmax12p5h.png`
  - `round03_dense_wall_clock_relative_l2_every1_train_test_generalization_xmax12p5h.png`

## Inference

The bundle is suitable for image-only sharing/inspection: it contains only PNG files, preserves the loss-specific polished-report image structure, includes the requested per-loss round03 p2q2 before/after attack images, adds the requested wall-clock `0..12.5h` clipped comparison views, and excludes sparse checkpoint-style heatmap+line plots.

## Sparse Checkpoint-Style Plot Cleanup - 2026-06-07T20:17:58Z

Observed cleanup: removed 6 sparse checkpoint-style heatmap+line PNGs from the bundle, matching `*checkpoint_style*heatmap_line_below*.png` across `loss1`, `loss2`, and `loss3` polished reports.

Removed file class: `polished_checkpoint_style_rmse_absolute11_heatmap_line_below_round03_long.png` and `polished_checkpoint_style_relative_l2_absolute11_heatmap_line_below_round03_long.png` for each loss. These were sparse checkpoint-column plots rather than every-epoch views.

Observed after sparse-plot cleanup and per-loss p2q2 image generation: bundle PNG count is `53`; non-PNG file count remains `0`.

## Per-Loss Round03 P2Q2 Attack Verification Images - 2026-06-07T20:44:37Z

Observed correction: removed the old copied `p2q2_reference_*` PNGs that had only been placed under `loss1/polished_report`, then generated fresh p=2,q=2 attack before/after PNGs separately for the round03 `loss1`, `loss2`, and `loss3` final self-training models.

Observed output summary: bundle now contains `53` PNG files and `0` non-PNG files. No `p2q2_reference_*.png` files remain.

Observed numeric source: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/summary.json`.

Observed final attack MSE means: baseline `0.026507497`; loss1 final model `0.0062939017`; loss2 final model `0.0092682792`; loss3 final model `0.0029296223`.



## Duplicate Comparison Directory Check - 2026-06-07T21:03:12Z

Observed from `diff -qr`: the bundle currently contains both `comparison_dense` and `burgers_loss3_selective_round03_long_training_comparison_dense_20260605` under `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607`.

Observed content comparison: `comparison_dense` has 23 files, all PNG; the long-name directory has 16 files, including 12 PNGs that are byte-identical to files also in `comparison_dense`, plus 4 non-PNG source/manifest files (`round03_dense_epoch_metrics_every1.csv`, `round03_dense_epoch_metrics_every5.csv`, `round03_dense_training_comparison_plot_manifest.txt`, and `round03_dense_plot_manifest.txt`).

Inference: `comparison_dense` is the intended image-only comparison folder. The long-name directory is an accidental copy of the original dense output directory and should not be part of the four-folder image-only bundle.


## Duplicate Directory Cleanup - 2026-06-07T21:04:48Z

Observed cleanup: removed the accidental long-name original dense-output directory from the bundle, leaving only the intended top-level folders `loss1`, `loss2`, `loss3`, and `comparison_dense`.

Observed post-cleanup verification: bundle contains 53 PNG files and 0 non-PNG files.

Inference: the image-only bundle now matches the requested four-folder structure, with `comparison_dense` as the sole comparison folder.


## Combined P2Q2 Comparison-Dense Images - 2026-06-07T21:21:56Z

Observed action: added six combined p=2,q=2 before/after attack PNGs to `comparison_dense`, generated from existing trace files by `tools/plot_burgers_round03_p2q2_combined_attack_panels.py`.

Observed output files:
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_step000_before_perturbation_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_step100_after_perturbation_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss1_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss2_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss3_before_after_overlay_two_column.png`

Observed post-generation verification: bundle now contains `59` PNG files and `0` non-PNG files. The four-column images are `7200x2528`; the two-column overlay images are `3600x2528`.

Inference: `comparison_dense` now contains the requested shared-baseline four-column before/after/overlay views plus the original-style two-column before/after overlays for loss1, loss2, and loss3.


## Overlay Before-Curve Alpha Correction - 2026-06-07T21:30:51Z

Observed issue: the first overlay version rendered before-perturbation main curves with alpha values `0.26`, `0.28`, and `0.30`, while after-perturbation curves used approximately `0.92..0.94`. The before curves were therefore too faint for visual inspection.

Correction: updated `tools/plot_burgers_round03_p2q2_combined_attack_panels.py` so all before-perturbation dashed main curves in overlay panels use `alpha=0.80`: delta, attacked input, solver before, model before, and model-solver diff before. Low-alpha fill bands remain light to avoid covering the after-step100 solid curves.

Regenerated overlay outputs:
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss1_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss2_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss3_before_after_overlay_two_column.png`

Observed verification: overlay PNG dimensions remain `7200x2528` for the four-column overlay and `3600x2528` for the two-column overlays. The image-only bundle still contains `59` PNG files and `0` non-PNG files.

Inference: the overlay figures now keep the before/after distinction via dashed vs solid line style while making the before curves visible enough for direct comparison.
