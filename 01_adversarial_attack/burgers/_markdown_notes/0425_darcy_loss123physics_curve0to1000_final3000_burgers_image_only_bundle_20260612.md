# Darcy Curve 0..1000 + Final 3000 Burgers-Style Image-Only Bundle - 2026-06-12

Status: generated a separate Darcy Flow image-only bundle that removes the stage2 discontinuity from training-epoch plots while keeping final 3000-epoch model/attack results.

- Bundle: `visualizations/darcy_loss123physics_curve0to1000_final3000_burgers_image_only_bundle_20260612`
- Work/stitched curve data: `analysis_outputs/darcy_curve0to1000_final3000_burgers_image_only_bundle_20260612_work`
- Training-epoch curves: `0..1000` only
- Attack heatmaps and final robustness/generalization summaries: copied from final `3000`-epoch bundle
- Top-level folders: `comparison_dense`, `loss1`, `loss2`, `loss3`, `physics`
- PNG count: `146`
- Non-PNG files in bundle: `0`

## What Changed From The Full3000 Bundle

- Replaced per-method polished reports with 0..1000 versions.
- Replaced `darcy_adv_training_*` comparison curves with 0..1000 versions.
- Replaced dense 5x5 generalization trajectory panels with `darcy_curve0to1000_*` versions.
- Kept `darcy_full50_*` final summaries and all `darcy_2d_attack_heatmaps/*` from the 3000-epoch bundle.

This is intentionally a new folder, so the old full 0..3000 bundle remains available for audit.

## More-Faded Heatmaps

- Added `loss3_more_faded_recommended` and `loss3_more_faded_maxfade_signsame` under `comparison_dense/darcy_2d_attack_heatmaps`.
- These are merged into the image-only bundle rather than left only as standalone visualization folders.

