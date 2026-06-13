# Darcy Random-Inclusive Burgers-Style Bundle - 2026-06-13

Status: regenerated Darcy Flow figures to include the two random-source training methods.

- Bundle: `visualizations/darcy_random_inclusive_burgers_style_bundle_20260613`
- Work manifest: `analysis_outputs/darcy_random_inclusive_burgers_style_bundle_20260613/bundle_manifest.json`
- Methods in training/loss curves: `loss1, loss2, loss3, physics loss, random clean y, random solver y`
- Heatmap rows/models: `baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y`
- Comparison PNGs: `10`
- Per-method PNGs: `18`
- Heatmap PNGs: `50` across `10` variants
- Total bundle PNGs: `78`

## Layout

- `comparison_dense/`: six-method loss, RMSE, wall-clock, delta, attack gain, and Jacobian/robustness comparison figures.
- `comparison_dense/darcy_2d_attack_heatmaps/`: seven-model 50-step binary attack heatmaps with loss-growth curves under each heatmap.
- `loss1/`, `loss2/`, `loss3/`, `physics/`, `random_clean_y/`, `random_solver_y/`: per-method train/test/generalization and training-dynamics panels.

## Heatmap Variants

- `index0`: `5` PNGs
- `loss3_best`: `5` PNGs
- `loss3_lower_quartile`: `5` PNGs
- `loss3_median`: `5` PNGs
- `loss3_more_faded_maxfade_signsame`: `5` PNGs
- `loss3_more_faded_recommended`: `5` PNGs
- `loss3_top10pct_visual_relaxed`: `5` PNGs
- `loss3_top5pct_visual_relaxed`: `5` PNGs
- `loss3_upper_quartile`: `5` PNGs
- `loss3_worst`: `5` PNGs

The visualization bundle itself is PNG-only; CSV/JSON summaries remain under `analysis_outputs`.
