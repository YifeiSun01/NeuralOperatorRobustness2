# Darcy Loss3 Top-Percentile Visual Heatmaps - 2026-06-12

This adds top-5% and top-10% Loss3-ranked Darcy five-model attack heatmap folders, plus relaxed visual-quality folders that enforce delta-shape similarity and residual sign consistency.

## Output Folders

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top5pct_visual`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top5pct_visual`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top5pct_visual_relaxed`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top5pct_visual_relaxed`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed`

The same four visualization folders were also copied into:

`visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612/comparison_dense/darcy_2d_attack_heatmaps/`

## Interpretation

- `loss3_top5pct_visual` and `loss3_top10pct_visual` are strict top-percentile folders selected from the existing 50-step Loss3-advantage ranking. Some highpass/bandpass strict candidates have very strong Loss3 margin but poor delta-shape similarity or residual sign flips.
- `loss3_top5pct_visual_relaxed` and `loss3_top10pct_visual_relaxed` are the recommended presentation folders for the visual story: candidates must satisfy delta cosine mean >= 0.70, delta cosine min >= 0.68, residual ratio min >= 1.16 against loss1/loss2/physics, and residual sign agreement = 1.

## Relaxed Selection Metrics

### loss3_top5pct_visual_relaxed

| sample | rank | margin | delta cos mean | delta cos min | residual ratio min | sign agree |
|---|---:|---:|---:|---:|---:|---:|
| `loss3_top5pct_visual_relaxed_01_maternfine_idx32_rank05` | 5 | 1.817e-06 | 0.775 | 0.771 | 1.263 | 1 |
| `loss3_top5pct_visual_relaxed_02_highpass_idx0_rank23` | 23 | 1.725e-06 | 0.775 | 0.762 | 1.231 | 1 |
| `loss3_top5pct_visual_relaxed_03_rectangles_idx18_rank04` | 4 | 1.978e-06 | 0.733 | 0.721 | 1.258 | 1 |
| `loss3_top5pct_visual_relaxed_04_highpass_idx0_rank22` | 22 | 1.905e-06 | 0.785 | 0.774 | 1.255 | 1 |
| `loss3_top5pct_visual_relaxed_05_bandpass_idx20_rank13` | 13 | 1.798e-06 | 0.811 | 0.810 | 1.239 | 1 |

### loss3_top10pct_visual_relaxed

| sample | rank | margin | delta cos mean | delta cos min | residual ratio min | sign agree |
|---|---:|---:|---:|---:|---:|---:|
| `loss3_top10pct_visual_relaxed_01_maternfine_idx25_rank13` | 13 | 1.472e-06 | 0.836 | 0.823 | 1.190 | 1 |
| `loss3_top10pct_visual_relaxed_02_highpass_idx39_rank38` | 38 | 1.581e-06 | 0.764 | 0.752 | 1.204 | 1 |
| `loss3_top10pct_visual_relaxed_03_rectangles_idx0_rank12` | 12 | 1.699e-06 | 0.785 | 0.780 | 1.216 | 1 |
| `loss3_top10pct_visual_relaxed_04_highpass_idx34_rank38` | 38 | 1.537e-06 | 0.788 | 0.774 | 1.197 | 1 |
| `loss3_top10pct_visual_relaxed_05_bandpass_idx42_rank25` | 25 | 1.570e-06 | 0.812 | 0.801 | 1.201 | 1 |

## Scripts

- `tools/build_darcy_loss3_top_percentile_visual_heatmaps_20260612.py`: strict top5/top10 plus candidate visual scoring.
- `tools/build_darcy_loss3_visual_relaxed_heatmaps_20260612.py`: relaxed visual-quality folders from existing arrays.
- `tools/plot_darcy_five_model_batch_ranked_heatmaps_20260612.py`: patched to clone tensor slices before JAX solver calls, avoiding XLA buffer alignment failures on sliced batches.
