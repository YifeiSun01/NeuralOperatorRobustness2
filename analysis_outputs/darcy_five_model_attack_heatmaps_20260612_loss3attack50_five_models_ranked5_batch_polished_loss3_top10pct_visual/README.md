# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual)

- Created: 2026-06-12T22:41:07+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/loss3_top10pct_visual_01_maternfine_idx32_rank05_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/loss3_top10pct_visual_02_highpass_idx49_rank02_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/loss3_top10pct_visual_03_rectangles_idx20_rank05_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/loss3_top10pct_visual_04_highpass_idx35_rank04_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/loss3_top10pct_visual_05_bandpass_idx12_rank01_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.48026e-06 | 0.179879 |
| loss1 | 5.71969e-06 | 0.187032 |
| loss2 | 4.9986e-06 | 0.171 |
| loss3 | 1.41061e-06 | 0.114477 |
| physics loss | 5.01175e-06 | 0.177137 |

## Visual Selection Criteria

Selected from the per-dataset top-percentile Loss3-advantage candidates using a display score that rewards smaller Loss3 residuals, delta-shape similarity to loss1/loss2/physics, and residual sign agreement. See `visual_selection_scores.csv`.
