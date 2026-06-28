# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade)

- Created: 2026-06-13T02:04:52+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/loss3_more_faded_maxfade_01_maternfine_idx30_rank01_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/loss3_more_faded_maxfade_02_highpass_idx33_rank01_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/loss3_more_faded_maxfade_03_rectangles_idx42_rank03_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/loss3_more_faded_maxfade_04_highpass_idx29_rank05_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/loss3_more_faded_maxfade_05_bandpass_idx35_rank02_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.63959e-06 | 0.180134 |
| loss1 | 5.92154e-06 | 0.187624 |
| loss2 | 5.06012e-06 | 0.170938 |
| loss3 | 2.04497e-07 | 0.0756343 |
| physics loss | 5.10234e-06 | 0.176714 |

## More-Faded Selection Criteria

This folder was selected from a wider candidate pool than the previous relaxed folders. The score emphasizes a larger Loss3 residual fade in `model - solver`, with residual mean-sign agreement, residual cosine, and delta cosine recorded in `visual_selection_scores.csv`.
