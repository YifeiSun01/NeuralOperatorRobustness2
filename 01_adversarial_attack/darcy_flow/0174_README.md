# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced)

- Created: 2026-06-13T02:04:45+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/loss3_more_faded_balanced_01_maternfine_idx32_rank05_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/loss3_more_faded_balanced_02_highpass_idx33_rank01_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/loss3_more_faded_balanced_03_rectangles_idx18_rank04_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/loss3_more_faded_balanced_04_highpass_idx44_rank01_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/loss3_more_faded_balanced_05_bandpass_idx34_rank10_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_balanced/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.30862e-06 | 0.180361 |
| loss1 | 5.62738e-06 | 0.187897 |
| loss2 | 4.81932e-06 | 0.17063 |
| loss3 | 1.82143e-06 | 0.116029 |
| physics loss | 4.88047e-06 | 0.177724 |

## More-Faded Selection Criteria

This folder was selected from a wider candidate pool than the previous relaxed folders. The score emphasizes a larger Loss3 residual fade in `model - solver`, with residual mean-sign agreement, residual cosine, and delta cosine recorded in `visual_selection_scores.csv`.
