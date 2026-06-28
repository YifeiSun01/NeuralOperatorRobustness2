# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame)

- Created: 2026-06-13T02:09:19+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/loss3_more_faded_maxfade_signsame_01_maternfine_idx32_rank05_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/loss3_more_faded_maxfade_signsame_02_highpass_idx33_rank01_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/loss3_more_faded_maxfade_signsame_03_rectangles_idx18_rank04_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/loss3_more_faded_maxfade_signsame_04_highpass_idx0_rank22_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/loss3_more_faded_maxfade_signsame_05_bandpass_idx34_rank10_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_more_faded_maxfade_signsame/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.30798e-06 | 0.180273 |
| loss1 | 5.62766e-06 | 0.188219 |
| loss2 | 4.84278e-06 | 0.171003 |
| loss3 | 2.42073e-06 | 0.129574 |
| physics loss | 4.87807e-06 | 0.177903 |

## More-Faded Selection Criteria

This folder was selected from a wider candidate pool than the previous relaxed folders. The score emphasizes a larger Loss3 residual fade in `model - solver`, with residual mean-sign agreement, residual cosine, and delta cosine recorded in `visual_selection_scores.csv`.
