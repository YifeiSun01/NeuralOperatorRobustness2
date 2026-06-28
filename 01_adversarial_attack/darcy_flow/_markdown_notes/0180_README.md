# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed)

- Created: 2026-06-12T22:44:56+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/loss3_top10pct_visual_relaxed_01_maternfine_idx25_rank13_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/loss3_top10pct_visual_relaxed_02_highpass_idx39_rank38_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/loss3_top10pct_visual_relaxed_03_rectangles_idx0_rank12_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/loss3_top10pct_visual_relaxed_04_highpass_idx34_rank38_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/loss3_top10pct_visual_relaxed_05_bandpass_idx42_rank25_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.51771e-06 | 0.178963 |
| loss1 | 5.78535e-06 | 0.186411 |
| loss2 | 5.03332e-06 | 0.170208 |
| loss3 | 3.45495e-06 | 0.154071 |
| physics loss | 5.11462e-06 | 0.177839 |

## Visual Relaxed Selection

These samples are selected by a hard visual filter: delta cosine mean >= 0.70, delta cosine min >= 0.68, residual ratio min >= 1.16, and residual sign agreement = 1 across loss1/loss2/physics. Within that filter, higher Loss3 margin and clearer residual fade are preferred.
