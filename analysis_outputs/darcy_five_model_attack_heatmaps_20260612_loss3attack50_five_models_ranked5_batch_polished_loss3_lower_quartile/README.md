# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile)

- Created: 2026-06-12T11:19:13+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/loss3_lower_quartile_01_maternfine_idx23_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/loss3_lower_quartile_02_highpass_idx39_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/loss3_lower_quartile_03_rectangles_idx8_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/loss3_lower_quartile_04_highpass_idx34_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/loss3_lower_quartile_05_bandpass_idx43_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.58327e-06 | 0.18098 |
| loss1 | 5.77772e-06 | 0.18712 |
| loss2 | 4.11463e-06 | 0.164183 |
| loss3 | 2.81585e-06 | 0.138195 |
| physics loss | 5.092e-06 | 0.179586 |
