# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile)

- Created: 2026-06-12T11:14:58+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/loss3_upper_quartile_01_maternfine_idx25_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/loss3_upper_quartile_02_highpass_idx25_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/loss3_upper_quartile_03_rectangles_idx3_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/loss3_upper_quartile_04_highpass_idx7_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/loss3_upper_quartile_05_bandpass_idx20_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.50586e-06 | 0.179342 |
| loss1 | 5.72604e-06 | 0.187 |
| loss2 | 5.00068e-06 | 0.170265 |
| loss3 | 2.14605e-06 | 0.129776 |
| physics loss | 5.05201e-06 | 0.176968 |
