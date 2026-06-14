# Darcy Seven-Model Shared-Range Attack Heatmaps (20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile)

- Created: 2026-06-13T21:06:33+00:00
- Models: baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples.
- The two random-source training methods are included as `random clean y` and `random solver y`.
- Color ranges are shared globally across all samples/models for each semantic panel type.

## Figures

- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/loss3_lower_quartile_01_maternfine_idx23_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/loss3_lower_quartile_02_highpass_idx39_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/loss3_lower_quartile_03_rectangles_idx8_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/loss3_lower_quartile_04_highpass_idx34_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/loss3_lower_quartile_05_bandpass_idx43_seven_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/summary.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/selected_samples.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/shared_color_ranges.json`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_lower_quartile/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.5825e-06 | 0.18093 |
| loss1 | 5.77733e-06 | 0.187175 |
| loss2 | 4.11367e-06 | 0.164013 |
| loss3 | 2.81196e-06 | 0.138106 |
| physics loss | 5.07661e-06 | 0.178847 |
| random clean y | 2.10144e-06 | 0.147625 |
| random solver y | 5.47489e-06 | 0.184107 |
