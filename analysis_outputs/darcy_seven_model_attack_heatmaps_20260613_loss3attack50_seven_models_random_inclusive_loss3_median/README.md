# Darcy Seven-Model Shared-Range Attack Heatmaps (20260613_loss3attack50_seven_models_random_inclusive_loss3_median)

- Created: 2026-06-13T11:42:17+00:00
- Models: baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples.
- The two random-source training methods are included as `random clean y` and `random solver y`.
- Color ranges are shared globally across all samples/models for each semantic panel type.

## Figures

- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/loss3_median_01_maternfine_idx15_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/loss3_median_02_highpass_idx1_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/loss3_median_03_rectangles_idx28_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/loss3_median_04_highpass_idx10_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/loss3_median_05_bandpass_idx42_seven_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/summary.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/selected_samples.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/shared_color_ranges.json`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_median/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.40194e-06 | 0.17808 |
| loss1 | 5.65997e-06 | 0.185332 |
| loss2 | 4.90142e-06 | 0.169102 |
| loss3 | 3.33791e-06 | 0.151635 |
| physics loss | 4.98604e-06 | 0.176262 |
| random clean y | 3.43524e-06 | 0.156746 |
| random solver y | 5.3592e-06 | 0.181312 |
