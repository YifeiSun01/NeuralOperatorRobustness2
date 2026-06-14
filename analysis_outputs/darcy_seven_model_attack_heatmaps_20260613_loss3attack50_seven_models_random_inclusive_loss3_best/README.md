# Darcy Seven-Model Shared-Range Attack Heatmaps (20260613_loss3attack50_seven_models_random_inclusive_loss3_best)

- Created: 2026-06-13T21:06:23+00:00
- Models: baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples.
- The two random-source training methods are included as `random clean y` and `random solver y`.
- Color ranges are shared globally across all samples/models for each semantic panel type.

## Figures

- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/loss3_best_01_maternfine_idx30_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/loss3_best_02_highpass_idx33_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/loss3_best_03_rectangles_idx23_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/loss3_best_04_highpass_idx44_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/loss3_best_05_bandpass_idx12_seven_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/summary.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/selected_samples.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/shared_color_ranges.json`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_best/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.64829e-06 | 0.181305 |
| loss1 | 5.91371e-06 | 0.188449 |
| loss2 | 5.08439e-06 | 0.171738 |
| loss3 | 2.16211e-07 | 0.0780704 |
| physics loss | 5.16336e-06 | 0.17875 |
| random clean y | 1.37338e-06 | 0.143038 |
| random solver y | 5.55873e-06 | 0.184147 |
