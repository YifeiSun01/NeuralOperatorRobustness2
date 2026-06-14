# Darcy Seven-Model Shared-Range Attack Heatmaps (20260613_loss3attack50_seven_models_random_inclusive_index0)

- Created: 2026-06-13T21:06:13+00:00
- Models: baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples.
- The two random-source training methods are included as `random clean y` and `random solver y`.
- Color ranges are shared globally across all samples/models for each semantic panel type.

## Figures

- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/index0_01_maternfine_idx0_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/index0_02_highpass_idx0_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/index0_03_rectangles_idx0_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/index0_04_highpass_idx0_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/index0_05_bandpass_idx0_seven_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/summary.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/selected_samples.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/shared_color_ranges.json`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_index0/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.4563e-06 | 0.179235 |
| loss1 | 5.63844e-06 | 0.185633 |
| loss2 | 4.94042e-06 | 0.170859 |
| loss3 | 3.29974e-06 | 0.153546 |
| physics loss | 4.88844e-06 | 0.174998 |
| random clean y | 1.36391e-06 | 0.145004 |
| random solver y | 5.37815e-06 | 0.181965 |
