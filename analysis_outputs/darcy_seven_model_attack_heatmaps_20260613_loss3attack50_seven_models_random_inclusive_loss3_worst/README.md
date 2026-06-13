# Darcy Seven-Model Shared-Range Attack Heatmaps (20260613_loss3attack50_seven_models_random_inclusive_loss3_worst)

- Created: 2026-06-13T11:48:01+00:00
- Models: baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples.
- The two random-source training methods are included as `random clean y` and `random solver y`.
- Color ranges are shared globally across all samples/models for each semantic panel type.

## Figures

- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/loss3_worst_01_maternfine_idx14_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/loss3_worst_02_highpass_idx21_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/loss3_worst_03_rectangles_idx41_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/loss3_worst_04_highpass_idx9_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/loss3_worst_05_bandpass_idx4_seven_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/summary.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/selected_samples.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/shared_color_ranges.json`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_worst/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 4.44955e-06 | 0.168383 |
| loss1 | 5.69436e-06 | 0.18751 |
| loss2 | 3.18041e-06 | 0.161355 |
| loss3 | 3.65047e-06 | 0.161216 |
| physics loss | 4.99698e-06 | 0.179175 |
| random clean y | 2.7499e-06 | 0.15389 |
| random solver y | 5.40707e-06 | 0.184831 |
