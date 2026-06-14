# Darcy Seven-Model Shared-Range Attack Heatmaps (20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed)

- Created: 2026-06-13T21:07:21+00:00
- Models: baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples.
- The two random-source training methods are included as `random clean y` and `random solver y`.
- Color ranges are shared globally across all samples/models for each semantic panel type.

## Figures

- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/loss3_top5pct_visual_relaxed_01_maternfine_idx32_rank05_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/loss3_top5pct_visual_relaxed_02_highpass_idx0_rank23_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/loss3_top5pct_visual_relaxed_03_rectangles_idx18_rank04_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/loss3_top5pct_visual_relaxed_04_highpass_idx0_rank22_seven_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/loss3_top5pct_visual_relaxed_05_bandpass_idx20_rank13_seven_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/summary.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/selected_samples.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/shared_color_ranges.json`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_loss3_top5pct_visual_relaxed/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.30698e-06 | 0.179821 |
| loss1 | 5.57105e-06 | 0.187645 |
| loss2 | 4.86337e-06 | 0.171124 |
| loss3 | 3.10517e-06 | 0.15098 |
| physics loss | 4.86336e-06 | 0.177409 |
| random clean y | 2.65216e-06 | 0.153206 |
| random solver y | 5.28612e-06 | 0.183297 |
