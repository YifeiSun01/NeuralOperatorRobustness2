# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack50_five_models_ranked5_batch_polished_index0)

- Created: 2026-06-12T11:05:55+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=50, epsilon_fraction=0.025.
- Samples: 5 selected samples (5 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/index0_01_maternfine_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/index0_02_highpass_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/index0_03_rectangles_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/index0_04_highpass_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/index0_05_bandpass_idx0_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.4563e-06 | 0.179235 |
| loss1 | 5.63138e-06 | 0.18553 |
| loss2 | 4.94252e-06 | 0.170862 |
| loss3 | 3.28231e-06 | 0.153202 |
| physics loss | 4.88892e-06 | 0.175 |
