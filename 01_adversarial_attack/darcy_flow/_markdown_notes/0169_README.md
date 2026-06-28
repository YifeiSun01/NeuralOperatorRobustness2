# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack20_five_models_sharedrange_with_losscurves)

- Created: 2026-06-12T09:18:28+00:00
- Models: baseline, loss1, loss2, loss3, fixed (physics).
- Attack: binary Darcy loss3 solver-consistent attack, steps=20, epsilon_fraction=0.025.
- Samples: 1 test sample plus 4 generalization samples.
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/test_original_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/gen_smooth_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/gen_highpass_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/gen_wave_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/gen_blocky_idx0_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange_with_losscurves/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 3.63662e-06 | 0.156661 |
| loss1 | 3.69891e-06 | 0.157255 |
| loss2 | 3.29655e-06 | 0.160397 |
| loss3 | 2.5561e-06 | 0.150055 |
| fixed | 3.20832e-06 | 0.150615 |
