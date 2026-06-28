# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack20_five_models_loss3best10_with_losscurves)

- Created: 2026-06-12T09:28:45+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=20, epsilon_fraction=0.025.
- Samples: 10 selected samples (10 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best01_maternfine_idx30_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best02_highpass_idx33_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best03_rectangles_idx23_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best04_highpass_idx13_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best05_maternfine_idx48_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best06_bandpass_idx6_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best07_rectangles_idx20_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best08_maternsmooth_idx2_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best09_bandpass_idx7_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/loss3best10_maternsmooth_idx21_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_loss3best10_with_losscurves/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.51847e-06 | 0.178171 |
| loss1 | 5.72296e-06 | 0.184323 |
| loss2 | 4.99657e-06 | 0.169758 |
| loss3 | 1.59576e-07 | 0.0676821 |
| physics loss | 4.91077e-06 | 0.173112 |
