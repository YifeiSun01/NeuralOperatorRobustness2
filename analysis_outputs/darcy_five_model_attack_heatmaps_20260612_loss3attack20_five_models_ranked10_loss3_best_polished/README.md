# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack20_five_models_ranked10_loss3_best_polished)

- Created: 2026-06-12T09:48:57+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=20, epsilon_fraction=0.025.
- Samples: 10 selected samples (10 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_01_maternfine_idx30_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_02_highpass_idx33_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_03_rectangles_idx23_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_04_highpass_idx13_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_05_bandpass_idx6_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_06_maternfine_idx48_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_07_rectangles_idx20_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_08_maternsmooth_idx2_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_09_maternfine_idx33_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/loss3_best_10_maternsmooth_idx26_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_loss3_best_polished/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.49917e-06 | 0.178229 |
| loss1 | 5.71649e-06 | 0.185326 |
| loss2 | 4.99701e-06 | 0.169847 |
| loss3 | 1.94494e-07 | 0.071677 |
| physics loss | 4.9771e-06 | 0.175265 |
