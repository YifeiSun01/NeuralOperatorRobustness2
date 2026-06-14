# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_loss3attack20_five_models_ranked10_index0_polished)

- Created: 2026-06-12T09:42:23+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=20, epsilon_fraction=0.025.
- Samples: 10 selected samples (10 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_01_maternfine_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_02_highpass_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_03_rectangles_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_04_highpass_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_05_bandpass_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_06_maternfine_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_07_rectangles_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_08_maternsmooth_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_09_maternfine_idx0_five_model_attack_heatmap_with_loss_curve.png`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/index0_10_maternsmooth_idx0_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_ranked10_index0_polished/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 5.19664e-06 | 0.178842 |
| loss1 | 5.32189e-06 | 0.184916 |
| loss2 | 4.67404e-06 | 0.170597 |
| loss3 | 3.19136e-06 | 0.150745 |
| physics loss | 4.59998e-06 | 0.173392 |
