# Darcy Five-Model Shared-Range Attack Heatmaps (20260612_launcher_debug_index0)

- Created: 2026-06-12T10:05:13+00:00
- Models: baseline, loss1, loss2, loss3, physics loss.
- Attack: binary Darcy loss3 solver-consistent attack, steps=5, epsilon_fraction=0.025.
- Samples: 1 selected samples (1 generalization, 0 non-generalization).
- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.
- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.

## Figures

- `visualizations/darcy_five_model_attack_heatmaps_20260612_launcher_debug_index0/index0_01_maternfine_idx0_five_model_attack_heatmap_with_loss_curve.png`

## Tables

- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_launcher_debug_index0/summary.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_launcher_debug_index0/selected_samples.csv`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_launcher_debug_index0/shared_color_ranges.json`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_launcher_debug_index0/column_color_ranges_applied.json`
- vectors: `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_launcher_debug_index0/arrays`

## Mean Attack Gain By Model

| model | mean attack gain | mean adv rel L2 |
|---|---:|---:|
| baseline | 3.0075e-06 | 0.155246 |
| loss1 | 3.36662e-06 | 0.165527 |
| loss2 | 2.61311e-06 | 0.146169 |
| loss3 | 2.22656e-06 | 0.136889 |
| physics loss | 2.80336e-06 | 0.150455 |
