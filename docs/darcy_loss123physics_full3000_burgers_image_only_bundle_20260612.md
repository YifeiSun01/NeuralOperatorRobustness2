# Darcy Full 0..3000 Burgers-Style Image-Only Bundle - 2026-06-12

Status: generated a Darcy Flow bundle matching the Burgers image-only bundle structure, with Darcy-specific 2D heatmap replacements for attack/sample views.

- Bundle: `visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612`
- Work/stitched data: `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work`
- Top-level folders: `comparison_dense`, `loss1`, `loss2`, `loss3`, `physics`
- PNG count: `136`
- Non-PNG files in bundle: `0`
- Epoch range: `0..3000`
- Copied Darcy 2D attack heatmap PNGs: `50`
- Generated per-generalization 5x5 dense panels: `8`

Important correction: previous Darcy figures were split across first-stage and stage2 directories. This bundle first stitches the formal runs into 0..3000 posthoc visualization runs, then regenerates the per-method and comparison figures.

## Main Locations

- `comparison_dense`: `visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612/comparison_dense`
- `loss1`: `visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612/loss1`
- `loss2`: `visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612/loss2`
- `loss3`: `visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612/loss3`
- `physics`: `visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612/physics`

## Notes

The image-only bundle contains PNG files only. CSV/JSON/Markdown and symlinked NPZ probe files are kept in the work directory, not inside the bundle.
The Burgers p=2,q=2 one-dimensional line panels are represented here by Darcy five-model two-dimensional attack heatmaps with shared color ranges and loss-growth curves.

## R2 Location

Uploaded under:

```text
bucket: neural-operator-robustness
prefix: machine-sync/NeuralOperatorRobustness2-selected/darcy_cflow_20260612
visualization bundle: visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612
stitched work data: analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work
```

R2 verification after upload:

```text
image-only bundle: 136 PNG files locally after top-percentile heatmap addition; R2 object counts are recorded in `docs/darcy_cflow_r2_sync_20260612.md`
stitched work directory: 202 objects, 655.146 MiB
```

## 2026-06-12 Top-Percentile Heatmap Addition

Added four additional Darcy 2D attack heatmap folders under `comparison_dense/darcy_2d_attack_heatmaps`:

```text
loss3attack50_five_models_ranked5_batch_polished_loss3_top5pct_visual
loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual
loss3attack50_five_models_ranked5_batch_polished_loss3_top5pct_visual_relaxed
loss3attack50_five_models_ranked5_batch_polished_loss3_top10pct_visual_relaxed
```

The relaxed folders are the recommended presentation folders when the desired visual criterion is: Loss3 delta has broadly similar shape to the other trained models while Loss3 `model - solver` residual is visibly lighter.
