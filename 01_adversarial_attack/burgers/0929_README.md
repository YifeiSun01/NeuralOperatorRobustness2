# Polished Burgers Adversarial Training Visualizations

## Checkpoint-Style Hybrid Figures

These figures follow the checkpoint-summary layout: a large absolute-value heatmap on top and a short, wide lineplot underneath. Loss lineplots use group means at the same 11 checkpoints and do not use epoch moving average. The FFT figure keeps the heatmap raw and uses 25-epoch moving average only for the bottom selected spectra.

- `polished_checkpoint_style_relative_l2_absolute11_heatmap_line_below.png`
- `polished_checkpoint_style_rmse_absolute11_heatmap_line_below.png`
- `polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png`
