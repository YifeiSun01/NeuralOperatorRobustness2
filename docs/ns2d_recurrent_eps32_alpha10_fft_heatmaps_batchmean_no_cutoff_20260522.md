# NS2D Recurrent FNO eps32 alpha10 Batch-Mean FFT Heatmaps Without Cutoff Overlays - 2026-05-22

## Status

Generated CPU-only batch-mean FFT log-magnitude heatmaps for saved `epsilon=32`, `alpha=10` baseline fields with no cutoff box, no axis line, and no guide marker.

This corrects the previous sample-0-only view. For each field, method, and loss/mode block, the script computes FFT per sample, takes the magnitude, averages magnitudes over the 10 saved samples, and then plots `log10(mean |FFT|)`. It does not average fields before FFT, because that would allow phase cancellation.

No model, solver, PyTorch, or JAX rerun was performed.

## Observed Evidence

- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Source arrays: saved `final_state_outputs.npz` files under each completed method directory.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_output_fft_heatmaps_no_cutoff_batchmean.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522`.
- Report JSON: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_report.json`.
- Averaging method: `FFT per sample -> magnitude -> arithmetic mean over 10 samples -> log10`.
- The command used `CUDA_VISIBLE_DEVICES=''` with `adv_robust/bin/python`; the script imports NumPy/matplotlib only.

## Generated No-Cutoff Batch-Mean Heatmaps

- Clean initial condition: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_x-clean_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Perturbed initial condition: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_x-adv_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Final perturbation delta: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_final-delta_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Clean FNO final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_clean-model-final_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Clean solver final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_clean-solver-final_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Clean FNO-solver difference: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_clean-model-minus-solver_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Perturbed FNO final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_adv-model-final_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Perturbed solver final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_adv-solver-final_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Perturbed FNO-solver difference: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_adv-model-minus-solver_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- FNO final change, adv-clean: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_model-final-change_fft_log_magnitude_batchmean_no_cutoff_grid.png`.
- Solver final change, adv-clean: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522/eps32_alpha10_solver-final-change_fft_log_magnitude_batchmean_no_cutoff_grid.png`.

## Interpretation

Observed evidence: these plots are the 10-sample batch-mean counterpart to the previous sample-0 no-cutoff heatmaps. They should be used to decide whether the spectral features are robust across the saved attack batch rather than artifacts of one initial condition.

Inference: averaging FFT magnitudes across the 10 samples should preserve common spectral support, preferred directions, and cutoff/axis structures while reducing sample-specific speckle. The clean/adv model output, clean/adv solver output, final delta, and model-solver difference can now be visually compared under the same no-overlay rule.
