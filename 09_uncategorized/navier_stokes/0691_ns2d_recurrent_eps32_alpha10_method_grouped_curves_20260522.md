# NS2D Recurrent FNO eps32 alpha10 Method-Grouped Curves - 2026-05-22

## Status

Generated CPU-only method-grouped curve plots for the completed `epsilon=32`, `alpha=10` baseline pair. These plots complement the earlier block-grouped views: instead of plotting four optimizers inside each loss/mode block, each optimizer panel overlays all completed loss/mode blocks.

No model, solver, PyTorch, or JAX rerun was performed.

## Observed Evidence

- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Source files: saved `per_step_metrics.csv` and `final_state_outputs.npz` files under each completed method directory.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_method_grouped_curves.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522`.
- Report JSON: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_method_grouped_curves_report.json`.
- The command used `CUDA_VISIBLE_DEVICES=''` with `adv_robust/bin/python`; the script imports NumPy/matplotlib only.

## Generated Plots

Loss curves grouped by optimizer:

- Linear y-scale: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_loss_curves_by_method_all_blocks_linear.png`.
- Log y-scale: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_loss_curves_by_method_all_blocks_logy.png`.

Radial FFT spectra grouped by optimizer:

- Final delta: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_final-delta_radial_fft_by_method_all_blocks.png`.
- Adversarial FNO final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_adv-model-final_radial_fft_by_method_all_blocks.png`.
- Adversarial solver final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_adv-solver-final_radial_fft_by_method_all_blocks.png`.
- Adversarial FNO-solver final difference: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_adv-model-minus-solver_radial_fft_by_method_all_blocks.png`.

## Frequency Markers

The spectral plots include two vertical markers:

- `rho = sqrt(2)/3 ~= 0.4714`: the per-axis `|kx| = 1/3` or `|ky| = 1/3` cutoff seen along coordinate axes.
- `rho = 2/3 ~= 0.6667`: the diagonal corner of the `|kx| <= 1/3`, `|ky| <= 1/3` retained square.

Inference: in radial averages, a 2/3-rule square cutoff does not appear as one infinitely sharp radial location. It appears as a drop-off band between roughly `rho=0.47` and `rho=0.67`, depending on angular direction.

## Interpretation

Observed from the method-grouped views:

- The original block-grouped plots are best for comparing optimizer methods inside a fixed loss/mode setting.
- The new method-grouped plots are better for comparing how one optimizer behaves across `loss1`, `loss2`, and the multiple `loss3` modes.
- The final-delta spectral view still shows `loss2` as the smoothest/lowest-frequency perturbation family and `loss3` as the family with more mid-frequency spatial structure.
- The FNO and solver final-output spectra show the same `2/3` drop-off band, supporting the working conclusion that the trained FNO has learned or mirrored the solver/training-data spectral cutoff.
- The FNO-solver difference spectra remain more mid/high-frequency than either output alone, because the shared low-frequency structure cancels under subtraction.

## Remaining Work

These are visualization-only summaries from the completed baseline pair. Future comparison should repeat the same grouped plots for the other `epsilon:alpha` pairs after those runs finish.
