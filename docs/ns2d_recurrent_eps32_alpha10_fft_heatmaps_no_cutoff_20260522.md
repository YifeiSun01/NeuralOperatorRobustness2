# NS2D Recurrent FNO eps32 alpha10 FFT Heatmaps Without Cutoff Overlays - 2026-05-22

## Status

Generated CPU-only FFT log-magnitude heatmaps for saved `epsilon=32`, `alpha=10` baseline fields with no cutoff box, no axis line, and no guide marker. This set is intended for visual inspection of whether the `2/3`/dealiasing spectral boundary is visible without drawing it.

No model, solver, PyTorch, or JAX rerun was performed.

## Observed Evidence

- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Source arrays: saved `final_state_outputs.npz` files under each completed method directory.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_output_fft_heatmaps_no_cutoff.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522`.
- Report JSON: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_fft_heatmaps_no_cutoff_sample0_report.json`.
- The command used `CUDA_VISIBLE_DEVICES=''` with `adv_robust/bin/python`; the script imports NumPy/matplotlib only.

## Existing Model/Solver Output FFT Plots With Cutoff Overlay

The previous overlayed output FFT plots are in `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522`:

- Perturbed model output: `eps32_alpha10_adv-model-final_fft_log_magnitude_sample0_grid.png`.
- Perturbed solver output: `eps32_alpha10_adv-solver-final_fft_log_magnitude_sample0_grid.png`.
- Perturbed model-solver difference: `eps32_alpha10_adv-model-minus-solver_fft_log_magnitude_sample0_grid.png`.
- Clean model output: `eps32_alpha10_clean-model-final_fft_log_magnitude_sample0_grid.png`.
- Clean solver output: `eps32_alpha10_clean-solver-final_fft_log_magnitude_sample0_grid.png`.
- Clean model-solver difference: `eps32_alpha10_clean-model-minus-solver_fft_log_magnitude_sample0_grid.png`.

## New No-Cutoff Heatmaps

These new heatmaps intentionally have no drawn `2/3` cutoff box/line:

- Clean initial condition: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_x-clean_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Perturbed initial condition: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_x-adv_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Final perturbation delta: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_final-delta_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Clean FNO final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_clean-model-final_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Clean solver final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_clean-solver-final_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Clean FNO-solver difference: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_clean-model-minus-solver_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Perturbed FNO final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_adv-model-final_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Perturbed solver final output: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_adv-solver-final_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Perturbed FNO-solver difference: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_adv-model-minus-solver_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- FNO final change, adv-clean: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_model-final-change_fft_log_magnitude_sample0_no_cutoff_grid.png`.
- Solver final change, adv-clean: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522/eps32_alpha10_solver-final-change_fft_log_magnitude_sample0_no_cutoff_grid.png`.

## Interpretation

Observed evidence: this plot set is visual-only and deliberately avoids guide lines. It should be used side-by-side with the overlayed plots and CSV metrics from `docs/ns2d_recurrent_eps32_alpha10_output_fft_dealias_analysis_20260522.md`.

Inference: if the boundary remains visible in these no-cutoff plots, then the apparent dealiasing boundary is not caused by the plotted guide box. The previous quantitative metrics already showed tiny outside-cutoff energy in model/solver outputs and axis-aligned residual energy in final deltas; these no-cutoff figures are the corresponding visual sanity check.

## Corrected Visual Interpretation

Observed from visual inspection of the no-cutoff heatmaps: the clean and adversarial FNO/model output FFT heatmaps do not show a clearly visible `2/3` cutoff box in the same way that the solver output and solver-change heatmaps do. The previous statement that the trained FNO output visibly mirrors the solver cutoff should therefore be weakened.

Corrected interpretation:

- The numerical outside-cutoff fractions for model outputs are small, but that does not by itself mean a sharp visual cutoff box is present.
- The most visually clear cutoff-box evidence appears in solver-related fields, especially solver final output and solver final change.
- The final perturbation delta can inherit the cutoff because the attack gradient passes through solver-generated frames or solver targets in the differentiable path. In a pseudo-spectral solver with dealiasing, the forward projection/mask also shapes the adjoint/backward gradient.
- `loss1/all_w` can still show the solver cutoff fingerprint because the target is `F(x)`, but the recurrent FNO input frames 2-10 are generated by the solver in `w` mode.
- `loss3` can show the fingerprint because its target path includes `G(x+delta)`, and the current modes keep target frame 20 in `w` mode.
- `loss2/all_a_target_w` is different: its active objective uses dictionary-provided input frames and a fixed clean target, so it does not contain the same differentiable perturbed-solver path. This explains why its final delta can lack the same obvious cutoff-box structure.

Terminology note: the visible box is the retained-mode `2/3` cutoff associated with the pseudo-spectral `3/2` dealiasing rule. The user sometimes calls it the `3/2` line; in these notes it is recorded as the `2/3 cutoff / 3/2-rule fingerprint`.

Additional interpretation from the initial-condition FFT: the clean initial condition appears lower-band-limited, consistent with being generated from a lower-resolution field and upsampled to `256 x 256`. The adversarial initial condition `x_adv = x_clean + delta` can therefore look like a superposition of the original low-frequency initial support and the solver-gradient-shaped `2/3` cutoff structure in `delta`.
