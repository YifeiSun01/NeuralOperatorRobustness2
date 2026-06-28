# NS2D Recurrent FNO eps32 alpha10 Output FFT Dealiasing Analysis - 2026-05-22

## Status

Generated CPU-only FFT/dealiasing analysis for saved final perturbations and final outputs from the completed `epsilon=32`, `alpha=10` baseline pair. No model, solver, PyTorch, or JAX rerun was performed.

## Observed Evidence

- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Source arrays: each method's saved `final_state_outputs.npz`.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_output_fft_dealias_analysis.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_output_fft_dealias_metrics_summary.csv`.
- Report JSON: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_output_fft_dealias_analysis_report.json`.
- The command used `CUDA_VISIBLE_DEVICES=''` with `adv_robust/bin/python`; the script imports NumPy/matplotlib only and does not use GPU.

## Generated Plots

Each log-magnitude FFT grid overlays a cyan dashed square at `|kx| <= 1/3`, `|ky| <= 1/3`, corresponding to the per-axis `2/3` retained-width cutoff.

- Final delta FFT: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_final-delta_fft_log_magnitude_sample0_grid.png`.
- Clean FNO final FFT: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_clean-model-final_fft_log_magnitude_sample0_grid.png`.
- Clean solver final FFT: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_clean-solver-final_fft_log_magnitude_sample0_grid.png`.
- Clean FNO-solver diff FFT: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_clean-model-minus-solver_fft_log_magnitude_sample0_grid.png`.
- Adv FNO final FFT: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_adv-model-final_fft_log_magnitude_sample0_grid.png`.
- Adv solver final FFT: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_adv-solver-final_fft_log_magnitude_sample0_grid.png`.
- Adv FNO-solver diff FFT: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_adv-model-minus-solver_fft_log_magnitude_sample0_grid.png`.

## Dealiasing Code Evidence

Observed from source code:

- Training-data solver generation sets `dealiasing_fraction=2 / 3` in `2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py` line 97.
- Dictionary generation sets `dealiasing_fraction=2 / 3` in `2D_NS_FNO2d_recurrent/data_generation/generate_ns_dictionary_batched.py` line 120.
- Attack `model_prediction` rolls out solver frames whenever mode characters are `d` or `w` in `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` lines 605-648.
- Active losses are `loss1 = ||F(x+delta)-F(x)||`, `loss2 = ||F(x+delta)-G(x)||`, and `loss3 = ||F(x+delta)-G(x+delta)||` in `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` lines 666-679.

Important implication: `loss1/all_w` does not use the solver as a final target, but it still uses solver-generated frames 2-10 as part of the FNO input path. Therefore `loss1/all_w` can inherit solver dealiasing structure in the gradient path.

## Metric Definitions

- `square_outside_frac`: fraction of non-DC FFT power outside the per-axis `|kx| <= 1/3`, `|ky| <= 1/3` square.
- `cross_outside_frac`: fraction of non-DC FFT power outside that square but lying near a coordinate axis, using `|kx| <= 4/256` or `|ky| <= 4/256`.
- `cross_share_of_outside`: `cross_outside_frac / square_outside_frac`.
- `spectral_centroid`: power-weighted normalized radial frequency, excluding DC.

## Aggregate Findings

Observed from `eps32_alpha10_output_fft_dealias_metrics_summary.csv`:

| Field | Loss | outside 2/3 square | outside cross arms | cross share outside | centroid | mid frac | high frac |
|---|---|---:|---:|---:|---:|---:|---:|
| `final_delta` | `loss1` | 1.361e-10 | 1.187e-10 | 0.852 | 0.01396 | 2.898e-04 | 8.479e-08 |
| `final_delta` | `loss2` | 1.183e-06 | 1.022e-06 | 0.836 | 0.01196 | 2.103e-05 | 2.124e-06 |
| `final_delta` | `loss3` | 3.490e-10 | 2.900e-10 | 0.835 | 0.01969 | 1.837e-03 | 1.012e-06 |
| `adv_model_final` | `loss1` | 1.233e-06 | 5.491e-08 | 0.075 | 0.01279 | 1.727e-03 | 1.796e-05 |
| `adv_model_final` | `loss2` | 1.494e-06 | 3.922e-08 | 0.042 | 0.01290 | 2.192e-03 | 2.181e-05 |
| `adv_model_final` | `loss3` | 1.148e-06 | 5.266e-08 | 0.091 | 0.01285 | 1.753e-03 | 1.492e-05 |
| `adv_solver_final` | `loss1` | 1.780e-06 | 9.736e-08 | 0.065 | 0.01392 | 3.767e-03 | 2.162e-04 |
| `adv_solver_final` | `loss2` | 1.886e-06 | 2.688e-08 | 0.033 | 0.01403 | 4.804e-03 | 1.922e-04 |
| `adv_solver_final` | `loss3` | 2.351e-06 | 8.995e-08 | 0.073 | 0.01434 | 4.451e-03 | 2.271e-04 |
| `adv_model_minus_solver` | `loss1` | 7.191e-05 | 3.284e-06 | 0.057 | 0.06652 | 7.946e-02 | 5.505e-03 |
| `adv_model_minus_solver` | `loss2` | 1.271e-04 | 2.540e-06 | 0.020 | 0.07915 | 1.088e-01 | 7.275e-03 |
| `adv_model_minus_solver` | `loss3` | 7.519e-05 | 2.926e-06 | 0.069 | 0.06140 | 6.247e-02 | 3.866e-03 |

## Interpretation

Observed evidence supports the user's visual observation:

- FNO final outputs and solver final outputs both have extremely small power outside the `2/3` cutoff square, around `1e-6` to `2e-6` outside-square fraction. This suggests the trained model has learned, or at least strongly mirrors, the solver/training-data spectral cutoff pattern.
- The model is not an exact hard dealiased projector: it has tiny outside-square leakage. But the leakage is small enough that the cutoff box is a real spectral feature, not just plotting noise.
- The FNO-solver difference has much more mid/high-frequency content than either output alone. This is expected because subtracting two similar low-frequency fields cancels shared low-frequency structure and emphasizes mismatch frequencies.
- For `final_delta`, the energy outside the `2/3` square is tiny. However, when outside-square energy exists, about `83%` to `85%` of it lies in coordinate-axis cross arms. This supports the observation of a faint cross structure extending outside the cutoff square.

Inference:

- `loss1/all_w` can show the `2/3` rule fingerprint because, although its objective target is `F(x)` rather than `G(x+delta)`, its FNO input frames 2-10 are generated through the solver in `w` mode. The solver path can therefore shape the attack gradient.
- `loss3` shows the fingerprint even more naturally because it uses the perturbed solver target for `G(x+delta)` when target mode is `w`, and many loss3 modes also use solver-generated input frames.
- `loss2/all_a_target_w` is different: its active objective uses a fixed clean target `G(x)` and dictionary-provided input frames rather than differentiating through a perturbed solver target. This helps explain why it appears smoother/low-frequency and lacks the same obvious cutoff-box boundary, even though it still has faint axis-cross leakage.
- The axis-cross structure should be interpreted carefully: the absolute outside-square energy is very small in `final_delta`, so log-magnitude plots make a tiny residual structure visually prominent. Still, the high `cross_share_of_outside` means the residual is not random; it is strongly axis-aligned.

## Working Conclusion

The saved evidence supports the hypothesis that the `2/3` dealiasing rule is visible not only in the solver outputs but also in the trained FNO outputs and in the attack perturbations. The most plausible explanation is a combination of solver-generated training targets, solver-generated recurrent input frames in `w`/`d` modes, the FNO's own spectral bias, and gradient propagation through these low-pass structures. The FNO-solver difference is the field where mid/high-frequency mismatch is most visible.

## Correction After No-Cutoff Visual Inspection

The previous interpretation that the FNO/model final outputs visibly show the same `2/3` cutoff as the solver should be treated cautiously. The no-cutoff heatmaps show that the model outputs have low outside-cutoff energy, but the sharp box is not visually obvious in the clean or adversarial model-output FFTs. The stronger visual cutoff evidence is in the solver outputs, solver changes, and in final deltas whose gradients pass through solver/dealiasing paths.

This distinction matters: small outside-cutoff energy is a quantitative spectral property, while a visible cutoff box is a stronger visual claim. The latter is clearly supported for solver-related fields and final deltas in solver-gradient-heavy losses, but not clearly supported for the model outputs alone.
