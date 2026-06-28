# Loss1/Loss2 Core-Four P2Q2 Baseline Marked/Angle Visuals - 2026-05-21

Status: completed on GPU from a new baseline run.

## Source Data

- Output root: `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2`
- Per-method source files: `per_step_metrics.csv`, `per_sample_step_metrics.csv`, `trajectory_samples.npz`, and `final_delta.npz`.
- Boundary-marker summary: `boundary_marker_hits_0to100.csv`.
- Angle summary: `angle_summary_0to100.csv`.
- GPU evidence: `Tesla V100-SXM2-32GB`, capability `sm_70`, PyTorch `2.8.0+cu126`, JAX backend `gpu`.

## Settings

- FNO / 1D Burgers `nu=0.001`; batch size `100`, dataset indices `0..99`.
- `epsilon=4.0`, `alpha=0.4`, `steps=100`, `p=q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- `loss1_original` uses tiny random start `1e-6`; `loss2_original` uses zero start.
- The `figures_0to100_marked_angles/` plots are explicitly restricted to `k=0..100`.
- Loss curves mark boundary arrivals at `25%`, `50%`, `75%`, and `99%` of the L2 budget. `100%` is not separately marked.
- Angle columns recorded in `per_step_metrics.csv`: `delta_prev_angle_degrees`, `direction_prev_angle_degrees`, and `delta_direction_angle_degrees`.

## Summary Table

| objective | method | final_optimized_loss_mean | final_fraction_of_best | step_to_95pct_best_final | final_delta_pnorm_mean | final_high_frequency_energy_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- |
| loss1_original | raw_add | 6.977 | 1 | 7 | 4 | 4.177e-10 |
| loss1_original | raw_replace | 6.942 | 0.9949 | 2 | 4 | 4.079e-10 |
| loss1_original | steepest_add | 6.977 | 0.9999 | 10 | 4 | 4.443e-10 |
| loss1_original | steepest_replace | 6.942 | 0.9949 | 2 | 4 | 4.079e-10 |
| loss2_original | raw_add | 7.07 | 0.9994 | 7 | 4 | 6.98e-10 |
| loss2_original | raw_replace | 7.045 | 0.9959 | 2 | 4 | 6.727e-10 |
| loss2_original | steepest_add | 7.074 | 1 | 10 | 4 | 7.18e-10 |
| loss2_original | steepest_replace | 7.045 | 0.9959 | 2 | 4 | 6.727e-10 |

## Figures

- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss1_original_mean_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss1_original_boundary_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss1_original_high_frequency_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss1_original_final_delta_selected_indices.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss1_original_delta_heatmap_index0.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss1_original_mean_vs_step_0to100_boundary_marked.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss1_original_boundary_ratio_0to100_marked.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss1_original_delta_prev_angle_degrees_0to100.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss1_original_direction_prev_angle_degrees_0to100.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss1_original_delta_direction_angle_degrees_0to100.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss2_original_mean_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss2_original_boundary_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss2_original_high_frequency_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss2_original_final_delta_selected_indices.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures/loss2_original_delta_heatmap_index0.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss2_original_mean_vs_step_0to100_boundary_marked.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss2_original_boundary_ratio_0to100_marked.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss2_original_delta_prev_angle_degrees_0to100.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss2_original_direction_prev_angle_degrees_0to100.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/loss2_original_delta_direction_angle_degrees_0to100.png`

## Observed Evidence

Observed from the generated CSVs and figures: replacement methods jump to the boundary immediately by construction, while additive methods grow the perturbation over many steps. For `p=q=2`, `raw_replace` and `steepest_replace` use the same normalized direction for these objectives and therefore are expected to nearly overlap.

Observed boundary-hit table: replacement methods hit the `99%` boundary at step `1`; `raw_add` hits it at step `8`; `steepest_add` hits it at step `11` for both `loss1_original` and `loss2_original`.

Observed angle table: step-to-step delta angles are now recorded for every step. The compact source table is `angle_summary_0to100.csv`; full per-step means/stds are in each method's `per_step_metrics.csv`.

Inference should stay within this single baseline setting unless more epsilon/alpha or P/Q combinations are run.
