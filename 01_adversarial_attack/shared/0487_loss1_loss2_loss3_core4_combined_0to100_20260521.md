# Loss1/Loss2/Loss3 Combined Core-Four 0..100 Visuals - 2026-05-21

Status: generated from existing completed outputs. No attack was rerun.

## Source Data

- Loss1/Loss2 source: `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2`
- Loss3 source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2`
- Combined output: `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521`
- Scope: FNO / 1D Burgers `nu=0.001`, `epsilon=4`, `alpha=0.4`, `p=q=2`, `k=0..100`.

## Figures

- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_mean_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_mean_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_mean_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_mean_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_normalized_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_normalized_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_normalized_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_normalized_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_boundary_ratio_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_boundary_ratio_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_boundary_ratio_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_boundary_ratio_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_pnorm_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_pnorm_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_pnorm_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_pnorm_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_prev_angle_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_prev_angle_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_prev_angle_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_prev_angle_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_direction_prev_angle_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_direction_prev_angle_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_direction_prev_angle_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_direction_prev_angle_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_direction_angle_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_direction_angle_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_direction_angle_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_delta_direction_angle_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_high_frequency_ratio_no_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_high_frequency_ratio_no_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_high_frequency_ratio_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_high_frequency_ratio_with_std_by_objective_0to100.png`

Figure naming convention:
- `*_no_std_by_method_*.png`: four method panels; each panel overlays `loss1`, `loss2`, and `loss3` as mean lines only.
- `*_with_std_by_method_*.png`: same method-panel layout with mean +/- std shading.
- `*_no_std_by_objective_*.png`: three objective rows; each row overlays the four optimizer methods as mean lines only.
- `*_with_std_by_objective_*.png`: same objective-row layout with mean +/- std shading. Rows share the same x/y axis ranges within each metric figure.

## Observed Boundary-Gain Evidence

Observed from `combined_boundary_gain_summary_0to100.csv`, using first mean boundary ratio >= 0.99.

| objective | method | first_boundary_k | loss_at_first_boundary_mean | final_loss_mean | gain_boundary_to_final | steps_after_boundary |
| --- | --- | --- | --- | --- | --- | --- |
| loss1 | raw_add | 8 | 6.767 | 6.977 | 0.2101 | 92 |
| loss2 | raw_add | 8 | 6.892 | 7.07 | 0.1774 | 92 |
| loss3 | raw_add | 43 | 2.623 | 2.877 | 0.2545 | 57 |
| loss1 | raw_replace | 1 | 6.221 | 6.942 | 0.7203 | 99 |
| loss2 | raw_replace | 1 | 6.352 | 7.045 | 0.6935 | 99 |
| loss3 | raw_replace | 1 | 1.569 | 3.047 | 1.478 | 99 |
| loss1 | steepest_add | 11 | 6.792 | 6.977 | 0.1851 | 89 |
| loss2 | steepest_add | 11 | 6.913 | 7.074 | 0.1609 | 89 |
| loss3 | steepest_add | 11 | 2.426 | 2.988 | 0.5622 | 89 |
| loss1 | steepest_replace | 1 | 6.221 | 6.942 | 0.7203 | 99 |
| loss2 | steepest_replace | 1 | 6.352 | 7.045 | 0.6935 | 99 |
| loss3 | steepest_replace | 1 | 1.569 | 3.047 | 1.478 | 99 |

## Final/Last-Finite Summary

| objective | method | final_loss_mean | final_boundary_ratio_mean | final_delta_pnorm_mean | last_finite_delta_prev_angle_k | last_finite_delta_prev_angle_degrees_mean | last_finite_direction_prev_angle_k | last_finite_direction_prev_angle_degrees_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | raw_add | 6.977 | 1 | 4 | 100 | 0.04206 | 99 | 0.05031 |
| loss2 | raw_add | 7.07 | 1 | 4 | 100 | 0.02814 | 99 | 0.03037 |
| loss3 | raw_add | 2.877 | 1 | 4 | 100 | 0.1888 | 100 | 0.383 |
| loss1 | raw_replace | 6.942 | 1 | 4 | 100 | 6.379 | 99 | 6.379 |
| loss2 | raw_replace | 7.045 | 1 | 4 | 100 | 5.127 | 99 | 5.127 |
| loss3 | raw_replace | 3.047 | 1 | 4 | 100 | 32.69 | 100 | 36.62 |
| loss1 | steepest_add | 6.977 | 1 | 4 | 100 | 0.03885 | 99 | 0.03296 |
| loss2 | steepest_add | 7.074 | 1 | 4 | 100 | 0.02423 | 99 | 0.016 |
| loss3 | steepest_add | 2.988 | 1 | 4 | 100 | 0.1328 | 100 | 4.065 |
| loss1 | steepest_replace | 6.942 | 1 | 4 | 100 | 6.379 | 99 | 6.379 |
| loss2 | steepest_replace | 7.045 | 1 | 4 | 100 | 5.127 | 99 | 5.127 |
| loss3 | steepest_replace | 3.047 | 1 | 4 | 100 | 32.69 | 100 | 36.62 |

## Notes

Both orientations are preserved: `by_method` keeps one panel per optimizer method, and `by_objective` keeps one panel per objective with four method curves.
Both no-std and with-std versions are generated. The with-std plots draw mean +/- std as translucent bands.
The `combined_loss_normalized_*_0to100.png` plots normalize each objective/method curve by its own max over `k=0..100`; use them to compare shape rather than scale.
Loss/boundary/delta-norm summary values are from `k=100`; angle summary values use the last finite value in `k=0..100`, because the final logged row can have no following direction for a step-to-step comparison.
Loss3 angle means for direction/delta-direction are derived from the saved cosine columns in the Loss3 baseline CSV; their shaded std bands are obtained by converting cosine mean +/- std into an approximate angle std band.
Inference from the overlaid curves and boundary-gain table: under this baseline p=q=2 setting, loss1/loss2 reach the epsilon boundary early and then add only a small amount of objective value by k=100; the normalized plot is the cleanest way to compare this shape against loss3 because the raw objective scales differ.
