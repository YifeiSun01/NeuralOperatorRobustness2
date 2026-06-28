# Loss3 Current Mechanism Validation Summary - 2026-05-21

Status: generated from current core4/PQ landscape, tangent, stepwise, and targeted Jacobian/SVD outputs.

## Source Outputs

- Landscape output: `forensics/loss3_core4_pq_landscape_probe_full_20260521`
- Early trajectory landscape output: `forensics/loss3_core4_pq_landscape_probe_trajectory_20260521`
- Tangent output: `forensics/loss3_all_p2_tangent_geometry_probe_20260520`
- Stepwise output: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520`
- Jacobian/SVD output: `forensics/loss3_current_core4_jacobian_svd_probe_20260521`
- Summary output: `forensics/loss3_current_mechanism_validation_summary_20260521`

## Question 1: Does GPI find the high-loss direction early?

Observed from ray profiles. For p2q2 steepest_replace, endpoint loss along early directions divided by endpoint loss along the final direction:

| pq | method | step | endpoint_loss | final_endpoint_loss | endpoint_to_final_ratio |
| --- | --- | --- | --- | --- | --- |
| p2q2 | steepest_replace | 1 | 1.481 | 3.634 | 0.4074 |
| p2q2 | steepest_replace | 5 | 3.494 | 3.634 | 0.9612 |
| p2q2 | steepest_replace | 10 | 3.664 | 3.634 | 1.008 |
| p2q2 | steepest_replace | 20 | 3.679 | 3.634 | 1.012 |

Interpretation: ratios near 1 by step 5-10 support the claim that replacement/GPI rapidly reaches a final-like high-loss direction.

## Question 2: Are method final deltas on a shared high-loss ridge?

Observed from p=2 boundary arcs. Values near 1 mean the arc between two final directions does not dip below the weaker endpoint much:

| pq | pair | min_arc_loss | min_over_weaker_endpoint | min_over_stronger_endpoint | arc_depth_from_weaker |
| --- | --- | --- | --- | --- | --- |
| p2q1 | raw_add__steepest_add | 41.74 | 0.9996 | 0.9742 | 0.01606 |
| p2q1 | steepest_replace__raw_add | 41.24 | 0.9878 | 0.8958 | 0.5082 |
| p2q1 | steepest_replace__steepest_add | 42.27 | 0.9868 | 0.9181 | 0.5675 |
| p2q2 | raw_add__steepest_add | 3.473 | 1 | 0.9998 | 0 |
| p2q2 | steepest_replace__raw_add | 3.426 | 0.9863 | 0.9616 | 0.04767 |
| p2q2 | steepest_replace__steepest_add | 3.426 | 0.9861 | 0.9616 | 0.04837 |
| p2qinf | raw_add__steepest_add | 0.4856 | 1 | 0.7959 | 0 |
| p2qinf | steepest_replace__raw_add | 0.415 | 0.9976 | 0.8547 | 0.001003 |
| p2qinf | steepest_replace__steepest_add | 0.416 | 1 | 0.6819 | 0 |

Interpretation: p2q2 arcs near 1 support a shared ridge; lower values or stronger dips are caveats, especially q=inf.

## Question 3: Is boundary hit different from boundary convergence?

Observed from all-p2 tangent residual rollup:

| pq | method | hit_step_99_mean | post_boundary_gain_mean | tangent_residual_at_hit_mean | mean_angle_after_hit_deg_mean | final_loss3_q_mean |
| --- | --- | --- | --- | --- | --- | --- |
| p2q1 | raw_add | 3.545 | 42.07 | 0.6179 | 8.848 | 105.1 |
| p2q1 | raw_replace | 1 | 70.32 | 0.8729 | 25.73 | 100.7 |
| p2q1 | steepest_add | 12.93 | 26.52 | 0.484 | 0.8846 | 100.6 |
| p2q1 | steepest_replace | 1 | 70.32 | 0.8729 | 25.73 | 100.7 |
| p2q2 | raw_add | 32.41 | 1.28 | 0.397 | 0.2812 | 5.639 |
| p2q2 | raw_replace | 1 | 3.412 | 0.8552 | 30.68 | 5.747 |
| p2q2 | steepest_add | 12.75 | 1.467 | 0.4349 | 0.605 | 5.817 |
| p2q2 | steepest_replace | 1 | 3.412 | 0.8552 | 30.68 | 5.747 |
| p2qinf | raw_add | 51.21 | 0.0233 | 0.7215 | 1.724 | 0.4976 |
| p2qinf | raw_replace | 1 | 0.1558 | 0.9381 | 66.89 | 0.5545 |
| p2qinf | steepest_add | 22.1 | 0.1764 | 0.8679 | 6.678 | 0.7864 |
| p2qinf | steepest_replace | 1 | 0.1558 | 0.9381 | 66.89 | 0.5545 |

Interpretation: high tangent residual at boundary hit means there is still sideways gradient on the boundary. Replacement can hit boundary immediately but still have large direction work left.

## Question 4: Does tangent size predict immediate next-step gain?

Observed from p2q2 stepwise post-boundary correlations:

| method | metric | pearson_r_with_next_loss_gain | pair_count |
| --- | --- | --- | --- |
| raw_add | actual_delta_angle_next | 0.3495 | 5.341e+05 |
| raw_add | actual_step_l2_over_eps_next | 0.3497 | 5.341e+05 |
| raw_add | direction_l2 | 0.03819 | 5.341e+05 |
| raw_add | direction_radial_abs_ratio | -0.3271 | 5.341e+05 |
| raw_add | direction_radial_signed_ratio | -0.3271 | 5.341e+05 |
| raw_add | direction_tangent_ratio | 0.3374 | 5.341e+05 |
| raw_add | grad_radial_abs_norm | -0.006004 | 5.341e+05 |
| raw_add | grad_radial_abs_ratio | -0.3271 | 5.341e+05 |
| raw_add | grad_radial_signed_norm | -0.006004 | 5.341e+05 |
| raw_add | grad_radial_signed_ratio | -0.3271 | 5.341e+05 |
| raw_add | grad_tangent_norm | 0.4916 | 5.341e+05 |
| raw_add | grad_tangent_ratio | 0.3374 | 5.341e+05 |
| raw_add | linear_gain_projected | 0.5996 | 5.341e+05 |
| raw_add | linear_gain_unprojected | 0.05672 | 5.341e+05 |
| raw_replace | actual_delta_angle_next | 0.03549 | 5.98e+05 |
| raw_replace | actual_step_l2_over_eps_next | 0.03402 | 5.98e+05 |
| raw_replace | direction_l2 | nan | 5.98e+05 |
| raw_replace | direction_radial_abs_ratio | -0.0442 | 5.98e+05 |
| raw_replace | direction_radial_signed_ratio | -0.04396 | 5.98e+05 |
| raw_replace | direction_tangent_ratio | 0.02877 | 5.98e+05 |
| raw_replace | grad_radial_abs_norm | -0.02116 | 5.98e+05 |
| raw_replace | grad_radial_abs_ratio | -0.0442 | 5.98e+05 |
| raw_replace | grad_radial_signed_norm | -0.02114 | 5.98e+05 |
| raw_replace | grad_radial_signed_ratio | -0.04396 | 5.98e+05 |
| raw_replace | grad_tangent_norm | 0.02772 | 5.98e+05 |
| raw_replace | grad_tangent_ratio | 0.02877 | 5.98e+05 |
| raw_replace | linear_gain_projected | 0.09711 | 5.98e+05 |
| raw_replace | linear_gain_unprojected | 0.09711 | 5.98e+05 |
| steepest_add | actual_delta_angle_next | 0.1631 | 5.739e+05 |
| steepest_add | actual_step_l2_over_eps_next | 0.1632 | 5.739e+05 |
| steepest_add | direction_l2 | nan | 5.739e+05 |
| steepest_add | direction_radial_abs_ratio | -0.2224 | 5.739e+05 |
| steepest_add | direction_radial_signed_ratio | -0.2224 | 5.739e+05 |
| steepest_add | direction_tangent_ratio | 0.2654 | 5.739e+05 |
| steepest_add | grad_radial_abs_norm | -0.006782 | 5.739e+05 |
| steepest_add | grad_radial_abs_ratio | -0.2224 | 5.739e+05 |
| steepest_add | grad_radial_signed_norm | -0.006782 | 5.739e+05 |
| steepest_add | grad_radial_signed_ratio | -0.2224 | 5.739e+05 |
| steepest_add | grad_tangent_norm | 0.408 | 5.739e+05 |
| steepest_add | grad_tangent_ratio | 0.2654 | 5.739e+05 |
| steepest_add | linear_gain_projected | 0.2991 | 5.739e+05 |
| steepest_add | linear_gain_unprojected | 0.05606 | 5.739e+05 |
| steepest_replace | actual_delta_angle_next | 0.03549 | 5.98e+05 |
| steepest_replace | actual_step_l2_over_eps_next | 0.03402 | 5.98e+05 |
| steepest_replace | direction_l2 | nan | 5.98e+05 |
| steepest_replace | direction_radial_abs_ratio | -0.0442 | 5.98e+05 |
| steepest_replace | direction_radial_signed_ratio | -0.04396 | 5.98e+05 |
| steepest_replace | direction_tangent_ratio | 0.02877 | 5.98e+05 |
| steepest_replace | grad_radial_abs_norm | -0.02116 | 5.98e+05 |
| steepest_replace | grad_radial_abs_ratio | -0.0442 | 5.98e+05 |
| steepest_replace | grad_radial_signed_norm | -0.02114 | 5.98e+05 |
| steepest_replace | grad_radial_signed_ratio | -0.04396 | 5.98e+05 |
| steepest_replace | grad_tangent_norm | 0.02772 | 5.98e+05 |
| steepest_replace | grad_tangent_ratio | 0.02877 | 5.98e+05 |
| steepest_replace | linear_gain_projected | 0.09711 | 5.98e+05 |
| steepest_replace | linear_gain_unprojected | 0.09711 | 5.98e+05 |

Interpretation: tangent magnitude is opportunity, not a complete one-step predictor. The actual projected direction and local nonlinearity matter.

## Question 5: What do 2D slices and curvature say?

2D slice summary:

| pq | center_method | center_loss | slice_loss_min | slice_loss_max | slice_range | center_minus_min |
| --- | --- | --- | --- | --- | --- | --- |
| p1qinf | steepest_add | 0.202 | 0.1817 | 0.2176 | 0.03586 | 0.02025 |
| p1qinf | steepest_replace | 0.2089 | 0.1894 | 0.2264 | 0.03698 | 0.01946 |
| p2q1 | steepest_add | 42.84 | 37.26 | 48.37 | 11.11 | 5.577 |
| p2q1 | steepest_replace | 46.04 | 40.09 | 52.29 | 12.2 | 5.951 |
| p2q2 | steepest_add | 3.474 | 3.129 | 3.796 | 0.667 | 0.3454 |
| p2q2 | steepest_replace | 3.562 | 3.161 | 3.938 | 0.777 | 0.4019 |
| p2qinf | steepest_add | 0.6101 | 0.5764 | 0.6225 | 0.04604 | 0.03367 |
| p2qinf | steepest_replace | 0.416 | 0.3816 | 0.4532 | 0.07156 | 0.0344 |

Interpretation: these contours show whether the final/center point sits on a broad ridge, a narrow peak, or a geometry-dependent irregular surface.

## Question 6: Is there direct dominant-mode evidence?

Targeted residual-Jacobian/SVD summary:

| pq | state_label | sigma1 | sigma2 | sigma1_over_sigma2 | top1_energy_fraction | top4_energy_fraction | top1_peakiness | top1_energy_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p2q2 | clean | 1.206 | 1.05 | 1.148 | 0.4038 | 0.803 | 3.928 | 0.004474 |
| p2q2 | steepest_replace_step1 | 4.68 | 3.252 | 1.439 | 0.5961 | 0.956 | 3.396 | 0.003761 |
| p2q2 | steepest_replace_step5 | 10.17 | 1.997 | 5.094 | 0.9245 | 0.9939 | 2.566 | 0.002804 |
| p2q2 | steepest_replace_step10 | 8.856 | 2.223 | 3.984 | 0.8752 | 0.9891 | 2.513 | 0.002733 |
| p2q2 | steepest_replace_final | 8.864 | 2.242 | 3.955 | 0.8751 | 0.9891 | 2.504 | 0.002722 |
| p2q2 | steepest_add_final | 8.344 | 1.616 | 5.163 | 0.9178 | 0.9853 | 3.228 | 0.00372 |
| p2qinf | clean | 1.206 | 1.05 | 1.148 | 0.4038 | 0.803 | 3.928 | 0.004474 |
| p2qinf | steepest_replace_final | 3.893 | 1.03 | 3.781 | 0.8299 | 0.9591 | 2.866 | 0.003629 |
| p2qinf | steepest_add_final | 7.097 | 1.719 | 4.127 | 0.892 | 0.98 | 3.484 | 0.00375 |
| p1qinf | clean | 1.206 | 1.05 | 1.148 | 0.4038 | 0.803 | 3.928 | 0.004474 |
| p1qinf | steepest_replace_final | 1.177 | 0.9532 | 1.235 | 0.4031 | 0.7789 | 3.97 | 0.00451 |

Top residual singular direction alignment with important deltas:

| pq | state_label | direction_label | abs_cos_top1_right_vs_direction | angle_deg_top1_right_vs_direction |
| --- | --- | --- | --- | --- |
| p1qinf | clean | raw_add_final | 0.2861 | 73.37 |
| p1qinf | clean | steepest_add_final | 0.003957 | 89.77 |
| p1qinf | clean | steepest_replace_final | 0.006199 | 89.64 |
| p1qinf | clean | steepest_replace_step5 | 0.006199 | 89.64 |
| p1qinf | steepest_replace_final | raw_add_final | 0.2783 | 73.84 |
| p1qinf | steepest_replace_final | steepest_add_final | 0.008299 | 89.52 |
| p1qinf | steepest_replace_final | steepest_replace_final | 0.0003424 | 89.98 |
| p1qinf | steepest_replace_final | steepest_replace_step5 | 0.0003424 | 89.98 |
| p2q2 | clean | raw_add_final | 0.02948 | 88.31 |
| p2q2 | clean | steepest_add_final | 0.02965 | 88.3 |
| p2q2 | clean | steepest_replace_final | 0.1086 | 83.77 |
| p2q2 | clean | steepest_replace_step5 | 0.3737 | 68.06 |
| p2q2 | steepest_add_final | raw_add_final | 0.6261 | 51.24 |
| p2q2 | steepest_add_final | steepest_add_final | 0.6248 | 51.33 |
| p2q2 | steepest_add_final | steepest_replace_final | 0.000145 | 89.99 |
| p2q2 | steepest_add_final | steepest_replace_step5 | 6.068e-05 | 90 |
| p2q2 | steepest_replace_final | raw_add_final | 0.01654 | 89.05 |
| p2q2 | steepest_replace_final | steepest_add_final | 0.01672 | 89.04 |
| p2q2 | steepest_replace_final | steepest_replace_final | 0.2813 | 73.66 |
| p2q2 | steepest_replace_final | steepest_replace_step5 | 0.1777 | 79.76 |
| p2q2 | steepest_replace_step1 | raw_add_final | 0.005329 | 89.69 |
| p2q2 | steepest_replace_step1 | steepest_add_final | 0.005561 | 89.68 |
| p2q2 | steepest_replace_step1 | steepest_replace_final | 0.172 | 80.1 |
| p2q2 | steepest_replace_step1 | steepest_replace_step5 | 0.3249 | 71.04 |
| p2q2 | steepest_replace_step10 | raw_add_final | 0.01619 | 89.07 |
| p2q2 | steepest_replace_step10 | steepest_add_final | 0.01637 | 89.06 |
| p2q2 | steepest_replace_step10 | steepest_replace_final | 0.2766 | 73.94 |
| p2q2 | steepest_replace_step10 | steepest_replace_step5 | 0.1748 | 79.94 |
| p2q2 | steepest_replace_step5 | raw_add_final | 0.01143 | 89.34 |
| p2q2 | steepest_replace_step5 | steepest_add_final | 0.01162 | 89.33 |
| p2q2 | steepest_replace_step5 | steepest_replace_final | 0.1926 | 78.89 |
| p2q2 | steepest_replace_step5 | steepest_replace_step5 | 0.1245 | 82.85 |
| p2qinf | clean | raw_add_final | 0.106 | 83.91 |
| p2qinf | clean | steepest_add_final | 0.07253 | 85.84 |
| p2qinf | clean | steepest_replace_final | 0.8289 | 34.02 |
| p2qinf | clean | steepest_replace_step5 | 0.04909 | 87.19 |
| p2qinf | steepest_add_final | raw_add_final | 0.01031 | 89.41 |
| p2qinf | steepest_add_final | steepest_add_final | 0.02704 | 88.45 |
| p2qinf | steepest_add_final | steepest_replace_final | 0.8701 | 29.54 |
| p2qinf | steepest_add_final | steepest_replace_step5 | 0.0315 | 88.2 |
| p2qinf | steepest_replace_final | raw_add_final | 0.1105 | 83.65 |
| p2qinf | steepest_replace_final | steepest_add_final | 0.07664 | 85.6 |
| p2qinf | steepest_replace_final | steepest_replace_final | 0.7453 | 41.81 |
| p2qinf | steepest_replace_final | steepest_replace_step5 | 0.5239 | 58.4 |

Interpretation: high spectral gap plus high cosine would support a literal dominant residual mode. Weak gap or weak cosine means the safer explanation remains empirical/local-surrogate rather than a strict spectral theorem.



## Final Interpretation After The New Probe

Observed evidence that **does support** the current explanation:

- `p2q2` replacement/GPI ray endpoint ratio reaches `0.961` by step 5, `1.008` by step 10, and `1.012` by step 20. So the early GPI direction is already almost as strong as the final direction in the ray-profile sense.
- `p2q2` boundary arc dips are small: replacement-to-additive arcs stay at about `0.986` of the weaker endpoint. This supports a shared high-loss ridge rather than completely separate basins.
- all-p2 tangent residual shows replacement hits boundary at step 1 with large tangent residual: about `0.855` for p2q2, `0.873` for p2q1, and `0.938` for p2qinf. So boundary hit is not convergence.
- `p2q2` residual-Jacobian spectrum becomes much more dominated after GPI moves: `sigma1/sigma2` rises from `1.148` at clean to `5.094` at step 5, and top-1 energy fraction rises from `0.404` to `0.925`. This supports the idea that the perturbed region has a strong local mode/ridge.

Observed evidence that **limits** the stronger spectral claim:

- The p2q2 top residual-Jacobian right singular vector is not strongly aligned with the final replacement delta. At `steepest_replace_final`, the cosine with `steepest_replace_final` is only `0.281`.
- At `steepest_replace_step5`, the cosine with `steepest_replace_final` is only `0.193`, and with the step5 delta is `0.125`.
- This means we should not claim that GPI/replacement is simply following the top singular vector of the local residual Jacobian.

Inference:

- The best-supported explanation is not a pure SVD theorem. It is a landscape/surrogate explanation: replacement/GPI rapidly moves into a high-loss boundary region, and in p2q2 that region has a strong local dominant structure/ridge. But the actual Loss3 optimizer direction is affected by the nonlinear/affine residual objective and boundary geometry, so pure residual-Jacobian top singular alignment is not enough to explain it.
- This actually matches the earlier warning: objective-gradient replacement is the successful surrogate, not literal pure JVP/VJP generalized power iteration for the full Loss3 objective.

PQ caveat from the new evidence:

- `p2qinf` still has a strong local spectrum, but final-ray and arc evidence show the cleaner p2q2 story weakens. For example, `p2qinf` steepest_replace-to-steepest_add arc has min/stronger-endpoint ratio only about `0.682`, and steepest_add final ray endpoint is much stronger than replacement.
- `p1qinf` replacement final has weak residual-Jacobian spectral gap (`sigma1/sigma2 = 1.235`) and the top direction is nearly orthogonal to the replacement final delta. This supports the warning that p=1/q=inf geometry is not governed by the same smooth dominant-ridge story.

## Bottom Line

The validation separates the original mystery into testable pieces: boundary arrival, boundary rotation, shared ridge, local slice/curvature, and residual-Jacobian dominant-mode evidence. The supported claim should stay conditional: replacement/GPI is a very strong aggressive full-budget surrogate in p2q2-like geometry, but q=inf and p=1 geometries can break the clean story.
