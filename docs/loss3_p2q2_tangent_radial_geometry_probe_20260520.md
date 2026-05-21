# Loss 3 p2q2 Tangent and Radial Geometry Probe - 2026-05-20

Status: generated from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.

## Tables

- Tangent diagnostics by setting/method: `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/tangent_kkt_by_setting_method.csv`
- Tangent diagnostics rollup: `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/tangent_kkt_rollup_by_method.csv`
- Radial growth diagnostics by setting/method: `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/radial_growth_by_setting_method.csv`
- Radial growth diagnostics rollup: `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/radial_growth_rollup_by_method.csv`
- Correlations: `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/geometry_probe_correlations.csv`

## Tangent Gradient Projection Diagnostic

For p=2, with `u = delta / ||delta||_2`, the normalized tangent-gradient residual is computed from the recorded gradient cosine:

```text
||(I - u u^T) grad L|| / ||grad L|| = sqrt(1 - cos(delta, grad)^2)
```

| method | row_count | mean_mean_hit_step | mean_mean_post_boundary_gain | mean_mean_tangent_grad_ratio_at_hit | mean_mean_tangent_grad_ratio_after_hit | mean_mean_tangent_grad_ratio_last_update | mean_mean_delta_angle_after_hit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_add | 20 | 32.41 | 1.285 | 0.397 | 0.08363 | 0.03876 | 0.2878 |
| raw_replace | 20 | 1 | 3.412 | 0.8552 | 0.4844 | 0.4802 | 30.68 |
| steepest_add | 20 | 12.75 | 1.47 | 0.4349 | 0.08228 | 0.04624 | 0.6172 |
| steepest_replace | 20 | 1 | 3.412 | 0.8552 | 0.4844 | 0.4802 | 30.68 |

## Radial Growth Diagnostic for Additive Methods

For p=2 additive updates, the exact unprojected radius formula is:

```text
r_next^2 = r^2 + 2 alpha r ||d|| cos(theta) + alpha^2 ||d||^2
```

Rows below use pre-boundary steps only, so projection should not dominate the comparison.

| method | row_count | mean_pre_boundary_step_count | mean_mean_cos_theta_delta_direction | mean_cv_cos_theta_delta_direction | mean_mean_direction_l2 | mean_cv_direction_l2 | mean_mean_actual_radial_increment | mean_cv_actual_radial_increment | mean_mae_exact_prediction_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_add | 20 | 3096 | 0.8693 | 0.2466 | 0.3595 | 0.6211 | 0.3058 | 0.6209 | 1.071e-07 |
| steepest_add | 20 | 1174 | 0.7 | 0.4775 | 1 | 3.295e-08 | 0.7815 | 0.11 | 1.229e-07 |

## Correlations

| scope | method | x | y | pearson_r | pair_count |
| --- | --- | --- | --- | --- | --- |
| setting_level_by_method | raw_add | post_boundary_gain | mean_tangent_grad_ratio_at_hit | 0.8159 | 20 |
| setting_level_by_method | raw_add | post_boundary_gain | mean_tangent_grad_ratio_after_hit | 0.4952 | 20 |
| setting_level_by_method | raw_add | post_boundary_gain | mean_tangent_grad_ratio_last_update | 0.6292 | 20 |
| setting_level_by_method | raw_add | post_boundary_gain | mean_delta_angle_after_hit | 0.4322 | 20 |
| setting_level_by_method | raw_replace | post_boundary_gain | mean_tangent_grad_ratio_at_hit | 0.6216 | 20 |
| setting_level_by_method | raw_replace | post_boundary_gain | mean_tangent_grad_ratio_after_hit | -0.08193 | 20 |
| setting_level_by_method | raw_replace | post_boundary_gain | mean_tangent_grad_ratio_last_update | -0.112 | 20 |
| setting_level_by_method | raw_replace | post_boundary_gain | mean_delta_angle_after_hit | -0.3606 | 20 |
| setting_level_by_method | steepest_add | post_boundary_gain | mean_tangent_grad_ratio_at_hit | 0.7018 | 20 |
| setting_level_by_method | steepest_add | post_boundary_gain | mean_tangent_grad_ratio_after_hit | 0.7242 | 20 |
| setting_level_by_method | steepest_add | post_boundary_gain | mean_tangent_grad_ratio_last_update | 0.5783 | 20 |
| setting_level_by_method | steepest_add | post_boundary_gain | mean_delta_angle_after_hit | 0.1601 | 20 |
| setting_level_by_method | steepest_replace | post_boundary_gain | mean_tangent_grad_ratio_at_hit | 0.6216 | 20 |
| setting_level_by_method | steepest_replace | post_boundary_gain | mean_tangent_grad_ratio_after_hit | -0.08193 | 20 |
| setting_level_by_method | steepest_replace | post_boundary_gain | mean_tangent_grad_ratio_last_update | -0.112 | 20 |
| setting_level_by_method | steepest_replace | post_boundary_gain | mean_delta_angle_after_hit | -0.3606 | 20 |
| radial_growth_setting_level | steepest_add | cv_actual_radial_increment | cv_cos_theta_delta_direction | 0.516 | 20 |
| radial_growth_setting_level | steepest_add | cv_actual_radial_increment | cv_direction_l2 | 0.2575 | 20 |
| radial_growth_setting_level | steepest_add | cv_actual_radial_increment | cv_predicted_radial_increment_exact | 1 | 20 |
| radial_growth_setting_level | steepest_add | cv_actual_radial_increment | cv_actual_radial_increment | 1 | 20 |
| radial_growth_setting_level | steepest_add | cv_actual_radial_increment | mae_exact_prediction_error | 0.7441 | 20 |
| radial_growth_setting_level | raw_add | cv_actual_radial_increment | cv_cos_theta_delta_direction | -0.3136 | 20 |
| radial_growth_setting_level | raw_add | cv_actual_radial_increment | cv_direction_l2 | 0.969 | 20 |
| radial_growth_setting_level | raw_add | cv_actual_radial_increment | cv_predicted_radial_increment_exact | 1 | 20 |
| radial_growth_setting_level | raw_add | cv_actual_radial_increment | cv_actual_radial_increment | 1 | 20 |
| radial_growth_setting_level | raw_add | cv_actual_radial_increment | mae_exact_prediction_error | 0.7266 | 20 |

## Interpretation

Observed evidence here directly tests the p=2 tangent-plane story using gradients recorded during the original runs. A high tangent residual after boundary hit means the optimizer is not boundary-stationary and can still improve by rotating on the L2 sphere.

The radial-growth table tests the LP-steepest straight-line explanation. If `steepest_add` has `direction_l2` near 1 and lower CV of actual/predicted radial increments than `raw_add`, then its straighter boundary-ratio curves are explained by normalized step size plus reasonably stable radial alignment. Raw PGD can curve because both direction norm and radial alignment vary.

Caveat: these diagnostics use metrics recorded at each step, not newly recomputed gradients. They are still based on the actual autograd gradients saved during the experiments, but full vector projection plots would require storing gradient vectors or rerunning a GPU diagnostic.

## Validation Verdicts

Observed from the p2q2 300-step sweep:

1. The tangent-gradient projection is genuinely large for replacement/GPI at the boundary. At the 99% boundary hit step, replacement has mean normalized tangent gradient residual about `0.855`, while raw add is about `0.397` and steepest add about `0.435`. This means the first replacement boundary point is very far from p=2 boundary stationarity.

2. Replacement/GPI keeps a large tangent residual even late in the run. At the last gradient-bearing step, replacement is still around `0.480`, while raw add is around `0.039` and steepest add around `0.046`. This suggests replacement is not converging by slowly killing the tangent KKT residual; it is aggressively moving/rotating on the boundary and may remain dynamically active.

3. Post-boundary gain is related to tangent residual at the hit step. Across the 20 p2q2 alpha/epsilon settings, setting-level correlation between post-boundary gain and tangent residual at hit is `0.816` for raw add, `0.702` for steepest add, and `0.622` for replacement. This supports the claim: if a method reaches the boundary while the tangent gradient is still large, there is still room for loss to grow by boundary-direction movement.

4. Tangent residual is not a complete predictor. For replacement, post-boundary gain is not positively correlated with mean tangent residual after the hit (`-0.082`) or last-update residual (`-0.112`). Interpretation: replacement almost always has substantial tangent motion; whether that motion improves loss depends on whether the selected new direction is actually useful for the current loss landscape.

5. The radial-growth formula is numerically validated. The exact p=2 unprojected radius prediction has mean absolute error around `1e-7`, so the formula

```text
r_next^2 = r^2 + 2 alpha r ||d|| cos(theta) + alpha^2 ||d||^2
```

matches the recorded trajectory before projection dominates.

6. LP-steepest's straighter norm growth is mainly due to normalized direction size. In the rollup, `steepest_add` has direction L2 mean `1.0` with CV about `3.3e-8`, while raw add has direction L2 mean about `0.359` with CV about `0.621`. The actual radial increment CV is about `0.110` for steepest add but `0.621` for raw add. This strongly supports the normalized-step explanation.

7. The earlier "cos theta is stable" explanation needs refinement. `steepest_add` does not have a lower cos-theta CV than raw add in this rollup; its cos-theta CV is about `0.478`, while raw add is about `0.247`. What matters more is that steepest_add removes gradient-norm variability. Its radial progress is much more synchronized even though the direction angle still changes.

Concrete baseline eps=4, alpha=0.4 example:

- raw add: hit step `29.01`, post-boundary gain `0.539`, tangent residual at hit `0.376`, last-update tangent residual `0.0157`, angle after hit `0.180 deg`.
- steepest add: hit step `11.13`, post-boundary gain `0.614`, tangent residual at hit `0.380`, last-update tangent residual `0.0115`, angle after hit `0.238 deg`.
- replacement/GPI: hit step `1`, post-boundary gain `1.493`, tangent residual at hit `0.882`, last-update tangent residual `0.514`, angle after hit `33.19 deg`.

Inference: for p2q2, the boundary story is now directly supported by the recorded gradients. Replacement/GPI reaches the boundary at a point with a very large tangent gradient component and then keeps making large boundary-direction moves. Additive methods reach the boundary later, with smaller tangent residual and much smaller angular motion.

