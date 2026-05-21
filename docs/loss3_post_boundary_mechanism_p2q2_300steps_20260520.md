# Loss3 Post-Boundary Mechanism Diagnostics - 2026-05-20

Status: generated from existing completed artifacts; no optimizer experiment was rerun.

## Source Data

Observed from sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`
Output root: `forensics/loss3_post_boundary_mechanism_p2q2_300steps_20260520`
Boundary threshold used here: `0.99`.

Tables:

- Per-setting diagnostics: `forensics/loss3_post_boundary_mechanism_p2q2_300steps_20260520/tables/post_boundary_mechanism_by_setting.csv`
- Method rollup: `forensics/loss3_post_boundary_mechanism_p2q2_300steps_20260520/tables/post_boundary_mechanism_rollup.csv`

## What The Diagnostics Mean

- `post_boundary_loss_gain_mean`: final mean loss minus mean loss at the first boundary-hit step. This is the direct measurement of how much useful optimization happens after the method has already reached the epsilon boundary.
- `loss_gain_10_steps_after_hit` and `loss_gain_25_steps_after_hit`: short-window post-boundary gain. These test whether the method improves quickly after hitting the boundary rather than only slowly over many steps.
- `cos_delta_direction_after_hit_mean`: cosine between the current perturbation direction and the proposed update direction after boundary hit. Near `1` means the update is mostly radial; smaller values mean more boundary-direction rotation.
- `tangent_direction_ratio_after_hit_p2_proxy`: for `p=2`, approximately `sqrt(1 - cos_delta_direction^2)`. Larger means more tangential/boundary-surface motion is available after the boundary is reached.
- `cos_direction_prev_after_hit_mean`: cosine between successive update directions. Smaller means the optimizer is still changing direction substantially; near `1` means directions have stabilized.
- `delta_prev_angle_degrees_after_hit_mean`: direct mean of `angle(delta_k, delta_{k-1})` after boundary hit, when available in newly generated runs.
- `delta_step_l2_over_epsilon_after_hit_mean`: actual perturbation-space step length after boundary hit, normalized by epsilon.
- `projection_shrink_after_hit_mean`: how much the proposed step is shrunk by projection. For additive PGD on the boundary, a lot of radial motion can be projected away, which can waste step length.
- `traj_final_cosine_from_hit_mean`: cosine between the selected-sample delta at boundary hit and final delta. Low values mean the boundary point still rotates a lot before the final solution.
- `traj_angular_travel_after_hit_deg_mean`: total selected-sample angular travel on/near the boundary after the hit step. This directly measures whether the optimizer keeps moving around the boundary.
- `traj_hf_ratio_change_hit_to_final_mean` and derivative/TV changes: whether the trajectory becomes smoother or rougher after boundary hit.

## Method Rollup

| method | setting_count | mean_hit_step | mean_post_boundary_loss_gain_mean | mean_loss_gain_10_steps_after_hit | mean_tangent_direction_ratio_after_hit_p2_proxy | mean_delta_prev_angle_degrees_after_hit_mean | mean_delta_step_l2_over_epsilon_after_hit_mean | mean_traj_final_cosine_from_hit_mean | mean_traj_angular_travel_after_hit_deg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_add | 20 | 49.2 | 0.8922 | 0.2356 | 0.1594 | 0.2462 | 0.004356 | 0.698 | 57.01 |
| raw_replace | 20 | 1 | 3.412 | 3.308 | 0.5667 | 30.68 | 0.5238 | 0.3192 | 7580 |
| steepest_add | 20 | 13.45 | 1.419 | 0.5926 | 0.1852 | 0.6115 | 0.01074 | 0.6871 | 59.5 |
| steepest_replace | 20 | 1 | 3.412 | 3.308 | 0.5667 | 30.68 | 0.5238 | 0.3192 | 7580 |

## Baseline eps=4, alpha=0.4 Preview

| method | hit_step | loss3_q_at_hit_mean | final_loss3_q_mean | post_boundary_loss_gain_mean | loss_gain_10_steps_after_hit | tangent_direction_ratio_after_hit_p2_proxy | delta_prev_angle_degrees_after_hit_mean | delta_step_l2_over_epsilon_after_hit_mean | traj_final_cosine_from_hit_mean | traj_angular_travel_after_hit_deg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_add | 43 | 2.623 | 3 | 0.3772 | 0.07483 | 0.1075 | 0.1436 | 0.002576 | 0.5584 | 65.62 |
| raw_replace | 1 | 1.569 | 3.062 | 1.493 | 1.356 | 0.6043 | 33.19 | 0.5654 | 0.3425 | 7940 |
| steepest_add | 11 | 2.426 | 3.065 | 0.6389 | 0.2532 | 0.1058 | 0.2391 | 0.004317 | 0.5324 | 67.79 |
| steepest_replace | 1 | 1.569 | 3.062 | 1.493 | 1.356 | 0.6043 | 33.19 | 0.5654 | 0.3425 | 7940 |

## Working Interpretation

Observed evidence from these diagnostics should be read together with the loss curves and GIF trajectories.

Inference: if GPI/replacement reaches the boundary at step 1 but still has a large `post_boundary_loss_gain_mean`, then the first boundary point is not the final good direction. The method is winning because it rapidly rotates/refines the direction on the boundary, not merely because it reaches the boundary.

Inference: if additive PGD methods show later boundary hit, high projection shrink, smaller tangential motion, or low post-boundary angular travel, then their post-boundary updates may be wasting effort in radial components that projection removes, or moving too slowly around the boundary surface.

Inference: if GPI's high-frequency/smoothness metrics decrease after the hit step, then the suspected behavior is confirmed: it may hit the boundary with a rougher intermediate perturbation and then smooth/organize the direction while staying near the boundary. If those metrics do not decrease, then the post-boundary gain is more likely directional alignment with the loss landscape rather than smoothing.
