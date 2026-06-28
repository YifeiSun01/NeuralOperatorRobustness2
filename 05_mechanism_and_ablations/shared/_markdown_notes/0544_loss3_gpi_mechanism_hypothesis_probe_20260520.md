# Loss 3 GPI Mechanism Hypothesis Probe - 2026-05-20

Status: generated from existing artifacts only; no neural-operator experiment was rerun.

## Output Tables

- Per-setting method metrics: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/mechanism_probe_by_setting_method.csv`
- Rollup by P/Q and method: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/mechanism_probe_rollup_by_pq_method.csv`
- Rollup by method: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/mechanism_probe_rollup_by_method.csv`
- Early-to-final trajectory probe: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/early_to_final_trajectory_probe.csv`
- Hypothesis tests: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/hypothesis_tests.csv`
- Objective-gradient versus JVP/VJP power variant probe: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/qaware_power_variant_probe_p2q2_eps8_alpha0p3.csv`

## What Was Tested

1. Whether post-boundary loss gain is accompanied by angular/tangent motion.
2. Whether replacement/GPI reaches a final-like delta direction early in saved trajectories.
3. Whether p/q geometry explains spike-prone perturbations via concentration metrics.
4. Whether immediate boundary arrival is enough to predict final loss.

## Key Rollup: p=2,q=2

| method | mean_hit_step_99 | mean_post_boundary_gain | mean_mean_angle_after_hit_deg | mean_mean_tangent_ratio_after_hit_p2_proxy | mean_final_loss_fraction_of_setting_best | mean_mean_peakiness_max_abs_over_rms |
| --- | --- | --- | --- | --- | --- | --- |
| raw_add | 49.2 | 0.8922 | 0.2462 | 0.1594 | 0.9758 | 4.128 |
| raw_replace | 1 | 3.412 | 30.68 | 0.5667 | 0.9799 | 3.633 |
| steepest_add | 13.45 | 1.419 | 0.6115 | 0.1852 | 0.9986 | 4.048 |
| steepest_replace | 1 | 3.412 | 30.68 | 0.5667 | 0.9799 | 3.633 |

## Key Rollup: p=1,q=inf

| method | mean_hit_step_99 | mean_post_boundary_gain | mean_mean_angle_after_hit_deg | mean_final_loss_fraction_of_setting_best | mean_mean_peakiness_max_abs_over_rms | mean_mean_top1_energy_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| raw_add | 12.8 | 0.001474 | 8.948 | 0.5285 | 8.281 | 0.08526 |
| raw_replace | 1 | 1.318e-05 | 44.86 | 0.4859 | 9.069 | 0.09851 |
| steepest_add | 12.25 | 0.005787 | 5.668 | 1 | 29.54 | 0.8636 |
| steepest_replace | 1 | -0.04598 | 23.43 | 0.8098 | 30.69 | 0.959 |

## Hypothesis Test Snapshot

| hypothesis | group | pearson_r | pair_count |
| --- | --- | --- | --- |
| post_boundary_gain_correlates_with_angle_motion | all_methods_all_pq | 0.04058 | 384 |
| post_boundary_gain_correlates_with_tangent_proxy | all_methods_all_pq | 0.001062 | 384 |
| early_boundary_hit_predicts_final_loss_fraction | all_methods_all_pq | 0.1803 | 384 |
| post_boundary_gain_correlates_with_angle_motion | replacement_only_all_pq | -0.0553 | 204 |
| post_boundary_gain_correlates_with_tangent_proxy | replacement_only_all_pq | -0.1641 | 204 |
| early_boundary_hit_predicts_final_loss_fraction | replacement_only_all_pq | nan | 204 |
| post_boundary_gain_correlates_with_angle_motion | additive_only_all_pq | 0.1835 | 180 |
| post_boundary_gain_correlates_with_tangent_proxy | additive_only_all_pq | 0.1246 | 180 |
| early_boundary_hit_predicts_final_loss_fraction | additive_only_all_pq | 0.1316 | 180 |
| post_boundary_gain_correlates_with_angle_motion | all_methods_p2_only | -0.03924 | 223 |
| post_boundary_gain_correlates_with_tangent_proxy | all_methods_p2_only | -0.1236 | 223 |
| early_boundary_hit_predicts_final_loss_fraction | all_methods_p2_only | 0.1578 | 223 |
| replacement_has_earlier_boundary_hit | p=1,q=2 |  |  |
| replacement_has_earlier_boundary_hit | p=1,q=inf |  |  |
| replacement_has_earlier_boundary_hit | p=2,q=1 |  |  |
| replacement_has_earlier_boundary_hit | p=2,q=2 |  |  |
| replacement_has_earlier_boundary_hit | p=2,q=inf |  |  |
| replacement_has_earlier_boundary_hit | p=inf,q=1 |  |  |
| steepest_or_qinf_geometry_increases_spikiness | p=1,q=2 |  |  |
| steepest_or_qinf_geometry_increases_spikiness | p=1,q=inf |  |  |
| steepest_or_qinf_geometry_increases_spikiness | p=2,q=1 |  |  |
| steepest_or_qinf_geometry_increases_spikiness | p=2,q=2 |  |  |
| steepest_or_qinf_geometry_increases_spikiness | p=2,q=inf |  |  |
| steepest_or_qinf_geometry_increases_spikiness | p=inf,q=1 |  |  |
| replacement_reaches_final_like_delta_early | trajectory_step_1 |  |  |
| replacement_reaches_final_like_delta_early | trajectory_step_5 |  |  |
| replacement_reaches_final_like_delta_early | trajectory_step_10 |  |  |
| replacement_reaches_final_like_delta_early | trajectory_step_20 |  |  |

## Objective-Gradient Replacement vs JVP/VJP Power Variants

| method | direction_family | final_loss3_q_mean | final_loss_fraction_of_best | final_boundary_ratio_mean | final_high_frequency_energy_ratio |
| --- | --- | --- | --- | --- | --- |
| power_replace__objective_gradient | objective_gradient_replacement | 6.805 | 1 | 1 | 8.099e-10 |
| raw_replace | objective_gradient_replacement | 6.805 | 1 | 1 | 8.099e-10 |
| steepest_replace | objective_gradient_replacement | 6.805 | 1 | 1 | 8.099e-10 |
| power_add__objective_gradient | objective_gradient_additive | 6.377 | 0.9372 | 1 | 7.998e-06 |
| steepest_add | objective_gradient_additive | 6.377 | 0.9372 | 1 | 7.998e-06 |
| unit_raw_add | objective_gradient_additive | 6.377 | 0.9372 | 1 | 7.998e-06 |
| raw_add | raw_gradient_additive | 5.39 | 0.792 | 0.9689 | 5.901e-08 |
| power_replace__affine_jvp_vjp | qaware_jvp_vjp_power | 3.842 | 0.5645 | 1 | 3.581e-06 |
| power_replace__pure_jvp_vjp | qaware_jvp_vjp_power | 2.324 | 0.3416 | 1 | 1.205e-05 |
| power_add__affine_jvp_vjp | qaware_jvp_vjp_power | 0.9658 | 0.1419 | 0.6209 | 5.261e-05 |
| power_add__pure_jvp_vjp | qaware_jvp_vjp_power | 0.6382 | 0.09379 | 0.6566 | 9.089e-05 |

## Interpretation

Observed evidence: replacement/GPI generally has the earliest boundary hit and the largest angular/tangent motion after boundary hit. This supports the mechanism that it is fast because it removes the radial phase and then rotates aggressively on the boundary.

Observed evidence: early-to-final trajectory cosine, where saved trajectories exist, tests whether the method reaches a final-like direction in a few steps. High values support the dominant-direction explanation; low values identify cases where the first boundary direction is not yet the final direction.

Observed evidence: concentration metrics such as `top1_energy_fraction`, `top5_energy_fraction`, and `peakiness` test the p/q geometry explanation for spikes. Higher concentration in p=1 or q=inf settings supports the claim that the geometry pushes optimization toward extreme coordinates.

Inference: none of these probes proves global optimality. They test whether the empirical GPI advantage is better explained by local-linear full-budget replacement plus boundary-direction rotation than by a true generalized-power theorem for the full nonlinear Loss 3.

Inference: if objective-gradient replacement beats the explicit JVP/VJP power variants, then the observed success should not be attributed to a literal generalized-power theorem. It is stronger evidence for a practical objective-gradient replacement surrogate.

Next diagnostic if stronger proof is needed: run a GPU gradient/JVP probe that records gradient-step cosine, local Jacobian spectrum, and local-linear predicted gain versus actual gain along the saved trajectories.


## Plain-Language Walkthrough

This probe was not a new model run. It did not recompute the neural operator or rerun the optimizer. It read existing saved experiment outputs and asked: which parts of the surprising GPI/replacement story are directly visible in the logs and arrays?

The input artifacts were:

- `per_step_metrics.csv`: per-iteration summaries such as loss, delta norm, boundary ratio, delta angle from previous step, gradient/direction cosine, projection shrink, and smoothness metrics.
- `final_deltas.npz`: final perturbation arrays for each method, used for shape/concentration comparisons.
- `trajectory_samples.npz`: saved intermediate deltas for selected samples and steps, used to compare early-step deltas with final deltas.

The unit of analysis called a `setting root` means one completed experiment folder, usually one combination of P, Q, epsilon, alpha, step count, and batch. The probe found 102 such folders. Since each folder usually has four core methods (`raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`), this produced 408 setting/method rows.

The 441 trajectory-probe rows are not 441 new experiments. They are comparisons made from saved trajectory files. For example, for a saved method trajectory, the probe compares `delta_5`, `delta_10`, and `delta_20` against `delta_final` and records cosine similarity.

The probe tested five questions:

1. Does replacement/GPI reach the boundary earlier than additive methods?
2. After reaching the boundary, does it still rotate direction substantially?
3. Is post-boundary loss growth associated with boundary-direction motion?
4. Do saved early deltas already look like final deltas, supporting a dominant-direction explanation?
5. Do p=1 or q=inf geometries produce measurable spike/concentration effects?

The q-aware comparison table asks a separate naming/theory question: is the strong method actually a literal JVP/VJP generalized P-Q power method, or is it objective-gradient replacement? In the checked p2q2 eps8/alpha0.3 ablation, objective-gradient replacement wins clearly over the explicit JVP/VJP power variants, so the current successful method should be interpreted as objective-gradient replacement rather than a true generalized-power theorem for the full nonlinear Loss 3.

The phrase "boundary后继续涨，对应boundary上转方向" means: once `boundary_ratio` is already near 1, radial movement is mostly finished. If loss still increases while the delta angle keeps changing, then the method is improving by changing the perturbation direction on the boundary surface, not by increasing perturbation norm.

## Mechanism Verdicts

Observed from the probe tables:

1. Replacement/GPI speed is strongly explained by the update geometry, not by a global theorem. In p2q2, replacement hits the 99% boundary at mean step `1`, while raw add hits at mean step `49.2` and steepest add at mean step `13.45`. Replacement also has much larger post-boundary angular motion: about `30.68 deg` mean angle after hit versus `0.246 deg` for raw add and `0.612 deg` for steepest add.

2. Boundary arrival is not convergence. In p2q2, replacement's mean post-boundary gain is `3.412`, about `49%` of final loss. This supports the boundary-surface rotation explanation: the first boundary point is not the final good point.

3. Angle motion is not sufficient by itself. Across all P/Q settings, the Pearson correlation between post-boundary gain and angle motion is only about `0.041`. This means "turning hard" is a mechanism for moving on the boundary, but the turn must align with a useful loss direction. p1qinf is the clearest counterexample: replacement has large angular motion but weak or negative post-boundary gain.

4. Early final-like direction is supported where saved trajectories exist. In the baseline p2q2 giftrace, replacement has mean cosine to its final delta of about `0.887` by step 5, `0.984` by step 10, and `0.995` by step 20. In the eps8/alpha0.3 p2q2 direction-proposal trajectory, objective-gradient replacement has about `0.788` by step 5, `0.966` by step 10, and `0.988` by step 20. This supports the dominant-direction hypothesis for p2q2, though it is trajectory-sample evidence rather than a full proof.

5. The explicit JVP/VJP power variants do not explain the success of the current GPI label. In the existing p2q2 eps8/alpha0.3 direction-proposal ablation, objective-gradient replacement reaches final mean loss `6.805`, while affine JVP/VJP replacement reaches `3.842` and pure JVP/VJP replacement reaches `2.324`. Therefore the successful method should be interpreted as objective-gradient replacement, not as a literal generalized P-Q power method for the full nonlinear objective.

6. P/Q spike behavior is explained by concentration metrics. In p1qinf, steepest methods have peakiness around `30` and top-1 energy fraction around `0.86-0.96`, while raw methods are much lower. In p2q2, all methods have peakiness around `3.6-4.1` and top-1 energy around `0.014-0.018`. This supports the geometry explanation: p=1 and q=inf push the update toward extreme/localized coordinates.

Inference from these probes:

- The best current mechanism is: objective-gradient replacement first removes the radial travel problem, then rapidly performs boundary-direction search. This is why it is fast.
- The surprising part is not boundary saturation; that is built into replacement. The surprising part is that, in p2q2-like regimes, the objective-gradient boundary directions align with a high-value dominant direction after only a few iterations.
- The caveat is important: large angular motion does not guarantee large loss gain. The direction must be useful under the current P/Q geometry and nonlinear loss landscape.

