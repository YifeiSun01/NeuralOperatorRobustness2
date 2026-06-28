# Loss3 Hypothesis Validation Status - 2026-05-21

Status: consolidated hypothesis-by-hypothesis validation map from current core4/PQ runs and 2026-05-21 mechanism probes. No new experiment was run while writing this document.

## Why This Document Exists

The current question is not just whether replacement/GPI works. The real question is:

```text
Why does it look so fast and strong, even though Loss3 is not a clean generalized-power/SVD objective?
```

This document separates the hypotheses that were actually checked from the hypotheses that remain only partially checked.

## Hypothesis Status Table

| Hypothesis | Status | Evidence | Conclusion |
| --- | --- | --- | --- |
| H1: GPI/replacement is fast because it reaches the epsilon boundary immediately. | Verified but incomplete explanation | all-p2 tangent rollup; replacement boundary hit step is `1` in p2q1/p2q2/p2qinf. | True, but boundary hit alone does not explain final performance. It solves the radius part, not the direction part. |
| H2: Boundary hit is not convergence; post-boundary direction rotation matters. | Strongly verified | p2q2 replacement tangent residual at hit about `0.855`; post-hit angle about `30.68 deg`; post-boundary gain about `3.412`. | Strong support. GPI reaches boundary while still having large tangent gradient and then rotates aggressively on the boundary. |
| H3: GPI finds a high-loss direction within a few steps. | Strongly verified for p2q2 | trajectory ray profile: p2q2 steepest_replace endpoint ratio vs final is `0.407` at step 1, `0.961` at step 5, `1.008` at step 10, `1.012` at step 20. | Strong support. GPI is not waiting until 300 steps to find the main high-loss direction. |
| H4: GPI/PGD/LP final deltas lie on a shared high-loss ridge rather than isolated peaks. | Verified for p2q2; partly for p2q1; caveat for p2qinf | p2q2 boundary arcs: replacement-to-additive arcs stay around `0.986` of weaker endpoint. p2qinf stronger-endpoint ratio can drop to about `0.682`. | p2q2 supports a shared high-loss ridge. q=inf weakens the clean story. |
| H5: The reason GPI works is that it follows the top singular vector of the residual Jacobian. | Tested and rejected as a strong explanation | current residual-Jacobian/SVD probe: p2q2 step5 has `sigma1/sigma2 = 5.094`, but top singular vector cosine with replacement final delta is low; replacement final state cosine with replacement final delta is only `0.281`. | Not supported. There can be a dominant local residual-Jacobian direction, but GPI final delta is not simply that direction. |
| H6: A better explanation is affine/nonlinear Loss3 structure, roughly `||b + J delta||`, not pure `||J delta||`. | Supported indirectly; not fully solved analytically | prior ablation: objective-gradient replacement final loss `6.805`, affine JVP/VJP replacement `3.842`, pure JVP/VJP replacement `2.324` in p2q2 eps8 alpha0.3. | Strong qualitative support. Pure operator-norm/SVD story is wrong; objective-gradient full Loss3 surrogate is better. A direct affine trust-region solve is still not done. |
| H7: Additive methods are slower because they have directional inertia. | Verified by angle traces and stepwise metrics | replacement post-hit angle about `30.68 deg`; raw_add after-hit angle about `0.288 deg`; steepest_add about `0.617 deg` in p2q2 300-step rollup. | Strong support. Replacement can rotate on boundary much faster than additive methods. |
| H8: Large tangent residual predicts future improvement. | Verified at setting level; weak at one-step level | p2q2 setting-level hit-to-final correlations: raw_add `0.816`, steepest_add `0.702`, replacement `0.622`. Stepwise tangent-to-next-gain correlations are much lower, replacement about `0.029`. | Tangent residual is a room-to-improve signal, not a complete next-step predictor. Direction and nonlinear geometry also matter. |
| H9: LP-steepest norm growth is straight because direction is fixed. | Rejected/refined | radial growth probe: steepest_add direction norm CV about `3.3e-8`, but cos(theta) CV about `0.478`. | Direction is not fixed. The straighter norm curve comes from fixed update norm plus mostly positive radial alignment. |
| H10: Raw PGD norm growth is curved because raw gradient scale varies. | Verified | raw_add direction L2 CV about `0.621`; actual radial increment CV also about `0.621`. | Strong support. Raw gradient magnitude variation explains much of raw PGD's uneven boundary approach. |
| H11: GPI is always best at final loss. | Rejected | p2q2 300-step sweep: some settings have steepest_add or raw_add final mean loss slightly higher than replacement. | GPI is best described as fast/early strong/stable, not unconditional final-loss winner. |
| H12: p/q geometry controls whether perturbations become spiky. | Strongly supported from existing visual/smoothness metrics | p1qinf steepest methods peakiness about `30`, top-1 energy fraction about `0.86-0.96`; p2q2 peakiness about `3.6-4.1`, top-1 energy about `0.014-0.018`. | Strong support. p=1 and q=inf can create concentrated/spiky perturbations; p2q2 is smoother. |
| H13: p2qinf is a warning case where tangent opportunity and angular motion are not enough. | Verified | all-p2 tangent residual: p2qinf replacement tangent at hit about `0.938`, but post-boundary gain is only about `0.156`; landscape shows steepest_add can be stronger. | Strong support. Direction must align with useful loss geometry; large tangent residual alone is not enough. |
| H14: p1qinf breaks the smooth dominant-ridge story. | Partly verified | current SVD: p1qinf replacement final `sigma1/sigma2 = 1.235`; top direction nearly orthogonal to replacement final delta. Existing spike metrics show severe concentration. | Supportive evidence. More samples would be needed for population-level statement. |
| H15: The current successful method should be described as objective-gradient replacement, not strict generalized-power iteration for Loss3. | Strongly verified | ablation + SVD mismatch + affine/nonlinear loss reasoning. | This is the safest interpretation. |

## What Was Actually Run/Used

Current 2026-05-21 outputs:

- `forensics/loss3_core4_pq_landscape_probe_full_20260521/`
- `forensics/loss3_core4_pq_landscape_probe_trajectory_20260521/`
- `forensics/loss3_current_core4_jacobian_svd_probe_20260521/`
- `forensics/loss3_current_mechanism_validation_summary_20260521/`

Current 2026-05-21 docs:

- `docs/loss3_core4_pq_landscape_probe_full_20260521.md`
- `docs/loss3_core4_pq_landscape_probe_trajectory_20260521.md`
- `docs/loss3_current_core4_jacobian_svd_probe_20260521.md`
- `docs/loss3_current_mechanism_validation_summary_20260521.md`

Earlier outputs reused as evidence:

- `forensics/loss3_all_p2_tangent_geometry_probe_20260520/`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/`
- `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`
- `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/`
- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/`
- `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520/`

## What Is Still Not Fully Verified

The following are not fully settled:

1. Full population-level residual-Jacobian/SVD evidence. The current SVD probe is targeted at sample index `0`, not all 100 samples.
2. Direct affine trust-region comparison for `max ||b + J delta||` against objective-gradient replacement. Existing ablation supports the affine/nonlinear explanation, but a direct solver comparison has not been run.
3. Full Hessian/eigenvalue landscape analysis across all PQ and alpha/epsilon settings. Current curvature is finite-difference pilot-scale.
4. A rigorous theorem that replacement/GPI should be optimal. Current evidence does not support such a theorem.

## Best Current Explanation

Observed evidence supports this explanation:

```text
GPI/replacement is strong because it is an aggressive full-budget objective-gradient surrogate.
It immediately uses the epsilon budget and can rotate rapidly on the boundary.
In p2q2-like geometry, the loss landscape contains a shared high-loss ridge/region,
so a few replacement steps already reach a direction whose boundary loss is close to final.
```

Observed evidence rejects this stronger explanation:

```text
GPI works because it simply follows the top singular vector of the residual Jacobian.
```

That stronger statement is not supported because the residual-Jacobian top singular direction has low cosine with replacement final delta in the current SVD probe.
