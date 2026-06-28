# Loss3 Key Findings Master Summary - 2026-05-21

Status: consolidated Markdown record from existing Loss3 experiment tables, figures, forensics outputs, and prior result notes. No new model experiment was run for this summary.

## Main Conclusion

Observed from the current Loss3 alpha/epsilon, p/q, 100-step, 300-step, giftrace, tangent-geometry, stepwise-gain, and landscape-probe records:

The main story is not simply "which method reaches the epsilon boundary." The stronger story is:

```text
Reaching the boundary solves only the radius problem.
The important part is how fast and how usefully the method can rotate direction on the boundary.
```

For the checked p2q2 and many p2q1 settings, replacement/GPI-style methods are strong because they:

- use the full epsilon budget immediately;
- still have large tangent gradient left at boundary hit;
- can make large direction changes on the boundary;
- reach a final-like perturbation direction within a few steps;
- often produce final perturbation shapes similar to additive PGD/LP-steepest methods, but much faster.

This does not mean replacement/GPI is the exact mathematical optimizer of Loss3. The better interpretation is:

```text
objective-gradient replacement is a very effective full-budget local surrogate
for the current Loss3 landscape, especially when the landscape has a dominant direction/ridge.
```

## Boundary Is Not Convergence

Observed from:

- `docs/loss3_post_boundary_mechanism_p2q2_300steps_20260520.md`
- `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/geometry_probe_summary_by_method.csv`

Key evidence in p2q2 300-step rollup:

| method | mean boundary hit step | mean post-boundary gain | tangent residual at hit | mean post-hit angle |
| --- | ---: | ---: | ---: | ---: |
| raw_add | 32.41 | 1.285 | 0.397 | 0.288 deg |
| steepest_add | 12.75 | 1.470 | 0.435 | 0.617 deg |
| raw_replace | 1 | 3.412 | 0.855 | 30.68 deg |
| steepest_replace | 1 | 3.412 | 0.855 | 30.68 deg |

Inference from this evidence:

- Replacement/GPI reaches the boundary at step 1, but at that point it is far from boundary stationary.
- Its tangent gradient residual is large, so there is still substantial room to improve by rotating along the boundary.
- Additive methods reach boundary later and rotate much more slowly after reaching it.

For p=2, the diagnostic quantity is:

```text
||(I - u u^T) grad L|| / ||grad L||, where u = delta / ||delta||_2
```

This measures how much of the gradient lies in the tangent plane of the L2 boundary. If this is large, being on the boundary does not mean the direction is optimized.

## Correlation Results Are Two Different Questions

Observed from:

- `docs/loss3_correlation_and_radial_growth_clarification_20260521.md`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/geometry_probe_correlations.csv`
- `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/tables/stepwise_geometry_correlations_by_method_scope.csv`

Setting-level hit-to-final question:

```text
Across 20 p2q2 alpha/epsilon settings, does tangent residual at first boundary hit
predict total remaining post-boundary gain?
```

Observed correlations:

| method | r |
| --- | ---: |
| raw_add | 0.816 |
| steepest_add | 0.702 |
| replacement/GPI | 0.622 |

Stepwise post-boundary next-step question:

```text
At each post-boundary step, does tangent residual at step k predict immediate gain L[k+1]-L[k]?
```

Observed correlations:

| method | r |
| --- | ---: |
| raw_add | 0.337 |
| steepest_add | 0.265 |
| replacement/GPI | 0.029 |

Inference:

- The high and low correlations are not contradictory.
- Tangent residual at boundary hit predicts total room to improve.
- Tangent residual at a single step does not fully predict the next step's loss gain, especially for replacement/GPI.
- For next-step gain, the actual projected proposal direction matters, not just tangent magnitude.

The more direct one-step predictor is:

```text
linear_gain_projected = grad L dot (projected_next_delta - current_delta)
```

Observed post-boundary correlations with immediate next gain:

| method | projected linear gain r |
| --- | ---: |
| raw_add | 0.600 |
| steepest_add | 0.299 |
| replacement/GPI | 0.097 |

## GPI / Replacement Is Fast, But Not Always Final-Loss Best

Observed from:

- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`
- `docs/loss3_post_boundary_mechanism_p2q2_300steps_20260520.md`
- `docs/loss3_gpi_overall_conclusion_20260520.md`

Inference from 300-step p2q2 records:

- Replacement/GPI is usually much faster in early optimization.
- It often reaches a strong perturbation direction in roughly 5-10 steps.
- It is not guaranteed to have the largest final 300-step mean loss.
- In some p2q2 alpha/epsilon settings, additive methods such as steepest_add or raw_add can slowly catch up or slightly exceed replacement/GPI final mean loss.

Practical interpretation:

```text
GPI/replacement is strong for speed, stability, and early high loss.
It should not be claimed as an unconditional global optimizer of Loss3.
```

## Early GPI Direction Becomes Final-Like Quickly

Observed from:

- `docs/loss3_gpi_early_step_comparison_20260520.md`
- `docs/loss3_core4_pq_landscape_probe_20260520.md`
- `forensics/loss3_gpi_early_step_comparison_20260520/`
- `forensics/loss3_core4_pq_landscape_probe_20260520/`

Representative p2q2 cosine-to-final evidence:

| setting | step 5 | step 10 | step 20 |
| --- | ---: | ---: | ---: |
| baseline eps=4 alpha=0.4 | 0.887 | 0.984 | 0.995 |
| eps=8 alpha=0.3 | 0.788 | 0.966 | 0.988 |

Landscape-probe p2q2 ray endpoint evidence:

| direction | endpoint loss at epsilon |
| --- | ---: |
| replacement step 1 | 1.481 |
| replacement step 5 | 3.494 |
| replacement step 10 | 3.664 |
| replacement step 20 | 3.679 |
| replacement final | 3.634 |

Inference:

- In p2q2, replacement/GPI seems to find a final-like high-loss direction very quickly.
- This supports a dominant-direction or high-loss-ridge explanation.
- It is evidence, not a proof of a spectral dominant mode. A stronger spectral claim would require a local Jacobian/SVD or Hessian-style probe on the current core4/PQ deltas.

## GPI Is Not The Strict Generalized-Power Optimizer Of Loss3

Observed from:

- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`
- `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/`

Important p2q2 eps=8 alpha=0.3 ablation:

| method family | final loss |
| --- | ---: |
| objective-gradient replacement / steepest_replace | 6.805 |
| objective-gradient additive / steepest_add | 6.377 |
| raw_add | 5.390 |
| affine JVP/VJP replacement | 3.842 |
| pure JVP/VJP replacement | 2.324 |

Inference:

- The method that performs well is objective-gradient replacement.
- It should not be described as the exact generalized power method for the full nonlinear Loss3 objective.
- The safer explanation is that objective-gradient replacement is a good local full-budget surrogate for this loss landscape.

## LP-Steepest Norm Growth Is Straight-Ish For A More Specific Reason

Observed from:

- `docs/loss3_correlation_and_radial_growth_clarification_20260521.md`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/radial_growth_rollup_by_method.csv`

The additive update radius obeys:

```text
r_next^2 = r^2 + 2 alpha r ||d|| cos(theta) + alpha^2 ||d||^2
```

So fixed step length alone does not mathematically guarantee linear norm growth. Direction alignment also matters.

Observed p2q2 20-setting rollup:

| method | mean cos(theta) | CV cos(theta) | mean direction L2 | CV direction L2 | CV radial increment |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_add | 0.869 | 0.247 | 0.359 | 0.621 | 0.621 |
| steepest_add | 0.700 | 0.478 | 1.000 | 3.3e-8 | 0.110 |

Inference:

- The evidence does not say steepest_add has a fixed direction.
- The evidence says steepest_add has fixed update norm and mostly positive radial alignment, so radial increments are much more regular.
- Raw PGD has strongly varying raw gradient norm, and that variation closely tracks its curved/uneven radius growth.

## Boundary-Ratio Standard Deviation Is A Useful Diagnostic

Observed from:

- `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`
- `docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md`
- `docs/loss3_post_boundary_mechanism_diagnostics_20260520.md`

Inference:

- Raw PGD has larger boundary-ratio standard deviation because raw gradient magnitudes vary by sample.
- LP-steepest has much smaller boundary-ratio standard deviation because its direction norm is normalized.
- Replacement/GPI has near-zero boundary-arrival variance when it jumps to the boundary at step 1.

This is useful because it separates:

```text
method cannot find a good direction
```

from:

```text
method simply takes inconsistent or tiny radial steps before reaching the boundary
```

## Final Delta Shapes Are Often Similar, But Not Always

Observed from:

- `docs/loss3_alpha_epsilon_core4_delta_similarity_20260520.md`
- `docs/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520.md`
- `docs/loss3_final_delta_similarity_results_20260518.md`

Inference:

- In p2q2 and p2q1, final perturbations from raw_add, steepest_add, and replacement/GPI are often shape-similar.
- This supports the idea that the methods are being attracted toward a shared high-loss direction or ridge.
- The difference is often path speed and boundary rotation, not an entirely different final object.
- This similarity becomes less reliable in p=1 or q=inf settings.

## P/Q Geometry Strongly Affects Smoothness And Spike Behavior

Observed from:

- `docs/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520.md`
- `docs/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520.md`
- `docs/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520.md`
- `docs/loss3_surprising_findings_validation_20260520.md`

Spike/concentration evidence:

| setting/method family | peakiness | top-1 energy fraction |
| --- | ---: | ---: |
| p1qinf steepest methods | about 30 | about 0.86-0.96 |
| p2q2 methods | about 3.6-4.1 | about 0.014-0.018 |

Inference:

- p=2 tends to spread perturbation energy more smoothly.
- p=1 has extreme-point geometry, so it can concentrate perturbation into sparse/spiky coordinates.
- q=inf focuses the output residual on the largest coordinate, so gradients can become localized.
- p2q1 looked relatively natural in the checked visualizations.
- p1qinf and p2qinf are warning cases for spike-like perturbations.

## Raw Replace And Steepest Replace Equivalence

Observed from:

- `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`
- `docs/loss3_surprising_findings_validation_20260520.md`
- `docs/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520.md`

Inference:

- In checked p=2 cases, raw_replace and steepest_replace often coincide or are effectively identical.
- This is consistent with the p=2 geometry where normalizing an objective-gradient direction and replacing delta by epsilon times that direction yields the same L2-boundary proposal.
- This should not be generalized blindly to p=1 or p=inf cases.

## PQ-Dependent Caveats

Observed from:

- `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- `docs/loss3_core4_pq_landscape_probe_20260520.md`
- `docs/loss3_alpha_epsilon_core4_pneq_q_stopped_100steps_result_20260520.md`

Important caveats:

- p2q2: strongest support for the GPI/replacement fast-boundary-rotation story.
- p2q1: broadly similar and often still natural-looking.
- p2qinf: high tangent residual and large angular motion do not always produce large gain; steepest_add can win final ray/loss in some probes.
- p1qinf: spike/concentration behavior is severe; angle motion can be large but post-boundary gain can be weak or negative.

Inference:

```text
Large tangent opportunity + large angular motion is not sufficient.
The movement also has to align with useful loss geometry.
```

## Most Defensible Paper-Style Wording

Based on the current evidence, the safe wording is:

```text
For Loss3, replacement/GPI-style objective-gradient updates are best understood
as aggressive full-budget local surrogate steps. They typically reach the
epsilon boundary immediately and then rapidly rotate along the boundary.
In p2q2 and related smooth geometries, this quickly approaches a shared
high-loss direction/ridge, explaining the strong early loss growth and
final-delta similarity. However, the method is not a guaranteed global optimizer:
with long enough additive optimization, other methods can match or exceed its
final loss, and p/q geometries such as q=inf or p=1 can produce spike-like,
less reliable behavior.
```

## Source Index

Primary synthesis/result docs:

- `docs/loss3_surprising_findings_synthesis_20260520.md`
- `docs/loss3_surprising_findings_validation_20260520.md`
- `docs/loss3_surprising_findings_math_explanation_20260520.md`
- `docs/loss3_gpi_overall_conclusion_20260520.md`
- `docs/loss3_boundary_geometry_derivation_and_validation_20260520.md`
- `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`
- `docs/loss3_p2q2_stepwise_gain_geometry_probe_20260520.md`
- `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- `docs/loss3_core4_pq_landscape_probe_20260520.md`
- `docs/loss3_correlation_and_radial_growth_clarification_20260521.md`

Primary forensics outputs:

- `forensics/loss3_alpha_epsilon_core4_analysis_20260519/`
- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/`
- `forensics/loss3_alpha_epsilon_core4_analysis_baseline_giftrace_20260520/`
- `forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520/`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/`
- `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`
- `forensics/loss3_all_p2_tangent_geometry_probe_20260520/`
- `forensics/loss3_core4_pq_landscape_probe_20260520/`

## Remaining Work

- If a stronger spectral/dominant-mode claim is needed, run a targeted local Jacobian/SVD probe on current p2q2 baseline, p2qinf warning case, and p1qinf spike case.
- If paper figures are needed, use both with-std and no-std curve versions, because standard-deviation bands can visually compress mean curves.
- Do not overclaim that replacement/GPI is the exact optimizer for Loss3; the current evidence supports a strong empirical/local-surrogate explanation.

## 2026-05-21 Current Mechanism Validation Update

New current-core4/PQ validation outputs:

- `docs/loss3_current_mechanism_validation_summary_20260521.md`
- `docs/loss3_core4_pq_landscape_probe_full_20260521.md`
- `docs/loss3_core4_pq_landscape_probe_trajectory_20260521.md`
- `docs/loss3_current_core4_jacobian_svd_probe_20260521.md`

Observed update:

- p2q2 early replacement ray ratios support fast convergence to a high-loss direction: step 5 `0.961`, step 10 `1.008`, step 20 `1.012` relative to final endpoint.
- p2q2 boundary arcs support a shared high-loss ridge: replacement-to-additive arcs stay near `0.986` of the weaker endpoint.
- p2q2 local residual-Jacobian spectrum becomes strongly dominated after GPI moves: `sigma1/sigma2` rises from `1.148` at clean to `5.094` at step 5.
- However, the top residual-Jacobian singular vector is not strongly aligned with the final replacement delta; at replacement final, cosine is only `0.281`.

Inference update:

- Keep the dominant-ridge/local-surrogate explanation.
- Do not upgrade it to "GPI follows the top residual-Jacobian singular vector." The new SVD probe argues against that stronger claim.
- This strengthens the wording that objective-gradient replacement works as an aggressive full-budget local surrogate for Loss3, not as the exact pure generalized-power/SVD optimizer.

