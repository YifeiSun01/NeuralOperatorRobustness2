# Loss3 Replace/Add Mechanism Master Log - 2026-06-24

## 0. Purpose

This log consolidates the discussion, theory, experiments, evidence, and final
interpretation for the question:

Why does `steepest_replace` work well on Burgers and Darcy Flow, but not on
NS2D, while `steepest_add` is more robust on NS2D?

This is the current source-of-truth record for the Loss3 replace/add mechanism
study. It intentionally records conclusions and artifact locations, but does not
record any cloud or Git credentials.

All paths below are relative to the repository root:

`/workspace/NeuralOperatorRobustness2`

## 1. Final Conclusion

The strongest current conclusion is:

\[
\texttt{steepest\_replace}
\text{ works when the full-budget jump lands in a broad, forgiving, high-loss basin.}
\]

For the three systems:

| System | Current mechanism conclusion |
| --- | --- |
| Burgers 1D | `steepest_replace` works because it reaches an optimizer-relevant high-loss basin that is broad enough that many nearby boundary directions still have high Loss3. |
| Darcy Flow | `replace` has a discrete analogue of the same phenomenon: many near-optimal flip sets exist, and replacement-like selection finds a strong set quickly. |
| NS2D | `steepest_replace` lands on the epsilon boundary but outside the high-loss basin reached by `steepest_add`; `add` wins because it follows a curved/path-dependent route and updates direction gradually. |

The short version is:

\[
\text{Burgers/Darcy: replace does not need to be very precise.}
\]

\[
\text{NS2D: replace must be precise, and it is not.}
\]

The supported explanation is not simply:

\[
\text{Burgers is linear, NS2D is nonlinear.}
\]

Linearity/direction stability matters, but the more direct experimental variable
is:

\[
\text{the size, width, and connectivity of the high-loss region on the feasible boundary.}
\]

## 2. What We Should Not Claim

The experiments do not justify these stronger statements:

- Burgers has high loss over the whole epsilon boundary.
- Burgers is purely linear.
- Burgers `replace` is good because its full-radius first-order prediction is accurate.
- Burgers is controlled only by one or two top singular/Jacobian modes.
- The whole mechanism is completely explained by top tangent directions.
- We mathematically proved a global theorem that `replace` is better than `add`.

The safe wording is:

> Experiments strongly support that Burgers `steepest_replace` reaches a broad optimizer-relevant high-loss basin, whereas NS2D `steepest_replace` remains outside the high-loss basin reached by `steepest_add`. Endpoint-local Jacobian/Gauss-Newton modes are causally important for preserving high loss near the Burgers replace endpoint, but the basin width is not reducible to a simple top-tangent-versus-orthogonal-tangent story alone.

## 3. Optimizer Definitions

Let the attack perturbation be \(\delta\), constrained by:

\[
\|\delta\| \le \epsilon.
\]

Let the Loss3 attack objective be:

\[
L(\delta).
\]

At step \(k\), define:

\[
g_k = \nabla_\delta L(\delta_k).
\]

Let \(d_k\) be the normalized steepest direction under the chosen norm geometry.
The two key update styles are:

### Add

\[
\delta_{k+1}
=
\Pi_{\|\delta\|\le \epsilon}
\left(
\delta_k + \alpha d_k
\right).
\]

Interpretation: `add` accumulates a path. It can keep rotating direction and
correct itself after reaching the boundary.

### Replace

\[
\delta_{k+1}
=
\Pi_{\|\delta\|\le \epsilon}
\left(
\epsilon d_k
\right).
\]

Interpretation: `replace` discards the previous perturbation and jumps directly
to the full-budget point in the current direction. It is fast if the current
direction already points into a good high-loss region, but fragile if the good
region is narrow or if the correct path bends.

## 4. Local Theory: Why Replace Can Look Like Power Iteration, But Only Under Conditions

For Loss3, write the model-solver error locally as:

\[
e(x+\delta)
=
F(x+\delta)-S(x+\delta).
\]

A local first-order expansion gives:

\[
e(x+\delta)
\approx
b + A\delta,
\]

where:

\[
b = e(x),
\qquad
A = J_F(x)-J_S(x).
\]

Then:

\[
L(\delta)
=
\|e(x+\delta)\|^2
\approx
\|b + A\delta\|^2.
\]

Expanding:

\[
L(\delta)
\approx
\|b\|^2
+
2 b^T A\delta
+
\delta^T A^T A\delta.
\]

So even in the locally linearized model, the Loss3 objective is not necessarily a
pure centered quadratic. It has both:

\[
\text{linear term } 2A^T b,
\]

and:

\[
\text{quadratic term } A^T A.
\]

If \(b=0\), the problem is closer to a pure quadratic on a sphere:

\[
L(\delta) \approx \delta^T H\delta,
\qquad H=A^TA.
\]

Then gradient replacement resembles a power-iteration-like update:

\[
\delta_{k+1}
\propto
\nabla L(\delta_k)
=
2H\delta_k.
\]

In that special case, `replace` can rapidly align with the dominant eigendirection.

But for the actual Loss3 problem:

\[
\nabla L(\delta)
\approx
2A^T(b+A\delta).
\]

This contains both the bias/residual term and the quadratic term. Also, for the
real neural operator and solver, \(A\) and higher-order terms can change along the
path. Therefore, there is no unconditional theory saying `replace` must beat
`add`.

The theory predicts conditions:

| Condition | Expected behavior |
| --- | --- |
| High-loss region is broad and connected | `replace` can work well because many full-budget directions are good. |
| Local direction is stable enough | `replace` can jump to a good region quickly. |
| Residual/bias term \(A^Tb\) is strong but direction is coherent | `replace` can still be fast even though it is not a pure quadratic method. |
| High-loss region is narrow or path-dependent | `replace` can miss; `add` is safer. |
| Direction changes strongly after reaching boundary | `add` can keep correcting; `replace` can keep resetting into bad regions. |
| Full-radius surrogate is inaccurate | `replace` loses theoretical reliability unless the basin is forgiving. |

This is exactly the pattern found experimentally.

## 5. Main Optimizer Results To Explain

The average Loss3 optimizer curves showed the empirical facts that the mechanism
experiments must explain.

| System | Sample count | Observation |
| --- | ---: | --- |
| Burgers 1D | N=100 formal curves | `steepest_add = 6.80804`, `raw_replace = 6.80783`, `steepest_replace = 6.80783`, `raw_add = 6.41518`. Add and replace are nearly tied; replace is very strong. |
| Darcy Flow | N=20 formal/flip-set diagnostics | `raw_replace = steepest_replace = 0.04996`, `raw_add = steepest_add = 0.04518`. Replace ends higher and reaches high fraction much faster. |
| NS2D | N=20 formal curves | `steepest_add = 278.54`, `raw_add = 129.86`, `raw_replace = steepest_replace = 96.42`. Replace is much worse than steepest add. |

Source note:

- `docs/loss3_high_loss_region_landscape_hypothesis_20260623.md`

## 6. Burgers Evidence

### 6.1 First-order prediction: Burgers replace is not simply accurate linear prediction

For a candidate next perturbation \(\delta_{\mathrm{cand}}\), define:

\[
\Delta L_{\mathrm{true}}
=
L(\delta_{\mathrm{cand}})-L(\delta_k),
\]

and first-order prediction:

\[
\Delta L_{\mathrm{pred}}
=
g_k^T(\delta_{\mathrm{cand}}-\delta_k).
\]

Relative prediction error:

\[
\mathrm{RelErr}
=
\frac{|\Delta L_{\mathrm{pred}}-\Delta L_{\mathrm{true}}|}
{|\Delta L_{\mathrm{true}}|+\eta}.
\]

Measured Burgers result:

| Candidate | Relative first-order error |
| --- | ---: |
| `add` | 0.0289 |
| `replace` | 3.407 |

Interpretation:

If Burgers `replace` worked only because the full-radius landscape was almost
linear and first-order prediction was accurate, the `replace` error should be
small. It is not. Therefore the stronger explanation is not “Burgers is linear,
so replace is accurate.”

The better explanation is:

\[
\text{Even if the full-budget prediction is inaccurate, replace still lands in a broad high-loss basin.}
\]

Sources:

- `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_first_order_prediction`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_first_order_prediction_N20`
- `docs/loss3_full_mechanism_validation_summary_20260623.md`

### 6.2 Burgers boundary arc: broad ridge between replace and add endpoints

For Burgers, boundary arc between:

\[
A=\delta_{\texttt{steepest\_replace}},
\qquad
B=\delta_{\texttt{steepest\_add}}.
\]

Representative endpoint losses:

| Endpoint | Loss3 |
| --- | ---: |
| `steepest_replace` | about 7.069 |
| `steepest_add` | about 6.339 |

The weaker endpoint is `steepest_add`. Along 17 sampled boundary-arc points:

\[
\min_{\mathrm{arc}} L = 6.172.
\]

Ratio to weaker endpoint:

\[
\frac{6.172}{6.339}=0.974.
\]

Also, 100% of the 17 arc points were above:

\[
0.95 \times L(\text{weaker endpoint}).
\]

Interpretation:

The high-loss region between Burgers `replace` and `add` endpoints is not a sharp
isolated point. It is a broad ridge. `replace` can be slightly wrong and still do
well.

Sources:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/boundary_arc.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/boundary_arc_aggregate.csv`
- `docs/loss3_high_loss_region_landscape_hypothesis_20260623.md`

### 6.3 Burgers 2D slice: local high-loss area is broad

Local 2D slice results around final perturbations:

| Center | Fraction >= 90% center loss | Fraction >= 95% center loss |
| --- | ---: | ---: |
| `steepest_replace` | 96% | 72% |
| `steepest_add` | 100% | 84% |

Interpretation:

Low-dimensional slices near the final Burgers solutions show a wide high-loss
area, not a needle-like optimum.

Sources:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/slice_2d_grid.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/slice_2d_aggregate.csv`

### 6.4 Burgers ray evidence: early replace directions already reach high-loss boundary

Ray endpoints along early `steepest_replace` trajectory directions:

| Direction source | Endpoint Loss3 |
| --- | ---: |
| trajectory step 5 | 6.883 |
| trajectory step 10 | 7.027 |
| trajectory step 20 | 7.105 |
| final | 7.069 |

Interpretation:

Burgers `replace` finds a final-like high-loss boundary direction very early.
This matches the broad-basin story.

Sources:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/ray_profile.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/ray_profile_aggregate.csv`

## 7. Direct High-Loss Region Tests: Same-sample Lmax Normalization

The stricter test defines, for each sample:

\[
L_{\max}
=
\max_m L(\delta_m),
\]

where \(m\) ranges over optimizer endpoints. Then cap points around an endpoint
are scored by:

\[
\frac{L(\delta_{\mathrm{cap}})}{L_{\max}}.
\]

The high-loss cap probability is:

\[
p_\tau(\theta)
=
\mathbb{P}[L(\delta_{\mathrm{cap}}(\theta))\ge \tau L_{\max}].
\]

This avoids a misleading situation where a bad endpoint looks locally flat just
because everything around it is similarly bad.

### 7.1 Burgers replace cap

| Endpoint | theta | N | p(L >= 0.95 Lmax) | mean L/Lmax |
| --- | ---: | ---: | ---: | ---: |
| Burgers `steepest_replace` | 0.00 | 40 | 0.600 | 0.907 |
| Burgers `steepest_replace` | 0.05 | 40 | 0.600 | 0.906 |
| Burgers `steepest_replace` | 0.10 | 40 | 0.400 | 0.902 |
| Burgers `steepest_replace` | 0.20 | 40 | 0.350 | 0.884 |
| Burgers `steepest_replace` | 0.40 | 40 | 0.000 | 0.814 |

### 7.2 NS2D replace cap

| Endpoint | theta | N | p(L >= 0.95 Lmax) | mean L/Lmax |
| --- | ---: | ---: | ---: | ---: |
| NS2D `steepest_replace` | 0.00 | 16 | 0.000 | 0.274 |
| NS2D `steepest_replace` | 0.05 | 16 | 0.000 | 0.274 |
| NS2D `steepest_replace` | 0.10 | 16 | 0.000 | 0.272 |
| NS2D `steepest_replace` | 0.20 | 16 | 0.000 | 0.267 |
| NS2D `steepest_replace` | 0.40 | 16 | 0.000 | 0.250 |

### 7.3 NS2D add cap

| Endpoint | theta | N | p(L >= 0.95 Lmax) | mean L/Lmax |
| --- | ---: | ---: | ---: | ---: |
| NS2D `steepest_add` | 0.00 | 16 | 1.000 | 1.000 |
| NS2D `steepest_add` | 0.05 | 16 | 1.000 | 0.999 |
| NS2D `steepest_add` | 0.10 | 16 | 1.000 | 0.996 |
| NS2D `steepest_add` | 0.20 | 16 | 1.000 | 0.982 |
| NS2D `steepest_add` | 0.40 | 16 | 0.000 | 0.923 |

Interpretation:

This is one of the cleanest pieces of evidence. Burgers `replace` is near the
sample-level best observed high-loss region. NS2D `replace` is not; its endpoint
neighborhood is only about 27% of \(L_{\max}\). NS2D `add` is the one sitting near
\(L_{\max}\).

Sources:

- `tools/probe_loss3_boundary_volume_20260623.py`
- `tools/analyze_loss3_boundary_cap_lmax_20260623.py`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/tables/endpoint_cap_volume_lmax_summary.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/figures/endpoint_cap_lmax_p095_by_theta.svg`
- `docs/loss3_boundary_cap_lmax_findings_20260623.md`

## 8. Dominant Subspace / Coherent Mode Evidence

Hypothesis tested:

Burgers loss might be dominated by a few coherent modes, so many directions that
excite these modes produce high Loss3.

The iso-projection test samples directions with controlled projection into an
endpoint/coherent subspace. Key result for \(k=8\), \(\beta=1\):

| System | Endpoint | k | beta | N | p(L >= 0.95 Lmax) | mean L/Lmax |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Burgers | `steepest_replace` | 8 | 1.0 | 80 | 0.600 | 0.919 |
| NS2D | `steepest_replace` | 8 | 1.0 | 32 | 0.000 | 0.274 |

For Burgers `steepest_replace`, even subspace-controlled perturbations remain
near high loss. For NS2D `steepest_replace`, they remain low.

Interpretation:

This supports a forgiving medium-dimensional local structure around the Burgers
replace endpoint, and confirms that NS2D replace is outside the high-loss basin.
It does not prove that only the top modes matter globally.

Sources:

- `tools/probe_loss3_dominant_subspace_iso_projection_20260623.py`
- `tools/run_loss3_dominant_subspace_iso_projection_20260623.sh`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/dominant_subspace_iso_projection_20260623/tables/iso_projection_summary.csv`

## 9. Jacobian/Gauss-Newton Mode Causal Ablation

This is the strongest causal evidence about whether endpoint-local Jacobian/GN
structure matters.

The test constructs endpoint-local bases and evaluates two operations:

- `keep k`: keep only the first \(k\) endpoint-local modes.
- `remove k`: remove the first \(k\) endpoint-local modes.

Key result for `steepest_replace`, endpoint basis, \(k=8\), threshold 0.95:

| System | Operation | k | N | p(L >= 0.95 Lmax) | mean L/Lmax |
| --- | --- | ---: | ---: | ---: | ---: |
| Burgers | keep | 8 | 5 | 0.600 | 0.944 |
| Burgers | remove | 8 | 5 | 0.000 | 0.261 |
| NS2D | keep | 8 | 2 | 0.000 | 0.157 |
| NS2D | remove | 8 | 2 | 0.000 | 0.246 |

Interpretation:

For Burgers, endpoint-local modes are causally important: keeping them preserves
high loss; removing them destroys high loss. For NS2D, keeping the replace
endpoint modes does not rescue the candidate, because the replace endpoint is
already outside the high-loss basin.

Sources:

- `tools/probe_loss3_jacobian_mode_causal_ablation_20260623.py`
- `tools/run_loss3_jacobian_mode_causal_ablation_20260623.sh`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623/tables/jacobian_mode_keep_remove_summary.csv`

## 10. Basin Attraction From Replace Endpoint

Experiment:

Start from the `steepest_replace` endpoint and continue steepest-style
optimization for 20 steps. If replace landed in a good basin, continuing should
reach or stay near high Loss3. If it landed outside, continuing may not reach the
best basin.

| System | Start | N | final mean L/Lmax | p(final >= 0.95 Lmax) | mean gain from start |
| --- | --- | ---: | ---: | ---: | ---: |
| Burgers | `steepest_replace` | 5 | 1.044 | 0.800 | 0.853 |
| NS2D | `steepest_replace` | 2 | 0.700 | 0.000 | 113.747 |

Interpretation:

Burgers replace endpoint is in or near a basin that can continue to near-maximal
loss. NS2D replace endpoint can improve, but still does not reach the add-level
high-loss basin; its final ratio remains only about 0.70 of \(L_{\max}\), with
zero samples above 0.95 \(L_{\max}\).

Source:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623/tables/basin_attraction_summary.csv`

## 11. Matched Tangent Width Check

Experiment:

From the replace endpoint, move by the same angular amount along:

- endpoint-local top-Jacobian tangent directions;
- orthogonal tangent directions.

Key results for `steepest_replace`, \(k=8\), threshold 0.95:

| System | Direction family | theta | N | p(L >= 0.95 Lmax) | mean L/Lmax |
| --- | --- | ---: | ---: | ---: | ---: |
| Burgers | orth tangent | 0.10 | 40 | 0.400 | 0.902 |
| Burgers | top tangent | 0.10 | 40 | 0.475 | 0.899 |
| Burgers | orth tangent | 0.20 | 40 | 0.400 | 0.884 |
| Burgers | top tangent | 0.20 | 40 | 0.350 | 0.873 |
| NS2D | orth tangent | 0.10 | 16 | 0.000 | 0.272 |
| NS2D | top tangent | 0.10 | 16 | 0.000 | 0.277 |
| NS2D | orth tangent | 0.20 | 16 | 0.000 | 0.268 |
| NS2D | top tangent | 0.20 | 16 | 0.000 | 0.270 |

Interpretation:

This test does not support a simple claim that only top tangent directions explain
the Burgers basin. Both top and orthogonal tangent families retain high loss at
small angles in Burgers. NS2D remains low for both families. Therefore the
supported statement is broader basin geometry, not a one-line top-mode story.

Sources:

- `tools/probe_loss3_jacobian_tangent_width_20260623.py`
- `tools/run_loss3_jacobian_tangent_width_20260623.sh`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_tangent_width_20260623/tables/jacobian_tangent_width_summary.csv`

## 12. Darcy Flow Evidence: Discrete Analogue Of Broad Near-optimal Choices

Darcy Flow is a binary/discrete flip-set problem, so the geometry is not the same
as continuous Burgers/NS2D boundary geometry. But the mechanism is analogous:
there are multiple good final flip sets, and replacement-like selection reaches a
strong set quickly.

### 12.1 Speed and final loss

| Method | N | final Loss3 mean | k50 mean | k90 mean | k95 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_add` / `steepest_add` | 20 | 0.04518 | 41.6 | 78.45 | 82.95 |
| `raw_replace` / `steepest_replace` | 20 | 0.04996 | 2.05 | 6.50 | 9.20 |

Interpretation:

Darcy replace reaches 90% of its final improvement in about 6.5 steps on average,
while add takes about 78.45 steps. Replace also ends higher on average.

### 12.2 Final flip-set overlap

Between `steepest_add` and `steepest_replace`:

| Metric | Mean |
| --- | ---: |
| Jaccard overlap | 0.2568 |
| overlap/min-count | 0.3538 |
| loss ratio replace/add | 1.1539 |

Interpretation:

Add and replace often choose different final flip sets, and replace is often
higher. This is not just the same final set reached faster; there are multiple
near-optimal discrete choices.

### 12.3 Near-optimal method count

Among the four method endpoints:

| Metric | Mean |
| --- | ---: |
| number of methods within 90% of best | 2.8 |
| number of methods within 95% of best | 2.8 |

Interpretation:

Darcy has a discrete near-optimal multiplicity: several method endpoints are close
to the best on each sample. This is the discrete version of a forgiving high-loss
region.

Sources:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/darcy_flipset_mechanism/speed_aggregate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/darcy_flipset_mechanism/final_pairwise_flip_overlap_aggregate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/darcy_flipset_mechanism/near_optimal_method_sets_aggregate.csv`

## 13. NS2D Evidence

### 13.1 First-order and linearity probes

NS2D exact mechanism probes:

| Run | N | add first-order relative error | replace first-order relative error | replace linearity c_phi mean |
| --- | ---: | ---: | ---: | ---: |
| NS2D exact | 3 | 12.45 | 26.19 | 14.20 |
| NS2D top-up | 2 | 7.88 | 18.53 | 6.63 |

Interpretation:

Full-budget first-order replacement prediction is poor in NS2D. Add is also not
perfectly predicted, but replace is worse. This supports path-dependence and
large-radius nonlinearity.

Sources:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism/first_order_by_method_candidate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism/linearity_aggregate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup/first_order_by_method_candidate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup/linearity_aggregate.csv`

### 13.2 NS2D boundary arc: replace and add endpoints are not connected by a high-loss ridge

N=3 exact probe:

| Quantity | Value |
| --- | ---: |
| `steepest_replace` endpoint | 85.09 |
| `steepest_add` endpoint | 255.48 |
| replace/add ratio | about 0.333 |
| arc points >= 95% add endpoint | 1/9 |

N=2 top-up:

| Quantity | Value |
| --- | ---: |
| `steepest_replace` endpoint | 114.54 |
| `steepest_add` endpoint | 337.49 |
| replace/add ratio | about 0.339 |
| arc points >= 95% add endpoint | 2/9 |

Interpretation:

NS2D replace and add endpoints are not sitting on one shared broad high-loss
ridge. Replace is far below add, and most of the arc does not reach add-level
high loss.

Sources:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism/boundary_arc_aggregate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup/boundary_arc_aggregate.csv`

### 13.3 NS2D post-boundary gain

| Run | Method | Post-boundary gain mean |
| --- | --- | ---: |
| N=3 exact | `steepest_replace` | -51.05 |
| N=3 exact | `steepest_add` | +43.43 |
| N=2 top-up | `steepest_replace` | -59.43 |
| N=2 top-up | `steepest_add` | +52.80 |

Interpretation:

Reaching the boundary is not enough. NS2D replace reaches the boundary quickly,
but after that it does not continue improving along the right boundary region.
Add still gains after boundary because it keeps rotating/correcting along the
path.

Source:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/cross_system_summary/mechanism_summary_rows.csv`

### 13.4 NS2D JVP/VJP candidate evidence

The NS2D JVP/VJP experiment estimates local Jacobian-vector and vector-Jacobian
products without materializing the full Jacobian. It tests local top-singular and
surrogate explanations.

Key qualitative result from the candidate comparisons:

| Candidate family | Mean true gain |
| --- | ---: |
| add-style candidates | positive on average, about +20.68 in the summarized comparison |
| replace-style candidates | negative on average, about -48.13 in the summarized comparison |

Even top-singular replacement-style full-radius moves often had negative true
gain. This means the problem is not simply “NS2D has no local ascent direction.”
It has local directions, but full-budget replacement moves can land in poor
regions.

Sources:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N3/candidate_comparison_aggregate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N3/quadratic_surrogate_aggregate.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/cross_system_summary/ns2d_jvp_vjp_key_metrics.csv`
- `docs/loss3_ns2d_jvp_vjp_integration_summary_20260623.md`

## 14. Answer To The Core Scientific Question

### Why is Burgers replace good?

Because Burgers `replace` reaches a high-loss basin that is broad enough to be
forgiving. Evidence:

- Main curve: replace is essentially tied with steepest add on N=100.
- Boundary arc: worst sampled arc point still has 97.4% of weaker endpoint loss.
- 2D slice: 72% of `steepest_replace` local grid points are above 95% center loss.
- Ray test: early directions already reach final-like high loss.
- Lmax cap: `steepest_replace` has p0.95 = 0.6 at theta 0 and 0.35 at theta 0.2.
- Iso-projection: k=8, beta=1 gives p0.95 = 0.6 and mean L/Lmax = 0.919.
- Jacobian keep/remove: keeping endpoint k=8 modes gives p0.95 = 0.6, removing gives p0.95 = 0.
- Basin attraction: continuing from replace reaches final mean L/Lmax = 1.044, p0.95 = 0.8.

### Why is this not just a linearity story?

Because Burgers full-budget replace first-order prediction error is large:

\[
\mathrm{RelErr}_{replace}=3.407.
\]

If the entire explanation were “Burgers is linear, so replace is accurately
predicted,” this number should be small. It is not. The better statement is:

\[
\text{Burgers replace can be inaccurate locally but still successful because the target high-loss basin is wide.}
\]

### Why is NS2D replace bad?

Because NS2D replace lands outside the high-loss basin reached by add. Evidence:

- Main curve: `steepest_add` around 278.54, replace around 96.42.
- Lmax cap: NS2D replace p0.95 = 0, mean L/Lmax about 0.274 near theta 0.
- NS2D add cap: p0.95 = 1.0 up to theta 0.2, mean L/Lmax near 1.
- Boundary arc: replace/add ratio about 0.333 to 0.339, and only 1/9 or 2/9 arc points reach 95% of add endpoint.
- Post-boundary gain: replace negative, add positive.
- JVP/VJP: full-radius replacement-style candidates often have negative true gain.
- Basin attraction: even after continuing from replace, NS2D reaches only mean L/Lmax = 0.700 with p0.95 = 0.

### Why is Darcy replace good?

Darcy is discrete, so the same story appears as near-optimal flip-set
multiplicity rather than continuous cap width:

- Replace reaches k90 in 6.5 steps vs add in 78.45.
- Replace final mean Loss3 is 0.04996 vs add 0.04518.
- Add/replace final flip sets differ substantially, Jaccard about 0.2568.
- On average 2.8 methods are within 95% of the best endpoint.

So Darcy has many good discrete choices, and replace finds one quickly.

## 15. Evidence Strength And Remaining Caveats

Strong evidence:

- NS2D replace is outside the high-loss basin reached by add.
- Burgers replace is near a forgiving high-loss endpoint region.
- Darcy has a discrete near-optimal multiplicity that favors replace-like selection.
- The explanation cannot be reduced to pure linearity because Burgers replace first-order prediction is poor.
- Endpoint-local Jacobian/GN modes matter causally for Burgers replace high loss.

Caveats:

- Some causal experiments use small N: Burgers N=5, NS2D N=2.
- Full high-dimensional boundary volume is hard to measure directly; cap/arc/slice tests are strong but still sampling-based.
- Endpoint-local Jacobian modes are not a global basis for the entire landscape.
- The matched tangent-width test does not prove a clean top-mode-only mechanism.
- The current statement should be phrased as strong experimental support, not a mathematical proof.

## 16. Final Paper-safe Wording

Recommended wording:

> Across Burgers, Darcy Flow, and NS2D, the difference between `replace` and `add` is best explained by the geometry of the high-loss region. `replace` succeeds when a full-budget jump lands in a broad or multiply represented high-loss region; it fails when the high-loss region is narrow, curved, or path-dependent. Burgers and Darcy provide forgiving high-loss regions or many near-optimal choices, while NS2D requires path-following correction that `steepest_add` supplies and `steepest_replace` does not.

Avoid wording:

> `replace` is better because Burgers is linear.

Better wording:

> Local linearity and direction stability may contribute, but the direct evidence points to optimizer-relevant high-loss region width and basin accessibility as the controlling factor.

## 17. Main Artifacts

### Documents

- `docs/loss3_high_loss_region_landscape_hypothesis_20260623.md`
- `docs/loss3_boundary_cap_lmax_findings_20260623.md`
- `docs/loss3_full_mechanism_validation_summary_20260623.md`
- `docs/loss3_replace_add_mechanism_final_conclusion_20260623.md`
- `docs/loss3_replace_add_mechanism_backup_record_20260623.md`
- `docs/loss3_ns2d_jvp_vjp_integration_summary_20260623.md`
- `docs/loss3_jvp_vjp_integration_experiment_plan_20260623.md`

### Code

- `tools/probe_loss3_boundary_volume_20260623.py`
- `tools/analyze_loss3_boundary_cap_lmax_20260623.py`
- `tools/plot_loss3_boundary_cap_lmax_svg_20260623.py`
- `tools/probe_loss3_dominant_subspace_iso_projection_20260623.py`
- `tools/run_loss3_dominant_subspace_iso_projection_20260623.sh`
- `tools/probe_loss3_jacobian_mode_causal_ablation_20260623.py`
- `tools/run_loss3_jacobian_mode_causal_ablation_20260623.sh`
- `tools/probe_loss3_jacobian_tangent_width_20260623.py`
- `tools/run_loss3_jacobian_tangent_width_20260623.sh`

### Data directories

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/dominant_subspace_iso_projection_20260623`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_tangent_width_20260623`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/darcy_flipset_mechanism`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N3`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N2_topup`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/cross_system_summary`

### Backup archives

R2 remote prefix:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Verified archive files under:

`analysis_outputs/backup_archives/20260623T014210Z/`

- `attack_objective_true_loss3_comparison_20260622_20260623T014210Z.tar` - 8,173,240,320 bytes
- `mechanism_20260622_partial_20260623T014210Z.tar` - 2,721,730,560 bytes
- `optimizer_ablation_20260622_20260623T014210Z.tar` - 1,669,314,560 bytes
- `code_docs_tools_commit45a7821_20260623T014210Z.tar` - 15,319,040 bytes
- `backup_manifests_20260623T014210Z.tar` - 256,000 bytes
- `SHA256SUMS.txt` - 835 bytes

Final mechanism archive:

- `analysis_outputs/backup_archives/loss3_replace_add_mechanism_final_20260623.tar.gz`
- `analysis_outputs/backup_archives/loss3_replace_add_mechanism_final_20260623.tar.gz.sha256`

SHA256 of final mechanism archive:

`644795e38d819a39fac2902ff2a57b383f1dddc0df1e9aa91a40f5625f17da91`

## 18. Current Bottom Line

The current explanation is reliable and coherent:

\[
\boxed{
\texttt{replace}\text{ succeeds when it does not need to be precise.}
}
\]

Burgers gives replace a broad optimizer-relevant high-loss basin. Darcy gives
replace many near-optimal discrete flip sets. NS2D gives replace a narrow,
path-dependent target; it reaches the boundary but misses the right region.

Therefore:

\[
\boxed{
\text{high-loss region width/connectivity explains the observed behavior better than a simple linear/nonlinear story.}
}
\]
