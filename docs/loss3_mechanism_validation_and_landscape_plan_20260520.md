# Loss 3 Mechanism Validation and Landscape Visualization Plan

Date: 2026-05-20

This note turns the current Loss 3 mechanism explanations into concrete validation experiments. It separates diagnostics already supported by saved artifacts from new GPU loss-evaluation probes needed for landscape/curvature visualization.

## Goal

The core hypothesis is:

> Objective-gradient replacement/GPI is fast because it immediately uses the epsilon budget and then rapidly changes direction on the boundary. In p2q2-like geometries, the loss landscape seems to contain a useful dominant direction/ridge, so replacement reaches a final-like direction in a few steps. In p=1 or q=inf geometries, the same aggressive motion can become spike-like or unhelpful.

We should validate this with explicit numeric and visual probes rather than only explanation.

## A. Diagnostics Already Validated From Existing Data

### A1. p2q2 tangent KKT residual

Existing output:
- `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/`

Metric:

```text
||(I - u u^T) grad L|| / ||grad L|| = sqrt(1 - cos(delta, grad)^2)
```

Observed:
- Replacement/GPI tangent residual at boundary hit is about `0.855`.
- Raw add and steepest add are about `0.397` and `0.435` at hit.
- Replacement/GPI post-boundary gain is much larger, and it keeps large angular motion.

Interpretation:
- Boundary arrival is not convergence.
- Replacement reaches the boundary while tangent gradient remains large, so it still has room to improve by boundary-direction rotation.

### A2. p2q2 radial-growth formula

Existing output:
- `radial_growth_by_setting_method.csv`
- `radial_growth_rollup_by_method.csv`

Formula:

```text
r_next^2 = r^2 + 2 alpha r ||d|| cos(theta) + alpha^2 ||d||^2
```

Observed:
- Exact formula matches recorded radius increments with mean absolute error around `1e-7`.
- LP-steepest has direction norm `1.0` with almost zero CV.
- Raw PGD direction norm CV is about `0.621`, explaining curved/synchronized radial differences.

Interpretation:
- LP-steepest's straighter norm curve is mainly normalized step size, not necessarily more stable angle.

## B. Post-Processing Extensions Without New Model Runs

These can be done from existing `per_sample_step_metrics.csv` and `final_deltas.npz` where the needed columns exist.

### B1. Extend tangent/KKT-style diagnostics to all available p=2 cases

Scope:
- p2q2 300-step
- p2q1 100-step
- p2qinf 100-step

Metric:

```text
tangent_residual_p2 = sqrt(1 - cos(delta, grad)^2)
```

Questions:
- Does p2q1 show the same useful high tangent residual and post-boundary gain pattern as p2q2?
- Does p2qinf show large tangent residual but weaker usefulness/spikier behavior?

Expected output:
- table by `pq, method, epsilon, alpha`
- rollup by `pq, method`
- scatter: tangent residual at hit vs post-boundary gain

### B2. General first-order gap for p=1 and p=2

For general p-ball linear maximization, the local linear optimum value is

```text
epsilon * ||grad||_{p*}
```

A normalized gap is

```text
gap_p = 1 - <grad, delta> / (epsilon * ||grad||_{p*})
```

For p=2, this is closely related to the tangent residual. For p=1, the dual norm is L_inf, and `grad_linf` plus `cos_delta_grad`, `delta_l2`, `grad_l2` can estimate the dot product from existing metrics.

Questions:
- Do settings with large first-order gap at boundary have more post-boundary gain?
- Do p=1 settings have large gap but still poor gain because the steepest feasible direction is too sparse/spiky?

Caveat:
- p=inf needs grad L1 norm for the exact dual gap; that is not currently saved, so p=inf requires a new GPU diagnostic or a rerun that records `grad_l1`.

## C. New GPU Landscape Visualization Experiments

These require evaluating the neural operator/solver loss at new points. They must follow the repo GPU rule: verify `nvidia-smi`, PyTorch CUDA, device name, compute capability, and JAX backend if used before official runs.

### C1. Ray profiles along method directions

For each selected sample, P/Q, epsilon/alpha, and method direction `u`, plot:

```text
L_q(x0 + t u),   t in [0, epsilon]
```

Directions to compare:
- raw_add final delta direction
- steepest_add final delta direction
- replacement/GPI final delta direction
- replacement/GPI early delta direction, e.g. step 1, 5, 10 if trajectory exists
- random control direction

Questions:
- Is loss mostly monotone toward the boundary along final directions?
- Does GPI's step-5/step-10 direction already have nearly the same ray profile as final?
- Do p1/qinf directions show sharp localized peaks or unstable profiles?

Expected figures:
- mean ± std ray profile by method
- per-sample ray profiles for representative samples
- endpoint loss and area-under-profile tables

### C2. 2D loss slices / contours

Choose a center point `delta_c` and two directions `u1, u2`. Evaluate a grid:

```text
L_q(x0 + delta_c + a u1 + b u2)
```

Useful centers:
- clean point: `delta_c=0`
- GPI boundary hit point
- GPI final point
- steepest_add final point

Useful planes:
- `u1 = GPI final direction`, `u2 = steepest_add final direction orthogonalized`
- `u1 = gradient direction`, `u2 = tangent direction`
- `u1 = GPI early-to-final difference`, `u2 = additive final difference`
- PCA plane from all final deltas or saved trajectory deltas

Questions:
- Is there a broad ridge in p2q2 that explains why methods converge to similar deltas?
- Is p1/qinf more jagged, narrow, or spike-like?
- Does replacement jump onto a ridge quickly while additive crawls toward it?

Expected figures:
- heatmap/contour of loss in 2D plane
- fitted affine/quadratic surface residual
- curvature ratio per P/Q/method/center

Existing nearby tool:
- `tools/run_loss3_2d_slice_planarity.py` already does a 2D local slice and quadratic fit for saved PGD trajectories. It should be adapted to read core4 `final_deltas.npz` and `trajectory_samples.npz` from the current alpha/epsilon/PQ sweeps.

### C3. Boundary arc interpolation between method final deltas

For p=2, compare two final unit directions `u_a` and `u_b` on the boundary:

```text
u(s) = normalize((1-s) u_a + s u_b)
delta(s) = epsilon * u(s)
L_q(x0 + delta(s)),  s in [0, 1]
```

Pairs:
- GPI final vs steepest_add final
- GPI final vs raw_add final
- raw_add final vs steepest_add final

Questions:
- Are the methods on the same broad ridge?
- Is there a valley between additive and replacement final deltas, or are they connected by high loss?

Expected figures:
- arc loss curves
- pairwise ridge connectivity score

For p=1/p=inf, use projected interpolation:

```text
delta(s) = Proj_{p,epsilon}((1-s) delta_a + s delta_b)
```

but interpret carefully because the boundary geometry is nonsmooth.

### C4. Local curvature / Hessian-vector probes

Avoid full Hessian. For selected center and direction `v`, compute finite-difference curvature:

```text
v^T H v ≈ [L(delta + h v) - 2 L(delta) + L(delta - h v)] / h^2
```

Directions:
- radial direction
- tangent gradient direction
- GPI final direction
- additive final direction
- random controls

Questions:
- Is p2q2 locally low-curvature/broad along the dominant direction?
- Are p1/qinf settings sharper or more nonsmooth?
- Does GPI move through high-curvature/rough regions early and settle onto smoother regions?

Existing nearby tool:
- `tools/run_loss3_path_directional_curvature.py` estimates curvature along saved PGD path tangents. It can be adapted to core4/GPI trajectories and PQ settings.

### C5. Local dominant-mode / spectral-gap probe

For small selected sample counts, estimate local residual Jacobian dominant directions around:
- clean point
- GPI boundary hit point
- GPI final point
- additive final point

Diagnostics:
- leading singular value estimate
- spectral gap, e.g. `sigma1 / sigma2`
- cosine between leading direction and final delta
- cosine between leading directions at different path points

Questions:
- Is p2q2 actually dominated by one strong mode?
- Does this dominant direction rotate along the boundary?
- Do p1/qinf settings have weaker spectral dominance or localized active coordinates?

Existing related tools/docs:
- `tools/analyze_local_jacobian_fno_deeponet.py`
- `tools/run_loss3_jacobian_subspace_rotation_path.py`
- `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`

This is the strongest but most expensive validation.

## Recommended Run Order

1. Post-process all p=2 cases with tangent residual and first-order gap diagnostics. Cheap and should be done first.
2. Run a small GPU ray-profile pilot for selected settings: p2q2, p2q1, p2qinf, p1qinf; one baseline alpha/epsilon and 4-8 representative samples.
3. Run 2D loss-slice heatmaps for the same samples/settings.
4. Run boundary arc interpolation for p2q2 and p2q1.
5. Only if needed for a paper-level mechanism claim, run local dominant-mode/Jacobian spectral probes.

## What Would Count As Evidence?

Evidence for the dominant-direction story:
- GPI early directions have ray profiles close to GPI final directions.
- 2D slices show a broad ridge leading toward similar final deltas.
- Boundary arc interpolation between GPI and additive final deltas stays high-loss.
- Local spectral gap is large and leading direction aligns with final deltas.

Evidence against the dominant-direction story:
- GPI early directions have poor ray profiles but happen to improve later by nonlinear jumps.
- Additive and GPI final deltas are separated by low-loss valleys in the 2D/arc plots.
- Local spectrum has no dominant gap, or leading directions do not align with final deltas.

Evidence for p/q geometry spike explanation:
- p=1/q=inf slices show narrow, sharp peaks.
- concentration metrics increase with slice curvature and high-frequency energy.
- projected interpolation in p=1/q=inf has abrupt loss changes compared with p2q2.

## Immediate Implementation Notes

The p2q2 tangent/radial probe is already implemented:

```bash
python3 tools/probe_loss3_p2q2_tangent_and_radial_geometry.py
```

A useful next script should be named something like:

```text
tools/run_loss3_pq_landscape_probe.py
```

It should read existing core4 outputs, select method deltas/trajectory deltas, evaluate `loss3_q` on rays/slices/arcs, and write figures/tables under:

```text
forensics/loss3_pq_landscape_probe_20260520/
```

Before running this script officially, verify the GPU path according to `AGENTS.md`.
