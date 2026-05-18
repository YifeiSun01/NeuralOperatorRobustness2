# Loss3 Optimizer Direction-Proposal Ablation Plan - 2026-05-17

Status: plan only. No numerical experiment has been run for this plan yet.

## Purpose

This experiment is designed to answer the user's optimizer question:

> When `generalized_power` reaches the `loss3_original` boundary optimum much
> faster than PGD or LP-steepest PGD, is the advantage caused by the direction
> rule, or by the replacement/boundary proposal rule?

The existing code already shows the key structural difference in
`tools/run_batch_three_loss_loss_only.py`:

```python
if method == "pgd":
    direction = grad
    delta = project_delta(delta + alpha * direction, epsilon, p_order)
elif method == "lp_steepest_pgd":
    direction = steepest_direction(grad, p_order)
    delta = project_delta(delta + alpha * direction, epsilon, p_order)
elif method == "generalized_power":
    direction = steepest_direction(grad, p_order)
    delta = project_delta(epsilon * direction, epsilon, p_order)
```

So the current three methods confound two choices:

1. direction rule: raw gradient, Lp-steepest gradient direction, or generalized-power direction;
2. proposal rule: additive PGD-style proposal or replacement/boundary proposal.

This plan separates those choices.

## Core Hypotheses

### H1: Boundary replacement explains the speed

If replacement variants reach high loss immediately regardless of whether the
direction is raw-gradient, Lp-steepest, or generalized-power, then the apparent
speed advantage mostly comes from the update

```text
delta_{k+1} = epsilon * d_k
```

rather than from generalized power as a special direction rule.

### H2: Direction choice still matters on the boundary

If all replacement variants reach the boundary quickly but their endpoint losses
are different, then the proposal rule explains boundary reach, while the
direction rule explains boundary quality.

### H3: Additive methods may catch up with tuned alpha

If PGD or LP-steepest PGD catches up when `alpha` is increased, or when we
evaluate their per-step directions after boundary normalization, then their
weakness at early steps is mostly path/radius speed rather than final direction
quality.

### H4: Fast replacement directions may be less physical

If replacement methods create perturbations with more high-frequency energy,
larger total variation, sharper local oscillations, or more out-of-distribution
input spectra, then speed may come with less physically plausible perturbations.

## Main Experimental Scope

Use the same core setting as the completed six-experiment story:

- Model/task: FNO / 1D Burgers.
- Viscosity: `nu=0.001`.
- Objective: `loss3_original` only.
- Input norm: start with L2.
- Output norm: L2.
- Epsilon: `8` as the main setting.
- Batch: first smoke `batch=5` or `20`; official run `batch=100`.
- Steps: `100` for additive methods; replacement methods still record all 100
  steps so trajectory stability can be compared.
- Initial delta: zero for the main run; random starts as robustness checks.
- GPU-only: use the repository GPU verification policy before official runs.

Secondary settings should be used only after the main result is clear:

- Epsilon sweep: `epsilon in {1.6, 4, 8, 16}`.
- Alpha sweep for additive methods: `alpha in {0.075, 0.15, 0.3, 0.6, 1.5, 3.0}`.
- P/Q geometry sweep: start with `p=2, q=2`, then run the matrix below.

## P/Q Geometry Extension

The optimizer comparison must not be restricted to the default `p=q=2`
geometry.  In this repository's notation, the perturbation budget is a
`p`-norm constraint on `delta`, while the optimized `loss3_original` value is a
`q`-norm on the output residual:

```text
||delta||_p <= epsilon,
loss3_original(delta) = ||f(x + delta) - g(x + delta)||_q.
```

The existing code already follows this convention:

- `tools/run_batch_three_loss_loss_only.py` uses `--p` for
  `project_delta(..., epsilon, p_order)` and uses `--q` inside
  `batch_norm(f_adv - g_adv, q_order)`.
- `run_three_loss_objective_attack.py` uses the older names `--input_p` and
  `--output_q`.  Its generalized-power variants use the q-side dual map
  `phi_q(y)=sign(y)|y|^{q-1}` before mapping the resulting input-side vector
  back to the p-ball.
- `three_loss_objective_experiment_plan.md` already proposed a p/q sweep with
  `(2,2)`, `(inf,2)`, `(2,inf)`, and `(inf,inf)`, plus optional sparse-input
  extensions.

Therefore this experiment has two independent geometry factors:

1. `p`: feasible set and projection for the perturbation `delta`;
2. `q`: residual norm used by the scalar loss and by q-aware generalized-power
   directions.

### Main p/q pairs

Run the optimizer ablation first at `p=2, q=2`, then repeat the key comparisons
under:

```text
(p=2,   q=2)    baseline Euclidean input / Euclidean residual
(p=inf, q=2)    pointwise-amplitude-bounded input / Euclidean residual
(p=2,   q=inf)  Euclidean input / worst-point residual
(p=inf, q=inf)  pointwise-amplitude-bounded input / worst-point residual
```

Optional extensions, after the four-pair matrix is stable:

```text
(p=1, q=2)      sparse perturbation budget / Euclidean residual
(p=1, q=inf)    sparse perturbation budget / worst-point residual
(p=2, q=1)      Euclidean input / distributed L1 residual
(p=inf, q=1)    pointwise input / distributed L1 residual
```

The `q=1` and `q=inf` cases are nonsmooth.  They should be treated as stress
tests of the geometry rather than as the first paper-facing claim unless the
optimizer traces are stable and the subgradient choices are explicitly recorded.

### What the p/q sweep should answer

The p/q extension adds these questions to the direction/proposal comparison:

- Does generalized-power replacement still look fast when the perturbation
  budget is `Linf` rather than `L2`?
- Does the advantage change when the residual objective is `Linf`, where the
  attack tries to maximize the worst output location rather than total residual
  energy?
- Are equivalences from the L2 plan still valid?  For example,
  `unit_raw_add == steepest_add` is guaranteed for `p=2`, but not for general
  `p`.
- Does a q-aware generalized-power direction, such as `pure_jvp_vjp` or
  `generalized_pq`, outperform the exact objective-gradient replacement when
  `q != 2`?
- Do non-L2 p/q choices produce rougher or less physical final deltas than the
  L2/L2 baseline?

### Required p/q diagnostics

Every row must record:

- `p_order`, `q_order`, and the exact CLI names used, either `--p/--q` or
  `--input_p/--output_q`.
- `delta_pnorm`, `delta_l2`, and `delta_linf`, so different p-budgets can be
  compared in common units.
- `loss3_q`, plus auxiliary `loss3_l2` and `loss3_linf` evaluations of the same
  final delta.  This prevents a `q=inf` run from looking better merely because
  it is scored under a different norm.
- gradient implementation source: for PGD, LP-steepest PGD, and
  `objective_gradient` replacement this should be `autograd(loss3_q)`, not a
  hand-coded q-norm residual-gradient formula.
- steepest-direction implementation source: the p-ball map used to convert
  `g_k` into `s_k`.
- q-aware direction information for generalized-power variants:
  `power_variant`, `output_dual_norm`, `jvp_qnorm`, `vjp_norm`, whether the
  q-side map was explicit or autograd-derived, and same-step cosine between the
  q-aware power direction and the exact objective-gradient steepest direction.
- physical diagnostics grouped by p/q pair, because `p=inf` and `q=inf` may
  naturally create more localized or sharper perturbations.

### Interpretation rule for p/q results

Do not rank methods across different q values using only their native optimized
loss, because `||r||_2` and `||r||_inf` are different scalar quantities.  Within
each p/q pair, compare optimizer speed, final native loss, boundary behavior,
and smoothness.  Across p/q pairs, compare shared auxiliary metrics and the
shape of the final perturbations.

## Autograd Gradient Versus Explicit Steepest Map

The implementation should separate two operations that are easy to confuse:

```text
objective gradient:  g_k = grad_delta L_q(delta_k)
steepest direction:  s_k = argmax_{||s||_p <= 1} g_k^T s
```

For `loss3_original`, compute the scalar objective gradient `g_k` with autograd:

```text
L_q(delta) = ||f(x + delta) - g_solver(x + delta)||_q.
```

Do not hand-code the full explicit q-norm residual-gradient formula for the
ordinary PGD, LP-steepest PGD, or `objective_gradient` replacement rows.  The
q-side derivative, solver/model Jacobians, and chain rule should come from
autograd.  This keeps the experiment focused on optimizer geometry instead of
adding another source of implementation error.

After `g_k` is available, the only explicit formula needed for LP-steepest PGD
is the p-ball linear maximizer:

```text
s_k = argmax_{||s||_p <= 1} g_k^T s.
```

Use the existing helper pattern from `steepest_direction` /
`maximize_linear_over_p_ball`:

```text
p = 2:        s_k = g_k / ||g_k||_2
p = inf:      s_k = sign(g_k)
p = 1:        s_k is supported on the largest |g_k| coordinate
1 < p < inf:  s_k = normalize_p(sign(g_k) * |g_k|^(p_dual - 1))
              where p_dual = p / (p - 1)
```

Thus the complicated explicit p/q power formulas should be reserved for the
q-aware generalized-power variants, such as `pure_jvp_vjp`, `affine_jvp_vjp`,
or `generalized_pq`.  Even there, the result should record whether the direction
was produced by an explicit q-dual helper or by differentiating a q-power
surrogate with autograd.

This gives a clean implementation split:

| Method family | Gradient source | Direction conversion |
|---|---|---|
| raw PGD | autograd of exact `loss3_q` | none, use `g_k` |
| unit raw / raw replacement | autograd of exact `loss3_q` | normalize `g_k` to p-ball |
| LP-steepest PGD / replacement | autograd of exact `loss3_q` | explicit p-steepest map `g_k -> s_k` |
| objective-gradient power replacement | autograd of exact `loss3_q` | explicit p-steepest or p-normalized gradient replacement |
| q-aware generalized power | JVP/VJP or q-power surrogate | q-aware map plus p-steepest normalization |

## Six Direction-Proposal Methods

Every method can be written as

```text
delta_{k+1} = Proj_{||delta||_p <= epsilon}(z_{k+1}).
```

The proposal is either additive

```text
z_add = delta_k + alpha * d_k
```

or replacement/boundary

```text
z_rep = tau * d_k, tau >= epsilon.
```

If `||d_k||_p = 1`, replacement is just

```text
delta_{k+1} = epsilon * d_k.
```

The factorial ablation is:

| Direction rule | Additive proposal | Replacement proposal |
|---|---|---|
| Raw gradient | `raw_add` | `raw_replace` |
| Lp-steepest direction | `steepest_add` | `steepest_replace` |
| Generalized-power direction | `power_add` | `power_replace` |

Definitions:

- `raw_add`: standard PGD, using `d_k = g_k` and
  `delta_{k+1}=Proj(delta_k + alpha g_k)`.
- `raw_replace`: normalize the raw gradient to a unit p-direction and replace:
  `delta_{k+1}=epsilon * normalize_p(g_k)`.
- `steepest_add`: LP-steepest PGD, using
  `s_k = argmax_{||s||_p <= 1} g_k^T s` and
  `delta_{k+1}=Proj(delta_k + alpha s_k)`.
- `steepest_replace`: replacement version of LP-steepest PGD,
  `delta_{k+1}=epsilon * s_k`.
- `power_add`: additive version of the generalized-power direction,
  `delta_{k+1}=Proj(delta_k + alpha u_k)`.
- `power_replace`: generalized power replacement,
  `delta_{k+1}=epsilon * u_k`.

Important L2 note:

- For `p=2`, `normalize_p(g_k)` and the Lp-steepest direction are the same
  direction. Therefore `raw_replace` and `steepest_replace` should match under
  L2, up to numerical details. This is a useful sanity check.
- If the current `generalized_power` implementation uses
  `steepest_direction(grad, p)` for `u_k`, then `power_replace` may also match
  `steepest_replace`. To test a genuinely different power direction, use
  generalized-power variants from `run_three_loss_objective_attack.py`, such as
  `pure_jvp_vjp`, `affine_jvp_vjp`, or `objective_gradient`, and record which
  one is used.

## Equivalence And Degeneracy Cases To Check Explicitly

The experiment must not treat method names as automatically distinct.  Several
rows can become mathematically identical or nearly identical depending on the
norm and on how `generalized_power` is implemented.

### Case 1: PGD can become LP-steepest PGD after normalization

Raw PGD uses the raw gradient direction:

```text
d_raw = g_k.
```

LP-steepest PGD uses

```text
s_k = argmax_{||s||_p <= 1} g_k^T s.
```

For `p=2`, this gives

```text
s_k = g_k / ||g_k||_2.
```

Therefore raw-gradient PGD and LP-steepest PGD differ in two ways only if raw
PGD keeps the gradient magnitude:

```text
raw_add:      delta_{k+1} = Proj(delta_k + alpha * g_k)
steepest_add: delta_{k+1} = Proj(delta_k + alpha * g_k / ||g_k||_2)
```

If raw PGD is modified to use a normalized gradient, or if `alpha` is rescaled
sample-by-sample by `||g_k||`, then raw PGD collapses to L2-steepest PGD.  The
experiment should therefore include both:

- `raw_add`: true raw-gradient additive update;
- `unit_raw_add`: additive update using `g_k / ||g_k||_p`.

For `p=2`, `unit_raw_add` should match `steepest_add`. If it does not, there is
an implementation bug or a norm-convention mismatch.

### Case 2: LP-steepest PGD and generalized power replacement can be the same

The current batch script implements `generalized_power` as:

```text
u_k = steepest_direction(g_k, p)
delta_{k+1} = epsilon * u_k.
```

Under that implementation,

```text
power_replace == steepest_replace
```

by construction.  In that case, generalized power is not a separate direction
rule; it is LP-steepest direction selection plus a replacement/boundary proposal.

So the first diagnostic should be:

```text
cos(power_replace_delta_k, steepest_replace_delta_k)
```

and the expected value is `1.0` whenever both use the same `u_k=s_k` and the same
initial conditions.

### Case 3: Generalized power is distinct only when `u_k` is distinct

To test whether generalized power has a direction advantage, we must run at
least one variant where

```text
u_k != s_k.
```

Candidate variants already documented in `run_three_loss_objective_attack.py`
include:

- `objective_gradient`: normalized exact objective-gradient replacement;
- `pure_jvp_vjp`: local operator-gain power direction;
- `affine_jvp_vjp`: affine / current-residual generalized-power direction.

The result table must therefore label generalized-power runs by both proposal
rule and power variant, for example:

```text
power_replace__steepest_gradient
power_replace__objective_gradient
power_replace__pure_jvp_vjp
power_replace__affine_jvp_vjp
```

Only the latter variants can support a claim that generalized power is doing
something beyond LP-steepest replacement.

### Case 4: Additive versus replacement is the cleanest causal contrast

The cleanest paired comparisons are:

| Question | Matched comparison |
|---|---|
| Does replacement explain speed for the same raw/unit gradient direction? | `unit_raw_add` vs `raw_replace` for `p=2` |
| Does replacement explain speed for the same LP-steepest direction? | `steepest_add` vs `steepest_replace` |
| Does replacement explain speed for the same power direction? | `power_add__variant` vs `power_replace__variant` |
| Does generalized power direction add value beyond LP-steepest? | `steepest_replace` vs `power_replace__pure_jvp_vjp` / `affine_jvp_vjp` |

This means the experiment should first prove which rows are equivalent, then
only interpret differences among rows that are not mathematically the same.

### Required sanity-check outputs

Add these checks to the analysis:

- `max_abs_delta_difference` between equivalent rows, e.g. `unit_raw_add` and
  `steepest_add` for L2.
- `mean_abs_final_delta_cosine` between equivalent rows.
- same-step direction cosine between `s_k` and `u_k`.
- warning flag if two methods are being compared but their direction/proposal
  definitions imply they should be identical.

Interpretation rule:

- If two methods are equivalent by definition, do not claim one is better.
  Instead say the experiment verifies the equivalence and then attribute any
  observed difference to implementation details, numerical tolerance, or a
  mismatched norm convention.

## Official Three-Method Baseline

Before the six-way ablation, reproduce the familiar three methods on the same
`loss3_original` setting:

| Method | Existing meaning | Key question |
|---|---|---|
| `pgd` | raw-gradient additive | Does slow growth mainly come from walking gradually to the boundary? |
| `lp_steepest_pgd` | Lp-steepest additive | Does normalizing into the p-steepest direction help compared with raw PGD? |
| `generalized_power` | current replacement/boundary method | Is the fast early loss caused by immediate boundary replacement? |

The baseline should save the same per-step and final diagnostics for all three
methods, even if some already exist in older directories, so the comparison is
perfectly paired.

## Metrics To Record

### Optimization performance

For each method, sample, and step:

- `loss3_original(delta_k)`.
- `loss3_original(delta_k) - loss3_original(0)`.
- best-so-far `loss3_original`.
- `delta_pnorm`, `delta_l2`, `delta_linf`.
- boundary ratio `||delta_k||_p / epsilon`.
- step when boundary ratio first exceeds `0.99`.
- step/time to reach `90%`, `95%`, and `99%` of the best final loss among all methods.
- area under the loss-vs-step curve, to measure early optimization speed.
- wall-clock time per step and total time.

### Direction/proposal diagnostics

For each method and step:

- raw gradient norm: `||g_k||_2`, `||g_k||_inf`.
- chosen direction norm: `||d_k||_p`.
- proposal norm before projection: `||z_{k+1}||_p`.
- projection shrink factor, if any.
- cosine between `delta_k` and `g_k`.
- cosine between `delta_k` and `d_k`.
- cosine between consecutive directions `d_k` and `d_{k-1}`.
- pairwise cosine between methods' directions at the same sample and same step.
- pairwise cosine between final deltas from all methods.

These diagnostics answer whether methods differ because they use different
directions, or because they use the same/similar direction with a different
proposal rule.

### Boundary-normalized direction quality

At every step, evaluate not only the actual perturbation `delta_k`, but also the
same direction rescaled to the boundary:

```text
delta_boundary_k = epsilon * delta_k / ||delta_k||_p.
```

Record `loss3_original(delta_boundary_k)` for additive methods.

This is critical: if PGD has a good direction early but small radius, its actual
loss will look weak while its boundary-normalized direction may already be good.
That separates direction quality from radius-growth speed.

### Physical / smoothness diagnostics for final deltas

For each final delta and for selected intermediate deltas:

- Fourier spectrum of `delta`.
- high-frequency energy ratio, for example energy above the top quarter of
  frequencies divided by total energy.
- spectral centroid.
- total variation of `delta`.
- first-derivative and second-derivative L2 norms.
- zero-crossing count.
- max absolute pointwise perturbation.
- spectrum of `x + delta` compared with the clean input spectrum.
- optional solver sanity: whether `x + delta` causes NaN/Inf or abnormal solver
  values.

This answers the user's physical-plausibility question: a method may produce a
large endpoint loss quickly, but the perturbation might be high-frequency,
rough, or visually implausible.

## Required Plots

### Optimization curves

1. Mean/std `loss3_original` vs step for all methods.
2. Mean/std `best_so_far_loss3_original` vs step.
3. Mean/std boundary ratio vs step.
4. Mean/std boundary-normalized `loss3_original` vs step.
5. Time-to-threshold bar plot: steps/time to reach `90%`, `95%`, `99%` of best
   final loss.

### Direction/proposal plots

1. Pairwise final-delta cosine heatmap.
2. Pairwise same-step direction cosine heatmaps at steps `0, 1, 5, 10, 25, 50, 100`.
3. Cosine between `delta_k` and `d_k` vs step.
4. Proposal norm before projection vs step.
5. Projection shrink factor vs step.

### Physical diagnostics

1. Final delta line plots for representative samples, e.g. indices `0, 7, 40, 47, 115`.
2. Fourier energy spectra of final deltas.
3. High-frequency energy ratio bar plot.
4. Total variation / derivative norm bar plots.
5. `x`, `x + delta`, and `delta` overlay plots for representative samples.

### Trajectory and GIF visualizations

The experiment must visualize not only the final `delta`, but also how `delta`
evolves over optimization steps.  This is required for answering whether a
method first creates high-frequency / spiky perturbations and later smooths out,
or whether it stays rough throughout.

Save per-step trajectories for representative samples, not necessarily for the
entire batch-100 run.  The default representative dataset indices should be:

```text
0, 7, 40, 47, 115
```

For each selected sample and method, save:

```text
trajectory_samples.npz:
  k
  dataset_index
  method
  delta[k, x]
  x_adv[k, x]
  loss3_q[k]
  loss3_l2[k]
  loss3_linf[k]
  delta_pnorm[k]
  delta_l2[k]
  delta_linf[k]
  boundary_ratio[k]
  high_frequency_energy_ratio[k]
  spectral_centroid[k]
  total_variation[k]
  first_derivative_l2[k]
  second_derivative_l2[k]
  zero_crossing_count[k]
```

The analysis must create three levels of visual comparison.

Static per-method/per-sample plots:

1. `delta` line plots at selected steps, e.g. `k = 0, 1, 2, 5, 10, 25, 50, 100`.
2. `x`, `x + delta`, and `delta` overlays at the same selected steps.
3. Fourier amplitude spectra of `delta_k` at the same selected steps.
4. loss curve and boundary-ratio curve on the same figure, with the current
   selected step marked.
5. roughness-over-step curves: high-frequency energy ratio, spectral centroid,
   total variation, first-derivative L2, and second-derivative L2.

Cross-method comparison plots:

1. One figure per selected sample with methods as rows and selected steps as
   columns, showing `delta_k` with common y-limits across all methods.
2. One figure per selected sample with methods as rows and selected steps as
   columns, showing Fourier spectra with common y-limits.
3. Time-space heatmap of `delta_k(x)` for each method: x-coordinate on the
   horizontal axis, step on the vertical axis, color = delta value.
4. Method-overlaid roughness curves, so it is visible whether one method starts
   spiky and later smooths, or remains high-frequency.
5. Method-overlaid loss curves next to method-overlaid roughness curves, so loss
   speed can be compared against physical smoothness.

GIF outputs:

1. `delta_evolution_<sample>_<method>.gif`: animated line plot of `delta_k` with
   fixed y-limits, plus current `loss3_q`, `||delta||_p/epsilon`, and roughness
   metrics in the title or side panel.
2. `delta_spectrum_evolution_<sample>_<method>.gif`: animated Fourier spectrum
   of `delta_k` with fixed y-limits.
3. `method_comparison_<sample>.gif`: multi-panel animation with one row per
   method and synchronized step index, showing `delta_k`, current loss, and
   high-frequency energy ratio.
4. Optional `input_output_evolution_<sample>_<method>.gif`: animated `x+delta`,
   model output, solver output, and residual if solver-forward visualization is
   affordable.

All GIFs should use fixed axes across time and across methods for the same
sample.  Otherwise a rough perturbation can look visually harmless simply
because the plot rescales every frame.

This visualization set should directly answer:

- Does replacement jump immediately to a high-amplitude or high-frequency
  perturbation?
- Does additive PGD gradually grow the same shape, or does it find a different
  smoother shape?
- Does a method's loss increase before its delta becomes physically smooth?
- Do p/q choices like `p=inf` or `q=inf` create more localized spikes than
  `p=2, q=2`?
- Are final deltas visually similar even when the optimization paths differ?

## Analysis Questions And Decision Rules

### Question A: Is GPI faster because it jumps to the boundary?

Compare:

- `steepest_add` vs `steepest_replace`;
- `power_add` vs `power_replace`;
- additive methods' actual loss vs boundary-normalized loss.

Interpretation:

- If replacement variants win early while additive variants' boundary-normalized
  directions are already good, the speed advantage is mostly boundary reach.
- If replacement variants still differ strongly from each other at the boundary,
  direction choice also matters.

### Question B: Is the generalized-power direction special?

Compare under the same proposal rule:

- `raw_replace` vs `steepest_replace` vs `power_replace`;
- `raw_add` vs `steepest_add` vs `power_add`.

Interpretation:

- If `power_replace` beats `steepest_replace` under the same replacement rule,
  the generalized-power direction itself adds value.
- If `power_replace` and `steepest_replace` are almost identical, then current
  generalized power is essentially using the same gradient-induced steepest
  direction; the observed advantage is mainly replacement.

### Question C: Can additive PGD catch up with alpha tuning?

Compare additive methods across `alpha`:

```text
alpha in {0.075, 0.15, 0.3, 0.6, 1.5, 3.0}
```

Interpretation:

- If larger `alpha` makes additive methods reach the same final loss quickly,
  the original PGD was mostly step-size limited.
- If too-large `alpha` causes oscillation, worse final loss, or rougher deltas,
  then gradual PGD may be slower but more stable.

### Question D: Are fast replacement deltas less physical?

Compare final delta smoothness and spectra across methods.

Interpretation:

- If replacement methods have much higher high-frequency energy or total
  variation, report a speed-versus-plausibility tradeoff.
- If replacement methods are just as smooth as PGD, then the fast boundary jump
  is not obviously producing artificial perturbations.

## Minimal Run Matrix

### Smoke run

Purpose: check code, GPU, output schema, and plots.

- batch: `5` samples, indices `0, 7, 40, 47, 115`.
- epsilon: `8`.
- alpha: `0.3`.
- steps: `20` or `30`.
- methods: all six direction-proposal combinations, plus `unit_raw_add` as an
  L2 sanity check against `steepest_add`.

### Official main run

Purpose: paired method comparison.

- batch: `100`, indices `0..99`.
- epsilon: `8`.
- alpha: `0.3` for additive methods.
- steps: `100`.
- methods: all six direction-proposal combinations plus `unit_raw_add`, and the
  existing official names `pgd`, `lp_steepest_pgd`, and `generalized_power` for
  compatibility.

### Alpha sensitivity run

Purpose: determine whether additive methods are merely step-size limited.

- batch: `100` if main run is cheap enough; otherwise batch `20`.
- epsilon: `8`.
- alpha: `{0.075, 0.15, 0.3, 0.6, 1.5, 3.0}`.
- additive methods only: `raw_add`, `steepest_add`, `power_add`.

### Epsilon sensitivity run

Purpose: determine whether conclusions hold at smaller/larger budgets.

- batch: `20` first, then `100` if needed.
- epsilon: `{1.6, 4, 8, 16}`.
- alpha: use `0.3`, plus a scaled alpha option if additive methods become unfair
  at small epsilon.
- methods: official three methods plus key ablations `steepest_replace` and
  `power_add`.

### P/Q geometry run

Purpose: test whether the direction/proposal conclusions are specific to
`p=2, q=2` or persist under different input-budget and output-loss geometries.

- batch: start with `20`; promote to `100` for pairs whose smoke traces are
  stable and scientifically interesting.
- epsilon: use `8` for the first comparison, but record common-unit norms
  (`delta_l2`, `delta_linf`) because the feasible set changes with `p`.
- alpha: `0.3` baseline for additive methods; add per-pair alpha sensitivity if
  boundary reach becomes unfair under `p=inf` or `p=1`.
- p/q pairs: `(2,2)`, `(inf,2)`, `(2,inf)`, `(inf,inf)`.
- optional extensions: `(1,2)`, `(1,inf)`, `(2,1)`, `(inf,1)`.
- methods: `raw_add`, `unit_raw_add`, `steepest_add`, `steepest_replace`,
  `power_replace__objective_gradient`, and at least one q-aware power variant
  such as `power_replace__pure_jvp_vjp` or `power_replace__generalized_pq`.
- analysis: compare methods within each p/q pair first; compare across p/q
  pairs only using shared auxiliary metrics and physical diagnostics.

## Implementation Plan

### New runner

Create:

```text
tools/run_loss3_direction_proposal_ablation.py
```

It should reuse as much as possible from `tools/run_batch_three_loss_loss_only.py` and the existing visualization patterns in `loss_attack_common.py` and `plot_burgers_corrected_oldstyle_5loss_gif.py`:

- model/solver loading;
- `loss3_original` objective computation;
- projection and norm helpers;
- autograd-based exact `loss3_q` gradient computation;
- p-steepest direction helper for `g_k -> s_k`;
- final delta diagnostics;
- all nine objective evaluations for compatibility.

New CLI ideas:

```text
--direction-rule raw_gradient | unit_raw_gradient | lp_steepest | power_objective_gradient | power_pure_jvp_vjp | power_affine_jvp_vjp
--proposal-rule additive | replacement
--alpha 0.3
--epsilon 8
--p 2                 # or --input_p 2 in scripts that use the older naming
--q 2                 # or --output_q 2 in scripts that use the older naming
--steps 100
--batch-size 100
--start-index 0
--init zero | random
--save-boundary-normalized-eval
--save-physical-diagnostics
--save-delta-trajectory
--trajectory-indices 0 7 40 47 115
--trajectory-save-every 1
--make-gifs
```

Output directory convention:

```text
forensics/loss3_optimizer_direction_proposal_ablation_20260517/
  fno_nu0p001_eps8_alpha0p3_batch100_steps100/
```

Required output files:

- `per_step_metrics.csv`
- `per_sample_step_metrics.csv`
- `final_delta_diagnostics.csv`
- `final_delta_summary.json`
- `final_deltas.npz`
- `direction_cosines.csv`
- `method_pairwise_final_delta_cosines.csv`
- `boundary_normalized_step_eval.csv`
- `physical_smoothness_metrics.csv`
- `delta_smoothness_by_step.csv`
- `trajectory_samples.npz`
- `delta_spectrum_by_step.npz`
- `cross_norm_final_metrics.csv`
- `pq_geometry_summary.csv`
- `manifest.json`
- figures under `figures/`, including loss curves, delta trajectory grids,
  spectrum grids, heatmaps, and roughness-over-step plots
- GIFs under `figures/gifs/`

The manifest must include GPU runtime evidence following the repository policy:
PyTorch version, CUDA version, GPU name, compute capability, PyTorch arch list,
JAX backend/devices, and no CPU fallback.  It must also include the p/q
configuration, norm helper source, exact power variant, gradient implementation
source, steepest-direction implementation source, and whether the runner used
`--p/--q` or `--input_p/--output_q` naming.

### Analysis script

Create:

```text
tools/analyze_loss3_direction_proposal_ablation.py
```

It should produce:

- method ranking table by final loss;
- method ranking table by time/steps to threshold;
- boundary reach table;
- boundary-normalized direction quality table;
- smoothness / physical diagnostic table;
- delta trajectory visual grids for representative samples;
- per-step spectrum and roughness plots;
- synchronized GIFs for delta and spectrum evolution;
- paired statistical tests across samples for key comparisons;
- compact Markdown summary.

### Result document

After the run, write:

```text
docs/loss3_optimizer_direction_proposal_ablation_result_YYYYMMDD.md
```

The result should answer these questions explicitly:

1. Is the speed advantage mainly boundary replacement?
2. Does the generalized-power direction add value beyond Lp-steepest gradient direction?
3. Can additive PGD catch up with alpha tuning or boundary-normalized evaluation?
4. Do fast replacement methods produce less smooth or less physical perturbations?
5. Do the conclusions persist under non-L2 p/q geometry?
6. When `q != 2`, do q-aware power variants behave differently from exact
   objective-gradient replacement?
7. What do the final and per-step perturbations look like visually, and do the
   methods differ in smoothness, frequency content, or step-by-step evolution?

## Expected Interpretations

Possible outcomes and how to read them:

| Observed pattern | Interpretation |
|---|---|
| All replacement methods jump to high loss immediately; additive boundary-normalized directions are already good | Main cause is proposal/radius, not special generalized power direction. |
| `power_replace` beats `steepest_replace` under same replacement rule | Generalized-power direction itself is better. |
| `power_add` is slow like additive PGD while `power_replace` is fast | Replacement proposal is the speed source. |
| Large-alpha additive PGD catches up without worse smoothness | Earlier PGD was step-size limited. |
| Large-alpha additive PGD oscillates or gets rougher | Gradual PGD has stability/smoothness advantages. |
| Replacement methods have much higher high-frequency energy | Fast methods may be less physically plausible. |
| Replacement and additive methods converge to similar final deltas and spectra | GPI is mainly a faster route to the same endpoint. |
| Replacement advantage persists across `(2,2)`, `(inf,2)`, `(2,inf)`, and `(inf,inf)` | The proposal-rule explanation is robust to p/q geometry. |
| Replacement advantage appears only at `(2,2)` | The conclusion is geometry-specific and should not be stated as universal. |
| q-aware power variants win mainly when `q != 2` | The q-side dual map is contributing a real direction effect beyond boundary replacement. |
| `q=inf` runs win native `loss3_q` but not auxiliary `loss3_l2` or smoothness | The method is exploiting worst-point residual geometry rather than improving the whole residual field. |
| Early high-frequency energy drops while loss continues rising | The optimizer first finds a sharp direction and then smooths/refines it. |
| Early high-frequency energy stays high through the final step | The method's apparent strength may depend on rough or less physical perturbations. |
| Additive and replacement methods end with similar final deltas but different GIF trajectories | The methods find similar endpoints through different paths; speed is path/proposal-driven. |

## Final Scientific Claim This Experiment Can Support

If the ablation behaves as expected, the paper-ready claim should be phrased like
this:

> The apparent speed advantage of generalized power iteration is not only an
> optimizer-name effect. It can be decomposed into a direction rule and a
> proposal rule. In the tested `loss3_original` attack, the replacement/boundary
> proposal can explain much of the rapid early loss increase, while the direction
> rule determines the quality and smoothness of the boundary point found.

The experiment should not claim that generalized power is universally better
unless it wins under matched proposal rules and matched physical/smoothness
constraints.
## Implementation Status - 2026-05-17

Status: code prepared, experiment not run.

Implemented runner / analyzer / visualizer:

```text
tools/run_loss3_direction_proposal_ablation.py
```

The script is designed as a single entry point for the planned experiment.  It
prepares the following pieces without requiring a separate plotting script:

- official aliases: `pgd`, `lp_steepest_pgd`, and `generalized_power`;
- direction/proposal ablations: raw additive, unit-raw additive/replacement,
  LP-steepest additive/replacement, objective-gradient power additive/replacement,
  and q-aware power variants;
- p/q geometry through `--p` and `--q`;
- non-contiguous sample selection through `--dataset-indices`;
- GPU evidence capture before any real run;
- per-step and per-sample metrics;
- final delta diagnostics and pairwise final-delta cosines;
- representative-sample delta trajectories;
- smoothness and spectrum diagnostics by step;
- static figures and optional GIF rendering;
- `analysis_checklist.md` for post-run interpretation.

Validation performed so far:

```text
python3 -m py_compile tools/run_loss3_direction_proposal_ablation.py
```

Observed result: syntax check passed.  No optimizer run, model load, solver call,
or GPU experiment was started during this implementation step.

Example smoke command to run later, after explicit approval:

```bash
./adv_robust/bin/python tools/run_loss3_direction_proposal_ablation.py \
  --out-root forensics/loss3_optimizer_direction_proposal_ablation_20260517/smoke_fno_nu0p001_eps8_p2_q2 \
  --dataset-indices 0 7 40 47 115 \
  --methods official3 main \
  --epsilon 8 \
  --alpha 0.3 \
  --steps 20 \
  --p 2 \
  --q 2 \
  --device cuda \
  --save-delta-trajectory \
  --make-gifs
```

Example official command to run later, after smoke is checked:

```bash
./adv_robust/bin/python tools/run_loss3_direction_proposal_ablation.py \
  --out-root forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2 \
  --batch-size 100 \
  --start-index 0 \
  --methods main \
  --epsilon 8 \
  --alpha 0.3 \
  --steps 100 \
  --p 2 \
  --q 2 \
  --device cuda \
  --trajectory-indices 0 7 40 47 \
  --save-delta-trajectory
```

P/Q geometry commands should repeat the same smoke/official pattern for:

```text
(p=2, q=2), (p=inf, q=2), (p=2, q=inf), (p=inf, q=inf)
```

When interpreting future results, compare methods within one p/q pair first.
Only compare across p/q pairs using shared auxiliary metrics such as `loss3_l2`,
`loss3_linf`, common-unit delta norms, and smoothness diagnostics.
## Smoke Runtime Result - 2026-05-17

Status: tiny GPU smoke completed; no scientific optimizer conclusion.

Observed smoke output root:

```text
forensics/loss3_optimizer_direction_proposal_ablation_20260517/smoke_tiny_runtime_p2_q2_idx0_7_steps3
```

Smoke setting: dataset indices `0` and `7`, methods `official3 main`,
`steps=3`, `epsilon=8`, `alpha=0.3`, `p=2`, `q=2`, trajectory saving enabled,
GIF rendering enabled.  The run completed and generated root CSV/NPZ outputs,
method-level diagnostics, 66 figure files, and 28 GIFs.

Observed method optimizer runtime sum: `32.27` seconds.  Observed output file
mtime span including diagnostics, plots, and GIFs: `74.10` seconds.

Estimated official p=2/q=2 batch100/steps100 `main` run: `30-40 minutes`
without GIFs, or about `50-75 minutes` with full GIF rendering.

Dedicated runtime note: `docs/loss3_direction_proposal_ablation_smoke_runtime_20260517.md`.
