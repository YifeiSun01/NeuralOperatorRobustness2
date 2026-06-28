# P/Q Geometry, Optimizer Curve Overlap, and Binary Darcy Notes

Date: 2026-05-29

This note summarizes the old markdown records about optimizer curve overlap in the loss-3/loss-2/loss-4 experiments, and adds the strict binary Darcy-flow case.

Main source records checked:

- `BATCH_LOSS_ONLY_OPTIMIZATION_METHODS.md`
- `docs/loss3_p_norm_method_equivalence_rules_20260518.md`
- `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`
- `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- `docs/ns2d_vs_1d_burgers_optimizer_curve_location_check_20260523.md`
- `docs/ns_burgers_optimizer_winner_and_equivalence_summary_20260524.md`
- `docs/optimizer_grouped_loss_curves_20260524.md`
- `2D_Darcy_FNO2d/DARCY_BINARY_ATTACK_NOTES.md`
- `analysis_outputs/darcy_flow_figures_clean_bundle_with_loss4_20260529/README.md`

## 1. Unified Setup

The attack is written as

```text
maximize_delta  O(delta)
subject to      ||delta||_p <= epsilon
```

Here:

- `p` is the input perturbation constraint geometry.
- `q` is the output/objective norm geometry.
- `g = grad_delta O(delta)` is the gradient of the chosen objective with respect to the perturbation.

The important optimizer labels are:

| Label | Meaning |
|---|---|
| `raw_add` | Add the unnormalized gradient direction, then project back to the `p` ball. |
| `unit_raw_add` | Add the `p`-normalized raw gradient direction, then project. |
| `raw_replace` | Replace `delta` by the boundary-normalized raw gradient direction. |
| `steepest_add` | Add the `p`-steepest-ascent direction, then project. |
| `steepest_replace` | Replace `delta` by the boundary `p`-steepest-ascent direction. |
| `power_add__objective_gradient` | Same as `steepest_add` in the current implementation. |
| `power_replace__objective_gradient` | Same as `steepest_replace` in the current implementation. |
| `power_*__generalized_pq` / `power_*__pure_jvp_vjp` | Duplicate labels in the current implementation for the same JVP/VJP route. |

The mathematical steepest-ascent direction under an input `p` constraint is

```text
s_p(g) = argmax_{||s||_p <= 1} <g, s>.
```

For `1 < p < infinity`, with dual exponent `p* = p / (p - 1)`,

```text
s_p(g) = normalize_p(sign(g) * |g|^(p* - 1)).
```

Special cases:

```text
p = 2        s_2(g)   = g / ||g||_2
p = infinity s_inf(g) = sign(g)
p = 1        s_1(g)   = one-hot at argmax_i |g_i|, with sign(g_i)
```

So exact overlap is mostly controlled by `p`. The value of `q` changes the objective gradient `g` and therefore changes performance, but for the core objective-gradient methods it does not change the algebraic fact that `p=2` makes the normalized raw direction equal to the steepest direction.

## 2. Continuous-Valued Attacks: General P/Q Rules

### Case A: `p = 2`

For continuous perturbations under an L2 ball,

```text
s_2(g) = g / ||g||_2.
```

Therefore the L2-normalized raw-gradient direction and the L2-steepest direction are exactly the same.

Exact method groups:

```text
unit_raw_add = steepest_add = power_add__objective_gradient
raw_replace = steepest_replace = power_replace__objective_gradient
```

Important distinction:

```text
raw_add != steepest_add
```

in general, because `raw_add` uses the unnormalized gradient magnitude. It has a different step length before projection, so it can follow a different trajectory even when the direction is collinear at each step.

For the core four-method plots:

```text
raw_add
raw_replace
steepest_add
steepest_replace
```

the exact deterministic overlap at `p=2` is normally:

```text
raw_replace = steepest_replace
```

while `raw_add` and `steepest_add` are generally separate. So the expected visual count is often three visible curves, not four.

If `raw_add` and `steepest_add` happen to be numerically close, the plot may look like only two curves, but that is visual closeness, not the exact p=2 equivalence.

### Case B: `p != 2`

For `p != 2`, normalized raw gradient and steepest direction are not generally the same.

Examples:

```text
p = infinity: steepest direction is sign(g)
p = 1:        steepest direction is one-hot argmax |g_i|
```

So the expected exact overlap between raw-gradient and steepest-gradient methods disappears:

```text
unit_raw_add != steepest_add
raw_replace  != steepest_replace
```

unless there is a special accidental gradient structure, duplicated code path, or plotting/rounding effect.

The old p=1 records confirm this: cosine similarities between `unit_raw_add` and `steepest_add` were far from 1, so they were not equivalent.

## 3. Continuous Burgers and Navier-Stokes Records

### 1D Burgers, continuous perturbation, `p=2, q=2`

Old records show that the replacement pair is exactly overlapping:

```text
raw_replace = steepest_replace
```

but the additive pair is not exactly overlapping:

```text
raw_add != steepest_add
```

Representative old run:

```text
Burgers loss3, eps=4, alpha=0.4, p=2, q=2
raw_replace vs steepest_replace: max_abs_diff = 0
raw_add vs steepest_add:         max_abs_diff > 0
```

Recorded final values from one downloaded per-step check:

| Method | Final loss3_q mean |
|---|---:|
| `raw_add` | 2.9997486937 |
| `raw_replace` | 3.0616590244 |
| `steepest_add` | 3.0650239444 |
| `steepest_replace` | 3.0616590244 |

Interpretation:

- `raw_replace` and `steepest_replace` are the same line.
- `raw_add` and `steepest_add` are different lines, although they can look close.
- If the plot contains only the core four methods, the mathematically expected count is three visible curves.

### 2D Navier-Stokes, continuous perturbation, `p=2, q=2`

The same p=2 replacement equivalence appears:

```text
raw_replace = steepest_replace
```

but additive methods can be very different.

Representative old run:

```text
NS2D loss3, eps=32, alpha=10, p=2, q=2
raw_replace vs steepest_replace: max_abs_diff = 0
raw_add vs steepest_add:         max_abs_diff large
```

Recorded final active-loss values:

| Method | Final active_loss_mean |
|---|---:|
| `raw_add` | 155.7122276306 |
| `raw_replace` | 106.3228977203 |
| `steepest_add` | 304.5929718018 |
| `steepest_replace` | 106.3228977203 |

Interpretation:

- `raw_replace` and `steepest_replace` are exactly the same curve.
- `raw_add` and `steepest_add` are not the same curve.
- In small-epsilon NS2D cases, additive methods can look visually close because the feasible ball is small, but exact data checks still distinguish them.

## 4. Strict Binary Darcy Flow

The binary Darcy case is different from the continuous Burgers/NS cases.

The coefficient field is strict binary:

```text
A_i in {3, 12}
```

Every valid flip has the same magnitude:

```text
|d_i| = |12 - 3| = 9.
```

At a pixel `i`, the only possible flip is:

```text
if A_i = 3:  d_i = +9
if A_i = 12: d_i = -9
```

For a first-order objective increase,

```text
O(delta + d_i e_i) - O(delta) approx g_i d_i.
```

The steepest binary choice scores candidate flips by

```text
g_i d_i.
```

The raw-gradient choice, after requiring sign compatibility, effectively ranks candidates by

```text
|g_i|.
```

But because every valid flip has the same magnitude 9,

```text
g_i d_i = 9 |g_i|
```

on sign-compatible flips.

Therefore strict binary Darcy collapses raw and steepest rankings:

```text
raw_add     = steepest_add
raw_replace = steepest_replace
```

provided the initialization, candidate set, tie-breaking, and random start are the same.

This is why the binary Darcy curves can have more overlap than continuous Burgers/NS curves.

### Binary Darcy Observations

Old notes record:

| Objective | Observed overlap |
|---|---|
| loss2 | `raw_add = steepest_add`, `raw_replace = steepest_replace` |
| loss3 | `raw_add = steepest_add`, `raw_replace = steepest_replace` |
| loss4 | `raw_add = steepest_add`, `raw_replace = steepest_replace` |
| loss1 | Not always exact, because `loss1_random_start=true` and tie/random effects can break equality. |

Recorded binary Darcy examples:

```text
loss2:
raw_add = steepest_add final true loss3 0.032282
raw_replace = steepest_replace final true loss3 0.031018

loss3:
raw_add = steepest_add final true loss3 0.048095
raw_replace = steepest_replace final true loss3 0.052848

loss1:
raw_add and steepest_add are close but not exact
raw_replace and steepest_replace are close but not exact
```

Interpretation:

- In strict binary Darcy, the add pair can also collapse, not only the replace pair.
- This is stronger overlap than the continuous `p=2` rule.
- Add-vs-replace can still remain different because add appends/accumulates flips, while replace recomputes a whole mask from the current gradient.

## 5. Why Continuous Burgers/NS and Binary Darcy Look Different

### Continuous p=2 case

The exact reason for overlap is:

```text
L2-normalized raw direction = L2-steepest direction.
```

This makes replacement methods identical:

```text
raw_replace = steepest_replace
```

If `unit_raw_add` exists, then:

```text
unit_raw_add = steepest_add
```

But ordinary `raw_add` is unnormalized, so:

```text
raw_add != steepest_add
```

usually.

### Binary Darcy case

The exact reason for overlap is different:

```text
all valid flips have equal magnitude 9.
```

So raw ranking and steepest binary ranking become the same ranking:

```text
g_i d_i = 9 |g_i|.
```

This can collapse both:

```text
raw_add     = steepest_add
raw_replace = steepest_replace
```

That is why binary Darcy may show only two curves in loss2/loss3/loss4:

```text
add-pair curve
replace-pair curve
```

while continuous Burgers/NS usually show at least three curves:

```text
raw_add
steepest_add
replace-pair curve
```

## 6. Expected Visible Line Counts

| Setting | Expected exact overlap | Typical visible lines |
|---|---|---:|
| Continuous, `p=2`, core four methods | `raw_replace = steepest_replace` | 3 |
| Continuous, `p=2`, with `unit_raw_add` / power labels | `unit_raw_add = steepest_add = power_add`, `raw_replace = steepest_replace = power_replace` | Depends on labels; exact groups collapse |
| Continuous, `p!=2`, core four methods | No raw/steepest exact overlap in general | 4 |
| Continuous, `p=2`, very small epsilon | Replacement exact; add curves may visually look close | 2 or 3 visually, but data may show 3 |
| Binary Darcy, loss2/loss3/loss4 | `raw_add = steepest_add`, `raw_replace = steepest_replace` | 2 |
| Binary Darcy, loss1 with random start | Equality can break due random start/ties | 3 or 4 possible |

## 7. Final Practical Rule

Use this rule when reading the plots:

```text
If it is continuous Burgers/NS with p=2:
    the guaranteed overlap is raw_replace = steepest_replace.
    raw_add and steepest_add are not guaranteed to overlap.

If it is strict binary Darcy with values {3, 12}:
    raw and steepest often collapse for both add and replace,
    because every valid flip has the same magnitude.

If p != 2 in continuous attacks:
    raw and steepest should usually separate.

If curves overlap visually:
    check saved per_step_metrics before calling them mathematically identical.
```

The key conclusion is:

```text
Binary Darcy is more degenerate than continuous Burgers/NS.
The binary constraint makes the optimizer choices simpler,
so more curves can become exactly identical.
```

