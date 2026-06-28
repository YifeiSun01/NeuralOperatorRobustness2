# Loss3 Selected Core Findings Summary - 2026-05-18

## One-Sentence Core Answer

For `p=2,q=2`, replacing PGD's additive update with direct boundary replacement
using the normalized gradient becomes the same final-delta method as the
GPI-style `steepest_replace` method.

This is because, for `p=2`, normalized gradient direction and L2-steepest
direction are the same direction.

## The Minimal Core Methods

| Method | What it means | Keep for core question? |
|---|---|---|
| `raw_add` | ordinary PGD: old delta plus alpha times raw gradient | yes |
| `raw_replace` | normalized raw-gradient direction, directly placed on boundary | yes |
| `steepest_add` | LP-steepest PGD: old delta plus alpha times steepest direction | yes |
| `steepest_replace` | GPI-style replacement: steepest direction directly placed on boundary | yes |
| JVP/VJP power variants | local operator-power directions using Jacobian push-forward/pull-back | no, separate question |
| affine JVP/VJP variants | biased local linearized operator-power direction | no, separate question |

## Important Naming Clarification

`raw_replace` is better understood as `unit_raw_replace`.

It is not ordinary PGD slowly walking until it hits the boundary. It is direct
boundary replacement using the normalized raw-gradient direction.

So:

`raw_replace` uses normalized gradient direction.

`steepest_replace` uses LP-steepest direction.

For `p=2`, these directions are the same.

For `p!=2`, they are generally different.

## p=2,q=2 Main Numeric Result

Source run:

`forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`

Final step `k=100`, batch size 100:

| Method | Final actual loss mean | Boundary ratio | High-frequency ratio | First-derivative L2 |
|---|---:|---:|---:|---:|
| `raw_add` | `5.389592` | `0.968878` | `5.90e-08` | `0.645722` |
| `raw_replace` | `6.804780` | `1.000000` | `8.10e-10` | `0.217019` |
| `steepest_add` | `6.377158` | `1.000000` | `8.00e-06` | `0.583815` |
| `steepest_replace` | `6.804780` | `1.000000` | `8.10e-10` | `0.217019` |

Interpretation:

`raw_replace` and `steepest_replace` give the same final loss and the same
roughness metrics for `p=2,q=2` because they are the same update in this setting.

## p=2,q=2 Final Delta Similarity

| Pair | Mean cosine | Mean relative L2 | Meaning |
|---|---:|---:|---|
| `raw_replace` vs `steepest_replace` | `1.000000` | `0.000000` | exactly same final delta |
| `raw_add` vs `steepest_add` | `0.858049` | `0.384210` | similar but not identical |
| `raw_add` vs `raw_replace` | `0.447369` | `0.933489` | boundary replacement changes final delta a lot |
| `steepest_add` vs `steepest_replace` | `0.530025` | `0.830928` | additive vs replacement matters |

## p=2 Method Equivalence Rule

For `p=2`, normalized raw-gradient direction equals L2-steepest direction.

Therefore, if all methods are present:

`unit_raw_add = steepest_add = power_add__objective_gradient`

and

`raw_replace = steepest_replace = power_replace__objective_gradient`

This explains why several similarity-matrix entries are exactly `1.000000`.

## p!=2 Rule

For `p!=2`, normalized raw-gradient direction and LP-steepest direction are
generally different.

So do not assume:

`raw_replace = steepest_replace`

for `p=1` or `p=inf`.

Observed support from completed `p=1` runs:

| Run | Pair | Mean cosine | Mean relative L2 | Meaning |
|---|---|---:|---:|---|
| `p=1,q=1` | `unit_raw_add` vs `steepest_add` | `0.113356` | `1.878867` | not equivalent |
| `p=1,q=2` | `unit_raw_add` vs `steepest_add` | `0.143339` | `1.852929` | not equivalent |
| `p=1,q=inf` | `unit_raw_add` vs `steepest_add` | `0.344037` | `1.776665` | not equivalent |

## Why JVP/VJP Was Removed From The Core Question

The core question is about changing the PGD update rule from additive to direct
boundary replacement.

JVP/VJP methods answer a different question: they try to find local
operator-power directions of the residual Jacobian.

Pure JVP/VJP looks at `J v` and ignores the current residual bias/base term.

But the local residual for the actual loss is closer to `b + J v`, where `b` is
the current residual.

So pure JVP/VJP can point toward a locally amplified Jacobian direction without
being the best direction for increasing the actual current loss.

Affine JVP/VJP partially adds the bias term back through `b + rho J v`, but it is
still a local linearized operator-power method, not the same as direct
objective-gradient replacement.

## Current Practical Conclusion

For the original research question, the clean experiment should ignore the extra
JVP/VJP variants and compare only:

1. `raw_add`
2. `raw_replace`
3. `steepest_add`
4. `steepest_replace`

For `p=2,q=2`, the result is already clear:

`raw_replace` is exactly the same final-delta method as `steepest_replace`.

For `p!=2`, the exact `raw_replace` comparison still needs to be run, because
some completed `pq_key` runs did not include `raw_replace`.

## Recommended Next Run

Run the minimal four-method set for non-2 p values:

`raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`

Recommended P/Q targets:

| Target | Why |
|---|---|
| `p=1,q=1` | strongest non-2 p geometry difference |
| `p=1,q=2` | compare with standard q=2 loss geometry |
| `p=1,q=inf` | check q-infinity behavior |
| `p=inf,q=2` | check coordinatewise p-bound geometry |
| `p=inf,q=inf` | hardest coordinatewise p/q geometry |

Report only actual loss, boundary ratio, final-delta similarity, roughness/high
frequency, and representative heatmap/3D delta surfaces.

## Simplified Figure-Only Folder

A compact figure-only folder for the `p=2,q=2` core comparison was generated at:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2`

It contains only three PNG files:

- `01_actual_loss_core4_mean_std.png`
- `02_delta_quality_core4_metrics.png`
- `03_final_delta_core4_selected_samples.png`

Detailed record: `docs/loss3_simplified_core_visualizations_20260518.md`.

