# Loss3 Simplified Core Visualizations - 2026-05-18

## Scope

Generated a simplified figure-only folder for the core `p=2,q=2` comparison.

Output folder:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2`

The folder contains only PNG images: `True`.

## Methods Included

Only the four core methods are included:

| Plot label | Method key | Meaning |
|---|---|---|
| PGD additive | `raw_add` | ordinary PGD, additive raw-gradient update |
| PGD boundary replacement | `raw_replace` | normalized raw-gradient direction placed directly on the boundary |
| Lp-steepest PGD | `steepest_add` | additive LP-steepest update |
| GPI / steepest replacement | `steepest_replace` | generalized-power-style boundary replacement |

No JVP/VJP, affine JVP/VJP, or generalized-pq operator-power variants are included.

## Figures

1. `01_actual_loss_core4_mean_std.png`

   Actual `loss3_q` curve for the four core methods, with mean and sample-standard-deviation shading across the 100 samples.

2. `02_delta_quality_core4_metrics.png`

   Compact delta diagnostics: boundary ratio, high-frequency energy ratio, first-derivative L2, and total variation.

3. `03_final_delta_core4_selected_samples.png`

   Final `delta` shape at step 100 for representative dataset indices `0`, `7`, `40`, and `47`.

## Important Scope Note

This simplified folder is currently for `p=2,q=2` only. It is the only completed
run that currently has all four required methods: `raw_add`, `raw_replace`,
`steepest_add`, and `steepest_replace`.

Other completed P/Q runs from the `pq_key` queue omitted `raw_replace`, so they
cannot yet produce the same four-method simplified plot without an additional
minimal run.

## Preservation Note

Existing figures were not deleted or overwritten. These simplified figures were
written to a new folder.


## 2026-05-18 01:41 UTC Update

`02_delta_quality_core4_metrics.png` was redrawn with mean curves, `+/- 1` sample-standard-deviation shading, and final-step mean `+/-` sample std bars.

## 2026-05-18 01:48 UTC Update

`02_delta_quality_core4_metrics.png` was redrawn again in a wide/flat layout after
visual inspection. The left side now uses four horizontally wide curve panels,
each with mean line and `+/- 1` sample-standard-deviation shading. The right side
keeps compact final-step mean `+/-` sample std summaries. The folder still
contains only the same three PNG files.

Other P/Q pairs currently do not have a real `raw_replace` run except
`p=2,q=2`. Matching four-method simplified figures for non-2 p values therefore
require a minimal `raw_replace`补跑. For `p=2` pairs only, `raw_replace` can be
inferred from `steepest_replace` by L2 geometry, but such a plot should be
labeled as inferred unless `raw_replace` is actually run.
