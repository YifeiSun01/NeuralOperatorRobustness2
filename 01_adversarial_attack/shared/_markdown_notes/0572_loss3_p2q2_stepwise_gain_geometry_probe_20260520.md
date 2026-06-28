# Loss 3 p2q2 Stepwise Geometry vs Next-Step Gain Probe - 2026-05-20

Status: generated from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.

## What This Tests

For each sample/method/step k, this probe compares geometry at step k with the immediate next loss increment:

```text
next_loss_gain = loss3_q[k+1] - loss3_q[k]
```

The goal is to test the user's finer question: does a larger tangent or radial component at a specific step predict a faster loss increase at the next step?

## Tables and Figures

- Stepwise means: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/tables/stepwise_geometry_mean_by_setting_method_step.csv`
- Method/scope correlations: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/tables/stepwise_geometry_correlations_by_method_scope.csv`
- Setting-level correlations: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/tables/stepwise_geometry_correlations_by_setting_method_scope.csv`
- Scatter sample: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/scatter_samples/post_boundary_scatter_sample.csv`
- Figures: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/figures`

## Post-Boundary Correlations With Next-Step Loss Gain

| method | metric | pearson_r_with_next_loss_gain | pair_count |
| --- | --- | --- | --- |
| raw_add | actual_delta_angle_next | 0.3495 | 5.341e+05 |
| raw_add | direction_tangent_ratio | 0.3374 | 5.341e+05 |
| raw_add | grad_radial_signed_ratio | -0.3271 | 5.341e+05 |
| raw_add | grad_tangent_ratio | 0.3374 | 5.341e+05 |
| raw_add | linear_gain_projected | 0.5996 | 5.341e+05 |
| raw_replace | actual_delta_angle_next | 0.03549 | 5.98e+05 |
| raw_replace | direction_tangent_ratio | 0.02877 | 5.98e+05 |
| raw_replace | grad_radial_signed_ratio | -0.04396 | 5.98e+05 |
| raw_replace | grad_tangent_ratio | 0.02877 | 5.98e+05 |
| raw_replace | linear_gain_projected | 0.09711 | 5.98e+05 |
| steepest_add | actual_delta_angle_next | 0.1631 | 5.739e+05 |
| steepest_add | direction_tangent_ratio | 0.2654 | 5.739e+05 |
| steepest_add | grad_radial_signed_ratio | -0.2224 | 5.739e+05 |
| steepest_add | grad_tangent_ratio | 0.2654 | 5.739e+05 |
| steepest_add | linear_gain_projected | 0.2991 | 5.739e+05 |
| steepest_replace | actual_delta_angle_next | 0.03549 | 5.98e+05 |
| steepest_replace | direction_tangent_ratio | 0.02877 | 5.98e+05 |
| steepest_replace | grad_radial_signed_ratio | -0.04396 | 5.98e+05 |
| steepest_replace | grad_tangent_ratio | 0.02877 | 5.98e+05 |
| steepest_replace | linear_gain_projected | 0.09711 | 5.98e+05 |

## Observed Conclusions

Observed from `stepwise_geometry_correlations_by_method_scope.csv`:

- For additive methods after the boundary is reached, tangent information is meaningfully related to the next-step loss gain, but it is not a perfect predictor. `grad_tangent_ratio` has correlation `r=0.337` for `raw_add` and `r=0.265` for `steepest_add`.
- The better one-step predictor is `linear_gain_projected`, because it includes both the local gradient geometry and the actual projected proposal made by the method. It has correlation `r=0.600` for `raw_add` and `r=0.299` for `steepest_add` post-boundary.
- The radial/normal signed ratio is negatively correlated with next-step gain post-boundary for additive methods (`r=-0.327` for `raw_add`, `r=-0.222` for `steepest_add`). In this p2q2 data, more tangent-facing geometry is more associated with immediate gain than more radial-facing geometry once the epsilon budget is already active.
- For replacement methods, the per-step post-boundary correlations are weak (`linear_gain_projected r=0.097`, `grad_tangent_ratio r=0.029`). This does not contradict the earlier hit-step result. The earlier result says replacement arrives at the boundary with a large tangent residual and has large total room to improve; this stepwise result says the immediate next-step gain is not explained well by tangent magnitude alone once replacement is already rotating aggressively on the boundary.

Inference from the above evidence:

- The user's finer question has a yes-but answer: tangent projection is related to next-step gain, especially for additive methods, but the actual projected local-linear gain is a cleaner predictor than tangent size by itself.
- "Tangent is large" means there is room to improve by rotating on the boundary. It does not guarantee that the optimizer's next step rotates in the useful direction.

## CV and LP-Steepest Norm Growth

`CV` means coefficient of variation: `standard deviation / mean`. It is just a relative wiggliness score. Larger CV means the quantity changes a lot relative to its average; smaller CV means it is steadier.

The earlier radial-growth probe showed:

| method | direction L2 CV | actual radial increment CV |
| --- | ---: | ---: |
| raw_add | 0.621 | 0.621 |
| steepest_add | 3.3e-8 | 0.110 |

This does **not** mean the `steepest_add` direction is fixed. Its direction can still change. The point is narrower:

- `steepest_add` uses a normalized direction, so the proposed step length is essentially fixed.
- `raw_add` uses the raw gradient, so the proposed step length changes a lot because the gradient norm changes a lot.
- Therefore the boundary-ratio/norm curve for `steepest_add` looks much straighter mainly because its step size is controlled. Direction changes still matter through the radial alignment term, but they are not the dominant source of wiggle in this data.

## Interpretation Guide

- `grad_tangent_ratio`: normalized size of `(I - u u^T) grad L`. It measures available tangent gradient, not whether the chosen update uses it correctly.
- `grad_radial_signed_ratio`: radial/normal alignment of the gradient with delta. Near 1 means mostly radial outward; near 0 means mostly tangent.
- `direction_tangent_ratio`: how tangent the method's proposed direction is relative to current delta.
- `linear_gain_projected`: local-linear prediction of the actual projected proposal's immediate gain, using recorded cosines and projection shrink. This is the most direct one-step predictor among these metrics.
- `actual_delta_angle_next`: the actual angle between delta[k] and delta[k+1].

Important caveat: a large tangent gradient is opportunity. It does not by itself guarantee the method's next proposal moves in the useful tangent direction. The one-step linear gain is the better test of whether the chosen step is aligned with the loss.
