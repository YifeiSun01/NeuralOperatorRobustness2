# Loss3 Correlation and Radial-Growth Clarification - 2026-05-21

Status: clarification from existing generated tables; no experiment was rerun.

## Why The Correlations Differ

There are two different correlation questions in the current records.

### 1. Setting-level hit-to-final correlation

Source table:

- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/geometry_probe_correlations.csv`

Question:

```text
Across 20 p2q2 alpha/epsilon settings, if tangent residual is large at the first boundary hit,
is the total post-boundary gain from hit step to final larger?
```

Observed:

| method | x | y | r | pair count |
| --- | --- | --- | ---: | ---: |
| raw_add | post_boundary_gain | mean_tangent_grad_ratio_at_hit | 0.816 | 20 |
| steepest_add | post_boundary_gain | mean_tangent_grad_ratio_at_hit | 0.702 | 20 |
| replacement/GPI | post_boundary_gain | mean_tangent_grad_ratio_at_hit | 0.622 | 20 |

Interpretation:

- This is a coarse setting-level / total-room-to-improve statistic.
- It supports: when the optimizer first reaches the boundary with more tangent gradient left, it tends to have more total room to keep improving later.

### 2. Stepwise post-boundary next-step correlation

Source table:

- `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/tables/stepwise_geometry_correlations_by_method_scope.csv`

Question:

```text
For every sample and every step k after boundary_ratio >= 0.99,
does the tangent quantity at step k predict the immediate next loss increment L[k+1]-L[k]?
```

Observed for `scope=post_boundary_099`:

| method | metric | r with next_loss_gain | pair count |
| --- | --- | ---: | ---: |
| raw_add | grad_tangent_ratio | 0.337 | 534099 |
| steepest_add | grad_tangent_ratio | 0.265 | 573927 |
| replacement/GPI | grad_tangent_ratio | 0.029 | 598000 |

Interpretation:

- This is a fine step-level / immediate-increment statistic.
- A large tangent component means opportunity, but it does not by itself say that the actual next update direction uses that opportunity correctly.
- Replacement/GPI has especially weak stepwise tangent-size correlation because it is already rotating aggressively on the boundary; immediate gain depends on the chosen direction and nonlinear loss geometry, not only tangent magnitude.

These two results are not contradictory. The first asks whether tangent residual at boundary hit predicts total remaining gain over the rest of the run. The second asks whether tangent residual at each post-boundary step predicts the very next step's gain.

## Tangent Ratio vs Projected Linear Gain

`grad_tangent_ratio` is only a size measure:

```text
||(I - u u^T) grad L|| / ||grad L||
```

It answers:

```text
How much sideways gradient exists on the p=2 boundary?
```

It does not answer:

```text
Will this optimizer's next proposal actually move in that useful sideways direction?
```

`linear_gain_projected` is a local first-order prediction of the actual proposed move after projection:

```text
grad L dot (projected_next_delta - current_delta)
```

For additive methods this approximates:

```text
grad L dot (Proj(delta + alpha d) - delta)
```

For replacement methods this approximates:

```text
grad L dot (Proj(epsilon d) - delta)
```

Observed post-boundary correlations with immediate next-step gain:

| method | linear_gain_projected r |
| --- | ---: |
| raw_add | 0.600 |
| steepest_add | 0.299 |
| replacement/GPI | 0.097 |

Interpretation:

- `linear_gain_projected` is closer to the actual one-step question because it includes both gradient geometry and the method's chosen next direction after projection.
- It still does not perfectly predict replacement/GPI, because replacement makes large boundary jumps and the loss is nonlinear.

## LP-Steepest Norm Growth Clarification

The user is right that fixed step length alone does not mathematically guarantee straight norm growth. For additive updates before projection:

```text
r_next^2 = r^2 + 2 alpha r ||d|| cos(theta) + alpha^2 ||d||^2
```

So norm growth depends on both:

1. step length `||d||`, and
2. radial alignment `cos(theta)` between current delta and the update direction.

Existing data source:

- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/radial_growth_rollup_by_method.csv`

Observed across 20 p2q2 settings:

| method | mean cos(theta) | CV cos(theta) | mean direction L2 | CV direction L2 | CV actual radial increment |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_add | 0.869 | 0.247 | 0.359 | 0.621 | 0.621 |
| steepest_add | 0.700 | 0.478 | 1.000 | 3.3e-8 | 0.110 |

Interpretation:

- This does not show that `steepest_add` direction is fixed. In fact its `cos(theta)` varies substantially.
- It does show that `steepest_add` has fixed update norm and mostly positive radial alignment, so the radial increment is much steadier than raw PGD.
- Raw PGD's raw gradient norm varies strongly, and its radial increment variability tracks that gradient/update norm variability closely.
- Therefore the correct statement is not: fixed norm alone guarantees a straight curve. The correct statement is: in this data, normalized update length plus mostly positive radial alignment makes `steepest_add` radial progress much more regular than raw PGD.
