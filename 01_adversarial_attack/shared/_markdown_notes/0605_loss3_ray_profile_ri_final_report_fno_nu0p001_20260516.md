# Loss3 Ray / RI Local-To-Global Profile Final Report - 2026-05-16

Status: completed for FNO / 1D Burgers `nu=0.001`.

This is the final cleaned record for the Ray / RI experiment. It uses only the
corrected fixed-sign PGD100 data directory:

```text
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/
```

The earlier non-fixed-sign PGD output is invalid and was removed locally. The
invalid interpretation should not be used.

## One-Sentence Conclusion

The experiment succeeded: a direction that is locally fastest near the clean
input can lose after following the same fixed ray to a large radius. In this
setting, clean local directions win the small-radius diagnostics, but direct
`loss3_original` PGD gives the strongest finite-radius endpoint `loss3` at
`r=8`.

## Experimental Question

For a clean input `x`, choose several directions `v`, then evaluate the same
straight ray:

```text
x(r) = x + r v
```

as `r` moves from `0` to `8`.

The purpose is not just to ask which attack loss is largest. The purpose is to
test a nonlinear local-to-global question:

```text
Does the direction that gives the fastest infinitesimal/local growth remain the
best direction after we keep walking along that same ray to a large radius?
```

The answer from this run is no.

## Settings

- Model/problem: FNO, 1D Burgers, `nu=0.001`.
- Samples: `0..99`, batch size 100.
- Endpoint radius: `epsilon = 8.0`.
- Attack optimizer for finite-radius directions: normal projected gradient
  ascent, not Adam.
- Attack steps: `100`.
- PGD alpha: `0.3`.
- Initialization: zero.
- Direction used for ray profile: final-step direction only.
- No best-over-steps and no multi-restart selection.
- Runtime device: Tesla V100-SXM2-32GB.
- GPU policy: no CPU fallback.

## Directions Compared

Let

```text
f(x) = neural operator output
g(x) = PDE solver output
e(x) = f(x) - g(x)
J_e  = local Jacobian of e at the clean input x
```

The ray directions are:

| direction name | type | how the direction is obtained |
| --- | --- | --- |
| `local_outward_growth` | clean local direction | `v_out = normalize(J_e^T e(x) / ||e(x)||)` |
| `local_residual_movement` | clean local direction | approximately `argmax_{||v||=1} ||J_e v||`; implemented as the small-radius residual-movement winner |
| `loss3_original_final` | finite-radius PGD direction | final PGD direction maximizing `||e(x+delta)||` under `||delta|| <= 8` |
| `loss3_increment_ratio_final` | finite-radius PGD direction | final PGD direction maximizing `(||e(x+delta)|| - ||e(x)||) / (||delta|| + eta)` |
| `loss3_regularized_final` | finite-radius PGD direction | final PGD direction maximizing `||e(x+delta)|| - C ||delta||` |
| `loss3_residual_increment_ratio_final` | finite-radius PGD direction | final PGD direction maximizing `||e(x+delta)-e(x)|| / (||delta|| + eta)` |
| `random` | baseline | random normalized direction |

The two local directions are clean-point directions. They are meant to describe
infinitesimal behavior near `x`. The PGD directions are finite-radius attack
directions.

## Losses And Diagnostics Plotted Along Each Ray

For each fixed direction `v`, the experiment evaluates:

```text
L1(r) = ||f(x+r v) - f(x)||
L2(r) = ||f(x+r v) - g(x)||
L3(r) = ||f(x+r v) - g(x+r v)|| = ||e(x+r v)||
G(r)  = (L3(r) - L3(0)) / (r + eta)
E(r)  = ||e(x+r v) - e(x)|| / (r + eta)
F(r)  = ||f(x+r v) - f(x)|| / (r + eta)
J(r)  = ||g(x+r v) - g(x)|| / (r + eta)
```

The ratio panels can look as if they jump immediately after `r=0`. That is a
normal ratio-plot artifact: at `r=0`, the numerator is zero, while as
`r -> 0+` the ratio approaches a nonzero directional derivative.

## Key Data

| direction | small-radius result | endpoint `L3` mean at `r=8` | endpoint wins |
| --- | --- | ---: | ---: |
| `local_outward_growth` | small norm-growth wins `100/100` | 2.720 | 0/100 |
| `local_residual_movement` | small residual-increment wins `100/100` | 4.139 | 31/100 |
| `loss3_original_final` | small local wins `0/100` | 5.447 | 58/100 among all directions |
| `loss3_original_final` among finite PGD objectives | direct `loss3` PGD attack | 5.447 | 81/100 among the three finite attack objectives |

Aggregate direction table:

| direction | endpoint mean | endpoint wins | small growth mean | small growth wins | small residual-ratio mean | small residual wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `loss3_original_final` | 5.447 | 58 | 0.1461 | 0 | 0.6226 | 0 |
| `loss3_increment_ratio_final` | 4.221 | 10 | 0.2317 | 0 | 0.6005 | 0 |
| `local_residual_movement` | 4.139 | 31 | 0.0121 | 0 | 1.101 | 100 |
| `local_outward_growth` | 2.720 | 0 | 0.2414 | 100 | 0.5311 | 0 |
| `loss3_regularized_final` | 2.068 | 1 | 0.0827 | 0 | 0.4542 | 0 |
| `random` | 0.5268 | 0 | -0.00099 | 0 | 0.0462 | 0 |
| `loss3_residual_increment_ratio_final` | 0.2976 | 0 | 0 | 0 | 0 | 0 |

These numbers show the central pattern:

- `local_outward_growth` is the clean local winner for residual norm growth.
- `local_residual_movement` is the clean local winner for residual field
  movement.
- `loss3_original_final` is not locally steepest, but it is strongest at the
  finite endpoint by `L3(r=8)`.

## Crossover Evidence

Dense zoom curves were generated for `r in [0,1]` and `r in [0,2]` to check
where direct `loss3_original` PGD overtakes local/ratio directions.

Observed mean-curve crossover locations:

| comparison | crossover location |
| --- | ---: |
| `loss3_original_final` crosses `local_outward_growth` | `r ~= 0.823321` |
| `loss3_original_final` crosses `loss3_increment_ratio_final` | `r ~= 0.929778` |
| `loss3_original_final` vs `local_residual_movement` | no below-then-above crossing by `r=1` or `r=2` |

This is exactly the local-to-global behavior the experiment was designed to
show: local or ratio-style directions can look better near the clean point, but
the direct `loss3` PGD direction becomes stronger after moving far enough along
the ray.

## Best Figures To Use

If only one figure is used, use the full final formula/labeled/std grid:

```text
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_all_metrics_by_direction_0to8_formula_labeled_std.png
```

For the crossover story, use the zoomed `loss3` figures:

```text
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to0p1_formula_labeled_std.png
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to0p5_formula_labeled_std.png
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to1p0_formula_labeled_std.png
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to2p0_formula_labeled_std.png
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to8p0_formula_labeled_std.png
```

All final formula figures include line labels, mean curves, and shaded standard
deviation bands across the 100 samples.

## R2 Backup

Large data and images were uploaded to Cloudflare R2.

```text
s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/
```

Upload manifest:

```text
/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/r2_upload_manifest_20260516.txt
```

Manifest summary:

```text
file_count=58
bytes_total=97423607
deleted_wrong_non_fixedsign_objects=0
```

Full related-artifact sync manifest:

```text
/workspace/NeuralOperatorRobustness2/docs/r2_sync_manifest_ray_profile_20260516.md
```

The broader Ray-profile artifact sync uploaded `198` files and `145207187` bytes under the same R2 base prefix.

## Historical Bug And Superseded Results

The earlier Ray wrapper had a manual-PGD sign bug:

```text
wrong: backpropagate -objective, then update delta += alpha * grad
```

That is descent for the target objective. The corrected PGD version does:

```text
correct: backpropagate objective, then update delta += alpha * grad, then project
```

Adam did not have this exact sign bug, but Adam and PGD are different
optimization protocols and Adam was not the historical PGD attack protocol used
for the direct `loss3` comparison.

The final valid conclusion should therefore be based on the corrected
`zero_fixedsign` PGD100 run only.
