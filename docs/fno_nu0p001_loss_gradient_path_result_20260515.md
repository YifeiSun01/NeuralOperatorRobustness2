# FNO nu=0.001 Loss-Gradient Path Result

Date: 2026-05-15 UTC

Result root:

```text
results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631
```

Postprocess output:

```text
results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis
```

## What Was Analyzed

The original run completed 5 initial conditions and 3 attack objectives:

- indices: `0, 7, 40, 47, 115`
- attack losses: `loss1`, `loss2`, `loss3`
- saved steps used for this analysis: `k=5,10,15,20,25,30,35,40,45,50`

That gives `5 x 3 x 10 = 150` analyzed trajectory points.

The saved `trajectory.npz` files contain `x_adv`, `delta`, model output, solver
output, and residual output at the saved steps. They do not contain all three
cross-loss gradients at each point. Therefore the postprocess script
`tools/analyze_loss_gradient_path_results.py` reloaded every saved point and
recomputed the exact original-objective gradients for:

```text
g1 = grad loss1(x + delta_k)
g2 = grad loss2(x + delta_k)
g3 = grad loss3(x + delta_k)
```

It then computed pairwise cosine similarities and angles.

## Main Gradient-Angle Result

Across all 150 points:

| gradient pair | mean cosine | mean angle |
| --- | ---: | ---: |
| `g1` vs `g2` | `0.9904` | `3.91 deg` |
| `g1` vs `g3` | `0.5269` | `56.51 deg` |
| `g2` vs `g3` | `0.5167` | `57.23 deg` |

Interpretation:

- `loss1` and `loss2` gradients are almost the same direction on this FNO
  `nu=0.001` trajectory experiment.
- `loss3` is genuinely a different direction. It is not just a small variant of
  `loss1` or `loss2`.
- This directly supports the distinction between the local SVD/Jacobian story
  and the actual one-step loss-gradient story: the optimized loss matters.

## Dependence On Attack Trajectory

Averaged over the 50 points for each attack loss:

| attack trajectory | angle `g1,g2` | angle `g1,g3` | angle `g2,g3` | mean `loss1` | mean `loss2` | mean `loss3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimized `loss1` path | `1.01 deg` | `55.79 deg` | `55.98 deg` | `9.781` | `9.810` | `3.152` |
| optimized `loss2` path | `1.12 deg` | `52.44 deg` | `52.67 deg` | `9.484` | `9.577` | `2.565` |
| optimized `loss3` path | `9.60 deg` | `61.30 deg` | `63.05 deg` | `2.629` | `2.614` | `2.369` |

Interpretation:

- Along `loss1` and `loss2` attack paths, `g1` and `g2` are essentially
  indistinguishable.
- Along the `loss3` attack path, `g1` and `g2` are still much closer to each
  other than either is to `g3`, but they are less perfectly aligned.
- `loss3` path grows differently: it keeps `loss1/loss2` much smaller while
  targeting the model-solver discrepancy.

## Dependence On Step k

Averaged over all indices and all three attack trajectories:

| k | angle `g1,g2` | angle `g1,g3` | angle `g2,g3` | mean `loss1` | mean `loss2` | mean `loss3` | mean budget ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | `12.20 deg` | `74.24 deg` | `76.43 deg` | `3.370` | `3.445` | `0.471` | `0.225` |
| 10 | `5.34 deg` | `68.47 deg` | `69.07 deg` | `4.889` | `4.938` | `0.949` | `0.387` |
| 15 | `3.87 deg` | `62.81 deg` | `63.21 deg` | `6.232` | `6.275` | `1.641` | `0.542` |
| 20 | `3.14 deg` | `60.43 deg` | `61.01 deg` | `7.349` | `7.391` | `2.416` | `0.688` |
| 25 | `2.93 deg` | `54.48 deg` | `55.09 deg` | `7.977` | `8.015` | `2.992` | `0.785` |
| 30 | `3.03 deg` | `52.82 deg` | `53.63 deg` | `8.169` | `8.201` | `3.245` | `0.820` |
| 35 | `2.20 deg` | `49.89 deg` | `50.47 deg` | `8.390` | `8.415` | `3.478` | `0.851` |
| 40 | `2.16 deg` | `47.93 deg` | `48.50 deg` | `8.645` | `8.666` | `3.697` | `0.877` |
| 45 | `2.14 deg` | `47.19 deg` | `47.68 deg` | `8.892` | `8.910` | `3.941` | `0.903` |
| 50 | `2.08 deg` | `46.81 deg` | `47.23 deg` | `9.065` | `9.081` | `4.128` | `0.921` |

Interpretation:

- Early in the attack, the three objectives are much more different. At `k=5`,
  `g1/g3` and `g2/g3` are about `75 deg` apart.
- As `delta_k` grows, the directions become less different. By `k=50`,
  `g1/g3` and `g2/g3` are still different, but closer, about `47 deg` apart.
- This matches the mathematical expectation that small-`delta` behavior is
  strongly affected by affine/bias terms such as `J^T b`, while later behavior
  is more influenced by the larger perturbation and nonlinear trajectory.

## Final Step Results

At `k=50`, averaged over the five initial conditions:

| optimized attack | mean `loss1` | mean `loss2` | mean `loss3` | mean budget ratio | angle `g1,g2` | angle `g1,g3` | angle `g2,g3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| optimize `loss1` | `11.218` | `11.243` | `4.068` | `1.000` | `0.87 deg` | `51.00 deg` | `51.24 deg` |
| optimize `loss2` | `10.854` | `10.945` | `3.698` | `1.000` | `0.97 deg` | `41.63 deg` | `41.90 deg` |
| optimize `loss3` | `5.123` | `5.055` | `4.617` | `0.764` | `4.41 deg` | `47.80 deg` | `48.55 deg` |

Interpretation:

- The `loss3` attack gives the largest mean final `loss3`: `4.617`.
- Compared with optimizing `loss1`, direct `loss3` optimization is about
  `1.135x` higher on final `loss3` on average.
- Compared with optimizing `loss2`, direct `loss3` optimization is about
  `1.248x` higher on final `loss3` on average.
- However, this is not uniform across every index. Some individual initial
  conditions still get higher final `loss3` from `loss1` or `loss2` attacks.
- The `loss3` attack also uses less of the L2 budget on average at step 50
  (`0.764` instead of `1.0`), because raw-gradient PGD with `alpha=0.3` did not
  always drive the perturbation to the boundary for `loss3`.

## Direction Interpretation And Averaging Convention

The correct interpretation is not that all three losses have three mutually
unrelated update directions. The actual pattern is more specific:

```text
loss1 direction ~= loss2 direction
loss3 direction != loss1/loss2 direction
```

For this run, `loss1` and `loss2` are almost the same optimization direction:
across all 150 analyzed points, `angle(g1,g2) = 3.91 deg` on average, with mean
cosine `0.9904`. Along the `loss1` and `loss2` attack paths specifically, their
mean angle is only about `1 deg`.

By contrast, `loss3` is a genuinely different update direction. Across all 150
points:

- `angle(g1,g3) = 56.51 deg` on average;
- `angle(g2,g3) = 57.23 deg` on average.

The difference is strongest early and shrinks as the perturbation grows:

| k | angle `g1,g3` | angle `g2,g3` | interpretation |
| ---: | ---: | ---: | --- |
| 5 | `74.24 deg` | `76.43 deg` | small-`delta` regime, affine/bias term is very important |
| 25 | `54.48 deg` | `55.09 deg` | intermediate regime |
| 50 | `46.81 deg` | `47.23 deg` | directions are more similar, but still clearly different |

So, as `delta_k` grows, the update directions become more similar overall. But
even at `k=50`, `loss3` is not reduced to `loss1/loss2`; it still differs by
roughly `47 deg` on average.

The reported angles are computed as `mean(angle(point))`, not as
`angle(mean gradient)`. For example, the `k=50` row averages the angles from 15
separate points: 5 initial conditions times 3 attack trajectories. The final
`k=50 by attack loss` table averages 5 angles, one for each initial condition.

## What This Teaches

The cleanest conclusion is:

```text
loss1 and loss2 are nearly the same optimization direction for this FNO setting,
but loss3 is a substantially different optimization direction.
```

This is the main answer to the question about whether the loss-gradient story is
separate from the Jacobian/SVD story. It is separate.

The SVD/Jacobian experiment studied operator-gain directions such as the top
singular direction of `J_f`, `J_g`, or `J_f - J_g`. This trajectory experiment
studies the actual one-step gradients of the nonlinear losses along an attack
path. Those are different objects.

The observed path behavior supports the earlier mathematical decomposition:

- `loss1` behaves like a model-output change objective.
- `loss2` adds the fixed baseline mismatch `f(x)-g(x)`, but in this FNO run its
  gradient remains almost parallel to `loss1`.
- `loss3` differentiates through the perturbed solver output and therefore uses
  a model-solver discrepancy gradient. That gradient is much less aligned with
  `loss1/loss2`.

Practical consequence:

- If the goal is just to make the FNO output move, `loss1` and `loss2` are
  almost interchangeable here.
- If the goal is to enlarge the FNO-vs-solver discrepancy, `loss3` is the more
  targeted objective, even though raw PGD does not always make it win on every
  individual initial condition.
- The next cleaner comparison should control for budget usage, for example by
  using L2-steepest PGD or by evaluating fixed-radius directions, because the
  current raw-gradient `loss3` runs often end below the full L2 budget.

## Generated Files

- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/per_point_loss_gradient_angles.csv`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary_by_attack_loss.csv`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary_by_k.csv`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary_by_attack_loss_and_k.csv`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/final_k_summary_by_attack_loss.csv`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary.md`
