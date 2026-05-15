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

## Figures

The following figures were generated from the same postprocessed CSV files:

- `docs/figures/fno_nu0p001_loss_gradient_path_dashboard_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_target_vs_loss3_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_angles_by_attack_20260515.png`

![FNO nu=0.001 loss-gradient path dashboard](figures/fno_nu0p001_loss_gradient_path_dashboard_20260515.png)

![FNO nu=0.001 target objective versus same-delta loss3](figures/fno_nu0p001_loss_gradient_path_target_vs_loss3_20260515.png)

![FNO nu=0.001 gradient angles by attack path](figures/fno_nu0p001_loss_gradient_path_angles_by_attack_20260515.png)

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

## Target Objective And Same-Delta Loss3

This table answers the specific question: when an attack optimizes `loss1` or
`loss2`, what is the target loss at that step, and what `loss3` does the same
`delta_k` produce? Rows are averaged over the five initial conditions.

| k | optimize loss1: target loss1 (same-delta loss3, budget) | optimize loss2: target loss2 (same-delta loss3, budget) | optimize loss3: direct loss3 (budget) | direct loss3 - loss1-path loss3 | direct loss3 - loss2-path loss3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | `4.9209 (L3=0.7918, b=0.323)` | `4.9880 (L3=0.2769, b=0.317)` | `0.3438 (b=0.036)` | `-0.4480` | `+0.0669` |
| 10 | `7.0218 (L3=1.6062, b=0.538)` | `6.9869 (L3=0.6956, b=0.530)` | `0.5438 (b=0.092)` | `-1.0624` | `-0.1518` |
| 15 | `8.8332 (L3=2.4386, b=0.734)` | `8.6381 (L3=1.5486, b=0.722)` | `0.9348 (b=0.171)` | `-1.5037` | `-0.6138` |
| 20 | `10.2920 (L3=3.2480, b=0.904)` | `9.9641 (L3=2.4301, b=0.892)` | `1.5692 (b=0.269)` | `-1.6788` | `-0.8609` |
| 25 | `10.9582 (L3=3.6986, b=0.992)` | `10.7516 (L3=3.1638, b=0.999)` | `2.1124 (b=0.365)` | `-1.5862` | `-1.0513` |
| 30 | `11.0809 (L3=3.8242, b=1.000)` | `10.8192 (L3=3.2967, b=1.000)` | `2.6134 (b=0.459)` | `-1.2108` | `-0.6833` |
| 35 | `11.1273 (L3=3.8887, b=1.000)` | `10.8617 (L3=3.4127, b=1.000)` | `3.1326 (b=0.554)` | `-0.7561` | `-0.2801` |
| 40 | `11.1638 (L3=3.9494, b=1.000)` | `10.8954 (L3=3.5186, b=1.000)` | `3.6236 (b=0.632)` | `-0.3258` | `+0.1050` |
| 45 | `11.1935 (L3=4.0084, b=1.000)` | `10.9228 (L3=3.6137, b=1.000)` | `4.1999 (b=0.708)` | `+0.1915` | `+0.5861` |
| 50 | `11.2179 (L3=4.0678, b=1.000)` | `10.9451 (L3=3.6982, b=1.000)` | `4.6171 (b=0.764)` | `+0.5493` | `+0.9188` |

This does not show that `loss1`/`loss2` always produce lower `loss3`. Early and
middle steps are confounded by budget usage: the `loss1` and `loss2` paths use
much more L2 budget, so their same-delta `loss3` can be larger than the
under-budget direct `loss3` path. At the final saved step, direct `loss3`
optimization gives the largest mean `loss3`: `4.6171`, compared with `4.0678`
on the `loss1` path and `3.6982` on the `loss2` path.

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
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/target_vs_loss3_by_attack_loss_and_k.csv`
- `docs/fno_nu0p001_loss_gradient_path_target_loss3_table_20260515.md`
- `docs/figures/fno_nu0p001_loss_gradient_path_dashboard_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_target_vs_loss3_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_angles_by_attack_20260515.png`
- `tools/plot_loss_gradient_path_figures.py`
