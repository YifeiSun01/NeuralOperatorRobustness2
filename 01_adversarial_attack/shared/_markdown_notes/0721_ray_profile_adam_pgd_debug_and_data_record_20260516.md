# Ray Profile Adam/PGD Debug And Data Record - 2026-05-16

Status: consolidated record of the Ray Profile debugging, optimizer comparison, and raw numerical summaries. This document is meant to prevent the experiment story from getting mixed up again.

## 1. Original Experiment Purpose

The Ray Profile / Local-to-Global Profile experiment was not originally just a generic loss-winner table.

The intended question was:

> If a direction has the fastest local growth near the clean input, does that same fixed ray remain the strongest direction at a large finite radius?

The experiment fixes a direction `v` and evaluates a straight ray:

```text
x(r) = x + r v
```

Then it plots quantities like:

```text
r -> loss3_original = ||f(x+r v) - g(x+r v)||
r -> loss3_increment_ratio
r -> residual-increment ratio
```

The key local-to-global claim is:

- At very small `r`, local directions can have the largest slope.
- At finite `r=epsilon`, those same local directions may not give the largest endpoint `loss3_original`.
- This gap is evidence of nonlinear finite-radius behavior: clean-point local structure does not fully determine the best large-radius attack direction.

Important local directions:

- `local_outward_growth`: clean-point direction maximizing immediate outward growth of `||e(x)||`, where `e=f-g`.
- `local_residual_movement`: direction maximizing small-radius residual movement `||e(x+r v)-e(x)||/r`.
- finite-radius attack directions: directions found by optimizing objectives such as `loss3_original`, `loss3_increment_ratio`, and `loss3_regularized`.

## 2. The Main Bug Found

A hand-written PGD branch in `tools/run_loss3_ray_profile_normal_batch.py` had a sign error.

Incorrect code logic:

```python
loss = -objective
loss.backward()
delta = delta + alpha * grad
```

Because `grad` here is the gradient of `-objective`, this update is actually:

```text
delta <- delta - alpha * grad(objective)
```

So it is gradient descent on the attack objective, not gradient ascent.

Correct manual PGD logic:

```python
objective.sum().backward()
delta = delta + alpha * grad(objective)
delta = project(delta, epsilon)
```

Equivalent alternative:

```python
(-objective).backward()
delta = delta - alpha * grad(-objective)
```

This bug affected the manual PGD Ray wrapper result. It did not explain the Adam runs, because Adam was minimizing `-objective` through the optimizer step, which is the correct ascent pattern for Adam.

The fixed commit was:

```text
466bb2e Fix PGD ray profile and add batch100 results
```

## 3. Historical PGD Cross-Check

A direct historical-script cross-check was run with:

```text
batch_size=100
epsilon=8
alpha=0.3
steps=100
optimizer=ordinary PGD
initial_delta=zero
model=FNO
Burgers nu=0.001
GPU=Tesla V100-SXM2-32GB
```

Historical-script output directory:

```text
results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/
```

Key historical-script boundary results:

| PGD objective | final delta norm mean | boundary loss3_original mean |
|---|---:|---:|
| `loss3_original_pgd` | 7.7510 | 5.4469 |
| `loss3_increment_ratio_pgd` | 7.8505 | 4.2215 |
| `loss3_regularized_pgd` | 0.3043 final, rescaled to 8.0 for boundary diagnostic | 2.0275 |

Conclusion:

- The old conclusion was reproduced: under the historical ordinary-PGD protocol, direct `loss3_original` gives the strongest endpoint `loss3_original` on average.
- Therefore the contradictory PGD Ray result was an implementation bug, not a real scientific reversal.

## 4. Corrected Ray PGD100 Result

Corrected Ray wrapper setting:

```text
batch=100
samples=0..99
epsilon=8
alpha=0.3
steps=100
optimizer=ordinary PGD
initialization=zero
selection=final step only
no best-over-steps
no multi-restart
```

Output directory:

```text
forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/
```

Endpoint `loss3_original` at `r=8`:

| direction | endpoint mean | endpoint wins among all profiled directions |
|---|---:|---:|
| `loss3_original_final` | 5.4469 | 58/100 |
| `loss3_increment_ratio_final` | 4.2215 | 10/100 |
| `loss3_regularized_final` | 2.0679 | 1/100 |
| `loss3_residual_increment_ratio_final` | 0.2976 | 0/100 |
| `local_outward_growth` | 2.7204 | 0/100 |
| `local_residual_movement` | 4.1391 | 31/100 |

Winner counts when comparing only the three finite-radius attack objectives by the same endpoint `loss3_original` metric:

| winner | count |
|---|---:|
| `loss3_original_final` | 81/100 |
| `loss3_increment_ratio_final` | 15/100 |
| `loss3_regularized_final` | 4/100 |

Small-radius diagnostics:

| diagnostic | winner | count |
|---|---|---:|
| small clean norm-growth | `local_outward_growth` | 100/100 |
| small residual-increment ratio | `local_residual_movement` | 100/100 |

Interpretation:

- Small-radius local directions win the local slope diagnostics.
- Large-radius endpoint behavior is strongest on average for direct `loss3_original` PGD.
- This is the intended local-to-global nonlinear gap.

## 5. Adam50 / Adam100 / PGD50 / PGD100 Comparison

The next question was whether Adam itself tends to produce smaller direct `loss3_original` attacks, or whether the earlier Adam result was just due to too few steps.

Four comparable runs were summarized:

| protocol | optimizer | steps | init | endpoint loss3_original mean for `loss3_original_final` | winner count among 3 finite objectives |
|---|---|---:|---|---:|---:|
| `Adam50_local` | Adam | 50 | local | 4.3653 | `loss3_original`: 3/100 |
| `Adam100_local` | Adam | 100 | local | 4.2093 | `loss3_original`: 2/100 |
| `PGD50_local` | ordinary PGD | 50 | local | 4.6497 | `loss3_original`: 85/100 |
| `PGD100_zero` | ordinary PGD | 100 | zero | 5.4469 | `loss3_original`: 81/100 |

Full three-objective winner counts:

| protocol | `loss3_original_final` | `loss3_increment_ratio_final` | `loss3_regularized_final` |
|---|---:|---:|---:|
| `Adam50_local` | 3 | 54 | 43 |
| `Adam100_local` | 2 | 57 | 41 |
| `PGD50_local` | 85 | 7 | 8 |
| `PGD100_zero` | 81 | 15 | 4 |

Endpoint means for the three finite-radius attack directions:

| protocol | original mean | increment-ratio mean | regularized mean |
|---|---:|---:|---:|
| `Adam50_local` | 4.3653 | 6.9918 | 5.7796 |
| `Adam100_local` | 4.2093 | 7.0524 | 5.7305 |
| `PGD50_local` | 4.6497 | 4.0266 | 2.0655 |
| `PGD100_zero` | 5.4469 | 4.2215 | 2.0679 |

Final norm means:

| protocol | original final norm | increment-ratio final norm | regularized final norm |
|---|---:|---:|---:|
| `Adam50_local` | 8.0000 | 7.5300 | 4.4785 |
| `Adam100_local` | 8.0000 | 7.3353 | 4.3951 |
| `PGD50_local` | 5.6245 | 1.5165 | 0.2407 |
| `PGD100_zero` | 7.7510 | 7.8505 | 0.3078 |

Initial conclusion from this comparison:

- The Adam discrepancy was not simply caused by `steps=50`; `Adam100_local` still showed the same ranking pattern.
- Switching only the optimizer from Adam to ordinary PGD while keeping `steps=50` and local initialization made direct `loss3_original` become the dominant winner again.
- Therefore Adam changes the constrained optimizer dynamics substantially.

## 6. Standard Deviation / Variance / Paired Significance

The user asked to add variance and standard deviation, not just means.

For endpoint `loss3_original` on the `loss3_original_final` direction:

| protocol | n | mean | std | variance | SE | approximate 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| `Adam50_local` | 100 | 4.365310 | 1.333548 | 1.778351 | 0.133355 | [4.100734, 4.629886] |
| `Adam100_local` | 100 | 4.209258 | 1.258978 | 1.585025 | 0.125898 | [3.959477, 4.459040] |
| `PGD50_local` | 100 | 4.649679 | 2.759004 | 7.612104 | 0.275900 | [4.102292, 5.197065] |
| `PGD100_zero` | 100 | 5.446888 | 2.704591 | 7.314813 | 0.270459 | [4.910297, 5.983479] |

Cross-protocol paired differences for the `loss3_original_final` direction:

| comparison | mean diff | std diff | SE diff | p-value | positive | negative |
|---|---:|---:|---:|---:|---:|---:|
| `PGD50_local - Adam50_local` | +0.284368 | 2.205702 | 0.220570 | 2.003e-01 | 49 | 51 |
| `PGD100_zero - Adam100_local` | +1.237630 | 2.128950 | 0.212895 | 7.496e-08 | 68 | 32 |
| `PGD100_zero - Adam50_local` | +1.081578 | 2.052120 | 0.205212 | 7.974e-07 | 65 | 35 |
| `PGD50_local - Adam100_local` | +0.440420 | 2.260050 | 0.226005 | 5.416e-02 | 50 | 50 |

Important nuance:

- `PGD50_local` has a higher mean than `Adam50_local` on the direct `loss3_original_final` direction, but this difference alone was not statistically significant (`p≈0.200`).
- The stronger and more significant observation is the within-protocol ranking: Adam makes `loss3_increment_ratio_final` and `loss3_regularized_final` beat direct `loss3_original_final`, while PGD makes direct `loss3_original_final` beat them.

Within-protocol paired differences on endpoint `loss3_original`:

| protocol | comparison | mean diff | std diff | SE diff | p-value | positive | negative |
|---|---|---:|---:|---:|---:|---:|---:|
| `Adam50_local` | original - increment_ratio | -2.626501 | 1.656713 | 0.165671 | 6.386e-29 | 6 | 94 |
| `Adam50_local` | original - regularized | -1.414256 | 3.088338 | 0.308834 | 1.359e-05 | 29 | 71 |
| `Adam100_local` | original - increment_ratio | -2.843095 | 1.637251 | 0.163725 | 8.241e-32 | 6 | 94 |
| `Adam100_local` | original - regularized | -1.521266 | 3.141793 | 0.314179 | 4.744e-06 | 30 | 70 |
| `PGD50_local` | original - increment_ratio | +0.623108 | 0.840015 | 0.084002 | 4.144e-11 | 91 | 9 |
| `PGD50_local` | original - regularized | +2.584161 | 2.568011 | 0.256801 | 7.977e-17 | 92 | 8 |
| `PGD100_zero` | original - increment_ratio | +1.225408 | 1.751128 | 0.175113 | 3.128e-10 | 84 | 16 |
| `PGD100_zero` | original - regularized | +3.379003 | 2.601970 | 0.260197 | 4.175e-23 | 96 | 4 |

Conclusion from the statistical checks:

- It is too strong to say Adam always makes the same direct `loss3_original_final` direction significantly smaller than PGD.
- It is supported to say Adam changes the optimizer dynamics and can reverse the ranking among attack objectives.
- Under Adam, direct `loss3_original` is significantly worse than increment-ratio and regularized directions in the `epsilon=8, alpha=0.3` local-init runs.
- Under ordinary PGD, direct `loss3_original` is significantly better than increment-ratio and regularized directions.

## 7. Two Additional Parameter Settings

The user requested two more `(epsilon, alpha)` settings to check whether Adam consistently makes the loss smaller.

A lightweight direct-comparison script was added:

```text
tools/compare_adam_pgd_loss3_original_params.py
```

This script only optimizes direct `loss3_original`; it does not recompute local Ray directions.

Settings:

```text
batch=100
samples=0..99
steps=50
objective=loss3_original
optimizer=Adam vs ordinary PGD
init=zero
epsilon=4, alpha=0.15
epsilon=12, alpha=0.45
```

Output directory:

```text
forensics/adam_vs_pgd_loss3_original_params_20260516/fno_nu0p001_batch100_steps50_eps4_eps12/
```

Summary table:

| setting | optimizer | boundary loss3 mean | std | variance | final norm mean |
|---|---|---:|---:|---:|---:|
| eps=4, alpha=0.15 | Adam | 1.6712 | 0.5538 | 0.3066 | 4.0000 |
| eps=4, alpha=0.15 | PGD | 2.3290 | 1.3685 | 1.8727 | 2.6570 final, rescaled to 4 |
| eps=12, alpha=0.45 | Adam | 8.1319 | 1.5485 | 2.3980 | 12.0000 |
| eps=12, alpha=0.45 | PGD | 7.2865 | 4.3876 | 19.2513 | 8.6585 final, rescaled to 12 |

Paired difference `PGD - Adam`:

| setting | metric | mean diff | std diff | p-value | PGD > Adam | Adam > PGD |
|---|---|---:|---:|---:|---:|---:|
| eps=4, alpha=0.15 | final loss3 | +0.3143 | 1.2265 | 1.189e-02 | 36 | 64 |
| eps=4, alpha=0.15 | boundary loss3 | +0.6578 | 0.9864 | 1.487e-09 | 75 | 25 |
| eps=12, alpha=0.45 | final loss3 | -1.5838 | 4.2216 | 2.961e-04 | 44 | 56 |
| eps=12, alpha=0.45 | boundary loss3 | -0.8454 | 3.7434 | 2.612e-02 | 48 | 52 |

Interpretation:

- At `epsilon=4`, PGD is significantly stronger than Adam at the boundary endpoint.
- At `epsilon=12`, Adam is stronger than PGD after 50 steps.
- This does not support a universal rule that Adam always gives smaller `loss3_original`.
- A likely reason for the `epsilon=12` reversal is boundary reach: Adam reaches the L2 boundary almost immediately, while PGD after 50 steps has final norm mean only about `8.66` before boundary rescaling.

## 8. Evaluation Metric Correction

One important source of confusion was how to define the "winner".

Correct winner definition for this Ray / attack comparison:

```text
Different objectives may be optimized,
but every resulting direction must be evaluated using the same endpoint metric.
```

For the main finite-radius attack comparison, the common endpoint metric is:

```text
loss3_original(x + epsilon v) = ||f(x + epsilon v) - g(x + epsilon v)||
```

So the comparison is not:

```text
loss3_original direction evaluated by loss3_original
loss3_increment_ratio direction evaluated by loss3_increment_ratio
loss3_regularized direction evaluated by loss3_regularized
```

The correct comparison is:

```text
loss3_original direction -> endpoint loss3_original
loss3_increment_ratio direction -> endpoint loss3_original
loss3_regularized direction -> endpoint loss3_original
local_outward_growth direction -> endpoint loss3_original
local_residual_movement direction -> endpoint loss3_original
```

In the batch-100 tables above, `winner count` means:

```text
number of samples, out of 100, for which that direction produced the largest endpoint loss3_original
```

It is an empirical win count / empirical win rate over the selected 100 samples.

This is also why the unified-plot script was added: the older `loss_curves` style plots often displayed the objective being optimized, while the corrected comparison needs plots where every trajectory is evaluated by the same metric, such as `loss3_original`.

## 9. User-Provided Historical Adam vs Normal PGD Curves

The user also provided earlier 2D Navier-Stokes plots comparing normal PGD and PGD-Adam. These are not new runs from this FNO/Burgers turn, but they are relevant historical observations because they show the same qualitative question: whether Adam tends to produce smaller or different attack loss trajectories than ordinary normalized PGD.

Values below are read from the plot legends.

### Historical Plot A: `alpha=1.0`, `epsilon=13.1072`

| loss/objective label | Adam final | normal PGD final |
|---|---:|---:|
| `detached1to5` | 2.72e4 | 3.51e4 |
| `detached1to9` | 7.34e3 | 4.06e3 |
| `detached5to9` | 4.04e3 | 8.54e3 |
| `withsolver` | 2.98e4 | 3.80e4 |

Observation:

- Normal PGD is larger for `detached1to5`, `detached5to9`, and `withsolver`.
- Adam is larger for `detached1to9`.
- This already suggests the optimizer effect is objective-dependent rather than a universal monotone rule.

### Historical Plot B: `alpha=1.0`, `epsilon=39.3216`

| loss/objective label | Adam final | normal PGD final |
|---|---:|---:|
| `detached1to5` | 9.00e4 | 1.27e5 |
| `detached1to9` | 2.36e4 | 3.37e4 |
| `detached5to9` | 2.69e4 | 2.88e4 |
| `withsolver` | 1.17e5 | 1.44e5 |

Observation:

- Normal PGD is larger for all four displayed objectives in this setting.

### Historical Plot C: `alpha=5.0`, `epsilon=13.1072`

| loss/objective label | Adam final | normal PGD final |
|---|---:|---:|
| `detached1to5` | 2.63e4 | 3.14e4 |
| `detached1to9` | 9.14e3 | 9.68e3 |
| `detached5to9` | 9.73e3 | 4.24e3 |
| `withsolver` | 2.84e4 | 3.48e4 |

Observation:

- Normal PGD is larger for `detached1to5`, `detached1to9`, and `withsolver`.
- Adam is larger for `detached5to9`.

### Historical Plot D: `alpha=5.0`, `epsilon=39.3216`

| loss/objective label | Adam final | normal PGD final |
|---|---:|---:|
| `detached1to5` | 8.94e4 | 1.08e5 |
| `detached1to9` | 2.89e4 | 1.98e4 |
| `detached5to9` | 3.31e4 | 2.08e4 |
| `withsolver` | 1.38e5 | 1.31e5 |

Observation:

- Adam is larger for `detached1to9`, `detached5to9`, and `withsolver`.
- Normal PGD is larger for `detached1to5`.
- This is another reason the final conclusion should not be "Adam always smaller"; the safer conclusion is that Adam changes the optimization dynamics and ranking, and the direction of the effect depends on objective, epsilon, alpha, and geometry.

## 10. Final Consolidated Interpretation

The clean, defensible story is:

1. The Ray Profile experiment is about local-to-global nonlinear mismatch, not just a flat loss leaderboard.
2. A real implementation bug occurred in the hand-written PGD branch: the update direction was reversed. This explains the first contradictory PGD result.
3. After fixing the PGD sign, the historical ordinary-PGD conclusion is restored: direct `loss3_original` is the strongest finite-radius endpoint objective under the historical protocol.
4. Adam did not have the PGD sign bug, but Adam changes constrained attack dynamics substantially.
5. For `epsilon=8, alpha=0.3`, Adam makes increment-ratio / regularized directions beat direct `loss3_original`, while ordinary PGD makes direct `loss3_original` win.
6. Across other `(epsilon, alpha)` settings, Adam is not always smaller. Its behavior depends on epsilon, alpha, step count, initialization, projection, and whether the optimizer reaches the boundary.
7. Therefore, Adam-vs-PGD should be treated as a separate optimizer-dynamics issue, not as a simple monotone rule.

## 11. Relevant Files

Scripts:

- `tools/run_loss3_ray_profile_normal_batch.py`
- `tools/compare_adam_pgd_loss3_original_params.py`
- `tools/plot_batch_eval_metric_matrix_curves.py`

Main docs:

- `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`
- `docs/adam_vs_pgd_loss3_original_param_control_20260516.md`
- `docs/unified_eval_metric_loss_curve_plots_20260516.md`
- this file: `docs/ray_profile_adam_pgd_debug_and_data_record_20260516.md`

Main output directories:

- `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`
- `forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/`
- `forensics/loss3_ray_profile_adam_control_20260516/fno_nu0p001_gpu_v100_batch100_steps100_local/`
- `forensics/adam_vs_pgd_loss3_original_params_20260516/fno_nu0p001_batch100_steps50_eps4_eps12/`

R2 backup prefixes include the corresponding `forensics/` directories under:

```text
neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/
```
