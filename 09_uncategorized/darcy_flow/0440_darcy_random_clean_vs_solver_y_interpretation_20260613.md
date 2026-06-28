# Darcy Random Clean-Y vs Random Solver-Y Results - 2026-06-13

This note records the random-source Darcy Flow training results and the current
interpretation of why `random_clean_y` can outperform `random_solver_y` on the
reported generalization and robustness metrics.

## Runs

Both random-source methods were trained from the same pretrained Darcy FNO
baseline for 1100 epochs.

| method | training mode | final checkpoint |
|---|---|---|
| `random_clean_y` | random binary perturbation, clean target held fixed | `adversarial_training_runs/darcy_binary_random_binary_fixed_y_1100ep_full50_20260613_random_binary_source_1100/darcy/checkpoints/darcy_epoch1100_step001100.pt` |
| `random_solver_y` | random binary perturbation, target recomputed by solver | `adversarial_training_runs/darcy_binary_random_binary_solver_y_1100ep_full50_20260613_random_binary_source_1100/darcy/checkpoints/darcy_epoch1100_step001100.pt` |

The random perturbation is binary-feasible: each coefficient pixel is projected
back to the per-sample two Darcy coefficient values after random flips. The
random field parameters vary across epochs/samples, using kernels from
`gaussian,matern,highpass,bandpass,mixed`, alpha values
`1.2,2.2,3.2,4.2,5.2`, lengthscale in `[0.035, 0.30]`, and flip fraction in
`[0.005, 0.05]`.

## Main Explanation

The surprising observation is real in the current results:
`random_clean_y` is better than `random_solver_y` on clean generalization and
20-step attack robustness.

The reason is that the two objectives optimize different invariances.

`random_solver_y` trains the physically correct perturbed mapping:

$$
F_\theta(a+\delta) \approx u(a+\delta).
$$

This is the correct PDE label for the perturbed input, but it encourages the
model to respond to the random binary coefficient flips.

`random_clean_y` trains a label-preserving consistency objective:

$$
F_\theta(a+\delta) \approx u(a).
$$

This target is not the exact PDE solution for the perturbed input, but it
directly enforces local output stability:

$$
F_\theta(a+\delta) \approx F_\theta(a).
$$

The robustness metrics here reward this local stability. In particular,
adversarial loss growth is more closely tied to error-aligned sensitivity,

$$
\nabla_a \frac{1}{2}\|F_\theta(a)-u(a)\|^2
=
J(a)^T (F_\theta(a)-u(a)),
$$

than to the physical correctness of labels on random off-manifold perturbations.

For Darcy Flow, the solver response to a 0.5%-5% binary coefficient flip can be
small enough that the label bias in `random_clean_y` is outweighed by the
stability regularization. By contrast, `random_solver_y` presents a moving target
that can increase training variance and preserve input sensitivity.

This does not mean solver labels are wrong. It means the current evaluation
criteria emphasize robustness/invariance around the original binary data
manifold, where `random_clean_y` acts like a strong consistency regularizer.

## Generalization Clean Loss

Lower is better.

| rank | model | mean relative L2 | median relative L2 | mean RMSE |
|---:|---|---:|---:|---:|
| 1 | `loss3` | 0.0614064 | 0.0623735 | 0.000649839 |
| 2 | `loss2` | 0.0721530 | 0.0733341 | 0.000770274 |
| 3 | `random_clean_y` | 0.0739331 | 0.0737779 | 0.000788917 |
| 4 | `physics_loss` | 0.0878201 | 0.0905458 | 0.000933881 |
| 5 | `loss1` | 0.0879285 | 0.0881902 | 0.000936259 |
| 6 | `baseline` | 0.0913369 | 0.0924669 | 0.000972351 |
| 7 | `random_solver_y` | 0.0959063 | 0.0995798 | 0.00101952 |

Conclusion: `loss3` is best; `random_clean_y` is third; `random_solver_y` is
worst on the 50 generalization datasets.

## Attack20 Generalization Robustness

The attack is a 20-step binary Darcy attack with `epsilon_fraction = 0.025`.
Lower attack gain and lower adversarial loss are better.

| rank | model | clean loss | attack gain | attacked loss |
|---:|---|---:|---:|---:|
| 1 | `loss3` | 4.76911e-07 | 2.73073e-06 | 3.20764e-06 |
| 2 | `random_clean_y` | 5.26291e-07 | 3.12722e-06 | 3.65351e-06 |
| 3 | `loss2` | 7.15598e-07 | 4.06879e-06 | 4.78439e-06 |
| 4 | `physics_loss` | 7.90779e-07 | 4.14508e-06 | 4.93586e-06 |
| 5 | `random_solver_y` | 8.61425e-07 | 4.59724e-06 | 5.45866e-06 |
| 6 | `loss1` | 9.74390e-07 | 4.85208e-06 | 5.82647e-06 |

Conclusion: `loss3` is best on actual attack robustness; `random_clean_y` is
second; `random_solver_y` is substantially worse.

## 25-Sample Robustness/Jacobian Metrics

Lower is better for all columns shown here except where explicitly stated.

| metric | best model | best value | full ranking |
|---|---|---:|---|
| actual attack gain | `loss3` | 2.75651e-06 | `loss3`, `random_clean_y`, `loss2`, `physics_loss`, `random_solver_y`, `loss1` |
| `||J^T e||` | `random_clean_y` | 9.85303e-05 | `random_clean_y`, `loss3`, `loss2`, `physics_loss`, `random_solver_y`, `loss1` |
| binary first-order gain | `random_clean_y` | 1.25688e-06 | `random_clean_y`, `loss3`, `loss2`, `physics_loss`, `loss1`, `random_solver_y` |
| top singular value `sigma_max` | `loss2` | 0.00172255 | `loss2`, `loss1`, `random_clean_y`, `random_solver_y`, `physics_loss`, `loss3` |

Key point: `random_clean_y` has the smallest `||J^T e||` and binary first-order
gain, but `loss3` still has the smallest measured attack gain. The pure spectral
norm `sigma_max` does not explain attack robustness here.

## 5-Sample Jacobian Probe

Lower is better.

| metric | best model | best value | full ranking |
|---|---|---:|---|
| clean relative L2 | `loss3` | 0.0785856 | `loss3`, `random_clean_y`, `loss2`, `physics_loss`, `random_solver_y`, `loss1` |
| `||J^T e||` | `random_clean_y` | 1.17679e-04 | `random_clean_y`, `loss3`, `loss2`, `physics_loss`, `random_solver_y`, `loss1` |
| `||J^T e||^2` | `random_clean_y` | 1.75566e-08 | `random_clean_y`, `loss3`, `loss2`, `physics_loss`, `random_solver_y`, `loss1` |
| `||J e||` | `random_clean_y` | 4.29411e-05 | `random_clean_y`, `loss3`, `loss2`, `random_solver_y`, `physics_loss`, `loss1` |
| top singular value `sigma_max` | `loss2` | 0.00175943 | `loss2`, `loss1`, `random_clean_y`, `random_solver_y`, `physics_loss`, `loss3` |

## Final Takeaway

The current result should be stated carefully:

`loss3` remains the best method for clean generalization and measured
adversarial robustness. `random_clean_y` is a strong second-place method and has
the smallest error-aligned Jacobian sensitivity in the sampled probes.
`random_solver_y` is more physically faithful for perturbed inputs, but it is
worse on the original clean/generalization and binary attack robustness metrics.

The likely mechanism is:

$$
\texttt{random_clean_y}
\quad\Rightarrow\quad
\text{local invariance / consistency regularization},
$$

while

$$
\texttt{random_solver_y}
\quad\Rightarrow\quad
\text{equivariance to random PDE label changes}.
$$

The present evaluation rewards the former more strongly.

## Stored Artifacts

- Random-source posthoc analysis:
  `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc`
- Six-model comparison report:
  `analysis_outputs/darcy_six_model_random_source_report_20260613`
- Random-inclusive visualization bundle:
  `visualizations/darcy_random_inclusive_burgers_style_bundle_20260613`
- Random-inclusive figure report:
  `docs/darcy_random_inclusive_burgers_style_bundle_20260613.md`
- This interpretation note:
  `docs/darcy_random_clean_vs_solver_y_interpretation_20260613.md`

