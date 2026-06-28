# Three-Loss Objective Experiment Plan

## Goal

This experiment separates three things that should not be mixed:

1. Which base loss is optimized.
2. Which objective form is used for that base loss.
3. Which optimizer/update rule is used.

For every run, the optimized objective is only one selected target, but the
evaluation must record all cross-loss quantities. In particular, every run
should save the full 3 x 3 evaluation table:

- `loss1_original`
- `loss1_increment_ratio`
- `loss1_regularized`
- `loss2_original`
- `loss2_increment_ratio`
- `loss2_regularized`
- `loss3_original`
- `loss3_increment_ratio`
- `loss3_regularized`

This is the main bookkeeping rule: even if the attack optimizes only one of
these nine objectives, all nine must be evaluated and written at every recorded
step and in the final summary.

## Fixed Parameters

The first round should not sweep epsilon, alpha, seed, or sample index, but
epsilon and alpha must not be chosen without looking at the input scale.

Deprecated tiny-smoke values:

```text
epsilon = 0.01
alpha   = 0.005
```

These values are too small for Burgers FNO L2 attacks on 1024-point inputs. For
`input_p=2`, `epsilon=0.01` means the dense per-point RMS perturbation is only

```text
0.01 / sqrt(1024) = 0.0003125
```

This is only useful as a local-linear smoke test, not as a serious attack scale.

Use scale-calibrated fixed values instead. For the historical Burgers FNO
`nu=0.001` L2 setting, the clean rerun used:

```text
epsilon = 8
alpha   = 0.15
```

The stronger 4x setting used:

```text
epsilon = 32
alpha   = 0.6
```

The current round-1 rerun requested for L2 uses:

```text
epsilon = 8
alpha   = 0.3
steps   = 100
seed    = 0
input_p = 2
output_q = 2
```

Index handling is outside the experiment grid. If the attack script supports a
batch of 100 samples, that batch already covers the sample dimension, so index
is not counted as a sweep axis in this plan.

## Epsilon and Alpha Scaling Rule

Before choosing `epsilon` and `alpha`, first record the clean input scale over
the index or batch being attacked:

```text
n_points
input_rms_mean = mean_i sqrt(mean_j x_ij^2)
input_rms_std
input_mean_abs_mean
input_max_abs_mean
```

Then choose the desired per-point perturbation scale. The preferred rule is:

```text
per_point_delta_rms = relative_rms * input_rms_mean
```

For example:

```text
relative_rms = 0.10  means average pointwise perturbation RMS is 10% of input RMS
relative_rms = 0.20  means average pointwise perturbation RMS is 20% of input RMS
```

Convert this per-point target into an input-p budget using the dense-perturbation
approximation:

```text
epsilon_p = per_point_delta_rms * n_points^(1/p),   finite p
epsilon_inf = per_point_delta_rms,                  p = inf
```

Important special cases:

```text
p = 2:
  epsilon = per_point_delta_rms * sqrt(n_points)

p = inf:
  epsilon = target per-point max/absolute scale
```

Choose `alpha` by deciding how many steps should be needed to reach the budget:

```text
alpha = epsilon / reach_epsilon_steps
```

For 100-step attacks, reasonable first choices are:

```text
reach_epsilon_steps = 25    aggressive
reach_epsilon_steps = 50    slower
```

Use the helper script before launching a new p/q sweep:

```bash
cd /workspace/NeuralOperatorRobustness2

adv_robust/bin/python recommend_attack_scale.py \
  --input_p 2 \
  --indices 0 \
  --relative_rms 0.10 \
  --reach_epsilon_steps 25
```

For a batch-style estimate:

```bash
adv_robust/bin/python recommend_attack_scale.py \
  --input_p 2 \
  --indices 0 1 2 3 4 5 6 7 8 9 \
  --relative_rms 0.20 \
  --reach_epsilon_steps 25
```

The output gives the input statistics, desired per-point perturbation, and
recommended `epsilon`/`alpha`. Record this output with the experiment notes.

## Base Losses

Run all three base losses:

```text
loss1 = ||f(x + delta) - f(x)||_q
loss2 = ||f(x + delta) - g(x)||_q
loss3 = ||f(x + delta) - g(x + delta)||_q
```

For `loss3`, use the same solver/frame convention throughout a given comparison
so that optimizer differences are not confounded with solver-gradient choices.

## Objective Variants

Run all three objective variants:

```text
original:
  L_i(delta)

increment_ratio:
  (L_i(delta) - L_i(0)) / (||delta||_p + eta)

regularized:
  L_i(delta) - C * ||delta||_p
```

For round 1, do not sweep `eta` or `C`. Use one fixed value for each:

```text
eta = 1e-6
C   = 1.0
```

`eta` must be positive. Do not use `eta=0` for the ratio objective, because the
run can start at `delta=0`, which makes the denominator zero and can create
`inf`, `nan`, or a zero-gradient dead start.

`C=1.0` is only a first-round placeholder. Later rounds should calibrate or
sweep `C`.

## Optimizers

Run all three optimizers:

```text
pgd
lp_steepest_pgd
power_iteration
```

For the main experiment, `power_iteration` should use:

```text
power_variant = objective_gradient
```

This makes the main comparison cleaner: all three optimizers are driven by
automatic differentiation of the selected scalar objective. The difference is
then the update geometry, not whether one method uses explicit matrix-free
JVP/VJP while another uses direct objective gradients.

Matrix-free variants such as `pure_jvp_vjp`, `affine_jvp_vjp`, and
`standard_l2` should be treated as a separate operator-gain comparison, not
mixed into the first main grid.

## Random Start Policy

Random starts are part of the experimental definition and must be recorded.

Use this first-round policy:

```text
original:
  pgd:              random_start = false
  lp_steepest_pgd:  random_start = false
  power_iteration:  no PGD-style random start; it initializes a random direction

increment_ratio:
  pgd:              random_start = true, eta > 0
  lp_steepest_pgd:  random_start = true, eta > 0
  power_iteration:  eta > 0; avoid taking an objective-gradient step at radius 0

regularized:
  pgd:              random_start = true
  lp_steepest_pgd:  random_start = true
  power_iteration:  prefer a nonzero radius/objective-gradient start
```

Reasoning:

- `increment_ratio` is unsafe at `delta=0` if `eta=0`, and even with a guard it
  can produce a dead first step. Use positive `eta` and random start for PGD
  style methods.
- `regularized` contains `||delta||_p`, which is nonsmooth at zero for common
  norms. A random start avoids an ambiguous zero start.
- `original` is the cleanest baseline, so keep PGD-style methods deterministic
  from `delta=0` in round 1.

If implementation details make `power_iteration + objective_gradient` evaluate
the objective at radius 0 before updating, set its first effective radius to a
small positive value or use a fixed positive radius schedule for the ratio and
regularized objectives.

## Round 1: Minimal Main Grid

Round 1 fixes:

```text
input_p = 2
output_q = 2
eta = 1e-6
C = 1.0
epsilon = 0.01
alpha = 0.005
steps = 100
seed = 0
```

Sweep:

```text
3 base losses
x 3 objective variants
x 3 optimizers
= 27 runs
```

Each of the 27 runs must record:

- the selected optimized objective value;
- all three base losses `loss1`, `loss2`, `loss3`;
- all three baseline-subtracted increments;
- all three ratio objectives using the fixed `eta`;
- all three regularized objectives using the fixed `C`;
- perturbation norms, especially `delta_norm_p`;
- gradient norms and update diagnostics;
- final summary values for all nine objective evaluations.

## Round 2: p/q Geometry Sweep

After round 1 is stable, keep `epsilon`, `alpha`, `steps`, `seed`, `eta`, and
`C` fixed. Sweep input/output norm geometry.

Recommended main p/q pairs:

```text
(input_p=2,   output_q=2)
(input_p=inf, output_q=2)
(input_p=2,   output_q=inf)
(input_p=inf, output_q=inf)
```

Optional sparse-input extension:

```text
(input_p=1, output_q=2)
(input_p=1, output_q=inf)
```

Round 2 should still record the full 3 x 3 evaluation table for every run.

## Round 3: eta and C Sensitivity

Only after round 1 and round 2 are stable, sweep objective hyperparameters.

For `increment_ratio`:

```text
eta in {1e-8, 1e-6, 1e-4}
```

For `regularized`, start with:

```text
C in {0.01, 0.1, 1.0, 10.0}
```

If the scale is poor, estimate a typical gain from round 1:

```text
g = loss_increment_q / delta_norm_p
```

Then run a calibrated sweep:

```text
C in {0.1g, 0.5g, 1.0g, 2.0g}
```

Do not sweep `epsilon`, `alpha`, or `seed` in this plan.

## Output Organization

Use output directory names that encode the swept axes:

```text
results/three_loss_objective_round1/
  burgers_loss1_original_pgd_p2_q2_eta1e-6_C1_eps0.01_alpha0.005_seed0/
  burgers_loss1_original_lp_steepest_pgd_p2_q2_eta1e-6_C1_eps0.01_alpha0.005_seed0/
  burgers_loss1_original_power_iteration_objective_gradient_p2_q2_eta1e-6_C1_eps0.01_alpha0.005_seed0/
  ...
```

For round 2 and round 3:

```text
results/three_loss_objective_pq_sweep/
results/three_loss_objective_eta_c_sweep/
```

## Implementation Notes

Before running the full grid, confirm the script supports:

1. `attack_method=power_iteration` in the grid script, or an explicit alias from
   `generalized_power` to `power_iteration`.
2. `power_variant=objective_gradient` for the main power-style optimizer.
3. positive `eta` for all `increment_ratio` runs.
4. random starts according to the policy above.
5. full 3 x 3 objective evaluation logging for every run, not only the active
   objective.

The first implementation check should be a 27-run smoke version with
`steps=3`. After that succeeds and output columns look correct, rerun round 1
with `steps=100`.
