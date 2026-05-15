# FNO nu=0.001 Loss-Gradient Direction Along Attack Path Plan

Date: 2026-05-15

## Question

We clarified that the top singular-vector directions from the local
Jacobian/SVD experiment are not the same object as one-step loss-gradient
directions during an attack.

The new question is:

> For FNO `nu=0.001`, as `delta_k` grows during an attack, do the one-step
> gradient directions of `loss1`, `loss2`, and `loss3` change, and do they
> become more or less aligned with each other and with the previously computed
> SVD directions?

This experiment is meant to test the mechanism directly along an actual attack
trajectory.

## Background

Use the same FNO/Burgers setting as the main mechanism experiment:

```text
model: FNO
dataset/task: 1D Burgers
nu: 0.001
nx: 1024
t_final: 1.0
dt: 0.001
domain: 2.0
input norm p: 2
output norm q: 2
epsilon: 8.0
alpha: 0.3
steps: 100
```

The existing FNO-vs-solver Jacobian/SVD comparison already computed, for sample
indices `0, 7, 40, 47, 115`:

```text
J_f = model Jacobian
J_j = solver Jacobian
J_e = J_f - J_j
v_f = top right singular directions of J_f
v_j = top right singular directions of J_j
v_e = top right singular directions of J_e
```

Those SVD directions answer a local-map question:

```text
Which input directions maximize ||J_f v||, ||J_j v||, or ||J_e v||?
```

This new experiment answers a different optimizer question:

```text
At the current attack iterate delta_k, which direction would each loss use for
its next gradient step?
```

## Local Gradient Formulas

Let:

```text
b = f(x) - j(x)
J_e = J_f - J_j
```

For squared L2 local objectives:

```text
loss1_sq(delta) ~= ||J_f delta||^2
g1(delta) = J_f^T J_f delta

loss2_sq(delta) ~= ||b + J_f delta||^2
g2(delta) = J_f^T b + J_f^T J_f delta

loss3_sq(delta) ~= ||b + J_e delta||^2
g3(delta) = J_e^T b + J_e^T J_e delta
```

At `delta=0`:

```text
g1(0) = 0
g2(0) = J_f^T b
g3(0) = J_e^T b
```

Therefore, when `delta_k` is tiny, `loss2` and `loss3` should be dominated by
their clean-residual outward terms.  As `||delta_k||` grows, the quadratic terms
`J_f^T J_f delta_k` and `J_e^T J_e delta_k` can become more important.

This is the effect we want to measure.

## Hypotheses

1. Early in the attack, `g2(delta_k)` and `g3(delta_k)` can differ strongly from
   the SVD directions because the `J^T b` outward terms dominate.
2. As `delta_k` grows, the quadratic terms can become larger, so `g1`, `g2`,
   and `g3` may rotate toward the corresponding quadratic/SVD structures.
3. The direction changes are not guaranteed to be monotone.  Actual PGD uses
   projection, finite step size, and nonlinear model/solver evaluations.
4. The clean-Jacobian prediction should be most accurate near small `delta_k`;
   if exact autograd gradients drift away from the clean-Jacobian gradients at
   later steps, that is evidence of local-to-global/path effects.

## Experimental Design

### Samples

Use the same five initial-condition indices as the local SVD comparison:

```text
0, 7, 40, 47, 115
```

### Attack Trajectories

Save attack trajectories with:

```text
--save_trajectory
--save_every 1
```

The existing attack script saves `trajectory.npz` with:

```text
k
x_clean
model_output_clean
solver_output_clean
x_adv
delta
model_output_adv
solver_output_adv
difference_adv
```

Primary trajectory set:

```text
loss1_original_pgd
loss2_original_pgd
loss3_original_pgd
```

Use a shared tiny random start for all three runs to avoid the squared-L2
`loss1` zero-gradient degeneracy:

```text
--random_start
--random_start_scale 1e-6
--seed 0
```

The clean-point diagnostic should still be computed separately from
`delta=0`, where `g1(0)=0`, `g2(0)=J_f^T b`, and `g3(0)=J_e^T b`.

### Steps To Report

Compute metrics for all saved steps, but make the tables and figures emphasize:

```text
k = 0, 1, 5, 10, 20, 30, 40, 50, 75, 100
```

The user's main requested checkpoints are:

```text
k = 10, 20, 30, 40, 50
```

### Two Gradient Families To Compute

#### A. Clean-Jacobian Squared-L2 Local Gradients

Use the saved explicit `1024 x 1024` Jacobians from:

```text
forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/
```

For each sample, trajectory, and saved `delta_k`, compute:

```text
g1_lin(k) = J_f^T J_f delta_k
g2_lin(k) = J_f^T b + J_f^T J_f delta_k
g3_lin(k) = J_e^T b + J_e^T J_e delta_k
```

These are the directions that directly connect to the previous SVD experiment.

Also store the decomposition:

```text
loss2 linear term:     a2 = J_f^T b
loss2 quadratic term:  q2(k) = J_f^T J_f delta_k

loss3 linear term:     a3 = J_e^T b
loss3 quadratic term:  q3(k) = J_e^T J_e delta_k
```

Record:

```text
||q2(k)|| / ||a2||
cos(a2, q2(k))
||q3(k)|| / ||a3||
cos(a3, q3(k))
```

These ratios directly test whether increasing `||delta_k||` makes the quadratic
part dominate.

#### B. Exact Autograd Gradients Of The Actual Objectives

For selected steps, also compute true autograd gradients at `x + delta_k`:

```text
grad_exact_loss1(k) = grad_delta ||f(x+delta_k)-f(x)||
grad_exact_loss2(k) = grad_delta ||f(x+delta_k)-j(x)||
grad_exact_loss3(k) = grad_delta ||f(x+delta_k)-j(x+delta_k)||
```

This measures the actual one-step direction used by the real attack objective.

Compare exact gradients to the clean-Jacobian local predictions:

```text
cos(grad_exact_loss1, g1_lin)
cos(grad_exact_loss2, g2_lin)
cos(grad_exact_loss3, g3_lin)
```

This separates clean local-linear behavior from nonlinear path effects.

## Metrics

For each sample, trajectory source, and step `k`, record:

### Direction Cosines

```text
cos(g1, g2)
cos(g1, g3)
cos(g2, g3)

cos(g1, v_f_top1)
cos(g2, v_f_top1)
cos(g2, v_e_top1)
cos(g3, v_f_top1)
cos(g3, v_e_top1)
```

Also record top-8 subspace projection norms:

```text
||Proj_{V_f,top8}(g_i)|| / ||g_i||
||Proj_{V_e,top8}(g_i)|| / ||g_i||
```

### Norm And Budget

```text
step k
||delta_k||_2
||delta_k||_2 / epsilon
||g1||
||g2||
||g3||
```

### Linear-vs-Quadratic Decomposition

```text
||J_f^T J_f delta_k|| / ||J_f^T b||
cos(J_f^T b, J_f^T J_f delta_k)

||J_e^T J_e delta_k|| / ||J_e^T b||
cos(J_e^T b, J_e^T J_e delta_k)
```

### Frequency Metrics

For normalized `g1`, `g2`, and `g3`, compute the same frequency diagnostics used
in the SVD plots:

```text
hi16, hi32, hi64, hi128, hi256
zero crossings
rough1
rough2
```

This will show whether gradient directions become smoother or more jagged as
the attack proceeds.

## Plots

Make figures with no standard-deviation shading.  Use individual sample lines
plus a clearly labeled mean line if needed.

Suggested plots:

1. `cos(g1,g2)`, `cos(g1,g3)`, `cos(g2,g3)` versus step.
2. `cos(g_i, v_f_top1)` and `cos(g_i, v_e_top1)` versus step.
3. Top-8 subspace projection of each gradient into `V_f` and `V_e` versus step.
4. Linear/quadratic norm ratio versus step for `loss2` and `loss3`.
5. Selected-step line plots of normalized `g1`, `g2`, `g3` for
   `k = 0, 10, 20, 30, 40, 50, 100`.
6. Fourier spectra of `g1`, `g2`, `g3` at selected steps.

## Expected Output Files

Proposed output root:

```text
forensics/fno_nu0p001_loss_gradient_path_diagnostics_20260515/
```

Files:

```text
gradient_direction_by_step.csv
gradient_direction_selected_steps.csv
linear_quadratic_decomposition.csv
svd_alignment_by_step.csv
frequency_metrics_by_step.csv
exact_vs_clean_jacobian_gradient_cosines.csv
summary.md
plots/
```

Suggested script:

```text
tools/analyze_fno_loss_gradient_path_directions.py
```

## Example Trajectory Command

For one sample and one loss:

```bash
python run_three_loss_objective_attack.py \
  --case burgers \
  --loss_type loss3 \
  --objective_variant original \
  --attack_method pgd \
  --norm 2 \
  --input_p 2 \
  --output_q 2 \
  --epsilon 8.0 \
  --alpha 0.3 \
  --steps 100 \
  --index 0 \
  --burgers-nu 0.001 \
  --burgers-t-final 1.0 \
  --burgers-dt 0.001 \
  --burgers-domain 2.0 \
  --random_start \
  --random_start_scale 1e-6 \
  --seed 0 \
  --save_trajectory \
  --save_every 1 \
  --output_dir results/fno_nu0p001_loss_gradient_path_20260515/index_000/loss3_original_pgd
```

Repeat for:

```text
loss_type = loss1, loss2, loss3
index = 0, 7, 40, 47, 115
```

## How This Proves Or Refutes The Claim

The claim is supported if:

1. At small `k`, `g2` and `g3` are dominated by `J^T b` and are far from the SVD
   directions.
2. As `||delta_k||` grows, the quadratic ratios
   `||J^T J delta_k|| / ||J^T b||` increase.
3. The cosine curves show gradient directions rotating over attack time.
4. Exact autograd gradients agree with clean-Jacobian predictions early, then
   drift later if nonlinear path effects become important.

The claim is not supported if:

1. `g1`, `g2`, and `g3` are already aligned from the first nonzero step.
2. The linear/quadratic ratios do not change with `k`.
3. The gradient directions remain essentially constant across the attack path.

Either result is useful: it tells us whether the difference between SVD
directions and one-step loss gradients is a small technicality or a major
mechanism in the FNO `nu=0.001` attack.
