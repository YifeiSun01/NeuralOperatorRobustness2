# FNO nu=0.001 Outward-Growth Direction Summary

Observed from this run:

- Output directory: `forensics/outward_growth_direction_20260515/fno_nu0p001`
- Jacobian source root: `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed`
- Indices: `[0, 7, 40, 47, 115]`
- Top-k singular directions recorded: `8`
- Random directions per sample: `64`
- Finite-difference radii: `[0.0001, 0.001, 0.01]`

This experiment adds the missing Experiment 2 row:

```text
v_growth = normalize((J_f - J_j)^T (b / ||b||_2))
```


## Conclusion

Observed from this completed FNO `nu=0.001` run:

- All five samples `0, 7, 40, 47, 115` completed and none had a degenerate
  clean-residual outward direction.
- `outward_growth` has outward component mean `0.166514`, much larger than
  `error_top` mean `0.0140571`, `fno_top` mean `0.00218766`, `solver_top` mean
  `-0.00819025`, and `random_best_by_outward_component` mean `0.0114951`.
- `error_top` has the larger mismatch-gain mean, `0.412169`, while
  `outward_growth` has mismatch-gain mean `0.368053`.
- Finite-difference checks match the linear outward prediction: for
  `outward_growth`, actual loss3 growth mean is `0.167286` at `rho=1e-4` and
  `0.166668` at `rho=1e-3`, versus predicted `0.166514`; the negative control
  gives approximately the opposite sign.

Inference from these observations:

- The missing Experiment 2 row is now completed for FNO `nu=0.001`.
- The top residual-Lipschitz direction `v_e*` and the clean-risk outward
  direction `v_growth* = normalize(A^T b/||b||)` are not the same diagnostic:
  one maximizes residual movement `||A v||`, while the other maximizes first-order
  increase of the current error norm.
- This supports the planned distinction between local mismatch movement, local
  outward risk growth, and finite-radius `loss3_original` endpoint attack.


## Plain-Language Interpretation

中文解释：这个结果主要说明三件事。

1. `error_top` / `v_e*` 和 `outward_growth` / `A^T b` 不是同一个东西。
   `error_top` 让误差场移动得最多，也就是 `||A v||` 最大；但是它不一定让
   当前误差范数 `||e(x)||` 立刻变大。这里 `error_top` 的 mismatch gain mean
   是 `0.412169`，比 `outward_growth` 的 `0.368053` 更大，但它的 outward
   component mean 只有 `0.0140571`。

2. `outward_growth` 找到的是“把当前 clean residual 往外推”的方向。它的
   outward component mean 是 `0.166514`，远大于 `fno_top`、`solver_top`、
   `error_top` 和 random-best directions。这说明 `A^T b` 这个方向确实补上了
   Experiment 2 里缺的 local outward risk growth 诊断。

3. 这个局部解释是可信的，因为 small-radius finite-difference 对上了线性预测。
   对 `outward_growth`，预测增长是 `0.166514`，实际增长在 `rho=1e-4` 是
   `0.167286`，在 `rho=1e-3` 是 `0.166668`；负方向给出相反符号。

因此，这个实验不是证明 `outward_growth` 是最终 finite-radius attack 的最优方向；
它证明的是：在局部一阶意义下，“误差场移动最大”和“当前误差范数增长最快”是两个
不同概念。这个结果支持论文主线：ratio / Lipschitz / local diagnostic 只能解释局部机制，
而真正有限半径攻击还是要看 `loss3_original` 的 endpoint error。




## Reset: What This Experiment Actually Ran

This experiment should be described narrowly.

It ran a **local residual-geometry experiment** for

\[
A=J_f-J_j,\qquad b=f(x)-j(x).
\]

The two main local questions were:

1. Pure residual-field movement:

\[
\max_{\|v\|_2=1}\|Av\|_2.
\]

This gives the top residual-movement / error-Jacobian singular direction. It asks:

> Which input direction moves the residual field \(e=f-j\) the most?

2. Clean-residual outward growth:

\[
\max_{\|v\|_2=1}
\left\langle \frac{b}{\|b\|_2}, Av\right\rangle.
\]

This gives

\[
v_{out}=\frac{A^Tb}{\|A^Tb\|_2}
\]

when \(A^Tb\neq0\). It asks:

> Which input direction immediately pushes the current clean residual \(b\) outward,
> increasing \(\|b+A\delta\|_2\) to first order?

So the core comparison was:

\[
\text{make }\|Av\|_2\text{ large}
\quad\text{versus}\quad
\text{make }\|b+A\delta\|_2\text{ grow immediately}.
\]

This is **not** a native `loss1` experiment. `loss1` is

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|,
\]

and its local Jacobian is \(J_f\), not \(J_f-J_j\). Therefore formulas involving
\(J_j\), \(J_f-J_j\), \(A\), or \(b=f-j\) should not be described as `loss1` angle
formulas.

The conceptual connection to `loss1` versus `loss3` is only this:

- `loss1`-style objectives reward large model movement, locally related to
  \(\|J_f\delta\|\);
- `loss3_original` rewards large endpoint residual error,
  \(\|f(x+\delta)-j(x+\delta)\|\), locally related to \(\|b+(J_f-J_j)\delta\|\);
- therefore, a direction that makes a movement term large need not be the direction
  that makes the endpoint residual error large.

But the existing outward-growth numbers directly support the narrower claim about
`loss3` residual movement versus `loss3` endpoint outward growth. They do **not** by
themselves compute pairwise gradient angles among native `loss1`, `loss2`, and `loss3`.

The direct experiment for native objective-direction differences would be:

\[
\angle(\nabla_\delta L_1(\delta_k),\nabla_\delta L_2(\delta_k)),\quad
\angle(\nabla_\delta L_1(\delta_k),\nabla_\delta L_3(\delta_k)),\quad
\angle(\nabla_\delta L_2(\delta_k),\nabla_\delta L_3(\delta_k))
\]

at the same saved \(\delta_k\) points. That has not been run in the current angle
records.


## Correction: This Is Not An Apples-To-Apples Optimizer Comparison

Important correction: this experiment should **not** be described as a fair comparison
between the optimizer direction of two objectives.

The user's concern is correct. The current comparison mixes two different levels:

1. `error_top` is a top singular-vector direction of

   \[
   A=J_f-J_j.
   \]

   It is the constrained optimum / power-iteration limit for the pure local residual
   movement problem

   \[
   \max_{\|v\|_2=1}\|Av\|_2.
   \]

   Equivalently, for the squared version, it is the final/global direction of

   \[
   \max_{\|\delta\|_2\le\varepsilon}\|A\delta\|_2^2.
   \]

2. `outward_growth` is the clean-point first-order direction of

   \[
   \|b+A\delta\|_2
   \]

   at \(\delta=0\). For the squared local endpoint objective,

   \[
   \nabla_\delta \|b+A\delta\|_2^2\big|_{\delta=0}=2A^Tb.
   \]

   So this is a first-step / infinitesimal-radius direction, not the finite-radius
   optimizer of \(\|b+A\delta\|_2\).

Therefore, the comparison

\[
error\_top \quad \text{versus} \quad outward\_growth
\]

is **not** comparing two final optimizer directions, and it is **not** comparing two
same-iterate gradients. It compares two local diagnostic directions from different
questions:

- residual field movement: \(\|Av\|\);
- clean residual outward growth at zero radius:
  \(\langle b/\|b\|,Av\rangle\).

This comparison is still useful as a mechanism diagnostic: it shows that a direction
can move the residual field a lot without moving it outward along the current residual.
But it should not be used to claim that one optimizer's final direction beats another
optimizer's final direction.

The apples-to-apples comparisons would be separate experiments:

### Same-iterate gradient comparison

Choose the same saved \(\delta_k\) and compare

\[
g_{endpoint}(\delta_k)=A^Tb+A^TA\delta_k
\]

with

\[
g_{movement}(\delta_k)=A^TA\delta_k.
\]

This directly asks how much the clean residual term \(A^Tb\) bends the next gradient
step at the same current perturbation.

### Same local finite-radius optimizer comparison

Solve both finite-radius local affine problems:

\[
\max_{\|\delta\|_2\le\varepsilon}\|A\delta\|_2^2
\]

and

\[
\max_{\|\delta\|_2\le\varepsilon}\|b+A\delta\|_2^2.
\]

Then compare their final \(\delta^*\) directions. This would compare final directions
within the same local affine level.

### Same nonlinear optimizer comparison

Run the actual nonlinear objectives from the same starts and compare final deltas and
final endpoint losses. This is the level relevant to finite-radius attacks.

So the corrected claim is narrow:

> The current outward-growth experiment validates a local zero-radius mechanism:
> top residual movement and first-order outward clean-error growth are different
> diagnostics. It does not by itself compare final optimizer directions or same-step
> gradient directions.

## Layered Interpretation: What Is Actually Being Compared

The main risk in interpreting this experiment is mixing several different levels of
analysis. They use similar symbols, but they are not the same mathematical object.

### Layer 0: The True Nonlinear Attack Objective

The real finite-radius regression attack is

\[
\max_{\|\delta\|_2\le \varepsilon}
\mathcal{L}_{true}(\delta)
=
\max_{\|\delta\|_2\le \varepsilon}
\|f(x+\delta)-j(x+\delta)\|_2.
\]

This is the actual `loss3_original` attack. In general it has no closed-form analytic
optimizer because both the model and solver/oracle response can change nonlinearly
along the path from \(x\) to \(x+\delta\). PGD and LP-steepest PGD are iterative methods
for this level.

This outward-growth experiment does **not** solve this level and does **not** claim to
identify the final nonlinear attack direction.

### Layer 1: The Local Affine Approximation

At a clean input \(x\), define

\[
b=e(x)=f(x)-j(x)
\]

and

\[
A=J_f(x)-J_j(x).
\]

For small perturbations,

\[
e(x+\delta)=f(x+\delta)-j(x+\delta)
\approx b+A\delta.
\]

This is the local model used by the outward-growth experiment. It is a mechanism
diagnostic, not the full finite-radius nonlinear attack.

### Layer 2: Three Different Local Questions

Once the local model is fixed, there are still multiple different questions.

#### 2.1 Pure Residual Movement

\[
M(\delta)=\|A\delta\|_2.
\]

This asks:

> How far does the error field move?

The squared version is

\[
M_{sq}(\delta)=\|A\delta\|_2^2=\delta^TA^TA\delta.
\]

The gradient at a current perturbation is

\[
\nabla_\delta M_{sq}(\delta)=2A^TA\delta.
\]

The constrained L2 optimum of the pure movement problem

\[
\max_{\|\delta\|_2\le \varepsilon}\|A\delta\|_2^2
\]

is

\[
\delta^*=\varepsilon v_1,
\]

where \(v_1\) is the top right singular vector of \(A\). This is the `error_top` / SVD
mechanism. But one gradient step from an arbitrary \(\delta_k\) is
\(A^TA\delta_k\), not automatically \(v_1\).

#### 2.2 Endpoint Error Under The Local Model

\[
S(\delta)=\|b+A\delta\|_2.
\]

This asks:

> How large is the final residual vector after perturbation?

The squared version expands as

\[
S_{sq}(\delta)=\|b+A\delta\|_2^2
=\|b\|_2^2+2b^TA\delta+\delta^TA^TA\delta.
\]

Its gradient is

\[
\nabla_\delta S_{sq}(\delta)=2A^Tb+2A^TA\delta.
\]

At \(\delta=0\), this becomes

\[
\nabla_\delta S_{sq}(0)=2A^Tb.
\]

So \(A^Tb\) is the clean-point first-order direction of endpoint error growth. It is
not the same thing as the top singular vector of \(A\), and it is not generally the
finite-radius optimizer of \(\|b+A\delta\|\).

#### 2.3 Same-Delta Gradient Difference

If one wants to compare the two local squared objectives at the same current
perturbation \(\delta_k\), the correct comparison is

\[
g_{endpoint}(\delta_k)=A^Tb+A^TA\delta_k
\]

versus

\[
g_{movement}(\delta_k)=A^TA\delta_k.
\]

Their difference is exactly

\[
g_{endpoint}(\delta_k)-g_{movement}(\delta_k)=A^Tb.
\]

This is a different diagnostic from the one run here. It would require loading actual
saved attack iterates \(\delta_k\) and measuring whether \(A^Tb\) is large enough to bend
the optimizer step away from the pure residual-movement step.

### Layer 3: Finite-Radius Local Affine Endpoint Problem

Even inside the local affine model, the finite-radius endpoint problem

\[
\max_{\|\delta\|_2\le\varepsilon}\|b+A\delta\|_2^2
\]

is not simply solved by either \(A^Tb\) or the top singular vector of \(A\).

Let

\[
Q=A^TA,
\quad
c=A^Tb.
\]

Then the problem is

\[
\max_{\|\delta\|_2\le\varepsilon}
\delta^TQ\delta+2c^T\delta.
\]

A boundary KKT stationary point satisfies

\[
Q\delta+c=\mu\delta.
\]

Equivalently, on a regular branch,

\[
\delta(\mu)=(\mu I-Q)^{-1}c,
\]

with \(\mu\) chosen so that

\[
\|\delta(\mu)\|_2=\varepsilon.
\]

This shows the finite-radius local affine endpoint direction mixes two forces:

\[
A^Tb
\]

and

\[
A^TA\delta.
\]

For very small radius, the linear term \(A^Tb\) dominates. As the radius grows, the
quadratic term \(A^TA\delta\) can pull the direction toward singular-vector structure.

### Layer 4: What This Experiment Actually Compared

This run compared **candidate directions**, not full optimizer trajectories.

The candidate directions were:

1. `error_top`: saved top right singular directions of \(A=J_f-J_j\). These are the
   local pure residual-movement directions for \(\|A\delta\|\).

2. `outward_growth`: the normalized clean-point outward direction

   \[
   v_{out}=\frac{A^Tb}{\|A^Tb\|_2}.
   \]

3. `fno_top`: top right singular directions of \(J_f\).

4. `solver_top`: top right singular directions of \(J_j\).

5. random and random-best directions.

For each direction \(v\), the run measured

\[
\text{mismatch gain}(v)=\|Av\|_2
\]

and

\[
\text{outward component}(v)=
\left\langle \frac{b}{\|b\|_2},Av\right\rangle.
\]

The important observed data are:

| direction group | mismatch gain mean | outward component mean | meaning |
|---|---:|---:|---|
| `error_top` | `0.412169` | `0.0140571` | moves the error field most, but barely pushes current residual outward |
| `outward_growth` | `0.368053` | `0.166514` | moves the error field slightly less, but strongly pushes current residual outward |
| `fno_top` | `0.337848` | `0.00218766` | model-sensitive direction is almost neutral for clean-error growth |
| `solver_top` | `0.340392` | `-0.00819025` | solver-sensitive direction is slightly inward on average |
| `random_best_by_outward_component` | `0.047458` | `0.0114951` | random search did not find comparable outward growth |

The key comparison is therefore

\[
0.412169 > 0.368053
\]

but

\[
0.0140571 \ll 0.166514.
\]

That means the direction that maximizes residual movement is not the direction that
maximizes local outward growth of the current residual norm.

### Layer 5: The Finite-Difference Validation

The experiment then checked whether the local outward-growth prediction is visible in
the actual nonlinear objective at very small radii. For \(\delta=\rho v\), it measured

\[
\frac{\|e(x+\rho v)\|_2-\|e(x)\|_2}{\rho}.
\]

For `outward_growth`, the predicted value was

\[
0.166514.
\]

Observed nonlinear finite differences were:

\[
0.167286\quad (\rho=10^{-4})
\]

and

\[
0.166668\quad (\rho=10^{-3}).
\]

The negative direction gave the opposite sign:

\[
-0.165528\quad (\rho=10^{-4}),
\]

versus predicted

\[
-0.166514.
\]

Observed from these values: the local \(A^Tb\) outward-growth interpretation is not just
symbolic algebra; it matches small-radius nonlinear measurements.

### Layer 6: What This Does And Does Not Prove

This experiment does prove, for the observed FNO `nu=0.001` samples:

1. The pure residual-movement direction and the clean-residual outward-growth direction
   are different local diagnostics.

2. A direction can have large \(\|Av\|\) while having small
   \(\langle b/\|b\|,Av\rangle\). In words: the error field can move a lot without
   strongly increasing the current error norm.

3. The clean residual term \(b\) matters for the local behavior of `loss3_original`.
   Removing it turns the objective into a different diagnostic.

This experiment does **not** prove:

1. That `outward_growth` is the final finite-radius optimizer of
   `loss3_original`.

2. That `error_top` is the per-step gradient direction of the pure movement objective
   from every \(\delta_k\). It is the SVD/global constrained direction for the pure local
   quadratic problem.

3. That the same-\(\delta_k\) gradients

   \[
   A^Tb+A^TA\delta_k
   \]

   and

   \[
   A^TA\delta_k
   \]

   have already been compared. That is a valid follow-up, but it is not what this run
   measured.

### Layer 7: Why This Matters For `loss3_original`

The local version of `loss3_original` is

\[
\|b+A\delta\|_2.
\]

The residual-increment / pure movement diagnostic is

\[
\|A\delta\|_2.
\]

The difference is the clean residual \(b\). Dropping \(b\) changes the question from

> how large is the final residual?

into

> how much did the residual vector move?

Those are not equivalent. The observed data show exactly this: `error_top` moves the
residual vector more, but `outward_growth` increases the current residual norm much
more at first order.

The correct role assignment is therefore:

| Object | Correct role |
|---|---|
| `loss3_original` / \(\|f(x+\delta)-j(x+\delta)\|\) | primary finite-radius regression attack objective |
| \(\|b+A\delta\|\) | local affine approximation to `loss3_original` |
| \(A^Tb\) | clean-point first-order outward-growth direction |
| \(\|A\delta\|\) | local residual-movement / mismatch diagnostic |
| top right singular vector of \(A\) | optimum of pure local residual movement, not final nonlinear attack |
| \(A^Tb+A^TA\delta_k\) vs \(A^TA\delta_k\) | same-iterate gradient comparison, not yet measured in this run |

### Layer 8: Natural Follow-Up If We Want The Same-Delta Gradient Question

To answer the question about optimizer-step differences directly, use saved attack
iterates \(\delta_k\) and compute

\[
\cos\left(A^Tb+A^TA\delta_k,
A^TA\delta_k\right),
\]

\[
\frac{\|A^Tb\|_2}{\|A^TA\delta_k\|_2},
\]

and

\[
\cos\left(A^Tb,
A^TA\delta_k\right).
\]

This would tell us whether, at each actual current perturbation \(\delta_k\), the clean
residual term \(A^Tb\) still significantly bends the endpoint-gradient direction away
from the pure residual-movement gradient.

That follow-up is a same-trajectory gradient diagnostic. The current outward-growth
experiment is a local candidate-direction diagnostic.




## Same-Delta Gradient Diagnostic Results

After the apples-to-apples concern, a new post-processing diagnostic was run using
existing saved trajectories and existing Jacobians. No attack or Jacobian recomputation
was needed.

Script:

`tools/analyze_same_delta_local_gradient_angles.py`

Outputs:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_diagnostics.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

For each saved attack iterate \(\delta_k\), it compares the two same-\(\delta_k\) local
squared-gradient directions:

\[
g_{endpoint}(\delta_k)=A^Tb+A^TA\delta_k
\]

and

\[
g_{movement}(\delta_k)=A^TA\delta_k.
\]

It records

\[
\cos(g_{endpoint},g_{movement}),
\]

\[
\angle(g_{endpoint},g_{movement}),
\]

and

\[
\frac{\|A^Tb\|_2}{\|A^TA\delta_k\|_2}.
\]

This is the fair same-current-perturbation comparison that the earlier candidate
direction diagnostic did not provide.

### Key Observed Values

Rows are averaged over the five samples `0, 7, 40, 47, 115`. The `k=0` rows have tiny
initial perturbations and nearly zero \(A^TA\delta_k\), so the more meaningful comparison
starts from `k=5`.

| attack trajectory | k | budget ratio mean | cos endpoint vs movement | angle | \(\|A^Tb\|/\|A^TA\delta_k\|\) |
|---|---:|---:|---:|---:|---:|
| `loss1_original_pgd` | 5 | `0.3226` | `0.999347` | `1.95 deg` | `0.0347` |
| `loss1_original_pgd` | 50 | `1.0000` | `0.999915` | `0.71 deg` | `0.0125` |
| `loss2_original_pgd` | 5 | `0.3171` | `0.999345` | `1.97 deg` | `0.0354` |
| `loss2_original_pgd` | 50 | `1.0000` | `0.999926` | `0.67 deg` | `0.0121` |
| `loss3_original_pgd` | 5 | `0.0359` | `0.930281` | `21.07 deg` | `0.8561` |
| `loss3_original_pgd` | 10 | `0.0924` | `0.978184` | `11.28 deg` | `0.3738` |
| `loss3_original_pgd` | 25 | `0.3649` | `0.996287` | `4.22 deg` | `0.1175` |
| `loss3_original_pgd` | 50 | `0.7637` | `0.999478` | `1.40 deg` | `0.0321` |

### Interpretation

Observed from the same-\(\delta_k\) diagnostic:

1. Along the `loss1` and `loss2` trajectories, \(A^TA\delta_k\) is already much larger
   than \(A^Tb\) by `k=5`, and the endpoint-gradient and movement-gradient directions
   are almost identical. Their angles are about `2 deg` at `k=5` and below `1 deg` by
   `k=50`.

2. Along the `loss3` trajectory, the perturbation is much smaller early on. At `k=5`,
   the ratio \(\|A^Tb\|/\|A^TA\delta_k\|\) is `0.8561`, and the endpoint-vs-movement
   angle is `21.07 deg`. So the clean residual term substantially bends the local
   endpoint-gradient direction early in the `loss3` path.

3. As \(\delta_k\) grows along the `loss3` path, \(A^TA\delta_k\) becomes dominant and
   the two gradients align: the angle falls to `11.28 deg` at `k=10`, `4.22 deg` at
   `k=25`, and `1.40 deg` at `k=50`.

Inference: the fair same-\(\delta_k\) comparison says the clean residual term \(A^Tb\)
matters most in the small-radius / early-trajectory regime. Once the perturbation is
large, the quadratic movement term \(A^TA\delta_k\) dominates the local squared-gradient
geometry. This is consistent with the layered interpretation: \(A^Tb\) is an important
zero-radius and early-step term, not a guarantee of a different final finite-radius
optimizer direction.

This diagnostic is still local-affine and squared-L2. It does not replace direct
nonlinear `loss3_original` evaluation; it explains how the local gradient components
behave along saved attack trajectories.

## Existing Angle Records And Whether A Rerun Is Needed

The original outward-growth run did record some angle information, but not the exact
same-\(\delta_k\) gradient angles discussed later.

### Already Recorded

The file

`forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`

records, for each sample, angles between `outward_growth` and the saved top singular
directions of:

- \(J_f\), recorded as `fno`;
- \(J_j\), recorded as `solver`;
- \(A=J_f-J_j\), recorded as `error`.

It includes columns:

- `dot`
- `abs_dot`
- `angle_deg`
- `subspace_projection_l2`
- `max_abs_dot`
- `mean_abs_dot`

Observed summary from that file:

| comparison with `outward_growth` | rows | dot mean | abs dot mean | angle mean |
|---|---:|---:|---:|---:|
| `error` top-8 singular directions | 40 | `0.085246` | `0.246550` | `84.68 deg` |
| `error` rank-1 singular direction only | 5 | `-0.015058` | `0.125291` | `90.87 deg` |
| `fno` top-8 singular directions | 40 | `0.013602` | `0.134222` | `89.20 deg` |
| `fno` rank-1 singular direction only | 5 | `0.032308` | `0.157988` | `88.14 deg` |
| `solver` top-8 singular directions | 40 | `-0.046311` | `0.138733` | `92.72 deg` |
| `solver` rank-1 singular direction only | 5 | `0.033007` | `0.149638` | `88.09 deg` |

Subspace summaries:

| subspace | rows | projection mean | max abs dot mean | mean abs dot mean |
|---|---:|---:|---:|---:|
| `error_top8_subspace` | 5 | `0.925912` | `0.654526` | `0.246550` |
| `fno_top8_subspace` | 5 | `0.465453` | `0.325827` | `0.134222` |
| `solver_top8_subspace` | 5 | `0.493495` | `0.358428` | `0.138733` |

Interpretation: `outward_growth` is nearly orthogonal to the rank-1 error singular
direction on average, but it lies strongly inside the top-8 error-Jacobian subspace.
That means \(A^Tb\) is not the dominant singular vector itself; it is more like a
combination of several error-Jacobian singular directions.

### Not Recorded Yet

The original run did **not** record the same-iterate gradient angle

\[
\cos\left(A^Tb+A^TA\delta_k,\ A^TA\delta_k\right).
\]

It also did not record

\[
\frac{\|A^Tb\|_2}{\|A^TA\delta_k\|_2}
\]

or

\[
\cos\left(A^Tb,\ A^TA\delta_k\right)
\]

for saved attack iterates \(\delta_k\).

### Does The Experiment Need To Be Rerun?

A full rerun of the outward-growth experiment is not necessary for the existing
candidate-direction angle data; those angles already exist in
`all_direction_similarity_table.csv`.

However, the more precise apples-to-apples question needs a new post-processing
diagnostic, not a full attack rerun. The repository already has saved attack
trajectories at

`results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`

and the Jacobians are available under

`forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/error/*_jacobian_svd.npz`.

So the right next step is to reuse those saved \(\delta_k\) values and compute the
same-\(\delta_k\) gradient diagnostics. That would answer the question about whether the
clean residual term \(A^Tb\) actually bends the optimizer step along real attack
trajectories.

## Are These Number Comparisons Valid?

中文澄清：这些数能比，但只能在同一个指标列里比，不能跨指标乱比。

### What Each Number Actually Is

For every unit input direction \(v\), the script computes two main local quantities:

\[
\text{mismatch gain}(v)=\|Av\|_2,
\]

and

\[
\text{outward component}(v)=
\left\langle \frac{b}{\|b\|_2},Av\right\rangle.
\]

They are not the same thing.

- `mismatch_gain` measures how far the error/residual field moves per unit input
  direction.
- `outward_component` measures the signed first-order growth of the current clean
  error norm along that direction.

So the valid comparison is:

1. Compare `mismatch_gain` to `mismatch_gain`.
2. Compare `outward_component` to `outward_component`.
3. Do not interpret a larger `mismatch_gain` as automatically meaning a larger
   `outward_component`.

### What The Reported Means Average Over

The aggregate rows have different counts:

- `error_top`: 40 rows = 5 samples times top-8 right singular directions of
  \(A=J_f-J_j\).
- `fno_top`: 40 rows = 5 samples times top-8 right singular directions of \(J_f\).
- `solver_top`: 40 rows = 5 samples times top-8 right singular directions of \(J_j\).
- `outward_growth`: 5 rows = one \(A^Tb\) direction per sample.

Therefore `error_top mismatch gain mean = 0.412169` is a top-8 group mean, not the
rank-1-only mean.

A rank-1-only check from `all_direction_response_table.csv` gives:

| direction set | rows | mismatch gain mean | outward component mean |
|---|---:|---:|---:|
| `error` top-8 group | 40 | `0.412169` | `0.0140571` |
| `error` rank-1 only | 5 | `0.868217` | `-0.00677836` |
| `outward_growth` | 5 | `0.368053` | `0.166514` |
| `fno` top-8 group | 40 | `0.337848` | `0.00218766` |
| `fno` rank-1 only | 5 | `0.674167` | `0.00326582` |
| `solver` top-8 group | 40 | `0.340392` | `-0.00819025` |
| `solver` rank-1 only | 5 | `0.680574` | `0.00304161` |

This makes the interpretation stronger, not weaker: the rank-1 error-Jacobian singular
direction moves the residual field even more, but its mean outward component is slightly
negative. So it is very much a residual-movement direction, not an immediate clean-error
norm growth direction.

### Which Comparison Is Correct

Correct comparison 1:

\[
0.412169 > 0.368053
\]

within `mismatch_gain` means the top-8 `error_top` directions move the error field more
than the `outward_growth` direction on average.

Correct comparison 2:

\[
0.166514 \gg 0.0140571
\]

within `outward_component` means `outward_growth` pushes the current residual norm
outward much more than the top-8 `error_top` directions.

Even stronger, for rank-1 only:

\[
0.868217 \gg 0.368053
\]

for mismatch movement, but

\[
-0.00677836 < 0.166514
\]

for outward growth.

So yes, the comparison is valid if stated this way:

> error/SVD directions are better for moving the residual field; the \(A^Tb\) direction
> is better for first-order growth of the current clean residual norm.

### What Would Be An Incorrect Interpretation

It would be incorrect to say:

> `outward_growth` is the final finite-radius attack optimum.

It would also be incorrect to say:

> this experiment already compared the two gradients
> \(A^Tb+A^TA\delta_k\) and \(A^TA\delta_k\) at the same saved attack iterate.

That same-\(\delta_k\) gradient comparison is a separate follow-up.

The current result supports the local distinction:

\[
\text{residual movement } \|Av\|
\neq
\text{outward clean-error growth }
\left\langle \frac{b}{\|b\|},Av\right\rangle.
\]

## Metric Glossary For The Four Key Numbers

For a unit input direction `v`, with `A = J_f - J_j` and clean residual
`b = e(x) = f(x)-j(x)`, the two reported quantities are:

\[
\text{mismatch gain}(v)=\|Av\|_2
\]

\[
\text{outward component}(v)=\left\langle \frac{b}{\|b\|_2}, Av \right\rangle
\]

The four headline numbers mean:

| number | row / column | mathematical meaning | interpretation |
|---:|---|---|---|
| `0.412169` | `error_top` mismatch gain mean | mean `||A v||_2` over the top-8 error-Jacobian singular directions across 5 samples | residual/error field movement is largest along `error_top` directions |
| `0.368053` | `outward_growth` mismatch gain mean | mean `||A v_growth||_2` across 5 samples | the outward-growth direction still moves the residual field substantially, but not as much as `error_top` |
| `0.0140571` | `error_top` outward component mean | mean `<b/||b||, A v>` over the top-8 error-Jacobian singular directions across 5 samples | `error_top` moves the residual field, but only weakly in the direction that increases the current error norm |
| `0.166514` | `outward_growth` outward component mean | mean `<b/||b||, A v_growth>` across 5 samples | `outward_growth = normalize(A^T b)` is the local direction that strongly pushes the current residual outward |

So `0.412169 > 0.368053` says `error_top` has larger raw residual movement,
but `0.166514 >> 0.0140571` says `outward_growth` has much larger local
clean-error growth.


## Where The Evidence Shows \(b+A\delta\) And \(A\delta\) Are Different

The intended comparison here is **not** native `loss1` if `loss1` means
\(\|f(x+\delta)-f(x)\|\). Native `loss1` linearizes to \(\|J_f\delta\|\), not
\(\|A\delta\|\).

The comparison this outward-growth experiment actually proves is:

\[
L_{endpoint}(\delta)
=\|f(x+\delta)-j(x+\delta)\|_2
=\|e(x+\delta)\|_2
\approx
\|b+A\delta\|_2,
\]

versus

\[
L_{movement}(\delta)
=\|e(x+\delta)-e(x)\|_2
\approx
\|A\delta\|_2.
\]

Here

\[
b=e(x)=f(x)-j(x),\qquad A=J_f-J_j.
\]

So the endpoint loss keeps the clean residual / bias term \(b\), while the residual
movement loss subtracts the clean residual away and only sees \(A\delta\). The point of
this experiment is to show that keeping or removing this constant residual term changes
the local direction.

The evidence is in three places.

### 1. Direction-response table

Observed from
`forensics/outward_growth_direction_20260515/fno_nu0p001/aggregate_direction_response_summary.csv`
and `all_direction_response_table.csv`:

| direction set | what it optimizes locally | mismatch gain mean \(\|Av\|\) | outward component mean \(\langle b/\|b\|,Av\rangle\) |
|---|---|---:|---:|
| `error_top` top-8 | large residual movement \(\|Av\|\) | `0.412169` | `0.0140571` |
| `error` rank-1 only | strongest singular direction of \(A\) | `0.868217` | `-0.00677836` |
| `outward_growth` | first-order growth of \(\|b+A\delta\|\) | `0.368053` | `0.166514` |

This is the cleanest numerical proof of the distinction:

- `error_top` / singular-vector directions make \(\|Av\|\) large.
- But they barely increase the current residual norm; the top-8 outward component is
  only `0.0140571`, and the rank-1 outward component is even slightly negative.
- `outward_growth = normalize(A^T b)` has a smaller raw movement gain than `error_top`,
  but its outward component is `0.166514`, much larger.

So the direction that is good for \(\|A\delta\|\) is not the direction that is good for
immediately increasing \(\|b+A\delta\|\).

### 2. Direction-angle table

Observed from
`forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`:

- `outward_growth` vs `error` top-8 angle mean: `84.68 deg`.
- `outward_growth` vs `error` rank-1 angle mean: `90.87 deg`.

This says the clean-point outward direction \(A^Tb\) is nearly orthogonal to the pure
residual-movement / singular-vector direction of \(A\). That is exactly the geometric
claim: the direction for \(\|b+A\delta\|\) and the direction for \(\|A\delta\|\) are not
the same.

### 3. Finite-difference validation

Observed from
`forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_summary.csv`:

| direction source | rho | actual loss3 growth mean | predicted growth mean |
|---|---:|---:|---:|
| `outward_growth` | `1e-4` | `0.167286` | `0.166514` |
| `outward_growth` | `1e-3` | `0.166668` | `0.166514` |
| `negative_outward_growth` | `1e-4` | `-0.165528` | `-0.166514` |
| `negative_outward_growth` | `1e-3` | `-0.166365` | `-0.166514` |

This confirms that the \(A^Tb\) direction is not just algebra: at small finite radius it
actually increases \(\|b+A\delta\|\) at the predicted rate, while the negative direction
reduces it.

### Relation To The Same-Point Nonlinear Gradient Table

The later same-point nonlinear endpoint-vs-movement table measures a related finite-point
question:

\[
\angle\left(
\nabla_z\|f(z)-j(z)\|_2,
\nabla_z\|(f(z)-j(z))-(f(x)-j(x))\|_2
\right),\qquad z=x+\delta_k.
\]

On the `loss3_original_pgd` path, the observed mean angles are:

| k | true nonlinear endpoint-vs-movement angle |
|---:|---:|
| 5 | `45.06 deg` |
| 10 | `36.36 deg` |
| 25 | `22.33 deg` |
| 50 | `8.15 deg` |

This means the two directions are very different early and in the middle of the path,
then become more similar later as the accumulated residual increment becomes large
relative to the clean residual \(b\). This is the finite-point version of the same idea:
keeping \(b\) versus subtracting \(b\) changes the gradient direction, especially before
\(A\delta\) dominates.


## Squared-Loss Gradient Clarification

For the local model

\[
e(x+\delta) \approx b + A\delta,
\]

there are two related but different local objectives.

Residual movement objective:

\[
M(\delta)=\|A\delta\|_2^2=\delta^T A^T A\delta.
\]

Its gradient is

\[
\nabla_\delta M(\delta)=2A^T A\delta.
\]

This objective measures how much the error field moves, independent of the
current residual direction `b`.

Squared endpoint error objective:

\[
S(\delta)=\|b+A\delta\|_2^2.
\]

Expanding exactly gives

\[
S(\delta)=\|b\|_2^2+2b^T A\delta+\delta^T A^T A\delta.
\]

Its exact local-model gradient is

\[
\nabla_\delta S(\delta)=2A^T b+2A^T A\delta.
\]

At `delta=0`, this becomes

\[
\nabla_\delta S(0)=2A^T b.
\]

So `A^T b` is not obtained by optimizing the residual movement objective. It is
the first-step gradient of the squared endpoint error objective. Dropping the
quadratic term is only a first-order, infinitesimal-radius diagnostic. For a
finite-radius attack, the quadratic term and nonlinear higher-order terms should
not be ignored; this is why the main finite-radius objective remains
`loss3_original`.

## Direction Summary

| direction group | n | mismatch gain mean | outward component mean | D_f mean |
|---|---:|---:|---:|---:|
| fno_top | 40 | 0.337848 | 0.00218766 | 0.191833 |
| solver_top | 40 | 0.340392 | -0.00819025 | 0.192255 |
| error_top | 40 | 0.412169 | 0.0140571 | 0.329056 |
| outward_growth | 5 | 0.368053 | 0.166514 | 0.355948 |
| negative_outward_growth | 5 | 0.368053 | -0.166514 | 0.355948 |
| random | 320 | 0.0453593 | -0.000143471 | 0.258463 |
| random_best_by_outward_component | 5 | 0.047458 | 0.0114951 | 0.284052 |
| random_best_by_mismatch_gain | 5 | 0.0760439 | 0.00173444 | 0.248026 |

## Finite-Difference Summary

| direction source | rho | n | actual loss3 growth mean | predicted growth mean | residual movement mean |
|---|---:|---:|---:|---:|---:|
| error | 1.0e-04 | 40 | 0.014074 | 0.0140571 | 0.412753 |
| error | 1.0e-03 | 40 | 0.0141318 | 0.0140571 | 0.412213 |
| error | 1.0e-02 | 40 | 0.0148188 | 0.0140571 | 0.412405 |
| fno | 1.0e-04 | 40 | 0.00209312 | 0.00218766 | 0.338168 |
| fno | 1.0e-03 | 40 | 0.00224155 | 0.00218766 | 0.337836 |
| fno | 1.0e-02 | 40 | 0.00236288 | 0.00218766 | 0.337758 |
| negative_outward_growth | 1.0e-04 | 5 | -0.165528 | -0.166514 | 0.367704 |
| negative_outward_growth | 1.0e-03 | 5 | -0.166365 | -0.166514 | 0.367903 |
| negative_outward_growth | 1.0e-02 | 5 | -0.165476 | -0.166514 | 0.366964 |
| outward_growth | 1.0e-04 | 5 | 0.167286 | 0.166514 | 0.368765 |
| outward_growth | 1.0e-03 | 5 | 0.166668 | 0.166514 | 0.368181 |
| outward_growth | 1.0e-02 | 5 | 0.167548 | 0.166514 | 0.369125 |
| random_best_by_mismatch_gain | 1.0e-04 | 5 | 0.00222016 | 0.00173444 | 0.0771404 |
| random_best_by_mismatch_gain | 1.0e-03 | 5 | 0.00167079 | 0.00173444 | 0.0761168 |
| random_best_by_mismatch_gain | 1.0e-02 | 5 | 0.00177817 | 0.00173444 | 0.0760449 |
| random_best_by_outward_component | 1.0e-04 | 5 | 0.0129368 | 0.0114951 | 0.0499933 |
| random_best_by_outward_component | 1.0e-03 | 5 | 0.0115535 | 0.0114951 | 0.0475233 |
| random_best_by_outward_component | 1.0e-02 | 5 | 0.0115446 | 0.0114951 | 0.0474824 |
| solver | 1.0e-04 | 40 | -0.00801186 | -0.00819025 | 0.340758 |
| solver | 1.0e-03 | 40 | -0.00812273 | -0.00819025 | 0.340372 |
| solver | 1.0e-02 | 40 | -0.00798748 | -0.00819025 | 0.339857 |

## Files

- `config.json`
- `manifest.json`
- `all_direction_response_table.csv`
- `aggregate_direction_response_summary.csv`
- `all_direction_similarity_table.csv`
- `finite_difference_growth_table.csv`
- `finite_difference_growth_summary.csv`
- `index_*/clean_model_output.npy`
- `index_*/clean_solver_output.npy`
- `index_*/clean_residual.npy`
- `index_*/outward_growth_direction.npy`

Observed source paths:
- `/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`
- `/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_000/error/error_index0_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_000/fno/fno_index0_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_000/solver/solver_index0_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_007/error/error_index7_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_007/fno/fno_index7_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_007/solver/solver_index7_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_040/error/error_index40_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_040/fno/fno_index40_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_040/solver/solver_index40_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_047/error/error_index47_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_047/fno/fno_index47_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_047/solver/solver_index47_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_115/error/error_index115_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_115/fno/fno_index115_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_115/solver/solver_index115_jacobian_svd.npz`


## Why This Supports `loss3_original` Over Pure Residual-Movement Objectives

This experiment should be read as local evidence for the importance of the clean
residual bias term in `loss3_original`.

Locally,

\[
loss3\_original(\delta)
=
\|f(x+\delta)-j(x+\delta)\|_2
\approx
\|b+A\delta\|_2,
\]

where

\[
b=f(x)-j(x),\quad A=J_f(x)-J_j(x).
\]

By contrast, the pure residual-movement / residual-increment objective is locally

\[
\|e(x+\delta)-e(x)\|_2
\approx
\|A\delta\|_2.
\]

The second objective removes the clean residual `b`. It asks only how far the error
field moves. The first objective keeps `b`. It asks whether the final residual vector
is large after the perturbation.

For squared local objectives,

\[
\|b+A\delta\|_2^2
=
\|b\|_2^2+2b^TA\delta+\delta^TA^TA\delta.
\]

Therefore the clean-point first-order gradient is

\[
\nabla_\delta \|b+A\delta\|_2^2\big|_{\delta=0}=2A^Tb.
\]

But the pure movement objective is

\[
\|A\delta\|_2^2=\delta^TA^TA\delta,
\]

with gradient

\[
2A^TA\delta.
\]

This explains why the top singular direction of `A` can be excellent for making the
error field move while still being poor at increasing the current clean error norm.
It optimizes the quadratic movement term, not the outward component from `b`.

Observed in this run:

- `error_top` had larger mismatch gain, mean `||A v|| = 0.412169`.
- But its outward component was only `0.0140571`.
- `outward_growth` had smaller mismatch gain, mean `||A v|| = 0.368053`.
- But its outward component was much larger, `0.166514`.

So `error_top` mostly moves the residual field sideways/rotationally relative to the
current residual `b`, while `outward_growth` pushes the current residual outward.

The finite-difference check confirms that this is not just algebraic notation: the
predicted outward growth `0.166514` matched the measured small-radius growth of
`loss3_original`, with observed values `0.167286` at `rho=1e-4` and `0.166668` at
`rho=1e-3`.

Inference: this supports the claim that `loss3_original` should not be replaced by a
pure residual-increment objective. The residual-increment/SVD objective is useful for
local mismatch diagnostics, but it drops the clean residual direction that determines
first-order growth of the current error norm. For finite-radius attacks, the full
nonlinear `loss3_original` still needs iterative optimization.


## Direction Type Clarification: Local Direction Versus Final Optimizer Direction

The outward-growth experiment compares local analytic directions, not final nonlinear
attack trajectories.

The compared objects are:

1. Pure residual-movement direction:

\[
v_{move}=\arg\max_{\|v\|_2=1}\|Av\|_2.
\]

For the local L2 model, this has an analytic solution: it is the top right singular
vector of `A`, equivalently the top eigenvector of \(A^TA\). This is the global solution
of

\[
\max_{\|\delta\|_2\le\varepsilon}\|A\delta\|_2^2,
\]

namely

\[
\delta^*=\varepsilon v_1,
\]

where \(v_1\) is the top right singular vector of \(A\).

2. Clean-residual outward-growth direction at zero radius:

\[
v_{out}=\arg\max_{\|v\|_2=1}
\left\langle \frac{b}{\|b\|_2},Av\right\rangle.
\]

For the local L2 first-order objective, this also has an analytic solution:

\[
v_{out}=\frac{A^Tb}{\|A^Tb\|_2},
\]

when \(A^Tb\neq0\). This is only the first-order direction at \(\delta=0\), not the
finite-radius optimum of \(\|b+A\delta\|_2\).

3. Finite-radius affine endpoint objective:

\[
\max_{\|\delta\|_2\le\varepsilon}\|b+A\delta\|_2^2.
\]

Expanding gives

\[
\|b+A\delta\|_2^2
=\|b\|_2^2+2b^TA\delta+\delta^TA^TA\delta.
\]

Let

\[
Q=A^TA,\quad c=A^Tb.
\]

The finite-radius local affine problem becomes

\[
\max_{\|\delta\|_2\le\varepsilon}
\delta^TQ\delta+2c^T\delta.
\]

Its KKT stationarity condition on the boundary is

\[
Q\delta+c=\mu\delta.
\]

In the generic case this can be written as

\[
\delta(\mu)=(\mu I-Q)^{-1}c,
\]

with \(\mu\) chosen so that

\[
\|\delta(\mu)\|_2=\varepsilon,
\quad \mu>\lambda_{max}(Q).
\]

So the finite-radius affine endpoint problem has an implicit analytic characterization
if the full matrix \(A\) is available, but it is not simply the top singular vector and
not simply \(A^Tb\). For very small \(\varepsilon\), the linear term dominates and the
solution approaches the \(A^Tb\) direction. For larger \(\varepsilon\), the quadratic
term \(A^TA\delta\) can bend the solution toward the dominant singular-vector
structure.

For the actual neural-operator attack objective

\[
\max_{\|\delta\|_2\le\varepsilon}
\|f(x+\delta)-j(x+\delta)\|_2,
\]

there is no closed-form analytic solution in general because \(f\), \(j\), and the
solver response are nonlinear along the path. PGD / LP-steepest PGD are iterative
methods for this finite nonlinear objective.

Therefore this experiment should be interpreted as comparing local mechanism
directions:

- `error_top` = analytic optimum of pure local residual movement \(\|A\delta\|\);
- `outward_growth` = analytic first-order direction for increasing current clean
  residual norm \(\|b+A\delta\|\) at \(\delta=0\).

It does not claim that either one is the final direction of a full nonlinear attack.


## Important Limitation: Not A Same-Delta Gradient Comparison

A later clarification is important: this experiment does **not** compare the two
objective gradients at the same finite perturbation \(\delta_k\).

For the squared local affine objectives, the gradients would be:

\[
g_{endpoint}(\delta)
=\nabla_\delta \|b+A\delta\|_2^2
=2A^Tb+2A^TA\delta,
\]

and

\[
g_{movement}(\delta)
=\nabla_\delta \|A\delta\|_2^2
=2A^TA\delta.
\]

At the same \(\delta\), the difference is exactly the clean-residual term:

\[
g_{endpoint}(\delta)-g_{movement}(\delta)=2A^Tb.
\]

That is the direct same-\(\delta\) gradient comparison. It would require choosing or
loading actual \(\delta_k\) values and then comparing the cosines, norms, and relative
sizes of \(A^Tb\) and \(A^TA\delta_k\).

The current outward-growth experiment did something different:

- `error_top` used the top right singular directions of \(A\), which solve the pure
  residual-movement problem \(\max_{\|v\|_2=1}\|Av\|_2\).
- `outward_growth` used the clean-point first-order direction \(A^Tb\), which solves
  \(\max_{\|v\|_2=1}\langle b/\|b\|,Av\rangle\) at \(\delta=0\).

So the experiment compares two local mechanism directions, not two per-step optimizer
gradients along the same trajectory.

This distinction matters because even for the pure quadratic movement objective,
one gradient step from a finite \(\delta_k\) is

\[
A^TA\delta_k,
\]

not automatically the top singular vector. The top singular vector is the global
constrained optimum / power-iteration limit of the pure quadratic objective, not the
one-step gradient direction from every possible \(\delta_k\).

A more direct follow-up diagnostic would compute, for saved attack iterates
\(\delta_k\):

\[
\cos\left(A^Tb+A^TA\delta_k,\ A^TA\delta_k\right),
\]

\[
\frac{\|A^Tb\|_2}{\|A^TA\delta_k\|_2},
\]

and

\[
\cos\left(A^Tb,\ A^TA\delta_k\right).
\]

That would answer the narrower question: at the same current perturbation, how much
does the clean residual term \(A^Tb\) bend the optimizer step away from the pure
residual-movement step?


## True Nonlinear Same-Point Gradient Comparison

This follow-up addresses the finite-\(\delta\) concern directly. The previous
same-\(\delta_k\) diagnostic compared

\[
A^Tb + A^TA\delta_k
\quad \text{versus} \quad
A^TA\delta_k,
\]

which is still a clean-point local-affine calculation. That calculation is useful for
understanding the terms in the Taylor model, but it can understate or misrepresent the
gradient difference when \(\delta_k\) is not small.

The new diagnostic does **not** use a fixed clean-point matrix \(A\). For each saved
trajectory point, it sets

\[
z_k = x+\delta_k
\]

and directly differentiates the true nonlinear objectives

\[
L_{\mathrm{endpoint}}(z_k)
=
\|f(z_k)-j(z_k)\|_2
\]

and

\[
L_{\mathrm{movement}}(z_k)
=
\|(f(z_k)-j(z_k))-(f(x)-j(x))\|_2.
\]

The compared vectors are therefore

\[
\nabla_{z_k}L_{\mathrm{endpoint}}(z_k)
\quad \text{and} \quad
\nabla_{z_k}L_{\mathrm{movement}}(z_k).
\]

Since \(z_k=x+\delta_k\), these are also the gradients with respect to the current
perturbation \(\delta_k\). This is an apples-to-apples gradient comparison at the same
finite point. It is still a **local gradient at the current point**, not a closed-form
solution for the final finite-radius optimizer.

Observed from:

- Script: `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`
- Input trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Output directory:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/`
- Main summary:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`

Selected observed means over indices `0, 7, 40, 47, 115`:

| trajectory | k | angle between nonlinear gradients | cosine | mean delta budget ratio |
|---|---:|---:|---:|---:|
| `loss1_original_pgd` | 5 | `13.48 deg` | `0.9374` | `0.3226` |
| `loss1_original_pgd` | 25 | `1.48 deg` | `0.9990` | `0.9916` |
| `loss1_original_pgd` | 50 | `0.55 deg` | `0.9999` | `1.0000` |
| `loss2_original_pgd` | 5 | `37.33 deg` | `0.7207` | `0.3171` |
| `loss2_original_pgd` | 25 | `1.67 deg` | `0.9989` | `0.9987` |
| `loss2_original_pgd` | 50 | `0.82 deg` | `0.9998` | `1.0000` |
| `loss3_original_pgd` | 5 | `45.06 deg` | `0.6933` | `0.0359` |
| `loss3_original_pgd` | 10 | `36.36 deg` | `0.7673` | `0.0924` |
| `loss3_original_pgd` | 25 | `22.33 deg` | `0.8758` | `0.3649` |
| `loss3_original_pgd` | 50 | `8.15 deg` | `0.9710` | `0.7637` |

Observed interpretation:

- Along the `loss1_original_pgd` and `loss2_original_pgd` trajectories, the true
  nonlinear endpoint gradient and true nonlinear residual-movement gradient become
  almost aligned once the perturbation is near the boundary.
- Along the `loss3_original_pgd` trajectory, they remain substantially different for
  longer: about `45 deg` at `k=5`, `36 deg` at `k=10`, `22 deg` at `k=25`, and still
  about `8 deg` at `k=50`.
- This is a stronger and fairer answer to the finite-\(\delta\) question than the
  fixed-\(A\) diagnostic, because it differentiates the actual model and solver at
  \(x+\delta_k\).

Inference:

- The user concern was correct: when \(\delta_k\) is not tiny, the fixed clean-point
  \(A=J_f(x)-J_j(x)\) comparison is only a local-affine approximation.
- The true nonlinear result shows that the endpoint objective
  \(\|f(z)-j(z)\|_2\) and the residual-movement objective
  \(\|(f(z)-j(z))-(f(x)-j(x))\|_2\) can have meaningfully different optimizer
  gradients at the same finite point, especially along the `loss3_original` path.
- Therefore the cleanest layered statement is:

\[
\text{local residual movement}
\neq
\text{local clean-error outward growth}
\neq
\text{true finite-point nonlinear endpoint gradient}
\neq
\text{final finite-radius attack direction}.
\]

中文总结：你这个质疑是对的。只要 \(\delta\) 已经不是无穷小，就不能只拿
clean 点的 \(A\) 去解释一切。真正公平的比较应该直接在当前
\(z_k=x+\delta_k\) 上对两个真实 nonlinear loss 求梯度。这个新结果说明，
至少在 `loss3_original_pgd` 路径上，endpoint error 的梯度和 residual movement
的梯度确实不是同一个方向；早期和中期差得很明显。它不是在证明某个方向就是最终全局最优，
而是在证明：有限半径路径上，真实 endpoint objective 的下一步优化方向不能简单等同于
“误差场移动最大”的方向。


## Analytic-Solution Hierarchy Note

A standalone version of the analytic-solution clarification is recorded in
`docs/analytic_solution_hierarchy_for_delta_objectives_20260516.md`.

The short version is:

\[
\max_{\|\delta\|\le\epsilon}\|A\delta\|^2
\]

has the simple top-right-singular-vector solution;

\[
\max_{\|\delta\|\le\epsilon}\|b+A\delta\|^2
\]

has a KKT / implicit local-affine solution satisfying

\[
A^TA\delta + A^Tb = \mu\delta;
\]

and the true nonlinear attack

\[
\max_{\|\delta\|\le\epsilon}\|f(x+\delta)-j(x+\delta)\|
\]

generally has no closed-form analytic solution and must be studied through iterative
optimization, actual trajectories, and final deltas.


## Angle Diagnostic Summary: Which Comparisons Are Valid

This section summarizes the angle diagnostics and separates the valid interpretation of
each one. The main source of confusion is that the experiments computed angles between
objects that live at different levels:

1. local candidate directions at clean point \(x\);
2. same-\(\delta_k\) gradients inside the clean-point local-affine approximation;
3. same-point true nonlinear gradients at \(z_k=x+\delta_k\).

These should not be described as the same comparison.

### Angle Set 1: Local Candidate Directions

Compared directions:

\[
v_{out}=\frac{A^Tb}{\|A^Tb\|_2}
\]

versus top right singular directions of

\[
A=J_f(x)-J_j(x).
\]

Observed angles:

- `outward_growth` vs `error` top-8 angle mean: `84.68 deg`.
- `outward_growth` vs `error` rank-1 angle mean: `90.87 deg`.

Valid interpretation:

- This is valid for the question: are the clean-point outward-growth direction and the
  pure residual-movement/SVD direction different local diagnostics?
- It shows that \(A^Tb\) and the top singular-vector direction of \(A\) are nearly
  orthogonal on average in this FNO `nu=0.001` sample set.

Invalid interpretation:

- This is **not** a comparison of two optimizer final directions.
- This is **not** a comparison of two gradients at the same finite \(\delta_k\).
- It should not be used to claim that one full nonlinear attack optimizer is better
  than another.

### Angle Set 2: Same-Delta Local-Affine Gradient Angles

Compared directions at saved attack iterates \(\delta_k\):

\[
g_{endpoint}^{local}(\delta_k)=A^Tb+A^TA\delta_k
\]

versus

\[
g_{movement}^{local}(\delta_k)=A^TA\delta_k.
\]

These are the squared-objective gradients inside the clean-point local-affine model,
up to the irrelevant factor of 2.

Observed mean angles:

| trajectory | k=5 | k=10 | k=25 | k=50 |
|---|---:|---:|---:|---:|
| `loss1_original_pgd` | `1.95 deg` | not highlighted | not highlighted | `0.71 deg` |
| `loss2_original_pgd` | `1.97 deg` | not highlighted | not highlighted | `0.67 deg` |
| `loss3_original_pgd` | `21.07 deg` | `11.28 deg` | `4.22 deg` | `1.40 deg` |

Valid interpretation:

- This is the right comparison for the **local-affine same-iterate** question:
  inside the approximation \(e(x+\delta)\approx b+A\delta\), how much does the
  clean residual term \(A^Tb\) bend the gradient away from pure residual movement?
- It shows that, in the clean-point local-affine model, the \(A^Tb\) term matters
  most early on the `loss3_original_pgd` trajectory and becomes less important as
  \(A^TA\delta_k\) grows.

Limitation:

- This still freezes \(A\) at the clean point \(x\).
- If \(\delta_k\) is finite and the model/solver response is nonlinear along the path,
  this can be a poor proxy for the true gradient at \(x+\delta_k\).
- Therefore this comparison is reasonable as a Taylor-model diagnostic, but it is not
  the final answer for finite-radius attack geometry.

### Angle Set 3: True Nonlinear Same-Point Gradient Angles

Compared directions at the same finite point

\[
z_k=x+\delta_k
\]

using actual autograd through model and solver:

\[
\nabla_{z_k}\|f(z_k)-j(z_k)\|_2
\]

versus

\[
\nabla_{z_k}\|(f(z_k)-j(z_k))-(f(x)-j(x))\|_2.
\]

Observed mean angles:

| trajectory | k=5 | k=10 | k=25 | k=50 |
|---|---:|---:|---:|---:|
| `loss1_original_pgd` | `13.48 deg` | `7.91 deg` | `1.48 deg` | `0.55 deg` |
| `loss2_original_pgd` | `37.33 deg` | `6.30 deg` | `1.67 deg` | `0.82 deg` |
| `loss3_original_pgd` | `45.06 deg` | `36.36 deg` | `22.33 deg` | `8.15 deg` |

Valid interpretation:

- This is the fairest angle comparison for finite saved trajectory points because it
  does not pretend that \(\delta_k\) is infinitesimal and does not rely on the fixed
  clean-point matrix \(A\).
- It directly answers: at the same actual finite point, do the endpoint-error loss and
  residual-movement loss produce the same next gradient direction?
- For `loss3_original_pgd`, the answer is clearly no for early/mid steps: `45.06 deg`
  at `k=5`, `36.36 deg` at `k=10`, and `22.33 deg` at `k=25`.

Comparison to Angle Set 2:

- The local-affine same-delta angles were much smaller than the true nonlinear angles,
  especially on the `loss3_original_pgd` path.
- Example for `loss3_original_pgd`:

| k | local-affine angle | true nonlinear angle |
|---:|---:|---:|
| 5 | `21.07 deg` | `45.06 deg` |
| 10 | `11.28 deg` | `36.36 deg` |
| 25 | `4.22 deg` | `22.33 deg` |
| 50 | `1.40 deg` | `8.15 deg` |

Inference:

- The user's concern is correct: using \(A^Tb+A^TA\delta_k\) is only a local-affine
  approximation. It is not wrong as a Taylor-model diagnostic, but it is not strong
  enough when \(\delta_k\) is not tiny.
- For finite-radius attack interpretation, the true nonlinear same-point gradient
  comparison should be preferred.
- The clean final hierarchy is:

\[
\text{candidate local direction angle}
\neq
\text{same-}\delta_k\text{ local-affine gradient angle}
\neq
\text{same-point true nonlinear gradient angle}
\neq
\text{final optimizer direction comparison}.
\]

Plain-language conclusion:

- Angle Set 1 was useful but easy to overinterpret.
- Angle Set 2 was a better same-iterate comparison, but still tied to the clean-point
  matrix \(A\).
- Angle Set 3 is the most relevant for the user's finite-\(\delta\) concern.
- Therefore, yes: for large or non-infinitesimal \(\delta\), the \(A^Tb+A^TA\delta_k\)
  comparison can be misleadingly optimistic about gradient alignment. It should be
  described as a local-affine approximation, not as the true finite-radius geometry.


## Additional Key Takeaways From The Nonlinear Gradient Run

This section records the most useful extra observations from the true nonlinear
same-point gradient diagnostic. These are observed from
`forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`.

### 1. The largest gradient-angle gap appears early and mid-path

For `loss3_original_pgd`, the true nonlinear endpoint-vs-movement gradient angle is
large early:

| k | mean delta budget ratio | true nonlinear angle |
|---:|---:|---:|
| 5 | `0.0359` | `45.06 deg` |
| 10 | `0.0924` | `36.36 deg` |
| 25 | `0.3649` | `22.33 deg` |
| 50 | `0.7637` | `8.15 deg` |

Observed evidence:

- At small and mid-sized radii, endpoint error and residual movement produce very
  different local gradient directions.
- As the perturbation grows, the angle shrinks, but it is still not zero at `k=50`.

Inference:

- Early in the attack, the clean residual and nonlinear solver/model response strongly
  affect the endpoint gradient.
- Later, the accumulated residual increment becomes large, so the endpoint objective
  and movement objective become more aligned, but they are still not identical.

### 2. Linearized gradients understate the real nonlinear angle gap

For `loss3_original_pgd`, the clean-point local-affine angle is much smaller than the
true nonlinear angle:

| k | local-affine angle | true nonlinear angle |
|---:|---:|---:|
| 5 | `21.07 deg` | `45.06 deg` |
| 10 | `11.28 deg` | `36.36 deg` |
| 25 | `4.22 deg` | `22.33 deg` |
| 50 | `1.40 deg` | `8.15 deg` |

Observed evidence:

- The local-affine model says the endpoint and movement gradients are already fairly
  close after early steps.
- Direct nonlinear autograd says they are much farther apart.

Inference:

- The fixed clean-point matrix \(A=J_f(x)-J_j(x)\) is too optimistic about gradient
  alignment once \(\delta_k\) is finite.
- This is exactly the distinction between a Taylor/local-affine diagnostic and the
  true finite-point nonlinear gradient.

### 3. Loss1 and loss2 trajectories align much faster than loss3

For true nonlinear same-point gradients:

| trajectory | k=5 angle | k=25 angle | k=50 angle |
|---|---:|---:|---:|
| `loss1_original_pgd` | `13.48 deg` | `1.48 deg` | `0.55 deg` |
| `loss2_original_pgd` | `37.33 deg` | `1.67 deg` | `0.82 deg` |
| `loss3_original_pgd` | `45.06 deg` | `22.33 deg` | `8.15 deg` |

Observed evidence:

- `loss1` and `loss2` paths reach near-boundary perturbations quickly, and their
  endpoint-vs-movement gradient angles collapse to near zero by `k=25` or `k=50`.
- `loss3` grows its perturbation more gradually in these saved trajectories and keeps
  a larger angle for longer.

Inference:

- The difference between endpoint error and residual movement is especially important
  on the actual `loss3_original` path.
- This supports the point that different losses can have substantially different local
  gradient geometry, especially early in optimization.

### 4. Endpoint loss and movement loss become closer as the perturbation dominates

For `loss3_original_pgd`:

| k | endpoint loss mean | movement loss mean |
|---:|---:|---:|
| 5 | `0.3438` | `0.1243` |
| 10 | `0.5438` | `0.3916` |
| 25 | `2.1124` | `2.0150` |
| 50 | `4.6171` | `4.5728` |

Observed evidence:

- Early on, endpoint loss is much larger than movement loss because endpoint loss
  includes the clean residual \(b=e(x)\).
- Later, the residual increment is large, so endpoint loss and movement loss become
  numerically close.

Inference:

- This helps explain why early gradients differ more: endpoint loss is still strongly
  influenced by the existing clean residual direction.
- As \(\|e(x+\delta)-e(x)\|\) becomes large compared with \(\|e(x)\|\), movement and
  endpoint objectives behave more similarly.

### 5. Movement-gradient norms can be much larger early on loss3

For `loss3_original_pgd`, the movement-to-endpoint gradient norm ratio is:

| k | movement / endpoint grad-norm ratio |
|---:|---:|
| 5 | `2.7067` |
| 10 | `1.8829` |
| 25 | `1.2774` |
| 50 | `1.0486` |

Observed evidence:

- The residual-movement loss can have a much larger gradient norm than the endpoint
  loss early in the path.
- The ratio approaches `1` as the trajectory progresses.

Inference:

- Residual-movement objectives can suggest a stronger local step, but not necessarily
  the same direction as endpoint-error growth.
- This is another reason not to treat residual movement as an equivalent replacement
  for `loss3_original`.

### Compact interpretation

The cleanest interpretation is:

> Under the clean-point linear assumption, the endpoint and movement gradients already
> differ, but the difference is moderate. Under the true nonlinear model/solver, the
> same-point gradient difference is larger, especially in the first part of the
> `loss3_original_pgd` trajectory. As the perturbation grows and the residual increment
> dominates the clean residual, endpoint and movement objectives become more aligned,
> but they do not become conceptually identical.

中文总结：可以这样讲。不同 loss 在局部梯度上确实有差别，尤其是 attack 早期差别很大。
如果做 clean-point 线性假设，差别会被看小；直接算真实 nonlinear gradient，差别更大。
后期 \(\delta\) 变大以后，residual increment 本身变大，endpoint loss 和 movement loss
越来越接近，所以梯度角度会缩小。但这不等于两个 objective 本质上相同，只是这条轨迹后期
增量项开始主导了。


## Raw Every-5-Step Angle Tables

The full every-5-step angle summary tables, now including `k=0`, are recorded in
`docs/angle_diagnostic_raw_tables_20260516.md`.

The table separates four angle families:

1. non-trajectory local candidate-direction angles;
2. local-affine endpoint-vs-movement angles;
3. local-affine bias-vs-movement angles;
4. true nonlinear same-point endpoint-vs-movement angles.

The exact source CSVs are:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_diagnostics.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`

