# Theory and Experiment Plan for `loss3_original` as the Main Regression Attack Objective

## 0. Central Goal

This document is organized around one core question:

> Why is `loss3_original`
> \[
> \max_{\|\delta\|_p\le \varepsilon}
> \|f(x+\delta)-j(x+\delta)\|_q
> \]
> the most appropriate finite-radius regression attack objective for neural operator / regression robustness?

Here:

- \(f\) is the model.
- \(j\) is the true solver / oracle.
- \(e(x)=f(x)-j(x)\) is the error field.
- \(b=e(x)\) is the existing residual at the clean input.
- \(\Delta f=f(x+\delta)-f(x)\).
- \(\Delta j=j(x+\delta)-j(x)\).
- \(\Delta e=e(x+\delta)-e(x)=\Delta f-\Delta j\).

Main conclusion:

> Regression robustness does not mean requiring \(f\) to be invariant. It means requiring \(f\) to co-vary correctly with \(j\).
> Therefore, the truly dangerous directions are not high model sensitivity directions, but high mismatch directions between \(f\) and \(j\).

In other words:

\[
\|f(x+\delta)-f(x)\|_q \text{ is large}
\]

does not imply:

\[
\|f(x+\delta)-j(x+\delta)\|_q \text{ is large}.
\]

The true solver \(j\) also changes when the input is perturbed.

## 1. Overall Proof Route

The argument has five steps.

### Step 1: Oracle Consistency

First, show that a true regression attack must compare \(f(x+\delta)\) with \(j(x+\delta)\).

The true objective is:

\[
L_3^{orig}(\delta)
=
\|f(x+\delta)-j(x+\delta)\|_q
=
\|e(x+\delta)\|_q.
\]

Counterexample:

If the model is perfect,

\[
f=j,
\]

then the true regression error under any perturbation should be:

\[
f(x+\delta)-j(x+\delta)=0.
\]

So a reasonable regression attack loss should be 0.

However:

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_q
\]

and:

\[
L_2(\delta)=\|f(x+\delta)-j(x)\|_q
\]

can still be large, because the true PDE / solver itself may be sensitive to the input.

Conclusion:

> `loss1` and `loss2` can produce false alerts: even when the model is exactly correct, they may still report a large attack effect.

### Step 2: Sources of Bias in Surrogate Losses

The three base losses are:

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_q=\|\Delta f\|_q.
\]

\[
L_2(\delta)=\|f(x+\delta)-j(x)\|_q=\|b+\Delta f\|_q.
\]

\[
L_3(\delta)=\|f(x+\delta)-j(x+\delta)\|_q
=\|b+\Delta f-\Delta j\|_q.
\]

Their meanings are:

- `loss1`: how much the model output changes by itself.
- `loss2`: how far the perturbed model output is from the clean solver target.
- `loss3`: the true regression error under the perturbed input.

`loss1` drops the response of \(j\).
`loss2` freezes the oracle and drops \(j(x+\delta)-j(x)\).
`loss3` keeps the perturbed-input oracle.

Conclusion:

> If \(j(x+\delta)-j(x)\) is not negligible, then `loss1` / `loss2` are not reliable surrogates for `loss3`.

### Step 3: Local Linearization Explains the Difference

Use a local expansion around \(x\):

\[
\Delta f \approx J_f(x)\delta.
\]

\[
\Delta j \approx J_j(x)\delta.
\]

\[
\Delta e
=e(x+\delta)-e(x)
\approx
(J_f(x)-J_j(x))\delta.
\]

Let:

\[
A=J_f(x)-J_j(x).
\]

Then:

\[
e(x+\delta)\approx b+A\delta.
\]

The three local sensitivities are:

\[
L_f=\|J_f\|_{p\to q}.
\]

\[
L_j=\|J_j\|_{p\to q}.
\]

\[
L_e=\|J_f-J_j\|_{p\to q}.
\]

Interpretation:

- \(L_f\): how sensitive the model itself is to input perturbations.
- \(L_j\): how sensitive the true solver itself is to input perturbations.
- \(L_e\): how different the local responses of the model and solver are.

Key case:

\[
L_f \text{ is large},\quad L_j \text{ is large},\quad L_e \text{ is small}.
\]

This means both \(f\) and \(j\) change rapidly, but they change with similar directions and magnitudes, so the error field is not sensitive.

Conclusion:

> Smoothness of \(f\) is not the same as regression robustness. What matters is whether the error field \(f-j\) is stable.

### Step 4: Separate Three Different Notions of Change

Under the local affine model:

\[
e(x+\delta)\approx b+A\delta,
\]

there are three different objects.

#### 4.1 Error Field Movement

\[
\frac{\|e(x+\delta)-e(x)\|_q}{\|\delta\|_p}
\approx
\frac{\|A\delta\|_q}{\|\delta\|_p}.
\]

This asks:

> How quickly does the error vector itself move?

The local limit corresponds to:

\[
\|A\|_{p\to q}=\|J_f-J_j\|_{p\to q}.
\]

This is the meaning of the residual increment ratio.

#### 4.2 Error Norm Growth

\[
\frac{\|e(x+\delta)\|_q-\|e(x)\|_q}{\|\delta\|_p}
=
\frac{\|b+A\delta\|_q-\|b\|_q}{\|\delta\|_p}.
\]

This asks:

> Does the current error norm get pushed outward per unit perturbation?

When \(q=2\) and \(b\neq 0\), a first-order Taylor expansion of the two-norm gives:

\[
\nabla_z\|z\|_2\big|_{z=b}=\frac{b}{\|b\|_2}.
\]

Therefore:

\[
\|b+A\delta\|_2-\|b\|_2
\approx
\left\langle
\frac{b}{\|b\|_2},
A\delta
\right\rangle.
\]

If \(\delta=\|\delta\|_p v\), then:

\[
\frac{\|b+A\delta\|_2-\|b\|_2}{\|\delta\|_p}
\approx
\left\langle
\frac{b}{\|b\|_2},
Av
\right\rangle.
\]

Interpretation:

> The norm increment ratio does not ask how long \(A\delta\) is. It asks how large the outward component of \(A\delta\) is along the current error direction \(b\).

So it is not a Lipschitz norm. It is local outward risk growth.

#### 4.3 Final Error

\[
\|e(x+\delta)\|_q=\|b+A\delta\|_q.
\]

This asks:

> How large is the final error after perturbation?

The true finite-radius attack is:

\[
\max_{\|\delta\|_p\le \varepsilon}
\|e(x+\delta)\|_q.
\]

This is the meaning of `loss3_original`.

#### 4.4 Key Distinction

The error field can move without increasing risk.

For example:

\[
b=(1,0),\quad A\delta=(-2,0).
\]

Then:

\[
\|A\delta\|=2
\]

is large, but:

\[
\|b+A\delta\|=\|(-1,0)\|=1=\|b\|.
\]

The error vector moves a lot, but the error norm does not grow.

Conclusion:

> A locally Lipschitz-large direction is not necessarily an adversarial-risk-increasing direction.

### Step 5: Connect Local and Global Behavior Through a Path Integral

Let the path be:

\[
\gamma(t)=x+t\delta,\quad t\in[0,1].
\]

By the fundamental theorem of calculus:

\[
e(x+\delta)-e(x)
=
\int_0^1
J_e(x+t\delta)\delta\,dt.
\]

So:

\[
e(x+\delta)
=
b+
\int_0^1
J_e(x+t\delta)\delta\,dt.
\]

Furthermore:

\[
\|e(x+\delta)-e(x)\|_q
\le
\int_0^1
\|J_e(x+t\delta)\delta\|_q\,dt
\]

\[
\le
\|\delta\|_p
\int_0^1
\|J_e(x+t\delta)\|_{p\to q}\,dt.
\]

Interpretation:

- Local Lipschitz analysis looks at the instantaneous amplification rate \(J_e(x)\).
- A finite-radius attack depends on the accumulated effect along the entire path \(J_e(x+t\delta)\).

Conclusion:

> Local Lipschitz behavior controls possible instantaneous growth, but it is not equal to finite-radius attack loss.

One direction may have a large local slope but then bend, saturate, or rotate along the path, producing a small endpoint error.
Another direction may not have the largest initial slope, but may accumulate the largest endpoint error along the path.

## 2. Meanings of the Nine Objectives

Each base loss has three objective variants.

### 2.1 Original Objective

\[
L(\delta).
\]

Question:

> How large is the endpoint value?

### 2.2 Increment Ratio

\[
\frac{L(\delta)-L(0)}{\|\delta\|_p+\eta}.
\]

Question:

> How much growth is obtained per unit perturbation?

It is suitable as a local diagnostic. It is not suitable as the main large-radius finite-radius attack loss.

### 2.3 Regularized Objective

\[
L(\delta)-C\|\delta\|_p.
\]

Question:

> After subtracting perturbation cost, is the growth worthwhile?

It may choose an interior radius and may not use the full \(\varepsilon\) budget.

### 2.4 3 by 3 Summary Table

| Objective | Mathematical form | Meaning | Suitable use |
|---|---|---|---|
| loss1_original | \(\|\Delta f\|\) | How much the model output changes by itself | Surrogate baseline / false-alert control |
| loss1_increment_ratio | \(\|\Delta f\|/\|\delta\|\) | Local sensitivity of the model itself | Estimate the local response of \(J_f\) |
| loss1_regularized | \(\|\Delta f\|-C\|\delta\|\) | Model-change gain minus perturbation cost | Cost-aware surrogate control |
| loss2_original | \(\|b+\Delta f\|\) | Fixed clean-target error | Fixed-target surrogate control |
| loss2_increment_ratio | \((\|b+\Delta f\|-\|b\|)/\|\delta\|\) | Local outward growth of fixed-target error | Local diagnostic |
| loss2_regularized | \(\|b+\Delta f\|-C\|\delta\|\) | Fixed-target gain minus perturbation cost | Cost-aware surrogate control |
| loss3_original | \(\|b+\Delta f-\Delta j\|\) | Perturbed-input regression error | Main finite-radius attack objective |
| loss3_increment_ratio | \((\|b+\Delta f-\Delta j\|-\|b\|)/\|\delta\|\) | Local outward growth of true error norm | Local diagnostic |
| loss3_regularized | \(\|b+\Delta f-\Delta j\|-C\|\delta\|\) | True-error gain minus perturbation cost | Cost-aware control |

Also recommended:

| Objective | Mathematical form | Meaning | Suitable use |
|---|---|---|---|
| loss3_residual_increment_ratio | \(\|\Delta f-\Delta j\|/\|\delta\|\) | Error field movement | Local Lipschitz / mismatch of \(J_f-J_j\) |

## 3. Method-Objective Matching

Different optimization methods match different mathematical structures.

### 3.1 Generalized Power Iteration

This is suitable for local homogeneous operator norms:

\[
\max_{\|v\|_p=1}\|Av\|_q.
\]

where \(A=J_f-J_j\) or \(J_f\).

It is suitable for:

- residual increment ratio;
- small-\(\varepsilon\) local Lipschitz analysis;
- local mechanism analysis.

It is not suitable as the only method for large-radius finite-radius attacks, because the global objective is nonlinear, non-homogeneous, and path dependent.

### 3.2 PGD / LP-Steepest PGD

These are suitable for finite-radius nonlinear attacks:

\[
\max_{\|\delta\|_p\le\varepsilon}\|e(x+\delta)\|_q.
\]

LP-steepest PGD chooses at each step:

\[
s_k=
\arg\max_{\|s\|_p\le1}
\langle \nabla_\delta L(\delta_k),s\rangle.
\]

It is better matched to \(p\)-ball geometry than a plain Euclidean gradient step, especially when \(p\neq 2\).

Conclusion:

> GPI / ratio objectives are suitable for local diagnostics. PGD / LP-steepest PGD on `loss3_original` are suitable for finite-radius attacks.

## 4. Core Claims to Validate

### Claim 1: `loss3_original` Is the Oracle-Consistent Main Attack Objective

Only `loss3_original` directly measures:

\[
f(x+\delta)-j(x+\delta).
\]

`loss1` / `loss2` are surrogates.

### Claim 2: `loss1` / `loss2` Can Produce False Alerts

They may find a high model sensitivity direction, but that is not the same as high regression error.

### Claim 3: In Many Directions, \(f\) and \(j\) May Co-Vary

Therefore, a large change in \(f\) is not necessarily an error. The true solver may be changing in the same way.

### Claim 4: The Dangerous Directions Are High Mismatch Directions

Dangerous directions satisfy:

\[
\Delta f-\Delta j \text{ is large}
\]

or:

\[
\cos(\Delta f,\Delta j) \text{ is low or negative}.
\]

### Claim 5: Ratio / Regularized Objectives Are Local or Cost-Aware Diagnostics, Not the Main Global Attack Objective

They can explain local mechanisms, but they cannot replace:

\[
\max_{\|\delta\|\le\varepsilon}\|e(x+\delta)\|.
\]

### Claim 6: A Locally Optimal Direction Is Not the Same as a Finite-Radius Global Endpoint-Optimal Direction

This follows from the path-integral view and nonlinear path effects.

## 5. Experiment Plan

The goal of these experiments is not to exhaustively run 27 combinations. Instead, each experiment should support a specific claim.

### Experiment 1: Main Objective Comparison

#### Purpose

Validate:

> `loss1_original` / `loss2_original` are not reliable substitutes for `loss3_original`.

#### Objects to Compare

- `loss1_original`
- `loss2_original`
- `loss3_original`

#### Method

On the same batch, with the same \(\varepsilon\), and using the same attack method, for example PGD / LP-steepest, separately optimize the three original objectives.

#### Metrics to Record

For each final delta, record:

\[
\|\Delta f\|_q
\]

\[
\|\Delta j\|_q
\]

\[
\|\Delta f-\Delta j\|_q
\]

\[
\|e(x+\delta)\|_q
\]

\[
\|e(x+\delta)\|_q-\|e(x)\|_q
\]

\[
\cos(\Delta f,\Delta j)
\]

tracking discount:

\[
D_f(\delta)=
\frac{\|\Delta f-\Delta j\|_q}{\|\Delta f\|_q+\eta}.
\]

Symmetric version:

\[
D_{sym}(\delta)=
\frac{\|\Delta f-\Delta j\|_q}
{\|\Delta f\|_q+\|\Delta j\|_q+\eta}.
\]

#### Expected Pattern

For the `loss1_original` direction:

- \(\|\Delta f\|\) is large;
- \(\|e(x+\delta)\|\) is not necessarily maximal;
- \(\cos(\Delta f,\Delta j)\) may be high;
- \(D_f\) may be small.

For the `loss3_original` direction:

- \(\|\Delta f\|\) is not necessarily maximal;
- \(\|e(x+\delta)\|\) is maximal;
- \(\|\Delta f-\Delta j\|\) is larger;
- \(\cos(\Delta f,\Delta j)\) is lower.

#### Claims Supported

- Claim 1
- Claim 2
- Claim 4

### Experiment 2: Local Response Decomposition Table

#### Purpose

Validate the local statement:

> The magnitude and direction of \(L_f,L_j,L_e\) are different; high \(L_f\) does not imply high \(L_e\).

#### Directions

Estimate the following directions:

\[
v_f^*=\arg\max_{\|v\|_p=1}\|J_fv\|_q.
\]

\[
v_j^*=\arg\max_{\|v\|_p=1}\|J_jv\|_q.
\]

\[
v_e^*=\arg\max_{\|v\|_p=1}\|(J_f-J_j)v\|_q.
\]

Local outward growth direction:

\[
v_{growth}^*
=
\arg\max_{\|v\|_p=1}
\left\langle
\frac{e(x)}{\|e(x)\|_2},
(J_f-J_j)v
\right\rangle
\]

Use this only when \(q=2\) and \(e(x)\neq0\).

Also include random directions.

#### Table to Record

| direction | \(\|\Delta f\|/\epsilon\) | \(\|\Delta j\|/\epsilon\) | \(\|\Delta f-\Delta j\|/\epsilon\) | \(\cos(\Delta f,\Delta j)\) | \(D_f\) |
|---|---:|---:|---:|---:|---:|
| \(v_f^*\) | | | | | |
| \(v_j^*\) | | | | | |
| \(v_e^*\) | | | | | |
| \(v_{growth}^*\) | | | | | |
| random mean | | | | | |

#### Expected Pattern

If along \(v_f^*\):

- \(\|\Delta f\|/\epsilon\) is large;
- \(\|\Delta j\|/\epsilon\) is also large;
- \(\cos(\Delta f,\Delta j)\) is high;
- \(\|\Delta f-\Delta j\|/\epsilon\) is relatively small;

then:

> The worst direction for `loss1` may be a false alert: the model changes a lot, but the solver changes with it.

If along \(v_e^*\) or \(v_{growth}^*\):

- \(\|\Delta f-\Delta j\|/\epsilon\) is large;
- \(\cos(\Delta f,\Delta j)\) is low or negative;

then:

> These are the true high mismatch directions.

#### Claims Supported

- Claim 2
- Claim 3
- Claim 4

### Experiment 3: Small-Epsilon Sweep

#### Purpose

Validate whether increment / residual ratios really characterize local structure.

#### Radii

\[
\epsilon\in\{10^{-4},10^{-3},10^{-2},10^{-1}\}.
\]

#### Estimators

\[
L_f(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|f(x+\epsilon v)-f(x)\|_q}{\epsilon}.
\]

\[
L_j(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|j(x+\epsilon v)-j(x)\|_q}{\epsilon}.
\]

\[
L_e(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|e(x+\epsilon v)-e(x)\|_q}{\epsilon}.
\]

Also measure norm growth:

\[
G_e(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|e(x+\epsilon v)\|_q-\|e(x)\|_q}{\epsilon}.
\]

#### Records

- Whether the values are stable across \(\epsilon\).
- Whether the worst direction \(v^*(\epsilon)\) is stable.
- Direction cosines across different \(\epsilon\) values:

\[
\cos(v^*(\epsilon_i),v^*(\epsilon_j)).
\]

#### Expected Pattern

If the values and directions are stable at small \(\epsilon\), then the ratio objective indeed characterizes local structure.

If the values and directions drift as \(\epsilon\) grows, then nonlinear path effects are appearing.

#### Claims Supported

- Claim 5
- Claim 6

#### Completion Update - 2026-05-16

Status: **completed for the current FNO / Burgers `nu=0.001` scope**.

The planned sweep over
\(\epsilon\in\{10^{-4},10^{-3},10^{-2},10^{-1}\}\) has been run on GPU and
recorded in `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`.
It records `L_f`, `L_j`, `L_e`, `G_e`, direction source counts, direction
stability, local-reference ratios, figures, and a GPU manifest. A second
version using projected gradient ascent on the direction `v` was also completed
and recorded in
`docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`.

Observed conclusion:

- For `epsilon <= 1e-2`, the finite-difference ratios and optimized directions
  are stable and agree with the clean local Jacobian / outward-growth
  references. This supports the intended epsilon-refinement interpretation: once
  epsilon is small enough, shrinking it further does not materially change the
  measured local structure.
- At `epsilon=0.1`, finite-radius drift appears, especially for `L_e` and
  `G_e`, showing that this larger radius is no longer purely local.
- `L_f` and `L_j` are around `4`, while `L_e` is around `0.87`, so the model and
  solver locally co-move; high model/solver sensitivity does not automatically
  mean high residual sensitivity.
- `G_e` is much smaller than `L_e` locally, around `0.166` versus `0.868`, so
  residual-field movement and outward growth of the current clean residual are
  distinct diagnostics.
- The gradient-optimization version confirms that the candidate-bank directions
  were not arbitrary. For small epsilon, projected gradient ascent recovers
  essentially the same local directions and values.
- The only notable direction-angle caveat is `L_j` at `idx47`: the first two
  solver singular values are nearly tied (`sigma2/sigma1 = 0.977468`), creating
  a flat top-2 subspace. In that near-degenerate case, top-2 subspace alignment
  is a better diagnostic than top-1 vector angle.

Current action item: no immediate rerun is needed for Experiment 3 in the
`nu=0.001` story. Broader model/viscosity sweeps would be separate extension
work, not required to mark this experiment complete.

### Experiment 4: Ray Profile / Local-to-Global Profile

#### Purpose

Validate:

> A locally good direction is not necessarily a good finite-radius endpoint direction.

#### Direction Sources

- `loss3_original` final direction
- `loss3_increment_ratio` final direction
- `loss3_residual_increment_ratio` final direction
- `loss3_regularized` final direction
- random direction

Normalize each final delta:

\[
v=\frac{\delta}{\|\delta\|_p}.
\]

#### Curves

For:

\[
r\in[0,\varepsilon],
\]

plot:

\[
r\mapsto \|e(x+rv)\|_q.
\]

\[
r\mapsto
\frac{\|e(x+rv)\|_q-\|e(x)\|_q}{r+\eta}.
\]

\[
r\mapsto
\frac{\|e(x+rv)-e(x)\|_q}{r+\eta}.
\]

#### Expected Pattern

For an increment-ratio direction:

- the growth rate is high for small \(r\);
- the curve may saturate or bend for large \(r\);
- the endpoint at \(r=\varepsilon\) is not necessarily maximal.

For an original-objective direction:

- the initial slope at small \(r\) is not necessarily maximal;
- but \(\|e(x+rv)\|\) at \(r=\varepsilon\) is maximal.

#### Claims Supported

- Claim 5
- Claim 6

### Experiment 5: Boundary-Rescaled Comparison

#### Purpose

Validate:

> Even after rescaling the directions found by ratio / regularized objectives to the same \(\varepsilon\), those directions do not necessarily maximize `loss3_original`.

#### Method

For each final delta:

\[
v=\frac{\delta}{\|\delta\|_p}.
\]

Construct:

\[
\delta_{bdry}=\varepsilon v.
\]

Compare:

- `loss3_original` on the final delta;
- `loss3_original` on the boundary-rescaled delta;
- `loss3_increment_ratio` on the final delta;
- `loss3_increment_ratio` on the boundary-rescaled delta.

#### Expected Pattern

- The final delta from the regularized objective may be small.
- Scaling it to the boundary may increase the loss, but still may not reach the original-attack direction.
- The increment-ratio direction may have high unit growth, but may not maximize endpoint loss.

#### Claims Supported

- Claim 5
- Claim 6

### Experiment 6: Direction Rotation Along Path

#### Purpose

Validate the path dependence of finite-radius attacks.

#### Method

Take an original attack direction \(\delta^*\), and define:

\[
x_t=x+t\delta^*,\quad t\in[0,1].
\]

At each \(x_t\), re-estimate:

\[
v_e^*(x_t)
=
\arg\max_{\|v\|_p=1}
\|J_e(x_t)v\|_q.
\]

Record:

\[
\cos(v_e^*(x_t),v_e^*(x_0)).
\]

Also optionally record adjacent-direction similarity:

\[
\cos(v_e^*(x_t),v_e^*(x_{t+\Delta t})).
\]

#### Expected Pattern

If the cosine drops substantially or oscillates, then:

> The local worst direction rotates along the path, so a single-point local ratio cannot represent the whole finite-radius attack.

#### Claims Supported

- Claim 6


## 5.1 Current Completion Status for the Minimal Six Experiments - 2026-05-17

This status note supersedes the 2026-05-16 planning snapshot below this heading.
The intended immediate scope is still **FNO / Burgers with `nu=0.001` only**.
Broader extensions such as other viscosities, DeepONet, or default-architecture
sweeps are separate follow-up work and are not required to mark this six-part
plan complete.

Current count:

- Done: 6 experiments.
- Partially done: 0 experiments.
- Not yet done as a full experiment: 0 experiments.

| Experiment | Current Status | What Is Already Done | What Still Needs Work |
|---|---|---|---|
| Experiment 1: Main Objective Comparison | Done for the current `nu=0.001` scope | The `loss1_original`, `loss2_original`, and `loss3_original` comparison has been run and interpreted in `docs/main_objective_mechanism_experiment1_result_20260514.md`. The current local tree contains the 27-run source directory `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/` and the core table `mechanism_diagnostics/mechanism_summary.csv`. The result supports that `loss1` / `loss2` are surrogate movement objectives and that `loss3_original` gives a different oracle-relative endpoint error. | No immediate rerun is required for the `nu=0.001` story. The older derived folder `results/main_objective_mechanism_summary_20260514/` has now been restored locally from R2 and contains `combined_mechanism_summary.csv`, `focused_original_objectives.csv`, and focused-objective plots. |
| Experiment 2: Local Response Decomposition Table | Done for the current `nu=0.001` scope | The decomposition has been completed with `v_f*`, `v_j*`, `v_e*`, random directions, and the previously missing outward-growth direction based on `A^T b`. Evidence is recorded in `docs/fno_solver_jacobian_similarity_result_20260514.md`, `docs/outward_growth_direction_result_20260515.md`, `forensics/fno_solver_jacobian_similarity_20260514/`, and `forensics/outward_growth_direction_20260515/fno_nu0p001/`. | No immediate rerun is required for `nu=0.001`. If the paper later wants a broader claim, repeat this for additional model / solver settings. |
| Experiment 3: Small-Epsilon Sweep | Done for the current `nu=0.001` scope | The full GPU sweep over `epsilon in {1e-4, 1e-3, 1e-2, 1e-1}` has been completed for five samples and recorded in `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md` with outputs under `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`. The second projected-gradient direction optimization check is recorded in `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md` with outputs under `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`. | No immediate rerun is required for the `nu=0.001` story. Broader model/viscosity sweeps are extension work. |
| Experiment 4: Ray Profile / Local-to-Global Profile | Done for the current `nu=0.001` scope | The corrected normal-protocol batch-100 GPU run is recorded in `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md` with outputs under `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`. It supersedes the earlier non-fixed-sign PGD wrapper output. It also includes a historical-script cross-check under `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`. | No immediate rerun is required for the `nu=0.001` story. Repeat only for broader settings or alternative epsilon values. |
| Experiment 5: Boundary-Rescaled Comparison | Done | Boundary-rescaled diagnostics are present in the restored final-boundary source directory `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/` and are summarized in `docs/loss3_original_plan_r2_completion_audit_20260515.md`. The 2026-05-16 local cross-check also records boundary values for the three `loss3_*` objectives in `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`. | No immediate rerun is required. A compact paper table from the JSON summaries is optional presentation work, not required completion work. |
| Experiment 6: Direction Rotation Along Path | Done for the current `nu=0.001` scope | The exact planned path-rotation diagnostic was run and recorded in `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`, with outputs under `forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/`. The expanded top-k subspace / spectral-norm diagnostic is recorded in `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`, with outputs under `forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/`. | No immediate rerun is required for the `nu=0.001` story. A denser or full-SVD path study would be extension work. |

Additional note, 2026-05-17: the older row that marked Experiment 6 as
"Partially done" is now stale. The later direction-rotation and expanded
Jacobian-subspace runs complete the originally requested test: along
`x_t = x + t delta*`, local residual-Jacobian directions were re-estimated and
compared to the clean direction, with adjacent-direction angles and broader
subspace/sketch diagnostics recorded.

Additional note, 2026-05-16: A strict normal-protocol batch-100 Ray Profile
rerun was completed without best-over-steps or multi-restart selection, using
zero initialization, ordinary PGD, `steps=100`, `alpha=0.3`, and endpoint
evaluation by the same `loss3_original` metric. The final corrected run is
`docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`.
It supersedes the earlier non-fixed-sign PGD wrapper output, which had a
manual-PGD sign error. The corrected run matches the historical script
cross-check: direct `loss3_original_pgd` has the largest endpoint mean loss3
(`5.4469`) compared with `loss3_increment_ratio_pgd` (`4.2215`) and
`loss3_regularized_pgd` (`2.0275` after boundary rescaling). Small-radius
diagnostics remain local: `local_outward_growth` wins small clean norm-growth
in 100/100 samples, and `local_residual_movement` wins small residual-increment
in 100/100 samples. This is the intended local-to-global nonlinear gap: local
slope winners are not necessarily the best finite-radius endpoint directions
along the same ray.

Recommended next order:

1. Treat the minimal six-experiment `nu=0.001` plan as complete.
2. Use additional runs only for broader scope, presentation polish, or remote
   backup verification.

## 5.2 Six-Experiment Results, Data, And Conclusions - 2026-05-17

This section records what each of the six experiments actually showed for the
current FNO / Burgers `nu=0.001` story.  Each item separates observed evidence
from the interpretation we should use in the writeup.

### Experiment 1: Main Objective Comparison

Data sources:

- Result note: `docs/main_objective_mechanism_experiment1_result_20260514.md`.
- Main source directory:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- Core numeric table:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/mechanism_diagnostics/mechanism_summary.csv`.
- Restored derived summaries:
  `results/main_objective_mechanism_summary_20260514/combined_mechanism_summary.csv`
  and `results/main_objective_mechanism_summary_20260514/focused_original_objectives.csv`.

Observed results:

| PGD optimized objective | model movement `||Delta f||` | solver movement `||Delta j||` | mismatch `||Delta f-Delta j||` | true error `||e(x+delta)||` | response cosine |
|---|---:|---:|---:|---:|---:|
| `loss1_original` | `11.172 +/- 0.918` | `10.630 +/- 0.993` | `4.351 +/- 1.837` | `4.340 +/- 1.830` | `0.910 +/- 0.068` |
| `loss2_original` | `11.295 +/- 0.989` | `10.700 +/- 0.854` | `4.377 +/- 1.908` | `4.372 +/- 1.917` | `0.912 +/- 0.063` |
| `loss3_original` | `5.866 +/- 3.237` | `5.839 +/- 2.582` | `5.374 +/- 2.773` | `5.395 +/- 2.748` | `0.565 +/- 0.219` |

Additional observed diagnostics:

| PGD optimized objective | error growth | `D_f` | `D_sym` | mismatch per perturbation norm |
|---|---:|---:|---:|---:|
| `loss1_original` | `4.042 +/- 1.822` | `0.385 +/- 0.150` | `0.199 +/- 0.084` | `0.544 +/- 0.230` |
| `loss2_original` | `4.074 +/- 1.913` | `0.382 +/- 0.152` | `0.197 +/- 0.082` | `0.547 +/- 0.239` |
| `loss3_original` | `5.097 +/- 2.718` | `1.012 +/- 0.376` | `0.469 +/- 0.113` | `0.682 +/- 0.336` |

Conclusion / role:

- Observed evidence shows that `loss1_original` and `loss2_original` produce
  large model movement, but the solver also moves strongly in almost the same
  direction.  Their response cosine is about `0.91`, so much of the model
  movement is co-movement with the oracle rather than regression failure.
- Observed evidence shows that `loss3_original` produces smaller raw model
  movement but larger model-solver mismatch and larger true endpoint error.
- Inference: `loss1` and `loss2` are not reliable substitutes for true
  oracle-relative error.  For finite-radius robustness, `loss3_original` is the
  objective that directly measures the failure we care about.

### Experiment 2: Local Response Decomposition Table

Data sources:

- Result notes: `docs/fno_solver_jacobian_similarity_result_20260514.md` and
  `docs/outward_growth_direction_result_20260515.md`.
- SVD/Jacobian outputs: `forensics/fno_solver_jacobian_similarity_20260514/`.
- Outward-growth outputs:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/`.

Observed results:

- The completed outward-growth run covered samples `0, 7, 40, 47, 115`.
- `outward_growth` outward component mean: `0.166514`.
- Control means for the same outward component:
  `error_top = 0.0140571`, `fno_top = 0.00218766`,
  `solver_top = -0.00819025`, and random-best `0.0114951`.
- `error_top` has larger residual-movement / mismatch-gain mean, `0.412169`,
  than `outward_growth`, `0.368053`.
- Finite-difference checks match the linear prediction for `outward_growth`:
  predicted `0.166514`, actual `0.167286` at `rho=1e-4`, and actual `0.166668`
  at `rho=1e-3`.

Conclusion / role:

- Observed evidence separates two local objects:
  `v_e*` maximizes residual-field movement `||A v||`, while
  `v_growth* = normalize(A^T b)` maximizes first-order outward growth of the
  current clean residual norm.
- The two directions are not the same diagnostic: `error_top` moves the residual
  field more, but `outward_growth` pushes the current residual norm outward much
  more strongly.
- Inference: local Lipschitz / residual-movement diagnostics help explain
  geometry, but they are not identical to the finite-radius endpoint objective.
  This supplies the missing local-decomposition row needed by the plan.

### Experiment 3: Small-Epsilon Sweep

Data sources:

- Result note:
  `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`.
- Direction-optimization confirmation:
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`.
- Output directories:
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/` and
  `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`.

Observed results:

- Samples: `0, 7, 40, 47, 115`.
- Radii: `epsilon = 1e-4, 1e-3, 1e-2, 1e-1`.
- Candidate evaluations: `5 samples x 4 epsilons x 178 directions = 3560`.
- Best source counts across all samples and epsilons:

| objective | winning direction source | count |
|---|---|---:|
| `L_f` | FNO `J_f` top direction | `20 / 20` |
| `L_j` | solver `J_j` top direction | `20 / 20` |
| `L_e` | residual/error `J_e` top direction | `20 / 20` |
| `G_e` | clean residual outward-growth direction | `20 / 20` |

Clean local reference means:

| objective | clean local reference mean |
|---|---:|
| `L_f` | `3.895` |
| `L_j` | `4.095` |
| `L_e` | `0.8682` |
| `G_e` | `0.1665` |

Best/local ratios by radius:

| objective | `1e-4` | `1e-3` | `1e-2` | `1e-1` |
|---|---:|---:|---:|---:|
| `L_f` | `1.0005` | `1.0000` | `1.0001` | `0.9901` |
| `L_j` | `0.9997` | `1.0000` | `0.9999` | `0.9854` |
| `L_e` | `1.0004` | `1.0002` | `1.0010` | `0.9608` |
| `G_e` | `0.9965` | `1.0005` | `1.0061` | `1.0576` |

Conclusion / role:

- Observed evidence shows an epsilon-refinement / local-convergence phenomenon:
  for `epsilon <= 1e-2`, finite-difference ratios match the clean local Jacobian
  references and selected direction subspaces are stable.
- Observed evidence also shows finite-radius drift beginning at `epsilon=0.1`,
  especially for `L_e` and `G_e`.
- Inference: ratio objectives are meaningful as local structure diagnostics when
  epsilon is sufficiently small, but they should not be overinterpreted as the
  full finite-radius attack objective.

### Experiment 4: Ray Profile / Local-to-Global Profile

Data sources:

- Corrected result note:
  `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`.
- Output directory:
  `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Historical-script cross-check:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.

Observed results:

| direction | endpoint mean `loss3` at `r=8` | endpoint wins | small clean-growth wins | small residual-increment wins |
|---|---:|---:|---:|---:|
| `loss3_original_final` | `5.447` | `58` | `0` | `0` |
| `loss3_increment_ratio_final` | `4.221` | `10` | `0` | `0` |
| `local_residual_movement` | `4.139` | `31` | `0` | `100` |
| `local_outward_growth` | `2.720` | `0` | `100` | `0` |
| `loss3_regularized_final` | `2.068` | `1` | `0` | `0` |
| `random` | `0.5268` | `0` | `0` | `0` |

Additional observed local-to-global gap metrics:

| quantity | mean | std | min | max |
|---|---:|---:|---:|---:|
| endpoint over small-growth endpoint `loss3` | `3.124` | `1.960` | `1.130` | `12.15` |
| endpoint over small-residual endpoint `loss3` | `2.716` | `4.144` | `1.000` | `25.01` |

Historical-script cross-check under the same normal PGD protocol:

| PGD objective | final delta norm mean | boundary `loss3_original` mean |
|---|---:|---:|
| `loss3_original_pgd` | `7.7510` | `5.4469` |
| `loss3_increment_ratio_pgd` | `7.8505` | `4.2215` |
| `loss3_regularized_pgd` | `0.3043` final, boundary-rescaled to `8.0` | `2.0275` |

Conclusion / role:

- Observed evidence shows that the clean local winners are not the finite-radius
  endpoint winners.  `local_outward_growth` wins the tiny-radius clean-growth
  diagnostic in `100/100` samples but has weak endpoint `loss3`; direct
  `loss3_original` is strongest at the endpoint.
- Inference: local optimality and finite-radius endpoint optimality are
  different.  This is the main local-to-global nonlinear gap result.
- The corrected fixed-sign PGD run supersedes the earlier non-fixed-sign wrapper
  output that had a manual-PGD sign error.

### Experiment 5: Boundary-Rescaled Comparison

Data sources:

- R2/local completion audit:
  `docs/loss3_original_plan_r2_completion_audit_20260515.md`.
- Final-boundary source directory:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- 2026-05-16 local cross-check:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.

Observed results from the 2026-05-14 final-boundary summaries:

| tag | final `||delta||_2` mean | boundary `||delta||_2` mean | final `loss3_original` mean | boundary `loss3_original` mean |
|---|---:|---:|---:|---:|
| `loss3_original_pgd` | `7.7524` | `8.0000` | `5.3949` | `5.4525` |
| `loss3_increment_ratio_pgd` | `7.8499` | `8.0000` | `4.1478` | `4.2225` |
| `loss3_regularized_pgd` | `0.3018` | `8.0000` | `0.4928` | `2.0995` |
| `loss2_increment_ratio_pgd` | `6.3068` | `8.0000` | `2.0591` | `3.1936` |
| `loss2_regularized_pgd` | `5.1939` | `8.0000` | `1.9873` | `3.6531` |

Conclusion / role:

- Observed evidence shows that regularized objectives can produce perturbations
  far inside the epsilon boundary.  For example, `loss3_regularized_pgd` has
  mean final norm only `0.3018` under `epsilon=8`.
- Boundary rescaling can increase endpoint `loss3_original`, but it still does
  not make ratio / regularized directions beat direct `loss3_original`.  For
  example, `loss3_regularized_pgd` rises from `0.4928` to `2.0995` after
  rescaling, still far below `loss3_original_pgd` at `5.4525`.
- Inference: local efficiency or cost-aware objectives do not automatically
  become good finite-radius endpoint attacks just because we rescale their
  directions to the full perturbation budget.

### Experiment 6: Direction Rotation Along Path

Data sources:

- Direction-rotation result:
  `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`.
- Expanded subspace result:
  `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`.
- Output directories:
  `forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/` and
  `forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/`.

Observed results from the direction-rotation run:

| path fraction `t` | mean angle to clean direction | mean adjacent angle | mean local gain ratio over clean direction |
|---|---:|---:|---:|
| `0` | `0` | `NA` | `1.000` |
| `0.25` | `50.2151 deg` | `50.2151 deg` | `2.6753` |
| `0.5` | `50.7279 deg` | `18.7846 deg` | `3.2327` |
| `0.75` | `55.4477 deg` | `12.7040 deg` | `3.1452` |
| `1` | `57.0735 deg` | `4.7341 deg` | `2.6840` |

Observed results from the expanded Jacobian-subspace run:

- Mean top-1 angle to the clean residual-Jacobian direction is already
  `40.55 deg` by `t=0.1` and reaches `57.64 deg` at `t=1`.
- The endpoint top-4 max principal angle to the clean subspace is `81.13 deg`;
  the endpoint top-8 max principal angle is `86.79 deg`.
- Mean spectral norm `sigma1(J_e(x_t))` grows from `0.868` at `t=0` to `8.495`
  at `t=1`.
- The random-probe response-sketch relative difference from clean reaches
  `6.767` at `t=1`.

Conclusion / role:

- Observed evidence shows that the local residual-Jacobian geometry changes
  rapidly after leaving the clean input.  The clean top direction is no longer a
  good representative of the endpoint local geometry.
- Observed evidence also shows that the residual Jacobian not only rotates but
  becomes much steeper along the path.
- Inference: a single clean-point linearization is only a starting snapshot.  It
  cannot fully explain the finite-radius `loss3_original` attack path, because
  the dangerous local directions and local amplification evolve along the path.

### Overall Story Supported By The Six Experiments

Taken together, the six experiments support the following chain:

1. `loss1` and `loss2` can find large model/solver co-movement, so they are not
   reliable proxies for true oracle-relative error.
2. Local Jacobian decomposition explains why movement, residual movement, and
   clean residual outward growth are different objects.
3. Small-epsilon sweeps show when ratio diagnostics really behave like local
   Jacobian quantities.
4. Ray profiles show that local winners do not necessarily remain endpoint
   winners at radius `epsilon=8`.
5. Boundary rescaling shows that simply forcing a local or regularized direction
   to use the full budget does not make it optimal for endpoint `loss3`.
6. Path-rotation diagnostics explain why: the residual-Jacobian geometry rotates
   and steepens along the finite-radius attack path.

The combined conclusion is that local diagnostics are useful for mechanism and
interpretability, but the finite-radius robustness objective must be evaluated
and optimized as its own endpoint problem.

## 6. How Existing Results Can Be Interpreted

Representative results recorded in the original plan:

```text
loss1_increment_ratio_generalized_power:
    L1_inc = 0.913
    L3_inc = 0.157
    L3_orig = 1.556

loss3_increment_ratio_generalized_power:
    L1_inc = 0.826
    L3_inc = 0.401
    L3_orig = 3.507
```

Interpretation:

- The `loss1` direction makes the model \(f\) itself change quickly.
- But the true error growth rate is low, suggesting that \(j\) may also be changing in the same direction.
- The `loss3` direction produces a slightly smaller model-only change, but a larger mismatch between \(f\) and \(j\).

Conclusion:

> A high model sensitivity direction is not the same as a high regression error growth direction.

Another representative result:

```text
loss1_original_pgd:
    L1_orig ~= 11.17
    L3_orig ~= 4.34

loss3_original_generalized_power:
    L1_orig ~= 8.53
    L3_orig ~= 6.86
```

Interpretation:

- The `loss1` attack makes \(f\) move more.
- But the `loss3` attack produces a larger true oracle-relative error.

Conclusion:

> A large change in model output does not imply a large error relative to the true solver.

## 7. Recommended Paper Narrative

### Main Line

```text
We aim to justify loss3_original as the primary adversarial objective for operator regression.
```

Restated:

> The goal is not to present many equivalent losses. The goal is to argue that `loss3_original` is the most direct and correct main objective for finite-radius adversarial attacks in regression / neural operator settings.

### Role Assignment

- `loss3_original`: main finite-radius regression attack objective.
- `loss1_original`, `loss2_original`: surrogate baselines used to expose the bias of simplified objectives.
- `loss3_increment_ratio`: local outward risk growth diagnostic.
- `loss3_residual_increment_ratio`: local error-field Lipschitz / mismatch diagnostic.
- `loss3_regularized`: cost-aware / interior-solution control.
- GPI: tool for local homogeneous operator norms.
- PGD / LP-steepest PGD: tools for finite-radius nonlinear attacks.

### Final Core Conclusions

1. Regression robustness is not invariance of \(f\); it is co-variation of \(f\) with \(j\).

2. High model sensitivity does not imply high regression error.

3. The dangerous directions are high-mismatch directions, not merely high-sensitivity directions.

4. Local Lipschitz / ratio objectives explain mechanisms, but finite-radius adversarial risk must be evaluated by `loss3_original`.

5. `loss1` and `loss2` can produce false alerts because they ignore or freeze the oracle response.

6. `loss3_original` directly answers the attack question:

\[
\max_{\|\delta\|_p\le\varepsilon}
\|f(x+\delta)-j(x+\delta)\|_q.
\]

## 8. Recommended Minimal Experiment Set

If the plan needs to be simplified, it is not recommended to continue centering the narrative on all 27 combinations. The recommended minimal but explanatory experiment set is:

1. `loss1_original` vs `loss2_original` vs `loss3_original`
   - Used to demonstrate surrogate bias / false alerts.

2. `loss3_original` vs `loss3_increment_ratio` vs `loss3_residual_increment_ratio` vs `loss3_regularized`
   - Used to explain the distinction between the main attack, local outward growth, local Lipschitz behavior, and cost-aware objectives.

3. local response decomposition table
   - Used to show that \(f\) and \(j\) co-vary in many directions, and that dangerous directions are mismatch directions.

4. small-\(\epsilon\) sweep
   - Used to show that ratio objectives have stable meaning only at local radii.

5. ray profile
   - Used to show that a locally optimal direction is not the same as a finite-radius endpoint-optimal direction.

6. boundary-rescaled comparison
   - Used to show that, even under the same budget, ratio / regularized directions still do not necessarily maximize `loss3_original`.

7. direction rotation
   - Used to show that finite-radius attacks are path-dependent nonlinear problems.

## 9. One-Sentence Summary

> `loss3_original` is the primary attack loss because it is the only objective among the considered losses that directly measures perturbed-input oracle-relative regression error.
> Other objectives are useful as surrogate baselines or local diagnostics, but they should not be interpreted as equivalent finite-radius attack objectives.

Restated:

> `loss3_original` is the main attack loss because it directly measures the true regression error on the perturbed input; the other losses can be used as controls or local mechanism diagnostics, but they cannot replace the true finite-radius attack objective.
