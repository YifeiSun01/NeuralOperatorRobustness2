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


## 5.1 Current Completion Status for the Minimal Six Experiments - 2026-05-16

This status note is for planning the next round of work. The current intended
scope is **FNO / Burgers with `nu=0.001` only**. Broader extensions such as
`nu=0.01` or default-architecture sweeps are intentionally not counted as
required for this immediate plan because they would make the next round too
expensive.

Current count:

- Done: 3 experiments.
- Partially done: 2 experiments.
- Not yet done as a full experiment: 1 experiment.

| Experiment | Current Status | What Is Already Done | What Still Needs Work |
|---|---|---|---|
| Experiment 1: Main Objective Comparison | Done | The `loss1_original`, `loss2_original`, and `loss3_original` comparison has been run and interpreted. The existing results support that `loss1` / `loss2` are surrogate objectives and that `loss3_original` gives a different true oracle-relative endpoint error. The later native gradient-angle analysis also directly supports that, at the same \(\delta_k\), the `loss3_original` optimization direction can differ strongly from the `loss1_original` / `loss2_original` directions. | No immediate rerun is required for the `nu=0.001` story. |
| Experiment 2: Local Response Decomposition Table | Done for the current `nu=0.001` scope | The decomposition has been completed with \(v_f^*\), \(v_j^*\), \(v_e^*\), random directions, and the previously missing outward-growth direction based on \(A^Tb\). The key result is that the direction maximizing error-field movement, \(\|Av\|\), is not the same as the direction maximizing immediate outward growth of the current residual norm. Finite-difference checks also support the local interpretation. | No immediate rerun is required for `nu=0.001`. If the paper later wants a broader claim, repeat this for additional model / solver settings. |
| Experiment 3: Small-Epsilon Sweep | Partially done | The outward-growth work already checked local behavior at small finite-difference radii such as \(\rho=10^{-4}\), \(10^{-3}\), and \(10^{-2}\), and the finite-difference values matched the predicted local outward growth. This supports the local Taylor explanation. | Still needs the full planned sweep over \(\epsilon\in\{10^{-4},10^{-3},10^{-2},10^{-1}\}\), with systematic records for \(L_f(\epsilon)\), \(L_j(\epsilon)\), \(L_e(\epsilon)\), \(G_e(\epsilon)\), direction stability, and cross-\(\epsilon\) direction cosine tables. |
| Experiment 4: Ray Profile / Local-to-Global Profile | Not yet done as a full experiment | The ray-profile idea has been designed and located in the plan, but the full ray-profile curves have not yet been produced as the main evidence. | This should be the top priority for the next run. For final directions from `loss3_original`, `loss3_increment_ratio`, `loss3_residual_increment_ratio`, `loss3_regularized`, and random directions, record curves for \(r\mapsto\|e(x+rv)\|_q\), norm increment ratio, and residual increment ratio over \(r\in[0,\epsilon]\). This is the cleanest experiment for showing that a locally good direction need not be the finite-radius endpoint-best direction. |
| Experiment 5: Boundary-Rescaled Comparison | Done | The boundary-rescaling diagnostic has been run and interpreted. The observed behavior is that ratio / regularized objectives can penalize \(\delta\), so their final perturbations may not reach the boundary. After rescaling those directions to radius \(\epsilon\), their `loss1`, `loss2`, and `loss3` values still do not necessarily beat the directions found by directly optimizing the corresponding original objective. | No immediate rerun is required. This result should be used as evidence that local or cost-aware directions do not automatically become best finite-radius directions after rescaling. |
| Experiment 6: Direction Rotation Along Path | Partially done | Related trajectory-gradient evidence exists: true nonlinear gradient angles along saved attack paths were computed, and they show that the difference between objectives can remain substantial for many steps. This supports the idea that finite-radius behavior is not fully captured by a single clean-point linearization. | The exact planned experiment is still missing: along \(x_t=x+t\delta^*\), re-estimate \(v_e^*(x_t)=\arg\max_{\|v\|_p=1}\|J_e(x_t)v\|_q\), then record \(\cos(v_e^*(x_t),v_e^*(x_0))\) and adjacent-direction cosines. This should be run after the ray-profile experiment unless the path-rotation claim becomes central. |

Recommended next order:

1. Run Experiment 4: Ray Profile / Local-to-Global Profile.
2. Complete Experiment 3: Small-Epsilon Sweep.
3. Complete Experiment 6: Direction Rotation Along Path.

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
