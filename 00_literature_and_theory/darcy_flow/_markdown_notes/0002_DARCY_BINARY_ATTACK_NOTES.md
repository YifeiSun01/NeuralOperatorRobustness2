# Darcy Binary Attack Notes

This note records the attack design for the 2D Darcy Flow case, especially the difference between continuous PGD-style perturbations and binary coefficient-field perturbations.

## Problem Setup

The Darcy coefficient field is binary:

$$
A_0(i) \in \{3, 12\},
$$

where $i$ indexes a spatial grid point. In the original Darcy data generation, a latent Gaussian random field is thresholded so that each pixel becomes either the low coefficient $3$ or the high coefficient $12$.

The model is an FNO map

$$
f(A) \approx U,
$$

and the differentiable Darcy solver is

$$
g(A) = U,
$$

where $U$ solves

$$
-\nabla \cdot (A \nabla U) = 1,
\qquad
U\vert_{\partial \Omega}=0.
$$

The attack is on the coefficient field $A$, not on an initial condition. This is different from Burgers or time-dependent Navier-Stokes, but the adversarial principle is similar.

## Why This Is Not Ordinary Continuous PGD

For Burgers or periodic 2D Navier-Stokes, the perturbation is usually continuous:

$$
x_{\mathrm{adv}} = x_0 + \delta,
$$

with a norm constraint such as

$$
\|\delta\|_2 \le \epsilon
\quad \text{or} \quad
\|\delta\|_\infty \le \epsilon.
$$

A typical gradient update is

$$
\delta_{t+1}
=
\Pi_{\epsilon}
\left(
\delta_t + \alpha S_t
\right),
$$

or a replacement-style update is

$$
\delta_{t+1}
=
\Pi_{\epsilon}
\left(S_t\right),
$$

where $S_t$ is a gradient-based direction, for example a normalized gradient, a steepest direction, or some Jacobian-gradient direction.

For Darcy, this cannot be used directly because $A$ is binary. We do not want values like $4.7$ or $8.2$. Each pixel must remain either $3$ or $12$.

Therefore the attack variable is a binary flip mask rather than a continuous perturbation.

## Binary Flip Representation

Define the flipped value of each pixel as

$$
A_{\mathrm{flip}}(i)
=
\begin{cases}
12, & A_0(i)=3, \\
3, & A_0(i)=12.
\end{cases}
$$

The attack variable is

$$
m_i \in \{0,1\}.
$$

The attacked coefficient field is

$$
A_{\mathrm{adv}}(m)_i
=
(1-m_i)A_0(i)+m_i A_{\mathrm{flip}}(i).
$$

Therefore, by construction,

$$
A_{\mathrm{adv}}(m)_i \in \{3,12\}.
$$

This is the main mechanism that guarantees the coefficient field stays binary. The code never performs an update of the form

$$
A \leftarrow A + \alpha g.
$$

Instead it updates the binary mask $m$, and reconstructs $A_{\mathrm{adv}}$ from $A_0$ and $A_{\mathrm{flip}}$.

## Hamming Budget

The perturbation budget is a Hamming budget:

$$
\sum_i m_i \le K.
$$

If the user specifies an epsilon fraction, then

$$
K
=
\left\lceil
\epsilon \cdot |\Omega_{\mathrm{valid}}|
\right\rceil,
$$

where $\Omega_{\mathrm{valid}}$ is the set of pixels allowed to flip. By default boundary pixels are excluded, because Darcy uses zero Dirichlet boundary for $U$ and it is cleaner to avoid boundary artifacts.

The per-step parameter $\alpha$ is also interpreted as a number of flips, not as a continuous step size. If an alpha fraction is specified, then

$$
K_\alpha
=
\left\lceil
\alpha \cdot |\Omega_{\mathrm{valid}}|
\right\rceil.
$$

If no alpha is specified, the default is roughly

$$
K_\alpha
=
\left\lceil
\frac{K}{T}
\right\rceil,
$$

where $T$ is the number of attack steps.

## Loss Definitions

The three losses are defined analogously to the previous Burgers / Navier-Stokes experiments.

Loss 1 compares the attacked model output to the clean model output:

$$
L_1(A)
=
\|f(A)-f(A_0)\|.
$$

Loss 2 compares the attacked model output to the clean solver output:

$$
L_2(A)
=
\|f(A)-g(A_0)\|.
$$

Loss 3 compares the attacked model output to the attacked solver output:

$$
L_3(A)
=
\|f(A)-g(A)\|.
$$

The optimized objective can be any one of

$$
L_1, \quad L_2, \quad L_3.
$$

However, the final true loss is always recorded as

$$
L_{\mathrm{true}}(A)=L_3(A).
$$

That is, even if the surrogate objective is $L_1$ or $L_2$, the final robustness metric is still the model-vs-solver error under the attacked coefficient field.

## Gradient-Based Flip Score

At attack step $t$, the current mask is $m_t$ and the current coefficient field is

$$
A_t = A_{\mathrm{adv}}(m_t).
$$

Let the optimized surrogate objective be

$$
L(A_t),
$$

where $L$ is one of $L_1,L_2,L_3$.

The gradient is

$$
g_i^{(t)}
=
\frac{\partial L}{\partial A_i}(A_t).
$$

If pixel $i$ is flipped, the coefficient change is

$$
\Delta A_i^{(t)}
=
A_{\mathrm{flip}}(i)-A_t(i).
$$

By first-order Taylor expansion,

$$
L\left(A_t+\Delta A_i^{(t)} e_i\right)-L(A_t)
\approx
 g_i^{(t)}\Delta A_i^{(t)}.
$$

So the binary flip score is

$$
s_i^{(t)}
=
g_i^{(t)}
\left(A_{\mathrm{flip}}(i)-A_t(i)\right).
$$

This score estimates how much the loss will increase if pixel $i$ is flipped.

If

$$
A_t(i)=3,
$$

then

$$
\Delta A_i^{(t)}=12-3=9,
$$

so

$$
s_i^{(t)}=9g_i^{(t)}.
$$

If

$$
A_t(i)=12,
$$

then

$$
\Delta A_i^{(t)}=3-12=-9,
$$

so

$$
s_i^{(t)}=-9g_i^{(t)}.
$$

Thus:

- if $g_i^{(t)}>0$, increasing $A_i$ tends to increase the loss, so $3\to12$ is favorable;
- if $g_i^{(t)}<0$, decreasing $A_i$ tends to increase the loss, so $12\to3$ is favorable.

The attack is therefore still gradient-based, but the gradient is used to rank binary flips rather than to create a continuous perturbation.

## Relation To Previous Continuous PGD Methods

In the continuous case, the add update is roughly

$$
\delta_{t+1}
=
\Pi_{\epsilon}
\left(
\delta_t+\alpha S_t
\right).
$$

In the Darcy binary case, the analogous add update is

$$
m_{t+1}
=
m_t
\cup
\operatorname{Top}_{K_\alpha}
\left(s^{(t)}\right),
$$

subject to

$$
\sum_i m_{t+1,i}\le K.
$$

In the continuous case, the replace update is roughly

$$
\delta_{t+1}
=
\Pi_{\epsilon}(S_t).
$$

In the Darcy binary case, the analogous replace update is

$$
m_{t+1}
=
\operatorname{Top}_{K}
\left(s^{(t)}\right).
$$

So the conceptual mapping is:

| Continuous attack | Darcy binary attack |
|---|---|
| continuous perturbation $\delta$ | binary flip mask $m$ |
| norm ball $\|\delta\|\le\epsilon$ | Hamming ball $\sum_i m_i\le K$ |
| step size $\alpha$ | number of new flips $K_\alpha$ |
| gradient direction $S_t$ | flip score $s_i^{(t)}$ |
| projection $\Pi_\epsilon$ | top-$K$ projection |

The method is therefore a Hamming-constrained, binary version of PGD.

Equivalently, the feasible set is

$$
\mathcal{A}_{\epsilon}
=
\left\{
A:
A_i\in\{3,12\},\
 d_H(A,A_0)\le K
\right\},
$$

where $d_H$ is Hamming distance.

## Steepest Add

For `steepest_add`, only currently unflipped pixels are eligible. Define

$$
\mathcal{I}_t
=
\left\{
 i\in\Omega_{\mathrm{valid}}:
 m_t(i)=0
\right\}.
$$

Select

$$
S_t
=
\operatorname{Top}_{K_\alpha}
\left(
\left\{s_i^{(t)}: i\in\mathcal{I}_t\right\}
\right).
$$

Then

$$
m_{t+1}(i)
=
\begin{cases}
1, & i\in S_t,\\
m_t(i), & \text{otherwise}.
\end{cases}
$$

This is the binary analogue of

$$
\delta_{t+1}
=
\Pi_\epsilon(\delta_t+\alpha S_t).
$$

It can add new flips but cannot remove old flips.

## Steepest Replace

For `steepest_replace`, every step recomputes the best mask from scratch:

$$
m_{t+1}
=
\operatorname{Top}_{K}
\left(
\left\{s_i^{(t)}: i\in\Omega_{\mathrm{valid}}\right\}
\right).
$$

This is the binary analogue of

$$
\delta_{t+1}
=
\Pi_\epsilon(S_t).
$$

It can effectively remove previously chosen flips because it replaces the entire mask.

## Raw Add And Raw Replace

The raw methods use the gradient sign more directly. They first infer whether the gradient wants the coefficient to increase or decrease:

$$
\operatorname{sign}\left(g_i^{(t)}\right).
$$

If

$$
g_i^{(t)}>0,
$$

then increasing $A_i$ increases the objective, so the favorable flip is

$$
3\to12.
$$

If

$$
g_i^{(t)}<0,
$$

then decreasing $A_i$ increases the objective, so the favorable flip is

$$
12\to3.
$$

The ranking score is approximately

$$
|g_i^{(t)}|.
$$

So `raw_add` is like steepest add, but uses the raw gradient sign and magnitude. `raw_replace` is like steepest replace, but again uses the raw gradient sign and magnitude.

The steepest methods are more faithful to the binary objective because they rank pixels by

$$
g_i^{(t)}\Delta A_i^{(t)},
$$

whereas the raw methods rank by something closer to

$$
|g_i^{(t)}|.
$$

## PGD / PAG Binary Variant

The script accepts `pgd`, and also treats `pag` as an alias for `pgd`.

The binary PGD version maintains an accumulated score

$$
q_i^{(t)}.
$$

It updates

$$
q_i^{(t+1)}
=
q_i^{(t)}
+
g_i^{(t)}
\left(A_{\mathrm{flip}}(i)-A_0(i)\right).
$$

Then it projects onto the Hamming ball by selecting the top-$K$ accumulated scores:

$$
m_{t+1}
=
\operatorname{Top}_{K}
\left(
\left\{q_i^{(t+1)}:i\in\Omega_{\mathrm{valid}}\right\}
\right).
$$

This is closest in spirit to continuous PGD because it accumulates gradient information over time and then projects back to the feasible perturbation set.

The important difference is that the projection is not onto an $\ell_2$ or $\ell_\infty$ ball. It is onto the binary Hamming ball:

$$
\left\{
 m\in\{0,1\}^{H\times W}:
 \sum_i m_i\le K
\right\}.
$$

## Loss 1 Random Start

For loss 1,

$$
L_1(A)=\|f(A)-f(A_0)\|,
$$

starting exactly at $A=A_0$ gives

$$
L_1(A_0)=0.
$$

In practice this can make the first gradient uninformative or zero. This is the same reason the earlier continuous loss-1 attacks often needed a random start.

Therefore the binary Darcy script uses a binary random start for loss 1 by default. It randomly flips a small number of valid pixels, still respecting the binary constraint

$$
A_i\in\{3,12\},
$$

and still respecting the Hamming budget

$$
\sum_i m_i\le K.
$$

This avoids the zero-loss starting point while keeping the attack in the correct discrete feasible set.

## What Gets Recorded

For every run, the script records the optimized surrogate loss and the true loss.

If the optimized loss is $L_j$, then the per-step surrogate is

$$
L_{\mathrm{surrogate}}^{(t)}=L_j(A_t).
$$

The true loss is always

$$
L_{\mathrm{true}}^{(t)}=L_3(A_t)=\|f(A_t)-g(A_t)\|.
$$

The final comparison therefore answers two questions:

1. Which surrogate objective increases fastest?
2. Which method produces the largest final true solver-model mismatch?


## Detailed Gradient-to-Update Flow

This section spells out exactly how the attack turns a gradient field into a binary update.

At step $t$, the current coefficient field is

$$
A_t=A_{\mathrm{adv}}(m_t).
$$

The optimized surrogate loss is one of

$$
L(A_t)\in\{L_1(A_t),L_2(A_t),L_3(A_t)\}.
$$

The attack computes the gradient with respect to the current coefficient field:

$$
J_t=\nabla_A L(A_t).
$$

Equivalently, each pixel has a scalar gradient

$$
J_t(i)=\frac{\partial L}{\partial A_i}(A_t).
$$

A continuous PGD method might try to use

$$
A_{t+1}=A_t+\alpha J_t.
$$

For Darcy this is not allowed, because it would usually produce non-binary values. The allowed update at pixel $i$ is only a flip:

$$
A_t(i)\longrightarrow A_{\mathrm{flip}}(i).
$$

The actual allowed one-pixel change is therefore

$$
\Delta A_t(i)=A_{\mathrm{flip}}(i)-A_t(i).
$$

Since the coefficient values are $3$ and $12$,

$$
\Delta A_t(i)=
\begin{cases}
+9, & A_t(i)=3,\\
-9, & A_t(i)=12.
\end{cases}
$$

The first-order Taylor approximation is

$$
L\bigl(A_t+\Delta A_t(i)e_i\bigr)
\approx
L(A_t)+J_t(i)\Delta A_t(i).
$$

Thus the approximate gain from flipping pixel $i$ is

$$
s_t(i)=J_t(i)\Delta A_t(i).
$$

Equivalently,

$$
s_t(i)=J_t(i)\bigl(A_{\mathrm{flip}}(i)-A_t(i)\bigr).
$$

This is the exact bridge from gradient to binary update:

$$
J_t
\longrightarrow
s_t(i)=J_t(i)\bigl(A_{\mathrm{flip}}(i)-A_t(i)\bigr)
\longrightarrow
\operatorname{TopK}(s_t)
\longrightarrow
m_{t+1}
\longrightarrow
A_{t+1}.
$$

If

$$
s_t(i)>0,
$$

then flipping pixel $i$ is predicted to increase the objective. If

$$
s_t(i)<0,
$$

then flipping pixel $i$ is predicted to decrease the objective, so it should not be selected when `positive_only` is enabled.

For `steepest_add`, define the available unflipped set

$$
\mathcal{I}_t=
\{i:m_t(i)=0,\ i\in\Omega_{\mathrm{valid}}\}.
$$

Then select

$$
S_t
=
\operatorname{Top}_{K_\alpha}
\left(\{s_t(i):i\in\mathcal{I}_t\}\right),
$$

and update

$$
m_{t+1}(i)=
\begin{cases}
1, & i\in S_t,\\
m_t(i), & i\notin S_t.
\end{cases}
$$

Finally reconstruct

$$
A_{t+1}(i)
=
(1-m_{t+1}(i))A_0(i)+m_{t+1}(i)A_{\mathrm{flip}}(i).
$$

For `steepest_replace`, the old mask is not preserved. Instead, each step recomputes the whole mask:

$$
m_{t+1}
=
\operatorname{Top}_{K_\epsilon}
\left(\{s_t(i):i\in\Omega_{\mathrm{valid}}\}\right).
$$

So the correspondence to the earlier continuous methods is:

$$
\delta_{t+1}=\Pi_\epsilon(\delta_t+\alpha S_t)
\quad\Longleftrightarrow\quad
m_{t+1}=m_t\cup\operatorname{Top}_{K_\alpha}(s_t),
$$

and

$$
\delta_{t+1}=\Pi_\epsilon(S_t)
\quad\Longleftrightarrow\quad
m_{t+1}=\operatorname{Top}_{K_\epsilon}(s_t).
$$

For `raw_add` and `raw_replace`, the update is cruder. It uses the sign of the gradient to decide which direction is favorable:

$$
J_t(i)>0
\quad\Rightarrow\quad
\text{increasing } A_i \text{ increases the objective},
$$

so the favorable flip is

$$
3\to12.
$$

Similarly,

$$
J_t(i)<0
\quad\Rightarrow\quad
\text{decreasing } A_i \text{ increases the objective},
$$

so the favorable flip is

$$
12\to3.
$$

The raw methods then rank eligible pixels approximately by

$$
|J_t(i)|.
$$

The steepest methods rank by

$$
J_t(i)\Delta A_t(i),
$$

so they explicitly account for the actual binary flip direction.


## Important Naming Caveat: Add/Replace Are Binary Analogues

The names `add` and `replace` are inherited from the earlier continuous perturbation experiments, but in the Darcy binary setting they are not literally the same mathematical operation.

In the continuous setting, the perturbation variable is

$$
\delta \in \mathbb{R}^{H\times W}.
$$

An add-style update means something like

$$
\delta_{t+1}
=
\Pi_\epsilon
\left(
\delta_t+\alpha S_t
\right),
$$

where $S_t$ is a gradient-based direction. This literally adds a continuous vector to the previous perturbation.

A replace-style update means something like

$$
\delta_{t+1}
=
\Pi_\epsilon(S_t),
$$

which replaces the current perturbation by the projected current direction.

For Darcy, however, the perturbation variable is not a continuous field. It is a binary flip mask:

$$
m\in\{0,1\}^{H\times W}.
$$

Therefore the operation

$$
m_t+\alpha S_t
$$

is not meaningful as a valid Darcy coefficient update. The attacked coefficient field must always satisfy

$$
A_i\in\{3,12\}.
$$

So in the binary Darcy setting, `add` and `replace` mean the following binary analogues.

### Binary Add

The binary add update means: keep the old flip mask and append new flips.

If $s_t(i)$ is the current flip score, then

$$
m_{t+1}
=
m_t
\cup
\operatorname{Top}_{K_\alpha}(s_t),
$$

with the Hamming budget constraint

$$
\sum_i m_{t+1}(i)\le K_\epsilon.
$$

This is a discrete analogue of accumulating perturbation, but it is not the same as adding a vector in a continuous space. Once a pixel has been flipped, add-style methods do not undo that flip.

### Binary Replace

The binary replace update means: ignore the old mask and choose a new full mask from the current scores.

$$
m_{t+1}
=
\operatorname{Top}_{K_\epsilon}(s_t).
$$

This is a discrete analogue of replacing the current perturbation direction. It can undo previous flips because the full mask is recomputed each step.

### Main Interpretation

Thus, in the binary Darcy setting:

| Name | Continuous intuition | Binary Darcy implementation |
|---|---|---|
| `add` | accumulate perturbation | append new flips to old mask |
| `replace` | replace perturbation by current direction | recompute full flip mask |

So the names are useful for conceptual continuity with the previous Burgers and Navier-Stokes experiments, but they should not be read as strict mathematical identity.

A more precise description is:

$$
\text{Darcy add/replace methods are binary Hamming-constrained analogues of continuous add/replace PGD methods.}
$$

They live in a different feasible set:

$$
\delta\in\mathbb{R}^{H\times W}
\quad\text{versus}\quad
m\in\{0,1\}^{H\times W}.
$$

The continuous case constrains a norm:

$$
\|\delta\|\le\epsilon.
$$

The Darcy binary case constrains a count:

$$
\sum_i m_i\le K.
$$

Therefore the algorithms are related in spirit but not identical in operation.

## Differences Among Raw Add, Raw Replace, Steepest Add, And Steepest Replace

The four method names combine two independent choices.

First choice: how to score candidate flips.

Second choice: how to update the flip mask.

### Raw Versus Steepest

The raw methods use the sign and magnitude of the gradient:

$$
J_t(i)=\frac{\partial L}{\partial A_i}(A_t).
$$

If

$$
J_t(i)>0,
$$

then increasing $A_i$ is predicted to increase the loss. So the favorable binary flip is

$$
3\to12.
$$

If

$$
J_t(i)<0,
$$

then decreasing $A_i$ is predicted to increase the loss. So the favorable binary flip is

$$
12\to3.
$$

The raw methods then rank eligible pixels approximately by

$$
|J_t(i)|.
$$

The steepest methods instead compute the actual first-order gain of the binary flip:

$$
s_t(i)=J_t(i)\bigl(A_{\mathrm{flip}}(i)-A_t(i)\bigr).
$$

This directly estimates the loss increase caused by flipping pixel $i$.

In the current strict $3/12$ binary Darcy setting,

$$
A_{\mathrm{flip}}(i)-A_t(i)\in\{+9,-9\}.
$$

Because the flip magnitude is always $9$, raw and steepest rankings can be very similar. The main difference is that steepest explicitly multiplies by the feasible flip direction, while raw first checks whether the sign of the gradient matches a feasible flip and then ranks by gradient magnitude.

If future Darcy variants use more than two coefficient values, unequal flip sizes, or spatially varying flip magnitudes, then steepest scoring becomes more meaningfully different from raw scoring.

### Add Versus Replace

The add/replace distinction is often more important in the binary setting.

`add` methods are monotone in the flip mask. They keep old flips and only add new ones:

$$
m_{t+1}
=
m_t
\cup
\operatorname{Top}_{K_\alpha}(\cdot).
$$

Thus they cannot reverse earlier decisions.

`replace` methods recompute the whole mask each step:

$$
m_{t+1}
=
\operatorname{Top}_{K_\epsilon}(\cdot).
$$

Thus they can remove previous flips and choose a new set.

### The Four Methods

`raw_add` uses raw gradient sign/magnitude for scoring and appends new flips:

$$
J_t(i)
\longrightarrow
\operatorname{sign}(J_t(i))
\longrightarrow
|J_t(i)|
\longrightarrow
m_t\cup\operatorname{Top}_{K_\alpha}.
$$

`raw_replace` uses raw gradient sign/magnitude for scoring and recomputes the whole mask:

$$
J_t(i)
\longrightarrow
\operatorname{sign}(J_t(i))
\longrightarrow
|J_t(i)|
\longrightarrow
\operatorname{Top}_{K_\epsilon}.
$$

`steepest_add` uses the first-order binary flip gain and appends new flips:

$$
J_t(i)
\longrightarrow
s_t(i)=J_t(i)\bigl(A_{\mathrm{flip}}(i)-A_t(i)\bigr)
\longrightarrow
m_t\cup\operatorname{Top}_{K_\alpha}.
$$

`steepest_replace` uses the first-order binary flip gain and recomputes the whole mask:

$$
J_t(i)
\longrightarrow
s_t(i)=J_t(i)\bigl(A_{\mathrm{flip}}(i)-A_t(i)\bigr)
\longrightarrow
\operatorname{Top}_{K_\epsilon}.
$$

In short:

$$
\text{raw vs steepest} = \text{score definition},
$$

and

$$
\text{add vs replace} = \text{mask update rule}.
$$

For the current strict binary Darcy case, the expected practical difference is:

1. `add` versus `replace` may produce visibly different trajectories because one cannot undo flips and the other can;
2. `raw` versus `steepest` may be less different because all flips have the same magnitude $9$;
3. `steepest` is still the more faithful first-order binary method because it scores the actual allowed flip direction.

## Computational Difficulty And Why No Exhaustive Search Is Used

The exact binary attack objective is a discrete combinatorial optimization problem:

$$
\max_{m\in\{0,1\}^{H\times W}} L(A_{\mathrm{adv}}(m))
$$

subject to

$$
\sum_i m_i\le K.
$$

If there are

$$
N=|\Omega_{\mathrm{valid}}|
$$

valid pixels, then brute force would require checking

$$
\sum_{k=0}^{K}\binom{N}{k}
$$

possible flip sets.

For a $211\times211$ field, excluding the boundary gives roughly

$$
N\approx209^2=43681.
$$

If

$$
\epsilon=0.01,
$$

then

$$
K\approx437.
$$

The brute-force search size would be

$$
\sum_{k=0}^{437}\binom{43681}{k},
$$

which is completely impossible to enumerate.

Therefore the implemented method does not attempt to solve the global combinatorial problem. It uses a local first-order approximation:

$$
L(A+\Delta A)
\approx
L(A)+\nabla_A L(A)^\top \Delta A.
$$

For a binary flip at pixel $i$, the candidate change is only

$$
\Delta A_i=A_{\mathrm{flip}}(i)-A_i.
$$

So each pixel gets a first-order gain estimate

$$
s_i=\frac{\partial L}{\partial A_i}\Delta A_i.
$$

Then the method selects the top-scoring pixels instead of enumerating all combinations.

Per step, the main operations are:

1. model forward, $f(A_t)$;
2. solver forward, $g(A_t)$, when required by the loss;
3. backward pass to compute $J_t=\nabla_A L(A_t)$;
4. compute all flip scores $s_t(i)$;
5. select the top-$K$ or top-$K_\alpha$ scores.

The top-k step is cheap compared with the PDE solve. Its cost is roughly

$$
O(N\log K),
$$

or similar to that order depending on the backend implementation.

For $211\times211$,

$$
N\approx4.4\times10^4,
$$

which is small enough that sorting or top-k selection is not the bottleneck.

The expensive part is the differentiable Darcy solver, especially for

$$
L_3(A)=\|f(A)-g(A)\|,
$$

because the attack needs gradients through

$$
g(A).
$$

So the practical difficulty is not the top-k sorting. The practical difficulty is the solver forward/backward cost.

This is directly analogous to continuous PGD. Continuous PGD also does not solve the exact global optimum. It uses a local first-order step:

$$
\delta_{t+1}=\Pi_\epsilon(\delta_t+\alpha\nabla L).
$$

The Darcy attack replaces the continuous projection with binary Hamming projection:

$$
m_{t+1}=\operatorname{TopK}(s_t).
$$

Thus the implemented algorithm is best understood as:

$$
\text{gradient}
\longrightarrow
\text{binary flip score}
\longrightarrow
\text{top-k Hamming projection}.
$$

It is not exhaustive combinatorial search; it is a gradient-greedy approximation to a hard discrete problem.

## Summary

The Darcy binary attack is gradient-based, like the earlier Burgers and Navier-Stokes attacks, but it changes the geometry of the perturbation space.

The continuous attacks update

$$
\delta
$$

inside an $\ell_p$ ball.

The Darcy attack updates

$$
m
$$

inside a binary Hamming ball.

The core gradient signal is still

$$
\nabla_A L(A_t),
$$

but the update uses that gradient to decide which pixels should flip between $3$ and $12$.

Thus the method is best described as a gradient-based, Hamming-constrained, binary PGD / greedy flip attack for Darcy coefficient fields.

## 2026-05-28 N50 211x211 attack observations

This section records the observations from the completed Darcy binary attack run:

```text
2D_Darcy_FNO2d/perturbation_results/binary_loss_method_sweep/
  darcy_binary_loss_method_grid_nx211_N50_eps001_alpha5_steps100_traceTrueEvery1_sample0_20260528
```

Visualization outputs were written to:

```text
analysis_outputs/darcy_binary_attack_nx211_N50_loss_curves_gifs_20260528
```

The run used:

```text
resolution: 211 x 211
batch size: 50
steps: 100
epsilon_fraction: 0.01
valid pixels: 209 x 209 = 43681
total Hamming budget: K_epsilon = ceil(0.01 * 43681) = 437
alpha_flips: 5
trace_true_loss3_every: 1
trace_sample_index: 0
```

### Main empirical result

The qualitative behavior is very similar to the earlier 1D Burgers observations: replace-style methods are the fastest to reach the constraint boundary, and they also give the largest final true loss when the optimized objective is loss3.

The clean batch mean true loss3 was:

$$
L_{3,\mathrm{clean}} \approx 0.026575.
$$

Final batch mean true loss3 values were:

| optimized loss | method | final true loss3 | increase over clean |
|---|---:|---:|---:|
| loss1 | raw_add | 0.031509 | 0.004934 |
| loss1 | raw_replace | 0.030554 | 0.003979 |
| loss1 | steepest_add | 0.031698 | 0.005123 |
| loss1 | steepest_replace | 0.030561 | 0.003986 |
| loss2 | raw_add | 0.032282 | 0.005707 |
| loss2 | raw_replace | 0.031018 | 0.004443 |
| loss2 | steepest_add | 0.032282 | 0.005707 |
| loss2 | steepest_replace | 0.031018 | 0.004443 |
| loss3 | raw_add | 0.048095 | 0.021520 |
| loss3 | raw_replace | 0.052848 | 0.026273 |
| loss3 | steepest_add | 0.048095 | 0.021520 |
| loss3 | steepest_replace | 0.052848 | 0.026273 |

The clear ranking is:

```text
loss3 replace-style methods > loss3 add-style methods >> loss1/loss2 objectives
```

In particular, `loss3/raw_replace` and `loss3/steepest_replace` were essentially tied for the largest final true loss3:

$$
L_{3,\mathrm{final}} \approx 0.052848.
$$

The `loss3/raw_add` and `loss3/steepest_add` runs reached:

$$
L_{3,\mathrm{final}} \approx 0.048095.
$$

The best loss1/loss2 runs were much weaker, around:

$$
L_{3,\mathrm{final}} \approx 0.0317\text{ to }0.0323.
$$

### Why replace reaches the boundary immediately

In this binary Darcy setting, `epsilon` is not a continuous norm radius. It is a Hamming budget. It defines the maximum number of pixels that can be flipped:

$$
\sum_i m_i \le K_\epsilon.
$$

Here,

$$
K_\epsilon = 437.
$$

The `alpha` used in this run is also not a continuous step length. It is the number of new pixels an add-style method can append per step:

$$
K_\alpha = 5.
$$

Therefore, for add-style methods, the number of flipped pixels grows slowly:

$$
|m_t| \approx |m_0| + 5t,
$$

until it reaches the total budget 437. Since

$$
437 / 5 \approx 87.4,
$$

add-style methods naturally reach the Hamming boundary only around step 87 to 90. This is why the add curves approach the budget boundary slowly in the plots.

For replace-style methods, the update is different. Each step directly selects a full top-$K_\epsilon$ mask:

$$
m_{t+1}=\operatorname{TopK}_{K_\epsilon}(s_t).
$$

Thus replace-style methods use the entire budget almost immediately. They are on the Hamming constraint boundary from the first update.

### No separate R method was used

No additional `R` method was used in this run. The relevant quantities were only:

```text
epsilon_fraction -> total Hamming budget K_epsilon
alpha_flips      -> per-step add budget K_alpha
```

For add-style methods:

$$
m_{t+1}=m_t \cup \operatorname{TopK}_{K_\alpha}(s_t),
$$

with the final mask clipped by the total budget:

$$
|m_{t+1}| \le K_\epsilon.
$$

For replace-style methods:

$$
m_{t+1}=\operatorname{TopK}_{K_\epsilon}(s_t).
$$

So `alpha_flips` controls how fast add methods move toward the boundary, while `epsilon_fraction` controls the boundary itself.

If add-style methods should reach the boundary faster in future runs, use a larger `alpha_flips`. For this same run:

```text
alpha_flips = 25   -> boundary in about 18 steps
alpha_flips = 50   -> boundary in about 9 steps
alpha_flips = 100  -> boundary in about 5 steps
```

### Raw and steepest are nearly identical here

In this Darcy binary coefficient setting, each valid pixel can only switch between 3 and 12. Thus every flip has magnitude:

$$
|\Delta A_i| = 9.
$$

Because the flip magnitude is constant, `raw_*` and `steepest_*` often rank pixels almost identically. That is why:

```text
raw_add       ~= steepest_add
raw_replace   ~= steepest_replace
```

in the final results. This may change if the coefficient values are not strictly binary, or if different pixels have different allowed flip magnitudes.

### Interpretation

The important observation is that the Darcy binary attack reproduces the same broad pattern seen in the earlier 1D Burgers attacks: replace-style updates are very aggressive because they project directly to the constraint boundary, while add-style updates depend strongly on the per-step alpha budget.

This similarity is striking because the Darcy input is a binary coefficient field and the perturbation geometry is Hamming-constrained, rather than a continuous field inside an $\ell_p$ ball. The common pattern suggests that, across these problems, quickly moving to the boundary of the allowed perturbation set is often more important than the detailed distinction between raw and steepest scoring, at least under the current loss3 objective and binary 3/12 coefficient setup.

## 2026-05-28 budget sweep and model-minus-solver bias

This section records the later budget-sweep observations from the same Darcy
binary attack setup. The main sweep fixed:

```text
resolution: 211 x 211
batch size: 50
steps: 100
method: steepest_replace
true metric: loss3 = ||f(A_adv) - g(A_adv)||
valid pixels: 209 x 209 = 43681
```

The loss3-only K sweep was stored under:

```text
2D_Darcy_FNO2d/perturbation_results/binary_loss3_steepest_replace_budget_sweep/
  darcy_loss3_steepest_replace_budget_sweep_nx211_N50_steps100_20260528
```

The loss1/loss2 completion runs for the larger K values were stored under:

```text
2D_Darcy_FNO2d/perturbation_results/binary_loss12_steepest_replace_budget_completion/
  darcy_loss12_steepest_replace_budget_completion_nx211_N50_steps100_20260528
```

### Loss3 is consistently the strongest objective

For the K values where all three optimized objectives were run with the same
`steepest_replace` method, the final true loss3 was largest every time when
the optimized objective itself was loss3.

| K flips | flip fraction | loss1 final true loss3 | loss2 final true loss3 | loss3 final true loss3 |
|---:|---:|---:|---:|---:|
| 437 | 1% | 0.030561 | 0.031018 | 0.052848 |
| 874 | 2% | 0.033707 | 0.034971 | 0.075608 |
| 2184 | 5% | 0.037051 | 0.048003 | 0.123735 |
| 4370 | 10% | 0.044592 | 0.054925 | 0.164392 |
| 10920 | 25% | 0.072096 | 0.071544 | 0.206095 |

The pattern is therefore not ambiguous:

```text
loss3 objective > loss2 objective >= loss1 objective
```

when the final metric is the true attacked solver-model discrepancy

$$
L_3(A_{\mathrm{adv}})=\|f(A_{\mathrm{adv}})-g(A_{\mathrm{adv}})\|.
$$

The gap also grows with K. At K=437, loss3 reaches about 0.0528 while loss1
and loss2 are around 0.031. At K=10920, loss3 reaches about 0.206 while loss1
and loss2 remain around 0.072.

### Loss3-only budget trend

The loss3 `steepest_replace` sweep shows a monotone increase with the number
of allowed binary flips.

| K flips | approximate flip fraction | final true loss3 | increase over clean |
|---:|---:|---:|---:|
| 44 | 0.1% | 0.029194 | 0.002619 |
| 87 | 0.2% | 0.031866 | 0.005291 |
| 219 | 0.5% | 0.040120 | 0.013545 |
| 437 | 1% | 0.052848 | 0.026273 |
| 874 | 2% | 0.075608 | 0.049032 |
| 2184 | 5% | 0.123735 | 0.097160 |
| 4370 | 10% | 0.164392 | 0.137817 |
| 10920 | 25% | 0.206095 | 0.179520 |

This is the expected direction: with a larger Hamming budget, the attacker can
flip more coefficient pixels and create a larger solver-model mismatch.

### Systematic negative model-minus-solver bias

A striking visual and numerical observation is that the attacked difference
field

$$
f(A_{\mathrm{adv}})-g(A_{\mathrm{adv}})
$$

is mostly negative, especially for the loss3 objective. In words, under the
loss3 attack, the model output is usually smaller than the differentiable
solver output.

For the loss3 `steepest_replace` K sweep:

| K flips | mean model-minus-solver diff | median diff | fraction negative pixels | fraction positive pixels |
|---:|---:|---:|---:|---:|
| 44 | -0.000129 | -0.000113 | 0.819 | 0.181 |
| 87 | -0.000146 | -0.000125 | 0.832 | 0.168 |
| 219 | -0.000198 | -0.000163 | 0.870 | 0.130 |
| 437 | -0.000281 | -0.000230 | 0.901 | 0.099 |
| 874 | -0.000453 | -0.000387 | 0.946 | 0.054 |
| 2184 | -0.000843 | -0.000823 | 0.966 | 0.034 |
| 4370 | -0.001224 | -0.001340 | 0.969 | 0.031 |
| 10920 | -0.001665 | -0.001827 | 0.969 | 0.031 |

So this is not just a colormap artifact. The saved tensors show that the
difference is numerically dominated by negative pixels.

The same sign bias exists for loss1 and loss2, but it is much weaker. For
example:

| K flips | optimized loss | mean model-minus-solver diff | fraction negative pixels | final true loss3 |
|---:|---|---:|---:|---:|
| 437 | loss1 | -0.000132 | 0.827 | 0.030561 |
| 437 | loss2 | -0.000130 | 0.816 | 0.031018 |
| 437 | loss3 | -0.000281 | 0.901 | 0.052848 |
| 2184 | loss1 | -0.000154 | 0.822 | 0.037051 |
| 2184 | loss2 | -0.000191 | 0.830 | 0.048003 |
| 2184 | loss3 | -0.000843 | 0.966 | 0.123735 |
| 10920 | loss1 | -0.000355 | 0.866 | 0.072096 |
| 10920 | loss2 | -0.000288 | 0.865 | 0.071544 |
| 10920 | loss3 | -0.001665 | 0.969 | 0.206095 |

### Interpretation of the negative bias

The current interpretation is that loss3 is not creating a symmetric positive
and negative error pattern. Instead, it is finding coefficient flips where the
true Darcy solver response increases more strongly than the FNO response.

This is plausible for this experiment because:

1. The FNO was trained at low resolution and evaluated at 211 resolution.
2. The input is a sharp binary coefficient field with hard 3/12 jumps.
3. The attack flips pixels in a way that can create local coefficient
   structures outside the model's most familiar training distribution.
4. FNOs often have a smoothing/low-frequency bias, so the model may underreact
   to sharp local binary changes.
5. The solver is the actual PDE solve, so it responds directly to the changed
   coefficient field.

Numerically, for K=437 in the loss3 `steepest_replace` run:

```text
clean model mean:       0.00550193
clean solver mean:      0.00561341
clean mean diff:       -0.00011148

attacked model mean:    0.00557776
attacked solver mean:   0.00585925
attacked mean diff:    -0.00028149
```

Thus the clean model already has a small negative bias relative to the solver.
The loss3 attack amplifies that existing bias. In this run, the model mean does
increase after attack, but the solver mean increases more:

```text
model mean increase after attack:  0.00007583
solver mean increase after attack: 0.00024584
```

The attacked discrepancy therefore becomes more negative.

For larger K, the same behavior becomes stronger. At K=10920:

```text
attacked model mean:  0.00630566
attacked solver mean: 0.00797079
mean diff:           -0.00166513
negative pixels:      96.9%
```

The important observation is not merely that the final loss is larger. The more
specific mechanism appears to be:

```text
loss3 finds binary coefficient flips that make the PDE solver output rise
more than the FNO output, exposing a systematic under-response of the model.
```

This is worth treating as a real experimental finding rather than a plotting
detail. Future checks should include:

1. testing whether the same sign bias appears for a model trained directly at
   211 resolution;
2. testing whether the bias persists at 421 resolution inference;
3. plotting histograms of `model - solver` for clean and attacked samples;
4. checking whether the chosen flips preferentially create high-conductivity
   or low-conductivity pathways;
5. comparing against a soft/continuous relaxation of the binary attack to see
   whether the sign bias is specific to hard flips.

## 2026-05-29 Loss4 Physics-Residual Attack With Boundary

A fourth Darcy attack objective was added to match the StablePDENet-style
physics-residual idea, while keeping the same binary coefficient attack setup
used for the previous loss1/loss2/loss3 experiments.

New script:

```text
2D_Darcy_FNO2d/perturbation_methods/attack_darcy_binary_physics_loss4.py
```

Loss4 is:

```math
L_4(A)=L_{4,pde}(A)+\lambda_{bc}L_{4,bc}(A),
\qquad \lambda_{bc}=1.0.
```

Here `L4_pde` is the differentiable residual for `-div(A grad u)=1`, and
`L4_bc` is the homogeneous Dirichlet boundary penalty on the FNO prediction.
So this is not only a PDE residual; the boundary condition is included by
default. The solver is not used inside the optimized loss4 objective, but
loss1/loss2/loss3 are still recorded at every step using the numerical solver.

Completed run matching the previous main Darcy binary setup:

```text
2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics/darcy_loss4_physics_steepest_replace_nx211_N50_eps001_alpha5_steps100_bc_20260529
```

Configuration:

```text
resolution:       211 x 211
samples:          50
steps:            100
epsilon_fraction: 0.01
max flips:        437
method:           steepest_replace
bc_weight:        1.0
```

Final mean results:

```text
clean loss1:          0.000000
clean loss2:          0.026588
clean loss3:          0.026588
clean loss4 physics: 17.712782
clean loss4_pde:     17.712658
clean loss4_bc:       0.000124

final loss1:          0.011647
final loss2:          0.028078
final true loss3:     0.025949
final loss4 physics: 36.435963
final loss4_pde:     36.435841
final loss4_bc:       0.000123

loss1 increase:       0.011647
loss2 increase:       0.001490
loss3 increase:      -0.000639
loss4 increase:      18.723179
loss4_pde increase:  18.723179
loss4_bc increase:   -0.000001
```

Interpretation: loss4 successfully attacks the physics residual surrogate, but
it did not increase the solver-consistent true loss3 on this run. This supports
the distinction discussed in the StablePDENet notes: physics-residual
adversarial objectives and solver-referenced evaluation errors are related, but
they are not the same objective.

