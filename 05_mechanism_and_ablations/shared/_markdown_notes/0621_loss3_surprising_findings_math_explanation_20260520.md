# Loss 3 Surprising Findings: Mathematical Interpretation

Date: 2026-05-20

This note explains the mathematical mechanisms that may underlie the observed Loss 3 results. It deliberately separates observed empirical behavior from theoretical interpretation. The main caveat is that the current `generalized_power` / `steepest_replace` method is not an exact generalized-power optimizer for the full nonlinear Loss 3 objective; it is an objective-gradient steepest replacement method.

## Setup

The constrained attack is approximately

\[
\max_\delta L(\delta)=\|r(x_0+\delta)\|_q,
\quad \|\delta\|_p\le \epsilon .
\]

For a differentiable point, with

\[
r_k=r(x_0+\delta_k), \quad J_k = D r(x_0+\delta_k),
\]

the ordinary objective gradient is

\[
g_k=\nabla_\delta L(\delta_k)=J_k^T \phi_q(r_k),
\]

where \(\phi_q\) is the q-norm dual/gradient map.

The four core methods are:

- `raw_add`: \(\delta_{k+1}=\Pi_{p,\epsilon}(\delta_k+\alpha g_k)\).
- `raw_replace`: normalize the raw gradient to the p-boundary and replace: \(\delta_{k+1}=\Pi_{p,\epsilon}(\epsilon\, g_k/\|g_k\|_p)\).
- `steepest_add`: use the p-steepest unit direction \(s_p(g_k)\): \(\delta_{k+1}=\Pi_{p,\epsilon}(\delta_k+\alpha s_p(g_k))\).
- `steepest_replace`: replace by the p-steepest boundary direction: \(\delta_{k+1}=\Pi_{p,\epsilon}(\epsilon s_p(g_k))\).

The p-steepest direction solves the local linear problem

\[
s_p(g)=\arg\max_{\|d\|_p\le 1}\langle g,d\rangle .
\]

For important p values:

- \(p=2\): \(s_p(g)=g/\|g\|_2\).
- \(p=1\): \(s_p(g)\) is a signed one-hot vector at the largest gradient coordinate.
- \(p=\infty\): \(s_p(g)=\operatorname{sign}(g)\).

## 1. Boundary Arrival Is Not Convergence

Observed: replacement/GPI can hit the epsilon boundary at step 1, but loss can continue to grow afterward.

Mathematical explanation: reaching \(\|\delta\|_p=\epsilon\) only means the norm constraint is active. It does not mean the constrained first-order condition is satisfied.

For the simple smooth \(p=2\) case, a boundary local maximum must satisfy that the gradient has no improving tangent component:

\[
(I-u u^T) g = 0,
\quad u=\delta/\|\delta\|_2 .
\]

Equivalently, \(g\) must be parallel to \(\delta\). If \(g\) has a tangent component, then one can rotate \(\delta\) along the sphere and still increase loss. Therefore a method can be exactly on the boundary and still be far from stationary.

This explains the central observation: GPI/replacement first solves the radial-budget problem, then it still has to solve the directional problem on the boundary.

The p=1/q=inf caveat is also reasonable: nonsmooth geometry can make the boundary direction jump between extreme coordinates, and the tangent-improvement picture is less smooth. In such settings, replacement can hit the boundary but not necessarily obtain positive post-boundary gain.

## 2. Why GPI/Replacement Is So Fast But Not Necessarily Final-Loss Best

Observed: `steepest_replace` is extremely fast, but 300-step additive methods can sometimes achieve larger final mean loss.

Mathematical explanation: `steepest_replace` greedily solves the local linearized full-budget subproblem

\[
\max_{\|\delta\|_p\le \epsilon}\langle g_k,\delta\rangle,
\]

whose solution is \(\epsilon s_p(g_k)\). That is why it immediately uses the full budget.

This is powerful when the local gradient direction is close to a dominant high-loss direction. If the local loss landscape is dominated by a leading mode, then one replacement step can land near the important direction very early.

But this is not a proof of global optimality for the true nonlinear objective. The true Loss 3 is not exactly the local linear model. Replacement is greedy, can overshoot curvature, and discards path memory. Additive PGD-like methods may follow a curved ascent path more faithfully over hundreds of steps. Thus GPI can be best early while not always being the 300-step final-loss winner.

Paper-safe interpretation: GPI/replacement is an aggressive local-linear surrogate optimizer, not an exact mathematical optimizer for full nonlinear Loss 3.

## 3. Why LP-Steepest Add Has Near-Linear Norm Growth While Raw PGD Curves

Observed: `steepest_add` boundary ratio often grows almost linearly; `raw_add` is curved and slows down.

Mathematical explanation: `steepest_add` uses a p-unit direction:

\[
\|s_p(g_k)\|_p=1.
\]

Before projection is active, each additive step has controlled p-size \(\alpha\). If directions are not changing too wildly, the radial budget grows roughly like

\[
\|\delta_k\|_p \approx k\alpha .
\]

This creates the visually straight boundary-ratio curves.

Raw PGD uses \(g_k\) itself, so the actual step size is \(\alpha\|g_k\|_p\). The gradient norm varies across samples and across time. As the loss saturates or directions rotate, radial progress can slow, producing a curved approach to the boundary.

The q=inf exception is expected: \(q=\infty\) focuses on the currently largest residual coordinate and is nonsmooth when the active coordinate changes. This can make even normalized steepest directions flip or lose radial alignment, so the clean linear story weakens.

## 4. Why Boundary-Ratio Standard Deviation Is Diagnostic

Observed: raw PGD has large boundary-ratio std; LP-steepest has much smaller std; replacement is near zero when it exactly enforces the boundary.

Mathematical explanation: boundary-ratio std measures radial synchronization across samples.

For raw PGD:

\[
\delta_{k+1}=\Pi(\delta_k+\alpha g_k),
\]

samples with large \(\|g_k\|_p\) move radially faster than samples with small \(\|g_k\|_p\). Therefore boundary arrival times differ strongly.

For LP-steepest add:

\[
\delta_{k+1}=\Pi(\delta_k+\alpha s_p(g_k)),
\quad \|s_p(g_k)\|_p=1,
\]

so the raw step size is synchronized across samples. Directional alignment can still differ, but the main scale variability is removed.

For replacement:

\[
\delta_{k+1}=\Pi(\epsilon d_k),
\quad \|d_k\|_p\approx 1,
\]

so all samples are placed on or very near the boundary immediately. Any nonzero std usually indicates zero/near-zero gradients, nonsmooth geometry, numerical effects, or a setting where the replacement direction is not exactly p-unit in practice.

## 5. Why GPI Has Large Angular Motion

Observed: replacement-family methods have the largest per-step delta angle changes.

Mathematical explanation: additive methods have angular inertia. For p=2, once \(\delta_k\) is near the boundary,

\[
\delta_{k+1}\approx \epsilon\frac{\delta_k+\alpha d_k}{\|\delta_k+\alpha d_k\|_2}.
\]

If \(\alpha\ll \epsilon\), the angular change is only about

\[
O((\alpha/\epsilon)\|P_T d_k\|),
\]

where \(P_T\) is the tangent projection. So additive PGD rotates slowly after it becomes large.

Replacement has no such inertia:

\[
\delta_{k+1}=\epsilon d_k.
\]

The new direction can be very different from the old direction. Therefore GPI/replacement can rotate hard on the boundary. This directly explains why boundary arrival and later loss growth separate: after step 1, GPI is mostly doing direction search on the boundary.

## 6. Why Final Delta Shapes Can Be Similar

Observed: in p2q2 and p2q1, final deltas from different methods often have similar broad shape; in p=1/q=inf settings this is much less reliable.

Mathematical explanation: if the local or semi-global loss landscape has a dominant direction, many reasonable ascent methods can converge toward the same basin or leading mode. A simplified local model is

\[
r(x_0+\delta)\approx r_0+J\delta.
\]

For p=q=2-like settings, maximizing \(\|r_0+J\delta\|_2\) over a ball is closely related to dominant singular/eigen directions of the residual Jacobian. Different methods can approach that same broad direction even if their paths differ.

This is why GPI can reach a final-like shape early and additive methods can eventually look similar.

But the similarity is not guaranteed. With p=1, the feasible ball has sparse extreme points. With q=inf, the loss focuses on the largest residual coordinate. Those geometries can create multiple competing active coordinates and spike-like solutions, so different methods may no longer share one smooth dominant direction.

## 7. Why P/Q Geometry Changes Perturbation Realism

Observed: p2q1 often looks natural; p1 and q=inf settings are more spike-prone.

Mathematical explanation: p controls the input perturbation geometry, and q controls which output residuals drive the gradient.

- p=2 is isotropic and smooth. It tends to distribute perturbation energy.
- p=1 has sparse extreme points. The steepest direction is one coordinate, so spikes are natural.
- p=inf has sign/corner geometry. The steepest direction is sign-like and can create non-smooth patterns.
- q=2 aggregates residuals smoothly.
- q=1 uses sign residuals across many coordinates, often still broad after VJP.
- q=inf focuses on the maximum residual coordinate, which can localize the gradient and create sharp peaks.

Thus p2q1 can look more physical because p=2 distributes input energy and q=1 still uses many residual coordinates. In contrast, p=1 or q=inf pushes the optimization toward sparse or localized extreme points.

## 8. Why raw_replace Equals steepest_replace When p=2

Observed: for checked p=2 groups, `raw_replace` and `steepest_replace` are exactly equivalent.

Mathematical explanation: for p=2,

\[
s_2(g)=g/\|g\|_2.
\]

`raw_replace` also normalizes the raw gradient to the p-boundary. Therefore both compute the same direction and then replace

\[
\delta_{k+1}=\epsilon g_k/\|g_k\|_2.
\]

This equivalence is independent of q as long as both methods use the same objective gradient \(g_k=\nabla L\). q changes the gradient, but once the same gradient is computed, the p=2 normalization is identical.

For p not equal to 2, the equivalence breaks:

- p=1: raw replacement gives an L1-normalized gradient, while steepest replacement gives a signed one-hot vector.
- p=inf: raw replacement divides by max absolute gradient, while steepest replacement gives a sign vector.
- general p: steepest replacement applies a nonlinear power map to gradient coordinates, not simple p-normalization.

The near-equivalence between `raw_add` and `raw_replace` in some p1q2 settings is therefore not a general theorem. It likely occurs when the raw gradient direction is stable over time and additive projection ends up near the same normalized gradient direction. It should be treated as empirical alignment, not method identity.

## Overall Mathematical Interpretation

The cleanest explanation is:

1. Standard fixed-budget loss maximization often wants to use the epsilon budget.
2. But using the budget only solves the radial part of the problem.
3. The remaining hard problem is directional optimization on the boundary.
4. Replacement/GPI is fast because it turns every step into a full-budget directional update.
5. Additive methods are slower because they have radial travel first and angular inertia later.
6. p/q geometry decides whether the directional update is smooth and distributed or sparse and spike-like.

The surprising part is not that replacement reaches the boundary. That is built into the update. The surprising part is that, in p=2 regimes, the local gradient-steepest replacement direction seems to align unusually well with the high-value Loss 3 directions, even though the full nonlinear Loss 3 objective is not mathematically a generalized-power objective.
