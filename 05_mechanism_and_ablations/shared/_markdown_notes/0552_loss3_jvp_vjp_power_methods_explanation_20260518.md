# Loss3 JVP/VJP Power Methods Explanation - 2026-05-18

## Main Point

The JVP/VJP power variants are not the same as the earlier objective-gradient
replacement generalized-power method.

The objective-gradient method uses the current residual itself.

The JVP/VJP power method uses how a trial perturbation direction is amplified by
the local residual Jacobian.

Those are different objects.

## Objects

Residual map:

\[
r(x)=F_	heta(x)-G(x)
\]

Current adversarial input:

\[
x_k=x_0+\delta_k
\]

Residual Jacobian at the current input:

\[
J_k = D r(x_k)
\]

Current residual:

\[
r_k=r(x_k)
\]

Current power state / trial direction:

\[
v_k
\]

## What JVP Means

JVP means Jacobian-vector product.

It pushes an input-space direction through the local residual Jacobian:

\[
J_k v_k
\]

Plain meaning: if the input is perturbed in direction \(v_k\), then \(J_kv_k\)
is the first-order change in the model-solver residual.

## What VJP Means

VJP means vector-Jacobian product.

It pulls an output-space vector back to input space:

\[
J_k^T w_k
\]

Plain meaning: after deciding which output-residual direction matters, VJP tells
which input perturbation direction causes that output change.

In PyTorch, this is basically a backpropagation operation.

## Objective-Gradient Direction

The ordinary loss-gradient direction is based on the current residual \(r_k\).

For the loss

\[
L(\delta)=\|r(x_0+\delta)\|_q,
\]

the gradient has the form

\[
g_k = J_k^T \phi_q(r_k),
\]

where \(\phi_q\) is the q-side dual/gradient map.

Then objective-gradient replacement uses

\[
d_k=s_p(g_k),
\]

and

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\epsilon d_k).
\]

This corresponds to `steepest_replace` and `power_replace__objective_gradient`.

## Pure JVP/VJP Power Direction

The pure JVP/VJP power method does not use \(r_k\) inside the q-side map.

It first pushes the current power direction through the Jacobian:

\[
y_k = J_k v_k.
\]

Then it applies the q-side map to that linearized output:

\[
w_k=\phi_q(y_k)=\phi_q(J_kv_k).
\]

Then it pulls back with VJP:

\[
h_k=J_k^T w_k = J_k^T\phi_q(J_kv_k).
\]

Then it converts this into the p-steepest perturbation direction:

\[
d_k=s_p(h_k).
\]

So the pure JVP/VJP replacement update is

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\epsilon s_p(J_k^T\phi_q(J_kv_k))).
\]

The additive update is

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\delta_k+lpha s_p(J_k^T\phi_q(J_kv_k))).
\]

## Affine JVP/VJP Power Direction

The affine variant mixes the current residual with the linearized effect of the
current power direction.

Let

\[

ho_k=\operatorname{mean}_{batch}\|\delta_k\|_p.
\]

Then it forms

\[
y_k=r_k+
ho_kJ_kv_k.
\]

Then

\[
h_k=J_k^T\phi_q(r_k+
ho_kJ_kv_k),
\]

and

\[
d_k=s_p(h_k).
\]

Replacement:

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\epsilon d_k).
\]

Additive:

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\delta_k+lpha d_k).
\]

## Why JVP/VJP Need Not Match Objective Gradient

Objective-gradient direction uses

\[
J_k^T\phi_q(r_k).
\]

Pure JVP/VJP power direction uses

\[
J_k^T\phi_q(J_kv_k).
\]

These are the same only in special cases where the current residual direction
\(r_k\) is aligned with the linearized output direction \(J_kv_k\).

In general, \(r_k\) and \(J_kv_k\) are different. Therefore the final deltas can
be very different.

For q=2, the difference is especially easy to see:

Objective gradient:

\[
g_k=J_k^T r_k.
\]

Pure JVP/VJP power:

\[
h_k=J_k^T J_k v_k.
\]

These solve different directional questions:

- \(J_k^T r_k\): how to increase the current residual norm.
- \(J_k^TJ_kv_k\): which local perturbation direction is amplified by the local
  residual Jacobian.

So JVP/VJP is closer to local operator-norm power iteration, while objective
gradient is direct nonlinear loss ascent.

## Add vs Replace

For any of these directions, the update rule is separate.

Additive:

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\delta_k+lpha d_k).
\]

Replacement:

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\epsilon d_k).
\]

So `power_add__pure_jvp_vjp` and `power_replace__pure_jvp_vjp` use the same type
of JVP/VJP direction, but they use different proposal rules.

## Observed Similarity Evidence

Source numeric tables:

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/*/final_delta_similarity/20260518_final_delta_similarity/final_delta_pairwise_similarity_summary.csv`

Selected observed final-delta similarities:

| Run | Pair | Mean cosine | Mean relative L2 | Interpretation |
|---|---|---:|---:|---|
| `p=2,q=2` | `steepest_replace` vs `power_replace__pure_jvp_vjp` | `0.121156` | `1.293200` | objective-gradient replacement and pure JVP/VJP power are different |
| `p=2,q=2` | `steepest_replace` vs `power_replace__affine_jvp_vjp` | `0.140963` | `1.262736` | objective-gradient replacement and affine JVP/VJP power are different |
| `p=2,q=2` | `power_replace__pure_jvp_vjp` vs `power_replace__affine_jvp_vjp` | `0.106569` | `1.282757` | pure and affine q-side maps give different directions |
| `p=2,q=2` | `power_add__pure_jvp_vjp` vs `power_replace__pure_jvp_vjp` | `-0.001293` | `1.449284` | add and replace proposals can lead to very different final deltas |
| `p=2,q=1` | `steepest_replace` vs `power_replace__pure_jvp_vjp` | `0.114126` | `1.312205` | same pattern in another q setting |
| `p=1,q=1` | `steepest_replace` vs `power_replace__pure_jvp_vjp` | `0.000000` | `1.414214` | p=1 sparse steepest geometry separates them strongly |
| `p=1,q=2` | `steepest_replace` vs `power_replace__pure_jvp_vjp` | `0.000000` | `1.414214` | p=1 sparse steepest geometry separates them strongly |
| `p=1,q=inf` | `steepest_replace` vs `power_replace__pure_jvp_vjp` | `0.000000` | `1.414214` | p=1 sparse steepest geometry separates them strongly |

## Practical Interpretation

The JVP/VJP variants should not be treated as the same as the earlier
three-method `generalized_power` unless the direction formula is explicitly the
objective-gradient replacement formula.

Use these names to avoid confusion:

| Concept | Method labels |
|---|---|
| Objective-gradient generalized-power replacement | `steepest_replace`, `power_replace__objective_gradient` |
| Pure local operator-power direction | `power_add__pure_jvp_vjp`, `power_replace__pure_jvp_vjp` |
| Affine local operator-power direction | `power_add__affine_jvp_vjp`, `power_replace__affine_jvp_vjp` |
| Current duplicate of pure JVP/VJP | `power_*__generalized_pq` |

The current `generalized_pq` implementation is the same as `pure_jvp_vjp`, so
that label is currently redundant.

## Bias Term Clarification: Pure JVP vs b + Jv

The user's key correction is that the actual local loss is not generally a pure
`Jv` problem. Around the current point, the residual has a bias/base term:

\[
b_k = r(x_k).
\]

A first-order local model of the residual after a perturbation direction is
closer to

\[
b_k + J_k v.
\]

Therefore, the local loss geometry relevant to the actual objective is closer to

\[
\|b_k + J_k v\|_q,
\]

not just

\[
\|J_k v\|_q.
\]

The pure JVP/VJP method uses only

\[
J_k v_k
\]

inside the q-side map. This means it is looking for a direction that is strongly
amplified by the local residual Jacobian, but it ignores the current residual
bias direction `b_k`. That can produce a direction that is good for local
operator norm growth but poor for increasing the actual current loss.

The affine JVP/VJP variant was intended to include this bias effect by using

\[
r(x_k) + ho_k J_k v_k.
\]

This is why affine JVP/VJP is conceptually closer to the real local objective
than pure JVP/VJP. However, it is still only a local linearized approximation and
is not the same as directly taking the objective-gradient replacement direction.

Practical conclusion: for the user's core experiment, which asks whether direct
boundary replacement of the PGD/objective-gradient direction explains the fast
loss increase, pure JVP/VJP should not be treated as a main comparison method.
It answers a different operator-power question. The minimal core comparison
should use `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.

