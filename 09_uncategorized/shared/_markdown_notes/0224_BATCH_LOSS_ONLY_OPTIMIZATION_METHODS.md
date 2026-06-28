# Batch Loss-Only Optimization Methods

This note records the optimization formulas used by the current batch loss-only attack code:

```text
tools/run_batch_three_loss_loss_only.py
```

The constrained optimization problem is

\[
\max_{\|\delta\|_p\le \epsilon} \mathcal O(\delta),
\]

where \(\mathcal O(\delta)\) can be one of the following objective variants:

\[
\mathcal O_i^{\mathrm{orig}}(\delta)=L_i(\delta),
\]

\[
\mathcal O_i^{\mathrm{inc}}(\delta)
=
\frac{L_i(\delta)-L_i(0)}{\|\delta\|_p+\eta},
\]

\[
\mathcal O_i^{\mathrm{reg}}(\delta)
=
L_i(\delta)-C\|\delta\|_p.
\]

At iteration \(k\), define the gradient

\[
g_k
=
\nabla_\delta \mathcal O(\delta_k).
\]

## 1. Projected Gradient Descent / Ascent

Since the objective is maximized, the update is more precisely projected gradient ascent:

\[
\delta_{k+1}
=
\Pi_{\|\delta\|_p\le \epsilon}
\left(
\delta_k+\alpha g_k
\right).
\]

Thus PGD depends on both

\[
\alpha
\quad\text{and}\quad
\epsilon.
\]

Current code:

```python
direction = grad
delta = project_delta(delta + args.alpha * direction, args.epsilon, args.p_order)
```

## 2. \(L_p\)-Steepest PGD

First convert the gradient into the steepest ascent direction under the \(L_p\) geometry:

\[
s_k
=
\arg\max_{\|s\|_p\le 1}
g_k^\top s.
\]

Then use the projected additive update:

\[
\delta_{k+1}
=
\Pi_{\|\delta\|_p\le \epsilon}
\left(
\delta_k+\alpha s_k
\right).
\]

Thus \(L_p\)-steepest PGD also depends on both

\[
\alpha
\quad\text{and}\quad
\epsilon.
\]

Common cases:

\[
p=2:
\qquad
s_k=\frac{g_k}{\|g_k\|_2},
\]

\[
p=\infty:
\qquad
s_k=\operatorname{sign}(g_k).
\]

Current code:

```python
direction = steepest_direction(grad, args.p_order)
delta = project_delta(delta + args.alpha * direction, args.epsilon, args.p_order)
```

## 3. Generalized Power Iteration

Generalized power iteration uses the same gradient

\[
g_k
=
\nabla_\delta \mathcal O(\delta_k),
\]

and the same \(L_p\)-steepest direction

\[
s_k
=
\arg\max_{\|s\|_p\le 1}
g_k^\top s.
\]

However, it is not an additive update. It does not use

\[
\delta_k+\alpha s_k.
\]

Instead, it directly replaces the perturbation by the current steepest direction at radius \(\epsilon\):

\[
\delta_{k+1}
=
\epsilon s_k.
\]

Therefore generalized power iteration depends on

\[
\epsilon
\]

but does not depend on

\[
\alpha.
\]

Current code:

```python
direction = steepest_direction(grad, args.p_order)
delta = project_delta(args.epsilon * direction, args.epsilon, args.p_order)
```

## Summary

\[
\text{PGD:}
\qquad
\delta_{k+1}
=
\Pi_{\|\delta\|_p\le \epsilon}
(\delta_k+\alpha g_k).
\]

\[
L_p\text{-steepest PGD:}
\qquad
\delta_{k+1}
=
\Pi_{\|\delta\|_p\le \epsilon}
(\delta_k+\alpha s_k).
\]

\[
\text{Generalized Power Iteration:}
\qquad
\delta_{k+1}
=
\epsilon s_k.
\]

where

\[
s_k
=
\arg\max_{\|s\|_p\le 1}
g_k^\top s.
\]

The current batch loss-only runner follows these formulas. In particular, the corrected `generalized_power` implementation no longer uses `alpha`; `alpha` appears only in `pgd` and `lp_steepest_pgd`.

