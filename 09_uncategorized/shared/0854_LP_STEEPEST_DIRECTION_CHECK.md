# LP-Steepest Direction Check

This note records the mathematical problem solved by the `LP-steepest PGD`
direction step and the corresponding implementation check.

## Direction Subproblem

At iteration \(k\), after computing the objective gradient

\[
g_k = \nabla_\delta O(\delta_k),
\]

the \(L_p\)-steepest direction is defined by

\[
s_k
=
\arg\max_{\|s\|_p \le 1} g_k^\top s.
\]

This is the direction that gives the largest first-order increase of the
objective under a unit \(L_p\) constraint.

## Closed-Form Solutions

For \(p=\infty\),

\[
s_k = \operatorname{sign}(g_k).
\]

For \(p=1\),

\[
s_k
=
\operatorname{sign}\bigl((g_k)_{i^\star}\bigr)e_{i^\star},
\qquad
i^\star = \arg\max_i |(g_k)_i|.
\]

For \(1<p<\infty\), let

\[
p^\ast = \frac{p}{p-1}.
\]

Then

\[
s_k
=
\frac{
\operatorname{sign}(g_k)|g_k|^{p^\ast-1}
}{
\left\|
\operatorname{sign}(g_k)|g_k|^{p^\ast-1}
\right\|_p
}.
\]

This direction satisfies

\[
\|s_k\|_p = 1
\]

and reaches the dual norm value

\[
g_k^\top s_k = \|g_k\|_{p^\ast}.
\]

## Code Correspondence

The implementation is in:

```text
tools/run_batch_three_loss_loss_only.py
```

Function:

```python
def steepest_direction(grad, p: float):
    flat = grad.reshape(grad.shape[0], -1)
    if math.isinf(p):
        return torch.sign(grad)
    if p == 1.0:
        out = torch.zeros_like(flat)
        idx = torch.argmax(flat.abs(), dim=1, keepdim=True)
        out.scatter_(1, idx, torch.sign(torch.gather(flat, 1, idx)))
        return out.reshape_as(grad)
    p_dual = p / (p - 1.0)
    mapped = torch.sign(grad) * torch.clamp(grad.abs(), min=1e-30).pow(p_dual - 1.0)
    return normalize_to_p_ball(mapped, p)
```

This code implements exactly the closed-form solution above.

## Numerical Sanity Check

A numerical check was run for

\[
p \in \{1,2,3,\infty\}.
\]

For each case, the computed direction \(s\) satisfied

\[
\|s\|_p = 1
\]

up to floating-point error, and also satisfied

\[
g^\top s = \|g\|_{p^\ast}.
\]

The largest observed absolute error was approximately

\[
4.7\times 10^{-7}.
\]

Therefore, the current `steepest_direction(grad, p)` implementation is correct
for the checked cases \(p=1,2,3,\infty\).
