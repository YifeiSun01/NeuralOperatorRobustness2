# Loss3 Optimizer Method Formulas, Readable Version - 2026-05-18

This document rewrites the optimizer formulas without code blocks.

## Basic Objects

Let the residual be

$$
r(x)=F_\theta(x)-G(x).
$$

The attack objective is

$$
L(\delta)=\|r(x_0+\delta)\|_q.
$$

The perturbation constraint is

$$
\|\delta\|_p \le \epsilon.
$$

At iteration $k$, the objective gradient is

$$
g_k=\nabla_\delta L(\delta_k).
$$

The projection back to the feasible $p$-ball is denoted by

$$
\Pi_{p,\epsilon}(z).
$$

For finite $p$, the code uses radial projection:

$$
\Pi_{p,\epsilon}(z)=z\min\left(1,\frac{\epsilon}{\|z\|_p}\right).
$$

For $p=\infty$, it clips each coordinate:

$$
\Pi_{\infty,\epsilon}(z)=\operatorname{clip}(z,-\epsilon,\epsilon).
$$

Define the normalized raw-gradient direction as

$$
u_p(g_k)=\frac{g_k}{\|g_k\|_p}.
$$

Define the $L^p$-steepest direction as

$$
s_p(g_k)=\arg\max_{\|s\|_p\le 1}\langle g_k,s\rangle.
$$

For $1<p<\infty$, with dual exponent $p^*=p/(p-1)$, the code implements

$$
s_p(g_k)=\operatorname{normalize}_p\left(\operatorname{sign}(g_k)|g_k|^{p^*-1}\right).
$$

For $p=2$, this simplifies to

$$
s_2(g_k)=\frac{g_k}{\|g_k\|_2}=u_2(g_k).
$$

This identity is the main reason many $p=2$ methods are identical.

## Objective-Gradient Family

### raw_add

$$
d_k=g_k.
$$

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\delta_k+\alpha g_k\right).
$$

This is raw PGD.

### unit_raw_add

$$
d_k=u_p(g_k).
$$

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\delta_k+\alpha u_p(g_k)\right).
$$

This is additive PGD using the normalized raw-gradient direction.

### raw_replace

$$
d_k=u_p(g_k).
$$

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\epsilon u_p(g_k)\right).
$$

This throws away the previous perturbation and jumps directly to the boundary in
the normalized raw-gradient direction.

### steepest_add

$$
d_k=s_p(g_k).
$$

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\delta_k+\alpha s_p(g_k)\right).
$$

This is $L^p$-steepest PGD.

### steepest_replace

$$
d_k=s_p(g_k).
$$

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\epsilon s_p(g_k)\right).
$$

This is the earlier three-method script's generalized-power-style update:
$L^p$-steepest direction plus replacement to the boundary.

### power_add__objective_gradient

$$
d_k=s_p(g_k).
$$

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\delta_k+\alpha s_p(g_k)\right).
$$

So in the current code,

$$
\text{power\_add\_\_objective\_gradient}=\text{steepest\_add}.
$$

### power_replace__objective_gradient

$$
d_k=s_p(g_k).
$$

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\epsilon s_p(g_k)\right).
$$

So in the current code,

$$
\text{power\_replace\_\_objective\_gradient}=\text{steepest\_replace}.
$$

## JVP/VJP Operator-Power Family

These methods use the Jacobian of the residual map instead of directly using
$g_k$ as the direction source.

Let

$$
x_k=x_0+\delta_k,
$$

and let

$$
J_k = Dr(x_k).
$$

Let $v_k$ be the current power-iteration state.

For the pure JVP/VJP variant,

$$
y_k=J_kv_k.
$$

Then the q-side dual vector is

$$
\phi_q(y_k).
$$

The VJP-side vector is

$$
h_k=J_k^T\phi_q(y_k).
$$

The perturbation-space direction is

$$
d_k=s_p(h_k).
$$

Then the additive version is

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\delta_k+\alpha d_k\right),
$$

and the replacement version is

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\epsilon d_k\right).
$$

Thus:

$$
\text{power\_add\_\_pure\_jvp\_vjp}:\quad
\delta_{k+1}=\Pi_{p,\epsilon}\left(\delta_k+\alpha s_p(J_k^T\phi_q(J_kv_k))\right).
$$

$$
\text{power\_replace\_\_pure\_jvp\_vjp}:\quad
\delta_{k+1}=\Pi_{p,\epsilon}\left(\epsilon s_p(J_k^T\phi_q(J_kv_k))\right).
$$

The current generalized-pq implementation uses the same formula as pure JVP/VJP:

$$
\text{power\_add\_\_generalized\_pq}=\text{power\_add\_\_pure\_jvp\_vjp},
$$

and

$$
\text{power\_replace\_\_generalized\_pq}=\text{power\_replace\_\_pure\_jvp\_vjp}.
$$

## Affine JVP/VJP Variant

Let

$$
\rho_k=\operatorname{mean}_{batch}\|\delta_k\|_p.
$$

The affine residual-side vector is

$$
y_k=r(x_k)+\rho_kJ_kv_k.
$$

Then

$$
h_k=J_k^T\phi_q(y_k),
$$

and

$$
d_k=s_p(h_k).
$$

So the additive affine update is

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\delta_k+\alpha s_p\left(J_k^T\phi_q(r(x_k)+\rho_kJ_kv_k)\right)\right),
$$

and the replacement affine update is

$$
\delta_{k+1}=\Pi_{p,\epsilon}\left(\epsilon s_p\left(J_k^T\phi_q(r(x_k)+\rho_kJ_kv_k)\right)\right).
$$

## Why The First Seven p=2,q=2 Methods Are So Similar

The first seven methods in the completed $p=2,q=2$ run are:

1. `raw_add`
2. `unit_raw_add`
3. `raw_replace`
4. `steepest_add`
5. `steepest_replace`
6. `power_add__objective_gradient`
7. `power_replace__objective_gradient`

All seven use the objective gradient family. None of these seven uses the JVP/VJP
operator-power direction.

For $p=2$,

$$
s_2(g_k)=u_2(g_k)=\frac{g_k}{\|g_k\|_2}.
$$

Therefore the following methods have identical update formulas:

$$
\text{unit\_raw\_add}
=
\text{steepest\_add}
=
\text{power\_add\_\_objective\_gradient}.
$$

And the following replacement methods also have identical update formulas:

$$
\text{raw\_replace}
=
\text{steepest\_replace}
=
\text{power\_replace\_\_objective\_gradient}.
$$

This is why their final-delta cosine similarity is exactly $1$ and their
relative L2 distance is exactly $0$.

Observed from the final-delta similarity analysis for $p=2,q=2$:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `unit_raw_add` vs `steepest_add` | 1.000000 | 0.000000 |
| `steepest_add` vs `power_add__objective_gradient` | 1.000000 | 0.000000 |
| `raw_replace` vs `steepest_replace` | 1.000000 | 0.000000 |
| `steepest_replace` vs `power_replace__objective_gradient` | 1.000000 | 0.000000 |

`raw_add` is not exactly identical because it uses $g_k$ rather than
$g_k/\|g_k\|_2$. But it can still be close because it follows the same gradient
field and is projected to the same feasible ball.

Observed:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `raw_add` vs `steepest_add` | 0.858049 | 0.384210 |

Additive and replacement updates are not the same, even when they use the same
direction family.

Observed:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `steepest_add` vs `steepest_replace` | 0.530025 | 0.830928 |

## Why p=1 Does Not Collapse The Same Way

For $p=1$,

$$
u_1(g_k)=\frac{g_k}{\|g_k\|_1},
$$

but

$$
s_1(g_k)=\operatorname{onehot}\left(\arg\max_i |g_{k,i}|\right)\operatorname{sign}(g_{k,i}).
$$

These are very different directions. Therefore `unit_raw_add` and
`steepest_add` are not equivalent for $p=1$.

Observed:

| Run | Pair | Mean cosine | Mean relative L2 |
|---|---|---:|---:|
| `p=1,q=1` | `unit_raw_add` vs `steepest_add` | 0.113356 | 1.878867 |
| `p=1,q=2` | `unit_raw_add` vs `steepest_add` | 0.143339 | 1.852929 |

So the high similarity among the first seven methods is mainly a $p=2$ formula
collapse, not a universal property of all P/Q settings.
