# Loss3 Optimizer Method Formulas And Equivalences - 2026-05-18

## Common Setup

Let

```text
r(x) = F_theta(x) - G(x)
L(delta) = || r(x0 + delta) ||_q
```

The perturbation is constrained by

```text
||delta||_p <= epsilon.
```

At step `k`:

```text
g_k = grad_delta L(delta_k)
```

The code uses the following radial/clipping projection operator `P_{p,epsilon}`:

```text
P_{p,epsilon}(z) = z * min(1, epsilon / ||z||_p),     finite p
P_{inf,epsilon}(z) = clip(z, -epsilon, epsilon),      p = inf
```

This projection is the code's budget enforcement operator. For finite `p` it is
radial scaling to the p-ball.

Define normalized raw-gradient direction:

```text
u_p(g) = g / ||g||_p
```

with the corresponding `p=inf` implementation using `||g||_inf` in the denominator.

Define the Lp-steepest direction:

```text
s_p(g) = arg max_{||s||_p <= 1} <g, s>.
```

The code implements:

```text
p = inf: s_inf(g) = sign(g)
p = 1:   s_1(g) is one-hot at argmax_i |g_i|, with the sign of g_i
1<p<inf: p* = p/(p-1)
         s_p(g) = normalize_p( sign(g) |g|^(p* - 1) )
```

Important special case:

```text
p = 2: s_2(g) = g / ||g||_2 = u_2(g)
```

This is why many `p=2` methods collapse to identical updates.

## Objective-Gradient Methods

These methods all use the same objective gradient `g_k = grad_delta L(delta_k)`.

### raw_add

```text
d_k = g_k
z_{k+1} = delta_k + alpha d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
```

This is raw PGD.

### unit_raw_add

```text
d_k = u_p(g_k) = g_k / ||g_k||_p
z_{k+1} = delta_k + alpha d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
```

This removes the raw gradient magnitude and keeps only its p-normalized direction.

### raw_replace

```text
d_k = u_p(g_k)
z_{k+1} = epsilon d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
```

This is normalized raw-gradient replacement to the boundary.

### steepest_add

```text
d_k = s_p(g_k)
z_{k+1} = delta_k + alpha d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
```

This is LP-steepest PGD.

### steepest_replace

```text
d_k = s_p(g_k)
z_{k+1} = epsilon d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
```

This is the previous three-method script's `generalized_power` implementation:
objective-gradient steepest direction plus replacement/boundary update.

### power_add__objective_gradient

```text
d_k = s_p(g_k)
z_{k+1} = delta_k + alpha d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
```

In the current code, this is mathematically the same as `steepest_add`.

### power_replace__objective_gradient

```text
d_k = s_p(g_k)
z_{k+1} = epsilon d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
```

In the current code, this is mathematically the same as `steepest_replace`.

## JVP/VJP Operator-Power Methods

These methods do not simply use `g_k = grad_delta L(delta_k)` as the direction
source. They use the Jacobian of the residual map.

Let

```text
x_k = x0 + delta_k
J_k = D r(x_k)
v_k = current power state
```

The q-side dual map in the code is:

```text
q = inf: phi_inf(y) is one-hot at argmax_i |y_i|, with sign(y_i)
q = 1:   phi_1(y) = sign(y)
finite q: phi_q(y) = sign(y) |y|^(q-1)
```

After computing a VJP-side vector `h_k`, the p-side direction is:

```text
d_k = s_p(h_k)
```

and then either additive or replacement proposal is used.

### power_add__pure_jvp_vjp

```text
y_k = J_k v_k
h_k = J_k^T phi_q(y_k)
d_k = s_p(h_k)
z_{k+1} = delta_k + alpha d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
v_{k+1} = d_k
```

### power_replace__pure_jvp_vjp

```text
y_k = J_k v_k
h_k = J_k^T phi_q(y_k)
d_k = s_p(h_k)
z_{k+1} = epsilon d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
v_{k+1} = d_k
```

### power_add__generalized_pq

Current code path is the same as `power_add__pure_jvp_vjp`:

```text
y_k = J_k v_k
h_k = J_k^T phi_q(y_k)
d_k = s_p(h_k)
z_{k+1} = delta_k + alpha d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
v_{k+1} = d_k
```

This alias exists for naming the intended P-Q generalized-power direction, but
in the current implementation it is not a distinct formula from `pure_jvp_vjp`.

### power_replace__generalized_pq

Current code path is the same as `power_replace__pure_jvp_vjp`:

```text
y_k = J_k v_k
h_k = J_k^T phi_q(y_k)
d_k = s_p(h_k)
z_{k+1} = epsilon d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
v_{k+1} = d_k
```

This explains the observed exact final-delta equality between
`power_replace__pure_jvp_vjp` and `power_replace__generalized_pq` in completed
`p=1` runs.

### power_add__affine_jvp_vjp

Let the code's radius scalar be

```text
rho_k = mean_batch ||delta_k||_p.
```

Then

```text
y_k = r(x_k) + rho_k J_k v_k
h_k = J_k^T phi_q(y_k)
d_k = s_p(h_k)
z_{k+1} = delta_k + alpha d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
v_{k+1} = d_k
```

### power_replace__affine_jvp_vjp

```text
y_k = r(x_k) + rho_k J_k v_k
h_k = J_k^T phi_q(y_k)
d_k = s_p(h_k)
z_{k+1} = epsilon d_k
delta_{k+1} = P_{p,epsilon}(z_{k+1})
v_{k+1} = d_k
```

## Why The First Seven Methods Are Highly Similar For p=2,q=2

Observed method order in `p=2,q=2/final_deltas.npz`:

```text
1. raw_add
2. unit_raw_add
3. raw_replace
4. steepest_add
5. steepest_replace
6. power_add__objective_gradient
7. power_replace__objective_gradient
8. power_add__pure_jvp_vjp
9. power_replace__pure_jvp_vjp
10. power_add__affine_jvp_vjp
11. power_replace__affine_jvp_vjp
```

The first seven methods are all objective-gradient methods. They do not use the
JVP/VJP operator-power direction. For `p=2`,

```text
s_2(g_k) = u_2(g_k) = g_k / ||g_k||_2.
```

Therefore these exact formula collapses happen:

```text
unit_raw_add = steepest_add = power_add__objective_gradient
raw_replace = steepest_replace = power_replace__objective_gradient
```

Observed final-delta similarity for `p=2,q=2` confirms this:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `unit_raw_add` vs `steepest_add` | 1.000000 | 0.000000 |
| `steepest_add` vs `power_add__objective_gradient` | 1.000000 | 0.000000 |
| `raw_replace` vs `steepest_replace` | 1.000000 | 0.000000 |
| `steepest_replace` vs `power_replace__objective_gradient` | 1.000000 | 0.000000 |

`raw_add` is not exactly the same because it uses the unnormalized gradient
`g_k`, not `g_k / ||g_k||_2`. But it is often similar because after projection
and many steps it tends to follow the same broad gradient direction:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `raw_add` vs `steepest_add` | 0.858049 | 0.384210 |

Additive and replacement updates are not identical even with the same direction
family:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `steepest_add` vs `steepest_replace` | 0.530025 | 0.830928 |

So if the loss curves look similar, that does not mean the final perturbations
are the same. For `p=2,q=2`, exact equality comes from duplicated formulas; high
but non-exact similarity comes from shared objective-gradient direction and the
same p-ball constraint.

## Why This Changes For p=1

For `p=1`,

```text
u_1(g) = g / ||g||_1
s_1(g) = one-hot coordinate at argmax_i |g_i|.
```

These are very different. That is why completed `p=1` runs do not show equality
between `unit_raw_add` and `steepest_add`.

Observed examples:

| Run | Pair | Mean cosine | Mean relative L2 |
|---|---|---:|---:|
| `p=1,q=1` | `unit_raw_add` vs `steepest_add` | 0.113356 | 1.878867 |
| `p=1,q=2` | `unit_raw_add` vs `steepest_add` | 0.143339 | 1.852929 |

Thus the high similarity among the first seven methods is mainly a `p=2`
collapse plus duplicate objective-gradient method definitions, not a universal
fact about every P/Q geometry.
