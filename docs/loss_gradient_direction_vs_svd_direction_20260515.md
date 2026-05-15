# Loss Gradient Direction vs SVD Direction

Date: 2026-05-15

This note records the key clarification reached after the local Jacobian/SVD
experiments: the top singular-vector direction is not the same object as the
one-step gradient direction used by a loss optimizer.

## Notation

At a fixed clean input `x`,

```text
f = learned model
j = solver / oracle
b = f(x) - j(x)
J_f = d f(x) / d x
J_j = d j(x) / d x
J_e = J_f - J_j
delta = input perturbation
```

The local linear approximations are:

```text
f(x + delta) - f(x) ~= J_f delta
j(x + delta) - j(x) ~= J_j delta
e(x + delta) ~= b + J_e delta
```

## The Three Base Losses

```text
loss1(delta) = ||f(x + delta) - f(x)||
             ~= ||J_f delta||

loss2(delta) = ||f(x + delta) - j(x)||
             ~= ||b + J_f delta||

loss3(delta) = ||f(x + delta) - j(x + delta)||
             ~= ||b + J_e delta||
```

For squared L2 local objectives:

```text
L1_sq(delta) ~= ||J_f delta||^2
grad L1_sq = 2 J_f^T J_f delta

L2_sq(delta) ~= ||b + J_f delta||^2
grad L2_sq = 2 J_f^T b + 2 J_f^T J_f delta

L3_sq(delta) ~= ||b + J_e delta||^2
grad L3_sq = 2 J_e^T b + 2 J_e^T J_e delta
```

With a half-squared convention the factor `2` disappears.  For unsquared L2,
the gradient contains the same geometric information but is normalized by the
current output residual norm.

## What The SVD Direction Means

For a Jacobian `J`, the top right singular vector solves:

```text
v_top = argmax_{||v||=1} ||J v||
```

Equivalently:

```text
J^T J v_top = sigma_top^2 v_top
```

So the top singular vector is the constrained optimum of a homogeneous
quadratic gain problem.  It is also the limiting direction of a normalized power
iteration such as:

```text
delta <- J^T J delta
delta <- delta / ||delta||
```

It is not generally the same as the one-step gradient at an arbitrary current
`delta_k`.

## What The One-Step Gradient Direction Means

At the current attack iterate `delta_k`, the local one-step gradient directions
are:

```text
g1(delta_k) = J_f^T J_f delta_k

g2(delta_k) = J_f^T b + J_f^T J_f delta_k

g3(delta_k) = J_e^T b + J_e^T J_e delta_k
```

These answer a different question:

```text
Given the current delta_k, which infinitesimal step most increases the chosen loss?
```

Only after repeated normalized updates on the pure homogeneous quadratic
`||J delta||^2` can the direction converge toward the top singular vector.

At `delta=0`:

```text
g1(0) = 0
g2(0) = J_f^T b
g3(0) = J_e^T b
```

This explains why `loss1` needs a nonzero/random start in the squared-L2 local
view, while `loss2` and `loss3` have clean-point outward directions determined
by the current residual `b`.

## What Was Already Computed

The existing local Jacobian/SVD experiments computed these directions:

| direction | status | mathematical object | meaning |
|---|---|---|---|
| `v_f` | computed | top right singular vectors of `J_f` | model-sensitivity / local `loss1` quadratic gain |
| `v_j` | computed | top right singular vectors of `J_j` | solver/oracle sensitivity |
| `v_e` | computed | top right singular vectors of `J_e=J_f-J_j` | pure mismatch / residual-increment gain |

Those results answer:

```text
Are the dominant local gain directions of J_f, J_j, and J_e similar?
```

They do not by themselves answer:

```text
Are the one-step optimizer gradients of loss1, loss2, and loss3 similar at the
same current delta_k?
```

## What Has Not Yet Been Computed

The following loss-gradient directions are still missing as explicit numerical
diagnostics:

| direction | formula | meaning |
|---|---|---|
| `g1(delta_k)` | `J_f^T J_f delta_k` | one-step local gradient of squared `loss1` |
| `g2(delta_k)` | `J_f^T b + J_f^T J_f delta_k` | one-step local gradient of squared `loss2` |
| `g3(delta_k)` | `J_e^T b + J_e^T J_e delta_k` | one-step local gradient of squared `loss3` |
| `g2(0)` | `J_f^T b` | clean-point outward direction for `loss2` |
| `g3(0)` | `J_e^T b` | clean-point outward direction for `loss3` |

The most direct next diagnostic is to compute all of these on the same five
indices used in the SVD comparison, then compare:

```text
cos(g1, g2)
cos(g1, g3)
cos(g2, g3)
cos(g1, v_f)
cos(g2, v_f)
cos(g2, v_e)
cos(g3, v_f)
cos(g3, v_e)
```

and also compare their frequency metrics such as `hi128`, zero crossings, and
roughness.

## Key Interpretation

The SVD experiment and the loss-gradient experiment are related but distinct:

```text
SVD/Jacobian experiment:
  Treat J_f, J_j, and J_e as local linear maps.
  Ask whether their dominant quadratic gain directions are similar.

Loss-gradient experiment:
  Fix a current perturbation delta_k.
  Ask whether loss1, loss2, and loss3 would take the same next optimizer step.
```

For `loss1`, the pure quadratic gain direction can converge to `v_f` after
repeated normalized updates.  But the one-step gradient at a finite `delta_k` is
`J_f^T J_f delta_k`, which need not already be `v_f`.

For `loss2` and `loss3`, the clean residual `b` adds a linear outward term.
Therefore their one-step directions can differ from the SVD directions even
more strongly:

```text
loss2: J_f^T b + J_f^T J_f delta_k
loss3: J_e^T b + J_e^T J_e delta_k
```

This is why the previously computed `v_f`, `v_j`, and `v_e` directions should be
described as local quadratic/Jacobian mechanism directions, not as the full
one-step gradient directions of `loss1`, `loss2`, and `loss3`.

## Practical Next Step

Use the saved explicit `1024 x 1024` Jacobians from the SVD artifacts and
recover the clean residual `b=f(x)-j(x)` for each sample.  Then compute
`g1`, `g2`, and `g3` for a shared set of current perturbations:

```text
delta_k = 0
delta_k = the shared tiny random loss1 start
delta_k = final PGD deltas from loss1/loss2/loss3, if available
```

The clean-point comparison should focus on `g2(0)=J_f^T b` and
`g3(0)=J_e^T b`; `g1(0)=0` is a degenerate case.
