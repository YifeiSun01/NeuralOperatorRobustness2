# Loss3 Generalized Power Naming Clarification - 2026-05-18

## Why The Name Is Confusing

In this experiment family, the phrase `generalized power iteration` has been
used for two related but not identical ideas:

1. A replacement/boundary update rule.
2. A P-Q operator-power direction rule built with JVP/VJP.

The current ablation script intentionally separates these two axes:

```text
direction rule  +  proposal/update rule
```

Therefore, two methods can both look "generalized-power-like" but differ in how
the direction is computed.

## Previous Three-Method Implementation

Observed from `tools/run_batch_three_loss_loss_only.py`, the previous
three-method runner implements:

```python
if method == "pgd":
    direction = grad
    delta = project_delta(delta + alpha * direction, epsilon, p)
elif method == "lp_steepest_pgd":
    direction = steepest_direction(grad, p)
    delta = project_delta(delta + alpha * direction, epsilon, p)
elif method == "generalized_power":
    direction = steepest_direction(grad, p)
    delta = project_delta(epsilon * direction, epsilon, p)
```

Here `grad = autograd(loss3_q)`. Thus this implementation of
`generalized_power` is:

```text
objective-gradient steepest direction + replacement/boundary update
```

In the new ablation names, this is represented by:

```text
steepest_replace
power_replace__objective_gradient
```

For `p=2,q=2`, the final-delta similarity analysis observed that these are
numerically identical to each other, and `raw_replace` also collapses to the
same result because L2 normalized raw gradient equals the L2 steepest direction.

## JVP/VJP Power Variants

The names

```text
power_*__pure_jvp_vjp
power_*__affine_jvp_vjp
power_*__generalized_pq
```

refer to a different direction construction. They do not simply use
`grad = autograd(loss3_q)` as the direction source. Instead they maintain a
power state `v_k` and compute a local P-Q operator-power direction using the
Jacobian of the residual map:

```text
r(x) = model(x) - solver(x)
J = derivative of r at the current base point
J v_k                      # JVP
phi_q(J v_k)               # q-side dual map
J^T phi_q(J v_k)            # VJP
steepest_direction(..., p)  # p-side direction
```

The `affine_jvp_vjp` variant applies the q-side dual map to an affine residual
approximation:

```text
r_base + radius * J v_k
```

instead of only `J v_k`.

The current `generalized_pq` implementation takes the same code branch as
`pure_jvp_vjp`, so the completed `p=1` runs observed exact final-delta equality
between `power_replace__pure_jvp_vjp` and `power_replace__generalized_pq`.

## Practical Naming Rule

To avoid ambiguity, future summaries should use these names:

- `objective-gradient replacement`: the user's earlier three-method
  `generalized_power` implementation.
- `JVP/VJP operator-power`: the q-aware local P-Q power-direction variants.
- `additive` vs `replacement`: the proposal/update rule, independent of how the
  direction is computed.

## Current Interpretation

If the user says `generalized power iteration` and refers to the earlier
`pgd / lp_steepest_pgd / generalized_power` comparison, then the matching method
in the new ablation is `steepest_replace` / `power_replace__objective_gradient`,
not the JVP/VJP variants.

If the user means the mathematical P-Q induced operator norm power iteration,
then the matching methods are the `power_*__jvp_vjp` / `power_*__generalized_pq`
variants. Those are related, but they are not the same implementation as the
previous objective-gradient replacement update.
