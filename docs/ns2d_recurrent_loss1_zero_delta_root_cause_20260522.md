# NS2D Recurrent loss1 Zero-Delta Root Cause - 2026-05-22

## Problem

The completed `loss1/canonical_solver_context` attack produced zero perturbation for all four update methods:

```text
raw_add          final delta L2 = 0, final delta Linf = 0
raw_replace      final delta L2 = 0, final delta Linf = 0
steepest_add     final delta L2 = 0, final delta Linf = 0
steepest_replace final delta L2 = 0, final delta Linf = 0
```

## Observed Evidence

Observed from each `per_step_metrics.csv` under:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_wwwwwwwwww_p2_q2_20260522_015452_UTC/batch_0000_0009/loss1/
```

For every method and every recorded step:

```text
loss1_mean = 0.0
delta_p_mean = 0.0
boundary_ratio_mean = 0.0
grad_l2_mean = 0.0
direction_l2_mean = 0.0
```

At final step `k=100`, `final_delta_and_metrics.npz` confirms `final_delta` is exactly zero.

## Code Path

Observed in `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`:

```text
run_one: delta = torch.zeros_like(problem.x0)
active_losses: loss1 = batch_norm(pred - self.f0, q_order)
__init__: self.f0 = self.model_prediction(self.x0, need_target=False)[0].detach()
```

At step `k=0`:

```text
delta = 0
x_adv = x0 + delta = x0
pred = F(x0)
f0 = F(x0)
loss1 = ||pred - f0|| = 0
```

PyTorch's gradient/subgradient of this norm at the exact zero residual is zero in this implementation. Therefore:

```text
grad = 0
direction = 0
```

The four update rules then all remain stuck at zero:

```text
raw_add:          delta <- Proj(delta + alpha * grad) = 0
raw_replace:      delta <- Proj(epsilon * normalize(grad)) = 0
steepest_add:     delta <- Proj(delta + alpha * s_p(grad)) = 0
steepest_replace: delta <- Proj(epsilon * s_p(grad)) = 0
```

Once the first update is zero, the same state repeats for all 100 steps.

## Conclusion

This is abnormal as an attack result, but it is explainable from the current algorithm:

- It is not because `epsilon=32` is too small.
- It is not because `alpha=1` is too small.
- It is not because projection clipped the perturbation away.
- It is because the `loss1` objective starts at an exact zero residual with zero gradient when initialized at `delta=0`.

## Recommended Fixes For Future loss1 Runs

Use one of these, preferably the first:

1. **Random start inside the L2 ball**: initialize `delta` to a small random nonzero vector, then project it into the epsilon ball before step 0. This breaks the zero-gradient fixed point.
2. **Seed direction**: for replacement-style methods, if `grad` is zero at `k=0`, use a random normalized direction for the first step.
3. **Finite-difference or power-iteration start**: estimate a nonzero local direction before the gradient attack loop.
4. **Objective change**: avoid an objective whose residual is exactly zero at initialization, or optimize a local linearization/Jacobian-based objective for `loss1`.

For the current run, the already completed `loss1` group should not be treated as a successful perturbation result. It is evidence that the current zero-initialized `loss1` attack stalls.
