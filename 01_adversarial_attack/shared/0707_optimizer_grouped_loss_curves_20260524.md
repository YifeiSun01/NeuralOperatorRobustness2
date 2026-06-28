# Optimizer-Grouped Loss Curves - 2026-05-24

Observed from saved `per_step_metrics.csv` files. No model inference, solver
rollout, attack update, PyTorch, JAX, or GPU work was run for this plotting
task.

## Outputs

- Burgers PNG:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/optimizer_grouped_loss_curves_20260524/burgers_optimizer_grouped_target_loss_curves_eps4_alpha0p4.png`
- NS2D PNG:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/optimizer_grouped_loss_curves_20260524/ns2d_optimizer_grouped_target_loss_curves_eps32_vs_eps1.png`
- Detailed report:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/optimizer_grouped_loss_curves_20260524.md`
- Bundle:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz`

## Plot Definition

Each panel fixes the target loss and overlays the four optimizer methods:
`raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.

- Burgers uses `epsilon = 4`, `alpha = 0.4`, `p = 2`, `q = 2`.
- NS2D compares `epsilon = 32, alpha = 10` against `epsilon = 1,
  alpha = 0.3125`.

## Interpretation

Inference from the plotted curves and update rule: with `p = 2`, steepest
direction is the L2-normalized gradient direction. The large oscillation of
`steepest_replace` is mainly caused by the `replace` rule discarding the
previous perturbation direction. If the gradient direction rotates from one step
to the next, the perturbation jumps to a different boundary direction. The
`add` variants preserve previous perturbation information and therefore usually
look smoother. At small epsilon the problem is closer to a local linear regime,
so method differences often shrink.
