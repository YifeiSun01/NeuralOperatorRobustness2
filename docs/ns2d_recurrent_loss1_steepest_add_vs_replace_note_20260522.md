# NS2D Recurrent FNO Loss1: Why Steepest Add Beat Replace - 2026-05-22

## Status

Inspected the completed baseline `epsilon=32, alpha=10`, `loss1/all_w` block after the user questioned why `steepest_add` produced larger objective/true-loss growth than `steepest_replace`.

## Observed Evidence

Source loss root:
`2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`

Source code inspected:
`2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`

Relevant implementation facts:

- `steepest_direction(grad, p_order=2)` returns the L2-normalized gradient.
- `raw_replace` uses `unit_raw_gradient` plus replacement.
- `steepest_replace` uses `lp_steepest` plus replacement.
- Therefore for `p=2`, `raw_replace` and `steepest_replace` are mathematically the same update direction in the current code.
- `replacement` proposes `delta = epsilon * direction`.
- `additive` proposes `delta = delta + alpha * direction`, then projects onto the epsilon ball.
- Metrics are recorded at the current `delta`; after recording, the projected proposal becomes the next step's `delta`.

Observed key metrics:

| Method | k=1 loss1 | k=1 true | k=1 boundary | k=100 loss1 | k=100 true | k=100 boundary |
|---|---:|---:|---:|---:|---:|---:|
| `raw_add` | 449.930322 | 92.174823 | 1.000000006 | 445.810025 | 93.041568 | 0.999999982 |
| `raw_replace` | 449.930487 | 92.175734 | 0.999999982 | 401.017053 | 98.326353 | 0.999999982 |
| `steepest_add` | 292.817056 | 71.224190 | 0.312507173 | 519.401230 | 158.283305 | 0.997810042 |
| `steepest_replace` | 449.930487 | 92.175734 | 0.999999982 | 401.017053 | 98.326353 | 0.999999982 |

## Interpretation

The observation is not automatically a code error.

For `p=q=2`, `raw_replace` and `steepest_replace` being identical is expected in this implementation. The apparent report value that replacement reaches exactly 100% boundary at step 61 is a strict floating-point threshold artifact: at step 1 its mean boundary ratio is already `0.999999982`, which is effectively on the boundary.

The generalized power iteration intuition applies to a fixed linear/quadratic local problem, such as maximizing a Rayleigh quotient or a fixed `delta^T A delta` over an L2 sphere. The current attack is not that problem:

- `loss1 = ||F(x + delta) - F(x)||_2` is evaluated through the nonlinear recurrent FNO.
- `epsilon=32` is not a tiny local perturbation, so the Jacobian/Hessian field changes a lot as `delta` moves.
- The plotted true loss is also different from the loss1 surrogate because it compares FNO and solver outputs.
- Replacement discards the previous `delta` every step and jumps to the current normalized gradient direction. In a nonlinear, twisting gradient field, that can oscillate or settle into a less useful boundary direction.
- `steepest_add` behaves like projected gradient ascent with normalized steps. It accumulates direction history and, once near the boundary, rotates the boundary point gradually instead of replacing it wholesale. In this observed block, that reached a better final nonlinear region.

## Inference

The completed `loss1` baseline result suggests `steepest_add` is currently more effective for this nonlinear finite-epsilon loss1 attack than the replacement update. This does not disprove the generalized-power intuition for a true fixed quadratic/local-linearized problem; it means this experiment is already outside that idealized regime.

## Suggested Follow-Up Checks

- Run the same comparison at smaller epsilon values, especially `epsilon=8` and `epsilon=16`; replacement/GPI should be more competitive if the objective is closer to local-linear.
- Add a fixed-linearized/JVP-only loss1 diagnostic to test whether replacement wins under the actual quadratic approximation.
- Plot sample-wise curves, not only batch means, to see whether `steepest_add` wins broadly or is driven by a few samples.
- Consider using squared loss for a dedicated power-method diagnostic, because the current norm loss is not exactly the same presentation as a Rayleigh quotient.
