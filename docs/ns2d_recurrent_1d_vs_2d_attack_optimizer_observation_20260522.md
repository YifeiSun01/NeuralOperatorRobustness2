# 1D Burgers vs 2D Navier-Stokes Attack Optimizer Observation - 2026-05-22

## Status

Recorded an experiment interpretation that the best adversarial-update rule may differ between the earlier 1D Burgers FNO experiments and the current 2D Navier-Stokes recurrent FNO experiments.

## Observed Evidence

Observed from the current 2D NS recurrent-FNO baseline attack block:

- Attack root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`.
- Settings: `epsilon=32`, `alpha=10`, `p=q=2`, `loss1`, `mode=all_w`, `attack_batch_size=10`, `steps=100`.
- Curves: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_overview_with_std.png`.
- Final-state panels: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522`.
- Detailed interpretation note: `docs/ns2d_recurrent_loss1_steepest_add_vs_replace_note_20260522.md`.

Observed final `loss1` summary for 2D NS baseline:

| Method | Final loss1 mean | Final true loss mean | True-loss increase from k=0 | Final boundary ratio |
|---|---:|---:|---:|---:|
| `raw_add` | 445.810025 | 93.041568 | 24.549836 | 0.999999982 |
| `raw_replace` | 401.017053 | 98.326353 | 29.834621 | 0.999999982 |
| `steepest_add` | 519.401230 | 158.283305 | 89.791573 | 0.997810042 |
| `steepest_replace` | 401.017053 | 98.326353 | 29.834621 | 0.999999982 |

Observed implementation detail:

- For `p=2`, `raw_replace` and `steepest_replace` are the same direction in the current code, because both use the L2-normalized gradient before replacement.
- `steepest_add` performs projected additive updates, so it can accumulate and rotate along the boundary rather than replacing the perturbation with the current boundary direction at every step.

Observed from the user's prior 1D Burgers experiments:

- The user's standing observation is that `steepest_replace` was usually the strongest method for 1D Burgers FNO attacks.
- This record treats the 1D Burgers statement as user-reported prior experimental evidence from this project context, not as a newly re-run local result in this turn.

## Interpretation

The current 2D result does not contradict the earlier 1D result. It suggests the optimizer ranking may depend on the PDE, dimension, rollout structure, perturbation scale, and loss definition.

Possible reasons:

- The 2D NS recurrent-FNO attack is more nonlinear than a one-step or simpler 1D setting. The perturbation affects a recurrent rollout and the solver trajectory, so the gradient field can rotate significantly as `delta` moves.
- `epsilon=32` is a finite, large-scale L2 budget for a `256 x 256` initial condition, not a tiny local perturbation. The local quadratic or local linearized intuition may be weak at this scale.
- Generalized power iteration is most directly justified for a fixed quadratic or fixed local-linear problem. The current `loss1 = ||F(x + delta) - F(x)||_2` passes through a nonlinear recurrent FNO, and the monitored true loss involves both FNO and solver outputs.
- Replacement updates can jump between boundary directions and discard path history. Additive projected updates can integrate direction information over multiple nonlinear states and may find a better boundary point.
- 1D Burgers may have a more dominant adversarial direction, where replacement/power-style updates work better; 2D NS may have more distributed or rotating unstable directions.

## Inference

The working conclusion is:

- For the prior 1D Burgers experiments, `steepest_replace` may remain the strongest observed optimizer.
- For the current 2D NS recurrent-FNO baseline block at `epsilon=32`, `alpha=10`, `loss1`, and `p=q=2`, `steepest_add` is the strongest observed method by final objective and true-loss growth.
- This difference is scientifically useful rather than merely a nuisance: it indicates that adversarial optimizer choice should be treated as PDE- and regime-dependent, not universal.

## Follow-Up Work

Recommended checks before turning this into a strong claim:

- Compare the same four optimizers for smaller epsilon values, especially `epsilon=8` and `epsilon=16`.
- Compare across `loss2` and `loss3`, because the solver-coupled objectives may change the ranking.
- Add sample-wise plots to check whether `steepest_add` wins broadly or is driven by a few samples.
- Run a fixed-linearized or JVP-only diagnostic for 2D NS; if the local quadratic theory is applicable there, replacement/power-style methods should become more competitive.
- Revisit the 1D Burgers records and document the exact settings where `steepest_replace` was strongest, so the comparison is apples-to-apples.
