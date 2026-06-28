# NS2D/Burgers Optimizer Winners And Equivalence Summary - 2026-05-24

Observed from saved reports, CSV files, and attack code only. No new GPU
experiment, model inference, solver rollout, or attack update was run for this
summary.

## Source Evidence

- NS2D active-target loss curves:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns2d_all_eps_optimizer_grouped_loss_curves_20260524.md`
- NS2D unified final true-loss winners:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/burgers_ns2d_existing_results_analysis_20260524_tables/ns2d_complete_group_winners.csv`
- Burgers optimizer-grouped active-target loss curves:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/optimizer_grouped_loss_curves_20260524.md`
- Burgers recorded key metrics:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/burgers_ns2d_existing_results_analysis_20260524_tables/burgers_recorded_key_metrics.csv`
- Method definitions:
  `/workspace/NeuralOperatorRobustness2_gitclean/2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`
- Existing equivalence notes:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/loss3_p2_q2_method_equivalence_summary_20260518.md`
  and
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/loss3_p_norm_method_equivalence_rules_20260518.md`

## Important Scope

There are two different "winner" questions:

1. Active-target winner: within a fixed loss target, which optimizer makes that
   same optimized loss largest at the final step.
2. Unified true-loss winner: after attacks are generated, evaluate all of them
   with the same final all-W true-loss metric.

The tables below keep these separate. The core four-method comparison is mostly
`p = 2, q = 2`. Burgers has many other epsilon/alpha records in R2/local notes,
but the locally evidenced complete same-protocol table with four optimizers and
loss1/loss2/loss3 is `epsilon = 4`, `alpha = 0.4`, `p = 2`, `q = 2`.

## NS2D Active-Target Winners

For complete four-method NS2D groups, `steepest_add` wins most often.

Complete four-method active-target winner count:

| method | wins |
|---|---:|
| Steepest Add | 16 / 26 |
| Raw Add | 9 / 26 |
| Raw Replace | 1 / 26 |
| Steepest Replace | 0 / 26 |

Detailed active-target winners:

| epsilon/alpha | target | winner | final active loss | methods present |
|---|---|---|---:|---:|
| eps160/alpha50 | Loss 1 / all W | Steepest Add | 511.784 | 1 |
| eps160/alpha50 | Loss 2 / all A -> W | Steepest Add | 311.971 | 1 |
| eps160/alpha50 | Loss 3 / all A -> W | Steepest Add | 487.022 | 1 |
| eps32/alpha10 | Loss 1 / all W | Steepest Add | 519.401 | 4 |
| eps32/alpha10 | Loss 2 / all A -> W | Raw Replace | 307.105 | 4 |
| eps32/alpha10 | Loss 3 / all W | Steepest Add | 304.593 | 4 |
| eps32/alpha10 | Loss 3 / all D -> W | Steepest Add | 216.414 | 4 |
| eps32/alpha10 | Loss 3 / W steps 1-5, D steps 6-9 -> W | Steepest Add | 207.858 | 4 |
| eps32/alpha10 | Loss 3 / D steps 1-5, W steps 6-9 -> W | Steepest Add | 289.227 | 4 |
| eps32/alpha10 | Loss 3 / A steps 1-5, D steps 6-9 -> W | Steepest Add | 237.705 | 4 |
| eps16/alpha5 | Loss 1 / all W | Steepest Add | 431.712 | 4 |
| eps16/alpha5 | Loss 2 / all A -> W | Raw Add | 314.166 | 4 |
| eps16/alpha5 | Loss 3 / all W | Steepest Add | 195.546 | 4 |
| eps8/alpha2.5 | Loss 1 / all W | Steepest Add | 304.813 | 4 |
| eps8/alpha2.5 | Loss 2 / all A -> W | Raw Add | 308.799 | 4 |
| eps8/alpha2.5 | Loss 3 / all W | Steepest Add | 139.456 | 4 |
| eps8/alpha2.5 | Loss 3 / all D -> W | Steepest Add | 84.7362 | 4 |
| eps8/alpha2.5 | Loss 3 / W steps 1-5, D steps 6-9 -> W | Steepest Add | 84.8339 | 4 |
| eps8/alpha2.5 | Loss 3 / D steps 1-5, W steps 6-9 -> W | Steepest Add | 132.805 | 4 |
| eps8/alpha2.5 | Loss 3 / A steps 1-5, D steps 6-9 -> W | Raw Add | 159.729 | 4 |
| eps4/alpha1.25 | Loss 1 / all W | Raw Add | 212.085 | 4 |
| eps4/alpha1.25 | Loss 2 / all A -> W | Raw Add | 320.921 | 4 |
| eps4/alpha1.25 | Loss 3 / all W | Steepest Add | 103.413 | 4 |
| eps2/alpha0.625 | Loss 1 / all W | Raw Add | 135.985 | 4 |
| eps2/alpha0.625 | Loss 2 / all A -> W | Raw Add | 320.658 | 4 |
| eps2/alpha0.625 | Loss 3 / all W | Steepest Add | 84.1793 | 4 |
| eps1/alpha0.3125 | Loss 1 / all W | Raw Add | 77.6735 | 4 |
| eps1/alpha0.3125 | Loss 2 / all A -> W | Raw Add | 320.527 | 4 |
| eps1/alpha0.3125 | Loss 3 / all W | Steepest Add | 75.6979 | 4 |

Observed evidence:

- `eps160/alpha50` is not a four-method comparison; only `steepest_add` is
  present locally for loss1, loss2, and loss3/all-A-to-W.
- For large/medium NS2D loss3 cases, `steepest_add` is the dominant
  active-target optimizer.
- At smaller epsilon, especially loss1/loss2, raw-add and steepest-add are often
  extremely close, and raw-add can be the numerical winner.
- `steepest_replace` has no unique active-target win in these complete NS2D
  four-method groups.

## NS2D Unified Final True-Loss Winners

Using the unified final all-W true-loss metric from
`ns2d_complete_group_winners.csv`, the winner count is:

| method | wins |
|---|---:|
| steepest_add | 15 / 17 |
| raw_replace | 2 / 17 |
| raw_add | 0 / 17 |
| steepest_replace | 0 / 17 |

Detailed unified true-loss winners:

| epsilon/alpha | target/mode | boundary-100 winner | final winner |
|---|---|---|---|
| eps32/alpha10 | loss1/all_w | steepest_add | steepest_add |
| eps32/alpha10 | loss2/all_a_target_w | raw_replace | steepest_add |
| eps32/alpha10 | loss3/all_w | steepest_add | steepest_add |
| eps32/alpha10 | loss3/all_d_target_w | steepest_add | steepest_add |
| eps32/alpha10 | loss3/w1_5_d6_9_target_w | steepest_add | steepest_add |
| eps32/alpha10 | loss3/d1_5_w6_9_target_w | steepest_add | steepest_add |
| eps32/alpha10 | loss3/a1_5_d6_9_target_w | steepest_add | steepest_add |
| eps16/alpha5 | loss1/all_w | raw_add | steepest_add |
| eps16/alpha5 | loss2/all_a_target_w | raw_replace | raw_replace |
| eps16/alpha5 | loss3/all_w | steepest_add | steepest_add |
| eps8/alpha2.5 | loss1/all_w | steepest_add | steepest_add |
| eps8/alpha2.5 | loss2/all_a_target_w | raw_add | raw_replace |
| eps8/alpha2.5 | loss3/all_w | steepest_add | steepest_add |
| eps8/alpha2.5 | loss3/all_d_target_w | steepest_add | steepest_add |
| eps8/alpha2.5 | loss3/w1_5_d6_9_target_w | steepest_add | steepest_add |
| eps8/alpha2.5 | loss3/d1_5_w6_9_target_w | steepest_add | steepest_add |
| eps8/alpha2.5 | loss3/a1_5_d6_9_target_w | steepest_add | steepest_add |

Inference from the observed NS2D tables: if the judging metric is final all-W
true-loss, `steepest_add` is the safest default in the currently evidenced
NS2D runs. Replacement can win specific loss2 cases, but it is not the dominant
final true-loss winner.

## Burgers Winners

The clean same-protocol Burgers comparison currently evidenced here is:
`epsilon = 4`, `alpha = 0.4`, `p = 2`, `q = 2`, four optimizers,
loss1/loss2/loss3.

Active-target final winners:

| target | winner | final active loss | comment |
|---|---|---:|---|
| Loss 1 / all W | Raw Add | 6.97733 | Steepest Add is nearly tied at 6.97673 |
| Loss 2 / all A -> W | Steepest Add | 7.07425 | Raw Add is close at 7.06981 |
| Loss 3 / all W | Raw Replace / Steepest Replace | 3.04661 | exact tie in this p=2 replacement group |

Observed from the recorded Burgers key metrics:

- In the corrected core-four `eps4/alpha0.4/p2q2` run, replacement reaches the
  boundary fastest.
- For loss1/loss2, additive methods catch up by the final step and are not
  worse by final loss.
- For loss3, replacement has a stronger endpoint in the recorded active-target
  curve, and the replacement trajectory rotates on the boundary.

Inference from these Burgers records: Burgers is not simply
"steepest_add always wins". For loss1/loss2, additive methods are enough and
nearly tied; for loss3, the replacement/GPI-like boundary jump is much more
competitive.

## Method Definitions In The Current Core-Four Code

Observed from `attack_ns2d_recurrent_core4.py`:

| method | direction rule | proposal rule |
|---|---|---|
| `raw_add` | raw gradient | `delta + alpha * direction`, then project |
| `raw_replace` | unit raw gradient | `epsilon * direction`, then project |
| `steepest_add` | Lp-steepest direction | `delta + alpha * direction`, then project |
| `steepest_replace` | Lp-steepest direction | `epsilon * direction`, then project |

The Lp-steepest direction is the maximizer of the linearized objective
`<grad, s>` subject to `||s||_p <= 1`.

## Equivalence Rules

Observed from the code and prior final-delta similarity notes:

### p = 2

For `p = 2`, the L2-steepest direction is exactly the L2-normalized raw
gradient:

`s = grad / ||grad||_2`.

Therefore:

- `raw_replace = steepest_replace` in the current core-four implementation,
  because both replace the perturbation with the same normalized direction on
  the L2 boundary.
- If `unit_raw_add` is included in an ablation, then
  `unit_raw_add = steepest_add`.
- `raw_add` is related but not identical to `steepest_add`: it uses raw
  gradient magnitude, while `steepest_add` uses normalized step length. They can
  be close after projection, but they are not duplicate methods.

Prior observed p2q2 final-delta similarity supports this:

| pair | mean cosine | mean relative L2 |
|---|---:|---:|
| `raw_replace` vs `steepest_replace` | 1.000000 | 0.000000 |
| `unit_raw_add` vs `steepest_add` | 1.000000 | 0.000000 |
| `raw_add` vs `steepest_add` | 0.858049 | 0.384210 |

### p != 2

For `p != 2`, normalized raw gradient is generally not the same as the
Lp-steepest direction.

Examples:

- For `p = inf`, the Lp-steepest direction is `sign(grad)`, not the raw
  normalized gradient.
- For `p = 1`, the Lp-steepest direction concentrates mass on the largest
  gradient coordinate, not the full raw-gradient vector.

Therefore `raw_replace` and `steepest_replace` should not be assumed identical
when `p != 2`, and `unit_raw_add` should not be assumed identical to
`steepest_add`.

Duplicate labels still remain if they are present in a given ablation:

- `steepest_add = power_add__objective_gradient`
- `steepest_replace = power_replace__objective_gradient`
- current `power_*__generalized_pq = power_*__pure_jvp_vjp`

### Role Of q

`q` changes the loss geometry and therefore changes the gradient being fed into
the optimizer. But once that gradient is fixed, the p-norm equivalence rule is
controlled by `p`.

So for `p = 2`, the normalized-gradient/steepest-direction equality still holds
for different `q`; the actual direction may change because the gradient from
the `q`-dependent objective changes.

## Bottom Line

Observed evidence:

- NS2D currently favors `steepest_add`, especially when judged by unified final
  all-W true-loss.
- Burgers `eps4/alpha0.4/p2q2` is more mixed: additive wins or nearly ties on
  loss1/loss2, while replacement wins loss3.
- For `p = 2`, several method labels are not independent algorithms. In
  particular, `raw_replace` and `steepest_replace` are the same update in the
  current code.

Inference:

- A fair paper-style comparison should not count p=2 `raw_replace` and
  `steepest_replace` as two independent pieces of evidence.
- The meaningful p=2 contrast is mostly additive raw magnitude vs additive
  normalized/steepest vs replacement boundary update.
- To make a stronger statement across `p/q`, more complete cross-`p/q`
  same-protocol tables are needed; existing evidence already shows that the
  p=2 equivalence breaks for p=1.
