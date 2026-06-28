# Loss3 Final Delta Similarity Results - 2026-05-18

## Scope

Observed completed runs analyzed:

- `p=2,q=2`: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`
- `p=1,q=1`: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1`
- `p=1,q=2`: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2`
- `p=1,q=inf`: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_qinf`
- `p=2,q=1`: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q1`

Observed not-yet-analyzed run:

- `p=2,q=inf` had manifest status `run_started` at the later status check, so it
  was not included yet.

Analysis script:

- `/workspace/NeuralOperatorRobustness2/tools/analyze_final_delta_similarity.py`

The script reads each completed run's `final_deltas.npz`. It does not rerun the
optimizer, model, or solver.

## Metrics

For each pair of methods and each dataset sample, the final perturbations are
flattened and compared.

- `cosine`: L2 cosine similarity between final deltas. `1` means same direction
  and shape, `0` means roughly orthogonal, negative means opposite direction.
- `relative_l2_distance`: `||delta_a - delta_b||_2 / (0.5*(||delta_a||_2 + ||delta_b||_2))`.
  `0` means identical final deltas; larger values mean more different final
  perturbation shapes and/or magnitudes.

The reported means/stds are across the 100 samples in the batch, not across
random seeds.

## Output Files

For each completed run, the new analysis directory is:

```text
final_delta_similarity/20260518_final_delta_similarity/
```

Each directory contains:

- `final_delta_pairwise_similarity_summary.csv`
- `final_delta_pairwise_similarity_per_sample.csv`
- `final_delta_cosine_mean_matrix.csv`
- `final_delta_cosine_std_matrix.csv`
- `final_delta_relative_l2_mean_matrix.csv`
- `figures/final_delta_cosine_mean_heatmap_all_methods.png`
- `figures/final_delta_relative_l2_mean_heatmap_all_methods.png`
- `figures/final_delta_cosine_mean_heatmap_core_methods.png`
- `figures/final_delta_relative_l2_mean_heatmap_core_methods.png`

Existing figures were not deleted or overwritten.

## Key Observations

### p=2, q=2

Observed exact final-delta equivalences:

- `unit_raw_add == steepest_add`
- `unit_raw_add == power_add__objective_gradient`
- `steepest_add == power_add__objective_gradient`
- `raw_replace == steepest_replace`
- `raw_replace == power_replace__objective_gradient`
- `steepest_replace == power_replace__objective_gradient`

Observed selected pairwise similarities:

| Method A | Method B | Mean cosine | Std cosine | Mean relative L2 |
|---|---|---:|---:|---:|
| `unit_raw_add` | `steepest_add` | 1.000000 | 0.000000 | 0.000000 |
| `steepest_add` | `power_add__objective_gradient` | 1.000000 | 0.000000 | 0.000000 |
| `steepest_replace` | `power_replace__objective_gradient` | 1.000000 | 0.000000 | 0.000000 |
| `raw_add` | `steepest_add` | 0.858049 | 0.197407 | 0.384210 |
| `steepest_add` | `steepest_replace` | 0.530025 | 0.444889 | 0.830928 |

Inference: for `p=2`, several method labels collapse to the same final delta
because the L2 steepest direction is the normalized raw gradient. The
`objective_gradient` power variants also use the same gradient-induced steepest
direction as the corresponding steepest method. Therefore their identical plots
are expected, not a separate empirical discovery.

The more meaningful `p=2,q=2` distinction is additive vs replacement. Even when
using the same direction family, `steepest_add` and `steepest_replace` have only
mean cosine `0.530025`, so they do not generally end at the same perturbation.

### p=1, q=1

Observed exact final-delta equivalences:

- `steepest_replace == power_replace__objective_gradient`
- `power_replace__pure_jvp_vjp == power_replace__generalized_pq`

Observed selected pairwise similarities:

| Method A | Method B | Mean cosine | Std cosine | Mean relative L2 |
|---|---|---:|---:|---:|
| `steepest_replace` | `power_replace__objective_gradient` | 1.000000 | 0.000000 | 0.000000 |
| `power_replace__pure_jvp_vjp` | `power_replace__generalized_pq` | 1.000000 | 0.000000 | 0.000000 |
| `raw_add` | `unit_raw_add` | 0.984738 | not listed here | 0.108011 |
| `raw_add` | `steepest_add` | 0.110365 | 0.041943 | 1.877980 |
| `steepest_add` | `steepest_replace` | 0.102297 | 0.283222 | 1.303215 |

Inference: under `p=1`, the steepest direction is no longer the same as a
normalized raw gradient. This is why `unit_raw_add` and `steepest_add` are not
similar here, unlike the `p=2` case.

### p=1, q=2

Observed exact final-delta equivalences:

- `steepest_replace == power_replace__objective_gradient`
- `power_replace__pure_jvp_vjp == power_replace__generalized_pq`

Observed selected pairwise similarities:

| Method A | Method B | Mean cosine | Std cosine | Mean relative L2 |
|---|---|---:|---:|---:|
| `steepest_replace` | `power_replace__objective_gradient` | 1.000000 | 0.000000 | 0.000000 |
| `power_replace__pure_jvp_vjp` | `power_replace__generalized_pq` | 1.000000 | 0.000000 | 0.000000 |
| `raw_add` | `unit_raw_add` | 0.997332 | not listed here | 0.047764 |
| `raw_add` | `steepest_add` | 0.142530 | 0.054330 | 1.851740 |
| `steepest_add` | `steepest_replace` | 0.167684 | 0.363200 | 1.220563 |

Inference: the same p=1 behavior appears with q=2: normalized raw-gradient
additive updates are close to raw PGD, but L1-steepest updates produce very
different final perturbations.



### Additional Completed Runs Added Later

After the initial report, `p=1,q=inf` and `p=2,q=1` completed and were analyzed
with the same tag `20260518_final_delta_similarity`.

Observed selected pairwise similarities:

| Run | Pair | Mean cosine | Mean relative L2 | Interpretation |
|---|---|---:|---:|---|
| `p=1,q=inf` | `unit_raw_add` vs `steepest_add` | `0.344037` | `1.776665` | not equivalent |
| `p=1,q=inf` | `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` | duplicate label |
| `p=1,q=inf` | `power_replace__pure_jvp_vjp` vs `power_replace__generalized_pq` | `1.000000` | `0.000000` | same current implementation |
| `p=2,q=1` | `unit_raw_add` vs `steepest_add` | `1.000000` | `0.000000` | p=2 identity |
| `p=2,q=1` | `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` | duplicate label |
| `p=2,q=1` | `power_replace__pure_jvp_vjp` vs `power_replace__generalized_pq` | `1.000000` | `0.000000` | same current implementation |

The general p-norm equivalence rule is recorded in
`docs/loss3_p_norm_method_equivalence_rules_20260518.md`.

## Scientific Takeaway

Observed evidence supports the user's concern: several method labels really are
redundant in the completed outputs.

For `p=2,q=2`, these are effectively the same final-perturbation method groups:

- additive normalized-gradient group: `unit_raw_add`, `steepest_add`,
  `power_add__objective_gradient`
- replacement normalized-gradient/steepest group: `raw_replace`,
  `steepest_replace`, `power_replace__objective_gradient`

For `p=1`, this collapse does not happen between normalized raw gradient and
Lp-steepest directions. The p-norm geometry matters: `raw_add`/`unit_raw_add`
can be very similar, while `steepest_add` is very different.

Therefore, future presentation figures should avoid plotting every redundant
variant together. A cleaner comparison should use one representative per
mathematically equivalent group, then separately show the q-aware JVP/VJP power
variants only when studying P/Q geometry.

## Dedicated p=2,q=2 Equivalence Summary

A concise interpretation of the crowded `p=2,q=2` similarity heatmap is recorded
in `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`. The key point is
that `unit_raw_add`, `steepest_add`, and `power_add__objective_gradient` are the
same update for `p=2`, and `raw_replace`, `steepest_replace`, and
`power_replace__objective_gradient` are also the same update for `p=2`.

## P-Norm Equivalence Rule Note

The general rule comparing `p=2` and `p!=2` method equivalences is recorded in
`docs/loss3_p_norm_method_equivalence_rules_20260518.md`.

