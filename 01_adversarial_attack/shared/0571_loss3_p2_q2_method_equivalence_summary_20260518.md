# Loss3 p=2,q=2 Method Equivalence Summary - 2026-05-18

## Purpose

This note records the interpretation of the `p=2,q=2` final-delta similarity
heatmap. It explains why several methods have cosine similarity exactly `1` and
why `raw_add` is highly similar but not identical to the normalized/steepest
additive group.

Source run:

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`

Source numeric table:

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/final_delta_similarity/20260518_final_delta_similarity/final_delta_pairwise_similarity_summary.csv`

Source heatmap:

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/final_delta_similarity/20260518_final_delta_similarity/figures/final_delta_cosine_mean_heatmap_all_methods.png`

## Method Name Clarification

The method named `raw_replace` is a normalized raw-gradient replacement method.
So when reading the heatmap, `raw_replace` means what one might call "unit raw
replace": it uses the normalized raw-gradient direction and then performs a
replacement/boundary update.

## Exact Equivalence Groups For p=2

For `p=2`, the L2-steepest direction equals the L2-normalized raw gradient.
Therefore, several method labels collapse to the same update formula.

Additive normalized-gradient group:

| Methods | Observed final-delta result |
|---|---|
| `unit_raw_add`, `steepest_add`, `power_add__objective_gradient` | Identical final deltas; pairwise cosine `1.000000`, relative L2 `0.000000` |

Replacement normalized-gradient group:

| Methods | Observed final-delta result |
|---|---|
| `raw_replace`, `steepest_replace`, `power_replace__objective_gradient` | Identical final deltas; pairwise cosine `1.000000`, relative L2 `0.000000` |

## Selected Observed Similarities

| Pair | Mean cosine | Mean relative L2 | Interpretation |
|---|---:|---:|---|
| `unit_raw_add` vs `steepest_add` | `1.000000` | `0.000000` | Same formula for `p=2` |
| `unit_raw_add` vs `power_add__objective_gradient` | `1.000000` | `0.000000` | Same formula for `p=2` |
| `steepest_add` vs `power_add__objective_gradient` | `1.000000` | `0.000000` | Same formula for `p=2` |
| `raw_replace` vs `steepest_replace` | `1.000000` | `0.000000` | Same formula for `p=2` |
| `raw_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` | Same formula for `p=2` |
| `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` | Same formula for `p=2` |
| `raw_add` vs `steepest_add` | `0.858049` | `0.384210` | Same broad gradient field, but `raw_add` uses unnormalized gradient |
| `steepest_add` vs `steepest_replace` | `0.530025` | `0.830928` | Same direction family, different proposal rule: additive vs replacement |

## Why `raw_add` Is High But Not 1

`raw_add` uses the raw gradient magnitude. The normalized/steepest additive group
uses only the normalized L2 direction. Under projection, these often move in a
similar broad direction, which explains the high cosine around `0.858`. But the
step scaling is not identical, so the final deltas are not exactly the same.

## Practical Conclusion

For `p=2,q=2`, the crowded similarity heatmap should not be interpreted as many
independent optimizer methods. Several entries are duplicate labels for the same
mathematical update.

For presentation or first-pass scientific interpretation, use one representative
from each exact-equivalence group:

| Concept | Representative method |
|---|---|
| Raw PGD additive | `raw_add` |
| Normalized/steepest additive | `steepest_add` or `unit_raw_add` |
| Normalized/steepest replacement / objective-gradient generalized power | `steepest_replace` |
| JVP/VJP operator-power variants | Keep separate only when studying q-aware P-Q power directions |

The main p=2 comparison should therefore focus on:

- `raw_add`
- one of `unit_raw_add`, `steepest_add`, or `power_add__objective_gradient`
- one of `raw_replace`, `steepest_replace`, or `power_replace__objective_gradient`
- selected JVP/VJP variants if the question is specifically about q-aware
  operator-power directions

## Relation To p=1 Results

This exact collapse is a `p=2` effect. For `p=1`, normalized raw-gradient
direction and L1-steepest direction are not the same. Existing completed `p=1`
runs show that `unit_raw_add` and `steepest_add` are not similar in final delta.

## P-Norm Equivalence Rule Note

The general rule comparing `p=2` and `p!=2` method equivalences is recorded in
`docs/loss3_p_norm_method_equivalence_rules_20260518.md`.

## Important Clarification: Why raw_replace Equals steepest_replace Only For p=2

`raw_replace` should not be read as ordinary PGD additive walking until it hits
the boundary. In this ablation, `raw_replace` is a boundary replacement method
using the normalized raw gradient direction.

So its direction is:

`normalized g_k`.

`steepest_replace` uses the Lp-steepest direction:

`s_k`.

In general these are not the same direction. But for `p=2`, the L2-steepest
direction is exactly the normalized gradient direction:

`s_k = g_k / ||g_k||_2`.

Therefore, in the `p=2,q=2` run:

`raw_replace = steepest_replace`.

This equality is not saying that raw-gradient PGD and steepest replacement are
always the same. It is saying that after replacing the update by a direct
boundary jump and using `p=2`, the normalized raw-gradient direction and the
L2-steepest direction coincide.

For `p != 2`, `normalized g_k` and `s_k` are generally different, so
`raw_replace` and `steepest_replace` should not be assumed equal.

