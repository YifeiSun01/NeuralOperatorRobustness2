# Loss3 P-Norm Method Equivalence Rules - 2026-05-18

## Short Answer

For `p=2`, there are extra exact equivalences because normalized raw gradient is
the same direction as the L2-steepest direction.

For `p != 2`, those extra equivalences disappear. The remaining exact
equivalences are mostly duplicate method labels in the current ablation code.

## Always-True Duplicate Labels In This Code

These equivalences are true for any `p`, because the current code defines them
with the same formula.

| Duplicate methods | Reason |
|---|---|
| `steepest_add = power_add__objective_gradient` | both use Lp-steepest direction from the objective gradient, then additive update |
| `steepest_replace = power_replace__objective_gradient` | both use Lp-steepest direction from the objective gradient, then replacement update |
| `power_*__generalized_pq = power_*__pure_jvp_vjp` | current `generalized_pq` branch uses the same implementation as `pure_jvp_vjp` |

Important: some runs do not include all these labels. For example, the `pq_key`
runs include `steepest_replace` and `power_replace__objective_gradient`, but do
not include `power_add__objective_gradient`.

## Extra Equivalences When p=2

When `p=2`, normalized raw gradient equals L2-steepest direction.

Therefore, for `p=2`, if all labels are present, the objective-gradient methods
collapse into two groups of three.

Additive group:

| Equivalent methods |
|---|
| `unit_raw_add = steepest_add = power_add__objective_gradient` |

Replacement group:

| Equivalent methods |
|---|
| `raw_replace = steepest_replace = power_replace__objective_gradient` |

Each group of three creates three pairwise cosine-1 entries in the similarity
matrix. So if all six labels are present, there are six exact pairwise matches
from these two groups alone.

Observed in completed `p=2,q=2`:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `unit_raw_add` vs `steepest_add` | `1.000000` | `0.000000` |
| `unit_raw_add` vs `power_add__objective_gradient` | `1.000000` | `0.000000` |
| `steepest_add` vs `power_add__objective_gradient` | `1.000000` | `0.000000` |
| `raw_replace` vs `steepest_replace` | `1.000000` | `0.000000` |
| `raw_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` |
| `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` |

Observed in completed `p=2,q=1` with the smaller `pq_key` method set:

| Pair | Mean cosine | Mean relative L2 |
|---|---:|---:|
| `unit_raw_add` vs `steepest_add` | `1.000000` | `0.000000` |
| `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` |
| `power_replace__pure_jvp_vjp` vs `power_replace__generalized_pq` | `1.000000` | `0.000000` |

The `p=2,q=1` run did not include `raw_replace` or
`power_add__objective_gradient`, so the full two-groups-of-three pattern is not
visible there even though the p=2 rule still applies.

## What Happens When p != 2

For `p != 2`, normalized raw-gradient direction is not generally the same as the
Lp-steepest direction.

Therefore these are generally not identical:

| Not generally identical when p != 2 |
|---|
| `unit_raw_add` and `steepest_add` |
| `raw_replace` and `steepest_replace` |

The duplicate-label equivalences still remain:

| Still identical when p != 2 |
|---|
| `steepest_add = power_add__objective_gradient`, if both are run |
| `steepest_replace = power_replace__objective_gradient`, if both are run |
| `power_*__generalized_pq = power_*__pure_jvp_vjp`, in the current implementation |

Observed completed `p=1` runs:

| Run | Pair | Mean cosine | Mean relative L2 | Interpretation |
|---|---|---:|---:|---|
| `p=1,q=1` | `unit_raw_add` vs `steepest_add` | `0.113356` | `1.878867` | not equivalent |
| `p=1,q=2` | `unit_raw_add` vs `steepest_add` | `0.143339` | `1.852929` | not equivalent |
| `p=1,q=inf` | `unit_raw_add` vs `steepest_add` | `0.344037` | `1.776665` | not equivalent |
| `p=1,q=1` | `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` | duplicate label |
| `p=1,q=2` | `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` | duplicate label |
| `p=1,q=inf` | `steepest_replace` vs `power_replace__objective_gradient` | `1.000000` | `0.000000` | duplicate label |
| `p=1,q=1` | `power_replace__pure_jvp_vjp` vs `power_replace__generalized_pq` | `1.000000` | `0.000000` | same current implementation |
| `p=1,q=2` | `power_replace__pure_jvp_vjp` vs `power_replace__generalized_pq` | `1.000000` | `0.000000` | same current implementation |
| `p=1,q=inf` | `power_replace__pure_jvp_vjp` vs `power_replace__generalized_pq` | `1.000000` | `0.000000` | same current implementation |

## Corrected Interpretation

It is not quite correct to say that `p != 2` gives "three methods all the same"
inside the raw/steepest/objective-gradient six-method family.

The cleaner rule is:

- For `p=2`: normalized raw-gradient direction equals steepest direction, so
  the objective-gradient family collapses into larger equivalence groups.
- For `p != 2`: normalized raw-gradient and steepest direction usually separate,
  so there are fewer exact similarities.
- Exact similarities that remain for `p != 2` mostly come from duplicate method
  labels, such as `steepest_replace = power_replace__objective_gradient`.

## Current Queue Note

At the time this note was written, `p=2,q=inf` was still running. Its final-delta
similarity should be analyzed after the manifest reports completion and
`final_deltas.npz` exists.
