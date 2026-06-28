# Loss3 Minimal Core Question Answer - 2026-05-18

## The Actual Question

The core question is simple:

If ordinary PGD uses this update:

PGD additive:

new delta = old delta + alpha times gradient, then project

what happens if we replace it with this update:

PGD boundary replacement:

new delta = epsilon times normalized gradient, then project

Does this become the same as generalized power replacement? Does it reach high
loss faster? Does the final delta look different, rougher, or more high-frequency?

## Minimal Methods Needed

Only these four methods are needed for the core experiment:

| Method | Meaning |
|---|---|
| `raw_add` | ordinary raw-gradient PGD additive update |
| `raw_replace` | normalized raw-gradient replacement directly to boundary |
| `steepest_add` | LP-steepest PGD additive update |
| `steepest_replace` | generalized-power-style replacement using LP-steepest direction |

Everything else, such as JVP/VJP or affine JVP/VJP power variants, is not needed
for this core question. Those are separate operator-power direction experiments.

## Answer From Completed p=2,q=2 Run

Observed source run:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`

Observed source tables:

- `per_step_metrics.csv`
- `final_delta_similarity/20260518_final_delta_similarity/final_delta_pairwise_similarity_summary.csv`

For `p=2`, normalized raw gradient equals L2-steepest direction. Therefore:

`raw_replace = steepest_replace`

So the answer for `p=2,q=2` is:

Yes. If PGD's additive update is replaced by a direct boundary update using the
normalized gradient, it becomes the same as generalized-power-style steepest
replacement in this p=2 setting.

## p=2,q=2 Final Loss And Delta Roughness

Final step `k=100`, batch size 100:

| Method | Final actual loss mean | Boundary ratio | High-frequency ratio | First-derivative L2 | Interpretation |
|---|---:|---:|---:|---:|---|
| `raw_add` | `5.389592` | `0.968878` | `5.90e-08` | `0.645722` | ordinary PGD additive; not fully on boundary on average |
| `raw_replace` | `6.804780` | `1.000000` | `8.10e-10` | `0.217019` | direct boundary PGD; identical to `steepest_replace` for p=2 |
| `steepest_add` | `6.377158` | `1.000000` | `8.00e-06` | `0.583815` | LP-steepest additive; reaches boundary but final delta differs |
| `steepest_replace` | `6.804780` | `1.000000` | `8.10e-10` | `0.217019` | GPI-style replacement; identical to `raw_replace` for p=2 |

Observed final-delta similarity:

| Pair | Mean cosine | Mean relative L2 | Meaning |
|---|---:|---:|---|
| `raw_replace` vs `steepest_replace` | `1.000000` | `0.000000` | exactly same final delta |
| `raw_add` vs `steepest_add` | `0.858049` | `0.384210` | similar but not identical |
| `raw_add` vs `raw_replace` | `0.447369` | `0.933489` | direct boundary replacement changes the final delta a lot |
| `steepest_add` vs `steepest_replace` | `0.530025` | `0.830928` | additive vs replacement matters |

## Direct Interpretation

For `p=2,q=2`, direct boundary PGD is not merely close to generalized power
replacement. It is exactly the same final-delta method as `steepest_replace` in
this implementation.

It also gives a higher final actual loss than ordinary raw PGD additive:

`raw_replace / steepest_replace`: final loss `6.804780`

`raw_add`: final loss `5.389592`

It does not look more high-frequency in the recorded roughness metrics. In fact,
for this run the replacement delta has lower first-derivative L2 and lower
high-frequency ratio than `raw_add` and `steepest_add`.

## What Is Still Missing For p != 2

The completed `p=1` and `p=2,q=1` `pq_key` runs did not include `raw_replace`.
They included `raw_add`, `unit_raw_add`, `steepest_add`, and `steepest_replace`,
but not the exact PGD-boundary-replacement method needed for the user's core
question.

Therefore, the current evidence fully answers the core question for `p=2`, but
not yet for `p=1` or `p=inf`.

To answer the core question for `p != 2`, run only this minimal method set:

`raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`

No JVP/VJP methods are needed.

## Recommended Clean Future Experiment

For each selected `(p,q)` pair, run only:

1. `raw_add`
2. `raw_replace`
3. `steepest_add`
4. `steepest_replace`

Then report only:

1. actual loss curve with mean and sample standard deviation;
2. boundary ratio curve;
3. final delta pairwise similarity;
4. final delta roughness / high-frequency metrics;
5. heatmap and 3D surface for a few representative samples.

This directly answers the original question without extra method clutter.

## Selected Findings Consolidated

A compact record of the selected results from the discussion is saved in:

`docs/loss3_selected_core_findings_summary_20260518.md`

Use that document as the short reference for the core experiment. It records:

- the four-method minimal comparison;
- the `p=2,q=2` result that `raw_replace = steepest_replace`;
- the final loss, boundary ratio, roughness, and final-delta similarity numbers;
- the `p=2` equivalence rule;
- the `p!=2` caveat;
- why JVP/VJP variants should not be part of the minimal core comparison.

