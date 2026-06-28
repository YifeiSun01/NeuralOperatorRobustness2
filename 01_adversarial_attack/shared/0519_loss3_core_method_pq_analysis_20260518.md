# Loss3 Core Method and P/Q Analysis - 2026-05-18

Status: completed numerical analysis from existing experiment outputs. No optimizer
experiment was launched for this step.

## Source Data

Original P/Q runs:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517`

Raw-replace backfill runs:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`

Generated analysis tables:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_core_method_pq_analysis_20260518`

Files:

- `core4_pq_method_summary.csv`
- `core4_pq_winner_summary.csv`
- `core4_pq_metric_ranks.csv`

Methods analyzed:

- `raw_add`: additive raw-gradient PGD.
- `raw_replace`: boundary replacement using normalized raw gradient.
- `steepest_add`: additive Lp-steepest PGD.
- `steepest_replace`: GPI / Lp-steepest boundary replacement.

## Important Caveat

Observed evidence: several `p=inf` runs have nonfinite losses. In particular:

| P/Q | Method | Final nonfinite count |
|---|---|---:|
| `p=inf,q=1` | `raw_add` | 100 |
| `p=inf,q=1` | `steepest_add` | 100 |
| `p=inf,q=2` | `raw_add` | 34 |
| `p=inf,q=2` | `steepest_add` | 100 |
| `p=inf,q=inf` | `steepest_add` | 100 |

Inference: the `p=inf` setting is numerically unstable or degenerate under the
current setup. Some `p=inf` replacement methods end with zero smoothness metrics
because the final perturbation collapses to a zero/constant-like state; that
should not be read as a physically good smooth perturbation by itself.

## Per-P/Q Summary

Observed from `core4_pq_winner_summary.csv` and `core4_pq_method_summary.csv`:

| P/Q | Largest final finite loss | Fastest early growth | Smoothest final delta by D1/TV | Interpretation |
|---|---|---|---|---|
| `p=1,q=1` | `steepest_add` | `steepest_replace` | `raw_replace` | Clear tradeoff: high loss from `steepest_add`, smoothness from `raw_replace`, fast early growth from GPI. |
| `p=1,q=2` | `steepest_add` | `steepest_replace` | `raw_replace` | Same tradeoff as `p=1,q=1`. |
| `p=1,q=inf` | `steepest_add` | `steepest_replace` | `raw_replace` | Same tradeoff; `steepest_add` is much rougher/high-frequency. |
| `p=2,q=1` | `raw_add` | `raw_replace = steepest_replace` | `raw_replace = steepest_replace` | `raw_add` has slightly larger final loss, but replacement/GPI is faster and smoother with close final loss. |
| `p=2,q=2` | `raw_replace = steepest_replace` | `raw_replace = steepest_replace` | `raw_replace = steepest_replace` | Best-balanced setting in the observed data: highest final loss, fastest growth, and smoothest final delta agree. |
| `p=2,q=inf` | `steepest_add` | `raw_replace = steepest_replace` | `raw_add` | Tradeoff setting: final loss, speed, and smoothness disagree. |
| `p=inf,q=1` | `raw_replace = steepest_replace` among finite finals | unreliable | degenerate replacement | `raw_add` and `steepest_add` become nonfinite; not recommended. |
| `p=inf,q=2` | `raw_add` among finite finals | unreliable | degenerate replacement | `raw_add` has 34 nonfinite samples at final; `steepest_add` fails; not clean. |
| `p=inf,q=inf` | `raw_add` among finite finals | unreliable | degenerate replacement | `steepest_add` fails; replacement curves show nonfinite intermediate behavior; not clean. |

## Final Loss Findings

Observed winner counts across the 9 P/Q settings, splitting exact ties:

- `steepest_add`: wins about 4 settings.
- `raw_add`: wins about 3 settings.
- `raw_replace`: wins about 1 setting.
- `steepest_replace`: wins about 1 setting.

Inference: if the only goal is the largest final loss, there is no single method
that dominates every P/Q setting. `steepest_add` is often strongest for `p=1` and
some `q=inf` cases, but it is also often rougher and can fail for `p=inf`.

## Speed Findings

Observed for stable `p=1` and `p=2` settings:

- For all `p=1` settings, `steepest_replace` has the fastest early growth, but
  `steepest_add` eventually reaches a larger final loss.
- For all `p=2` settings, `raw_replace` and `steepest_replace` have the fastest
  early growth. They are identical when `p=2` because normalized raw-gradient
  replacement equals L2-steepest replacement.

Inference: replacement-style updates are the fastest way to increase loss early
because they jump directly to the boundary. Additive methods can eventually catch
up or exceed them in some P/Q settings, but they spend many steps walking toward
large-radius perturbations.

## Smoothness / Low-Frequency Findings

Observed from final-step `high_frequency_energy_ratio`, `first_derivative_l2`,
and `total_variation`:

- For `p=1`, `raw_replace` is consistently the smoothest or near-smoothest, but
  its final loss is much lower than `steepest_add`.
- For `p=2,q=1` and `p=2,q=2`, `raw_replace = steepest_replace` is smoother than
  additive methods by D1/TV and also grows faster.
- For `p=2,q=inf`, `raw_add` is smoothest by D1/TV, while `steepest_add` gives
  the largest final loss and replacement/GPI is fastest early.
- For `p=inf`, zero smoothness metrics in replacement methods are not a reliable
  positive signal because the loss curves show nonfinite/degenerate behavior.

Inference: for a practical balance of high final loss, fast growth, and smooth
final perturbation, `p=2,q=2` with `raw_replace/steepest_replace` is the cleanest
observed result. `p=1` gives a strong loss-vs-smoothness tradeoff, and `p=inf`
should be treated as not suitable in this current experiment configuration.

## Practical Recommendation

Recommended primary setting:

- `p=2,q=2`, using `steepest_replace` / GPI. Since `p=2`, this is identical to
  `raw_replace` boundary replacement in the current objective-gradient version.

Reason:

- It has the largest final loss in its P/Q setting.
- It has the fastest early loss growth.
- It has the lowest or tied-lowest high-frequency ratio, first-derivative L2, and
  total variation.
- It avoids the `p=inf` numerical pathologies.

Secondary setting worth reporting:

- `p=2,q=1`, where `raw_add` gets the largest final loss, but GPI/replacement is
  faster and smoother with close final loss.

Settings to be cautious about:

- `p=1`: useful for showing the final-loss vs smoothness tradeoff, but
  `steepest_add` final perturbations are much rougher.
- `p=inf`: not recommended without fixing numerical/nonfinite behavior.
- `q=inf`: produces more tradeoffs and less clean agreement between final loss,
  speed, and smoothness than `q=2`.
