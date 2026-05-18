# Loss3 Direction-Proposal Conclusions and 2026-05-19 Plan

Status: planning/summary document. No new optimizer experiment was run for this
note.

## Source Evidence

Observed from existing outputs:

- Original P/Q runs:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517`
- Raw-replace backfill runs:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`
- Per-P/Q four-figure folders:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_per_pq_four_figures_20260518`
- Numerical analysis tables:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_method_pq_analysis_20260518`
- Detailed analysis note:
  `docs/loss3_core_method_pq_analysis_20260518.md`

## Main Conclusions So Far

Observed evidence:

- `p=2,q=2` is the cleanest balanced case.
- In `p=2,q=2`, `raw_replace` and `steepest_replace` are identical in the
  observed objective-gradient implementation.
- In `p=2,q=2`, that replacement/GPI method reaches the largest final loss, grows
  fastest early, and has the smoothest/lowest-frequency final delta by
  high-frequency ratio, first-derivative L2, and total variation.
- `p=2,q=1` is a useful secondary comparison: `raw_add` gets the largest final
  loss, but `raw_replace = steepest_replace` grows faster and is smoother.
- `p=1,*` settings show a tradeoff: `steepest_add` often gives the largest final
  loss, `steepest_replace` grows fastest early, and `raw_replace` tends to give
  the smoothest final delta.
- `p=inf,*` settings are numerically problematic in the current configuration.
  Several methods produce nonfinite losses or degenerate final perturbations.

Inference:

- P/Q choice is part of the method behavior, not just a harmless evaluation
  detail.
- `p=2,q=2` should be the primary setting for the main story because final loss,
  growth speed, and smoothness all agree.
- `p=2,q=1` is worth keeping as a secondary case because it shows a meaningful
  final-loss-vs-speed/smoothness tradeoff.
- `p=1` is useful for explaining tradeoffs, but not as the cleanest main setting.
- `p=inf` should not be used as a main conclusion until the nonfinite/degenerate
  behavior is fixed or deliberately controlled.
- `q=inf` is less clean than `q=2`; it often makes final loss, speed, and
  smoothness disagree.

## Why Some Curves Break

Observed evidence:

- In `p=inf,q=1`, `raw_add` and `steepest_add` have final nonfinite counts of
  `100/100`.
- In `p=inf,q=2`, `steepest_add` has final nonfinite count `100/100`, and
  `raw_add` has final nonfinite count `34/100`.
- In `p=inf,q=inf`, `steepest_add` has final nonfinite count `100/100`.

Inference:

- The broken curves are not plotting bugs. The underlying losses become
  nonfinite (`NaN` or `Inf`), so the plotted lines disappear or stop being
  meaningful.
- `p=inf` with `epsilon=8` is much less restrictive than `p=2` with
  `epsilon=8`, because each coordinate can move by up to 8. This can push the
  input far outside the model/solver's stable regime.
- `q=inf` can further emphasize extreme residual points, which makes the
  optimization less smooth and less numerically clean.

## Working Recommendation

Primary result to report:

- `p=2,q=2`, method `steepest_replace` / GPI.
- In `p=2`, this is equivalent to `raw_replace` boundary replacement for the
  current objective-gradient implementation.

Secondary result to report:

- `p=2,q=1`, because it shows that additive PGD can sometimes get a slightly
  larger final loss, while replacement/GPI is faster and smoother.

Tradeoff result:

- `p=1,*`, especially to show that largest final loss and smoothest final delta
  need not come from the same optimizer.

Do not use as main result yet:

- `p=inf,*`, because the current runs show nonfinite or degenerate behavior.

## 2026-05-19 Next Experiment: Larger Alpha With Fixed Epsilon

Goal:

- Keep `epsilon = 8` fixed.
- Increase `alpha` for additive methods so they reach the boundary quickly.
- Test whether additive PGD / Lp-steepest PGD still differ from replacement/GPI
  after the additive methods are no longer limited by slow radius growth.

Main question:

- Is replacement/GPI better because it directly jumps to the boundary, or because
  its direction is genuinely better after radius progress is no longer the
  bottleneck?

Recommended P/Q settings:

1. Primary:
   - `p=2,q=2`
   - Reason: cleanest setting; no current numerical pathology; best balance of
     final loss, growth speed, and smoothness.

2. Secondary:
   - `p=2,q=1`
   - Reason: `raw_add` has larger final loss, while replacement/GPI is faster and
     smoother. Good for testing whether larger `alpha` lets additive PGD close
     the speed gap.

3. Optional tradeoff setting:
   - `p=1,q=2` or `p=1,q=1`
   - Reason: demonstrates final-loss vs smoothness tradeoffs.

Avoid for the first alpha sweep:

- `p=inf,*`
- Reason: current outputs show nonfinite losses and degenerate behavior.

Suggested methods:

- `raw_add`
- `raw_replace`
- `steepest_add`
- `steepest_replace`

Suggested alpha values:

- Keep baseline: `alpha = 0.3`
- Try larger values: `alpha = 0.6`, `1.0`, `2.0`, `4.0`
- If additive methods still approach the boundary too slowly for a chosen P/Q,
  add `alpha = 8.0`.

Metrics to compare:

- Actual `loss3_q` curve, mean and sample std.
- Step to reach 90% and 95% of the best final loss within a fixed P/Q.
- Final actual `loss3_q`.
- Boundary ratio curve, to verify whether additive methods reach the boundary
  quickly.
- Final delta smoothness:
  - high-frequency energy ratio
  - spectral centroid
  - first-derivative L2
  - total variation
- Final delta shapes for selected samples, especially sample `40`.

Expected interpretation:

- If larger `alpha` makes `raw_add` or `steepest_add` behave like
  replacement/GPI, then the old advantage was mostly a radius-progress advantage.
- If replacement/GPI remains better even when additive methods reach the boundary
  quickly, then the direction/replacement rule itself is contributing more than
  just faster boundary arrival.
- If larger `alpha` creates rougher deltas or nonfinite losses, then additive
  updates may be less stable even when they become faster.

Minimum deliverable for 2026-05-19:

- Run alpha sweep for `p=2,q=2`.
- Generate the same four per-P/Q figures for each alpha.
- Make one summary table comparing final loss, step-to-95%, boundary ratio, and
  smoothness.

Next deliverable if the first run is clean:

- Repeat for `p=2,q=1`.
- Optionally repeat for one `p=1` setting to show the tradeoff case.

## Interpretation Update: No Universal Best Optimizer

Observed evidence from the completed P/Q runs:

- There is no single optimizer that wins every P/Q setting by final loss.
- `steepest_add` reaches the largest final loss in several `p=1` settings and
  in `p=2,q=inf`.
- `raw_replace` / `steepest_replace` is fastest in many stable settings,
  especially `p=2,*`.
- `raw_replace` / `steepest_replace` is smoothest in `p=2,q=2`, but not every
  high-loss setting produces a physically plausible perturbation.
- Several `p=inf` settings have nonfinite or degenerate behavior.

Inference:

The experiment should not be reported as "method A is always better than method
B." A more accurate conclusion is:

- optimizer choice, P/Q geometry, final loss, convergence speed, and perturbation
  physical plausibility are coupled;
- optimizing only for final loss can select spiky/high-frequency perturbations;
- some P/Q settings encourage mathematically valid but physically suspicious
  delta shapes.

## Dirac-Delta / Spike Concern

Observed from the delta figures:

- Several perturbations look spike-like or highly localized rather than like
  smooth physical input perturbations.
- The problem is especially concerning in settings involving `p=1`, `q=inf`, or
  `p=inf`.

Mechanistic interpretation:

- `p=1` constraints naturally encourage sparse perturbations. The Lp-steepest
  direction for `p=1` puts mass on the largest-gradient coordinates, which can
  look like a Dirac-delta spike.
- `q=inf` objectives focus on the largest residual point, so the optimizer can
  chase a single extreme location instead of distributing error smoothly.
- `p=inf` permits each coordinate to move up to `epsilon`, which is a very large
  feasible set in a 1024-dimensional input. This can produce unstable or
  nonphysical inputs.

Inference:

A larger loss is not automatically a better adversarial perturbation for the
research goal. If the final delta looks like a spike, it may be exploiting the
mathematical norm constraint rather than representing a plausible physical
perturbation.

## Revised Scientific Message

The result of the current experiment is not a simple optimizer ranking. The
stronger message is:

1. Replacement/GPI methods are very good at fast boundary arrival.
2. Additive Lp-steepest PGD can sometimes reach larger final loss, especially in
   P/Q geometries that permit sparse or extreme perturbations.
3. Some high-loss solutions are not physically satisfactory because the final
   delta can become spike-like/high-frequency.
4. Therefore the next experiment should treat smoothness/physical plausibility as
   a first-class criterion, not just as a diagnostic after maximizing loss.

## Additional Next-Step Recommendation

For the 2026-05-19 alpha sweep, do not only ask whether larger `alpha` improves
final loss. Also ask:

- Does larger `alpha` make additive methods reach the boundary faster without
  creating worse spikes?
- Does the high-loss method produce a smooth enough delta to be physically
  meaningful?
- If not, should the attack objective include a smoothness penalty or a spectral
  low-pass parameterization?

Possible follow-up constraints/regularizers:

- Add total-variation or first-derivative penalty to the objective.
- Optimize a low-pass-filtered delta instead of arbitrary pointwise delta.
- Parameterize delta with a small number of Fourier modes.
- Enforce a smoothness budget in addition to the p-norm budget.
- Compare original final delta against low-pass-filtered final delta to see how
  much loss depends on high-frequency spikes.
