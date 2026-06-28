# Loss3 GPI / Replacement Overall Conclusion - 2026-05-20

Status: synthesis from existing local experiment artifacts; no new experiment was run for this note.

## Scope

This note summarizes the current evidence for using `steepest_replace` / generalized power iteration (GPI-style replacement) as a fast optimizer for `loss3` under the tested alpha/epsilon sweeps.

Observed source artifacts:

- 300-step p2q2 summary table: `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`
- 300-step p2q2 visual root: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`
- GPI early-step comparison: `forensics/loss3_gpi_early_step_comparison_20260520/`
- GPI early-step result note: `docs/loss3_gpi_early_step_comparison_20260520.md`
- Baseline trajectory/GIF source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/`

## Main Conclusion

Observed from the current experiments, `steepest_replace` / GPI is a very strong practical optimizer for this `loss3` setting.

Its main advantage is not that it always has the largest final loss after a very long run. Its main advantage is that it reaches a strong boundary perturbation extremely quickly:

- It saturates the p-norm perturbation budget almost immediately.
- It quickly moves along the boundary with much larger angular updates than additive PGD-style methods.
- In the saved baseline trajectory, its perturbation shape is already close to the final 300-step perturbation after only a few steps.
- Its final perturbation shape is often visually and cosine-wise similar to PGD / LP-steepest results, but it reaches that kind of shape much faster.

## Key Observations

Observed from the p2q2 alpha/epsilon sweeps:

- Replacement/GPI reaches the p-norm boundary at step 1.
- Raw PGD can take many steps to reach the boundary, and its boundary-arrival speed depends strongly on raw gradient scale.
- LP-steepest additive PGD reaches the boundary faster than raw PGD because its update direction is normalized in the p-ball geometry, but it still has additive-update inertia once the perturbation is large.
- Replacement/GPI has much faster angular motion along the boundary than the additive methods.

Observed from `docs/loss3_gpi_early_step_comparison_20260520.md` for the saved baseline samples (`epsilon=4`, `alpha=0.4`, `p=q=2`):

- Mean `cos(delta_5, delta_300)` for GPI is `0.8868`.
- Mean `cos(delta_10, delta_300)` for GPI is `0.9840`.
- Mean `cos(delta_20, delta_300)` for GPI is `0.9949`.
- Mean selected-sample GPI loss is `3.4935` at k=5, `3.6642` at k=10, `3.6791` at k=20, and `3.6344` at k=300.

This supports the interpretation that the GPI perturbation shape is already mostly formed by about 5-10 steps in the baseline case.

## Why This Happens

Inference from the implementation and recorded diagnostics:

- Raw PGD uses the unnormalized raw gradient `J`. Because `||J||` varies across samples, different samples move toward the epsilon boundary at very different speeds. This explains the large `boundary_ratio` standard deviation for PGD.
- LP-steepest PGD computes a steepest direction `S` from `J` and normalizes it in the p-ball geometry. This makes radial budget usage much more synchronized across samples.
- GPI / replacement also uses a normalized steepest direction, but instead of adding a small update to the old perturbation, it replaces the perturbation with a full-budget boundary perturbation in the current direction.
- Because replacement has little angular inertia, it can rotate aggressively along the epsilon boundary and quickly find a strong direction.

## Important Caveat

The current evidence should not be written as "GPI is always the final-loss optimum."

Observed from the 300-step p2q2 runs:

- In some alpha/epsilon settings, `steepest_add` or raw PGD can match or slightly exceed GPI in final mean loss after 300 steps.
- Therefore, GPI's cleanest advantage is speed, stability, and early strong performance, not unconditional final-loss dominance.

A safe conclusion is:

> GPI/replacement achieves a strong Pareto advantage: it reaches the perturbation boundary immediately, converges to a final-like perturbation shape within a few steps, and obtains comparable final perturbations and loss much faster than additive PGD-style methods.

## Practical Recommendation

For future loss3 experiments, it is reasonable to include GPI / `steepest_replace` as a main optimizer or fast default.

Suggested variants to report:

- `GPI-5`: run GPI for 5 steps.
- `GPI-10`: run GPI for 10 steps.
- `GPI-300`: long-run reference, when needed.
- Compare against raw PGD and LP-steepest additive PGD on final loss, boundary-arrival step, angular motion, and perturbation smoothness.

The current baseline evidence suggests `GPI-10` may already recover a perturbation shape close to the long-run GPI result.

## Remaining Checks

Before making a strong paper-level claim, the following would be useful:

- Repeat early-step GPI comparisons for more alpha/epsilon settings, not only the saved baseline trajectory.
- Compute full-batch early-stop metrics if intermediate deltas are saved for all 100 samples.
- Compare high-frequency energy, derivative/TV smoothness, and peakiness at k=5/k=10 versus k=300.
- Report the cases where long-run additive methods slightly exceed GPI final loss, to keep the conclusion honest.
