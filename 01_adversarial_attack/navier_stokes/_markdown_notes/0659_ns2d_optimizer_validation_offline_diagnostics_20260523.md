# NS2D Optimizer Validation Offline Diagnostics

Updated: 2026-05-23 00:36:15 UTC

Status: CPU-only offline post-processing completed. No solver call, model
inference, attack step, JAX import, PyTorch import, or GPU computation was
started by this analysis script.

## Source Evidence

Observed from completed method directories under:

- `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522`

A method directory was included only when these files existed:

- `final_delta_and_metrics.npz`
- `per_step_metrics.csv`
- `per_sample_step_metrics.csv`
- `step_sample_trace.npz`
- `step_sample_trace_metrics.csv`

## Scope

- Completed method directories analyzed: `45`
- Epsilon/loss/mode groups represented: `12`
- Early-to-final cosine rows: `405`
- Boundary-matched true-loss rows: `180`
- Gradient-rotation step rows: `4545`
- Generated images: `36`
- Wall time: `33.3 s`

## Outputs

Records:

- `2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523/records/input_method_dirs.csv`
- `2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523/records/early_to_final_cosine.csv`
- `2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523/records/boundary_matched_true_loss.csv`
- `2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523/records/gradient_rotation_step_sample.csv`
- `2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523/records/gradient_rotation_summary.csv`

Images:

- `2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523/images`

## Interpretation Notes

Observed evidence:

- Early-to-final cosine uses the saved representative sample trace. It tests
  whether an optimizer direction quickly resembles its final perturbation.
- Boundary-matched true loss uses the first recorded step where the mean delta
  norm reaches 25%, 50%, 75%, and 100% of epsilon. This helps separate
  "arrived at the boundary earlier" from "better direction at the same norm".
- Gradient rotation uses saved representative-sample arrays for gradient,
  update direction, and delta. It records step-to-step direction changes and
  the angle between the current delta and current update direction.

Inference:

- These diagnostics are sufficient for a first pass on the nonlinear/path
  dependence hypothesis without disturbing the running GPU attack.
- Full-batch gradient rotation is not evidenced by this offline pass; it would
  require storing all-sample gradients or rerunning attacks with much larger
  trace output.


## First-Pass Results

Observed from `records/boundary_matched_true_loss.csv` at the first step where
mean delta norm reached `100%` of epsilon:

- `eps32_alpha10 / loss1 / all_w`: `steepest_add` has the largest true loss
  (`141.3314` at step `12`), above `raw_add` (`92.1748` at step `1`) and
  replacement methods (`86.5344` at step `61`).
- `eps32_alpha10 / loss3 / all_w`: `steepest_add` has the largest true loss
  (`261.6749` at step `10`), above `raw_add` (`146.0830` at step `1`) and
  replacement methods (`123.0626` at step `14`).
- `eps32_alpha10 / loss3` W/D/A mixed modes: `steepest_add` is the 100%-boundary
  true-loss winner in all five completed loss3 modes.
- `eps32_alpha10 / loss2 / all_a_target_w`: `raw_add` is slightly ahead at the
  100%-boundary comparison (`82.3287`), with `steepest_add` close (`81.3908`).
- `eps8_alpha2p5 / loss1 / all_w`: the gap is much smaller. `steepest_add`
  reaches `75.7010`, while `raw_add` reaches `75.4601`, and replacement methods
  reach `72.4402`.
- `eps8_alpha2p5 / loss2 / all_a_target_w`: replacement methods are the 100%-
  boundary winners (`72.4378`), slightly above additive methods.
- `eps8_alpha2p5 / loss3 / all_w`: `steepest_add` remains ahead (`136.9439`),
  above `raw_add` (`106.7041`) and replacement methods (`86.5195`).
- `eps8_alpha2p5 / loss3 / w1_5_d6_9_target_w` is partial in the local records:
  only `raw_add` was complete at the time of this offline pass, so it should not
  be used for four-method comparison yet.

Observed from `records/early_to_final_cosine.csv`:

- `eps32_alpha10 / loss1 / all_w`: at `k=10`, `steepest_add` already has
  `cos(delta_k, delta_final)=0.7081`, while replacement methods are only
  `0.2243`; by `k=20`, `steepest_add` is `0.8563`, replacement is `0.5295`.
- `eps32_alpha10 / loss3 / all_w`: at `k=10`, `steepest_add` is `0.6694`, while
  replacement methods are negative (`-0.0758`); by `k=20`, `steepest_add` is
  `0.8233`, replacement is still negative (`-0.1594`).
- `eps8_alpha2p5 / loss1 / all_w`: all methods are essentially aligned with
  their final direction by `k=10` or `k=20`, consistent with the smaller epsilon
  behaving more locally/linearly.

Observed from `records/gradient_rotation_summary.csv`:

- For `eps32_alpha10 / loss1 / all_w`, `steepest_add` has a much smoother delta
  trajectory: median `angle(delta_k, delta_{k-1}) = 12.42 deg`, while replacement
  methods have about `66.47 deg`.
- For `eps32_alpha10 / loss3 / all_w`, `steepest_add` again has a smoother delta
  trajectory: median `angle(delta_k, delta_{k-1}) = 11.61 deg`, while replacement
  methods have about `78.09 deg`.
- For replacement methods, `angle(delta_k, delta_{k-1})` and
  `angle(delta_k, update_k)` are effectively the same diagnostic, because each
  step replaces delta with the current direction rather than adding a new update
  to the old delta.

Inference from this offline pass:

- The finite-radius/nonlinear/path-dependent explanation is supported by the
  completed records. At `eps32_alpha10`, `steepest_add` tends to win not merely
  because it reaches the boundary, but because at the same boundary level it has
  higher true loss in the main `loss1`/`loss3` cases.
- Smaller epsilon weakens the advantage: `eps8_alpha2p5 / loss1 / all_w` nearly
  collapses the method gap, which is consistent with the local-linear regime
  becoming more replacement/GPI-friendly.
- The replacement methods often show large step-to-step direction rotation in
  the full 2D NS case, so resetting delta every step can discard useful path
  history. Additive LP-steepest PGD preserves history and produces smoother
  final perturbation trajectories in the observed `eps32_alpha10` cases.


## Interpretation: Why 2D NS Favors Steepest Add While 1D Burgers Favored Replacement

Question addressed: why the best optimizer in the 2D NS recurrent FNO attack is
not the same as in the earlier 1D Burgers attack.

Observed evidence from this offline pass:

- At `eps32_alpha10 / loss1 / all_w`, `steepest_add` has higher 100%-boundary
  true loss than replacement methods: `141.3314` versus `86.5344`.
- At `eps32_alpha10 / loss3 / all_w`, `steepest_add` has higher 100%-boundary
  true loss than replacement methods: `261.6749` versus `123.0626`.
- At `eps32_alpha10`, every completed `loss3` mode has `steepest_add` as the
  100%-boundary true-loss winner.
- At `eps8_alpha2p5 / loss1 / all_w`, the method gap nearly collapses:
  `steepest_add=75.7010`, `raw_add=75.4601`, replacement methods `72.4402`.
- For `eps32_alpha10 / loss3 / all_w`, the median delta step angle is much
  smoother for `steepest_add`: `angle(delta_k, delta_{k-1}) = 11.61 deg`, while
  replacement methods are about `78.09 deg`.

Interpretation:

1. The large-epsilon 2D NS attack is not behaving like a fixed quadratic or a
   locally linear singular-vector problem.

   If it were close to a stable local-linear problem, generalized power / replace
   would be expected to quickly find a stable dominant direction. Instead, in
   the `eps32_alpha10` full attack, replacement directions rotate substantially
   across iterations and do not consistently align early with the final
   perturbation.

2. Replacement/GPI discards path history.

   Replacement methods set the new perturbation essentially from the current
   direction. That is ideal when the current direction is already the right
   dominant direction, as in a simple or locally linear regime. But when the
   objective changes along the path, each new gradient can point somewhere
   substantially different. In that case, replacement can jump around and lose
   useful components accumulated in previous steps.

3. LP-steepest additive PGD integrates a path of changing directions.

   `steepest_add` updates by adding the LP-steepest direction and then projecting
   back to the epsilon ball. This keeps useful older components while adding the
   new direction. In the observed 2D NS runs, this produces a much smoother delta
   trajectory and higher true loss at the same boundary level. It acts less like
   "choose the best current vector" and more like "accumulate a useful nonlinear
   path".

4. The epsilon dependence supports the finite-radius/nonlinear explanation.

   When epsilon is smaller (`eps8_alpha2p5`), the `loss1/all_w` gap between
   methods becomes small. That is what the hypothesis predicts: smaller epsilon
   stays closer to the local-linear regime, where replacement/GPI should become
   more competitive. At larger epsilon (`eps32_alpha10`), the nonlinear/path
   dependence is stronger, so additive steepest PGD wins more clearly.

5. The result is not just "steepest_add reaches the boundary faster".

   Boundary-matched comparison shows that at the same `100%` epsilon boundary,
   `steepest_add` has higher true loss in the main `loss1` and `loss3` cases.
   That means the advantage is not only speed-to-boundary; it is also a better
   finite-radius direction/path.

Connection to 1D Burgers:

- The earlier 1D Burgers case appears more compatible with a stable dominant
  direction picture: replacement/GPI quickly moves toward the final useful
  perturbation direction and therefore performs very well.
- The 2D NS recurrent case is higher-dimensional, more nonlinear, recurrent in
  the model rollout, and coupled to solver/detach/dictionary mode choices. The
  effective objective changes more as delta grows, so a one-shot replacement of
  delta by the current direction is less reliable.

Current working conclusion:

- 1D Burgers: closer to local-linear / stable dominant-direction behavior, so
  generalized power / steepest replacement can be best.
- 2D NS recurrent FNO: more finite-radius nonlinear and path-dependent, so
  additive LP-steepest PGD is better because it accumulates useful directions
  instead of resetting the perturbation every step.

Remaining validation:

- A frozen-linearized 2D NS diagnostic is still the strongest direct test. If
  replacement/GPI wins in the frozen-linearized version but `steepest_add` wins
  in the full nonlinear version, the explanation above would be strongly
  confirmed.
