# NS2D vs 1D Burgers Optimizer-Geometry Hypothesis Check

Updated: 2026-05-22 22:29:03 UTC

Status: literature/record inspection only. No model, solver, GPU, or attack computation was run for this note.

## Question

Check whether the previous 1D Burgers / FNO optimizer-mechanism experiments support the current hypothesis for the 2D NS recurrent FNO results:

- 1D Burgers often favors generalized-power / replacement-style attacks because the practical attack geometry quickly enters a stable high-loss boundary direction regime.
- 2D NS recurrent FNO often favors `steepest_add` because the full objective is more path-dependent, recurrent, solver-coupled, and gradient-rotation dominated.

## Burgers Evidence Inspected

Observed from:

- `docs/three_loss_burgers_optimizer_findings_summary_20260521.md`
- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`
- `docs/gpi_fast_optimizer_three_losses_interpretation_20260521.md`
- `docs/loss3_gpi_overall_conclusion_20260520.md`
- `docs/loss3_hypothesis_validation_status_20260521.md`
- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`
- `docs/loss3_gpi_early_step_comparison_20260520.md`
- `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
- `docs/local_jacobian_svd_experiment_purpose_20260515.md`
- `docs/loss3_finite_difference_linearity_fno_nu0p001_steps100_result_20260517.md`
- `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`
- `docs/loss3_pgd_gradient_rotation_fno_nu0p001_steps100_result_20260517.md`
- `docs/loss3_path_geometry_theory_and_angle_evidence_20260517.md`
- `docs/loss3_current_mechanism_validation_summary_20260521.md`
- `docs/loss3_current_core4_jacobian_svd_probe_20260521.md`

## Observed Burgers Findings

1. GPI / replacement is fastest early in the 1D Burgers records.
   - For `loss1_original`, k95 is `26` for PGD, `27` for LP-steepest PGD, and `3` for generalized power.
   - For `loss2_original`, k95 is `24` for PGD, `27` for LP-steepest PGD, and `2` for generalized power.
   - For `loss3_original`, generalized power rises sharply in the first few steps and reaches strong values early.

2. The strongest Burgers claim is early speed, not unconditional final-loss dominance.
   - Prior records explicitly warn that long additive runs can match or slightly exceed replacement/GPI in some alpha/epsilon settings.
   - The safe conclusion is that replacement/GPI is a fast early optimizer and a strong practical boundary-direction method.

3. Boundary arrival alone was not enough in Burgers.
   - Replacement/GPI reaches the epsilon boundary at step `1`, but records show large post-boundary angular motion and post-boundary gain.
   - In p2q2 loss3 diagnostics, replacement/GPI post-hit angle is about `30.68 deg`, while raw-add and steepest-add post-hit angles are much smaller.

4. GPI early directions quickly become final-like in the saved Burgers trajectories.
   - Mean `cos(delta_k, delta_300)` for GPI is about `0.8868` at k=5, `0.9840` at k=10, and `0.9949` at k=20 for the saved baseline samples.
   - This supports an early high-loss direction / boundary-corridor story for the tested Burgers setting.

5. Small-epsilon Burgers diagnostics support local-linear stability.
   - In the small-epsilon sweep, selected candidate directions remain stable across `epsilon = 1e-4, 1e-3, 1e-2, 1e-1` with absolute cosine `1.0`.
   - For `epsilon <= 1e-2`, finite-epsilon ratios agree closely with clean local Jacobian references.

6. But the older Burgers records reject a pure SVD/GPI theorem.
   - The explicit JVP/VJP power variants were weaker than objective-gradient replacement in the checked p2q2 eps8/alpha0.3 ablation.
   - Current SVD probes show local residual-Jacobian dominant modes can exist, but the top singular vector is not always strongly aligned with the final replacement delta.
   - Therefore the successful Burgers method is best described as objective-gradient replacement / full-budget boundary search, not a strict generalized-power theorem for the full nonlinear loss.

7. Burgers also had path dependence, but it behaved in a way favorable to replacement.
   - Straight-line residual-Jacobian diagnostics show early rotation away from the clean geometry and then entry into a more stable high-amplification corridor.
   - Adjacent top-1 angles become small late along the path even though endpoint geometry is far from clean geometry.
   - This explains why replacement can work well: after a few aggressive boundary moves, the trajectory enters a stable useful region.

## How This Compares To Current 2D NS Evidence

Observed from current 2D NS eps32/alpha10 records:

- `steepest_add` / LP-steepest PGD is often strongest, especially for W-heavy `loss3` cases.
- Replacement methods reach the boundary but do not consistently achieve the best true-loss increase.
- FFT diagnostics show solver/dealiasing fingerprints and ADW-mode dependence in final deltas.

Inference:

- The Burgers evidence supports the general framework: optimizer ranking depends on boundary geometry, direction stability, and whether a method can exploit a high-loss corridor.
- The Burgers evidence does not say replacement is universally best. It says replacement is excellent when early full-budget directions quickly become final-like and when the boundary corridor is stable.
- The current 2D NS result can be explained if its full recurrent/solver-coupled objective has more rotating, multi-component, frequency-filtered gradient structure. In that case, `steepest_add` can win because it accumulates useful history instead of discarding it each replacement step.

## Updated Mechanism Statement

The refined comparison should be:

```text
1D Burgers: replacement/GPI is fast because it immediately uses the radius budget and quickly enters a stable high-loss boundary corridor. The evidence supports objective-gradient replacement as a practical fast optimizer, not a pure top-singular-vector theorem.

2D NS recurrent FNO: steepest_add appears stronger because the useful perturbation is more path-dependent and frequency/mode coupled. Additive normalized updates can preserve and rotate accumulated structure, while replacement can discard useful history by resetting to the current gradient direction.
```

## What Would Falsify Or Strengthen This

Strong supporting check for 2D NS:

- If a frozen-linearized or small-epsilon 2D NS diagnostic makes replacement/GPI stronger, that would mirror the Burgers local-linear evidence and support the hypothesis.

Potential falsifier:

- If 2D NS replacement directions are already final-like by k=5/k=10 but still have worse true loss, then the issue is not direction instability; it may be surrogate/true-loss mismatch, solver target coupling, or alpha/epsilon scaling.

Next recommended diagnostics:

- Run a 2D NS small-epsilon sweep (`epsilon = 4, 8, 16, 32`) with boundary-matched milestones.
- Record `cos(delta_k, delta_final)` and true-loss ratios for replacement methods at k=1,5,10,20.
- Add a frozen-linearized/JVP diagnostic for one or two samples to see whether replacement becomes strong in the local-linear regime.

## Plain-Language Clarification: Nonlinearity And Why Loss1 Can Differ From GPI Intuition

Updated: 2026-05-22 22:31:41 UTC

Plain-language clarification:

Yes, the current working interpretation is that the 2D NS recurrent FNO attack can be nonlinear enough that replacement/GPI-style intuition weakens, even for `loss1`.

The simple local theory would be:

```text
F(x + delta) - F(x) ~= J_F(x) delta
```

If this approximation is accurate and the perturbation radius is small, then maximizing `||F(x + delta) - F(x)||_2` over an L2 ball is close to a fixed linear/Jacobian problem. In that regime, replacement / generalized-power-style updates should be very strong because they repeatedly move directly to the current best full-budget direction.

Why the current 2D NS `loss1` may not obey that simple theory:

- The perturbation radius (`epsilon=32` in the baseline analysis) is finite and not infinitesimal, so the attack may leave the clean local-linear region.
- The 2D NS FNO is recurrent/autoregressive over time, so the final output is the result of repeated model application rather than one fixed linear map.
- The FNO input is not just a raw single frame in the simple 1D sense. Because the recurrent model needs the first 10 frames, perturbing the initial condition can pass through a solver/warm-up trajectory before the FNO final output is evaluated.
- Therefore even `loss1`, although it does not use the solver as the final target, can still contain nonlinear preprocessing / rollout geometry before the model output is compared.

Inference:

- If `epsilon` were much smaller, or if the model/rollout were frozen into a local linearized Jacobian, replacement/GPI should become more competitive.
- If `steepest_add` remains stronger at the current finite radius, that suggests the useful perturbation direction changes along the path. Additive normalized updates can preserve useful past components, while replacement keeps resetting to the current direction.

This is why the next clean test is a small-epsilon or frozen-linearized diagnostic.

## Existing-Data Validation: Does 2D NS Support The Complexity/Nonlinearity Explanation?

Updated: 2026-05-22 22:35:25 UTC

Status: existing-result inspection only. No new attack, solver, model, plotting, or GPU computation was run.

Question: does current evidence support the idea that 2D NS is a more complex / more nonlinear optimization problem, where full replacement/GPI-style updates are less effective and LP-steepest additive PGD can be better?

Observed from existing local 2D NS summaries under:

`2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/`

| pair | mode | loss | best true method | best true | steepest_replace true | steepest_add / replace true ratio | best surrogate method | steepest_add / replace surrogate ratio |
|---|---|---|---|---:|---:|---:|---|---:|
| eps32_alpha10 | `aaaaaaaaaw` | loss2 | steepest_add | 82.271 | 81.345 | 1.011 | raw_replace | 0.948 |
| eps32_alpha10 | `aaaaaddddw` | loss3 | steepest_add | 177.048 | 98.614 | 1.795 | steepest_add | 1.619 |
| eps32_alpha10 | `dddddddddw` | loss3 | steepest_add | 216.414 | 116.944 | 1.851 | steepest_add | 1.851 |
| eps32_alpha10 | `dddddwwwww` | loss3 | steepest_add | 289.227 | 137.871 | 2.098 | steepest_add | 2.098 |
| eps32_alpha10 | `wwwwwddddw` | loss3 | steepest_add | 207.858 | 114.180 | 1.820 | steepest_add | 1.820 |
| eps32_alpha10 | `wwwwwwwwww` | loss1 | steepest_add | 158.283 | 98.326 | 1.610 | steepest_add | 1.295 |
| eps32_alpha10 | `wwwwwwwwww` | loss3 | steepest_add | 304.593 | 106.323 | 2.865 | steepest_add | 2.865 |
| eps8_alpha2p5 | `aaaaaaaaaw` | loss2 | raw_replace | 72.438 | 72.438 | 0.981 | raw_add | 1.084 |
| eps8_alpha2p5 | `wwwwwwwwww` | loss1 | steepest_add | 75.255 | 72.655 | 1.036 | steepest_add | 1.002 |
| eps8_alpha2p5 | `wwwwwwwwww` | loss3 | steepest_add | 139.456 | 97.812 | 1.426 | steepest_add | 1.426 |

Observed evidence:

- At `eps32_alpha10`, `steepest_add` is much stronger than `steepest_replace` for `loss1/all_w` and most `loss3` modes.
- At smaller `eps8_alpha2p5`, the `loss1/all_w` gap almost disappears: true-loss ratio `steepest_add / steepest_replace` is only about `1.036`, and surrogate ratio is about `1.002`.
- `loss3/all_w` still favors `steepest_add` at `eps8`, but the advantage is smaller than at `eps32`: true-loss ratio drops from about `2.865` to about `1.426`.
- `loss2/all_a_target_w` is close across methods and can even favor replacement/raw-replacement on true loss, which is consistent with `loss2` being a simpler/fixed-target/dictionary-like objective.

Inference:

- Existing 2D NS data support the user's interpretation: as the problem becomes more finite-radius / nonlinear / path-dependent, replacement/GPI-style updates lose their clear advantage, while LP-steepest additive PGD can become better.
- The `eps32 -> eps8` comparison is especially important. Smaller epsilon makes `loss1` look much more replacement-like again, while larger epsilon makes `steepest_add` clearly stronger. This is exactly the pattern expected if the failure mode is finite-radius nonlinearity rather than a simple code or naming issue.
- The evidence is not yet a full proof. A frozen-linearized 2D NS diagnostic would be the strongest confirmation: if replacement/GPI wins on the frozen local Jacobian version but loses on the full recurrent/solver-coupled attack, then the explanation is essentially confirmed.

Plain conclusion:

```text
Yes: current evidence supports the idea that Wendy/1D Burgers was easier or more stable for replacement/GPI, while current 2D NS is more complex. In the more complex 2D NS case, LP-steepest additive PGD can be better because it keeps useful perturbation history instead of replacing delta with only the current gradient direction.
```
