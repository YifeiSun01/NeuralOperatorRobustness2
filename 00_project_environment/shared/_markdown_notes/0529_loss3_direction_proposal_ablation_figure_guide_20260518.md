# Loss3 Direction-Proposal Ablation Figure Guide - 2026-05-18

Observed source run:
`/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.

Observed source code:
`tools/run_loss3_direction_proposal_ablation.py` and imported norm/projection helpers from
`tools/run_batch_three_loss_loss_only.py`.

## Main Reading Rule

The primary optimization result should be read from the actual loss curves:

- `figures/curves/loss3_q_mean_vs_step.png`
- `figures/true_loss_progression/actual_loss3_q_mean_methods_p2_q2_clean_20260517.png`

These use the actual iterates `delta_k` produced by each optimizer and plot the
mean observed `loss3_q` over the batch. For this completed run, `p=2,q=2`, so
`loss3_q` is the same q-side norm used by the attack objective.

`boundary_loss3_q_mean_vs_step.png` is not the primary loss. It is a diagnostic
that rescales the current nonzero `delta_k` to the p-boundary and recomputes the
loss there. It answers a different question: "if the current direction were
stretched to the full epsilon radius, what loss would it have?" It should not be
used as the main convergence/result curve.

## Curve Figures

- `loss3_q_mean_vs_step.png`: actual objective loss progression. X axis is
  optimizer step `k`; Y axis is batch mean `||F(x+delta_k)-G(x+delta_k)||_q`.
  This is the main figure for comparing final loss and speed of loss growth.

- `actual_loss3_q_mean_methods_p2_q2_clean_20260517.png`: newly added cleaner
  version of the same actual-loss comparison. It intentionally avoids
  boundary-normalized loss.

- `boundary_ratio_mean_vs_step.png`: mean `||delta_k||_p / epsilon`. This is a
  constraint-radius diagnostic, not a loss. A value near `1` means the method is
  on the p-norm boundary. Replacement methods typically jump to `1` immediately;
  additive PGD-style methods may approach it gradually.

- `boundary_loss3_q_mean_vs_step.png`: diagnostic loss after rescaling the
  current `delta_k` direction to the boundary. Useful for separating "direction
  quality" from "step/radius progress", but it is a virtual evaluation, not the
  actual optimizer trajectory.

- `high_frequency_energy_ratio_mean.png`: fraction of Fourier energy in the
  highest-frequency band of `delta`. Larger values mean more high-frequency,
  potentially noisy perturbations.

- `spectral_centroid_mean.png`: average frequency location of the perturbation
  spectrum. Larger values mean the perturbation energy is shifted toward higher
  frequencies.

- `total_variation_mean.png`: sum of absolute adjacent differences in `delta`.
  Larger values mean less smooth / more jagged perturbations.

- `first_derivative_l2_mean.png`: L2 norm of the first finite difference
  `diff(delta)`. Larger values mean sharper local changes.

- `second_derivative_l2_mean.png`: L2 norm of the second finite difference
  `diff(delta, n=2)`. Larger values mean stronger curvature/oscillation in the
  perturbation.

## Delta Shape Figures

- `figures/delta_grids/delta_grid_sample_*.png`: for one dataset index, each row
  is a method and each column is a saved step. It shows the 1D perturbation
  `delta` as a curve, so it is good for seeing whether a method jumps, slowly
  grows, becomes spiky, or becomes smoother over time.

- `figures/delta_heatmaps/delta_heatmap_<method>_sample_*.png`: time-space view
  for one method and one sample. X axis is spatial index; Y axis is saved step;
  color is signed delta value. This is the best static plot for seeing how the
  perturbation evolves over iterations.

- `figures/final_input_delta_overlays/final_input_delta_overlay_sample_*.png`:
  final visual comparison for one sample: clean input `x`, perturbed input
  `x + final_delta`, and final `delta`. This answers whether the final attack
  visibly changes the initial condition and what shape the final perturbation has.

- `figures/final_delta_overlaid/final_delta_overlay_sample_*.png`: final deltas
  from different methods overlaid on the same axes for the same sample. This is
  useful for comparing whether methods find similar or very different final
  perturbation shapes.

- `figures/loss_progression_by_method/loss3_q_and_boundary_ratio_methods_p2_q2.png`:
  combined convenience figure with actual loss and boundary ratio together. The
  loss panel is meaningful as actual loss; the boundary-ratio panel is only a
  radius/constraint diagnostic.

## Method Labels In The Legends

- `raw_add`: raw-gradient PGD, `delta <- project(delta + alpha * grad)`.
- `unit_raw_add`: normalized raw-gradient additive update. For p=2 this is close
  to the L2-steepest direction case.
- `raw_replace`: normalized raw gradient with replacement/boundary update,
  `delta <- epsilon * direction`.
- `steepest_add`: Lp-steepest direction with additive PGD update. This is the
  LP-steepest PGD comparison.
- `steepest_replace`: Lp-steepest direction with replacement update. In the
  gradient-induced case, this is the generalized-power-style boundary jump.
- `power_add__objective_gradient` and `power_replace__objective_gradient`: power
  labels using the objective-gradient-induced steepest direction, with additive
  or replacement proposal.
- `power_add__pure_jvp_vjp` and `power_replace__pure_jvp_vjp`: q-aware
  JVP/VJP-based power direction, with additive or replacement proposal.
- `power_add__affine_jvp_vjp` and `power_replace__affine_jvp_vjp`: q-aware
  JVP/VJP direction using an affine residual-side q map, with additive or
  replacement proposal.

## Interpretation Guidance

Observed evidence as of the 2026-05-18 status check: completed static
figure sets exist for `p=2,q=2`, `p=1,q=1`, and `p=1,q=2`. The `p=1,q=inf`
run is currently started/running and had no figure files at the check. Later P/Q
pairs should not be described as completed unless their manifest and figures exist.

Inference from the figure definitions: use actual loss curves for optimizer
quality, boundary ratio for how fast the method reaches the epsilon constraint,
and roughness/spectrum/delta plots for whether the final perturbation looks
smooth, high-frequency, or physically questionable.

Preservation note: existing figures should be kept. New explanatory or cleaned
plots should be written under new file names or new figure directories unless
replacement is explicitly requested.

## Line Count And Averaging Clarification - 2026-05-18

Observed from `per_step_metrics.csv` and `per_sample_step_metrics.csv` for the
completed runs `p=2,q=2`, `p=1,q=1`, and `p=1,q=2`:

- Each method line in the existing curve plots is the batch mean column
  `loss3_q_mean` at each optimization step.
- The batch size is `100` samples. This is not 100 independent repeated runs;
  it is one run where each optimizer method is applied to 100 dataset samples
  and then averaged at each step.
- The existing old curve plots did not draw uncertainty shading even though the
  CSV contains `loss3_q_std`. That standard deviation is across the 100 samples
  at the same step, not across random seeds.
- `p=2,q=2` full `main` figures have 11 lines because they include many
  direction/proposal ablation variants. `p=1,q=1` and `p=1,q=2` `pq_key`
  figures have 7 lines.

New actual-loss plots with sample-standard-deviation shading were generated
without deleting or overwriting old figures:

- `p=2,q=2`:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/actual_loss_with_std_clean/actual_loss3_q_mean_std_core3.png`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/actual_loss_with_std_clean/actual_loss3_q_mean_std_power_replacement_variants.png`
- `p=1,q=1`:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1/figures/actual_loss_with_std_clean/actual_loss3_q_mean_std_core3.png`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1/figures/actual_loss_with_std_clean/actual_loss3_q_mean_std_power_replacement_variants.png`
- `p=1,q=2`:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2/figures/actual_loss_with_std_clean/actual_loss3_q_mean_std_core3.png`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2/figures/actual_loss_with_std_clean/actual_loss3_q_mean_std_power_replacement_variants.png`

Recommended reading order:

1. `actual_loss3_q_mean_std_core3.png`: compare the three main ideas only:
   raw PGD, LP-steepest PGD, and GPI-style replacement.
2. `actual_loss3_q_mean_std_power_replacement_variants.png`: compare the
   replacement/power direction variants after the main three-method comparison.
3. Old all-method plots are useful for debugging, but they are too crowded for
   presentation or first-pass scientific interpretation.

## 3D Delta Surface Figures - 2026-05-18

New 3D surface plots are available under each completed run:

```text
figures/delta_surfaces_3d_20260518/
```

They are the 3D version of `delta_heatmaps`: X axis is space index, Y axis is
step `0..100`, and Z axis is the perturbation value. They were generated from
saved `trajectory_samples.npz` files with `--space-stride 4` and `--step-stride 1`.
Existing heatmaps and old figures were not deleted or overwritten.

Detailed record: `docs/loss3_delta_3d_surface_visualizations_20260518.md`.

