# Three-Loss 1D Burgers Optimizer Findings Summary - 2026-05-21

Status: consolidated interpretation from existing local experiment artifacts and
figures. No new attack run was performed for this summary.

## Source Evidence

Observed from these local records and artifacts:

- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`
- `docs/loss1_loss2_core4_p2q2_baseline_marked_angles_0to100_20260521.md`
- `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`
- `docs/gpi_fast_optimizer_three_losses_interpretation_20260521.md`
- `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`
- `docs/loss3_hypothesis_validation_status_20260521.md`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`
- Organized PNG folder: `all_requested_figures_20260521/`

## Finding 1: GPI Is the Fastest Early Optimizer Across the Three Original Losses

Observed from the saved 1D Burgers FNO three-method results
(`epsilon=8`, `alpha=0.3`, `p=q=2`, batch size `100`, `steps=100`):

| objective | PGD k95 | LP-steepest PGD k95 | generalized power k95 |
| --- | ---: | ---: | ---: |
| `loss1_original` | 26 | 27 | 3 |
| `loss2_original` | 24 | 27 | 2 |

Observed for `loss3_original`: the generalized-power curve evaluated by
`loss3_original` rises from `0.298` at step `0` to `2.721` at step `1`, `4.683`
at step `2`, `6.581` at step `5`, `6.680` at step `10`, and `6.857` at step
`100`.

Inference: generalized power / GPI-style replacement is best described as the
fastest early optimizer in step count for the three original losses. It reaches
strong or near-final objective values much earlier than additive PGD-style
methods.

Important caveat: this does not mean GPI always has the largest final 300-step
mean loss. Later `loss3` p2q2 alpha/epsilon sweeps show that long additive runs,
especially `steepest_add`, can slightly exceed replacement/GPI in final mean
loss for some settings. The robust claim is speed and early strength, not
unconditional final-loss dominance.

## Finding 2: Loss1 and Loss2 Plateau After Boundary Arrival

Observed from the corrected loss1/loss2 core-four baseline run
(`epsilon=4`, `alpha=0.4`, `p=q=2`, batch size `100`, `steps=100`): all methods
end at the L2 boundary with mean `||delta||_2` about `4.0` and boundary ratio
about `1.0`.

Boundary-gain evidence from `combined_boundary_gain_summary_0to100.csv`:

| objective | method | first boundary k >= 0.99 | loss at boundary | final loss | gain after boundary |
| --- | --- | ---: | ---: | ---: | ---: |
| `loss1` | `raw_add` | 8 | 6.767 | 6.977 | 0.210 |
| `loss2` | `raw_add` | 8 | 6.892 | 7.070 | 0.177 |
| `loss1` | `steepest_add` | 11 | 6.792 | 6.977 | 0.185 |
| `loss2` | `steepest_add` | 11 | 6.913 | 7.074 | 0.161 |
| `loss1` | `raw_replace` | 1 | 6.221 | 6.942 | 0.720 |
| `loss2` | `raw_replace` | 1 | 6.352 | 7.045 | 0.694 |

Inference: for `loss1` and `loss2`, once the perturbation reaches the epsilon
boundary, the remaining optimization space is limited in this baseline setting.
The loss curves mostly plateau, especially for additive methods.

## Finding 3: Loss3 Has Much Larger Boundary-Direction Rotation

Observed from the loss3 mechanism records and p2q2 rollups:

- Replacement/GPI boundary hit step is `1` in the main p2q2 setting.
- Replacement/GPI post-hit angle is about `30.68 deg`.
- Raw-add post-hit angle is about `0.288 deg`.
- Steepest-add post-hit angle is about `0.617 deg`.
- p2q2 replacement/GPI endpoint ratio versus final is `0.961` by step `5`,
  `1.008` by step `10`, and `1.012` by step `20` in the trajectory ray-profile
  evidence.

Inference: for `loss3`, GPI/replacement is not just fast because it reaches the
boundary immediately. It also rotates/refines the perturbation direction much
more aggressively along the boundary, which explains why it can keep improving
after boundary arrival.

## Finding 4: Angle Metrics Mean Different Things

Definitions used in the figures:

- `Delta Prev Angle degrees`: mean `angle(delta_k, delta_{k-1})`. This measures
  how much the actual perturbation changes direction from one step to the next.
- `Direction Prev Angle degrees`: mean `angle(direction_k, direction_{k-1})`.
  This measures how much the proposed optimization/search direction changes.
- `Delta Direction Angle degrees`: mean `angle(delta_k, direction_k)`. This
  measures whether the proposed direction is mostly radial or has substantial
  tangential/boundary-surface component.
- `mean angle` means batch mean over the evaluated samples, not a separate kind
  of angle.

Inference: small `Delta Prev Angle` means the actual perturbation has nearly
stopped rotating. Small `Delta Direction Angle` near the boundary means the
optimizer is mostly pushing radially outward into the epsilon constraint, which
projection will suppress. Large values, especially for loss3 replacement/GPI,
indicate active boundary-surface motion.

## Finding 5: Combined Figures and Download Folder

The clean single folder for downloading the requested figures is:

`all_requested_figures_20260521/`

Observed contents:

- Total PNG count: `62`.
- Non-PNG files in the folder: `0`.
- PNG readability/nonblank check: `62` readable, `0` blank.

Key subfolders:

- `combined/no_std/by_objective/`: three objective rows (`loss1`, `loss2`,
  `loss3`), four optimizer curves per row, mean only.
- `combined/with_std/by_objective/`: same layout with mean +/- std shading.
- `combined/no_std/by_method/`: four method panels, three objective curves per
  panel, mean only.
- `combined/with_std/by_method/`: same layout with mean +/- std shading.
- `loss1_loss2_0to100/`: corrected loss1/loss2 0..100 plots with boundary
  markers and angle diagnostics.
- `loss1_loss2_0to300/`: earlier 0..300 loss1/loss2 baseline plots.

## Final Working Interpretation

The strongest evidence-backed summary is:

> GPI-style replacement is the fastest early optimizer across the three original
> losses in the locally recorded 1D Burgers experiments. For loss1 and loss2,
> the loss mostly saturates after boundary arrival. For loss3, GPI/replacement
> reaches the boundary immediately and then rotates the perturbation direction
> much more aggressively along the boundary, which gives it especially strong
> early performance.

The claim should be written as an empirical optimization-speed and boundary-
geometry result, not as a theorem that GPI is always the globally optimal or
largest-final-loss method.
