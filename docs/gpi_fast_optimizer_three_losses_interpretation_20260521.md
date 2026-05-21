# GPI Fast Optimizer Interpretation Across Three Losses - 2026-05-21

Status: interpretation recorded from existing experiment artifacts. No experiment was rerun.

## Source Evidence

Observed from:

- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`
- `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`
- `docs/loss3_hypothesis_validation_status_20260521.md`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`

## Observed Speed Evidence

For `loss1_original` and `loss2_original` on the saved 1D Burgers FNO setting
(`epsilon=8`, `alpha=0.3`, `p=q=2`, batch size `100`, `steps=100`), the first
step reaching 95% of each method's final mean is:

| objective | PGD k95 | LP-steepest PGD k95 | generalized power k95 |
| --- | ---: | ---: | ---: |
| `loss1_original` | 26 | 27 | 3 |
| `loss2_original` | 24 | 27 | 2 |

For `loss3_original`, the recorded generalized-power curve evaluated by
`loss3_original` rises from `0.298` at step 0 to `2.721` at step 1, `4.683` at
step 2, `6.581` at step 5, `6.680` at step 10, and `6.857` at step 100. The
recorded final mean for `loss3_original_generalized_power` is `6.8573`, versus
`6.3782` for `loss3_original_lp_steepest_pgd` and `5.3949` for
`loss3_original_pgd` in that saved comparison.

## Observed Angular/Delta Evidence

For later `loss3` core-four p2q2 diagnostics, `steepest_replace` / GPI-style
replacement reaches the boundary at step `1`, but boundary arrival is not the
whole story. Existing mechanism records note:

- p2q2 replacement/GPI post-hit angle is about `30.68 deg`.
- raw-add post-hit angle is about `0.288 deg` and steepest-add about `0.617 deg`
  in the p2q2 300-step rollup.
- p2q2 replacement/GPI endpoint ratio versus final is `0.961` by step `5`,
  `1.008` by step `10`, and `1.012` by step `20` in the trajectory ray-profile
  evidence.

Observed evidence therefore supports the user's reading: GPI-style replacement
has much faster angular motion, especially for `loss3`, and this fast boundary
direction rotation is a key part of its practical speed.

## Interpretation

It is fair to say that, in the locally recorded 1D Burgers evidence, generalized
power / GPI-style replacement is the fastest optimizer in step count for the
three original losses: it reaches strong or near-final objective values much
earlier than additive PGD-style methods.

The safer paper wording is:

> GPI-style replacement is the fastest early optimizer and the strongest
> practical boundary-direction method in these settings; it reaches the
> perturbation boundary immediately and then changes the perturbation direction
> much more aggressively, especially for `loss3`.

Important caveat: this should not be phrased as "GPI always has the largest
final 300-step mean loss." Existing `loss3` p2q2 300-step sweeps show that long
additive runs, especially `steepest_add`, can slightly exceed replacement/GPI in
final mean loss for some alpha/epsilon settings. The robust claim is speed and
early strength, not unconditional final-loss dominance.
