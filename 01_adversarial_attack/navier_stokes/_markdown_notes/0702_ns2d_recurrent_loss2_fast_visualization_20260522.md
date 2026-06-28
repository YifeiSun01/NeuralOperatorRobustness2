# NS2D Recurrent FNO Loss2 Fast Visualization - 2026-05-22

## Scope

This note records the completed `loss2` portion of the interrupted NS2D recurrent FNO core4 attack launch. The purpose was to check whether the current `epsilon=32`, `alpha=1`, `p=2`, `q=2` setting produced meaningful perturbations and whether the scale looked visually reasonable.

No model or solver computation was rerun for this visualization. The plots were generated on CPU from saved arrays and CSV metrics only.

## Source Evidence

Observed from:

- Attack root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_aaaaaaaaaw_p2_q2_20260522_030849_UTC/batch_0000_0009/loss2`
- Visualization report: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_fast_panels_20260522/loss2_fast_panel_report.json`
- Curve plot: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_fast_panels_20260522/loss2_method_curves.png`
- Fast panel example: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_fast_panels_20260522/loss2_steepest_add_samplepos0_idx0_fast_panels.png`

## Observed Metrics

The completed `loss2/all_a_target_w` group produced nonzero perturbations for all four methods.

| method | final delta L2 mean | final boundary ratio | loss2 final/start | true loss final/start |
| --- | ---: | ---: | ---: | ---: |
| raw_add | 12.0754 | 0.3774 | 0.9516 | 1.0892 |
| raw_replace | 32.0000 | 1.0000 | 0.9960 | 1.1877 |
| steepest_add | 32.0000 | 1.0000 | 0.9830 | 1.1984 |
| steepest_replace | 32.0000 | 1.0000 | 0.9960 | 1.1877 |

Observed per-sample final `L_inf` perturbation magnitudes in the plotted samples were about `0.05` to `0.37`.

## Interpretation

Observed evidence says this `loss2` run is meaningful in the basic sense: the attack did not stall at zero, the final deltas are nonzero, and the true all-W loss increased by about `9%` to `20%` depending on method.

Inference from the curves:

- `epsilon=32` is not absurdly tiny. Three methods reached the L2 boundary.
- `raw_add` with `alpha=1` is comparatively weak for 100 steps: it used only about `38%` of the available L2 budget.
- `raw_replace` and `steepest_replace` jump directly to the epsilon boundary, so they are useful as max-budget directional tests, but they are aggressive iterative dynamics.
- `steepest_add` looks like the best candidate from this old single-scale run: it reaches the boundary and gives the largest final true-loss ratio in this group.
- The surrogate `loss2` objective itself did not clearly increase at the final step. It ended near or below its starting value, while the tracked true loss increased. That means this single scale should be treated as a diagnostic, not as proof that `epsilon=32, alpha=1` is optimal.

## Recommendation

For the next corrected run, use the newly added epsilon/alpha sweep support instead of trusting this single value. A small paired sweep is the safest next step:

```bash
EPSILON_ALPHA_PAIRS="16:0.5 32:1 64:2"
```

If runtime needs to stay smaller, run only:

```bash
EPSILON_ALPHA_PAIRS="16:0.5 32:1"
```

The current evidence suggests `epsilon=32` is visually plausible, but the replace methods saturate immediately and `raw_add` is too slow at `alpha=1`. The sweep is needed to see whether smaller epsilon gives cleaner perturbations or whether larger epsilon gives useful true-loss growth without visually unreasonable high-frequency artifacts.

## Remaining Work

- Generate full model-vs-solver panels later if GPU time is available and exact output comparison is needed.
- Do not treat the interrupted full ADW launch as a complete experiment; only `loss1/all_w` and `loss2/all_a_target_w` completed before the process was stopped.
