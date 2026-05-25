# NS2D eps32 alpha10 alt-loss cleanstyle comparison - 2026-05-25

CPU-only redraw from saved attack artifacts. No model, solver, or attack optimization was rerun for these plots.

## Scope

- Problem: NS2D final-state all-W loss3 path.
- Attack setting: `epsilon=32`, `alpha=10`, `optimizer=steepest_add`, `batch_size=10`.
- Compared rows:
  - Initial condition, sample position 0.
  - Normal baseline `Loss 3` / `all_w`.
  - `DISTS`.
  - `MS-SSIM`.
  - `Scattering2D`.
  - `Affine + DISTS`.
  - `Local warp + DISTS`.

## Input Artifacts

- Baseline Loss 3 source:
  `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/steepest_add`
- Alternative-loss source:
  `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps32_alpha10_steepest_add_loss3_allw_alt5_b10_20260525_035657_UTC/eps32_alpha10`
- Plotting script:
  `tools/plot_ns2d_eps32_alpha10_altloss_cleanstyle_comparison_20260525.py`

## Outputs

- Mean curve summary:
  `docs/ns2d_eps32_alpha10_altloss_cleanstyle_comparison_20260525/ns2d_eps32_alpha10_altloss_mean_curves_cleanstyle.png`
- Sample-0 cleanstyle comparison:
  `docs/ns2d_eps32_alpha10_altloss_cleanstyle_comparison_20260525/ns2d_eps32_alpha10_altloss_sample0_cleanstyle_comparison.png`
- Sample-0 model-solver error grid:
  `docs/ns2d_eps32_alpha10_altloss_cleanstyle_comparison_20260525/ns2d_eps32_alpha10_altloss_sample0_model_solver_error_grid.png`
- Summary CSV:
  `docs/ns2d_eps32_alpha10_altloss_cleanstyle_comparison_20260525/ns2d_eps32_alpha10_altloss_summary.csv`
- Manifest:
  `docs/ns2d_eps32_alpha10_altloss_cleanstyle_comparison_20260525/manifest.json`

## Batch-Mean Result Snapshot

| method | active metric initial | active metric final | active ratio | original qnorm clean | original qnorm adv | qnorm increase | final boundary ratio | runtime min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Loss 3 baseline | 68.492 | 304.593 | 4.447 | 68.492 | 304.593 | 236.101 | 1.000 | 30.317 |
| DISTS | 0.168056 | 0.303863 | 1.808 | 68.491 | 204.801 | 136.310 | 0.985 | 20.361 |
| MS-SSIM | 0.091933 | 0.602451 | 6.553 | 68.491 | 283.688 | 215.197 | 0.989 | 20.405 |
| Scattering2D | 6.201e-06 | 0.000126 | 20.372 | 68.491 | 299.939 | 231.448 | 1.000 | 21.225 |
| Affine + DISTS | 0.168773 | 0.295282 | 1.750 | 68.491 | 200.579 | 132.088 | 1.000 | 20.794 |
| Local warp + DISTS | 0.167275 | 0.292725 | 1.750 | 68.491 | 200.135 | 131.644 | 0.992 | 20.465 |

## Verification

- Generated PNGs opened successfully with PIL.
- PNG dimensions:
  - Mean curves: `3267 x 1947`.
  - Sample-0 comparison: `3335 x 3471`.
  - Model-solver error grid: `3914 x 615`.
- Final five-loss sweep scan:
  - Unexpected CSV NaN/Inf: `0`.
  - NPZ NaN/Inf: `0`.
  - Negative loss/norm anomalies: `0`.

