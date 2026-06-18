# Darcy Fixed vs Random Budget Generalization Comparison

Date: 2026-06-18

This note records the comparison requested for Darcy/SIR20 adversarial training:
for the same training objective (`loss1`, `loss2`, `loss3`), does randomizing the
Darcy budget help the generalization Relative L2 decrease more than using a fixed
budget?

## Runs Compared

- Fixed budget run: old Darcy fixed `epsilon=1` / fixed budget logs from
  `adversarial_training_runs/darcy_binary_loss3targeted_*_20260612*`.
- Random budget run: current Darcy/SIR20 run with per-batch/per-sample random
  budget multiplier `0.25` to `1.75`, under
  `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617`.

The primary metric below is the 50-dataset mean generalization `relative_l2`.
For the main conclusion, each run is compared against its own epoch 0:

```text
delta_vs_epoch0 = relative_l2(epoch) - relative_l2(epoch 0)
```

Negative is better because it means the generalization Relative L2 decreased from
that run's own starting point.

## Main Result

| Method | Better for generalization decrease? | Fixed final delta | Random final delta | Fixed best delta | Random best delta | Fixed late mean delta | Random late mean delta |
|---|---|---:|---:|---:|---:|---:|---:|
| loss1 | fixed budget is better | -0.003408 | +0.008754 | -0.038076 | -0.009987 | +0.001428 | +0.007218 |
| loss2 | random budget is slightly better | -0.015251 | -0.020987 | -0.024225 | -0.028162 | -0.009211 | -0.014829 |
| loss3 | random budget is clearly better | -0.038263 | -0.073581 | -0.046561 | -0.074326 | -0.038264 | -0.068279 |

Common-final checkpoints:

| Method | Fixed common epoch | Fixed final mean | Random common epoch | Random final mean |
|---|---:|---:|---:|---:|
| loss1 | 3000 | 0.087929 | 3000 | 0.131502 |
| loss2 | 3070 | 0.076086 | 3070 | 0.101762 |
| loss3 | 3030 | 0.053074 | 3030 | 0.049167 |

Best-checkpoint values within the common range:

| Method | Fixed best epoch | Fixed best mean | Random best epoch | Random best mean |
|---|---:|---:|---:|---:|
| loss1 | 1005 | 0.053261 | 916 | 0.112762 |
| loss2 | 3059 | 0.067112 | 2780 | 0.094587 |
| loss3 | 2596 | 0.044776 | 2700 | 0.048423 |

## Interpretation

- `loss1`: random budget hurts this run. Fixed budget gives a better absolute
  final loss and a much better best checkpoint. Relative to each run's own epoch
  0, fixed budget has the stronger decrease.
- `loss2`: random budget is modestly better if judged by improvement from its own
  epoch 0, but the absolute loss is still lower in the old fixed-budget run.
- `loss3`: random budget gives the strongest improvement from its own epoch 0 and
  also has the lower common-final absolute Relative L2 among the two runs.

Short answer: random budget is not uniformly better. It helps `loss3` a lot, helps
`loss2` mildly by the delta-vs-epoch0 criterion, and hurts `loss1`.

## Caveats

This is not a perfectly controlled A/B experiment. The old fixed-budget run and
the current random-budget run differ in more than the randomization itself:

- Epoch 0 generalization means differ: fixed starts around `0.091337`, random
  starts around `0.122749`.
- The old run used the older Darcy baseline/checkpoint setup, while the current
  run uses the screen baseline/setup.
- Previous checks also showed differences in evaluation sample settings between
  historical logs and the current run.

Because of those differences, the safest reading is:

- Use `delta_vs_epoch0` to judge whether training made each run improve relative
  to itself.
- Use absolute final means as descriptive evidence, not as a pure causal proof
  that randomization alone caused the difference.

## Artifacts

Generated local artifacts:

- Clear per-dataset fixed-vs-random generalization panels:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_generalization_panels_clear/`
- Loss3 absolute Relative L2 panels:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_generalization_panels_clear/darcy_loss3_fixed_vs_random_relative_l2_generalization_part01.png`
  and
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_generalization_panels_clear/darcy_loss3_fixed_vs_random_relative_l2_generalization_part02.png`
- Loss3 delta-vs-own-epoch0 panels:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_generalization_panels_clear/darcy_loss3_fixed_vs_random_delta_vs_epoch0_generalization_part01.png`
  and
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_generalization_panels_clear/darcy_loss3_fixed_vs_random_delta_vs_epoch0_generalization_part02.png`
- Loss1/Loss2 direct fixed-vs-random panels:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_generalization_panels_clear/darcy_loss1_loss2_fixed_vs_random_relative_l2_generalization_part01.png`
  and
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_generalization_panels_clear/darcy_loss1_loss2_fixed_vs_random_relative_l2_generalization_part02.png`
- Train/test/generalization split-mean fixed-vs-random curves:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_fixed_random_split_curves/`.
  This directory contains RMSE and Relative L2 split curves by epoch for
  `loss1`, `loss2`, and `loss3`, plus 3x3 overview grids.
- Image-only download folder:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/darcy_fixed_random_generalization_figures_download_only_20260618/`.
  This folder contains only PNG files. It includes the 50-dataset generalization
  panels, the clearer fixed-vs-random generalization panels, and the split-mean
  train/test/generalization curves.
- Summary figure:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_50dataset_panels/darcy_fixed_vs_random_generalization_improvement_summary.png`
- Mean curves and delta curves:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/figures_50dataset_panels/darcy_fixed_vs_random_loss123_mean_curves_absolute_and_delta.png`
- Numeric summary CSV:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/generalization_improvement_fixed_vs_random_summary.csv`
- Full per-epoch curve means:
  `analysis_outputs/darcy_fixed_vs_random_eps_loss123_20260618/generalization_curve_means.csv`
