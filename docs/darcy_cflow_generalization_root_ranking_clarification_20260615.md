# Darcy CFlow Generalization Root Ranking Clarification - 2026-06-15

Status: clarification after audit.

## Problem

Two different Darcy CFlow generalization roots were being discussed together:

1. The organized-release training curves and polished report figures use:

```text
generalization_datasets_darcy_binary_loss3targeted_20260611/
```

2. The final attack20/SVD25 bundle and aligned 52 x 7 x 8 matrix use:

```text
generalization_datasets_darcy_lossdrop50_selected_20260607/
```

Because these are different 50-dataset suites, their model rankings are not
interchangeable.

## What "lossdrop50 selected" Means

`lossdrop50 selected` refers to the local dataset root:

```text
generalization_datasets_darcy_lossdrop50_selected_20260607/
```

It contains the selected 50 Darcy generalization datasets used by the final
attack20/SVD25 run. The final clean evaluation, attack sample table, and
SVD/Jacobian diagnostics were aligned to this root in:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/
```

## Organized-Release Curve Ranking

The user's visual reading of the organized-release curves is correct for the
curve source root `generalization_datasets_darcy_binary_loss3targeted_20260611`.

Evidence:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_52dataset_metric_long_ranked.csv
```

Generalization mean clean metrics on that root:

| method | RMSE mean | Relative L2 mean | best datasets out of 50 |
|---|---:|---:|---:|
| loss3 | 0.000595 | 0.056154 | 47 |
| loss2 | 0.000870 | 0.081638 | 1 |
| random clean | 0.000884 | 0.082881 | 2 |
| loss1 | 0.000936 | 0.087929 | 0 |
| Physics Loss | 0.000939 | 0.088293 | 0 |
| baseline | 0.000972 | 0.091337 | 0 |
| random solver | 0.001067 | 0.100569 | 0 |

So on the organized-release curve root, `loss3` is clearly best and
`random clean` is better than `random solver`.

## Lossdrop50-Selected Final Matrix Ranking

The final matrix uses the separate lossdrop50 selected root:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/final_52dataset_7model_8metric_matrix.csv
```

Generalization mean clean/attack metrics on that root:

| method | clean RMSE | clean RelL2 | adv loss | loss increase |
|---|---:|---:|---:|---:|
| random solver | 0.000264 | 0.043099 | 1.0384e-07 | 3.3746e-08 |
| loss3 | 0.000283 | 0.046147 | 1.9758e-07 | 1.1736e-07 |
| Physics Loss | 0.000364 | 0.059502 | 2.1870e-07 | 8.5025e-08 |
| loss1 | 0.000421 | 0.068733 | 2.4658e-07 | 6.8343e-08 |
| loss2 | 0.000455 | 0.074334 | 2.7887e-07 | 7.0467e-08 |
| baseline | 0.000572 | 0.093380 | 7.8148e-07 | 4.5306e-07 |
| random clean | 0.000592 | 0.096630 | 4.6508e-07 | 1.1332e-07 |

On this separate root, `random solver` is first on clean RMSE, clean Relative L2,
attack final loss, and absolute attack loss increase.

## Corrected Interpretation

- Do not use the final lossdrop50-selected matrix to describe the
  organized-release training curves.
- Do not use the organized-release curve ranking to describe the final
  lossdrop50-selected attack20/SVD25 matrix.
- For the plotted organized-release curves: `loss3` is best, and `random clean`
  is better than `random solver`.
- For the final lossdrop50-selected attack20/SVD25 matrix: `random solver` is
  best on the main clean/absolute-attack metrics, with `loss3` second on clean
  metrics and attack final loss.
