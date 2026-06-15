# Darcy CFlow Final 8-Metric Matrix Summary - 2026-06-15

Status: complete.

Observed from the final seven-model clean evaluation, attack20 outputs, and
SVD/Jacobian outputs in:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/
```

## 52 x 7 x 8 Matrix

The final aligned matrix exists at:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/final_52dataset_7model_8metric_matrix.csv
```

Shape:

- `52` datasets.
- `7` models.
- `364` rows.
- `8` scalar metrics per row.
- No missing metric values.

The eight metrics are:

- `clean_rmse`
- `clean_relative_l2`
- `attack_clean_loss_mean`
- `attack_adv_loss_mean`
- `attack_loss_increase_mean`
- `attack_relative_increase_mean`
- `attack_delta_l2_rms_mean`
- `attack_delta_linf_mean`

Clean metrics were evaluated on the same lossdrop50 selected dataset root used
by attack20/SVD:

```text
generalization_datasets_darcy_lossdrop50_selected_20260607/
```

## Generalization Mean Results

| method | clean RMSE | clean RelL2 | adv loss | loss increase | relative increase |
|---|---:|---:|---:|---:|---:|
| baseline | 5.7172e-04 | 0.093380 | 7.8148e-07 | 4.5306e-07 | 1.733156 |
| loss1 | 4.2089e-04 | 0.068733 | 2.4658e-07 | 6.8343e-08 | 0.465106 |
| loss2 | 4.5517e-04 | 0.074334 | 2.7887e-07 | 7.0467e-08 | 0.378074 |
| loss3 | 2.8260e-04 | 0.046147 | 1.9758e-07 | 1.1736e-07 | 2.417832 |
| Physics Loss | 3.6436e-04 | 0.059502 | 2.1870e-07 | 8.5025e-08 | 0.727942 |
| random clean | 5.9168e-04 | 0.096630 | 4.6508e-07 | 1.1332e-07 | 0.350687 |
| random solver | 2.6392e-04 | 0.043099 | 1.0384e-07 | 3.3746e-08 | 0.676693 |

## Generalization Best Counts

Lower is better for these scalar loss/error metrics. Delta magnitudes are
diagnostic and are not by themselves a robustness claim.

| method | clean RMSE | clean RelL2 | attack clean | adv loss | loss increase | relative increase |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 0 | 0 | 0 |
| loss1 | 0 | 0 | 0 | 0 | 0 | 1 |
| loss2 | 0 | 0 | 0 | 0 | 0 | 15 |
| loss3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Physics Loss | 0 | 0 | 0 | 0 | 0 | 0 |
| random clean | 0 | 0 | 0 | 0 | 0 | 34 |
| random solver | 50 | 50 | 50 | 50 | 50 | 0 |

Conclusion from these final lossdrop50 results:

- `random solver` is best on clean RMSE, clean Relative L2, attack clean loss,
  attack final loss, and absolute attack loss increase across all 50
  generalization datasets.
- `loss3` is the strongest among the four adversarial/self-attack objectives on
  clean generalization error, and it is far better than baseline, loss1, loss2,
  Physics Loss, and random clean on the clean metrics.
- `relative increase` has a different denominator effect: random clean and
  loss2 win many relative-increase counts because their clean losses are larger
  or their proportional gain is smaller, but their absolute clean/adv losses are
  not best.

## SVD/Jacobian

SVD/Jacobian is intentionally smaller, as requested, because it is expensive:

- `25` fixed samples.
- `7` models.
- `175` rows.

The SVD/Jacobian metrics are in:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/svd_jacobian_metrics.csv
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/final_svd25_model_means.csv
```

Mean SVD/Jacobian diagnostics:

| method | sigma | J^T error norm | SVD attack cos | J^T error attack cos | top-k attack subspace cos |
|---|---:|---:|---:|---:|---:|
| baseline | 0.001142 | 5.1659e-05 | 0.0772 | -0.1369 | 0.4590 |
| loss1 | 0.001330 | 3.9641e-05 | 0.0904 | -0.1210 | 0.3855 |
| loss2 | 0.001296 | 4.3672e-05 | 0.0860 | -0.1355 | 0.3946 |
| loss3 | 0.001572 | 2.1018e-05 | 0.0381 | 0.1481 | 0.3491 |
| Physics Loss | 0.001354 | 3.4213e-05 | 0.0408 | -0.0078 | 0.3885 |
| random clean | 0.001300 | 5.7718e-05 | 0.0545 | -0.1254 | 0.3989 |
| random solver | 0.001346 | 2.0164e-05 | 0.0645 | -0.0697 | 0.3897 |

SVD/Jacobian interpretation:

- `random solver` has the lowest mean `J^T error` norm, with `loss3` very close.
- `loss3` has the largest mean top singular value in this 25-sample diagnostic,
  so top singular value alone does not explain the final attack robustness.
- Single-vector cosine alignments between the top singular vector, `J^T error`,
  and the actual attack delta are weak on average; the attack delta is not simply
  the top singular vector.

## Generated Files

- `data/final_eval_metrics.csv`
- `data/final_eval_summary_by_model_split.csv`
- `data/final_52dataset_7model_8metric_matrix.csv`
- `data/final_52dataset_7model_8metric_ranked.csv`
- `data/final_8metric_model_means.csv`
- `data/final_8metric_best_counts.csv`
- `data/final_svd25_model_means.csv`
- `reports/final_8metric_matrix_summary.md`

Implementation:

```text
tools/summarize_darcy_final_8metric_matrix_20260615.py
```
