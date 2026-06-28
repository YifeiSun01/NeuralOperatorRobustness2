# Darcy CFlow Residual Jacobian SVD

Date: 2026-06-15

Scope: final seven Darcy CFlow models on the fixed 25 SVD/Jacobian samples, aligned to the 50-step attack manifest. Generalization samples are from `generalization_datasets_darcy_binary_loss3targeted_20260611`. No `generalization_datasets_darcy_lossdrop50_selected_20260607` data is used here.

## What Was Computed

- Existing model result: `J_model` block/2 top-10 SVD was already available.
- Newly computed solver result: `J_solver` block/2 top-10 SVD.
- Newly computed residual result: `J_model - J_solver` block/2 top-10 SVD.
- Newly computed residual gradient:
  `(J_model - J_solver)^T (model(x) - solver(x))`.
- Attack comparison uses the existing final attack table with `attack_steps = 50`.

The residual spectral norm is:

```text
residual_sigma1 = ||P_out (J_model - J_solver) L_in||_2
```

The residual gradient norm is:

```text
residual_jt_error_norm = ||(J_model - J_solver)^T (model(x) - solver(x))||_2
```

## Main Result

On the 25 fixed samples, Loss3 is best on the residual operator scalar metrics:

| metric | Loss3 wins |
|---|---:|
| residual spectral norm | 25/25 |
| residual error L2 norm | 16/25 |
| residual JT error norm | 20/25 |
| attack adv loss | 21/25 |
| attack loss increase | 21/25 |

By-model means:

| method | clean loss | adv loss | loss increase | residual sigma1 | residual error L2 | residual JT norm |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 9.1093e-07 | 9.8564e-06 | 8.9455e-06 | 0.00246356 | 0.073329 | 1.9314e-04 |
| loss1 | 4.6244e-07 | 4.1118e-06 | 3.6494e-06 | 0.00207122 | 0.0513343 | 1.1425e-04 |
| loss2 | 4.0213e-07 | 3.7950e-06 | 3.3929e-06 | 0.00207742 | 0.0481466 | 1.0684e-04 |
| loss3 | 1.7521e-07 | 1.8914e-06 | 1.7162e-06 | 0.00170213 | 0.0334209 | 4.7257e-05 |
| Physics Loss | 4.5121e-07 | 4.1647e-06 | 3.7135e-06 | 0.00204244 | 0.0508341 | 1.1251e-04 |
| random clean | 4.4480e-07 | 3.8977e-06 | 3.4529e-06 | 0.002143 | 0.0503255 | 1.1327e-04 |
| random solver | 6.0717e-07 | 5.0465e-06 | 4.4394e-06 | 0.00200877 | 0.0593998 | 1.3103e-04 |

## Correlations

Across all 25 fixed samples x 7 models:

| pair | Pearson | Spearman |
|---|---:|---:|
| loss increase vs residual JT norm | 0.504257 | 0.502891 |
| loss increase vs residual sigma1 | 0.561659 | 0.438184 |
| loss increase vs residual error L2 | 0.534601 | 0.495813 |
| residual JT norm vs residual sigma1 | 0.909114 | 0.945159 |

Across the 21 fixed generalization samples x 7 models:

| pair | Pearson | Spearman |
|---|---:|---:|
| loss increase vs residual JT norm | 0.353294 | 0.209620 |
| loss increase vs residual sigma1 | 0.346862 | 0.114519 |
| loss increase vs residual error L2 | 0.345888 | 0.201369 |
| residual JT norm vs residual sigma1 | 0.918346 | 0.916724 |

## Interpretation

The earlier counterintuitive spectral-norm result came from the model-only Jacobian:

```text
||P_out J_model L_in||_2
```

That is not the model-minus-solver residual operator. Once the residual operator is computed directly, Loss3 has the smallest residual spectral norm on every one of the 25 fixed samples and the smallest residual gradient norm on most samples.

## Output Files

- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_svd_25samples_7models.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_svd_by_model.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_aligned_25samples_7models.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_by_model_25samples_20260615.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_correlations_20260615.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_winner_counts_20260615.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_summary_20260615.md`
