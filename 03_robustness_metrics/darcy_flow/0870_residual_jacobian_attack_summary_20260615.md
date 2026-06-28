# Darcy CFlow Residual Jacobian / Attack Summary

Scope: final seven models, 25 fixed SVD/Jacobian samples, aligned with the attack50 manifest. Generalization samples are from `generalization_datasets_darcy_binary_loss3targeted_20260611`; no `lossdrop50_selected_20260607` data is used in this table.

Definitions: `residual_sigma1 = ||P_out (J_model - J_solver) L_in||_2`; `residual_jt_error_norm = ||(J_model - J_solver)^T (model(x)-solver(x))||_2`; attack loss is 50-step MSE(model, solver) attack.

## By-Model Means
| method | clean loss | adv loss | loss increase | relative increase | residual sigma1 | residual error L2 | residual JT norm | top10 residual vs delta angle | residual JT vs delta angle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 9.1093e-07 | 9.8564e-06 | 8.9455e-06 | 16.8514 | 0.00246356 | 0.073329 | 1.9314e-04 | 76.0416 | 83.4014 |
| loss1 | 4.6244e-07 | 4.1118e-06 | 3.6494e-06 | 14.0503 | 0.00207122 | 0.0513343 | 1.1425e-04 | 75.2805 | 80.0533 |
| loss2 | 4.0213e-07 | 3.7950e-06 | 3.3929e-06 | 17.9768 | 0.00207742 | 0.0481466 | 1.0684e-04 | 77.059 | 81.3309 |
| loss3 | 1.7521e-07 | 1.8914e-06 | 1.7162e-06 | 17.5665 | 0.00170213 | 0.0334209 | 4.7257e-05 | 71.497 | 80.2697 |
| Physics Loss | 4.5121e-07 | 4.1647e-06 | 3.7135e-06 | 16.2162 | 0.00204244 | 0.0508341 | 1.1251e-04 | 76.8222 | 81.3293 |
| random clean | 4.4480e-07 | 3.8977e-06 | 3.4529e-06 | 16.0817 | 0.002143 | 0.0503255 | 1.1327e-04 | 76.2319 | 81.0739 |
| random solver | 6.0717e-07 | 5.0465e-06 | 4.4394e-06 | 19.5897 | 0.00200877 | 0.0593998 | 1.3103e-04 | 76.8983 | 81.4494 |

## Winner Counts, 25 Fixed Samples
- `clean_loss` lower-is-better wins: baseline 0/25, loss1 1/25, loss2 2/25, loss3 16/25, Physics Loss 1/25, random clean 1/25, random solver 4/25
- `adv_loss` lower-is-better wins: baseline 0/25, loss1 1/25, loss2 0/25, loss3 21/25, Physics Loss 3/25, random clean 0/25, random solver 0/25
- `loss_increase` lower-is-better wins: baseline 0/25, loss1 1/25, loss2 0/25, loss3 21/25, Physics Loss 3/25, random clean 0/25, random solver 0/25
- `relative_increase` lower-is-better wins: baseline 0/25, loss1 4/25, loss2 2/25, loss3 8/25, Physics Loss 1/25, random clean 2/25, random solver 8/25
- `residual_block2_sigma1` lower-is-better wins: baseline 0/25, loss1 0/25, loss2 0/25, loss3 25/25, Physics Loss 0/25, random clean 0/25, random solver 0/25
- `residual_error_l2_norm` lower-is-better wins: baseline 0/25, loss1 1/25, loss2 2/25, loss3 16/25, Physics Loss 1/25, random clean 1/25, random solver 4/25
- `residual_jt_error_l2_norm` lower-is-better wins: baseline 0/25, loss1 0/25, loss2 0/25, loss3 20/25, Physics Loss 1/25, random clean 2/25, random solver 2/25

## Correlations
### all_25_samples_x_7_models
| x | y | Pearson | Spearman |
|---|---|---:|---:|
| `loss_increase` | `residual_jt_error_l2_norm` | 0.504257 | 0.502891 |
| `loss_increase` | `residual_block2_sigma1` | 0.561659 | 0.438184 |
| `loss_increase` | `residual_error_l2_norm` | 0.534601 | 0.495813 |
| `adv_loss` | `residual_error_l2_norm` | 0.648374 | 0.706155 |
| `adv_loss` | `residual_jt_error_l2_norm` | 0.623827 | 0.706061 |
| `clean_loss` | `residual_error_l2_norm` | 0.959554 | 1 |
| `residual_jt_error_l2_norm` | `residual_block2_sigma1` | 0.909114 | 0.945159 |
| `relative_increase` | `residual_jt_error_l2_norm` | -0.435353 | -0.479004 |
| `relative_increase` | `residual_block2_sigma1` | -0.41024 | -0.417382 |

### generalization_21_samples_x_7_models
| x | y | Pearson | Spearman |
|---|---|---:|---:|
| `loss_increase` | `residual_jt_error_l2_norm` | 0.353294 | 0.20962 |
| `loss_increase` | `residual_block2_sigma1` | 0.346862 | 0.114519 |
| `loss_increase` | `residual_error_l2_norm` | 0.345888 | 0.201369 |
| `adv_loss` | `residual_error_l2_norm` | 0.504001 | 0.544383 |
| `adv_loss` | `residual_jt_error_l2_norm` | 0.511645 | 0.542516 |
| `clean_loss` | `residual_error_l2_norm` | 0.974629 | 1 |
| `residual_jt_error_l2_norm` | `residual_block2_sigma1` | 0.918346 | 0.916724 |
| `relative_increase` | `residual_jt_error_l2_norm` | -0.544346 | -0.720583 |
| `relative_increase` | `residual_block2_sigma1` | -0.511399 | -0.644217 |

## Key Takeaway
Loss3 is the best model on the residual operator scalar metrics in this 25-sample diagnostic: residual sigma1 is 25/25, residual JT norm is 20/25, attack loss increase is 21/25. The earlier counterintuitive model-only spectral-norm result should not be used as the residual-operator conclusion.
