# Darcy CFlow Final Robustness on Binary 20260611 Data - 2026-06-15

This note records the completed final robustness run for the Darcy/SIR20
time-matched model set. It supersedes any earlier smoke, one-epoch, or
June-7 `lossdrop50_selected` robustness summaries.

## Scope

- Output bundle: `outputs/darcy_cflow_final_robustness_20260615/`
- Generalization root: `generalization_datasets_darcy_binary_loss3targeted_20260611/`
- Rejected old root: `generalization_datasets_darcy_lossdrop50_selected_20260607/`
- Dataset count: train + test + 50 generalization datasets
- Models: baseline, loss1, loss2, loss3, Physics Loss, random clean, random solver
- Attack: 50 PGD steps, 50 samples per dataset
- SVD/Jacobian: fixed 25 samples shared across all models

The provenance check reports:

- `old_lossdrop50_token_found = false`
- `generalization_dataset_prefix_all_current_20260611 = true`
- `attack_rows = 18200`
- `svd_rows = 175`
- `attack_steps = [50]`

## Main Result

On the 50 binary 20260611 generalization datasets, loss3 is the best model on the
main clean and attack robustness metrics:

| model | samples | clean loss | adv loss | loss increase | relative increase |
|---|---:|---:|---:|---:|---:|
| loss3 | 2500 | 2.88169e-07 | 1.98716e-06 | 1.699e-06 | 8.48128 |
| loss2 | 2500 | 6.29486e-07 | 4.19417e-06 | 3.56469e-06 | 16.466 |
| random clean | 2500 | 6.68449e-07 | 4.35924e-06 | 3.69079e-06 | 15.2786 |
| loss1 | 2500 | 7.38492e-07 | 4.66556e-06 | 3.92706e-06 | 14.0778 |
| Physics Loss | 2500 | 7.22671e-07 | 4.69654e-06 | 3.97387e-06 | 15.1758 |
| random solver | 2500 | 9.31897e-07 | 5.75595e-06 | 4.82405e-06 | 13.3171 |
| baseline | 2500 | 1.4816e-06 | 1.11625e-05 | 9.68093e-06 | 17.7507 |

On the fixed 25 SVD/Jacobian samples, loss3 also has the lowest attack loss
increase, error norm, and `J^T error` norm. The top singular value alone does not
rank loss3 best, so it should not be used by itself as the robustness explanation.

| model | samples | attack increase | J^T error norm | sigma input right | angle singular-delta | angle J^T error-delta |
|---|---:|---:|---:|---:|---:|---:|
| loss3 | 25 | 1.71618e-06 | 6.1568e-05 | 0.00235389 | 80.405 | 86.131 |
| loss2 | 25 | 3.39291e-06 | 9.22537e-05 | 0.00175079 | 92.111 | 93.637 |
| random clean | 25 | 3.45293e-06 | 9.11171e-05 | 0.00172493 | 91.172 | 93.728 |
| loss1 | 25 | 3.64937e-06 | 9.9221e-05 | 0.00182142 | 85.902 | 93.764 |
| Physics Loss | 25 | 3.7135e-06 | 0.000103396 | 0.0018555 | 88.440 | 89.332 |
| random solver | 25 | 4.43937e-06 | 0.00012357 | 0.00189828 | 87.847 | 89.196 |
| baseline | 25 | 8.94546e-06 | 9.29277e-05 | 0.00127159 | 92.766 | 96.406 |

## Files

- Raw attack samples: `outputs/darcy_cflow_final_robustness_20260615/data/robustness_attack_52datasets_samples.csv`
- SVD/Jacobian samples: `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv`
- Attack deltas: `outputs/darcy_cflow_final_robustness_20260615/data/robustness_deltas/`
- SVD/Jacobian vectors: `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_vectors/`
- Summary report: `outputs/darcy_cflow_final_robustness_20260615/reports/final_robustness_summary_20260615.md`
- Provenance: `outputs/darcy_cflow_final_robustness_20260615/data/final_robustness_provenance.json`

## Interpretation

For the current binary 20260611 Darcy/SIR20 setting, the final 3000-3500 epoch
loss3 model is the strongest overall model by the requested clean generalization
and attack50 robustness metrics. Random clean is usually the next strongest among
the random baselines, and random solver is weaker on this binary 20260611
generalization root. This is the dataset family used by the organized-release
loss curves.
