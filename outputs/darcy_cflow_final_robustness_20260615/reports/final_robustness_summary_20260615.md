# Darcy CFlow Final Robustness Summary - 2026-06-15

## Scope

Observed from the final post-hoc robustness run in this bundle.

- Generalization root: `generalization_datasets_darcy_binary_loss3targeted_20260611/`
- Generalization datasets: `50`
- Generalization prefix check: `True`
- Old lossdrop50 token found in raw summary inputs: `False`
- Attack rows: `18200`
- SVD/Jacobian rows: `175`
- Attack steps: `[50]`

The 52-dataset attack set is train + test + 50 binary 20260611 generalization datasets.

## Attack50 Generalization Means

| model | samples | clean loss | adv loss | loss increase | relative increase | delta L2 RMS |
|---|---:|---:|---:|---:|---:|---:|
| loss3 | 2500 | 2.88169e-07 | 1.98716e-06 | 1.699e-06 | 8.48128 | 3.68202 |
| loss2 | 2500 | 6.29486e-07 | 4.19417e-06 | 3.56469e-06 | 16.466 | 3.67253 |
| random clean | 2500 | 6.68449e-07 | 4.35924e-06 | 3.69079e-06 | 15.2786 | 3.6725 |
| loss1 | 2500 | 7.38492e-07 | 4.66556e-06 | 3.92706e-06 | 14.0778 | 3.68674 |
| Physics Loss | 2500 | 7.22671e-07 | 4.69654e-06 | 3.97387e-06 | 15.1758 | 3.69384 |
| random solver | 2500 | 9.31897e-07 | 5.75595e-06 | 4.82405e-06 | 13.3171 | 3.72643 |
| baseline | 2500 | 1.4816e-06 | 1.11625e-05 | 9.68093e-06 | 17.7507 | 4.19273 |

## SVD/Jacobian Fixed-25 Means

| model | samples | attack increase | J^T error norm | sigma input right | angle singular-delta | angle J^T error-delta |
|---|---:|---:|---:|---:|---:|---:|
| loss3 | 25 | 1.71618e-06 | 6.1568e-05 | 0.00235389 | 80.405 | 86.131 |
| loss2 | 25 | 3.39291e-06 | 9.22537e-05 | 0.00175079 | 92.111 | 93.637 |
| random clean | 25 | 3.45293e-06 | 9.11171e-05 | 0.00172493 | 91.172 | 93.728 |
| loss1 | 25 | 3.64937e-06 | 9.9221e-05 | 0.00182142 | 85.902 | 93.764 |
| Physics Loss | 25 | 3.7135e-06 | 0.000103396 | 0.0018555 | 88.440 | 89.332 |
| random solver | 25 | 4.43937e-06 | 0.00012357 | 0.00189828 | 87.847 | 89.196 |
| baseline | 25 | 8.94546e-06 | 9.29277e-05 | 0.00127159 | 92.766 | 96.406 |

## Winner Summary

| metric system | scope | metric | best model | best value | ranking |
|---|---|---|---|---:|---|
| attack50_52datasets_50samples | generalization | clean_loss_mean | loss3 | 2.88169e-07 | `loss3,loss2,random_clean,physics_loss,loss1,random_solver,baseline` |
| attack50_52datasets_50samples | generalization | adv_loss_mean | loss3 | 1.98716e-06 | `loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | generalization | loss_increase_mean | loss3 | 1.699e-06 | `loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | generalization | relative_increase_mean | loss3 | 8.48128 | `loss3,random_solver,loss1,physics_loss,random_clean,loss2,baseline` |
| attack50_52datasets_50samples | generalization | delta_l2_rms_mean | random clean | 3.6725 | `random_clean,loss2,loss3,loss1,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | generalization | delta_linf_mean | baseline | 9 | `baseline,loss1,loss2,loss3,physics_loss,random_clean,random_solver` |
| attack50_52datasets_50samples | test | clean_loss_mean | random solver | 2.99052e-08 | `random_solver,loss2,physics_loss,loss1,random_clean,loss3,baseline` |
| attack50_52datasets_50samples | test | adv_loss_mean | random clean | 1.04726e-06 | `random_clean,loss1,loss2,loss3,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | test | loss_increase_mean | random clean | 1.00479e-06 | `random_clean,loss1,loss2,loss3,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | test | relative_increase_mean | random clean | 25.1191 | `random_clean,loss3,loss1,baseline,loss2,physics_loss,random_solver` |
| attack50_52datasets_50samples | test | delta_l2_rms_mean | random clean | 3.76007 | `random_clean,loss1,loss2,physics_loss,loss3,random_solver,baseline` |
| attack50_52datasets_50samples | test | delta_linf_mean | baseline | 9 | `baseline,loss1,loss2,loss3,physics_loss,random_clean,random_solver` |
| attack50_52datasets_50samples | train | clean_loss_mean | random solver | 2.43531e-08 | `random_solver,physics_loss,loss2,loss1,random_clean,baseline,loss3` |
| attack50_52datasets_50samples | train | adv_loss_mean | random clean | 8.39911e-07 | `random_clean,loss2,loss1,loss3,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | train | loss_increase_mean | random clean | 8.06238e-07 | `random_clean,loss2,loss1,loss3,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | train | relative_increase_mean | random clean | 25.2693 | `random_clean,loss3,loss1,loss2,physics_loss,random_solver,baseline` |
| attack50_52datasets_50samples | train | delta_l2_rms_mean | random clean | 3.84302 | `random_clean,loss1,loss2,loss3,physics_loss,baseline,random_solver` |
| attack50_52datasets_50samples | train | delta_linf_mean | baseline | 9 | `baseline,loss1,loss2,loss3,physics_loss,random_clean,random_solver` |
| svd_jacobian_25samples | fixed_25_samples | attack_loss_increase_mean | loss3 | 1.71618e-06 | `loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline` |
| svd_jacobian_25samples | fixed_25_samples | attack_relative_increase_mean | loss1 | 14.0503 | `loss1,random_clean,physics_loss,baseline,loss3,loss2,random_solver` |
| svd_jacobian_25samples | fixed_25_samples | error_l2_norm_mean | loss3 | 0.0388067 | `loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline` |
| svd_jacobian_25samples | fixed_25_samples | jt_error_l2_norm_mean | loss3 | 6.1568e-05 | `loss3,random_clean,loss2,baseline,loss1,physics_loss,random_solver` |
| svd_jacobian_25samples | fixed_25_samples | sigma_input_right_mean | baseline | 0.00127159 | `baseline,random_clean,loss2,loss1,physics_loss,random_solver,loss3` |
| svd_jacobian_25samples | fixed_25_samples | block2_sigma1_mean | baseline | 0.00124975 | `baseline,random_clean,loss2,loss1,physics_loss,random_solver,loss3` |
| svd_jacobian_25samples | fixed_25_samples | angle_singular_attack_delta_deg_mean | loss3 | 80.4053 | `loss3,loss1,random_solver,physics_loss,random_clean,loss2,baseline` |
| svd_jacobian_25samples | fixed_25_samples | angle_jt_error_attack_delta_deg_mean | loss3 | 86.1309 | `loss3,random_solver,physics_loss,loss2,random_clean,loss1,baseline` |

## Correlation Files

- `data/svd_scalar_correlations.csv` contains scalar correlations for same-sample SVD/Jacobian/attack metrics.
- `data/svd_jacobian_metrics.csv` contains the per-model, per-sample vector angle/cosine/correlation diagnostics.
