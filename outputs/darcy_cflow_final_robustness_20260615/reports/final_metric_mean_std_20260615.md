# Darcy Final Robustness Mean/Std Tables

This report records explicit mean/std tables derived from the final attack50 and SVD/Jacobian raw outputs.

## Provenance

- Attack rows: `18200` = 7 models x 52 datasets x 50 samples.
- Attack steps: `[50]`.
- SVD/Jacobian rows: `175` = 7 models x 25 fixed samples.
- Old lossdrop50 token found: `False`.
- Smoke token found: `False`.

## Attack50 Generalization Mean/Std

| method_display | sample_count | dataset_count | clean_loss_mean | clean_loss_std | adv_loss_mean | adv_loss_std | loss_increase_mean | loss_increase_std | relative_increase_mean | relative_increase_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2500 | 50 | 1.4816e-06 | 1.18366e-06 | 1.11625e-05 | 7.44979e-07 | 9.68093e-06 | 1.36386e-06 | 17.7507 | 26.5887 |
| loss1 | 2500 | 50 | 7.38492e-07 | 6.1274e-07 | 4.66556e-06 | 9.43964e-07 | 3.92706e-06 | 9.9652e-07 | 14.0778 | 17.6537 |
| loss2 | 2500 | 50 | 6.29486e-07 | 5.39989e-07 | 4.19417e-06 | 9.0255e-07 | 3.56469e-06 | 9.17269e-07 | 16.466 | 22.3986 |
| loss3 | 2500 | 50 | 2.88169e-07 | 2.23217e-07 | 1.98716e-06 | 9.52954e-07 | 1.699e-06 | 8.86811e-07 | 8.48128 | 8.27228 |
| Physics Loss | 2500 | 50 | 7.22671e-07 | 5.80885e-07 | 4.69654e-06 | 8.29735e-07 | 3.97387e-06 | 9.11407e-07 | 15.1758 | 22.5812 |
| random clean | 2500 | 50 | 6.68449e-07 | 5.74389e-07 | 4.35924e-06 | 1.02939e-06 | 3.69079e-06 | 1.04659e-06 | 15.2786 | 20.0931 |
| random solver | 2500 | 50 | 9.31897e-07 | 6.7665e-07 | 5.75595e-06 | 3.19607e-07 | 4.82405e-06 | 7.41465e-07 | 13.3171 | 19.0807 |

## SVD/Jacobian Mean/Std

| method_display | sample_count | dataset_count | sigma_input_right_mean | sigma_input_right_std | jt_error_l2_norm_mean | jt_error_l2_norm_std | topk_subspace_cos_jt_error_mean | topk_subspace_cos_jt_error_std | topk_subspace_cos_attack_delta_mean | topk_subspace_cos_attack_delta_std | angle_jt_error_attack_delta_deg_mean | angle_jt_error_attack_delta_deg_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 25 | 23 | 0.00127159 | 0.00013504 | 9.29277e-05 | 4.64608e-05 | 0.981275 | 0.00973193 | 0.416569 | 0.096246 | 96.4064 | 5.63187 |
| loss1 | 25 | 23 | 0.00182142 | 0.000265135 | 9.9221e-05 | 5.78332e-05 | 0.97528 | 0.0121014 | 0.318049 | 0.120427 | 93.7638 | 9.45093 |
| loss2 | 25 | 23 | 0.00175079 | 0.000254529 | 9.22537e-05 | 5.20638e-05 | 0.974519 | 0.0163788 | 0.307293 | 0.110943 | 93.6368 | 6.80019 |
| loss3 | 25 | 23 | 0.00235389 | 0.000399373 | 6.1568e-05 | 3.48649e-05 | 0.967706 | 0.0138495 | 0.292265 | 0.122517 | 86.1309 | 11.3964 |
| Physics Loss | 25 | 23 | 0.0018555 | 0.000277948 | 0.000103396 | 5.89499e-05 | 0.971787 | 0.0223673 | 0.308166 | 0.118103 | 89.3321 | 7.1795 |
| random clean | 25 | 23 | 0.00172493 | 0.000260395 | 9.11171e-05 | 5.35073e-05 | 0.974833 | 0.0133883 | 0.337281 | 0.118777 | 93.7278 | 8.75038 |
| random solver | 25 | 23 | 0.00189828 | 0.000286945 | 0.00012357 | 6.88485e-05 | 0.967935 | 0.0372588 | 0.289419 | 0.117049 | 89.1958 | 4.92235 |

## Files

- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/attack50_52dataset_7model_mean_std.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/attack50_by_model_split_mean_std.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_jacobian_25sample_7model_mean_std.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_jacobian_25sample_by_model_split_mean_std.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_block2_top10_singular_values_raw.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_block2_top10_singular_values_by_model_mean_std.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_jacobian_vector_manifest_with_metrics.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/provenance.json`
