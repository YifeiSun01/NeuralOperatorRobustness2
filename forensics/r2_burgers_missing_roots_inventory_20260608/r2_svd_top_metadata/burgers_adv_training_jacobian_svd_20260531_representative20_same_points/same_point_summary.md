# Representative Same-Point Burgers Jacobian/SVD Summary

This run compares the same fixed input points `x` across three model checkpoints.
The labels `adv_only` and `clean_plus_adv` refer to model checkpoints, not input sample sources.

For each fixed point:

```text
J_error_baseline(x)       = J_model_baseline(x) - J_solver(x)
J_error_adv_only(x)       = J_model_adv_only_trained(x) - J_solver(x)
J_error_clean_plus_adv(x) = J_model_clean_plus_adv_trained(x) - J_solver(x)
```

## Sample Mix

- `generalization`: 10 fixed input points
- `test`: 4 fixed input points
- `train`: 6 fixed input points

## J_error Spectral Norm Split Summary

| source_split | n_points | j_error_spectral_baseline_mean | j_error_spectral_adv_only_trained_mean | j_error_spectral_clean_plus_adv_trained_mean | adv_only_lower_count | clean_plus_adv_lower_count | adv_only_mean_ratio_to_baseline | clean_plus_adv_mean_ratio_to_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 10 | 3.54311 | 1.49217 | 1.70049 | 10 | 9 | 0.471689 | 0.572851 |
| test | 4 | 0.769029 | 0.438301 | 0.705828 | 4 | 2 | 0.585922 | 0.952391 |
| train | 6 | 1.508 | 0.489776 | 0.754874 | 6 | 5 | 0.480284 | 0.713471 |
| ALL | 20 | 2.37776 | 0.980677 | 1.21787 | 20 | 16 | 0.497114 | 0.690945 |

## Main Reading

- ADV-only trained model has lower `J_error` spectral norm than baseline on 20/20 fixed points.
- clean+ADV trained model has lower `J_error` spectral norm than baseline on 16/20 fixed points.
- Mean ratio ADV-only / baseline: 0.4971.
- Mean ratio clean+ADV / baseline: 0.6909.

## Key Files

- `representative20_sample_manifest.csv`
- `same_point_j_error_spectral_norm_comparison.csv`
- `same_point_model_solver_error_spectral_norm_comparison.csv`
- `same_point_j_error_spectral_norm_split_summary.csv`
- `same_point_j_error_top20_singular_values_wide.csv`
- `mean_j_error_top20_singular_values_by_split_model.csv`
- `jacobian_svd_summary.csv`
- `top_singular_values_long.csv`
