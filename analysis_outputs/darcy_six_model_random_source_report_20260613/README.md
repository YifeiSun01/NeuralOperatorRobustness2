# Darcy Six-Model Random-Source Report

Created: 2026-06-13T10:51:00.557538+00:00

This report follows the attached runbook: old Darcy artifacts are reused, and only
the two random-source models are newly trained/evaluated. The random-source
training completed at 1100 epochs for both methods.

## Model Names

- `baseline`: original pretrained Darcy FNO, available in clean eval.
- `loss1`, `loss2`, `loss3`: adversarial training checkpoints from the existing 1000-ish run.
- `physics_loss`: existing physics/fixed adversarial training checkpoint, retained because the local old robustness artifacts include it.
- `random_clean_y`: random binary source perturbation with clean target `y` held fixed.
- `random_solver_y`: random binary source perturbation with solver target recomputed after perturbing `a`.

## Important Completeness Notes

- Old clean eval includes `baseline/loss1/loss2/loss3/physics_loss`.
- Old attack20, metric-correlation, and Jacobian/SVD artifacts include `loss1/loss2/loss3/physics_loss`; they do not include baseline.
- Random-source artifacts include `random_clean_y/random_solver_y`.
- baseline attack20/Jacobian/SVD metrics are absent in the reused old robustness artifacts.

## Generalization Clean Loss

| model | dataset_count | mean_relative_l2 | median_relative_l2 | mean_rmse |
| --- | --- | --- | --- | --- |
| loss3 | 50 | 0.0614064 | 0.0623735 | 0.000649839 |
| loss2 | 50 | 0.072153 | 0.0733341 | 0.000770274 |
| random_clean_y | 50 | 0.0739331 | 0.0737779 | 0.000788917 |
| physics_loss | 50 | 0.0878201 | 0.0905458 | 0.000933881 |
| loss1 | 50 | 0.0879285 | 0.0881902 | 0.000936259 |
| baseline | 50 | 0.0913369 | 0.0924669 | 0.000972351 |
| random_solver_y | 50 | 0.0959063 | 0.0995798 | 0.00101952 |

## Attack20 Generalization Robustness

| model | trained_epochs | dataset_count | sample_count | mean_clean_loss | mean_attack_loss_gain | mean_adv_loss |
| --- | --- | --- | --- | --- | --- | --- |
| loss3 | 1011 | 50 | 2500 | 4.76911e-07 | 2.73073e-06 | 3.20764e-06 |
| random_clean_y | 1100 | 50 | 2500 | 5.26291e-07 | 3.12722e-06 | 3.65351e-06 |
| loss2 | 1026 | 50 | 2500 | 7.15598e-07 | 4.06879e-06 | 4.78439e-06 |
| physics_loss | 1040 | 50 | 2500 | 7.90779e-07 | 4.14508e-06 | 4.93586e-06 |
| random_solver_y | 1100 | 50 | 2500 | 8.61425e-07 | 4.59724e-06 | 5.45866e-06 |
| loss1 | 1000 | 50 | 2500 | 9.7439e-07 | 4.85208e-06 | 5.82647e-06 |

## 25-Sample Metric Means

| model | mean_attack_gain | mean_jt_error | mean_sigma | mean_binary_first_order |
| --- | --- | --- | --- | --- |
| loss3 | 2.75651e-06 | 0.000105769 | 0.00207277 | 1.32177e-06 |
| random_clean_y | 2.94164e-06 | 9.85303e-05 | 0.00181992 | 1.25688e-06 |
| loss2 | 4.03097e-06 | 0.000116678 | 0.00172255 | 1.42332e-06 |
| physics_loss | 4.13983e-06 | 0.000132158 | 0.00186073 | 1.68308e-06 |
| random_solver_y | 4.54417e-06 | 0.000136492 | 0.00184049 | 1.73765e-06 |
| loss1 | 4.78855e-06 | 0.000139188 | 0.00179944 | 1.70468e-06 |

## 5-Sample Jacobian Probe Means

| model | mean_relative_l2 | mean_jt_error_l2_norm | mean_jt_error_l2_norm_sq | mean_j_error_l2_norm | mean_spectral_norm_top_sigma |
| --- | --- | --- | --- | --- | --- |
| random_clean_y | 0.0806844 | 0.000117679 | 1.75566e-08 | 4.29411e-05 | 0.00184299 |
| loss3 | 0.0785856 | 0.00012645 | 1.85496e-08 | 5.04957e-05 | 0.00210832 |
| loss2 | 0.0943854 | 0.000134626 | 2.24368e-08 | 5.52805e-05 | 0.00175943 |
| physics_loss | 0.0982516 | 0.000153299 | 2.75102e-08 | 6.23465e-05 | 0.00188965 |
| random_solver_y | 0.10239 | 0.00015803 | 2.93647e-08 | 6.02202e-05 | 0.00186718 |
| loss1 | 0.106734 | 0.000160822 | 3.01177e-08 | 6.57173e-05 | 0.00182829 |

## Main Output Files

- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_clean_by_dataset.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_clean_by_split.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_attack20_by_split.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_attack20_by_dataset_model.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_metric25_model_means.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_metric25_samples.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_metric25_correlations.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_jacobian5_model_means.csv`
- `analysis_outputs/darcy_six_model_random_source_report_20260613/artifact_manifest.json`

## Source Artifacts

- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_full50_eval_merged.csv`
- `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/summary_by_model_split.csv`
- `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/summary_by_dataset_model.csv`
- `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/model_metric_means.csv`
- `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/metrics_by_model_sample.csv`
- `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/correlations.csv`
- `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/summary_by_model.csv`
- `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc`
- `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/clean_eval_by_model_dataset.csv`
- `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples/summary_by_model_split.csv`
- `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples/summary_by_dataset_model.csv`
- `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples/metrics_by_model_sample.csv`
- `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples/correlations.csv`
- `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen/summary_by_model.csv`
