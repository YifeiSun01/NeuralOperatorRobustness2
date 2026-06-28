# Burgers Robustness Numeric Snapshot, 2026-06-14

Generated from completed ranked metric CSVs. No training, attack, Jacobian, or SVD computation was rerun.

Important protocol note: the latest direct robustness evidence for `loss3 e1000` is the 25-sample robustness/SVD table and the dense visual trace set. The recovered full-52 attack appendix is mixed historical evidence; see `docs/burgers_attack52_protocol_recheck_20260614.md`.

Comparability note: the 54 robustness metrics are not all six-model-common
quality metrics. The integrity audit classifies them as 28 six-model-common
metrics, 14 old-four-only metrics, and 12 random-only metrics. Use
`outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/integrity_audit_20260614/robustness_metric_comparability_audit.csv`
for the exact per-metric class.

## Files

- `../outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/robustness_numeric_snapshot_20260614/robustness_all_25sample_54metric_model_means.csv`
- `../outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/robustness_numeric_snapshot_20260614/robustness_core_numeric_table.csv`
- `../outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/robustness_numeric_snapshot_20260614/robustness_all_25sample_non_loss3_best_rows.csv`
- `../outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/robustness_numeric_snapshot_20260614/svd_error_spectrum_topk_model_means.csv`
- `../outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/robustness_numeric_snapshot_20260614/svd_error_spectrum_rank1_to_rank20_model_means.csv`
- `../outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/robustness_numeric_snapshot_20260614/loss3_vs_random_solver_selected_tests.csv`

## Non-loss3 Best Rows

The core table below is intentionally a selected quality/mechanism table, so it
mostly shows rows where `loss3` is best. The full 54-metric table has 20 rows
where the best model is not `loss3`; those rows are isolated in
`robustness_all_25sample_non_loss3_best_rows.csv`. Most of them are old4-only,
random-only, direction-cosine, or perturbation/budget/process fields, not clean
six-model quality metrics.

## Core 25-Sample Robustness Numbers

| metric | direction | baseline | loss1 | loss2 | loss3 | random_clean_y | random_solver_y | best_model | runner_up_model | best_vs_runner_t_q_one_sided_better_bh_fdr | best_vs_runner_significant_q05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attack_initial_mse | lower | 0.001095 | 8.231e-04 | 8.558e-04 | 2.439e-04 | 0.009137 | 7.423e-04 | loss3 | random_solver_y | 0.003843 | 1 |
| attack_final_mse | lower | 0.010259 | 0.006661 | 0.006479 | 0.003307 | 0.051981 | 0.008745 | loss3 | loss2 | 1.220e-06 | 1 |
| attack_loss_increase | lower | 0.009164 | 0.005838 | 0.005623 | 0.003063 | 0.042844 | 0.008003 | loss3 | loss2 | 4.570e-06 | 1 |
| clean_residual_mse_recomputed | lower | 0.001095 | 8.231e-04 | 8.558e-04 | 2.439e-04 | 0.008772 | 7.127e-04 | loss3 | random_solver_y | 0.004034 | 1 |
| clean_residual_norm_l2 | lower | 0.914193 | 0.714167 | 0.742491 | 0.411851 | 2.75033 | 0.637226 | loss3 | random_solver_y | 0.001818 | 1 |
| bias_gradient_norm | lower | 0.462951 | 0.315854 | 0.343340 | 0.111950 | 4.78477 | 0.371206 | loss3 | loss1 | 1.417e-04 | 1 |
| bias_gradient_rms | lower | 0.014467 | 0.009870 | 0.010729 | 0.003498 | 0.149524 | 0.011600 | loss3 | loss1 | 1.417e-04 | 1 |
| j_error_transpose_error_l2 | lower | 0.462951 | 0.315854 | 0.343340 | 0.111950 | 4.78477 | 0.371206 | loss3 | loss1 | 1.417e-04 | 1 |
| j_error_transpose_error_rms | lower | 0.014467 | 0.009870 | 0.010729 | 0.003498 | 0.149524 | 0.011600 | loss3 | loss1 | 1.417e-04 | 1 |
| error_spectral_norm | lower | 2.36703 | 1.70247 | 1.85511 | 1.27153 | 4.13091 | 1.55252 | loss3 | random_solver_y | 0.057041 | 0.000e+00 |
| error_fro_norm_comparable | lower | 3.57462 | 2.78114 | 2.88781 | 1.89853 | 7.22482 | 2.51221 | loss3 | random_solver_y | 0.004541 | 1 |
| error_effective_rank | lower | 10.2846 | 18.1362 | 11.9306 | 6.37332 | 7.0791 | 7.97519 | loss3 | random_clean_y | 0.117104 | 0.000e+00 |
| model_solver_top1_right_abs_cos | higher | 0.853861 | 0.937746 | 0.940075 | 0.960089 | 0.229040 | 0.920981 | loss3 | loss2 | 0.303225 | 0.000e+00 |
| model_solver_top1_left_abs_cos | higher | 0.692770 | 0.841553 | 0.833315 | 0.913083 | 0.104539 | 0.837769 | loss3 | loss1 | 0.033148 | 1 |
| model_solver_top5_right_subspace_mean_cos | higher | 0.902259 | 0.924231 | 0.924449 | 0.936441 | 0.533103 | 0.921183 | loss3 | loss2 | 0.238099 | 0.000e+00 |
| model_solver_top5_left_subspace_mean_cos | higher | 0.773360 | 0.818937 | 0.821315 | 0.897940 | 0.377899 | 0.834929 | loss3 | random_solver_y | 0.007219 | 1 |
| model_solver_top10_right_subspace_mean_cos | higher | 0.886490 | 0.911366 | 0.904067 | 0.957945 | 0.618067 | 0.903616 | loss3 | loss1 | 9.192e-05 | 1 |
| model_solver_top10_left_subspace_mean_cos | higher | 0.799476 | 0.845780 | 0.838075 | 0.933519 | 0.539678 | 0.854451 | loss3 | random_solver_y | 1.020e-04 | 1 |
| model_solver_top20_right_subspace_mean_cos | higher | 0.731655 | 0.764206 | 0.760543 | 0.886612 | 0.646238 | 0.831291 | loss3 | random_solver_y | 2.921e-07 | 1 |
| model_solver_top20_left_subspace_mean_cos | higher | 0.800560 | 0.859059 | 0.868047 | 0.943287 | 0.687855 | 0.914269 | loss3 | random_solver_y | 2.552e-04 | 1 |
| j_error_delta_l2 | lower |  |  |  |  | 9.70486 | 2.09097 | random_solver_y | random_clean_y | 5.551e-10 | 1 |
| j_error_delta_rms | lower |  |  |  |  | 0.303277 | 0.065343 | random_solver_y | random_clean_y | 5.551e-10 | 1 |
| attack_delta_svd_abs_cos | higher | 0.304880 | 0.209058 | 0.196515 | 0.091452 | 0.393474 | 0.142775 | random_clean_y | baseline | 0.047047 | 1 |
| attack_delta_outward_abs_cos | higher | 0.793450 | 0.803056 | 0.798714 | 0.726458 |  |  | loss1 | loss2 | 0.401928 | 0.000e+00 |
| attack_final_delta_rms | lower | 0.120000 | 0.120000 | 0.120000 | 0.120000 | 0.119844 | 0.120000 | random_clean_y | loss1 | 0.179617 | 0.000e+00 |
| delta_l2 | lower |  |  |  |  | 3.83501 | 3.84 | random_clean_y | random_solver_y | 0.179617 | 0.000e+00 |

## SVD Error Spectrum Top-k Numbers

| metric | direction | baseline | loss1 | loss2 | loss3 | random_clean_y | random_solver_y | best_model | runner_up_model | best_vs_runner_t_q_one_sided_better_bh_fdr | best_vs_runner_significant_q05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| error_singular_value_top20_all | lower | 0.520427 | 0.430666 | 0.445088 | 0.277697 | 1.13533 | 0.395152 | loss3 | random_solver_y | 3.464e-21 | 1 |
| error_singular_values_top01_l2 | lower | 2.36703 | 1.70247 | 1.85511 | 1.27153 | 4.13091 | 1.55252 | loss3 | random_solver_y | 0.057041 | 0.000e+00 |
| error_singular_values_top01_mean | lower | 2.36703 | 1.70247 | 1.85511 | 1.27153 | 4.13091 | 1.55252 | loss3 | random_solver_y | 0.057041 | 0.000e+00 |
| error_singular_values_top05_l2 | lower | 3.32192 | 2.53924 | 2.65192 | 1.77905 | 6.65476 | 2.31171 | loss3 | random_solver_y | 0.010109 | 1 |
| error_singular_values_top05_mean | lower | 1.32598 | 1.05167 | 1.0887 | 0.707426 | 2.84056 | 0.957895 | loss3 | random_solver_y | 0.002626 | 1 |
| error_singular_values_top10_l2 | lower | 3.48199 | 2.69227 | 2.81309 | 1.86313 | 7.1186 | 2.45136 | loss3 | random_solver_y | 0.005909 | 1 |
| error_singular_values_top10_mean | lower | 0.880253 | 0.716717 | 0.743533 | 0.465913 | 1.94307 | 0.650993 | loss3 | random_solver_y | 5.175e-04 | 1 |
| error_singular_values_top20_l2 | lower | 3.53332 | 2.74131 | 2.86194 | 1.89052 | 7.21329 | 2.50246 | loss3 | random_solver_y | 0.004650 | 1 |
| error_singular_values_top20_mean | lower | 0.520427 | 0.430666 | 0.445088 | 0.277697 | 1.13533 | 0.395152 | loss3 | random_solver_y | 2.172e-04 | 1 |

## SVD Error Spectrum Rank 1-20 Numbers

| metric | direction | baseline | loss1 | loss2 | loss3 | random_clean_y | random_solver_y | best_model | runner_up_model | best_vs_runner_t_q_one_sided_better_bh_fdr | best_vs_runner_significant_q05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| error_singular_value_rank01 | lower | 2.36703 | 1.70247 | 1.85511 | 1.27153 | 4.13091 | 1.55252 | loss3 | random_solver_y | 0.057041 | 0.000e+00 |
| error_singular_value_rank02 | lower | 1.61305 | 1.24478 | 1.23818 | 0.839499 | 3.2816 | 1.11577 | loss3 | random_solver_y | 0.013717 | 1 |
| error_singular_value_rank03 | lower | 1.13749 | 0.929710 | 0.964504 | 0.578942 | 2.76421 | 0.859274 | loss3 | random_solver_y | 3.418e-04 | 1 |
| error_singular_value_rank04 | lower | 0.825305 | 0.756065 | 0.771293 | 0.457247 | 2.23526 | 0.698096 | loss3 | random_solver_y | 3.123e-04 | 1 |
| error_singular_value_rank05 | lower | 0.687057 | 0.625320 | 0.614398 | 0.389914 | 1.79083 | 0.563819 | loss3 | random_solver_y | 8.529e-04 | 1 |
| error_singular_value_rank06 | lower | 0.584930 | 0.507367 | 0.548112 | 0.316174 | 1.48175 | 0.455935 | loss3 | random_solver_y | 1.299e-04 | 1 |
| error_singular_value_rank07 | lower | 0.489710 | 0.417590 | 0.440846 | 0.258355 | 1.23095 | 0.380549 | loss3 | random_solver_y | 5.468e-05 | 1 |
| error_singular_value_rank08 | lower | 0.419429 | 0.364953 | 0.380239 | 0.212435 | 0.984078 | 0.331423 | loss3 | random_solver_y | 1.536e-05 | 1 |
| error_singular_value_rank09 | lower | 0.371347 | 0.329047 | 0.327883 | 0.179941 | 0.828505 | 0.290979 | loss3 | random_solver_y | 3.110e-06 | 1 |
| error_singular_value_rank10 | lower | 0.307188 | 0.289863 | 0.294757 | 0.155090 | 0.702568 | 0.261567 | loss3 | random_solver_y | 5.456e-06 | 1 |
| error_singular_value_rank11 | lower | 0.269788 | 0.241704 | 0.252186 | 0.135438 | 0.578811 | 0.229998 | loss3 | random_solver_y | 1.044e-05 | 1 |
| error_singular_value_rank12 | lower | 0.233060 | 0.213431 | 0.213200 | 0.119922 | 0.488713 | 0.197453 | loss3 | random_solver_y | 5.641e-06 | 1 |
| error_singular_value_rank13 | lower | 0.201961 | 0.182263 | 0.187556 | 0.108750 | 0.414709 | 0.171415 | loss3 | random_solver_y | 2.684e-05 | 1 |
| error_singular_value_rank14 | lower | 0.178941 | 0.163542 | 0.163551 | 0.099241 | 0.363832 | 0.149029 | loss3 | random_solver_y | 1.299e-04 | 1 |
| error_singular_value_rank15 | lower | 0.157492 | 0.137306 | 0.138716 | 0.090994 | 0.321304 | 0.136691 | loss3 | random_solver_y | 2.353e-04 | 1 |
| error_singular_value_rank16 | lower | 0.137090 | 0.121821 | 0.125094 | 0.081708 | 0.275723 | 0.123915 | loss3 | loss1 | 1.823e-09 | 1 |
| error_singular_value_rank17 | lower | 0.124573 | 0.112176 | 0.112077 | 0.073341 | 0.243725 | 0.111905 | loss3 | random_solver_y | 9.220e-05 | 1 |
| error_singular_value_rank18 | lower | 0.111987 | 0.099559 | 0.101693 | 0.067257 | 0.215538 | 0.099678 | loss3 | loss1 | 5.672e-12 | 1 |
| error_singular_value_rank19 | lower | 0.101091 | 0.091753 | 0.091113 | 0.061714 | 0.195024 | 0.090742 | loss3 | random_solver_y | 6.545e-05 | 1 |
| error_singular_value_rank20 | lower | 0.090018 | 0.082587 | 0.081245 | 0.056447 | 0.178619 | 0.082290 | loss3 | loss2 | 1.977e-07 | 1 |

## Strongest Correlations With Attack Loss Increase

| scope | metric | target | n | pearson | spearman | pearson_q_bh_fdr | spearman_q_bh_fdr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_two_50 | solver_subspace_mismatch_top20_left | attack_loss_increase | 50 | 0.904947 | 0.914622 | 6.292e-19 | 3.498e-20 |
| random_two_50 | model_solver_top20_left_subspace_mean_cos | attack_loss_increase | 50 | -0.904947 | -0.914622 | 6.292e-19 | 3.498e-20 |
| random_two_50 | bias_gradient_rms | attack_loss_increase | 50 | 0.884352 | 0.911357 | 4.440e-17 | 7.701e-20 |
| random_two_50 | bias_gradient_norm | attack_loss_increase | 50 | 0.884352 | 0.911357 | 4.440e-17 | 7.701e-20 |
| all_six_150 | j_error_delta_l2 | attack_loss_increase | 48 | 0.882571 | 0.900456 | 2.647e-16 | 5.355e-18 |
| random_two_50 | j_error_delta_l2 | attack_loss_increase | 48 | 0.882571 | 0.900456 | 2.647e-16 | 5.355e-18 |
| all_six_150 | bias_gradient_rms | attack_loss_increase | 150.000 | 0.899726 | 0.897556 | 1.027e-53 | 4.613e-53 |
| all_six_150 | bias_gradient_norm | attack_loss_increase | 150.000 | 0.899726 | 0.897556 | 1.027e-53 | 4.613e-53 |
| random_two_50 | error_fro_norm_comparable | attack_loss_increase | 50 | 0.859156 | 0.896279 | 3.146e-15 | 2.710e-18 |
| generalization_all_six_126 | j_error_delta_l2 | attack_loss_increase | 42 | 0.862165 | 0.879588 | 4.073e-13 | 2.420e-14 |
| generalization_all_six_126 | bias_gradient_norm | attack_loss_increase | 126.000 | 0.925377 | 0.875762 | 6.079e-53 | 5.410e-40 |
| generalization_all_six_126 | bias_gradient_rms | attack_loss_increase | 126.000 | 0.925377 | 0.875762 | 6.079e-53 | 5.410e-40 |
| all_six_150 | error_fro_norm_comparable | attack_loss_increase | 150.000 | 0.803927 | 0.872713 | 2.217e-34 | 1.134e-46 |
| random_two_50 | clean_residual_norm_l2 | attack_loss_increase | 50 | 0.858377 | 0.865546 | 3.177e-15 | 7.519e-16 |
| random_two_50 | clean_residual_mse_recomputed | attack_loss_increase | 50 | 0.788598 | 0.865546 | 1.732e-11 | 7.519e-16 |
