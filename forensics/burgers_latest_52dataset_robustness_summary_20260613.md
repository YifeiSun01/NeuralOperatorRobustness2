# Burgers Latest 52-Dataset Result Check, 2026-06-13

This file is a corrected provenance-aware summary. It does not merge incompatible
experiments into a fake single ranking.

## What Exists

- Same wide-parameter loss3-targeted clean generalization, 50 generalization
  datasets, 10,000 samples: baseline/loss1/loss2/loss3 plus random_clean_y and
  random_solver_y can be compared directly.
- Full 52-dataset P2Q2 attack exists for the old first_master 4-model run:
  baseline/loss1/loss2/loss3.
- Full 52-dataset P2Q2 attack exists for the new random-field 2-model run:
  random_clean_y/random_solver_y.
- A single finished 52-dataset, same-sample, same-attack, 5-model or 6-model
  robustness table was not found locally or on R2.

## Clean Generalization, Same 50 Wideparam Datasets

Lower is better.

| rank | model | RMSE mean | MSE mean | relative L2 mean |
|---:|---|---:|---:|---:|
| 1 | loss3 | 0.012050 | 0.0001978 | 0.023644 |
| 2 | random_solver_y | 0.019972 | 0.0005776 | 0.038941 |
| 3 | loss1 | 0.021027 | 0.0005822 | 0.041171 |
| 4 | loss2 | 0.022292 | 0.0006477 | 0.043542 |
| 5 | baseline | 0.029902 | 0.0012222 | 0.057737 |
| 6 | random_clean_y | 0.088542 | 0.0090267 | 0.168778 |

Conclusion: random_clean_y is not good on generalization. random_solver_y is
competitive with loss1/loss2, but loss3 is still the best clean-generalization
model on this wide-parameter set.

## Robustness Metric 1: P2Q2 Attack Increase

These are the latest 52-dataset attack summaries, but the old 4-model table and
the random-field 2-model table are different runs/data families. Do not use this
as a strict 6-model ranking.

Generalization split, 50 datasets / 10,000 samples:

| model | source | initial loss | final attack loss | attack increase | final diff RMS |
|---|---|---:|---:|---:|---:|
| random_solver_y | random_field_52_2models_20260613 | 0.000578 | 0.008338 | 0.007761 | 0.083409 |
| random_clean_y | random_field_52_2models_20260613 | 0.009027 | 0.043808 | 0.034781 | 0.204783 |
| loss2 | first_master_52_4models_20260608 | 0.025641 | 0.063833 | 0.038191 | 0.156609 |
| loss3 | first_master_52_4models_20260608 | 0.027885 | 0.069603 | 0.041718 | 0.150078 |
| baseline | first_master_52_4models_20260608 | 0.029338 | 0.072910 | 0.043572 | 0.202537 |
| loss1 | first_master_52_4models_20260608 | 0.038382 | 0.088790 | 0.050408 | 0.172880 |

The old first_master loss2-vs-loss3 ordering should not override the newer
wideparam dense-figure result. On the correct wideparam group00-04 visual subset,
loss3 beats loss2.

## Robustness Metric 2: Error-Jacobian Spectral Norm

Lower `J_error = J_model - J_solver` spectral norm is better. These are not all
the same sample manifest, so use them as separate evidence, not a single exact
ranking.

Old round03 final-extension SVD, 20 samples:

| model | all-split error spectral mean | generalization error spectral mean |
|---|---:|---:|
| baseline | 3.467930 | 5.774620 |
| loss1 | 1.509120 | 2.753180 |
| loss2 | 1.806150 | 3.322260 |
| loss3 | 1.673370 | 2.499600 |

Random-field SVD, 25 samples:

| model | all-split error spectral mean | generalization error spectral mean |
|---|---:|---:|
| random_clean_y | 3.828716 | 3.685240 |
| random_solver_y | 1.735561 | 2.009350 |

Conclusion: random_solver_y has much smaller error-Jacobian norm than
random_clean_y. Among old loss models, loss3 has the best generalization
error-Jacobian norm in the round03 SVD summary; loss1 has the best all-split mean.

## Robustness Metric 3: J_error Applied to Attack Delta

This was computed for the random-field full suite only.

| model | n | mean ||J_error delta||_2 | mean RMS | mean abs cosine with top error right singular vector |
|---|---:|---:|---:|---:|
| random_clean_y | 24 | 9.017980 | 0.281812 | 0.399197 |
| random_solver_y | 24 | 2.528930 | 0.079029 | 0.277103 |

Conclusion: random_solver_y is much better than random_clean_y by this local
linearized error-response metric.

## Corrected Bottom Line

- Best clean generalization on the comparable wideparam 50-dataset table: loss3.
- random_solver_y is strong and beats loss1/loss2 on clean RMSE in that table,
  but it does not beat loss3.
- random_clean_y is bad on clean generalization and should not be described as
  a best model.
- The currently available robustness files do not contain a strict same-run
  52-dataset 5-model/6-model comparison. Any single ranking across old
  first_master and new random-field attacks would be mixing experimental
  contexts.

## Source Files

- `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/clean_loss/per_dataset_clean_metrics.csv`
- `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608/summary_by_model_dataset.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/summary_by_model_dataset.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_jacobian_svd_summary.md`
- `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/jacobian_svd_summary.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/j_error_times_attack_delta.csv`
