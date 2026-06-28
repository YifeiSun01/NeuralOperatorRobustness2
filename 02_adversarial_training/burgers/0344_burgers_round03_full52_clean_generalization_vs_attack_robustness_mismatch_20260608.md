# Burgers Round03 Full-52 Clean Generalization vs Attack Robustness Mismatch - 2026-06-08

## Scope

This note records data support for the claim that clean prediction/generalization loss and fixed-budget P2Q2 attack robustness do not necessarily have the same ranking.

This is a post-processing audit of existing full-dataset evaluation artifacts. No new model training or solver evaluation was run in this step.

## Source Evidence

Observed from:

- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/summary.json`
- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/summary_by_model_dataset.csv`
- `forensics/burgers_round03_full_p2q2_clean_loss_per_dataset_20260608/clean_and_attack_loss_ratios_by_dataset.csv`
- `forensics/burgers_round03_full_p2q2_clean_loss_per_dataset_20260608/sample_weighted_before_after_solver_gap_summary.csv`

Derived tables written on 2026-06-08:

- `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/model_sample_weighted_means.csv`
- `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/winner_by_dataset.csv`
- `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/winner_and_sample_weighted_summary.csv`
- `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/clean_attack_correlation_summary.csv`
- `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/loss3_pairwise_ratios_vs_loss12.csv`
- `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/trained_models_clean_attack_long_by_dataset.csv`
- `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/metadata.json`

Attack setting observed from the source summary: Burgers P2Q2 RMS-L2, epsilon RMS `0.12`, alpha RMS `0.012`, `20` steps, `52` datasets, `10200` samples, V100 GPU path recorded in the source config.

Dataset scope: `train_original_gaussian_corr0p03_first50`, `test_original_gaussian_corr0p03`, and `50` generalization datasets from the current round03 full-P2Q2 evaluation suite.

## Sample-Weighted Model Means

Observed from `model_sample_weighted_means.csv`:

| scope | model | clean MSE | final attack MSE | attack increase MSE |
|---|---|---:|---:|---:|
| all 52 | baseline | 0.0003210206038 | 0.02971614201 | 0.02939512141 |
| all 52 | loss1_epoch8000 | 0.00000337123809 | 0.003606513503 | 0.003603142265 |
| all 52 | loss2_epoch2000 | 0.000008680059203 | 0.004632377368 | 0.004623697309 |
| all 52 | loss3_epoch1500 | 0.00004514041727 | 0.001694242130 | 0.001649101713 |
| generalization 50 | baseline | 0.0003256726507 | 0.03016204335 | 0.02983637070 |
| generalization 50 | loss1_epoch8000 | 0.000003424036656 | 0.003650282826 | 0.003646858790 |
| generalization 50 | loss2_epoch2000 | 0.000008806193828 | 0.004692557324 | 0.004683751130 |
| generalization 50 | loss3_epoch1500 | 0.00004564248688 | 0.001710434542 | 0.001664792055 |

Observed ranking among loss1/loss2/loss3:

- Clean MSE: `loss1 < loss2 < loss3`.
- Final attack MSE: `loss3 < loss1 < loss2`.
- Attack increase MSE: `loss3 < loss1 < loss2`.

## Winner Counts

Observed from `winner_and_sample_weighted_summary.csv`:

| scope | datasets | clean winner counts among trained models | final attack winner counts among trained models | attack increase winner counts among trained models |
|---|---:|---|---|---|
| all 52 | 52 | loss1: 52, loss2: 0, loss3: 0 | loss1: 0, loss2: 0, loss3: 52 | loss1: 0, loss2: 0, loss3: 52 |
| generalization 50 | 50 | loss1: 50, loss2: 0, loss3: 0 | loss1: 0, loss2: 0, loss3: 50 | loss1: 0, loss2: 0, loss3: 50 |
| train/test 2 | 2 | loss1: 2, loss2: 0, loss3: 0 | loss1: 0, loss2: 0, loss3: 2 | loss1: 0, loss2: 0, loss3: 2 |

This is direct dataset-level support for a mismatch: the clean-loss winner and the attack-robustness winner are different on every dataset in this evaluation suite.

## Pairwise Loss3 Ratios

Observed from `loss3_pairwise_ratios_vs_loss12.csv`:

| scope | comparison | clean MSE ratio | final attack MSE ratio | attack increase MSE ratio |
|---|---|---:|---:|---:|
| all 52 | loss3 / loss1 | 13.38986333 | 0.4697728509 | 0.4576843187 |
| all 52 | loss3 / loss2 | 5.200473431 | 0.3657392296 | 0.3566629912 |
| generalization 50 | loss3 / loss1 | 13.33002285 | 0.4685758949 | 0.4565002791 |
| generalization 50 | loss3 / loss2 | 5.182998213 | 0.3644994454 | 0.3554399047 |

Observed implication: on the 50 generalization datasets, loss3 has about `13.33x` higher clean MSE than loss1, but about `0.469x` the final attacked MSE and about `0.457x` the attack increase. In other words, loss3 is much worse by clean MSE and much better by fixed-budget attack robustness on the same suite.

## Correlation

Observed from `clean_attack_correlation_summary.csv`:

| scope | n | method | clean vs final attack | clean vs attack increase |
|---|---:|---|---:|---:|
| trained all-52 dataset rows | 156 | Pearson | -0.504984 | -0.513058 |
| trained all-52 dataset rows | 156 | Spearman | -0.332841 | -0.346611 |
| trained generalization-50 dataset rows | 150 | Pearson | -0.549044 | -0.556730 |
| trained generalization-50 dataset rows | 150 | Spearman | -0.400500 | -0.409894 |
| all-52 trained-model means | 3 | Pearson | -0.892404 | -0.895446 |
| all-52 trained-model means | 3 | Spearman | -0.500000 | -0.500000 |
| generalization-50 trained-model means | 3 | Pearson | -0.892056 | -0.895095 |
| generalization-50 trained-model means | 3 | Spearman | -0.500000 | -0.500000 |

The all-four-model mean correlation including baseline is positive because baseline is both bad clean and bad attacked. That aggregate is not the relevant comparison for loss1/loss2/loss3 method selection. Among the trained methods, the observed correlations are negative in this suite.

## Interpretation

Observed evidence:

- For all 50 generalization datasets in this suite, loss1 is the clean-MSE winner among trained models.
- For all 50 generalization datasets in this suite, loss3 is the fixed-budget final-attack-MSE winner and attack-increase winner among trained models.
- The same complete mismatch also holds for the 52-dataset suite including train/test.

Inference from the evidence:

- On this existing Burgers round03 full-P2Q2 evaluation suite, clean prediction/generalization loss and fixed-budget P2Q2 attack robustness do not match as model-selection criteria.
- This does not prove the statement for every possible dataset distribution or every attack budget. It is direct evidence for this specific suite, budget, models, and attack implementation.

Caveat:

- This supports the clean-vs-robustness mismatch claim on the current full-P2Q2 suite. It does not by itself solve the separate validity concern about whether this generalization suite is the ideal external generalization benchmark.
