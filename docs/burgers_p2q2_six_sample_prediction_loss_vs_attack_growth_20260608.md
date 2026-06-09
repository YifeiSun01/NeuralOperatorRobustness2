# Burgers P2Q2 Six-Sample Prediction Loss vs Attack Growth - 2026-06-08

## Scope

This note answers whether the clean prediction/generalization loss on the six display samples is directly associated with the loss after a fixed-budget zero-start attack.

This is a post-processing audit of existing attack curves; no new model or solver training/evaluation was run in this step.

## Source Evidence

Observed from:

- `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/sample_manifest.json`
- `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss1/attack_loss_curves.csv`
- `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss2/attack_loss_curves.csv`
- `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss3/attack_loss_curves.csv`

Derived tables written on 2026-06-08:

- `forensics/burgers_p2q2_six_sample_prediction_vs_attack_growth_20260608/six_sample_clean_attack_growth.csv`
- `forensics/burgers_p2q2_six_sample_prediction_vs_attack_growth_20260608/model_mean_clean_attack_growth.csv`
- `forensics/burgers_p2q2_six_sample_prediction_vs_attack_growth_20260608/clean_vs_attack_correlations.csv`
- `forensics/burgers_p2q2_six_sample_prediction_vs_attack_growth_20260608/metadata.json`

Important parsing detail: each attack directory writes the trained model as `target`; the audit relabels those rows by source directory into `loss1`, `loss2`, and `loss3`. The baseline rows are duplicated across the three source directories, so only the baseline copy from `loss1/attack_loss_curves.csv` is used.

Attack setting recorded in the source run: Burgers P2Q2 RMS-L2, epsilon RMS `0.12`, alpha RMS `0.012`, `100` steps. `clean_mse_step0` is the zero-attack prediction MSE. `attack_increase_mse` is `attack_mse_step100 - clean_mse_step0`.

## Six Samples

Observed from the sample manifest:

| sample | split | dataset_id | index |
|---:|---|---|---:|
| 1 | test | `test_original_gaussian_corr0p03` | 23 |
| 2 | generalization | `burgers_near_gaussian_corr0p1` | 17 |
| 3 | generalization | `burgers_target_gaussian_corr0p2` | 42 |
| 4 | generalization | `burgers_target_matern_corr0p6_nu3` | 73 |
| 5 | generalization | `burgers_mid_matern_corr1_nu4` | 29 |
| 6 | generalization | `burgers_far_sawtooth_add_scale0p3_shift0` | 11 |

## Mean Results Over Six Samples

Observed from the derived model-mean table:

| model | clean prediction MSE, step 0 | attacked MSE, step 100 | attack increase MSE |
|---|---:|---:|---:|
| baseline | 0.0002251479018 | 0.02650749679 | 0.02628234889 |
| loss1 | 0.000007584736987 | 0.006293901601 | 0.006286316864 |
| loss2 | 0.000008816111498 | 0.009268279517 | 0.009259463406 |
| loss3 | 0.00003560450356 | 0.002929622463 | 0.002894017959 |

Clean prediction ranking on these six samples:

`loss1 < loss2 < loss3 < baseline`.

Fixed-budget attacked-loss ranking on these six samples:

`loss3 < loss1 < loss2 < baseline`.

So on this six-sample display set, loss3 is not the best clean predictor, but it is the most robust after the fixed-budget P2Q2 attack.

## Correlation

Observed from `clean_vs_attack_correlations.csv`:

| scope | n | Pearson clean vs increase | Spearman clean vs increase | Pearson clean vs final attacked | Spearman clean vs final attacked |
|---|---:|---:|---:|---:|---:|
| baseline only | 6 | 0.484372 | 0.600000 | 0.496628 | 0.657143 |
| loss1 only | 6 | 0.188354 | 0.771429 | 0.190235 | 0.771429 |
| loss2 only | 6 | 0.832941 | 0.771429 | 0.833264 | 0.771429 |
| loss3 only | 6 | 0.101395 | 0.428571 | 0.117064 | 0.428571 |
| all 24 sample-model rows | 24 | 0.668182 | 0.638261 | 0.674491 | 0.641739 |
| four model means | 4 | 0.934085 | 0.400000 | 0.935351 | 0.400000 |

## Interpretation

Observed evidence:

- Within a fixed model, some samples with higher clean loss also tend to have higher attacked loss, but this is uneven. It is strong for loss2, weak for loss1 and loss3, and moderate for baseline.
- Across the model comparison that matters here, clean prediction loss does not preserve the robustness ranking. Loss1 has the lowest clean MSE, while loss3 has the lowest final attacked MSE and smallest attack increase.

Inference from the evidence:

- Clean prediction/generalization loss and zero-start fixed-budget robustness are related but not equivalent.
- Clean loss measures the error at the unperturbed input. The fixed-budget attack loss also depends on the local sensitivity, gradient geometry, and curvature of the model around that input.
- Therefore a model can have a higher clean prediction loss but lower attack growth if its input-output map is locally flatter or harder for the attack to exploit. The six-sample Burgers P2Q2 display set is an example of that: loss3 has worse clean MSE than loss1/loss2, but lower attacked MSE after the same budget.

Remaining caveat:

- These are the six display samples only. They should not be treated as a replacement for full-50 clean generalization or full-52-dataset attack evaluation.
