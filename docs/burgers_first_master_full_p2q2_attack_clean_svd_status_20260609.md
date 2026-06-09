# Burgers First Master Full P2Q2 Attack Clean/SVD Status 20260609

## Status

The first/master generalization-root full P2Q2 attack/tag run completed for all four models. The queued second `round03_selective` run was then stopped; its partial output should not be used as a formal result.

Observed first/master attack root:

- `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608`
- Source table: `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608/summary_by_model_dataset.csv`
- Derived analysis: `forensics/burgers_first_master_full_p2q2_clean_attack_correlation_20260609`

## Attack Setup

- Data root: `generalization_datasets/burgers`
- Datasets: `52` total = train first50, test, and 50 generalization datasets
- Samples: `10200` total
- Models: `baseline`, `loss1_epoch8000`, `loss2_epoch2000`, `loss3_epoch1500`
- Attack: P2Q2, `epsilon_rms=0.12`, `alpha_rms=0.012`, `20` steps, `batch_size=500`

## Weighted Attack Means

| split | model | initial loss | final loss | attack increase | final/initial |
|---|---|---:|---:|---:|---:|
| test | baseline | 9.1081e-05 | 0.00765143 | 0.00756035 | 84.0068 |
| test | loss1_epoch8000 | 7.77436e-07 | 0.00147068 | 0.0014699 | 1891.7 |
| test | loss2_epoch2000 | 2.49355e-06 | 0.00165785 | 0.00165536 | 664.856 |
| test | loss3_epoch1500 | 2.04998e-05 | 0.000897277 | 0.000876777 | 43.7701 |
| generalization | baseline | 0.0293379 | 0.0729095 | 0.0435716 | 2.48516 |
| generalization | loss1_epoch8000 | 0.0383823 | 0.0887901 | 0.0504078 | 2.31331 |
| generalization | loss2_epoch2000 | 0.0256413 | 0.0638325 | 0.0381912 | 2.48944 |
| generalization | loss3_epoch1500 | 0.027885 | 0.0696031 | 0.0417181 | 2.49608 |

## Clean Loss Versus Attack Damage Correlations

| scope | n | x | y | Pearson | Spearman |
|---|---:|---|---|---:|---:|
| all_rows_all_models | 208 | clean_initial_loss | attack_final_loss | 0.982369 | 0.820054 |
| all_rows_all_models | 208 | clean_initial_loss | attack_increase | 0.924343 | 0.813025 |
| generalization_all_models | 200 | clean_initial_loss | attack_final_loss | 0.982409 | 0.820932 |
| generalization_all_models | 200 | clean_initial_loss | attack_increase | 0.924292 | 0.814273 |

Per-model generalization correlations:

| scope | n | y | Pearson | Spearman |
|---|---:|---|---:|---:|
| generalization_baseline | 50 | attack_final_loss | 0.984139 | 0.897335 |
| generalization_baseline | 50 | attack_increase | 0.912718 | 0.891188 |
| generalization_loss1_epoch8000 | 50 | attack_final_loss | 0.987123 | 0.818391 |
| generalization_loss1_epoch8000 | 50 | attack_increase | 0.943372 | 0.809364 |
| generalization_loss2_epoch2000 | 50 | attack_final_loss | 0.978285 | 0.747995 |
| generalization_loss2_epoch2000 | 50 | attack_increase | 0.911828 | 0.740504 |
| generalization_loss3_epoch1500 | 50 | attack_final_loss | 0.982165 | 0.888307 |
| generalization_loss3_epoch1500 | 50 | attack_increase | 0.932833 | 0.87928 |

## Winner Counts On 50 Generalization Datasets

| metric | baseline | loss1 | loss2 | loss3 |
|---|---:|---:|---:|---:|
| clean_initial_loss_winner | 3 | 26 | 2 | 19 |
| attack_final_loss_winner | 5 | 0 | 7 | 38 |
| attack_increase_winner | 8 | 0 | 5 | 37 |

## SVD Availability

Observed locally for first/master SVD directory:

- Directory: `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608`
- Completed `jacobian_svd_summary.csv` visible: `False`
- Error spectral norm CSV visible: `False`

Conclusion: first/master attack-vs-SVD correlation is not currently computable from local evidence, because the first/master SVD artifact only shows preflight/manifest/partial sample files and no completed spectral-norm summary table.

## Interpretation

Observed evidence from the completed attack run supports dataset-level comparison of clean/generalization loss and fixed-budget attack damage on the true first/master root. It does not yet support a first/master SVD-vs-attack correlation claim.

Inference: if we need the actual first/master SVD correlation, the missing prerequisite is a completed first/master SVD summary over the same sample points. The existing `round01/aligned` SVD correlation result cannot be silently substituted for this master-root result.
