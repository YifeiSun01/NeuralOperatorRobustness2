# Darcy CFlow Attack50 Per-Dataset T-Tests - 2026-06-15

Status: complete.

Observed source:
- Attack sample table:
  `outputs/darcy_cflow_final_robustness_20260615/data/robustness_attack_52datasets_samples.csv`
- Generated t-test table:
  `outputs/darcy_cflow_timematched_organized_release_20260614/data/per_dataset_ttests_20260615/attack50_per_dataset_first_vs_second_ttests.csv`
- Generated summary:
  `outputs/darcy_cflow_timematched_organized_release_20260614/data/per_dataset_ttests_20260615/attack50_per_dataset_ttest_summary.csv`
- Report:
  `outputs/darcy_cflow_timematched_organized_release_20260614/reports/per_dataset_attack50_ttests_20260615.md`

Provenance checks:
- Final attack table uses 7 models and 52 datasets with 50 matched samples per dataset.
- `attack_steps = 50`.
- The final robustness provenance records `old_lossdrop50_token_found = False`.
- A text search over the new per-dataset t-test outputs found no `lossdrop50_selected_20260607`, no smoke checkpoint token, and no `attack_steps=1` token.

Test definition:
- For each dataset and each metric, models were ranked by the 50-sample mean.
- A paired t-test was run on matched `source_sample_index` pairs for first-ranked model vs second-ranked model.
- Alternative hypothesis: `second - best > 0`.
- Small p-value means the first-ranked model is significantly lower than the second-ranked model inside that dataset.
- Benjamini-Hochberg correction was also computed per metric over all 52 dataset tests.

All-52 results:

| Metric | First-model counts | Loss3 first | Raw p<0.05 | BH p<0.05 | Conclusion |
|---|---:|---:|---:|---:|---|
| clean MSE loss | loss3:44; loss2:4; random solver:2; Physics Loss:2 | 44/52 | 50/52 | 50/52 | Not all 52 significant |
| clean RMSE from per-sample MSE | loss3:44; loss2:4; random solver:2; Physics Loss:2 | 44/52 | 48/52 | 48/52 | Not all 52 significant |
| adversarial MSE loss | loss3:50; random clean:2 | 50/52 | 52/52 | 52/52 | All 52 significant |
| absolute loss increase | loss3:50; random clean:2 | 50/52 | 52/52 | 52/52 | All 52 significant |
| relative loss increase | random solver:21; loss3:16; random clean:8; loss1:6; loss2:1 | 16/52 | 29/52 | 28/52 | Not all 52 significant |
| delta L2 RMS | loss3:21; random clean:13; loss2:10; loss1:4; random solver:3; Physics Loss:1 | 21/52 | 28/52 | 24/52 | Not all 52 significant |
| delta Linf | baseline:52 | 0/52 | 0/52 | 0/52 | No meaningful significance; Linf is tied/saturated |

Generalization-50 results:

| Metric | First-model counts | Loss3 first | Raw p<0.05 | BH p<0.05 | Conclusion |
|---|---:|---:|---:|---:|---|
| clean MSE loss | loss3:44; loss2:4; Physics Loss:2 | 44/50 | 49/50 | 49/50 | Not all 50 significant |
| clean RMSE from per-sample MSE | loss3:44; loss2:4; Physics Loss:2 | 44/50 | 47/50 | 47/50 | Not all 50 significant |
| adversarial MSE loss | loss3:50 | 50/50 | 50/50 | 50/50 | All 50 significant |
| absolute loss increase | loss3:50 | 50/50 | 50/50 | 50/50 | All 50 significant |
| relative loss increase | random solver:21; loss3:16; random clean:6; loss1:6; loss2:1 | 16/50 | 29/50 | 28/50 | Not all 50 significant |
| delta L2 RMS | loss3:21; random clean:11; loss2:10; loss1:4; random solver:3; Physics Loss:1 | 21/50 | 26/50 | 23/50 | Not all 50 significant |
| delta Linf | baseline:50 | 0/50 | 0/50 | 0/50 | No meaningful significance; Linf is tied/saturated |

Important limitation:
- The full clean evaluation table in the organized release stores aggregate RMSE/Relative L2 values, not per-sample clean errors.
- Therefore, a true per-dataset t-test for full clean RMSE/Relative L2 requires rerunning or reconstructing clean evaluation with per-sample errors saved.
- The completed 52-per-dataset tests above are valid for the final attack50 sample table, including `clean_loss` and `sqrt(clean_loss)` on the 50 attack-manifest samples.
