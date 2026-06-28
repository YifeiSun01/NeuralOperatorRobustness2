# R2 Burgers Three-Dataset Conclusion Audit - 2026-06-08

Status: inspected local records and the user-provided R2 bucket prefix for the Burgers three-generalization-dataset question. No new training or attack run was launched for this note.

## User-Facing Dataset Numbering

For this note, the user's "first / second / third" Burgers generalization datasets mean:

1. First: `generalization_datasets/burgers`
2. Second: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`
3. Third: `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`

Observed from R2 under `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`:

| dataset | R2 object count | R2 bytes | note |
|---|---:|---:|---|
| first root `generalization_datasets/burgers` | 50 | 82205042 | semantic/master Burgers root |
| second root `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` | 50 | 82206074 | semantic target-band/ns50 root |
| third root `generalization_datasets_burgers_loss3_selective_search/round_03/burgers` | 50 | 20744250 | loss3-selective attack-generated stress root |
| partial `generalization_datasets_burgers_loss3_aligned_search/round_02/burgers` | 10 | 4139730 | partial/unofficial; not one of the three complete roots |

Observed R2 artifact for the full fixed-budget attack/tag audit:

- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`
- Object count: 21
- Bytes: 160859222
- Source data in the local copy: `train=50`, `test=150`, `generalization=50*200=10000`, total `10200`.
- The `10000` generalization samples use the second root, `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`, with round03-trained checkpoints.

## First Dataset: `generalization_datasets/burgers`

Observed clean/generalization evidence from `docs/burgers_run1_run2_generalization_roots_audit_20260608.md` and `docs/burgers_master_semantic_final_models_20260608.md`:

| model | generalization RMSE | relative L2 | MAE |
|---|---:|---:|---:|
| baseline | 0.091671 | 0.248715 | 0.060993 |
| loss1_epoch8000 | 0.085163 | 0.212877 | 0.059952 |
| loss2_epoch2000 | 0.073342 | 0.188096 | 0.050274 |
| loss3_epoch1500 | 0.074097 | 0.176689 | 0.051457 |

Observed per-generalization winner counts:

- RMSE: baseline `3`, loss1 `26`, loss2 `2`, loss3 `19`.
- relative L2: baseline `3`, loss1 `26`, loss2 `2`, loss3 `19`.
- MAE: baseline `3`, loss1 `26`, loss2 `3`, loss3 `18`.

Inference:

- It is not accurate to say loss1/loss2 both simply dominate loss3 on the first root.
- By aggregate RMSE/MAE, loss2 is slightly better than loss3.
- By aggregate relative L2, loss3 is better than loss2 and loss1.
- By per-dataset winner counts, loss1 wins the most datasets, but loss3 also wins many.

Observed robustness evidence from R2/local records:

- I did not find a first-root full fixed-budget P2Q2 attack/tag table analogous to the second-root `10200`-sample artifact.
- Therefore, "first dataset robustness loss1/loss2 both better than loss3" is not currently evidenced by a direct first-root full attack table.

## Second Dataset: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`

Observed clean/generalization evidence from `docs/burgers_neutral_generalization_final_models_20260608.md`:

| model | generalization RMSE | relative L2 | MAE |
|---|---:|---:|---:|
| baseline | 0.0179323 | 0.0310194 | 0.00603362 |
| loss1_epoch8000 | 0.00176997 | 0.00306454 | 0.000909169 |
| loss2_epoch2000 | 0.00291557 | 0.00503321 | 0.00118401 |
| loss3_epoch1500 | 0.00672954 | 0.0116154 | 0.00187665 |

Observed per-generalization winner counts:

- RMSE: loss1 wins `50/50`.
- relative L2: loss1 wins `50/50`.
- MAE: loss1 wins `50/50`.

Observed robustness evidence from `docs/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608.md`:

| scope | clean wins loss1/loss2/loss3 | final attack wins loss1/loss2/loss3 | attack-increase wins loss1/loss2/loss3 |
|---|---|---|---|
| generalization 10000 samples | 9195 / 770 / 35 | 729 / 355 / 8916 | 683 / 325 / 8992 |

Observed generalization means from the same full-tag artifact:

| model | clean MSE mean | final attack MSE mean | attack increase mean |
|---|---:|---:|---:|
| loss1_epoch8000 | 0.000003424036669 | 0.003650282849 | 0.003646858812 |
| loss2_epoch2000 | 0.000008806193927 | 0.004692557371 | 0.004683751177 |
| loss3_epoch1500 | 0.00004564248681 | 0.001710434531 | 0.001664792044 |

Inference:

- The second dataset conclusion is strongly supported: clean/generalization favors loss1, then loss2, then loss3.
- Fixed-budget P2Q2 attack/tag robustness strongly favors loss3.
- This is the clearest "clean generalization and robustness ranking mismatch" result.

## Third Dataset: `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`

Observed provenance evidence from `docs/burgers_round03_attack_generated_generalization_validity_caveat_20260608.md`:

- The third root is loss3-selective and attack-generated from original train/test examples.
- It is a stress set, not an unbiased neutral generalization benchmark.

Observed clean/generated-generalization evidence from `docs/burgers_loss3_selective_round03_complete_report_20260605.md` and `docs/burgers_loss3_selective_round03_current_advantage_assessment_20260605.md`:

- Epoch-50 generated-generalization RMSE loss1/loss2/loss3 = `0.055974 / 0.056011 / 0.045295`.
- Round03 generated50 per-dataset audit reported loss3 winning `50/50` generated datasets at epoch10 and epoch50.
- Later long/final round03 summaries also show loss3 advantage on the selected generated root.

Observed robustness evidence:

- The available round03 final-model six-sample 100-step P2Q2 verification shows loss3 has the lowest attacked mean MSE: loss1/loss2/loss3 = `0.0062939017 / 0.0092682792 / 0.0029296223`.
- The available full `10200`-sample 20-step P2Q2 artifact shows loss3 robustness advantage, but that full artifact's generalization samples are from the second root, not the third attack-generated root.

Inference:

- The third dataset clearly supports loss3 on the selected generated/stress clean loss.
- Robustness evidence is consistent with loss3 being better, but the strongest full-sample attack/tag table currently found is on the second root, not a separate full third-root attack table.
- Because the third root is attack-generated and loss3-selective, it should be reported as a stress-set win, not broad neutral generalization proof.

## Corrected Answer To The User's Proposed Summary

The proposed three-line summary is not exactly right.

Observed/inferred correction:

1. First dataset: not a clean "loss1/loss2 both beat loss3" story. Loss2 beats loss3 on aggregate RMSE/MAE, loss3 beats loss2 on aggregate relative L2, and loss1 wins the most per-dataset counts. I did not find a direct first-root full attack/tag robustness table on R2/local records.
2. Second dataset: yes. Clean/generalization favors loss1/loss2, especially loss1, while fixed-budget P2Q2 robustness strongly favors loss3.
3. Third dataset: yes for the selected stress-set clean/generated loss, and consistent for robustness, but the claim must carry the caveat that the data root is attack-generated/loss3-selective.

