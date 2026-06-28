# Burgers Three Generalization Dataset Unified Status

Date: 2026-06-08

## Scope

This note records the current unified status for the user's three Burgers
generalization roots:

1. First/master semantic root: `generalization_datasets/burgers`
2. Second/target-band ns50 root:
   `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`
3. Third/round03 selective stress root:
   `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`

The question is whether clean/generalization and robustness conclusions have
been run and whether they agree across all three roots.

## Short Answer

No: it is not correct to say all three roots now have the same complete
clean-plus-robustness evidence.

The current status is:

- The second root has the strongest complete evidence: clean/generalization
  favors loss1/loss2, while fixed-budget P2Q2 robustness strongly favors loss3.
- The first root has clean/generalization evaluation, but no direct full
  first-root fixed-budget P2Q2 attack/tag table was found in local/R2 records.
- The third root supports loss3 on the selected stress-set clean/generated loss
  and has robustness evidence consistent with loss3, but it is attack-generated
  and loss3-selective. It should be reported as a stress-set result, not broad
  neutral generalization.

## First Root: `generalization_datasets/burgers`

Observed clean/generalization evidence:

- Source doc: `docs/burgers_master_semantic_final_models_20260608.md`
- Generalization dataset count: `50`
- This root has no attack metadata in the local provenance check.

Observed aggregate generalization metrics:

| model | RMSE | relative L2 | MAE |
|---|---:|---:|---:|
| baseline | `0.091671` | `0.248715` | `0.060993` |
| loss1_epoch8000 | `0.085163` | `0.212877` | `0.059952` |
| loss2_epoch2000 | `0.073342` | `0.188096` | `0.050274` |
| loss3_epoch1500 | `0.074097` | `0.176689` | `0.051457` |

Observed per-dataset winner counts:

- RMSE: baseline `3`, loss1 `26`, loss2 `2`, loss3 `19`
- relative L2: baseline `3`, loss1 `26`, loss2 `2`, loss3 `19`
- MAE: baseline `3`, loss1 `26`, loss2 `3`, loss3 `18`

Inference:

- This is not a simple "loss1/loss2 both beat loss3" root.
- Loss2 is slightly better than loss3 by aggregate RMSE and MAE.
- Loss3 is better than loss2 by aggregate relative L2.
- Loss1 wins the most individual datasets, but loss3 also wins many.

Robustness status:

- No direct first-root full fixed-budget P2Q2 attack/tag table analogous to the
  second-root `10200`-sample artifact was found in the inspected local/R2
  records.
- Therefore a first-root robustness ranking is not fully evidenced yet.

## Second Root: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`

Observed clean/generalization evidence:

- Source doc: `docs/burgers_neutral_generalization_final_models_20260608.md`
- Generalization dataset count: `50`
- This root has no attack metadata in the local provenance check.

Observed aggregate generalization metrics:

| model | RMSE | relative L2 | MAE |
|---|---:|---:|---:|
| baseline | `0.0179323` | `0.0310194` | `0.00603362` |
| loss1_epoch8000 | `0.00176997` | `0.00306454` | `0.000909169` |
| loss2_epoch2000 | `0.00291557` | `0.00503321` | `0.00118401` |
| loss3_epoch1500 | `0.00672954` | `0.0116154` | `0.00187665` |

Observed clean/generalization winner counts:

- loss1 wins `50/50` by RMSE, relative L2, and MAE.

Observed robustness evidence:

- Source docs:
  - `docs/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608.md`
  - `docs/burgers_round03_full52_clean_generalization_vs_attack_robustness_mismatch_20260608.md`
- Source artifact:
  `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`
- The source artifact has `10200` samples:
  `50` train, `150` test, and `10000` generalization samples.
- The `10000` generalization samples use this second root.

Observed generalization winner counts over `10000` samples:

| metric | loss1 | loss2 | loss3 |
|---|---:|---:|---:|
| clean MSE winner | `9195` | `770` | `35` |
| final attack MSE winner | `729` | `355` | `8916` |
| attack-increase winner | `683` | `325` | `8992` |

Observed sample means on the `10000` generalization samples:

| model | clean MSE mean | final attack MSE mean | attack increase mean |
|---|---:|---:|---:|
| loss1_epoch8000 | `0.000003424036669` | `0.003650282849` | `0.003646858812` |
| loss2_epoch2000 | `0.000008806193927` | `0.004692557371` | `0.004683751177` |
| loss3_epoch1500 | `0.00004564248681` | `0.001710434531` | `0.001664792044` |

Inference:

- This is the cleanest and strongest full-result root.
- Clean/generalization strongly favors loss1, then loss2, then loss3.
- Fixed-budget P2Q2 attack/tag robustness strongly favors loss3.
- This root gives the clearest evidence that clean generalization and
  fixed-budget adversarial robustness can rank the methods differently.

## Third Root: `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`

Observed provenance evidence:

- Source doc: `docs/burgers_round03_attack_generated_generalization_validity_caveat_20260608.md`
- This root is attack-generated and loss3-selective.
- It is a stress set, not a neutral/unbiased generalization benchmark.

Observed clean/generated-stress evidence:

- Source docs:
  - `docs/burgers_loss3_selective_round03_complete_report_20260605.md`
  - `docs/burgers_loss3_selective_round03_current_advantage_assessment_20260605.md`
- Epoch-50 generated/generalization RMSE:
  loss1/loss2/loss3 = `0.055974 / 0.056011 / 0.045295`
- Round03 generated50 per-dataset audit reported loss3 winning `50/50`
  generated datasets at epoch10 and epoch50.
- Later long/final round03 summaries also support a loss3 advantage on this
  selected generated/stress root.

Observed robustness evidence:

- The available six-sample 100-step P2Q2 verification shows attacked mean MSE
  loss1/loss2/loss3 = `0.0062939017 / 0.0092682792 / 0.0029296223`, favoring
  loss3.
- The strongest full `10200`-sample 20-step attack/tag table found so far uses
  the second root, not this third root.

Inference:

- This root supports loss3 as a stress-set winner.
- Robustness evidence is consistent with loss3 being better.
- But because the root is attack-generated/loss3-selective, it should not be
  used as broad neutral generalization evidence.

## Unified Conclusion

The unified conclusion is not "all three roots give the same complete story."

The correct unified status is:

1. First root:
   clean/generalization has been evaluated, but the result is mixed and no
   direct full first-root robustness/tag table is currently evidenced.
2. Second root:
   clean/generalization and robustness/tag have both been evaluated in the
   strongest way; clean favors loss1/loss2, especially loss1, while robustness
   strongly favors loss3.
3. Third root:
   loss3 wins on the selected stress-set clean/generated loss, and robustness
   evidence is consistent with loss3, but the root is selection-biased and
   attack-generated.

Paper-safe wording:

> Across the currently audited Burgers roots, the strongest neutral/semantic
> full-tag evidence comes from the second target-band ns50 root. There, loss3 is
> generally worse by clean/generalization loss but much better by fixed-budget
> P2Q2 robustness. The first master semantic root has clean/generalization
> evidence but lacks a matched full robustness/tag table. The third round03 root
> supports loss3 on a selected stress set, but because it is attack-generated and
> loss3-selective it should be reported as stress-set evidence rather than broad
> neutral generalization.

## Remaining Work

- Run or locate a direct full fixed-budget P2Q2 attack/tag table for the first
  root if a first-root robustness conclusion is needed.
- Run or locate a separate full third-root attack/tag table if the third-root
  robustness claim needs the same strength as the second-root `10200`-sample
  artifact.
