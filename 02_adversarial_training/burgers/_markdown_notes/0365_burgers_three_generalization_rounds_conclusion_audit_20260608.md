# Burgers three generalization/self-training conclusion audit - 2026-06-08

Status: conclusion alignment from existing local records. No new training or attack run was launched for this note.

## Question

The user asked whether the three Burgers generalization/self-training attempts should be summarized as:

- first run: loss3 is bad on both clean/generalization loss and robustness;
- second run: loss3 has better robustness but worse clean/generalization loss;
- third run: loss3 is good on both clean/generalization loss and robustness.

## Observed Evidence

Observed from `docs/burgers_generalization_round00_round01_direction_analysis_20260605.md`:

- `round00` is a hand-crafted aggressive OOD set. It is hard, but not loss3-selective.
- Round00 10-step generalization-gradient cosine means are loss1/loss2/loss3 = `-0.0155 / 0.0374 / 0.0342`.
- Inference from that record: round00 did not provide evidence that loss3 was the useful direction.

Observed from `forensics/burgers_loss3_aligned_round01_report_20260604.md` and `docs/burgers_generalization_round00_round01_direction_analysis_20260605.md`:

- `round01` is not a neutral random generalization set. It is a loss3-aligned adversarial/generated set built from raw loss3 adversarial inputs with `epsilon_fraction=0.06`, `5` attack steps, and train sources.
- Round01 50-step generalization-gradient cosine means are loss1/loss2/loss3 = `0.235689 / 0.166701 / 0.735448`.
- Round01 best observed generated-generalization RMSE was loss3 at epoch `463`, RMSE `0.007302`.
- At final checkpoints, generated-generalization RMSE was loss1/loss2/loss3 = `0.009127 / 0.012053 / 0.009939`, so loss1 final was still better than loss3 final.
- Inference from that record: round01 showed loss3 mechanism/gradient alignment and a best-epoch signal, but it did not establish a clean final-checkpoint or equal-wall-clock loss3 win.

Observed from `docs/burgers_round03_attack_generated_generalization_validity_caveat_20260608.md` and `docs/burgers_loss3_selective_round03_complete_report_20260605.md`:

- `round03` is a loss3-selective attack-generated stress set, not a neutral 50-distribution generalization benchmark.
- The selected round03 data come from attacking original train/test examples and selecting candidates by loss3-favorable geometry.
- Round03 final epoch-50 generated-generalization RMSE was loss1/loss2/loss3 = `0.055974 / 0.056011 / 0.045295`.
- Round03 generated50 per-dataset audit showed loss3 winning `50/50` generated datasets at epoch10 and epoch50, while losing on the original train/test rows.
- Inference from that record: round03 supports a targeted stress-set/generalization-to-loss3-selected-data claim, but has low persuasive power as a neutral OOD generalization claim.

Observed from `docs/burgers_neutral_generalization_final_models_20260608.md`:

- The neutral/non-attack semantic target-band root is `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- On that neutral semantic root, final-model clean/generalization RMSE means were baseline/loss1/loss2/loss3 = `0.0179323 / 0.00176997 / 0.00291557 / 0.00672954`.
- Loss1 was best on all `50/50` neutral semantic generalization datasets by RMSE, relative L2, and MAE.
- Inference from that record: the neutral semantic clean/generalization evaluation does not support loss3 being best.

Observed from `docs/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608.md`:

- The full per-sample tagged artifact used the same neutral/non-attack semantic target-band root, not the problematic round03 attack-generated root.
- On `10000` neutral semantic generalization samples, clean winners were loss1/loss2/loss3 = `9195 / 770 / 35`.
- On the same samples after fixed-budget P2Q2 attack/tag, final-loss winners were loss1/loss2/loss3 = `729 / 355 / 8916`.
- Attack-increase winners were loss1/loss2/loss3 = `683 / 325 / 8992`.
- Inference from that record: on this neutral semantic root, loss3 is usually worse for clean/generalization prediction loss but usually better for fixed-budget adversarial robustness.

Observed from `docs/burgers_round03_characteristics_vs_round01_round02_train_test_20260605.md`:

- Local `round02` is partial and unofficial: only `10` `.pt` files, no manifest/summary, and no formal evaluation record in that note.
- Inference: round02 should not be treated as one of the completed three formal conclusions unless a separate full evaluation artifact is located.

## Corrected Summary

The user's high-level memory is directionally right that the conclusions changed across datasets, but the exact mapping needs correction:

1. The early hard/generalization attempt (`round00`) did not show loss3 as useful. Existing records support "loss3 not good / not selected by the target direction" for that dataset. I do not see a completed full robustness/tag table for round00 in the inspected records, so "robustness also bad" should be stated only if a separate round00 attack artifact is located.
2. The neutral semantic target-band evaluation supports the strong mismatch conclusion: loss3 is usually worse on clean/generalization loss but usually better under fixed-budget attack/tag. This is the clearest evidence for "generalization loss bad, robustness good."
3. The round03 loss3-selective stress set supports "loss3 clean/generated loss good on that stress set," and it is aligned with the intended loss3 robustness/stress mechanism. But because round03 is attack-generated and selection-biased, it should not be described as proof that loss3 is broadly good on neutral generalization.


## User Three-Run Terminology Mapping

Observed on 2026-06-08 after rechecking local roots:

- If the user’s "first/second/third generalization datasets" means the three practical Burgers generalization dataset attempts where the first two are normal semantic roots and the third is the problematic attack-generated root, then:
- first dataset: the broad semantic master-style root `generalization_datasets/burgers`;
- second dataset: the target-band/ns50 semantic root `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`;
- third dataset: the loss3-selective attack-generated stress root `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`.

Under that user-facing numbering, the `10200`-sample full-tag/per-sample result is the **second** dataset, not the first and not the third. It used `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` for the `10000` generalization samples, with round03-trained checkpoints.

## Practical Interpretation

For reporting, separate the axes:

- Neutral clean generalization: use `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`; current final-model result favors loss1, then loss2, then loss3.
- Neutral fixed-budget robustness/tag: use the full52 per-sample P2Q2 artifact; current result strongly favors loss3 after attack/tag despite worse clean loss.
- Loss3-selected stress generalization: use `generalization_datasets_burgers_loss3_selective_search/round_03`; current result favors loss3, but the claim must be labeled as stress-set/selective, not neutral OOD generalization.

