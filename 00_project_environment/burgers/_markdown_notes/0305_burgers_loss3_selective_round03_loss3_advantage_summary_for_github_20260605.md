# Burgers Round03 Loss3 Advantage Summary

Date: 2026-06-05 UTC.

## Bottom Line

The user's phrase "LlaMA 3" in the discussion is interpreted here as `loss3`.

Yes: on the round03 Burgers generated-generalization dataset, the `loss3` advantage is more visible than on the two earlier generalization datasets. The strongest evidence is not just one final RMSE number. It is the combination of:

- prediction RMSE/relative-L2 advantage on generated50;
- loss3 winning all 50 generated datasets against both loss1 and loss2 at the checked epochs/final checkpoints;
- stronger gradient alignment between the loss3 adversarial-training update and the round03 generalization objective;
- final Jacobian/SVD evidence showing better generated-generalization `J_model - J_solver` spectral norm and stronger solver-like singular subspace alignment.

This conclusion should stay narrow: round03 shows a targeted generated-OOD generalization advantage for loss3. It does not show loss3 is uniformly better on the original train/test distribution, and strict wall-clock comparison against the loss1-final time budget is still not favorable to loss3.

## Dataset Comparison

| dataset | status | main design | baseline difficulty | loss3-selective signal | conclusion |
| --- | --- | --- | ---: | --- | --- |
| round00 aggressive | official old dataset | hand-crafted large range/spectrum/shape shifts | generated RMSE `0.3743`, rel L2 `0.7405` | weak: 10-step gradient cosine loss1/loss2/loss3 `-0.0155 / 0.0374 / 0.0342` | hard, but not loss3-selective |
| round01 loss3-aligned | official old dataset | train-source loss3 adversarial inputs, epsilon fraction `0.06`, 5 steps | generated RMSE `0.04254`, rel L2 `0.07883` | strong gradient signal: 50-step cosine loss1/loss2/loss3 `0.2357 / 0.1667 / 0.7354` | mechanism signal exists, but final prediction/SVD were mixed |
| round03 loss3-selective | current official dataset | mixed train/test sources, epsilon fractions `0.08/0.10/0.12`, 8 or 10 attack steps, selected by loss3 cosine margin | generated RMSE `0.088998`, rel L2 `0.159756` | selection cosine means loss1/loss2/loss3 `0.364237 / 0.152738 / 0.740184`, margin `0.375216` | stronger than round01 while preserving loss3-selective geometry |

Observed round03-vs-round01 input/attack differences:

- round03 baseline generated RMSE is about `2.09x` round01.
- round03 mean out-of-bound amount `x_oob_mean` is `0.019239`, about `3.20x` round01's `0.006013`.
- round03 delta RMS and delta L_inf are about `1.90x` and `1.70x` round01.
- round03 is not the round00 style of arbitrary large shift; it was filtered by loss3-selective first-order geometry.

## Prediction Evidence

Final generated50 generalization metrics:

| loss | epoch | wall h | gen RMSE | gen rel L2 |
| --- | ---: | ---: | ---: | ---: |
| loss1 | 1000 | 1.39523 | 0.0365568 | 0.0655582 |
| loss2 | 500 | 3.02754 | 0.0396661 | 0.0711470 |
| loss3 | 500 | 6.63540 | 0.0236606 | 0.0424415 |

Observed from these final metrics:

- loss3 is about `35.3%` lower RMSE than loss1 on round03 generated50.
- loss3 is about `40.4%` lower RMSE than loss2 on round03 generated50.

Best generated50 generalization metrics reached during each run:

| loss | best epoch | best wall h | best gen RMSE | best gen rel L2 |
| --- | ---: | ---: | ---: | ---: |
| loss1 | 965 | 1.34075 | 0.0351621 | 0.0630595 |
| loss2 | 486 | 2.94171 | 0.0383002 | 0.0686989 |
| loss3 | 492 | 6.51218 | 0.0230768 | 0.0414002 |

Observed from best-epoch metrics:

- loss3 is about `34.4%` lower RMSE than loss1's own best generated checkpoint.
- loss3 is about `39.7%` lower RMSE than loss2's own best generated checkpoint.

Per-generated-dataset win counts:

| comparison | baseline | datasets | loss3 wins | win fraction | mean advantage pct | min advantage pct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| same_epoch_10 | loss1 | 50 | 50 | 1.00 | 17.05 | 9.92 |
| same_epoch_10 | loss2 | 50 | 50 | 1.00 | 15.90 | 8.66 |
| same_epoch_300 | loss1 | 50 | 50 | 1.00 | 30.30 | 22.77 |
| same_epoch_300 | loss2 | 50 | 50 | 1.00 | 32.71 | 24.39 |
| same_epoch_500 | loss1 | 50 | 50 | 1.00 | 40.98 | 30.96 |
| same_epoch_500 | loss2 | 50 | 50 | 1.00 | 41.32 | 31.41 |
| run_final | loss1 | 50 | 50 | 1.00 | 35.93 | 27.31 |
| run_final | loss2 | 50 | 50 | 1.00 | 41.32 | 31.41 |

This is the cleanest reason round03 is more convincing than round01: the loss3 prediction advantage is broad across all 50 generated datasets, not driven by a few easy wins.

## Wall-Clock Caveat

Wall-clock-aligned generated50 metrics:

| comparison | baseline | baseline epoch | loss3 epoch | baseline gen RMSE | loss3 gen RMSE | loss3/base RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| same_wall_as_loss1_final | loss1 | 1000 | 104 | 0.0365568 | 0.0462736 | 1.2658 |
| same_wall_as_loss1_final | loss2 | 230 | 104 | 0.0441223 | 0.0462736 | 1.0488 |
| same_wall_as_loss2_final | loss1 | 1000 | 229 | 0.0365568 | 0.0340254 | 0.9308 |
| same_wall_as_loss2_final | loss2 | 500 | 229 | 0.0396661 | 0.0340254 | 0.8578 |
| same_wall_as_loss3_final | loss1 | 1000 | 500 | 0.0365568 | 0.0236606 | 0.6472 |
| same_wall_as_loss3_final | loss2 | 500 | 500 | 0.0396661 | 0.0236606 | 0.5965 |

Observed interpretation:

- At the loss1-final wall-clock budget, loss3 has not caught loss1.
- At the loss2-final wall-clock budget, loss3 already beats both observed loss1/loss2 generated50 metrics.
- At loss3's own final checkpoint, loss3 is clearly best on generated50, but it used longer wall time.

## Train/Test Caveat

Final original train/test metrics:

| loss | split | RMSE | rel L2 |
| --- | --- | ---: | ---: |
| loss1 | train | 0.0025765 | 0.0048460 |
| loss1 | test | 0.0026812 | 0.0049882 |
| loss2 | train | 0.0019206 | 0.0036124 |
| loss2 | test | 0.0021087 | 0.0039230 |
| loss3 | train | 0.0058810 | 0.0110613 |
| loss3 | test | 0.0061688 | 0.0114765 |

Observed interpretation: loss3 is worse on the original train/test distribution. The round03 claim is therefore not "loss3 is globally better"; it is "loss3 is better on this targeted generated generalization distribution and its local Jacobian geometry."

## Gradient Alignment Evidence

Mean gradient-alignment cosine:

| variant | clean train | clean test | generalization mixed4 |
| --- | ---: | ---: | ---: |
| loss1_raw | 0.900233 | 0.906246 | 0.0338632 |
| loss2_raw | 0.899806 | 0.854951 | 0.119059 |
| loss3_raw | 0.935371 | 0.849992 | 0.276633 |

Observed interpretation: on round03 generalization, loss3's adversarial update gradient is about `2.3x` loss2 and about `8.2x` loss1 in mean cosine. This supports the mechanism that round03 was selected to expose, rather than only making the loss numerically large.

## Final Jacobian/SVD Evidence

Final/basic SVD scope:

- loss1 epoch1000
- loss2 epoch500
- loss3 epoch500
- train/test/generalization samples
- rep20/top100
- same-wall/time-matched SVD was cancelled by user request and is not included here.

Mean spectral norm of `J_model - J_solver`:

| split | baseline | loss1 | loss2 | loss3 | best trained loss |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 1.08462 | 0.41549 | 0.28852 | 0.88538 | loss2 |
| test | 1.23788 | 0.50024 | 0.36496 | 0.68253 | loss2 |
| generalization | 5.77462 | 3.39560 | 3.93652 | 2.65625 | loss3 |

Baseline-relative drop in `J_model - J_solver` spectral norm:

| split | loss1 drop | loss2 drop | loss3 drop | largest drop |
| --- | ---: | ---: | ---: | --- |
| train | 61.69% | 73.40% | 18.37% | loss2 |
| test | 59.59% | 70.52% | 44.86% | loss2 |
| generalization | 41.20% | 31.83% | 54.00% | loss3 |

Observed generalization per-sample SVD wins:

- loss3 has lower error spectral norm than loss1 on 8/10 generalization samples.
- loss3 has lower error spectral norm than loss2 on 8/10 generalization samples.
- loss3 is best among loss1/loss2/loss3 on 7/10 generalization samples.

Generalization model-vs-solver top-k singular subspace similarity, mean principal cosine:

| top-k | metric | baseline | loss1 | loss2 | loss3 | best trained loss |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 5 | right/input | 0.914152 | 0.967460 | 0.949149 | 0.975547 | loss3 |
| 10 | right/input | 0.876361 | 0.925909 | 0.929847 | 0.950825 | loss3 |
| 20 | right/input | 0.732393 | 0.777392 | 0.774694 | 0.843830 | loss3 |
| 5 | left/output | 0.714969 | 0.904195 | 0.869640 | 0.907541 | loss3 |
| 10 | left/output | 0.770438 | 0.897435 | 0.889833 | 0.920445 | loss3 |
| 20 | left/output | 0.767185 | 0.850798 | 0.849171 | 0.875296 | loss3 |

Generalization top-k improvement over baseline:

| top-k | metric | loss1 delta | loss2 delta | loss3 delta | largest delta |
| ---: | --- | ---: | ---: | ---: | --- |
| 5 | right/input | +0.053308 | +0.034997 | +0.061395 | loss3 |
| 10 | right/input | +0.049549 | +0.053487 | +0.074465 | loss3 |
| 20 | right/input | +0.044998 | +0.042301 | +0.111437 | loss3 |
| 5 | left/output | +0.189226 | +0.154670 | +0.192572 | loss3 |
| 10 | left/output | +0.126997 | +0.119395 | +0.150007 | loss3 |
| 20 | left/output | +0.083613 | +0.081986 | +0.108110 | loss3 |

Observed interpretation: final SVD strengthens the round03 story. On the generated-generalization set, loss3 has the best trained-model error operator norm against the solver and the strongest solver-like singular subspace alignment. On train/test, loss2 remains better in Jacobian error magnitude.

## Why Round03 Is More Convincing Than Round00/Round01

Round00 proved that one can make a very hard OOD dataset, but its gradient direction was not loss3-selective. Round01 proved that loss3 geometry could be selected, but the final checkpoint comparison remained mixed: loss3 had good top-10/top-20 right singular subspace behavior, yet final prediction and `J_model - J_solver` spectral norm did not clearly beat loss1.

Round03 is better aligned with the user's goal because three kinds of evidence now point in the same generated-generalization direction:

- prediction: final generated50 RMSE loss3 `0.0236606` vs loss1 `0.0365568` and loss2 `0.0396661`;
- first-order training direction: generalization gradient cosine loss3 `0.276633` vs loss1 `0.0338632` and loss2 `0.119059`;
- local operator geometry: generalization `J_model - J_solver` mean loss3 `2.65625` vs loss1 `3.39560` and loss2 `3.93652`, plus loss3-best top5/top10/top20 subspaces.

That is why the round03 dataset makes loss3 look more clearly advantageous than the previous two generalization datasets, while still requiring honest caveats about wall-clock and train/test degradation.

## Source Paths

Observed evidence was taken from:

- `docs/burgers_generalization_round00_round01_direction_analysis_20260605.md`
- `docs/burgers_round03_characteristics_vs_round01_round02_train_test_20260605.md`
- `docs/burgers_loss3_selective_round03_current_advantage_assessment_20260605.md`
- `docs/burgers_loss3_selective_round03_final_svd_interpretation_20260605.md`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/epoch_aligned_split_metrics.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_pairwise_ratios.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/per_dataset_generated_advantage_summary.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/best_epoch_by_split.csv`
- `forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_50step_long_pipeline_20260605/gradient_alignment_mean_by_variant.csv`
- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_error_spectral_norm_aggregate.csv`
- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_solver_similarity_subspaces.csv`
- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_solver_similarity_rankwise.csv`

No new training or SVD job was started for this Markdown summary.
