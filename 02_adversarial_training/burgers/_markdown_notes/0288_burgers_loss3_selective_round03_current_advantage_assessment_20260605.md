# Burgers Round03 Current Loss3 Advantage Assessment

Date: 2026-06-05 UTC.

## Scope

This is a current interpretation of the completed long-training prediction and gradient-alignment outputs for the round03 Burgers generalization dataset. The final Jacobian/SVD posthoc stage is still running, so this note does not claim a finished spectral/Jacobian conclusion.

## Source Evidence

Observed from:

- `docs/burgers_loss3_selective_round03_long_training_report_20260605.md`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/epoch_aligned_split_metrics.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_pairwise_ratios.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/per_dataset_generated_advantage_summary.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/best_epoch_by_split.csv`
- `forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_50step_long_pipeline_20260605/gradient_alignment_mean_by_variant.csv`
- `adversarial_training_runs/burgers_loss3_selective_round03_full_pipeline_20260605_logs/svd_final_rep20_top100.log`

## Prediction Evidence

Observed final-run generated50 generalization metrics:

| loss | epoch | wall h | gen RMSE | gen rel L2 |
| --- | ---: | ---: | ---: | ---: |
| loss1 | 1000 | 1.39523 | 0.0365568 | 0.0655582 |
| loss2 | 500 | 3.02754 | 0.0396661 | 0.0711470 |
| loss3 | 500 | 6.63540 | 0.0236606 | 0.0424415 |

Observed inference from final-run generated50 metrics: loss3 is about `35.3%` lower RMSE than loss1 final and about `40.4%` lower RMSE than loss2 final on round03 generalization.

Observed best generated50 generalization metrics over each run:

| loss | best epoch | best wall h | best gen RMSE | best gen rel L2 |
| --- | ---: | ---: | ---: | ---: |
| loss1 | 965 | 1.34075 | 0.0351621 | 0.0630595 |
| loss2 | 486 | 2.94171 | 0.0383002 | 0.0686989 |
| loss3 | 492 | 6.51218 | 0.0230768 | 0.0414002 |

Observed inference from best-epoch metrics: even comparing each loss at its own best observed generalization epoch, loss3 is about `34.4%` lower RMSE than loss1 and about `39.7%` lower RMSE than loss2.

Observed generated-dataset win counts:

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

Observed inference from per-dataset win counts: on round03 generated50, loss3 advantage is not coming from a few lucky generated datasets. At final comparison it wins all 50 generated datasets against both baselines, with the weakest final generated-dataset RMSE advantage still above `27%` versus loss1 and above `31%` versus loss2.

## Wall-Clock Evidence

Observed wall-clock-aligned generated50 metrics:

| comparison | baseline | baseline epoch | loss3 epoch | baseline gen RMSE | loss3 gen RMSE | loss3/base RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| same_wall_as_loss1_final | loss1 | 1000 | 104 | 0.0365568 | 0.0462736 | 1.2658 |
| same_wall_as_loss1_final | loss2 | 230 | 104 | 0.0441223 | 0.0462736 | 1.0488 |
| same_wall_as_loss2_final | loss1 | 1000 | 229 | 0.0365568 | 0.0340254 | 0.9308 |
| same_wall_as_loss2_final | loss2 | 500 | 229 | 0.0396661 | 0.0340254 | 0.8578 |
| same_wall_as_loss3_final | loss1 | 1000 | 500 | 0.0365568 | 0.0236606 | 0.6472 |
| same_wall_as_loss3_final | loss2 | 500 | 500 | 0.0396661 | 0.0236606 | 0.5965 |

Observed inference from wall-clock metrics: at the loss1-final wall-clock budget, loss3 has not yet caught up on round03 generalization. At the loss2-final wall-clock budget, loss3 already beats both observed baselines on generated50 generalization. At loss3 final, it is clearly better, but that uses a longer observed wall time than the final loss1/loss2 runs.

## Train/Test Caveat

Observed final train/test metrics:

| loss | split | RMSE | rel L2 |
| --- | --- | ---: | ---: |
| loss1 | train | 0.0025765 | 0.0048460 |
| loss1 | test | 0.0026812 | 0.0049882 |
| loss2 | train | 0.0019206 | 0.0036124 |
| loss2 | test | 0.0021087 | 0.0039230 |
| loss3 | train | 0.0058810 | 0.0110613 |
| loss3 | test | 0.0061688 | 0.0114765 |

Observed inference from train/test metrics: loss3 is not a uniform train/test winner. It is substantially worse on the original train/test distribution, while being much better on the targeted round03 generated generalization distribution.

## Gradient Alignment Evidence

Observed mean gradient-alignment cosine:

| variant | clean train | clean test | generalization mixed4 |
| --- | ---: | ---: | ---: |
| loss1_raw | 0.900233 | 0.906246 | 0.0338632 |
| loss2_raw | 0.899806 | 0.854951 | 0.119059 |
| loss3_raw | 0.935371 | 0.849992 | 0.276633 |

Observed inference from gradient alignment: loss3's adversarial update gradient is much more aligned with the round03 generalization-loss gradient than loss1/loss2. The generalization cosine is about `2.3x` loss2 and about `8.2x` loss1, which supports the intended mechanism rather than just a loss-scale artifact.

## Jacobian/SVD Status

Observed from `svd_final_rep20_top100.log`: the posthoc final Jacobian/SVD job is active and currently processing `sample_000`; the output directory currently has `config.json`, `round03_long_final_sample_manifest.csv`, and `sample_000/` only. Therefore the current evidence does not yet include the final spectral norm, singular-vector, or subspace-alignment conclusion.

## Current Conclusion

Inference from the completed prediction and gradient outputs: yes, the round03 dataset currently shows a clear loss3 advantage for the intended generated50 generalization target. The evidence is strong because loss3 wins all 50 generated datasets at final comparison, has a roughly 35-41% final generalization RMSE advantage, and has substantially better gradient alignment with the round03 generalization objective.

However, the conclusion should be stated narrowly: this is a round03 out-of-distribution generalization advantage, not a train/test advantage. For wall-clock fairness, loss3 is not already better at the loss1-final time budget, but it is better by the loss2-final time budget and at its own final checkpoint. The final Jacobian/SVD result is still pending and should be used before making the stronger spectral/subspace claim.
