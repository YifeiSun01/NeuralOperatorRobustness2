# Burgers Round03 Long Training Report

Date: 2026-06-05 UTC.

Observed from local run directories:
- `loss1`: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605` final epoch 1000 wall 1.39523 h
- `loss2`: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605` final epoch 500 wall 3.02754 h
- `loss3`: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605` final epoch 500 wall 6.6354 h

Round03 generalization root: `generalization_datasets_burgers_loss3_selective_search/round_03`.
The long runs use full train/test/generated50 evaluation every epoch and checkpoint every 100 epochs plus configured wall-clock checkpoints.

## Same-Epoch Metrics

| epoch | loss | split | wall h | RMSE | rel L2 |
| --- | --- | --- | --- | --- | --- |
| 10 | loss1 | test | 0.0139408 | 0.0054194 | 0.0100823 |
| 10 | loss1 | generalization | 0.0139408 | 0.0683111 | 0.122591 |
| 10 | loss2 | test | 0.0612345 | 0.0051944 | 0.00966367 |
| 10 | loss2 | generalization | 0.0612345 | 0.0673916 | 0.120941 |
| 10 | loss3 | test | 0.128456 | 0.0181212 | 0.0337127 |
| 10 | loss3 | generalization | 0.128456 | 0.0565652 | 0.101501 |
| 50 | loss1 | test | 0.0697523 | 0.00405189 | 0.00753814 |
| 50 | loss1 | generalization | 0.0697523 | 0.055702 | 0.099942 |
| 50 | loss2 | test | 0.301789 | 0.00424173 | 0.00789132 |
| 50 | loss2 | generalization | 0.301789 | 0.056329 | 0.101069 |
| 50 | loss3 | test | 0.663747 | 0.01465 | 0.0272549 |
| 50 | loss3 | generalization | 0.663747 | 0.0491649 | 0.0882075 |
| 100 | loss1 | test | 0.143621 | 0.00348712 | 0.00648746 |
| 100 | loss1 | generalization | 0.143621 | 0.0519137 | 0.0931389 |
| 100 | loss2 | test | 0.604144 | 0.00375729 | 0.00699007 |
| 100 | loss2 | generalization | 0.604144 | 0.0514299 | 0.0922715 |
| 100 | loss3 | test | 1.33454 | 0.0132686 | 0.024685 |
| 100 | loss3 | generalization | 1.33454 | 0.0451856 | 0.0810725 |
| 200 | loss1 | test | 0.28197 | 0.00289399 | 0.00538398 |
| 200 | loss1 | generalization | 0.28197 | 0.0453257 | 0.081313 |
| 200 | loss2 | test | 1.22121 | 0.00360497 | 0.00670669 |
| 200 | loss2 | generalization | 1.22121 | 0.0459183 | 0.0823749 |
| 200 | loss3 | test | 2.63325 | 0.0122325 | 0.0227573 |
| 200 | loss3 | generalization | 2.63325 | 0.0383609 | 0.0688192 |
| 300 | loss1 | test | 0.420783 | 0.00282532 | 0.00525624 |
| 300 | loss1 | generalization | 0.420783 | 0.0401741 | 0.0720649 |
| 300 | loss2 | test | 1.83013 | 0.00233949 | 0.00435239 |
| 300 | loss2 | generalization | 1.83013 | 0.0414719 | 0.0743903 |
| 300 | loss3 | test | 3.97228 | 0.00744155 | 0.0138443 |
| 300 | loss3 | generalization | 3.97228 | 0.028279 | 0.0507264 |
| 400 | loss1 | test | 0.559675 | 0.002828 | 0.00526122 |
| 400 | loss1 | generalization | 0.559675 | 0.0412766 | 0.0740467 |
| 400 | loss2 | test | 2.43036 | 0.0025613 | 0.00476506 |
| 400 | loss2 | generalization | 2.43036 | 0.0410153 | 0.073572 |
| 400 | loss3 | test | 5.29331 | 0.00687267 | 0.0127859 |
| 400 | loss3 | generalization | 5.29331 | 0.0258739 | 0.0464296 |
| 500 | loss1 | test | 0.699672 | 0.00235694 | 0.00438486 |
| 500 | loss1 | generalization | 0.699672 | 0.0394174 | 0.0707024 |
| 500 | loss2 | test | 3.02754 | 0.00210867 | 0.00392297 |
| 500 | loss2 | generalization | 3.02754 | 0.0396661 | 0.071147 |
| 500 | loss3 | test | 6.6354 | 0.00616881 | 0.0114765 |
| 500 | loss3 | generalization | 6.6354 | 0.0236606 | 0.0424415 |

## Final Metrics

| loss | epoch | split | wall h | RMSE | rel L2 |
| --- | --- | --- | --- | --- | --- |
| loss1 | 1000 | train | 1.39523 | 0.00257647 | 0.00484599 |
| loss1 | 1000 | test | 1.39523 | 0.00268123 | 0.00498817 |
| loss1 | 1000 | generalization | 1.39523 | 0.0365568 | 0.0655582 |
| loss2 | 500 | train | 3.02754 | 0.0019206 | 0.00361238 |
| loss2 | 500 | test | 3.02754 | 0.00210867 | 0.00392297 |
| loss2 | 500 | generalization | 3.02754 | 0.0396661 | 0.071147 |
| loss3 | 500 | train | 6.6354 | 0.00588098 | 0.0110613 |
| loss3 | 500 | test | 6.6354 | 0.00616881 | 0.0114765 |
| loss3 | 500 | generalization | 6.6354 | 0.0236606 | 0.0424415 |

## Wall-Clock Pair Ratios

| comparison | split | base | base ep | loss3 ep | loss3/base RMSE | loss3/base relL2 |
| --- | --- | --- | --- | --- | --- | --- |
| same_wall_as_loss1_final | test | loss1 | 1000 | 104 | 4.93287 | 4.93287 |
| same_wall_as_loss1_final | test | loss2 | 230 | 104 | 4.09618 | 4.09618 |
| same_wall_as_loss1_final | generalization | loss1 | 1000 | 104 | 1.2658 | 1.26633 |
| same_wall_as_loss1_final | generalization | loss2 | 230 | 104 | 1.04876 | 1.04888 |
| same_wall_as_loss1_final | ALL | loss1 | 1000 | 104 | 1.27655 | 1.27754 |
| same_wall_as_loss1_final | ALL | loss2 | 230 | 104 | 1.05774 | 1.05825 |
| same_wall_as_loss2_final | test | loss1 | 1000 | 229 | 3.95469 | 3.95469 |
| same_wall_as_loss2_final | test | loss2 | 500 | 229 | 5.02851 | 5.02851 |
| same_wall_as_loss2_final | generalization | loss1 | 1000 | 229 | 0.930755 | 0.931134 |
| same_wall_as_loss2_final | generalization | loss2 | 500 | 229 | 0.857795 | 0.85799 |
| same_wall_as_loss2_final | ALL | loss1 | 1000 | 229 | 0.939583 | 0.94034 |
| same_wall_as_loss2_final | ALL | loss2 | 500 | 229 | 0.866661 | 0.867236 |

## Generated Dataset Win Counts

| comparison | base | n | loss3 wins | win frac | adv pct mean | adv pct min |
| --- | --- | --- | --- | --- | --- | --- |
| same_epoch_10 | loss1 | 50 | 50 | 1 | 17.0476 | 9.92413 |
| same_epoch_10 | loss2 | 50 | 50 | 1 | 15.901 | 8.65523 |
| same_epoch_50 | loss1 | 50 | 49 | 0.98 | 11.1253 | -4.75502 |
| same_epoch_50 | loss2 | 50 | 49 | 0.98 | 12.2527 | -1.16843 |
| same_epoch_100 | loss1 | 50 | 49 | 0.98 | 12.2451 | -3.88428 |
| same_epoch_100 | loss2 | 50 | 49 | 0.98 | 11.4684 | -4.23891 |
| same_epoch_200 | loss1 | 50 | 49 | 0.98 | 14.7935 | -8.15211 |
| same_epoch_200 | loss2 | 50 | 49 | 0.98 | 16.0504 | -3.9042 |
| same_epoch_300 | loss1 | 50 | 50 | 1 | 30.3005 | 22.7718 |
| same_epoch_300 | loss2 | 50 | 50 | 1 | 32.7087 | 24.3888 |
| same_epoch_400 | loss1 | 50 | 50 | 1 | 37.8357 | 29.939 |
| same_epoch_400 | loss2 | 50 | 50 | 1 | 37.3462 | 28.5767 |
| same_epoch_500 | loss1 | 50 | 50 | 1 | 40.9795 | 30.9557 |
| same_epoch_500 | loss2 | 50 | 50 | 1 | 41.3158 | 31.4137 |
| run_final | loss1 | 50 | 50 | 1 | 35.9292 | 27.3128 |
| run_final | loss2 | 50 | 50 | 1 | 41.3158 | 31.4137 |

## Attack Delta Summary

| loss | epochs | mean l2 rms | mean linf | final epoch | final l2 rms | final linf |
| --- | --- | --- | --- | --- | --- | --- |
| loss1 | 1000 | 0.0599928 | 0.106296 | 1000 | 0.0600432 | 0.106548 |
| loss2 | 500 | 0.059987 | 0.106571 | 500 | 0.0596108 | 0.105766 |
| loss3 | 500 | 0.059987 | 0.270536 | 500 | 0.0596108 | 0.291309 |

## Output CSVs

- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/epoch_aligned_split_metrics.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_checkpoint_selection.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_aligned_split_metrics.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_pairwise_ratios.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/per_dataset_generated_advantage.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/per_dataset_generated_advantage_summary.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/best_epoch_by_split.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/attack_delta_selected_epochs.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/attack_delta_summary_by_loss.csv`

Inference from these tables should be made after the Jacobian/SVD diagnostics are added, because prediction and local linearization can diverge.
