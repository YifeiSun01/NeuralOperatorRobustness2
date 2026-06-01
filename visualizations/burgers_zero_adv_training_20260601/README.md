# Burgers zero adversarial-training visualization summary

Generated from:

```text
adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601
```

## Final split metrics at epoch 1000

| task | phase | epoch | global_step | progress_fraction | split | dataset_count | total_samples_evaluated | eval_wall_sec | rmse_dataset_mean | rmse_dataset_min | rmse_dataset_max | mae_dataset_mean | mae_dataset_min | mae_dataset_max | relative_l2_dataset_mean | relative_l2_dataset_min | relative_l2_dataset_max | accuracy_score_dataset_mean | accuracy_score_dataset_min | accuracy_score_dataset_max | invalid_value_fraction_dataset_mean | invalid_value_fraction_dataset_min | invalid_value_fraction_dataset_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers | during_adversarial_training | 1000 | 3000 | 1 | ALL | 52 | 11500 | 0.712865 | 0.00801961 | 0.0042053 | 0.0120691 | 0.00252524 | 0.00188154 | 0.00318365 | 0.0138184 | 0.00790958 | 0.0203329 | 98.6384 | 98.0072 | 99.2152 | 0 | 0 | 0 |
| burgers | during_adversarial_training | 1000 | 3000 | 1 | train | 1 | 1350 | 0.712865 | 0.0042053 | 0.0042053 | 0.0042053 | 0.00188154 | 0.00188154 | 0.00188154 | 0.00790958 | 0.00790958 | 0.00790958 | 99.2152 | 99.2152 | 99.2152 | 0 | 0 | 0 |
| burgers | during_adversarial_training | 1000 | 3000 | 1 | test | 1 | 150 | 0.712865 | 0.00431479 | 0.00431479 | 0.00431479 | 0.0019283 | 0.0019283 | 0.0019283 | 0.00802724 | 0.00802724 | 0.00802724 | 99.2037 | 99.2037 | 99.2037 | 0 | 0 | 0 |
| burgers | during_adversarial_training | 1000 | 3000 | 1 | generalization | 50 | 10000 | 0.712865 | 0.00816999 | 0.00513891 | 0.0120691 | 0.00255005 | 0.00213465 | 0.00318365 | 0.0140524 | 0.00986996 | 0.0203329 | 98.6156 | 98.0072 | 99.0227 | 0 | 0 | 0 |

## Loss reduction summary

- Mean absolute Relative L2 drop across 52 datasets: `0.0166743`.
- Mean fractional Relative L2 drop across 52 datasets: `55.128%`.
- Mean absolute RMSE drop across 52 datasets: `0.00957925`.

## Fixed-probe attack/frequency trend

- Mean fixed-probe attack gain changed from `0.000366458` to `2.92745e-05`.
- Mean fixed-probe high-frequency ratio changed from `0.014817` to `0.0351265`.
- High-frequency ratio slope per epoch: `2.67527e-05`.
- High-frequency ratio Pearson correlation with epoch: `0.936544`.
- Top 1% highest-mode energy share changed from `0.000246505` to `0.000628912`.
- Top 10% highest-mode energy share changed from `0.00229428` to `0.0053782`.
- Top 50% highest-mode energy share changed from `0.0147294` to `0.0349194`.

## Main figures

- `relative_l2_all_52_datasets.png`
- `rmse_all_52_datasets.png`
- `relative_l2_grouped_shared_y.png`
- `rmse_grouped_shared_y.png`
- `relative_l2_grouped_shared_y_distinct_datasets.png`
- `rmse_grouped_shared_y_distinct_datasets.png`
- `relative_l2_group_mean_std.png`
- `rmse_group_mean_std.png`
- `relative_l2_heatmap_52_datasets.png`
- `relative_l2_reduction_by_dataset.png`
- `relative_l2_checkpoint_change_heatmap.png`
- `relative_l2_checkpoint_change_heatmap_50epoch.png`
- `rmse_checkpoint_change_heatmap_50epoch.png`
- `attack_losses_progress.png`
- `attack_relative_gain_progress.png`
- `epsilon_bucket_attack_loss_gain.png`
- `epsilon_bucket_attack_loss_gain_relative.png`
- `fixed_probe_delta_frequency_metrics.png`
- `delta_high_frequency_energy_share_progress.png`
- `delta_checkpoint_shapes_all_probes.png`
- `delta_checkpoint_shapes_probe0.png`
- `delta_checkpoint_fft_probe0.png`
