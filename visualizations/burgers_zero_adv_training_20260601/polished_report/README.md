# Polished Burgers Adversarial Training Visualizations

This folder contains presentation-style visualizations for the completed Burgers ADV-only run.

## Key Numbers

- Relative L2 improved on `52/52` datasets.
- Mean Relative L2 changed from `0.0304927` to `0.0138184`.
- Mean fractional Relative L2 reduction: `55.128%`.
- Mean attack gain changed from `0.000279671` to `2.28501e-05`.
- Top 50% high-frequency energy share changed from `0.0147294` to `0.0349194`.

## Figures

- `polished_report_dashboard.png`
- `polished_loss_reduction_by_dataset.png`
- `polished_relative_l2_heatmap.png`
- `polished_checkpoint_loss_summary.png`
- `polished_attack_loss_buckets.png`
- `polished_high_frequency_energy_share.png`
- `polished_delta_probe_checkpoint.png`
- `polished_report.pdf`

## CSV Outputs

- `polished_attack_loss_epoch_decile_summary.csv`
- `polished_checkpoint_relative_l2_changes.csv`
- `polished_high_frequency_energy_share_trend_summary.csv`
- `polished_relative_l2_reduction_by_dataset.csv`
- `polished_rmse_reduction_by_dataset.csv`

## Raw Attack-Loss Addendum

These figures remove the 25-epoch moving average and show raw epoch means with standard-deviation shading.

- `polished_attack_loss_raw_std_no_moving_average.png`
- `polished_epsilon_bucket_adv_after_and_gain_raw_std.png`
- `polished_attack_loss_raw_std_summary.csv`
