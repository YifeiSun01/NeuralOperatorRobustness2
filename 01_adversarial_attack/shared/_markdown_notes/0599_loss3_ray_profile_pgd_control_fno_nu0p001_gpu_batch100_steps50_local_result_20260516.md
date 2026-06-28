# Normal-Protocol Loss3 Experiment 4 Ray Profile Result - 2026-05-16

Status: GPU run completed for FNO / 1D Burgers `nu=0.001`, batch size 100.

## Important Protocol Note

This run intentionally does not use best-over-steps and does not use multi-restart selection. Each attack objective uses one initialization, runs `pgd` to the final step, and the final direction is the ray direction.

## Scope And Settings

- Samples: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`.
- Epsilon: `8.0`.
- Attack optimizer: `pgd`.
- Attack initialization: `local`.
- Attack steps: `50`.
- Attack learning rate / PGD alpha: `0.3`.
- Output directory: `forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local`.
- Runtime device: `Tesla V100-SXM2-32GB`.
- Runtime seconds: `280.2`.

## Winner Counts

- Best endpoint `loss3` at `r=epsilon`: `{'loss3_original_final': 48, 'local_residual_movement': 43, 'loss3_regularized_final': 3, 'loss3_increment_ratio_final': 5, 'local_outward_growth': 1}`.
- Best small-radius clean residual norm-growth ratio: `{'local_outward_growth': 100}`.
- Best small-radius residual-increment ratio: `{'local_residual_movement': 100}`.

## Aggregate Direction Table

| direction | endpoint mean | endpoint wins | small growth mean | small growth wins | small residual-ratio mean | small residual wins |
| --- | --- | --- | --- | --- | --- | --- |
| original-final | 4.650 | 48 | 0.1742 | 0 | 0.5914 | 0 |
| local-residual | 4.139 | 43 | 0.0121 | 0 | 1.101 | 100 |
| inc-ratio-final | 4.027 | 5 | 0.1913 | 0 | 0.5896 | 0 |
| local-outward | 2.720 | 1 | 0.2414 | 100 | 0.5311 | 0 |
| regularized-final | 2.066 | 3 | 0.0876 | 0 | 0.4717 | 0 |
| resid-ratio-final | 1.454 | 0 | 0.0145 | 0 | 0.2080 | 0 |
| random | 0.5268 | 0 | -0.00099 | 0 | 0.0462 | 0 |

## Local-To-Global Gap Summary

| quantity | mean | std | min | max |
| --- | --- | --- | --- | --- |
| endpoint_over_small_growth_endpoint_loss3 | 2.807 | 1.855 | 1.000 | 12.15 |
| endpoint_over_small_residual_endpoint_loss3 | 2.479 | 4.075 | 1.000 | 23.17 |

Interpretation: values larger than 1 mean the endpoint winner has larger `loss3` at `r=8` than the direction that won the corresponding small-radius diagnostic. This is the direct ray-profile evidence for local-to-global mismatch.

## Artifacts

- `ray_profile.csv`: every sample, direction, and radius.
- `ray_profile_aggregate_curves.csv`: mean curves for all losses/ratios.
- `ray_direction_summary.csv`: endpoint and small-radius summaries per sample/direction.
- `ray_direction_aggregate.csv`: aggregate ranks and winner counts.
- `ray_winner_summary.csv`: per-sample winners.
- `local_to_global_gap_summary.csv`: per-sample comparison of local winners against endpoint winner.
- `attack_trace.csv`: final-step single-init attack traces.
- `attack_final_by_sample.csv`: final attack values per sample.
- `manifest.json`: GPU/runtime metadata.

## Visualizations

![normal_batch100_mean_loss3_vs_r](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_mean_loss3_vs_r.png)

![normal_batch100_mean_loss1_loss2_loss3_vs_r](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_mean_loss1_loss2_loss3_vs_r.png)

![normal_batch100_mean_ratio_curves_vs_r](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_mean_ratio_curves_vs_r.png)

![normal_batch100_ray_profile_index_000](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_000.png)

![normal_batch100_ray_profile_index_001](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_001.png)

![normal_batch100_ray_profile_index_002](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_002.png)

![normal_batch100_ray_profile_index_007](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_007.png)

![normal_batch100_ray_profile_index_010](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_010.png)

![normal_batch100_ray_profile_index_040](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_040.png)

![normal_batch100_ray_profile_index_047](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_047.png)

![normal_batch100_ray_profile_index_099](../forensics/loss3_ray_profile_pgd_control_20260516/fno_nu0p001_gpu_v100_batch100_steps50_local/figures/normal_batch100_ray_profile_index_099.png)
