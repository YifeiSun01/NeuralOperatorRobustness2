# Normal-Protocol Loss3 Experiment 4 Ray Profile Result - 2026-05-16

Status: GPU run completed for FNO / 1D Burgers `nu=0.001`, batch size 100.

## Important Protocol Note

This run intentionally does not use best-over-steps and does not use multi-restart selection. Each attack objective uses one initialization, runs to the final Adam step, and the final direction is the ray direction.

## Scope And Settings

- Samples: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`.
- Epsilon: `8.0`.
- Attack steps: `50`.
- Attack learning rate: `0.3`.
- Output directory: `forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100`.
- Runtime device: `Tesla V100-SXM2-32GB`.
- Runtime seconds: `278.7`.

## Winner Counts

- Best endpoint `loss3` at `r=epsilon`: `{'loss3_regularized_final': 43, 'loss3_residual_increment_ratio_final': 2, 'loss3_increment_ratio_final': 47, 'local_residual_movement': 7, 'loss3_original_final': 1}`.
- Best small-radius clean residual norm-growth ratio: `{'local_outward_growth': 100}`.
- Best small-radius residual-increment ratio: `{'local_residual_movement': 100}`.

## Aggregate Direction Table

| direction | endpoint mean | endpoint wins | small growth mean | small growth wins | small residual-ratio mean | small residual wins |
| --- | --- | --- | --- | --- | --- | --- |
| inc-ratio-final | 6.992 | 47 | 0.0855 | 0 | 0.6485 | 0 |
| regularized-final | 5.780 | 43 | 0.0652 | 0 | 0.5500 | 0 |
| resid-ratio-final | 5.249 | 2 | 0.0391 | 0 | 0.6652 | 0 |
| local-residual | 4.139 | 7 | 0.0121 | 0 | 1.101 | 100 |
| original-final | 4.365 | 1 | 0.0699 | 0 | 0.5244 | 0 |
| local-outward | 2.720 | 0 | 0.2414 | 100 | 0.5311 | 0 |
| random | 0.5268 | 0 | -0.00099 | 0 | 0.0462 | 0 |

## Local-To-Global Gap Summary

| quantity | mean | std | min | max |
| --- | --- | --- | --- | --- |
| endpoint_over_small_growth_endpoint_loss3 | 4.247 | 3.053 | 1.131 | 18.20 |
| endpoint_over_small_residual_endpoint_loss3 | 3.469 | 5.032 | 1.000 | 29.41 |

Interpretation: values larger than 1 mean the endpoint winner has larger `loss3` at `r=8` than the direction that won the corresponding small-radius diagnostic. This is the direct ray-profile evidence for local-to-global mismatch.


## Direct Answer To The Batch-100 Question

Under the requested normal protocol, increasing the sample count to 100 did **not** make `loss3_original_final` the best endpoint direction. With single initialization, final-step-only selection, and no multi-restart, the endpoint winners at `r=8` are:

- `loss3_increment_ratio_final`: 47/100
- `loss3_regularized_final`: 43/100
- `local_residual_movement`: 7/100
- `loss3_residual_increment_ratio_final`: 2/100
- `loss3_original_final`: 1/100

So the earlier five-sample behavior was not just small-sample randomness. Under this exact old-style protocol, the same pattern persists at batch 100: local/ratio-related directions often beat the final-step `loss3_original` direction at the finite endpoint.

But the ray-profile claim is slightly different and is now very clear: the local winner and endpoint winner are different. At small radius, `local_outward_growth` wins the clean residual norm-growth slope in 100/100 samples, and `local_residual_movement` wins residual-increment slope in 100/100 samples. At `r=8`, neither local direction is generally endpoint-best.

## Selected Radius Values

This table is the direct curve evidence. Read it row-wise by radius. At tiny radius, `local-outward` has the largest norm-growth ratio and the highest mean `loss3` among the near-clean rays. By `r=2`, `r=4`, and `r=8`, the finite-radius optimized ratio/regularized directions overtake it strongly.

| r | direction | mean loss3 | mean norm-growth ratio | mean residual-increment ratio |
| --- | --- | --- | --- | --- |
| 0.0001 | local-outward | 0.2976 | 0.2373 | 0.5245 |
| 0.0001 | inc-ratio-final | 0.2976 | 0.0837 | 0.6400 |
| 0.0001 | regularized-final | 0.2976 | 0.0641 | 0.5428 |
| 0.0001 | original-final | 0.2976 | 0.0692 | 0.5186 |
| 0.0001 | resid-ratio-final | 0.2976 | 0.0382 | 0.6576 |
| 0.0001 | local-residual | 0.2976 | 0.0112 | 1.090 |
| 0.0001 | random | 0.2976 | -0.00104 | 0.0475 |
| 0.001 | local-outward | 0.2978 | 0.2394 | 0.5289 |
| 0.001 | inc-ratio-final | 0.2977 | 0.0844 | 0.6457 |
| 0.001 | regularized-final | 0.2977 | 0.0644 | 0.5476 |
| 0.001 | original-final | 0.2977 | 0.0695 | 0.5231 |
| 0.001 | resid-ratio-final | 0.2976 | 0.0385 | 0.6636 |
| 0.001 | local-residual | 0.2976 | 0.0112 | 1.100 |
| 0.001 | random | 0.2976 | -0.00099 | 0.0462 |
| 0.0100 | local-outward | 0.3000 | 0.2414 | 0.5311 |
| 0.0100 | inc-ratio-final | 0.2985 | 0.0855 | 0.6485 |
| 0.0100 | regularized-final | 0.2982 | 0.0652 | 0.5500 |
| 0.0100 | original-final | 0.2983 | 0.0699 | 0.5244 |
| 0.0100 | resid-ratio-final | 0.2980 | 0.0391 | 0.6652 |
| 0.0100 | local-residual | 0.2977 | 0.0121 | 1.101 |
| 0.0100 | random | 0.2976 | -0.00099 | 0.0462 |
| 0.1000 | local-outward | 0.3234 | 0.2583 | 0.5456 |
| 0.1000 | inc-ratio-final | 0.3072 | 0.0963 | 0.6519 |
| 0.1000 | regularized-final | 0.3050 | 0.0736 | 0.5565 |
| 0.1000 | original-final | 0.3049 | 0.0733 | 0.5246 |
| 0.1000 | resid-ratio-final | 0.3020 | 0.0444 | 0.6588 |
| 0.1000 | local-residual | 0.2997 | 0.0209 | 1.046 |
| 0.1000 | random | 0.2975 | -0.000726 | 0.0462 |
| 1.000 | local-outward | 0.6302 | 0.3326 | 0.5220 |
| 1.000 | inc-ratio-final | 0.5381 | 0.2405 | 0.4698 |
| 1.000 | regularized-final | 0.5279 | 0.2303 | 0.4438 |
| 1.000 | original-final | 0.4177 | 0.1201 | 0.3487 |
| 1.000 | resid-ratio-final | 0.4314 | 0.1338 | 0.3791 |
| 1.000 | local-residual | 0.3984 | 0.1008 | 0.3555 |
| 1.000 | random | 0.2996 | 0.00202 | 0.0462 |
| 2.000 | local-outward | 0.9748 | 0.3386 | 0.4596 |
| 2.000 | inc-ratio-final | 1.103 | 0.4025 | 0.5374 |
| 2.000 | regularized-final | 1.157 | 0.4297 | 0.5570 |
| 2.000 | original-final | 0.6584 | 0.1804 | 0.3103 |
| 2.000 | resid-ratio-final | 0.7955 | 0.2489 | 0.3887 |
| 2.000 | local-residual | 0.6155 | 0.1589 | 0.2954 |
| 2.000 | random | 0.3081 | 0.00527 | 0.0462 |
| 4.000 | local-outward | 1.569 | 0.3179 | 0.3859 |
| 4.000 | inc-ratio-final | 2.890 | 0.6481 | 0.7210 |
| 4.000 | regularized-final | 2.728 | 0.6076 | 0.6769 |
| 4.000 | original-final | 1.519 | 0.3052 | 0.3760 |
| 4.000 | resid-ratio-final | 2.111 | 0.4533 | 0.5287 |
| 4.000 | local-residual | 1.408 | 0.2775 | 0.3506 |
| 4.000 | random | 0.3482 | 0.0127 | 0.0475 |
| 8.000 | local-outward | 2.720 | 0.3029 | 0.3392 |
| 8.000 | inc-ratio-final | 6.992 | 0.8368 | 0.8741 |
| 8.000 | regularized-final | 5.780 | 0.6852 | 0.7212 |
| 8.000 | original-final | 4.365 | 0.5085 | 0.5455 |
| 8.000 | resid-ratio-final | 5.249 | 0.6189 | 0.6570 |
| 8.000 | local-residual | 4.139 | 0.4802 | 0.5186 |
| 8.000 | random | 0.5268 | 0.0286 | 0.0538 |

## Compact Aggregate Recheck

| direction | endpoint mean loss3 | endpoint wins | small growth mean | small growth wins | small residual ratio mean | small residual wins |
| --- | --- | --- | --- | --- | --- | --- |
| inc-ratio-final | 6.992 | 47 | 0.0855 | 0 | 0.6485 | 0 |
| regularized-final | 5.780 | 43 | 0.0652 | 0 | 0.5500 | 0 |
| resid-ratio-final | 5.249 | 2 | 0.0391 | 0 | 0.6652 | 0 |
| local-residual | 4.139 | 7 | 0.0121 | 0 | 1.101 | 100 |
| original-final | 4.365 | 1 | 0.0699 | 0 | 0.5244 | 0 |
| local-outward | 2.720 | 0 | 0.2414 | 100 | 0.5311 | 0 |
| random | 0.5268 | 0 | -0.00099 | 0 | 0.0462 | 0 |

## Interpretation For The Ray Experiment

The graph you asked for is now generated as `normal_batch100_mean_loss3_vs_r.png`: the x-axis is radius `r`, and the y-axis is mean `loss3` along each fixed ray direction. The result shows exactly the local-to-global effect: a direction can be the best infinitesimal/small-radius direction, but after following that same ray outward to large radius, another direction can produce much larger endpoint loss.

This batch-100 run should be cited as the **normal-protocol ray-profile check**. It is intentionally not a strongest-possible endpoint attack comparison; it is the old-style single-path/final-step protocol you requested.

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

![normal_batch100_mean_loss3_vs_r](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_mean_loss3_vs_r.png)

![normal_batch100_mean_loss1_loss2_loss3_vs_r](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_mean_loss1_loss2_loss3_vs_r.png)

![normal_batch100_mean_ratio_curves_vs_r](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_mean_ratio_curves_vs_r.png)

![normal_batch100_ray_profile_index_000](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_000.png)

![normal_batch100_ray_profile_index_001](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_001.png)

![normal_batch100_ray_profile_index_002](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_002.png)

![normal_batch100_ray_profile_index_007](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_007.png)

![normal_batch100_ray_profile_index_010](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_010.png)

![normal_batch100_ray_profile_index_040](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_040.png)

![normal_batch100_ray_profile_index_047](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_047.png)

![normal_batch100_ray_profile_index_099](../forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/figures/normal_batch100_ray_profile_index_099.png)

