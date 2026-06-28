# Normal-Protocol Loss3 Experiment 4 Ray Profile Result - 2026-05-16

Status: GPU run completed for FNO / 1D Burgers `nu=0.001`, batch size 100.

## Important Protocol Note

This run intentionally does not use best-over-steps and does not use multi-restart selection. Each attack objective uses one initialization, runs `pgd` to the final step, and the final direction is the ray direction.

## Scope And Settings

- Samples: `[0, 1, 2, 3]`.
- Epsilon: `8.0`.
- Attack optimizer: `pgd`.
- Attack initialization: `zero`.
- Attack steps: `2`.
- Attack learning rate / PGD alpha: `0.3`.
- Output directory: `forensics/loss3_ray_profile_pgd_20260516/smoke_gpu`.
- Runtime device: `Tesla V100-SXM2-32GB`.
- Runtime seconds: `22.15`.

## Winner Counts

- Best endpoint `loss3` at `r=epsilon`: `{'local_outward_growth': 1, 'local_residual_movement': 3}`.
- Best small-radius clean residual norm-growth ratio: `{'local_outward_growth': 4}`.
- Best small-radius residual-increment ratio: `{'local_residual_movement': 4}`.

## Aggregate Direction Table

| direction | endpoint mean | endpoint wins | small growth mean | small growth wins | small residual-ratio mean | small residual wins |
| --- | --- | --- | --- | --- | --- | --- |
| local-residual | 2.352 | 3 | -0.0199 | 0 | 0.8260 | 4 |
| local-outward | 1.880 | 1 | 0.1747 | 4 | 0.3755 | 0 |
| original-final | 1.076 | 0 | -0.1731 | 0 | 0.3741 | 0 |
| regularized-final | 1.049 | 0 | -0.1732 | 0 | 0.3739 | 0 |
| inc-ratio-final | 1.027 | 0 | -0.1732 | 0 | 0.3734 | 0 |
| random | 0.5883 | 0 | 0.00226 | 0 | 0.0451 | 0 |
| resid-ratio-final | 0.2878 | 0 | 0 | 0 | 0 | 0 |

## Local-To-Global Gap Summary

| quantity | mean | std | min | max |
| --- | --- | --- | --- | --- |
| endpoint_over_small_growth_endpoint_loss3 | 1.454 | 0.4370 | 1.000 | 2.045 |
| endpoint_over_small_residual_endpoint_loss3 | 1.039 | 0.0677 | 1.000 | 1.156 |

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

![normal_batch100_mean_loss3_vs_r](../forensics/loss3_ray_profile_pgd_20260516/smoke_gpu/figures/normal_batch100_mean_loss3_vs_r.png)

![normal_batch100_mean_loss1_loss2_loss3_vs_r](../forensics/loss3_ray_profile_pgd_20260516/smoke_gpu/figures/normal_batch100_mean_loss1_loss2_loss3_vs_r.png)

![normal_batch100_mean_ratio_curves_vs_r](../forensics/loss3_ray_profile_pgd_20260516/smoke_gpu/figures/normal_batch100_mean_ratio_curves_vs_r.png)

![normal_batch100_ray_profile_index_000](../forensics/loss3_ray_profile_pgd_20260516/smoke_gpu/figures/normal_batch100_ray_profile_index_000.png)

![normal_batch100_ray_profile_index_001](../forensics/loss3_ray_profile_pgd_20260516/smoke_gpu/figures/normal_batch100_ray_profile_index_001.png)

![normal_batch100_ray_profile_index_002](../forensics/loss3_ray_profile_pgd_20260516/smoke_gpu/figures/normal_batch100_ray_profile_index_002.png)
