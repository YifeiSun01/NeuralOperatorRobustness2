# Corrected Loss3 Experiment 4 Ray Profile Result - 2026-05-16

Status: corrected GPU run completed for FNO / 1D Burgers `nu=0.001`, batch size 20.

## Superseded Point

The earlier ray-profile endpoint-winner statement is superseded. It used the last `loss3_original` step and did not give direct endpoint optimization a fair best-over-steps / restart comparison.

## Scope And Settings

- Samples: `[0, 7]`.
- Epsilon: `8.0`.
- Attack steps per restart: `2`.
- Attack learning rate: `0.3`.
- Output directory: `forensics/loss3_ray_profile_corrected_20260516/smoke_gpu`.
- Runtime device: `Tesla V100-SXM2-32GB`.

## Aggregate Direction Summary

| direction | endpoint loss3 mean | small growth mean | small resid-ratio mean | endpoint rank mean | endpoint wins | small growth wins | small resid wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss3_original_pgd_best | 4.932 | 0.1931 | 0.501 | 1 | 2 | 0 | 0 |
| loss3_increment_ratio_pgd_best | 4.501 | 0.1944 | 0.4887 | 2 | 0 | 0 | 0 |
| local_outward_growth | 3.74 | 0.2012 | 0.4393 | 3.5 | 0 | 0 | 0 |
| loss3_regularized_pgd_best | 3.74 | 0.2012 | 0.4393 | 3.5 | 0 | 2 | 0 |
| loss3_residual_increment_ratio_pgd_best | 2.629 | 0.04368 | 0.8629 | 5 | 0 | 0 | 1 |
| local_residual_movement | 2.629 | 0.04368 | 0.8629 | 6 | 0 | 0 | 1 |
| random | 0.6391 | 0.003926 | 0.04505 | 7 | 0 | 0 | 0 |

## Winner Counts

- Best finite-radius endpoint `loss3_original`: `{'loss3_original_pgd_best': 2}`.
- Best small-radius clean residual norm growth ratio: `{'loss3_regularized_pgd_best': 2}`.
- Best small-radius residual increment ratio: `{'loss3_residual_increment_ratio_pgd_best': 1, 'local_residual_movement': 1}`.

## Direction Alignment Diagnostics

| source_a | source_b | mean angle | std angle | max angle |
| --- | --- | --- | --- | --- |
| loss3_increment_ratio_pgd_best | local_outward_growth | 15.53 | 2.295 | 17.82 |
| loss3_residual_increment_ratio_pgd_best | local_residual_movement | 1.129e-06 | 1.129e-06 | 2.259e-06 |
| loss3_original_pgd_best | local_outward_growth | 16.93 | 2.598 | 19.53 |
| loss3_original_pgd_best | loss3_increment_ratio_pgd_best | 3.526 | 1.066 | 4.592 |
| loss3_original_pgd_best | loss3_regularized_pgd_best | 16.93 | 2.598 | 19.53 |

## Interpretation

This corrected experiment should be read as a ray diagnostic, not just a winner table. The intended evidence is: local outward growth controls the small-radius norm-growth slope, local residual movement controls the small-radius residual-increment slope, and direct `loss3_original` PGD is the fair finite-radius endpoint attack baseline.

The direct `loss3_original` direction is computed after the control directions and is restarted from their boundary-normalized rays. Therefore, if a control ray has a high endpoint value, direct endpoint PGD is allowed to start there and improve it. This removes the unfair endpoint comparison in the first run.

## Raw Artifacts

- `ray_profile.csv`: every sample, direction, and radius.
- `ray_direction_summary.csv`: endpoint and small-radius summaries per sample/direction.
- `ray_direction_aggregate.csv`: aggregate means/ranks/win counts by direction.
- `ray_winner_summary.csv`: per-sample endpoint and small-radius winners.
- `attack_trace.csv`: PGD trace for every source/restart.
- `attack_best_by_sample.csv`: best step/restart per sample.
- `direction_alignment.csv`: angles between local and finite-radius directions.
- `directions.npz`: normalized directions used by the ray profiles.
- `deltas.npz`: best deltas before ray normalization.
- `manifest.json`: GPU/runtime/source metadata.

## Visualizations

![ray_profile_corrected_index_000](../forensics/loss3_ray_profile_corrected_20260516/smoke_gpu/figures/ray_profile_corrected_index_000.png)

![ray_profile_corrected_index_007](../forensics/loss3_ray_profile_corrected_20260516/smoke_gpu/figures/ray_profile_corrected_index_007.png)

![ray_profile_corrected_aggregate_mean](../forensics/loss3_ray_profile_corrected_20260516/smoke_gpu/figures/ray_profile_corrected_aggregate_mean.png)
