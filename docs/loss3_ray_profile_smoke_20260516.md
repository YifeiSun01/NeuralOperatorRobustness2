# Loss3 Experiment 4 Ray Profile Result - 2026-05-16

Status: completed for FNO / 1D Burgers `nu=0.001` with GPU-only execution.

## Scope And Settings

- Samples: `[0]`.
- Epsilon: `0.5`.
- Attack steps per optimized direction: `2`.
- Attack learning rate: `0.3`.
- Regularization C: `1.0`.
- Output directory: `forensics/loss3_ray_profile_20260516/smoke_gpu`.
- Runtime device: `Tesla V100-SXM2-32GB`.

## What Was Measured

For each direction `v`, this run evaluates `x + r v` for radii from `0` to `epsilon` and records endpoint residual norm, clean-residual norm growth ratio, residual increment ratio, model/solver movement ratios, and model-solver movement cosine.

The key comparison is local versus finite-radius behavior: the best small-radius ratio direction need not be the best endpoint `loss3_original` direction at `r=epsilon`.

## Aggregate Direction Summary

| direction | endpoint loss3 mean | small growth mean | small resid-ratio mean | endpoint rank mean | endpoint wins | small growth wins | small resid wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| local_outward_growth | 0.4528 | 0.2059 | 0.406 | 1 | 1 | 1 | 0 |
| loss3_increment_ratio | 0.4437 | 0.1662 | 0.3787 | 2 | 0 | 0 | 0 |
| loss3_original | 0.4365 | 0.1744 | 0.3648 | 3 | 0 | 0 | 0 |
| loss3_residual_increment_ratio | 0.3806 | 0.0511 | 0.5569 | 4 | 0 | 0 | 0 |
| loss3_regularized | 0.358 | 0.0526 | 0.3426 | 5 | 0 | 0 | 0 |
| random | 0.335 | 0.002343 | 0.07868 | 6 | 0 | 0 | 0 |
| local_error_svd | 0.3327 | -0.0247 | 1.204 | 7 | 0 | 0 | 1 |

## Winner Counts

- Best finite-radius endpoint `loss3_original`: `{'local_outward_growth': 1}`.
- Best small-radius clean residual norm growth ratio: `{'local_outward_growth': 1}`.
- Best small-radius residual increment ratio: `{'local_error_svd': 1}`.

## Visualizations

![ray_profile_index_000](../forensics/loss3_ray_profile_20260516/smoke_gpu/figures/ray_profile_index_000.png)

![ray_profile_aggregate_mean](../forensics/loss3_ray_profile_20260516/smoke_gpu/figures/ray_profile_aggregate_mean.png)

## Interpretation

This experiment is the finite-radius companion to the small-epsilon sweep. The small-epsilon experiment showed that ratio diagnostics are meaningful local objects; this ray-profile experiment asks whether those local objects remain endpoint-best over a large radius.

The raw evidence to inspect is:

- `ray_profile.csv`: every sample, direction, and radius.
- `ray_direction_summary.csv`: endpoint and small-radius summaries per sample/direction.
- `ray_direction_aggregate.csv`: aggregate means/ranks/win counts by direction.
- `ray_winner_summary.csv`: per-sample winners for endpoint and small-radius criteria.
- `attack_trace.csv`: optimization traces for regenerated finite-radius directions.
- `directions.npz`: normalized directions used by the ray profiles.
- `manifest.json`: GPU/runtime/source metadata.

Conclusion should be read through winner changes and curve crossings: if the small-radius ratio winners differ from endpoint winners, the experiment demonstrates the nonlinear local-to-global gap that Experiment 4 was designed to expose.
