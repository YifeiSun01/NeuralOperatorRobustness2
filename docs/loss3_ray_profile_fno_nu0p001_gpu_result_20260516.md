# Loss3 Experiment 4 Ray Profile Result - 2026-05-16

Status: completed for FNO / 1D Burgers `nu=0.001` with GPU-only execution.

## Scope And Settings

- Samples: `[0, 7, 40, 47, 115]`.
- Epsilon: `8.0`.
- Attack steps per optimized direction: `50`.
- Attack learning rate: `0.3`.
- Regularization C: `1.0`.
- Output directory: `forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100`.
- Runtime device: `Tesla V100-SXM2-32GB`.

## What Was Measured

For each direction `v`, this run evaluates `x + r v` for radii from `0` to `epsilon` and records endpoint residual norm, clean-residual norm growth ratio, residual increment ratio, model/solver movement ratios, and model-solver movement cosine.

The key comparison is local versus finite-radius behavior: the best small-radius ratio direction need not be the best endpoint `loss3_original` direction at `r=epsilon`.

## Key Conclusion

This run directly shows the nonlinear local-to-global gap. At small radius, the
local reference directions win exactly as the clean Jacobian analysis predicts:
`local_outward_growth` is best for clean residual norm growth in all 5/5
samples, and `local_error_svd` is best for residual-field movement in all 5/5
samples. But at the finite endpoint `r=8`, neither local direction is endpoint
best. The endpoint winners are `loss3_regularized` in 3/5 samples and
`loss3_increment_ratio` in 2/5 samples.

So the local objects are real local diagnostics, not noise, but they are not the
same as the best finite-radius attack directions. The curves are therefore
measuring the intended nonlinearity: a direction can have the best infinitesimal
slope and still lose after traveling a large distance along the ray.

## Aggregate Direction Summary

| direction | endpoint loss3 mean | small growth mean | small resid-ratio mean | endpoint rank mean | endpoint wins | small growth wins | small resid wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss3_increment_ratio | 7.608 | 0.03872 | 0.5376 | 1.6 | 2 | 0 | 0 |
| loss3_regularized | 7.831 | 0.0227 | 0.4987 | 1.8 | 3 | 0 | 0 |
| loss3_residual_increment_ratio | 7.032 | 0.04968 | 0.5126 | 2.6 | 0 | 0 | 0 |
| loss3_original | 4.98 | 0.03907 | 0.4632 | 4.2 | 0 | 0 | 0 |
| local_error_svd | 3.28 | -0.005787 | 0.8679 | 5.2 | 0 | 0 | 5 |
| local_outward_growth | 2.618 | 0.1675 | 0.3691 | 5.6 | 0 | 5 | 0 |
| random | 0.5854 | 0.0008522 | 0.04918 | 7 | 0 | 0 | 0 |

## Winner Counts

- Best finite-radius endpoint `loss3_original`: `{'loss3_regularized': 3, 'loss3_increment_ratio': 2}`.
- Best small-radius clean residual norm growth ratio: `{'local_outward_growth': 5}`.
- Best small-radius residual increment ratio: `{'local_error_svd': 5}`.

Per-sample winners:

| sample index | endpoint `loss3_original` winner | small norm-growth winner | small residual-increment winner |
| --- | --- | --- | --- |
| 0 | `loss3_regularized` | `local_outward_growth` | `local_error_svd` |
| 7 | `loss3_regularized` | `local_outward_growth` | `local_error_svd` |
| 40 | `loss3_increment_ratio` | `local_outward_growth` | `local_error_svd` |
| 47 | `loss3_regularized` | `local_outward_growth` | `local_error_svd` |
| 115 | `loss3_increment_ratio` | `local_outward_growth` | `local_error_svd` |

## Visualizations

![ray_profile_index_000](../forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/figures/ray_profile_index_000.png)

![ray_profile_index_007](../forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/figures/ray_profile_index_007.png)

![ray_profile_index_040](../forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/figures/ray_profile_index_040.png)

![ray_profile_index_047](../forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/figures/ray_profile_index_047.png)

![ray_profile_index_115](../forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/figures/ray_profile_index_115.png)

![ray_profile_aggregate_mean](../forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/figures/ray_profile_aggregate_mean.png)

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

Conclusion from this run: the small-radius ratio winners differ systematically from the endpoint winners, so Experiment 4 succeeds as a local-to-global nonlinearity diagnostic for the current FNO / Burgers `nu=0.001` scope.
