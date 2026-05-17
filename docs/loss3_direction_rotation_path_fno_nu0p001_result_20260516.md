# Loss3 Direction Rotation Along Path Result

Date: 2026-05-16 UTC

Scope: FNO / 1D Burgers `nu=0.001` only.

## Purpose

This is Experiment 6, the direction-rotation-along-path diagnostic. Given a finite-radius direct `loss3_original` attack perturbation `delta*`, the experiment walks along the straight path

```text
x_t = x + t delta*,    t in [0, 1]
```

At each path point it re-estimates the local residual-Jacobian top input direction. In local notation,

```text
J_e(x_t) = J_f(x_t) - J_j(x_t)
v_e*(x_t) = argmax_{||v||_2 = 1} ||J_e(x_t) v||_2
```

Operationally, the run estimates that top right singular direction with the small-epsilon objective

```text
L_e(x_t, v) = ||e(x_t + eps v) - e(x_t)||_2 / eps,    e = f - j
```

using `eps = 0.001`. The clean `t=0` reference direction comes from the saved exact clean-point SVD of `J_e(x_0)`; path-point directions are optimized by Adam on the unit L2 sphere with best-over-steps selection.

## Angle Definition

For each path point, let

```text
v_0 = v_e*(x_0)
v_t = v_e*(x_t)
```

Both are normalized input-space vectors in `R^1024`. The reported clean-reference similarity is sign-invariant:

```text
abs_cos_to_clean = |<v_t, v_0>|
angle_to_clean_deg = arccos(abs_cos_to_clean) * 180 / pi
```

The absolute value is necessary because a singular vector and its negative represent the same singular direction. Therefore `angle_to_clean_deg` lies in `[0, 90]` degrees. The adjacent angle uses the same formula between consecutive path points.

## Run Settings

- Output directory: `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot`
- Delta source: `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/deltas.npz` key `loss3_original_pgd_best`
- Sample indices: `[0, 7, 40, 47, 115]`
- Path fractions: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- Local epsilon: `0.001`
- Direction optimizer: Adam on unit L2 direction with best-over-steps selection
- Optimizer steps: `6`
- Learning rate: `0.15`
- Random starts: `0`
- Start policy: `warm_clean`
- Runtime seconds: `1258.3`
- GPU: `Tesla V100-SXM2-32GB`

## Main Result

At `t=1`, the mean sign-invariant angle from the clean local residual direction is `57.07` degrees. The mean adjacent-step angle at `t=1` is `4.73` degrees.

At `t=1`, the mean ratio

```text
best local gain at x_t / gain of clean direction at x_t
```

is `2.684`. Thus, after moving to the finite-radius endpoint, re-estimating the local residual-Jacobian top direction gives about `2.68x` more local residual movement than continuing to use the clean-point top direction.

## Aggregate By Path Fraction

| t | n | angle clean mean | angle clean std | angle clean min | angle clean max | adj angle mean | adj angle std | gain ratio mean | gain ratio std | gain ratio min | gain ratio max | |cos(v_t,delta*)| mean | |cos(v_t,delta*)| std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 2.958e-07 | 5.915e-07 | 0 | 1.479e-06 | nan | nan | 1 | 0 | 1 | 1 | 0.3651 | 0.3644 |
| 0.25 | 5 | 50.2151 | 32.7168 | 9.9091 | 87.8084 | 50.2151 | 32.7168 | 2.6753 | 1.9353 | 1.0152 | 5.0685 | 0.6957 | 0.1205 |
| 0.5 | 5 | 50.7279 | 30.4513 | 17.9585 | 87.6992 | 18.7846 | 9.0829 | 3.2327 | 2.6021 | 1.0548 | 6.878 | 0.6091 | 0.1404 |
| 0.75 | 5 | 55.4477 | 26.6567 | 24.6404 | 87.6414 | 12.704 | 9.7932 | 3.1452 | 2.5217 | 1.105 | 7.4407 | 0.5339 | 0.1273 |
| 1 | 5 | 57.0735 | 25.5428 | 28.8153 | 88.2566 | 4.7341 | 2.914 | 2.684 | 2.4335 | 1.1494 | 7.5005 | 0.5049 | 0.1208 |

## Per-Sample Endpoint Summary

| sample | path pts | endpoint angle to clean | endpoint gain ratio | mean nonzero angle | max nonzero angle | mean adjacent angle |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 35.5814 | 1.2125 | 38.3286 | 46.0363 | 15.5592 |
| 7 | 5 | 88.2566 | 2.1337 | 87.8514 | 88.2566 | 30.0388 |
| 40 | 5 | 87.0572 | 7.5005 | 87.1203 | 87.3722 | 34.9595 |
| 47 | 5 | 28.8153 | 1.1494 | 20.3308 | 28.8153 | 10.0378 |
| 115 | 5 | 45.6569 | 1.4238 | 33.1992 | 45.6569 | 17.4521 |

## Clean-Point Jacobian Metadata

These are from the saved clean-point SVD of `J_e(x_0)`. The ratio `sigma2/sigma1` is included because a near-tied top singular pair can make top-1 vector angles less stable.

| sample | ||delta*||2 | clean sigma1 | clean sigma2 | sigma2/sigma1 | |cos(v0,delta*)| | angle(v0,delta*) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 8 | 1.2058 | 1.05 | 0.8708 | 0.0661 | 86.2098 |
| 7 | 8 | 0.8402 | 0.7502 | 0.8929 | 0.1091 | 83.7342 |
| 40 | 8 | 0.6168 | 0.5616 | 0.9105 | 0.0298 | 88.2946 |
| 47 | 8 | 0.9487 | 0.7342 | 0.7739 | 0.8206 | 34.8557 |
| 115 | 8 | 0.7297 | 0.5367 | 0.7355 | 0.8 | 36.8695 |

## Full Path Rows

This table records every analyzed `(sample, t)` point. `best_value` is the estimated local `L_e(x_t, v_t)` value. `clean_direction_value` is the same local objective evaluated at the clean-point direction `v_e*(x_0)`. `best_over_clean_direction_value` is their ratio.

| sample | t | ||t delta*||2 | best start | best L_e | clean-dir L_e | gain ratio | angle to clean | adjacent angle | |cos(v_t,delta*)| | angle to delta* | best-second gap | starts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | clean_plus | 1.2057 | 1.2057 | 1 | 0 | nan | 0.0661 | 86.2098 | nan | 1 |
| 0 | 0.25 | 2 | previous | 4.3916 | 3.6406 | 1.2063 | 46.0363 | 46.0363 | 0.462 | 62.4809 | 0 | 2 |
| 0 | 0.5 | 4 | previous | 9.3623 | 7.7274 | 1.2116 | 36.1152 | 12.7264 | 0.3676 | 68.4304 | 0.0003 | 2 |
| 0 | 0.75 | 6 | previous | 9.8882 | 8.1537 | 1.2127 | 35.5814 | 3.474 | 0.3566 | 69.1109 | 0.0008 | 2 |
| 0 | 1 | 8 | previous | 10.1674 | 8.3853 | 1.2125 | 35.5814 | 0 | 0.3566 | 69.1109 | 0.0005 | 2 |
| 7 | 0 | 0 | clean_plus | 0.8402 | 0.8402 | 1 | 0 | nan | 0.1091 | 83.7342 | nan | 1 |
| 7 | 0.25 | 2 | previous | 5.4995 | 1.0955 | 5.02 | 87.8084 | 87.8084 | 0.7197 | 43.9706 | 0 | 2 |
| 7 | 0.5 | 4 | clean_plus | 7.7169 | 1.3044 | 5.9162 | 87.6992 | 22.7526 | 0.6014 | 53.0272 | 0.0016 | 2 |
| 7 | 0.75 | 6 | clean_plus | 8.18 | 1.7682 | 4.6261 | 87.6414 | 4.4866 | 0.579 | 54.6194 | 0.0005 | 2 |
| 7 | 1 | 8 | previous | 8.4637 | 3.9667 | 2.1337 | 88.2566 | 5.1077 | 0.5521 | 56.492 | 0.0005 | 2 |
| 40 | 0 | 0 | clean_plus | 0.6171 | 0.6171 | 1 | 0 | nan | 0.0298 | 88.2946 | nan | 1 |
| 40 | 0.25 | 2 | previous | 3.8667 | 0.7629 | 5.0685 | 87.3722 | 87.3722 | 0.7469 | 41.6734 | 0 | 2 |
| 40 | 0.5 | 4 | clean_plus | 6.513 | 0.9469 | 6.878 | 86.9815 | 34.8187 | 0.5844 | 54.2379 | 0.0004 | 2 |
| 40 | 0.75 | 6 | clean_plus | 7.9803 | 1.0725 | 7.4407 | 87.0702 | 13.7667 | 0.4679 | 62.1022 | 0.0007 | 2 |
| 40 | 1 | 8 | clean_plus | 8.6371 | 1.1515 | 7.5005 | 87.0572 | 3.8806 | 0.4293 | 64.5774 | 0.0001 | 2 |
| 47 | 0 | 0 | clean_plus | 0.9486 | 0.9486 | 1 | 0 | nan | 0.8206 | 34.8557 | nan | 1 |
| 47 | 0.25 | 2 | previous | 5.0322 | 4.9569 | 1.0152 | 9.9091 | 9.9091 | 0.809 | 36.004 | 0 | 2 |
| 47 | 0.5 | 4 | clean_plus | 5.4726 | 5.1883 | 1.0548 | 17.9585 | 10.1177 | 0.7787 | 38.8561 | 0.0019 | 2 |
| 47 | 0.75 | 6 | clean_plus | 5.7961 | 5.2456 | 1.105 | 24.6404 | 11.1123 | 0.7415 | 42.1418 | 0.0025 | 2 |
| 47 | 1 | 8 | clean_plus | 6.1069 | 5.3133 | 1.1494 | 28.8153 | 9.012 | 0.7104 | 44.7285 | 0.0097 | 2 |
| 115 | 0 | 0 | clean_plus | 0.7298 | 0.7298 | 1 | 1.479e-06 | nan | 0.8 | 36.8695 | nan | 1 |
| 115 | 0.25 | 2 | previous | 5.5416 | 5.1949 | 1.0667 | 19.9496 | 19.9496 | 0.7407 | 42.2112 | 0 | 2 |
| 115 | 0.5 | 4 | clean_plus | 6.1495 | 5.5761 | 1.1028 | 24.885 | 13.5078 | 0.7134 | 44.4906 | 0.0007 | 2 |
| 115 | 0.75 | 6 | clean_plus | 8.522 | 6.3515 | 1.3417 | 42.3055 | 30.6804 | 0.5245 | 58.3651 | 0.0003 | 2 |
| 115 | 1 | 8 | clean_plus | 9.1007 | 6.392 | 1.4238 | 45.6569 | 5.6704 | 0.4762 | 61.5622 | 0.0001 | 2 |

## Output Files

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/direction_rotation_by_sample_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/direction_rotation_aggregate_by_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/direction_rotation_summary_by_sample.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/optimization_trace.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/optimized_directions.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/manifest.json`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/figures/angle_to_clean_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/figures/adjacent_angle_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/figures/best_over_clean_gain_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/figures/angle_to_clean_by_sample.png`

## Interpretation

The clean sanity check works: at `t=0`, the angle to the clean direction is numerically zero and the gain ratio is exactly `1.0`.

The direction changes quickly after leaving the clean point. By `t=0.25`, the mean angle to the clean top direction is about `50.22` degrees. By `t=1.0`, the mean angle is about `57.07` degrees.

The gain-ratio column shows that this is not merely a cosmetic angle change. At nonzero path points, the re-estimated path-local direction typically gives substantially larger local residual movement than the clean direction. The effect is strongest for sample `40`, where the endpoint gain ratio is `7.50x`.

This supports the Experiment 6 claim for the current FNO / Burgers `nu=0.001` scope: along the finite-radius direct `loss3_original` attack path, the top input direction of the residual Jacobian is path-dependent. The clean-point local Jacobian top direction is therefore not a faithful representative of the local Jacobian geometry encountered later along the finite-radius path.
