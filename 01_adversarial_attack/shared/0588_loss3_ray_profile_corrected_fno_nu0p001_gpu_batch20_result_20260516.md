# Corrected Loss3 Experiment 4 Ray Profile Result - 2026-05-16

Status: corrected GPU run completed for FNO / 1D Burgers `nu=0.001`, batch size 20.

## Superseded Point

The earlier ray-profile endpoint-winner statement is superseded. It used the last `loss3_original` step and did not give direct endpoint optimization a fair best-over-steps / restart comparison.

## Scope And Settings

- Samples: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 40, 47, 115]`.
- Epsilon: `8.0`.
- Attack steps per restart: `100`.
- Attack learning rate: `0.3`.
- Output directory: `forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20`.
- Runtime device: `Tesla V100-SXM2-32GB`.

## Aggregate Direction Summary

| direction | endpoint loss3 mean | small growth mean | small resid-ratio mean | endpoint rank mean | endpoint wins | small growth wins | small resid wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss3_original_pgd_best | 7.728 | 0.03245 | 0.6162 | 1 | 20 | 0 | 0 |
| loss3_residual_increment_ratio_pgd_best | 4.736 | 0.008183 | 0.9237 | 3.2 | 0 | 0 | 18 |
| loss3_increment_ratio_pgd_best | 3.866 | 0.1324 | 0.4946 | 3.3 | 0 | 1 | 0 |
| local_residual_movement | 3.997 | 0.02998 | 0.9715 | 3.4 | 0 | 0 | 2 |
| local_outward_growth | 2.036 | 0.1985 | 0.4269 | 5.05 | 0 | 4 | 0 |
| loss3_regularized_pgd_best | 2.036 | 0.1985 | 0.4269 | 5.2 | 0 | 15 | 0 |
| random | 0.5465 | -0.0003485 | 0.04918 | 6.85 | 0 | 0 | 0 |

## Winner Counts

- Best finite-radius endpoint `loss3_original`: `{'loss3_original_pgd_best': 20}`.
- Best small-radius clean residual norm growth ratio: `{'loss3_regularized_pgd_best': 15, 'local_outward_growth': 4, 'loss3_increment_ratio_pgd_best': 1}`.
- Best small-radius residual increment ratio: `{'loss3_residual_increment_ratio_pgd_best': 18, 'local_residual_movement': 2}`.

## Direction Alignment Diagnostics

| source_a | source_b | mean angle | std angle | max angle |
| --- | --- | --- | --- | --- |
| loss3_increment_ratio_pgd_best | local_outward_growth | 49 | 20.21 | 85.62 |
| loss3_residual_increment_ratio_pgd_best | local_residual_movement | 7.095 | 21.36 | 76.7 |
| loss3_original_pgd_best | local_outward_growth | 75.41 | 12.15 | 89.1 |
| loss3_original_pgd_best | loss3_increment_ratio_pgd_best | 64.5 | 26.36 | 89.48 |
| loss3_original_pgd_best | loss3_regularized_pgd_best | 75.41 | 12.15 | 89.1 |

## Interpretation

This corrected experiment should be read as a ray diagnostic, not just a winner table. The intended evidence is: local outward growth controls the small-radius norm-growth slope, local residual movement controls the small-radius residual-increment slope, and direct `loss3_original` PGD is the fair finite-radius endpoint attack baseline.

The direct `loss3_original` direction is computed after the control directions and is restarted from their boundary-normalized rays. Therefore, if a control ray has a high endpoint value, direct endpoint PGD is allowed to start there and improve it. This removes the unfair endpoint comparison in the first run.


## Corrected Core Conclusion

This corrected batch-20 run fixes the earlier unfair endpoint comparison. The direct endpoint attack `loss3_original_pgd_best` is now optimized with best-over-steps and multiple restarts, including restarts from the boundary-normalized ratio and regularized directions. Under this fair comparison, `loss3_original_pgd_best` wins the finite-radius endpoint at `r=8` in **20/20 samples**.

The local story is different and is exactly the point of the Ray experiment: local/ratio directions explain the small-radius slopes, but they are not the same as the best large-radius endpoint direction. The ray curves therefore show the local-to-global nonlinear gap: a direction can be locally steep but stop being endpoint-best after traveling far along the same fixed ray.

A practical correction to the first five-sample run is also recorded here: the earlier statement that regularized/increment-ratio were endpoint-best is superseded. That happened because the first implementation compared the final `loss3_original` step without a strong best-over-steps, multi-restart endpoint baseline.

## Corrected Aggregate Table

| direction | endpoint mean | endpoint wins | small growth mean | small growth wins | small residual-ratio mean | small residual wins |
| --- | --- | --- | --- | --- | --- | --- |
| original | 7.728 | 20 | 0.0324 | 0 | 0.6162 | 0 |
| resid-ratio | 4.736 | 0 | 0.00818 | 0 | 0.9237 | 18 |
| inc-ratio | 3.866 | 0 | 0.1324 | 1 | 0.4946 | 0 |
| local-residual | 3.997 | 0 | 0.0300 | 0 | 0.9715 | 2 |
| local-outward | 2.036 | 0 | 0.1985 | 4 | 0.4269 | 0 |
| regularized | 2.036 | 0 | 0.1985 | 15 | 0.4269 | 0 |
| random | 0.5465 | 0 | -0.000349 | 0 | 0.0492 | 0 |

## Per-Sample Endpoint `loss3_original` At `r=8`

| sample | original | inc-ratio | resid-ratio | regularized | local-outward | local-residual | random | endpoint winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8.226 | 6.376 | 5.376 | 2.929 | 2.929 | 5.376 | 0.5893 | original |
| 1 | 6.610 | 2.395 | 4.895 | 1.432 | 1.432 | 4.895 | 0.6398 | original |
| 2 | 8.439 | 2.036 | 3.063 | 2.264 | 2.264 | 3.063 | 0.6930 | original |
| 3 | 5.111 | 2.193 | 2.663 | 0.8952 | 0.8952 | 2.663 | 0.4311 | original |
| 4 | 7.622 | 2.530 | 4.012 | 1.410 | 1.410 | 4.012 | 0.2880 | original |
| 5 | 7.832 | 1.741 | 5.906 | 1.026 | 1.026 | 5.906 | 0.4134 | original |
| 6 | 7.200 | 6.687 | 5.442 | 3.410 | 3.410 | 5.442 | 0.6171 | original |
| 7 | 8.897 | 7.060 | 7.024 | 4.552 | 4.552 | 0.6617 | 0.8317 | original |
| 8 | 8.606 | 8.200 | 6.458 | 5.425 | 5.425 | 6.458 | 0.4635 | original |
| 9 | 7.762 | 2.330 | 4.517 | 1.156 | 1.156 | 4.517 | 0.4077 | original |
| 10 | 10.69 | 0.5844 | 8.891 | 0.5844 | 0.5844 | 0.4815 | 0.3409 | original |
| 11 | 7.739 | 2.546 | 6.174 | 2.313 | 2.313 | 6.174 | 0.4501 | original |
| 12 | 7.289 | 2.708 | 5.279 | 2.223 | 2.223 | 5.279 | 0.5432 | original |
| 13 | 9.299 | 2.761 | 0.2899 | 0.6944 | 0.6944 | 0.2899 | 0.4926 | original |
| 14 | 8.007 | 7.183 | 6.861 | 2.647 | 2.647 | 6.861 | 1.053 | original |
| 15 | 6.706 | 3.035 | 3.192 | 1.518 | 1.518 | 3.192 | 0.6295 | original |
| 16 | 6.681 | 1.185 | 3.142 | 0.6369 | 0.6369 | 3.142 | 0.5718 | original |
| 40 | 7.018 | 2.212 | 3.216 | 1.386 | 1.386 | 3.216 | 0.4681 | original |
| 47 | 6.740 | 6.273 | 1.589 | 2.977 | 2.977 | 1.589 | 0.6012 | original |
| 115 | 8.075 | 7.293 | 6.725 | 1.245 | 1.245 | 6.725 | 0.4052 | original |

## Per-Sample Small-Radius Clean Residual Norm Growth

These values are the small-radius slope diagnostic for `||e(x+r v)||`. The local-outward and regularized columns are often numerically tied because regularized PGD collapses toward the clean outward-growth direction under this setting.

| sample | original growth | inc-ratio growth | resid-ratio growth | regularized growth | local-outward growth | local-residual growth | random growth | growth winner | resid winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0766 | 0.0281 | 0.0269 | 0.2059 | 0.2059 | 0.0269 | 0.00639 | regularized | resid-ratio |
| 1 | 0.0830 | 0.1733 | 0.0522 | 0.2234 | 0.2234 | 0.0522 | 0.00408 | regularized | resid-ratio |
| 2 | 0.0130 | 0.1205 | 0.0341 | 0.1706 | 0.1706 | 0.0341 | 0.000698 | regularized | resid-ratio |
| 3 | -0.00669 | 0.0668 | -0.0178 | 0.0989 | 0.0989 | -0.0178 | -0.00214 | regularized | resid-ratio |
| 4 | 0.1179 | 0.2685 | -0.000212 | 0.3061 | 0.3061 | -0.000212 | -7.57e-05 | regularized | resid-ratio |
| 5 | 0.0281 | 0.2042 | 0.0211 | 0.2281 | 0.2281 | 0.0211 | -0.00189 | regularized | resid-ratio |
| 6 | 0.0679 | 0.0844 | 0.0480 | 0.1606 | 0.1606 | 0.0480 | 0.00334 | local-outward | resid-ratio |
| 7 | 0.0494 | 0.1156 | 0.0689 | 0.1965 | 0.1965 | -0.00141 | 0.00404 | regularized | local-residual |
| 8 | 0.2177 | 0.2680 | 0.1984 | 0.3646 | 0.3646 | 0.1984 | -0.00111 | local-outward | resid-ratio |
| 9 | 0.00833 | 0.0853 | 0.0150 | 0.1167 | 0.1167 | 0.0150 | 0.00244 | regularized | resid-ratio |
| 10 | -0.2645 | 0.3678 | -0.2243 | 0.3678 | 0.3678 | 0.2820 | -0.0171 | inc-ratio | local-residual |
| 11 | 0.0825 | 0.1510 | 0.0702 | 0.1993 | 0.1993 | 0.0702 | -0.00727 | regularized | resid-ratio |
| 12 | 0.00361 | 0.2045 | -0.0441 | 0.2338 | 0.2338 | -0.0441 | 0.002 | regularized | resid-ratio |
| 13 | 0.0397 | 0.0810 | -0.0891 | 0.2014 | 0.2014 | -0.0891 | -0.0135 | regularized | resid-ratio |
| 14 | 0.0372 | 0.0459 | 0.0277 | 0.1493 | 0.1493 | 0.0277 | -0.0032 | local-outward | resid-ratio |
| 15 | 0.0140 | 0.0582 | 0.00484 | 0.1387 | 0.1387 | 0.00484 | -0.000195 | local-outward | resid-ratio |
| 16 | -0.0029 | 0.1448 | -0.0239 | 0.1740 | 0.1740 | -0.0239 | 0.00607 | regularized | resid-ratio |
| 40 | 0.00982 | 0.0774 | 0.0234 | 0.1187 | 0.1187 | 0.0234 | 0.00546 | regularized | resid-ratio |
| 47 | 0.0669 | 0.0905 | -0.0392 | 0.1852 | 0.1852 | -0.0392 | 0.00231 | regularized | resid-ratio |
| 115 | 0.00747 | 0.0125 | 0.0115 | 0.1313 | 0.1313 | 0.0115 | 0.00266 | regularized | resid-ratio |

## Per-Sample Small-Radius Residual-Increment Ratio

This is the small-radius diagnostic for residual-field movement, `||e(x+r v)-e(x)||/r`. It is different from outward growth of the residual norm.

| sample | original | inc-ratio | resid-ratio | regularized | local-outward | local-residual | random | resid winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.3858 | 0.8093 | 1.205 | 0.4060 | 0.4060 | 1.205 | 0.0463 | resid-ratio |
| 1 | 0.9646 | 0.3601 | 1.353 | 0.4863 | 0.4863 | 1.353 | 0.0483 | resid-ratio |
| 2 | 0.1903 | 0.2755 | 0.8430 | 0.3646 | 0.3646 | 0.8430 | 0.0487 | resid-ratio |
| 3 | 0.3247 | 0.1805 | 0.4043 | 0.2451 | 0.2451 | 0.4043 | 0.0370 | resid-ratio |
| 4 | 0.2466 | 0.3765 | 1.591 | 0.3942 | 0.3942 | 1.591 | 0.0527 | resid-ratio |
| 5 | 0.8999 | 0.3124 | 1.063 | 0.3482 | 0.3482 | 1.063 | 0.0240 | resid-ratio |
| 6 | 0.6040 | 0.5850 | 0.7777 | 0.3374 | 0.3375 | 0.7777 | 0.0341 | resid-ratio |
| 7 | 0.4932 | 0.6899 | 0.5153 | 0.4725 | 0.4725 | 0.8397 | 0.0722 | local-residual |
| 8 | 0.8934 | 0.9076 | 1.148 | 0.7580 | 0.7580 | 1.148 | 0.0550 | resid-ratio |
| 9 | 0.3179 | 0.3229 | 0.5620 | 0.2785 | 0.2785 | 0.5620 | 0.0444 | resid-ratio |
| 10 | 0.7165 | 0.9268 | 0.5381 | 0.9268 | 0.9268 | 1.170 | 0.0804 | local-residual |
| 11 | 1.090 | 0.3132 | 1.303 | 0.5382 | 0.5382 | 1.303 | 0.0821 | resid-ratio |
| 12 | 1.138 | 0.4091 | 1.350 | 0.4304 | 0.4304 | 1.350 | 0.0357 | resid-ratio |
| 13 | 0.4495 | 0.4301 | 1.157 | 0.6362 | 0.6362 | 1.157 | 0.0397 | resid-ratio |
| 14 | 0.7469 | 0.7834 | 0.8971 | 0.3484 | 0.3484 | 0.8971 | 0.0866 | resid-ratio |
| 15 | 0.5263 | 0.3592 | 0.6565 | 0.2640 | 0.2640 | 0.6565 | 0.0361 | resid-ratio |
| 16 | 0.5419 | 0.2699 | 0.8145 | 0.3370 | 0.3370 | 0.8145 | 0.0428 | resid-ratio |
| 40 | 0.4203 | 0.2659 | 0.6181 | 0.2899 | 0.2899 | 0.6181 | 0.0434 | resid-ratio |
| 47 | 0.7832 | 0.7177 | 0.9469 | 0.3478 | 0.3478 | 0.9469 | 0.0325 | resid-ratio |
| 115 | 0.5909 | 0.5962 | 0.7303 | 0.3291 | 0.3291 | 0.7303 | 0.0415 | resid-ratio |

## Per-Sample Direction Angles

Angles use sign-invariant comparison, so `v` and `-v` count as the same direction. Large angles between `original` and local/ratio directions are expected here: the endpoint-optimized direction is not the clean-point local direction.

| sample | inc vs local-outward | resid vs local-residual | original vs local-outward | original vs inc-ratio |
| --- | --- | --- | --- | --- |
| 0 | 82.31 deg | 1.21e-06 deg | 68.12 deg | 81.63 deg |
| 1 | 39.45 deg | 0 deg | 68.26 deg | 71.71 deg |
| 2 | 45.51 deg | 8.54e-07 deg | 85.65 deg | 83.85 deg |
| 3 | 47.82 deg | 8.54e-07 deg | 86.09 deg | 86.92 deg |
| 4 | 28.93 deg | 1.71e-06 deg | 67.35 deg | 67.18 deg |
| 5 | 26.75 deg | 0 deg | 82.80 deg | 85.80 deg |
| 6 | 58.79 deg | 1.21e-06 deg | 65.33 deg | 21.82 deg |
| 7 | 54.66 deg | 76.70 deg | 75.49 deg | 72.01 deg |
| 8 | 43.25 deg | 0 deg | 53.72 deg | 17.42 deg |
| 9 | 43.66 deg | 0 deg | 86.24 deg | 87.91 deg |
| 10 | 1.91e-06 deg | 65.19 deg | 44.28 deg | 44.28 deg |
| 11 | 41.11 deg | 0 deg | 65.37 deg | 75.00 deg |
| 12 | 29.26 deg | 0 deg | 89.10 deg | 88.86 deg |
| 13 | 66.61 deg | 8.54e-07 deg | 77.94 deg | 89.48 deg |
| 14 | 75.47 deg | 1.71e-06 deg | 77.47 deg | 28.31 deg |
| 15 | 66.02 deg | 1.21e-06 deg | 84.36 deg | 70.29 deg |
| 16 | 34.01 deg | 0 deg | 89.03 deg | 83.42 deg |
| 40 | 49.83 deg | 0 deg | 85.21 deg | 89.01 deg |
| 47 | 60.97 deg | 0 deg | 68.95 deg | 21.17 deg |
| 115 | 85.62 deg | 0 deg | 87.55 deg | 23.87 deg |

## Best Restart / Best Step Audit

This table verifies the corrected `loss3_original` comparison used best-over-steps and multiple restarts.

| sample | original restart | original step | inc-ratio restart | inc-ratio step | resid-ratio restart | resid-ratio step | regularized restart | regularized step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | boundary_from:loss3_regularized_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 1 | boundary_from:loss3_regularized_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 2 | random_boundary_00 | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 3 | boundary_from:loss3_residual_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 4 | boundary_from:loss3_regularized_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 5 | boundary_from:loss3_residual_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 6 | boundary_from:loss3_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 7 | boundary_from:loss3_residual_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 100 | local_outward_small | 0 |
| 8 | boundary_from:loss3_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 9 | random_boundary_00 | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 10 | boundary_from:loss3_residual_increment_ratio_pgd_best | 100 | local_outward_small | 0 | local_residual_small | 100 | local_outward_small | 0 |
| 11 | boundary_from:loss3_regularized_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 12 | boundary_from:loss3_residual_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 13 | boundary_from:loss3_regularized_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 14 | boundary_from:loss3_increment_ratio_pgd_best | 100 | local_outward_small | 99 | local_residual_small | 0 | local_outward_small | 0 |
| 15 | boundary_from:loss3_residual_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 16 | random_boundary_01 | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 40 | random_boundary_00 | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 47 | boundary_from:loss3_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |
| 115 | boundary_from:loss3_increment_ratio_pgd_best | 100 | local_outward_small | 100 | local_residual_small | 0 | local_outward_small | 0 |

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

![ray_profile_corrected_index_000](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_000.png)

![ray_profile_corrected_index_001](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_001.png)

![ray_profile_corrected_index_002](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_002.png)

![ray_profile_corrected_index_003](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_003.png)

![ray_profile_corrected_index_004](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_004.png)

![ray_profile_corrected_index_005](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_005.png)

![ray_profile_corrected_index_006](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_006.png)

![ray_profile_corrected_index_007](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_007.png)

![ray_profile_corrected_index_008](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_008.png)

![ray_profile_corrected_index_009](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_009.png)

![ray_profile_corrected_index_010](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_010.png)

![ray_profile_corrected_index_011](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_011.png)

![ray_profile_corrected_index_012](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_012.png)

![ray_profile_corrected_index_013](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_013.png)

![ray_profile_corrected_index_014](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_014.png)

![ray_profile_corrected_index_015](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_015.png)

![ray_profile_corrected_index_016](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_016.png)

![ray_profile_corrected_index_040](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_040.png)

![ray_profile_corrected_index_047](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_047.png)

![ray_profile_corrected_index_115](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_index_115.png)

![ray_profile_corrected_aggregate_mean](../forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/figures/ray_profile_corrected_aggregate_mean.png)

