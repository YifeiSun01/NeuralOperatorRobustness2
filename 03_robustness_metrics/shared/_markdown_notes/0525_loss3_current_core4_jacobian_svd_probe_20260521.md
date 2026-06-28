# Loss3 Current Core4/PQ Jacobian SVD Probe - 2026-05-21

Status: completed targeted GPU explicit residual-Jacobian/SVD probe on current core4/PQ states.

This is not the old PGD-only path diagnostic. It uses current core4/PQ deltas and selected perturbed states.

## Scope

- Output directory: `forensics/loss3_current_core4_jacobian_svd_probe_20260521`
- Sample index: `0`
- Probe states: `11`
- GPU runtime recorded in: `forensics/loss3_current_core4_jacobian_svd_probe_20260521/manifest.json`

## Main Tables

- State SVD summary: `forensics/loss3_current_core4_jacobian_svd_probe_20260521/tables/state_svd_summary.csv`
- Top direction alignment: `forensics/loss3_current_core4_jacobian_svd_probe_20260521/tables/top_direction_alignment.csv`
- Top vector metrics: `forensics/loss3_current_core4_jacobian_svd_probe_20260521/tables/top_vector_frequency_metrics.csv`

## Residual-Jacobian Spectral Summary

| pq | state_label | sigma1 | sigma2 | sigma1_over_sigma2 | top1_energy_fraction | top4_energy_fraction | top1_peakiness | top1_energy_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p2q2 | clean | 1.206 | 1.05 | 1.148 | 0.4038 | 0.803 | 3.928 | 0.004474 |
| p2q2 | steepest_replace_step1 | 4.68 | 3.252 | 1.439 | 0.5961 | 0.956 | 3.396 | 0.003761 |
| p2q2 | steepest_replace_step5 | 10.17 | 1.997 | 5.094 | 0.9245 | 0.9939 | 2.566 | 0.002804 |
| p2q2 | steepest_replace_step10 | 8.856 | 2.223 | 3.984 | 0.8752 | 0.9891 | 2.513 | 0.002733 |
| p2q2 | steepest_replace_final | 8.864 | 2.242 | 3.955 | 0.8751 | 0.9891 | 2.504 | 0.002722 |
| p2q2 | steepest_add_final | 8.344 | 1.616 | 5.163 | 0.9178 | 0.9853 | 3.228 | 0.00372 |
| p2qinf | clean | 1.206 | 1.05 | 1.148 | 0.4038 | 0.803 | 3.928 | 0.004474 |
| p2qinf | steepest_replace_final | 3.893 | 1.03 | 3.781 | 0.8299 | 0.9591 | 2.866 | 0.003629 |
| p2qinf | steepest_add_final | 7.097 | 1.719 | 4.127 | 0.892 | 0.98 | 3.484 | 0.00375 |
| p1qinf | clean | 1.206 | 1.05 | 1.148 | 0.4038 | 0.803 | 3.928 | 0.004474 |
| p1qinf | steepest_replace_final | 1.177 | 0.9532 | 1.235 | 0.4031 | 0.7789 | 3.97 | 0.00451 |

## Top Residual Singular Direction Alignment Snapshot

| pq | state_label | direction_label | abs_cos_top1_right_vs_direction | angle_deg_top1_right_vs_direction |
| --- | --- | --- | --- | --- |
| p1qinf | clean | raw_add_final | 0.2861 | 73.37 |
| p1qinf | clean | steepest_add_final | 0.003957 | 89.77 |
| p1qinf | clean | steepest_replace_final | 0.006199 | 89.64 |
| p1qinf | clean | steepest_replace_step5 | 0.006199 | 89.64 |
| p1qinf | steepest_replace_final | raw_add_final | 0.2783 | 73.84 |
| p1qinf | steepest_replace_final | steepest_add_final | 0.008299 | 89.52 |
| p1qinf | steepest_replace_final | steepest_replace_final | 0.0003424 | 89.98 |
| p1qinf | steepest_replace_final | steepest_replace_step5 | 0.0003424 | 89.98 |
| p2q2 | clean | raw_add_final | 0.02948 | 88.31 |
| p2q2 | clean | steepest_add_final | 0.02965 | 88.3 |
| p2q2 | clean | steepest_replace_final | 0.1086 | 83.77 |
| p2q2 | clean | steepest_replace_step5 | 0.3737 | 68.06 |
| p2q2 | steepest_add_final | raw_add_final | 0.6261 | 51.24 |
| p2q2 | steepest_add_final | steepest_add_final | 0.6248 | 51.33 |
| p2q2 | steepest_add_final | steepest_replace_final | 0.000145 | 89.99 |
| p2q2 | steepest_add_final | steepest_replace_step5 | 6.068e-05 | 90 |
| p2q2 | steepest_replace_final | raw_add_final | 0.01654 | 89.05 |
| p2q2 | steepest_replace_final | steepest_add_final | 0.01672 | 89.04 |
| p2q2 | steepest_replace_final | steepest_replace_final | 0.2813 | 73.66 |
| p2q2 | steepest_replace_final | steepest_replace_step5 | 0.1777 | 79.76 |
| p2q2 | steepest_replace_step1 | raw_add_final | 0.005329 | 89.69 |
| p2q2 | steepest_replace_step1 | steepest_add_final | 0.005561 | 89.68 |
| p2q2 | steepest_replace_step1 | steepest_replace_final | 0.172 | 80.1 |
| p2q2 | steepest_replace_step1 | steepest_replace_step5 | 0.3249 | 71.04 |
| p2q2 | steepest_replace_step10 | raw_add_final | 0.01619 | 89.07 |
| p2q2 | steepest_replace_step10 | steepest_add_final | 0.01637 | 89.06 |
| p2q2 | steepest_replace_step10 | steepest_replace_final | 0.2766 | 73.94 |
| p2q2 | steepest_replace_step10 | steepest_replace_step5 | 0.1748 | 79.94 |
| p2q2 | steepest_replace_step5 | raw_add_final | 0.01143 | 89.34 |
| p2q2 | steepest_replace_step5 | steepest_add_final | 0.01162 | 89.33 |
| p2q2 | steepest_replace_step5 | steepest_replace_final | 0.1926 | 78.89 |
| p2q2 | steepest_replace_step5 | steepest_replace_step5 | 0.1245 | 82.85 |
| p2qinf | clean | raw_add_final | 0.106 | 83.91 |
| p2qinf | clean | steepest_add_final | 0.07253 | 85.84 |
| p2qinf | clean | steepest_replace_final | 0.8289 | 34.02 |
| p2qinf | clean | steepest_replace_step5 | 0.04909 | 87.19 |
| p2qinf | steepest_add_final | raw_add_final | 0.01031 | 89.41 |
| p2qinf | steepest_add_final | steepest_add_final | 0.02704 | 88.45 |
| p2qinf | steepest_add_final | steepest_replace_final | 0.8701 | 29.54 |
| p2qinf | steepest_add_final | steepest_replace_step5 | 0.0315 | 88.2 |
| p2qinf | steepest_replace_final | raw_add_final | 0.1105 | 83.65 |
| p2qinf | steepest_replace_final | steepest_add_final | 0.07664 | 85.6 |
| p2qinf | steepest_replace_final | steepest_replace_final | 0.7453 | 41.81 |
| p2qinf | steepest_replace_final | steepest_replace_step5 | 0.5239 | 58.4 |

## Interpretation Guide

- A large `sigma1_over_sigma2` or large `top1_energy_fraction` supports a dominant-mode story.
- A high cosine between the residual-Jacobian top right singular vector and a final/early delta supports the idea that the optimizer rapidly follows a locally dominant direction.
- Low cosine or weak spectral gap means the previous GPI explanation should stay empirical/local-surrogate rather than being upgraded to a strict spectral theorem.
- High `top1_peakiness` or `top1_energy_concentration` supports the spike/concentration concern in p=1 or q=inf geometries.
