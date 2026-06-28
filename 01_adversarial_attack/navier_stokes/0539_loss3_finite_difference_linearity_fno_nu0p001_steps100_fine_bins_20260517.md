# Loss3 Finite-Difference Local Linearity Fine-Bin Tables

Date: 2026-05-17 UTC

## What This Measures

At PGD step `k`, the path point is exactly:

`z_k = x0 + delta_k`.

For each `z_k`, direction `u`, and `rho=0.16`, the experiment evaluates three points:

`z_k - rho u`, `z_k`, `z_k + rho u`.

The scalar Loss3 linearity score is:

`C_L = |L3(z+rho u)-2L3(z)+L3(z-rho u)| / (|L3(z+rho u)-L3(z-rho u)| + eps)`.

The residual-map linearity score is:

`C_E = ||e(z+rho u)-2e(z)+e(z-rho u)|| / (||e(z+rho u)-e(z-rho u)|| + eps)`, where `e(z)=f(z)-j(z)`.

Smaller `C_L` or `C_E` means more locally linear along that direction.

## Averaging

The raw data has 5 samples, 21 saved `k` values, and directions `grad`, `radial`, `step`, `random_0..random_3`.

For each nonzero saved step, there are usually `5 samples x 7 directions = 35` rows. At `k=0`, there is no previous-step direction, so there are `5 samples x 6 directions = 30` rows.

The tables below are ordinary averages/medians over the rows in each group, not weighted by loss or gradient norm.

## Per-k Fine Table

This is the most direct time/path view. It uses all directions together.

| k | rho | n_points | delta_budget_ratio_mean | loss_linearity_score_mean | loss_linearity_score_median | residual_linearity_score_mean | residual_linearity_score_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.16 | 30 | 9.988e-07 | 0.2779 | 0.1600 | 0.0713 | 0.0413 |
| 5 | 0.16 | 35 | 0.0359 | 0.2236 | 0.0904 | 0.1145 | 0.0483 |
| 10 | 0.16 | 35 | 0.0924 | 0.9206 | 0.0525 | 0.1064 | 0.0452 |
| 15 | 0.16 | 35 | 0.1713 | 0.0786 | 0.0306 | 0.1057 | 0.0410 |
| 20 | 0.16 | 35 | 0.2687 | 0.0926 | 0.0153 | 0.0937 | 0.0390 |
| 25 | 0.16 | 35 | 0.3649 | 0.0373 | 0.0122 | 0.0957 | 0.0357 |
| 30 | 0.16 | 35 | 0.4595 | 0.0257 | 0.0096 | 0.1035 | 0.0340 |
| 35 | 0.16 | 35 | 0.5543 | 0.0330 | 0.0081 | 0.1213 | 0.0323 |
| 40 | 0.16 | 35 | 0.6319 | 0.0155 | 0.0093 | 0.1341 | 0.0328 |
| 45 | 0.16 | 35 | 0.7077 | 0.0113 | 0.0054 | 0.1285 | 0.0284 |
| 50 | 0.16 | 35 | 0.7636 | 0.0158 | 0.0042 | 0.1214 | 0.0275 |
| 55 | 0.16 | 35 | 0.8228 | 0.0289 | 0.0050 | 0.1170 | 0.0257 |
| 60 | 0.16 | 35 | 0.8769 | 0.0223 | 0.0050 | 0.1113 | 0.0327 |
| 65 | 0.16 | 35 | 0.9156 | 0.0131 | 0.0047 | 0.1102 | 0.0326 |
| 70 | 0.16 | 35 | 0.9312 | 0.0093 | 0.0031 | 0.1090 | 0.0325 |
| 75 | 0.16 | 35 | 0.9409 | 0.0104 | 0.0043 | 0.1083 | 0.0324 |
| 80 | 0.16 | 35 | 0.9506 | 0.0137 | 0.0038 | 0.1108 | 0.0323 |
| 85 | 0.16 | 35 | 0.9606 | 0.0340 | 0.0053 | 0.1156 | 0.0322 |
| 90 | 0.16 | 35 | 0.9711 | 0.0207 | 0.0049 | 0.1181 | 0.0321 |
| 95 | 0.16 | 35 | 0.9825 | 0.0141 | 0.0058 | 0.1181 | 0.0320 |
| 100 | 0.16 | 35 | 0.9948 | 0.0147 | 0.0048 | 0.1181 | 0.0319 |

## 10 Radius-Bin Table

Bins are by `delta_budget_ratio = ||delta_k||_2 / epsilon`, not by wall-clock time.

| radius_bin10 | rho | n_points | k_mean | delta_budget_ratio_mean | loss_linearity_score_mean | loss_linearity_score_median | residual_linearity_score_mean | residual_linearity_score_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0-0.1 | 0.16 | 107 | 6.2150 | 0.0444 | 0.4566 | 0.0938 | 0.0938 | 0.0456 |
| 0.1-0.2 | 0.16 | 49 | 18.5714 | 0.1665 | 0.0581 | 0.0195 | 0.0906 | 0.0406 |
| 0.2-0.3 | 0.16 | 28 | 28.7500 | 0.2516 | 0.0269 | 0.0091 | 0.0564 | 0.0319 |
| 0.3-0.4 | 0.16 | 56 | 31.2500 | 0.3444 | 0.0739 | 0.0142 | 0.0972 | 0.0325 |
| 0.4-0.5 | 0.16 | 28 | 38.7500 | 0.4637 | 0.0459 | 0.0110 | 0.1190 | 0.0280 |
| 0.5-0.6 | 0.16 | 35 | 45.0000 | 0.5413 | 0.0397 | 0.0093 | 0.1000 | 0.0247 |
| 0.6-0.7 | 0.16 | 42 | 48.3333 | 0.6517 | 0.0146 | 0.0068 | 0.1161 | 0.0270 |
| 0.7-0.8 | 0.16 | 28 | 60.0000 | 0.7468 | 0.0094 | 0.0059 | 0.1152 | 0.0262 |
| 0.8-0.9 | 0.16 | 35 | 61.0000 | 0.8374 | 0.0220 | 0.0059 | 0.1317 | 0.0341 |
| 0.9-1.0 | 0.16 | 322 | 74.8913 | 0.9942 | 0.0145 | 0.0038 | 0.1250 | 0.0332 |

## 10 Radius-Bin By Direction Table

| radius_bin10 | direction_type | rho | n_points | loss_linearity_score_mean | loss_linearity_score_median | residual_linearity_score_mean | residual_linearity_score_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0-0.1 | grad | 0.16 | 16 | 0.0795 | 0.0807 | 0.1987 | 0.1809 |
| 0.0-0.1 | radial | 0.16 | 16 | 0.0979 | 0.0545 | 0.1456 | 0.1544 |
| 0.0-0.1 | step | 0.16 | 11 | 0.0504 | 0.0462 | 0.1882 | 0.1855 |
| 0.0-0.1 | random_0 | 0.16 | 16 | 0.2119 | 0.1373 | 0.0376 | 0.0338 |
| 0.0-0.1 | random_1 | 0.16 | 16 | 0.0853 | 0.0562 | 0.0365 | 0.0359 |
| 0.0-0.1 | random_2 | 0.16 | 16 | 0.5335 | 0.3113 | 0.0377 | 0.0351 |
| 0.0-0.1 | random_3 | 0.16 | 16 | 2.0107 | 0.1405 | 0.0419 | 0.0383 |
| 0.1-0.2 | grad | 0.16 | 7 | 0.0186 | 0.0168 | 0.1686 | 0.1112 |
| 0.1-0.2 | radial | 0.16 | 7 | 0.0057 | 0.0058 | 0.1826 | 0.1778 |
| 0.1-0.2 | step | 0.16 | 7 | 0.0121 | 0.0114 | 0.1671 | 0.1306 |
| 0.1-0.2 | random_0 | 0.16 | 7 | 0.1249 | 0.0849 | 0.0271 | 0.0236 |
| 0.1-0.2 | random_1 | 0.16 | 7 | 0.0483 | 0.0140 | 0.0296 | 0.0287 |
| 0.1-0.2 | random_2 | 0.16 | 7 | 0.1212 | 0.1006 | 0.0292 | 0.0281 |
| 0.1-0.2 | random_3 | 0.16 | 7 | 0.0756 | 0.0771 | 0.0302 | 0.0279 |
| 0.2-0.3 | grad | 0.16 | 4 | 0.0069 | 0.0068 | 0.0959 | 0.0982 |
| 0.2-0.3 | radial | 0.16 | 4 | 0.0094 | 0.0088 | 0.1202 | 0.1256 |
| 0.2-0.3 | step | 0.16 | 4 | 0.0032 | 0.0026 | 0.0829 | 0.0785 |
| 0.2-0.3 | random_0 | 0.16 | 4 | 0.0566 | 0.0616 | 0.0230 | 0.0208 |
| 0.2-0.3 | random_1 | 0.16 | 4 | 0.0076 | 0.0059 | 0.0260 | 0.0246 |
| 0.2-0.3 | random_2 | 0.16 | 4 | 0.0822 | 0.0925 | 0.0228 | 0.0246 |
| 0.2-0.3 | random_3 | 0.16 | 4 | 0.0227 | 0.0205 | 0.0238 | 0.0242 |
| 0.3-0.4 | grad | 0.16 | 8 | 0.0234 | 0.0136 | 0.2442 | 0.2978 |
| 0.3-0.4 | radial | 0.16 | 8 | 0.0108 | 0.0104 | 0.1468 | 0.1285 |
| 0.3-0.4 | step | 0.16 | 8 | 0.0116 | 0.0045 | 0.2003 | 0.1839 |
| 0.3-0.4 | random_0 | 0.16 | 8 | 0.2846 | 0.0398 | 0.0208 | 0.0191 |
| 0.3-0.4 | random_1 | 0.16 | 8 | 0.0178 | 0.0165 | 0.0226 | 0.0210 |
| 0.3-0.4 | random_2 | 0.16 | 8 | 0.1419 | 0.0244 | 0.0224 | 0.0216 |
| 0.3-0.4 | random_3 | 0.16 | 8 | 0.0274 | 0.0254 | 0.0236 | 0.0203 |
| 0.4-0.5 | grad | 0.16 | 4 | 0.0088 | 0.0035 | 0.2946 | 0.3508 |
| 0.4-0.5 | radial | 0.16 | 4 | 0.0079 | 0.0087 | 0.1867 | 0.1582 |
| 0.4-0.5 | step | 0.16 | 4 | 0.0061 | 0.0049 | 0.2702 | 0.3354 |
| 0.4-0.5 | random_0 | 0.16 | 4 | 0.0896 | 0.0367 | 0.0173 | 0.0160 |
| 0.4-0.5 | random_1 | 0.16 | 4 | 0.1251 | 0.0139 | 0.0199 | 0.0178 |
| 0.4-0.5 | random_2 | 0.16 | 4 | 0.0573 | 0.0081 | 0.0191 | 0.0188 |
| 0.4-0.5 | random_3 | 0.16 | 4 | 0.0264 | 0.0224 | 0.0249 | 0.0189 |
| 0.5-0.6 | grad | 0.16 | 5 | 0.0054 | 0.0048 | 0.2201 | 0.2224 |
| 0.5-0.6 | radial | 0.16 | 5 | 0.0091 | 0.0103 | 0.1884 | 0.1926 |
| 0.5-0.6 | step | 0.16 | 5 | 0.0058 | 0.0054 | 0.2189 | 0.2339 |
| 0.5-0.6 | random_0 | 0.16 | 5 | 0.1017 | 0.0124 | 0.0163 | 0.0164 |
| 0.5-0.6 | random_1 | 0.16 | 5 | 0.0226 | 0.0171 | 0.0192 | 0.0171 |
| 0.5-0.6 | random_2 | 0.16 | 5 | 0.0383 | 0.0038 | 0.0203 | 0.0181 |
| 0.5-0.6 | random_3 | 0.16 | 5 | 0.0946 | 0.0141 | 0.0168 | 0.0169 |
| 0.6-0.7 | grad | 0.16 | 6 | 0.0032 | 0.0030 | 0.2581 | 0.3133 |
| 0.6-0.7 | radial | 0.16 | 6 | 0.0061 | 0.0059 | 0.2249 | 0.2717 |
| 0.6-0.7 | step | 0.16 | 6 | 0.0037 | 0.0032 | 0.2531 | 0.3037 |
| 0.6-0.7 | random_0 | 0.16 | 6 | 0.0296 | 0.0272 | 0.0160 | 0.0151 |
| 0.6-0.7 | random_1 | 0.16 | 6 | 0.0143 | 0.0128 | 0.0188 | 0.0169 |
| 0.6-0.7 | random_2 | 0.16 | 6 | 0.0357 | 0.0102 | 0.0208 | 0.0174 |
| 0.6-0.7 | random_3 | 0.16 | 6 | 0.0096 | 0.0104 | 0.0210 | 0.0162 |
| 0.7-0.8 | grad | 0.16 | 4 | 0.0067 | 0.0041 | 0.2768 | 0.2892 |
| 0.7-0.8 | radial | 0.16 | 4 | 0.0040 | 0.0037 | 0.1976 | 0.1822 |
| 0.7-0.8 | step | 0.16 | 4 | 0.0029 | 8.879e-04 | 0.2596 | 0.2656 |
| 0.7-0.8 | random_0 | 0.16 | 4 | 0.0212 | 0.0227 | 0.0137 | 0.0136 |
| 0.7-0.8 | random_1 | 0.16 | 4 | 0.0125 | 0.0127 | 0.0193 | 0.0168 |
| 0.7-0.8 | random_2 | 0.16 | 4 | 0.0160 | 0.0183 | 0.0173 | 0.0157 |
| 0.7-0.8 | random_3 | 0.16 | 4 | 0.0025 | 0.0025 | 0.0223 | 0.0159 |
| 0.8-0.9 | grad | 0.16 | 5 | 0.0122 | 0.0050 | 0.3150 | 0.3073 |
| 0.8-0.9 | radial | 0.16 | 5 | 0.0031 | 0.0031 | 0.2290 | 0.2955 |
| 0.8-0.9 | step | 0.16 | 5 | 0.0085 | 0.0023 | 0.2988 | 0.3036 |
| 0.8-0.9 | random_0 | 0.16 | 5 | 0.0135 | 0.0144 | 0.0187 | 0.0147 |
| 0.8-0.9 | random_1 | 0.16 | 5 | 0.0151 | 0.0145 | 0.0182 | 0.0186 |
| 0.8-0.9 | random_2 | 0.16 | 5 | 0.0176 | 0.0132 | 0.0216 | 0.0173 |
| 0.8-0.9 | random_3 | 0.16 | 5 | 0.0837 | 0.0031 | 0.0206 | 0.0156 |
| 0.9-1.0 | grad | 0.16 | 46 | 0.0027 | 0.0016 | 0.3245 | 0.3308 |
| 0.9-1.0 | radial | 0.16 | 46 | 0.0027 | 0.0024 | 0.3264 | 0.3319 |
| 0.9-1.0 | step | 0.16 | 46 | 0.0174 | 0.0111 | 0.1385 | 0.0953 |
| 0.9-1.0 | random_0 | 0.16 | 46 | 0.0066 | 0.0067 | 0.0218 | 0.0161 |
| 0.9-1.0 | random_1 | 0.16 | 46 | 0.0606 | 0.0241 | 0.0191 | 0.0189 |
| 0.9-1.0 | random_2 | 0.16 | 46 | 0.0072 | 0.0037 | 0.0227 | 0.0234 |
| 0.9-1.0 | random_3 | 0.16 | 46 | 0.0040 | 0.0030 | 0.0217 | 0.0162 |

## Takeaway

The per-k table shows why three coarse bins were too blunt. The scalar score `C_L` drops sharply after the first few saved points, with a large early outlier at `k=10`; from about `k=25` onward the median `C_L` is already very small, and the late path stays small.

The residual score `C_E` is direction-dependent. Random directions are generally small and often smaller farther out, but `grad` and `radial` directions can have larger residual bending later. So this experiment supports scalar Loss3 local linearity more strongly than full residual-map linearity.

## Output CSVs

- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_robust_by_k.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_robust_by_k_direction.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_robust_by_radius_bin10.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_robust_by_radius_bin10_direction.csv`
