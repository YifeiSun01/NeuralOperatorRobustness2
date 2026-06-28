# Loss3 Jacobian Subspace Rotation Along Path Result

Date: 2026-05-16 UTC

Scope: FNO / 1D Burgers `nu=0.001` only.

## Purpose

This is the expanded Experiment 6 diagnostic. The core question is whether the local residual-Jacobian geometry stays aligned with the clean point while we walk along the adversarial direction, or whether the dangerous local directions rotate and the local amplification changes.

Concretely, for each selected sample it walks along `x_t = x_0 + t delta*`, where `delta*` is the previously found Loss3 PGD endpoint direction. At each path point it estimates the residual Jacobian `J_e(x_t) = J_f(x_t) - J_j(x_t)`. The experiment is meant to support the bigger mechanism claim: Loss3 succeeds not only because the clean-point top singular direction is bad, but because moving outward changes the residual Jacobian itself. If the top direction, top-k subspaces, response sketch, and spectral norm all change, then a fixed clean local SVD is not enough to explain the attack geometry.

## What This Run Proves

- The top-1 right singular direction of `J_e` rotates away from the clean point quickly: the mean clean-reference angle is already about `40.55` degrees by `t=0.1` and reaches `57.64` degrees at the endpoint.
- The adjacent-step angle becomes small near the end: mean top-1 previous-reference angle drops from `40.55` degrees at `t=0.1` to `1.89` degrees at `t=1.0`. So the geometry changes sharply early, then evolves smoothly along the later part of this particular path.
- The broader top-k geometry also changes strongly. At the endpoint, the mean max principal angle to the clean top-4 subspace is `81.13` degrees and to the clean top-8 subspace is `86.79` degrees.
- The estimated spectral norm grows along the path: mean `sigma1(J_e(x_t))` goes from `0.868` at `t=0` to `8.495` at `t=1`. This means the residual amplification surface is not just rotating; it is becoming much steeper along the path.
- The random-probe response sketch changes substantially: mean relative difference from the clean response sketch reaches `6.767` at `t=1`. This is the broader-Jacobian check beyond only the top singular vector.

## Definitions

- `path_fraction`: normalized progress from `x_0` to the endpoint `x_0 + delta*`; it is not physical PDE time.
- `||t delta*||2`: actual L2 distance traveled from the clean input. The endpoint path length is about `8.0` for every sample in this run.
- `top1 clean deg`: `acos(abs(<v1(x_t), v1(x_0)>))`, where `v1` is the top right singular vector of `J_e`. The absolute value removes the arbitrary sign flip of singular vectors.
- `top1 prev deg`: same top-1 angle, but compared to the previous path point `x_{k-1}` instead of `x_0`. It is `NA` at `t=0`.
- `top-k clean/prev max/mean`: principal angles between the top-k right-singular subspaces. Each table cell below is `max principal angle / mean principal angle`, in degrees.
- `sigma1`: estimated spectral norm of `J_e(x_t)`. `sigma2/sigma1` records how dominant the first singular direction is.
- `sketch rel clean/prev`: relative difference between random-probe responses of the residual Jacobian, used as a broader Jacobian-level comparison.

## Important Caveat

This is a top-8 randomized SVD / sketch diagnostic at each off-clean path point, not a full exact 1024-rank SVD at every point. The clean reference uses the saved exact clean SVD. A full dense SVD along all 55 path points would be much more expensive; this run records the top-k structure plus random-probe response changes as the practical broader-Jacobian check.

## Settings

- Output directory: `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001`
- Delta source: `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/deltas.npz` key `loss3_original_pgd_best`
- Sample indices: `[0, 7, 40, 47, 115]`
- Path points: `11` fractions from 0 to 1: `['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']`
- Finite-difference epsilon: `0.001`
- Randomized SVD rank: `8`; oversample: `4`; probe count: `12`
- Subspace k values: `[1, 2, 4, 8]`
- Runtime seconds: `1875.6`
- GPU: `Tesla V100-SXM2-32GB`

## Aggregate By Path Fraction

| t | n | mean ||t delta*||2 | top1 clean deg | top1 prev deg | top2 clean max | top2 prev max | top4 clean max | top4 prev max | top8 clean max | top8 prev max | sigma1 | sigma2/sigma1 | sketch rel clean | sketch rel prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0 | 0 | NA | 0.005 | NA | 0.006 | NA | 0.007 | NA | 0.868 | 0.837 | 0 | NA |
| 0.1 | 5 | 0.8 | 40.552 | 40.552 | 40.08 | 40.08 | 70.229 | 70.229 | 69.829 | 69.829 | 1.973 | 0.499 | 1.566 | 1.566 |
| 0.2 | 5 | 1.6 | 54.061 | 19.436 | 87.157 | 54.19 | 66.899 | 22.059 | 73.272 | 31.652 | 3.991 | 0.341 | 2.917 | 1.883 |
| 0.3 | 5 | 2.4 | 49.612 | 24.043 | 89.165 | 13.341 | 63.441 | 24.877 | 76.997 | 35.54 | 5.809 | 0.25 | 4.015 | 1.893 |
| 0.4 | 5 | 3.2 | 50.241 | 5.651 | 78.633 | 27.451 | 73.678 | 28.013 | 78.961 | 36.451 | 6.44 | 0.254 | 4.545 | 1.394 |
| 0.5 | 5 | 4 | 51.417 | 9.046 | 78.585 | 28.674 | 79.578 | 33.895 | 83.332 | 35.743 | 7.045 | 0.299 | 5.026 | 1.31 |
| 0.6 | 5 | 4.8 | 53.795 | 7.533 | 78.633 | 12.053 | 81.158 | 16.358 | 85.498 | 30.446 | 7.641 | 0.317 | 5.524 | 1.286 |
| 0.7 | 5 | 5.6 | 55.735 | 4.22 | 78.622 | 6.692 | 81.563 | 23.579 | 86.553 | 16.531 | 7.973 | 0.33 | 5.864 | 1.239 |
| 0.8 | 5 | 6.4 | 56.546 | 2.456 | 78.516 | 13.125 | 81.221 | 25.444 | 87.064 | 11.76 | 8.158 | 0.351 | 6.144 | 1.213 |
| 0.9 | 5 | 7.2 | 57.139 | 2.072 | 78.452 | 21.132 | 81.393 | 25.091 | 86.265 | 13.072 | 8.319 | 0.402 | 6.468 | 1.194 |
| 1 | 5 | 8 | 57.642 | 1.89 | 78.468 | 10.259 | 81.128 | 21.409 | 86.786 | 13.728 | 8.495 | 0.441 | 6.767 | 1.168 |

## Per-Sample Endpoint Summary

| sample | endpoint top1 clean | endpoint top4 max | endpoint top8 max | endpoint sigma1 | endpoint sketch rel clean | max top1 clean | mean top1 prev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 34.982 | 81.256 | 86.159 | 10.171 | 6.12 | 64.759 | 15.903 |
| 7 | 88.959 | 58.202 | 88.913 | 8.466 | 7.475 | 89.669 | 14.363 |
| 40 | 88.439 | 88.887 | 85.175 | 8.633 | 6.568 | 89.154 | 16.329 |
| 47 | 29.806 | 89.57 | 87.064 | 6.11 | 5.083 | 29.806 | 4.438 |
| 115 | 46.026 | 87.728 | 86.619 | 9.095 | 8.591 | 46.026 | 7.417 |

## Full Path Rows: Top-1, Spectral Norm, Response Sketch

| sample | t | ||t delta*||2 | top1 clean deg | top1 prev deg | sigma1 | sigma2/sigma1 | sketch cos clean | sketch rel clean | sketch cos prev | sketch rel prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | NA | 1.206 | 0.871 | 1 | 0 | NA | NA |
| 0 | 0.1 | 0.8 | 3.522 | 3.522 | 1.661 | 0.76 | 0.687 | 1.036 | 0.687 | 1.036 |
| 0 | 0.2 | 1.6 | 64.759 | 66.833 | 2.493 | 0.728 | 0.41 | 1.708 | 0.542 | 1.132 |
| 0 | 0.3 | 2.4 | 37.184 | 79.301 | 7.634 | 0.306 | 0.621 | 4.114 | 0.335 | 2.373 |
| 0 | 0.4 | 3.2 | 35.133 | 3.146 | 9.053 | 0.267 | 0.186 | 5.213 | 0.261 | 1.305 |
| 0 | 0.5 | 4 | 35.123 | 1.565 | 9.366 | 0.31 | 0.152 | 5.331 | 0.356 | 1.144 |
| 0 | 0.6 | 4.8 | 35.095 | 1.211 | 9.6 | 0.359 | 0.129 | 5.441 | 0.391 | 1.113 |
| 0 | 0.7 | 5.6 | 35.01 | 1.001 | 9.802 | 0.412 | 0.08 | 5.593 | 0.421 | 1.086 |
| 0 | 0.8 | 6.4 | 34.962 | 0.881 | 9.959 | 0.469 | 0.005 | 5.764 | 0.448 | 1.06 |
| 0 | 0.9 | 7.2 | 34.962 | 0.812 | 10.077 | 0.53 | -0.081 | 5.944 | 0.472 | 1.036 |
| 0 | 1 | 8 | 34.982 | 0.756 | 10.171 | 0.589 | -0.164 | 6.12 | 0.494 | 1.015 |
| 7 | 0 | 0 | 0 | NA | 0.84 | 0.893 | 1 | 0 | NA | NA |
| 7 | 0.1 | 0.8 | 89.5 | 89.5 | 1.853 | 0.602 | 0.327 | 1.625 | 0.327 | 1.625 |
| 7 | 0.2 | 1.6 | 89.394 | 8.981 | 4.393 | 0.358 | 0.026 | 3.187 | 0.08 | 2.032 |
| 7 | 0.3 | 2.4 | 89.385 | 25.224 | 6.405 | 0.319 | 0.064 | 4.577 | -0.161 | 1.919 |
| 7 | 0.4 | 3.2 | 89.468 | 8.861 | 7.377 | 0.333 | 0.104 | 5.293 | 0.117 | 1.447 |
| 7 | 0.5 | 4 | 89.531 | 2.37 | 7.721 | 0.358 | 0.049 | 5.657 | 0.255 | 1.257 |
| 7 | 0.6 | 4.8 | 89.669 | 1.967 | 7.941 | 0.374 | 0.002 | 5.913 | 0.288 | 1.216 |
| 7 | 0.7 | 5.6 | 89.602 | 1.728 | 8.111 | 0.388 | -0.021 | 6.107 | 0.302 | 1.199 |
| 7 | 0.8 | 6.4 | 89.285 | 1.634 | 8.251 | 0.459 | -0.036 | 6.481 | 0.326 | 1.197 |
| 7 | 0.9 | 7.2 | 89.282 | 1.787 | 8.353 | 0.635 | -0.084 | 7.08 | 0.386 | 1.159 |
| 7 | 1 | 8 | 88.959 | 1.578 | 8.466 | 0.723 | -0.129 | 7.475 | 0.43 | 1.096 |
| 40 | 0 | 0 | 0 | NA | 0.617 | 0.911 | 1 | 0 | NA | NA |
| 40 | 0.1 | 0.8 | 89.154 | 89.154 | 1.117 | 0.53 | 0.515 | 1.181 | 0.515 | 1.181 |
| 40 | 0.2 | 1.6 | 87.72 | 10.201 | 3.114 | 0.251 | 0.223 | 2.713 | 0.079 | 2.232 |
| 40 | 0.3 | 2.4 | 87.835 | 8.439 | 4.257 | 0.217 | 0.153 | 3.702 | -0.233 | 1.858 |
| 40 | 0.4 | 3.2 | 87.96 | 8.585 | 4.704 | 0.215 | 0.158 | 4.083 | 0.06 | 1.447 |
| 40 | 0.5 | 4 | 88.357 | 28.253 | 6.51 | 0.327 | 0.118 | 5.208 | 0.194 | 1.455 |
| 40 | 0.6 | 4.8 | 88.473 | 11.735 | 7.645 | 0.279 | 0.155 | 5.832 | 0.198 | 1.352 |
| 40 | 0.7 | 5.6 | 88.458 | 2.021 | 7.873 | 0.236 | 0.112 | 5.995 | 0.227 | 1.256 |
| 40 | 0.8 | 6.4 | 88.443 | 1.645 | 8.094 | 0.195 | 0.102 | 6.113 | 0.226 | 1.256 |
| 40 | 0.9 | 7.2 | 88.439 | 1.685 | 8.365 | 0.211 | 0.098 | 6.3 | 0.239 | 1.252 |
| 40 | 1 | 8 | 88.439 | 1.575 | 8.633 | 0.274 | 0.088 | 6.568 | 0.272 | 1.232 |
| 47 | 0 | 0 | 0 | NA | 0.949 | 0.774 | 1 | 0 | NA | NA |
| 47 | 0.1 | 0.8 | 4.964 | 4.964 | 2.24 | 0.32 | 0.495 | 1.317 | 0.495 | 1.317 |
| 47 | 0.2 | 1.6 | 8.517 | 6.102 | 4.581 | 0.174 | 0.296 | 2.431 | 0.148 | 1.845 |
| 47 | 0.3 | 2.4 | 12.184 | 4.049 | 5.139 | 0.189 | 0.249 | 2.803 | -0.183 | 1.646 |
| 47 | 0.4 | 3.2 | 15.742 | 4.264 | 5.301 | 0.216 | 0.215 | 3.013 | 0.095 | 1.39 |
| 47 | 0.5 | 4 | 18.857 | 4.335 | 5.476 | 0.246 | 0.17 | 3.268 | 0.213 | 1.3 |
| 47 | 0.6 | 4.8 | 21.632 | 4.446 | 5.626 | 0.285 | 0.125 | 3.58 | 0.263 | 1.267 |
| 47 | 0.7 | 5.6 | 24.278 | 4.602 | 5.743 | 0.323 | 0.093 | 3.955 | 0.301 | 1.244 |
| 47 | 0.8 | 6.4 | 26.524 | 4.377 | 5.861 | 0.358 | 0.066 | 4.375 | 0.318 | 1.231 |
| 47 | 0.9 | 7.2 | 28.264 | 3.789 | 5.989 | 0.39 | 0.054 | 4.758 | 0.325 | 1.215 |
| 47 | 1 | 8 | 29.806 | 3.448 | 6.11 | 0.419 | 0.048 | 5.083 | 0.331 | 1.198 |
| 115 | 0 | 0 | 0 | NA | 0.73 | 0.736 | 1 | 0 | NA | NA |
| 115 | 0.1 | 0.8 | 15.621 | 15.621 | 2.994 | 0.285 | 0.15 | 2.67 | 0.15 | 2.67 |
| 115 | 0.2 | 1.6 | 19.915 | 5.061 | 5.371 | 0.195 | 0.112 | 4.548 | -0.213 | 2.175 |
| 115 | 0.3 | 2.4 | 21.473 | 3.201 | 5.611 | 0.22 | 0.093 | 4.877 | -0.301 | 1.67 |
| 115 | 0.4 | 3.2 | 22.902 | 3.4 | 5.767 | 0.239 | 0.082 | 5.122 | 0.091 | 1.382 |
| 115 | 0.5 | 4 | 25.217 | 8.706 | 6.153 | 0.255 | 0.053 | 5.667 | 0.128 | 1.39 |
| 115 | 0.6 | 4.8 | 34.106 | 18.309 | 7.392 | 0.287 | 0.067 | 6.854 | 0.118 | 1.48 |
| 115 | 0.7 | 5.6 | 41.327 | 11.747 | 8.335 | 0.289 | 0.069 | 7.67 | 0.12 | 1.409 |
| 115 | 0.8 | 6.4 | 43.516 | 3.742 | 8.624 | 0.272 | 0.04 | 7.986 | 0.159 | 1.321 |
| 115 | 0.9 | 7.2 | 44.75 | 2.288 | 8.811 | 0.243 | -0.004 | 8.26 | 0.173 | 1.305 |
| 115 | 1 | 8 | 46.026 | 2.09 | 9.095 | 0.198 | 0.016 | 8.591 | 0.193 | 1.299 |

## Full Path Rows: Top-k Subspace Angles

Each angle cell is `max principal angle / mean principal angle`, in degrees.

| sample | t | k=2 clean max/mean | k=2 prev max/mean | k=4 clean max/mean | k=4 prev max/mean | k=8 clean max/mean | k=8 prev max/mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 / 0 | NA / NA | 0 / 0 | NA / NA | 0.01 / 0 | NA / NA |
| 0 | 0.1 | 4.16 / 3.83 | 4.16 / 3.83 | 87.15 / 29.5 | 87.15 / 29.5 | 47.38 / 18.66 | 47.38 / 18.66 |
| 0 | 0.2 | 86.8 / 47.89 | 86.91 / 46.93 | 87.43 / 29.55 | 16.73 / 8.4 | 58.41 / 23.64 | 33.2 / 11.33 |
| 0 | 0.3 | 89.27 / 49.52 | 10.46 / 9.35 | 89.45 / 31.23 | 9.55 / 6.09 | 77.38 / 28 | 52.08 / 11.18 |
| 0 | 0.4 | 35.11 / 23.08 | 89.87 / 46.08 | 89.83 / 32.31 | 5.36 / 3.13 | 78.91 / 29.34 | 15.08 / 4.85 |
| 0 | 0.5 | 35.1 / 23.86 | 1.85 / 1.71 | 89.4 / 33.13 | 3.95 / 2.57 | 80.32 / 30.16 | 12.28 / 4.25 |
| 0 | 0.6 | 35.07 / 24.59 | 1.73 / 1.47 | 88.23 / 33.75 | 4.07 / 2.34 | 83.67 / 34.41 | 40.3 / 7.27 |
| 0 | 0.7 | 34.99 / 25.23 | 1.6 / 1.3 | 86.68 / 34.27 | 4.73 / 2.31 | 84.56 / 35.98 | 18.64 / 4.56 |
| 0 | 0.8 | 34.94 / 25.88 | 1.55 / 1.21 | 82.63 / 34.16 | 8.06 / 2.99 | 87.65 / 36.87 | 8.38 / 2.98 |
| 0 | 0.9 | 34.93 / 26.53 | 1.46 / 1.13 | 82.78 / 39.71 | 77.17 / 20.32 | 84.96 / 36.44 | 13.23 / 3.48 |
| 0 | 1 | 34.95 / 27.14 | 1.36 / 1.05 | 81.26 / 39.95 | 8.19 / 2.98 | 86.16 / 36.91 | 6.59 / 2.61 |
| 7 | 0 | 0 / 0 | NA / NA | 0 / 0 | NA / NA | 0.01 / 0 | NA / NA |
| 7 | 0.1 | 89.45 / 46.96 | 89.45 / 46.96 | 84.34 / 25.55 | 84.34 / 25.55 | 65.67 / 21.19 | 65.67 / 21.19 |
| 7 | 0.2 | 89.42 / 49.27 | 9.01 / 7.56 | 48.42 / 18.47 | 39.35 / 12.53 | 58.94 / 23.16 | 32.85 / 10.62 |
| 7 | 0.3 | 89.36 / 51.43 | 25.23 / 15.02 | 18.52 / 12.84 | 37.38 / 11.73 | 64.82 / 25.05 | 14.56 / 6.14 |
| 7 | 0.4 | 89.43 / 53.13 | 8.88 / 6.26 | 30.7 / 17.35 | 20.95 / 7.02 | 68.39 / 27.45 | 59.7 / 10.56 |
| 7 | 0.5 | 89.5 / 54.34 | 2.69 / 2.52 | 48.49 / 22.83 | 20.6 / 6.68 | 74.71 / 29.58 | 20.24 / 5.12 |
| 7 | 0.6 | 89.65 / 55.07 | 4.36 / 3.16 | 53.26 / 25.1 | 7.02 / 3.68 | 80.6 / 31.41 | 14.62 / 4.39 |
| 7 | 0.7 | 89.6 / 55.11 | 7.05 / 4.39 | 55.17 / 27.67 | 9.73 / 4.52 | 84.52 / 33.51 | 11.76 / 4.83 |
| 7 | 0.8 | 89.35 / 55.91 | 31.91 / 16.76 | 56.41 / 26.82 | 11.97 / 4.91 | 86.73 / 34.67 | 12.45 / 4.69 |
| 7 | 0.9 | 89.5 / 54.53 | 19.12 / 10.45 | 57.34 / 27.66 | 5.41 / 3.15 | 88 / 35.57 | 11.55 / 4.19 |
| 7 | 1 | 89.35 / 53.95 | 4.64 / 3.1 | 58.2 / 28.55 | 3.99 / 2.32 | 88.91 / 35.85 | 11.3 / 3.8 |
| 40 | 0 | 0.01 / 0.01 | NA / NA | 0.01 / 0 | NA / NA | 0.01 / 0 | NA / NA |
| 40 | 0.1 | 11.81 / 8.83 | 11.81 / 8.83 | 89.8 / 28.57 | 89.8 / 28.57 | 70.61 / 25.66 | 70.61 / 25.66 |
| 40 | 0.2 | 85.78 / 49.63 | 83.9 / 45.31 | 86.64 / 30.23 | 25.6 / 10.59 | 86.44 / 32.2 | 44.37 / 13.46 |
| 40 | 0.3 | 88.61 / 54.24 | 11.08 / 9.11 | 84.93 / 31.69 | 10.74 / 7.2 | 88.89 / 37.82 | 46.27 / 14.34 |
| 40 | 0.4 | 89.69 / 57.71 | 15.7 / 10.62 | 84.71 / 33.86 | 20 / 8.6 | 89.45 / 39.63 | 48.19 / 11.62 |
| 40 | 0.5 | 89.95 / 58.95 | 71.27 / 37.34 | 87.54 / 50.18 | 77.7 / 22.77 | 88.52 / 38.08 | 28.24 / 8.35 |
| 40 | 0.6 | 89.93 / 59.56 | 4.73 / 3.76 | 86.96 / 49.4 | 9.71 / 4.49 | 86.62 / 38.66 | 14.2 / 5.16 |
| 40 | 0.7 | 89.98 / 59.03 | 3.97 / 2.75 | 88.6 / 51.54 | 60.85 / 17.87 | 84.85 / 39.97 | 15.83 / 5.28 |
| 40 | 0.8 | 89.93 / 56.62 | 23.17 / 12.27 | 89.73 / 51.65 | 17.61 / 6.91 | 83.45 / 41.42 | 13.07 / 4.94 |
| 40 | 0.9 | 89.65 / 70.67 | 76.97 / 39.18 | 89.36 / 49.05 | 16.79 / 6.07 | 83.26 / 43.29 | 15.2 / 5.46 |
| 40 | 1 | 89.62 / 71.46 | 9.13 / 5.17 | 88.89 / 37.02 | 69.22 / 18.91 | 85.17 / 44.87 | 15.46 / 5.59 |
| 47 | 0 | 0.01 / 0 | NA / NA | 0.01 / 0 | NA / NA | 0.01 / 0 | NA / NA |
| 47 | 0.1 | 5.9 / 5.11 | 5.9 / 5.11 | 50.96 / 16.59 | 50.96 / 16.59 | 84.68 / 21.64 | 84.68 / 21.64 |
| 47 | 0.2 | 84.17 / 46.17 | 80.27 / 42.18 | 64.05 / 24.01 | 17.26 / 9.15 | 86.04 / 23.97 | 18.24 / 7.38 |
| 47 | 0.3 | 88.76 / 50.29 | 12.04 / 7.99 | 72.49 / 36.81 | 50.47 / 16.65 | 81.11 / 25.92 | 21.39 / 7 |
| 47 | 0.4 | 89.05 / 52.31 | 15.3 / 9.77 | 87.17 / 44.02 | 32.82 / 12.13 | 81.21 / 29.45 | 28.18 / 7.64 |
| 47 | 0.5 | 88.75 / 53.79 | 47.45 / 25.87 | 88.67 / 45.59 | 9.14 / 4.24 | 85.98 / 32.64 | 41.58 / 9.52 |
| 47 | 0.6 | 89.15 / 55.39 | 13.34 / 8.82 | 89.22 / 46.71 | 8.4 / 3.57 | 89.18 / 35.12 | 23.79 / 7.05 |
| 47 | 0.7 | 89.26 / 56.76 | 4.78 / 3.82 | 89.4 / 47.78 | 7.97 / 3.22 | 89.33 / 37.23 | 17.06 / 5.94 |
| 47 | 0.8 | 89.22 / 57.87 | 4.4 / 2.83 | 89.45 / 48.8 | 7.32 / 2.95 | 88.26 / 37.54 | 11.87 / 4.75 |
| 47 | 0.9 | 89.19 / 58.73 | 3.79 / 2.24 | 89.52 / 49.71 | 6.88 / 2.74 | 87.67 / 37.81 | 12.94 / 4.26 |
| 47 | 1 | 89.22 / 59.51 | 3.45 / 2 | 89.57 / 50.56 | 6.75 / 2.64 | 87.06 / 38.74 | 18.2 / 5.03 |
| 115 | 0 | 0.01 / 0 | NA / NA | 0.01 / 0 | NA / NA | 0.01 / 0 | NA / NA |
| 115 | 0.1 | 89.08 / 49.79 | 89.08 / 49.79 | 38.9 / 15.4 | 38.9 / 15.4 | 80.81 / 28.07 | 80.81 / 28.07 |
| 115 | 0.2 | 89.62 / 52.26 | 10.86 / 7.19 | 47.95 / 19.92 | 11.35 / 6.92 | 76.52 / 31.11 | 29.59 / 9.62 |
| 115 | 0.3 | 89.82 / 53.68 | 7.9 / 5.52 | 51.82 / 24.96 | 16.26 / 8.26 | 72.78 / 34.25 | 43.39 / 13.11 |
| 115 | 0.4 | 89.88 / 54.84 | 7.51 / 5.42 | 75.97 / 38.22 | 60.94 / 18.69 | 76.85 / 38.59 | 31.11 / 13.28 |
| 115 | 0.5 | 89.62 / 57.01 | 20.11 / 11.71 | 83.78 / 43.58 | 58.08 / 19.64 | 87.14 / 47.7 | 76.38 / 24.01 |
| 115 | 0.6 | 89.36 / 60.31 | 36.1 / 19.51 | 88.13 / 48.93 | 52.59 / 16.43 | 87.41 / 49.21 | 59.31 / 15.62 |
| 115 | 0.7 | 89.28 / 61.15 | 16.06 / 9.14 | 87.96 / 54.77 | 34.61 / 11.8 | 89.5 / 49.84 | 19.37 / 7.52 |
| 115 | 0.8 | 89.14 / 61.55 | 4.6 / 3.34 | 87.88 / 55.38 | 82.26 / 23.29 | 89.23 / 50.34 | 13.04 / 5.03 |
| 115 | 0.9 | 88.99 / 62.19 | 4.32 / 3.13 | 87.96 / 52.1 | 19.21 / 7.75 | 87.43 / 50.36 | 12.44 / 4.86 |
| 115 | 1 | 89.21 / 67.28 | 32.72 / 17.25 | 87.73 / 56.4 | 18.89 / 7.63 | 86.62 / 50.47 | 17.09 / 5.61 |

## Singular Values By Sample And Path Point

| sample | t | ||t delta*||2 | sigma1 | sigma2 | sigma3 | sigma4 | sigma5 | sigma6 | sigma7 | sigma8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1.2058 | 1.05 | 0.487 | 0.3125 | 0.3064 | 0.3012 | 0.2664 | 0.207 |
| 0 | 0.1 | 0.8 | 1.6614 | 1.2623 | 1.103 | 0.5897 | 0.3072 | 0.282 | 0.1956 | 0.1767 |
| 0 | 0.2 | 1.6 | 2.4929 | 1.8146 | 1.5927 | 0.9379 | 0.3439 | 0.3134 | 0.223 | 0.1986 |
| 0 | 0.3 | 2.4 | 7.6341 | 2.3355 | 1.9783 | 1.0301 | 0.3333 | 0.275 | 0.2343 | 0.1912 |
| 0 | 0.4 | 3.2 | 9.0531 | 2.4135 | 2.0311 | 1.0033 | 0.3375 | 0.2942 | 0.248 | 0.169 |
| 0 | 0.5 | 4 | 9.3664 | 2.9008 | 2.0592 | 0.8731 | 0.3548 | 0.3289 | 0.2443 | 0.1631 |
| 0 | 0.6 | 4.8 | 9.6 | 3.4418 | 2.1393 | 0.7342 | 0.3831 | 0.3574 | 0.2374 | 0.1556 |
| 0 | 0.7 | 5.6 | 9.8025 | 4.0358 | 2.2134 | 0.6241 | 0.4237 | 0.3909 | 0.2335 | 0.1695 |
| 0 | 0.8 | 6.4 | 9.9591 | 4.6728 | 2.2777 | 0.5477 | 0.4692 | 0.414 | 0.2307 | 0.1796 |
| 0 | 0.9 | 7.2 | 10.0768 | 5.3365 | 2.3331 | 0.5158 | 0.5077 | 0.4222 | 0.2279 | 0.1895 |
| 0 | 1 | 8 | 10.1706 | 5.9939 | 2.3795 | 0.5615 | 0.494 | 0.3907 | 0.2251 | 0.194 |
| 7 | 0 | 0 | 0.8402 | 0.7502 | 0.5526 | 0.3635 | 0.3057 | 0.2771 | 0.2559 | 0.2337 |
| 7 | 0.1 | 0.8 | 1.8533 | 1.1148 | 0.8943 | 0.5676 | 0.3855 | 0.3408 | 0.2112 | 0.1798 |
| 7 | 0.2 | 1.6 | 4.3935 | 1.5708 | 0.9991 | 0.9532 | 0.6509 | 0.4856 | 0.2382 | 0.1849 |
| 7 | 0.3 | 2.4 | 6.4047 | 2.0434 | 1.7167 | 1.0973 | 0.9555 | 0.6028 | 0.287 | 0.1751 |
| 7 | 0.4 | 3.2 | 7.3767 | 2.4545 | 1.5341 | 1.1862 | 1.0358 | 0.6714 | 0.3398 | 0.1901 |
| 7 | 0.5 | 4 | 7.7206 | 2.7633 | 1.5989 | 1.2551 | 0.8269 | 0.7017 | 0.3966 | 0.2255 |
| 7 | 0.6 | 4.8 | 7.941 | 2.9667 | 1.7271 | 1.2816 | 0.7205 | 0.6417 | 0.4413 | 0.2629 |
| 7 | 0.7 | 5.6 | 8.1113 | 3.1459 | 1.8427 | 1.3281 | 0.7106 | 0.5631 | 0.4622 | 0.2903 |
| 7 | 0.8 | 6.4 | 8.2505 | 3.7908 | 2.0309 | 1.9452 | 0.7844 | 0.6011 | 0.4454 | 0.305 |
| 7 | 0.9 | 7.2 | 8.3532 | 5.3036 | 2.0472 | 1.8158 | 0.8184 | 0.6823 | 0.4413 | 0.3182 |
| 7 | 1 | 8 | 8.466 | 6.1195 | 2.1406 | 1.6049 | 0.8779 | 0.779 | 0.4503 | 0.3285 |
| 40 | 0 | 0 | 0.6168 | 0.5616 | 0.4373 | 0.2825 | 0.2294 | 0.211 | 0.202 | 0.183 |
| 40 | 0.1 | 0.8 | 1.1167 | 0.5919 | 0.3939 | 0.3101 | 0.2558 | 0.1724 | 0.1588 | 0.1301 |
| 40 | 0.2 | 1.6 | 3.1142 | 0.7812 | 0.647 | 0.4188 | 0.2602 | 0.1669 | 0.1528 | 0.1367 |
| 40 | 0.3 | 2.4 | 4.257 | 0.9232 | 0.7223 | 0.4872 | 0.3336 | 0.2666 | 0.1656 | 0.153 |
| 40 | 0.4 | 3.2 | 4.7036 | 1.011 | 0.8002 | 0.5816 | 0.5006 | 0.2685 | 0.2447 | 0.1686 |
| 40 | 0.5 | 4 | 6.5097 | 2.1312 | 0.9931 | 0.8639 | 0.6578 | 0.4408 | 0.2754 | 0.1666 |
| 40 | 0.6 | 4.8 | 7.6452 | 2.1319 | 1.0316 | 0.9221 | 0.803 | 0.4104 | 0.2878 | 0.1633 |
| 40 | 0.7 | 5.6 | 7.8733 | 1.8577 | 1.1547 | 1.0189 | 0.9627 | 0.3533 | 0.3017 | 0.1629 |
| 40 | 0.8 | 6.4 | 8.0942 | 1.5746 | 1.3328 | 1.26 | 1.0092 | 0.3247 | 0.2957 | 0.1632 |
| 40 | 0.9 | 7.2 | 8.3648 | 1.7665 | 1.6144 | 1.0863 | 1.0408 | 0.353 | 0.2511 | 0.1632 |
| 40 | 1 | 8 | 8.6329 | 2.365 | 1.7773 | 1.0753 | 0.8368 | 0.389 | 0.2303 | 0.1628 |
| 47 | 0 | 0 | 0.9487 | 0.7342 | 0.4914 | 0.2947 | 0.2644 | 0.2389 | 0.2043 | 0.1854 |
| 47 | 0.1 | 0.8 | 2.2403 | 0.7172 | 0.5068 | 0.5001 | 0.2726 | 0.2044 | 0.179 | 0.1322 |
| 47 | 0.2 | 1.6 | 4.5811 | 0.7973 | 0.6904 | 0.6165 | 0.4589 | 0.2049 | 0.1896 | 0.1324 |
| 47 | 0.3 | 2.4 | 5.1391 | 0.9706 | 0.805 | 0.6513 | 0.6111 | 0.2172 | 0.1979 | 0.1358 |
| 47 | 0.4 | 3.2 | 5.3007 | 1.1455 | 1.0543 | 0.7776 | 0.522 | 0.2662 | 0.1995 | 0.1356 |
| 47 | 0.5 | 4 | 5.4757 | 1.3481 | 1.2517 | 0.9111 | 0.3941 | 0.3431 | 0.2091 | 0.1443 |
| 47 | 0.6 | 4.8 | 5.6256 | 1.6045 | 1.3648 | 1.0463 | 0.4411 | 0.2693 | 0.2149 | 0.1677 |
| 47 | 0.7 | 5.6 | 5.7432 | 1.8563 | 1.454 | 1.1789 | 0.505 | 0.25 | 0.2217 | 0.1886 |
| 47 | 0.8 | 6.4 | 5.8611 | 2.1001 | 1.5169 | 1.3136 | 0.5274 | 0.3513 | 0.2349 | 0.2021 |
| 47 | 0.9 | 7.2 | 5.9891 | 2.3352 | 1.5544 | 1.4491 | 0.5424 | 0.5178 | 0.2519 | 0.2123 |
| 47 | 1 | 8 | 6.1101 | 2.5602 | 1.5946 | 1.5687 | 0.7493 | 0.5237 | 0.2736 | 0.2159 |
| 115 | 0 | 0 | 0.7297 | 0.5367 | 0.3315 | 0.3025 | 0.2083 | 0.2015 | 0.1897 | 0.186 |
| 115 | 0.1 | 0.8 | 2.9936 | 0.8528 | 0.4468 | 0.3133 | 0.1784 | 0.1488 | 0.147 | 0.1437 |
| 115 | 0.2 | 1.6 | 5.3712 | 1.0474 | 0.3691 | 0.3466 | 0.1759 | 0.1654 | 0.1455 | 0.143 |
| 115 | 0.3 | 2.4 | 5.6106 | 1.2334 | 0.3982 | 0.2906 | 0.2205 | 0.1616 | 0.149 | 0.1442 |
| 115 | 0.4 | 3.2 | 5.7671 | 1.3774 | 0.4379 | 0.2941 | 0.2582 | 0.1904 | 0.1552 | 0.1422 |
| 115 | 0.5 | 4 | 6.1529 | 1.5714 | 0.6367 | 0.445 | 0.2977 | 0.2555 | 0.2182 | 0.1641 |
| 115 | 0.6 | 4.8 | 7.3919 | 2.1218 | 0.9755 | 0.4896 | 0.4507 | 0.3674 | 0.3294 | 0.2328 |
| 115 | 0.7 | 5.6 | 8.3352 | 2.4102 | 1.0954 | 0.5591 | 0.519 | 0.4981 | 0.4366 | 0.3083 |
| 115 | 0.8 | 6.4 | 8.6243 | 2.3454 | 1.1888 | 0.6463 | 0.6243 | 0.5524 | 0.4649 | 0.3756 |
| 115 | 0.9 | 7.2 | 8.8107 | 2.1368 | 1.3448 | 0.8074 | 0.7789 | 0.5699 | 0.4618 | 0.3579 |
| 115 | 1 | 8 | 9.0952 | 1.7991 | 1.5266 | 0.9873 | 0.9701 | 0.5982 | 0.4388 | 0.3086 |

## Output Files

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/jacobian_subspace_rotation_by_sample_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/jacobian_subspace_rotation_aggregate_by_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/jacobian_subspace_rotation_summary_by_sample.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/singular_values_by_sample_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/manifest.json`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/figures/top1_angle_to_clean_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/figures/spectral_norm_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/figures/response_sketch_rel_diff_to_clean_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/figures/top1_subspace_clean_max_angle_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/figures/top2_subspace_clean_max_angle_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/figures/top4_subspace_clean_max_angle_vs_t.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/figures/top8_subspace_clean_max_angle_vs_t.png`

## Interpretation

This run gives the evidence the original Experiment 6 needed: along the Loss3 direction, the residual Jacobian does not keep the same local orientation as the clean point. The top-1 direction changes, the top-k subspaces change, and the spectral norm grows by almost an order of magnitude on these samples. The clean-point Jacobian is therefore a starting snapshot, not the whole explanation of the attack path.

The adjacent-angle columns are useful because they separate two effects: large accumulated rotation from `x_0`, and local smoothness from one path point to the next. Here the accumulated clean-reference rotation is large, while the adjacent angle becomes small near the endpoint, suggesting the path enters a different but locally coherent high-amplification region.
