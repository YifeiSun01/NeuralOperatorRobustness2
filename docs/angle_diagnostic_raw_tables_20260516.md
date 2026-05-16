# Angle Diagnostic Raw Tables

Date: 2026-05-16

This note records the every-5-step angle data requested for the FNO `nu=0.001`
trajectory diagnostics. It separates the different angle families so they are not mixed.

## Source Files

Observed from:

- True nonlinear summary: `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- True nonlinear per-index raw rows: `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv`
- Local-affine summary: `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`
- Local-affine per-index raw rows: `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_diagnostics.csv`
- Non-trajectory candidate-direction angles: `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`
- Trajectory source root:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`

The trajectory files contain saved `k` values:

```text
0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
```

## How Many Angle Families Were Computed?

There are four angle families in the current records.

1. **Local candidate-direction angles**, not indexed by attack step:

\[
\angle\left(\frac{A^Tb}{\|A^Tb\|_2},\ v_i(A)\right),
\]

where \(v_i(A)\) are top right singular directions of \(A=J_f-J_j\). These live in
`all_direction_similarity_table.csv`; they are not `k=0,5,...` trajectory angles.

2. **Same-delta local-affine endpoint-vs-movement angle**:

\[
\angle\left(A^Tb+A^TA\delta_k,\ A^TA\delta_k\right).
\]

This is a clean-point local-affine/Taylor-model angle.

3. **Same-delta local-affine bias-vs-movement angle**:

\[
\angle\left(A^Tb,\ A^TA\delta_k\right).
\]

This records how the clean-residual term itself aligns with the local movement term.

4. **True nonlinear same-point endpoint-vs-movement angle**:

\[
\angle\left(
\nabla_{z_k}\|f(z_k)-j(z_k)\|_2,
\nabla_{z_k}\|(f(z_k)-j(z_k))-(f(x)-j(x))\|_2
\right),
\qquad z_k=x+\delta_k.
\]

This is the finite-saved-point autograd comparison and is the fairest comparison for
non-infinitesimal \(\delta_k\).

## Important Note About k=0

`k=0` exists in the saved trajectories, but the saved perturbation is not exactly zero:
its mean L2 norm is about `8.0046e-06`, or budget ratio `1.0006e-06`. Therefore the
`k=0` angles below should be read as near-zero initialization diagnostics. At exactly
\(\delta=0\), movement-style gradients can be degenerate or numerically unstable.

## Interpretation Correction: The Column Names Are Delta Sources, Not Angle Objectives

The `loss1`, `loss2`, and `loss3` columns in the angle tables should be read as
**delta-source labels**: `delta from loss1 trajectory`, `delta from loss2 trajectory`,
and `delta from loss3 trajectory`. They should not be read as `loss1 angle`,
`loss2 angle`, or `loss3 angle`.

For the true nonlinear endpoint-vs-movement table, the compared gradients are both
defined from the `loss3` residual field:

\[
\nabla_{z_k}\|f(z_k)-j(z_k)\|_2
\quad\text{versus}\quad
\nabla_{z_k}\|(f(z_k)-j(z_k))-(f(x)-j(x))\|_2.
\]

Therefore:

- the `loss3` column is the only column whose trajectory label matches the two
  `loss3`-related gradients being compared;
- the `loss1` column means: take \(\delta_k\) from the `loss1_original_pgd`
  trajectory, then compute this `loss3` endpoint-vs-movement angle at that point;
- the `loss2` column means the same thing using \(\delta_k\) from the
  `loss2_original_pgd` trajectory;
- these columns must not be described as angles for the `loss1` or `loss2`
  objectives themselves, because `loss1` does not contain the perturbed oracle
  \(j(x+\delta)\), and the formulas in these tables explicitly use \(J_f-J_j\)
  or `loss3` residual-field gradients.

The same caveat applies to the local-affine tables. Their formulas use
\(A=J_f-J_j\) and \(b=f(x)-j(x)\), so the reported numbers are angles from the
local `loss3` residual geometry, evaluated at \(\delta_k\) values saved from
different attack trajectories. They are not `loss1` or `loss2` angle formulas.
In particular, if a formula contains \(J_j\), \(J_f-J_j\), or the residual
\(f-j\), it should not be interpreted as a native `loss1` diagnostic. These are
angle tables, not loss-growth tables.

The bias-vs-movement table is especially limited:

\[
\angle\left(A^Tb,\ A^TA\delta_k\right).
\]

It is useful only as a decomposition check inside the clean-point affine model:
it tells whether the clean-residual outward term and the local movement term are
aligned. It is not a finite-radius attack objective, and by itself it should not
be used as evidence that one attack is better than another.


## Clarification: These Tables Are Not Pairwise loss1/loss2/loss3 Gradient Angles

A cleaner pairwise-gradient experiment would fix the same point \(\delta_k\) and compute
angles among the actual objective gradients:

\[
\angle(\nabla_\delta L_1(\delta_k),\nabla_\delta L_2(\delta_k)),\quad
\angle(\nabla_\delta L_1(\delta_k),\nabla_\delta L_3(\delta_k)),\quad
\angle(\nabla_\delta L_2(\delta_k),\nabla_\delta L_3(\delta_k)).
\]

That would directly answer whether the `loss1_original`, `loss2_original`, and
`loss3_original` update directions differ at the same \(\delta_k\).

The current true nonlinear endpoint-vs-movement table does **not** do that. It fixes a
saved point \(z_k=x+\delta_k\) from a trajectory and computes the angle between two
`loss3`-related gradients:

\[
\nabla_{z_k}\|f(z_k)-j(z_k)\|_2
\quad\text{and}\quad
\nabla_{z_k}\|(f(z_k)-j(z_k))-(f(x)-j(x))\|_2.
\]

So its direct meaning is: at this saved point, how different is the local update
direction for the `loss3_original` endpoint error from the local update direction for
residual-field movement? It supports a claim about endpoint-error geometry versus
movement/rank-style geometry, especially along the `loss3` trajectory. It should not be
presented as a direct pairwise comparison of `loss1`, `loss2`, and `loss3` gradients.

## Raw Summary Table 1: True Nonlinear Endpoint-vs-Movement Angle

Metric: `endpoint_movement_angle_deg` from `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`.

| k | loss1 mean | loss1 std | loss1 min | loss1 max | loss2 mean | loss2 std | loss2 min | loss2 max | loss3 mean | loss3 std | loss3 min | loss3 max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 85.3498 | 8.7552 | 72.9460 | 100.1667 | 85.3498 | 8.7552 | 72.9460 | 100.1667 | 85.3498 | 8.7552 | 72.9460 | 100.1667 |
| 5 | 13.4841 | 15.6031 | 0.7697 | 40.0477 | 37.3297 | 26.2570 | 3.0369 | 83.1190 | 45.0634 | 10.7231 | 25.3069 | 56.0404 |
| 10 | 7.9088 | 8.9697 | 0.7244 | 22.4943 | 6.3010 | 6.6992 | 0.5647 | 18.0925 | 36.3602 | 17.2274 | 5.2001 | 56.0812 |
| 15 | 4.8338 | 5.4338 | 0.2867 | 13.9848 | 2.3134 | 3.2755 | 0.4538 | 8.8400 | 28.0486 | 19.9863 | 3.9601 | 52.7146 |
| 20 | 2.1962 | 2.7784 | 0.2819 | 7.7174 | 2.1501 | 3.3821 | 0.2862 | 8.9074 | 23.6289 | 17.9346 | 2.5928 | 43.9144 |
| 25 | 1.4815 | 2.0732 | 0.2824 | 5.6232 | 1.6723 | 2.1745 | 0.2995 | 5.9712 | 22.3301 | 18.8516 | 1.4564 | 48.0007 |
| 30 | 1.5683 | 2.2748 | 0.2999 | 6.1131 | 1.4151 | 1.6891 | 0.3219 | 4.7311 | 18.8716 | 16.0922 | 1.4421 | 38.4415 |
| 35 | 1.0238 | 1.2314 | 0.3028 | 3.4769 | 1.1767 | 1.2689 | 0.3369 | 3.6367 | 12.9802 | 11.9746 | 1.5775 | 34.7261 |
| 40 | 0.7072 | 0.6248 | 0.2934 | 1.9336 | 1.0109 | 0.9987 | 0.3266 | 2.9172 | 9.2389 | 12.0246 | 1.6663 | 33.2190 |
| 45 | 0.5845 | 0.3780 | 0.2803 | 1.2998 | 0.9006 | 0.8380 | 0.2896 | 2.4781 | 8.4512 | 11.7608 | 1.3403 | 31.9201 |
| 50 | 0.5471 | 0.2606 | 0.2682 | 1.0162 | 0.8173 | 0.7280 | 0.2614 | 2.1713 | 8.1530 | 11.3178 | 1.2159 | 30.7008 |

## Raw Summary Table 2: Local-Affine Endpoint-vs-Movement Angle

Metric: `endpoint_vs_movement_angle_deg` from `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`.

| k | loss1 mean | loss1 std | loss1 min | loss1 max | loss2 mean | loss2 std | loss2 min | loss2 max | loss3 mean | loss3 std | loss3 min | loss3 max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 96.9639 | 18.0143 | 74.5121 | 118.7086 | 96.9639 | 18.0143 | 74.5121 | 118.7086 | 96.9639 | 18.0143 | 74.5121 | 118.7086 |
| 5 | 1.9477 | 0.7029 | 1.0111 | 2.9257 | 1.9740 | 0.6340 | 1.0739 | 2.7345 | 21.0709 | 4.4774 | 13.9904 | 27.0605 |
| 10 | 1.2066 | 0.4389 | 0.6131 | 1.7686 | 1.1874 | 0.3728 | 0.6492 | 1.6774 | 11.2759 | 4.1013 | 4.5304 | 17.2177 |
| 15 | 0.9041 | 0.3238 | 0.4552 | 1.2778 | 0.8745 | 0.2726 | 0.4758 | 1.2789 | 7.1926 | 3.5364 | 2.2294 | 12.0965 |
| 20 | 0.7372 | 0.2517 | 0.3897 | 1.0514 | 0.7132 | 0.2193 | 0.3957 | 1.0476 | 5.2761 | 3.0548 | 1.3530 | 9.1438 |
| 25 | 0.6764 | 0.2344 | 0.3894 | 0.9814 | 0.6350 | 0.1771 | 0.3950 | 0.9085 | 4.2205 | 2.5673 | 1.0120 | 7.3050 |
| 30 | 0.6787 | 0.2364 | 0.3883 | 0.9787 | 0.6425 | 0.1855 | 0.3933 | 0.9389 | 3.5139 | 2.2097 | 0.8272 | 6.0880 |
| 35 | 0.6856 | 0.2369 | 0.3872 | 0.9787 | 0.6492 | 0.1919 | 0.3913 | 0.9622 | 2.8682 | 1.9169 | 0.7034 | 5.2403 |
| 40 | 0.6926 | 0.2376 | 0.3864 | 0.9804 | 0.6552 | 0.1974 | 0.3895 | 0.9822 | 2.1705 | 1.5353 | 0.6313 | 4.6229 |
| 45 | 0.6995 | 0.2383 | 0.3858 | 0.9831 | 0.6608 | 0.2024 | 0.3879 | 0.9996 | 1.6440 | 1.3137 | 0.5465 | 4.1555 |
| 50 | 0.7063 | 0.2391 | 0.3853 | 0.9861 | 0.6659 | 0.2067 | 0.3865 | 1.0148 | 1.4039 | 1.2072 | 0.5385 | 3.7891 |

## Raw Summary Table 3: Local-Affine Bias-vs-Movement Angle

Metric: `bias_vs_movement_angle_deg` from `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`.

| k | loss1 mean | loss1 std | loss1 min | loss1 max | loss2 mean | loss2 std | loss2 min | loss2 max | loss3 mean | loss3 std | loss3 min | loss3 max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 96.9642 | 18.0142 | 74.5124 | 118.7087 | 96.9642 | 18.0142 | 74.5124 | 118.7087 | 96.9642 | 18.0142 | 74.5124 | 118.7087 |
| 5 | 93.8192 | 10.5095 | 81.0627 | 108.5001 | 98.9407 | 11.0602 | 78.7519 | 112.1912 | 48.6056 | 7.7641 | 35.3030 | 58.4840 |
| 10 | 93.7032 | 10.3696 | 79.8926 | 107.6902 | 98.9187 | 10.9367 | 79.2059 | 112.4912 | 49.0891 | 10.4090 | 31.1601 | 62.4803 |
| 15 | 93.5175 | 10.0123 | 79.4535 | 106.5535 | 98.9042 | 10.8177 | 79.5776 | 112.6367 | 50.4990 | 13.0140 | 29.0480 | 68.1290 |
| 20 | 93.2498 | 9.6715 | 79.2012 | 105.2594 | 99.1277 | 10.8304 | 79.8356 | 112.7689 | 53.0114 | 14.6241 | 29.7810 | 71.5791 |
| 25 | 93.0409 | 9.4779 | 78.9987 | 104.3727 | 99.3824 | 10.8889 | 80.0101 | 112.8730 | 55.2984 | 15.0990 | 32.9446 | 72.9748 |
| 30 | 92.8808 | 9.3514 | 78.8280 | 103.7289 | 99.5692 | 10.9345 | 80.1410 | 112.9356 | 57.8002 | 14.8307 | 39.0338 | 75.4480 |
| 35 | 92.7637 | 9.2828 | 78.6405 | 103.2854 | 99.7023 | 10.9724 | 80.2498 | 113.0335 | 61.5018 | 14.7244 | 44.3261 | 83.0306 |
| 40 | 92.6845 | 9.2492 | 78.4594 | 102.9852 | 99.8022 | 11.0010 | 80.3455 | 113.1283 | 65.8951 | 15.2271 | 44.4215 | 88.9191 |
| 45 | 92.6345 | 9.2356 | 78.2934 | 102.7780 | 99.8826 | 11.0231 | 80.4319 | 113.2081 | 68.5819 | 16.5000 | 44.5616 | 91.7458 |
| 50 | 92.6061 | 9.2330 | 78.1452 | 102.6287 | 99.9496 | 11.0403 | 80.5108 | 113.2701 | 69.5856 | 17.1836 | 44.7273 | 93.0726 |


## Count Summary And Mean/Std Tables

The step-indexed angle diagnostics contain **three** angle families, not four:

1. local-affine endpoint-vs-movement angle;
2. local-affine bias-vs-movement angle;
3. true nonlinear endpoint-vs-movement angle.

Each of these has:

- `11` saved steps: `k = 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`;
- `3` attack trajectories: `loss1_original_pgd`, `loss2_original_pgd`, `loss3_original_pgd`;
- `5` samples per mean/std aggregation: indices `0, 7, 40, 47, 115`.

So the step-indexed angle tables contain:

```text
3 angle families x 11 steps x 3 trajectories = 99 mean/std entries
```

The non-step angle family is different:

\[
\angle\left(\frac{A^Tb}{\|A^Tb\|_2},\ v_i(A)\right).
\]

It compares the clean-point outward-growth direction with the saved top singular
vectors of \(A=J_f-J_j\). It is **not** a per-step optimizer direction. It has no
`k=0,5,...` axis unless one recomputes \(A\), \(b\), and the SVD at every current
point \(x+\delta_k\), which was not what this experiment did.

The entries below are `mean (std)` in degrees, aggregated over the five samples.

### Mean/Std Table A: True Nonlinear Endpoint-vs-Movement Angle

| k | loss1 | loss2 | loss3 |
| ---: | ---: | ---: | ---: |
| 0 | 85.35 (8.76) | 85.35 (8.76) | 85.35 (8.76) |
| 5 | 13.48 (15.60) | 37.33 (26.26) | 45.06 (10.72) |
| 10 | 7.91 (8.97) | 6.30 (6.70) | 36.36 (17.23) |
| 15 | 4.83 (5.43) | 2.31 (3.28) | 28.05 (19.99) |
| 20 | 2.20 (2.78) | 2.15 (3.38) | 23.63 (17.93) |
| 25 | 1.48 (2.07) | 1.67 (2.17) | 22.33 (18.85) |
| 30 | 1.57 (2.27) | 1.42 (1.69) | 18.87 (16.09) |
| 35 | 1.02 (1.23) | 1.18 (1.27) | 12.98 (11.97) |
| 40 | 0.71 (0.62) | 1.01 (1.00) | 9.24 (12.02) |
| 45 | 0.58 (0.38) | 0.90 (0.84) | 8.45 (11.76) |
| 50 | 0.55 (0.26) | 0.82 (0.73) | 8.15 (11.32) |

### Mean/Std Table B: Local-Affine Endpoint-vs-Movement Angle

| k | loss1 | loss2 | loss3 |
| ---: | ---: | ---: | ---: |
| 0 | 96.96 (18.01) | 96.96 (18.01) | 96.96 (18.01) |
| 5 | 1.95 (0.70) | 1.97 (0.63) | 21.07 (4.48) |
| 10 | 1.21 (0.44) | 1.19 (0.37) | 11.28 (4.10) |
| 15 | 0.90 (0.32) | 0.87 (0.27) | 7.19 (3.54) |
| 20 | 0.74 (0.25) | 0.71 (0.22) | 5.28 (3.05) |
| 25 | 0.68 (0.23) | 0.63 (0.18) | 4.22 (2.57) |
| 30 | 0.68 (0.24) | 0.64 (0.19) | 3.51 (2.21) |
| 35 | 0.69 (0.24) | 0.65 (0.19) | 2.87 (1.92) |
| 40 | 0.69 (0.24) | 0.66 (0.20) | 2.17 (1.54) |
| 45 | 0.70 (0.24) | 0.66 (0.20) | 1.64 (1.31) |
| 50 | 0.71 (0.24) | 0.67 (0.21) | 1.40 (1.21) |

### Mean/Std Table C: Local-Affine Bias-vs-Movement Angle

| k | loss1 | loss2 | loss3 |
| ---: | ---: | ---: | ---: |
| 0 | 96.96 (18.01) | 96.96 (18.01) | 96.96 (18.01) |
| 5 | 93.82 (10.51) | 98.94 (11.06) | 48.61 (7.76) |
| 10 | 93.70 (10.37) | 98.92 (10.94) | 49.09 (10.41) |
| 15 | 93.52 (10.01) | 98.90 (10.82) | 50.50 (13.01) |
| 20 | 93.25 (9.67) | 99.13 (10.83) | 53.01 (14.62) |
| 25 | 93.04 (9.48) | 99.38 (10.89) | 55.30 (15.10) |
| 30 | 92.88 (9.35) | 99.57 (10.93) | 57.80 (14.83) |
| 35 | 92.76 (9.28) | 99.70 (10.97) | 61.50 (14.72) |
| 40 | 92.68 (9.25) | 99.80 (11.00) | 65.90 (15.23) |
| 45 | 92.63 (9.24) | 99.88 (11.02) | 68.58 (16.50) |
| 50 | 92.61 (9.23) | 99.95 (11.04) | 69.59 (17.18) |

## Short Readout

Observed evidence:

- The true nonlinear table now includes `k=0` and every 5 steps through `k=50`.
- The local-affine table already included `k=0` and every 5 steps through `k=50`.
- For `loss3`, the local-affine endpoint-vs-movement angle decreases from `21.0709 deg`
  at `k=5` to `1.4039 deg` at `k=50`, while the true nonlinear endpoint-vs-movement
  angle decreases from `45.0634 deg` at `k=5` to `8.1530 deg` at `k=50`.

Inference:

- The clean-point local-affine model understates the true nonlinear gradient-angle gap.
- The gap is largest early/mid trajectory and shrinks later, but the true nonlinear
  angle remains larger than the local-affine angle across the saved `loss3` path.
