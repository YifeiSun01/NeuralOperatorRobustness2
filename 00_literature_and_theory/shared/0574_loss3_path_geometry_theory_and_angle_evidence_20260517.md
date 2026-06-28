# Loss3 Path Geometry: Theory, Angle Definitions, And Experiment Results

Date: 2026-05-17 UTC

Scope: FNO / 1D Burgers `nu=0.001`; samples `0, 7, 40, 47, 115`.

## Executive Claim

The earlier PGD-trajectory experiment and the straight-line Jacobian experiment are related but they measure different mathematical objects. The trajectory experiment measures gradient-angle differences at the same attack point. The straight-line experiment measures how the residual Jacobian singular geometry changes across path points. Together they support the same qualitative mechanism: near the clean point the geometry changes quickly; farther along the adversarial direction, adjacent local geometry becomes more stable. This does not mean the endpoint geometry returns to the clean-point linearization. It means the path appears to leave the clean local geometry and enter a different, more stable high-amplification region.

## Data Sources

- `/workspace/NeuralOperatorRobustness2/results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary_by_k.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_diagnostics.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/jacobian_subspace_rotation_aggregate_by_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/jacobian_subspace_rotation_by_sample_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/jacobian_subspace_rotation_summary_by_sample.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/singular_values_by_sample_t.csv`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/manifest.json`

## Common Notation

Let the neural operator be $f$, the numerical solver be $j$, and the residual be

$$e(z)=f(z)-j(z).$$

The clean input is $x_0$. A PGD trajectory point is

$$z_k=x_0+\delta_k,$$

with clean residual

$$b=e(x_0),$$

residual increment

$$r_k=e(z_k)-e(x_0)=e(z_k)-b,$$

and current residual Jacobian

$$J_k=D e(z_k).$$

## Angle Family 1: True Nonlinear Endpoint-Vs-Movement Gradient Angle

This is the most faithful finite-point gradient comparison from the earlier trajectory run. It compares two true autograd gradients at the same point $z_k$.

Endpoint loss:

$$L_{\mathrm{end}}(z)=\|e(z)\|_2.$$

Movement loss:

$$L_{\mathrm{mov}}(z)=\|e(z)-e(x_0)\|_2.$$

The gradients at $z_k$ are

$$g_{\mathrm{end}}^{\mathrm{NL}}(k)=\nabla_z L_{\mathrm{end}}(z_k)=J_k^\top\frac{e(z_k)}{\|e(z_k)\|_2}=J_k^\top\frac{b+r_k}{\|b+r_k\|_2},$$

$$g_{\mathrm{mov}}^{\mathrm{NL}}(k)=\nabla_z L_{\mathrm{mov}}(z_k)=J_k^\top\frac{e(z_k)-e(x_0)}{\|e(z_k)-e(x_0)\|_2}=J_k^\top\frac{r_k}{\|r_k\|_2}.$$

The recorded angle is

$$\theta_{\mathrm{NL}}(k)=\arccos\frac{\langle g_{\mathrm{end}}^{\mathrm{NL}}(k),g_{\mathrm{mov}}^{\mathrm{NL}}(k)\rangle}{\|g_{\mathrm{end}}^{\mathrm{NL}}(k)\|_2\|g_{\mathrm{mov}}^{\mathrm{NL}}(k)\|_2}.$$

This angle is not $\angle(g(k),g(k-5))$. It asks: at the same point $z_k$, how different are the endpoint-error gradient and the residual-movement gradient?

Mechanism: since $e(z_k)=b+r_k$, the clean residual $b$ can bend the endpoint gradient early. As $\|r_k\|$ grows relative to $\|b\|$, the normalized directions $(b+r_k)/\|b+r_k\|$ and $r_k/\|r_k\|$ become closer, so the two gradients tend to align if the pullback by $J_k^\top$ does not undo that alignment.

### True Nonlinear Every-5-Step Mean/Std Angles

Entries are `mean (std)` degrees over five samples.

| k | loss1 path | loss2 path | loss3 path |
| --- | --- | --- | --- |
| 0 | 85.35 (8.76) | 85.35 (8.76) | 85.35 (8.76) |
| 5 | 13.48 (15.6) | 37.33 (26.26) | 45.06 (10.72) |
| 10 | 7.91 (8.97) | 6.3 (6.7) | 36.36 (17.23) |
| 15 | 4.83 (5.43) | 2.31 (3.28) | 28.05 (19.99) |
| 20 | 2.2 (2.78) | 2.15 (3.38) | 23.63 (17.93) |
| 25 | 1.48 (2.07) | 1.67 (2.17) | 22.33 (18.85) |
| 30 | 1.57 (2.27) | 1.42 (1.69) | 18.87 (16.09) |
| 35 | 1.02 (1.23) | 1.18 (1.27) | 12.98 (11.97) |
| 40 | 0.71 (0.62) | 1.01 (1) | 9.24 (12.02) |
| 45 | 0.58 (0.38) | 0.9 (0.84) | 8.45 (11.76) |
| 50 | 0.55 (0.26) | 0.82 (0.73) | 8.15 (11.32) |

### True Nonlinear Loss3-Path Detailed Means

| k | budget ratio | endpoint loss | movement loss | angle deg | cos | mov/end grad norm |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.282 | 0 | 85.35 | 0.0804 | 1.207 |
| 5 | 0.0359 | 0.344 | 0.124 | 45.06 | 0.6933 | 2.707 |
| 10 | 0.0924 | 0.544 | 0.392 | 36.36 | 0.7673 | 1.883 |
| 15 | 0.1713 | 0.935 | 0.817 | 28.05 | 0.829 | 1.4 |
| 20 | 0.2687 | 1.569 | 1.454 | 23.63 | 0.8714 | 1.271 |
| 25 | 0.3649 | 2.112 | 2.015 | 22.33 | 0.8758 | 1.277 |
| 30 | 0.4595 | 2.613 | 2.538 | 18.87 | 0.9094 | 1.238 |
| 35 | 0.5543 | 3.133 | 3.068 | 12.98 | 0.9537 | 1.073 |
| 40 | 0.6319 | 3.624 | 3.571 | 9.24 | 0.9659 | 1.073 |
| 45 | 0.7077 | 4.2 | 4.156 | 8.45 | 0.9688 | 1.055 |
| 50 | 0.7637 | 4.617 | 4.573 | 8.15 | 0.971 | 1.049 |

## Angle Family 2: Clean-Point Local-Affine Endpoint-Vs-Movement Angle

This is the clean-point linearized proxy for Angle Family 1. It freezes the residual Jacobian at the clean point:

$$A=D e(x_0).$$

The local-affine approximation is

$$e(x_0+\delta)\approx b+A\delta.$$

The squared endpoint proxy is

$$\widehat L_{\mathrm{end}}(\delta)=\frac{1}{2}\|b+A\delta\|_2^2,$$

with gradient

$$\widehat g_{\mathrm{end}}(k)=\nabla_\delta \widehat L_{\mathrm{end}}(\delta_k)=A^\top(b+A\delta_k)=A^\top b+A^\top A\delta_k.$$

The squared movement proxy is

$$\widehat L_{\mathrm{mov}}(\delta)=\frac{1}{2}\|A\delta\|_2^2,$$

with gradient

$$\widehat g_{\mathrm{mov}}(k)=\nabla_\delta \widehat L_{\mathrm{mov}}(\delta_k)=A^\top A\delta_k.$$

The recorded local-affine angle is

$$\theta_{\mathrm{aff}}(k)=\arccos\frac{\langle A^\top b+A^\top A\delta_k,A^\top A\delta_k\rangle}{\|A^\top b+A^\top A\delta_k\|_2\|A^\top A\delta_k\|_2}.$$

This asks: inside the clean-point linear model, how much does the fixed bias term $A^\top b$ bend the endpoint gradient away from the movement gradient? It is not a finite-point nonlinear gradient and it is not an adjacent-step angle.

### Local-Affine Endpoint-Vs-Movement Every-5-Step Mean/Std Angles

Entries are `mean (std)` degrees over five samples.

| k | loss1 path | loss2 path | loss3 path |
| --- | --- | --- | --- |
| 0 | 96.96 (18.01) | 96.96 (18.01) | 96.96 (18.01) |
| 5 | 1.95 (0.7) | 1.97 (0.63) | 21.07 (4.48) |
| 10 | 1.21 (0.44) | 1.19 (0.37) | 11.28 (4.1) |
| 15 | 0.9 (0.32) | 0.87 (0.27) | 7.19 (3.54) |
| 20 | 0.74 (0.25) | 0.71 (0.22) | 5.28 (3.05) |
| 25 | 0.68 (0.23) | 0.63 (0.18) | 4.22 (2.57) |
| 30 | 0.68 (0.24) | 0.64 (0.19) | 3.51 (2.21) |
| 35 | 0.69 (0.24) | 0.65 (0.19) | 2.87 (1.92) |
| 40 | 0.69 (0.24) | 0.66 (0.2) | 2.17 (1.54) |
| 45 | 0.7 (0.24) | 0.66 (0.2) | 1.64 (1.31) |
| 50 | 0.71 (0.24) | 0.67 (0.21) | 1.4 (1.21) |

### Local-Affine Loss3-Path Detailed Means

| k | budget ratio | end-vs-mov angle | end-vs-mov cos | bias-vs-mov angle | ||A^T b|| | ||A^T A delta|| | bias/mov norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 96.96 | -0.1144 | 96.96 | 0.048 | 0 | 451358.6023 |
| 5 | 0.0359 | 21.07 | 0.9303 | 48.61 | 0.048 | 0.0712 | 0.8561 |
| 10 | 0.0924 | 11.28 | 0.9782 | 49.09 | 0.048 | 0.2145 | 0.3738 |
| 15 | 0.1713 | 7.19 | 0.9902 | 50.5 | 0.048 | 0.4537 | 0.2256 |
| 20 | 0.2687 | 5.28 | 0.9943 | 53.01 | 0.048 | 0.8123 | 0.1565 |
| 25 | 0.3649 | 4.22 | 0.9963 | 55.3 | 0.048 | 1.165 | 0.1175 |
| 30 | 0.4595 | 3.51 | 0.9974 | 57.8 | 0.048 | 1.511 | 0.0909 |
| 35 | 0.5543 | 2.87 | 0.9982 | 61.5 | 0.048 | 1.9065 | 0.0684 |
| 40 | 0.6319 | 2.17 | 0.9989 | 65.9 | 0.048 | 2.3231 | 0.049 |
| 45 | 0.7077 | 1.64 | 0.9993 | 68.58 | 0.048 | 2.7757 | 0.0374 |
| 50 | 0.7637 | 1.4 | 0.9995 | 69.59 | 0.048 | 3.0885 | 0.0321 |

## Angle Family 3: Clean-Point Local-Affine Bias-Vs-Movement Angle

This auxiliary angle compares the two terms inside the clean-point linear model:

$$A^\top b \quad \text{and} \quad A^\top A\delta_k.$$

The angle is

$$\theta_{\mathrm{bias}}(k)=\arccos\frac{\langle A^\top b,A^\top A\delta_k\rangle}{\|A^\top b\|_2\|A^\top A\delta_k\|_2}.$$

This is not a true attack gradient and not a nonlinear geometry measurement. It is only useful for interpreting the clean-point Taylor model.

### Local-Affine Bias-Vs-Movement Every-5-Step Mean/Std Angles

| k | loss1 path | loss2 path | loss3 path |
| --- | --- | --- | --- |
| 0 | 96.96 (18.01) | 96.96 (18.01) | 96.96 (18.01) |
| 5 | 93.82 (10.51) | 98.94 (11.06) | 48.61 (7.76) |
| 10 | 93.7 (10.37) | 98.92 (10.94) | 49.09 (10.41) |
| 15 | 93.52 (10.01) | 98.9 (10.82) | 50.5 (13.01) |
| 20 | 93.25 (9.67) | 99.13 (10.83) | 53.01 (14.62) |
| 25 | 93.04 (9.48) | 99.38 (10.89) | 55.3 (15.1) |
| 30 | 92.88 (9.35) | 99.57 (10.93) | 57.8 (14.83) |
| 35 | 92.76 (9.28) | 99.7 (10.97) | 61.5 (14.72) |
| 40 | 92.68 (9.25) | 99.8 (11) | 65.9 (15.23) |
| 45 | 92.63 (9.24) | 99.88 (11.02) | 68.58 (16.5) |
| 50 | 92.61 (9.23) | 99.95 (11.04) | 69.59 (17.18) |

## Angle Family 4: Three-Loss Gradient Angles Along The PGD Trajectory

This is a separate diagnostic. At the same point $z_k$, it compares gradients of three different objectives:

$$g_i(k)=\nabla_z L_i(z_k),\qquad i=1,2,3.$$

The pairwise angle is

$$\theta_{ij}(k)=\arccos\frac{\langle g_i(k),g_j(k)\rangle}{\|g_i(k)\|_2\|g_j(k)\|_2}.$$

This answers which direction different losses want to move at the same point. It does not directly measure nonlinearity or adjacent-step rotation.

### Three-Loss Gradient Every-5-Step Means

| k | n | angle g1,g2 | angle g1,g3 | angle g2,g3 | budget ratio | mean loss1 | mean loss2 | mean loss3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 15 | 12.2 | 74.24 | 76.43 | 0.225 | 3.369 | 3.445 | 0.471 |
| 10 | 15 | 5.34 | 68.47 | 69.07 | 0.387 | 4.889 | 4.938 | 0.949 |
| 15 | 15 | 3.87 | 62.81 | 63.21 | 0.543 | 6.232 | 6.275 | 1.641 |
| 20 | 15 | 3.14 | 60.43 | 61.01 | 0.688 | 7.349 | 7.391 | 2.416 |
| 25 | 15 | 2.93 | 54.48 | 55.09 | 0.785 | 7.977 | 8.015 | 2.992 |
| 30 | 15 | 3.03 | 52.82 | 53.63 | 0.82 | 8.169 | 8.2 | 3.245 |
| 35 | 15 | 2.2 | 49.89 | 50.47 | 0.851 | 8.39 | 8.415 | 3.478 |
| 40 | 15 | 2.16 | 47.93 | 48.5 | 0.877 | 8.645 | 8.666 | 3.697 |
| 45 | 15 | 2.14 | 47.19 | 47.68 | 0.903 | 8.892 | 8.91 | 3.941 |
| 50 | 15 | 2.08 | 46.81 | 47.23 | 0.921 | 9.065 | 9.081 | 4.128 |

## Angle Family 5: Straight-Line Residual-Jacobian Rotation

This is the newer Experiment 6 run. It walks on a fixed straight line

$$x_t=x_0+t\delta^\star,$$

where $\delta^\star$ is the Loss3 PGD endpoint perturbation. It estimates the residual Jacobian

$$J(t)=D e(x_t),$$

and its right singular geometry. If

$$J(t)=U(t)\Sigma(t)V(t)^\top,$$

then the top right singular direction is $v_1(t)$. The clean-reference top-1 angle is

$$\alpha_1^{\mathrm{clean}}(t)=\arccos\left(|v_1(t)^\top v_1(0)|\right),$$

and the adjacent-step top-1 angle is

$$\alpha_1^{\mathrm{prev}}(t_m)=\arccos\left(|v_1(t_m)^\top v_1(t_{m-1})|\right).$$

For top-$k$ subspaces, let

$$V_k(t)=[v_1(t),\dots,v_k(t)].$$

For two points $a,b$, compute singular values $s_i$ of

$$C=V_k(a)^\top V_k(b).$$

The principal angles are

$$\phi_i(a,b)=\arccos(s_i).$$

The report records max and mean principal angles. This angle family really is a path-rotation measurement: clean-reference angles compare against $x_0$, and previous-reference angles compare against the immediately previous path point.

### Randomized SVD Caveat

The off-clean Jacobian run uses a top-8 randomized SVD/sketch diagnostic, not a full exact 1024-rank SVD at every path point. The clean reference uses saved exact clean SVD. This is enough for top-k dominant geometry and spectral norm evidence, but it should be described as an estimated top-8 residual-Jacobian geometry diagnostic.

### Straight-Line Aggregate Results

Top-k cells are `max principal angle / mean principal angle`, in degrees.

| t | n | mean ||t delta*||2 | top1 clean | top1 prev | top2 clean | top2 prev | top4 clean | top4 prev | top8 clean | top8 prev | sigma1 | sigma2/sigma1 | sketch rel clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0 | 0 | NA | 0 / 0 | NA / NA | 0.01 / 0 | NA / NA | 0.01 / 0 | NA / NA | 0.868 | 0.837 | 0 |
| 0.1 | 5 | 0.8 | 40.552 | 40.552 | 40.08 / 22.9 | 40.08 / 22.9 | 70.23 / 23.12 | 70.23 / 23.12 | 69.83 / 23.04 | 69.83 / 23.04 | 1.973 | 0.499 | 1.566 |
| 0.2 | 5 | 1.6 | 54.061 | 19.436 | 87.16 / 49.05 | 54.19 / 29.84 | 66.9 / 24.44 | 22.06 / 9.52 | 73.27 / 26.81 | 31.65 / 10.48 | 3.991 | 0.341 | 2.917 |
| 0.3 | 5 | 2.4 | 49.612 | 24.043 | 89.16 / 51.83 | 13.34 / 9.4 | 63.44 / 27.51 | 24.88 / 9.99 | 77 / 30.21 | 35.54 / 10.36 | 5.809 | 0.25 | 4.015 |
| 0.4 | 5 | 3.2 | 50.241 | 5.651 | 78.63 / 48.22 | 27.45 / 15.63 | 73.68 / 33.15 | 28.01 / 9.92 | 78.96 / 32.89 | 36.45 / 9.59 | 6.44 | 0.254 | 4.545 |
| 0.5 | 5 | 4 | 51.417 | 9.046 | 78.59 / 49.59 | 28.67 / 15.83 | 79.58 / 39.06 | 33.9 / 11.18 | 83.33 / 35.63 | 35.74 / 10.25 | 7.045 | 0.299 | 5.026 |
| 0.6 | 5 | 4.8 | 53.795 | 7.533 | 78.63 / 50.98 | 12.05 / 7.34 | 81.16 / 40.78 | 16.36 / 6.1 | 85.5 / 37.76 | 30.45 / 7.9 | 7.641 | 0.317 | 5.524 |
| 0.7 | 5 | 5.6 | 55.735 | 4.22 | 78.62 / 51.46 | 6.69 / 4.28 | 81.56 / 43.21 | 23.58 / 7.95 | 86.55 / 39.31 | 16.53 / 5.63 | 7.973 | 0.33 | 5.864 |
| 0.8 | 5 | 6.4 | 56.546 | 2.456 | 78.52 / 51.57 | 13.13 / 7.28 | 81.22 / 43.36 | 25.44 / 8.21 | 87.06 / 40.17 | 11.76 / 4.48 | 8.158 | 0.351 | 6.144 |
| 0.9 | 5 | 7.2 | 57.139 | 2.072 | 78.45 / 54.53 | 21.13 / 11.23 | 81.39 / 43.64 | 25.09 / 8.01 | 86.27 / 40.69 | 13.07 / 4.45 | 8.319 | 0.402 | 6.468 |
| 1 | 5 | 8 | 57.642 | 1.89 | 78.47 / 55.87 | 10.26 / 5.71 | 81.13 / 42.5 | 21.41 / 6.9 | 86.79 / 41.37 | 13.73 / 4.53 | 8.495 | 0.441 | 6.767 |

### Straight-Line Endpoint Summary By Sample

| sample | endpoint top1 clean | endpoint top4 max | endpoint top8 max | endpoint sigma1 | endpoint sketch rel clean | max top1 clean | mean top1 prev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 34.982 | 81.256 | 86.159 | 10.171 | 6.12 | 64.759 | 15.903 |
| 7 | 88.959 | 58.202 | 88.913 | 8.466 | 7.475 | 89.669 | 14.363 |
| 40 | 88.439 | 88.887 | 85.175 | 8.633 | 6.568 | 89.154 | 16.329 |
| 47 | 29.806 | 89.57 | 87.064 | 6.11 | 5.083 | 29.806 | 4.438 |
| 115 | 46.026 | 87.728 | 86.619 | 9.095 | 8.591 | 46.026 | 7.417 |

### Straight-Line Full Raw Rows: Top-1, Spectral Norm, Response Sketch

| sample | t | ||t delta*||2 | top1 clean | top1 prev | sigma1 | sigma2/sigma1 | sketch cos clean | sketch rel clean | sketch cos prev | sketch rel prev |
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

### Straight-Line Full Raw Rows: Top-k Subspace Angles

Each angle cell is `max principal angle / mean principal angle`, in degrees.

| sample | t | top2 clean | top2 prev | top4 clean | top4 prev | top8 clean | top8 prev |
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

### Straight-Line Singular Values By Sample And Path Point

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

## Mathematical Interpretation

### What The Earlier Gradient Tables Prove

The true nonlinear endpoint-vs-movement gradient angle mainly measures how much the clean residual $b=e(x_0)$ still matters at the current finite point. Early on, $r_k$ is small and $b$ can strongly affect the endpoint gradient. Later, $r_k$ becomes large and the endpoint residual $b+r_k$ points more like $r_k$, so endpoint and movement gradients become more aligned. The data support this on the Loss3 path: the true nonlinear angle decreases from `45.06 deg` at `k=5` to `8.15 deg` at `k=50`.

The local-affine endpoint-vs-movement angle is the same story inside the clean-point Taylor model. With $p=A^\top b$ and $q=A^\top A\delta_k$, the two directions are $p+q$ and $q$. If $\|q\|$ grows while $\|p\|$ is fixed, their angle tends to zero. That explains the clean monotonic drop from `21.07 deg` at `k=5` to `1.40 deg` at `k=50` on the Loss3 path. But this is only a clean-point linearized proxy, not the true finite-radius geometry.

### What The Straight-Line Jacobian Tables Prove

The straight-line run measures a different object: pathwise rotation of the dominant residual-Jacobian singular geometry. The key result is that accumulated rotation from the clean point is large, but adjacent-step rotation becomes small. Mean top-1 clean-reference angle reaches `57.64 deg` at `t=1`, while mean top-1 previous-reference angle drops to `1.89 deg`. Top-4 and top-8 clean-reference max principal angles at `t=1` are `81.13 deg` and `86.79 deg`, so the endpoint geometry is not close to the clean-point geometry.

A matrix perturbation view explains why adjacent singular geometry can stabilize. Since right singular vectors are eigenvectors of $M(t)=J(t)^\top J(t)$, for adjacent points $J_2=J_1+\Delta J$,

$$\Delta M=J_2^\top J_2-J_1^\top J_1=J_1^\top\Delta J+\Delta J^\top J_1+\Delta J^\top\Delta J.$$

Davis-Kahan/Wedin-style perturbation bounds say, schematically,

$$\sin\angle(\mathcal V_1,\mathcal V_2)\lesssim\frac{\|\Delta M\|_2}{\mathrm{spectral\ gap}}.$$

For top-1, a rough version is

$$\sin\angle(v_1(t+h),v_1(t))\lesssim\frac{\|M(t+h)-M(t)\|_2}{\sigma_1(t)^2-\sigma_2(t)^2}.$$

Thus adjacent angles become small when the adjacent Jacobian perturbation is small or when the relevant singular gap is large. The observed mean $\sigma_2/\sigma_1$ decreases from `0.837` at `t=0` to `0.441` at `t=1`, which is consistent with the top direction becoming more dominant.

### Final Synthesis

The strongest safe conclusion is not that the endpoint returns to the clean-point linear model. The evidence says the opposite: the endpoint singular geometry is far from the clean singular geometry. The better conclusion is:

$$\text{near }x_0:\ \text{bias, curvature, and Jacobian change make directions unstable;}$$

$$\text{farther along the adversarial path}:\ \text{residual increment dominates and adjacent local Jacobian geometry stabilizes.}$$

In short: this is evidence for an early transition away from clean geometry and entry into a different locally stable high-amplification corridor. The phrase `early transition, late corridor` is a descriptive name for this observed mechanism, not a standard theorem.

## What This Does And Does Not Prove

- Supported: the clean residual bias becomes less important along the Loss3 PGD trajectory, as measured by endpoint-vs-movement gradient angles.
- Supported: along the fixed straight Loss3 direction, dominant residual-Jacobian geometry rotates far away from the clean point but becomes more stable between adjacent later path points.
- Supported: the endpoint is not well represented by the clean-point top singular geometry.
- Not proven: the entire far region is globally linear.
- Not proven by the local-affine table alone: true finite-radius nonlinear geometry. The true nonlinear table and the straight-line Jacobian table are the stronger evidence.

## Suggested Next Hardening Diagnostics

To make the mechanism quantitatively harder, record

$$\rho_{\mathrm{grad}}(k)=\frac{\|e(z_k)-e(x_0)\|_2}{\|e(x_0)\|_2},$$

and compare it against $\theta_{\mathrm{NL}}(k)$. Also record

$$R_k(t)=\frac{\|J(t+h)-J(t)\|_2}{\sigma_k(t)^2-\sigma_{k+1}(t)^2},$$

or its randomized-sketch approximation, and compare it against top-$k$ adjacent principal angles. If these ratios decrease with the observed angles, the story becomes much closer to a quantitative proof rather than only a mechanistic explanation.
