# Loss3 2D Local Slice Planarity Result

Date: 2026-05-17 UTC

Experiment name:

\[
\textbf{Experiment 3: Small 2D Local Loss Slice Planarity Plot}
\]

This is the optional visualization experiment from the simple path-linearity plan. It is run after the gradient-rotation and finite-difference local-linearity experiments showed a useful signal.

## Setup

- model/problem: FNO Burgers `nu=0.001`;
- objective: scalar Loss3, `L3(z)=||f(z)-j(z)||_2`;
- path: saved 100-step `loss3_original_pgd` trajectory;
- samples: `0, 40, 115`;
- path points: `k=0,5,10,...,100`;
- grid: `a,b in {-rho, -rho/2, 0, rho/2, rho}` with `rho=0.16`;
- plane directions: `u1 = normalized Loss3 gradient`; `u2 = radial direction orthogonalized against u1`.

At each path point:

\[
z_k=x_0+\delta_k,
\]

then the grid evaluates:

\[
L_3(z_k+a u_1+b u_2).
\]

An affine plane is fit to the 25 grid values:

\[
\widehat L(a,b)=c_0+c_1a+c_2b.
\]

The planarity score is:

\[
R_{\mathrm{plane}}=\frac{\sqrt{N^{-1}\sum_i(L_i-\widehat L_i)^2}}{\mathrm{std}(L_i)+\varepsilon_{\mathrm{num}}}.
\]

Smaller `R_plane` means the local 2D loss slice is closer to an affine plane.

## Aggregate By k

| k | n_points | delta_budget_ratio_mean | center_loss_mean | plane_residual_score_mean | plane_residual_score_std | plane_residual_score_median | quadratic_improvement_ratio_mean | quadratic_to_linear_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 9.984e-07 | 0.2736 | 0.0544 | 0.0041 | 0.0570 | 0.9117 | 0.1294 |
| 5 | 3 | 0.0312 | 0.3200 | 0.1231 | 0.0314 | 0.1259 | 0.9384 | 0.2640 |
| 10 | 3 | 0.0704 | 0.3957 | 0.0893 | 0.0222 | 0.0857 | 0.9568 | 0.1949 |
| 15 | 3 | 0.1177 | 0.5093 | 0.0618 | 0.0154 | 0.0581 | 0.9482 | 0.1351 |
| 20 | 3 | 0.1717 | 0.6586 | 0.0427 | 0.0098 | 0.0425 | 0.9653 | 0.0938 |
| 25 | 3 | 0.2299 | 0.8312 | 0.0330 | 0.0045 | 0.0337 | 0.9470 | 0.0704 |
| 30 | 3 | 0.2907 | 1.0232 | 0.0415 | 0.0105 | 0.0426 | 0.9374 | 0.0854 |
| 35 | 3 | 0.3545 | 1.2729 | 0.0618 | 0.0425 | 0.0401 | 0.9440 | 0.1304 |
| 40 | 3 | 0.4262 | 1.7013 | 0.0305 | 0.0216 | 0.0216 | 0.9135 | 0.0658 |
| 45 | 3 | 0.5128 | 2.3827 | 0.0102 | 0.0067 | 0.0072 | 0.9370 | 0.0223 |
| 50 | 3 | 0.6061 | 3.0405 | 0.0095 | 0.0059 | 0.0060 | 0.9599 | 0.0210 |
| 55 | 3 | 0.7047 | 3.7494 | 0.0130 | 0.0028 | 0.0131 | 0.6582 | 0.0226 |
| 60 | 3 | 0.7948 | 4.4954 | 0.0085 | 0.0047 | 0.0072 | 0.9245 | 0.0189 |
| 65 | 3 | 0.8593 | 5.0533 | 0.0072 | 0.0043 | 0.0045 | 0.9582 | 0.0159 |
| 70 | 3 | 0.8853 | 5.3404 | 0.0071 | 0.0058 | 0.0049 | 0.9422 | 0.0144 |
| 75 | 3 | 0.9016 | 5.5244 | 0.0127 | 0.0104 | 0.0056 | 0.9723 | 0.0259 |
| 80 | 3 | 0.9177 | 5.6901 | 0.0202 | 0.0190 | 0.0088 | 0.9521 | 0.0420 |
| 85 | 3 | 0.9343 | 5.8517 | 0.0207 | 0.0229 | 0.0046 | 0.9276 | 0.0432 |
| 90 | 3 | 0.9518 | 6.0030 | 0.0149 | 0.0144 | 0.0054 | 0.9678 | 0.0313 |
| 95 | 3 | 0.9708 | 6.1595 | 0.0093 | 0.0076 | 0.0041 | 0.9754 | 0.0198 |
| 100 | 3 | 0.9914 | 6.3292 | 0.0069 | 0.0045 | 0.0038 | 0.9742 | 0.0147 |

## Raw Summary By Sample And k

| sample_index | k | delta_budget_ratio | center_loss | plane_residual_score | plane_rmse | loss_std_on_grid | quadratic_improvement_ratio | quadratic_to_linear_ratio | grad_radial_angle_deg | u2_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 9.976e-07 | 0.3324 | 0.0487 | 0.0011 | 0.0230 | 0.9459 | 0.1163 | 91.2601 | radial_orthogonalized_to_grad |
| 0 | 5 | 0.0431 | 0.4151 | 0.0833 | 0.0026 | 0.0308 | 0.9463 | 0.1728 | 9.2261 | radial_orthogonalized_to_grad |
| 0 | 10 | 0.0984 | 0.5545 | 0.0641 | 0.0025 | 0.0387 | 0.9721 | 0.1313 | 15.8915 | radial_orthogonalized_to_grad |
| 0 | 15 | 0.1641 | 0.7614 | 0.0450 | 0.0020 | 0.0453 | 0.9211 | 0.0913 | 22.0111 | radial_orthogonalized_to_grad |
| 0 | 20 | 0.2365 | 1.0236 | 0.0308 | 0.0015 | 0.0488 | 0.9656 | 0.0657 | 23.6998 | radial_orthogonalized_to_grad |
| 0 | 25 | 0.3116 | 1.3101 | 0.0271 | 0.0014 | 0.0503 | 0.9158 | 0.0555 | 24.9217 | radial_orthogonalized_to_grad |
| 0 | 30 | 0.3880 | 1.6210 | 0.0538 | 0.0029 | 0.0547 | 0.9246 | 0.1098 | 30.8754 | radial_orthogonalized_to_grad |
| 0 | 35 | 0.4693 | 2.0714 | 0.0401 | 0.0029 | 0.0728 | 0.9584 | 0.0831 | 43.6208 | radial_orthogonalized_to_grad |
| 0 | 40 | 0.5655 | 2.8538 | 0.0095 | 8.444e-04 | 0.0884 | 0.8790 | 0.0191 | 44.7152 | radial_orthogonalized_to_grad |
| 0 | 45 | 0.6741 | 3.7420 | 0.0038 | 3.229e-04 | 0.0844 | 0.9096 | 0.0079 | 38.4724 | radial_orthogonalized_to_grad |
| 0 | 50 | 0.7870 | 4.5709 | 0.0047 | 4.014e-04 | 0.0848 | 0.9465 | 0.0098 | 33.9128 | radial_orthogonalized_to_grad |
| 0 | 55 | 0.9104 | 5.6094 | 0.0131 | 0.0016 | 0.1188 | 0.0598 | 0.0106 | 46.1058 | radial_orthogonalized_to_grad |
| 0 | 60 | 1.0000 | 6.6694 | 0.0035 | 3.446e-04 | 0.0998 | 0.9761 | 0.0081 | 35.8339 | radial_orthogonalized_to_grad |
| 0 | 65 | 1.0000 | 7.0105 | 0.0039 | 3.840e-04 | 0.0986 | 0.9878 | 0.0082 | 30.8911 | radial_orthogonalized_to_grad |
| 0 | 70 | 1.0000 | 7.2744 | 0.0049 | 4.837e-04 | 0.0995 | 0.9888 | 0.0098 | 27.0801 | radial_orthogonalized_to_grad |
| 0 | 75 | 1.0000 | 7.4880 | 0.0051 | 5.122e-04 | 0.1012 | 0.9812 | 0.0102 | 23.8737 | radial_orthogonalized_to_grad |
| 0 | 80 | 1.0000 | 7.6624 | 0.0048 | 4.910e-04 | 0.1029 | 0.9756 | 0.0097 | 21.0771 | radial_orthogonalized_to_grad |
| 0 | 85 | 1.0000 | 7.8042 | 0.0044 | 4.540e-04 | 0.1043 | 0.9720 | 0.0089 | 18.6316 | radial_orthogonalized_to_grad |
| 0 | 90 | 1.0000 | 7.9188 | 0.0040 | 4.210e-04 | 0.1054 | 0.9703 | 0.0083 | 16.5020 | radial_orthogonalized_to_grad |
| 0 | 95 | 1.0000 | 8.0113 | 0.0037 | 3.983e-04 | 0.1063 | 0.9699 | 0.0079 | 14.6530 | radial_orthogonalized_to_grad |
| 0 | 100 | 1.0000 | 8.0857 | 0.0036 | 3.851e-04 | 0.1069 | 0.9704 | 0.0078 | 13.0482 | radial_orthogonalized_to_grad |
| 40 | 0 | 9.978e-07 | 0.2684 | 0.0577 | 7.638e-04 | 0.0132 | 0.9436 | 0.1371 | 91.0993 | radial_orthogonalized_to_grad |
| 40 | 5 | 0.0240 | 0.2938 | 0.1259 | 0.0021 | 0.0165 | 0.9569 | 0.2623 | 7.7032 | radial_orthogonalized_to_grad |
| 40 | 10 | 0.0537 | 0.3333 | 0.0857 | 0.0018 | 0.0206 | 0.9677 | 0.1802 | 12.2841 | radial_orthogonalized_to_grad |
| 40 | 15 | 0.0899 | 0.3934 | 0.0581 | 0.0014 | 0.0250 | 0.9738 | 0.1231 | 14.9819 | radial_orthogonalized_to_grad |
| 40 | 20 | 0.1324 | 0.4777 | 0.0425 | 0.0012 | 0.0288 | 0.9642 | 0.0904 | 16.6556 | radial_orthogonalized_to_grad |
| 40 | 25 | 0.1803 | 0.5853 | 0.0337 | 0.0011 | 0.0318 | 0.9497 | 0.0720 | 17.9158 | radial_orthogonalized_to_grad |
| 40 | 30 | 0.2320 | 0.7119 | 0.0281 | 9.529e-04 | 0.0339 | 0.9373 | 0.0607 | 18.8402 | radial_orthogonalized_to_grad |
| 40 | 35 | 0.2860 | 0.8513 | 0.0243 | 8.500e-04 | 0.0350 | 0.9312 | 0.0534 | 19.3666 | radial_orthogonalized_to_grad |
| 40 | 40 | 0.3412 | 0.9969 | 0.0216 | 7.618e-04 | 0.0353 | 0.9330 | 0.0483 | 19.5242 | radial_orthogonalized_to_grad |
| 40 | 45 | 0.3964 | 1.1425 | 0.0195 | 6.841e-04 | 0.0350 | 0.9403 | 0.0445 | 19.4132 | radial_orthogonalized_to_grad |
| 40 | 50 | 0.4508 | 1.2838 | 0.0179 | 6.137e-04 | 0.0343 | 0.9500 | 0.0411 | 19.1488 | radial_orthogonalized_to_grad |
| 40 | 55 | 0.5041 | 1.4186 | 0.0164 | 5.474e-04 | 0.0334 | 0.9597 | 0.0378 | 18.8513 | radial_orthogonalized_to_grad |
| 40 | 60 | 0.5560 | 1.5462 | 0.0148 | 4.812e-04 | 0.0325 | 0.9680 | 0.0339 | 18.6648 | radial_orthogonalized_to_grad |
| 40 | 65 | 0.6066 | 1.6673 | 0.0133 | 4.231e-04 | 0.0318 | 0.9725 | 0.0293 | 18.8093 | radial_orthogonalized_to_grad |
| 40 | 70 | 0.6560 | 1.7836 | 0.0150 | 4.703e-04 | 0.0313 | 0.9668 | 0.0303 | 19.7175 | radial_orthogonalized_to_grad |
| 40 | 75 | 0.7047 | 1.8987 | 0.0273 | 8.637e-04 | 0.0316 | 0.9670 | 0.0556 | 22.2835 | radial_orthogonalized_to_grad |
| 40 | 80 | 0.7532 | 2.0198 | 0.0470 | 0.0016 | 0.0334 | 0.9811 | 0.0976 | 27.8521 | radial_orthogonalized_to_grad |
| 40 | 85 | 0.8028 | 2.1650 | 0.0530 | 0.0020 | 0.0382 | 0.9878 | 0.1106 | 36.5284 | radial_orthogonalized_to_grad |
| 40 | 90 | 0.8553 | 2.3699 | 0.0352 | 0.0016 | 0.0463 | 0.9626 | 0.0735 | 44.4743 | radial_orthogonalized_to_grad |
| 40 | 95 | 0.9124 | 2.6625 | 0.0201 | 0.0011 | 0.0538 | 0.9769 | 0.0419 | 48.2824 | radial_orthogonalized_to_grad |
| 40 | 100 | 0.9742 | 3.0384 | 0.0132 | 7.881e-04 | 0.0597 | 0.9737 | 0.0274 | 49.5721 | radial_orthogonalized_to_grad |
| 115 | 0 | 9.997e-07 | 0.2199 | 0.0570 | 8.224e-04 | 0.0144 | 0.8457 | 0.1347 | 89.0595 | radial_orthogonalized_to_grad |
| 115 | 5 | 0.0265 | 0.2510 | 0.1602 | 0.0029 | 0.0182 | 0.9119 | 0.3568 | 10.1795 | radial_orthogonalized_to_grad |
| 115 | 10 | 0.0591 | 0.2993 | 0.1181 | 0.0027 | 0.0228 | 0.9305 | 0.2731 | 13.6650 | radial_orthogonalized_to_grad |
| 115 | 15 | 0.0992 | 0.3731 | 0.0823 | 0.0023 | 0.0275 | 0.9497 | 0.1910 | 14.5499 | radial_orthogonalized_to_grad |
| 115 | 20 | 0.1461 | 0.4746 | 0.0549 | 0.0017 | 0.0313 | 0.9662 | 0.1254 | 15.3473 | radial_orthogonalized_to_grad |
| 115 | 25 | 0.1978 | 0.5984 | 0.0381 | 0.0013 | 0.0336 | 0.9755 | 0.0836 | 16.6687 | radial_orthogonalized_to_grad |
| 115 | 30 | 0.2521 | 0.7368 | 0.0426 | 0.0015 | 0.0353 | 0.9503 | 0.0856 | 19.6488 | radial_orthogonalized_to_grad |
| 115 | 35 | 0.3080 | 0.8961 | 0.1212 | 0.0050 | 0.0413 | 0.9425 | 0.2547 | 31.4680 | radial_orthogonalized_to_grad |
| 115 | 40 | 0.3720 | 1.2533 | 0.0603 | 0.0049 | 0.0811 | 0.9285 | 0.1300 | 54.0301 | radial_orthogonalized_to_grad |
| 115 | 45 | 0.4680 | 2.2635 | 0.0072 | 6.799e-04 | 0.0940 | 0.9612 | 0.0147 | 47.5413 | radial_orthogonalized_to_grad |
| 115 | 50 | 0.5804 | 3.2669 | 0.0060 | 5.445e-04 | 0.0910 | 0.9833 | 0.0121 | 39.7120 | radial_orthogonalized_to_grad |
| 115 | 55 | 0.6997 | 4.2201 | 0.0094 | 8.601e-04 | 0.0911 | 0.9550 | 0.0193 | 35.9588 | radial_orthogonalized_to_grad |
| 115 | 60 | 0.8284 | 5.2706 | 0.0072 | 7.082e-04 | 0.0987 | 0.8293 | 0.0147 | 34.0922 | radial_orthogonalized_to_grad |
| 115 | 65 | 0.9712 | 6.4821 | 0.0045 | 4.784e-04 | 0.1061 | 0.9143 | 0.0101 | 29.7820 | radial_orthogonalized_to_grad |
| 115 | 70 | 1.0000 | 6.9630 | 0.0015 | 1.596e-04 | 0.1088 | 0.8711 | 0.0030 | 25.4379 | radial_orthogonalized_to_grad |
| 115 | 75 | 1.0000 | 7.1865 | 0.0056 | 6.164e-04 | 0.1100 | 0.9688 | 0.0119 | 22.6076 | radial_orthogonalized_to_grad |
| 115 | 80 | 1.0000 | 7.3882 | 0.0088 | 9.925e-04 | 0.1131 | 0.8995 | 0.0187 | 22.1845 | radial_orthogonalized_to_grad |
| 115 | 85 | 1.0000 | 7.5858 | 0.0046 | 5.218e-04 | 0.1133 | 0.8230 | 0.0102 | 19.8048 | radial_orthogonalized_to_grad |
| 115 | 90 | 1.0000 | 7.7202 | 0.0054 | 5.975e-04 | 0.1104 | 0.9706 | 0.0121 | 15.7132 | radial_orthogonalized_to_grad |
| 115 | 95 | 1.0000 | 7.8046 | 0.0041 | 4.409e-04 | 0.1087 | 0.9793 | 0.0095 | 13.0250 | radial_orthogonalized_to_grad |
| 115 | 100 | 1.0000 | 7.8635 | 0.0038 | 4.105e-04 | 0.1080 | 0.9785 | 0.0091 | 11.1731 | radial_orthogonalized_to_grad |

## Interpretation

This experiment is mainly a visualization/communication layer, not the strongest proof. It tests a tiny 2D plane at every saved 5-step path location from `k=0` to `k=100`.

The numeric score should be read as: lower `R_plane` means the scalar Loss3 slice is more planar around that path point in the gradient/radial plane.

Observed mean `R_plane` trend: `k=0: 0.0544 -> k=5: 0.1231 -> k=10: 0.0893 -> k=15: 0.0618 -> k=20: 0.0427 -> k=25: 0.0330 -> k=30: 0.0415 -> k=35: 0.0618 -> k=40: 0.0305 -> k=45: 0.0102 -> k=50: 0.0095 -> k=55: 0.0130 -> k=60: 0.0085 -> k=65: 0.0072 -> k=70: 0.0071 -> k=75: 0.0127 -> k=80: 0.0202 -> k=85: 0.0207 -> k=90: 0.0149 -> k=95: 0.0093 -> k=100: 0.0069`.

The trend is not perfectly monotone, so use this figure as qualitative support rather than a standalone proof.

This result should be combined with the Local Taylor Error Scanner result: the finite-difference tables are the stronger numerical evidence that scalar Loss3 becomes more locally linear; this 2D slice gives a visual explanation of the same idea.

## Output Files

- output directory: `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100_dense`
- `loss3_2d_slice_planarity_by_sample_k.csv`
- `loss3_2d_slice_grid_values.csv`
- `loss3_2d_slice_planarity_aggregate_by_k.csv`
- `manifest.json`
- `summary.md`

## Figures

- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100_dense/figures/plane_residual_score_vs_k.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100_dense/figures/local_2d_slice_loss_centered.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100_dense/figures/local_2d_slice_affine_residual_normalized.png`

## R-plane Definition And Dense Result Notes

This section records the explicit definition and the dense every-5-step result discussed after the run.

The planarity score \(R_{\mathrm{plane}}\) means: how far this local 2D Loss3 slice is from the best fitted affine plane, after normalizing by the variation of the loss values on that slice.

At a path point

\[
z_k=x_0+\delta_k,
\]

the first plane direction is the normalized scalar Loss3 gradient:

\[
u_1=\frac{\nabla_z L_3(z_k)}{\|\nabla_z L_3(z_k)\|_2}.
\]

The second direction is the radial direction orthogonalized against \(u_1\):

\[
\tilde u_2=\frac{\delta_k}{\|\delta_k\|_2}
-
\left\langle \frac{\delta_k}{\|\delta_k\|_2},u_1\right\rangle u_1,
\qquad
u_2=\frac{\tilde u_2}{\|\tilde u_2\|_2}.
\]

At \(k=0\), the radial direction is not meaningful because \(\delta_0\approx 0\). In the dense implementation, \(k=0\) is kept for completeness, but it should be interpreted cautiously; the more reliable early reference is \(k=5\).

On the 2D plane, the experiment evaluates 25 points:

\[
z_k+a_i u_1+b_i u_2,
\]

where

\[
a_i,b_i\in\{-\rho,-\rho/2,0,\rho/2,\rho\},
\qquad
\rho=0.16.
\]

For each grid point, compute scalar Loss3:

\[
L_i=L_3(z_k+a_i u_1+b_i u_2).
\]

Then fit the best affine plane by least squares:

\[
\widehat L(a,b)=c_0+c_1a+c_2b,
\]

with

\[
(c_0,c_1,c_2)
=
\arg\min_{c_0,c_1,c_2}
\sum_i
\left(L_i-(c_0+c_1a_i+c_2b_i)\right)^2.
\]

The plane residual RMSE is

\[
\mathrm{RMSE}_{\mathrm{plane}}
=
\sqrt{
\frac{1}{N}
\sum_i
\left(L_i-\widehat L_i\right)^2
}.
\]

The normalized planarity score is

\[
R_{\mathrm{plane}}
=
\frac{
\mathrm{RMSE}_{\mathrm{plane}}
}{
\mathrm{std}(L_i)+\varepsilon_{\mathrm{num}}
}.
\]

So:

\[
R_{\mathrm{plane}}\approx 0
\]

means the local 2D Loss3 slice is very close to an affine plane, hence more locally linear. Larger \(R_{\mathrm{plane}}\) means the slice is more warped/non-planar.

### Dense Every-5-Step Mean Trend

The dense run evaluates every saved point:

\[
k=0,5,10,\ldots,100.
\]

Mean \(R_{\mathrm{plane}}\) over samples `0, 40, 115`:

| k | mean \(R_{\mathrm{plane}}\) |
| ---: | ---: |
| 0 | 0.0544 |
| 5 | 0.1231 |
| 10 | 0.0893 |
| 15 | 0.0618 |
| 20 | 0.0427 |
| 25 | 0.0330 |
| 30 | 0.0415 |
| 35 | 0.0618 |
| 40 | 0.0305 |
| 45 | 0.0102 |
| 50 | 0.0095 |
| 55 | 0.0130 |
| 60 | 0.0085 |
| 65 | 0.0072 |
| 70 | 0.0071 |
| 75 | 0.0127 |
| 80 | 0.0202 |
| 85 | 0.0207 |
| 90 | 0.0149 |
| 95 | 0.0093 |
| 100 | 0.0069 |

Interpretation: the dense trend is not perfectly monotone. There are visible bumps around \(k=30/35\) and \(k=75/85\). But the overall pattern is still clear: the early path, especially around \(k=5\), is much less planar, while the later path is mostly in a low \(R_{\mathrm{plane}}\) regime.

This should be read as qualitative/visual support for the scalar Loss3 local-linearity story, not as the main proof. The Local Taylor Error Scanner tables remain the stronger numerical evidence.

### Dense Figure Paths

- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100_dense/figures/plane_residual_score_vs_k.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100_dense/figures/local_2d_slice_loss_centered.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100_dense/figures/local_2d_slice_affine_residual_normalized.png`

