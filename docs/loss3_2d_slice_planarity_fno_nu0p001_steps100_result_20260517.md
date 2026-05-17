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
- path points: `k=5,25,50`;
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
| 5 | 3 | 0.0312 | 0.3200 | 0.1231 | 0.0314 | 0.1259 | 0.9384 | 0.2640 |
| 25 | 3 | 0.2299 | 0.8312 | 0.0330 | 0.0045 | 0.0337 | 0.9470 | 0.0704 |
| 50 | 3 | 0.6061 | 3.0405 | 0.0095 | 0.0059 | 0.0060 | 0.9599 | 0.0210 |

## Raw Summary By Sample And k

| sample_index | k | delta_budget_ratio | center_loss | plane_residual_score | plane_rmse | loss_std_on_grid | quadratic_improvement_ratio | quadratic_to_linear_ratio | grad_radial_angle_deg | u2_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0.0431 | 0.4151 | 0.0833 | 0.0026 | 0.0308 | 0.9463 | 0.1728 | 9.2261 | radial_orthogonalized_to_grad |
| 0 | 25 | 0.3116 | 1.3101 | 0.0271 | 0.0014 | 0.0503 | 0.9158 | 0.0555 | 24.9217 | radial_orthogonalized_to_grad |
| 0 | 50 | 0.7870 | 4.5709 | 0.0047 | 4.014e-04 | 0.0848 | 0.9465 | 0.0098 | 33.9128 | radial_orthogonalized_to_grad |
| 40 | 5 | 0.0240 | 0.2938 | 0.1259 | 0.0021 | 0.0165 | 0.9569 | 0.2623 | 7.7032 | radial_orthogonalized_to_grad |
| 40 | 25 | 0.1803 | 0.5853 | 0.0337 | 0.0011 | 0.0318 | 0.9497 | 0.0720 | 17.9158 | radial_orthogonalized_to_grad |
| 40 | 50 | 0.4508 | 1.2838 | 0.0179 | 6.137e-04 | 0.0343 | 0.9500 | 0.0411 | 19.1488 | radial_orthogonalized_to_grad |
| 115 | 5 | 0.0265 | 0.2510 | 0.1602 | 0.0029 | 0.0182 | 0.9119 | 0.3568 | 10.1795 | radial_orthogonalized_to_grad |
| 115 | 25 | 0.1978 | 0.5984 | 0.0381 | 0.0013 | 0.0336 | 0.9755 | 0.0836 | 16.6687 | radial_orthogonalized_to_grad |
| 115 | 50 | 0.5804 | 3.2669 | 0.0060 | 5.445e-04 | 0.0910 | 0.9833 | 0.0121 | 39.7120 | radial_orthogonalized_to_grad |

## Interpretation

This experiment is mainly a visualization/communication layer, not the strongest proof. It tests only a tiny 2D plane at three selected path locations.

The numeric score should be read as: lower `R_plane` means the scalar Loss3 slice is more planar around that path point in the gradient/radial plane.

Observed mean `R_plane` trend: `k=5: 0.1231 -> k=25: 0.0330 -> k=50: 0.0095`.

This matches the clean optional-experiment success criterion: the 2D scalar Loss3 slice is most non-planar early and becomes more planar later.

This result should be combined with the Local Taylor Error Scanner result: the finite-difference tables are the stronger numerical evidence that scalar Loss3 becomes more locally linear; this 2D slice gives a visual explanation of the same idea.

## Output Files

- output directory: `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100`
- `loss3_2d_slice_planarity_by_sample_k.csv`
- `loss3_2d_slice_grid_values.csv`
- `loss3_2d_slice_planarity_aggregate_by_k.csv`
- `manifest.json`
- `summary.md`

## Figures

- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100/figures/plane_residual_score_vs_k.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100/figures/local_2d_slice_loss_centered.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment3_2d_loss_slice_planarity_steps100/figures/local_2d_slice_affine_residual_normalized.png`
