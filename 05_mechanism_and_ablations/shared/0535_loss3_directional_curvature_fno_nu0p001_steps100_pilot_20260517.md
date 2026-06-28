# Loss3 Directional Curvature Along Path Pilot

Date: 2026-05-17 UTC

Experiment name:

\[
\textbf{Experiment D: Hessian / Curvature Along Path}
\]

This pilot uses the cheap directional version of the Hessian experiment. It does not form the full Hessian and does not estimate the top eigenvalue.

## Definition

At each saved PGD point \(z_k=x_0+\delta_k\), define the local path tangent

\[
\tau_k=\frac{z_k-z_{k-5}}{\|z_k-z_{k-5}\|_2}.
\]

The target scalar loss is

\[
L_3(z)=\|f(z)-j(z)\|_2.
\]

The intended curvature is

\[
\kappa_k=|\tau_k^\top \nabla_z^2 L_3(z_k)\tau_k|.
\]

Because exact second-order differentiation through the JAX-to-Torch solver bridge needs extra validation, this pilot estimates it by centered gradient differences:

\[
\tau_k^\top H_k\tau_k \approx \frac{\langle \nabla L_3(z_k+h\tau_k)-\nabla L_3(z_k-h\tau_k),\tau_k\rangle}{2h}.
\]

This run uses `h=0.02` and samples `[0, 40, 115]`.

The scalar-loss second difference is also recorded as a consistency check:

\[
\frac{L_3(z_k+h\tau_k)-2L_3(z_k)+L_3(z_k-h\tau_k)}{h^2}.
\]

## Aggregate By k

| k | n_points | delta_budget_ratio_mean | center_loss_mean | tangent_curvature_abs_mean | tangent_curvature_abs_std | loss_second_diff_abs_mean | fd_grad_loss_seconddiff_relative_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 3 | 0.0312 | 0.3200 | 0.1060 | 0.0180 | 0.1060 | 0.0012 |
| 10 | 3 | 0.0704 | 0.3957 | 0.1004 | 0.0199 | 0.1008 | 0.0055 |
| 15 | 3 | 0.1177 | 0.5093 | 0.0677 | 0.0338 | 0.0672 | 0.0060 |
| 20 | 3 | 0.1717 | 0.6586 | 0.0420 | 0.0230 | 0.0431 | 0.0931 |
| 25 | 3 | 0.2299 | 0.8312 | 0.0225 | 0.0141 | 0.0233 | 0.1058 |
| 30 | 3 | 0.2907 | 1.0232 | 0.0221 | 0.0080 | 0.0231 | 0.0491 |
| 35 | 3 | 0.3545 | 1.2729 | 0.0631 | 0.0406 | 0.0638 | 0.0731 |
| 40 | 3 | 0.4262 | 1.7013 | 0.1079 | 0.1449 | 0.1082 | 0.0476 |
| 45 | 3 | 0.5128 | 2.3827 | 0.0333 | 0.0153 | 0.0331 | 0.0279 |
| 50 | 3 | 0.6061 | 3.0405 | 0.0188 | 0.0134 | 0.0187 | 0.1649 |
| 55 | 3 | 0.7047 | 3.7494 | 0.0559 | 0.0562 | 0.0562 | 0.0285 |
| 60 | 3 | 0.7948 | 4.4954 | 0.0364 | 0.0320 | 0.0370 | 0.0207 |
| 65 | 3 | 0.8593 | 5.0533 | 0.0149 | 0.0061 | 0.0146 | 0.0301 |
| 70 | 3 | 0.8853 | 5.3404 | 0.0104 | 0.0048 | 0.0102 | 0.0203 |
| 75 | 3 | 0.9016 | 5.5244 | 0.0292 | 0.0209 | 0.0283 | 0.1099 |
| 80 | 3 | 0.9177 | 5.6901 | 0.0522 | 0.0277 | 0.0523 | 0.0026 |
| 85 | 3 | 0.9343 | 5.8517 | 0.0568 | 0.0211 | 0.0569 | 0.0043 |
| 90 | 3 | 0.9518 | 6.0030 | 0.0666 | 0.0251 | 0.0663 | 0.0035 |
| 95 | 3 | 0.9708 | 6.1595 | 0.0406 | 0.0175 | 0.0404 | 0.0059 |
| 100 | 3 | 0.9914 | 6.3292 | 0.0305 | 0.0167 | 0.0302 | 0.0116 |

## Trend

Mean finite-difference tangent curvature trend: `k=5: 0.106 -> k=10: 0.1004 -> k=15: 0.06771 -> k=20: 0.04204 -> k=25: 0.02255 -> k=30: 0.02206 -> k=35: 0.0631 -> k=40: 0.1079 -> k=45: 0.03327 -> k=50: 0.01875 -> k=55: 0.05592 -> k=60: 0.03643 -> k=65: 0.01488 -> k=70: 0.01035 -> k=75: 0.02916 -> k=80: 0.0522 -> k=85: 0.05676 -> k=90: 0.06661 -> k=95: 0.04058 -> k=100: 0.03046`.

This pilot is meant to answer feasibility first. If the trend is useful, it can be expanded to more samples and optional top-eigenvalue estimation.

## Output Files

- output directory: `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment4_directional_curvature_steps100_pilot`
- `directional_curvature_by_sample_k.csv`
- `directional_curvature_aggregate_by_k.csv`
- `manifest.json`

## Figures

- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment4_directional_curvature_steps100_pilot/figures/directional_curvature_vs_k.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment4_directional_curvature_steps100_pilot/figures/directional_curvature_vs_radius.png`

## Feasibility Notes

This experiment is implementable, but the reliable implementation is not the raw exact autograd HVP path yet.

### Exact HVP Smoke Test

A direct PyTorch second-order HVP call was tested at sample `0`, `k=5`, along the path tangent. It returned a value, so it did not crash:

- objective value: `0.4151228075`;
- exact-autograd reported `tau^T H tau`: `0.2662635446`;
- exact-autograd HVP norm: `0.9630646706`.

However, centered finite-difference gradients at the same point were very stable across step sizes:

| h | finite-difference `tau^T H tau` |
| ---: | ---: |
| 0.01 | 0.123452 |
| 0.02 | 0.123468 |
| 0.05 | 0.123588 |
| 0.10 | 0.123994 |

Because the finite-difference estimate is stable but the exact-autograd HVP differs by about a factor of two, the exact HVP should not be trusted yet. The likely reason is the custom JAX-to-Torch solver bridge: first-order gradients work, but second-order differentiation through the bridge needs separate validation.

### Reliable Pilot Choice

The pilot therefore uses centered gradient differences:

\[
\tau_k^\top H_k\tau_k \approx
\frac{\langle \nabla L_3(z_k+h\tau_k)-\nabla L_3(z_k-h\tau_k),\tau_k\rangle}{2h}.
\]

This is slower than a true HVP but still feasible. For this pilot, `3` samples and `20` saved path points finished successfully.

A useful consistency check is that the gradient-difference curvature and scalar-loss second difference agree closely in most rows. This supports the finite-difference directional curvature estimate.

## Interpretation Update

The pilot supports feasibility and gives a real curvature signal, but the trend is not a clean monotone curve.

The early values are high:

\[
k=5: 0.1060,\qquad k=10: 0.1004.
\]

They drop strongly by the middle:

\[
k=25: 0.0225,\qquad k=30: 0.0221,\qquad k=50: 0.0188.
\]

But there are local curvature spikes:

\[
k=40: 0.1079,\qquad k=80: 0.0522,\qquad k=85: 0.0568,\qquad k=90: 0.0666.
\]

So the best conclusion is:

\[
\text{directional curvature is measurable and often lower after the early transition, but it has local spikes along the PGD path.}
\]

This means Experiment D is feasible and useful, but it should be presented as a sharper curvature diagnostic, not as a perfectly monotone confirmation. It complements the smoother scalar finite-difference and 2D planarity results.

