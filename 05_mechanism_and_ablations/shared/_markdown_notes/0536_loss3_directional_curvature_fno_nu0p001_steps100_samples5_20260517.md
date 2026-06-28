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

This run uses `h=0.02` and samples `[0, 7, 40, 47, 115]`.

The scalar-loss second difference is also recorded as a consistency check:

\[
\frac{L_3(z_k+h\tau_k)-2L_3(z_k)+L_3(z_k-h\tau_k)}{h^2}.
\]

## Aggregate By k

| k | n_points | delta_budget_ratio_mean | center_loss_mean | tangent_curvature_abs_mean | tangent_curvature_abs_std | loss_second_diff_abs_mean | fd_grad_loss_seconddiff_relative_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 5 | 0.0359 | 0.3438 | 0.2094 | 0.1532 | 0.2093 | 8.367e-04 |
| 10 | 5 | 0.0924 | 0.5438 | 0.1320 | 0.0738 | 0.1321 | 0.0069 |
| 15 | 5 | 0.1713 | 0.9348 | 0.1270 | 0.0819 | 0.1271 | 0.0058 |
| 20 | 5 | 0.2687 | 1.5692 | 0.0649 | 0.0554 | 0.0659 | 0.0619 |
| 25 | 5 | 0.3649 | 2.1124 | 0.0337 | 0.0265 | 0.0342 | 0.0709 |
| 30 | 5 | 0.4595 | 2.6134 | 0.0221 | 0.0062 | 0.0225 | 0.0371 |
| 35 | 5 | 0.5543 | 3.1326 | 0.0387 | 0.0434 | 0.0391 | 0.1598 |
| 40 | 5 | 0.6319 | 3.6236 | 0.0775 | 0.1192 | 0.0778 | 0.0362 |
| 45 | 5 | 0.7077 | 4.1997 | 0.0268 | 0.0158 | 0.0268 | 0.0336 |
| 50 | 5 | 0.7636 | 4.6169 | 0.0187 | 0.0104 | 0.0184 | 0.1117 |
| 55 | 5 | 0.8228 | 5.0604 | 0.0413 | 0.0471 | 0.0420 | 0.0406 |
| 60 | 5 | 0.8769 | 5.5226 | 0.0319 | 0.0254 | 0.0324 | 0.0195 |
| 65 | 5 | 0.9156 | 5.8692 | 0.0212 | 0.0096 | 0.0210 | 0.0193 |
| 70 | 5 | 0.9312 | 6.0512 | 0.0201 | 0.0136 | 0.0200 | 0.0181 |
| 75 | 5 | 0.9409 | 6.1696 | 0.0326 | 0.0180 | 0.0319 | 0.0732 |
| 80 | 5 | 0.9506 | 6.2756 | 0.0475 | 0.0235 | 0.0474 | 0.0053 |
| 85 | 5 | 0.9606 | 6.3779 | 0.0511 | 0.0197 | 0.0508 | 0.0126 |
| 90 | 5 | 0.9711 | 6.4733 | 0.0580 | 0.0240 | 0.0577 | 0.0058 |
| 95 | 5 | 0.9825 | 6.5710 | 0.0433 | 0.0172 | 0.0429 | 0.0114 |
| 100 | 5 | 0.9948 | 6.6760 | 0.0382 | 0.0193 | 0.0383 | 0.0126 |

## Trend

Mean finite-difference tangent curvature trend: `k=5: 0.2094 -> k=10: 0.132 -> k=15: 0.127 -> k=20: 0.06488 -> k=25: 0.03372 -> k=30: 0.02207 -> k=35: 0.03868 -> k=40: 0.07751 -> k=45: 0.0268 -> k=50: 0.01868 -> k=55: 0.04131 -> k=60: 0.03186 -> k=65: 0.02117 -> k=70: 0.0201 -> k=75: 0.03261 -> k=80: 0.04746 -> k=85: 0.05115 -> k=90: 0.05799 -> k=95: 0.04331 -> k=100: 0.03818`.

This pilot is meant to answer feasibility first. If the trend is useful, it can be expanded to more samples and optional top-eigenvalue estimation.

## Output Files

- output directory: `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment4_directional_curvature_steps100_samples5`
- `directional_curvature_by_sample_k.csv`
- `directional_curvature_aggregate_by_k.csv`
- `manifest.json`

## Figures

- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment4_directional_curvature_steps100_samples5/figures/directional_curvature_vs_k.png`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment4_directional_curvature_steps100_samples5/figures/directional_curvature_vs_radius.png`

## Five-Sample Interpretation

This rerun uses all currently saved 100-step trajectories:

\[
\{0,7,40,47,115\}.
\]

So it is no longer just the original 3-sample pilot. The full every-5-step table above contains all saved points from \(k=5\) to \(k=100\).

### Region Averages

Using the finite-difference tangent curvature

\[
\kappa_k=|\tau_k^\top H_k\tau_k|,
\]

the region averages are:

| region | k range | mean \(\kappa\) | median \(\kappa\) | n |
| --- | --- | ---: | ---: | ---: |
| early | 5,10,15 | 0.1561 | 0.1141 | 15 |
| transition | 20,25,30 | 0.0402 | 0.0228 | 15 |
| middle/low corridor | 45,50,55,60,65,70 | 0.0267 | 0.0199 | 30 |
| late boundary | 75,80,85,90,95,100 | 0.0451 | 0.0368 | 30 |

This strengthens the main statement:

\[
\text{early curvature is high, and after the early transition the path usually enters a lower-curvature regime.}
\]

But it also shows that the late boundary region is not perfectly flat. There is a moderate rebound near the boundary.

### Per-Sample Region Means

| sample | early 5-15 | transition 20-30 | middle/low 45-70 | late boundary 75-100 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0719 | 0.0158 | 0.0463 | 0.0324 |
| 7 | 0.2804 | 0.0913 | 0.0280 | 0.0578 |
| 40 | 0.1066 | 0.0416 | 0.0162 | 0.0549 |
| 47 | 0.2260 | 0.0231 | 0.0205 | 0.0299 |
| 115 | 0.0957 | 0.0292 | 0.0223 | 0.0505 |

The early-to-transition drop appears in every sample. This means the early-high curvature is not only caused by one sample.

However, the exact spikes are sample-dependent.

### Largest Individual Spikes

| sample | k | \(\kappa\) |
| ---: | ---: | ---: |
| 7 | 5 | 0.4991 |
| 115 | 40 | 0.3127 |
| 47 | 10 | 0.2758 |
| 7 | 15 | 0.2593 |
| 47 | 5 | 0.2296 |
| 47 | 15 | 0.1725 |
| 7 | 20 | 0.1693 |
| 0 | 55 | 0.1353 |
| 0 | 5 | 0.1235 |
| 115 | 10 | 0.1149 |
| 40 | 10 | 0.1141 |
| 40 | 5 | 0.1134 |

So the refined conclusion is:

\[
\text{the early-to-mid curvature drop is robust across the available samples, but localized curvature spikes are sample-specific.}
\]

This means Experiment D supports the local-linearity story, but it should be phrased carefully. It does not prove monotone curvature decay. It shows a high-curvature early transition, a lower-curvature middle corridor, and occasional curvature spikes, especially near specific path events or near the boundary.

