# Loss3 Finite-Difference Local Linearity: CL/CE Explanation

Date: 2026-05-17 UTC

This note records the discussion around the symmetric finite-difference local-linearity experiment for FNO `nu=0.001`, `loss3_original_pgd`, `epsilon=8`, `steps=100`.

The detailed fine-bin tables are in:

- `docs/loss3_finite_difference_linearity_fno_nu0p001_steps100_fine_bins_20260517.md`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_by_sample_k_direction.csv`


## Experiment Name

Recommended name:

\[
\textbf{Experiment A: Local Taylor Error Scanner}
\]

Plain-language name:

\[
\textbf{Symmetric Finite-Difference Local Linearity Scanner}
\]

Informal alias used in discussion:

\[
\textbf{Local Tailor Scanner}
\]

The mathematically correct spelling is `Taylor`, because the experiment is checking the local Taylor linear approximation error by using symmetric three-point finite differences. It does not explicitly compute Hessians or Jacobians.

## 1. Path Point Definition

At PGD step \(k\), the saved perturbation is \(\delta_k\). The experiment evaluates local behavior around

\[
z_k = x_0 + \delta_k.
\]

So yes: \(z\) is the adversarial/path point, and it is exactly clean input plus the current PGD perturbation.

At each \(z_k\), for a unit direction \(u\), the experiment evaluates three symmetric points:

\[
z_k-\rho u,\qquad z_k,\qquad z_k+\rho u.
\]

In this run:

\[
\rho = 0.16 = 0.02\epsilon,\qquad \epsilon=8.
\]

The directions are:

- `grad`: current Loss3 gradient direction;
- `radial`: current perturbation direction \(\delta_k/\|\delta_k\|_2\);
- `step`: previous path-step direction, when available;
- `random_0`, `random_1`, `random_2`, `random_3`: fixed random probe directions.

The saved path points are:

\[
k=0,5,10,\ldots,100.
\]

## 2. Residual Vector And Scalar Loss3

The residual/error vector field is

\[
e(z)=f(z)-j(z),
\]

where \(f(z)\) is the neural operator prediction and \(j(z)\) is the solver/oracle output.

This is a vector or field, not a scalar.

The scalar Loss3 objective is the residual norm:

\[
L_3(z)=\|e(z)\|_2=\|f(z)-j(z)\|_2.
\]

So the scalar loss is indeed the magnitude of the residual vector. The difference between the two diagnostics is not that they come from unrelated objects. The difference is:

\[
e(z) \longrightarrow \|e(z)\|_2.
\]

The residual vector keeps direction/component information. The scalar Loss3 throws away residual direction and keeps only the norm.

## 3. Scalar Loss Curvature Score \(C_L\)

For the three points around \(z\), define:

\[
e_- = e(z-\rho u),\qquad e_0=e(z),\qquad e_+=e(z+\rho u).
\]

The scalar values are:

\[
L_-=\|e_-\|_2,\qquad L_0=\|e_0\|_2,\qquad L_+=\|e_+\|_2.
\]

If the scalar function \(L_3\) is locally linear along direction \(u\), then the middle value should lie near the midpoint of the two side values:

\[
L_+ - 2L_0 + L_- \approx 0.
\]

The scalar Loss3 curvature/linearity score is

\[
C_L(z,u,\rho)
=
\frac{
\left|L_3(z+\rho u)-2L_3(z)+L_3(z-\rho u)\right|
}{
\left|L_3(z+\rho u)-L_3(z-\rho u)\right|+\varepsilon_{\mathrm{num}}
}.
\]

Interpretation:

- numerator: three-point second difference of the scalar loss;
- denominator: first-difference scale, used for normalization;
- smaller \(C_L\): scalar Loss3 looks more locally linear along this direction.

This is not a gradient angle and not a Jacobian/SVD measurement. It is a three-point finite-difference test of local bending.

## 4. Residual-Map Curvature Score \(C_E\)

The vector residual map is tested before taking the norm.

If the residual map \(e(z)\) is locally affine along direction \(u\), then:

\[
e(z+\rho u)-2e(z)+e(z-\rho u)\approx 0.
\]

The residual-map curvature/linearity score is

\[
C_E(z,u,\rho)
=
\frac{
\left\|e(z+\rho u)-2e(z)+e(z-\rho u)\right\|_2
}{
\left\|e(z+\rho u)-e(z-\rho u)\right\|_2+\varepsilon_{\mathrm{num}}
}.
\]

Interpretation:

- numerator: vector second difference of the residual field;
- denominator: vector first-difference scale;
- smaller \(C_E\): the full residual vector field looks more locally affine along this direction.

The key distinction is:

\[
C_L \text{ tests whether } \|e(z)\|_2 \text{ is locally linear.}
\]

\[
C_E \text{ tests whether } e(z) \text{ itself is locally affine.}
\]

Because \(\|e(z)\|_2\) discards direction information, \(C_L\) can be small even when \(C_E\) is not small. For example, a residual vector can rotate while maintaining an almost linearly changing norm.

## 5. Early/Middle/Late Definition

The original coarse summary used radius/budget bins, not wall-clock time bins. Define:

\[
r_k=\frac{\|\delta_k\|_2}{\epsilon}.
\]

Then:

\[
\text{early}: r_k\le 0.2,
\]

\[
\text{middle}: 0.2<r_k<0.6,
\]

\[
\text{late}: r_k\ge 0.6.
\]

So early/middle/late means how far the perturbation has moved away from \(x_0\) relative to the allowed \(L_2\) budget, not simply the step index.

The coarse three-bin scalar result was:

| region | mean \(C_L\) | median \(C_L\) |
| --- | ---: | ---: |
| early | 0.3314 | 0.0641 |
| middle | 0.0515 | 0.0109 |
| late | 0.0148 | 0.0046 |

This supports the narrow claim:

\[
\text{Along the PGD path, scalar Loss3 becomes more locally linear farther from } x_0.
\]

But three bins are too coarse, so the finer per-\(k\) and ten-radius-bin summaries below are more informative.

## 6. Averaging

The raw data contains:

- 5 samples;
- 21 saved path steps: \(k=0,5,\ldots,100\);
- directions: `grad`, `radial`, `step`, `random_0`, `random_1`, `random_2`, `random_3`.

For each nonzero saved \(k\), the aggregation usually has:

\[
5 \text{ samples} \times 7 \text{ directions} = 35 \text{ rows}.
\]

At \(k=0\), there is no previous-step direction, so it has:

\[
5 \text{ samples} \times 6 \text{ directions} = 30 \text{ rows}.
\]

The reported means/medians are ordinary means/medians over those rows. They are not weighted by loss size, gradient norm, or sample importance.

## 7. Per-Step Fine Table

This is the direct path/time view, using all directions together.

| k | budget ratio mean | mean \(C_L\) | median \(C_L\) | mean \(C_E\) | median \(C_E\) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9.988e-07 | 0.2779 | 0.1600 | 0.0713 | 0.0413 |
| 5 | 0.0359 | 0.2236 | 0.0904 | 0.1145 | 0.0483 |
| 10 | 0.0924 | 0.9206 | 0.0525 | 0.1064 | 0.0452 |
| 15 | 0.1713 | 0.0786 | 0.0306 | 0.1057 | 0.0410 |
| 20 | 0.2687 | 0.0926 | 0.0153 | 0.0937 | 0.0390 |
| 25 | 0.3649 | 0.0373 | 0.0122 | 0.0957 | 0.0357 |
| 30 | 0.4595 | 0.0257 | 0.0096 | 0.1035 | 0.0340 |
| 35 | 0.5543 | 0.0330 | 0.0081 | 0.1213 | 0.0323 |
| 40 | 0.6319 | 0.0155 | 0.0093 | 0.1341 | 0.0328 |
| 45 | 0.7077 | 0.0113 | 0.0054 | 0.1285 | 0.0284 |
| 50 | 0.7636 | 0.0158 | 0.0042 | 0.1214 | 0.0275 |
| 55 | 0.8228 | 0.0289 | 0.0050 | 0.1170 | 0.0257 |
| 60 | 0.8769 | 0.0223 | 0.0050 | 0.1113 | 0.0327 |
| 65 | 0.9156 | 0.0131 | 0.0047 | 0.1102 | 0.0326 |
| 70 | 0.9312 | 0.0093 | 0.0031 | 0.1090 | 0.0325 |
| 75 | 0.9409 | 0.0104 | 0.0043 | 0.1083 | 0.0324 |
| 80 | 0.9506 | 0.0137 | 0.0038 | 0.1108 | 0.0323 |
| 85 | 0.9606 | 0.0340 | 0.0053 | 0.1156 | 0.0322 |
| 90 | 0.9711 | 0.0207 | 0.0049 | 0.1181 | 0.0321 |
| 95 | 0.9825 | 0.0141 | 0.0058 | 0.1181 | 0.0320 |
| 100 | 0.9948 | 0.0147 | 0.0048 | 0.1181 | 0.0319 |

Main observation:

- \(C_L\) drops quickly after the first few saved points.
- The \(k=10\) mean \(C_L\) is a large outlier, but the median is already much smaller.
- From about \(k=25\) onward, median \(C_L\) is consistently very small.
- \(C_E\) does not show the same strong monotone decrease.

## 8. Ten Radius-Bin Summary

This is the radius view, using bins of \(r_k=\|\delta_k\|_2/\epsilon\).

| radius bin | median \(C_L\) | median \(C_E\) |
| --- | ---: | ---: |
| 0.0-0.1 | 0.0938 | 0.0456 |
| 0.1-0.2 | 0.0195 | 0.0406 |
| 0.2-0.3 | 0.0091 | 0.0319 |
| 0.3-0.4 | 0.0142 | 0.0325 |
| 0.4-0.5 | 0.0110 | 0.0280 |
| 0.5-0.6 | 0.0093 | 0.0247 |
| 0.6-0.7 | 0.0068 | 0.0270 |
| 0.7-0.8 | 0.0059 | 0.0262 |
| 0.8-0.9 | 0.0059 | 0.0341 |
| 0.9-1.0 | 0.0038 | 0.0332 |

The ten-bin view makes the scalar-loss trend clearer:

\[
\text{median } C_L:\quad 0.0938 \rightarrow 0.0195 \rightarrow 0.0091 \rightarrow \cdots \rightarrow 0.0038.
\]

This is stronger evidence than the three-bin early/middle/late table.

## 9. What This Experiment Proves And Does Not Prove

This experiment supports:

\[
\text{Along the PGD path, the scalar Loss3 landscape becomes more locally linear farther from } x_0.
\]

It does not by itself prove:

\[
\text{The full residual vector field } e(z)=f(z)-j(z) \text{ becomes uniformly more linear in all directions.}
\]

Why? Because \(L_3(z)=\|e(z)\|_2\) only observes residual magnitude. The residual vector can still rotate or bend internally while its norm behaves almost linearly.

The residual-map result is direction-dependent:

- random directions have relatively small \(C_E\) and often get smaller farther out;
- adversarial directions such as `grad` and `radial` can retain or even increase residual-map bending later.

So the safest statement is:

\[
\text{The scalar attack objective becomes locally more linear along the path,}
\]

but

\[
\text{the full residual geometry remains direction-dependent and is not globally proven affine.}
\]

## 10. Relation To The Bigger Claim

This finite-difference experiment is useful because it directly tests local bending using only three points. It does not require Hessians, Jacobians, or SVDs.

Its strongest contribution is to the scalar-objective story:

\[
\text{near } x_0: \text{ scalar Loss3 has stronger local bending;}
\]

\[
\text{farther out: scalar Loss3 behaves more like a locally linear function.}
\]

For the stronger geometric claim about the full residual map or the dominant Jacobian directions, the straight-line Jacobian/subspace rotation experiment remains stronger evidence.

## 11. Why This Does Not Contradict The Straight-Line Direction-Rotation Experiment

The finite-difference result and the earlier straight-line direction-rotation result are not contradictory. They are measuring different layers of the same object.

The finite-difference experiment shows:

\[
C_L \text{ decreases strongly along the path.}
\]

That means the scalar Loss3 objective,

\[
L_3(z)=\|e(z)\|_2,
\]

becomes more locally linear farther from \(x_0\).

At the same time, the residual-map score \(C_E\) does not decrease as cleanly. That means the full residual vector field,

\[
e(z)=f(z)-j(z),
\]

is not proven to become uniformly affine in all directions.

These two statements can both be true because \(L_3(z)\) only keeps the magnitude of \(e(z)\), while \(e(z)\) itself also contains direction, phase, and component information.

So it is possible that

\[
\|e(z)\|_2 \text{ changes almost linearly}
\]

while

\[
e(z) \text{ itself still rotates or bends as a vector field.}
\]

This also connects to the earlier gradient/direction experiments. The Loss3 gradient is

\[
\nabla_z L_3(z)
=
J_e(z)^\top
\frac{e(z)}{\|e(z)\|_2}.
\]

Therefore the attack gradient is not the residual vector \(e(z)\) itself. It is the normalized residual direction,

\[
\frac{e(z)}{\|e(z)\|_2},
\]

pulled back into input space through the residual Jacobian \(J_e(z)^\top\).

This gives three distinct layers:

\[
e(z)
\]

is the full residual vector field.

\[
L_3(z)=\|e(z)\|_2
\]

is the scalar residual norm / scalar attack loss.

\[
\nabla_z L_3(z)=J_e(z)^\top e(z)/\|e(z)\|_2
\]

is the loss-gradient direction that the attack actually follows.

The experiments fit together as follows:

1. The finite-difference \(C_L\) experiment shows that scalar Loss3 becomes more locally linear farther along the PGD path.

2. The finite-difference \(C_E\) result shows that the full residual vector map does not uniformly become more linear in every direction.

3. The straight-line gradient/Jacobian direction-rotation experiments show that attack-relevant directions, such as the Loss3 gradient or dominant Jacobian singular directions, become more stable farther away from the clean point.

So the shared conclusion is not:

\[
e(z) \text{ becomes globally linear in the far region.}
\]

That would be too strong for the current evidence.

The safer and more accurate conclusion is:

\[
\text{Along adversarial paths or adversarial rays, the scalar Loss3 landscape and attack-relevant dominant directions become more stable and more locally linear,}
\]

while

\[
\text{the complete residual vector geometry remains direction-dependent and can still be nonlinear.}
\]

This resolves the apparent tension: the scalar objective can become easier/more linear for the attack even when the underlying residual vector field has not become uniformly affine.

