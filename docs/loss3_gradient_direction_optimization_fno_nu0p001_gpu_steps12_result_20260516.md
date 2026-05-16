# Loss3 Gradient Direction Optimization Result - 2026-05-16

Status: completed for FNO / 1D Burgers `nu=0.001` with GPU-only execution.

## What Changed From The Candidate Sweep

The previous experiment tried a fixed bank of theoretically meaningful directions. This second version directly optimizes the direction `v` by projected gradient ascent on the unit sphere.

For each clean input `x`, radius `epsilon`, and objective, the optimized problem is:

```text
maximize objective(x, epsilon, v) subject to ||v||_2 = 1
delta = epsilon v
```

The script uses Adam ascent on `v`, then renormalizes `v` after every step. This is not model training; only the perturbation direction is updated.

## Objectives

```text
L_f = ||f(x+epsilon v)-f(x)|| / epsilon
L_j = ||j(x+epsilon v)-j(x)|| / epsilon
L_e = ||e(x+epsilon v)-e(x)|| / epsilon, e=f-j
G_e = (||e(x+epsilon v)||-||e(x)||) / epsilon
```

`L_f`, `L_j`, and `L_e` have local pullback eigenvector references: maximize `||Jv||^2 = v^T J^T J v`. The maximizer is the largest eigenvector of `J^T J`, equivalently the top right singular vector of `J`.

`G_e` is different. Its first-order expansion is:

```text
d/d epsilon ||e(x)+epsilon J_e v|| at epsilon=0
= <e(x)/||e(x)||, J_e v>
= <J_e^T e(x)/||e(x)||, v>
```

Therefore the local outward-growth direction is `normalize(J_e^T e(x)/||e(x)||)`, not the top eigenvector of `J_e^T J_e`.

## Run Settings

- Samples: `[0, 7, 40, 47, 115]`.
- Epsilons: `[0.0001, 0.001, 0.01, 0.1]`.
- Steps per start: `12`.
- Adam learning rate: `0.2`.
- Random starts per case: `1` plus analytic plus/minus starts.
- Output directory: `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12`.
- Runtime device: `Tesla V100-SXM2-32GB`.

## Per-Sample Value And Direction Results

Cell format: `optimized/reference; angle`. For `L_f`, `L_j`, and `L_e`,
`reference` is the clean local top singular value. For `G_e`, `reference` is
the clean local outward-growth rate `||J_e^T (e/||e||)||`, not a singular
value.

| objective | epsilon | idx0 | idx7 | idx40 | idx47 | idx115 |
| --- | ---: | --- | --- | --- | --- | --- |
| `L_f` | `0.0001` | `4.5037/4.5044; 4.83 deg` | `3.3279/3.3289; 1.67 deg` | `4.0636/4.0608; 1.41 deg` | `3.8713/3.8713; 3.54 deg` | `3.7063/3.7077; 1.37 deg` |
| `L_f` | `0.001` | `4.4990/4.5044; 9.39 deg` | `3.3275/3.3289; 1.56 deg` | `4.0608/4.0608; 0.51 deg` | `3.8653/3.8713; 6.42 deg` | `3.7075/3.7077; 0.51 deg` |
| `L_f` | `0.01` | `4.4981/4.5044; 10.74 deg` | `3.3285/3.3289; 1.20 deg` | `4.0599/4.0608; 0.42 deg` | `3.8696/3.8713; 1.85 deg` | `3.7051/3.7077; 0.93 deg` |
| `L_f` | `0.1` | `4.4682/4.5044; 8.00 deg` | `3.2994/3.3289; 1.26 deg` | `4.0044/4.0608; 1.56 deg` | `3.8140/3.8713; 10.17 deg` | `3.6840/3.7077; 2.52 deg` |
| `L_j` | `0.0001` | `4.8722/4.8751; 1.56 deg` | `3.4713/3.4742; 1.14 deg` | `4.1568/4.1568; 0.41 deg` | `3.8697/3.8842; 24.24 deg` | `4.0831/4.0841; 1.20 deg` |
| `L_j` | `0.001` | `4.8733/4.8751; 1.60 deg` | `3.4733/3.4742; 1.30 deg` | `4.1573/4.1568; 0.32 deg` | `3.8682/3.8842; 23.86 deg` | `4.0847/4.0841; 0.32 deg` |
| `L_j` | `0.01` | `4.8720/4.8751; 1.57 deg` | `3.4733/3.4742; 0.94 deg` | `4.1544/4.1568; 1.70 deg` | `3.8688/3.8842; 24.33 deg` | `4.0836/4.0841; 0.44 deg` |
| `L_j` | `0.1` | `4.8101/4.8751; 4.96 deg` | `3.4257/3.4742; 1.07 deg` | `4.0863/4.1568; 1.55 deg` | `3.8227/3.8842; 31.88 deg` | `4.0201/4.0841; 0.57 deg` |
| `L_e` | `0.0001` | `1.2059/1.2058; 1.00 deg` | `0.8403/0.8402; 2.31 deg` | `0.6169/0.6168; 1.44 deg` | `0.9489/0.9487; 1.48 deg` | `0.7303/0.7297; 1.52 deg` |
| `L_e` | `0.001` | `1.2056/1.2058; 1.13 deg` | `0.8400/0.8402; 3.01 deg` | `0.6167/0.6168; 1.92 deg` | `0.9483/0.9487; 1.41 deg` | `0.7293/0.7297; 1.33 deg` |
| `L_e` | `0.01` | `1.2057/1.2058; 1.27 deg` | `0.8397/0.8402; 3.69 deg` | `0.6178/0.6168; 5.83 deg` | `0.9492/0.9487; 1.17 deg` | `0.7273/0.7297; 1.04 deg` |
| `L_e` | `0.1` | `1.1272/1.2058; 12.64 deg` | `0.8213/0.8402; 25.26 deg` | `0.6167/0.6168; 5.40 deg` | `0.9194/0.9487; 5.50 deg` | `0.6518/0.7297; 5.96 deg` |
| `G_e` | `0.0001` | `0.2031/0.2049; 0.42 deg` | `0.1985/0.1951; 0.55 deg` | `0.1186/0.1180; 1.58 deg` | `0.1858/0.1840; 2.07 deg` | `0.1315/0.1306; 3.54 deg` |
| `G_e` | `0.001` | `0.2049/0.2049; 0.22 deg` | `0.1954/0.1951; 0.55 deg` | `0.1180/0.1180; 0.43 deg` | `0.1842/0.1840; 2.28 deg` | `0.1306/0.1306; 2.72 deg` |
| `G_e` | `0.01` | `0.2060/0.2049; 0.63 deg` | `0.1965/0.1951; 1.01 deg` | `0.1187/0.1180; 0.91 deg` | `0.1851/0.1840; 1.78 deg` | `0.1311/0.1306; 3.60 deg` |
| `G_e` | `0.1` | `0.2156/0.2049; 5.46 deg` | `0.2120/0.1951; 10.02 deg` | `0.1255/0.1180; 7.83 deg` | `0.1966/0.1840; 5.47 deg` | `0.1384/0.1306; 8.69 deg` |

## Table Definitions

The `optimized/reference; angle` cells should be read as follows:

```text
optimized = final objective value after gradient ascent on v
reference = clean local reference value
angle = arccos(abs(cos(v_grad, v_ref))) in degrees
```

For `L_f`, `L_j`, and `L_e`, the clean local reference value is the top
singular value of the corresponding Jacobian: `sigma_1(J_f)`, `sigma_1(J_j)`,
or `sigma_1(J_e)`. For these three objectives, the reference direction is the
top right singular vector, equivalently the largest eigenvector of `J^T J`.

For `G_e`, the reference is not a singular value. The local reference value is
`||J_e^T (e/||e||)||`, and the reference direction is
`normalize(J_e^T (e/||e||))`. This is the clean residual norm's outward-growth
direction.

The angle is computed per sample. It is not `arccos(mean cosine)`. Earlier
aggregate cosine summaries were useful as a quick check, but the per-sample
angles above are the primary record.

## L_j Top-2 Degeneracy Diagnostic

The large `L_j` angle at `idx47` is not evidence that gradient ascent failed.
It is caused by a near-degenerate solver Jacobian spectrum. At `idx47`, the
first two solver singular values are very close, so the objective is flat in the
top-2 singular subspace. A mixture of the first two singular directions can have
a large angle to `v1` while preserving almost the same `L_j` value.

Solver Jacobian top singular gaps:

| index | sigma1 | sigma2 | sigma2/sigma1 | relative gap |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `4.875141` | `4.518274` | `0.926799` | `7.32%` |
| `7` | `3.474184` | `2.882186` | `0.829601` | `17.04%` |
| `40` | `4.156828` | `2.710343` | `0.652022` | `34.80%` |
| `47` | `3.884161` | `3.796643` | `0.977468` | `2.25%` |
| `115` | `4.084114` | `2.479935` | `0.607215` | `39.28%` |

`idx47` optimized `L_j` direction decomposition into the solver SVD basis:

| epsilon | cos to v1 | cos to v2 | top-2 projection | value/reference |
| ---: | ---: | ---: | ---: | ---: |
| `1e-4` | `0.9118` | `0.4090` | `0.99936` | `3.8697/3.8842` |
| `1e-3` | `0.9145` | `0.4030` | `0.99940` | `3.8682/3.8842` |
| `1e-2` | `0.9112` | `0.4118` | `0.99992` | `3.8688/3.8842` |
| `1e-1` | `0.8491` | `0.5282` | `0.99998` | `3.8227/3.8842` |

This is the important interpretation: for near-degenerate cases, top-k subspace
alignment is more meaningful than top-1 vector alignment. At `idx47`, the
optimized direction is almost entirely in the solver top-2 plane, even though
its angle to the single top vector `v1` is large.

## Pullback Eigenvector Checks

| index | objective | eigenvalue | sigma^2 | abs cos eig/svd | outward formula/source cos |
| --- | --- | --- | --- | --- | --- |
| 0 | L_f | 20.29 | 20.29 | 1 |  |
| 0 | L_j | 23.77 | 23.77 | 1 |  |
| 0 | L_e | 1.454 | 1.454 | 1 |  |
| 0 | G_e | nan | nan | nan | 1 |
| 7 | L_f | 11.08 | 11.08 | 1 |  |
| 7 | L_j | 12.07 | 12.07 | 1 |  |
| 7 | L_e | 0.7059 | 0.7059 | 1 |  |
| 7 | G_e | nan | nan | nan | 1 |
| 40 | L_f | 16.49 | 16.49 | 1 |  |
| 40 | L_j | 17.28 | 17.28 | 1 |  |
| 40 | L_e | 0.3804 | 0.3804 | 1 |  |
| 40 | G_e | nan | nan | nan | 1 |
| 47 | L_f | 14.99 | 14.99 | 1 |  |
| 47 | L_j | 15.09 | 15.09 | 1 |  |
| 47 | L_e | 0.9 | 0.9 | 1 |  |
| 47 | G_e | nan | nan | nan | 1 |
| 115 | L_f | 13.75 | 13.75 | 1 |  |
| 115 | L_j | 16.68 | 16.68 | 1 |  |
| 115 | L_e | 0.5325 | 0.5325 | 1 |  |
| 115 | G_e | nan | nan | nan | 1 |

## Visualizations

![gradient vs candidate value ratio](../forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/figures/gradient_vs_candidate_value_ratio.png)

![gradient direction alignment](../forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/figures/gradient_direction_alignment.png)

![random start alignment](../forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/figures/random_start_alignment.png)

## Interpretation

This run directly optimizes the perturbation direction `v` by projected gradient
ascent. It is therefore the gradient-based version of the earlier candidate-bank
sweep, not another fixed-direction sweep.

Key observations recorded from the per-sample table:

- For `L_f`, the optimized values stay very close to the clean local reference
  values across all epsilons. Most direction angles are small; the larger angles
  at `idx0` and `idx47` still preserve nearly the same objective value.
- For `L_j`, four of the five samples align tightly with the top solver singular
  direction, usually around `0.3-1.7 deg` for `epsilon <= 1e-2`. The exception is
  `idx47`, where the top two solver singular values are nearly tied. The
  optimized direction rotates inside the top-2 plane, so the top-1 angle becomes
  `23.86-24.33 deg` at small epsilon and `31.88 deg` at `epsilon=0.1`, while the
  value remains close to the local reference.
- For `L_e`, the small-epsilon directions are very stable. At `epsilon <= 1e-2`,
  most samples stay close to the residual/error top singular direction. At
  `epsilon=0.1`, finite-radius drift becomes visible, especially `idx0` and
  `idx7`.
- For `G_e`, the small-epsilon directions align with the outward-growth formula
  direction, not with a top SVD direction. At `epsilon <= 1e-2`, all five samples
  remain close to the outward-growth reference. At `epsilon=0.1`, every sample
  moves farther away, which is consistent with leaving the purely local regime.

The `epsilon` trend is consistent with an epsilon-refinement or local-convergence
check: when epsilon is small, the optimized finite-difference direction agrees
with the clean local Jacobian/outward-growth reference. When epsilon is large
(`0.1`), nonlinear finite-radius effects become visible and the direction can
move away from the clean local direction.

## Clear Conclusion

The gradient experiment confirms the main small-epsilon conclusion. For FNO / 1D
Burgers / `nu=0.001`, gradient ascent on `v` recovers essentially the same local
objects that the previous candidate-bank sweep selected:

- `L_f` recovers the FNO Jacobian top-response direction.
- `L_j` recovers the solver Jacobian top-response direction except when the
  solver spectrum is nearly degenerate, as at `idx47`; there the right diagnostic
  is top-2 subspace alignment rather than top-1 vector angle.
- `L_e` recovers the residual/error Jacobian top-response direction at small
  epsilon.
- `G_e` recovers the residual outward-growth direction
  `normalize(J_e^T (e/||e||))`, not a top SVD direction.

The important nuance is the `L_j/idx47` case. Its first two solver singular
values satisfy `sigma2/sigma1 = 0.977468`, so the maximum direction is not sharp.
Gradient ascent can converge to a `v1/v2` mixture with a large angle to `v1`, but
because `sigma2` is almost as large as `sigma1`, the objective value barely
drops. This explains why top-1 angle alone can look bad while the optimized
value and top-2 subspace alignment are both excellent.

Observed evidence is stored in the raw CSVs and vector files under
`forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12`.
The key raw tables are `gradient_best_by_case.csv`, `gradient_run_summary.csv`,
`gradient_trajectory.csv`, `gradient_summary_by_objective_epsilon.csv`, and
`pullback_eigen_summary.csv`.

## Output Files

- `gradient_run_summary.csv`: every objective/index/epsilon/start final result.
- `gradient_trajectory.csv`: recorded optimization steps.
- `gradient_best_by_case.csv`: best overall and best random-start result for each case.
- `gradient_summary_by_objective_epsilon.csv`: aggregate table used above.
- `pullback_eigen_summary.csv`: `J^T J` eigenvector and outward-growth checks.
- `manifest.json`: reproducibility manifest with GPU evidence.
