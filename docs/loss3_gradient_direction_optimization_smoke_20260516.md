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

- Samples: `[0]`.
- Epsilons: `[0.001]`.
- Steps per start: `3`.
- Adam learning rate: `0.15`.
- Random starts per case: `1` plus analytic plus/minus starts.
- Output directory: `forensics/loss3_gradient_direction_optimization_20260516/smoke_gpu`.
- Runtime device: `Tesla V100-SXM2-32GB`.

## Aggregate Results

| objective | epsilon | grad/candidate | grad/local | cos ref | random cos ref |
| --- | --- | --- | --- | --- | --- |
| G_e | 0.001 | 0.9771 | 0.975 | 0.9781 | 0.9781 |
| L_e | 0.001 | 0.922 | 0.9221 | 0.7227 | 0.7227 |
| L_f | 0.001 | 0.9805 | 0.9805 | 0.8455 | 0.8455 |
| L_j | 0.001 | 0.9578 | 0.9579 | 0.8001 | 0.8001 |

## Pullback Eigenvector Checks

| index | objective | eigenvalue | sigma^2 | abs cos eig/svd | outward formula/source cos |
| --- | --- | --- | --- | --- | --- |
| 0 | L_f | 20.29 | 20.29 | 1 |  |
| 0 | L_j | 23.77 | 23.77 | 1 |  |
| 0 | L_e | 1.454 | 1.454 | 1 |  |
| 0 | G_e | nan | nan | nan | 1 |

## Visualizations

![gradient vs candidate value ratio](../forensics/loss3_gradient_direction_optimization_20260516/smoke_gpu/figures/gradient_vs_candidate_value_ratio.png)

![gradient direction alignment](../forensics/loss3_gradient_direction_optimization_20260516/smoke_gpu/figures/gradient_direction_alignment.png)

![random start alignment](../forensics/loss3_gradient_direction_optimization_20260516/smoke_gpu/figures/random_start_alignment.png)

## Interpretation

Observed evidence is in `gradient_best_by_case.csv`, `gradient_run_summary.csv`, and `pullback_eigen_summary.csv`.

The key comparison is whether the gradient-optimized direction has high cosine with the same local reference direction selected by the candidate-bank sweep. If `grad/candidate` is near `1` and `cos ref` is near `1`, the gradient optimizer has recovered the same local direction rather than merely selecting it from a prebuilt list.

For `L_f`, `L_j`, and `L_e`, the pullback eigenvector check should match the saved SVD direction up to sign. For `G_e`, the check is the outward formula direction, not a pullback eigenvector.

## Output Files

- `gradient_run_summary.csv`: every objective/index/epsilon/start final result.
- `gradient_trajectory.csv`: recorded optimization steps.
- `gradient_best_by_case.csv`: best overall and best random-start result for each case.
- `gradient_summary_by_objective_epsilon.csv`: aggregate table used above.
- `pullback_eigen_summary.csv`: `J^T J` eigenvector and outward-growth checks.
- `manifest.json`: reproducibility manifest with GPU evidence.
