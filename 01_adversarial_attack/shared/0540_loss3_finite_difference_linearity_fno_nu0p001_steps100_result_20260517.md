# Loss3 Finite-Difference Local Linearity Result

Date: 2026-05-17 UTC

## Scope

- experiment: symmetric finite-difference local linearity, Experiment 2
- model/task: FNO, 1D Burgers, `nu=0.001`
- path: existing `loss3_original_pgd` 100-step trajectory
- saved points: `k=0,5,10,...,100`
- samples: `0, 7, 40, 47, 115`
- radius: `rho=0.16 = 0.02 epsilon`
- directions: `grad`, `radial`, `step`, and `random_0..random_3`
- no Hessian, no dense Jacobian, no SVD

Output root:

`forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/`

## Scores

For scalar Loss3:

`C_L = |L(z+rho u)-2L(z)+L(z-rho u)| / (|L(z+rho u)-L(z-rho u)| + eps)`

For residual map `e(z)=f(z)-j(z)`:

`C_e = ||e(z+rho u)-2e(z)+e(z-rho u)|| / (||e(z+rho u)-e(z-rho u)|| + eps)`

Smaller means more locally linear along that direction at radius `rho`.

## Robust Aggregate By Path Bin

| path_bin | rho | n_points | delta_budget_ratio_mean | loss_linearity_score_mean | loss_linearity_score_median | loss_linearity_score_p90 | residual_linearity_score_mean | residual_linearity_score_median | residual_linearity_score_p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early | 0.16 | 156 | 0.0827 | 0.3314 | 0.0641 | 0.3232 | 0.0928 | 0.0439 | 0.2233 |
| late | 0.16 | 427 | 0.9314 | 0.0148 | 0.0046 | 0.0247 | 0.1240 | 0.0328 | 0.3512 |
| middle | 0.16 | 147 | 0.3964 | 0.0515 | 0.0109 | 0.0770 | 0.0942 | 0.0298 | 0.3154 |

## Robust Aggregate By Direction

| path_bin | direction_type | rho | n_points | loss_linearity_score_mean | loss_linearity_score_median | residual_linearity_score_mean | residual_linearity_score_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early | grad | 0.16 | 23 | 0.0609 | 0.0590 | 0.1895 | 0.1553 |
| early | radial | 0.16 | 23 | 0.0698 | 0.0372 | 0.1569 | 0.1684 |
| early | step | 0.16 | 18 | 0.0355 | 0.0320 | 0.1800 | 0.1709 |
| early | random_0 | 0.16 | 23 | 0.1855 | 0.0938 | 0.0344 | 0.0330 |
| early | random_1 | 0.16 | 23 | 0.0741 | 0.0347 | 0.0344 | 0.0352 |
| early | random_2 | 0.16 | 23 | 0.4080 | 0.2261 | 0.0351 | 0.0340 |
| early | random_3 | 0.16 | 23 | 1.4217 | 0.1107 | 0.0383 | 0.0370 |
| middle | grad | 0.16 | 21 | 0.0132 | 0.0048 | 0.2198 | 0.2224 |
| middle | radial | 0.16 | 21 | 0.0096 | 0.0097 | 0.1592 | 0.1351 |
| middle | step | 0.16 | 21 | 0.0075 | 0.0049 | 0.1957 | 0.1836 |
| middle | random_0 | 0.16 | 21 | 0.1605 | 0.0413 | 0.0195 | 0.0175 |
| middle | random_1 | 0.16 | 21 | 0.0374 | 0.0143 | 0.0219 | 0.0187 |
| middle | random_2 | 0.16 | 21 | 0.0898 | 0.0230 | 0.0213 | 0.0204 |
| middle | random_3 | 0.16 | 21 | 0.0423 | 0.0222 | 0.0223 | 0.0199 |
| late | grad | 0.16 | 61 | 0.0038 | 0.0018 | 0.3141 | 0.3306 |
| late | radial | 0.16 | 61 | 0.0032 | 0.0026 | 0.3000 | 0.3288 |
| late | step | 0.16 | 61 | 0.0144 | 0.0094 | 0.1709 | 0.1243 |
| late | random_0 | 0.16 | 61 | 0.0104 | 0.0079 | 0.0204 | 0.0159 |
| late | random_1 | 0.16 | 61 | 0.0492 | 0.0146 | 0.0190 | 0.0187 |
| late | random_2 | 0.16 | 61 | 0.0114 | 0.0047 | 0.0221 | 0.0234 |
| late | random_3 | 0.16 | 61 | 0.0110 | 0.0031 | 0.0216 | 0.0161 |

## Main Interpretation

The scalar Loss3 score strongly decreases along the path:

- mean `C_L`: early `0.3314`, middle `0.0515`, late `0.0148`;
- median `C_L`: early `0.0641`, middle `0.0109`, late `0.0046`.

So the scalar objective `L3(z)=||f(z)-j(z)||` becomes much more locally linear under this finite-difference test.

The residual-map score is mixed:

- all-direction mean `C_e`: early `0.0928`, middle `0.0942`, late `0.1240`;
- all-direction median `C_e`: early `0.0439`, middle `0.0298`, late `0.0328`.

This does not support a blanket claim that the residual map itself becomes more affine in every direction.

Direction-specific behavior is the key:

- random directions: residual-map `C_e` generally decreases from about `0.034-0.038` early to about `0.019-0.022` late;
- gradient/radial directions: residual-map `C_e` increases, for example `grad` mean `0.1895 -> 0.2198 -> 0.3141`, and `radial` mean `0.1569 -> 0.1592 -> 0.3000`;
- step direction is mixed: mean `0.1800 -> 0.1957 -> 0.1709`.

The safest conclusion is:

`Along the Loss3 PGD path, the scalar Loss3 objective becomes much more locally linear farther out.  The residual vector map does not uniformly become more linear; it becomes more linear in generic random directions, but adversarial grad/radial directions retain or even increase residual-map bending.`

This is still useful, but it proves a different and narrower statement than the straight-line Jacobian/subspace experiment.  It is strong evidence for scalar objective local linearization, not a complete proof that the full residual geometry is globally less nonlinear in all important directions.

## Output Files

- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_by_sample_k_direction.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_aggregate_by_k_direction.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_aggregate_by_bin_direction.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_aggregate_by_k.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_aggregate_by_bin.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/finite_difference_linearity_aggregate_by_direction.csv`
- `forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment2_finite_difference_linearity_steps100/summary.md`
- `finite_difference_linearity_robust_by_bin.csv`
- `finite_difference_linearity_robust_by_bin_direction.csv`

Analysis script:

`tools/run_loss3_finite_difference_linearity.py`
