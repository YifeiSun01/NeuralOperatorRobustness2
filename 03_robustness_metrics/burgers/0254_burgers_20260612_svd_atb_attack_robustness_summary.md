# Burgers 2026-06-12 Full-Jacobian SVD, ATB, And Attack Robustness Summary

This note records the full 1024 by 1024 Jacobian SVD results, biased local ATB direction analysis, direction-angle comparisons, and correlation rankings with adversarial attack loss increase.

## Definitions

Let the local model-solver residual be defined at a sample x:

A = J_model(x) - J_solver(x)

b = f_model(x) - f_solver(x)

ATB = A transpose times b

The pure SVD local sensitivity asks for the top singular direction of A. The actual local affine attack growth for squared residual includes both a quadratic term and a linear bias term:

loss increase approximately equals delta transpose A transpose A delta plus 2 times ATB transpose delta.

Therefore, when the attack radius is very small, the linear term dominates and the leading direction is closer to ATB than to the pure top singular vector.

## Source Files

SVD/ATB root:

`forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611`

Important CSV files:

- `biased_direction_metrics.csv`
- `biased_direction_correlations.csv`
- `biased_direction_quantile_summary.csv`
- `direction_pairwise_angle_summary_20260612.csv`
- `atb_svd_loss_increase_correlation_ranking_20260612.csv`

The analysis set has 100 model-sample pairs, from 25 samples times 4 models.

## Direction Angles

Angles are computed with absolute cosine and reported in degrees from 0 to 90 degrees. This avoids the arbitrary sign of singular vectors.

| pair of directions | mean angle | median angle | q25 | q75 |
|---|---:|---:|---:|---:|
| attack delta vs pure top SVD | 77.60 | 83.56 | 77.16 | 86.30 |
| attack delta vs ATB/outward | 37.70 | 38.11 | 31.98 | 44.44 |
| pure top SVD vs ATB/outward | 75.20 | 80.40 | 72.38 | 85.58 |
| SVD angle minus ATB angle to attack | 39.90 | 43.65 | 33.12 | 51.00 |

Conclusion: the actual nonlinear attack delta is much closer to ATB than to the pure top SVD direction. On average, ATB is about 39.9 degrees closer to the true attack delta.

## Correlation With Attack Loss Increase

The target is attack loss increase, equal to final attacked MSE minus initial MSE.

| metric | Pearson | Spearman |
|---|---:|---:|
| ATB norm | 0.745 | 0.851 |
| ATB norm squared | 0.638 | 0.851 |
| ATB local gain at epsilon | 0.626 | 0.872 |
| Frobenius norm of A | 0.666 | 0.810 |
| clean residual norm b | 0.638 | 0.797 |
| spectral norm sigma1 of A | 0.637 | 0.778 |
| SVD local gain at epsilon | 0.577 | 0.780 |

Ranking interpretation:

- By Pearson, ATB norm is the strongest single scalar predictor of attack loss increase.
- By Spearman, ATB local gain at epsilon is the strongest ranked predictor, but it is derived from the same ATB/outward local direction.
- Spectral norm sigma1 of A is positively correlated, but it is weaker than ATB-based indicators.

## Correlation With Final Attacked MSE

| metric | Pearson | Spearman |
|---|---:|---:|
| ATB norm | 0.828 | 0.887 |
| ATB norm squared | 0.724 | 0.887 |
| ATB local gain at epsilon | 0.662 | 0.895 |
| Frobenius norm of A | 0.747 | 0.851 |
| clean residual norm b | 0.743 | 0.840 |
| spectral norm sigma1 of A | 0.724 | 0.816 |
| SVD local gain at epsilon | 0.681 | 0.818 |

Conclusion: ATB-based metrics also explain final attacked MSE better than pure SVD spectral norm.

## Why ATB Norm And ATB Norm Squared Have The Same Spearman

ATB norm squared is a monotone transformation of ATB norm, so ranks do not change. Spearman correlation is rank-based, so the Spearman values are identical. Pearson changes because squaring changes linear scale.

## Main Scientific Interpretation

The pure SVD spectral norm measures worst-case local sensitivity of the error Jacobian A. That is useful, but the actual attack objective is not just the pure quadratic norm of A delta. It also includes the current residual b. Because the local loss-increase gradient is controlled by ATB, the ATB direction and ATB magnitude explain the observed attack growth more directly.

This supports a possible paper contribution: for solver-consistent neural-operator robustness, a biased local residual-gradient metric based on ATB can be a more faithful and cheaper robustness indicator than pure SVD spectral norm alone.

## Caveat

These SVD/ATB metrics were computed for the 25-sample full-Jacobian workflow and its model set. The latest comparison-dense P2Q2 plots use the latest retrained checkpoints from 2026-06-12. If exact SVD/ATB statistics are required for those exact retrained checkpoints, the full SVD workflow should be rerun with those checkpoint paths.
