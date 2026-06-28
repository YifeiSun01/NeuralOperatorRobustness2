# Burgers Biased Local Attack Direction Analysis - 2026-06-11

This analyzes the completed full `1024 x 1024` Burgers error-Jacobian SVD run and asks whether the local attack direction should be the pure top singular vector or the biased finite-radius direction caused by the clean residual `b`.

## Local Math

Let `b = f_model(x) - f_solver(x)` and `A = J_model(x) - J_solver(x)`.  The pure SVD movement metric solves

```text
max_{||delta|| <= r} ||A delta||^2
```

so its direction is the top right singular vector of `A`.  But the local affine attack-loss model is

```text
max_{||delta|| <= r} ||b + A delta||^2 - ||b||^2
= max delta^T A^T A delta + 2 (A^T b)^T delta.
```

Therefore as `r -> 0`, the direction tends to `A^T b / ||A^T b||`, not the top singular vector.  At finite radius, the trust-region KKT form is

```text
delta(mu) = (mu I - A^T A)^(-1) A^T b,  mu > lambda_max(A^T A),  ||delta(mu)|| = r.
```

This script computes all three directions: `top_svd`, `outward=A^T b`, and `affine_trust_region` at the attack radius.

## Files

- source SVD root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611`
- output directory: `forensics/burgers_wideparam_loss3targeted_biased_local_direction_20260611`
- per-pair metrics: `forensics/burgers_wideparam_loss3targeted_biased_local_direction_20260611/biased_direction_metrics.csv`
- correlations: `forensics/burgers_wideparam_loss3targeted_biased_local_direction_20260611/biased_direction_correlations.csv`
- summary JSON: `forensics/burgers_wideparam_loss3targeted_biased_local_direction_20260611/biased_direction_summary.json`

## Headline Numbers

- pairs analyzed: `12`
- mean abs angle between top SVD direction and `A^T b`: `80.777` degrees
- mean abs cosine, final nonlinear attack delta vs top SVD: `0.1537`
- mean abs cosine, final nonlinear attack delta vs `A^T b`: `0.7730`
- mean abs cosine, final nonlinear attack delta vs finite-radius affine direction: `0.1679`

## Per Model Mean Direction Agreement

| model | angle SVD vs A^T b deg | attack cos SVD | attack cos A^T b | attack cos affine eps |
|---|---:|---:|---:|---:|
| baseline | 80.7953 | 0.313126 | 0.741439 | 0.340124 |
| loss1 | 78.2556 | 0.154677 | 0.80578 | 0.182628 |
| loss2 | 81.2142 | 0.106322 | 0.807138 | 0.0980208 |
| loss3 | 82.8418 | 0.0406373 | 0.737664 | 0.0507659 |

## Compact Per Pair Table

| sample | model | sigma1 | angle_svd_vs_Atb_deg | cos_attack_svd | cos_attack_Atb | cos_attack_affine | actual_growth | svd_gain | affine_gain |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 2.36802 | 86.4495 | 0.0316196 | 0.832821 | 0.0796711 | 0.0333194 | 0.0811511 | 0.0813243 |
| 0 | loss1 | 3.02591 | 74.8808 | 0.237807 | 0.712122 | 0.249762 | 0.0108567 | 0.13306 | 0.1331 |
| 0 | loss2 | 3.07656 | 79.9648 | 0.0647416 | 0.741123 | 0.0471148 | 0.00766046 | 0.137337 | 0.137406 |
| 0 | loss3 | 1.06904 | 84.4276 | 0.00441821 | 0.760831 | 0.0477549 | 0.00632803 | 0.0165894 | 0.0166483 |
| 1 | baseline | 1.71212 | 75.9939 | 0.126244 | 0.758792 | 0.152445 | 0.00437249 | 0.0428246 | 0.0428718 |
| 1 | loss1 | 1.99931 | 78.5502 | 0.125629 | 0.869603 | 0.168368 | 0.00754316 | 0.0581908 | 0.0582664 |
| 1 | loss2 | 1.97707 | 78.0083 | 0.171126 | 0.794364 | 0.18647 | 0.00674436 | 0.0567107 | 0.0567302 |
| 1 | loss3 | 1.70113 | 77.882 | 0.0919157 | 0.753722 | 0.103135 | 0.00396794 | 0.0419253 | 0.0419342 |
| 2 | baseline | 1.95388 | 79.9425 | 0.781514 | 0.632704 | 0.788255 | 0.00870896 | 0.0552268 | 0.0552365 |
| 2 | loss1 | 1.22419 | 81.3359 | 0.100595 | 0.835616 | 0.129753 | 0.00271454 | 0.0217529 | 0.0217728 |
| 2 | loss2 | 1.21643 | 85.6695 | 0.0830966 | 0.885926 | 0.0604777 | 0.00235327 | 0.0213816 | 0.0213939 |
| 2 | loss3 | 0.420253 | 86.2159 | 0.0255779 | 0.69844 | 0.00140744 | 0.00146813 | 0.00255264 | 0.00255496 |

## All-Pair Correlations

| x | y | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| `error_spectral_norm` | `attack_loss_growth_abs` | 12 | 0.476677 | 0.846154 |
| `error_spectral_norm` | `attack_final_mse` | 12 | 0.512738 | 0.881119 |
| `bias_gradient_norm` | `attack_loss_growth_abs` | 12 | 0.749221 | 0.888112 |
| `bias_gradient_norm` | `attack_final_mse` | 12 | 0.779204 | 0.923077 |
| `svd_local_gain_eps_mse` | `attack_loss_growth_abs` | 12 | 0.430769 | 0.846154 |
| `svd_local_gain_eps_mse` | `attack_final_mse` | 12 | 0.469991 | 0.881119 |
| `outward_local_gain_eps_mse` | `attack_loss_growth_abs` | 12 | 0.75101 | 0.832168 |
| `outward_local_gain_eps_mse` | `attack_final_mse` | 12 | 0.77824 | 0.874126 |
| `affine_local_gain_eps_mse` | `attack_loss_growth_abs` | 12 | 0.431559 | 0.846154 |
| `affine_local_gain_eps_mse` | `attack_final_mse` | 12 | 0.47077 | 0.881119 |

## Interpretation

The angle between top SVD and `A^T b` is the clean way to measure your concern: if it is large, the infinitesimal attack direction is not the top singular vector direction.  The finite-radius affine direction is the more mathematically faithful local comparator for attack loss growth because it includes both the linear `2 b^T A delta` term and the quadratic `delta^T A^T A delta` term.

For the actual 15-step nonlinear attack, the final delta can still differ from all three local directions because the model and PDE solver Jacobians change along the path.  So the right conclusion is not that SVD is wrong; it is that pure SVD measures local error sensitivity, while biased affine direction measures local attack-loss increase.
