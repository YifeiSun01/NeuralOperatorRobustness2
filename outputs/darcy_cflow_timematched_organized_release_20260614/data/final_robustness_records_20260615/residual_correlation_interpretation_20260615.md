# Darcy CFlow Residual Correlation Interpretation - 20260615

This file mirrors the dedicated docs note:

`docs/darcy_cflow_residual_correlation_interpretation_20260615.md`

Summary:

- The small-epsilon derivation predicts attack loss increase should be closest
  to residual `J_res^T error`.
- The final attack50 table is finite-step, not infinitesimal.
- In all-25 x 7 residual correlations, residual sigma1 has the highest Pearson
  correlation with loss increase, while residual JT norm has the stronger rank
  interpretation among the direct loss-increase pairs.
- This does not contradict the derivation; it means attack50 can exploit
  high-gain residual singular directions.
- Loss3 remains the strongest overall model: closest model-solver top-k
  subspaces, smallest residual operator metrics, and lowest attack loss
  increase mean.

Key numbers:

| pair | Pearson | Spearman |
|---|---:|---:|
| loss increase vs residual JT norm | 0.504257 | 0.502891 |
| loss increase vs residual sigma1 | 0.561659 | 0.438184 |
| loss increase vs residual error L2 | 0.534601 | 0.495813 |
| residual JT norm vs residual sigma1 | 0.909114 | 0.945159 |

Loss3 residual means:

| metric | value |
|---|---:|
| residual sigma1 mean | 0.00170213 |
| residual error L2 mean | 0.0334209 |
| residual JT norm mean | 4.7257e-05 |
| attack loss increase mean | 1.7162e-06 |

Main wording:

> The residual Jacobian analysis supports Loss3 robustness. Loss3 has the
> closest model-solver local singular subspaces and the smallest residual
> operator metrics. The only unexpected feature is that, under the finite
> attack50 protocol, attack loss increase correlates slightly more strongly in
> Pearson correlation with residual spectral norm than with residual
> `J_res^T error`. This does not contradict the infinitesimal derivation:
> the derivation predicts `J_res^T error` dominance in the small-epsilon
> first-order regime, whereas attack50 is a finite-step optimization that can
> exploit high-gain residual singular directions.

