# Darcy CFlow Loss Increase, J^T Error Norm, And Spectral Norm Correlations - 2026-06-15

Observed from:

- Source table: `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv`
- Generated correlation table:
  `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_lossincrease_jt_spectral_correlations_20260615.csv`

This uses the final seven-model Darcy CFlow run on
`generalization_datasets_darcy_binary_loss3targeted_20260611`. It does not use
the older `lossdrop50_selected_20260607` data.

## Metrics

- `attack_loss_increase`: 50-step attack loss increase on the same fixed sample.
- `jt_error_l2_norm`: `||J^T error||_2` for the trained model Jacobian.
- `block2_sigma1`: exact block/2 projected model-Jacobian spectral norm.
- `sigma_input_right`: full-input one-step sigma estimate initialized from the
  lifted block/2 top right singular vector.

## Overall Correlations

All rows below use the 25 fixed SVD/Jacobian samples per model unless otherwise
noted.

| scope | n | x | y | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---|---|---:|---:|---:|---:|
| all 7 models | 175 | loss increase | J^T error norm | 0.382521 | 1.7503e-07 | 0.429304 | 3.06525e-09 |
| all 7 models | 175 | loss increase | block/2 spectral norm | -0.319092 | 1.6775e-05 | -0.215947 | 0.00410232 |
| all 7 models | 175 | loss increase | full-input sigma estimate | -0.318688 | 1.72154e-05 | -0.211796 | 0.00489736 |
| all 7 models | 175 | J^T error norm | block/2 spectral norm | 0.357818 | 1.16499e-06 | 0.408939 | 1.92496e-08 |
| all 7 models | 175 | J^T error norm | full-input sigma estimate | 0.351658 | 1.82429e-06 | 0.404174 | 2.90785e-08 |
| all 7 models | 175 | block/2 spectral norm | full-input sigma estimate | 0.999172 | 1.67664e-242 | 0.998990 | 4.68971e-235 |

Generalization-only rows:

| scope | n | x | y | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---|---|---:|---:|---:|---:|
| generalization only | 147 | loss increase | J^T error norm | 0.0649105 | 0.434741 | 0.0952595 | 0.251087 |
| generalization only | 147 | loss increase | block/2 spectral norm | -0.816533 | 2.04238e-36 | -0.726549 | 2.11787e-25 |
| generalization only | 147 | loss increase | full-input sigma estimate | -0.822020 | 2.79435e-37 | -0.719609 | 9.82365e-25 |
| generalization only | 147 | J^T error norm | block/2 spectral norm | 0.058778 | 0.479459 | 0.133322 | 0.107434 |
| generalization only | 147 | J^T error norm | full-input sigma estimate | 0.0443474 | 0.593787 | 0.124247 | 0.133777 |

## Interpretation

Observed:

- `loss increase` and `J^T error norm` have a moderate positive correlation when
  all 175 rows are pooled.
- `loss increase` and spectral norm are negatively correlated when all seven
  models are pooled, especially on the generalization-only subset.
- `block2_sigma1` and `sigma_input_right` are almost identical as rankings
  because the latter is initialized from the lifted block/2 top direction.

Inference:

- The negative pooled correlation between loss increase and spectral norm is a
  cross-model effect. Loss3 has the largest spectral norm but much smaller clean
  residual and `J^T error`, and it also has the smallest attack loss increase.
- Spectral norm alone is not a reliable explanation of robustness in this final
  Darcy table. `J^T error norm` is more aligned with attack loss increase, but
  even that relationship weakens in the generalization-only pooled subset.

## Per-Model Primary Correlations

| model | n | pair | Pearson r | Spearman rho |
|---|---:|---|---:|---:|
| baseline | 25 | loss increase vs J^T error norm | 0.557450 | -0.189231 |
| baseline | 25 | loss increase vs block/2 spectral norm | 0.378930 | 0.0838462 |
| baseline | 25 | J^T error norm vs block/2 spectral norm | 0.418586 | 0.422308 |
| loss1 | 25 | loss increase vs J^T error norm | 0.556789 | 0.105385 |
| loss1 | 25 | loss increase vs block/2 spectral norm | 0.646814 | 0.030000 |
| loss1 | 25 | J^T error norm vs block/2 spectral norm | 0.731020 | 0.672308 |
| loss2 | 25 | loss increase vs J^T error norm | 0.543386 | 0.111538 |
| loss2 | 25 | loss increase vs block/2 spectral norm | 0.668495 | -0.0130769 |
| loss2 | 25 | J^T error norm vs block/2 spectral norm | 0.846612 | 0.824615 |
| loss3 | 25 | loss increase vs J^T error norm | 0.368319 | 0.371538 |
| loss3 | 25 | loss increase vs block/2 spectral norm | -0.132198 | -0.117692 |
| loss3 | 25 | J^T error norm vs block/2 spectral norm | 0.604355 | 0.558462 |
| Physics Loss | 25 | loss increase vs J^T error norm | 0.609199 | 0.181538 |
| Physics Loss | 25 | loss increase vs block/2 spectral norm | 0.617535 | -0.0592308 |
| Physics Loss | 25 | J^T error norm vs block/2 spectral norm | 0.767345 | 0.706923 |
| random clean | 25 | loss increase vs J^T error norm | 0.504632 | 0.122308 |
| random clean | 25 | loss increase vs block/2 spectral norm | 0.597206 | 0.00153846 |
| random clean | 25 | J^T error norm vs block/2 spectral norm | 0.724027 | 0.728462 |
| random solver | 25 | loss increase vs J^T error norm | 0.593815 | 0.0592308 |
| random solver | 25 | loss increase vs block/2 spectral norm | 0.513267 | -0.103846 |
| random solver | 25 | J^T error norm vs block/2 spectral norm | 0.755110 | 0.709231 |

