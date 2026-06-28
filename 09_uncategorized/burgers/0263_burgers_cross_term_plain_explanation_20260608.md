# Burgers Cross Term Plain Explanation

Date: 2026-06-08

## Question

Clarify what the endpoint-growth decomposition means, what the cross term is,
which quantities were correlated, and where the concrete data lives.

## Definitions

For one input sample:

- `E(x)` is the clean model-solver error vector: model prediction minus solver output at the clean input.
- `E(x+delta)` is the attacked endpoint model-solver error vector.
- `DeltaE = E(x+delta) - E(x)` is how much the error vector moved because of the perturbation.

The endpoint growth metric is:

\[
\|E(x+\delta)\|^2 - \|E(x)\|^2.
\]

This means: after attack, did the final model-solver error norm become larger
than the clean model-solver error norm, and by how much?

## Decomposition

Because:

\[
E(x+\delta)=E(x)+\Delta E,
\]

we have:

\[
\|E(x+\delta)\|^2 - \|E(x)\|^2
= \|\Delta E\|^2 + 2\langle E(x),\Delta E\rangle.
\]

The cross term is:

\[
2\langle E(x),\Delta E\rangle.
\]

In the code/CSV this is `cross_term_mse`, using the same per-grid-point mean
convention as the MSE losses.

## Plain Meaning

`||DeltaE||^2` asks: how much did the error vector move?

The cross term asks: did that movement point along the old clean error direction
or against it?

- positive cross term: the error movement goes outward along the old error, so endpoint loss grows more;
- negative cross term: the error movement partly goes against the old error, so it cancels part of the movement energy;
- near-zero cross term: the error movement is mostly sideways relative to the old error.

Therefore:

\[
endpoint\ growth - residual\ movement\ energy = cross\ term.
\]

Equivalently:

\[
cross\ term = (final\ loss - initial\ loss) - \|E(x+\delta)-E(x)\|^2.
\]

## Which Correlation Was Computed

The x-variable was:

\[
\|J_{model}-J_{solver}\|_2,
\]

stored as `error_spectral_norm` in the joined CSVs.

The y-variable for the cross-term correlation was:

\[
2\langle E(x), E(x+\delta)-E(x)\rangle,
\]

stored as `cross_term_mse`.

So the phrase "cross term is negatively correlated with spectral norm" means:
across the joined model/sample rows, larger `error_spectral_norm` often comes
with a more negative `cross_term_mse`.

This does not mean the spectral norm is negative. It means the extra endpoint
correction term tends to subtract more when the error-Jacobian spectral norm is
larger.

## Concrete Correlation Data

Source CSV:

- `forensics/burgers_existing_svd_residual_change_correlation_key_summary_20260608.csv`

All samples, all models:

| data/SVD group | n | `error_spectral_norm` vs `cross_term_mse` Pearson/Spearman | endpoint growth Pearson/Spearman | residual-change MSE Pearson/Spearman |
|---|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 120 | `-0.808 / -0.571` | `0.516 / 0.655` | `0.555 / 0.677` |
| second ns50 loss1 epoch2000 | 40 | `-0.681 / -0.274` | `0.843 / 0.835` | `0.850 / 0.820` |
| second ns50 loss2 epoch0900 | 40 | `-0.701 / -0.489` | `0.834 / 0.775` | `0.841 / 0.769` |
| round01 aligned final loss123 | 80 | `-0.785 / -0.488` | `0.821 / 0.764` | `0.854 / 0.749` |
| round03 long-final loss123 | 80 | `-0.513 / -0.425` | `0.850 / 0.824` | `0.875 / 0.826` |
| round03 final-extension loss123 | 80 | `-0.447 / -0.320` | `0.833 / 0.808` | `0.867 / 0.802` |

Generalization rows, all models:

| data/SVD group | n | `error_spectral_norm` vs `cross_term_mse` Pearson/Spearman | endpoint growth Pearson/Spearman | residual-change MSE Pearson/Spearman |
|---|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 60 | `-0.814 / -0.723` | `0.406 / 0.480` | `0.454 / 0.530` |
| second ns50 loss1 epoch2000 | 20 | `-0.681 / -0.507` | `0.827 / 0.783` | `0.836 / 0.792` |
| second ns50 loss2 epoch0900 | 20 | `-0.678 / -0.562` | `0.817 / 0.780` | `0.827 / 0.800` |
| round01 aligned final loss123 | 40 | `-0.831 / -0.660` | `0.806 / 0.737` | `0.849 / 0.700` |
| round03 long-final loss123 | 40 | `-0.495 / -0.634` | `0.807 / 0.803` | `0.847 / 0.816` |
| round03 final-extension | 40 | `-0.468 / -0.518` | `0.769 / 0.781` | `0.829 / 0.777` |

## Per-Sample Data Location

Per-sample values are in each root's joined rows, for example:

- `forensics/burgers_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_round03_long_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_run2_ns50_p2q2_loss3_checkpoint_series_svd20_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`

Important columns:

- `error_spectral_norm`: the SVD spectral norm `||J_model-J_solver||_2`;
- `endpoint_growth_mse`: endpoint growth, `final_loss - initial_loss`;
- `residual_change_mse`: `||DeltaE||^2`;
- `cross_term_mse`: `2<E(x), DeltaE>`;
- `endpoint_decomp_error`: numerical check that `endpoint_growth_mse = residual_change_mse + cross_term_mse`.

## Short Interpretation

The spectral norm is more directly connected to `residual_change_mse`, because
both concern movement of the error field. Endpoint growth is residual movement
plus the cross term. Since the cross term is often negative and negatively
correlated with spectral norm, it partially cancels the residual movement energy
in the endpoint metric.
