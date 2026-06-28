# Burgers SVD Residual-Change Correlation

Date: 2026-06-08

## Scope

This experiment recomputes the metric paired with existing Burgers SVD20 attack
deltas. It does not rerun PGD and does not recompute any Jacobian/SVD.

The previous correlation used endpoint attack damage:

\[
\|e(x+\delta)\|^2-\|e(x)\|^2.
\]

This run also computes the direct residual-change quantity:

\[
\Delta e=e(x+\delta)-e(x),
\quad
\|\Delta e\|,
\quad
\|\Delta e\|^2.
\]

It also decomposes endpoint growth as:

\[
\|e(x+\delta)\|^2-\|e(x)\|^2
=
2\langle e(x),\Delta e\rangle+\|\Delta e\|^2.
\]

## GPU / Run Evidence

Observed GPU path:

- `nvidia-smi`: `forensics/burgers_existing_svd_residual_change_nvidia_smi_20260608.txt`
- PyTorch: `2.8.0+cu126`
- CUDA runtime reported by PyTorch: `12.6`
- GPU: Tesla V100-SXM2-32GB
- Compute capability: `7.0`
- PyTorch arch list includes `sm_70`
- CUDA matmul sanity: `128.0`

Script:

- `tools/compute_burgers_svd20_residual_change_correlation_20260608.py`

Aggregate outputs:

- `forensics/burgers_existing_svd_residual_change_correlation_key_summary_20260608.csv`
- `forensics/burgers_existing_svd_residual_change_correlation_key_summary_20260608_all_summary.csv`

Per-root outputs written beside the existing attack outputs:

- `error_svd_residual_change_joined_rows.csv`
- `error_svd_residual_change_correlation_summary.csv`
- `residual_change_metrics_config.json`

## All Samples, All Models

Observed correlations below use
`x = ||J_model-J_solver||_2`.

| data/SVD group | n | endpoint growth Pearson/Spearman | residual-change MSE Pearson/Spearman | residual-change RMS Pearson/Spearman | cross term Pearson/Spearman |
|---|---:|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 120 | `0.516 / 0.655` | `0.555 / 0.677` | `0.610 / 0.677` | `-0.808 / -0.571` |
| second ns50 loss1 epoch2000 | 40 | `0.843 / 0.835` | `0.850 / 0.820` | `0.826 / 0.820` | `-0.681 / -0.274` |
| second ns50 loss2 epoch0900 | 40 | `0.834 / 0.775` | `0.841 / 0.769` | `0.805 / 0.769` | `-0.701 / -0.489` |
| round01 aligned final loss123 | 80 | `0.821 / 0.764` | `0.854 / 0.749` | `0.833 / 0.749` | `-0.785 / -0.488` |
| round03 long-final loss123 | 80 | `0.850 / 0.824` | `0.875 / 0.826` | `0.883 / 0.826` | `-0.513 / -0.425` |
| round03 final-extension loss123 | 80 | `0.833 / 0.808` | `0.867 / 0.802` | `0.862 / 0.802` | `-0.447 / -0.320` |

Inference: replacing endpoint growth with residual-change MSE increases Pearson
correlation in all six all-sample/all-model groups. Spearman rank correlation is
similar, but not uniformly higher.

## Generalization Samples, All Models

Observed correlations below use
`x = ||J_model-J_solver||_2`.

| data/SVD group | n | endpoint growth Pearson/Spearman | residual-change MSE Pearson/Spearman | residual-change RMS Pearson/Spearman | cross term Pearson/Spearman |
|---|---:|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 60 | `0.406 / 0.480` | `0.454 / 0.530` | `0.506 / 0.530` | `-0.814 / -0.723` |
| second ns50 loss1 epoch2000 | 20 | `0.827 / 0.783` | `0.836 / 0.792` | `0.815 / 0.792` | `-0.681 / -0.507` |
| second ns50 loss2 epoch0900 | 20 | `0.817 / 0.780` | `0.827 / 0.800` | `0.791 / 0.800` | `-0.678 / -0.562` |
| round01 aligned final loss123 | 40 | `0.806 / 0.737` | `0.849 / 0.700` | `0.820 / 0.700` | `-0.831 / -0.660` |
| round03 long-final loss123 | 40 | `0.807 / 0.803` | `0.847 / 0.816` | `0.851 / 0.816` | `-0.495 / -0.634` |
| round03 final-extension loss123 | 40 | `0.769 / 0.781` | `0.829 / 0.777` | `0.812 / 0.777` | `-0.468 / -0.518` |

Inference: on generalization rows, residual-change MSE again gives higher
Pearson correlation in all six groups. Spearman is higher in four groups and
slightly lower in two.

## Decomposition Check

The endpoint decomposition was numerically exact up to floating-point error.
Maximum absolute decomposition error per root was between `1.86e-09` and
`7.45e-09`.

Mean values over joined rows:

| data/SVD group | endpoint growth mean | residual-change MSE mean | cross term mean | clean/change cosine mean |
|---|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | `0.005622` | `0.005882` | `-0.000260` | `-0.015` |
| second ns50 loss1 epoch2000 | `0.010607` | `0.010779` | `-0.000172` | `0.167` |
| second ns50 loss2 epoch0900 | `0.011414` | `0.011594` | `-0.000179` | `0.128` |
| round01 aligned final loss123 | `0.006199` | `0.006705` | `-0.000506` | `0.103` |
| round03 long-final loss123 | `0.011160` | `0.012182` | `-0.001023` | `0.064` |
| round03 final-extension loss123 | `0.011221` | `0.011901` | `-0.000681` | `0.107` |

Inference: the cross term is often negative on average. This confirms the older
outward-growth evidence: the saved endpoint-PGD deltas can produce large
residual movement while not always moving outward along the clean residual.
Endpoint growth remains positive because the residual-change energy term is
larger than the negative cross term.

## Relation To Prior Direction Experiments

Observed from `docs/outward_growth_direction_result_20260515.md` and
`docs/angle_experiment_inventory_20260516.md`:

- top error-Jacobian singular directions had high residual movement but small or
  negative outward component;
- the first-order outward-growth direction `A^T b` was nearly orthogonal to the
  top residual-movement singular directions;
- the negative outward-growth direction had large residual movement but negative
  endpoint growth;
- true nonlinear endpoint-vs-movement gradient angles on the `loss3` path were
  `45.06 deg` at `k=5`, `36.36 deg` at `k=10`, `22.33 deg` at `k=25`, and
  `8.15 deg` at `k=50`.

Inference: the current residual-change correlation result is consistent with
that prior direction evidence. The spectral norm is more directly tied to
residual movement than endpoint growth, but endpoint growth is still correlated
because the residual-change energy is a large component of the endpoint damage
under these fixed-budget attacks.

## Conclusion

Observed answer to the question: yes, using residual-change MSE
`||e(x+delta)-e(x)||^2` makes the Pearson correlation with
`||J_model-J_solver||_2` higher in every tested SVD/attack group. The improvement
is modest, not a dramatic jump, because endpoint growth was already strongly
driven by residual-change energy. Rank correlation is not uniformly higher.

The cleanest statement is:

`||J_model-J_solver||_2` correlates strongly with endpoint attack damage, and it
correlates slightly more directly with the residual-change metric that matches
the local spectral-norm theorem. The difference between the two is exactly the
cross/outward term involving the nonzero clean residual.
