# Burgers Error-Operator SVD And Attack-Damage Paper Record

Date: 2026-06-08

## Purpose

This note consolidates the paper-useful theory, data, experiment results, and
wording from the Burgers error-operator/SVD/attack-damage discussion and the
new residual-change correlation computation.

It is intended as a writing record for the paper, not as a replacement for the
raw artifacts. Source files and exact CSV paths are listed at the end.

## Executive Thesis

For the Burgers neural operator experiments, the model-solver error Jacobian

\[
J_e(x) = J_{model}(x) - J_{solver}(x)
\]

has a strong empirical relationship with finite-budget adversarial damage. The
relationship becomes slightly more direct when the attack metric is changed from
endpoint error growth

\[
\|e(x+\delta)\|^2 - \|e(x)\|^2
\]

to residual-field movement

\[
\|e(x+\delta)-e(x)\|^2.
\]

This matches the local Lipschitz/SVD theorem: the spectral norm of \(J_e\)
controls infinitesimal movement of the error field, not endpoint error growth by
itself.

The endpoint attack objective remains useful, but it includes an additional
cross/outward term involving the existing clean residual. That term explains why
large residual movement need not always mean endpoint error grows in the same
way.

## Notation

Let:

\[
e(x) = M_\theta(x)-S(x)
\]

where \(M_\theta\) is the neural operator and \(S\) is the numerical solver.
Let:

\[
b=e(x), \quad \Delta e=e(x+\delta)-e(x), \quad A=J_e(x).
\]

Local linearization gives:

\[
\Delta e \approx A\delta.
\]

The SVD quantity used in the experiments is the spectral norm:

\[
\|J_{model}-J_{solver}\|_2 = \|A\|_2.
\]

## Mathematical Distinction

The theorem-side residual movement metric is:

\[
\|e(x+\delta)-e(x)\|^2 = \|\Delta e\|^2.
\]

Endpoint attack damage is:

\[
\|e(x+\delta)\|^2 - \|e(x)\|^2.
\]

Expanding endpoint attack damage gives:

\[
\|b+\Delta e\|^2 - \|b\|^2
= 2\langle b,\Delta e\rangle + \|\Delta e\|^2.
\]

Under the local affine approximation:

\[
\|b+A\delta\|^2 - \|b\|^2
= 2\langle b,A\delta\rangle + \|A\delta\|^2.
\]

So the two quantities differ by the cross/outward term:

\[
2\langle b,\Delta e\rangle.
\]

This term can be positive, small, or negative. Therefore a perturbation can move
the error vector a lot while moving sideways or inward relative to the clean
residual direction.

## What The Previous Direction Experiment Showed

The older outward-growth experiment separated two local objectives:

1. residual movement: maximize \(\|A v\|\);
2. endpoint outward growth: maximize \(\langle b/\|b\|, A v\rangle\), whose
   first-order direction is \(A^T b\).

Observed direction-response table:

| direction set | local role | mismatch gain mean `||Av||` | outward component mean `<b/||b||, Av>` |
|---|---|---:|---:|
| `error_top` top-8 | large residual movement | `0.412169` | `0.0140571` |
| `error` rank-1 only | strongest singular direction of `A` | `0.868217` | `-0.00677836` |
| `outward_growth` | first-order growth of `||b+A delta||` | `0.368053` | `0.166514` |

Interpretation: the top singular direction of the error Jacobian can have very
large residual movement but almost no outward endpoint growth. The rank-1 error
singular direction even has a slightly negative outward component.

Observed direction angles:

| comparison | observed mean angle |
|---|---:|
| `outward_growth` vs `error` top-8 | `84.68 deg` |
| `outward_growth` vs `error` rank-1 | `90.87 deg` |

Interpretation: the first-order endpoint-growth direction \(A^T b\) is almost
orthogonal to the pure residual-movement/SVD direction.

Observed finite-difference validation:

| direction source | rho | actual endpoint growth mean | predicted growth mean | residual movement mean |
|---|---:|---:|---:|---:|
| `outward_growth` | `1e-4` | `0.167286` | `0.166514` | `0.368765` |
| `negative_outward_growth` | `1e-4` | `-0.165528` | `-0.166514` | `0.367704` |

Interpretation: the negative outward direction has almost the same residual
movement as the positive outward direction, but endpoint growth is negative.
This is direct evidence that large \(\|e(x+\delta)-e(x)\|\) does not guarantee
large endpoint \(\|e(x+\delta)\|\).

Observed true nonlinear endpoint-vs-movement gradient angles on the `loss3`
path:

| k | true nonlinear endpoint-vs-movement angle |
|---:|---:|
| `5` | `45.06 deg` |
| `10` | `36.36 deg` |
| `25` | `22.33 deg` |
| `50` | `8.15 deg` |

Interpretation: early and mid attack paths show materially different endpoint
and residual-movement directions. They become closer later as the accumulated
residual increment becomes large relative to the clean residual.

## Previous Endpoint Attack-Damage Correlations

Before the residual-change recomputation, the main metric was endpoint attack
damage:

\[
\|e(x+\delta)\|^2 - \|e(x)\|^2.
\]

Observed all-sample/all-model correlations between
`||J_model-J_solver||_2` and endpoint attack damage:

| data/SVD group | Pearson | Spearman |
|---|---:|---:|
| second ns50 loss3 checkpoint series | `0.516` | `0.655` |
| second ns50 loss1 epoch2000 | `0.843` | `0.835` |
| second ns50 loss2 epoch0900 | `0.834` | `0.775` |
| round01 aligned final loss123 | `0.821` | `0.764` |
| round03 long-final loss123 | `0.850` | `0.824` |
| round03 final-extension loss123 | `0.833` | `0.808` |

Interpretation: across existing SVD/attack audits, error-Jacobian spectral norm
is strongly positively associated with endpoint attack damage. This is strong
empirical evidence for a robustness connection, but it is not yet the direct
local theorem metric because endpoint damage contains the clean-residual cross
term.

## New Residual-Change Correlation Experiment

New script:

- `tools/compute_burgers_svd20_residual_change_correlation_20260608.py`

The computation reused saved `final_delta_by_sample.npz` files from six existing
SVD20 P2Q2 attack roots. It did not rerun PGD and did not recompute Jacobian/SVD.
For each saved attack delta it computed:

\[
\Delta e=e(x+\delta)-e(x),
\quad
\|\Delta e\|,
\quad
\|\Delta e\|^2,
\quad
2\langle e(x),\Delta e\rangle.
\]

GPU path was verified before the run:

| item | observed value |
|---|---|
| GPU | Tesla V100-SXM2-32GB |
| PyTorch | `2.8.0+cu126` |
| PyTorch CUDA | `12.6` |
| compute capability | `7.0` |
| arch list | includes `sm_70` |
| CUDA matmul sanity | `128.0` |

Aggregate output files:

- `forensics/burgers_existing_svd_residual_change_correlation_key_summary_20260608.csv`
- `forensics/burgers_existing_svd_residual_change_correlation_key_summary_20260608_all_summary.csv`

Per-root outputs:

- `error_svd_residual_change_joined_rows.csv`
- `error_svd_residual_change_correlation_summary.csv`
- `residual_change_metrics_config.json`

## Endpoint Growth Versus Residual Change: All Samples

Observed correlations below use `x = ||J_model-J_solver||_2`.
Each table entry is `Pearson / Spearman`.

| data/SVD group | n | endpoint growth | residual-change MSE `||Delta e||^2` | residual-change RMS `||Delta e||` | cross term |
|---|---:|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 120 | `0.516 / 0.655` | `0.555 / 0.677` | `0.610 / 0.677` | `-0.808 / -0.571` |
| second ns50 loss1 epoch2000 | 40 | `0.843 / 0.835` | `0.850 / 0.820` | `0.826 / 0.820` | `-0.681 / -0.274` |
| second ns50 loss2 epoch0900 | 40 | `0.834 / 0.775` | `0.841 / 0.769` | `0.805 / 0.769` | `-0.701 / -0.489` |
| round01 aligned final loss123 | 80 | `0.821 / 0.764` | `0.854 / 0.749` | `0.833 / 0.749` | `-0.785 / -0.488` |
| round03 long-final loss123 | 80 | `0.850 / 0.824` | `0.875 / 0.826` | `0.883 / 0.826` | `-0.513 / -0.425` |
| round03 final-extension loss123 | 80 | `0.833 / 0.808` | `0.867 / 0.802` | `0.862 / 0.802` | `-0.447 / -0.320` |

Observed answer: replacing endpoint growth with residual-change MSE increases
Pearson correlation in every tested group. The increase is modest because
endpoint growth was already strongly driven by residual-change energy. Spearman
rank correlation is similar but not uniformly higher.

## Endpoint Growth Versus Residual Change: Generalization Rows

Observed correlations below use `x = ||J_model-J_solver||_2`.
Each table entry is `Pearson / Spearman`.

| data/SVD group | n | endpoint growth | residual-change MSE `||Delta e||^2` | residual-change RMS `||Delta e||` | cross term |
|---|---:|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 60 | `0.406 / 0.480` | `0.454 / 0.530` | `0.506 / 0.530` | `-0.814 / -0.723` |
| second ns50 loss1 epoch2000 | 20 | `0.827 / 0.783` | `0.836 / 0.792` | `0.815 / 0.792` | `-0.681 / -0.507` |
| second ns50 loss2 epoch0900 | 20 | `0.817 / 0.780` | `0.827 / 0.800` | `0.791 / 0.800` | `-0.678 / -0.562` |
| round01 aligned final loss123 | 40 | `0.806 / 0.737` | `0.849 / 0.700` | `0.820 / 0.700` | `-0.831 / -0.660` |
| round03 long-final loss123 | 40 | `0.807 / 0.803` | `0.847 / 0.816` | `0.851 / 0.816` | `-0.495 / -0.634` |
| round03 final-extension loss123 | 40 | `0.769 / 0.781` | `0.829 / 0.777` | `0.812 / 0.777` | `-0.468 / -0.518` |

Observed answer: on generalization rows, residual-change MSE again gives higher
Pearson correlation in all six groups. Spearman is higher in four groups and
slightly lower in two.

## Decomposition Check

The decomposition

\[
endpoint\ growth = \|\Delta e\|^2 + 2\langle e(x),\Delta e\rangle
\]

was numerically exact up to floating-point error. Maximum absolute decomposition
error per root was between `1.86e-09` and `7.45e-09`.

Mean values over joined rows:

| data/SVD group | endpoint growth mean | residual-change MSE mean | cross term mean | clean/change cosine mean |
|---|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | `0.005622` | `0.005882` | `-0.000260` | `-0.015` |
| second ns50 loss1 epoch2000 | `0.010607` | `0.010779` | `-0.000172` | `0.167` |
| second ns50 loss2 epoch0900 | `0.011414` | `0.011594` | `-0.000179` | `0.128` |
| round01 aligned final loss123 | `0.006199` | `0.006705` | `-0.000506` | `0.103` |
| round03 long-final loss123 | `0.011160` | `0.012182` | `-0.001023` | `0.064` |
| round03 final-extension loss123 | `0.011221` | `0.011901` | `-0.000681` | `0.107` |

Interpretation: the cross term is often negative on average. Endpoint growth
remains positive because the residual-change energy term is larger than the
negative cross term. This directly matches the older negative-direction
validation: residual movement and outward endpoint growth are related but not
identical.

## Paper-Ready Claims

A careful main claim:

> The spectral norm of the model-solver error Jacobian is a local measure of
> how strongly input perturbations can move the model-solver residual field. In
> Burgers experiments, this spectral norm is strongly positively correlated with
> finite-budget endpoint adversarial damage, and correlates even more directly
> in Pearson correlation with the residual-change metric that matches the local
> theorem.

A precise theorem-side claim:

> The local Lipschitz/SVD object \(\|J_{model}-J_{solver}\|_2\) controls
> infinitesimal residual-field movement \(\|e(x+h)-e(x)\|\), not endpoint error
> growth alone.

A precise endpoint-attack claim:

> Endpoint adversarial damage decomposes into residual-field movement energy
> plus a clean-residual cross term. The cross term can be negative, explaining
> why large residual movement need not always produce equally large endpoint
> error growth.

A data-backed empirical claim:

> Across six existing Burgers SVD20/P2Q2 attack audits, replacing endpoint
> growth with residual-change MSE increases the Pearson correlation with
> \(\|J_{model}-J_{solver}\|_2\) in every tested group. Spearman correlations are
> similar but not uniformly higher.

## Claims To Avoid Or Qualify

Do not say:

> The SVD spectral norm directly equals adversarial endpoint loss growth.

Better:

> The SVD spectral norm is directly tied to residual-field movement, while
> endpoint loss growth additionally depends on the current clean residual
> direction.

Do not say:

> A large \(\|e(x+\delta)-e(x)\|\) always means endpoint error increases.

Better:

> A large residual change can move the residual vector outward, sideways, or
> inward relative to the clean residual. The endpoint effect is decided by the
> decomposition \(\|\Delta e\|^2+2\langle e(x),\Delta e\rangle\).

Do not use percentage growth as the primary theorem metric. Percentage growth
adds an extra denominator involving clean error, can become unstable when clean
error is small, and is not the Taylor/SVD object.

## Suggested Paper Tables Or Figures

1. Direction-response table showing `error_top`, `error rank-1`, and
   `outward_growth` mismatch gain versus outward component.
2. Endpoint-vs-residual-movement decomposition equation and a small schematic
   showing \(b\), \(\Delta e\), and the outward component.
3. Correlation table comparing endpoint growth and residual-change MSE against
   \(\|J_{model}-J_{solver}\|_2\).
4. Cross-term table showing that the cross term is often negative on average.
5. True nonlinear endpoint-vs-movement angle table across attack steps.

## Source Artifacts

Theory and prior direction evidence:

- `docs/error_operator_robustness_local_lipschitz_pgd_equivalence_20260608.md`
- `docs/loss_objective_direction_experiments_error_operator_framework_20260608.md`
- `docs/delta_loss_formula_taxonomy_20260515.md`
- `docs/outward_growth_direction_result_20260515.md`
- `docs/angle_experiment_inventory_20260516.md`
- `docs/burgers_error_movement_vs_endpoint_growth_prior_evidence_20260608.md`

Endpoint correlation and residual-change correlation records:

- `docs/burgers_existing_svd_attack_correlation_sweep_20260608.md`
- `docs/burgers_svd_attack_correlation_metric_choice_20260608.md`
- `docs/burgers_svd_residual_change_correlation_20260608.md`
- `forensics/burgers_existing_svd_attack_correlation_key_summary_20260608.csv`
- `forensics/burgers_existing_svd_residual_change_correlation_key_summary_20260608.csv`
- `forensics/burgers_existing_svd_residual_change_correlation_key_summary_20260608_all_summary.csv`

Scripts:

- `tools/run_burgers_svd20_p2q2_attack_correlation_20260608.py`
- `tools/run_burgers_existing_svd_attack_correlations_20260608.sh`
- `tools/refresh_burgers_existing_svd_attack_correlation_joins_20260608.sh`
- `tools/compute_burgers_svd20_residual_change_correlation_20260608.py`

Per-root residual-change outputs:

- `forensics/burgers_run2_ns50_p2q2_loss3_checkpoint_series_svd20_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_run2_ns50_p2q2_loss3_checkpoint_series_svd20_attack_correlation_20260608/error_svd_residual_change_correlation_summary.csv`
- `forensics/burgers_run2_ns50_p2q2_loss1_epoch2000_svd20_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_run2_ns50_p2q2_loss1_epoch2000_svd20_attack_correlation_20260608/error_svd_residual_change_correlation_summary.csv`
- `forensics/burgers_run2_ns50_p2q2_loss2_epoch900_svd20_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_run2_ns50_p2q2_loss2_epoch900_svd20_attack_correlation_20260608/error_svd_residual_change_correlation_summary.csv`
- `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_correlation_summary.csv`
- `forensics/burgers_round03_long_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_round03_long_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_correlation_summary.csv`
- `forensics/burgers_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_joined_rows.csv`
- `forensics/burgers_svd20_p2q2_attack_correlation_20260608/error_svd_residual_change_correlation_summary.csv`

## Remaining Work

1. If the paper needs a clean theorem-validation table, report residual-change
   metrics separately from endpoint-damage metrics.
2. If a full first-root generalization SVD exists on another machine or R2, join
   it to the same residual-change analysis before making a first-root claim.
3. For larger finite epsilon, treat SVD as a local diagnostic and endpoint PGD as
   a finite-radius stress test. Agreement is strong evidence; disagreement is
   not automatically a contradiction.
