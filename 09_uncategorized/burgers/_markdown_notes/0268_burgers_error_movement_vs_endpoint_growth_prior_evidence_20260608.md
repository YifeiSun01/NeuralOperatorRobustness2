# Burgers Error Movement Versus Endpoint Growth Prior Evidence

Date: 2026-06-08

## Scope

This note records the prior evidence for the distinction between:

\[
\Delta e = e(x+\delta)-e(x)
\]

and endpoint error growth:

\[
\|e(x+\delta)\|^2-\|e(x)\|^2.
\]

It was written after rechecking earlier Markdown records, not after running a
new attack or recomputing any Jacobian/SVD.

## Current Correlation Result Being Reworded

Observed from `docs/burgers_existing_svd_attack_correlation_sweep_20260608.md`
and `forensics/burgers_existing_svd_attack_correlation_key_summary_20260608.csv`:
existing Burgers SVD sample audits show positive correlations between
`error_spectral_norm = ||J_model - J_solver||_2` and endpoint attack damage
`attack_increase = final_loss - initial_loss`.

Key observed Pearson/Spearman correlations:

| data/SVD group | `||J_model-J_solver||_2` vs endpoint `attack_increase` |
|---|---:|
| second ns50 loss3 checkpoint series | `0.516 / 0.655` |
| second ns50 loss1 epoch2000 | `0.843 / 0.835` |
| second ns50 loss2 epoch0900 | `0.834 / 0.775` |
| round01 aligned final loss123 | `0.821 / 0.764` |
| round03 long-final loss123 | `0.850 / 0.824` |
| round03 final-extension loss123 | `0.833 / 0.808` |

Correct wording: these are correlations between error-Jacobian spectral norm
and endpoint attack damage. They are not direct measurements of the theorem
quantity `||e(x+delta)-e(x)||`.

## Algebraic Distinction

Let:

\[
b=e(x), \quad \Delta e=e(x+\delta)-e(x).
\]

Then:

\[
\|e(x+\delta)\|^2-\|e(x)\|^2
=
\|b+\Delta e\|^2-\|b\|^2
=
2\langle b,\Delta e\rangle+\|\Delta e\|^2.
\]

The theorem-side residual movement energy is:

\[
\|\Delta e\|^2.
\]

The endpoint attack damage includes the additional cross/outward term:

\[
2\langle b,\Delta e\rangle.
\]

Under the local affine approximation \(\Delta e \approx A\delta\), where
\(A=J_model-J_solver\), endpoint growth is:

\[
2\langle b,A\delta\rangle+\|A\delta\|^2.
\]

Therefore a direction can have large residual movement \(\|A\delta\|\) while
moving sideways or inward relative to the existing clean residual \(b\). In that
case endpoint error may grow weakly, or even decrease.

## Prior Experimental Evidence

Observed from `docs/outward_growth_direction_result_20260515.md`:

| direction set | local role | mismatch gain mean `||Av||` | outward component mean `<b/||b||, Av>` |
|---|---|---:|---:|
| `error_top` top-8 | large residual movement | `0.412169` | `0.0140571` |
| `error` rank-1 only | strongest singular direction of `A` | `0.868217` | `-0.00677836` |
| `outward_growth` | first-order growth of `||b+A delta||` | `0.368053` | `0.166514` |

Inference from this table: top error-Jacobian singular directions make the
residual vector move, but they barely move it outward in the current residual
direction. The first-order outward direction has smaller raw movement but much
larger endpoint-growth component.

Observed direction angles from
`forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`,
as recorded in `docs/outward_growth_direction_result_20260515.md` and
`docs/angle_experiment_inventory_20260516.md`:

- `outward_growth` versus `error` top-8 angle mean: `84.68 deg`.
- `outward_growth` versus `error` rank-1 angle mean: `90.87 deg`.

Inference: the first-order endpoint-growth direction \(A^T b\) is almost
orthogonal to the pure residual-movement/spectral direction.

Observed finite-difference validation from
`forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_summary.csv`:

| direction source | rho | actual endpoint growth mean | predicted growth mean | residual movement mean |
|---|---:|---:|---:|---:|
| `outward_growth` | `1e-4` | `0.167286` | `0.166514` | `0.368765` |
| `negative_outward_growth` | `1e-4` | `-0.165528` | `-0.166514` | `0.367704` |

Inference: the negative outward direction has almost the same residual movement
as the positive outward direction, but endpoint growth is negative. This is the
direct prior evidence for the user's concern: large
`||e(x+delta)-e(x)||` does not guarantee large endpoint
`||e(x+delta)||`.

Observed from `docs/angle_experiment_inventory_20260516.md` for the true
nonlinear endpoint-vs-movement gradients on the `loss3` path:

| k | true nonlinear endpoint-vs-movement angle |
|---:|---:|
| `5` | `45.06 deg` |
| `10` | `36.36 deg` |
| `25` | `22.33 deg` |
| `50` | `8.15 deg` |

Inference: early and mid attack paths have materially different endpoint and
residual-movement directions. They become closer later, after the accumulated
residual increment becomes large relative to the clean residual.

## Practical Consequence

It is plausible that, on the same saved attack deltas, the direct residual
change metric

\[
\|e(x+\delta)-e(x)\|
\]

could correlate more directly with `||J_model-J_solver||_2` than endpoint
`attack_increase` does. But this is not guaranteed for endpoint-PGD deltas,
because those deltas were optimized for endpoint loss, not necessarily for pure
residual movement.

Strict theorem validation should therefore compute, on the same saved attack
deltas and without recomputing SVD:

\[
\|e(x+\delta)-e(x)\|, \quad
\frac12\|e(x+\delta)-e(x)\|^2,
\quad
2\langle e(x), e(x+\delta)-e(x)\rangle.
\]

Then report correlations separately for:

- `error_spectral_norm` versus residual-change norm/energy;
- `error_spectral_norm` versus endpoint attack damage;
- residual-change energy versus outward/cross term;
- endpoint damage decomposition into cross term plus residual-change energy.

## Conclusion

Observed evidence supports the following careful statement:

`||J_model-J_solver||_2` is positively correlated with finite-budget endpoint
attack damage across the existing Burgers SVD/attack audits. However, the direct
spectral-norm theorem quantity is residual-field movement
`||e(x+delta)-e(x)||`, not endpoint damage. Prior outward-growth experiments
already showed that large residual movement can be nearly orthogonal to, or even
opposite from, endpoint error growth because the clean residual vector is
nonzero.
