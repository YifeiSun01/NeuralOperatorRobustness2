# Burgers SVD/Attack Correlation Metric Choice 20260608

## Question

Which attack-side quantity is mathematically closest to the local error-Jacobian spectral norm?

## Correction

The clean theorem in the error-operator notes is about **error-field movement**, not endpoint-loss growth. The directly spectral-norm-linked quantity is

```text
||E(a+h) - E(a)||
```

or, in squared-energy form,

```text
1/2 ||E(a+h) - E(a)||^2.
```

This is different from endpoint error growth:

```text
1/2 ||E(a+h)||^2 - 1/2 ||E(a)||^2.
```

The endpoint-growth quantity keeps the clean residual `E(a)` and therefore has the additional first-order residual term.

## Observed Source Documents

- `docs/error_operator_robustness_local_lipschitz_pgd_equivalence_20260608.md` defines the clean local theorem using `R_epsilon(a) = sup ||E(a+h)-E(a)||`, and states that `R_epsilon(a)/epsilon -> ||D E(a)||`. It separately lists practical residual energy loss `1/2||E(a+h)||^2` as a different quantity with an extra first-order term.
- `docs/loss_objective_direction_experiments_error_operator_framework_20260608.md` explicitly says under “Three Quantities That Must Not Be Mixed” that error-field movement `||E(a+h)-E(a)||^2 ≈ ||Ah||^2` is directly controlled by the local error Jacobian, while endpoint error energy `||E(a+h)||^2` contains `||e0||^2 + 2<e0,Ah> + ||Ah||^2`.
- `docs/delta_loss_formula_taxonomy_20260515.md` distinguishes “Error Field Movement” from “Error Norm Growth”. The former is `||e(x+delta)-e(x)||/||delta|| ≈ ||A delta||/||delta||` and is connected to `||A||`; the latter asks whether the residual moves outward in the current residual direction.
- `docs/burgers_p2q2_training_visualization_jacobian_summary_20260602.md` defines logged attack loss gain as `adversarial loss after attack - clean loss before attack`; that is useful logging, but it is not identical to the clean spectral-norm theorem quantity.

## Correct Metric Hierarchy

1. Best theoretical match to `||J_model - J_solver||_2`:

```text
residual_change_norm = ||E(a+h) - E(a)||
residual_change_energy = 1/2 ||E(a+h) - E(a)||^2
```

For small L2 perturbations, `sup residual_change_norm / epsilon` tends to the spectral norm.

2. Practical endpoint attack damage:

```text
attack_increase = final_loss - initial_loss
```

This is still an additive difference and is better than percentage growth for attack-damage reporting, but it is an endpoint error-growth metric, not the direct local Lipschitz/spectral-norm theorem metric.

3. Relative/percentage growth:

```text
(final_loss - initial_loss) / initial_loss
```

This can be a secondary normalized diagnostic, but it introduces an extra denominator and can become unstable when initial loss is very small. It is not the natural Taylor/small-epsilon spectral-norm object.

## Consequence For Recent SVD20 Correlations

The recent SVD20 correlation tables using `attack_increase = final_loss - initial_loss` should be described as correlations between error-Jacobian spectral norm and endpoint attack damage. They should not be overstated as the direct theorem quantity unless we also compute `||E(a+h)-E(a)||` or its squared energy on the same perturbations.
