# Burgers Actual PGD Cross Term vs Top-Singular-Direction Cross Term

Date: 2026-06-08

## Question

Clarify whether the negative endpoint cross term means that the PGD attack is
moving along the error-Jacobian top singular vector, and compare actual PGD
endpoint-optimized perturbations with a clean-point top singular vector
counterfactual.

## Definitions

Let:

```text
E(x) = model(x) - solver(x)
DeltaE_actual = E(x + delta_pgd) - E(x)
A = J_model(x) - J_solver(x)
```

For the actual saved PGD endpoint attack:

```text
actual_cross = 2 mean(E(x) * DeltaE_actual)
actual_residual_change = mean(DeltaE_actual^2)
actual_endpoint_growth = actual_residual_change + actual_cross
```

For the top-singular-direction counterfactual:

```text
v1 = top right singular vector of A
delta_top = ||delta_pgd||_2 * v1
DeltaE_top_linear = A delta_top
top1_cross_raw = 2 mean(E(x) * DeltaE_top_linear)
```

The sign of `v1` in an SVD file is arbitrary. Therefore, for an endpoint-growth
counterfactual, the correct best-sign quantity is:

```text
top1_cross_outward = abs(top1_cross_raw)
```

If someone simply uses the raw saved sign of `v1`, the cross term can be positive
or negative for meaningless sign-convention reasons.

## Main Answer

The actual PGD perturbation is not the same object as the top singular vector.
Actual PGD optimizes nonlinear endpoint loss through the model/solver and ends
with `DeltaE_actual`. The top singular vector is a clean-point linearized
direction that maximizes `||A delta||`, not necessarily endpoint growth.

The new comparison shows:

- actual PGD cross term is often near zero and can be negative, especially on
  generalization rows;
- top-1 raw cross term is about half positive and half negative because the SVD
  sign is arbitrary;
- top-1 outward-sign cross term is positive by construction, but small relative
  to the enormous top-1 residual movement;
- top-1 output direction is mostly close to orthogonal to `E(x)`: mean abs
  cosine is only `0.0936` all rows and `0.0775` on generalization rows.

So the correct interpretation is **not**:

> top singular direction would necessarily have a more negative cross term.

The correct interpretation is:

> top singular direction creates much larger linearized residual movement, but
> its output movement is mostly not aligned with the clean residual. Endpoint
> PGD chooses a nonlinear perturbation that is not top-1; on generalization rows
> its actual residual movement still often has a mildly negative orientation
> term, but that term only partially offsets the positive movement energy.

## Observed Artifacts

Source and output files:

- script: `tools/compute_burgers_actual_vs_top_singular_cross_term_20260608.py`
- rows: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/actual_vs_top1_cross_term_rows.csv`
- summary: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/actual_vs_top1_cross_term_summary.csv`
- correlations: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/actual_vs_top1_cross_term_correlations.csv`
- manifest: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/manifest.json`
- GPU preflight: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/gpu_preflight.json`
- nvidia-smi: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/nvidia_smi.txt`
- previous direction rows used as source: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/attack_delta_vs_error_svd_direction_rows.csv`

The run matched all saved rows:

```text
n_rows = 440
n_missing = 0
```

The check `||A v1|| - sigma1` was numerically tiny:

```text
mean abs error = 8.26e-10
max abs error = 2.43e-08
```

## Aggregate Numbers

All `440` rows:

```text
actual_cross mean/median = -5.045636e-04 / 7.221332e-06
actual_cross negative fraction = 0.479545
actual residual-clean cosine mean/median = 0.072723 / 0.018151

top1_raw_cross mean/median = 5.876126e-05 / -1.646525e-07
top1_raw_cross negative fraction = 0.502273

top1_outward_cross mean/median = 1.157287e-03 / 6.770594e-05
top1 output abs cosine with E(x) mean/median = 0.093563 / 0.051896

top1 linear residual-change mean/median = 1.007458e-01 / 1.607166e-02
actual residual-change mean/median = 9.236001e-03 / 2.592878e-03
```

Generalization `220` rows:

```text
actual_cross mean/median = -1.000210e-03 / -6.494438e-05
actual_cross negative fraction = 0.627273
actual residual-clean cosine mean/median = -0.011381 / -0.067133

top1_raw_cross mean/median = 1.100690e-04 / -1.475462e-06
top1_raw_cross negative fraction = 0.522727

top1_outward_cross mean/median = 2.215496e-03 / 2.951988e-04
top1 output abs cosine with E(x) mean/median = 0.077477 / 0.051896

top1 linear residual-change mean/median = 1.866151e-01 / 8.740677e-02
actual residual-change mean/median = 1.517049e-02 / 6.293141e-03
```

Mean-level cancellation size:

```text
all rows: actual_cross_mean / actual_residual_change_mean = -5.46%
generalization rows: actual_cross_mean / actual_residual_change_mean = -6.59%
```

Per-row cross/residual ratios:

```text
all rows actual cross/residual mean/median = 0.008141 / 0.001774
generalization actual cross/residual mean/median = -0.012990 / -0.013220

top1 outward cross/residual mean/median = 0.010205 / 0.005582 all
top1 outward cross/residual mean/median = 0.009201 / 0.005279 generalization
```

## Interpretation

Observed evidence says actual endpoint PGD does **not** make `DeltaE_actual`
strongly outward on generalization rows. The actual cross term is often slightly
negative there.

But this does not mean the attack is failing to increase endpoint loss. The
endpoint loss can still grow because:

```text
actual_residual_change > |actual_cross|
```

In mean terms on generalization rows:

```text
actual_residual_change_mean ~= 0.01517
actual_cross_mean ~= -0.00100
endpoint_growth_mean ~= 0.01417
```

So the cross term cancels a small-to-moderate part of the movement energy; it
does not dominate it.

The top singular vector counterfactual gives a different lesson. With the sign
chosen outward, the top-1 cross term is positive, not negative. But the top-1
output direction has tiny alignment with `E(x)` and its main effect is the huge
linearized movement energy `||A delta_top||^2`.

Therefore, for paper wording:

> The error-Jacobian spectral norm controls the largest possible local residual
> movement. The realized endpoint PGD perturbation is not the top singular
> direction; it is a nonlinear high-singular-subspace perturbation. Endpoint
> growth differs from residual movement by an orientation term relative to the
> clean residual, and on generalization rows this term is often mildly negative,
> partially canceling but not overturning the residual movement.

## Caveat

`top1_*` metrics are clean-point linearized counterfactuals using saved
Jacobians and the same input L2 norm as the actual saved PGD delta. They are not
new nonlinear attacks along `v1`. A direct nonlinear top-1 perturbation test
would require evaluating `E(x +/- ||delta|| v1)` through the model and solver.
