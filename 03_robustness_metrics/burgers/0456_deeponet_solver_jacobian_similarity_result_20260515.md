# DeepONet vs Solver Local Jacobian Similarity Result

Date: 2026-05-15

## Question

This is the DeepONet counterpart of the FNO-vs-solver local-Jacobian
experiment.  It uses the same five test initial conditions, but switches the
Burgers viscosity to the DeepONet model setting:

```text
nu: 0.01
sample indices: 0, 7, 40, 47, 115
```

The compared local linear maps are:

```text
J_d = DeepONet local Jacobian
J_j = solver/oracle local Jacobian
J_e = J_d - J_j
```

## Setting

```text
model: DeepONet / default-net
solver/oracle: JAX/Exponax Burgers solver
nu: 0.01
nx: 1024
t_final: 1.0
dt: 0.001
domain: 2.0
sample indices: 0, 7, 40, 47, 115
Jacobian shape: 1024 x 1024
device: cuda
DeepONet Jacobian: explicit row-wise autograd
solver Jacobian: explicit row-wise VJP/autograd through JAX bridge
```

Script:

```text
tools/analyze_deeponet_solver_jacobian_similarity.py
```

Output directory:

```text
forensics/deeponet_solver_jacobian_similarity_20260515/
```

Main generated files:

```text
summary.md
aggregate_direction_comovement_summary.csv
aggregate_singular_value_summary.csv
aggregate_principal_angles_summary.csv
all_direction_comovement_table.csv
all_singular_vector_overlap.csv
all_fourier_gain_comparison.csv
index_*/...
```

## Aggregate Singular Values

Values are mean +/- std across the five sampled indices.

| Jacobian | spectral norm | Frobenius norm | effective rank | top-8 energy | top-32 energy |
|---|---:|---:|---:|---:|---:|
| `J_d` DeepONet | 7.202 +/- 0.702 | 11.680 +/- 0.722 | 7.043 +/- 0.738 | 0.943 +/- 0.009 | 1.000 +/- 0.000 |
| `J_j` solver | 1.373 +/- 0.096 | 2.262 +/- 0.079 | 4.747 +/- 0.410 | 0.995 +/- 0.001 | 1.000 +/- 0.000 |
| `J_e = J_d - J_j` | 7.100 +/- 0.696 | 11.479 +/- 0.726 | 7.077 +/- 0.725 | 0.941 +/- 0.009 | 1.000 +/- 0.000 |

Interpretation:

- DeepONet has a much larger local spectral norm than the `nu=0.01` solver:
  about `7.20` vs `1.37`.
- The residual/error Jacobian is almost as large as DeepONet itself:
  spectral norm `7.10`, not a small residual.
- This is the opposite of the FNO-vs-solver `nu=0.001` result, where the model
  and solver leading Jacobians were close and the residual was smaller.

## Top Singular Vector Frequency

The largest DeepONet right singular directions are high-frequency.  The solver
top singular directions are low-frequency/smooth.

Top-1 right singular vector metrics:

| sample index | DeepONet `hi128` | DeepONet zero crossings | solver `hi128` | solver zero crossings | error `hi128` | error zero crossings |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.770 | 508 | 1.78e-17 | 0 | 0.808 | 532 |
| 7 | 0.757 | 507 | 9.74e-18 | 0 | 0.800 | 511 |
| 40 | 0.755 | 499 | 3.61e-17 | 0 | 0.796 | 509 |
| 47 | 0.761 | 510 | 7.82e-18 | 0 | 0.805 | 522 |
| 115 | 0.764 | 512 | 1.37e-17 | 0 | 0.808 | 506 |

Mean over the five samples:

| directions | singular value mean | `hi128` mean | zero crossings mean |
|---|---:|---:|---:|
| DeepONet top-1 | 7.202 | 0.761 | 507.2 |
| solver top-1 | 1.373 | 1.70e-17 | 0.0 |
| error top-1 | 7.100 | 0.803 | 516.0 |
| DeepONet top-8 | 3.664 | 0.745 | 503.3 |
| solver top-8 | 0.672 | 1.82e-16 | 4.0 |
| error top-8 | 3.597 | 0.775 | 517.2 |

Interpretation:

- DeepONet's dominant local input directions are strongly high-frequency and
  jagged.
- The solver's dominant local input directions are essentially low-frequency;
  the top-1 direction has no zero crossings in all five samples.
- The error Jacobian follows the DeepONet high-frequency structure, which means
  the dominant local mismatch is the DeepONet high-frequency response that the
  physical solver does not share.

## Direction Co-Movement

For directions `v` taken from top right singular vectors of each Jacobian, this
table records:

```text
||J_d v||
||J_j v||
||J_d v - J_j v||
cos(J_d v, J_j v)
D_d = ||J_d v - J_j v|| / (||J_d v|| + eta)
```

Values are aggregated over top-8 directions from each source and over five
sample indices.

| direction source | `||J_d v||` | `||J_j v||` | `||J_d v - J_j v||` | `cos(J_d v,J_j v)` | `D_d` |
|---|---:|---:|---:|---:|---:|
| top `J_d` directions | 3.664 +/- 1.651 | 0.177 +/- 0.074 | 3.529 +/- 1.594 | 0.718 +/- 0.197 | 0.964 +/- 0.016 |
| top `J_j` directions | 0.660 +/- 0.406 | 0.672 +/- 0.431 | 0.129 +/- 0.070 | 0.958 +/- 0.053 | 0.254 +/- 0.147 |
| top `J_e` directions | 3.593 +/- 1.622 | 0.031 +/- 0.023 | 3.597 +/- 1.624 | -0.159 +/- 0.347 | 1.001 +/- 0.003 |
| random directions | 0.344 +/- 0.105 | 0.065 +/- 0.020 | 0.339 +/- 0.103 | 0.162 +/- 0.393 | 0.991 +/- 0.085 |

Interpretation:

- Along top DeepONet directions, the solver response is tiny relative to the
  model response: `0.177` vs `3.664`.
- The mismatch gain is almost the same as the DeepONet gain, with `D_d` about
  `0.964`.
- Along top solver directions, DeepONet and solver are much more aligned, but
  those are not DeepONet's dominant sensitive directions.

## Principal-Angle / Subspace Similarity

Top-k right-singular-vector subspace comparison between `J_d` and `J_j`:

| k | mean principal cosine | mean angle |
|---:|---:|---:|
| 1 | 0.125 | 82.83 deg |
| 2 | 0.164 | 80.52 deg |
| 4 | 0.182 | 79.49 deg |
| 8 | 0.170 | 80.19 deg |
| 16 | 0.123 | 82.89 deg |
| 32 | 0.149 | 81.34 deg |

By contrast, DeepONet vs error is almost the same leading subspace:

```text
k=1 mean principal cosine for J_d vs J_e: 0.985
k=8 mean principal cosine for J_d vs J_e: 0.981
```

Interpretation:

- DeepONet and solver leading local input subspaces are nearly orthogonal.
- DeepONet and error leading subspaces are almost identical.
- This again says the leading local mismatch is dominated by DeepONet's own
  high-frequency Jacobian directions.

## Fourier Gain

Selected mean Fourier gains across five indices:

| frequency k | `||J_d phi_k||` | `||J_j phi_k||` | `||J_e phi_k||` |
|---:|---:|---:|---:|
| 1 | 1.035 | 1.063 | 0.167 |
| 2 | 0.669 | 0.671 | 0.140 |
| 4 | 0.200 | 0.189 | 0.077 |
| 8 | 0.050 | 0.042 | 0.055 |
| 16 | 0.220 | 0.00195 | 0.220 |
| 32 | 0.311 | 4.01e-6 | 0.311 |
| 64 | 0.429 | 1.38e-8 | 0.429 |
| 128 | 0.296 | 2.45e-9 | 0.296 |
| 256 | 0.420 | 1.91e-9 | 0.420 |
| 512 | 0.429 | 0.00840 | 0.434 |

Interpretation:

- At the lowest frequencies, DeepONet and solver gains are comparable.
- Starting around `k=16`, the physical solver strongly damps the perturbation,
  while DeepONet keeps a large high-frequency response.
- Therefore the high-frequency error gain is essentially the DeepONet gain.

## Main Conclusion

For the `nu=0.01` DeepONet/default-net model on the same five initial
conditions, the local Jacobian evidence is:

```text
DeepONet dominant directions: high-frequency, jagged, large-gain.
solver dominant directions: low-frequency/smooth, much smaller at high k.
error dominant directions: almost the same as DeepONet dominant directions.
```

This supports the user's observed visual result: the DeepONet-generated local
perturbation can be highly oscillatory, while the solver locally responds like a
diffusive low-pass map.  In this setting, maximizing DeepONet movement can
directly expose large oracle-relative mismatch because `J_d` and `J_j` do not
share the same dominant subspace.
