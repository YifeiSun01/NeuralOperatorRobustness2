# DeepONet vs Solver Local Jacobian Similarity Summary

This experiment compares the local DeepONet Jacobian `J_d`, solver Jacobian
`J_j`, and residual/error-field Jacobian `J_e = J_d - J_j` across sampled
Burgers `nu=0.01` test inputs.

## Aggregate Direction Co-Movement

| direction source | cos mean +/- std | mismatch gain mean +/- std | D_model mean +/- std |
|---|---:|---:|---:|
| deeponet | 0.718 +/- 0.1971 | 3.529 +/- 1.594 | 0.9642 +/- 0.01589 |
| solver | 0.9584 +/- 0.05269 | 0.1289 +/- 0.06992 | 0.2537 +/- 0.1473 |
| error | -0.1586 +/- 0.3466 | 3.597 +/- 1.624 | 1.001 +/- 0.002617 |
| random | 0.1616 +/- 0.3926 | 0.3393 +/- 0.1035 | 0.9909 +/- 0.08539 |

## Aggregate Singular Values

| jacobian | spectral norm mean +/- std | fro norm mean +/- std | effective rank mean +/- std |
|---|---:|---:|---:|
| deeponet | 7.202 +/- 0.7021 | 11.68 +/- 0.7221 | 7.043 +/- 0.7376 |
| solver | 1.373 +/- 0.09647 | 2.262 +/- 0.07898 | 4.747 +/- 0.4104 |
| error | 7.1 +/- 0.6965 | 11.48 +/- 0.726 | 7.077 +/- 0.7246 |

## Files

- `aggregate_direction_comovement_summary.csv`
- `aggregate_principal_angles_summary.csv`
- `aggregate_singular_value_summary.csv`
- `index_*/`: per-index Jacobians, SVDs, overlap tables, Fourier gains, and plots
