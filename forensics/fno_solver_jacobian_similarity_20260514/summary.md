# FNO vs Solver Local Jacobian Similarity Summary

This experiment compares the local FNO Jacobian `J_f`, solver Jacobian
`J_j`, and residual/error-field Jacobian `J_e = J_f - J_j` across sampled
Burgers test inputs.

## Aggregate Direction Co-Movement

| direction source | cos mean +/- std | mismatch gain mean +/- std | D_f mean +/- std |
|---|---:|---:|---:|
| fno | 0.981 +/- 0.01668 | 0.3378 +/- 0.301 | 0.1918 +/- 0.08301 |
| solver | 0.9813 +/- 0.01542 | 0.3404 +/- 0.3027 | 0.1923 +/- 0.07973 |
| error | 0.9451 +/- 0.07038 | 0.4122 +/- 0.2587 | 0.3291 +/- 0.1901 |
| random | 0.9663 +/- 0.01788 | 0.04212 +/- 0.01537 | 0.27 +/- 0.07195 |

## Aggregate Singular Values

| jacobian | spectral norm mean +/- std | fro norm mean +/- std | effective rank mean +/- std |
|---|---:|---:|---:|
| fno | 3.895 +/- 0.3886 | 6.047 +/- 0.5296 | 4.963 +/- 0.7579 |
| solver | 4.095 +/- 0.4566 | 6.349 +/- 0.5097 | 4.809 +/- 0.7578 |
| error | 0.8682 +/- 0.2018 | 1.493 +/- 0.2437 | 15.08 +/- 4.966 |

## Files

- `aggregate_direction_comovement_summary.csv`
- `aggregate_principal_angles_summary.csv`
- `aggregate_singular_value_summary.csv`
- `index_*/`: per-index Jacobians, SVDs, overlap tables, Fourier gains, and plots
