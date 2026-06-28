# FNO vs Solver Local Jacobian Similarity Summary

This experiment compares the local FNO Jacobian `J_f`, solver Jacobian
`J_j`, and residual/error-field Jacobian `J_e = J_f - J_j` across sampled
Burgers test inputs.

## Aggregate Direction Co-Movement

| direction source | cos mean +/- std | mismatch gain mean +/- std | D_f mean +/- std |
|---|---:|---:|---:|
| fno | 0.9993 +/- 0.001029 | 0.01552 +/- 0.006485 | 0.0326 +/- 0.02169 |
| solver | 0.9993 +/- 0.0009723 | 0.01541 +/- 0.006543 | 0.03204 +/- 0.02082 |
| error | 0.3524 +/- 0.3647 | 0.03185 +/- 0.003565 | 0.8029 +/- 0.3135 |
| random | 0.9443 +/- 0.04045 | 0.01834 +/- 0.001242 | 0.3077 +/- 0.1103 |

## Aggregate Singular Values

| jacobian | spectral norm mean +/- std | fro norm mean +/- std | effective rank mean +/- std |
|---|---:|---:|---:|
| fno | 1.379 +/- 0.1055 | 2.347 +/- 0.07766 | 8.358 +/- 1.318 |
| solver | 1.373 +/- 0.09647 | 2.262 +/- 0.07898 | 4.747 +/- 0.4104 |
| error | 0.0373 +/- 0.005834 | 0.5883 +/- 0.03984 | 782 +/- 66.34 |

## Files

- `aggregate_direction_comovement_summary.csv`
- `aggregate_principal_angles_summary.csv`
- `aggregate_singular_value_summary.csv`
- `index_*/`: per-index Jacobians, SVDs, overlap tables, Fourier gains, and plots
