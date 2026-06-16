# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `175`
- SVD/Jacobian rows: `0`
- Sample manifest: `outputs/darcy_cflow_epsilon_sweep_20260615/eps_0p1x/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 6.4449e-07 | 6.13723e-07 | 26.77 |
| baseline | test | 2 | 1.43908e-07 | 4.22277e-06 | 4.07886e-06 | 19.1349 |
| baseline | generalization | 21 | 1.0678e-06 | 7.96368e-06 | 6.89588e-06 | 9.9848 |
| loss1 | train | 2 | 3.23248e-08 | 2.12185e-07 | 1.7986e-07 | 9.13756 |
| loss1 | test | 2 | 3.8904e-08 | 2.48256e-07 | 2.09352e-07 | 5.97418 |
| loss1 | generalization | 21 | 5.43745e-07 | 3.8016e-06 | 3.25786e-06 | 10.6593 |
| loss2 | train | 2 | 2.94289e-08 | 2.94399e-07 | 2.6497e-07 | 24.8645 |
| loss2 | test | 2 | 3.80187e-08 | 1.16156e-06 | 1.12354e-06 | 28.335 |
| loss2 | generalization | 21 | 4.72299e-07 | 3.54241e-06 | 3.07011e-06 | 12.2652 |
| loss3 | train | 2 | 2.70434e-08 | 2.47477e-07 | 2.20434e-07 | 7.1785 |
| loss3 | test | 2 | 2.99718e-08 | 8.03127e-07 | 7.73155e-07 | 22.5351 |
| loss3 | generalization | 21 | 2.03159e-07 | 1.51822e-06 | 1.31506e-06 | 7.95369 |
| Physics Loss | train | 2 | 1.55644e-08 | 1.51751e-07 | 1.36187e-07 | 18.8728 |
| Physics Loss | test | 2 | 2.4253e-08 | 1.15377e-06 | 1.12951e-06 | 36.5678 |
| Physics Loss | generalization | 21 | 5.33366e-07 | 3.77975e-06 | 3.24638e-06 | 9.87701 |
| random clean | train | 2 | 3.0829e-08 | 3.162e-07 | 2.85371e-07 | 9.94775 |
| random clean | test | 2 | 6.195e-08 | 3.2234e-07 | 2.6039e-07 | 4.54738 |
| random clean | generalization | 21 | 5.20686e-07 | 3.5034e-06 | 2.98271e-06 | 11.443 |
| random solver | train | 2 | 1.4804e-08 | 1.44122e-07 | 1.29318e-07 | 23.6004 |
| random solver | test | 2 | 2.36844e-08 | 1.10945e-06 | 1.08577e-06 | 35.7987 |
| random solver | generalization | 21 | 7.19158e-07 | 4.54888e-06 | 3.82972e-06 | 7.55636 |
