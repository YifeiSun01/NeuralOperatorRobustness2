# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `175`
- SVD/Jacobian rows: `0`
- Sample manifest: `outputs/darcy_cflow_epsilon_sweep_20260615/eps_10x/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 5.24171e-07 | 4.93403e-07 | 30.0794 |
| baseline | test | 2 | 1.43908e-07 | 4.77294e-06 | 4.62903e-06 | 22.6134 |
| baseline | generalization | 21 | 1.0678e-06 | 8.90886e-06 | 7.84106e-06 | 12.4041 |
| loss1 | train | 2 | 3.23248e-08 | 2.64595e-06 | 2.61362e-06 | 116.338 |
| loss1 | test | 2 | 3.8904e-08 | 1.2518e-06 | 1.2129e-06 | 63.0036 |
| loss1 | generalization | 21 | 5.43745e-07 | 2.31211e-06 | 1.76836e-06 | 7.49728 |
| loss2 | train | 2 | 2.94289e-08 | 9.59683e-08 | 6.65394e-08 | 7.55026 |
| loss2 | test | 2 | 3.80187e-08 | 2.28287e-06 | 2.24485e-06 | 59.0925 |
| loss2 | generalization | 21 | 4.72299e-07 | 1.93356e-06 | 1.46126e-06 | 7.56602 |
| loss3 | train | 2 | 2.70434e-08 | 1.61348e-06 | 1.58644e-06 | 61.8396 |
| loss3 | test | 2 | 2.99718e-08 | 1.52211e-06 | 1.49214e-06 | 54.9085 |
| loss3 | generalization | 21 | 2.03159e-07 | 9.8026e-07 | 7.77101e-07 | 5.29861 |
| Physics Loss | train | 2 | 1.55644e-08 | 2.42754e-06 | 2.41197e-06 | 281.459 |
| Physics Loss | test | 2 | 2.4253e-08 | 2.52929e-06 | 2.50503e-06 | 116.957 |
| Physics Loss | generalization | 21 | 5.33366e-07 | 2.2375e-06 | 1.70414e-06 | 6.58659 |
| random clean | train | 2 | 3.0829e-08 | 2.14946e-06 | 2.11863e-06 | 71.8591 |
| random clean | test | 2 | 6.195e-08 | 1.70509e-07 | 1.08559e-07 | 2.1304 |
| random clean | generalization | 21 | 5.20686e-07 | 2.06547e-06 | 1.54478e-06 | 8.04999 |
| random solver | train | 2 | 1.4804e-08 | 2.23718e-06 | 2.22237e-06 | 315.601 |
| random solver | test | 2 | 2.36844e-08 | 2.7842e-06 | 2.76051e-06 | 135.558 |
| random solver | generalization | 21 | 7.19158e-07 | 2.72882e-06 | 2.00966e-06 | 4.80078 |
