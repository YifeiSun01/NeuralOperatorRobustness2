# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `175`
- SVD/Jacobian rows: `0`
- Sample manifest: `outputs/darcy_cflow_epsilon_sweep_20260615/eps_0p01x/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 1.70685e-07 | 1.39917e-07 | 7.17379 |
| baseline | test | 2 | 1.43908e-07 | 6.08944e-07 | 4.65036e-07 | 3.31689 |
| baseline | generalization | 21 | 1.0678e-06 | 2.19153e-06 | 1.12373e-06 | 1.31824 |
| loss1 | train | 2 | 3.23248e-08 | 1.31345e-07 | 9.90205e-08 | 4.77191 |
| loss1 | test | 2 | 3.8904e-08 | 1.3534e-07 | 9.64359e-08 | 3.05088 |
| loss1 | generalization | 21 | 5.43745e-07 | 1.0369e-06 | 4.93153e-07 | 1.34488 |
| loss2 | train | 2 | 2.94289e-08 | 1.29884e-07 | 1.00455e-07 | 9.20233 |
| loss2 | test | 2 | 3.80187e-08 | 1.90714e-07 | 1.52695e-07 | 3.9685 |
| loss2 | generalization | 21 | 4.72299e-07 | 9.23789e-07 | 4.5149e-07 | 1.43003 |
| loss3 | train | 2 | 2.70434e-08 | 4.38057e-08 | 1.67623e-08 | 0.575491 |
| loss3 | test | 2 | 2.99718e-08 | 1.4451e-07 | 1.14538e-07 | 3.69467 |
| loss3 | generalization | 21 | 2.03159e-07 | 3.93211e-07 | 1.90052e-07 | 1.0474 |
| Physics Loss | train | 2 | 1.55644e-08 | 8.30193e-08 | 6.74549e-08 | 8.94911 |
| Physics Loss | test | 2 | 2.4253e-08 | 1.42093e-07 | 1.1784e-07 | 4.82265 |
| Physics Loss | generalization | 21 | 5.33366e-07 | 1.01708e-06 | 4.83714e-07 | 1.26714 |
| random clean | train | 2 | 3.0829e-08 | 1.48682e-07 | 1.17853e-07 | 4.14109 |
| random clean | test | 2 | 6.195e-08 | 1.98428e-07 | 1.36478e-07 | 2.45258 |
| random clean | generalization | 21 | 5.20686e-07 | 9.68833e-07 | 4.48148e-07 | 1.28612 |
| random solver | train | 2 | 1.4804e-08 | 8.00246e-08 | 6.52206e-08 | 10.6288 |
| random solver | test | 2 | 2.36844e-08 | 1.36073e-07 | 1.12389e-07 | 4.66496 |
| random solver | generalization | 21 | 7.19158e-07 | 1.31543e-06 | 5.96276e-07 | 1.08253 |
