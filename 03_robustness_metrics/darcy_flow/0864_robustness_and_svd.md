# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `175`
- SVD/Jacobian rows: `0`
- Sample manifest: `outputs/darcy_cflow_epsilon_sweep_20260615/eps_0p2x/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 6.25113e-07 | 5.94345e-07 | 25.4975 |
| baseline | test | 2 | 1.43908e-07 | 4.92698e-06 | 4.78307e-06 | 25.7595 |
| baseline | generalization | 21 | 1.0678e-06 | 1.01939e-05 | 9.12612e-06 | 13.6966 |
| loss1 | train | 2 | 3.23248e-08 | 1.94771e-07 | 1.62446e-07 | 7.95804 |
| loss1 | test | 2 | 3.8904e-08 | 2.05925e-07 | 1.67021e-07 | 4.64504 |
| loss1 | generalization | 21 | 5.43745e-07 | 4.60474e-06 | 4.06099e-06 | 14.3915 |
| loss2 | train | 2 | 2.94289e-08 | 2.84733e-07 | 2.55304e-07 | 22.9709 |
| loss2 | test | 2 | 3.80187e-08 | 1.20775e-06 | 1.16973e-06 | 29.4553 |
| loss2 | generalization | 21 | 4.72299e-07 | 4.17048e-06 | 3.69818e-06 | 16.2056 |
| loss3 | train | 2 | 2.70434e-08 | 5.68515e-07 | 5.41471e-07 | 19.4045 |
| loss3 | test | 2 | 2.99718e-08 | 9.31794e-07 | 9.01822e-07 | 26.4622 |
| loss3 | generalization | 21 | 2.03159e-07 | 1.73652e-06 | 1.53336e-06 | 10.2672 |
| Physics Loss | train | 2 | 1.55644e-08 | 1.4821e-07 | 1.32646e-07 | 18.3948 |
| Physics Loss | test | 2 | 2.4253e-08 | 1.25089e-06 | 1.22664e-06 | 39.3853 |
| Physics Loss | generalization | 21 | 5.33366e-07 | 4.54994e-06 | 4.01657e-06 | 13.1375 |
| random clean | train | 2 | 3.0829e-08 | 3.02573e-07 | 2.71744e-07 | 9.38735 |
| random clean | test | 2 | 6.195e-08 | 3.09816e-07 | 2.47866e-07 | 4.28304 |
| random clean | generalization | 21 | 5.20686e-07 | 4.30365e-06 | 3.78296e-06 | 16.2257 |
| random solver | train | 2 | 1.4804e-08 | 1.39854e-07 | 1.2505e-07 | 22.3956 |
| random solver | test | 2 | 2.36844e-08 | 1.26213e-06 | 1.23844e-06 | 40.4916 |
| random solver | generalization | 21 | 7.19158e-07 | 5.42822e-06 | 4.70906e-06 | 9.82967 |
