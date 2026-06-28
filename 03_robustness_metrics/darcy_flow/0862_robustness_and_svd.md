# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `175`
- SVD/Jacobian rows: `0`
- Sample manifest: `outputs/darcy_cflow_epsilon_sweep_20260615/eps_0p05x/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 5.26515e-07 | 4.95747e-07 | 23.601 |
| baseline | test | 2 | 1.43908e-07 | 2.24893e-06 | 2.10503e-06 | 11.604 |
| baseline | generalization | 21 | 1.0678e-06 | 5.31861e-06 | 4.2508e-06 | 5.79187 |
| loss1 | train | 2 | 3.23248e-08 | 2.39028e-07 | 2.06703e-07 | 10.3279 |
| loss1 | test | 2 | 3.8904e-08 | 2.28137e-07 | 1.89233e-07 | 5.65083 |
| loss1 | generalization | 21 | 5.43745e-07 | 2.3975e-06 | 1.85375e-06 | 5.65817 |
| loss2 | train | 2 | 2.94289e-08 | 2.85077e-07 | 2.55648e-07 | 23.9824 |
| loss2 | test | 2 | 3.80187e-08 | 7.63306e-07 | 7.25287e-07 | 18.3855 |
| loss2 | generalization | 21 | 4.72299e-07 | 2.21891e-06 | 1.74661e-06 | 6.19283 |
| loss3 | train | 2 | 2.70434e-08 | 1.47849e-07 | 1.20805e-07 | 4.22781 |
| loss3 | test | 2 | 2.99718e-08 | 4.89556e-07 | 4.59584e-07 | 14.299 |
| loss3 | generalization | 21 | 2.03159e-07 | 9.58869e-07 | 7.5571e-07 | 4.41511 |
| Physics Loss | train | 2 | 1.55644e-08 | 1.72945e-07 | 1.57381e-07 | 22.3606 |
| Physics Loss | test | 2 | 2.4253e-08 | 6.1161e-07 | 5.87357e-07 | 19.9717 |
| Physics Loss | generalization | 21 | 5.33366e-07 | 2.37595e-06 | 1.84259e-06 | 5.30638 |
| random clean | train | 2 | 3.0829e-08 | 3.31987e-07 | 3.01158e-07 | 10.5421 |
| random clean | test | 2 | 6.195e-08 | 3.41829e-07 | 2.79879e-07 | 4.83443 |
| random clean | generalization | 21 | 5.20686e-07 | 2.16423e-06 | 1.64354e-06 | 5.52817 |
| random solver | train | 2 | 1.4804e-08 | 1.46385e-07 | 1.31581e-07 | 25.1432 |
| random solver | test | 2 | 2.36844e-08 | 6.05111e-07 | 5.81426e-07 | 20.0715 |
| random solver | generalization | 21 | 7.19158e-07 | 2.93554e-06 | 2.21638e-06 | 4.26187 |
