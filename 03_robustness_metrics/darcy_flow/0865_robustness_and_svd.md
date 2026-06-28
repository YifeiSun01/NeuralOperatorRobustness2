# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `175`
- SVD/Jacobian rows: `0`
- Sample manifest: `outputs/darcy_cflow_epsilon_sweep_20260615/eps_0p5x/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 6.02035e-07 | 5.71267e-07 | 23.2563 |
| baseline | test | 2 | 1.43908e-07 | 5.5296e-06 | 5.3857e-06 | 27.6947 |
| baseline | generalization | 21 | 1.0678e-06 | 1.07861e-05 | 9.71825e-06 | 14.8958 |
| loss1 | train | 2 | 3.23248e-08 | 1.41621e-07 | 1.09296e-07 | 4.69835 |
| loss1 | test | 2 | 3.8904e-08 | 1.62121e-07 | 1.23217e-07 | 3.27457 |
| loss1 | generalization | 21 | 5.43745e-07 | 4.7924e-06 | 4.24865e-06 | 15.1744 |
| loss2 | train | 2 | 2.94289e-08 | 2.41073e-07 | 2.11644e-07 | 16.233 |
| loss2 | test | 2 | 3.80187e-08 | 1.18251e-06 | 1.14449e-06 | 28.7601 |
| loss2 | generalization | 21 | 4.72299e-07 | 4.36043e-06 | 3.88813e-06 | 17.2238 |
| loss3 | train | 2 | 2.70434e-08 | 9.37776e-07 | 9.10732e-07 | 32.2092 |
| loss3 | test | 2 | 2.99718e-08 | 1.25613e-06 | 1.22616e-06 | 41.005 |
| loss3 | generalization | 21 | 2.03159e-07 | 1.85156e-06 | 1.6484e-06 | 10.9845 |
| Physics Loss | train | 2 | 1.55644e-08 | 1.08496e-07 | 9.29314e-08 | 13.046 |
| Physics Loss | test | 2 | 2.4253e-08 | 1.32033e-06 | 1.29608e-06 | 40.9528 |
| Physics Loss | generalization | 21 | 5.33366e-07 | 4.8198e-06 | 4.28643e-06 | 14.3762 |
| random clean | train | 2 | 3.0829e-08 | 2.74336e-07 | 2.43507e-07 | 8.06647 |
| random clean | test | 2 | 6.195e-08 | 2.54032e-07 | 1.92082e-07 | 3.10658 |
| random clean | generalization | 21 | 5.20686e-07 | 4.5441e-06 | 4.02342e-06 | 17.885 |
| random solver | train | 2 | 1.4804e-08 | 1.04743e-07 | 8.9939e-08 | 13.579 |
| random solver | test | 2 | 2.36844e-08 | 1.30284e-06 | 1.27916e-06 | 41.2309 |
| random solver | generalization | 21 | 7.19158e-07 | 5.70645e-06 | 4.98729e-06 | 10.6422 |
