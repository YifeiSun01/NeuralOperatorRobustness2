# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `175`
- SVD/Jacobian rows: `0`
- Sample manifest: `outputs/darcy_cflow_epsilon_sweep_20260615/eps_5x/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 6.1263e-07 | 5.81862e-07 | 33.7511 |
| baseline | test | 2 | 1.43908e-07 | 5.73285e-06 | 5.58894e-06 | 25.1891 |
| baseline | generalization | 21 | 1.0678e-06 | 1.08344e-05 | 9.76663e-06 | 15.0911 |
| loss1 | train | 2 | 3.23248e-08 | 1.56293e-06 | 1.53061e-06 | 105.278 |
| loss1 | test | 2 | 3.8904e-08 | 1.29753e-06 | 1.25863e-06 | 65.1872 |
| loss1 | generalization | 21 | 5.43745e-07 | 3.72179e-06 | 3.17805e-06 | 12.4152 |
| loss2 | train | 2 | 2.94289e-08 | 1.08105e-07 | 7.86763e-08 | 5.58579 |
| loss2 | test | 2 | 3.80187e-08 | 1.77894e-06 | 1.74092e-06 | 45.3349 |
| loss2 | generalization | 21 | 4.72299e-07 | 3.21957e-06 | 2.74727e-06 | 12.9776 |
| loss3 | train | 2 | 2.70434e-08 | 1.68699e-06 | 1.65995e-06 | 64.5918 |
| loss3 | test | 2 | 2.99718e-08 | 1.78777e-06 | 1.75779e-06 | 63.8773 |
| loss3 | generalization | 21 | 2.03159e-07 | 1.44004e-06 | 1.23689e-06 | 8.29856 |
| Physics Loss | train | 2 | 1.55644e-08 | 2.52293e-06 | 2.50736e-06 | 295.958 |
| Physics Loss | test | 2 | 2.4253e-08 | 2.71534e-06 | 2.69109e-06 | 118.981 |
| Physics Loss | generalization | 21 | 5.33366e-07 | 3.5945e-06 | 3.06114e-06 | 10.887 |
| random clean | train | 2 | 3.0829e-08 | 7.52842e-08 | 4.44552e-08 | 1.47421 |
| random clean | test | 2 | 6.195e-08 | 1.35814e-07 | 7.38638e-08 | 0.978132 |
| random clean | generalization | 21 | 5.20686e-07 | 3.41155e-06 | 2.89086e-06 | 13.3603 |
| random solver | train | 2 | 1.4804e-08 | 2.6326e-06 | 2.6178e-06 | 366.55 |
| random solver | test | 2 | 2.36844e-08 | 2.59552e-06 | 2.57184e-06 | 120.866 |
| random solver | generalization | 21 | 7.19158e-07 | 4.31084e-06 | 3.59168e-06 | 8.17851 |
