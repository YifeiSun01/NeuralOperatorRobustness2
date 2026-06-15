# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `18200`
- SVD/Jacobian rows: `175`
- Sample manifest: `outputs/darcy_cflow_final_robustness_20260615/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 50 | 4.29983e-08 | 4.07106e-06 | 4.02806e-06 | 184.046 |
| baseline | test | 50 | 9.96066e-08 | 4.60882e-06 | 4.50921e-06 | 63.2473 |
| baseline | generalization | 2500 | 1.4816e-06 | 1.11625e-05 | 9.68093e-06 | 17.7507 |
| loss1 | train | 50 | 2.95497e-08 | 1.25311e-06 | 1.22356e-06 | 42.1317 |
| loss1 | test | 50 | 3.57893e-08 | 1.37401e-06 | 1.33822e-06 | 44.9953 |
| loss1 | generalization | 2500 | 7.38492e-07 | 4.66556e-06 | 3.92706e-06 | 14.0778 |
| loss2 | train | 50 | 2.94268e-08 | 1.20035e-06 | 1.17092e-06 | 65.1937 |
| loss2 | test | 50 | 3.03455e-08 | 1.41365e-06 | 1.38331e-06 | 64.0131 |
| loss2 | generalization | 2500 | 6.29486e-07 | 4.19417e-06 | 3.56469e-06 | 16.466 |
| loss3 | train | 50 | 5.00365e-08 | 1.50446e-06 | 1.45443e-06 | 36.9346 |
| loss3 | test | 50 | 6.69125e-08 | 1.4672e-06 | 1.40029e-06 | 35.0706 |
| loss3 | generalization | 2500 | 2.88169e-07 | 1.98716e-06 | 1.699e-06 | 8.48128 |
| Physics Loss | train | 50 | 2.5973e-08 | 1.88847e-06 | 1.8625e-06 | 86.2895 |
| Physics Loss | test | 50 | 3.17203e-08 | 1.82731e-06 | 1.79559e-06 | 70.3739 |
| Physics Loss | generalization | 2500 | 7.22671e-07 | 4.69654e-06 | 3.97387e-06 | 15.1758 |
| random clean | train | 50 | 3.3673e-08 | 8.39911e-07 | 8.06238e-07 | 25.2693 |
| random clean | test | 50 | 4.24682e-08 | 1.04726e-06 | 1.00479e-06 | 25.1191 |
| random clean | generalization | 2500 | 6.68449e-07 | 4.35924e-06 | 3.69079e-06 | 15.2786 |
| random solver | train | 50 | 2.43531e-08 | 2.50651e-06 | 2.48216e-06 | 137.722 |
| random solver | test | 50 | 2.99052e-08 | 2.14919e-06 | 2.11929e-06 | 109.248 |
| random solver | generalization | 2500 | 9.31897e-07 | 5.75595e-06 | 4.82405e-06 | 13.3171 |
