# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `728`
- SVD/Jacobian rows: `21`
- Sample manifest: `outputs/darcy_sir20_timematched_full_20260614_smoke_initial/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 1.04019e-07 | 7.32514e-08 | 2.91664 |
| baseline | test | 2 | 1.43908e-07 | 7.68115e-07 | 6.24207e-07 | 3.33489 |
| baseline | generalization | 100 | 3.57459e-07 | 4.7201e-07 | 1.14551e-07 | 0.391007 |
| loss1 | train | 2 | 2.01942e-07 | 4.5167e-07 | 2.49728e-07 | 1.2395 |
| loss1 | test | 2 | 1.80298e-07 | 6.7295e-07 | 4.92652e-07 | 3.52422 |
| loss1 | generalization | 100 | 7.35615e-07 | 9.29886e-07 | 1.94271e-07 | 0.295857 |
| loss2 | train | 2 | 1.3861e-07 | 3.41862e-07 | 2.03252e-07 | 1.48025 |
| loss2 | test | 2 | 1.58014e-07 | 7.6116e-07 | 6.03146e-07 | 3.95364 |
| loss2 | generalization | 100 | 6.17383e-07 | 7.93132e-07 | 1.75749e-07 | 0.320301 |
| loss3 | train | 2 | 1.48492e-07 | 3.15385e-07 | 1.66894e-07 | 1.10908 |
| loss3 | test | 2 | 4.0413e-07 | 1.37936e-06 | 9.7523e-07 | 2.55663 |
| loss3 | generalization | 100 | 1.88968e-07 | 2.29275e-07 | 4.03063e-08 | 0.226754 |
| physics_loss | train | 2 | 1.20536e-07 | 5.8599e-07 | 4.65454e-07 | 3.80502 |
| physics_loss | test | 2 | 4.39217e-07 | 1.5071e-06 | 1.06789e-06 | 3.14478 |
| physics_loss | generalization | 100 | 1.35769e-07 | 1.72205e-07 | 3.64354e-08 | 0.334471 |
| random_clean | train | 2 | 1.12021e-07 | 1.83883e-07 | 7.1862e-08 | 0.668057 |
| random_clean | test | 2 | 4.00374e-07 | 1.24213e-06 | 8.41755e-07 | 1.31014 |
| random_clean | generalization | 100 | 2.61202e-07 | 3.26034e-07 | 6.48319e-08 | 0.283605 |
| random_solver | train | 2 | 1.7642e-07 | 4.02862e-07 | 2.26442e-07 | 1.28559 |
| random_solver | test | 2 | 1.65956e-07 | 6.5009e-07 | 4.84134e-07 | 3.40749 |
| random_solver | generalization | 100 | 6.94467e-07 | 8.78527e-07 | 1.84059e-07 | 0.298434 |
