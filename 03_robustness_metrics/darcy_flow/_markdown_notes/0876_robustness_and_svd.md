# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics

Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.

- Attack rows: `28`
- SVD/Jacobian rows: `14`
- Sample manifest: `outputs/darcy_cflow_final_robustness_20260615_preflight/data/attack_50sample_manifest.csv`

| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |
|---|---|---:|---:|---:|---:|---:|
| baseline | train | 2 | 3.07679e-08 | 1.04019e-07 | 7.32514e-08 | 2.91664 |
| baseline | test | 2 | 1.43908e-07 | 7.68115e-07 | 6.24207e-07 | 3.33489 |
| loss1 | train | 2 | 3.23248e-08 | 5.73245e-08 | 2.49997e-08 | 0.863713 |
| loss1 | test | 2 | 3.8904e-08 | 6.72625e-08 | 2.83585e-08 | 0.842375 |
| loss2 | train | 2 | 2.94289e-08 | 7.77113e-08 | 4.82824e-08 | 3.10812 |
| loss2 | test | 2 | 3.80187e-08 | 9.72996e-08 | 5.92808e-08 | 1.5275 |
| loss3 | train | 2 | 2.70434e-08 | 4.78886e-08 | 2.08452e-08 | 0.731383 |
| loss3 | test | 2 | 2.99718e-08 | 9.08086e-08 | 6.08367e-08 | 1.90282 |
| Physics Loss | train | 2 | 1.55644e-08 | 3.63734e-08 | 2.08089e-08 | 2.96492 |
| Physics Loss | test | 2 | 2.4253e-08 | 7.94714e-08 | 5.52184e-08 | 2.06026 |
| random clean | train | 2 | 3.0829e-08 | 8.33997e-08 | 5.25707e-08 | 1.78147 |
| random clean | test | 2 | 6.195e-08 | 1.19364e-07 | 5.74141e-08 | 0.962387 |
| random solver | train | 2 | 1.4804e-08 | 3.38985e-08 | 1.90946e-08 | 2.68099 |
| random solver | test | 2 | 2.36844e-08 | 5.74842e-08 | 3.37999e-08 | 1.3857 |
