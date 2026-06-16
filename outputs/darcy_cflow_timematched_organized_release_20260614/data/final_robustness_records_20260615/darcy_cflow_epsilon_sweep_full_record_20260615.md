# Darcy CFlow Epsilon Sweep Full Record - 20260615

本文件专门记录 epsilon budget sweep 的最终数据和结论。实验口径：固定 25 samples x 7 models，attack_steps=50；1x 使用已有 final attack50 表筛固定 175 行，不重跑；其余 epsilon 为新 rerun。

## 1. Provenance
| epsilon | multiplier | epsilon_fraction | source | raw_rows | used_rows | attack_steps | old_20260607_rows | usable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0p01x | 0.01 | 0.00025000 | sweep | 175 | 175 | 50 | 0 | True |
| 0p05x | 0.05 | 0.00125000 | sweep | 175 | 175 | 50 | 0 | True |
| 0p1x | 0.10 | 0.00250000 | sweep | 175 | 175 | 50 | 0 | True |
| 0p2x | 0.20 | 0.00500000 | sweep | 175 | 175 | 50 | 0 | True |
| 0p5x | 0.50 | 0.01250000 | sweep | 175 | 175 | 50 | 0 | True |
| 1x | 1.00 | 0.02500000 | existing_final_attack50 | 18200 | 175 | 50 | 0 | True |
| 5x | 5.00 | 0.12500000 | sweep | 175 | 175 | 50 | 0 | True |
| 10x | 10.00 | 0.25000000 | sweep | 175 | 175 | 50 | 0 | True |

## 2. Main Conclusion

- Loss3 is mean-best for attack `loss_increase` at every tested epsilon budget: 0.01x, 0.05x, 0.1x, 0.2x, 0.5x, 1x, 5x, and 10x.
- Loss3 is not pointwise-best on every sample. It is majority-best: 17/25 to 22/25 depending on epsilon.
- Smaller epsilon makes `loss_increase` correlate more strongly with residual first-order quantities, especially residual `J_res^T error`; 10x weakens the rank/monotone relation substantially.
- Attack delta is not well aligned with the single residual top singular vector. It aligns more with residual `J_res^T error` and with the residual top10 input subspace, but this vector similarity is not strictly monotone as epsilon decreases.

## 3. Loss3 Mean-Best Check
| epsilon | multiplier | Loss3 loss_increase mean | second model | second mean | gap | relative gap | Loss3 pointwise | mean-best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0p01x | 0.01 | 1.701e-07 | random clean | 3.968e-07 | 2.266e-07 | 57.1% | 22/25 | True |
| 0p05x | 0.05 | 6.812e-07 | random clean | 1.427e-06 | 7.458e-07 | 52.3% | 21/25 | True |
| 0p1x | 0.10 | 1.184e-06 | random clean | 2.549e-06 | 1.365e-06 | 53.5% | 22/25 | True |
| 0p2x | 0.20 | 1.403e-06 | random clean | 3.219e-06 | 1.816e-06 | 56.4% | 21/25 | True |
| 0p5x | 0.50 | 1.556e-06 | loss2 | 3.375e-06 | 1.819e-06 | 53.9% | 21/25 | True |
| 1x | 1.00 | 1.716e-06 | loss2 | 3.393e-06 | 1.677e-06 | 49.4% | 21/25 | True |
| 5x | 5.00 | 1.312e-06 | random clean | 2.438e-06 | 1.125e-06 | 46.2% | 21/25 | True |
| 10x | 10.00 | 8.991e-07 | loss2 | 1.412e-06 | 5.133e-07 | 36.3% | 17/25 | True |

## 4. Scalar Correlation: Generalization 21 Samples x 7 Models

Format: Pearson r / Spearman rho. All pairs use attack `loss_increase`.
| epsilon | multiplier | vs residual J^T error norm | vs residual sigma1 | vs residual error L2 |
| --- | --- | --- | --- | --- |
| 0p01x | 0.01 | 0.589 / 0.620 | 0.421 / 0.425 | 0.652 / 0.669 |
| 0p05x | 0.05 | 0.592 / 0.648 | 0.474 / 0.478 | 0.639 / 0.679 |
| 0p1x | 0.10 | 0.567 / 0.655 | 0.506 / 0.506 | 0.584 / 0.664 |
| 0p2x | 0.20 | 0.447 / 0.378 | 0.447 / 0.297 | 0.434 / 0.361 |
| 0p5x | 0.50 | 0.398 / 0.298 | 0.399 / 0.209 | 0.386 / 0.284 |
| 1x | 1.00 | 0.353 / 0.210 | 0.347 / 0.115 | 0.346 / 0.201 |
| 5x | 5.00 | 0.302 / 0.050 | 0.307 / -0.013 | 0.290 / 0.033 |
| 10x | 10.00 | 0.227 / -0.096 | 0.235 / -0.153 | 0.220 / -0.103 |

## 5. Scalar Correlation: All 25 Samples x 7 Models

Format: Pearson r / Spearman rho. All pairs use attack `loss_increase`.
| epsilon | multiplier | vs residual J^T error norm | vs residual sigma1 | vs residual error L2 |
| --- | --- | --- | --- | --- |
| 0p01x | 0.01 | 0.659 / 0.754 | 0.544 / 0.628 | 0.711 / 0.781 |
| 0p05x | 0.05 | 0.675 / 0.765 | 0.623 / 0.655 | 0.725 / 0.781 |
| 0p1x | 0.10 | 0.659 / 0.769 | 0.668 / 0.673 | 0.698 / 0.773 |
| 0p2x | 0.20 | 0.579 / 0.605 | 0.634 / 0.552 | 0.602 / 0.592 |
| 0p5x | 0.50 | 0.541 / 0.556 | 0.602 / 0.496 | 0.568 / 0.545 |
| 1x | 1.00 | 0.504 / 0.503 | 0.562 / 0.438 | 0.535 / 0.496 |
| 5x | 5.00 | 0.386 / 0.311 | 0.406 / 0.266 | 0.390 / 0.296 |
| 10x | 10.00 | 0.243 / -0.053 | 0.234 / -0.094 | 0.236 / -0.064 |

## 6. Vector Similarity: Attack Delta vs Residual Directions

Scope: all 25 samples x 7 models. Format: mean cosine / mean angle degrees.
| epsilon | multiplier | delta vs residual J^T error vector | delta vs residual top singular vector | delta vs residual top10 subspace |
| --- | --- | --- | --- | --- |
| 0p01x | 0.01 | 0.1228 / 82.91 deg | -0.0108 / 90.57 deg | 0.1668 / 80.33 deg |
| 0p05x | 0.05 | 0.1687 / 80.19 deg | -0.0144 / 90.68 deg | 0.2246 / 76.84 deg |
| 0p1x | 0.10 | 0.1712 / 80.03 deg | -0.0157 / 90.72 deg | 0.2379 / 76.03 deg |
| 0p2x | 0.20 | 0.1650 / 80.36 deg | -0.0084 / 90.28 deg | 0.2415 / 75.77 deg |
| 0p5x | 0.50 | 0.1600 / 80.62 deg | -0.0016 / 89.84 deg | 0.2435 / 75.58 deg |
| 1x | 1.00 | 0.1484 / 81.27 deg | 0.0016 / 89.66 deg | 0.2415 / 75.69 deg |
| 5x | 5.00 | 0.0928 / 84.53 deg | 0.0138 / 89.06 deg | 0.2167 / 77.23 deg |
| 10x | 10.00 | 0.0247 / 88.52 deg | 0.0383 / 87.73 deg | 0.2154 / 77.41 deg |

## 7. Source Data Files
| file | contents |
| --- | --- |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_provenance_checks.csv | 8 epsilon provenance/sanity checks |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_attack_rows_fixed25_7models.csv | 1400 attack rows: 8 epsilon x 175 rows |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_attack_residual_joined_fixed25_7models.csv | attack rows joined to residual scalar metrics |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_by_model_mean_std.csv | by-model mean/std for attack and residual scalar metrics |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_winner_counts_by_epsilon.csv | pointwise winner counts per epsilon and metric |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_residual_correlations.csv | all Pearson/Spearman scalar correlations |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_vector_angles_rows.csv | row-level vector angles/similarities |
| outputs/darcy_cflow_epsilon_sweep_20260615/data/epsilon_sweep_vector_angles_overall.csv | overall vector-angle mean/std per epsilon |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/epsilon_sweep_raw_attack_sources_20260615/ | raw attack CSVs, delta NPZs, and logs for sweep reruns |

## 8. Final Short Answer

Loss3 is the most robust model by mean attack `loss_increase` at every tested epsilon budget. The scalar-correlation data support the small-epsilon first-order interpretation: smaller budgets strengthen the relationship between attack loss increase and residual `J_res^T error`, while large budgets, especially 10x, weaken it. The vector-similarity data show that attack delta is mostly not the residual top singular vector itself; the residual top10 subspace and residual `J_res^T error` vector are more informative.
