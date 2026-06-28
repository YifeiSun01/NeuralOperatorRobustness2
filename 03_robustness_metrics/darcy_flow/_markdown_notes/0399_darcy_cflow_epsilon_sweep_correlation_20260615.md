# Darcy CFlow Epsilon Sweep Attack/Residual Correlation Summary

## 1. Provenance
本实验只重跑 attack，不重训、不重算 residual Jacobian/SVD；口径是固定 25 样本 x 7 模型，用于和已有 residual `J_model - J_solver` 指标逐行对齐。
| epsilon | source | raw_rows | rows | models | splits | attack_steps | epsilon_fraction | 20260611_only | old_rows | usable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0p01x | sweep | 175 | 175 | 7 | generalization,test,train | 50 | 0.00025000 | True | 0 | True |
| 0p05x | sweep | 175 | 175 | 7 | generalization,test,train | 50 | 0.00125000 | True | 0 | True |
| 0p1x | sweep | 175 | 175 | 7 | generalization,test,train | 50 | 0.00250000 | True | 0 | True |
| 0p2x | sweep | 175 | 175 | 7 | generalization,test,train | 50 | 0.00500000 | True | 0 | True |
| 0p5x | sweep | 175 | 175 | 7 | generalization,test,train | 50 | 0.01250000 | True | 0 | True |
| 1x | existing_final_attack50 | 18200 | 175 | 7 | generalization,test,train | 50 | 0.02500000 | True | 0 | True |
| 5x | sweep | 175 | 175 | 7 | generalization,test,train | 50 | 0.12500000 | True | 0 | True |
| 10x | sweep | 175 | 175 | 7 | generalization,test,train | 50 | 0.25000000 | True | 0 | True |

## 2. 核心相关性结论
Generalization 21 samples x 7 models：
| epsilon | loss_inc~JT Pearson/Spearman | loss_inc~sigma1 Pearson/Spearman | loss_inc~errorL2 Pearson/Spearman | Pearson abs strongest |
| --- | --- | --- | --- | --- |
| 0p01x | 0.588872/0.620364 | 0.421332/0.425364 | 0.651664/0.668883 | error_L2 |
| 0p05x | 0.591960/0.648146 | 0.474381/0.477748 | 0.638526/0.678554 | error_L2 |
| 0p1x | 0.567467/0.654825 | 0.506495/0.506033 | 0.583907/0.663832 | error_L2 |
| 0p2x | 0.447471/0.378035 | 0.446991/0.297176 | 0.433758/0.360705 | JT |
| 0p5x | 0.397756/0.298238 | 0.399102/0.209257 | 0.385921/0.283602 | sigma1 |
| 1x | 0.353294/0.209620 | 0.346862/0.114519 | 0.345888/0.201369 | JT |
| 5x | 0.302126/0.049717 | 0.306626/-0.013468 | 0.289939/0.032660 | sigma1 |
| 10x | 0.227265/-0.096015 | 0.234954/-0.153401 | 0.219544/-0.102502 | sigma1 |

All 25 samples x 7 models：
| epsilon | loss_inc~JT Pearson/Spearman | loss_inc~sigma1 Pearson/Spearman | loss_inc~errorL2 Pearson/Spearman | Pearson abs strongest |
| --- | --- | --- | --- | --- |
| 0p01x | 0.658555/0.753842 | 0.543971/0.628417 | 0.711126/0.781373 | error_L2 |
| 0p05x | 0.674528/0.764536 | 0.622718/0.654680 | 0.724575/0.780909 | error_L2 |
| 0p1x | 0.658889/0.769111 | 0.667559/0.672806 | 0.698067/0.772588 | error_L2 |
| 0p2x | 0.578763/0.605255 | 0.633927/0.551523 | 0.601891/0.592056 | sigma1 |
| 0p5x | 0.540979/0.556120 | 0.602031/0.496160 | 0.568101/0.544626 | sigma1 |
| 1x | 0.504257/0.502891 | 0.561659/0.438184 | 0.534601/0.495813 | sigma1 |
| 5x | 0.385627/0.311153 | 0.406477/0.266352 | 0.389652/0.296028 | sigma1 |
| 10x | 0.242870/-0.053247 | 0.234327/-0.093820 | 0.236418/-0.064172 | JT |

结论：在最小 epsilon=0p01x 的 generalization 口径下，loss increase 和 residual JT norm 的 Pearson 相关强于 residual sigma1，这支持“小 budget 更接近一阶 JT-error 推导”的方向。
但两者不是互斥解释：finite-step PGD 的方向仍可能被 residual operator 的大奇异方向影响。

## 3. Loss3 Attack Mean/Std Across Epsilon
| epsilon | adv_loss mean/std | loss_increase mean/std | relative_increase mean/std | delta_l2_rms mean/std | delta_linf mean/std |
| --- | --- | --- | --- | --- | --- |
| 0p01x | 3.453625e-07/1.961210e-07 | 1.701477e-07/9.779415e-08 | 1.221432/0.848058 | 1.045814/0.065050 | 9.000000/0 |
| 0p05x | 8.564422e-07/4.445584e-07 | 6.812274e-07/3.667507e-07 | 5.190830/3.661749 | 2.186866/0.164000 | 9.000000/0 |
| 0p1x | 1.359357e-06/7.893073e-07 | 1.184142e-06/7.277502e-07 | 9.058188/7.403633 | 2.859423/0.422066 | 9.000000/0 |
| 0p2x | 1.578699e-06/9.203691e-07 | 1.403484e-06/8.830717e-07 | 12.293819/11.473072 | 3.196894/0.409932 | 9.000000/0 |
| 0p5x | 1.730826e-06/9.419332e-07 | 1.555611e-06/9.135353e-07 | 15.084116/14.133303 | 3.529247/0.382260 | 9.000000/0 |
| 1x | 1.891393e-06/8.783259e-07 | 1.716178e-06/8.543290e-07 | 17.566485/16.897701 | 3.819950/0.425012 | 9.000000/0 |
| 5x | 1.487618e-06/7.100791e-07 | 1.312403e-06/7.196521e-07 | 17.248316/23.339794 | 4.615791/0.587142 | 9.000000/0 |
| 10x | 1.074266e-06/5.061813e-07 | 8.990511e-07/5.510103e-07 | 13.790685/21.550216 | 5.121135/0.638813 | 9.000000/0 |

## 4. Winner Counts
| epsilon | metric | top winner | Loss3 |
| --- | --- | --- | --- |
| 0p01x | adv_loss | loss3 21/25 | 21/25 |
| 0p01x | loss_increase | loss3 22/25 | 22/25 |
| 0p01x | relative_increase | random solver 9/25 | 8/25 |
| 0p05x | adv_loss | loss3 21/25 | 21/25 |
| 0p05x | loss_increase | loss3 21/25 | 21/25 |
| 0p05x | relative_increase | random solver 8/25 | 7/25 |
| 0p1x | adv_loss | loss3 22/25 | 22/25 |
| 0p1x | loss_increase | loss3 22/25 | 22/25 |
| 0p1x | relative_increase | loss3 8/25 | 8/25 |
| 0p2x | adv_loss | loss3 21/25 | 21/25 |
| 0p2x | loss_increase | loss3 21/25 | 21/25 |
| 0p2x | relative_increase | loss3 8/25 | 8/25 |
| 0p5x | adv_loss | loss3 21/25 | 21/25 |
| 0p5x | loss_increase | loss3 21/25 | 21/25 |
| 0p5x | relative_increase | loss3 9/25 | 9/25 |
| 1x | adv_loss | loss3 21/25 | 21/25 |
| 1x | loss_increase | loss3 21/25 | 21/25 |
| 1x | relative_increase | loss3 8/25 | 8/25 |
| 5x | adv_loss | loss3 21/25 | 21/25 |
| 5x | loss_increase | loss3 21/25 | 21/25 |
| 5x | relative_increase | random solver 8/25 | 7/25 |
| 10x | adv_loss | loss3 20/25 | 20/25 |
| 10x | loss_increase | loss3 17/25 | 17/25 |
| 10x | relative_increase | random solver 8/25 | 6/25 |

## 5. Vector Angle Summary
| epsilon | singular vs delta angle/cos | JT vs delta angle/cos | top10 subspace vs delta angle/cos |
| --- | --- | --- | --- |
| 0p01x | 90.568561/-0.010795 | 82.908964/0.122768 | 80.329435/0.166792 |
| 0p05x | 90.684345/-0.014438 | 80.190673/0.168713 | 76.844953/0.224642 |
| 0p1x | 90.721398/-0.015667 | 80.029138/0.171231 | 76.028077/0.237894 |
| 0p2x | 90.276274/-0.008401 | 80.363730/0.165023 | 75.771325/0.241452 |
| 0p5x | 89.843585/-0.001555 | 80.617559/0.160045 | 75.582924/0.243481 |
| 1x | 89.657261/0.001624 | 81.272553/0.148373 | 75.690078/0.241540 |
| 5x | 89.059149/0.013844 | 84.531300/0.092815 | 77.229097/0.216688 |
| 10x | 87.732143/0.038350 | 88.520574/0.024717 | 77.406760/0.215356 |

## 6. Output Files
| file | meaning |
| --- | --- |
| data/epsilon_sweep_attack_rows_fixed25_7models.csv | 8 epsilon x 175 attack rows |
| data/epsilon_sweep_attack_residual_joined_fixed25_7models.csv | attack rows aligned with residual scalar metrics |
| data/epsilon_sweep_residual_correlations.csv | Pearson/Spearman correlations |
| data/epsilon_sweep_by_model_mean_std.csv | by-model attack/residual mean/std |
| data/epsilon_sweep_vector_angles_rows.csv | row-level vector angles for new deltas |
| data/epsilon_sweep_vector_angles_by_model.csv | by-model vector angle mean/std |
| data/epsilon_sweep_winner_counts_by_epsilon.csv | per-epsilon pointwise winner counts |
