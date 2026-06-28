# FNO nu=0.001 Loss-Gradient Path Analysis

Date: 2026-05-15 UTC

Result root: `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631`
Analyzed points: `150` = 5 indices x 3 attack losses x 10 saved steps.

This postprocess recomputes exact original-objective gradients for `loss1`, `loss2`, and `loss3` at each saved `x + delta_k` point. The trajectory files themselves save `x_adv`, `delta`, model output, solver output, and residual output; they do not save all three cross-loss gradients at every point.

## Overall Gradient Angles

| n_points | cos_g1_g2_mean | angle_g1_g2_deg_mean | cos_g1_g3_mean | angle_g1_g3_deg_mean | cos_g2_g3_mean | angle_g2_g3_deg_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 150 | 0.9904 | 3.9081 | 0.5269 | 56.5076 | 0.5167 | 57.2331 |

## By Attack Loss

| attack_loss | n_points | angle_g1_g2_deg_mean | angle_g1_g3_deg_mean | angle_g2_g3_deg_mean | cos_g1_g2_mean | cos_g1_g3_mean | cos_g2_g3_mean | loss1_value_mean | loss2_value_mean | loss3_value_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | 50 | 1.0078 | 55.7887 | 55.9767 | 0.9998 | 0.5460 | 0.5435 | 9.7810 | 9.8096 | 3.1522 |
| loss2 | 50 | 1.1215 | 52.4368 | 52.6732 | 0.9998 | 0.5849 | 0.5818 | 9.4836 | 9.5773 | 2.5655 |
| loss3 | 50 | 9.5950 | 61.2974 | 63.0495 | 0.9716 | 0.4498 | 0.4250 | 2.6289 | 2.6140 | 2.3691 |

## Final Step k=max

| attack_loss | n_points | loss1_value_mean | loss2_value_mean | loss3_value_mean | angle_g1_g2_deg_mean | angle_g1_g3_deg_mean | angle_g2_g3_deg_mean | delta_budget_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | 5 | 11.2179 | 11.2427 | 4.0678 | 0.8690 | 51.0046 | 51.2385 | 1.0000 |
| loss2 | 5 | 10.8544 | 10.9451 | 3.6982 | 0.9663 | 41.6277 | 41.8983 | 1.0000 |
| loss3 | 5 | 5.1226 | 5.0549 | 4.6171 | 4.4105 | 47.7987 | 48.5523 | 0.7637 |

## By Saved Step

| k | n_points | angle_g1_g2_deg_mean | angle_g1_g3_deg_mean | angle_g2_g3_deg_mean | loss1_value_mean | loss2_value_mean | loss3_value_mean | delta_budget_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 15 | 12.1978 | 74.2424 | 76.4345 | 3.3695 | 3.4450 | 0.4708 | 0.2252 |
| 10 | 15 | 5.3387 | 68.4714 | 69.0737 | 4.8886 | 4.9379 | 0.9486 | 0.3867 |
| 15 | 15 | 3.8707 | 62.8148 | 63.2059 | 6.2316 | 6.2746 | 1.6407 | 0.5425 |
| 20 | 15 | 3.1432 | 60.4330 | 61.0127 | 7.3491 | 7.3905 | 2.4158 | 0.6883 |
| 25 | 15 | 2.9255 | 54.4790 | 55.0857 | 7.9772 | 8.0153 | 2.9916 | 0.7851 |
| 30 | 15 | 3.0266 | 52.8210 | 53.6348 | 8.1694 | 8.2005 | 3.2448 | 0.8198 |
| 35 | 15 | 2.2042 | 49.8927 | 50.4663 | 8.3900 | 8.4148 | 3.4780 | 0.8514 |
| 40 | 15 | 2.1565 | 47.9260 | 48.5042 | 8.6453 | 8.6662 | 3.6972 | 0.8773 |
| 45 | 15 | 2.1360 | 47.1855 | 47.6839 | 8.8924 | 8.9103 | 3.9407 | 0.9026 |
| 50 | 15 | 2.0819 | 46.8103 | 47.2297 | 9.0650 | 9.0809 | 4.1277 | 0.9212 |

## Files

- `per_point_loss_gradient_angles.csv`
- `summary_by_attack_loss.csv`
- `summary_by_k.csv`
- `summary_by_attack_loss_and_k.csv`
- `summary_by_index_and_attack_loss.csv`
- `final_k_summary_by_attack_loss.csv`
- `summary.json`
