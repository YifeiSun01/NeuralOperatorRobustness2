# Darcy Binary 20260611 Loss3 Advantage Metric Tables

## Scope

- Clean metrics source: `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_52dataset_metric_long_ranked.csv`
- Attack50 source: `outputs/darcy_cflow_final_robustness_20260615/data/attack50_summary_by_dataset_model.csv`
- Generalization root: `generalization_datasets_darcy_binary_loss3targeted_20260611/`
- Final robustness provenance old-root check: `False`
- Attack rows: `18200`, SVD rows: `175`, attack steps: `[50]`

Lower is better for every metric in the tables below. `loss3_dataset_wins` means Loss3 is lowest or tied for lowest; `loss3_dataset_strict_wins` counts only non-tied Loss3 wins. `delta_linf_mean` is included as one of the six attack columns, but it is fixed by the epsilon box and is not a meaningful model-quality winner.

## Main Generalization Evidence

| system | split | metric | metric_label | datasets | loss3_dataset_wins | loss3_dataset_strict_wins | loss3_dataset_win_fraction | best_mean_model | best_mean_model_display | is_loss3_best_by_mean | loss3_mean | next_best_model | next_best_model_display | next_best_mean | loss3_relative_improvement_vs_next_best | mean_ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | generalization | rmse | rmse | 50 | 47 | 47 | 0.94 | loss3 | loss3 | True | 0.00059457 | loss2 | loss2 | 0.00087015 | 0.31670451 | loss3,loss2,random_clean,loss1,physics_loss,baseline,random_solver |
| clean | generalization | relative_l2 | relative_l2 | 50 | 47 | 47 | 0.94 | loss3 | loss3 | True | 0.0561539 | loss2 | loss2 | 0.08163754 | 0.31215594 | loss3,loss2,random_clean,loss1,physics_loss,baseline,random_solver |
| attack50 | generalization | clean_loss_mean | attack clean loss | 50 | 44 | 44 | 0.88 | loss3 | loss3 | True | 2.9e-07 | loss2 | loss2 | 6.3e-07 | 0.54221513 | loss3,loss2,random_clean,physics_loss,loss1,random_solver,baseline |
| attack50 | generalization | adv_loss_mean | attack adv loss | 50 | 50 | 50 | 1 | loss3 | loss3 | True | 1.99e-06 | loss2 | loss2 | 4.19e-06 | 0.52620829 | loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline |
| attack50 | generalization | loss_increase_mean | attack loss increase | 50 | 50 | 50 | 1 | loss3 | loss3 | True | 1.7e-06 | loss2 | loss2 | 3.56e-06 | 0.52338165 | loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline |
| attack50 | generalization | relative_increase_mean | attack relative increase | 50 | 16 | 16 | 0.32 | loss3 | loss3 | True | 8.4812805 | random_solver | random solver | 13.317063 | 0.36312682 | loss3,random_solver,loss1,physics_loss,random_clean,loss2,baseline |

Interpretation: Loss3 is best by mean on clean RMSE, clean Relative L2, attack clean loss, attack adv loss, attack loss increase, and attack relative increase. The strongest per-dataset result is attack adv loss and attack loss increase, where Loss3 wins 50/50 generalization datasets.

## Generalization Mean Values By Model

| system | split | metric | metric_label | model | model_display | mean_value | mean_rank_lower_is_better | is_loss3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | generalization | rmse | rmse | baseline | baseline | 0.0009723512 | 6 | False |
| clean | generalization | rmse | rmse | loss1 | loss1 | 0.0009362592 | 4 | False |
| clean | generalization | rmse | rmse | loss2 | loss2 | 0.0008701539 | 2 | False |
| clean | generalization | rmse | rmse | loss3 | loss3 | 0.0005945722 | 1 | True |
| clean | generalization | rmse | rmse | physics_loss | Physics Loss | 0.000938999 | 5 | False |
| clean | generalization | rmse | rmse | random_clean | random clean | 0.0008839682 | 3 | False |
| clean | generalization | rmse | rmse | random_solver | random solver | 0.0010673193 | 7 | False |
| clean | generalization | relative_l2 | relative_l2 | baseline | baseline | 0.091336901 | 6 | False |
| clean | generalization | relative_l2 | relative_l2 | loss1 | loss1 | 0.087928525 | 4 | False |
| clean | generalization | relative_l2 | relative_l2 | loss2 | loss2 | 0.081637542 | 2 | False |
| clean | generalization | relative_l2 | relative_l2 | loss3 | loss3 | 0.056153898 | 1 | True |
| clean | generalization | relative_l2 | relative_l2 | physics_loss | Physics Loss | 0.088293132 | 5 | False |
| clean | generalization | relative_l2 | relative_l2 | random_clean | random clean | 0.082880698 | 3 | False |
| clean | generalization | relative_l2 | relative_l2 | random_solver | random solver | 0.10056936 | 7 | False |
| attack50 | generalization | clean_loss_mean | attack clean loss | baseline | baseline | 1.4816e-06 | 7 | False |
| attack50 | generalization | clean_loss_mean | attack clean loss | loss1 | loss1 | 7.385e-07 | 5 | False |
| attack50 | generalization | clean_loss_mean | attack clean loss | loss2 | loss2 | 6.295e-07 | 2 | False |
| attack50 | generalization | clean_loss_mean | attack clean loss | loss3 | loss3 | 2.882e-07 | 1 | True |
| attack50 | generalization | clean_loss_mean | attack clean loss | physics_loss | Physics Loss | 7.227e-07 | 4 | False |
| attack50 | generalization | clean_loss_mean | attack clean loss | random_clean | random clean | 6.684e-07 | 3 | False |
| attack50 | generalization | clean_loss_mean | attack clean loss | random_solver | random solver | 9.319e-07 | 6 | False |
| attack50 | generalization | adv_loss_mean | attack adv loss | baseline | baseline | 1.11625e-05 | 7 | False |
| attack50 | generalization | adv_loss_mean | attack adv loss | loss1 | loss1 | 4.6656e-06 | 4 | False |
| attack50 | generalization | adv_loss_mean | attack adv loss | loss2 | loss2 | 4.1942e-06 | 2 | False |
| attack50 | generalization | adv_loss_mean | attack adv loss | loss3 | loss3 | 1.9872e-06 | 1 | True |
| attack50 | generalization | adv_loss_mean | attack adv loss | physics_loss | Physics Loss | 4.6965e-06 | 5 | False |
| attack50 | generalization | adv_loss_mean | attack adv loss | random_clean | random clean | 4.3592e-06 | 3 | False |
| attack50 | generalization | adv_loss_mean | attack adv loss | random_solver | random solver | 5.7559e-06 | 6 | False |
| attack50 | generalization | loss_increase_mean | attack loss increase | baseline | baseline | 9.6809e-06 | 7 | False |
| attack50 | generalization | loss_increase_mean | attack loss increase | loss1 | loss1 | 3.9271e-06 | 4 | False |
| attack50 | generalization | loss_increase_mean | attack loss increase | loss2 | loss2 | 3.5647e-06 | 2 | False |
| attack50 | generalization | loss_increase_mean | attack loss increase | loss3 | loss3 | 1.699e-06 | 1 | True |
| attack50 | generalization | loss_increase_mean | attack loss increase | physics_loss | Physics Loss | 3.9739e-06 | 5 | False |
| attack50 | generalization | loss_increase_mean | attack loss increase | random_clean | random clean | 3.6908e-06 | 3 | False |
| attack50 | generalization | loss_increase_mean | attack loss increase | random_solver | random solver | 4.8241e-06 | 6 | False |
| attack50 | generalization | relative_increase_mean | attack relative increase | baseline | baseline | 17.750654 | 7 | False |
| attack50 | generalization | relative_increase_mean | attack relative increase | loss1 | loss1 | 14.077819 | 3 | False |
| attack50 | generalization | relative_increase_mean | attack relative increase | loss2 | loss2 | 16.465996 | 6 | False |
| attack50 | generalization | relative_increase_mean | attack relative increase | loss3 | loss3 | 8.4812805 | 1 | True |
| attack50 | generalization | relative_increase_mean | attack relative increase | physics_loss | Physics Loss | 15.175773 | 4 | False |
| attack50 | generalization | relative_increase_mean | attack relative increase | random_clean | random clean | 15.27862 | 5 | False |
| attack50 | generalization | relative_increase_mean | attack relative increase | random_solver | random solver | 13.317063 | 2 | False |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | baseline | baseline | 4.1927349 | 7 | False |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | loss1 | loss1 | 3.6867407 | 4 | False |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | loss2 | loss2 | 3.6725337 | 2 | False |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | loss3 | loss3 | 3.6820211 | 3 | True |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | physics_loss | Physics Loss | 3.6938381 | 5 | False |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | random_clean | random clean | 3.672501 | 1 | False |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | random_solver | random solver | 3.7264271 | 6 | False |
| attack50 | generalization | delta_linf_mean | delta Linf | baseline | baseline | 9 | 1 | False |
| attack50 | generalization | delta_linf_mean | delta Linf | loss1 | loss1 | 9 | 1 | False |
| attack50 | generalization | delta_linf_mean | delta Linf | loss2 | loss2 | 9 | 1 | False |
| attack50 | generalization | delta_linf_mean | delta Linf | loss3 | loss3 | 9 | 1 | True |
| attack50 | generalization | delta_linf_mean | delta Linf | physics_loss | Physics Loss | 9 | 1 | False |
| attack50 | generalization | delta_linf_mean | delta Linf | random_clean | random clean | 9 | 1 | False |
| attack50 | generalization | delta_linf_mean | delta Linf | random_solver | random solver | 9 | 1 | False |

## Attack50 Generalization Six-Metric Summary

| system | split | metric | metric_label | datasets | loss3_dataset_wins | loss3_dataset_strict_wins | loss3_dataset_win_fraction | best_mean_model | best_mean_model_display | is_loss3_best_by_mean | loss3_mean | next_best_model | next_best_model_display | next_best_mean | loss3_relative_improvement_vs_next_best | mean_ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attack50 | generalization | clean_loss_mean | attack clean loss | 50 | 44 | 44 | 0.88 | loss3 | loss3 | True | 2.9e-07 | loss2 | loss2 | 6.3e-07 | 0.54221513 | loss3,loss2,random_clean,physics_loss,loss1,random_solver,baseline |
| attack50 | generalization | adv_loss_mean | attack adv loss | 50 | 50 | 50 | 1 | loss3 | loss3 | True | 1.99e-06 | loss2 | loss2 | 4.19e-06 | 0.52620829 | loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline |
| attack50 | generalization | loss_increase_mean | attack loss increase | 50 | 50 | 50 | 1 | loss3 | loss3 | True | 1.7e-06 | loss2 | loss2 | 3.56e-06 | 0.52338165 | loss3,loss2,random_clean,loss1,physics_loss,random_solver,baseline |
| attack50 | generalization | relative_increase_mean | attack relative increase | 50 | 16 | 16 | 0.32 | loss3 | loss3 | True | 8.4812805 | random_solver | random solver | 13.317063 | 0.36312682 | loss3,random_solver,loss1,physics_loss,random_clean,loss2,baseline |
| attack50 | generalization | delta_l2_rms_mean | delta L2 RMS | 50 | 21 | 21 | 0.42 | random_clean | random clean | False | 3.6820212 | random_clean | random clean | 3.672501 | -0.00259228 | random_clean,loss2,loss3,loss1,physics_loss,random_solver,baseline |
| attack50 | generalization | delta_linf_mean | delta Linf | 50 | 50 | 0 | 1 | baseline,loss1,loss2,loss3,physics_loss,random_clean,random_solver | baseline,loss1,loss2,loss3,Physics Loss,random clean,random solver | True | 9 | baseline | baseline | 9 | 0 | baseline,loss1,loss2,loss3,physics_loss,random_clean,random_solver |

## Clean Generalization RMSE/Relative L2 Summary

| system | split | metric | metric_label | datasets | loss3_dataset_wins | loss3_dataset_strict_wins | loss3_dataset_win_fraction | best_mean_model | best_mean_model_display | is_loss3_best_by_mean | loss3_mean | next_best_model | next_best_model_display | next_best_mean | loss3_relative_improvement_vs_next_best | mean_ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | generalization | rmse | rmse | 50 | 47 | 47 | 0.94 | loss3 | loss3 | True | 0.00059457 | loss2 | loss2 | 0.00087015 | 0.31670451 | loss3,loss2,random_clean,loss1,physics_loss,baseline,random_solver |
| clean | generalization | relative_l2 | relative_l2 | 50 | 47 | 47 | 0.94 | loss3 | loss3 | True | 0.0561539 | loss2 | loss2 | 0.08163754 | 0.31215594 | loss3,loss2,random_clean,loss1,physics_loss,baseline,random_solver |

## Files

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/clean_52dataset_7model_rmse_relative_l2_long.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/attack50_52dataset_7model_6metrics_long.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/loss3_win_summary_by_split_metric.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/model_mean_values_by_split_metric.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/per_dataset_metric_winners.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/generalization50_metric_winners_wide.csv`

## Caveat

The precise claim supported by these tables is that Loss3 is the best overall model on the main binary-20260611 generalization and attack50 robustness metrics. It is not literally the strict winner of every auxiliary per-dataset diagnostic: clean RMSE/Relative L2 are 47/50 per-dataset wins, attack clean loss is 44/50, attack relative increase is best by mean but only 16/50 strict per-dataset wins, and delta magnitude metrics are not primary robustness-quality metrics.
