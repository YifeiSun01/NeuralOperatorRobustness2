# Darcy Loss4 Observer Curves

Setting: `K=10920`, method `steepest_replace`, sample00 step traces.

This evaluates the physics loss4 metric offline along attacks optimized for loss1/loss2/loss3/loss4.
For every step, `A_adv` and `model_u` from `step_sample_trace.npz` are plugged into the same Darcy physics-loss function used by the loss4 attack:

`loss4 = loss4_pde + loss4_bc` with `bc_weight=1.0` and `physics_metric=rel_l2`.

## Final Values

| optimized loss | step0 loss4 | final loss4 | increase | ratio | final true loss3 | source |
|---|---:|---:|---:|---:|---:|---|
| loss1 | 21.479 | 33.7417 | 12.2627 | 1.57091 | 0.0529069 | `2D_Darcy_FNO2d/perturbation_results/binary_loss12_steepest_replace_budget_completion/darcy_loss12_steepest_replace_budget_completion_nx211_N50_steps100_20260528/runs/loss12_steepest_replace_nx211_N50_steps100_K10920_pct25_alpha125_20260528/loss_method_grid/loss1/steepest_replace/step_sample_trace.npz` |
| loss2 | 14.6585 | 28.121 | 13.4625 | 1.91841 | 0.0443383 | `2D_Darcy_FNO2d/perturbation_results/binary_loss12_steepest_replace_budget_completion/darcy_loss12_steepest_replace_budget_completion_nx211_N50_steps100_20260528/runs/loss12_steepest_replace_nx211_N50_steps100_K10920_pct25_alpha125_20260528/loss_method_grid/loss2/steepest_replace/step_sample_trace.npz` |
| loss3 | 14.6585 | 39.5498 | 24.8913 | 2.69807 | 0.21267 | `2D_Darcy_FNO2d/perturbation_results/binary_loss3_steepest_replace_budget_sweep/darcy_loss3_steepest_replace_budget_sweep_nx211_N50_steps100_20260528/runs/loss3_steepest_replace_nx211_N50_steps100_K10920_pct25_alpha125_20260528/loss_method_grid/loss3/steepest_replace/step_sample_trace.npz` |
| loss4 | 14.6419 | 186.846 | 172.204 | 12.7611 | 0.0784569 | `2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics_budget_sweep/darcy_loss4_physics_steepest_replace_nx211_N50_steps100_K10920_pct0p25_alpha125_bc_20260529/step_sample_trace.npz` |

## Outputs

- `darcy_K10920_steepest_replace_loss4_observer_curves.png`
- `darcy_K10920_steepest_replace_loss4_observer_components.png`
- `darcy_K10920_steepest_replace_loss4_observer_curves.csv`
