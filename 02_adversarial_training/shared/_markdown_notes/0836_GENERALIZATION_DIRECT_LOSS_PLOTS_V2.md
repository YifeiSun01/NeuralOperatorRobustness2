# Generalization Direct-Loss Bar Plots V2

This is the second visualization version requested by the user. It does not use `100 / (1 + relative_l2)` and does not plot relative L2.

The bar height is direct evaluation loss measured as `RMSE` from `generalization_eval/metrics_sorted_by_similarity.csv`; lower is better. Train and test are placed first, followed by generated datasets sorted from closest to farthest using `feature_distance_to_train`.

## Plots

- `generalization_eval/burgers_generalization_direct_loss_rmse_barplot_v2.png`
- `generalization_eval/darcy_generalization_direct_loss_rmse_barplot_v2.png`
- `generalization_eval/ns2d_generalization_direct_loss_rmse_barplot_v2.png`

## Notes

- The existing `metrics.csv` also contains `MAE` and `relative_l2`, but this v2 plot uses only `RMSE` as the direct loss.
- NS2D non-finite rollout values were handled by the finite-value mask already recorded in `invalid_value_fraction`.
