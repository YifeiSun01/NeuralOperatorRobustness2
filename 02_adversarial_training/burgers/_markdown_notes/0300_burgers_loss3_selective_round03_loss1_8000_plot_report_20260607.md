# Burgers Round03 Final Extension Plot Report - 2026-06-06

This report joins original and continuation run directories so curves extend through loss1 epoch8000, loss2 epoch2000, loss3 epoch1500.

## Final And Best Generated Generalization

| loss | final epoch | final wall h | final gen RMSE | final gen rel L2 | best gen epoch | best gen RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 8000 | 11.9919 | 0.0309312 | 0.0554492 | 7955 | 0.0301536 |
| loss2 | 2000 | 12.423 | 0.0357396 | 0.064081 | 1784 | 0.034426 |
| loss3 | 1500 | 19.3331 | 0.0216375 | 0.0387743 | 893 | 0.0193611 |

## Outputs

- Dense output directory: `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605`
- Compact output directory: `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_epoch_metrics_every1.csv`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_same_epoch_rmse_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_wall_clock_rmse_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_same_epoch_relative_l2_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_wall_clock_relative_l2_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_epoch_metrics_every5.csv`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_same_epoch_rmse_every5_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_wall_clock_rmse_every5_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_same_epoch_relative_l2_every5_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_wall_clock_relative_l2_every5_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_final_rmse_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_generated_final_vs_best_rmse_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_same_epoch_rmse.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_wall_clock_rmse.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_final_rmse_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_generated_final_vs_best_rmse_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_final_relative_l2_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_generated_final_vs_best_relative_l2_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_same_epoch_relative_l2.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_wall_clock_relative_l2.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_final_relative_l2_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_generated_final_vs_best_relative_l2_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_final_generated50_loss3_advantage_hist.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_plot_manifest.txt`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_long_training_comparison_plot_manifest.txt`

## Folder Consolidation - 2026-06-07T20:05:14Z

Observed current official output folders: `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605`, `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`, and `visualizations/burgers_loss3_selective_round03_loss1_8000ep_long_20260605_plots`.

Observed deletion/rename cleanup: the standalone `visualizations/burgers_loss3_selective_round03_loss1_8000_dense_comparison_20260607` and `visualizations/burgers_loss3_selective_round03_loss1_8000_comparison_20260607` folders were deleted; the single-run output now uses the renamed `loss1_8000ep` folder rather than a separate newly-created folder.

Observed every-epoch evidence: `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_epoch_metrics_every1.csv` has loss1 epochs `0..8000`, with `8001` points for each of `train`, `test`, and `generalization`. The every5 files remain only as optional companion outputs.

Observed FFT selected-epoch fix: selected spectra now include `50, 200, 400, 800, 1200, 2000, 3000, 4000, 5000, 6000, 7000, 8000` for the 8000-epoch single-run plot.

