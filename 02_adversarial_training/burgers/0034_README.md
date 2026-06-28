# Loss1 vs Loss3 Wall-Clock Comparison

Key alignment points:

- Loss1 epoch 1000 took about 1.530 h; same wall-clock for loss3 is epoch 103.
- Loss3 1/5 is epoch 200 at about 2.968 h; same wall-clock for loss1 is epoch 1946.
- Loss1 epoch 2000 is the saved checkpoint near that time, but it is about 5.13 minutes later than exact loss3 1/5.

Main files:

- `loss1_epoch1000_and_1over5_wallclock_comparison.csv`
- `loss1_epoch1000_vs_loss3_same_wall_rmse_bars.png`
- `loss1_epoch1000_vs_loss3_same_wall_relative_l2_bars.png`
- `loss1_same_wall_as_loss3_1over5_rmse_bars.png`
- `loss1_same_wall_as_loss3_1over5_relative_l2_bars.png`
- `loss1_loss3_wallclock_test_rmse_curve.png`
- `loss1_loss3_wallclock_generated50_rmse_curve.png`
- `loss1_loss3_wallclock_test_relative_l2_curve.png`
- `loss1_loss3_wallclock_generated50_relative_l2_curve.png`
