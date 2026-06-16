# Darcy SIR20 Required Figures Only

This folder intentionally contains only the requested figures.

The current figures are six-method common-range plots:
`loss1`, `loss2`, `loss3`, `physics`, `random clean`, `random solver`.
Because the available random runs end at epoch 1100, the epoch plots are
truncated to the common 0-1100 range and work-clock plots are truncated to the
common work-clock range shared by all six methods.

Included:

- Six-method RMSE train/test/generalization mean vs epoch.
- Six-method RMSE train/test/generalization mean vs work-clock.
- Six-method Relative L2 train/test/generalization mean vs epoch.
- Six-method Relative L2 train/test/generalization mean vs work-clock.
- Six-method RMSE 50 generalization datasets split into part01/part02 vs epoch.
- Six-method RMSE 50 generalization datasets split into part01/part02 vs work-clock.
- Six-method Relative L2 50 generalization datasets split into part01/part02 vs epoch.
- Six-method Relative L2 50 generalization datasets split into part01/part02 vs work-clock.
- One Darcy 2D attack heatmap selected for loss3 advantage.

Excluded:

- zoom figures;
- optimizer-loss figures;
- grad-norm figures;
- bar charts;
- contact sheets;
- extra random-inclusive legacy figures.
