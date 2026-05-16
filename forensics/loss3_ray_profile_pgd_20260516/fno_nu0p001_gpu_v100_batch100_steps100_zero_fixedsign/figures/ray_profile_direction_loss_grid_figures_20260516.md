# Ray Profile Direction/Loss Grid Figures - 2026-05-16

Input data: `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/ray_profile.csv`.

These figures use the corrected PGD100 batch-100 Ray Profile run. Each line fixes one direction `v` and evaluates `x + r v` as `r` increases. Values are means over 100 samples.

- Full radius all-metric grid: `normal_batch100_all_metrics_by_direction_full_r_grid.png`
- Local zoom all-metric grid: `normal_batch100_all_metrics_by_direction_local_zoom_grid.png`
- Loss3 mean/std main plot: `normal_batch100_loss3_by_direction_mean_std.png`

Actual sampled radii: 45 points, including `0`, `1e-4`, `1e-3`, `1e-2`, `0.1`, and then every `0.2` from `0.2` to `8.0`. No interpolation was used.
