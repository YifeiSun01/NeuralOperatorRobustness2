# Corrected Loss3 Experiment 4 Ray Profile Plan - 2026-05-16

Scope: FNO / 1D Burgers `nu=0.001`, GPU-only, batch size 20.

This corrected run supersedes the first ray-profile endpoint-winner interpretation. The first run used the final `loss3_original` step, not best-over-steps, and therefore was not a fair endpoint comparison.

## Corrected Design

- Keep the ray diagnostic: fix a direction `v`, evaluate `x + r v` for `r in [0, epsilon]`.
- Compute `local_outward_growth = normalize(grad_x ||e(x)||)`, the small-radius limit of `loss3_increment_ratio`.
- Compute a local residual-movement direction by optimizing the small-radius residual increment ratio.
- Recompute finite-radius control directions with PGD: `loss3_increment_ratio`, `loss3_residual_increment_ratio`, and `loss3_regularized`.
- Recompute the main endpoint direction with direct `loss3_original` PGD, best-over-steps, and multiple restarts.
- The `loss3_original` restarts include the boundary-normalized control directions, so the endpoint comparison is fair: direct endpoint optimization gets a chance to improve from every control ray.

## Settings

- Sample indices: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 40, 47, 115]`.
- Epsilon: `8.0`.
- Attack steps per restart: `100`.
- Attack update: `raw` PGD, learning rate `0.3`.
- Original-loss restarts: `['zero', 'local_outward_small', 'local_residual_small', 'random_boundary_00', 'random_boundary_01', 'boundary_from:loss3_increment_ratio_pgd_best', 'boundary_from:loss3_residual_increment_ratio_pgd_best', 'boundary_from:loss3_regularized_pgd_best']`.
- Control directions: `['loss3_increment_ratio_pgd_best', 'loss3_residual_increment_ratio_pgd_best', 'loss3_regularized_pgd_best']`.
- Official run must use GPU; CPU fallback is refused.
