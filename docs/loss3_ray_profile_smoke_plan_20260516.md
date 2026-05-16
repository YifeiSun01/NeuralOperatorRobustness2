# Loss3 Experiment 4 Ray Profile Plan - 2026-05-16

Scope: FNO / 1D Burgers `nu=0.001`, GPU-only.

## Purpose

This experiment checks the local-to-global gap: a direction can look optimal at very small radius but fail to maximize the finite-radius endpoint residual. The diagnostic walks along fixed directions `v` and records the whole curve `x + r v` for `r in [0, epsilon]`.

## Direction Sources

- `loss3_original`: direction from finite-radius optimization of `||e(x+delta)||`.
- `loss3_increment_ratio`: direction from finite-radius optimization of `(||e(x+delta)||-||e(x)||)/(||delta||+eta)`.
- `loss3_residual_increment_ratio`: direction from finite-radius optimization of `||e(x+delta)-e(x)||/(||delta||+eta)`.
- `loss3_regularized`: direction from finite-radius optimization of `||e(x+delta)|| - C||delta||`.
- `local_error_svd`: clean local top direction of `J_e`.
- `local_outward_growth`: clean local direction `normalize(J_e^T e/||e||)`.
- `random`: seeded random baseline direction.

## Curves Recorded

For each sample, direction, and radius:

```text
endpoint residual norm    = ||e(x+r v)||_2
norm growth ratio         = (||e(x+r v)||_2 - ||e(x)||_2) / (r + eta)
residual increment ratio  = ||e(x+r v)-e(x)||_2 / (r + eta)
model movement ratio      = ||f(x+r v)-f(x)||_2 / (r + eta)
solver movement ratio     = ||j(x+r v)-j(x)||_2 / (r + eta)
cos_delta_f_delta_j       = cos(f movement, solver movement)
```

## Settings

- Sample indices: `[0]`.
- Epsilon: `0.5`.
- Attack steps per optimized direction: `2`.
- Attack optimizer: Adam ascent on `delta`, projected to the L2 ball after every step.
- Ray radii: linear grid with `5` points plus small radii `1e-4`, `1e-3`, `1e-2`, `1e-1`.
- Official run must use GPU; CPU fallback is refused.

## Expected Nonlinear Evidence

- If a local ratio direction has high small-radius ratio but loses at `r=epsilon`, this shows the local objective is not the finite-radius endpoint objective.
- If the winner changes between the small-radius ratio and endpoint `||e(x+epsilon v)||`, this directly demonstrates local-to-global nonlinear drift.
- If curves bend, saturate, or cross, then a single clean-point direction is insufficient to explain the finite-radius attack.
