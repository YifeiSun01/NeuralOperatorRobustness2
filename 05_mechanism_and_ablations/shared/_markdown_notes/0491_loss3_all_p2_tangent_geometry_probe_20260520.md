# Loss3 All p=2 Tangent Geometry Probe - 2026-05-20

Status: generated from existing core4 per-sample metrics; no optimizer/model experiment was rerun.

## Scope

- Setting roots processed: `60`
- Sample/method rows: `24000`
- Output directory: `forensics/loss3_all_p2_tangent_geometry_probe_20260520`

This extends the p2q2 tangent KKT residual check to all available p=2 settings in the current core4 sweeps: p2q2, p2q1, and p2qinf where files exist.

Metric:

```text
tangent_residual = sqrt(1 - cos(delta, grad)^2)
```

For p=2 this equals `||(I-u u^T) grad L|| / ||grad L||`.

## Rollup By PQ And Method

| pq | method | n_sample_rows | hit_step_99_mean | post_boundary_gain_mean | tangent_residual_at_hit_mean | tangent_residual_last_grad_mean | mean_angle_after_hit_deg_mean | final_loss3_q_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p2q1 | raw_add | 2000 | 3.545 | 42.07 | 0.6179 | 0.2014 | 8.848 | 105.1 |
| p2q1 | raw_replace | 2000 | 1 | 70.32 | 0.8729 | 0.3989 | 25.73 | 100.7 |
| p2q1 | steepest_add | 2000 | 12.93 | 26.52 | 0.484 | 0.07231 | 0.8846 | 100.6 |
| p2q1 | steepest_replace | 2000 | 1 | 70.32 | 0.8729 | 0.3989 | 25.73 | 100.7 |
| p2q2 | raw_add | 2000 | 32.41 | 1.28 | 0.397 | 0.03876 | 0.2812 | 5.639 |
| p2q2 | raw_replace | 2000 | 1 | 3.412 | 0.8552 | 0.4802 | 30.68 | 5.747 |
| p2q2 | steepest_add | 2000 | 12.75 | 1.467 | 0.4349 | 0.04624 | 0.605 | 5.817 |
| p2q2 | steepest_replace | 2000 | 1 | 3.412 | 0.8552 | 0.4802 | 30.68 | 5.747 |
| p2qinf | raw_add | 2000 | 51.21 | 0.0233 | 0.7215 | 0.7503 | 1.724 | 0.4976 |
| p2qinf | raw_replace | 2000 | 1 | 0.1558 | 0.9381 | 0.8525 | 66.89 | 0.5545 |
| p2qinf | steepest_add | 2000 | 22.1 | 0.1764 | 0.8679 | 0.7912 | 6.678 | 0.7864 |
| p2qinf | steepest_replace | 2000 | 1 | 0.1558 | 0.9381 | 0.8525 | 66.89 | 0.5545 |

## Correlations With Post-Boundary Gain

| pq | method | metric | pearson_r_with_post_boundary_gain | pair_count |
| --- | --- | --- | --- | --- |
| p2q1 | raw_add | tangent_residual_at_hit | 0.5816 | 2000 |
| p2q1 | raw_add | radial_signed_ratio_at_hit | -0.6177 | 2000 |
| p2q1 | raw_add | tangent_residual_last_grad | 0.1522 | 2000 |
| p2q1 | raw_add | mean_angle_after_hit_deg | 0.2562 | 2000 |
| p2q1 | raw_add | mean_tangent_residual_after_hit | 0.1794 | 2000 |
| p2q1 | raw_replace | tangent_residual_at_hit | 0.4266 | 2000 |
| p2q1 | raw_replace | radial_signed_ratio_at_hit | -0.453 | 2000 |
| p2q1 | raw_replace | tangent_residual_last_grad | 0.02308 | 2000 |
| p2q1 | raw_replace | mean_angle_after_hit_deg | -0.03426 | 2000 |
| p2q1 | raw_replace | mean_tangent_residual_after_hit | 0.01507 | 2000 |
| p2q1 | steepest_add | tangent_residual_at_hit | 0.5861 | 2000 |
| p2q1 | steepest_add | radial_signed_ratio_at_hit | -0.6262 | 2000 |
| p2q1 | steepest_add | tangent_residual_last_grad | 0.03387 | 2000 |
| p2q1 | steepest_add | mean_angle_after_hit_deg | 0.1453 | 2000 |
| p2q1 | steepest_add | mean_tangent_residual_after_hit | 0.2549 | 2000 |
| p2q1 | steepest_replace | tangent_residual_at_hit | 0.4266 | 2000 |
| p2q1 | steepest_replace | radial_signed_ratio_at_hit | -0.453 | 2000 |
| p2q1 | steepest_replace | tangent_residual_last_grad | 0.02308 | 2000 |
| p2q1 | steepest_replace | mean_angle_after_hit_deg | -0.03426 | 2000 |
| p2q1 | steepest_replace | mean_tangent_residual_after_hit | 0.01507 | 2000 |
| p2q2 | raw_add | tangent_residual_at_hit | 0.5713 | 1996 |
| p2q2 | raw_add | radial_signed_ratio_at_hit | -0.5945 | 1996 |
| p2q2 | raw_add | tangent_residual_last_grad | -0.02803 | 1996 |
| p2q2 | raw_add | mean_angle_after_hit_deg | 0.1139 | 1996 |
| p2q2 | raw_add | mean_tangent_residual_after_hit | 0.2001 | 1996 |
| p2q2 | raw_replace | tangent_residual_at_hit | 0.4981 | 2000 |
| p2q2 | raw_replace | radial_signed_ratio_at_hit | -0.5188 | 2000 |
| p2q2 | raw_replace | tangent_residual_last_grad | -0.1148 | 2000 |
| p2q2 | raw_replace | mean_angle_after_hit_deg | -0.1575 | 2000 |
| p2q2 | raw_replace | mean_tangent_residual_after_hit | -0.1062 | 2000 |
| p2q2 | steepest_add | tangent_residual_at_hit | 0.5671 | 1998 |
| p2q2 | steepest_add | radial_signed_ratio_at_hit | -0.5599 | 1998 |
| p2q2 | steepest_add | tangent_residual_last_grad | -0.05921 | 1998 |
| p2q2 | steepest_add | mean_angle_after_hit_deg | -0.005755 | 1998 |
| p2q2 | steepest_add | mean_tangent_residual_after_hit | 0.1219 | 1998 |
| p2q2 | steepest_replace | tangent_residual_at_hit | 0.4981 | 2000 |
| p2q2 | steepest_replace | radial_signed_ratio_at_hit | -0.5188 | 2000 |
| p2q2 | steepest_replace | tangent_residual_last_grad | -0.1148 | 2000 |
| p2q2 | steepest_replace | mean_angle_after_hit_deg | -0.1575 | 2000 |
| p2q2 | steepest_replace | mean_tangent_residual_after_hit | -0.1062 | 2000 |
| p2qinf | raw_add | tangent_residual_at_hit | 0.144 | 761 |
| p2qinf | raw_add | radial_signed_ratio_at_hit | -0.1125 | 761 |
| p2qinf | raw_add | tangent_residual_last_grad | 0.03015 | 765 |
| p2qinf | raw_add | mean_angle_after_hit_deg | 0.2047 | 761 |
| p2qinf | raw_add | mean_tangent_residual_after_hit | 0.1196 | 761 |
| p2qinf | raw_replace | tangent_residual_at_hit | 0.1891 | 1996 |
| p2qinf | raw_replace | radial_signed_ratio_at_hit | -0.1971 | 1996 |
| p2qinf | raw_replace | tangent_residual_last_grad | -0.4026 | 1996 |
| p2qinf | raw_replace | mean_angle_after_hit_deg | -0.3317 | 1996 |
| p2qinf | raw_replace | mean_tangent_residual_after_hit | -0.3111 | 1996 |
| p2qinf | steepest_add | tangent_residual_at_hit | 0.1974 | 1873 |
| p2qinf | steepest_add | radial_signed_ratio_at_hit | -0.166 | 1873 |
| p2qinf | steepest_add | tangent_residual_last_grad | -0.03077 | 1875 |
| p2qinf | steepest_add | mean_angle_after_hit_deg | 0.07002 | 1873 |
| p2qinf | steepest_add | mean_tangent_residual_after_hit | 0.04642 | 1873 |
| p2qinf | steepest_replace | tangent_residual_at_hit | 0.1891 | 1996 |
| p2qinf | steepest_replace | radial_signed_ratio_at_hit | -0.1971 | 1996 |
| p2qinf | steepest_replace | tangent_residual_last_grad | -0.4026 | 1996 |
| p2qinf | steepest_replace | mean_angle_after_hit_deg | -0.3317 | 1996 |
| p2qinf | steepest_replace | mean_tangent_residual_after_hit | -0.3111 | 1996 |

## Interpretation

- Observed evidence here can support or qualify the p2 boundary-stationarity story across q values.
- This still does not evaluate new landscape points. It only uses the gradients/cosines already saved during the completed optimizer runs.
- If p2qinf has high tangent residual but weak gain or high peakiness elsewhere, that supports the idea that tangent opportunity must align with useful/non-spiky loss geometry.

## Observed Conclusions

Observed from `p2_tangent_rollup_by_pq_method.csv`:

- Replacement reaches the 99% boundary at step 1 for all checked p=2 geometries (`p2q1`, `p2q2`, `p2qinf`).
- Tangent residual at boundary hit is high for replacement across q: about `0.873` for `p2q1`, `0.855` for `p2q2`, and `0.938` for `p2qinf`.
- High tangent residual alone is not enough. `p2qinf` has very high replacement tangent residual and very large angular motion (`66.9 deg` mean after hit), but post-boundary gain is tiny (`0.156`) compared with `p2q1` (`70.3`) and `p2q2` (`3.41`). This strengthens the caveat that tangent opportunity must align with a useful loss geometry.
- For additive methods, tangent residual at hit correlates positively with post-boundary gain in `p2q1` and `p2q2` (`r ~ 0.57-0.59`) but much more weakly in `p2qinf` (`r ~ 0.14-0.20`).

Inference:

- The p=2 boundary-stationarity explanation is real, but q changes whether that tangent opportunity turns into useful gain.
- `p2qinf` is the strongest warning case: it has the geometry signal, but the loss landscape/active-coordinate behavior makes the signal less predictive.

