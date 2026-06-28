# Loss3 Small-Epsilon Sweep Result - 2026-05-16

Status: completed for FNO / 1D Burgers `nu=0.001` using a fixed candidate-direction bank.

## Scope

- Model: FNO1d, `nu=0.001`.
- Indices: `[0]`.
- Epsilons: `[0.0001]`.
- Input/output norm: L2.
- Output directory: `forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001`.

The candidate bank uses clean-point top right singular vectors from `J_f`, `J_j`, and `J_e=J_f-J_j`, both signs of those directions, outward-growth directions, and seeded random directions.

## Local References

| objective | local reference mean | local reference std |
| --- | --- | --- |
| G_e | 0.2049 | 0 |
| L_e | 1.206 | 0 |
| L_f | 4.504 | 0 |
| L_j | 4.875 | 0 |

## Aggregate Sweep Values

| objective | epsilon | best mean | best/local | abs cos local |
| --- | --- | --- | --- | --- |
| G_e | 0.0001 | 0.202 | 0.9856 | 1 |
| L_e | 0.0001 | 1.209 | 1.003 | 1 |
| L_f | 0.0001 | 4.509 | 1.001 | 1 |
| L_j | 0.0001 | 4.873 | 0.9995 | 1 |

## Direction Stability

| objective | eps a | eps b | abs cosine mean | abs cosine min |
| --- | --- | --- | --- | --- |

## Visualizations

![small epsilon best values](../forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001/figures/small_epsilon_best_values.png)

![small epsilon local reference ratio](../forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001/figures/small_epsilon_local_reference_ratio.png)

![small epsilon direction stability heatmap](../forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001/figures/small_epsilon_direction_stability_heatmap.png)

![small epsilon best direction sources](../forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001/figures/small_epsilon_best_direction_sources.png)

## Conclusion

Observed from the completed sweep:

- At the smallest radius, `L_f` is `4.509` on average, while `L_e` is `1.209`. This preserves the main mechanism: model-only sensitivity is much larger than residual/error-field sensitivity.
- The local clean-error outward-growth diagnostic `G_e` is much smaller than `L_e`: at the smallest radius, `G_e` is `0.202` on average.
- At the largest tested local radius, `G_e` is `0.202` on average, so this diagnostic remains distinct from the residual-movement quantity `L_e`.
- Direction-stability tables record whether the selected candidate direction remains stable as epsilon grows; low cross-epsilon cosine means nonlinear/local-to-finite-radius drift is already visible.

Inference:

The Experiment 3 evidence supports the intended distinction in the `loss3_original` plan: local ratio-style diagnostics are meaningful at small radii, but they are different objects from the finite-radius endpoint objective. In particular, `L_e` measures how fast the error field moves, while `G_e` measures whether the current clean residual norm is pushed outward.

## Output Files

- `candidate_metrics.csv`: every evaluated candidate direction.
- `best_by_objective_epsilon_index.csv`: per-index selected direction/value.
- `aggregate_best_by_objective_epsilon.csv`: aggregate values used above.
- `direction_stability.csv` and `direction_stability_summary.csv`: cross-epsilon selected-direction cosines.
- `local_reference_by_index.csv` and `local_reference_summary.csv`: clean-point local references.
- `manifest.json`: reproducibility manifest.
