# Loss3 replace/add validation status - 2026-06-23

Status at `2026-06-23 00:53 UTC`: no attack/mechanism jobs are running. GPU is idle.

## Completed Runs

- Main three-system Loss3 optimizer mean-curve figure was already generated under `analysis_outputs/optimizer_ablation_20260622/figures/`.
- NS2D mechanism trace completed:
  - Root: `analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace`
  - Parameters: `eps=32`, `alpha=10`, `steps=100`, `mode=all_w`, `N=3`, four methods.
  - Done marker: `analysis_outputs/mechanism_20260622/run_logs/ns2d_eps32_alpha10_steps100_N3_trace_20260622.done`
  - Completed trace files: `12` method/sample summaries.
- Burgers validation trajectory trace completed:
  - Root: `analysis_outputs/mechanism_20260622/replace_add_validation/raw/burgers_eps8_alpha0p3_steps100_N5_trace`
  - Parameters: `eps=8`, `alpha=0.3`, `steps=100`, `N=5`, four methods.
- Burgers first-order prediction probe completed:
  - Root: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_first_order_prediction`
- Burgers landscape/ridge probe completed:
  - Root: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe`
  - Rows: ray `1700`, boundary arc `255`, 2D slice `250`, curvature `20`.
- NS2D direction stability diagnostic completed:
  - Root: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/ns2d_direction_stability`

## NS2D Direction Diagnostic, N=3

Aggregate means from `step_sample_direction_summary.csv`:

| method | final true loss | first boundary k | loss at boundary | gain after boundary | mean direction cos to previous | cos(final delta, k1) | cos(final delta, k5) | cos(final delta, k10) | cos(final delta, k20) | cos(final delta, k50) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_add | 121.892 | 1 | 136.140 | -14.248 | 0.145 | 0.119 | 0.407 | 0.269 | -0.023 | 0.396 |
| raw_replace | 85.086 | 1 | 136.139 | -51.053 | 0.155 | 0.054 | 0.087 | 0.178 | -0.122 | 0.027 |
| steepest_add | 255.475 | 6 | 212.043 | 43.432 | -0.315 | 0.383 | 0.587 | 0.724 | 0.843 | 0.940 |
| steepest_replace | 85.086 | 1 | 136.139 | -51.053 | 0.155 | 0.054 | 0.087 | 0.178 | -0.122 | 0.027 |

Interpretation: in this small NS2D mechanism batch, replacement reaches the boundary immediately but then loses substantial true loss relative to the boundary point. `steepest_add` reaches the boundary later and its perturbation direction becomes progressively aligned with the final direction. This supports the explanation that NS2D needs path-dependent direction accumulation/rotation; a full-budget replacement jump can lock onto a poor moving-target direction.

## Burgers First-Order Prediction Probe, N=5

Aggregate from `first_order_prediction_by_candidate.csv`:

| candidate | n | Pearson(pred,true) | predicted gain mean | true gain mean | relative abs error mean | relative abs error median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| add_from_saved_direction | 140 | 0.998 | 0.084 | 0.085 | 0.029 | 0.017 |
| replace_from_saved_direction | 140 | 0.639 | 1.566 | 0.824 | 3.407 | 0.416 |
| actual_next_step | 140 | 0.721 | 1.107 | 0.405 | 3.302 | 0.118 |
| final_delta | 160 | 0.190 | 0.542 | 2.697 | 0.831 | 0.698 |
| random_boundary | 160 | 0.931 | -3.230 | -2.527 | 1.872 | 0.539 |

Interpretation: local additive steps are very well predicted by the local linear model. Full replacement jumps are not uniformly well predicted by one-step first-order linearization, and are often over-predicted. Therefore Burgers replacement success should not be explained as "the full-budget local linear model is globally accurate." The better explanation is fast boundary arrival plus a forgiving high-loss boundary/ridge structure and enough direction quality.

## Burgers Landscape/Ridge Probe, N=5

Final-radius ray means:

| method | final-radius Loss3 mean | std |
| --- | ---: | ---: |
| raw_add | 4.295 | 2.163 |
| raw_replace | 6.115 | 1.100 |
| steepest_add | 6.392 | 2.085 |
| steepest_replace | 6.115 | 1.100 |

Boundary arc aggregate:

| pair | endpoint s=0 | endpoint s=1 | min along arc | max along arc |
| --- | ---: | ---: | ---: | ---: |
| raw_add -> steepest_add | 4.295 | 6.392 | 4.295 | 6.392 |
| steepest_replace -> raw_add | 6.115 | 4.295 | 4.295 | 6.144 |
| steepest_replace -> steepest_add | 6.115 | 6.392 | 5.916 | 6.392 |

Curvature finite-difference means near the boundary were small:

| center | direction | mean second difference |
| --- | --- | ---: |
| steepest_add | radial | -0.0197 |
| steepest_add | toward_steepest_replace | -0.0784 |
| steepest_replace | radial | -0.0256 |
| steepest_replace | toward_steepest_add | 0.0108 |

Interpretation: for this Burgers validation batch, replacement is not best at the final endpoint compared with `steepest_add`, but it lands in the same high-loss boundary neighborhood quickly. The boundary arc between `steepest_replace` and `steepest_add` remains high-loss, with minimum `5.916`, so directional mistakes between these two are not catastrophic. This supports the "broad high-loss boundary ridge" part of the Burgers explanation.

## Current Mechanistic Takeaway

The evidence now points to:

\[
\text{stable/forgiving boundary landscape} + \text{fast boundary arrival}
\Rightarrow \text{replacement can look very strong}
\]

and

\[
\text{moving target directions} + \text{narrow/path-dependent high-loss region}
\Rightarrow \text{additive steepest updates are safer and can achieve larger final loss}.
\]

For Burgers, the new probe weakens the overly simple claim that replacement wins because the whole objective is accurately first-order or quadratic over the full radius. The more defensible claim is that replacement gets to a high-loss boundary region quickly, and the Burgers boundary region tolerates directional error. For NS2D, the small trace directly shows replacement jumps to the boundary immediately but then underperforms, while `steepest_add` keeps changing direction and reaches much higher final loss.
