# Loss3 boundary cap volume: same-sample Lmax normalization

Date: 2026-06-23

This note records the stricter follow-up to the boundary-volume experiment. The
first endpoint-cap table normalized each cap point by its own endpoint loss:

\[
\frac{L(\delta_{\mathrm{cap}})}{L(\delta_{\mathrm{endpoint}})}.
\]

That is useful for local flatness, but it can make a bad endpoint look good if it
is locally flat around a low loss. The stricter test normalizes by the best
observed optimizer endpoint for the same sample:

\[
\frac{L(\delta_{\mathrm{cap}})}{L_{\max}},
\qquad
L_{\max} =
\max_m L(\delta_m).
\]

The high-loss cap probability is then

\[
p_\tau(\theta)
=
\mathbb{P}\!\left[
L(\delta_{\mathrm{cap}}(\theta)) \ge \tau L_{\max}
\right].
\]

## Files

- Script: `tools/analyze_loss3_boundary_cap_lmax_20260623.py`
- SVG plot: `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/figures/endpoint_cap_lmax_p095_by_theta.svg`
- Rescored table: `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/tables/endpoint_cap_volume_lmax.csv`
- Summary table: `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/tables/endpoint_cap_volume_lmax_summary.csv`
- Width table: `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/tables/endpoint_cap_width_lmax_summary.csv`

## Main result at tau = 0.95

For Burgers, the optimizer-relevant cap around `steepest_replace` still contains
near-maximal points:

| system | endpoint | theta | p(loss >= 0.95 Lmax) | mean loss/Lmax |
| --- | --- | ---: | ---: | ---: |
| Burgers | steepest_replace | 0.00 | 0.600 | 0.907 |
| Burgers | steepest_replace | 0.05 | 0.600 | 0.906 |
| Burgers | steepest_replace | 0.10 | 0.400 | 0.902 |
| Burgers | steepest_replace | 0.20 | 0.350 | 0.884 |
| Burgers | steepest_replace | 0.40 | 0.000 | 0.814 |

For NS2D, the cap around `steepest_replace` is not close to the best observed
loss at all:

| system | endpoint | theta | p(loss >= 0.95 Lmax) | mean loss/Lmax |
| --- | --- | ---: | ---: | ---: |
| NS2D | steepest_replace | 0.00 | 0.000 | 0.274 |
| NS2D | steepest_replace | 0.05 | 0.000 | 0.274 |
| NS2D | steepest_replace | 0.10 | 0.000 | 0.272 |
| NS2D | steepest_replace | 0.20 | 0.000 | 0.267 |
| NS2D | steepest_replace | 0.40 | 0.000 | 0.250 |

By contrast, NS2D `steepest_add` is near the best observed loss:

| system | endpoint | theta | p(loss >= 0.95 Lmax) | mean loss/Lmax |
| --- | --- | ---: | ---: | ---: |
| NS2D | steepest_add | 0.00 | 1.000 | 1.000 |
| NS2D | steepest_add | 0.05 | 1.000 | 0.999 |
| NS2D | steepest_add | 0.10 | 1.000 | 0.996 |
| NS2D | steepest_add | 0.20 | 1.000 | 0.982 |
| NS2D | steepest_add | 0.40 | 0.000 | 0.923 |

## Interpretation

This stricter normalization supports the current mechanism story:

1. Burgers `steepest_replace` is not merely locally flat around a weak endpoint.
   It actually lands near the sample-level best observed loss for a nontrivial
   cap around the endpoint.
2. NS2D `steepest_replace` is locally measurable, but it is not a high-loss
   region relative to the best loss that `steepest_add` finds. Its mean
   loss/Lmax is only about 0.27 near theta 0.
3. The more precise statement is therefore not that the whole Burgers boundary
   has high loss. Global random boundary sampling found high-loss points rare
   for both Burgers and NS2D. The supported statement is that the
   optimizer-relevant high-loss region around the Burgers replace endpoint is
   much more forgiving than the NS2D replace endpoint region.

This is more direct evidence for the high-loss-region explanation than the older
endpoint-relative cap result.
