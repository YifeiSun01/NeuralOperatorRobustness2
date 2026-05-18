# Loss3 Delta Smoothness Metrics Interpretation - 2026-05-18

This note records the interpretation of the delta smoothness diagnostics for the
`p=2,q=2` core-four comparison.

Source run:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`

New figure-only output folder:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2`

Generated/superseded PNG status:

- `01_final_smoothness_core4_mean_std_annotated.png` remains in the annotated
  folder as a record of the final-step mean/std bar diagnostics.
- `02_sample040_final_delta_with_smoothness_numbers.png` was deleted on
  2026-05-18 01:41 UTC because the user judged it redundant.
- `03_sample040_smoothness_over_steps.png` was deleted on 2026-05-18 01:41 UTC
  because the user judged it redundant.
- The useful mean/std information was moved into the simplified core figure
  `forensics/loss3_core_simplified_figures_20260518_p2_q2/02_delta_quality_core4_metrics.png`,
  which now includes sample-standard-deviation shading and final-step
  mean/std bars.

## Metric Meaning

Observed from `tools/run_loss3_direction_proposal_ablation.py`:

- `boundary_ratio = ||delta||_p / epsilon`. This is not a smoothness metric. It
  only says whether the perturbation uses the allowed p-norm radius. A value near
  `1` means the perturbation is on the constraint boundary.
- `high_frequency_energy_ratio` is the fraction of Fourier energy in the highest
  frequency band of `delta`. Larger values mean more high-frequency perturbation
  energy.
- `spectral_centroid` is the average frequency location of the perturbation
  spectrum. Larger values mean energy shifts toward higher frequencies.
- `first_derivative_l2 = ||diff(delta)||_2`. Larger values mean sharper adjacent
  changes, so the perturbation is less smooth.
- `total_variation = sum(abs(diff(delta)))`. Larger values mean the curve is more
  jagged overall.
- `second_derivative_l2 = ||diff(delta, n=2)||_2`. Larger values mean stronger
  curvature or oscillation.

In short: for `high_frequency_energy_ratio`, `spectral_centroid`,
`first_derivative_l2`, `total_variation`, and `second_derivative_l2`, smaller is
smoother / less high-frequency. `boundary_ratio` is separate and should not be
read as smoothness.

## Observed p=2,q=2 Final-Step Evidence

Observed from per-method `per_step_metrics.csv` at step `k=100`:

| Method | Actual loss mean | High-frequency ratio mean | First-derivative L2 mean | Total variation mean | Boundary ratio mean |
|---|---:|---:|---:|---:|---:|
| `raw_add` | 5.389592 | 5.900997e-08 | 0.645722 | 8.573150 | 0.968878 |
| `raw_replace` | 6.804780 | 8.098748e-10 | 0.217019 | 2.685119 | 1.000000 |
| `steepest_add` | 6.377158 | 7.997723e-06 | 0.583815 | 6.464769 | 1.000000 |
| `steepest_replace` | 6.804780 | 8.098748e-10 | 0.217019 | 2.685119 | 1.000000 |

Observed from `per_sample_step_metrics.csv` for dataset index `40` at final step:

| Method | High-frequency ratio | First-derivative L2 | Total variation |
|---|---:|---:|---:|
| `raw_add` | 6.957584e-09 | 0.949167 | 14.660805 |
| `raw_replace` | 6.571385e-10 | 0.250301 | 2.685880 |
| `steepest_add` | 1.301442e-09 | 0.299778 | 4.275679 |
| `steepest_replace` | 6.571385e-10 | 0.250301 | 2.685880 |

## Interpretation

Observed evidence supports the user's visual observation for sample `40`: the
`raw_add` final perturbation has much larger first-derivative L2 and total
variation than the replacement/GPI perturbation, which quantitatively matches a
more jagged or less smooth delta.

For the 100-sample final-step average, `raw_replace` and `steepest_replace` have
both the highest actual loss and the lowest recorded first-derivative L2 / total
variation among the four core methods. This means that in the observed
`p=2,q=2` run, the GPI-style replacement did not pay for faster loss growth by
creating a rougher final perturbation; it actually produced a smoother final
perturbation by these diagnostics.

Caveat: smoothness metrics quantify shape, but they do not by themselves prove
physical validity. The plotted final delta curves and any domain-specific
constraints should still be inspected alongside the metrics.
