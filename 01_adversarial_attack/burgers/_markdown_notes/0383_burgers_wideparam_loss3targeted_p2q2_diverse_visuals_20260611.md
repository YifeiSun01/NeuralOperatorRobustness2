# Burgers Wide-Parameter Loss3-Targeted P2Q2 Diverse Visuals - 2026-06-11

## Status

Complete. The new loss3-targeted wide-parameter Burgers dataset was first
checked for parameter and curve-shape diversity, then used to run a batched P2Q2
attack and render five comparison-dense multi-sample groups.

## Inputs

- Dataset root:
  `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers`
- Dataset audit:
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611/wide_parameter_dataset_audit.csv`
- Runner:
  `tools/run_burgers_wideparam_loss3targeted_round00_p2q2_diverse_multi_visuals_batched_20260611.py`
- Base comparison plotter:
  `tools/plot_burgers_round03_p2q2_combined_attack_panels.py`

## Diversity Check

Observed from the audit table and selected-group manifest:

- The full dataset has 50 unique parameter sets.
- Family counts are: `gaussian` 12, `matern` 10, `powerlaw_fourier` 14,
  `sine_mixture` 12, `sawtooth` 1, `square_wave` 1, `spike_train` 0.
- Actual value range remains controlled: global `x` range is approximately
  `[-0.65, 1.50]`.
- Spectral centroids span low-to-high regimes:
  about 6 for smooth Matern, 8-10 for Gaussian, 10-15 for power-law Fourier,
  20 for square wave, and 23-32 for high-frequency sine mixtures.
- A preview figure was written at:
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611/diverse_group_initial_condition_preview.png`.
  The PNG is valid and has size 3780x2160.

The five P2Q2 groups use 25 distinct generalization datasets with no reuse.
Each group contains one test sample plus five generalization samples selected to
mix smooth kernels, Gaussian fields, power-law spectra, sine mixtures, and
sharp/high-frequency examples.

## P2Q2 Run

GPU path was verified before the run:

- GPU: Tesla V100-SXM2-32GB
- PyTorch: `2.8.0+cu126`, CUDA available, device capability `(7, 0)`
- JAX backend: `gpu`

Attack settings:

- `p=2`, `q=2`
- 100 attack steps
- record every 2 steps
- RMS epsilon `0.12`
- alpha RMS `0.012`
- 5 groups were batched together as 30 samples for each model attack.

Trace root:
`forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611`

Visualization root:
`visualizations/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_comparison_dense_diverse_multi_sample_batched_bundle_20260611/comparison_dense`

## Outputs

The run rendered 30 valid PNG files:

- 5 groups
- 6 PNG files per group
- Four-column images are 7200x2528.
- Two-column baseline-vs-loss images are 3600x2528.

Each group contains:

- `baseline_loss1_loss2_loss3_step000_before_perturbation_four_column.png`
- `baseline_loss1_loss2_loss3_step100_after_perturbation_four_column.png`
- `baseline_loss1_loss2_loss3_before_after_overlay_four_column.png`
- `baseline_vs_loss1_before_after_overlay_two_column.png`
- `baseline_vs_loss2_before_after_overlay_two_column.png`
- `baseline_vs_loss3_before_after_overlay_two_column.png`

## Final Attack Loss Means

Observed from each group summary:

| group | baseline | loss1 | loss2 | loss3 |
| --- | ---: | ---: | ---: | ---: |
| group00 | 1.2439e-02 | 6.0062e-03 | 6.0574e-03 | 3.2049e-03 |
| group01 | 1.6045e-02 | 6.9050e-03 | 7.7222e-03 | 4.0798e-03 |
| group02 | 1.2800e-02 | 6.5433e-03 | 5.9747e-03 | 3.9501e-03 |
| group03 | 1.9085e-02 | 7.8953e-03 | 6.4595e-03 | 5.1924e-03 |
| group04 | 1.5852e-02 | 7.3700e-03 | 6.8700e-03 | 4.1244e-03 |

Observed conclusion:

- Loss3 has the lowest final attacked MSE mean in all 5 displayed groups.
- The selected visual groups contain visibly different family and spectral
  regimes rather than adjacent copies from the same dataset family.

## Remaining Work

- Inspect the generated PNGs visually and decide whether another hand-picked
  group set should be rendered.
- Sync outputs to durable storage before recycling/destroying the Vast instance,
  because `/workspace` is not persistent.


## Wrapped Label Rerender

The comparison-dense panels were rerendered with multi-line row labels so long
semantic dataset names no longer intrude into the delta / initial-condition / output
panels.

Updated code:

- `tools/plot_burgers_round03_p2q2_combined_attack_panels.py`
- `tools/run_burgers_wideparam_loss3targeted_round00_p2q2_diverse_multi_visuals_batched_20260611.py`

New visualization root, leaving the previous image folder intact:

`visualizations/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_comparison_dense_diverse_multi_sample_batched_wrapped_labels_20260611/comparison_dense`

Label format now uses separate lines such as:

- `S2 generalization`
- `Matern`
- `C=0.08`
- `Nu=2.5`
- `Range=[-0.5,1.5]`
- `Centroid=5.97`
- `TV=0.0165`
- `idx=7`

The rerender produced 30 PNGs again: 5 groups x 6 PNGs. Four-column images remain
7200x2528, and two-column images remain 3600x2528.

Programmatic text-bounding-box check:

- four-column panels: maximum label right edge `x=0.0306`, first delta panel starts at `x=0.045`, margin `0.0144` figure-width units.
- two-column panels: maximum label right edge `x=0.0522`, first delta panel starts at `x=0.082`, margin `0.0298` figure-width units.

Inference: the rewritten multi-line labels fit in the left label gutter and should
not cover the plotted panels.
