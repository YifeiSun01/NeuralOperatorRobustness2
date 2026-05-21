# Loss3 Alpha/Epsilon Core-Four Final Delta Similarity - 2026-05-20

Generated: 2026-05-20T22:06:17.294335+00:00

## Scope

Observed from completed `p=2,q=2` alpha/epsilon sweep artifacts only. No optimizer experiment was rerun for this analysis.

Similarity is computed per sample after aligning methods by `dataset_index`, then summarized over the batch and across settings.

Metrics:

- `cosine`: signed direction similarity of final delta.
- `centered_cosine`: cosine after subtracting each delta's spatial mean; closer to shape similarity.
- `spectral_magnitude_cosine`: cosine between Fourier magnitude spectra; closer to frequency-content similarity.
- `relative_l2_distance`: actual final perturbation distance normalized by average norm.
- `sign_agreement`: fraction of grid points with matching sign.

## Key Outputs

- Pairwise summary by setting: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_summary.csv`
- Per-sample pairwise table: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_per_sample.csv`
- Across-setting rollup: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_rollup.csv`
- Mean cosine by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/figures/pairwise_mean_cosine_by_setting.png`
- Mean centered-cosine by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/figures/pairwise_mean_centered_cosine_by_setting.png`
- Mean spectral-cosine by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/figures/pairwise_mean_spectral_cosine_by_setting.png`
- Mean relative-L2 by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/figures/pairwise_mean_relative_l2_by_setting.png`
- Across-setting mean cosine matrix: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/figures/rollup_mean_cosine_matrix.png`
- Across-setting mean relative-L2 matrix: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/figures/rollup_mean_relative_l2_matrix.png`

## Rollup Highlights

Highest mean cosine pairs across settings:

| pair | mean cosine | mean centered cosine | mean spectral cosine | mean relative L2 |
|---|---:|---:|---:|---:|
| raw replace vs steepest replace | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| raw add vs steepest add | 0.8153 | 0.8102 | 0.9284 | 0.4276 |
| raw add vs raw replace | 0.7255 | 0.7221 | 0.9247 | 0.5523 |
| raw add vs steepest replace | 0.7255 | 0.7221 | 0.9247 | 0.5523 |
| raw replace vs steepest add | 0.6528 | 0.6466 | 0.8750 | 0.6712 |
| steepest add vs steepest replace | 0.6528 | 0.6466 | 0.8750 | 0.6712 |

Smallest relative L2 pairs across settings:

| pair | mean relative L2 | mean cosine | mean sign agreement |
|---|---:|---:|---:|
| raw replace vs steepest replace | 0.0000 | 1.0000 | 1.0000 |
| raw add vs steepest add | 0.4276 | 0.8153 | 0.8346 |
| raw add vs raw replace | 0.5523 | 0.7255 | 0.7678 |
| raw add vs steepest replace | 0.5523 | 0.7255 | 0.7678 |
| raw replace vs steepest add | 0.6712 | 0.6528 | 0.7291 |
| steepest add vs steepest replace | 0.6712 | 0.6528 | 0.7291 |

## Interpretation

Observed evidence should be read with the existing smoothness and high-frequency heatmaps. High delta similarity means the methods find broadly similar final perturbation shapes; it does not by itself decide which optimizer reaches that shape faster or with better post-boundary loss growth.

Inference: if `steepest_replace` / GPI has high similarity to the other final deltas while also showing faster boundary arrival, stronger post-boundary loss gain, and low high-frequency/peakiness metrics, then it is a stronger practical optimizer for this fixed-budget `p=2,q=2` loss3 setting.

## Inputs

- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps1_alpha0p1_batch100_steps100_p2_q1` (e1/a0.1, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps1_alpha0p2_batch100_steps100_p2_q1` (e1/a0.2, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps2_alpha0p2_batch100_steps100_p2_q1` (e2/a0.2, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps2_alpha0p4_batch100_steps100_p2_q1` (e2/a0.4, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps2_alpha0p8_batch100_steps100_p2_q1` (e2/a0.8, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha0p2_batch100_steps100_p2_q1` (e4/a0.2, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q1` (e4/a0.4, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha0p8_batch100_steps100_p2_q1` (e4/a0.8, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha1p2_batch100_steps100_p2_q1` (e4/a1.2, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha1p6_batch100_steps100_p2_q1` (e4/a1.6, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha0p2_batch100_steps100_p2_q1` (e8/a0.2, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q1` (e8/a0.3, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha0p4_batch100_steps100_p2_q1` (e8/a0.4, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha0p8_batch100_steps100_p2_q1` (e8/a0.8, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha1p6_batch100_steps100_p2_q1` (e8/a1.6, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha2p4_batch100_steps100_p2_q1` (e8/a2.4, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps12_alpha0p6_batch100_steps100_p2_q1` (e12/a0.6, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps12_alpha1p2_batch100_steps100_p2_q1` (e12/a1.2, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps16_alpha1p6_batch100_steps100_p2_q1` (e16/a1.6, p=2, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps16_alpha3p2_batch100_steps100_p2_q1` (e16/a3.2, p=2, q=1)
