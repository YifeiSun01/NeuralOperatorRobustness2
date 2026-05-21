# Loss3 Alpha/Epsilon Core-Four Final Delta Similarity - 2026-05-20

Generated: 2026-05-20T22:06:30.040844+00:00

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

- Pairwise summary by setting: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_summary.csv`
- Per-sample pairwise table: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_per_sample.csv`
- Across-setting rollup: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_rollup.csv`
- Mean cosine by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/figures/pairwise_mean_cosine_by_setting.png`
- Mean centered-cosine by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/figures/pairwise_mean_centered_cosine_by_setting.png`
- Mean spectral-cosine by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/figures/pairwise_mean_spectral_cosine_by_setting.png`
- Mean relative-L2 by setting heatmap: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/figures/pairwise_mean_relative_l2_by_setting.png`
- Across-setting mean cosine matrix: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/figures/rollup_mean_cosine_matrix.png`
- Across-setting mean relative-L2 matrix: `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/figures/rollup_mean_relative_l2_matrix.png`

## Rollup Highlights

Highest mean cosine pairs across settings:

| pair | mean cosine | mean centered cosine | mean spectral cosine | mean relative L2 |
|---|---:|---:|---:|---:|
| raw add vs steepest add | 0.5118 | 0.4173 | 0.8221 | 0.9345 |
| raw replace vs steepest add | 0.3925 | 0.2749 | 0.8106 | 1.2583 |
| raw add vs raw replace | 0.3625 | 0.2748 | 0.7661 | 1.2649 |
| raw replace vs steepest replace | 0.3085 | 0.2432 | 0.7796 | 1.3953 |
| steepest add vs steepest replace | 0.2124 | 0.1231 | 0.7454 | 1.2327 |
| raw add vs steepest replace | 0.1723 | 0.0992 | 0.7091 | 1.2715 |

Smallest relative L2 pairs across settings:

| pair | mean relative L2 | mean cosine | mean sign agreement |
|---|---:|---:|---:|
| raw add vs steepest add | 0.9345 | 0.5118 | 0.6894 |
| steepest add vs steepest replace | 1.2327 | 0.2124 | 0.5653 |
| raw replace vs steepest add | 1.2583 | 0.3925 | 0.6163 |
| raw add vs raw replace | 1.2649 | 0.3625 | 0.6307 |
| raw add vs steepest replace | 1.2715 | 0.1723 | 0.5642 |
| raw replace vs steepest replace | 1.3953 | 0.3085 | 0.5963 |

## Interpretation

Observed evidence should be read with the existing smoothness and high-frequency heatmaps. High delta similarity means the methods find broadly similar final perturbation shapes; it does not by itself decide which optimizer reaches that shape faster or with better post-boundary loss growth.

Inference: if `steepest_replace` / GPI has high similarity to the other final deltas while also showing faster boundary arrival, stronger post-boundary loss gain, and low high-frequency/peakiness metrics, then it is a stronger practical optimizer for this fixed-budget `p=2,q=2` loss3 setting.

## Inputs

- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps1_alpha0p1_batch100_steps100_pinf_q1` (e1/a0.1, p=inf, q=1)
- `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps1_alpha0p2_batch100_steps100_pinf_q1` (e1/a0.2, p=inf, q=1)
