# Darcy Sigma1 From Block Vector Estimate (20260612_loss3_6samples_blockvec_jvp)

- Created: 2026-06-12T07:39:24+00:00
- Model: `loss3`
- Samples: 6 generalization samples; first 3 reused prior full/block SVD outputs, last 3 computed new full truth.
- Method: compute block/2 top right singular vector `v_b`, lift it to full `85x85` input space, estimate `||J_full L v_b||`; also test one power-iteration refinement.

## Mean Absolute Relative Error vs Full Sigma1

| estimate | mean abs rel error |
|---|---:|
| block/2 sigma1 | 2.89% |
| lifted `||Jv||` | 2.83% |
| one power step | 0.03% |

## Per-sample Results

| sample | full sigma1 | block/2 sigma1 | block err | lifted Jv | lifted err | one-power | one-power err | reused full/block? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smooth_idx0 | 0.00231427 | 0.00227061 | -1.89% | 0.00227231 | -1.81% | 0.00231346 | -0.04% | True |
| highpass_idx0 | 0.00194708 | 0.0018593 | -4.51% | 0.00186088 | -4.43% | 0.00194628 | -0.04% | True |
| wave_idx0 | 0.00231545 | 0.00225953 | -2.42% | 0.00226099 | -2.35% | 0.00231462 | -0.04% | True |
| bandpass_idx0 | 0.00229337 | 0.00223824 | -2.40% | 0.00223937 | -2.35% | 0.00229295 | -0.02% | False |
| blocky_idx0 | 0.00195464 | 0.00191609 | -1.97% | 0.0019171 | -1.92% | 0.00195377 | -0.04% | False |
| rectangles_idx0 | 0.00196784 | 0.0018859 | -4.16% | 0.00188737 | -4.09% | 0.00196777 | -0.00% | False |

## Files

- `analysis_outputs/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp/sigma1_estimate_summary.csv`
- `analysis_outputs/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp/selected_samples.csv`
- `visualizations/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp/sigma1_full_vs_block_lifted_estimates.png`
- `visualizations/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp/relative_error_vs_full_sigma1.png`
- `visualizations/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp/timing_full_reference_vs_lifted_estimate.png`
