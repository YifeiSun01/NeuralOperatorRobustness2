# Darcy Full vs Block/2 SVD Vector Comparison (20260612_loss3_3samples_top20)

- Created: 2026-06-12T07:17:27+00:00
- Model: `loss3`
- Compared full `85x85` Jacobian SVD against orthonormal block/2 projection on the `84x84` crop.
- Top ranks compared: `20`.

## Timing and Top Singular Value

| sample | full total sec | block/2 total sec | full sigma1 | block/2 sigma1 | rel error |
|---|---:|---:|---:|---:|---:|
| smooth_idx0 | 76.45 | 11.60 | 0.00231427 | 0.00227061 | -1.89% |
| highpass_idx0 | 75.70 | 11.34 | 0.00194708 | 0.0018593 | -4.51% |
| wave_idx0 | 74.31 | 11.16 | 0.00231545 | 0.00225953 | -2.42% |

## Mean Top-k Singular Value Relative Error

| k | mean abs rel error |
|---:|---:|
| 5 | 3.82% |
| 10 | 4.28% |
| 20 | 4.92% |

## Same-rank Vector Alignment After Projecting Full Vectors to Block/2

| k | mean right abs dot | mean left abs dot | mean right projection energy | mean left projection energy |
|---:|---:|---:|---:|---:|
| 5 | 0.999 | 0.999 | 0.930 | 0.995 |
| 10 | 0.991 | 0.990 | 0.924 | 0.992 |
| 20 | 0.972 | 0.971 | 0.917 | 0.985 |

## Subspace Alignment

| k | mean right principal cosine | mean left principal cosine |
|---:|---:|---:|
| 5 | 1.000 | 0.999 |
| 10 | 0.999 | 0.998 |
| 20 | 0.997 | 0.996 |

## Files

- `analysis_outputs/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20/sample_timing_summary.csv`
- `analysis_outputs/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20/singular_value_comparison_topk.csv`
- `analysis_outputs/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20/vector_alignment_topk.csv`
- `analysis_outputs/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20/subspace_alignment_topk.csv`
- `visualizations/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20`
