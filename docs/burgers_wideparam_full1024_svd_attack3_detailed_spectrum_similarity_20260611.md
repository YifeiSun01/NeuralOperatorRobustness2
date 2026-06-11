# Burgers Full-1024 SVD Attack3 Detailed Spectrum/Similarity Results - 2026-06-11

Source root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611`. This reads the saved full `1024 x 1024` SVD NPZ files; no block projection or top-k approximation is used.

## Samples

| sample | dataset | index | family |
|---:|---|---:|---|
| 0 | `burgers_widevis_l3target_d31` | 10 | `powerlaw_fourier` |
| 1 | `burgers_widevis_l3target_d17` | 177 | `matern` |
| 2 | `burgers_widevis_l3target_d05` | 45 | `gaussian` |

## Model Mean Metrics Across 3 Samples

| model_key | err_sigma1 | model_spectral_norm | solver_spectral_norm | right_v1_cos | left_u1_cos | right_top10_mean | left_top10_mean | err_v1_delta_cos | init_mse | final_mse | growth_abs | growth_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2.01134 | 2.82197 | 3.24638 | 0.940287 | 0.840368 | 0.887814 | 0.798228 | 0.313126 | 0.000757474 | 0.0162244 | 0.0154669 | 18.8755 |
| loss1 | 2.08314 | 3.20474 | 3.24638 | 0.986033 | 0.939893 | 0.939044 | 0.847184 | 0.154677 | 0.000747912 | 0.00778604 | 0.00703813 | 10.3962 |
| loss2 | 2.09002 | 3.12797 | 3.24638 | 0.981972 | 0.971943 | 0.936032 | 0.847439 | 0.106322 | 0.000764422 | 0.00635045 | 0.00558603 | 11.6968 |
| loss3 | 1.06347 | 3.2908 | 3.24638 | 0.992065 | 0.992449 | 0.967363 | 0.946596 | 0.0406373 | 0.000165457 | 0.00408682 | 0.00392136 | 30.169 |

## Per-Sample Compact Metrics

| sample_id | dataset_id | family | model_key | err_sigma1 | err_sv5 | err_sv10 | err_sv20 | err_top20_energy | right_v1_cos | left_u1_cos | right_top10_mean | left_top10_mean | err_v1_delta_cos | init_mse | final_mse | growth_abs | growth_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | burgers_widevis_l3target_d31 | powerlaw_fourier | baseline | 2.36802 | 0.782298 | 0.398828 | 0.125598 | 0.974785 | 0.849395 | 0.882995 | 0.896214 | 0.800051 | 0.0316196 | 0.0011872 | 0.0345066 | 0.0333194 | 29.0655 |
| 0 | burgers_widevis_l3target_d31 | powerlaw_fourier | loss1 | 3.02591 | 0.653577 | 0.342697 | 0.115888 | 0.984882 | 0.988057 | 0.998649 | 0.950982 | 0.836363 | 0.237807 | 0.00119666 | 0.0120534 | 0.0108567 | 10.0725 |
| 0 | burgers_widevis_l3target_d31 | powerlaw_fourier | loss2 | 3.07656 | 0.768454 | 0.404698 | 0.11604 | 0.989408 | 0.975962 | 0.95258 | 0.952029 | 0.822849 | 0.0647416 | 0.0016665 | 0.00932695 | 0.00766046 | 5.59674 |
| 0 | burgers_widevis_l3target_d31 | powerlaw_fourier | loss3 | 1.06904 | 0.3801 | 0.223275 | 0.0797783 | 0.985802 | 0.98777 | 0.988272 | 0.979924 | 0.967344 | 0.00441821 | 0.000204258 | 0.00653228 | 0.00632803 | 31.9806 |
| 1 | burgers_widevis_l3target_d17 | matern | baseline | 1.71212 | 0.71337 | 0.413375 | 0.0984854 | 0.974408 | 0.980358 | 0.87379 | 0.877827 | 0.787673 | 0.126244 | 0.000612469 | 0.00498496 | 0.00437249 | 8.13913 |
| 1 | burgers_widevis_l3target_d17 | matern | loss1 | 1.99931 | 0.900503 | 0.408838 | 0.106598 | 0.988224 | 0.97861 | 0.908526 | 0.88801 | 0.792766 | 0.125629 | 0.000745306 | 0.00828846 | 0.00754316 | 11.1209 |
| 1 | burgers_widevis_l3target_d17 | matern | loss2 | 1.97707 | 0.652855 | 0.386158 | 0.103126 | 0.988265 | 0.97792 | 0.984548 | 0.890983 | 0.80705 | 0.171126 | 0.000363448 | 0.00710781 | 0.00674436 | 19.5566 |
| 1 | burgers_widevis_l3target_d17 | matern | loss3 | 1.70113 | 0.460975 | 0.183271 | 0.0714887 | 0.992461 | 0.989935 | 0.995194 | 0.925355 | 0.882787 | 0.0919157 | 0.000256348 | 0.00422428 | 0.00396794 | 16.4787 |
| 2 | burgers_widevis_l3target_d05 | gaussian | baseline | 1.95388 | 0.624787 | 0.357279 | 0.12391 | 0.976278 | 0.991107 | 0.76432 | 0.889401 | 0.806958 | 0.781514 | 0.000472753 | 0.00918172 | 0.00870896 | 19.4218 |
| 2 | burgers_widevis_l3target_d05 | gaussian | loss1 | 1.22419 | 0.459407 | 0.219476 | 0.107041 | 0.975941 | 0.991431 | 0.912503 | 0.978139 | 0.912423 | 0.100595 | 0.000301773 | 0.00301631 | 0.00271454 | 9.9953 |
| 2 | burgers_widevis_l3target_d05 | gaussian | loss2 | 1.21643 | 0.47035 | 0.226864 | 0.110789 | 0.975596 | 0.992033 | 0.978701 | 0.965083 | 0.912417 | 0.0830966 | 0.000263318 | 0.00261659 | 0.00235327 | 9.937 |
| 2 | burgers_widevis_l3target_d05 | gaussian | loss3 | 0.420253 | 0.286318 | 0.122206 | 0.0524481 | 0.975449 | 0.998491 | 0.99388 | 0.996809 | 0.989657 | 0.0255779 | 3.57665e-05 | 0.0015039 | 0.00146813 | 42.0476 |

## Loss3 Reductions Versus Other Models

| metric | reference | loss3_value | reference_value | loss3_minus_reference | loss3_reduction_fraction |
|---|---|---|---|---|---|
| error_spectral_norm_sv1 | baseline | 1.06347 | 2.01134 | -0.947866 | 0.471261 |
| error_spectral_norm_sv1 | loss1 | 1.06347 | 2.08314 | -1.01966 | 0.489484 |
| error_spectral_norm_sv1 | loss2 | 1.06347 | 2.09002 | -1.02654 | 0.491165 |
| attack_final_mse | baseline | 0.00408682 | 0.0162244 | -0.0121376 | 0.748107 |
| attack_final_mse | loss1 | 0.00408682 | 0.00778604 | -0.00369922 | 0.475109 |
| attack_final_mse | loss2 | 0.00408682 | 0.00635045 | -0.00226363 | 0.356452 |
| attack_growth_abs | baseline | 0.00392136 | 0.0154669 | -0.0115456 | 0.746468 |
| attack_growth_abs | loss1 | 0.00392136 | 0.00703813 | -0.00311677 | 0.44284 |
| attack_growth_abs | loss2 | 0.00392136 | 0.00558603 | -0.00166467 | 0.298005 |
| model_spectral_norm | baseline | 3.2908 | 2.82197 | 0.468823 | -0.166133 |
| model_spectral_norm | loss1 | 3.2908 | 3.20474 | 0.0860587 | -0.0268536 |
| model_spectral_norm | loss2 | 3.2908 | 3.12797 | 0.162825 | -0.0520545 |
| right_top10_mean_cos | baseline | 0.967363 | 0.887814 | 0.0795486 | -0.0896005 |
| right_top10_mean_cos | loss1 | 0.967363 | 0.939044 | 0.0283189 | -0.0301572 |
| right_top10_mean_cos | loss2 | 0.967363 | 0.936032 | 0.0313308 | -0.033472 |
| left_top10_mean_cos | baseline | 0.946596 | 0.798228 | 0.148368 | -0.185872 |
| left_top10_mean_cos | loss1 | 0.946596 | 0.847184 | 0.0994118 | -0.117344 |
| left_top10_mean_cos | loss2 | 0.946596 | 0.847439 | 0.0991571 | -0.117008 |

## Output Tables

- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/detailed_spectrum_similarity_20260611/compact_per_sample_model_metrics.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/detailed_spectrum_similarity_20260611/model_mean_metrics.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/detailed_spectrum_similarity_20260611/singular_values_energy_topk.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/detailed_spectrum_similarity_20260611/model_solver_subspace_similarity_topk.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/detailed_spectrum_similarity_20260611/loss3_reductions_vs_others.csv`
