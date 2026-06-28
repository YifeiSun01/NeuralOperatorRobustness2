# Darcy Attack50 Per-Dataset First-vs-Second T-Tests

Each row is one paired t-test inside one dataset using 50 matched attack samples. The test is `second - best > 0`, so a small p-value means the first-ranked model is significantly lower than the second-ranked model for that dataset.

Important limitation: the clean 52-dataset RMSE/Relative L2 release table is aggregate-only, so per-dataset t-tests for full clean RMSE/Relative L2 require rerunning clean evaluation with per-sample errors saved. This report uses the sample-level final attack50 table, including `clean_loss` and `sqrt(clean_loss)` for the same 50 samples per dataset.

## Summary

| scope | metric | metric_label | tests | best_model_counts | loss3_best_count | p05_significant_count | bh_significant_count | median_p | max_p | non_sig_dataset_short |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization50 | clean_loss | clean MSE loss | 50 | loss3:44; loss2:4; physics_loss:2 | 44 | 49 | 49 | 2.02835e-15 | 0.442782 | 02_highpass_grf_frac0p2_a1p39914_t8p09047 |
| generalization50 | clean_rmse_from_loss | clean RMSE from per-sample MSE | 50 | loss3:44; loss2:4; physics_loss:2 | 44 | 47 | 47 | 1.51533e-18 | 0.303078 | 33_matern_fine_frac0p24_a1p77702_t7p67384; 02_highpass_grf_frac0p2_a1p39914_t8p09047; 06_rectangles_frac0p2_a1p44134_t6p32822 |
| generalization50 | adv_loss | adversarial MSE loss | 50 | loss3:50 | 50 | 50 | 50 | 3.49087e-27 | 0.0417448 |  |
| generalization50 | loss_increase | absolute loss increase | 50 | loss3:50 | 50 | 50 | 50 | 9.65899e-23 | 0.0026337 |  |
| generalization50 | relative_increase | relative loss increase | 50 | random_solver:21; loss3:16; random_clean:6; loss1:6; loss2:1 | 16 | 29 | 28 | 0.0019534 | 0.497234 | 00_matern_smooth_frac0p12_a3p65909_t2p05625; 03_bandpass_grf_frac0p16_a2p16503_t3p04694; 16_matern_smooth_frac0p2_a3p63184_t2p95736; 17_matern_fine_frac0p16_a1p05208_t9p11004; 24_matern_smooth_frac0p24_a4p16883_t1p66293; 32_matern_smooth_frac0p12_a4p06817_t2p8029; 35_bandpass_grf_frac0p16_a1p27246_t5p44182; 40_matern_smooth_frac0p16_a4p67408_t1p27431; 43_bandpass_grf_frac0p2_a1p22554_t2p62907; 49_matern_fine_frac0p16_a1p39654_t9p9182; 07_cellular_blobs_frac0p16_a1p97385_t4p60937; 13_blocky_tiles_frac0p12_a2p00536_t4p79207; 14_rectangles_frac0p24_a1p67693_t9p58326; 15_cellular_blobs_frac0p2_a3p27206_t2p59767; 22_rectangles_frac0p12_a1p43234_t10p2242; 23_cellular_blobs_frac0p24_a2p77492_t4p18018; 26_highpass_grf_frac0p16_a1p59204_t12p8363; 31_cellular_blobs_frac0p12_a2p19293_t5p77748; 39_cellular_blobs_frac0p16_a3p47176_t5p78996; 45_blocky_tiles_frac0p12_a2p72728_t2p09368 |
| all52 | clean_loss | clean MSE loss | 52 | loss3:44; loss2:4; random_solver:2; physics_loss:2 | 44 | 50 | 50 | 4.02431e-15 | 0.442782 | test; 02_highpass_grf_frac0p2_a1p39914_t8p09047 |
| all52 | clean_rmse_from_loss | clean RMSE from per-sample MSE | 52 | loss3:44; loss2:4; random_solver:2; physics_loss:2 | 44 | 48 | 48 | 3.1519e-18 | 0.303078 | test; 33_matern_fine_frac0p24_a1p77702_t7p67384; 02_highpass_grf_frac0p2_a1p39914_t8p09047; 06_rectangles_frac0p2_a1p44134_t6p32822 |
| all52 | adv_loss | adversarial MSE loss | 52 | loss3:50; random_clean:2 | 50 | 52 | 52 | 7.65008e-27 | 0.0438581 |  |
| all52 | loss_increase | absolute loss increase | 52 | loss3:50; random_clean:2 | 50 | 52 | 52 | 3.08678e-22 | 0.0422821 |  |
| all52 | relative_increase | relative loss increase | 52 | random_solver:21; loss3:16; random_clean:8; loss1:6; loss2:1 | 16 | 29 | 28 | 0.00863458 | 0.497234 | train; test; 00_matern_smooth_frac0p12_a3p65909_t2p05625; 03_bandpass_grf_frac0p16_a2p16503_t3p04694; 16_matern_smooth_frac0p2_a3p63184_t2p95736; 17_matern_fine_frac0p16_a1p05208_t9p11004; 24_matern_smooth_frac0p24_a4p16883_t1p66293; 32_matern_smooth_frac0p12_a4p06817_t2p8029; 35_bandpass_grf_frac0p16_a1p27246_t5p44182; 40_matern_smooth_frac0p16_a4p67408_t1p27431; 43_bandpass_grf_frac0p2_a1p22554_t2p62907; 49_matern_fine_frac0p16_a1p39654_t9p9182; 07_cellular_blobs_frac0p16_a1p97385_t4p60937; 13_blocky_tiles_frac0p12_a2p00536_t4p79207; 14_rectangles_frac0p24_a1p67693_t9p58326; 15_cellular_blobs_frac0p2_a3p27206_t2p59767; 22_rectangles_frac0p12_a1p43234_t10p2242; 23_cellular_blobs_frac0p24_a2p77492_t4p18018; 26_highpass_grf_frac0p16_a1p59204_t12p8363; 31_cellular_blobs_frac0p12_a2p19293_t5p77748 |

## Generalization50 Rows for Main Metrics

| dataset_short | metric_label | best_model_display | second_model_display | best_mean | second_mean | mean_diff_second_minus_best | t_stat | p_one_sided_best_lower | p_bh_by_metric_generalization50 | significant_p05 | significant_bh_gen50_p05 | loss3_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | clean MSE loss | loss3 | Physics Loss | 4.51197e-07 | 1.30028e-06 | 8.49078e-07 | 39.1843 | 5.9321e-39 | 1.48303e-37 | True | True | 1 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | clean MSE loss | loss2 | random clean | 1.02252e-07 | 1.11107e-07 | 8.85438e-09 | 2.87919 | 0.00294762 | 0.00342746 | True | True | 5 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | clean MSE loss | loss3 | loss2 | 2.3457e-07 | 6.28484e-07 | 3.93914e-07 | 10.5519 | 1.64131e-14 | 2.73552e-14 | True | True | 1 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | clean MSE loss | loss3 | Physics Loss | 2.68903e-07 | 7.12645e-07 | 4.43742e-07 | 18.9907 | 1.53434e-24 | 5.11447e-24 | True | True | 1 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | clean MSE loss | loss3 | loss2 | 2.982e-07 | 7.01141e-07 | 4.02942e-07 | 22.8311 | 4.55341e-28 | 2.52967e-27 | True | True | 1 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | clean MSE loss | loss2 | random clean | 7.47225e-08 | 7.88623e-08 | 4.13987e-09 | 2.75208 | 0.00413959 | 0.00449956 | True | True | 5 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | clean MSE loss | loss3 | loss2 | 2.02712e-07 | 5.12895e-07 | 3.10183e-07 | 7.7782 | 2.0888e-10 | 2.90112e-10 | True | True | 1 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | clean MSE loss | loss3 | random clean | 1.3292e-07 | 2.14144e-07 | 8.12239e-08 | 17.2411 | 9.54714e-23 | 2.98348e-22 | True | True | 1 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | clean MSE loss | loss3 | loss2 | 1.19523e-07 | 1.70816e-07 | 5.12934e-08 | 3.31679 | 0.000860513 | 0.00102442 | True | True | 1 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | clean MSE loss | loss3 | loss1 | 9.71217e-08 | 2.5193e-07 | 1.54808e-07 | 7.562 | 4.49093e-10 | 6.06882e-10 | True | True | 1 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | clean MSE loss | loss3 | loss2 | 1.38901e-07 | 1.89547e-07 | 5.06461e-08 | 4.14321 | 6.74712e-05 | 8.2282e-05 | True | True | 1 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | clean MSE loss | loss3 | loss2 | 2.51893e-07 | 6.63416e-07 | 4.11523e-07 | 28.2725 | 2.66649e-32 | 2.22208e-31 | True | True | 1 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | clean MSE loss | loss3 | loss2 | 4.74496e-07 | 1.423e-06 | 9.48505e-07 | 23.1297 | 2.54016e-28 | 1.5876e-27 | True | True | 1 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | clean MSE loss | loss3 | loss2 | 1.38143e-07 | 1.65431e-07 | 2.72878e-08 | 1.71378 | 0.0464448 | 0.0473927 | True | True | 1 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | clean MSE loss | loss3 | loss2 | 1.54686e-07 | 2.47171e-07 | 9.24844e-08 | 10.698 | 1.02206e-14 | 1.76218e-14 | True | True | 1 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | clean MSE loss | loss3 | loss2 | 2.48058e-07 | 9.21093e-07 | 6.73035e-07 | 15.112 | 2.23242e-20 | 5.07367e-20 | True | True | 1 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | clean MSE loss | loss3 | loss2 | 2.75359e-07 | 6.55521e-07 | 3.80162e-07 | 29.2454 | 5.56461e-33 | 6.95577e-32 | True | True | 1 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | clean MSE loss | loss3 | loss2 | 1.33404e-07 | 2.16931e-07 | 8.35267e-08 | 6.82417 | 6.2108e-09 | 7.76351e-09 | True | True | 1 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | clean MSE loss | loss3 | loss2 | 1.69046e-07 | 5.27664e-07 | 3.58618e-07 | 10.1239 | 6.67813e-14 | 1.07712e-13 | True | True | 1 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | clean MSE loss | loss3 | loss2 | 1.63898e-07 | 2.89063e-07 | 1.25165e-07 | 7.95392 | 1.12356e-10 | 1.60509e-10 | True | True | 1 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | clean MSE loss | loss2 | random clean | 5.16404e-08 | 5.17335e-08 | 9.31016e-11 | 0.144671 | 0.442782 | 0.442782 | False | False | 5 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | clean MSE loss | loss3 | loss2 | 6.43165e-07 | 1.54064e-06 | 8.9747e-07 | 34.9101 | 1.40476e-36 | 2.34126e-35 | True | True | 1 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | clean MSE loss | loss3 | loss2 | 2.35873e-07 | 2.82328e-07 | 4.64554e-08 | 2.43349 | 0.0093237 | 0.00991883 | True | True | 1 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | clean MSE loss | loss3 | loss2 | 1.24406e-07 | 1.69129e-07 | 4.47226e-08 | 2.01104 | 0.0249196 | 0.0259579 | True | True | 1 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | clean MSE loss | loss3 | loss2 | 4.36389e-07 | 1.02318e-06 | 5.86792e-07 | 15.4329 | 9.49911e-21 | 2.37478e-20 | True | True | 1 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | clean MSE loss | Physics Loss | loss1 | 3.00532e-08 | 4.06331e-08 | 1.05799e-08 | 14.9456 | 3.49319e-20 | 7.5939e-20 | True | True | 7 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | clean MSE loss | loss3 | random clean | 3.81914e-07 | 8.60307e-07 | 4.78393e-07 | 28.3881 | 2.20769e-32 | 2.20769e-31 | True | True | 1 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | clean MSE loss | loss3 | loss2 | 6.2768e-07 | 1.38865e-06 | 7.60968e-07 | 20.6421 | 4.02614e-26 | 1.67756e-25 | True | True | 1 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | clean MSE loss | loss3 | loss2 | 2.18312e-07 | 5.04518e-07 | 2.86205e-07 | 7.46302 | 6.38116e-10 | 8.39626e-10 | True | True | 1 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | clean MSE loss | loss3 | loss2 | 3.28795e-07 | 6.90609e-07 | 3.61814e-07 | 9.32234 | 9.80556e-13 | 1.53212e-12 | True | True | 1 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | clean MSE loss | loss3 | random clean | 2.02115e-07 | 5.02688e-07 | 3.00574e-07 | 48.2975 | 2.71789e-43 | 1.35894e-41 | True | True | 1 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | clean MSE loss | loss3 | random clean | 2.33196e-07 | 4.37899e-07 | 2.04703e-07 | 10.9607 | 4.3901e-15 | 8.12981e-15 | True | True | 1 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | clean MSE loss | loss3 | loss2 | 3.82736e-07 | 8.5076e-07 | 4.68024e-07 | 21.4103 | 8.007e-27 | 4.0035e-26 | True | True | 1 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | clean MSE loss | loss3 | random clean | 6.34737e-07 | 1.80457e-06 | 1.16984e-06 | 26.9492 | 2.42975e-31 | 1.73553e-30 | True | True | 1 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | clean MSE loss | loss3 | loss2 | 3.06107e-07 | 4.8929e-07 | 1.83182e-07 | 8.23746 | 4.15008e-11 | 6.10306e-11 | True | True | 1 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | clean MSE loss | loss3 | random clean | 1.25899e-07 | 2.14885e-07 | 8.89861e-08 | 19.8956 | 2.02699e-25 | 7.23924e-25 | True | True | 1 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | clean MSE loss | loss3 | random clean | 1.73137e-07 | 2.53949e-07 | 8.08126e-08 | 8.45897 | 1.91447e-11 | 2.90072e-11 | True | True | 1 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | clean MSE loss | loss3 | loss2 | 2.93601e-07 | 4.76317e-07 | 1.82717e-07 | 11.7222 | 3.9819e-16 | 7.9638e-16 | True | True | 1 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | clean MSE loss | loss3 | loss2 | 4.20646e-07 | 1.10621e-06 | 6.85562e-07 | 15.1393 | 2.07489e-20 | 4.94021e-20 | True | True | 1 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | clean MSE loss | loss3 | loss2 | 7.31497e-07 | 1.60573e-06 | 8.74235e-07 | 17.1796 | 1.11016e-22 | 3.26518e-22 | True | True | 1 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | clean MSE loss | loss2 | random clean | 5.11228e-08 | 5.34434e-08 | 2.3206e-09 | 2.81651 | 0.00348879 | 0.00387643 | True | True | 5 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | clean MSE loss | loss3 | loss2 | 6.7633e-07 | 1.4359e-06 | 7.59575e-07 | 20.7269 | 3.36074e-26 | 1.52761e-25 | True | True | 1 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | clean MSE loss | loss3 | loss2 | 1.76734e-07 | 2.6231e-07 | 8.55758e-08 | 6.95886 | 3.8411e-09 | 4.92449e-09 | True | True | 1 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | clean MSE loss | loss3 | loss2 | 3.2054e-07 | 8.34019e-07 | 5.13479e-07 | 12.0685 | 1.3706e-16 | 2.85542e-16 | True | True | 1 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | clean MSE loss | loss3 | loss2 | 4.07401e-07 | 1.00548e-06 | 5.98076e-07 | 17.1345 | 1.24034e-22 | 3.44539e-22 | True | True | 1 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | clean MSE loss | Physics Loss | loss1 | 3.12077e-08 | 3.81108e-08 | 6.90313e-09 | 10.8977 | 5.37195e-15 | 9.59277e-15 | True | True | 7 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | clean MSE loss | loss3 | loss2 | 4.78281e-07 | 1.0266e-06 | 5.48322e-07 | 20.2107 | 1.01879e-25 | 3.91842e-25 | True | True | 1 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | clean MSE loss | loss3 | loss2 | 5.59549e-07 | 1.20901e-06 | 6.49462e-07 | 16.1396 | 1.50834e-21 | 3.96933e-21 | True | True | 1 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | clean MSE loss | loss3 | loss2 | 1.45701e-07 | 2.17787e-07 | 7.20861e-08 | 2.82583 | 0.00340289 | 0.00386692 | True | True | 1 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | clean MSE loss | loss3 | loss2 | 4.34069e-07 | 8.42608e-07 | 4.08539e-07 | 11.0177 | 3.65851e-15 | 7.0356e-15 | True | True | 1 |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | clean RMSE from per-sample MSE | loss3 | Physics Loss | 0.000657619 | 0.00113368 | 0.000476057 | 48.1191 | 3.24639e-43 | 8.11598e-42 | True | True | 1 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | clean RMSE from per-sample MSE | loss2 | random clean | 0.000306048 | 0.000322552 | 1.65041e-05 | 3.60736 | 0.00036215 | 0.000431131 | True | True | 5 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000469083 | 0.000767832 | 0.000298749 | 12.1285 | 1.14089e-16 | 1.90149e-16 | True | True | 1 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | clean RMSE from per-sample MSE | loss3 | Physics Loss | 0.000507087 | 0.000833136 | 0.000326049 | 19.1168 | 1.15211e-24 | 2.88027e-24 | True | True | 1 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000540235 | 0.000830495 | 0.00029026 | 29.9308 | 1.89709e-33 | 1.18568e-32 | True | True | 1 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | clean RMSE from per-sample MSE | loss2 | random clean | 0.00026773 | 0.000276574 | 8.84364e-06 | 3.42522 | 0.000625606 | 0.000727449 | True | True | 5 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000435643 | 0.000697685 | 0.000262043 | 7.77033 | 2.14768e-10 | 2.75344e-10 | True | True | 1 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | clean RMSE from per-sample MSE | loss3 | random clean | 0.000363065 | 0.000460193 | 9.71281e-05 | 19.8094 | 2.45035e-25 | 6.44828e-25 | True | True | 1 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000334829 | 0.000391855 | 5.70262e-05 | 3.09909 | 0.0016062 | 0.00178466 | True | True | 1 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | clean RMSE from per-sample MSE | loss3 | loss1 | 0.00029844 | 0.000485914 | 0.000187474 | 8.11199 | 6.44379e-11 | 8.94972e-11 | True | True | 1 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000364248 | 0.000420507 | 5.62589e-05 | 4.43421 | 2.6073e-05 | 3.17963e-05 | True | True | 1 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000495582 | 0.000808251 | 0.000312669 | 41.7232 | 2.98757e-40 | 4.97928e-39 | True | True | 1 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000670228 | 0.00117985 | 0.000509617 | 26.1513 | 9.64598e-31 | 5.35888e-30 | True | True | 1 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.00036322 | 0.000385339 | 2.21196e-05 | 1.33695 | 0.0937065 | 0.0956188 | False | False | 1 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000389924 | 0.000491096 | 0.000101172 | 12.1859 | 9.58089e-17 | 1.65188e-16 | True | True | 1 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000486636 | 0.000946244 | 0.000459608 | 15.7862 | 3.75963e-21 | 8.17312e-21 | True | True | 1 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000523342 | 0.000806869 | 0.000283527 | 37.2392 | 6.63944e-38 | 6.63944e-37 | True | True | 1 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000356183 | 0.00045122 | 9.50373e-05 | 6.6544 | 1.13835e-08 | 1.42294e-08 | True | True | 1 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000404589 | 0.000706569 | 0.00030198 | 10.2181 | 4.89407e-14 | 7.64699e-14 | True | True | 1 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000396311 | 0.000520657 | 0.000124346 | 8.06616 | 7.56954e-11 | 1.02291e-10 | True | True | 1 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | clean RMSE from per-sample MSE | loss2 | random clean | 0.000226127 | 0.000226853 | 7.25125e-07 | 0.518915 | 0.303078 | 0.303078 | False | False | 5 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000791862 | 0.00123508 | 0.000443219 | 39.8018 | 2.8208e-39 | 3.526e-38 | True | True | 1 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000468224 | 0.000504213 | 3.59895e-05 | 2.32098 | 0.0122458 | 0.0130275 | True | True | 1 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000346576 | 0.000373958 | 2.73825e-05 | 1.50853 | 0.0689197 | 0.0717913 | False | False | 1 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000639661 | 0.000993288 | 0.000353627 | 18.8163 | 2.28561e-24 | 5.44193e-24 | True | True | 1 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | clean RMSE from per-sample MSE | Physics Loss | loss1 | 0.000173185 | 0.000201097 | 2.79124e-05 | 16.1891 | 1.32857e-21 | 3.01947e-21 | True | True | 7 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | clean RMSE from per-sample MSE | loss3 | random clean | 0.000607399 | 0.00092015 | 0.000312751 | 36.3433 | 2.10195e-37 | 1.75163e-36 | True | True | 1 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.00078084 | 0.00117092 | 0.000390083 | 22.6074 | 7.08085e-28 | 2.36028e-27 | True | True | 1 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000450255 | 0.000658827 | 0.000208572 | 8.82688 | 5.34783e-12 | 7.63975e-12 | True | True | 1 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000550399 | 0.000799071 | 0.000248672 | 12.3324 | 6.14279e-17 | 1.09693e-16 | True | True | 1 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | clean RMSE from per-sample MSE | loss3 | random clean | 0.000449021 | 0.000708013 | 0.000258992 | 60.9154 | 3.75779e-48 | 1.87889e-46 | True | True | 1 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | clean RMSE from per-sample MSE | loss3 | random clean | 0.000473601 | 0.000644507 | 0.000170906 | 13.3796 | 2.77702e-18 | 5.34043e-18 | True | True | 1 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000608662 | 0.000914847 | 0.000306185 | 24.7702 | 1.14652e-29 | 5.73258e-29 | True | True | 1 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | clean RMSE from per-sample MSE | loss3 | random clean | 0.000781071 | 0.00133398 | 0.00055291 | 31.7522 | 1.20613e-34 | 8.6152e-34 | True | True | 1 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000538419 | 0.000678007 | 0.000139588 | 9.54195 | 4.66213e-13 | 6.85608e-13 | True | True | 1 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | clean RMSE from per-sample MSE | loss3 | random clean | 0.000354292 | 0.0004619 | 0.000107608 | 22.7884 | 4.95261e-28 | 1.76879e-27 | True | True | 1 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | clean RMSE from per-sample MSE | loss3 | random clean | 0.000408723 | 0.000493281 | 8.45575e-05 | 9.70071 | 2.73304e-13 | 4.14097e-13 | True | True | 1 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000529873 | 0.000679134 | 0.00014926 | 13.2971 | 3.52677e-18 | 6.53105e-18 | True | True | 1 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.00063131 | 0.00102501 | 0.000393698 | 21.897 | 2.94766e-27 | 9.21143e-27 | True | True | 1 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000826639 | 0.00124303 | 0.000416392 | 21.4656 | 7.14046e-27 | 2.10014e-26 | True | True | 1 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | clean RMSE from per-sample MSE | loss2 | random clean | 0.000224078 | 0.000230203 | 6.12418e-06 | 3.35631 | 0.000766591 | 0.000871126 | True | True | 5 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000812493 | 0.00118836 | 0.000375869 | 24.2348 | 3.08997e-29 | 1.40453e-28 | True | True | 1 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000410459 | 0.000501875 | 9.14161e-05 | 7.97838 | 1.03082e-10 | 1.35634e-10 | True | True | 1 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000552677 | 0.000881696 | 0.000329019 | 15.2565 | 1.51735e-20 | 3.16115e-20 | True | True | 1 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000616813 | 0.000984449 | 0.000367636 | 23.1067 | 2.65652e-28 | 1.02174e-27 | True | True | 1 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | clean RMSE from per-sample MSE | Physics Loss | loss1 | 0.000176257 | 0.000194747 | 1.84899e-05 | 11.3157 | 1.42075e-15 | 2.29153e-15 | True | True | 7 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.00068073 | 0.00100488 | 0.000324148 | 23.3108 | 1.78828e-28 | 7.45116e-28 | True | True | 1 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000727753 | 0.00108055 | 0.000352794 | 21.2814 | 1.04645e-26 | 2.90679e-26 | True | True | 1 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000371262 | 0.00042218 | 5.09183e-05 | 2.49262 | 0.00805484 | 0.00875526 | True | True | 1 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | clean RMSE from per-sample MSE | loss3 | loss2 | 0.000637081 | 0.000892465 | 0.000255384 | 14.2228 | 2.53647e-19 | 5.07293e-19 | True | True | 1 |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | adversarial MSE loss | loss3 | loss2 | 2.30966e-06 | 4.4895e-06 | 2.17985e-06 | 37.5631 | 4.40506e-38 | 1.69425e-37 | True | True | 1 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | adversarial MSE loss | loss3 | loss2 | 1.39883e-06 | 3.24383e-06 | 1.845e-06 | 9.04884 | 2.49324e-12 | 2.54412e-12 | True | True | 1 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | adversarial MSE loss | loss3 | loss2 | 2.14941e-06 | 4.41045e-06 | 2.26104e-06 | 18.8391 | 2.16896e-24 | 3.28631e-24 | True | True | 1 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | adversarial MSE loss | loss3 | loss2 | 2.08911e-06 | 4.48448e-06 | 2.39537e-06 | 21.6144 | 5.25358e-27 | 1.0103e-26 | True | True | 1 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | adversarial MSE loss | loss3 | loss2 | 2.60831e-06 | 4.52628e-06 | 1.91797e-06 | 89.0686 | 3.65307e-56 | 6.08845e-55 | True | True | 1 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | adversarial MSE loss | loss3 | random clean | 8.36558e-07 | 3.76616e-06 | 2.9296e-06 | 11.6095 | 5.65398e-16 | 6.42497e-16 | True | True | 1 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | adversarial MSE loss | loss3 | random clean | 1.54194e-06 | 4.29219e-06 | 2.75025e-06 | 16.194 | 1.31208e-21 | 1.68216e-21 | True | True | 1 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | adversarial MSE loss | loss3 | loss2 | 1.57804e-06 | 4.49765e-06 | 2.91961e-06 | 19.7042 | 3.09116e-25 | 4.98575e-25 | True | True | 1 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | adversarial MSE loss | loss3 | loss2 | 1.2553e-06 | 3.96978e-06 | 2.71447e-06 | 14.5594 | 9.99827e-20 | 1.19027e-19 | True | True | 1 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | adversarial MSE loss | loss3 | loss2 | 2.15037e-06 | 4.43334e-06 | 2.28297e-06 | 22.6541 | 6.4549e-28 | 1.34477e-27 | True | True | 1 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | adversarial MSE loss | loss3 | loss2 | 1.50483e-06 | 4.4218e-06 | 2.91697e-06 | 18.1337 | 1.11827e-23 | 1.59753e-23 | True | True | 1 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | adversarial MSE loss | loss3 | loss2 | 2.57533e-06 | 4.52798e-06 | 1.95265e-06 | 42.5397 | 1.18516e-40 | 5.92579e-40 | True | True | 1 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | adversarial MSE loss | loss3 | loss2 | 2.33007e-06 | 4.52765e-06 | 2.19757e-06 | 35.1012 | 1.08595e-36 | 3.01654e-36 | True | True | 1 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | adversarial MSE loss | loss3 | loss2 | 1.80297e-06 | 3.94182e-06 | 2.13885e-06 | 11.4176 | 1.03071e-15 | 1.12033e-15 | True | True | 1 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | adversarial MSE loss | loss3 | loss2 | 1.90838e-06 | 4.51199e-06 | 2.60361e-06 | 18.4687 | 5.10162e-24 | 7.50238e-24 | True | True | 1 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | adversarial MSE loss | loss3 | loss2 | 1.62483e-06 | 4.44576e-06 | 2.82092e-06 | 19.9314 | 1.87379e-25 | 3.12298e-25 | True | True | 1 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | adversarial MSE loss | loss3 | loss2 | 2.62352e-06 | 4.56855e-06 | 1.94503e-06 | 104.756 | 1.34338e-59 | 3.35844e-58 | True | True | 1 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | adversarial MSE loss | loss3 | loss2 | 1.76307e-06 | 4.39806e-06 | 2.63499e-06 | 16.7546 | 3.18025e-22 | 4.29763e-22 | True | True | 1 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | adversarial MSE loss | loss3 | random clean | 1.53673e-06 | 4.38988e-06 | 2.85315e-06 | 17.4049 | 6.40358e-23 | 8.89386e-23 | True | True | 1 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | adversarial MSE loss | loss3 | loss2 | 1.89049e-06 | 4.49335e-06 | 2.60286e-06 | 16.4007 | 7.75074e-22 | 1.01983e-21 | True | True | 1 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | adversarial MSE loss | loss3 | random clean | 4.77704e-07 | 4.08005e-06 | 3.60234e-06 | 15.2504 | 1.54206e-20 | 1.88056e-20 | True | True | 1 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | adversarial MSE loss | loss3 | loss2 | 2.66348e-06 | 4.43004e-06 | 1.76656e-06 | 58.3264 | 3.0679e-47 | 3.0679e-46 | True | True | 1 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | adversarial MSE loss | loss3 | loss2 | 2.28708e-06 | 4.03078e-06 | 1.74369e-06 | 11.2794 | 1.59339e-15 | 1.69509e-15 | True | True | 1 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | adversarial MSE loss | loss3 | loss2 | 8.89381e-07 | 4.43868e-06 | 3.5493e-06 | 28.4252 | 2.07836e-32 | 5.46936e-32 | True | True | 1 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | adversarial MSE loss | loss3 | loss2 | 2.50417e-06 | 4.44137e-06 | 1.93719e-06 | 36.88 | 1.05058e-37 | 3.28536e-37 | True | True | 1 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | adversarial MSE loss | loss3 | loss1 | 4.59757e-07 | 5.27602e-07 | 6.78453e-08 | 11.5473 | 6.86488e-16 | 7.62765e-16 | True | True | 1 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | adversarial MSE loss | loss3 | loss2 | 2.57211e-06 | 4.36979e-06 | 1.79768e-06 | 36.8795 | 1.05132e-37 | 3.28536e-37 | True | True | 1 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | adversarial MSE loss | loss3 | loss2 | 2.58212e-06 | 4.47877e-06 | 1.89665e-06 | 37.1151 | 7.77711e-38 | 2.77754e-37 | True | True | 1 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | adversarial MSE loss | loss3 | loss2 | 1.99459e-06 | 4.18234e-06 | 2.18774e-06 | 16.0767 | 1.77268e-21 | 2.21585e-21 | True | True | 1 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | adversarial MSE loss | loss3 | loss2 | 2.2512e-06 | 4.33223e-06 | 2.08102e-06 | 20.8793 | 2.43271e-26 | 4.34412e-26 | True | True | 1 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | adversarial MSE loss | loss3 | loss2 | 2.54301e-06 | 4.60283e-06 | 2.05983e-06 | 175.36 | 1.56375e-70 | 7.81875e-69 | True | True | 1 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | adversarial MSE loss | loss3 | loss2 | 2.31671e-06 | 4.24878e-06 | 1.93206e-06 | 22.1608 | 1.72816e-27 | 3.45632e-27 | True | True | 1 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | adversarial MSE loss | loss3 | loss2 | 2.70309e-06 | 4.42077e-06 | 1.71768e-06 | 54.0794 | 1.17903e-45 | 8.42165e-45 | True | True | 1 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | adversarial MSE loss | loss3 | loss2 | 2.38707e-06 | 4.38922e-06 | 2.00215e-06 | 36.7531 | 1.23671e-37 | 3.63739e-37 | True | True | 1 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | adversarial MSE loss | loss3 | loss2 | 2.42535e-06 | 4.2972e-06 | 1.87185e-06 | 20.1339 | 1.20381e-25 | 2.07553e-25 | True | True | 1 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | adversarial MSE loss | loss3 | loss2 | 1.19733e-06 | 4.49472e-06 | 3.29739e-06 | 23.9114 | 5.67523e-29 | 1.23375e-28 | True | True | 1 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | adversarial MSE loss | loss3 | loss2 | 2.35961e-06 | 4.12417e-06 | 1.76456e-06 | 24.0827 | 4.10878e-29 | 9.33814e-29 | True | True | 1 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | adversarial MSE loss | loss3 | loss2 | 2.61441e-06 | 4.36875e-06 | 1.75433e-06 | 63.1482 | 6.58124e-49 | 8.22656e-48 | True | True | 1 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | adversarial MSE loss | loss3 | loss2 | 2.31322e-06 | 4.39058e-06 | 2.07736e-06 | 24.5887 | 1.60102e-29 | 4.00254e-29 | True | True | 1 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | adversarial MSE loss | loss3 | loss2 | 2.53929e-06 | 4.4313e-06 | 1.89201e-06 | 39.2156 | 5.71099e-39 | 2.5959e-38 | True | True | 1 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | adversarial MSE loss | loss3 | random clean | 4.92723e-07 | 3.57481e-06 | 3.08208e-06 | 11.2616 | 1.68528e-15 | 1.7555e-15 | True | True | 1 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | adversarial MSE loss | loss3 | loss2 | 2.5655e-06 | 4.46093e-06 | 1.89543e-06 | 45.4059 | 5.24626e-42 | 2.91459e-41 | True | True | 1 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | adversarial MSE loss | loss3 | loss2 | 2.46028e-06 | 4.36063e-06 | 1.90035e-06 | 19.4805 | 5.08287e-25 | 7.94198e-25 | True | True | 1 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | adversarial MSE loss | loss3 | loss2 | 2.12932e-06 | 4.32991e-06 | 2.20059e-06 | 21.301 | 1.00466e-26 | 1.86048e-26 | True | True | 1 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | adversarial MSE loss | loss3 | loss2 | 2.55192e-06 | 4.37651e-06 | 1.82459e-06 | 39.0249 | 7.19922e-39 | 2.99968e-38 | True | True | 1 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | adversarial MSE loss | loss3 | loss1 | 4.47144e-07 | 5.96589e-07 | 1.49445e-07 | 1.7668 | 0.0417448 | 0.0417448 | True | True | 1 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | adversarial MSE loss | loss3 | loss2 | 2.70577e-06 | 4.45645e-06 | 1.75068e-06 | 51.9514 | 8.15556e-45 | 5.09723e-44 | True | True | 1 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | adversarial MSE loss | loss3 | loss2 | 2.53871e-06 | 4.47413e-06 | 1.93541e-06 | 57.5993 | 5.62347e-47 | 4.68623e-46 | True | True | 1 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | adversarial MSE loss | loss3 | random clean | 1.45603e-06 | 3.92032e-06 | 2.46429e-06 | 12.3602 | 5.64762e-17 | 6.567e-17 | True | True | 1 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | adversarial MSE loss | loss3 | loss2 | 2.45241e-06 | 4.3144e-06 | 1.86199e-06 | 24.3636 | 2.4301e-29 | 5.78595e-29 | True | True | 1 |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | absolute loss increase | loss3 | loss2 | 1.85846e-06 | 3.17591e-06 | 1.31745e-06 | 19.8748 | 2.12165e-25 | 6.24015e-25 | True | True | 1 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | absolute loss increase | loss3 | loss2 | 1.26056e-06 | 3.14158e-06 | 1.88102e-06 | 9.33456 | 9.40676e-13 | 9.59873e-13 | True | True | 1 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | absolute loss increase | loss3 | loss2 | 1.91484e-06 | 3.78196e-06 | 1.86712e-06 | 15.0343 | 2.75025e-20 | 4.29756e-20 | True | True | 1 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | absolute loss increase | loss3 | loss2 | 1.82021e-06 | 3.6928e-06 | 1.87259e-06 | 14.0967 | 3.6085e-19 | 4.87635e-19 | True | True | 1 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | absolute loss increase | loss3 | loss2 | 2.31011e-06 | 3.82514e-06 | 1.51503e-06 | 58.0918 | 3.72734e-47 | 6.21223e-46 | True | True | 1 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | absolute loss increase | loss3 | random clean | 7.19152e-07 | 3.6873e-06 | 2.96815e-06 | 11.7562 | 3.58442e-16 | 3.98269e-16 | True | True | 1 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | absolute loss increase | loss3 | random clean | 1.33922e-06 | 3.66846e-06 | 2.32924e-06 | 13.4613 | 2.19402e-18 | 2.61193e-18 | True | True | 1 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | absolute loss increase | loss3 | loss2 | 1.44512e-06 | 4.27858e-06 | 2.83346e-06 | 18.9536 | 1.66966e-24 | 4.63795e-24 | True | True | 1 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | absolute loss increase | loss3 | loss2 | 1.13578e-06 | 3.79896e-06 | 2.66318e-06 | 13.9991 | 4.746e-19 | 6.24473e-19 | True | True | 1 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | absolute loss increase | loss3 | loss2 | 2.05325e-06 | 4.14583e-06 | 2.09258e-06 | 18.5498 | 4.22563e-24 | 1.11201e-23 | True | True | 1 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | absolute loss increase | loss3 | loss2 | 1.36593e-06 | 4.23225e-06 | 2.86632e-06 | 17.309 | 8.08854e-23 | 1.68511e-22 | True | True | 1 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | absolute loss increase | loss3 | loss2 | 2.32344e-06 | 3.86456e-06 | 1.54112e-06 | 32.9905 | 2.00898e-35 | 1.67415e-34 | True | True | 1 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | absolute loss increase | loss3 | random clean | 1.85558e-06 | 3.0308e-06 | 1.17523e-06 | 14.5946 | 9.07721e-20 | 1.33488e-19 | True | True | 1 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | absolute loss increase | loss3 | loss2 | 1.66483e-06 | 3.77639e-06 | 2.11157e-06 | 11.4853 | 8.33632e-16 | 8.86843e-16 | True | True | 1 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | absolute loss increase | loss3 | loss2 | 1.75369e-06 | 4.26482e-06 | 2.51113e-06 | 17.5133 | 4.92326e-23 | 1.07027e-22 | True | True | 1 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | absolute loss increase | loss3 | random clean | 1.37678e-06 | 3.37825e-06 | 2.00147e-06 | 14.3263 | 1.9022e-19 | 2.64195e-19 | True | True | 1 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | absolute loss increase | loss3 | loss2 | 2.34816e-06 | 3.91303e-06 | 1.56486e-06 | 61.7881 | 1.88809e-48 | 4.72023e-47 | True | True | 1 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | absolute loss increase | loss3 | loss2 | 1.62966e-06 | 4.18112e-06 | 2.55146e-06 | 16.014 | 2.08348e-21 | 3.7205e-21 | True | True | 1 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | absolute loss increase | loss3 | random clean | 1.36769e-06 | 3.74479e-06 | 2.3771e-06 | 13.8727 | 6.77884e-19 | 8.47354e-19 | True | True | 1 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | absolute loss increase | loss3 | loss2 | 1.7266e-06 | 4.20429e-06 | 2.47769e-06 | 14.6904 | 6.98323e-20 | 1.05806e-19 | True | True | 1 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | absolute loss increase | loss3 | random clean | 3.68818e-07 | 4.02831e-06 | 3.6595e-06 | 15.5148 | 7.6533e-21 | 1.31954e-20 | True | True | 1 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | absolute loss increase | loss3 | loss2 | 2.02031e-06 | 2.8894e-06 | 8.69092e-07 | 23.2455 | 2.02894e-28 | 7.80361e-28 | True | True | 1 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | absolute loss increase | loss3 | loss2 | 2.05121e-06 | 3.74845e-06 | 1.69724e-06 | 10.9587 | 4.41757e-15 | 4.60164e-15 | True | True | 1 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | absolute loss increase | loss3 | loss2 | 7.64975e-07 | 4.26955e-06 | 3.50458e-06 | 25.1425 | 5.81552e-30 | 2.90776e-29 | True | True | 1 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | absolute loss increase | loss3 | loss2 | 2.06779e-06 | 3.41819e-06 | 1.3504e-06 | 24.812 | 1.06175e-29 | 4.82616e-29 | True | True | 1 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | absolute loss increase | loss3 | loss1 | 3.14234e-07 | 4.86969e-07 | 1.72735e-07 | 28.5368 | 1.73395e-32 | 9.63303e-32 | True | True | 1 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | absolute loss increase | loss3 | loss2 | 2.1902e-06 | 3.49231e-06 | 1.30211e-06 | 23.3318 | 1.71732e-28 | 7.15551e-28 | True | True | 1 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | absolute loss increase | loss3 | loss2 | 1.95444e-06 | 3.09012e-06 | 1.13569e-06 | 18.2111 | 9.31968e-24 | 2.21897e-23 | True | True | 1 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | absolute loss increase | loss3 | loss2 | 1.77628e-06 | 3.67782e-06 | 1.90154e-06 | 12.2663 | 7.50342e-17 | 8.7249e-17 | True | True | 1 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | absolute loss increase | loss3 | random clean | 1.92241e-06 | 3.61792e-06 | 1.69551e-06 | 15.1203 | 2.18295e-20 | 3.63826e-20 | True | True | 1 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | absolute loss increase | loss3 | loss2 | 2.34089e-06 | 4.08175e-06 | 1.74085e-06 | 134.649 | 6.37313e-65 | 3.18656e-63 | True | True | 1 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | absolute loss increase | loss3 | loss2 | 2.08352e-06 | 3.76661e-06 | 1.68309e-06 | 17.195 | 1.06899e-22 | 2.05575e-22 | True | True | 1 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | absolute loss increase | loss3 | loss2 | 2.32035e-06 | 3.57001e-06 | 1.24966e-06 | 35.2357 | 9.06704e-37 | 9.06704e-36 | True | True | 1 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | absolute loss increase | loss3 | loss2 | 1.75233e-06 | 2.57269e-06 | 8.20359e-07 | 13.9826 | 4.97124e-19 | 6.37338e-19 | True | True | 1 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | absolute loss increase | loss3 | loss2 | 2.11924e-06 | 3.80791e-06 | 1.68867e-06 | 17.6993 | 3.14395e-23 | 7.14533e-23 | True | True | 1 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | absolute loss increase | loss3 | loss2 | 1.07143e-06 | 4.26003e-06 | 3.1886e-06 | 22.8321 | 4.54496e-28 | 1.6232e-27 | True | True | 1 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | absolute loss increase | loss3 | loss2 | 2.18647e-06 | 3.832e-06 | 1.64552e-06 | 21.648 | 4.90352e-27 | 1.63451e-26 | True | True | 1 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | absolute loss increase | loss3 | loss2 | 2.32081e-06 | 3.89243e-06 | 1.57162e-06 | 45.2194 | 6.38922e-42 | 7.98653e-41 | True | True | 1 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | absolute loss increase | loss3 | loss2 | 1.89257e-06 | 3.28437e-06 | 1.3918e-06 | 13.8032 | 8.2553e-19 | 1.00674e-18 | True | True | 1 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | absolute loss increase | loss3 | random clean | 1.80779e-06 | 2.82327e-06 | 1.01548e-06 | 14.4925 | 1.20175e-19 | 1.71679e-19 | True | True | 1 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | absolute loss increase | loss3 | random clean | 3.81154e-07 | 3.52136e-06 | 3.14021e-06 | 11.5112 | 7.68656e-16 | 8.35495e-16 | True | True | 1 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | absolute loss increase | loss3 | random clean | 1.88917e-06 | 3.02183e-06 | 1.13266e-06 | 20.6948 | 3.59857e-26 | 1.12455e-25 | True | True | 1 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | absolute loss increase | loss3 | loss2 | 2.28355e-06 | 4.09832e-06 | 1.81478e-06 | 18.3202 | 7.21496e-24 | 1.80374e-23 | True | True | 1 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | absolute loss increase | loss3 | loss2 | 1.80878e-06 | 3.49589e-06 | 1.68711e-06 | 15.0343 | 2.75044e-20 | 4.29756e-20 | True | True | 1 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | absolute loss increase | loss3 | random clean | 2.14452e-06 | 3.35757e-06 | 1.21306e-06 | 17.2825 | 8.62811e-23 | 1.72562e-22 | True | True | 1 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | absolute loss increase | loss3 | loss1 | 3.12191e-07 | 5.58478e-07 | 2.46287e-07 | 2.92066 | 0.0026337 | 0.0026337 | True | True | 1 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | absolute loss increase | loss3 | loss2 | 2.22749e-06 | 3.42985e-06 | 1.20235e-06 | 29.6345 | 3.01261e-33 | 1.88288e-32 | True | True | 1 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | absolute loss increase | loss3 | loss2 | 1.97916e-06 | 3.26512e-06 | 1.28595e-06 | 31.3578 | 2.16342e-34 | 1.5453e-33 | True | True | 1 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | absolute loss increase | loss3 | random clean | 1.31033e-06 | 3.68324e-06 | 2.37291e-06 | 11.7777 | 3.35351e-16 | 3.81081e-16 | True | True | 1 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | absolute loss increase | loss3 | loss2 | 2.01834e-06 | 3.47179e-06 | 1.45345e-06 | 16.5659 | 5.10457e-22 | 9.4529e-22 | True | True | 1 |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | relative loss increase | random clean | loss2 | 2.57786 | 2.75196 | 0.174101 | 1.03771 | 0.15225 | 0.223897 | False | False | 7 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | relative loss increase | loss3 | loss1 | 11.9567 | 36.5116 | 24.5549 | 7.09839 | 2.33549e-09 | 7.29839e-09 | True | True | 1 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | relative loss increase | random solver | loss1 | 7.64653 | 7.9244 | 0.277865 | 1.29329 | 0.100988 | 0.157793 | False | False | 6 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | relative loss increase | random clean | random solver | 5.69708 | 6.54127 | 0.844191 | 3.7115 | 0.000263365 | 0.000526729 | True | True | 6 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | relative loss increase | random solver | loss1 | 4.35023 | 5.24701 | 0.896777 | 13.2561 | 3.97182e-18 | 2.83702e-17 | True | True | 7 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | relative loss increase | loss3 | random solver | 6.45231 | 29.9038 | 23.4515 | 10.264 | 4.20846e-14 | 1.61864e-13 | True | True | 1 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | relative loss increase | random clean | loss2 | 8.60487 | 10.1077 | 1.50286 | 1.57507 | 0.0608373 | 0.101395 | False | False | 3 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | relative loss increase | random solver | loss3 | 9.90796 | 10.7161 | 0.808133 | 0.68932 | 0.246937 | 0.322691 | False | False | 2 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | relative loss increase | loss3 | random clean | 11.674 | 30.6183 | 18.9442 | 5.38907 | 1.00606e-06 | 2.79461e-06 | True | True | 1 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | relative loss increase | loss2 | random clean | 17.5108 | 18.101 | 0.590167 | 0.562744 | 0.288087 | 0.351326 | False | False | 7 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | relative loss increase | loss3 | random solver | 11.3649 | 14.6879 | 3.32306 | 1.84797 | 0.0353234 | 0.0609024 | True | False | 1 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | relative loss increase | random solver | baseline | 4.51259 | 5.45059 | 0.938007 | 4.15973 | 6.39681e-05 | 0.000139061 | True | True | 7 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | relative loss increase | random clean | loss1 | 2.40321 | 2.50641 | 0.103203 | 0.674128 | 0.251699 | 0.322691 | False | False | 7 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | relative loss increase | loss3 | random solver | 14.8345 | 25.5997 | 10.7651 | 4.13438 | 6.94187e-05 | 0.000144622 | True | True | 1 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | relative loss increase | random solver | loss3 | 10.0941 | 11.8038 | 1.70971 | 1.52525 | 0.0668127 | 0.107762 | False | False | 2 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | relative loss increase | random clean | random solver | 4.11491 | 4.2095 | 0.0945972 | 0.315307 | 0.376934 | 0.410706 | False | False | 6 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | relative loss increase | random solver | Physics Loss | 4.09933 | 5.27219 | 1.17286 | 29.3722 | 4.55265e-33 | 4.55265e-32 | True | True | 7 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | relative loss increase | loss3 | random solver | 14.1163 | 16.4085 | 2.29227 | 1.13544 | 0.130857 | 0.198269 | False | False | 1 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | relative loss increase | random clean | loss2 | 8.42324 | 10.5252 | 2.10191 | 2.04058 | 0.0233501 | 0.0416966 | True | True | 3 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | relative loss increase | random solver | loss3 | 10.568 | 11.0886 | 0.520627 | 0.344243 | 0.366068 | 0.410706 | False | False | 2 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | relative loss increase | loss3 | random solver | 3.42656 | 24.6023 | 21.1757 | 43.3781 | 4.66629e-41 | 2.33315e-39 | True | True | 1 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | relative loss increase | random solver | loss1 | 1.80718 | 1.894 | 0.0868194 | 4.84156 | 6.64532e-06 | 1.74877e-05 | True | True | 7 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | relative loss increase | loss3 | loss1 | 11.2373 | 17.3609 | 6.12361 | 4.2155 | 5.33966e-05 | 0.000121356 | True | True | 1 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | relative loss increase | loss3 | random solver | 6.26967 | 16.9967 | 10.727 | 6.87566 | 5.16852e-09 | 1.52015e-08 | True | True | 1 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | relative loss increase | loss1 | random solver | 3.50566 | 3.5578 | 0.0521357 | 0.765177 | 0.223918 | 0.310997 | False | False | 7 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | relative loss increase | loss3 | random clean | 2.18232 | 9.74651 | 7.56419 | 31.2011 | 2.73417e-34 | 4.55695e-33 | True | True | 1 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | relative loss increase | random solver | Physics Loss | 3.14545 | 3.81187 | 0.66642 | 12.4747 | 4.00024e-17 | 2.22235e-16 | True | True | 7 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | relative loss increase | random solver | loss1 | 2.22562 | 2.23399 | 0.00837175 | 0.269659 | 0.394277 | 0.410706 | False | False | 7 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | relative loss increase | loss3 | random solver | 10.1791 | 10.3346 | 0.155545 | 0.111157 | 0.455973 | 0.465279 | False | False | 1 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | relative loss increase | loss1 | random clean | 6.98056 | 7.33771 | 0.357149 | 0.772184 | 0.221857 | 0.310997 | False | False | 6 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | relative loss increase | random solver | baseline | 4.76693 | 6.18367 | 1.41675 | 25.8308 | 1.69586e-30 | 1.41321e-29 | True | True | 7 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | relative loss increase | random solver | Physics Loss | 6.21243 | 8.38715 | 2.17471 | 7.83076 | 1.73481e-10 | 5.78271e-10 | True | True | 6 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | relative loss increase | random solver | Physics Loss | 3.61829 | 4.00042 | 0.382133 | 10.2766 | 4.03738e-14 | 1.61864e-13 | True | True | 7 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | relative loss increase | loss1 | random solver | 1.51494 | 1.52585 | 0.0109083 | 0.421827 | 0.337498 | 0.401783 | False | False | 7 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | relative loss increase | loss3 | random solver | 8.72509 | 8.97865 | 0.253562 | 0.289812 | 0.386591 | 0.410706 | False | False | 1 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | relative loss increase | loss3 | random solver | 8.54216 | 8.86483 | 0.322666 | 0.27769 | 0.391209 | 0.410706 | False | False | 1 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | relative loss increase | random solver | Physics Loss | 8.34195 | 11.445 | 3.10309 | 12.851 | 1.30272e-17 | 8.14203e-17 | True | True | 5 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | relative loss increase | random solver | Physics Loss | 6.44645 | 7.47154 | 1.02509 | 11.1223 | 2.62112e-15 | 1.19142e-14 | True | True | 7 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | relative loss increase | random solver | loss1 | 3.35701 | 3.68503 | 0.328013 | 2.80025 | 0.00364343 | 0.0070066 | True | True | 7 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | relative loss increase | loss1 | random solver | 2.07301 | 2.09831 | 0.0253042 | 0.709039 | 0.24083 | 0.322691 | False | False | 7 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | relative loss increase | loss3 | random solver | 3.48475 | 25.7981 | 22.3134 | 30.3854 | 9.40572e-34 | 1.17572e-32 | True | True | 1 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | relative loss increase | loss1 | random solver | 1.97893 | 2.09467 | 0.115735 | 4.44017 | 2.55633e-05 | 6.08649e-05 | True | True | 7 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | relative loss increase | random solver | Physics Loss | 11.6274 | 13.6569 | 2.02952 | 11.5373 | 7.08319e-16 | 3.54159e-15 | True | True | 3 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | relative loss increase | random solver | Physics Loss | 5.13996 | 5.98923 | 0.849264 | 2.27603 | 0.0136257 | 0.0252328 | True | True | 5 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | relative loss increase | loss1 | random clean | 3.59711 | 3.66046 | 0.0633523 | 0.569635 | 0.285764 | 0.351326 | False | False | 7 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | relative loss increase | loss3 | random clean | 2.33897 | 10.4846 | 8.14558 | 32.9526 | 2.12018e-35 | 5.30046e-34 | True | True | 1 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | relative loss increase | random solver | Physics Loss | 3.02395 | 3.24456 | 0.220609 | 9.97502 | 1.09365e-13 | 3.9059e-13 | True | True | 7 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | relative loss increase | random solver | loss1 | 2.78518 | 2.7855 | 0.000316112 | 0.00696906 | 0.497234 | 0.497234 | False | False | 7 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | relative loss increase | loss3 | baseline | 9.05567 | 20.0311 | 10.9754 | 4.50663 | 2.05057e-05 | 5.12643e-05 | True | True | 1 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | relative loss increase | random solver | loss1 | 4.71133 | 4.75064 | 0.0393157 | 0.358277 | 0.360838 | 0.410706 | False | False | 6 |

## Files

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/per_dataset_ttests_20260615/attack50_per_dataset_first_vs_second_ttests.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/per_dataset_ttests_20260615/attack50_per_dataset_ttest_summary.csv`