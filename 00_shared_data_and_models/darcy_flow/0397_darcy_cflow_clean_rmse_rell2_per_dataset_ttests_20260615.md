# Darcy Clean RMSE / Relative L2 Per-Dataset T-Tests

Each row is one paired t-test inside one dataset using matched clean samples. The test is `second - best > 0`, so a small p-value means the first-ranked model is significantly lower than the second-ranked model in that dataset.

This run is locked to `generalization_datasets_darcy_binary_loss3targeted_20260611` and refuses the obsolete `lossdrop50_selected_20260607` root.

## Summary

| scope | metric | metric_label | tests | best_model_counts | loss3_best_count | p05_significant_count | bh_significant_count | median_p | max_p | non_sig_dataset_short |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization50 | rmse | RMSE | 50 | loss3:47; random_clean:2; loss2:1 | 47 | 49 | 49 | 7.27357e-22 | 0.362113 | 01_matern_fine_frac0p24_a1p77634_t11p8641 |
| generalization50 | relative_l2 | Relative L2 | 50 | loss3:47; random_clean:2; loss2:1 | 47 | 49 | 49 | 1.39948e-21 | 0.338247 | 01_matern_fine_frac0p24_a1p77634_t11p8641 |
| all52 | rmse | RMSE | 52 | loss3:47; random_solver:2; random_clean:2; loss2:1 | 47 | 50 | 50 | 7.27357e-22 | 0.362113 | test; 01_matern_fine_frac0p24_a1p77634_t11p8641 |
| all52 | relative_l2 | Relative L2 | 52 | loss3:47; random_solver:2; random_clean:2; loss2:1 | 47 | 50 | 50 | 1.39948e-21 | 0.338247 | test; 01_matern_fine_frac0p24_a1p77634_t11p8641 |

## Generalization50 Rows

| dataset_short | metric_label | best_model_display | second_model_display | best_mean | second_mean | mean_diff_second_minus_best | t_stat | p_one_sided_best_lower | p_bh_by_metric_generalization50 | significant_p05 | significant_bh_gen50_p05 | loss3_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | RMSE | loss3 | Physics Loss | 0.000684267 | 0.00116473 | 0.000480466 | 48.7259 | 1.77847e-43 | 2.96411e-42 | True | True | 1 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | RMSE | loss2 | loss3 | 0.000381305 | 0.000386057 | 4.75262e-06 | 0.354848 | 0.362113 | 0.362113 | False | False | 2 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | RMSE | loss3 | loss2 | 0.000511576 | 0.000841563 | 0.000329987 | 12.9637 | 9.34207e-18 | 1.55701e-17 | True | True | 1 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | RMSE | loss3 | Physics Loss | 0.000528648 | 0.000863559 | 0.000334911 | 20.3886 | 6.93382e-26 | 1.65091e-25 | True | True | 1 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | RMSE | loss3 | loss2 | 0.000644147 | 0.000980464 | 0.000336317 | 29.7144 | 2.65813e-33 | 1.47674e-32 | True | True | 1 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | RMSE | loss3 | random clean | 0.0004105 | 0.000493361 | 8.28611e-05 | 7.74561 | 2.34384e-10 | 2.54765e-10 | True | True | 1 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | RMSE | loss3 | loss2 | 0.000440731 | 0.00072459 | 0.000283859 | 8.42703 | 2.13987e-11 | 2.43168e-11 | True | True | 1 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | RMSE | loss3 | random clean | 0.000488885 | 0.000685955 | 0.00019707 | 28.6075 | 1.54635e-32 | 7.73177e-32 | True | True | 1 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | RMSE | loss3 | loss2 | 0.000363142 | 0.000493181 | 0.000130039 | 7.40339 | 7.88657e-10 | 8.38997e-10 | True | True | 1 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | RMSE | loss3 | loss1 | 0.000309003 | 0.000515916 | 0.000206913 | 9.33971 | 9.2436e-13 | 1.07484e-12 | True | True | 1 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | RMSE | loss3 | loss2 | 0.000438136 | 0.000602219 | 0.000164084 | 12.3591 | 5.66601e-17 | 8.0943e-17 | True | True | 1 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | RMSE | loss3 | loss2 | 0.00062268 | 0.00100038 | 0.000377703 | 38.9162 | 8.21905e-39 | 6.84921e-38 | True | True | 1 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | RMSE | loss3 | loss2 | 0.000693939 | 0.00121108 | 0.000517141 | 26.541 | 4.89668e-31 | 2.04028e-30 | True | True | 1 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | RMSE | loss3 | loss2 | 0.000405346 | 0.000465454 | 6.01074e-05 | 3.40867 | 0.000657032 | 0.000684408 | True | True | 1 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | RMSE | loss3 | random clean | 0.0005938 | 0.000817742 | 0.000223942 | 17.9229 | 1.84251e-23 | 3.68503e-23 | True | True | 1 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | RMSE | loss3 | loss2 | 0.000503435 | 0.000988509 | 0.000485074 | 16.1587 | 1.43629e-21 | 2.76209e-21 | True | True | 1 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | RMSE | loss3 | loss2 | 0.000674311 | 0.00102064 | 0.000346329 | 40.2864 | 1.58623e-39 | 1.98279e-38 | True | True | 1 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | RMSE | loss3 | loss2 | 0.000463653 | 0.000663391 | 0.000199738 | 12.5789 | 2.92625e-17 | 4.57227e-17 | True | True | 1 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | RMSE | loss3 | loss2 | 0.000405613 | 0.000739394 | 0.000333781 | 11.6869 | 4.44346e-16 | 6.00467e-16 | True | True | 1 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | RMSE | loss3 | loss2 | 0.000464423 | 0.000661375 | 0.000196952 | 12.5681 | 3.0231e-17 | 4.58046e-17 | True | True | 1 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | RMSE | loss3 | random clean | 0.000391006 | 0.000487461 | 9.6455e-05 | 28.538 | 1.73052e-32 | 7.86598e-32 | True | True | 1 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | RMSE | loss3 | loss2 | 0.000900329 | 0.0013595 | 0.000459171 | 39.3047 | 5.12729e-39 | 5.12729e-38 | True | True | 1 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | RMSE | loss3 | loss2 | 0.000529565 | 0.000573599 | 4.40336e-05 | 2.59829 | 0.00617032 | 0.00629625 | True | True | 1 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | RMSE | loss3 | random clean | 0.000430046 | 0.000600302 | 0.000170256 | 10.3054 | 3.67305e-14 | 4.59131e-14 | True | True | 1 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | RMSE | loss3 | loss2 | 0.000720986 | 0.00109405 | 0.000373064 | 19.0989 | 1.19975e-24 | 2.49947e-24 | True | True | 1 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | RMSE | random clean | loss2 | 0.000199683 | 0.000227809 | 2.81262e-05 | 11.8192 | 2.94882e-16 | 4.09558e-16 | True | True | 3 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | RMSE | loss3 | random clean | 0.000791837 | 0.00113594 | 0.000344107 | 37.5007 | 4.76651e-38 | 3.40465e-37 | True | True | 1 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | RMSE | loss3 | loss2 | 0.000848885 | 0.00125537 | 0.000406488 | 22.4706 | 9.29085e-28 | 2.58079e-27 | True | True | 1 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | RMSE | loss3 | loss2 | 0.000560728 | 0.00082895 | 0.000268222 | 12.7293 | 1.86849e-17 | 3.0137e-17 | True | True | 1 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | RMSE | loss3 | loss2 | 0.000610319 | 0.000879108 | 0.00026879 | 13.0201 | 7.9145e-18 | 1.36457e-17 | True | True | 1 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | RMSE | loss3 | random clean | 0.000565819 | 0.000899576 | 0.000333757 | 113.849 | 2.31184e-61 | 1.15592e-59 | True | True | 1 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | RMSE | loss3 | random clean | 0.000685639 | 0.000901121 | 0.000215482 | 19.1673 | 1.02754e-24 | 2.23379e-24 | True | True | 1 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | RMSE | loss3 | loss2 | 0.000736345 | 0.00106617 | 0.000329827 | 24.9395 | 8.4105e-30 | 3.00375e-29 | True | True | 1 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | RMSE | loss3 | random clean | 0.000871448 | 0.00144149 | 0.000570041 | 31.746 | 1.21725e-34 | 7.60784e-34 | True | True | 1 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | RMSE | loss3 | loss2 | 0.000643311 | 0.000802941 | 0.00015963 | 10.7629 | 8.28723e-15 | 1.06247e-14 | True | True | 1 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | RMSE | loss3 | random clean | 0.000559599 | 0.000805804 | 0.000246206 | 64.3589 | 2.62352e-49 | 6.55881e-48 | True | True | 1 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | RMSE | loss3 | random clean | 0.000627981 | 0.000748003 | 0.000120022 | 12.4746 | 4.00171e-17 | 5.88487e-17 | True | True | 1 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | RMSE | loss3 | loss2 | 0.000696356 | 0.000871439 | 0.000175083 | 14.0581 | 4.02076e-19 | 7.17992e-19 | True | True | 1 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | RMSE | loss3 | loss2 | 0.000733187 | 0.00116416 | 0.000430972 | 25.461 | 3.27604e-30 | 1.26001e-29 | True | True | 1 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | RMSE | loss3 | loss2 | 0.000898631 | 0.00133105 | 0.000432424 | 21.8328 | 3.35966e-27 | 8.84122e-27 | True | True | 1 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | RMSE | loss3 | random clean | 0.000337339 | 0.000381822 | 4.44831e-05 | 8.03467 | 8.45582e-11 | 9.39535e-11 | True | True | 1 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | RMSE | loss3 | loss2 | 0.000884103 | 0.00127121 | 0.000387104 | 24.7002 | 1.30384e-29 | 4.34615e-29 | True | True | 1 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | RMSE | loss3 | loss2 | 0.000530335 | 0.00065597 | 0.000125635 | 9.59255 | 3.93115e-13 | 4.79409e-13 | True | True | 1 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | RMSE | loss3 | loss2 | 0.00066291 | 0.00103665 | 0.000373735 | 19.7103 | 3.04951e-25 | 6.93069e-25 | True | True | 1 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | RMSE | loss3 | loss2 | 0.00068878 | 0.00107239 | 0.000383614 | 23.9577 | 5.19968e-29 | 1.6249e-28 | True | True | 1 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | RMSE | random clean | loss2 | 0.000225717 | 0.000256489 | 3.07724e-05 | 11.4631 | 8.93708e-16 | 1.17593e-15 | True | True | 3 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | RMSE | loss3 | loss2 | 0.000800636 | 0.0011479 | 0.000347259 | 23.8215 | 6.72906e-29 | 1.97914e-28 | True | True | 1 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | RMSE | loss3 | loss2 | 0.000785267 | 0.00114904 | 0.000363772 | 21.7149 | 4.27469e-27 | 1.06867e-26 | True | True | 1 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | RMSE | loss3 | loss2 | 0.000460165 | 0.000619858 | 0.000159693 | 9.57583 | 4.15891e-13 | 4.95108e-13 | True | True | 1 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | RMSE | loss3 | loss2 | 0.000740416 | 0.00101833 | 0.000277914 | 14.7165 | 6.50482e-20 | 1.2046e-19 | True | True | 1 |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | Relative L2 | loss3 | Physics Loss | 0.058936 | 0.100393 | 0.0414573 | 48.8002 | 1.653e-43 | 2.75501e-42 | True | True | 1 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | Relative L2 | loss2 | loss3 | 0.0408146 | 0.041418 | 0.000603369 | 0.419763 | 0.338247 | 0.338247 | False | False | 2 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | Relative L2 | loss3 | loss2 | 0.0479309 | 0.0784836 | 0.0305528 | 13.3615 | 2.92608e-18 | 4.71948e-18 | True | True | 1 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | Relative L2 | loss3 | Physics Loss | 0.0483004 | 0.0787839 | 0.0304836 | 20.4501 | 6.07409e-26 | 1.38047e-25 | True | True | 1 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | Relative L2 | loss3 | loss2 | 0.0577908 | 0.0878757 | 0.0300849 | 31.2791 | 2.43307e-34 | 1.35171e-33 | True | True | 1 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | Relative L2 | loss3 | random clean | 0.0419188 | 0.0502748 | 0.00835596 | 7.79246 | 1.98613e-10 | 2.15884e-10 | True | True | 1 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | Relative L2 | loss3 | loss2 | 0.042621 | 0.0692784 | 0.0266574 | 8.29009 | 3.45195e-11 | 3.92267e-11 | True | True | 1 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | Relative L2 | loss3 | random clean | 0.0468406 | 0.0656746 | 0.018834 | 29.6038 | 3.161e-33 | 1.5805e-32 | True | True | 1 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | Relative L2 | loss3 | loss2 | 0.0377318 | 0.050927 | 0.0131952 | 7.41049 | 7.69027e-10 | 8.18114e-10 | True | True | 1 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | Relative L2 | loss3 | loss1 | 0.0316591 | 0.0527792 | 0.0211202 | 9.08323 | 2.21625e-12 | 2.57704e-12 | True | True | 1 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | Relative L2 | loss3 | loss2 | 0.0440543 | 0.0603406 | 0.0162864 | 12.8994 | 1.12906e-17 | 1.76416e-17 | True | True | 1 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | Relative L2 | loss3 | loss2 | 0.0554506 | 0.0890682 | 0.0336176 | 41.8748 | 2.51303e-40 | 2.51303e-39 | True | True | 1 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | Relative L2 | loss3 | loss2 | 0.0597964 | 0.104201 | 0.0444042 | 27.5927 | 8.20022e-32 | 3.41676e-31 | True | True | 1 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | Relative L2 | loss3 | loss2 | 0.0429334 | 0.0489112 | 0.00597775 | 3.2881 | 0.000935441 | 0.000974418 | True | True | 1 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | Relative L2 | loss3 | random clean | 0.0559966 | 0.077 | 0.0210033 | 18.487 | 4.88961e-24 | 9.77922e-24 | True | True | 1 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | Relative L2 | loss3 | loss2 | 0.0455704 | 0.0886471 | 0.0430768 | 15.9005 | 2.79407e-21 | 5.37322e-21 | True | True | 1 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | Relative L2 | loss3 | loss2 | 0.060374 | 0.0913326 | 0.0309587 | 42.5064 | 1.23023e-40 | 1.53778e-39 | True | True | 1 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | Relative L2 | loss3 | loss2 | 0.0459797 | 0.0656074 | 0.0196276 | 12.7343 | 1.84108e-17 | 2.70747e-17 | True | True | 1 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | Relative L2 | loss3 | loss2 | 0.0390022 | 0.0705694 | 0.0315672 | 11.6962 | 4.31742e-16 | 5.83434e-16 | True | True | 1 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | Relative L2 | loss3 | loss2 | 0.0447763 | 0.0635916 | 0.0188153 | 12.8566 | 1.28128e-17 | 1.94133e-17 | True | True | 1 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | Relative L2 | loss3 | random clean | 0.0399453 | 0.049796 | 0.00985071 | 28.6771 | 1.38175e-32 | 6.28067e-32 | True | True | 1 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | Relative L2 | loss3 | loss2 | 0.0777204 | 0.117331 | 0.0396107 | 41.1561 | 5.73626e-40 | 4.78021e-39 | True | True | 1 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | Relative L2 | loss3 | loss2 | 0.0556294 | 0.0598385 | 0.00420913 | 2.39446 | 0.0102575 | 0.0104668 | True | True | 1 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | Relative L2 | loss3 | random clean | 0.0432078 | 0.0601096 | 0.0169018 | 10.8418 | 6.42864e-15 | 8.0358e-15 | True | True | 1 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | Relative L2 | loss3 | loss2 | 0.0661585 | 0.100142 | 0.0339836 | 20.0532 | 1.43516e-25 | 3.11991e-25 | True | True | 1 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | Relative L2 | random clean | loss2 | 0.022053 | 0.0251594 | 0.00310637 | 11.8182 | 2.95814e-16 | 4.10853e-16 | True | True | 3 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | Relative L2 | loss3 | random clean | 0.0726953 | 0.104314 | 0.0316183 | 37.9641 | 2.66308e-38 | 1.9022e-37 | True | True | 1 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | Relative L2 | loss3 | loss2 | 0.0738515 | 0.109088 | 0.0352362 | 23.2136 | 2.1583e-28 | 5.99529e-28 | True | True | 1 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | Relative L2 | loss3 | loss2 | 0.0566342 | 0.0833746 | 0.0267404 | 13.3617 | 2.92445e-18 | 4.71948e-18 | True | True | 1 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | Relative L2 | loss3 | loss2 | 0.0591532 | 0.0850074 | 0.0258542 | 13.422 | 2.45711e-18 | 4.2364e-18 | True | True | 1 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | Relative L2 | loss3 | random clean | 0.0513472 | 0.0816352 | 0.0302881 | 114.773 | 1.55782e-61 | 7.78912e-60 | True | True | 1 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | Relative L2 | loss3 | random clean | 0.0663045 | 0.0870931 | 0.0207886 | 19.9893 | 1.65058e-25 | 3.43871e-25 | True | True | 1 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | Relative L2 | loss3 | loss2 | 0.0680121 | 0.0983734 | 0.0303612 | 25.9499 | 1.37405e-30 | 4.58016e-30 | True | True | 1 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | Relative L2 | loss3 | random clean | 0.0741363 | 0.1226 | 0.0484634 | 32.7478 | 2.84037e-35 | 1.77523e-34 | True | True | 1 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | Relative L2 | loss3 | loss2 | 0.0652752 | 0.0812039 | 0.0159287 | 11.1994 | 2.05217e-15 | 2.63098e-15 | True | True | 1 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | Relative L2 | loss3 | random clean | 0.0531959 | 0.0765983 | 0.0234025 | 64.8356 | 1.83509e-49 | 4.58774e-48 | True | True | 1 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | Relative L2 | loss3 | random clean | 0.0645853 | 0.0768732 | 0.0122879 | 12.6566 | 2.3201e-17 | 3.31443e-17 | True | True | 1 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | Relative L2 | loss3 | loss2 | 0.0678897 | 0.084863 | 0.0169732 | 14.2986 | 2.05449e-19 | 3.66873e-19 | True | True | 1 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | Relative L2 | loss3 | loss2 | 0.0665739 | 0.105626 | 0.0390525 | 27.2318 | 1.50369e-31 | 5.78342e-31 | True | True | 1 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | Relative L2 | loss3 | loss2 | 0.0777833 | 0.115145 | 0.0373615 | 22.8764 | 4.1655e-28 | 1.04137e-27 | True | True | 1 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | Relative L2 | loss3 | random clean | 0.0348772 | 0.0394755 | 0.00459832 | 8.03519 | 8.44027e-11 | 9.37807e-11 | True | True | 1 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | Relative L2 | loss3 | loss2 | 0.0768406 | 0.110307 | 0.0334662 | 26.2373 | 8.29973e-31 | 2.96419e-30 | True | True | 1 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | Relative L2 | loss3 | loss2 | 0.0551801 | 0.0680794 | 0.0128992 | 9.75321 | 2.29204e-13 | 2.72862e-13 | True | True | 1 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | Relative L2 | loss3 | loss2 | 0.0634706 | 0.0990414 | 0.0355708 | 21.0628 | 1.65261e-26 | 3.93479e-26 | True | True | 1 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | Relative L2 | loss3 | loss2 | 0.0628548 | 0.0978826 | 0.0350278 | 24.9335 | 8.50283e-30 | 2.65713e-29 | True | True | 1 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | Relative L2 | random clean | loss2 | 0.0248626 | 0.0282517 | 0.0033891 | 11.4648 | 8.88716e-16 | 1.16936e-15 | True | True | 3 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | Relative L2 | loss3 | loss2 | 0.0727722 | 0.10422 | 0.0314474 | 24.8285 | 1.03014e-29 | 3.02983e-29 | True | True | 1 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | Relative L2 | loss3 | loss2 | 0.0691387 | 0.101023 | 0.0318839 | 23.158 | 2.40429e-28 | 6.32709e-28 | True | True | 1 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | Relative L2 | loss3 | loss2 | 0.047715 | 0.0640317 | 0.0163167 | 10.0111 | 9.70292e-14 | 1.18328e-13 | True | True | 1 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | Relative L2 | loss3 | loss2 | 0.0706939 | 0.0971038 | 0.0264099 | 15.2466 | 1.55755e-20 | 2.88436e-20 | True | True | 1 |

## Files

- Per-sample clean metrics: `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_per_sample_ttests_20260615/clean_per_sample_rmse_relative_l2_52datasets_7models.csv`
- Per-dataset t-tests: `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_per_sample_ttests_20260615/clean_rmse_relative_l2_per_dataset_first_vs_second_ttests.csv`
- Summary: `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_per_sample_ttests_20260615/clean_rmse_relative_l2_per_dataset_ttest_summary.csv`
- Provenance: `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_per_sample_ttests_20260615/clean_per_sample_ttest_provenance.json`
