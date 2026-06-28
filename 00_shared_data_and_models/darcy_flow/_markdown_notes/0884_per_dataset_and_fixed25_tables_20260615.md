# Darcy CFlow/SIR20 Per-Dataset And Fixed-25 Tables

只使用 20260611 generalization 数据；未发现 forbidden lossdrop50/smoke；attack_steps 全是 50。

## Source Files

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/clean_52dataset_7model_rmse_wide.csv`: 52 rows
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/clean_52dataset_7model_relative_l2_wide.csv`: 52 rows
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/attack50_52dataset_7model_mean_std.csv`: 364 rows
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_aligned_25samples_7models.csv`: 175 rows
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_25samples_7models.csv`: 175 rows

## Clean RMSE + Relative L2: 52 datasets x 7 models

| split | dataset | RMSE winner | RelL2 winner | baseline RMSE | loss1 RMSE | loss2 RMSE | loss3 RMSE | Physics Loss RMSE | random clean RMSE | random solver RMSE | baseline RelL2 | loss1 RelL2 | loss2 RelL2 | loss3 RelL2 | Physics Loss RelL2 | random clean RelL2 | random solver RelL2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 | loss3 | loss3 | 0.001217 | 0.001202 | 0.00114 | 6.6100e-04 | 0.001153 | 0.001189 | 0.001265 | 0.104217 | 0.102866 | 0.097618 | 0.056588 | 0.098733 | 0.101752 | 0.10833 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 | loss2 | loss2 | 4.6666e-04 | 4.7750e-04 | 4.1286e-04 | 4.3516e-04 | 4.7211e-04 | 4.4168e-04 | 5.4911e-04 | 0.051092 | 0.05228 | 0.045202 | 0.047644 | 0.051689 | 0.048358 | 0.060119 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 | loss3 | loss3 | 6.4904e-04 | 5.9430e-04 | 5.3083e-04 | 3.9162e-04 | 6.4449e-04 | 4.8585e-04 | 8.3821e-04 | 0.066406 | 0.060805 | 0.054312 | 0.040068 | 0.06594 | 0.049709 | 0.08576 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | loss3 | loss3 | 8.9788e-04 | 9.1417e-04 | 8.6603e-04 | 4.9559e-04 | 9.0645e-04 | 9.2295e-04 | 0.001003 | 0.083533 | 0.085048 | 0.08057 | 0.046106 | 0.08433 | 0.085865 | 0.09335 |
| generalization | 04_wave_mix_frac0p12_a2p34332_t2p21748 | loss3 | loss3 | 0.001526 | 0.001455 | 0.001381 | 9.1447e-04 | 0.00146 | 0.00139 | 0.001605 | 0.131796 | 0.125661 | 0.119274 | 0.078966 | 0.126101 | 0.120059 | 0.138575 |
| generalization | 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | loss3 | loss3 | 5.9916e-04 | 6.4365e-04 | 5.4077e-04 | 5.2421e-04 | 6.2279e-04 | 5.9795e-04 | 6.6215e-04 | 0.063988 | 0.06874 | 0.057753 | 0.055984 | 0.066512 | 0.063859 | 0.070716 |
| generalization | 06_rectangles_frac0p2_a1p44134_t6p32822 | loss3 | loss3 | 7.3404e-04 | 6.7902e-04 | 6.0839e-04 | 4.5153e-04 | 7.0814e-04 | 5.8906e-04 | 8.7698e-04 | 0.074477 | 0.068894 | 0.061728 | 0.045813 | 0.071849 | 0.059767 | 0.08898 |
| generalization | 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | loss3 | loss3 | 0.001309 | 0.00128 | 0.001185 | 8.2392e-04 | 0.001251 | 0.001229 | 0.00136 | 0.119965 | 0.117279 | 0.10861 | 0.075499 | 0.114589 | 0.112581 | 0.124602 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 | loss3 | loss3 | 9.3484e-04 | 8.6813e-04 | 9.0663e-04 | 4.8876e-04 | 8.4179e-04 | 8.7270e-04 | 9.2842e-04 | 0.085721 | 0.079603 | 0.083134 | 0.044817 | 0.077188 | 0.080023 | 0.085132 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 | loss3 | loss3 | 0.001146 | 0.001089 | 9.9114e-04 | 6.2317e-04 | 0.001073 | 0.001003 | 0.001242 | 0.102344 | 0.097295 | 0.088543 | 0.05567 | 0.095832 | 0.089568 | 0.110978 |
| generalization | 10_highpass_grf_frac0p24_a1p11037_t11p7271 | random clean | random clean | 3.3722e-04 | 2.8327e-04 | 2.3290e-04 | 2.7002e-04 | 3.3684e-04 | 2.0785e-04 | 5.1383e-04 | 0.037236 | 0.03128 | 0.025718 | 0.029817 | 0.037194 | 0.022952 | 0.056739 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | loss3 | loss3 | 5.9052e-04 | 5.7077e-04 | 5.1854e-04 | 4.4747e-04 | 6.0502e-04 | 4.9692e-04 | 7.5313e-04 | 0.060975 | 0.058936 | 0.053543 | 0.046205 | 0.062473 | 0.051311 | 0.077767 |
| generalization | 12_wave_mix_frac0p16_a2p26209_t4p32867 | loss3 | loss3 | 0.001274 | 0.001213 | 0.001151 | 7.9232e-04 | 0.001244 | 0.00113 | 0.001413 | 0.116555 | 0.111015 | 0.105317 | 0.072501 | 0.113874 | 0.103421 | 0.129264 |
| generalization | 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | loss3 | loss3 | 0.00141 | 0.00136 | 0.001263 | 8.6315e-04 | 0.001335 | 0.001299 | 0.001468 | 0.123343 | 0.119022 | 0.110537 | 0.075525 | 0.116783 | 0.113683 | 0.128469 |
| generalization | 14_rectangles_frac0p24_a1p67693_t9p58326 | loss3 | loss3 | 0.001034 | 9.6343e-04 | 9.1165e-04 | 5.9191e-04 | 9.3614e-04 | 9.1924e-04 | 0.001033 | 0.103546 | 0.096467 | 0.091281 | 0.059266 | 0.093733 | 0.092042 | 0.103473 |
| generalization | 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | loss3 | loss3 | 9.5918e-04 | 9.1400e-04 | 8.7359e-04 | 5.8038e-04 | 9.0191e-04 | 8.8574e-04 | 9.6822e-04 | 0.093004 | 0.088623 | 0.084705 | 0.056275 | 0.087451 | 0.085883 | 0.09388 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 | loss3 | loss3 | 7.1837e-04 | 7.8773e-04 | 7.3191e-04 | 4.4540e-04 | 7.6476e-04 | 8.2145e-04 | 8.4288e-04 | 0.068911 | 0.075565 | 0.07021 | 0.042726 | 0.073361 | 0.078799 | 0.080855 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 | loss3 | loss3 | 8.6297e-04 | 8.0786e-04 | 7.1918e-04 | 5.1713e-04 | 8.3165e-04 | 7.0655e-04 | 0.001021 | 0.082716 | 0.077433 | 0.068934 | 0.049567 | 0.079714 | 0.067723 | 0.097896 |
| generalization | 18_highpass_grf_frac0p12_a1p26489_t6p96292 | loss3 | loss3 | 0.001078 | 0.001009 | 9.2230e-04 | 5.7515e-04 | 0.001023 | 9.0858e-04 | 0.001232 | 0.097817 | 0.091618 | 0.083719 | 0.052207 | 0.092886 | 0.082473 | 0.111866 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | loss3 | loss3 | 6.4193e-04 | 5.9841e-04 | 5.9177e-04 | 4.4963e-04 | 6.0343e-04 | 6.0514e-04 | 6.7111e-04 | 0.065902 | 0.061434 | 0.060753 | 0.04616 | 0.06195 | 0.062125 | 0.068898 |
| generalization | 20_wave_mix_frac0p2_a2p69422_t4p60678 | loss3 | loss3 | 9.9846e-04 | 9.3951e-04 | 8.8127e-04 | 6.1988e-04 | 9.7854e-04 | 8.5361e-04 | 0.001152 | 0.09723 | 0.091489 | 0.085818 | 0.060363 | 0.09529 | 0.083124 | 0.112191 |
| generalization | 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | loss3 | loss3 | 0.001175 | 0.001147 | 0.001057 | 7.3773e-04 | 0.00116 | 0.001081 | 0.001296 | 0.108861 | 0.106286 | 0.097924 | 0.068377 | 0.107486 | 0.10021 | 0.120168 |
| generalization | 22_rectangles_frac0p12_a1p43234_t10p2242 | loss3 | loss3 | 0.001669 | 0.001543 | 0.00148 | 8.6256e-04 | 0.001514 | 0.001457 | 0.001664 | 0.140984 | 0.130327 | 0.125021 | 0.072843 | 0.127828 | 0.123038 | 0.140518 |
| generalization | 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | loss3 | loss3 | 8.6173e-04 | 8.5865e-04 | 7.7627e-04 | 6.0934e-04 | 8.1920e-04 | 8.1694e-04 | 8.8024e-04 | 0.085956 | 0.08565 | 0.077432 | 0.060781 | 0.081714 | 0.081488 | 0.087803 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 | loss3 | loss3 | 5.3714e-04 | 4.8801e-04 | 5.6411e-04 | 2.9714e-04 | 5.1685e-04 | 5.3818e-04 | 5.5570e-04 | 0.054166 | 0.049211 | 0.056885 | 0.029964 | 0.05212 | 0.054271 | 0.056037 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 | loss3 | loss3 | 7.0350e-04 | 6.7808e-04 | 5.9183e-04 | 4.5553e-04 | 6.9524e-04 | 5.9266e-04 | 8.3850e-04 | 0.071483 | 0.068901 | 0.060137 | 0.046287 | 0.070644 | 0.060221 | 0.085201 |
| generalization | 26_highpass_grf_frac0p16_a1p59204_t12p8363 | loss3 | loss3 | 9.6945e-04 | 9.1067e-04 | 8.3763e-04 | 5.5818e-04 | 9.4623e-04 | 8.0672e-04 | 0.001147 | 0.092155 | 0.086567 | 0.079625 | 0.05306 | 0.089948 | 0.076686 | 0.109078 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | loss3 | loss3 | 0.001153 | 0.001104 | 0.001033 | 6.7705e-04 | 0.00109 | 0.001025 | 0.001265 | 0.103204 | 0.098783 | 0.09248 | 0.060584 | 0.097532 | 0.091726 | 0.113207 |
| generalization | 28_wave_mix_frac0p24_a1p71757_t2p96649 | loss3 | loss3 | 8.7735e-04 | 8.2632e-04 | 7.9442e-04 | 6.4436e-04 | 8.8706e-04 | 7.3035e-04 | 0.001047 | 0.090704 | 0.085429 | 0.082131 | 0.066617 | 0.091709 | 0.075507 | 0.10822 |
| generalization | 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | loss3 | loss3 | 0.001075 | 0.001059 | 9.5757e-04 | 7.4061e-04 | 0.001069 | 9.8305e-04 | 0.001194 | 0.103634 | 0.102059 | 0.092279 | 0.071371 | 0.102972 | 0.094734 | 0.115058 |
| generalization | 30_rectangles_frac0p16_a1p24225_t5p53206 | loss3 | loss3 | 0.001374 | 0.001255 | 0.001219 | 7.9175e-04 | 0.001234 | 0.001182 | 0.001368 | 0.124112 | 0.113425 | 0.110122 | 0.071531 | 0.111443 | 0.106832 | 0.1236 |
| generalization | 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | loss3 | loss3 | 0.001399 | 0.001346 | 0.001271 | 8.7218e-04 | 0.001327 | 0.001304 | 0.001446 | 0.122687 | 0.117989 | 0.111427 | 0.07646 | 0.116301 | 0.114298 | 0.126798 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 | loss3 | loss3 | 0.001262 | 0.001274 | 0.001217 | 6.7808e-04 | 0.001247 | 0.001295 | 0.001356 | 0.10835 | 0.109389 | 0.104435 | 0.058212 | 0.107041 | 0.111173 | 0.116425 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 | loss3 | loss3 | 5.3227e-04 | 5.4424e-04 | 4.4824e-04 | 3.9633e-04 | 5.3554e-04 | 5.1184e-04 | 6.0859e-04 | 0.055528 | 0.056777 | 0.046761 | 0.041346 | 0.055869 | 0.053397 | 0.06349 |
| generalization | 34_highpass_grf_frac0p2_a1p38501_t10p2829 | loss3 | loss3 | 5.4870e-04 | 4.9931e-04 | 4.3085e-04 | 3.3685e-04 | 5.4523e-04 | 3.9190e-04 | 7.3569e-04 | 0.056625 | 0.051528 | 0.044462 | 0.034762 | 0.056266 | 0.040443 | 0.075921 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | loss3 | loss3 | 9.7389e-04 | 9.2704e-04 | 8.4728e-04 | 6.0025e-04 | 9.4855e-04 | 8.4161e-04 | 0.001125 | 0.092193 | 0.087757 | 0.080207 | 0.056822 | 0.089793 | 0.07967 | 0.106452 |
| generalization | 36_wave_mix_frac0p12_a1p72323_t5p41614 | loss3 | loss3 | 0.001489 | 0.001438 | 0.001299 | 8.8670e-04 | 0.001398 | 0.001386 | 0.001524 | 0.127589 | 0.123289 | 0.111383 | 0.076003 | 0.119812 | 0.11876 | 0.13063 |
| generalization | 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | loss3 | loss3 | 7.7981e-04 | 7.3699e-04 | 6.7402e-04 | 5.4986e-04 | 7.7006e-04 | 6.6886e-04 | 8.7650e-04 | 0.080659 | 0.07623 | 0.069716 | 0.056874 | 0.07965 | 0.069183 | 0.090659 |
| generalization | 38_rectangles_frac0p2_a1p49751_t8p36934 | loss3 | loss3 | 0.001197 | 0.001123 | 0.001049 | 6.6024e-04 | 0.001123 | 0.001069 | 0.001227 | 0.114577 | 0.107494 | 0.100413 | 0.063202 | 0.107453 | 0.102293 | 0.117454 |
| generalization | 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | loss3 | loss3 | 0.001225 | 0.001156 | 0.001059 | 7.1978e-04 | 0.001134 | 0.001111 | 0.001233 | 0.113438 | 0.107091 | 0.098106 | 0.066654 | 0.105006 | 0.102875 | 0.114166 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 | loss3 | loss3 | 0.00103 | 0.001101 | 0.001048 | 5.3230e-04 | 0.001099 | 0.001146 | 0.001204 | 0.092741 | 0.099131 | 0.094401 | 0.047947 | 0.098984 | 0.103269 | 0.108458 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 | loss3 | loss3 | 0.001152 | 0.001103 | 0.001003 | 6.5256e-04 | 0.001107 | 0.001015 | 0.001298 | 0.103686 | 0.099338 | 0.090277 | 0.05875 | 0.099648 | 0.091394 | 0.116882 |
| generalization | 42_highpass_grf_frac0p24_a1p18205_t11p6872 | random clean | random clean | 3.5098e-04 | 2.9193e-04 | 2.4937e-04 | 2.8243e-04 | 3.4465e-04 | 2.2187e-04 | 5.1856e-04 | 0.038682 | 0.032174 | 0.027483 | 0.031127 | 0.037985 | 0.024453 | 0.057151 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | loss3 | loss3 | 7.7455e-04 | 7.5240e-04 | 6.8443e-04 | 4.6762e-04 | 7.2985e-04 | 6.9098e-04 | 8.6173e-04 | 0.076669 | 0.074477 | 0.067748 | 0.046287 | 0.072244 | 0.068397 | 0.085298 |
| generalization | 44_wave_mix_frac0p16_a1p82574_t5p31661 | loss3 | loss3 | 0.001226 | 0.001168 | 0.001099 | 7.4385e-04 | 0.001189 | 0.0011 | 0.001308 | 0.111948 | 0.106669 | 0.100363 | 0.067922 | 0.10853 | 0.100458 | 0.119421 |
| generalization | 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | loss3 | loss3 | 0.001375 | 0.001334 | 0.001225 | 8.6257e-04 | 0.001317 | 0.001296 | 0.001441 | 0.120797 | 0.117221 | 0.107631 | 0.07577 | 0.115711 | 0.113805 | 0.126576 |
| generalization | 46_rectangles_frac0p24_a1p57209_t7p6296 | loss3 | loss3 | 8.2334e-04 | 7.9882e-04 | 7.0728e-04 | 5.0230e-04 | 7.9104e-04 | 7.2283e-04 | 9.0420e-04 | 0.084803 | 0.082277 | 0.072849 | 0.051736 | 0.081477 | 0.074451 | 0.093132 |
| generalization | 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | loss3 | loss3 | 0.001138 | 0.001096 | 0.001024 | 7.5567e-04 | 0.001118 | 0.001075 | 0.001213 | 0.109356 | 0.105384 | 0.098447 | 0.072635 | 0.107463 | 0.103353 | 0.116588 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 | loss3 | loss3 | 7.3668e-04 | 8.1145e-04 | 7.7617e-04 | 4.1038e-04 | 8.0357e-04 | 8.5613e-04 | 8.8533e-04 | 0.071099 | 0.078315 | 0.07491 | 0.039607 | 0.077554 | 0.082628 | 0.085445 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 | loss3 | loss3 | 8.2571e-04 | 7.8951e-04 | 7.2330e-04 | 4.8253e-04 | 8.0088e-04 | 7.2751e-04 | 9.4430e-04 | 0.079387 | 0.075907 | 0.069542 | 0.046392 | 0.077 | 0.069946 | 0.09079 |
| test | test_original_binary_grf_alpha2_tau3 | baseline | baseline | 1.5579e-04 | 1.8630e-04 | 1.7405e-04 | 2.6898e-04 | 1.9786e-04 | 1.8804e-04 | 1.9053e-04 | 0.022536 | 0.02695 | 0.025177 | 0.03891 | 0.028622 | 0.027201 | 0.027561 |
| train | train_original_binary_grf_alpha2_tau3 | random solver | random solver | 1.3228e-04 | 1.3275e-04 | 1.3042e-04 | 1.6933e-04 | 1.0887e-04 | 1.1380e-04 | 1.0247e-04 | 0.019132 | 0.019201 | 0.018864 | 0.024492 | 0.015746 | 0.01646 | 0.01482 |

## Attack50 adv_loss mean: 52 datasets x 7 models

| split | dataset | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 | 1.1051e-05 | 4.9622e-06 | 4.4895e-06 | 2.3097e-06 | 4.9797e-06 | 4.7084e-06 | 5.8617e-06 | loss3 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 | 1.0747e-05 | 4.0331e-06 | 3.2438e-06 | 1.3988e-06 | 4.2966e-06 | 3.6963e-06 | 5.5235e-06 | loss3 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 | 1.1498e-05 | 5.0169e-06 | 4.4174e-06 | 4.7770e-07 | 4.9456e-06 | 4.0800e-06 | 5.8810e-06 | loss3 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | 1.1144e-05 | 4.8740e-06 | 4.4104e-06 | 2.1494e-06 | 4.7726e-06 | 4.5302e-06 | 5.7299e-06 | loss3 |
| generalization | 04_wave_mix_frac0p12_a2p34332_t2p21748 | 1.1145e-05 | 4.8803e-06 | 4.4300e-06 | 2.6635e-06 | 4.8853e-06 | 4.6765e-06 | 5.8061e-06 | loss3 |
| generalization | 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | 1.0502e-05 | 4.6746e-06 | 4.0308e-06 | 2.2871e-06 | 4.5687e-06 | 4.3256e-06 | 5.5647e-06 | loss3 |
| generalization | 06_rectangles_frac0p2_a1p44134_t6p32822 | 1.1263e-05 | 4.9573e-06 | 4.4387e-06 | 8.8938e-07 | 4.9467e-06 | 4.6063e-06 | 5.8645e-06 | loss3 |
| generalization | 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | 1.1126e-05 | 4.8602e-06 | 4.4414e-06 | 2.5042e-06 | 4.8197e-06 | 4.5892e-06 | 5.7545e-06 | loss3 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 | 1.1020e-05 | 4.8861e-06 | 4.4845e-06 | 2.0891e-06 | 4.9052e-06 | 4.5551e-06 | 5.8313e-06 | loss3 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 | 1.1218e-05 | 5.0252e-06 | 4.5263e-06 | 2.6083e-06 | 4.9995e-06 | 4.8174e-06 | 5.9630e-06 | loss3 |
| generalization | 10_highpass_grf_frac0p24_a1p11037_t11p7271 | 1.1677e-05 | 5.2760e-07 | 5.9571e-07 | 4.5976e-07 | 1.4836e-06 | 7.8190e-07 | 5.8249e-06 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | 1.1355e-05 | 4.9791e-06 | 3.9545e-06 | 8.3656e-07 | 4.8838e-06 | 3.7662e-06 | 5.9007e-06 | loss3 |
| generalization | 12_wave_mix_frac0p16_a2p26209_t4p32867 | 1.1121e-05 | 4.8570e-06 | 4.3698e-06 | 2.5721e-06 | 4.8369e-06 | 4.7292e-06 | 5.7397e-06 | loss3 |
| generalization | 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | 1.1173e-05 | 4.9305e-06 | 4.4788e-06 | 2.5821e-06 | 4.8939e-06 | 4.6663e-06 | 5.8052e-06 | loss3 |
| generalization | 14_rectangles_frac0p24_a1p67693_t9p58326 | 1.1106e-05 | 4.4937e-06 | 4.1823e-06 | 1.9946e-06 | 4.5902e-06 | 4.2843e-06 | 5.4560e-06 | loss3 |
| generalization | 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | 1.1123e-05 | 4.7112e-06 | 4.3322e-06 | 2.2512e-06 | 4.6974e-06 | 4.3789e-06 | 5.6027e-06 | loss3 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 | 1.1065e-05 | 4.7847e-06 | 4.3129e-06 | 1.5419e-06 | 4.6701e-06 | 4.2922e-06 | 5.6114e-06 | loss3 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 | 1.1329e-05 | 5.0325e-06 | 4.4976e-06 | 1.5780e-06 | 4.9866e-06 | 4.8684e-06 | 5.9396e-06 | loss3 |
| generalization | 18_highpass_grf_frac0p12_a1p26489_t6p96292 | 1.1306e-05 | 5.0766e-06 | 4.6028e-06 | 2.5430e-06 | 5.0649e-06 | 4.9253e-06 | 5.9991e-06 | loss3 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | 1.0515e-05 | 4.4838e-06 | 3.9698e-06 | 1.2553e-06 | 4.5508e-06 | 4.1920e-06 | 5.6554e-06 | loss3 |
| generalization | 20_wave_mix_frac0p2_a2p69422_t4p60678 | 1.1213e-05 | 4.8095e-06 | 4.2488e-06 | 2.3167e-06 | 4.7841e-06 | 4.6986e-06 | 5.6753e-06 | loss3 |
| generalization | 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | 1.1061e-05 | 4.9135e-06 | 4.4208e-06 | 2.7031e-06 | 4.8766e-06 | 4.6844e-06 | 5.8295e-06 | loss3 |
| generalization | 22_rectangles_frac0p12_a1p43234_t10p2242 | 1.1257e-05 | 4.8043e-06 | 4.3892e-06 | 2.3871e-06 | 4.7465e-06 | 4.5142e-06 | 5.6835e-06 | loss3 |
| generalization | 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | 1.1162e-05 | 4.6636e-06 | 4.2972e-06 | 2.4253e-06 | 4.7043e-06 | 4.4137e-06 | 5.5826e-06 | loss3 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 | 1.0796e-05 | 4.8628e-06 | 4.4333e-06 | 2.1504e-06 | 4.8455e-06 | 4.5264e-06 | 5.7828e-06 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 | 1.1306e-05 | 4.8934e-06 | 4.4218e-06 | 1.5048e-06 | 4.8592e-06 | 4.6628e-06 | 5.7763e-06 | loss3 |
| generalization | 26_highpass_grf_frac0p16_a1p59204_t12p8363 | 1.1380e-05 | 5.0671e-06 | 4.4947e-06 | 1.1973e-06 | 5.0036e-06 | 4.9316e-06 | 5.9437e-06 | loss3 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | 1.1238e-05 | 5.0147e-06 | 4.5280e-06 | 2.5753e-06 | 5.0095e-06 | 4.8232e-06 | 5.9332e-06 | loss3 |
| generalization | 28_wave_mix_frac0p24_a1p71757_t2p96649 | 1.1122e-05 | 4.6269e-06 | 4.1242e-06 | 2.3596e-06 | 4.6432e-06 | 4.5309e-06 | 5.5178e-06 | loss3 |
| generalization | 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | 1.1083e-05 | 4.8319e-06 | 4.3687e-06 | 2.6144e-06 | 4.8503e-06 | 4.6735e-06 | 5.7666e-06 | loss3 |
| generalization | 30_rectangles_frac0p16_a1p24225_t5p53206 | 1.1159e-05 | 4.8212e-06 | 4.3906e-06 | 2.3132e-06 | 4.8183e-06 | 4.5592e-06 | 5.7187e-06 | loss3 |
| generalization | 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | 1.1199e-05 | 4.8055e-06 | 4.4313e-06 | 2.5393e-06 | 4.7479e-06 | 4.4971e-06 | 5.6560e-06 | loss3 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 | 1.0994e-05 | 4.9715e-06 | 4.5276e-06 | 2.3301e-06 | 4.9172e-06 | 4.6705e-06 | 5.8948e-06 | loss3 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 | 1.0948e-05 | 4.7876e-06 | 3.9418e-06 | 1.8030e-06 | 4.6871e-06 | 4.3976e-06 | 5.7343e-06 | loss3 |
| generalization | 34_highpass_grf_frac0p2_a1p38501_t10p2829 | 1.1533e-05 | 4.9938e-06 | 4.3100e-06 | 4.9272e-07 | 4.9326e-06 | 3.5748e-06 | 5.8788e-06 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | 1.1355e-05 | 5.0182e-06 | 4.5120e-06 | 1.9084e-06 | 4.9547e-06 | 4.8270e-06 | 5.8907e-06 | loss3 |
| generalization | 36_wave_mix_frac0p12_a1p72323_t5p41614 | 1.1161e-05 | 4.9076e-06 | 4.4609e-06 | 2.5655e-06 | 4.8733e-06 | 4.6127e-06 | 5.8084e-06 | loss3 |
| generalization | 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | 1.1086e-05 | 4.8185e-06 | 4.3606e-06 | 2.4603e-06 | 4.7956e-06 | 4.5778e-06 | 5.6954e-06 | loss3 |
| generalization | 38_rectangles_frac0p2_a1p49751_t8p36934 | 1.1183e-05 | 4.7231e-06 | 4.3299e-06 | 2.1293e-06 | 4.7078e-06 | 4.3810e-06 | 5.6216e-06 | loss3 |
| generalization | 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | 1.1185e-05 | 4.7569e-06 | 4.3765e-06 | 2.5519e-06 | 4.7594e-06 | 4.4835e-06 | 5.6624e-06 | loss3 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 | 1.1061e-05 | 4.9312e-06 | 4.4458e-06 | 1.6248e-06 | 4.8603e-06 | 4.4918e-06 | 5.6314e-06 | loss3 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 | 1.1236e-05 | 5.0552e-06 | 4.5686e-06 | 2.6235e-06 | 4.9908e-06 | 4.8376e-06 | 5.9436e-06 | loss3 |
| generalization | 42_highpass_grf_frac0p24_a1p18205_t11p6872 | 1.1675e-05 | 5.9659e-07 | 6.0014e-07 | 4.4714e-07 | 2.1104e-06 | 7.7299e-07 | 5.8134e-06 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | 1.1240e-05 | 4.8588e-06 | 4.3981e-06 | 1.7631e-06 | 4.8445e-06 | 4.5378e-06 | 5.7640e-06 | loss3 |
| generalization | 44_wave_mix_frac0p16_a1p82574_t5p31661 | 1.1157e-05 | 4.8657e-06 | 4.4565e-06 | 2.7058e-06 | 4.8622e-06 | 4.6274e-06 | 5.7562e-06 | loss3 |
| generalization | 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | 1.1177e-05 | 4.9240e-06 | 4.4741e-06 | 2.5387e-06 | 4.9077e-06 | 4.6802e-06 | 5.8159e-06 | loss3 |
| generalization | 46_rectangles_frac0p24_a1p57209_t7p6296 | 1.1226e-05 | 4.4157e-06 | 3.9662e-06 | 1.4560e-06 | 4.6136e-06 | 3.9203e-06 | 5.5799e-06 | loss3 |
| generalization | 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | 1.1196e-05 | 4.6746e-06 | 4.3144e-06 | 2.4524e-06 | 4.6948e-06 | 4.4262e-06 | 5.6142e-06 | loss3 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 | 1.0999e-05 | 4.8558e-06 | 4.4143e-06 | 1.5367e-06 | 4.7172e-06 | 4.3899e-06 | 5.6505e-06 | loss3 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 | 1.1255e-05 | 4.9620e-06 | 4.4934e-06 | 1.8905e-06 | 4.9312e-06 | 4.7674e-06 | 5.8608e-06 | loss3 |
| test | test | 4.6088e-06 | 1.3740e-06 | 1.4137e-06 | 1.4672e-06 | 1.8273e-06 | 1.0473e-06 | 2.1492e-06 | random clean |
| train | train | 4.0711e-06 | 1.2531e-06 | 1.2004e-06 | 1.5045e-06 | 1.8885e-06 | 8.3991e-07 | 2.5065e-06 | random clean |


## Attack50 loss_increase mean: 52 datasets x 7 models

| split | dataset | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 | 8.6622e-06 | 3.5720e-06 | 3.1759e-06 | 1.8585e-06 | 3.6795e-06 | 3.3217e-06 | 4.3114e-06 | loss3 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 | 1.0398e-05 | 3.9075e-06 | 3.1416e-06 | 1.2606e-06 | 4.1728e-06 | 3.5852e-06 | 5.3533e-06 | loss3 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 | 1.1159e-05 | 4.9372e-06 | 4.3657e-06 | 3.6882e-07 | 4.8480e-06 | 4.0283e-06 | 5.6474e-06 | loss3 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | 9.9753e-06 | 4.1637e-06 | 3.7820e-06 | 1.9148e-06 | 4.0937e-06 | 3.8467e-06 | 4.8700e-06 | loss3 |
| generalization | 04_wave_mix_frac0p12_a2p34332_t2p21748 | 7.8870e-06 | 3.1314e-06 | 2.8894e-06 | 2.0203e-06 | 3.1504e-06 | 3.1139e-06 | 3.6699e-06 | loss3 |
| generalization | 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | 9.7511e-06 | 4.3043e-06 | 3.7484e-06 | 2.0512e-06 | 4.2282e-06 | 4.0050e-06 | 5.1736e-06 | loss3 |
| generalization | 06_rectangles_frac0p2_a1p44134_t6p32822 | 1.0522e-05 | 4.7283e-06 | 4.2696e-06 | 7.6497e-07 | 4.7080e-06 | 4.4315e-06 | 5.4646e-06 | loss3 |
| generalization | 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | 8.7019e-06 | 3.6215e-06 | 3.4182e-06 | 2.0678e-06 | 3.6377e-06 | 3.4599e-06 | 4.3322e-06 | loss3 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 | 9.5578e-06 | 4.1151e-06 | 3.6928e-06 | 1.8202e-06 | 4.1926e-06 | 3.7917e-06 | 4.9566e-06 | loss3 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 | 9.3501e-06 | 4.1679e-06 | 3.8251e-06 | 2.3101e-06 | 4.1554e-06 | 4.0891e-06 | 4.7870e-06 | loss3 |
| generalization | 10_highpass_grf_frac0p24_a1p11037_t11p7271 | 1.1578e-05 | 4.8697e-07 | 5.5139e-07 | 3.1423e-07 | 1.4536e-06 | 7.0679e-07 | 5.7575e-06 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | 1.1045e-05 | 4.8723e-06 | 3.8798e-06 | 7.1915e-07 | 4.7709e-06 | 3.6873e-06 | 5.6830e-06 | loss3 |
| generalization | 12_wave_mix_frac0p16_a2p26209_t4p32867 | 9.1786e-06 | 3.8422e-06 | 3.4923e-06 | 2.1902e-06 | 3.7637e-06 | 3.8689e-06 | 4.2824e-06 | loss3 |
| generalization | 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | 8.1064e-06 | 3.3303e-06 | 3.0901e-06 | 1.9544e-06 | 3.3687e-06 | 3.2113e-06 | 3.9349e-06 | loss3 |
| generalization | 14_rectangles_frac0p24_a1p67693_t9p58326 | 9.5969e-06 | 3.8821e-06 | 3.6778e-06 | 1.7763e-06 | 4.0107e-06 | 3.7513e-06 | 4.7370e-06 | loss3 |
| generalization | 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | 9.5500e-06 | 3.8752e-06 | 3.6416e-06 | 1.9224e-06 | 3.9329e-06 | 3.6179e-06 | 4.7057e-06 | loss3 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 | 1.0268e-05 | 4.2372e-06 | 3.8000e-06 | 1.3392e-06 | 4.1295e-06 | 3.6685e-06 | 4.9547e-06 | loss3 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 | 1.0468e-05 | 4.7280e-06 | 4.2786e-06 | 1.4451e-06 | 4.6640e-06 | 4.6543e-06 | 5.3810e-06 | loss3 |
| generalization | 18_highpass_grf_frac0p12_a1p26489_t6p96292 | 9.7190e-06 | 4.4266e-06 | 4.0817e-06 | 2.3409e-06 | 4.3965e-06 | 4.4226e-06 | 4.9526e-06 | loss3 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | 9.9178e-06 | 4.2914e-06 | 3.7990e-06 | 1.1358e-06 | 4.3635e-06 | 4.0111e-06 | 5.4160e-06 | loss3 |
| generalization | 20_wave_mix_frac0p2_a2p69422_t4p60678 | 9.8993e-06 | 4.2556e-06 | 3.7666e-06 | 2.0835e-06 | 4.1656e-06 | 4.2607e-06 | 4.7708e-06 | loss3 |
| generalization | 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | 9.1535e-06 | 3.8863e-06 | 3.5700e-06 | 2.3203e-06 | 3.8358e-06 | 3.7871e-06 | 4.4973e-06 | loss3 |
| generalization | 22_rectangles_frac0p12_a1p43234_t10p2242 | 6.8598e-06 | 2.7787e-06 | 2.5727e-06 | 1.7523e-06 | 2.8310e-06 | 2.7096e-06 | 3.3315e-06 | loss3 |
| generalization | 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | 9.8168e-06 | 4.0388e-06 | 3.8079e-06 | 2.1192e-06 | 4.1137e-06 | 3.8578e-06 | 4.8914e-06 | loss3 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 | 1.0363e-05 | 4.6109e-06 | 4.1458e-06 | 2.0533e-06 | 4.5803e-06 | 4.2526e-06 | 5.4727e-06 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 | 1.0573e-05 | 4.6353e-06 | 4.2323e-06 | 1.3659e-06 | 4.5959e-06 | 4.4691e-06 | 5.3593e-06 | loss3 |
| generalization | 26_highpass_grf_frac0p16_a1p59204_t12p8363 | 1.0478e-05 | 4.7548e-06 | 4.2600e-06 | 1.0714e-06 | 4.6591e-06 | 4.7168e-06 | 5.3348e-06 | loss3 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | 9.3641e-06 | 4.2222e-06 | 3.8646e-06 | 2.3234e-06 | 4.2379e-06 | 4.1512e-06 | 4.8263e-06 | loss3 |
| generalization | 28_wave_mix_frac0p24_a1p71757_t2p96649 | 1.0268e-05 | 4.2794e-06 | 3.8320e-06 | 2.1865e-06 | 4.2238e-06 | 4.2769e-06 | 4.8758e-06 | loss3 |
| generalization | 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | 9.8973e-06 | 4.2299e-06 | 3.8924e-06 | 2.3208e-06 | 4.2143e-06 | 4.1677e-06 | 4.9246e-06 | loss3 |
| generalization | 30_rectangles_frac0p16_a1p24225_t5p53206 | 8.3470e-06 | 3.5274e-06 | 3.2844e-06 | 1.8926e-06 | 3.5966e-06 | 3.4214e-06 | 4.1990e-06 | loss3 |
| generalization | 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | 7.8350e-06 | 2.9637e-06 | 2.8256e-06 | 1.8078e-06 | 3.0158e-06 | 2.8233e-06 | 3.5608e-06 | loss3 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 | 8.7704e-06 | 3.3815e-06 | 3.1046e-06 | 1.8556e-06 | 3.3805e-06 | 3.0308e-06 | 4.0654e-06 | loss3 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 | 1.0365e-05 | 4.5514e-06 | 3.7764e-06 | 1.6648e-06 | 4.4729e-06 | 4.1931e-06 | 5.4563e-06 | loss3 |
| generalization | 34_highpass_grf_frac0p2_a1p38501_t10p2829 | 1.1195e-05 | 4.9135e-06 | 4.2589e-06 | 3.8115e-07 | 4.8365e-06 | 3.5214e-06 | 5.6516e-06 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | 1.0454e-05 | 4.6859e-06 | 4.2648e-06 | 1.7537e-06 | 4.6092e-06 | 4.5730e-06 | 5.3273e-06 | loss3 |
| generalization | 36_wave_mix_frac0p12_a1p72323_t5p41614 | 7.8535e-06 | 3.1568e-06 | 3.0250e-06 | 1.8892e-06 | 3.2246e-06 | 3.0218e-06 | 3.8279e-06 | loss3 |
| generalization | 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | 1.0291e-05 | 4.4807e-06 | 4.0983e-06 | 2.2835e-06 | 4.4251e-06 | 4.3036e-06 | 5.1963e-06 | loss3 |
| generalization | 38_rectangles_frac0p2_a1p49751_t8p36934 | 8.9945e-06 | 3.7341e-06 | 3.4959e-06 | 1.8088e-06 | 3.7589e-06 | 3.5060e-06 | 4.4597e-06 | loss3 |
| generalization | 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | 8.7709e-06 | 3.5367e-06 | 3.3710e-06 | 2.1445e-06 | 3.6182e-06 | 3.3576e-06 | 4.2945e-06 | loss3 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 | 9.9398e-06 | 3.9275e-06 | 3.5247e-06 | 1.3768e-06 | 3.8456e-06 | 3.3782e-06 | 4.4029e-06 | loss3 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 | 9.4778e-06 | 4.2440e-06 | 3.9130e-06 | 2.3482e-06 | 4.1792e-06 | 4.1728e-06 | 4.7607e-06 | loss3 |
| generalization | 42_highpass_grf_frac0p24_a1p18205_t11p6872 | 1.1567e-05 | 5.5848e-07 | 5.5914e-07 | 3.1219e-07 | 2.0792e-06 | 7.0386e-07 | 5.7381e-06 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | 1.0586e-05 | 4.5759e-06 | 4.1811e-06 | 1.6297e-06 | 4.5710e-06 | 4.3069e-06 | 5.3524e-06 | loss3 |
| generalization | 44_wave_mix_frac0p16_a1p82574_t5p31661 | 8.8119e-06 | 3.6734e-06 | 3.4298e-06 | 2.2275e-06 | 3.6520e-06 | 3.5758e-06 | 4.2556e-06 | loss3 |
| generalization | 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | 8.5541e-06 | 3.4754e-06 | 3.2651e-06 | 1.9792e-06 | 3.5257e-06 | 3.3595e-06 | 4.1290e-06 | loss3 |
| generalization | 46_rectangles_frac0p24_a1p57209_t7p6296 | 1.0338e-05 | 4.1232e-06 | 3.7484e-06 | 1.3103e-06 | 4.3314e-06 | 3.6832e-06 | 5.1817e-06 | loss3 |
| generalization | 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | 9.2055e-06 | 3.6467e-06 | 3.4718e-06 | 2.0183e-06 | 3.6917e-06 | 3.4809e-06 | 4.4343e-06 | loss3 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 | 1.0300e-05 | 4.2997e-06 | 3.8867e-06 | 1.3677e-06 | 4.1563e-06 | 3.7448e-06 | 4.9711e-06 | loss3 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 | 1.0324e-05 | 4.5842e-06 | 4.2043e-06 | 1.7266e-06 | 4.5456e-06 | 4.4591e-06 | 5.2824e-06 | loss3 |
| test | test | 4.5092e-06 | 1.3382e-06 | 1.3833e-06 | 1.4003e-06 | 1.7956e-06 | 1.0048e-06 | 2.1193e-06 | random clean |
| train | train | 4.0281e-06 | 1.2236e-06 | 1.1709e-06 | 1.4544e-06 | 1.8625e-06 | 8.0624e-07 | 2.4822e-06 | random clean |


## Attack50 delta_l2_rms mean: 52 datasets x 7 models

| split | dataset | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 | 3.892503 | 3.33304 | 3.289165 | 3.409469 | 3.294603 | 3.233204 | 3.273968 | random clean |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 | 4.671646 | 4.231617 | 4.181948 | 4.024415 | 4.217046 | 4.219072 | 4.262547 | loss3 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 | 4.483176 | 4.116549 | 4.041369 | 4.077755 | 4.083733 | 4.169581 | 4.085849 | loss2 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | 4.07868 | 3.525206 | 3.472846 | 3.699704 | 3.493928 | 3.452112 | 3.526536 | random clean |
| generalization | 04_wave_mix_frac0p12_a2p34332_t2p21748 | 3.701948 | 3.119021 | 3.080855 | 3.147987 | 3.10227 | 3.113895 | 3.130977 | loss2 |
| generalization | 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | 4.426335 | 4.102954 | 4.123224 | 3.856013 | 4.109929 | 4.089765 | 4.161388 | loss3 |
| generalization | 06_rectangles_frac0p2_a1p44134_t6p32822 | 4.347806 | 4.003916 | 3.965001 | 4.034054 | 3.995354 | 4.01957 | 4.00791 | loss2 |
| generalization | 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | 3.946339 | 3.401581 | 3.456489 | 3.445009 | 3.427131 | 3.408863 | 3.476531 | loss1 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 | 4.161912 | 3.596636 | 3.598453 | 3.715919 | 3.623669 | 3.519381 | 3.633683 | random clean |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 | 3.789872 | 3.265413 | 3.266325 | 3.327857 | 3.267477 | 3.269023 | 3.276657 | loss1 |
| generalization | 10_highpass_grf_frac0p24_a1p11037_t11p7271 | 4.782326 | 3.909147 | 3.856388 | 3.736237 | 4.104072 | 3.787067 | 4.400659 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | 4.524009 | 4.090488 | 4.106732 | 4.091198 | 4.094205 | 4.132326 | 4.086168 | random solver |
| generalization | 12_wave_mix_frac0p16_a2p26209_t4p32867 | 4.096994 | 3.533746 | 3.490668 | 3.402424 | 3.514823 | 3.59101 | 3.520219 | loss3 |
| generalization | 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | 3.677133 | 3.116111 | 3.11453 | 3.205267 | 3.141432 | 3.114128 | 3.159033 | random clean |
| generalization | 14_rectangles_frac0p24_a1p67693_t9p58326 | 4.334834 | 3.928625 | 3.968348 | 3.825048 | 3.998907 | 3.99694 | 4.057661 | loss3 |
| generalization | 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | 4.252223 | 3.683244 | 3.703367 | 3.613009 | 3.729286 | 3.663088 | 3.78819 | loss3 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 | 4.429689 | 3.834367 | 3.754445 | 4.013066 | 3.791124 | 3.721767 | 3.839982 | random clean |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 | 4.161943 | 3.739693 | 3.725576 | 4.131599 | 3.727547 | 3.762995 | 3.731122 | loss2 |
| generalization | 18_highpass_grf_frac0p12_a1p26489_t6p96292 | 3.84022 | 3.345066 | 3.359395 | 3.375152 | 3.350691 | 3.361501 | 3.335085 | random solver |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | 4.55736 | 4.22723 | 4.210336 | 4.001154 | 4.229186 | 4.180397 | 4.259717 | loss3 |
| generalization | 20_wave_mix_frac0p2_a2p69422_t4p60678 | 4.391433 | 3.935389 | 3.849594 | 3.682861 | 3.909209 | 3.998823 | 3.909776 | loss3 |
| generalization | 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | 4.059434 | 3.517037 | 3.488103 | 3.441162 | 3.493028 | 3.526575 | 3.536239 | loss3 |
| generalization | 22_rectangles_frac0p12_a1p43234_t10p2242 | 3.508143 | 3.002326 | 2.967315 | 3.054567 | 2.997541 | 2.98704 | 3.053521 | loss2 |
| generalization | 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | 4.470003 | 4.023732 | 4.053791 | 3.793774 | 4.054294 | 4.02927 | 4.101442 | loss3 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 | 4.82136 | 4.321456 | 4.258028 | 4.178579 | 4.309568 | 4.193324 | 4.345882 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 | 4.414892 | 3.979483 | 3.959072 | 4.04997 | 3.965153 | 3.986634 | 3.972392 | loss2 |
| generalization | 26_highpass_grf_frac0p16_a1p59204_t12p8363 | 4.165922 | 3.741333 | 3.701652 | 4.43357 | 3.728016 | 3.763999 | 3.722624 | loss2 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | 3.820519 | 3.329262 | 3.301846 | 3.428833 | 3.31123 | 3.304958 | 3.306163 | loss2 |
| generalization | 28_wave_mix_frac0p24_a1p71757_t2p96649 | 4.61638 | 4.186836 | 4.105473 | 3.825118 | 4.157906 | 4.255094 | 4.167242 | loss3 |
| generalization | 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | 4.337215 | 3.879928 | 3.875576 | 3.619208 | 3.881191 | 3.927919 | 3.908128 | loss3 |
| generalization | 30_rectangles_frac0p16_a1p24225_t5p53206 | 3.830342 | 3.378455 | 3.356914 | 3.399123 | 3.404145 | 3.379843 | 3.439988 | loss2 |
| generalization | 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | 3.642709 | 2.992977 | 3.028247 | 3.081724 | 2.99018 | 2.972635 | 3.01517 | random clean |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 | 3.83081 | 3.139486 | 3.13877 | 3.351015 | 3.099157 | 3.009734 | 3.119819 | random clean |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 | 4.559436 | 4.155372 | 4.232401 | 3.845642 | 4.141227 | 4.135133 | 4.237493 | loss3 |
| generalization | 34_highpass_grf_frac0p2_a1p38501_t10p2829 | 4.497135 | 4.113012 | 4.039678 | 4.037574 | 4.074298 | 4.131172 | 4.077052 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | 4.168498 | 3.730537 | 3.718452 | 3.922668 | 3.705815 | 3.734981 | 3.708025 | Physics Loss |
| generalization | 36_wave_mix_frac0p12_a1p72323_t5p41614 | 3.636334 | 3.094697 | 3.150346 | 3.150626 | 3.107269 | 3.070669 | 3.141659 | random clean |
| generalization | 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | 4.55931 | 4.186795 | 4.164592 | 3.921065 | 4.165737 | 4.204905 | 4.191872 | loss3 |
| generalization | 38_rectangles_frac0p2_a1p49751_t8p36934 | 4.141167 | 3.691035 | 3.704611 | 3.543151 | 3.699886 | 3.667713 | 3.761617 | loss3 |
| generalization | 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | 3.922295 | 3.337034 | 3.373317 | 3.460498 | 3.390933 | 3.333415 | 3.422371 | random clean |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 | 4.221563 | 3.500012 | 3.451178 | 3.838946 | 3.507417 | 3.360919 | 3.488732 | random clean |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 | 3.828702 | 3.323537 | 3.327187 | 3.333837 | 3.308792 | 3.317401 | 3.307502 | random solver |
| generalization | 42_highpass_grf_frac0p24_a1p18205_t11p6872 | 4.77571 | 3.940983 | 3.900271 | 3.784138 | 4.21242 | 3.80642 | 4.387313 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | 4.345403 | 3.883494 | 3.88089 | 3.906044 | 3.905652 | 3.933666 | 3.921571 | loss2 |
| generalization | 44_wave_mix_frac0p16_a1p82574_t5p31661 | 3.990966 | 3.492769 | 3.46749 | 3.398216 | 3.480025 | 3.498772 | 3.502263 | loss3 |
| generalization | 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | 3.699479 | 3.118023 | 3.13936 | 3.170504 | 3.127386 | 3.119089 | 3.140148 | loss1 |
| generalization | 46_rectangles_frac0p24_a1p57209_t7p6296 | 4.448745 | 4.044904 | 4.041623 | 3.830549 | 4.12253 | 4.080265 | 4.159488 | loss3 |
| generalization | 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | 4.13313 | 3.682919 | 3.708314 | 3.595201 | 3.683211 | 3.675073 | 3.751108 | loss3 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 | 4.492141 | 3.883342 | 3.846918 | 4.019762 | 3.83037 | 3.766096 | 3.874362 | random clean |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 | 4.15062 | 3.627221 | 3.630212 | 3.870792 | 3.631903 | 3.647834 | 3.63651 | loss1 |
| test | test | 4.354155 | 3.959111 | 4.164808 | 4.289312 | 4.236803 | 3.760067 | 4.350571 | random clean |
| train | train | 4.420606 | 4.030085 | 4.193032 | 4.254761 | 4.36184 | 3.843025 | 4.589495 | random clean |


## Fixed-25 adv_loss: 25 samples x 7 models

| split | sample | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 / idx0 | 1.0950e-05 | 5.0350e-06 | 4.5359e-06 | 2.2565e-06 | 5.0311e-06 | 4.8591e-06 | 5.8753e-06 | loss3 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 / idx0 | 1.0334e-05 | 4.1870e-06 | 3.8259e-06 | 2.6547e-06 | 4.1889e-06 | 3.8762e-06 | 5.4225e-06 | loss3 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 / idx0 | 1.1844e-05 | 4.8858e-06 | 4.2467e-06 | 4.6805e-07 | 4.6944e-06 | 4.8469e-06 | 5.8061e-06 | loss3 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 / idx0 | 1.0665e-05 | 4.2544e-06 | 3.7107e-06 | 2.6825e-06 | 4.0032e-06 | 3.4506e-06 | 4.8385e-06 | loss3 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 / idx0 | 1.1117e-05 | 4.8850e-06 | 4.4629e-06 | 2.2626e-06 | 5.0585e-06 | 4.5095e-06 | 5.9743e-06 | loss3 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 / idx0 | 1.0848e-05 | 4.9408e-06 | 4.4704e-06 | 2.6389e-06 | 4.9264e-06 | 4.6843e-06 | 5.9354e-06 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 / idx0 | 1.1329e-05 | 5.0118e-06 | 4.3972e-06 | 2.7157e-06 | 4.9241e-06 | 4.8954e-06 | 5.8768e-06 | loss3 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 / idx0 | 1.1519e-05 | 4.9892e-06 | 4.4315e-06 | 4.2953e-07 | 5.1184e-06 | 4.8101e-06 | 5.9519e-06 | loss3 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 / idx0 | 1.1572e-05 | 5.1245e-06 | 4.5929e-06 | 2.4358e-06 | 4.9968e-06 | 5.0056e-06 | 6.0648e-06 | loss3 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 / idx0 | 1.1137e-05 | 4.7809e-06 | 4.2438e-06 | 1.2895e-06 | 4.2816e-06 | 3.8815e-06 | 5.3326e-06 | loss3 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 / idx0 | 1.1020e-05 | 5.1001e-06 | 4.6739e-06 | 2.7310e-06 | 5.1052e-06 | 4.8766e-06 | 5.9913e-06 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 / idx0 | 1.0750e-05 | 4.9607e-06 | 4.5510e-06 | 2.4664e-06 | 4.9982e-06 | 4.7418e-06 | 5.9044e-06 | loss3 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 / idx0 | 1.1590e-05 | 4.9632e-06 | 4.5292e-06 | 2.2974e-06 | 5.0852e-06 | 4.9004e-06 | 6.0368e-06 | loss3 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 / idx0 | 1.0993e-05 | 5.1016e-06 | 4.5567e-06 | 2.6424e-06 | 5.1275e-06 | 5.0056e-06 | 6.0840e-06 | loss3 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 / idx0 | 1.1846e-05 | 4.5987e-06 | 4.1809e-06 | 2.8383e-06 | 4.6393e-06 | 4.3441e-06 | 5.4862e-06 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 / idx0 | 1.1144e-05 | 5.0789e-06 | 4.6500e-06 | 2.6183e-06 | 5.0894e-06 | 5.0013e-06 | 6.0608e-06 | loss3 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 / idx0 | 1.0892e-05 | 4.8219e-06 | 4.3188e-06 | 4.0239e-07 | 4.8880e-06 | 4.2239e-06 | 5.5163e-06 | loss3 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 / idx0 | 1.1053e-05 | 5.0820e-06 | 4.6131e-06 | 2.5634e-06 | 5.0302e-06 | 4.8588e-06 | 5.9490e-06 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 / idx0 | 1.1410e-05 | 4.6369e-06 | 4.2604e-06 | 5.0470e-07 | 4.6346e-06 | 4.7206e-06 | 5.5101e-06 | loss3 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 / idx0 | 1.0687e-05 | 4.7145e-06 | 4.2811e-06 | 5.7113e-07 | 4.3273e-06 | 4.2609e-06 | 5.2451e-06 | loss3 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 / idx0 | 1.1202e-05 | 5.1837e-06 | 4.5738e-06 | 2.1952e-06 | 4.9824e-06 | 4.8608e-06 | 6.0235e-06 | loss3 |
| test | test / idx0 | 1.0984e-05 | 1.7721e-07 | 2.2614e-06 | 1.8268e-06 | 2.7898e-06 | 2.8314e-07 | 2.6264e-06 | loss1 |
| test | test / idx1 | 4.0570e-07 | 5.9767e-08 | 1.0413e-07 | 1.0604e-06 | 4.4944e-08 | 1.3042e-07 | 2.4832e-06 | Physics Loss |
| train | train / idx0 | 7.9150e-07 | 1.3353e-07 | 2.6052e-07 | 1.6048e-06 | 8.4376e-08 | 2.4493e-07 | 9.7516e-08 | Physics Loss |
| train | train / idx1 | 3.2695e-07 | 8.8319e-08 | 1.4287e-07 | 1.1284e-06 | 6.7803e-08 | 1.7089e-07 | 7.0992e-08 | Physics Loss |


## Fixed-25 loss_increase: 25 samples x 7 models

| split | sample | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 / idx0 | 8.5642e-06 | 3.4601e-06 | 3.3393e-06 | 1.7673e-06 | 3.6160e-06 | 3.3939e-06 | 4.1792e-06 | loss3 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 / idx0 | 9.9140e-06 | 3.9763e-06 | 3.6464e-06 | 2.4450e-06 | 4.0341e-06 | 3.6879e-06 | 5.1904e-06 | loss3 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 / idx0 | 1.1547e-05 | 4.8151e-06 | 4.2025e-06 | 3.5990e-07 | 4.6065e-06 | 4.8000e-06 | 5.5870e-06 | loss3 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 / idx0 | 9.5982e-06 | 3.8375e-06 | 3.2965e-06 | 2.4451e-06 | 3.6305e-06 | 3.1257e-06 | 4.3289e-06 | loss3 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 / idx0 | 9.4146e-06 | 3.9295e-06 | 3.3901e-06 | 1.8608e-06 | 4.1683e-06 | 3.5403e-06 | 4.8782e-06 | loss3 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 / idx0 | 9.6799e-06 | 4.2215e-06 | 3.8925e-06 | 2.3312e-06 | 4.2121e-06 | 4.1005e-06 | 4.9360e-06 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 / idx0 | 1.1014e-05 | 4.8427e-06 | 4.2679e-06 | 2.5599e-06 | 4.7397e-06 | 4.7637e-06 | 5.5572e-06 | loss3 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 / idx0 | 9.8571e-06 | 4.5092e-06 | 4.0468e-06 | 1.6335e-07 | 4.6758e-06 | 4.2804e-06 | 5.3815e-06 | loss3 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 / idx0 | 1.0771e-05 | 4.7679e-06 | 4.3419e-06 | 2.2787e-06 | 4.6184e-06 | 4.7561e-06 | 5.4246e-06 | loss3 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 / idx0 | 1.0336e-05 | 4.6621e-06 | 3.9744e-06 | 1.1594e-06 | 4.0954e-06 | 3.6774e-06 | 5.1020e-06 | loss3 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 / idx0 | 1.0766e-05 | 4.7539e-06 | 4.3393e-06 | 2.6726e-06 | 4.7163e-06 | 4.4546e-06 | 5.5499e-06 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 / idx0 | 1.0407e-05 | 4.7880e-06 | 4.4305e-06 | 2.3758e-06 | 4.7893e-06 | 4.6132e-06 | 5.5771e-06 | loss3 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 / idx0 | 9.3301e-06 | 4.0620e-06 | 3.8440e-06 | 2.0574e-06 | 4.2942e-06 | 4.2031e-06 | 4.8539e-06 | loss3 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 / idx0 | 8.2538e-06 | 3.4536e-06 | 3.3115e-06 | 2.3093e-06 | 3.6920e-06 | 3.3663e-06 | 4.3061e-06 | loss3 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 / idx0 | 1.1244e-05 | 4.3146e-06 | 3.8816e-06 | 2.6936e-06 | 4.3131e-06 | 4.0773e-06 | 5.0723e-06 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 / idx0 | 1.0332e-05 | 4.7212e-06 | 4.3882e-06 | 2.4861e-06 | 4.6987e-06 | 4.7209e-06 | 5.4317e-06 | loss3 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 / idx0 | 9.9457e-06 | 4.0226e-06 | 3.4964e-06 | 2.5778e-07 | 4.0343e-06 | 3.2349e-06 | 4.4870e-06 | loss3 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 / idx0 | 9.8912e-06 | 4.4093e-06 | 4.1030e-06 | 2.3294e-06 | 4.3316e-06 | 4.2901e-06 | 4.9031e-06 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 / idx0 | 1.0730e-05 | 4.5122e-06 | 4.1577e-06 | 3.9419e-07 | 4.5161e-06 | 4.6238e-06 | 5.3067e-06 | loss3 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 / idx0 | 1.0107e-05 | 4.2211e-06 | 3.7152e-06 | 3.5664e-07 | 3.7807e-06 | 3.5668e-06 | 4.5877e-06 | loss3 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 / idx0 | 9.7745e-06 | 4.6373e-06 | 4.1230e-06 | 2.0947e-06 | 4.3673e-06 | 4.4025e-06 | 5.1426e-06 | loss3 |
| test | test / idx0 | 1.0727e-05 | 1.1833e-07 | 2.2213e-06 | 1.7877e-06 | 2.7575e-06 | 2.0093e-07 | 2.5948e-06 | loss1 |
| test | test / idx1 | 3.7490e-07 | 4.0846e-08 | 6.8118e-08 | 1.0396e-06 | 2.8729e-08 | 8.8732e-08 | 2.4674e-06 | Physics Loss |
| train | train / idx0 | 7.4219e-07 | 8.3314e-08 | 2.0841e-07 | 1.5728e-06 | 5.8137e-08 | 2.0916e-07 | 7.1690e-08 | Physics Loss |
| train | train / idx1 | 3.1472e-07 | 7.3887e-08 | 1.3612e-07 | 1.1063e-06 | 6.2913e-08 | 1.4500e-07 | 6.7209e-08 | Physics Loss |


## Fixed-25 residual sigma1: 25 samples x 7 models

| split | sample | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 / idx0 | 0.003438 | 0.002959 | 0.002928 | 0.002455 | 0.002915 | 0.003055 | 0.002858 | loss3 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 / idx0 | 0.002315 | 0.002025 | 0.002015 | 0.001821 | 0.001975 | 0.00205 | 0.00196 | loss3 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 / idx0 | 0.001969 | 0.001571 | 0.001573 | 0.001303 | 0.001556 | 0.001608 | 0.001544 | loss3 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 / idx0 | 0.002592 | 0.002038 | 0.002074 | 0.001688 | 0.001996 | 0.002085 | 0.001971 | loss3 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 / idx0 | 0.003334 | 0.002868 | 0.002904 | 0.002305 | 0.002803 | 0.002969 | 0.002753 | loss3 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 / idx0 | 0.002606 | 0.002138 | 0.002143 | 0.001773 | 0.002123 | 0.002194 | 0.002087 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 / idx0 | 0.00212 | 0.00174 | 0.001738 | 0.001478 | 0.001717 | 0.001769 | 0.001696 | loss3 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 / idx0 | 0.002665 | 0.002209 | 0.002225 | 0.001843 | 0.002163 | 0.002295 | 0.002134 | loss3 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 / idx0 | 0.002451 | 0.002028 | 0.00203 | 0.001686 | 0.002008 | 0.002073 | 0.001986 | loss3 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 / idx0 | 0.00237 | 0.001771 | 0.00186 | 0.001294 | 0.001759 | 0.001924 | 0.001702 | loss3 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 / idx0 | 0.002649 | 0.00238 | 0.002365 | 0.001911 | 0.002366 | 0.002502 | 0.002334 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 / idx0 | 0.0021 | 0.001814 | 0.001813 | 0.001572 | 0.00179 | 0.001855 | 0.001771 | loss3 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 / idx0 | 0.003143 | 0.002568 | 0.002553 | 0.002111 | 0.002532 | 0.002622 | 0.002506 | loss3 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 / idx0 | 0.003361 | 0.00289 | 0.002851 | 0.002317 | 0.00285 | 0.003003 | 0.002805 | loss3 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 / idx0 | 0.002638 | 0.0023 | 0.00233 | 0.001958 | 0.002266 | 0.002379 | 0.002221 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 / idx0 | 0.002464 | 0.002066 | 0.002052 | 0.001747 | 0.002049 | 0.00213 | 0.002017 | loss3 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 / idx0 | 0.003147 | 0.002766 | 0.002747 | 0.002152 | 0.002722 | 0.002878 | 0.002659 | loss3 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 / idx0 | 0.002661 | 0.002235 | 0.002224 | 0.001875 | 0.002219 | 0.002294 | 0.00219 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 / idx0 | 0.00237 | 0.001903 | 0.001898 | 0.001572 | 0.001855 | 0.001945 | 0.001822 | loss3 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 / idx0 | 0.00289 | 0.002587 | 0.002574 | 0.00195 | 0.002544 | 0.002713 | 0.002481 | loss3 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 / idx0 | 0.002793 | 0.002278 | 0.002274 | 0.001842 | 0.00225 | 0.00232 | 0.002217 | loss3 |
| test | test / idx0 | 0.001641 | 0.001209 | 0.0013 | 0.001014 | 0.001215 | 0.001264 | 0.001184 | loss3 |
| test | test / idx1 | 0.001132 | 0.001089 | 0.001074 | 9.0346e-04 | 0.001063 | 0.001151 | 0.00104 | loss3 |
| train | train / idx0 | 0.001229 | 0.001028 | 0.001041 | 9.2192e-04 | 0.001036 | 0.001071 | 0.001021 | loss3 |
| train | train / idx1 | 0.001508 | 0.00132 | 0.001346 | 0.001063 | 0.00129 | 0.001424 | 0.001258 | loss3 |


## Fixed-25 residual error L2 norm: 25 samples x 7 models

| split | sample | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 / idx0 | 0.131297 | 0.106669 | 0.092981 | 0.059453 | 0.101114 | 0.10289 | 0.110697 | loss3 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 / idx0 | 0.055061 | 0.039018 | 0.036013 | 0.038929 | 0.033441 | 0.036881 | 0.040944 | Physics Loss |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 / idx0 | 0.046278 | 0.022596 | 0.017869 | 0.027953 | 0.025199 | 0.018395 | 0.03978 | loss2 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 / idx0 | 0.087801 | 0.054882 | 0.054708 | 0.041422 | 0.051895 | 0.048451 | 0.060676 | loss3 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 / idx0 | 0.110918 | 0.083085 | 0.088042 | 0.053884 | 0.080201 | 0.083681 | 0.08899 | loss3 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 / idx0 | 0.091859 | 0.072094 | 0.064621 | 0.047148 | 0.071839 | 0.064946 | 0.084975 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 / idx0 | 0.047722 | 0.034953 | 0.030558 | 0.033546 | 0.036504 | 0.030845 | 0.048049 | loss2 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 / idx0 | 0.109565 | 0.05889 | 0.052725 | 0.043853 | 0.05655 | 0.061862 | 0.064194 | loss3 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 / idx0 | 0.076112 | 0.050754 | 0.042592 | 0.033697 | 0.052288 | 0.042466 | 0.068009 | loss3 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 / idx0 | 0.076068 | 0.029297 | 0.044123 | 0.030655 | 0.036675 | 0.038402 | 0.040824 | loss1 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 / idx0 | 0.042842 | 0.050013 | 0.049167 | 0.020544 | 0.053008 | 0.055217 | 0.056473 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 / idx0 | 0.049758 | 0.035323 | 0.029501 | 0.025589 | 0.038852 | 0.030481 | 0.048626 | loss3 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 / idx0 | 0.127782 | 0.080691 | 0.070359 | 0.04164 | 0.075601 | 0.070982 | 0.092448 | loss3 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 / idx0 | 0.140668 | 0.109116 | 0.094849 | 0.049056 | 0.101842 | 0.108829 | 0.113337 | loss3 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 / idx0 | 0.065929 | 0.045302 | 0.046502 | 0.032326 | 0.048544 | 0.043905 | 0.054688 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 / idx0 | 0.076629 | 0.050835 | 0.043491 | 0.030906 | 0.053132 | 0.045008 | 0.06742 | loss3 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 / idx0 | 0.082674 | 0.075993 | 0.077083 | 0.032324 | 0.078537 | 0.084529 | 0.086237 | loss3 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 / idx0 | 0.091614 | 0.069715 | 0.060707 | 0.041121 | 0.071045 | 0.0641 | 0.086928 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 / idx0 | 0.070103 | 0.030011 | 0.027242 | 0.028258 | 0.029264 | 0.026435 | 0.038332 | random clean |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 / idx0 | 0.064705 | 0.059705 | 0.063946 | 0.039366 | 0.062844 | 0.070813 | 0.068918 | loss3 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 / idx0 | 0.101558 | 0.062835 | 0.057066 | 0.026936 | 0.066667 | 0.057542 | 0.07978 | loss3 |
| test | test / idx0 | 0.043093 | 0.020627 | 0.017005 | 0.016818 | 0.015274 | 0.024372 | 0.015109 | random solver |
| test | test / idx1 | 0.014917 | 0.011692 | 0.01613 | 0.012258 | 0.010824 | 0.017355 | 0.010675 | random solver |
| train | train / idx0 | 0.018875 | 0.019048 | 0.019403 | 0.0152 | 0.013769 | 0.016076 | 0.01366 | random solver |
| train | train / idx1 | 0.009399 | 0.010211 | 0.006983 | 0.012638 | 0.005944 | 0.013676 | 0.005227 | random solver |


## Fixed-25 residual JT error norm: 25 samples x 7 models

| split | sample | baseline | loss1 | loss2 | loss3 | Physics Loss | random clean | random solver | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 / idx0 | 4.3975e-04 | 3.1125e-04 | 2.6930e-04 | 1.3887e-04 | 2.9121e-04 | 3.0941e-04 | 3.1364e-04 | loss3 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 / idx0 | 1.1816e-04 | 7.1753e-05 | 5.9098e-05 | 5.9705e-05 | 5.9088e-05 | 5.7382e-05 | 7.7501e-05 | random clean |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 / idx0 | 9.0320e-05 | 2.9415e-05 | 1.9970e-05 | 2.2223e-05 | 3.6152e-05 | 1.4211e-05 | 6.4223e-05 | random clean |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 / idx0 | 2.2512e-04 | 1.0855e-04 | 1.1100e-04 | 5.3499e-05 | 1.0004e-04 | 9.6095e-05 | 1.1745e-04 | loss3 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 / idx0 | 3.6350e-04 | 2.3672e-04 | 2.5412e-04 | 1.1556e-04 | 2.2260e-04 | 2.4696e-04 | 2.4367e-04 | loss3 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 / idx0 | 2.3780e-04 | 1.5248e-04 | 1.3747e-04 | 6.9223e-05 | 1.5136e-04 | 1.3764e-04 | 1.7913e-04 | loss3 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 / idx0 | 9.8107e-05 | 5.3263e-05 | 4.5247e-05 | 3.1682e-05 | 5.9074e-05 | 3.9182e-05 | 8.2527e-05 | loss3 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 / idx0 | 2.8463e-04 | 1.2128e-04 | 1.1025e-04 | 6.1936e-05 | 1.1400e-04 | 1.3196e-04 | 1.2835e-04 | loss3 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 / idx0 | 1.8709e-04 | 1.0173e-04 | 8.4666e-05 | 4.3852e-05 | 1.0507e-04 | 8.2439e-05 | 1.3881e-04 | loss3 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 / idx0 | 1.6756e-04 | 3.5502e-05 | 7.0112e-05 | 2.7509e-05 | 5.6269e-05 | 5.5295e-05 | 6.0222e-05 | loss3 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 / idx0 | 9.2928e-05 | 1.1183e-04 | 1.1014e-04 | 3.3520e-05 | 1.1844e-04 | 1.3015e-04 | 1.2401e-04 | loss3 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 / idx0 | 9.1694e-05 | 5.8547e-05 | 4.5472e-05 | 2.8597e-05 | 6.5277e-05 | 4.7142e-05 | 8.4770e-05 | loss3 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 / idx0 | 4.0218e-04 | 2.0738e-04 | 1.8057e-04 | 7.3512e-05 | 1.9115e-04 | 1.8540e-04 | 2.3263e-04 | loss3 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 / idx0 | 4.5952e-04 | 3.0741e-04 | 2.6469e-04 | 9.7993e-05 | 2.8223e-04 | 3.1322e-04 | 3.1109e-04 | loss3 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 / idx0 | 1.6862e-04 | 9.7849e-05 | 1.0267e-04 | 4.9989e-05 | 1.0492e-04 | 9.7399e-05 | 1.1849e-04 | loss3 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 / idx0 | 1.8697e-04 | 1.0395e-04 | 8.7406e-05 | 3.9559e-05 | 1.0883e-04 | 9.1140e-05 | 1.3900e-04 | loss3 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 / idx0 | 2.5425e-04 | 2.0887e-04 | 2.1055e-04 | 3.6108e-05 | 2.1209e-04 | 2.4187e-04 | 2.2805e-04 | loss3 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 / idx0 | 2.4250e-04 | 1.5563e-04 | 1.3438e-04 | 6.4848e-05 | 1.5745e-04 | 1.4531e-04 | 1.9240e-04 | loss3 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 / idx0 | 1.6493e-04 | 4.5501e-05 | 3.8863e-05 | 2.1392e-05 | 4.2207e-05 | 3.1132e-05 | 6.4815e-05 | loss3 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 / idx0 | 1.7311e-04 | 1.5288e-04 | 1.6337e-04 | 5.3891e-05 | 1.5829e-04 | 1.9063e-04 | 1.6963e-04 | loss3 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 / idx0 | 2.8498e-04 | 1.4201e-04 | 1.2846e-04 | 2.8504e-05 | 1.4911e-04 | 1.2970e-04 | 1.7808e-04 | loss3 |
| test | test / idx0 | 6.6776e-05 | 1.1939e-05 | 1.3524e-05 | 1.1010e-05 | 9.7049e-06 | 1.5591e-05 | 9.0754e-06 | random solver |
| test | test / idx1 | 8.7543e-06 | 7.5295e-06 | 1.2507e-05 | 4.4282e-06 | 7.9736e-06 | 1.6992e-05 | 7.2454e-06 | loss3 |
| train | train / idx0 | 1.3790e-05 | 1.4129e-05 | 1.2847e-05 | 8.2017e-06 | 7.6316e-06 | 1.0170e-05 | 8.3263e-06 | Physics Loss |
| train | train / idx1 | 5.4288e-06 | 8.7737e-06 | 4.2743e-06 | 5.8262e-06 | 2.6947e-06 | 1.5407e-05 | 2.6566e-06 | random solver |


## Fixed-25 model-solver top10 subspace similarity summary by sample

| split | sample | right overlap winner | loss3 right overlap | right angle winner | loss3 right angle | left overlap winner | loss3 left overlap | left angle winner | loss3 left angle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | 00_matern_smooth_frac0p12_a3p65909_t2p05625 / idx0 | loss3 | 0.769651 | loss3 | 39.399184 | random clean | 0.873415 | random clean | 25.573952 |
| generalization | 01_matern_fine_frac0p24_a1p77634_t11p8641 / idx0 | loss3 | 0.701378 | loss3 | 45.999777 | random clean | 0.823347 | random clean | 31.808437 |
| generalization | 02_highpass_grf_frac0p2_a1p39914_t8p09047 / idx0 | random clean | 0.772891 | random clean | 38.755129 | random clean | 0.889337 | random clean | 23.468842 |
| generalization | 03_bandpass_grf_frac0p16_a2p16503_t3p04694 / idx0 | loss3 | 0.784473 | loss3 | 37.649325 | loss3 | 0.88964 | loss3 | 23.457575 |
| generalization | 08_matern_smooth_frac0p16_a3p55666_t2p38988 / idx0 | loss3 | 0.798047 | loss3 | 36.087846 | loss3 | 0.880436 | loss3 | 24.569658 |
| generalization | 09_matern_fine_frac0p12_a1p37232_t13p3592 / idx0 | loss2 | 0.766536 | loss2 | 39.766838 | random clean | 0.880106 | random clean | 24.74869 |
| generalization | 11_bandpass_grf_frac0p2_a1p80507_t4p09153 / idx0 | loss2 | 0.77027 | loss2 | 38.990446 | random clean | 0.882733 | random clean | 24.515233 |
| generalization | 16_matern_smooth_frac0p2_a3p63184_t2p95736 / idx0 | loss3 | 0.756463 | loss3 | 40.688795 | Physics Loss | 0.847681 | Physics Loss | 28.471775 |
| generalization | 17_matern_fine_frac0p16_a1p05208_t9p11004 / idx0 | loss3 | 0.75933 | loss3 | 40.135385 | random clean | 0.872347 | random clean | 25.338301 |
| generalization | 19_bandpass_grf_frac0p24_a2p18225_t4p94437 / idx0 | loss3 | 0.798363 | loss3 | 36.101515 | loss3 | 0.891389 | loss3 | 23.464809 |
| generalization | 24_matern_smooth_frac0p24_a4p16883_t1p66293 / idx0 | loss3 | 0.731937 | loss3 | 43.320645 | random clean | 0.821708 | random clean | 31.666619 |
| generalization | 25_matern_fine_frac0p2_a1p32486_t12p8246 / idx0 | loss3 | 0.743728 | loss3 | 42.553133 | loss2 | 0.863383 | loss2 | 27.482888 |
| generalization | 27_bandpass_grf_frac0p12_a1p8965_t7p66724 / idx0 | loss3 | 0.758209 | loss3 | 40.566647 | random clean | 0.856256 | random clean | 28.062154 |
| generalization | 32_matern_smooth_frac0p12_a4p06817_t2p8029 / idx0 | loss3 | 0.741113 | loss3 | 42.182074 | loss3 | 0.845535 | loss3 | 28.836024 |
| generalization | 33_matern_fine_frac0p24_a1p77702_t7p67384 / idx0 | loss3 | 0.717746 | loss3 | 44.136346 | loss3 | 0.848166 | loss3 | 29.146681 |
| generalization | 35_bandpass_grf_frac0p16_a1p27246_t5p44182 / idx0 | loss3 | 0.770384 | loss3 | 39.088381 | random clean | 0.895376 | random clean | 23.361065 |
| generalization | 40_matern_smooth_frac0p16_a4p67408_t1p27431 / idx0 | loss3 | 0.762414 | loss3 | 41.096567 | random clean | 0.861593 | random clean | 28.075342 |
| generalization | 41_matern_fine_frac0p12_a1p13343_t8p3613 / idx0 | loss3 | 0.759001 | loss3 | 40.18044 | random clean | 0.878667 | random clean | 25.121991 |
| generalization | 43_bandpass_grf_frac0p2_a1p22554_t2p62907 / idx0 | loss3 | 0.736883 | loss3 | 42.554716 | random clean | 0.86395 | random clean | 26.920976 |
| generalization | 48_matern_smooth_frac0p2_a3p3086_t2p22775 / idx0 | loss3 | 0.762095 | loss3 | 40.203863 | loss3 | 0.860576 | loss3 | 27.275736 |
| generalization | 49_matern_fine_frac0p16_a1p39654_t9p9182 / idx0 | loss3 | 0.726488 | loss3 | 44.546793 | loss3 | 0.851646 | Physics Loss | 29.507234 |
| test | test / idx0 | loss3 | 0.717846 | loss3 | 44.579917 | loss3 | 0.82551 | loss3 | 32.751376 |
| test | test / idx1 | loss3 | 0.6853 | loss3 | 47.905462 | loss3 | 0.794057 | loss3 | 36.149849 |
| train | train / idx0 | loss3 | 0.68815 | loss3 | 48.605135 | loss3 | 0.787407 | loss3 | 36.778997 |
| train | train / idx1 | loss3 | 0.7241 | loss3 | 43.947202 | loss3 | 0.809549 | loss3 | 34.094214 |
