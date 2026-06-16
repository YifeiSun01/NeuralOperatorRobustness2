# Clean 52-Dataset Winner Summary

Source: `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_52dataset_metric_long_ranked.csv`.

Lower is better. `relative_gap_vs_second = (second - best) / second`. This is a practical gap, not a statistical p-value; the table does not contain per-sample variance for a formal significance test.

## Winner Counts

| split | metric | metric_label | datasets | loss3_best_count | loss3_best_fraction | best_model_counts | mean_relative_gap_vs_second | median_relative_gap_vs_second | min_relative_gap_vs_second | max_relative_gap_vs_second | gap_ge_1pct_count | gap_ge_5pct_count | gap_ge_10pct_count | gap_ge_20pct_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | data_mse | generalization loss / data MSE | 1 | 0 | 0 | random_solver:1 | 0.114127 | 0.114127 | 0.114127 | 0.114127 | 1 | 1 | 1 | 0 |
| train | rmse | RMSE | 1 | 0 | 0 | random_solver:1 | 0.0587915 | 0.0587915 | 0.0587915 | 0.0587915 | 1 | 1 | 0 | 0 |
| train | relative_l2 | Relative L2 | 1 | 0 | 0 | random_solver:1 | 0.0587915 | 0.0587915 | 0.0587915 | 0.0587915 | 1 | 1 | 0 | 0 |
| test | data_mse | generalization loss / data MSE | 1 | 0 | 0 | baseline:1 | 0.198831 | 0.198831 | 0.198831 | 0.198831 | 1 | 1 | 1 | 0 |
| test | rmse | RMSE | 1 | 0 | 0 | baseline:1 | 0.10492 | 0.10492 | 0.10492 | 0.10492 | 1 | 1 | 1 | 0 |
| test | relative_l2 | Relative L2 | 1 | 0 | 0 | baseline:1 | 0.10492 | 0.10492 | 0.10492 | 0.10492 | 1 | 1 | 1 | 0 |
| generalization | data_mse | generalization loss / data MSE | 50 | 47 | 0.94 | loss3:47; loss2:1; random_clean:2 | 0.481545 | 0.525199 | 0.0603119 | 0.732712 | 50 | 50 | 48 | 47 |
| generalization | rmse | RMSE | 50 | 47 | 0.94 | loss3:47; loss2:1; random_clean:2 | 0.287802 | 0.310948 | 0.0306249 | 0.483001 | 50 | 49 | 47 | 40 |
| generalization | relative_l2 | Relative L2 | 50 | 47 | 0.94 | loss3:47; loss2:1; random_clean:2 | 0.287802 | 0.310948 | 0.0306249 | 0.483001 | 50 | 49 | 47 | 40 |

## Generalization Mean Values

| split | metric | metric_label | model | model_display | mean_value | mean_rank |
| --- | --- | --- | --- | --- | --- | --- |
| generalization | data_mse | generalization loss / data MSE | loss3 | loss3 | 3.82036e-07 | 1 |
| generalization | data_mse | generalization loss / data MSE | loss2 | loss2 | 8.42343e-07 | 2 |
| generalization | data_mse | generalization loss / data MSE | random_clean | random clean | 8.74675e-07 | 3 |
| generalization | data_mse | generalization loss / data MSE | physics | Physics Loss | 9.65135e-07 | 4 |
| generalization | data_mse | generalization loss / data MSE | loss1 | loss1 | 9.68283e-07 | 5 |
| generalization | data_mse | generalization loss / data MSE | baseline | baseline | 1.0453e-06 | 6 |
| generalization | data_mse | generalization loss / data MSE | random_solver | random solver | 1.2271e-06 | 7 |
| generalization | rmse | RMSE | loss3 | loss3 | 0.000594572 | 1 |
| generalization | rmse | RMSE | loss2 | loss2 | 0.000870154 | 2 |
| generalization | rmse | RMSE | random_clean | random clean | 0.000883968 | 3 |
| generalization | rmse | RMSE | loss1 | loss1 | 0.000936259 | 4 |
| generalization | rmse | RMSE | physics | Physics Loss | 0.000938999 | 5 |
| generalization | rmse | RMSE | baseline | baseline | 0.000972351 | 6 |
| generalization | rmse | RMSE | random_solver | random solver | 0.00106732 | 7 |
| generalization | relative_l2 | Relative L2 | loss3 | loss3 | 0.0561539 | 1 |
| generalization | relative_l2 | Relative L2 | loss2 | loss2 | 0.0816375 | 2 |
| generalization | relative_l2 | Relative L2 | random_clean | random clean | 0.0828807 | 3 |
| generalization | relative_l2 | Relative L2 | loss1 | loss1 | 0.0879285 | 4 |
| generalization | relative_l2 | Relative L2 | physics | Physics Loss | 0.0882931 | 5 |
| generalization | relative_l2 | Relative L2 | baseline | baseline | 0.0913369 | 6 |
| generalization | relative_l2 | Relative L2 | random_solver | random solver | 0.100569 | 7 |

## Full 50-Generalization Dataset Winners

| dataset_short | data_mse_best_model | data_mse_second_model | data_mse_relative_gap_vs_second | rmse_best_model | rmse_second_model | rmse_relative_gap_vs_second | relative_l2_best_model | relative_l2_second_model | relative_l2_relative_gap_vs_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_matern_smooth_frac0p12_a3p65909_t2p05625 | loss3 | loss2 | 0.663961 | loss3 | loss2 | 0.420311 | loss3 | loss2 | 0.420311 |
| 01_matern_fine_frac0p24_a1p77634_t11p8641 | loss2 | loss3 | 0.0998896 | loss2 | loss3 | 0.0512585 | loss2 | loss3 | 0.0512585 |
| 02_highpass_grf_frac0p2_a1p39914_t8p09047 | loss3 | random_clean | 0.350284 | loss3 | random_clean | 0.19395 | loss3 | random_clean | 0.19395 |
| 03_bandpass_grf_frac0p16_a2p16503_t3p04694 | loss3 | loss2 | 0.672532 | loss3 | loss2 | 0.427752 | loss3 | loss2 | 0.427752 |
| 04_wave_mix_frac0p12_a2p34332_t2p21748 | loss3 | loss2 | 0.561687 | loss3 | loss2 | 0.337948 | loss3 | loss2 | 0.337948 |
| 05_blocky_tiles_frac0p24_a3p85379_t3p60021 | loss3 | loss2 | 0.0603119 | loss3 | loss2 | 0.0306249 | loss3 | loss2 | 0.0306249 |
| 06_rectangles_frac0p2_a1p44134_t6p32822 | loss3 | random_clean | 0.412423 | loss3 | random_clean | 0.233464 | loss3 | random_clean | 0.233464 |
| 07_cellular_blobs_frac0p16_a1p97385_t4p60937 | loss3 | loss2 | 0.516788 | loss3 | loss2 | 0.304865 | loss3 | loss2 | 0.304865 |
| 08_matern_smooth_frac0p16_a3p55666_t2p38988 | loss3 | physics | 0.662878 | loss3 | physics | 0.419378 | loss3 | physics | 0.419378 |
| 09_matern_fine_frac0p12_a1p37232_t13p3592 | loss3 | loss2 | 0.604688 | loss3 | loss2 | 0.371262 | loss3 | loss2 | 0.371262 |
| 10_highpass_grf_frac0p24_a1p11037_t11p7271 | random_clean | loss2 | 0.203537 | random_clean | loss2 | 0.107552 | random_clean | loss2 | 0.107552 |
| 11_bandpass_grf_frac0p2_a1p80507_t4p09153 | loss3 | random_clean | 0.189112 | loss3 | random_clean | 0.099507 | loss3 | random_clean | 0.099507 |
| 12_wave_mix_frac0p16_a2p26209_t4p32867 | loss3 | random_clean | 0.508555 | loss3 | random_clean | 0.298969 | loss3 | random_clean | 0.298969 |
| 13_blocky_tiles_frac0p12_a2p00536_t4p79207 | loss3 | loss2 | 0.533155 | loss3 | loss2 | 0.316739 | loss3 | loss2 | 0.316739 |
| 14_rectangles_frac0p24_a1p67693_t9p58326 | loss3 | loss2 | 0.578447 | loss3 | loss2 | 0.350729 | loss3 | loss2 | 0.350729 |
| 15_cellular_blobs_frac0p2_a3p27206_t2p59767 | loss3 | loss2 | 0.558627 | loss3 | loss2 | 0.335641 | loss3 | loss2 | 0.335641 |
| 16_matern_smooth_frac0p2_a3p63184_t2p95736 | loss3 | baseline | 0.615583 | loss3 | baseline | 0.379986 | loss3 | baseline | 0.379986 |
| 17_matern_fine_frac0p16_a1p05208_t9p11004 | loss3 | random_clean | 0.464306 | loss3 | random_clean | 0.268089 | loss3 | random_clean | 0.268089 |
| 18_highpass_grf_frac0p12_a1p26489_t6p96292 | loss3 | random_clean | 0.599281 | loss3 | random_clean | 0.366977 | loss3 | random_clean | 0.366977 |
| 19_bandpass_grf_frac0p24_a2p18225_t4p94437 | loss3 | loss2 | 0.422709 | loss3 | loss2 | 0.240203 | loss3 | loss2 | 0.240203 |
| 20_wave_mix_frac0p2_a2p69422_t4p60678 | loss3 | random_clean | 0.472662 | loss3 | random_clean | 0.273819 | loss3 | random_clean | 0.273819 |
| 21_blocky_tiles_frac0p16_a3p29541_t4p96638 | loss3 | loss2 | 0.512421 | loss3 | loss2 | 0.301731 | loss3 | loss2 | 0.301731 |
| 22_rectangles_frac0p12_a1p43234_t10p2242 | loss3 | random_clean | 0.64949 | loss3 | random_clean | 0.407962 | loss3 | random_clean | 0.407962 |
| 23_cellular_blobs_frac0p24_a2p77492_t4p18018 | loss3 | loss2 | 0.383843 | loss3 | loss2 | 0.215043 | loss3 | loss2 | 0.215043 |
| 24_matern_smooth_frac0p24_a4p16883_t1p66293 | loss3 | loss1 | 0.629259 | loss3 | loss1 | 0.391115 | loss3 | loss1 | 0.391115 |
| 25_matern_fine_frac0p2_a1p32486_t12p8246 | loss3 | loss2 | 0.407569 | loss3 | loss2 | 0.230305 | loss3 | loss2 | 0.230305 |
| 26_highpass_grf_frac0p16_a1p59204_t12p8363 | loss3 | random_clean | 0.521254 | loss3 | random_clean | 0.308085 | loss3 | random_clean | 0.308085 |
| 27_bandpass_grf_frac0p12_a1p8965_t7p66724 | loss3 | random_clean | 0.563753 | loss3 | random_clean | 0.33951 | loss3 | random_clean | 0.33951 |
| 28_wave_mix_frac0p24_a1p71757_t2p96649 | loss3 | random_clean | 0.221626 | loss3 | random_clean | 0.117745 | loss3 | random_clean | 0.117745 |
| 29_blocky_tiles_frac0p2_a2p06162_t2p29776 | loss3 | loss2 | 0.401809 | loss3 | loss2 | 0.226572 | loss3 | loss2 | 0.226572 |
| 30_rectangles_frac0p16_a1p24225_t5p53206 | loss3 | random_clean | 0.551679 | loss3 | random_clean | 0.330432 | loss3 | random_clean | 0.330432 |
| 31_cellular_blobs_frac0p12_a2p19293_t5p77748 | loss3 | loss2 | 0.529144 | loss3 | loss2 | 0.313811 | loss3 | loss2 | 0.313811 |
| 32_matern_smooth_frac0p12_a4p06817_t2p8029 | loss3 | loss2 | 0.689311 | loss3 | loss2 | 0.442605 | loss3 | loss2 | 0.442605 |
| 33_matern_fine_frac0p24_a1p77702_t7p67384 | loss3 | loss2 | 0.21822 | loss3 | loss2 | 0.115817 | loss3 | loss2 | 0.115817 |
| 34_highpass_grf_frac0p2_a1p38501_t10p2829 | loss3 | random_clean | 0.261217 | loss3 | random_clean | 0.140475 | loss3 | random_clean | 0.140475 |
| 35_bandpass_grf_frac0p16_a1p27246_t5p44182 | loss3 | random_clean | 0.491323 | loss3 | random_clean | 0.286784 | loss3 | random_clean | 0.286784 |
| 36_wave_mix_frac0p12_a1p72323_t5p41614 | loss3 | loss2 | 0.534392 | loss3 | loss2 | 0.317645 | loss3 | loss2 | 0.317645 |
| 37_blocky_tiles_frac0p24_a2p23756_t3p79868 | loss3 | random_clean | 0.324179 | loss3 | random_clean | 0.177917 | loss3 | random_clean | 0.177917 |
| 38_rectangles_frac0p2_a1p49751_t8p36934 | loss3 | loss2 | 0.603823 | loss3 | loss2 | 0.370574 | loss3 | loss2 | 0.370574 |
| 39_cellular_blobs_frac0p16_a3p47176_t5p78996 | loss3 | loss2 | 0.538401 | loss3 | loss2 | 0.320589 | loss3 | loss2 | 0.320589 |
| 40_matern_smooth_frac0p16_a4p67408_t1p27431 | loss3 | baseline | 0.732712 | loss3 | baseline | 0.483001 | loss3 | baseline | 0.483001 |
| 41_matern_fine_frac0p12_a1p13343_t8p3613 | loss3 | loss2 | 0.57649 | loss3 | loss2 | 0.349224 | loss3 | loss2 | 0.349224 |
| 42_highpass_grf_frac0p24_a1p18205_t11p6872 | random_clean | loss2 | 0.208372 | random_clean | loss2 | 0.110265 | random_clean | loss2 | 0.110265 |
| 43_bandpass_grf_frac0p2_a1p22554_t2p62907 | loss3 | loss2 | 0.5332 | loss3 | loss2 | 0.316772 | loss3 | loss2 | 0.316772 |
| 44_wave_mix_frac0p16_a1p82574_t5p31661 | loss3 | loss2 | 0.541988 | loss3 | loss2 | 0.323234 | loss3 | loss2 | 0.323234 |
| 45_blocky_tiles_frac0p12_a2p72728_t2p09368 | loss3 | loss2 | 0.504412 | loss3 | loss2 | 0.29602 | loss3 | loss2 | 0.29602 |
| 46_rectangles_frac0p24_a1p57209_t7p6296 | loss3 | loss2 | 0.495651 | loss3 | loss2 | 0.289824 | loss3 | loss2 | 0.289824 |
| 47_cellular_blobs_frac0p2_a3p35513_t6p18204 | loss3 | loss2 | 0.455651 | loss3 | loss2 | 0.2622 | loss3 | loss2 | 0.2622 |
| 48_matern_smooth_frac0p2_a3p3086_t2p22775 | loss3 | baseline | 0.689675 | loss3 | baseline | 0.442932 | loss3 | baseline | 0.442932 |
| 49_matern_fine_frac0p16_a1p39654_t9p9182 | loss3 | loss2 | 0.554955 | loss3 | loss2 | 0.332883 | loss3 | loss2 | 0.332883 |

## Files

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean52_winner_tables_20260615/clean52_dataset_winners_wide.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean52_winner_tables_20260615/clean52_metric_winners_long.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean52_winner_tables_20260615/clean52_model_mean_by_split_metric.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/clean52_winner_tables_20260615/clean52_winner_summary_by_split_metric.csv`