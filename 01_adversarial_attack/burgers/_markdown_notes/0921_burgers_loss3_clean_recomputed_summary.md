# Clean Recomputed Burgers Attack Summary

This report is rebuilt from per-index `true_loss_summary.csv` files and does not trust existing ratio-summary `model` fields.

- parameter groups: `80`
- model-evidence mismatch samples: `0`

## Overview

| model | nu | norm | eps | alpha | n | best | best ratio | best wins | loss3 ratio | loss3 wins | root |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepONet | 0.01 | 2 | 8 | 0.15 | 1 | loss3 | 1749.62x | 1/1 | 1749.62x | 1/1 | `burgers_deeponet_corrected_oldstyle_5loss_nu0p01` |
| DeepONet | 0.01 | inf | 0.5 | 0.02 | 1 | loss3 | 2699.75x | 1/1 | 2699.75x | 1/1 | `burgers_deeponet_corrected_oldstyle_5loss_nu0p01` |
| DeepONet | 0.01 | 2 | 0.5 | 0.009375 | 1 | loss3 | 73.2504x | 1/1 | 73.2504x | 1/1 | `burgers_deeponet_corrected_oldstyle_5loss_nu0p01_one16` |
| DeepONet | 0.01 | inf | 0.03125 | 0.00125 | 1 | loss2_dict_N2000 | 173.782x | 1/1 | 158.682x | 0/1 | `burgers_deeponet_corrected_oldstyle_5loss_nu0p01_one16` |
| DeepONet | 0.01 | 2 | 2 | 0.0375 | 1 | loss2_dict_N20000 | 618.901x | 1/1 | 453.788x | 0/1 | `burgers_deeponet_corrected_oldstyle_5loss_nu0p01_quarter` |
| DeepONet | 0.01 | inf | 0.125 | 0.005 | 1 | loss2_dict_N20000 | 1054.32x | 1/1 | 749.611x | 0/1 | `burgers_deeponet_corrected_oldstyle_5loss_nu0p01_quarter` |
| DeepONet | 0.01 | 2 | 4 | 0.075 | 50 | loss3 | 1808.74x | 42/50 | 1808.74x | 42/50 | `burgers_deeponet_nu0p01_l2_eps4_alpha0p075_batch50_random50_loss3_candidate` |
| DeepONet | 0.01 | inf | 0.25 | 0.01 | 50 | loss3 | 2883.11x | 44/50 | 2883.11x | 44/50 | `burgers_deeponet_nu0p01_linf_eps0p25_alpha0p01_batch50_random50_loss3_candidate` |
| DeepONet | 0.01 | 2 | 0.5 | 0.009 | 100 | loss3 | 106.826x | 89/100 | 106.826x | 89/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_l2_eps0p5_alpha0p009_batch100_seed20260511` |
| DeepONet | 0.01 | 2 | 12 | 0.225 | 100 | loss3 | 6196.79x | 96/100 | 6196.79x | 96/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_l2_eps12p0_alpha0p225_batch100_seed20260511` |
| DeepONet | 0.01 | 2 | 1 | 0.01875 | 100 | loss3 | 365.275x | 82/100 | 365.275x | 82/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_l2_eps1p0_alpha0p01875_batch100_seed20260511` |
| DeepONet | 0.01 | 2 | 2 | 0.0375 | 100 | loss3 | 1001.16x | 83/100 | 1001.16x | 83/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_l2_eps2p0_alpha0p0375_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.025 | 0.001 | 100 | loss3 | 173.521x | 80/100 | 173.521x | 80/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_linf_eps0p025_alpha0p001_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.05 | 0.002 | 100 | loss3 | 555.677x | 78/100 | 555.677x | 78/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_linf_eps0p05_alpha0p002_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.075 | 0.003 | 100 | loss3 | 978.541x | 77/100 | 978.541x | 77/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_linf_eps0p075_alpha0p003_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.75 | 0.03 | 100 | loss3 | 1.13e+04x | 100/100 | 1.13e+04x | 100/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_linf_eps0p75_alpha0p03_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 1 | 0.04 | 100 | loss3 | 1.495e+04x | 100/100 | 1.495e+04x | 100/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/deeponet_nu0p01_linf_eps1p0_alpha0p04_batch100_seed20260511` |
| DeepONet | 0.01 | 2 | 1 | 0.01875 | 100 | loss3 | 365.275x | 82/100 | 365.275x | 82/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/deeponet_nu0p01_l2_eps1p0_alpha0p01875_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.05 | 0.002 | 100 | loss3 | 555.677x | 78/100 | 555.677x | 78/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/deeponet_nu0p01_linf_eps0p05_alpha0p002_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.75 | 0.03 | 100 | loss3 | 1.13e+04x | 100/100 | 1.13e+04x | 100/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/deeponet_nu0p01_linf_eps0p75_alpha0p03_batch100_seed20260511` |
| DeepONet | 0.01 | 2 | 4 | 0.075 | 100 | loss3 | 2019.56x | 87/100 | 2019.56x | 87/100 | `burgers_loss3_candidate_batch100_random100/deeponet_nu0p01_l2_eps4_alpha0p075_batch100_seed20260510` |
| DeepONet | 0.01 | 2 | 6 | 0.1125 | 100 | loss3 | 2939.69x | 92/100 | 2939.69x | 92/100 | `burgers_loss3_candidate_batch100_random100/deeponet_nu0p01_l2_eps6_alpha0p1125_batch100_seed20260510` |
| DeepONet | 0.01 | inf | 0.25 | 0.01 | 100 | loss3 | 3247.43x | 89/100 | 3247.43x | 89/100 | `burgers_loss3_candidate_batch100_random100/deeponet_nu0p01_linf_eps0p25_alpha0p01_batch100_seed20260510` |
| DeepONet | 0.01 | inf | 0.375 | 0.015 | 100 | loss3 | 4985.15x | 95/100 | 4985.15x | 95/100 | `burgers_loss3_candidate_batch100_random100/deeponet_nu0p01_linf_eps0p375_alpha0p015_batch100_seed20260510` |
| DeepONet | 0.01 | 2 | 7.5 | 0.14 | 100 | loss3 | 3796.15x | 93/100 | 3796.15x | 93/100 | `burgers_loss3_good_bad_7settings_batch100_random100_losses/deeponet_nu0p01_l2_eps7p5_alpha0p14_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.125 | 0.005 | 100 | loss3 | 1744.32x | 83/100 | 1744.32x | 83/100 | `burgers_loss3_good_bad_7settings_batch100_random100_losses/deeponet_nu0p01_linf_eps0p125_alpha0p005_batch100_seed20260511` |
| DeepONet | 0.01 | inf | 0.45 | 0.018 | 100 | loss3 | 6413.15x | 96/100 | 6413.15x | 96/100 | `burgers_loss3_good_bad_7settings_batch100_random100_losses/deeponet_nu0p01_linf_eps0p45_alpha0p018_batch100_seed20260511` |
| FNO | 0.001 | 2 | 8 | 0.15 | 5 | loss2_dict_N20000 | 298.394x | 1/5 | 263.71x | 2/5 | `burgers_corrected_oldstyle_5loss_clean_rerun` |
| FNO | 0.001 | inf | 0.5 | 0.02 | 5 | loss2_dict_N2000 | 572.17x | 2/5 | 412.759x | 1/5 | `burgers_corrected_oldstyle_5loss_clean_rerun` |
| FNO | 0.01 | 2 | 8 | 0.15 | 5 | loss3 | 1318.66x | 3/5 | 1318.66x | 3/5 | `burgers_corrected_oldstyle_5loss_clean_rerun` |
| FNO | 0.01 | inf | 0.5 | 0.02 | 5 | loss3 | 3139.18x | 5/5 | 3139.18x | 5/5 | `burgers_corrected_oldstyle_5loss_clean_rerun` |
| FNO | 0.001 | 2 | 32 | 0.6 | 5 | loss2_dict_N20000 | 1.769e+04x | 2/5 | 1.715e+04x | 0/5 | `burgers_corrected_oldstyle_5loss_nu0p001_4x` |
| FNO | 0.001 | inf | 2 | 0.08 | 5 | loss2_dict_N2000 | 2.125e+04x | 1/5 | 1.497e+04x | 0/5 | `burgers_corrected_oldstyle_5loss_nu0p001_4x` |
| FNO | 0.001 | inf | 0.01 | 0.15 | 3 | loss3 | 1.38943x | 3/3 | 1.38943x | 3/3 | `burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only` |
| FNO | 0.001 | inf | 0.1 | 0.01 | 3 | loss3 | 13.9799x | 3/3 | 13.9799x | 3/3 | `burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only` |
| FNO | 0.01 | 2 | 4 | 0.075 | 50 | loss3 | 94.2174x | 40/50 | 94.2174x | 40/50 | `burgers_fno_nu0p01_l2_eps4_alpha0p075_batch50_random50_loss3_candidate` |
| FNO | 0.01 | inf | 0.25 | 0.01 | 50 | loss3 | 127.638x | 39/50 | 127.638x | 39/50 | `burgers_fno_nu0p01_linf_eps0p25_alpha0p01_batch50_random50_loss3_candidate` |
| FNO | 0.01 | 2 | 10 | 0.1875 | 100 | loss3 | 3542.63x | 84/100 | 3542.63x | 84/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps10p0_alpha0p1875_batch100_seed20260511` |
| FNO | 0.01 | 2 | 12 | 0.225 | 100 | loss3 | 6471.81x | 82/100 | 6471.81x | 82/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps12p0_alpha0p225_batch100_seed20260511` |
| FNO | 0.01 | 2 | 16 | 0.3 | 100 | loss3 | 1.501e+04x | 83/100 | 1.501e+04x | 83/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps16p0_alpha0p30_batch100_seed20260511` |
| FNO | 0.01 | 2 | 2 | 0.0375 | 100 | loss3 | 7.46757x | 94/100 | 7.46757x | 94/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps2p0_alpha0p0375_batch100_seed20260511` |
| FNO | 0.01 | 2 | 8 | 0.15 | 100 | loss3 | 1576.58x | 82/100 | 1576.58x | 82/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps8p0_alpha0p15_batch100_seed20260511` |
| FNO | 0.01 | 2 | 9 | 0.09 | 100 | loss3 | 1627.24x | 71/100 | 1627.24x | 71/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps9p0_alpha0p09_batch100_seed20260511` |
| FNO | 0.01 | 2 | 9 | 0.17 | 100 | loss3 | 2443.39x | 82/100 | 2443.39x | 82/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps9p0_alpha0p17_batch100_seed20260511` |
| FNO | 0.01 | 2 | 9 | 0.3 | 100 | loss3 | 2720.25x | 88/100 | 2720.25x | 88/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_l2_eps9p0_alpha0p30_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.2 | 0.008 | 100 | loss3 | 46.7879x | 85/100 | 46.7879x | 85/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_linf_eps0p2_alpha0p008_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.3 | 0.003 | 100 | loss3 | 179.77x | 51/100 | 179.77x | 51/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_linf_eps0p3_alpha0p003_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.3 | 0.03 | 100 | loss3 | 278.36x | 83/100 | 278.36x | 83/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_linf_eps0p3_alpha0p03_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.45 | 0.018 | 100 | loss3 | 1770.01x | 91/100 | 1770.01x | 91/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_linf_eps0p45_alpha0p018_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.75 | 0.03 | 100 | loss3 | 1.298e+04x | 96/100 | 1.298e+04x | 96/100 | `burgers_future_loss3_bad_search_batch100_random100_sequential/fno_nu0p01_linf_eps0p75_alpha0p03_batch100_seed20260511` |
| FNO | 0.001 | inf | 0.15 | 0.015 | 100 | loss3 | 34.9612x | 80/100 | 34.9612x | 80/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p001_linf_eps0p15_alpha0p015_batch100_seed20260511` |
| FNO | 0.001 | inf | 0.25 | 0.025 | 100 | loss3 | 107.54x | 43/100 | 107.54x | 43/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p001_linf_eps0p25_alpha0p025_batch100_seed20260511` |
| FNO | 0.001 | inf | 0.35 | 0.035 | 100 | loss3 | 243.612x | 36/100 | 243.612x | 36/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p001_linf_eps0p35_alpha0p035_batch100_seed20260511` |
| FNO | 0.001 | inf | 0.5 | 0.005 | 100 | loss2_fixed | 417.041x | 38/100 | 325x | 22/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p001_linf_eps0p5_alpha0p005_batch100_seed20260511` |
| FNO | 0.001 | inf | 0.5 | 0.08 | 100 | loss3 | 672.908x | 52/100 | 672.908x | 52/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p001_linf_eps0p5_alpha0p08_batch100_seed20260511` |
| FNO | 0.001 | inf | 1 | 0.04 | 100 | loss3 | 3565.66x | 46/100 | 3565.66x | 46/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p001_linf_eps1p0_alpha0p04_batch100_seed20260511` |
| FNO | 0.01 | 2 | 2 | 0.0375 | 100 | loss3 | 7.46757x | 94/100 | 7.46757x | 94/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_l2_eps2p0_alpha0p0375_batch100_seed20260511` |
| FNO | 0.01 | 2 | 9 | 0.17 | 100 | loss3 | 2443.39x | 82/100 | 2443.39x | 82/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_l2_eps9p0_alpha0p17_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.15 | 0.006 | 100 | loss3 | 16.0929x | 95/100 | 16.0929x | 95/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.2 | 0.008 | 100 | loss3 | 46.7879x | 85/100 | 46.7879x | 85/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p2_alpha0p008_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.3 | 0.003 | 100 | loss3 | 179.77x | 51/100 | 179.77x | 51/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p3_alpha0p003_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.3 | 0.03 | 100 | loss3 | 278.36x | 83/100 | 278.36x | 83/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p3_alpha0p03_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.45 | 0.018 | 100 | loss3 | 1770.01x | 91/100 | 1770.01x | 91/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p45_alpha0p018_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.75 | 0.03 | 100 | loss3 | 1.298e+04x | 96/100 | 1.298e+04x | 96/100 | `burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p75_alpha0p03_batch100_seed20260511` |
| FNO | 0.001 | inf | 0.05 | 0.005 | 100 | loss3 | 5.22121x | 99/100 | 5.22121x | 99/100 | `burgers_loss3_candidate_batch100_random100/fno_nu0p001_linf_eps0p05_alpha0p005_batch100_seed20260510` |
| FNO | 0.001 | inf | 0.1 | 0.01 | 100 | loss3 | 15.9765x | 94/100 | 15.9765x | 94/100 | `burgers_loss3_candidate_batch100_random100/fno_nu0p001_linf_eps0p1_alpha0p01_batch100_seed20260510` |
| FNO | 0.001 | inf | 0.2 | 0.02 | 100 | loss3 | 66.7127x | 56/100 | 66.7127x | 56/100 | `burgers_loss3_candidate_batch100_random100/fno_nu0p001_linf_eps0p2_alpha0p02_batch100_seed20260510` |
| FNO | 0.01 | 2 | 4 | 0.075 | 100 | loss3 | 90.2389x | 84/100 | 90.2389x | 84/100 | `burgers_loss3_candidate_batch100_random100/fno_nu0p01_l2_eps4_alpha0p075_batch100_seed20260510` |
| FNO | 0.01 | 2 | 6 | 0.1125 | 100 | loss3 | 487.218x | 83/100 | 487.218x | 83/100 | `burgers_loss3_candidate_batch100_random100/fno_nu0p01_l2_eps6_alpha0p1125_batch100_seed20260510` |
| FNO | 0.01 | inf | 0.25 | 0.01 | 100 | loss3 | 128.678x | 83/100 | 128.678x | 83/100 | `burgers_loss3_candidate_batch100_random100/fno_nu0p01_linf_eps0p25_alpha0p01_batch100_seed20260510` |
| FNO | 0.01 | inf | 0.375 | 0.015 | 100 | loss3 | 824.424x | 86/100 | 824.424x | 86/100 | `burgers_loss3_candidate_batch100_random100/fno_nu0p01_linf_eps0p375_alpha0p015_batch100_seed20260510` |
| FNO | 0.01 | 2 | 1 | 0.01875 | 100 | loss3 | 2.06243x | 99/100 | 2.06243x | 99/100 | `burgers_loss3_fno_nu0p01_eps_sweep_plus_deeponet_batch100_random100_losses/fno_nu0p01_l2_eps1p0_alpha0p01875_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.05 | 0.002 | 100 | loss3 | 2.00295x | 98/100 | 2.00295x | 98/100 | `burgers_loss3_fno_nu0p01_eps_sweep_plus_deeponet_batch100_random100_losses/fno_nu0p01_linf_eps0p05_alpha0p002_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.1 | 0.004 | 100 | loss3 | 5.27062x | 96/100 | 5.27062x | 96/100 | `burgers_loss3_fno_nu0p01_eps_sweep_plus_deeponet_batch100_random100_losses/fno_nu0p01_linf_eps0p10_alpha0p004_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.6 | 0.024 | 100 | loss3 | 5835.27x | 96/100 | 5835.27x | 96/100 | `burgers_loss3_fno_nu0p01_eps_sweep_plus_deeponet_batch100_random100_losses/fno_nu0p01_linf_eps0p60_alpha0p024_batch100_seed20260511` |
| FNO | 0.01 | inf | 1 | 0.04 | 100 | loss3 | 3.092e+04x | 91/100 | 3.092e+04x | 91/100 | `burgers_loss3_fno_nu0p01_eps_sweep_plus_deeponet_batch100_random100_losses/fno_nu0p01_linf_eps1p00_alpha0p040_batch100_seed20260511` |
| FNO | 0.001 | inf | 0.5 | 0.02 | 100 | loss2_dict_N200 | 584.96x | 26/100 | 551.847x | 32/100 | `burgers_loss3_good_bad_7settings_batch100_random100_losses/fno_nu0p001_linf_eps0p5_alpha0p02_batch100_seed20260511` |
| FNO | 0.001 | inf | 2 | 0.08 | 100 | loss3 | 2.683e+04x | 22/100 | 2.683e+04x | 22/100 | `burgers_loss3_good_bad_7settings_batch100_random100_losses/fno_nu0p001_linf_eps2p0_alpha0p08_batch100_seed20260511` |
| FNO | 0.01 | 2 | 7 | 0.13 | 100 | loss3 | 935.193x | 80/100 | 935.193x | 80/100 | `burgers_loss3_good_bad_7settings_batch100_random100_losses/fno_nu0p01_l2_eps7p0_alpha0p13_batch100_seed20260511` |
| FNO | 0.01 | inf | 0.3 | 0.012 | 100 | loss3 | 274.993x | 82/100 | 274.993x | 82/100 | `burgers_loss3_good_bad_7settings_batch100_random100_losses/fno_nu0p01_linf_eps0p30_alpha0p012_batch100_seed20260511` |

## Model Evidence Mismatches

No model-evidence mismatches detected.

## DeepONet Method Ratios With Standard Deviation

Each method cell is `wins; mean ratio +/- std ratio`.
Rows are de-duplicated by `(model, nu, norm, epsilon, alpha, steps, index)`, so older batch50 or duplicate parent-directory runs do not inflate `n`.

### DeepONet nu=0.01 L2

| eps | alpha | n | loss1 | loss2_fixed | N200 | N2000 | N20000 | loss3_sg | loss3 | best |
|---:|---:|---:|---|---|---|---|---|---|---|---|
| 0.500 | 0.009 | 100 | 0/100; 65.03 +/- 27.68x | 0/100; 103.4 +/- 51.19x | 3/100; 77.23 +/- 44.23x | 4/100; 79.47 +/- 44.08x | 4/100; 82.74 +/- 49.66x | 0/100; 103.3 +/- 51.37x | 89/100; 106.8 +/- 52.90x | loss3; 89/100; 106.8 +/- 52.90x |
| 0.500 | 0.009 | 1 | 0/1; 50.21 +/- 0.000x | 0/1; 70.63 +/- 0.000x | 0/1; 46.01 +/- 0.000x | 0/1; 22.68 +/- 0.000x | 0/1; 22.68 +/- 0.000x | 0/1; 70.87 +/- 0.000x | 1/1; 73.25 +/- 0.000x | loss3; 1/1; 73.25 +/- 0.000x |
| 1.000 | 0.019 | 100 | 0/100; 136.4 +/- 49.33x | 1/100; 351.2 +/- 180.2x | 3/100; 282.4 +/- 165.2x | 8/100; 293.9 +/- 168.1x | 6/100; 300.2 +/- 176.4x | 0/100; 351.1 +/- 180.7x | 82/100; 365.3 +/- 186.8x | loss3; 82/100; 365.3 +/- 186.8x |
| 2.000 | 0.037 | 100 | 0/100; 159.7 +/- 55.81x | 0/100; 935.0 +/- 487.0x | 5/100; 762.1 +/- 465.4x | 7/100; 797.1 +/- 475.6x | 5/100; 821.0 +/- 482.6x | 0/100; 940.3 +/- 486.2x | 83/100; 1001 +/- 517.4x | loss3; 83/100; 1001 +/- 517.4x |
| 4.000 | 0.075 | 100 | 0/100; 105.9 +/- 37.34x | 0/100; 1643 +/- 913.6x | 2/100; 1429 +/- 930.2x | 7/100; 1473 +/- 946.8x | 4/100; 1512 +/- 943.3x | 0/100; 1659 +/- 911.3x | 87/100; 2020 +/- 1164x | loss3; 87/100; 2020 +/- 1164x |
| 6.000 | 0.113 | 100 | 0/100; 72.13 +/- 25.33x | 0/100; 2058 +/- 1161x | 1/100; 1802 +/- 1164x | 3/100; 1868 +/- 1186x | 4/100; 1925 +/- 1190x | 0/100; 2085 +/- 1152x | 92/100; 2940 +/- 1712x | loss3; 92/100; 2940 +/- 1712x |
| 7.500 | 0.140 | 100 | 0/100; 60.29 +/- 20.73x | 0/100; 2443 +/- 1279x | 1/100; 2031 +/- 1247x | 4/100; 2144 +/- 1290x | 2/100; 2165 +/- 1265x | 0/100; 2464 +/- 1276x | 93/100; 3796 +/- 1965x | loss3; 93/100; 3796 +/- 1965x |
| 8.000 | 0.150 | 1 | 0/1; 45.20 +/- 0.000x | 0/1; 1235 +/- 0.000x | 0/1; 1068 +/- 0.000x | 0/1; 708.3 +/- 0.000x | 0/1; 639.0 +/- 0.000x | 0/1; 1250 +/- 0.000x | 1/1; 1750 +/- 0.000x | loss3; 1/1; 1750 +/- 0.000x |
| 12.00 | 0.225 | 100 | 0/100; 41.20 +/- 13.40x | 0/100; 2894 +/- 1521x | 0/100; 2409 +/- 1511x | 3/100; 2535 +/- 1551x | 1/100; 2549 +/- 1520x | 0/100; 2947 +/- 1531x | 96/100; 6197 +/- 3234x | loss3; 96/100; 6197 +/- 3234x |

### DeepONet nu=0.01 Linf

| eps | alpha | n | loss1 | loss2_fixed | N200 | N2000 | N20000 | loss3_sg | loss3 | best |
|---:|---:|---:|---|---|---|---|---|---|---|---|
| 0.025 | 0.001 | 100 | 9/100; 124.0 +/- 61.57x | 0/100; 167.3 +/- 82.36x | 1/100; 136.2 +/- 77.34x | 4/100; 140.6 +/- 78.15x | 6/100; 145.9 +/- 81.86x | 0/100; 167.7 +/- 82.24x | 80/100; 173.5 +/- 84.73x | loss3; 80/100; 173.5 +/- 84.73x |
| 0.031 | 0.001 | 1 | 0/1; 156.2 +/- 0.000x | 0/1; 152.5 +/- 0.000x | 0/1; 91.19 +/- 0.000x | 1/1; 173.8 +/- 0.000x | 0/1; 173.8 +/- 0.000x | 0/1; 152.2 +/- 0.000x | 0/1; 158.7 +/- 0.000x | loss2_dict_N2000; 1/1; 173.8 +/- 0.000x |
| 0.050 | 0.002 | 100 | 2/100; 196.7 +/- 83.23x | 0/100; 530.2 +/- 270.4x | 5/100; 444.5 +/- 264.3x | 9/100; 466.6 +/- 271.5x | 6/100; 480.3 +/- 277.2x | 0/100; 532.7 +/- 269.9x | 78/100; 555.7 +/- 280.1x | loss3; 78/100; 555.7 +/- 280.1x |
| 0.075 | 0.003 | 100 | 1/100; 173.1 +/- 64.24x | 0/100; 918.1 +/- 471.2x | 6/100; 765.2 +/- 458.3x | 8/100; 804.0 +/- 480.9x | 8/100; 832.3 +/- 490.2x | 0/100; 922.3 +/- 470.8x | 77/100; 978.5 +/- 498.1x | loss3; 77/100; 978.5 +/- 498.1x |
| 0.125 | 0.005 | 100 | 0/100; 104.5 +/- 29.62x | 0/100; 1531 +/- 792.2x | 2/100; 1287 +/- 778.2x | 5/100; 1339 +/- 806.4x | 10/100; 1377 +/- 803.6x | 0/100; 1540 +/- 786.2x | 83/100; 1744 +/- 893.0x | loss3; 83/100; 1744 +/- 893.0x |
| 0.250 | 0.010 | 100 | 0/100; 38.50 +/- 11.58x | 0/100; 2348 +/- 1347x | 0/100; 2027 +/- 1317x | 5/100; 2128 +/- 1340x | 6/100; 2185 +/- 1353x | 0/100; 2349 +/- 1338x | 89/100; 3247 +/- 1909x | loss3; 89/100; 3247 +/- 1909x |
| 0.375 | 0.015 | 100 | 0/100; 20.99 +/- 9.015x | 0/100; 2825 +/- 1665x | 0/100; 2428 +/- 1644x | 3/100; 2605 +/- 1659x | 2/100; 2688 +/- 1703x | 0/100; 2860 +/- 1741x | 95/100; 4985 +/- 2955x | loss3; 95/100; 4985 +/- 2955x |
| 0.450 | 0.018 | 100 | 0/100; 16.44 +/- 10.64x | 0/100; 3113 +/- 1595x | 0/100; 2680 +/- 1593x | 3/100; 2787 +/- 1593x | 1/100; 2874 +/- 1622x | 0/100; 3162 +/- 1577x | 96/100; 6413 +/- 3370x | loss3; 96/100; 6413 +/- 3370x |
| 0.500 | 0.020 | 1 | 0/1; 6.801 +/- 0.000x | 0/1; 1525 +/- 0.000x | 0/1; 1162 +/- 0.000x | 0/1; 1275 +/- 0.000x | 0/1; 1080 +/- 0.000x | 0/1; 1629 +/- 0.000x | 1/1; 2700 +/- 0.000x | loss3; 1/1; 2700 +/- 0.000x |
| 0.750 | 0.030 | 100 | 0/100; 7.105 +/- 4.534x | 0/100; 3342 +/- 1776x | 0/100; 3000 +/- 1781x | 0/100; 3065 +/- 1879x | 0/100; 3180 +/- 1862x | 0/100; 3494 +/- 1827x | 100/100; 1.13e+04 +/- 6022x | loss3; 100/100; 1.13e+04 +/- 6022x |
| 1.000 | 0.040 | 100 | 0/100; 5.645 +/- 5.924x | 0/100; 3461 +/- 1912x | 0/100; 3176 +/- 1979x | 0/100; 3270 +/- 1964x | 0/100; 3280 +/- 1846x | 0/100; 3689 +/- 1908x | 100/100; 1.495e+04 +/- 7826x | loss3; 100/100; 1.495e+04 +/- 7826x |
