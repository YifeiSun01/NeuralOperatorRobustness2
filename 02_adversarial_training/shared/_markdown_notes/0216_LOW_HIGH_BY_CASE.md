# Loss position by generated case

Diagnostics were recomputed with `max_samples=50`; train/test shaded bands use +/-1 sample RMSE std from those 50 diagnostic samples. Bar heights use full evaluator RMSE from `generalization_eval_repaired_all/metrics.csv`.

## Counts

| task | n | rmse_below_train | rmse_between_train_test | rmse_above_or_equal_test | rel_l2_below_train | rel_l2_between_train_test | rel_l2_above_or_equal_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| burgers | 50 | 0 | 2 | 48 | 0 | 1 | 49 |
| darcy | 50 | 9 | 4 | 37 | 9 | 4 | 37 |
| ns2d | 49 | 21 | 10 | 18 | 21 | 10 | 18 |

## Train/Test Reference Std

| task | split | case_label | rmse_mean_line | rmse_std_band_sample_50 | relative_l2_mean | relative_l2_sample_std_sample_50 |
| --- | --- | --- | --- | --- | --- | --- |
| burgers | test | Burgers test set | 0.009544 | 0.002705 | 0.01775 | 0.004575 |
| burgers | train | Burgers train set | 0.008677 | 0.00202 | 0.01639 | 0.003185 |
| darcy | test | Darcy/C-flow test set | 0.0001552 | 5.335e-05 | 0.02279 | 0.007079 |
| darcy | train | Darcy/C-flow train set | 0.0001348 | 4.935e-05 | 0.01984 | 0.005774 |
| ns2d | test | NS2D test set | 0.1336 | 0.05989 | 0.0957 | 0.0457 |
| ns2d | train | NS2D train set | 0.07132 | 0.0182 | 0.0513 | 0.01329 |

## RMSE Below Test

### burgers

| case_label | plot_range | plot_family | rmse_position | rmse | rmse_sample_std_50 | relative_l2 | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burgers: Gaussian initial field, correlation length=0.75 | base range | gaussian | between_train_and_test | 0.009048 | 0.002533 | 0.01674 | 0.6664 | 0.001823 |
| Burgers: Gaussian initial field, correlation length=0.025 | base range | gaussian | between_train_and_test | 0.009458 | 0.001727 | 0.01817 | 0.607 | 0.002296 |

### darcy

| case_label | plot_range | plot_family | rmse_position | rmse | rmse_sample_std_50 | relative_l2 | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=5, tau=2 | coeff 3/12 | binary coefficient | below_train | 7.679e-05 | 2.32e-05 | 0.01131 | 0.0138 | 0.000207 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=6, tau=1.5 | coeff 3/12 | binary coefficient | below_train | 7.721e-05 | 2.071e-05 | 0.01136 | 0.01386 | 0.0002093 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=3.5, tau=3 | coeff 3/12 | binary coefficient | below_train | 0.0001038 | 5.34e-05 | 0.01522 | 0.01384 | 0.0002102 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=4, tau=4 | coeff 3/12 | binary coefficient | below_train | 0.0001116 | 6.338e-05 | 0.01624 | 0.0136 | 0.0002094 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2, tau=1 | coeff 3/12 | binary coefficient | below_train | 0.0001251 | 4.755e-05 | 0.01834 | 0.01342 | 0.0002061 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2.5, tau=3.5 | coeff 3/12 | binary coefficient | below_train | 0.0001283 | 4.989e-05 | 0.01898 | 0.01339 | 0.0002079 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2, tau=2 | coeff 3/12 | binary coefficient | below_train | 0.0001304 | 6.433e-05 | 0.0194 | 0.0129 | 0.000202 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2.3, tau=3 | coeff 3/12 | binary coefficient | below_train | 0.000131 | 6.561e-05 | 0.01944 | 0.01341 | 0.000206 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=3, tau=4 | coeff 3/12 | binary coefficient | below_train | 0.0001341 | 4.638e-05 | 0.0195 | 0.01335 | 0.000204 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2.15, tau=3 | coeff 3/12 | binary coefficient | between_train_and_test | 0.0001439 | 5.567e-05 | 0.02127 | 0.01327 | 0.0002071 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=4, tau=8 | coeff 3/12 | binary coefficient | between_train_and_test | 0.0001459 | 5.62e-05 | 0.02192 | 0.01325 | 0.0002079 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=1.85, tau=3 | coeff 3/12 | binary coefficient | between_train_and_test | 0.0001485 | 4.12e-05 | 0.02215 | 0.01315 | 0.0002057 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2, tau=2.5 | coeff 3/12 | binary coefficient | between_train_and_test | 0.0001496 | 5.797e-05 | 0.02212 | 0.01327 | 0.0002093 |

### ns2d

| case_label | plot_range | plot_family | rmse_position | rmse | rmse_sample_std_50 | relative_l2 | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NS2D: Gaussian random vorticity, alpha=4, tau=4 | base range | gaussian RF alpha/tau | below_train | 0.026 | 0.00295 | 0.01782 | 5.25 | 0.04404 |
| NS2D: squared-and-centered initial field, scale=1 | base range | squared | below_train | 0.02612 | 0.003949 | 0.01779 | 5.345 | 0.04205 |
| NS2D: Gaussian random vorticity, alpha=3, tau=3 | base range | gaussian RF alpha/tau | below_train | 0.02827 | 0.002807 | 0.01941 | 5.261 | 0.04551 |
| NS2D: Gaussian random vorticity, alpha=4.5, tau=7 | base range | gaussian RF alpha/tau | below_train | 0.03012 | 0.003226 | 0.02093 | 5.284 | 0.04732 |
| NS2D: Gaussian random vorticity, alpha=5, tau=8 | base range | gaussian RF alpha/tau | below_train | 0.03086 | 0.003591 | 0.02136 | 5.289 | 0.04826 |
| NS2D: Gaussian random vorticity, alpha=6, tau=10 | base range | gaussian RF alpha/tau | below_train | 0.03146 | 0.004551 | 0.02175 | 5.294 | 0.04838 |
| NS2D: scale initial vorticity x0.5 | smaller amplitude | scale | below_train | 0.03522 | 0.00434 | 0.02443 | 5.324 | 0.04781 |
| NS2D: Gaussian random vorticity, alpha=2, tau=2 | base range | gaussian RF alpha/tau | below_train | 0.03822 | 0.005043 | 0.0266 | 5.342 | 0.04936 |
| NS2D: log-absolute centered initial field, scale=1 | base range | log-abs | below_train | 0.03832 | 0.004309 | 0.0262 | 5.338 | 0.04393 |
| NS2D: Gaussian random vorticity, alpha=3.2, tau=6 | base range | gaussian RF alpha/tau | below_train | 0.03851 | 0.00725 | 0.02681 | 5.375 | 0.05145 |
| NS2D: squared-and-centered initial field, scale=2 | larger amplitude | squared | below_train | 0.04218 | 0.01265 | 0.02892 | 5.35 | 0.04413 |
| NS2D: Gaussian random vorticity, alpha=4, tau=12 | base range | gaussian RF alpha/tau | below_train | 0.04486 | 0.00832 | 0.03149 | 5.408 | 0.05145 |
| NS2D: Gaussian random vorticity, alpha=3.5, tau=10 | base range | gaussian RF alpha/tau | below_train | 0.04562 | 0.008828 | 0.0319 | 5.429 | 0.05117 |
| NS2D: Gaussian random vorticity, alpha=5, tau=20 | base range | gaussian RF alpha/tau | below_train | 0.04634 | 0.00633 | 0.03238 | 5.382 | 0.04803 |
| NS2D: Gaussian random vorticity, alpha=2.5, tau=5 | base range | gaussian RF alpha/tau | below_train | 0.04779 | 0.009851 | 0.03363 | 5.48 | 0.05413 |
| NS2D: Gaussian random vorticity, alpha=3, tau=8 | base range | gaussian RF alpha/tau | below_train | 0.05003 | 0.01004 | 0.03511 | 5.453 | 0.05353 |
| NS2D: Gaussian random vorticity, alpha=2.8, tau=7 | base range | gaussian RF alpha/tau | below_train | 0.05276 | 0.0127 | 0.03725 | 5.463 | 0.05591 |
| NS2D: Gaussian random vorticity, alpha=2.2, tau=4 | base range | gaussian RF alpha/tau | below_train | 0.05581 | 0.01815 | 0.03935 | 5.503 | 0.05358 |
| NS2D: Gaussian random vorticity, alpha=2.5, tau=6 | base range | gaussian RF alpha/tau | below_train | 0.05694 | 0.01391 | 0.04002 | 5.553 | 0.0543 |
| NS2D: Gaussian random vorticity, alpha=2.6, tau=7 | base range | gaussian RF alpha/tau | below_train | 0.06821 | 0.02264 | 0.04813 | 5.562 | 0.05791 |
| NS2D: log-absolute centered initial field, scale=2 | larger amplitude | log-abs | below_train | 0.06989 | 0.0133 | 0.04832 | 5.456 | 0.04879 |
| NS2D: Gaussian random vorticity, alpha=2.4, tau=7 | base range | gaussian RF alpha/tau | between_train_and_test | 0.07308 | 0.02033 | 0.05163 | 5.617 | 0.05598 |
| NS2D: Gaussian random vorticity, alpha=2.5, tau=8 | base range | gaussian RF alpha/tau | between_train_and_test | 0.07313 | 0.01898 | 0.052 | 5.577 | 0.05553 |
| NS2D: add sawtooth vorticity pattern, amplitude=0.1 | sawtooth add | sawtooth | between_train_and_test | 0.07364 | 0.02267 | 0.05186 | 5.566 | 0.05674 |
| NS2D: Gaussian random vorticity, alpha=2.5, tau=9 | base range | gaussian RF alpha/tau | between_train_and_test | 0.08225 | 0.02425 | 0.05818 | 5.619 | 0.05673 |
| NS2D: Gaussian random vorticity, alpha=3, tau=20 | base range | gaussian RF alpha/tau | between_train_and_test | 0.08454 | 0.01643 | 0.05902 | 5.551 | 0.05081 |
| NS2D: Gaussian random vorticity, alpha=2.2, tau=7 | base range | gaussian RF alpha/tau | between_train_and_test | 0.08921 | 0.02539 | 0.06424 | 5.658 | 0.05981 |
| NS2D: Gaussian random vorticity, alpha=2, tau=6 | base range | gaussian RF alpha/tau | between_train_and_test | 0.1018 | 0.03029 | 0.07275 | 5.702 | 0.05887 |
| NS2D: add sawtooth vorticity pattern, amplitude=0.25 | sawtooth add | sawtooth | between_train_and_test | 0.107 | 0.03166 | 0.07584 | 5.71 | 0.05806 |
| NS2D: Gaussian random vorticity, alpha=1.8, tau=5 | base range | gaussian RF alpha/tau | between_train_and_test | 0.1153 | 0.04186 | 0.08118 | 5.75 | 0.06039 |
| NS2D: Gaussian random vorticity, alpha=2, tau=15 | base range | gaussian RF alpha/tau | between_train_and_test | 0.1308 | 0.02392 | 0.09093 | 5.772 | 0.05697 |

## Highest RMSE Per Task

### burgers

| case_label | plot_range | plot_family | rmse_position | rmse | rmse_sample_std_50 | relative_l2 | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burgers: centered amplitude x2.5, offset -0.25 | negative offset | scale + offset | above_or_equal_test | 0.6021 | 0.1168 | 1.429 | 1.021 | 0.003117 |
| Burgers: binary sign initial field, scale=1, offset=0 | binary/sign | binary sign | above_or_equal_test | 0.5921 | 0.1727 | 1.287 | 1.189 | 0.003517 |
| Burgers: centered amplitude x2, offset 0 | larger amplitude | scale + offset | above_or_equal_test | 0.3616 | 0.09846 | 1.237 | 0.9152 | 0.002953 |
| Burgers: add negative offset -1 | negative offset | offset | above_or_equal_test | 0.3376 | 0.05444 | 0.6328 | 0.6522 | 0.002392 |
| Burgers: centered amplitude x1, offset -0.4 | negative offset | scale + offset | above_or_equal_test | 0.3265 | 0.05686 | 0.7415 | 0.6426 | 0.002373 |
| Burgers: add positive offset 1 | positive offset | offset | above_or_equal_test | 0.2902 | 0.04395 | 0.1924 | 0.6524 | 0.002385 |
| Burgers: centered amplitude x2.5, offset 0.25 | positive offset | scale + offset | above_or_equal_test | 0.2897 | 0.09684 | 0.6998 | 1.014 | 0.003106 |
| Burgers: centered amplitude x1.5, offset 0 | larger amplitude | scale + offset | above_or_equal_test | 0.2787 | 0.06855 | 1.163 | 0.7955 | 0.002732 |
| Burgers: add negative offset -0.5 | negative offset | offset | above_or_equal_test | 0.1904 | 0.03883 | 1.061 | 0.6361 | 0.002349 |
| Burgers: force initial field to zero mean | zero mean | zero mean | above_or_equal_test | 0.1864 | 0.03559 | 1.153 | 0.63 | 0.002355 |
| Burgers: add positive offset 0.5 | positive offset | offset | above_or_equal_test | 0.172 | 0.03189 | 0.17 | 0.6352 | 0.002351 |
| Burgers: binary sign initial field, scale=0.5, offset=0.25 | binary/sign | binary sign | above_or_equal_test | 0.1611 | 0.05193 | 0.4305 | 0.8286 | 0.003 |

### darcy

| case_label | plot_range | plot_family | rmse_position | rmse | rmse_sample_std_50 | relative_l2 | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Darcy/C-flow: binary coefficient 1/12, GRF alpha=3, tau=1.5 | coeff 1/12 | binary coefficient | above_or_equal_test | 0.006713 | 0.001636 | 0.4797 | 0.03259 | 0.0004676 |
| Darcy/C-flow: binary coefficient 1/12, GRF alpha=3.5, tau=6 | coeff 1/12 | binary coefficient | above_or_equal_test | 0.005945 | 0.001883 | 0.4534 | 0.02834 | 0.0004186 |
| Darcy/C-flow: binary coefficient 1/12, GRF alpha=2, tau=3 | coeff 1/12 | binary coefficient | above_or_equal_test | 0.005537 | 0.001296 | 0.4319 | 0.02851 | 0.0004205 |
| Darcy/C-flow: binary coefficient 1/20, GRF alpha=2, tau=12 | coeff 1/20 | binary coefficient | above_or_equal_test | 0.003457 | 0.0008624 | 0.3967 | 0.01879 | 0.0003092 |
| Darcy/C-flow: threshold bias -0.8 | coeff 3/12 | threshold bias | above_or_equal_test | 0.002325 | 0.0002475 | 0.1723 | 0.02438 | 0.0003936 |
| Darcy/C-flow: threshold bias -0.5 | coeff 3/12 | threshold bias | above_or_equal_test | 0.001989 | 0.0004309 | 0.1545 | 0.02356 | 0.0003791 |
| Darcy/C-flow: squared-centered latent field, threshold bias -0.2 | coeff 3/12 | squared | above_or_equal_test | 0.001819 | 0.000537 | 0.1443 | 0.02325 | 0.0003742 |
| Darcy/C-flow: log-absolute centered latent field, threshold bias -0.2 | coeff 3/12 | log-abs | above_or_equal_test | 0.001628 | 0.0004546 | 0.133 | 0.02262 | 0.0003634 |
| Darcy/C-flow: binary coefficient 2/20, GRF alpha=6, tau=6 | coeff 2/20 | binary coefficient | above_or_equal_test | 0.001399 | 0.0003857 | 0.1927 | 0.01655 | 0.0002395 |
| Darcy/C-flow: binary coefficient 2/15, GRF alpha=2, tau=3 | coeff 2/15 | binary coefficient | above_or_equal_test | 0.001149 | 0.0003028 | 0.1506 | 0.01637 | 0.0002448 |
| Darcy/C-flow: binary coefficient 2/15, GRF alpha=1.5, tau=6 | coeff 2/15 | binary coefficient | above_or_equal_test | 0.00103 | 0.0002861 | 0.1439 | 0.01389 | 0.0002285 |
| Darcy/C-flow: binary coefficient 3/20, GRF alpha=4.5, tau=10 | coeff 3/20 | binary coefficient | above_or_equal_test | 0.001025 | 0.0002175 | 0.1946 | 0.01061 | 0.0001653 |

### ns2d

| case_label | plot_range | plot_family | rmse_position | rmse | rmse_sample_std_50 | relative_l2 | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NS2D: binary sign vorticity field, scale=1 | binary/sign | binary sign | above_or_equal_test | 1.06 | 0.174 | 0.77 | 6.368 | 0.08718 |
| NS2D: scale x2 and offset 0.5 | positive offset | scale + offset | above_or_equal_test | 0.6727 | 0.1105 | 0.4601 | 6.296 | 0.07127 |
| NS2D: scale x2 and offset -0.5 | negative offset | scale + offset | above_or_equal_test | 0.6527 | 0.09446 | 0.4497 | 6.066 | 0.06772 |
| NS2D: add negative vorticity offset -0.5 | negative offset | offset | above_or_equal_test | 0.5484 | 0.03739 | 0.3626 | 5.519 | 0.05339 |
| NS2D: Gaussian random vorticity, alpha=1, tau=2 | base range | gaussian RF alpha/tau | above_or_equal_test | 0.5468 | 0.1363 | 0.3949 | 7.185 | 0.08142 |
| NS2D: add positive vorticity offset 0.5 | positive offset | offset | above_or_equal_test | 0.5375 | 0.02087 | 0.3559 | 5.521 | 0.05467 |
| NS2D: Gaussian random vorticity, alpha=1, tau=10 | base range | gaussian RF alpha/tau | above_or_equal_test | 0.3951 | 0.09173 | 0.2842 | 6.698 | 0.07348 |
| NS2D: Gaussian random vorticity, alpha=1.2, tau=3 | base range | gaussian RF alpha/tau | above_or_equal_test | 0.3511 | 0.1045 | 0.2538 | 6.525 | 0.07447 |
| NS2D: binary sign vorticity field, scale=0.5 | binary/sign | binary sign | above_or_equal_test | 0.3313 | 0.0522 | 0.2454 | 6.11 | 0.07438 |
| NS2D: scale x1.5 and offset -0.25 | negative offset | scale + offset | above_or_equal_test | 0.3272 | 0.04001 | 0.234 | 5.862 | 0.06424 |
| NS2D: scale x1.5 and offset 0.25 | positive offset | scale + offset | above_or_equal_test | 0.3082 | 0.03344 | 0.2191 | 5.809 | 0.06177 |
| NS2D: add negative vorticity offset -0.25 | negative offset | offset | above_or_equal_test | 0.2848 | 0.02149 | 0.1989 | 5.508 | 0.05555 |

