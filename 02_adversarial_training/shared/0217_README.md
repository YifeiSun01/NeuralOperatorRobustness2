# Generalization loss pattern diagnostics

This report compares low-loss and high-loss generated datasets for Burgers, Darcy/C-flow, and NS2D using the same retained FNO checkpoints.

Terminology: `ground truth final` means the solver/PDE target at the final predicted time or the Darcy solution target. The previous `soft output target last` wording was incorrect and is not used here.

Hard `sign` generated datasets are kept in the CSV but excluded from the soft-focus example selection and main soft-pattern correlations. That avoids using the artificial -1/1 style cases as evidence for smooth continuous generalization.

## Files

- `sample_pattern_metrics.csv`: per-sample RMSE/relative-L2 plus range, TV, and spectral features.
- `dataset_pattern_summary.csv`: per-dataset means/stds of those features.
- `*_loss_vs_pattern_metrics.png`: scatter plots linking loss with visual/spectral descriptors.
- `*_rmse_ranked.png`: ranked RMSE per task, sorted from low to high RMSE; generated bars use discrete range colors from the old `coolwarm` 0.15-0.85 palette, and hatch encodes generator family.
- `*_rmse_grouped_by_generator.png`: same bars grouped by generator family and discrete range label.
- `*_low_vs_high_target_final_spectrum.png`: average final-state spectra for low vs high RMSE datasets.
- `examples/`: low/high RMSE panels with input, ground truth, model output, difference, and spectrum.

## Numeric summary

### Burgers

- train: RMSE=0.0086768, relative_L2=0.0163895
- test: RMSE=0.0095436, relative_L2=0.0177549
- soft-focus generated below test RMSE: 2/48
- soft-focus generated below test relative L2: 1/48

Strongest soft-focus correlations:

| feature | corr_RMSE | corr_relative_L2 |
| --- | --- | --- |
| target_final_tv_mean_mean | 0.631866 | 0.518577 |
| target_final_highfreq_frac_mean | 0.404178 | 0.30196 |
| target_final_range_mean | 0.227744 | 0.138231 |
| target_final_spectral_centroid_mean | 0.208813 | 0.205313 |
| input_highfreq_frac_mean | -0.174481 | -0.168953 |
| input_spectral_centroid_mean | -0.154206 | -0.147813 |
| input_tv_mean_mean | -0.102384 | -0.109295 |
| target_final_rms_mean | -0.0737414 | -0.483898 |

Lowest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers: Gaussian initial field, correlation length=0.75 | 0.00904838 | 0.0167386 | 0.0402621 | 1.87115e-07 | 0.666413 | 0.00182349 |
| Burgers: Gaussian initial field, correlation length=0.025 | 0.00945809 | 0.0181661 | 1.32243e-09 | 3.59487e-08 | 0.607013 | 0.00229602 |
| Burgers: Gaussian initial field, correlation length=0.035 | 0.0097846 | 0.0186187 | 1.0469e-09 | 8.28903e-08 | 0.676651 | 0.00229756 |
| Burgers: Matern initial field, correlation length=0.1, smoothness nu=5 | 0.0100939 | 0.0187912 | 3.27363e-14 | 1.00111e-07 | 0.66898 | 0.00234041 |
| Burgers: Gaussian initial field, correlation length=0.04 | 0.0102829 | 0.0194147 | 8.46273e-10 | 1.03129e-07 | 0.706176 | 0.00228369 |
| Burgers: Gaussian initial field, correlation length=0.02 | 0.0103738 | 0.0200568 | 1.31564e-09 | 9.8371e-09 | 0.561313 | 0.00225678 |
| Burgers: Matern initial field, correlation length=0.02, smoothness nu=0.8 | 0.0105218 | 0.0208457 | 0.0443689 | 4.70725e-13 | 0.200898 | 0.00118269 |
| Burgers: Matern initial field, correlation length=0.08, smoothness nu=3.5 | 0.0106579 | 0.0201409 | 1.08131e-09 | 1.95804e-08 | 0.57463 | 0.00229901 |

Highest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers: centered amplitude x2.5, offset -0.25 | 0.6021 | 1.4294 | 1.01147e-09 | 9.01586e-06 | 1.02057 | 0.00311709 |
| Burgers: centered amplitude x2, offset 0 | 0.361561 | 1.23694 | 1.06011e-09 | 3.50158e-06 | 0.91518 | 0.00295313 |
| Burgers: add negative offset -1 | 0.337561 | 0.63277 | 9.67578e-10 | 1.57516e-07 | 0.652153 | 0.00239185 |
| Burgers: centered amplitude x1, offset -0.4 | 0.326527 | 0.741491 | 1.02207e-09 | 1.58755e-07 | 0.642598 | 0.00237294 |
| Burgers: add positive offset 1 | 0.290221 | 0.192426 | 1.02423e-09 | 1.46688e-07 | 0.65236 | 0.00238543 |
| Burgers: centered amplitude x2.5, offset 0.25 | 0.289728 | 0.699841 | 1.01739e-09 | 8.79569e-06 | 1.01385 | 0.00310605 |
| Burgers: centered amplitude x1.5, offset 0 | 0.278718 | 1.16296 | 1.06418e-09 | 1.4779e-06 | 0.795492 | 0.00273161 |
| Burgers: add negative offset -0.5 | 0.190449 | 1.06058 | 1.04016e-09 | 1.02502e-07 | 0.636128 | 0.00234882 |

### Darcy/C-flow

- train: RMSE=0.000134776, relative_L2=0.0198393
- test: RMSE=0.000155193, relative_L2=0.0227883
- soft-focus generated below test RMSE: 13/50
- soft-focus generated below test relative L2: 13/50

Strongest soft-focus correlations:

| feature | corr_RMSE | corr_relative_L2 |
| --- | --- | --- |
| target_final_range_mean | 0.819982 | 0.627032 |
| target_final_tv_mean_mean | 0.792065 | 0.601508 |
| target_final_rms_mean | 0.727783 | 0.525921 |
| target_final_spectral_centroid_mean | 0.558036 | 0.629072 |
| target_final_highfreq_frac_mean | 0.247705 | 0.37044 |
| input_highfreq_frac_mean | -0.135529 | -0.0927889 |
| input_spectral_centroid_mean | -0.131593 | -0.0919186 |
| input_tv_mean_mean | -0.100653 | -0.0263723 |

Lowest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=5, tau=2 | 7.6794e-05 | 0.0113148 | 0.0241003 | 8.88665e-05 | 0.0138017 | 0.000206981 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=6, tau=1.5 | 7.72079e-05 | 0.0113635 | 0.023804 | 8.90422e-05 | 0.0138635 | 0.000209322 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=3.5, tau=3 | 0.000103843 | 0.0152159 | 0.0260246 | 9.04157e-05 | 0.0138382 | 0.000210205 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=4, tau=4 | 0.000111553 | 0.0162406 | 0.0264056 | 9.33509e-05 | 0.0135967 | 0.000209367 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2, tau=1 | 0.000125148 | 0.0183382 | 0.0355928 | 9.21639e-05 | 0.0134202 | 0.000206056 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2.5, tau=3.5 | 0.000128268 | 0.0189807 | 0.0316432 | 9.15711e-05 | 0.0133866 | 0.000207943 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2, tau=2 | 0.000130436 | 0.0194024 | 0.039676 | 9.8295e-05 | 0.0129046 | 0.000202003 |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=2.3, tau=3 | 0.000130968 | 0.0194376 | 0.0323624 | 8.97165e-05 | 0.013407 | 0.000206025 |

Highest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Darcy/C-flow: binary coefficient 1/12, GRF alpha=3, tau=1.5 | 0.0067133 | 0.479666 | 0.025922 | 0.000127204 | 0.032587 | 0.000467608 |
| Darcy/C-flow: binary coefficient 1/12, GRF alpha=3.5, tau=6 | 0.005945 | 0.453444 | 0.0308923 | 0.000151445 | 0.0283394 | 0.000418575 |
| Darcy/C-flow: binary coefficient 1/12, GRF alpha=2, tau=3 | 0.00553693 | 0.431882 | 0.0442295 | 0.000152655 | 0.0285145 | 0.000420514 |
| Darcy/C-flow: binary coefficient 1/20, GRF alpha=2, tau=12 | 0.00345732 | 0.396703 | 0.11549 | 0.000333551 | 0.018787 | 0.000309237 |
| Darcy/C-flow: threshold bias -0.8 | 0.00232523 | 0.172339 | 0.0818754 | 6.57315e-05 | 0.0243767 | 0.000393581 |
| Darcy/C-flow: threshold bias -0.5 | 0.00198914 | 0.154495 | 0.144082 | 6.41557e-05 | 0.0235564 | 0.000379065 |
| Darcy/C-flow: squared-centered latent field, threshold bias -0.2 | 0.00181885 | 0.14427 | 0.104916 | 6.10453e-05 | 0.0232538 | 0.000374209 |
| Darcy/C-flow: log-absolute centered latent field, threshold bias -0.2 | 0.00162832 | 0.132978 | 0.105881 | 6.13077e-05 | 0.022624 | 0.000363442 |

### NS2D

- train: RMSE=0.0713213, relative_L2=0.0512957
- test: RMSE=0.133571, relative_L2=0.0957001
- soft-focus generated below test RMSE: 31/47
- soft-focus generated below test relative L2: 31/47

Strongest soft-focus correlations:

| feature | corr_RMSE | corr_relative_L2 |
| --- | --- | --- |
| target_final_highfreq_frac_mean | 0.69635 | 0.720406 |
| target_final_tv_mean_mean | 0.687601 | 0.712413 |
| target_final_range_mean | 0.680376 | 0.70769 |
| target_final_spectral_centroid_mean | 0.630911 | 0.652044 |
| target_final_rms_mean | -0.43536 | -0.466502 |
| input_tv_mean_mean | 0.412209 | 0.437885 |
| input_spectral_centroid_mean | 0.389373 | 0.414652 |
| input_highfreq_frac_mean | 0.388499 | 0.413116 |

Lowest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| NS2D: Gaussian random vorticity, alpha=4, tau=4 | 0.0259965 | 0.0178177 | 5.49116e-11 | 5.06498e-06 | 5.24961 | 0.0440358 |
| NS2D: squared-and-centered initial field, scale=1 | 0.0261181 | 0.0177915 | 4.07479e-05 | 4.48705e-07 | 5.34493 | 0.0420452 |
| NS2D: Gaussian random vorticity, alpha=3, tau=3 | 0.0282695 | 0.0194117 | 9.20352e-08 | 1.59799e-05 | 5.26053 | 0.0455072 |
| NS2D: Gaussian random vorticity, alpha=4.5, tau=7 | 0.0301152 | 0.0209342 | 7.27736e-12 | 3.26261e-05 | 5.28373 | 0.0473198 |
| NS2D: Gaussian random vorticity, alpha=5, tau=8 | 0.0308551 | 0.0213649 | 4.68482e-13 | 4.33375e-05 | 5.28927 | 0.0482568 |
| NS2D: Gaussian random vorticity, alpha=6, tau=10 | 0.0314619 | 0.021755 | 1.85646e-14 | 4.49471e-05 | 5.2937 | 0.0483803 |
| NS2D: scale initial vorticity x0.5 | 0.0352213 | 0.0244264 | 1.28568e-05 | 3.87931e-05 | 5.32367 | 0.0478102 |
| NS2D: Gaussian random vorticity, alpha=2, tau=2 | 0.0382196 | 0.0266001 | 0.000206557 | 5.46969e-05 | 5.34184 | 0.0493593 |

Highest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| NS2D: scale x2 and offset 0.5 | 0.672659 | 0.460134 | 1.32411e-05 | 0.000384366 | 6.29559 | 0.0712709 |
| NS2D: scale x2 and offset -0.5 | 0.652734 | 0.449686 | 1.58555e-05 | 0.000305144 | 6.06553 | 0.0677164 |
| NS2D: add negative vorticity offset -0.5 | 0.548399 | 0.362623 | 1.67994e-05 | 0.00010332 | 5.51942 | 0.0533901 |
| NS2D: Gaussian random vorticity, alpha=1, tau=2 | 0.546791 | 0.394879 | 0.186163 | 0.000423693 | 7.18469 | 0.0814184 |
| NS2D: add positive vorticity offset 0.5 | 0.537486 | 0.355913 | 1.49251e-05 | 0.000138516 | 5.52078 | 0.054668 |
| NS2D: Gaussian random vorticity, alpha=1, tau=10 | 0.395076 | 0.284173 | 0.222358 | 0.00035906 | 6.69761 | 0.0734777 |
| NS2D: Gaussian random vorticity, alpha=1.2, tau=3 | 0.351083 | 0.253752 | 0.0660112 | 0.000396701 | 6.52524 | 0.0744666 |
| NS2D: scale x1.5 and offset -0.25 | 0.327167 | 0.234037 | 1.32545e-05 | 0.000303646 | 5.86192 | 0.0642387 |

