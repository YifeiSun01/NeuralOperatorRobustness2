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
- soft-focus generated below test RMSE: 0/50
- soft-focus generated below test relative L2: 0/50

Strongest soft-focus correlations:

| feature | corr_RMSE | corr_relative_L2 |
| --- | --- | --- |
| target_final_highfreq_frac_mean | 0.787702 | 0.505176 |
| target_final_range_mean | 0.436337 | 0.0261455 |
| target_final_rms_mean | 0.310115 | -0.147181 |
| input_highfreq_frac_mean | 0.193087 | 0.319715 |
| target_final_tv_mean_mean | -0.180505 | 0.189309 |
| input_tv_mean_mean | -0.101088 | 0.117688 |
| target_final_spectral_centroid_mean | -0.0842193 | 0.305973 |
| input_spectral_centroid_mean | -0.0537509 | 0.299383 |

Lowest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| burgers_target_gaussian_corr0p085 | 0.0146485 | 0.0257573 | 6.49927e-10 | 1.0882e-06 | 0.86591 | 0.00211213 |
| Burgers: Gaussian initial field, correlation length=0.08 | 0.0148483 | 0.0266531 | 6.83586e-10 | 8.41772e-07 | 0.83226 | 0.00210599 |
| burgers_target_matern_corr0p42_nu3 | 0.0148645 | 0.0263761 | 2.00759e-12 | 1.80124e-06 | 0.874687 | 0.00202662 |
| burgers_target_matern_corr0p34_nu4 | 0.014931 | 0.0265218 | 9.25005e-15 | 1.69497e-06 | 0.879629 | 0.00201711 |
| burgers_target_matern_corr0p28_nu5 | 0.0149327 | 0.0265696 | 7.93841e-15 | 1.20588e-06 | 0.875374 | 0.00204639 |
| burgers_target_gaussian_corr0p09 | 0.0151897 | 0.0265745 | 6.15065e-10 | 1.22262e-06 | 0.87725 | 0.00208313 |
| burgers_target_matern_corr0p3_nu5 | 0.0152952 | 0.027088 | 8.03698e-15 | 1.55271e-06 | 0.886507 | 0.00201924 |
| burgers_target_gaussian_corr0p095 | 0.0157059 | 0.0273568 | 6.48368e-10 | 1.38968e-06 | 0.883914 | 0.00205448 |

Highest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers: add sawtooth pattern, amplitude=0.3 | 0.021171 | 0.0398251 | 0.00177041 | 1.98748e-07 | 0.657173 | 0.00245437 |
| Burgers: Matern initial field, correlation length=1, smoothness nu=4 | 0.0210295 | 0.035642 | 7.66085e-15 | 4.28528e-06 | 0.965191 | 0.00190165 |
| Burgers: Gaussian initial field, correlation length=0.5 | 0.0208401 | 0.0351268 | 2.14281e-06 | 5.4857e-06 | 0.987924 | 0.00193078 |
| Burgers: Gaussian initial field, correlation length=0.4 | 0.0208182 | 0.0350715 | 6.94076e-08 | 5.53203e-06 | 0.989068 | 0.00193306 |
| Burgers: Gaussian initial field, correlation length=0.3 | 0.0207156 | 0.0348855 | 7.53338e-09 | 5.53534e-06 | 0.98935 | 0.00193368 |
| burgers_target_gaussian_corr0p45 | 0.0204406 | 0.0344421 | 2.00503e-07 | 5.5e-06 | 0.988398 | 0.00193175 |
| burgers_target_gaussian_corr0p28 | 0.0203831 | 0.0343394 | 3.40391e-09 | 5.50488e-06 | 0.989329 | 0.00193371 |
| burgers_target_gaussian_corr0p26 | 0.0203812 | 0.034332 | 2.64469e-09 | 5.42304e-06 | 0.988974 | 0.00193306 |

### Darcy/C-flow

- train: RMSE=0.000134776, relative_L2=0.0198393
- test: RMSE=0.000155193, relative_L2=0.0227883
- soft-focus generated below test RMSE: 0/50
- soft-focus generated below test relative L2: 0/50

Strongest soft-focus correlations:

| feature | corr_RMSE | corr_relative_L2 |
| --- | --- | --- |
| target_final_range_mean | -0.608581 | -0.860812 |
| target_final_tv_mean_mean | -0.602204 | -0.863706 |
| target_final_rms_mean | -0.592595 | -0.862617 |
| target_final_highfreq_frac_mean | -0.300322 | -0.332938 |
| input_spectral_centroid_mean | -0.279931 | -0.531427 |
| input_highfreq_frac_mean | -0.2776 | -0.532697 |
| input_tv_mean_mean | -0.235566 | -0.489647 |
| target_final_spectral_centroid_mean | 0.13124 | 0.405746 |

Lowest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Darcy/C-flow: binary coefficient 3/12, GRF alpha=0.7, tau=2 | 0.000238652 | 0.038727 | 0.770956 | 0.000137357 | 0.0111422 | 0.000185217 |
| darcy_target_identity_bias0p18 | 0.000246128 | 0.0495603 | 0.049637 | 0.000105476 | 0.00952151 | 0.000150971 |
| darcy_target_square_centered_bias0p02 | 0.000251053 | 0.0347583 | 0.0709828 | 7.80346e-05 | 0.0141314 | 0.000225016 |
| darcy_target_log_abs_centered_bias0p08 | 0.000252626 | 0.051605 | 0.102935 | 9.59766e-05 | 0.00953066 | 0.000152334 |
| darcy_target_identity_bias0p2 | 0.000271331 | 0.0569921 | 0.0544823 | 0.000101083 | 0.00882927 | 0.000141598 |
| darcy_target_identity_bias0p22 | 0.000286136 | 0.0617457 | 0.0521687 | 0.000105004 | 0.00883604 | 0.000141006 |
| darcy_target_alpha0p55_tau1p5_bin3_12 | 0.00029671 | 0.0488684 | 0.830034 | 0.000118736 | 0.0109614 | 0.000180991 |
| darcy_target_alpha0p55_tau3_bin3_12 | 0.000299166 | 0.0493049 | 0.833704 | 0.000122118 | 0.0109672 | 0.000181234 |

Highest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| darcy_target_identity_bias0p46 | 0.00045815 | 0.123544 | 0.0973268 | 9.0618e-05 | 0.00668844 | 0.000110519 |
| darcy_target_log_abs_centered_bias0p16 | 0.000457384 | 0.123591 | 0.175591 | 8.41767e-05 | 0.00683822 | 0.000110283 |
| darcy_target_identity_bias0p44 | 0.000455845 | 0.121589 | 0.099086 | 8.85259e-05 | 0.00653589 | 0.000107767 |
| Darcy/C-flow: binary coefficient 3/20, GRF alpha=0.5, tau=15 | 0.000453389 | 0.106212 | 0.850823 | 0.000164534 | 0.00772161 | 0.000129279 |
| darcy_target_identity_bias0p42 | 0.000451903 | 0.119229 | 0.122894 | 9.1828e-05 | 0.0066316 | 0.000109873 |
| darcy_target_square_centered_bias0p1 | 0.000447552 | 0.108958 | 0.013006 | 6.73272e-05 | 0.00772642 | 0.000122321 |
| darcy_target_identity_bias0p4 | 0.000447221 | 0.118495 | 0.118586 | 9.16741e-05 | 0.00689505 | 0.000112551 |
| darcy_target_identity_bias0p38 | 0.000428331 | 0.109308 | 0.0797765 | 9.37323e-05 | 0.00701957 | 0.000114807 |

### NS2D

- train: RMSE=0.0713213, relative_L2=0.0512957
- test: RMSE=0.133571, relative_L2=0.0957001
- soft-focus generated below test RMSE: 0/48
- soft-focus generated below test relative L2: 0/48

Strongest soft-focus correlations:

| feature | corr_RMSE | corr_relative_L2 |
| --- | --- | --- |
| input_tv_mean_mean | 0.648071 | 0.631868 |
| target_final_range_mean | 0.629319 | 0.640457 |
| input_highfreq_frac_mean | 0.608364 | 0.586375 |
| target_final_tv_mean_mean | 0.603242 | 0.638208 |
| input_spectral_centroid_mean | 0.562028 | 0.540553 |
| target_final_spectral_centroid_mean | 0.497027 | 0.545237 |
| target_final_highfreq_frac_mean | 0.492561 | 0.535947 |
| target_final_rms_mean | -0.485307 | -0.546191 |

Lowest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| NS2D: add sawtooth vorticity pattern, amplitude=0.5 | 0.208036 | 0.14725 | 0.0138437 | 0.000186923 | 5.86321 | 0.0611845 |
| NS2D: Gaussian random vorticity, alpha=1.5, tau=7 | 0.212079 | 0.152086 | 0.0146831 | 0.000268602 | 6.06607 | 0.0665734 |
| NS2D: scale initial vorticity x2 | 0.22976 | 0.168167 | 1.55402e-05 | 0.000340887 | 6.15985 | 0.068485 |
| ns_target_grf_alpha1p4_tau4 | 0.23227 | 0.167402 | 0.0203101 | 0.000272773 | 6.11726 | 0.0666967 |
| ns_target_grf_alpha1p35_tau12 | 0.234095 | 0.164823 | 0.0509592 | 0.000203683 | 6.19031 | 0.0635853 |
| ns_target_grf_alpha1p4_tau5 | 0.234819 | 0.16768 | 0.0228588 | 0.000363428 | 6.12621 | 0.0673153 |
| ns_target_grf_alpha1p4_tau10 | 0.235837 | 0.168361 | 0.0346222 | 0.000294778 | 6.14599 | 0.0650437 |
| ns_target_grf_alpha1p4_tau8 | 0.239534 | 0.170209 | 0.0290407 | 0.000262725 | 6.25295 | 0.0660895 |

Highest RMSE soft-focus generated datasets:

| case_label | full_rmse_mean | full_relative_l2_mean | input_highfreq_frac_mean | target_final_highfreq_frac_mean | target_final_range_mean | target_final_tv_mean_mean |
| --- | --- | --- | --- | --- | --- | --- |
| ns_target_grf_alpha1p1_tau4 | 0.400073 | 0.292801 | 0.120795 | 0.000409005 | 6.61358 | 0.0743633 |
| NS2D: Gaussian random vorticity, alpha=1, tau=10 | 0.395076 | 0.284173 | 0.222358 | 0.00035906 | 6.69761 | 0.0734777 |
| ns_target_grf_alpha1p15_tau3 | 0.39446 | 0.282128 | 0.0907979 | 0.000373364 | 6.63174 | 0.07498 |
| ns_target_grf_alpha1p15_tau4 | 0.380146 | 0.277407 | 0.094092 | 0.000306323 | 6.6196 | 0.070179 |
| ns_target_grf_alpha1p2_tau4 | 0.374744 | 0.27123 | 0.0669563 | 0.000343249 | 6.54821 | 0.0722319 |
| ns_target_grf_alpha1p15_tau5 | 0.369277 | 0.266533 | 0.0981349 | 0.000423131 | 6.52239 | 0.0748346 |
| ns_target_grf_alpha1p2_tau5 | 0.365871 | 0.264963 | 0.0718963 | 0.000339514 | 6.57526 | 0.0726696 |
| ns_target_grf_alpha1p1_tau8 | 0.364441 | 0.262995 | 0.14002 | 0.000325016 | 6.59852 | 0.0701916 |

