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

