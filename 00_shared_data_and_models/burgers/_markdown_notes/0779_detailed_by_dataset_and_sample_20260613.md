# Detailed By-Dataset and By-Sample Appendix: Six Burgers Models

Date: 2026-06-13

This appendix expands the corrected six-model summary into explicit rows. It separates the data granularities so the numbers are not mixed:

- Clean generalization/training/test loss: all 52 datasets, six models.
- Full random-model attack suite: all 52 datasets, two random models only, because that is the full-52 attack table currently present for the random models.
- Robustness/SVD/attack mechanism detail: the matched 25-sample latest wideparam SVD/attack manifest, six models, 150 model-sample rows.

Dataset-id note: old-four clean rows name train/test as `burgers_original_*_seed45`; random-model rows name the same data as `train_original_gaussian_corr0p03` and `test_original_gaussian_corr0p03`. The table below uses the canonical short id and keeps the old id in the CSV.

## Files

- `forensics/burgers_six_model_latest_wideparam_summary_20260613/clean_52dataset_six_models_detailed.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/random_models_attack_52dataset_detailed.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_manifest.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_six_models_detailed_with_subspace.csv`

## 1. Dataset Manifest: All 52 Clean Datasets

| order | split | dataset_id | n | family | display_label |
| --- | --- | --- | --- | --- | --- |
| 1 | train | train_original_gaussian_corr0p03 | 1350 | train_test_gaussian | Original Burgers train; Gaussian GRF corr=0.03; range observed [0,1] |
| 2 | test | test_original_gaussian_corr0p03 | 150 | train_test_gaussian | Original Burgers test; Gaussian GRF corr=0.03; range observed [0,1] |
| 3 | generalization | burgers_widevis_l3target_d00 | 200 | gaussian | Gaussian GRF corr=0.012; range [-0.2,1.2] |
| 4 | generalization | burgers_widevis_l3target_d01 | 200 | gaussian | Gaussian GRF corr=0.012; range [-0.3,1.3] |
| 5 | generalization | burgers_widevis_l3target_d02 | 200 | gaussian | Gaussian GRF corr=0.012; range [0.15,1.25] |
| 6 | generalization | burgers_widevis_l3target_d03 | 200 | gaussian | Gaussian GRF corr=0.012; range [0,1.2] |
| 7 | generalization | burgers_widevis_l3target_d04 | 200 | gaussian | Gaussian GRF corr=0.009; range [-0.3,1.3] |
| 8 | generalization | burgers_widevis_l3target_d05 | 200 | gaussian | Gaussian GRF corr=0.009; range [0.15,1.25] |
| 9 | generalization | burgers_widevis_l3target_d06 | 200 | gaussian | Gaussian GRF corr=0.012; range [-0.1,1.1] |
| 10 | generalization | burgers_widevis_l3target_d07 | 200 | gaussian | Gaussian GRF corr=0.009; range [0,1.2] |
| 11 | generalization | burgers_widevis_l3target_d08 | 200 | gaussian | Gaussian GRF corr=0.012; range [-0.5,1.5] |
| 12 | generalization | burgers_widevis_l3target_d09 | 200 | gaussian | Gaussian GRF corr=0.009; range [-0.5,1.5] |
| 13 | generalization | burgers_widevis_l3target_d10 | 200 | gaussian | Gaussian GRF corr=0.009; range [-0.2,1.2] |
| 14 | generalization | burgers_widevis_l3target_d11 | 200 | gaussian | Gaussian GRF corr=0.012; range [0.05,1.05] |
| 15 | generalization | burgers_widevis_l3target_d12 | 200 | matern | Matern GRF c=0.055, nu=4; range [-0.3,1.3] |
| 16 | generalization | burgers_widevis_l3target_d13 | 200 | matern | Matern GRF c=0.04, nu=2.5; range [-0.3,1.3] |
| 17 | generalization | burgers_widevis_l3target_d14 | 200 | matern | Matern GRF c=0.04, nu=2.5; range [-0.5,1.5] |
| 18 | generalization | burgers_widevis_l3target_d15 | 200 | matern | Matern GRF c=0.04, nu=2.5; range [0.15,1.25] |
| 19 | generalization | burgers_widevis_l3target_d16 | 200 | matern | Matern GRF c=0.08, nu=2.5; range [-0.3,1.3] |
| 20 | generalization | burgers_widevis_l3target_d17 | 200 | matern | Matern GRF c=0.04, nu=2.5; range [-0.65,1.35] |
| 21 | generalization | burgers_widevis_l3target_d18 | 200 | matern | Matern GRF c=0.04, nu=2.5; range [0,1.5] |
| 22 | generalization | burgers_widevis_l3target_d19 | 200 | matern | Matern GRF c=0.08, nu=2.5; range [-0.5,1.5] |
| 23 | generalization | burgers_widevis_l3target_d20 | 200 | matern | Matern GRF c=0.055, nu=4; range [-0.5,1.5] |
| 24 | generalization | burgers_widevis_l3target_d21 | 200 | matern | Matern GRF c=0.03, nu=1.5; range [0,1.5] |
| 25 | generalization | burgers_widevis_l3target_d22 | 200 | powerlaw_fourier | Power-law Fourier alpha=2.5, k0=18; range [-0.2,1.2] |
| 26 | generalization | burgers_widevis_l3target_d23 | 200 | powerlaw_fourier | Power-law Fourier alpha=1.5, k0=10; range [-0.5,1.5] |
| 27 | generalization | burgers_widevis_l3target_d24 | 200 | powerlaw_fourier | Power-law Fourier alpha=3, k0=24; range [-0.2,1.2] |
| 28 | generalization | burgers_widevis_l3target_d25 | 200 | powerlaw_fourier | Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5] |
| 29 | generalization | burgers_widevis_l3target_d26 | 200 | powerlaw_fourier | Power-law Fourier alpha=2.5, k0=18; range [-0.1,1.1] |
| 30 | generalization | burgers_widevis_l3target_d27 | 200 | powerlaw_fourier | Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5] |
| 31 | generalization | burgers_widevis_l3target_d28 | 200 | powerlaw_fourier | Power-law Fourier alpha=2.2, k0=16; range [-0.2,1.2] |
| 32 | generalization | burgers_widevis_l3target_d29 | 200 | powerlaw_fourier | Power-law Fourier alpha=3.5, k0=28; range [0.15,1.25] |
| 33 | generalization | burgers_widevis_l3target_d30 | 200 | powerlaw_fourier | Power-law Fourier alpha=3.5, k0=28; range [-0.5,1.5] |
| 34 | generalization | burgers_widevis_l3target_d31 | 200 | powerlaw_fourier | Power-law Fourier alpha=4, k0=36; range [-0.5,1.5] |
| 35 | generalization | burgers_widevis_l3target_d32 | 200 | powerlaw_fourier | Power-law Fourier alpha=4, k0=36; range [0.15,1.25] |
| 36 | generalization | burgers_widevis_l3target_d33 | 200 | powerlaw_fourier | Power-law Fourier alpha=3, k0=24; range [-0.1,1.1] |
| 37 | generalization | burgers_widevis_l3target_d34 | 200 | powerlaw_fourier | Power-law Fourier alpha=1.8, k0=12; range [-0.5,1.5] |
| 38 | generalization | burgers_widevis_l3target_d35 | 200 | powerlaw_fourier | Power-law Fourier alpha=3, k0=24; range [-0.5,1.5] |
| 39 | generalization | burgers_widevis_l3target_d36 | 200 | sine_mixture | Sine mix f=[5, 9, 15, 23], decay=0.25; range [0,1.5] |
| 40 | generalization | burgers_widevis_l3target_d37 | 200 | sine_mixture | Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5] |
| 41 | generalization | burgers_widevis_l3target_d38 | 200 | sine_mixture | Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.3,1.3] |
| 42 | generalization | burgers_widevis_l3target_d39 | 200 | sine_mixture | Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.3,1.3] |
| 43 | generalization | burgers_widevis_l3target_d40 | 200 | sine_mixture | Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.2,1.2] |
| 44 | generalization | burgers_widevis_l3target_d41 | 200 | sine_mixture | Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.65,1.35] |
| 45 | generalization | burgers_widevis_l3target_d42 | 200 | sine_mixture | Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.5,1.5] |
| 46 | generalization | burgers_widevis_l3target_d43 | 200 | sine_mixture | Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.2,1.2] |
| 47 | generalization | burgers_widevis_l3target_d44 | 200 | sine_mixture | Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.5,1.5] |
| 48 | generalization | burgers_widevis_l3target_d45 | 200 | sine_mixture | Sine mix f=[5, 10, 20, 40, 80], decay=0.9; range [-0.1,1.1] |
| 49 | generalization | burgers_widevis_l3target_d46 | 200 | sine_mixture | Sine mix f=[4, 7, 11], decay=0.35; range [-0.3,1.3] |
| 50 | generalization | burgers_widevis_l3target_d47 | 200 | sine_mixture | Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.65,1.35] |
| 51 | generalization | burgers_widevis_l3target_d48 | 200 | sawtooth | Sawtooth freq=2; range [0.15,1.25] |
| 52 | generalization | burgers_widevis_l3target_d49 | 200 | square_wave | Square wave freq=7, duty=0.35; range [0,1.2] |

## 2. Clean RMSE: 52 Datasets x 6 Models

| order | split | dataset_id | best | baseline | loss1 | loss2 | loss3 | random_clean_y | random_solver_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | train | train_original_gaussian_corr0p03 | loss1 | 0.008756 | 0.000747 | 0.001402 | 0.004250 | 0.007395 | 0.001372 |
| 2 | test | test_original_gaussian_corr0p03 | loss1 | 0.009232 | 0.000836 | 0.001497 | 0.004374 | 0.070656 | 0.001527 |
| 3 | generalization | burgers_widevis_l3target_d00 | loss3 | 0.020888 | 0.015091 | 0.016997 | 0.008614 | 0.082199 | 0.013815 |
| 4 | generalization | burgers_widevis_l3target_d01 | loss3 | 0.027672 | 0.020034 | 0.022459 | 0.011412 | 0.096383 | 0.019503 |
| 5 | generalization | burgers_widevis_l3target_d02 | loss3 | 0.025101 | 0.011765 | 0.012962 | 0.006844 | 0.081537 | 0.012515 |
| 6 | generalization | burgers_widevis_l3target_d03 | loss3 | 0.019186 | 0.011511 | 0.013052 | 0.006707 | 0.078570 | 0.011073 |
| 7 | generalization | burgers_widevis_l3target_d04 | loss3 | 0.025473 | 0.020986 | 0.022898 | 0.012178 | 0.084944 | 0.019159 |
| 8 | generalization | burgers_widevis_l3target_d05 | loss3 | 0.022631 | 0.012963 | 0.013702 | 0.007636 | 0.074628 | 0.013120 |
| 9 | generalization | burgers_widevis_l3target_d06 | loss3 | 0.016760 | 0.010937 | 0.012521 | 0.006488 | 0.070311 | 0.009665 |
| 10 | generalization | burgers_widevis_l3target_d07 | loss3 | 0.019386 | 0.013500 | 0.014647 | 0.008005 | 0.068943 | 0.012403 |
| 11 | generalization | burgers_widevis_l3target_d08 | loss3 | 0.050017 | 0.035929 | 0.038104 | 0.021160 | 0.129116 | 0.037325 |
| 12 | generalization | burgers_widevis_l3target_d09 | loss3 | 0.040383 | 0.034050 | 0.036023 | 0.020090 | 0.110890 | 0.033955 |
| 13 | generalization | burgers_widevis_l3target_d10 | loss3 | 0.020726 | 0.015908 | 0.017486 | 0.009648 | 0.074209 | 0.013875 |
| 14 | generalization | burgers_widevis_l3target_d11 | loss3 | 0.014129 | 0.008113 | 0.009309 | 0.004964 | 0.064209 | 0.007242 |
| 15 | generalization | burgers_widevis_l3target_d12 | loss3 | 0.031493 | 0.016040 | 0.018466 | 0.009752 | 0.119787 | 0.017217 |
| 16 | generalization | burgers_widevis_l3target_d13 | loss3 | 0.024607 | 0.019219 | 0.020424 | 0.011440 | 0.083356 | 0.017240 |
| 17 | generalization | burgers_widevis_l3target_d14 | loss3 | 0.040699 | 0.031609 | 0.032796 | 0.019281 | 0.112304 | 0.031191 |
| 18 | generalization | burgers_widevis_l3target_d15 | loss3 | 0.022417 | 0.011730 | 0.012441 | 0.007261 | 0.074658 | 0.011935 |
| 19 | generalization | burgers_widevis_l3target_d16 | loss3 | 0.032854 | 0.014689 | 0.017399 | 0.009404 | 0.115259 | 0.016269 |
| 20 | generalization | burgers_widevis_l3target_d17 | loss3 | 0.053124 | 0.035878 | 0.035441 | 0.022168 | 0.120842 | 0.038185 |
| 21 | generalization | burgers_widevis_l3target_d18 | loss3 | 0.055691 | 0.024611 | 0.026806 | 0.015951 | 0.107200 | 0.030233 |
| 22 | generalization | burgers_widevis_l3target_d19 | loss3 | 0.065774 | 0.034025 | 0.037197 | 0.022744 | 0.156183 | 0.040313 |
| 23 | generalization | burgers_widevis_l3target_d20 | loss3 | 0.060027 | 0.033016 | 0.036748 | 0.021996 | 0.155641 | 0.038589 |
| 24 | generalization | burgers_widevis_l3target_d21 | loss3 | 0.031912 | 0.017895 | 0.018233 | 0.012269 | 0.077515 | 0.019268 |
| 25 | generalization | burgers_widevis_l3target_d22 | loss3 | 0.020071 | 0.014939 | 0.015911 | 0.008577 | 0.076902 | 0.013041 |
| 26 | generalization | burgers_widevis_l3target_d23 | loss3 | 0.039022 | 0.026142 | 0.028073 | 0.015604 | 0.117379 | 0.026996 |
| 27 | generalization | burgers_widevis_l3target_d24 | loss3 | 0.019932 | 0.015800 | 0.016637 | 0.009419 | 0.072541 | 0.013514 |
| 28 | generalization | burgers_widevis_l3target_d25 | loss3 | 0.041626 | 0.032111 | 0.033582 | 0.018751 | 0.115063 | 0.031897 |
| 29 | generalization | burgers_widevis_l3target_d26 | loss3 | 0.016534 | 0.011114 | 0.012056 | 0.006749 | 0.066929 | 0.009503 |
| 30 | generalization | burgers_widevis_l3target_d27 | loss3 | 0.033830 | 0.022152 | 0.022926 | 0.013533 | 0.105322 | 0.021607 |
| 31 | generalization | burgers_widevis_l3target_d28 | loss3 | 0.019215 | 0.013751 | 0.014950 | 0.008353 | 0.075679 | 0.012324 |
| 32 | generalization | burgers_widevis_l3target_d29 | loss3 | 0.021955 | 0.012169 | 0.013005 | 0.007506 | 0.074278 | 0.012252 |
| 33 | generalization | burgers_widevis_l3target_d30 | loss3 | 0.039177 | 0.030732 | 0.032112 | 0.018782 | 0.108161 | 0.030406 |
| 34 | generalization | burgers_widevis_l3target_d31 | loss3 | 0.037001 | 0.033389 | 0.033737 | 0.020449 | 0.099914 | 0.031343 |
| 35 | generalization | burgers_widevis_l3target_d32 | loss3 | 0.020138 | 0.012560 | 0.013234 | 0.007732 | 0.067280 | 0.012447 |
| 36 | generalization | burgers_widevis_l3target_d33 | loss3 | 0.016644 | 0.011714 | 0.012660 | 0.007174 | 0.065229 | 0.009884 |
| 37 | generalization | burgers_widevis_l3target_d34 | loss3 | 0.042284 | 0.027699 | 0.029245 | 0.016942 | 0.119466 | 0.028547 |
| 38 | generalization | burgers_widevis_l3target_d35 | loss3 | 0.037797 | 0.032115 | 0.032886 | 0.019569 | 0.107791 | 0.030739 |
| 39 | generalization | burgers_widevis_l3target_d36 | loss3 | 0.037549 | 0.024549 | 0.024522 | 0.010642 | 0.093273 | 0.023157 |
| 40 | generalization | burgers_widevis_l3target_d37 | loss3 | 0.027876 | 0.018915 | 0.021274 | 0.008616 | 0.037147 | 0.022291 |
| 41 | generalization | burgers_widevis_l3target_d38 | loss3 | 0.022650 | 0.020632 | 0.021694 | 0.010124 | 0.081058 | 0.017004 |
| 42 | generalization | burgers_widevis_l3target_d39 | loss3 | 0.023021 | 0.021906 | 0.022153 | 0.010504 | 0.042219 | 0.014262 |
| 43 | generalization | burgers_widevis_l3target_d40 | loss3 | 0.018930 | 0.017719 | 0.017947 | 0.008544 | 0.039456 | 0.011794 |
| 44 | generalization | burgers_widevis_l3target_d41 | loss3 | 0.033706 | 0.035034 | 0.034487 | 0.017008 | 0.074518 | 0.029534 |
| 45 | generalization | burgers_widevis_l3target_d42 | loss3 | 0.031532 | 0.030942 | 0.033510 | 0.015254 | 0.097181 | 0.027455 |
| 46 | generalization | burgers_widevis_l3target_d43 | loss3 | 0.019510 | 0.016016 | 0.016636 | 0.008244 | 0.072430 | 0.012822 |
| 47 | generalization | burgers_widevis_l3target_d44 | loss3 | 0.030622 | 0.029937 | 0.029996 | 0.015180 | 0.057648 | 0.019251 |
| 48 | generalization | burgers_widevis_l3target_d45 | loss3 | 0.011166 | 0.008160 | 0.008614 | 0.004264 | 0.068556 | 0.006297 |
| 49 | generalization | burgers_widevis_l3target_d46 | loss3 | 0.029482 | 0.014125 | 0.018727 | 0.008549 | 0.103633 | 0.015580 |
| 50 | generalization | burgers_widevis_l3target_d47 | loss3 | 0.029870 | 0.029298 | 0.027531 | 0.016086 | 0.057535 | 0.021064 |
| 51 | generalization | burgers_widevis_l3target_d48 | loss3 | 0.015944 | 0.015919 | 0.012852 | 0.004576 | 0.107579 | 0.008192 |
| 52 | generalization | burgers_widevis_l3target_d49 | loss3 | 0.037051 | 0.022288 | 0.027553 | 0.009736 | 0.084259 | 0.021315 |

## 3. Clean Relative L2: 52 Datasets x 6 Models

| order | split | dataset_id | best | baseline | loss1 | loss2 | loss3 | random_clean_y | random_solver_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | train | train_original_gaussian_corr0p03 | loss1 | 0.016815 | 0.001447 | 0.002706 | 0.008117 | 0.014322 | 0.002651 |
| 2 | test | test_original_gaussian_corr0p03 | loss1 | 0.017514 | 0.001612 | 0.002880 | 0.008325 | 0.134975 | 0.002919 |
| 3 | generalization | burgers_widevis_l3target_d00 | loss3 | 0.040852 | 0.029363 | 0.033117 | 0.016857 | 0.162367 | 0.026825 |
| 4 | generalization | burgers_widevis_l3target_d01 | loss3 | 0.054399 | 0.039032 | 0.043805 | 0.022327 | 0.190958 | 0.038078 |
| 5 | generalization | burgers_widevis_l3target_d02 | loss3 | 0.034575 | 0.016573 | 0.018308 | 0.009570 | 0.114376 | 0.017491 |
| 6 | generalization | burgers_widevis_l3target_d03 | loss3 | 0.031060 | 0.018917 | 0.021479 | 0.010998 | 0.128713 | 0.018045 |
| 7 | generalization | burgers_widevis_l3target_d04 | loss3 | 0.051356 | 0.042307 | 0.046184 | 0.024672 | 0.171564 | 0.038621 |
| 8 | generalization | burgers_widevis_l3target_d05 | loss3 | 0.031389 | 0.018317 | 0.019396 | 0.010719 | 0.104892 | 0.018412 |
| 9 | generalization | burgers_widevis_l3target_d06 | loss3 | 0.032797 | 0.021319 | 0.024410 | 0.012699 | 0.138408 | 0.018779 |
| 10 | generalization | burgers_widevis_l3target_d07 | loss3 | 0.031765 | 0.022202 | 0.024201 | 0.013191 | 0.112918 | 0.020298 |
| 11 | generalization | burgers_widevis_l3target_d08 | loss3 | 0.100065 | 0.070278 | 0.074775 | 0.041390 | 0.255107 | 0.073455 |
| 12 | generalization | burgers_widevis_l3target_d09 | loss3 | 0.082095 | 0.068375 | 0.072355 | 0.040607 | 0.224044 | 0.068460 |
| 13 | generalization | burgers_widevis_l3target_d10 | loss3 | 0.041524 | 0.031896 | 0.035153 | 0.019415 | 0.148963 | 0.027833 |
| 14 | generalization | burgers_widevis_l3target_d11 | loss3 | 0.025492 | 0.014647 | 0.016822 | 0.008954 | 0.115810 | 0.013038 |
| 15 | generalization | burgers_widevis_l3target_d12 | loss3 | 0.062431 | 0.031456 | 0.036238 | 0.019257 | 0.235116 | 0.034185 |
| 16 | generalization | burgers_widevis_l3target_d13 | loss3 | 0.048587 | 0.038219 | 0.040539 | 0.022788 | 0.165717 | 0.034181 |
| 17 | generalization | burgers_widevis_l3target_d14 | loss3 | 0.080737 | 0.062830 | 0.064965 | 0.038455 | 0.222171 | 0.061967 |
| 18 | generalization | burgers_widevis_l3target_d15 | loss3 | 0.031068 | 0.016462 | 0.017514 | 0.010143 | 0.104658 | 0.016672 |
| 19 | generalization | burgers_widevis_l3target_d16 | loss3 | 0.063610 | 0.028566 | 0.033843 | 0.018147 | 0.222291 | 0.031648 |
| 20 | generalization | burgers_widevis_l3target_d17 | loss3 | 0.164134 | 0.103489 | 0.102434 | 0.065134 | 0.339045 | 0.114231 |
| 21 | generalization | burgers_widevis_l3target_d18 | loss3 | 0.070424 | 0.031630 | 0.034416 | 0.020271 | 0.139632 | 0.038271 |
| 22 | generalization | burgers_widevis_l3target_d19 | loss3 | 0.128807 | 0.066814 | 0.072625 | 0.044721 | 0.302645 | 0.079256 |
| 23 | generalization | burgers_widevis_l3target_d20 | loss3 | 0.118415 | 0.065505 | 0.072656 | 0.043506 | 0.299387 | 0.076922 |
| 24 | generalization | burgers_widevis_l3target_d21 | loss3 | 0.040899 | 0.023417 | 0.023821 | 0.015933 | 0.101019 | 0.024919 |
| 25 | generalization | burgers_widevis_l3target_d22 | loss3 | 0.039524 | 0.029348 | 0.031332 | 0.016911 | 0.152463 | 0.025621 |
| 26 | generalization | burgers_widevis_l3target_d23 | loss3 | 0.077308 | 0.051930 | 0.055685 | 0.031000 | 0.228842 | 0.053523 |
| 27 | generalization | burgers_widevis_l3target_d24 | loss3 | 0.039408 | 0.031243 | 0.032980 | 0.018632 | 0.143033 | 0.026680 |
| 28 | generalization | burgers_widevis_l3target_d25 | loss3 | 0.083546 | 0.063590 | 0.066625 | 0.037448 | 0.229626 | 0.063597 |
| 29 | generalization | burgers_widevis_l3target_d26 | loss3 | 0.032754 | 0.022017 | 0.023922 | 0.013395 | 0.133187 | 0.018816 |
| 30 | generalization | burgers_widevis_l3target_d27 | loss3 | 0.066056 | 0.042757 | 0.044324 | 0.026288 | 0.204328 | 0.041667 |
| 31 | generalization | burgers_widevis_l3target_d28 | loss3 | 0.037251 | 0.026649 | 0.028989 | 0.016238 | 0.146271 | 0.023805 |
| 32 | generalization | burgers_widevis_l3target_d29 | loss3 | 0.030577 | 0.017156 | 0.018374 | 0.010558 | 0.104642 | 0.017204 |
| 33 | generalization | burgers_widevis_l3target_d30 | loss3 | 0.079030 | 0.061155 | 0.063327 | 0.037509 | 0.215938 | 0.060469 |
| 34 | generalization | burgers_widevis_l3target_d31 | loss3 | 0.072923 | 0.066212 | 0.066730 | 0.040751 | 0.199246 | 0.061993 |
| 35 | generalization | burgers_widevis_l3target_d32 | loss3 | 0.028715 | 0.018023 | 0.019008 | 0.011072 | 0.096063 | 0.017809 |
| 36 | generalization | burgers_widevis_l3target_d33 | loss3 | 0.032883 | 0.023127 | 0.025074 | 0.014183 | 0.128499 | 0.019510 |
| 37 | generalization | burgers_widevis_l3target_d34 | loss3 | 0.082117 | 0.054257 | 0.057268 | 0.033139 | 0.231573 | 0.055914 |
| 38 | generalization | burgers_widevis_l3target_d35 | loss3 | 0.075583 | 0.063861 | 0.065403 | 0.038879 | 0.216826 | 0.060878 |
| 39 | generalization | burgers_widevis_l3target_d36 | loss3 | 0.049647 | 0.032490 | 0.032452 | 0.014081 | 0.123461 | 0.030643 |
| 40 | generalization | burgers_widevis_l3target_d37 | loss3 | 0.037107 | 0.025178 | 0.028318 | 0.011468 | 0.049454 | 0.029670 |
| 41 | generalization | burgers_widevis_l3target_d38 | loss3 | 0.044598 | 0.040620 | 0.042718 | 0.019924 | 0.159712 | 0.033478 |
| 42 | generalization | burgers_widevis_l3target_d39 | loss3 | 0.045830 | 0.043613 | 0.044097 | 0.020895 | 0.084168 | 0.028352 |
| 43 | generalization | burgers_widevis_l3target_d40 | loss3 | 0.037705 | 0.035296 | 0.035744 | 0.017004 | 0.078641 | 0.023469 |
| 44 | generalization | burgers_widevis_l3target_d41 | loss3 | 0.092958 | 0.096485 | 0.094928 | 0.046865 | 0.205541 | 0.081304 |
| 45 | generalization | burgers_widevis_l3target_d42 | loss3 | 0.061954 | 0.060763 | 0.065817 | 0.029941 | 0.191092 | 0.053906 |
| 46 | generalization | burgers_widevis_l3target_d43 | loss3 | 0.038440 | 0.031551 | 0.032778 | 0.016237 | 0.142760 | 0.025261 |
| 47 | generalization | burgers_widevis_l3target_d44 | loss3 | 0.060989 | 0.059626 | 0.059732 | 0.030208 | 0.115052 | 0.038244 |
| 48 | generalization | burgers_widevis_l3target_d45 | loss3 | 0.022281 | 0.016434 | 0.017310 | 0.008742 | 0.140286 | 0.012631 |
| 49 | generalization | burgers_widevis_l3target_d46 | loss3 | 0.064629 | 0.030578 | 0.040118 | 0.018295 | 0.224045 | 0.033999 |
| 50 | generalization | burgers_widevis_l3target_d47 | loss3 | 0.084212 | 0.082649 | 0.077639 | 0.045410 | 0.162218 | 0.059596 |
| 51 | generalization | burgers_widevis_l3target_d48 | loss3 | 0.022285 | 0.022250 | 0.017963 | 0.006396 | 0.150366 | 0.011450 |
| 52 | generalization | burgers_widevis_l3target_d49 | loss3 | 0.080071 | 0.048071 | 0.059412 | 0.020965 | 0.181778 | 0.045959 |

## 4. Clean MSE: 52 Datasets x 6 Models

| order | split | dataset_id | best | baseline | loss1 | loss2 | loss3 | random_clean_y | random_solver_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | train | train_original_gaussian_corr0p03 | loss1 | 8.071e-05 | 5.990e-07 | 2.141e-06 | 1.938e-05 | 5.711e-05 | 1.974e-06 |
| 2 | test | test_original_gaussian_corr0p03 | loss1 | 9.108e-05 | 7.774e-07 | 2.493e-06 | 2.050e-05 | 0.005476 | 2.499e-06 |
| 3 | generalization | burgers_widevis_l3target_d00 | loss3 | 0.000468 | 0.000264 | 0.000333 | 8.344e-05 | 0.007224 | 0.000227 |
| 4 | generalization | burgers_widevis_l3target_d01 | loss3 | 0.000865 | 0.000472 | 0.000590 | 0.000150 | 0.009907 | 0.000453 |
| 5 | generalization | burgers_widevis_l3target_d02 | loss3 | 0.000777 | 0.000154 | 0.000186 | 5.055e-05 | 0.007045 | 0.000179 |
| 6 | generalization | burgers_widevis_l3target_d03 | loss3 | 0.000410 | 0.000146 | 0.000188 | 4.854e-05 | 0.006518 | 0.000138 |
| 7 | generalization | burgers_widevis_l3target_d04 | loss3 | 0.000687 | 0.000486 | 0.000571 | 0.000164 | 0.007647 | 0.000410 |
| 8 | generalization | burgers_widevis_l3target_d05 | loss3 | 0.000590 | 0.000182 | 0.000203 | 6.279e-05 | 0.005884 | 0.000189 |
| 9 | generalization | burgers_widevis_l3target_d06 | loss3 | 0.000293 | 0.000133 | 0.000172 | 4.618e-05 | 0.005264 | 0.000105 |
| 10 | generalization | burgers_widevis_l3target_d07 | loss3 | 0.000397 | 0.000197 | 0.000231 | 6.950e-05 | 0.005075 | 0.000167 |
| 11 | generalization | burgers_widevis_l3target_d08 | loss3 | 0.002995 | 0.001544 | 0.001719 | 0.000545 | 0.017778 | 0.001673 |
| 12 | generalization | burgers_widevis_l3target_d09 | loss3 | 0.001882 | 0.001286 | 0.001435 | 0.000460 | 0.013026 | 0.001319 |
| 13 | generalization | burgers_widevis_l3target_d10 | loss3 | 0.000448 | 0.000276 | 0.000330 | 0.000101 | 0.005801 | 0.000213 |
| 14 | generalization | burgers_widevis_l3target_d11 | loss3 | 0.000206 | 7.074e-05 | 9.372e-05 | 2.602e-05 | 0.004356 | 5.679e-05 |
| 15 | generalization | burgers_widevis_l3target_d12 | loss3 | 0.001248 | 0.000326 | 0.000418 | 0.000109 | 0.015256 | 0.000389 |
| 16 | generalization | burgers_widevis_l3target_d13 | loss3 | 0.000653 | 0.000405 | 0.000453 | 0.000144 | 0.007418 | 0.000332 |
| 17 | generalization | burgers_widevis_l3target_d14 | loss3 | 0.001900 | 0.001130 | 0.001197 | 0.000418 | 0.013263 | 0.001107 |
| 18 | generalization | burgers_widevis_l3target_d15 | loss3 | 0.000590 | 0.000148 | 0.000166 | 5.619e-05 | 0.005927 | 0.000156 |
| 19 | generalization | burgers_widevis_l3target_d16 | loss3 | 0.001440 | 0.000258 | 0.000368 | 0.000100 | 0.014200 | 0.000339 |
| 20 | generalization | burgers_widevis_l3target_d17 | loss3 | 0.003668 | 0.001468 | 0.001455 | 0.000572 | 0.015464 | 0.001867 |
| 21 | generalization | burgers_widevis_l3target_d18 | loss3 | 0.003910 | 0.000723 | 0.000881 | 0.000326 | 0.012216 | 0.001231 |
| 22 | generalization | burgers_widevis_l3target_d19 | loss3 | 0.005505 | 0.001501 | 0.001777 | 0.000689 | 0.026094 | 0.002285 |
| 23 | generalization | burgers_widevis_l3target_d20 | loss3 | 0.004673 | 0.001387 | 0.001701 | 0.000638 | 0.025825 | 0.002047 |
| 24 | generalization | burgers_widevis_l3target_d21 | loss3 | 0.001198 | 0.000339 | 0.000355 | 0.000161 | 0.006349 | 0.000414 |
| 25 | generalization | burgers_widevis_l3target_d22 | loss3 | 0.000430 | 0.000258 | 0.000287 | 8.342e-05 | 0.006321 | 0.000197 |
| 26 | generalization | burgers_widevis_l3target_d23 | loss3 | 0.001868 | 0.000795 | 0.000921 | 0.000275 | 0.014684 | 0.000869 |
| 27 | generalization | burgers_widevis_l3target_d24 | loss3 | 0.000419 | 0.000282 | 0.000309 | 9.827e-05 | 0.005550 | 0.000208 |
| 28 | generalization | burgers_widevis_l3target_d25 | loss3 | 0.002123 | 0.001172 | 0.001295 | 0.000413 | 0.014292 | 0.001190 |
| 29 | generalization | burgers_widevis_l3target_d26 | loss3 | 0.000288 | 0.000137 | 0.000160 | 4.971e-05 | 0.004772 | 9.963e-05 |
| 30 | generalization | burgers_widevis_l3target_d27 | loss3 | 0.001435 | 0.000568 | 0.000607 | 0.000207 | 0.012143 | 0.000584 |
| 31 | generalization | burgers_widevis_l3target_d28 | loss3 | 0.000389 | 0.000213 | 0.000250 | 7.756e-05 | 0.006076 | 0.000173 |
| 32 | generalization | burgers_widevis_l3target_d29 | loss3 | 0.000558 | 0.000163 | 0.000185 | 6.019e-05 | 0.005866 | 0.000167 |
| 33 | generalization | burgers_widevis_l3target_d30 | loss3 | 0.001754 | 0.001088 | 0.001182 | 0.000401 | 0.012694 | 0.001073 |
| 34 | generalization | burgers_widevis_l3target_d31 | loss3 | 0.001562 | 0.001254 | 0.001280 | 0.000464 | 0.010854 | 0.001138 |
| 35 | generalization | burgers_widevis_l3target_d32 | loss3 | 0.000440 | 0.000170 | 0.000188 | 6.301e-05 | 0.004821 | 0.000167 |
| 36 | generalization | burgers_widevis_l3target_d33 | loss3 | 0.000288 | 0.000152 | 0.000176 | 5.617e-05 | 0.004493 | 0.000108 |
| 37 | generalization | burgers_widevis_l3target_d34 | loss3 | 0.002138 | 0.000892 | 0.000994 | 0.000333 | 0.015353 | 0.000977 |
| 38 | generalization | burgers_widevis_l3target_d35 | loss3 | 0.001649 | 0.001203 | 0.001256 | 0.000449 | 0.012421 | 0.001131 |
| 39 | generalization | burgers_widevis_l3target_d36 | loss3 | 0.001498 | 0.000648 | 0.000659 | 0.000118 | 0.008817 | 0.000581 |
| 40 | generalization | burgers_widevis_l3target_d37 | loss3 | 0.000778 | 0.000358 | 0.000453 | 7.438e-05 | 0.001396 | 0.000498 |
| 41 | generalization | burgers_widevis_l3target_d38 | loss3 | 0.000530 | 0.000486 | 0.000523 | 0.000110 | 0.006858 | 0.000329 |
| 42 | generalization | burgers_widevis_l3target_d39 | loss3 | 0.000531 | 0.000481 | 0.000491 | 0.000111 | 0.001858 | 0.000205 |
| 43 | generalization | burgers_widevis_l3target_d40 | loss3 | 0.000359 | 0.000315 | 0.000323 | 7.321e-05 | 0.001611 | 0.000140 |
| 44 | generalization | burgers_widevis_l3target_d41 | loss3 | 0.001213 | 0.001375 | 0.001347 | 0.000318 | 0.005710 | 0.001004 |
| 45 | generalization | burgers_widevis_l3target_d42 | loss3 | 0.001047 | 0.001067 | 0.001231 | 0.000257 | 0.009805 | 0.000851 |
| 46 | generalization | burgers_widevis_l3target_d43 | loss3 | 0.000390 | 0.000293 | 0.000308 | 7.169e-05 | 0.005437 | 0.000185 |
| 47 | generalization | burgers_widevis_l3target_d44 | loss3 | 0.000940 | 0.000897 | 0.000901 | 0.000231 | 0.003425 | 0.000375 |
| 48 | generalization | burgers_widevis_l3target_d45 | loss3 | 0.000138 | 7.717e-05 | 8.570e-05 | 1.978e-05 | 0.005121 | 4.746e-05 |
| 49 | generalization | burgers_widevis_l3target_d46 | loss3 | 0.001009 | 0.000260 | 0.000460 | 7.913e-05 | 0.011298 | 0.000317 |
| 50 | generalization | burgers_widevis_l3target_d47 | loss3 | 0.000896 | 0.000859 | 0.000761 | 0.000260 | 0.003355 | 0.000448 |
| 51 | generalization | burgers_widevis_l3target_d48 | loss3 | 0.000255 | 0.000253 | 0.000165 | 2.095e-05 | 0.011641 | 6.714e-05 |
| 52 | generalization | burgers_widevis_l3target_d49 | loss3 | 0.001383 | 0.000498 | 0.000762 | 9.575e-05 | 0.007127 | 0.000455 |

## 5. Full 52-Dataset Attack Table Available for Random Models

This is the full post-training adversarial attack-by-dataset table for `random_clean_y` and `random_solver_y`. It is not a six-model table because the comparable latest wideparam full-52 attack table for the old four models is not present in this source set; the six-model robustness mechanism comparison is the 25-sample manifest in sections 6-9.

| idx | split | dataset_id | n | init_random_clean_y | final_random_clean_y | increase_random_clean_y | delta_rms_random_clean_y | init_random_solver_y | final_random_solver_y | increase_random_solver_y | delta_rms_random_solver_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | train | train_original_gaussian_corr0p03_first50 | 50 | 5.420e-05 | 0.036903 | 0.036849 | 0.114878 | 1.804e-06 | 0.001463 | 0.001461 | 0.120000 |
| 1 | test | test_original_gaussian_corr0p03 | 150 | 0.005476 | 0.044544 | 0.039067 | 0.114558 | 2.499e-06 | 0.001714 | 0.001712 | 0.120000 |
| 2 | generalization | burgers_widevis_l3target_d00 | 200 | 0.007224 | 0.044384 | 0.037160 | 0.117064 | 0.000227 | 0.006222 | 0.005995 | 0.120000 |
| 3 | generalization | burgers_widevis_l3target_d01 | 200 | 0.009907 | 0.052927 | 0.043020 | 0.119353 | 0.000453 | 0.008380 | 0.007927 | 0.120000 |
| 4 | generalization | burgers_widevis_l3target_d02 | 200 | 0.007045 | 0.041189 | 0.034144 | 0.118556 | 0.000179 | 0.007463 | 0.007284 | 0.120000 |
| 5 | generalization | burgers_widevis_l3target_d03 | 200 | 0.006518 | 0.038976 | 0.032457 | 0.115327 | 0.000138 | 0.005124 | 0.004986 | 0.120000 |
| 6 | generalization | burgers_widevis_l3target_d04 | 200 | 0.007647 | 0.043910 | 0.036264 | 0.118128 | 0.000410 | 0.007156 | 0.006745 | 0.120000 |
| 7 | generalization | burgers_widevis_l3target_d05 | 200 | 0.005884 | 0.036928 | 0.031044 | 0.118440 | 0.000189 | 0.005634 | 0.005445 | 0.120000 |
| 8 | generalization | burgers_widevis_l3target_d06 | 200 | 0.005264 | 0.035197 | 0.029933 | 0.111404 | 0.000105 | 0.004178 | 0.004074 | 0.120000 |
| 9 | generalization | burgers_widevis_l3target_d07 | 200 | 0.005075 | 0.033645 | 0.028570 | 0.114220 | 0.000167 | 0.004441 | 0.004274 | 0.120000 |
| 10 | generalization | burgers_widevis_l3target_d08 | 200 | 0.017778 | 0.066848 | 0.049070 | 0.119617 | 0.001673 | 0.017360 | 0.015687 | 0.120000 |
| 11 | generalization | burgers_widevis_l3target_d09 | 200 | 0.013026 | 0.055955 | 0.042929 | 0.119795 | 0.001319 | 0.013201 | 0.011882 | 0.120000 |
| 12 | generalization | burgers_widevis_l3target_d10 | 200 | 0.005801 | 0.036980 | 0.031179 | 0.114289 | 0.000213 | 0.005341 | 0.005128 | 0.120000 |
| 13 | generalization | burgers_widevis_l3target_d11 | 200 | 0.004356 | 0.028274 | 0.023918 | 0.106066 | 5.679e-05 | 0.002801 | 0.002745 | 0.120000 |
| 14 | generalization | burgers_widevis_l3target_d12 | 200 | 0.015256 | 0.063326 | 0.048071 | 0.119965 | 0.000389 | 0.010552 | 0.010162 | 0.120000 |
| 15 | generalization | burgers_widevis_l3target_d13 | 200 | 0.007418 | 0.045054 | 0.037636 | 0.118003 | 0.000332 | 0.006736 | 0.006404 | 0.120000 |
| 16 | generalization | burgers_widevis_l3target_d14 | 200 | 0.013263 | 0.058124 | 0.044861 | 0.119772 | 0.001107 | 0.012822 | 0.011716 | 0.120000 |
| 17 | generalization | burgers_widevis_l3target_d15 | 200 | 0.005927 | 0.036976 | 0.031048 | 0.117101 | 0.000156 | 0.005988 | 0.005832 | 0.120000 |
| 18 | generalization | burgers_widevis_l3target_d16 | 200 | 0.014200 | 0.062908 | 0.048708 | 0.119841 | 0.000339 | 0.011105 | 0.010767 | 0.120000 |
| 19 | generalization | burgers_widevis_l3target_d17 | 200 | 0.015464 | 0.059441 | 0.043977 | 0.120000 | 0.001867 | 0.020328 | 0.018461 | 0.119906 |
| 20 | generalization | burgers_widevis_l3target_d18 | 200 | 0.012216 | 0.049434 | 0.037217 | 0.119056 | 0.001231 | 0.020029 | 0.018798 | 0.120000 |
| 21 | generalization | burgers_widevis_l3target_d19 | 200 | 0.026094 | 0.079518 | 0.053424 | 0.120000 | 0.002285 | 0.026064 | 0.023779 | 0.119989 |
| 22 | generalization | burgers_widevis_l3target_d20 | 200 | 0.025825 | 0.079291 | 0.053466 | 0.120000 | 0.002047 | 0.023238 | 0.021191 | 0.120000 |
| 23 | generalization | burgers_widevis_l3target_d21 | 200 | 0.006349 | 0.037307 | 0.030958 | 0.119262 | 0.000414 | 0.009094 | 0.008680 | 0.120000 |
| 24 | generalization | burgers_widevis_l3target_d22 | 200 | 0.006321 | 0.039052 | 0.032731 | 0.113673 | 0.000197 | 0.005122 | 0.004925 | 0.120000 |
| 25 | generalization | burgers_widevis_l3target_d23 | 200 | 0.014684 | 0.060836 | 0.046151 | 0.119623 | 0.000869 | 0.012272 | 0.011403 | 0.120000 |
| 26 | generalization | burgers_widevis_l3target_d24 | 200 | 0.005550 | 0.035942 | 0.030391 | 0.112679 | 0.000208 | 0.004967 | 0.004759 | 0.120000 |
| 27 | generalization | burgers_widevis_l3target_d25 | 200 | 0.014292 | 0.058687 | 0.044395 | 0.119886 | 0.001190 | 0.013295 | 0.012105 | 0.120000 |
| 28 | generalization | burgers_widevis_l3target_d26 | 200 | 0.004772 | 0.032364 | 0.027592 | 0.109947 | 9.963e-05 | 0.003857 | 0.003757 | 0.120000 |
| 29 | generalization | burgers_widevis_l3target_d27 | 200 | 0.012143 | 0.057431 | 0.045287 | 0.118671 | 0.000584 | 0.010421 | 0.009837 | 0.120000 |
| 30 | generalization | burgers_widevis_l3target_d28 | 200 | 0.006076 | 0.040079 | 0.034003 | 0.116218 | 0.000173 | 0.004825 | 0.004652 | 0.120000 |
| 31 | generalization | burgers_widevis_l3target_d29 | 200 | 0.005866 | 0.036227 | 0.030361 | 0.117918 | 0.000167 | 0.005175 | 0.005008 | 0.120000 |
| 32 | generalization | burgers_widevis_l3target_d30 | 200 | 0.012694 | 0.055859 | 0.043165 | 0.119528 | 0.001073 | 0.011390 | 0.010317 | 0.120000 |
| 33 | generalization | burgers_widevis_l3target_d31 | 200 | 0.010854 | 0.051359 | 0.040505 | 0.118817 | 0.001138 | 0.011207 | 0.010069 | 0.120000 |
| 34 | generalization | burgers_widevis_l3target_d32 | 200 | 0.004821 | 0.033177 | 0.028356 | 0.117053 | 0.000167 | 0.004080 | 0.003913 | 0.120000 |
| 35 | generalization | burgers_widevis_l3target_d33 | 200 | 0.004493 | 0.027963 | 0.023470 | 0.104391 | 0.000108 | 0.003357 | 0.003249 | 0.120000 |
| 36 | generalization | burgers_widevis_l3target_d34 | 200 | 0.015353 | 0.062544 | 0.047191 | 0.119767 | 0.000977 | 0.014553 | 0.013576 | 0.119957 |
| 37 | generalization | burgers_widevis_l3target_d35 | 200 | 0.012421 | 0.056014 | 0.043593 | 0.119307 | 0.001131 | 0.011754 | 0.010623 | 0.120000 |
| 38 | generalization | burgers_widevis_l3target_d36 | 200 | 0.008817 | 0.031629 | 0.022812 | 0.120000 | 0.000581 | 0.006119 | 0.005539 | 0.120000 |
| 39 | generalization | burgers_widevis_l3target_d37 | 200 | 0.001396 | 0.025631 | 0.024235 | 0.120000 | 0.000498 | 0.003818 | 0.003320 | 0.120000 |
| 40 | generalization | burgers_widevis_l3target_d38 | 200 | 0.006858 | 0.038166 | 0.031308 | 0.119412 | 0.000329 | 0.006107 | 0.005779 | 0.120000 |
| 41 | generalization | burgers_widevis_l3target_d39 | 200 | 0.001858 | 0.018898 | 0.017041 | 0.106214 | 0.000205 | 0.002300 | 0.002095 | 0.120000 |
| 42 | generalization | burgers_widevis_l3target_d40 | 200 | 0.001611 | 0.015920 | 0.014308 | 0.099943 | 0.000140 | 0.001975 | 0.001835 | 0.120000 |
| 43 | generalization | burgers_widevis_l3target_d41 | 200 | 0.005710 | 0.031754 | 0.026044 | 0.120000 | 0.001004 | 0.008689 | 0.007685 | 0.120000 |
| 44 | generalization | burgers_widevis_l3target_d42 | 200 | 0.009805 | 0.044293 | 0.034488 | 0.120000 | 0.000851 | 0.008304 | 0.007453 | 0.120000 |
| 45 | generalization | burgers_widevis_l3target_d43 | 200 | 0.005437 | 0.032984 | 0.027548 | 0.116136 | 0.000185 | 0.004983 | 0.004798 | 0.120000 |
| 46 | generalization | burgers_widevis_l3target_d44 | 200 | 0.003425 | 0.027211 | 0.023786 | 0.119233 | 0.000375 | 0.002957 | 0.002582 | 0.120000 |
| 47 | generalization | burgers_widevis_l3target_d45 | 200 | 0.005121 | 0.027378 | 0.022257 | 0.110135 | 4.746e-05 | 0.001271 | 0.001223 | 0.120000 |
| 48 | generalization | burgers_widevis_l3target_d46 | 200 | 0.011298 | 0.049739 | 0.038442 | 0.120000 | 0.000317 | 0.007657 | 0.007340 | 0.120000 |
| 49 | generalization | burgers_widevis_l3target_d47 | 200 | 0.003355 | 0.025131 | 0.021776 | 0.120000 | 0.000448 | 0.003645 | 0.003197 | 0.120000 |
| 50 | generalization | burgers_widevis_l3target_d48 | 200 | 0.011641 | 0.055814 | 0.044173 | 0.120000 | 6.715e-05 | 0.006406 | 0.006339 | 0.120000 |
| 51 | generalization | burgers_widevis_l3target_d49 | 200 | 0.007127 | 0.031738 | 0.024611 | 0.120000 | 0.000455 | 0.003152 | 0.002696 | 0.120000 |

## 6. Robustness/SVD Sample Manifest: 25 Samples

These are the exact samples used for the six-model SVD/attack mechanism tables below.

| sample | split | dataset_id | local_index |
| --- | --- | --- | --- |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 |
| 3 | train | train_original_gaussian_corr0p03 | 1 |
| 4 | train | train_original_gaussian_corr0p03 | 128 |
| 5 | test | test_original_gaussian_corr0p03 | 45 |
| 6 | test | test_original_gaussian_corr0p03 | 71 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 |

## 7. Robustness Indicator 1: Attack Delta and Loss Increase

| sample | split | dataset_id | local | model | init_mse | final_mse | increase | delta_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | baseline | 0.001187 | 0.034506 | 0.033319 | 0.120000 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss1 | 0.001197 | 0.012053 | 0.010857 | 0.120000 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss2 | 0.001666 | 0.009327 | 0.007661 | 0.120000 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss3 | 0.000204 | 0.006532 | 0.006328 | 0.120000 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | random_clean_y | 0.011164 | 0.058306 | 0.047142 | 0.120000 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | random_solver_y | 0.001358 | 0.025460 | 0.024102 | 0.120000 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | baseline | 0.000612 | 0.004985 | 0.004372 | 0.120000 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss1 | 0.000745 | 0.008288 | 0.007543 | 0.120000 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss2 | 0.000363 | 0.007108 | 0.006744 | 0.120000 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss3 | 0.000256 | 0.004224 | 0.003968 | 0.120000 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | random_clean_y | 0.009404 | 0.047789 | 0.038385 | 0.120000 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | random_solver_y | 0.000328 | 0.007556 | 0.007228 | 0.120000 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | baseline | 0.000473 | 0.009182 | 0.008709 | 0.120000 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss1 | 0.000302 | 0.003016 | 0.002715 | 0.120000 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss2 | 0.000263 | 0.002617 | 0.002353 | 0.120000 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss3 | 3.577e-05 | 0.001504 | 0.001468 | 0.120000 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | random_clean_y | 0.006376 | 0.038939 | 0.032563 | 0.120000 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | random_solver_y | 0.000218 | 0.002586 | 0.002367 | 0.120000 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | baseline | 4.476e-05 | 0.001518 | 0.001474 | 0.120000 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss1 | 3.252e-07 | 0.000679 | 0.000679 | 0.120000 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss2 | 2.931e-06 | 0.001253 | 0.001250 | 0.120000 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss3 | 1.954e-05 | 0.000769 | 0.000750 | 0.120000 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | random_clean_y | 3.209e-05 | 0.034426 | 0.034394 | 0.110938 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | random_solver_y | 1.250e-06 | 0.000852 | 0.000851 | 0.120000 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | baseline | 4.994e-05 | 0.013576 | 0.013526 | 0.120000 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss1 | 6.593e-07 | 0.001570 | 0.001569 | 0.120000 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss2 | 2.084e-06 | 0.001582 | 0.001580 | 0.120000 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss3 | 3.348e-05 | 0.000627 | 0.000594 | 0.120000 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | random_clean_y | 0.000992 | 0.036337 | 0.035345 | 0.090213 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | random_solver_y | 2.389e-06 | 0.001679 | 0.001676 | 0.120000 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | baseline | 2.926e-05 | 0.002359 | 0.002329 | 0.120000 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss1 | 7.260e-07 | 0.001786 | 0.001786 | 0.120000 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss2 | 2.650e-06 | 0.001858 | 0.001856 | 0.120000 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss3 | 1.809e-05 | 0.000963 | 0.000945 | 0.120000 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | random_clean_y | 0.012422 | 0.045477 | 0.033056 | 0.120000 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | random_solver_y | 1.041e-06 | 0.001895 | 0.001894 | 0.120000 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | baseline | 7.155e-05 | 0.002524 | 0.002452 | 0.120000 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss1 | 3.967e-07 | 0.000936 | 0.000935 | 0.120000 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss2 | 1.709e-06 | 0.001782 | 0.001781 | 0.120000 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss3 | 2.183e-05 | 0.000403 | 0.000381 | 0.120000 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | random_clean_y | 0.009022 | 0.051190 | 0.042168 | 0.120000 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | random_solver_y | 3.015e-06 | 0.001653 | 0.001650 | 0.120000 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | baseline | 0.000111 | 0.004941 | 0.004831 | 0.120000 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss1 | 7.335e-05 | 0.005519 | 0.005445 | 0.120000 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss2 | 8.061e-05 | 0.004955 | 0.004874 | 0.120000 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss3 | 3.398e-05 | 0.001001 | 0.000967 | 0.120000 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | random_clean_y | 0.007374 | 0.047246 | 0.039873 | 0.120000 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | random_solver_y | 5.151e-05 | 0.005206 | 0.005155 | 0.120000 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | baseline | 0.004422 | 0.011838 | 0.007416 | 0.120000 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss1 | 0.004861 | 0.015522 | 0.010661 | 0.120000 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss2 | 0.005907 | 0.017116 | 0.011209 | 0.120000 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss3 | 0.001728 | 0.009513 | 0.007784 | 0.120000 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | random_clean_y | 0.005358 | 0.040751 | 0.035393 | 0.120000 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | random_solver_y | 0.005318 | 0.015801 | 0.010483 | 0.120000 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | baseline | 0.000568 | 0.010015 | 0.009447 | 0.120000 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss1 | 6.048e-05 | 0.003478 | 0.003418 | 0.120000 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss2 | 9.148e-05 | 0.002272 | 0.002181 | 0.120000 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss3 | 6.931e-05 | 0.001231 | 0.001162 | 0.120000 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | random_clean_y | 0.008719 | 0.048343 | 0.039624 | 0.120000 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | random_solver_y | 9.722e-05 | 0.003128 | 0.003031 | 0.120000 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | baseline | 0.000682 | 0.004792 | 0.004110 | 0.120000 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss1 | 0.000132 | 0.003129 | 0.002997 | 0.120000 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss2 | 0.000188 | 0.009297 | 0.009109 | 0.120000 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss3 | 6.105e-05 | 0.001519 | 0.001457 | 0.120000 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | random_clean_y | 0.004549 | 0.026417 | 0.021868 | 0.088545 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | random_solver_y | 7.677e-05 | 0.008952 | 0.008876 | 0.120000 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | baseline | 0.000574 | 0.009634 | 0.009060 | 0.120000 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss1 | 0.003635 | 0.016343 | 0.012708 | 0.120000 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss2 | 0.002027 | 0.009807 | 0.007780 | 0.120000 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss3 | 0.000817 | 0.010284 | 0.009467 | 0.120000 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | random_clean_y | 0.010287 | 0.043007 | 0.032719 | 0.120000 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | random_solver_y | 0.002321 | 0.012621 | 0.010300 | 0.120000 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | baseline | 0.003430 | 0.022907 | 0.019477 | 0.120000 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss1 | 0.000753 | 0.013225 | 0.012471 | 0.120000 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss2 | 0.000920 | 0.013902 | 0.012982 | 0.120000 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss3 | 0.000633 | 0.008441 | 0.007808 | 0.120000 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | random_clean_y | 0.022387 | 0.077425 | 0.055038 | 0.120000 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | random_solver_y | 0.001163 | 0.010405 | 0.009242 | 0.120000 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | baseline | 0.001011 | 0.021369 | 0.020358 | 0.120000 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss1 | 7.863e-05 | 0.006812 | 0.006733 | 0.120000 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss2 | 0.000214 | 0.007848 | 0.007634 | 0.120000 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss3 | 3.473e-05 | 0.002178 | 0.002144 | 0.120000 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | random_clean_y | 0.014434 | 0.073149 | 0.058714 | 0.120000 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | random_solver_y | 0.000142 | 0.008351 | 0.008209 | 0.120000 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | baseline | 0.000183 | 0.002431 | 0.002248 | 0.120000 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss1 | 8.047e-05 | 0.002570 | 0.002490 | 0.120000 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss2 | 8.934e-05 | 0.003216 | 0.003127 | 0.120000 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss3 | 2.061e-05 | 0.000884 | 0.000863 | 0.120000 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | random_clean_y | 0.002778 | 0.029770 | 0.026992 | 0.120000 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | random_solver_y | 8.132e-05 | 0.002897 | 0.002815 | 0.120000 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | baseline | 0.001214 | 0.004935 | 0.003721 | 0.120000 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss1 | 0.000697 | 0.004748 | 0.004051 | 0.120000 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss2 | 0.000573 | 0.003471 | 0.002898 | 0.120000 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss3 | 0.000217 | 0.002667 | 0.002450 | 0.120000 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | random_clean_y | 0.003778 | 0.030500 | 0.026722 | 0.120000 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | random_solver_y | 0.000706 | 0.005047 | 0.004341 | 0.120000 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | baseline | 0.001546 | 0.011520 | 0.009974 | 0.120000 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss1 | 0.001257 | 0.009761 | 0.008504 | 0.120000 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss2 | 0.001735 | 0.012874 | 0.011138 | 0.120000 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss3 | 0.000393 | 0.005960 | 0.005567 | 0.120000 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | random_clean_y | 0.006606 | 0.039238 | 0.032632 | 0.120000 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | random_solver_y | 0.001434 | 0.012325 | 0.010891 | 0.120000 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | baseline | 0.001167 | 0.010517 | 0.009351 | 0.120000 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss1 | 0.001048 | 0.010744 | 0.009696 | 0.120000 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss2 | 0.001129 | 0.010208 | 0.009079 | 0.120000 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss3 | 0.000236 | 0.004493 | 0.004257 | 0.120000 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | random_clean_y | 0.003523 | 0.031853 | 0.028330 | 0.120000 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | random_solver_y | 0.001073 | 0.010646 | 0.009573 | 0.120000 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | baseline | 0.001368 | 0.017550 | 0.016182 | 0.120000 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss1 | 0.002022 | 0.009668 | 0.007646 | 0.120000 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss2 | 0.001905 | 0.007707 | 0.005801 | 0.120000 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss3 | 0.000218 | 0.002889 | 0.002671 | 0.120000 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | random_clean_y | 0.012537 | 0.063221 | 0.050684 | 0.120000 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | random_solver_y | 0.001492 | 0.009319 | 0.007827 | 0.120000 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | baseline | 0.001167 | 0.008062 | 0.006895 | 0.120000 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss1 | 0.000173 | 0.007072 | 0.006899 | 0.120000 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss2 | 0.000281 | 0.005258 | 0.004977 | 0.120000 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss3 | 8.669e-05 | 0.001940 | 0.001853 | 0.120000 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | random_clean_y | 0.030511 | 0.077479 | 0.046968 | 0.120000 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | random_solver_y | 0.000101 | 0.004688 | 0.004587 | 0.120000 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | baseline | 0.004585 | 0.022403 | 0.017818 | 0.120000 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss1 | 0.000816 | 0.007185 | 0.006369 | 0.120000 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss2 | 0.001181 | 0.007476 | 0.006296 | 0.120000 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss3 | 0.000392 | 0.004514 | 0.004122 | 0.120000 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | random_clean_y | 0.021850 | 0.063975 | 0.042125 | 0.120000 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | random_solver_y | 0.000899 | 0.018936 | 0.018037 | 0.120000 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | baseline | 0.000210 | 0.006447 | 0.006237 | 0.120000 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss1 | 3.417e-05 | 0.003119 | 0.003084 | 0.120000 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss2 | 4.894e-05 | 0.003211 | 0.003162 | 0.120000 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss3 | 3.477e-05 | 0.001070 | 0.001035 | 0.120000 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | random_clean_y | 0.012876 | 0.056817 | 0.043942 | 0.120000 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | random_solver_y | 2.358e-05 | 0.004363 | 0.004339 | 0.120000 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | baseline | 0.000797 | 0.008097 | 0.007301 | 0.120000 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss1 | 0.001305 | 0.010675 | 0.009370 | 0.120000 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss2 | 0.001296 | 0.010390 | 0.009094 | 0.120000 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss3 | 0.000212 | 0.005347 | 0.005136 | 0.120000 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | random_clean_y | 0.004206 | 0.031850 | 0.027645 | 0.120000 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | random_solver_y | 0.000897 | 0.010760 | 0.009863 | 0.120000 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | baseline | 0.000803 | 0.005500 | 0.004697 | 0.120000 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss1 | 0.000371 | 0.003648 | 0.003276 | 0.120000 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss2 | 0.000443 | 0.003060 | 0.002617 | 0.120000 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss3 | 7.788e-05 | 0.001586 | 0.001508 | 0.120000 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | random_clean_y | 0.001174 | 0.024024 | 0.022850 | 0.120000 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | random_solver_y | 0.000505 | 0.003665 | 0.003159 | 0.120000 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | baseline | 0.001058 | 0.004861 | 0.003803 | 0.120000 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss1 | 0.000933 | 0.004980 | 0.004047 | 0.120000 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss2 | 0.000982 | 0.004386 | 0.003403 | 0.120000 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss3 | 0.000243 | 0.002138 | 0.001895 | 0.120000 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | random_clean_y | 0.003421 | 0.023269 | 0.019848 | 0.120000 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | random_solver_y | 0.000314 | 0.002897 | 0.002583 | 0.120000 |

## 8. Robustness Indicator 2: Error-Jacobian/SVD Scalars

| sample | split | dataset_id | local | model | err_spec | err_fro | err_eff_rank | model_spec | solver_spec | top_err_sv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | baseline | 2.368024 | 3.741819 | 8.527455 | 2.636727 | 3.270614 | 2.368024 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss1 | 3.025912 | 3.626148 | 4.154918 | 3.317314 | 3.270614 | 3.025912 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss2 | 3.076558 | 3.838913 | 4.777933 | 2.983047 | 3.270614 | 3.076558 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss3 | 1.069037 | 1.787472 | 8.137996 | 3.275095 | 3.270614 | 1.069037 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | random_clean_y | 3.149859 | 5.779617 | 8.468183 | 2.094794 | 3.270614 | 3.149859 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | random_solver_y | 3.176701 | 3.825829 | 3.765998 | 3.041676 | 3.270614 | 3.176701 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | baseline | 1.712117 | 3.083574 | 9.513850 | 3.233032 | 3.476166 | 1.712117 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss1 | 1.999307 | 3.701908 | 7.698343 | 3.663870 | 3.476166 | 1.999307 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss2 | 1.977068 | 2.987887 | 7.833975 | 3.686292 | 3.476166 | 1.977068 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss3 | 1.701135 | 2.241785 | 4.916453 | 3.569335 | 3.476166 | 1.701135 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | random_clean_y | 3.517420 | 7.021082 | 7.392356 | 3.740091 | 3.476166 | 3.517420 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | random_solver_y | 1.428675 | 2.639607 | 7.420595 | 3.719172 | 3.476166 | 1.428675 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | baseline | 1.953882 | 2.789339 | 7.777084 | 2.596158 | 2.992355 | 1.953882 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss1 | 1.224193 | 2.482901 | 9.084983 | 2.633026 | 2.992355 | 1.224193 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss2 | 1.216430 | 2.262923 | 9.311051 | 2.714572 | 2.992355 | 1.216430 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss3 | 0.420253 | 0.933419 | 12.558880 | 3.027957 | 2.992355 | 0.420253 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | random_clean_y | 3.142629 | 6.077037 | 8.291551 | 2.447746 | 2.992355 | 3.142629 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | random_solver_y | 1.054541 | 1.998419 | 7.755245 | 2.816016 | 2.992355 | 1.054541 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | baseline | 0.562423 | 1.076882 | 38.717285 | 3.541703 | 3.675378 | 0.562423 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss1 | 0.230608 | 0.623416 | 96.810653 | 3.667327 | 3.675378 | 0.230608 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss2 | 0.298257 | 0.689725 | 39.422768 | 3.665086 | 3.675378 | 0.298257 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss3 | 0.671966 | 0.802107 | 3.066006 | 3.620701 | 3.675378 | 0.671966 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | random_clean_y | 3.683963 | 5.903705 | 4.816051 | 0.889530 | 3.675378 | 3.683963 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | random_solver_y | 0.199889 | 0.451108 | 12.694174 | 3.658550 | 3.675378 | 0.199889 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | baseline | 0.713240 | 1.273662 | 15.362671 | 4.643264 | 4.924169 | 0.713240 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss1 | 0.282315 | 0.613508 | 62.645233 | 4.910130 | 4.924169 | 0.282315 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss2 | 0.311704 | 0.655804 | 33.500847 | 4.889284 | 4.924169 | 0.311704 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss3 | 1.072046 | 1.185242 | 2.112405 | 4.880766 | 4.924169 | 1.072046 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | random_clean_y | 4.780229 | 6.615272 | 3.472405 | 0.955718 | 4.924169 | 4.780229 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | random_solver_y | 0.364570 | 0.623250 | 9.378031 | 4.903430 | 4.924169 | 0.364570 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | baseline | 0.435788 | 1.064090 | 32.427841 | 4.731237 | 4.731434 | 0.435788 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss1 | 0.295706 | 0.605026 | 47.587482 | 4.705448 | 4.731434 | 0.295706 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss2 | 0.303508 | 0.642141 | 31.100905 | 4.695914 | 4.731434 | 0.303508 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss3 | 0.852245 | 0.894250 | 1.775264 | 4.687473 | 4.731434 | 0.852245 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | random_clean_y | 5.599870 | 7.777482 | 5.540532 | 4.186464 | 4.731434 | 5.599870 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | random_solver_y | 0.268098 | 0.497182 | 11.629324 | 4.728409 | 4.731434 | 0.268098 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | baseline | 1.063944 | 1.560530 | 10.112319 | 4.500153 | 4.678557 | 1.063944 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss1 | 0.300514 | 0.650856 | 47.600076 | 4.673946 | 4.678557 | 0.300514 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss2 | 0.315856 | 0.699654 | 28.329585 | 4.657341 | 4.678557 | 0.315856 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss3 | 0.797558 | 0.949276 | 2.498503 | 4.599768 | 4.678557 | 0.797558 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | random_clean_y | 4.263829 | 7.509106 | 5.484927 | 2.746551 | 4.678557 | 4.263829 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | random_solver_y | 0.360125 | 0.614176 | 8.635656 | 4.657614 | 4.678557 | 0.360125 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | baseline | 0.982439 | 1.783218 | 15.914028 | 3.837820 | 3.760348 | 0.982439 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss1 | 0.771032 | 1.425225 | 13.287556 | 3.761181 | 3.760348 | 0.771032 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss2 | 0.782679 | 1.495357 | 11.817193 | 3.695495 | 3.760348 | 0.782679 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss3 | 0.797000 | 1.086109 | 4.192063 | 3.665734 | 3.760348 | 0.797000 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | random_clean_y | 3.177350 | 6.344445 | 7.294403 | 2.141581 | 3.760348 | 3.177350 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | random_solver_y | 0.593918 | 1.171226 | 8.895280 | 3.735696 | 3.760348 | 0.593918 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | baseline | 5.450374 | 6.546980 | 3.732957 | 3.806096 | 3.986943 | 5.450374 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss1 | 4.335012 | 5.769881 | 4.991490 | 2.935766 | 3.986943 | 4.335012 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss2 | 4.820957 | 6.141759 | 4.289616 | 2.989348 | 3.986943 | 4.820957 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss3 | 4.127125 | 4.849156 | 2.849976 | 3.302784 | 3.986943 | 4.127125 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | random_clean_y | 3.912249 | 6.292123 | 6.659051 | 1.341260 | 3.986943 | 3.912249 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | random_solver_y | 5.105792 | 6.218225 | 3.472771 | 3.411943 | 3.986943 | 5.105792 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | baseline | 2.463942 | 3.400931 | 5.786836 | 4.269007 | 4.639815 | 2.463942 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss1 | 0.478847 | 1.114171 | 19.519721 | 4.658851 | 4.639815 | 0.478847 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss2 | 0.969504 | 1.437623 | 9.158768 | 4.658077 | 4.639815 | 0.969504 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss3 | 1.235990 | 1.530194 | 3.608366 | 4.686644 | 4.639815 | 1.235990 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | random_clean_y | 4.548937 | 7.188410 | 5.251371 | 1.382874 | 4.639815 | 4.548937 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | random_solver_y | 1.433141 | 1.909826 | 4.591284 | 4.495786 | 4.639815 | 1.433141 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | baseline | 2.517054 | 3.322429 | 5.745495 | 3.324345 | 3.765484 | 2.517054 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss1 | 0.871945 | 1.729667 | 12.986809 | 3.545009 | 3.765484 | 0.871945 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss2 | 0.844874 | 1.656652 | 11.968177 | 3.321080 | 3.765484 | 0.844874 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss3 | 0.611584 | 1.193624 | 9.851333 | 3.641921 | 3.765484 | 0.611584 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | random_clean_y | 4.459299 | 7.895197 | 7.003903 | 5.153374 | 3.765484 | 4.459299 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | random_solver_y | 0.688620 | 1.189274 | 9.004654 | 3.347703 | 3.765484 | 0.688620 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | baseline | 1.553642 | 2.917255 | 10.560410 | 2.997013 | 3.659257 | 1.553642 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss1 | 3.487212 | 5.407694 | 5.420834 | 3.872805 | 3.659257 | 3.487212 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss2 | 3.175533 | 4.626412 | 5.900675 | 4.179110 | 3.659257 | 3.175533 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss3 | 2.672006 | 3.608564 | 4.840270 | 3.271449 | 3.659257 | 2.672006 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | random_clean_y | 3.299453 | 6.408577 | 7.098969 | 1.893398 | 3.659257 | 3.299453 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | random_solver_y | 3.378216 | 5.344286 | 5.839916 | 4.833199 | 3.659257 | 3.378216 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | baseline | 4.499534 | 7.465172 | 4.205090 | 3.864217 | 4.027183 | 4.499534 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss1 | 2.450320 | 3.854779 | 7.024650 | 4.053382 | 4.027183 | 2.450320 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss2 | 3.188570 | 4.740439 | 5.817093 | 4.024368 | 4.027183 | 3.188570 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss3 | 2.374605 | 3.399082 | 4.664347 | 3.945812 | 4.027183 | 2.374605 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | random_clean_y | 3.880644 | 7.211033 | 5.681117 | 1.600951 | 4.027183 | 3.880644 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | random_solver_y | 3.450447 | 5.001066 | 4.584074 | 4.001041 | 4.027183 | 3.450447 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | baseline | 3.246409 | 4.436329 | 4.413752 | 4.113513 | 4.389275 | 3.246409 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss1 | 0.995321 | 1.595762 | 10.252787 | 4.288864 | 4.389275 | 0.995321 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss2 | 1.396208 | 2.498169 | 7.693383 | 4.268441 | 4.389275 | 1.396208 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss3 | 0.684247 | 1.107207 | 5.607120 | 4.278123 | 4.389275 | 0.684247 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | random_clean_y | 4.674353 | 8.324120 | 7.075140 | 3.558831 | 4.389275 | 4.674353 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | random_solver_y | 1.672366 | 2.203931 | 4.459070 | 4.409905 | 4.389275 | 1.672366 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | baseline | 0.824200 | 1.658133 | 19.999967 | 2.036220 | 2.015097 | 0.824200 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss1 | 0.621946 | 1.251178 | 24.672299 | 2.059818 | 2.015097 | 0.621946 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss2 | 0.750253 | 1.313077 | 18.171853 | 2.069547 | 2.015097 | 0.750253 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss3 | 0.427831 | 0.777955 | 13.365116 | 2.011480 | 2.015097 | 0.427831 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | random_clean_y | 2.293354 | 4.792719 | 10.860124 | 2.586514 | 2.015097 | 2.293354 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | random_solver_y | 0.634369 | 1.161240 | 11.995324 | 2.026026 | 2.015097 | 0.634369 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | baseline | 3.114906 | 4.095137 | 4.511718 | 2.756592 | 3.069114 | 3.114906 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss1 | 2.299518 | 3.553530 | 6.493301 | 2.981972 | 3.069114 | 2.299518 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss2 | 1.972822 | 3.148232 | 6.495836 | 2.875197 | 3.069114 | 1.972822 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss3 | 1.119665 | 2.049804 | 9.682577 | 2.957500 | 3.069114 | 1.119665 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | random_clean_y | 3.086348 | 5.272913 | 8.190710 | 2.365421 | 3.069114 | 3.086348 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | random_solver_y | 2.506936 | 3.578184 | 5.006643 | 2.974047 | 3.069114 | 2.506936 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | baseline | 1.868395 | 3.773138 | 8.879562 | 2.810447 | 3.257861 | 1.868395 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss1 | 2.039940 | 4.340793 | 9.058795 | 2.956791 | 3.257861 | 2.039940 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss2 | 2.334766 | 4.610658 | 8.866026 | 2.836687 | 3.257861 | 2.334766 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss3 | 1.439749 | 2.690649 | 7.330331 | 3.070531 | 3.257861 | 1.439749 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | random_clean_y | 3.326230 | 6.497961 | 8.011427 | 2.471371 | 3.257861 | 3.326230 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | random_solver_y | 2.041697 | 4.141793 | 8.633032 | 2.996673 | 3.257861 | 2.041697 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | baseline | 2.740100 | 3.972144 | 6.505833 | 3.298521 | 3.400892 | 2.740100 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss1 | 3.056563 | 4.367501 | 5.015501 | 3.430219 | 3.400892 | 3.056563 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss2 | 2.731035 | 3.941714 | 5.484471 | 3.461999 | 3.400892 | 2.731035 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss3 | 1.306184 | 2.330902 | 6.572911 | 3.310314 | 3.400892 | 1.306184 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | random_clean_y | 3.041840 | 5.893716 | 8.576320 | 2.337708 | 3.400892 | 3.041840 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | random_solver_y | 2.610078 | 3.946639 | 5.999622 | 3.298825 | 3.400892 | 2.610078 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | baseline | 3.838771 | 5.074160 | 4.894879 | 3.427531 | 3.657557 | 3.838771 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss1 | 3.204478 | 5.225343 | 6.913525 | 3.952101 | 3.657557 | 3.204478 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss2 | 3.263919 | 4.758728 | 5.859746 | 3.834154 | 3.657557 | 3.263919 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss3 | 1.583629 | 2.416743 | 7.241125 | 3.559239 | 3.657557 | 1.583629 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | random_clean_y | 3.822097 | 6.912173 | 7.679719 | 3.983625 | 3.657557 | 3.822097 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | random_solver_y | 2.898543 | 4.459863 | 5.748512 | 3.604581 | 3.657557 | 2.898543 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | baseline | 4.374951 | 6.238346 | 3.088186 | 4.594967 | 5.445839 | 4.374951 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss1 | 1.337991 | 1.931526 | 7.166362 | 5.254076 | 5.445839 | 1.337991 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss2 | 2.167449 | 2.823146 | 4.409428 | 5.241657 | 5.445839 | 2.167449 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss3 | 1.622361 | 2.041828 | 2.999132 | 5.493232 | 5.445839 | 1.622361 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | random_clean_y | 5.454612 | 8.912439 | 4.821085 | 3.618512 | 5.445839 | 5.454612 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | random_solver_y | 1.053546 | 1.723233 | 6.498340 | 5.199681 | 5.445839 | 1.053546 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | baseline | 4.664260 | 6.382761 | 4.178297 | 3.683242 | 3.813986 | 4.664260 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss1 | 2.521648 | 3.800790 | 5.662765 | 3.600017 | 3.813986 | 2.521648 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss2 | 2.907321 | 4.294123 | 5.640375 | 3.627551 | 3.813986 | 2.907321 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss3 | 1.723512 | 2.718160 | 5.511738 | 3.407639 | 3.813986 | 1.723512 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | random_clean_y | 3.920019 | 6.959542 | 7.314365 | 2.308093 | 3.813986 | 3.920019 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | random_solver_y | 2.463651 | 3.634764 | 5.017514 | 3.592920 | 3.813986 | 2.463651 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | baseline | 1.855179 | 2.699863 | 6.511996 | 4.024939 | 4.643709 | 1.855179 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss1 | 0.693094 | 1.361195 | 12.510971 | 4.571011 | 4.643709 | 0.693094 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss2 | 0.845617 | 1.570530 | 9.226861 | 4.521829 | 4.643709 | 0.845617 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss3 | 0.696566 | 1.145380 | 5.504107 | 4.462451 | 4.643709 | 0.696566 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | random_clean_y | 4.578716 | 8.299306 | 5.864593 | 2.154337 | 4.643709 | 4.578716 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | random_solver_y | 0.769704 | 1.171651 | 6.557751 | 4.549034 | 4.643709 | 0.769704 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | baseline | 2.881901 | 4.193108 | 6.017617 | 3.427281 | 3.644273 | 2.881901 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss1 | 3.004633 | 4.675045 | 5.760873 | 2.989343 | 3.644273 | 3.004633 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss2 | 3.420075 | 5.194232 | 5.134411 | 3.044195 | 3.644273 | 3.420075 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss3 | 1.129792 | 2.128602 | 7.395789 | 3.283092 | 3.644273 | 1.129792 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | random_clean_y | 3.475438 | 6.823139 | 6.760207 | 1.830657 | 3.644273 | 3.475438 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | random_solver_y | 2.649121 | 4.468388 | 5.168173 | 3.115362 | 3.644273 | 2.649121 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | baseline | 1.752561 | 3.101131 | 9.683798 | 2.113183 | 2.402523 | 1.752561 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss1 | 1.029404 | 2.122437 | 12.183121 | 2.084603 | 2.402523 | 1.029404 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss2 | 1.374278 | 2.414101 | 9.346435 | 2.133882 | 2.402523 | 1.374278 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss3 | 0.578937 | 1.403526 | 13.343183 | 2.055840 | 2.402523 | 0.578937 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | random_clean_y | 2.581578 | 5.737472 | 8.401591 | 1.015707 | 2.402523 | 2.581578 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | random_solver_y | 1.267372 | 2.382276 | 9.249466 | 2.047130 | 2.402523 | 1.267372 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | baseline | 1.737589 | 3.715320 | 10.045352 | 2.121573 | 2.542912 | 1.737589 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss1 | 2.004303 | 3.698282 | 8.901083 | 2.230619 | 2.542912 | 2.004303 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss2 | 1.932549 | 3.753162 | 8.707527 | 2.151940 | 2.542912 | 1.932549 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss3 | 1.073222 | 2.192192 | 9.707994 | 2.054088 | 2.542912 | 1.073222 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | random_clean_y | 4.047574 | 6.532911 | 7.568831 | 4.329749 | 2.542912 | 4.047574 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | random_solver_y | 1.318908 | 2.519035 | 7.961693 | 2.182344 | 2.542912 | 1.318908 |

## 9. Robustness Indicator 3: Direction, Subspace, A^T b, and J_error delta

For old-four rows, `atb_norm` is the stored `bias_gradient_norm` / A^T b style scalar. For random rows, `jerr_delta_l2` and `jerr_delta_rms` are the stored `J_error @ delta` response. Blank random `jerr_delta` entries mean that the attack-delta projection row was not present in `j_error_times_attack_delta.csv`; sample 4 is the current missing case. The full singular-vector arrays are in the NPZ paths listed in the CSV.

| sample | split | dataset_id | local | model | top1_R_solver | top1_L_solver | top5_R_sub | top10_R_sub | top20_R_sub | top5_L_sub | top10_L_sub | top20_L_sub | abs_cos_delta_top_errSV | delta_l2 | atb_norm | jerr_delta_l2 | jerr_delta_rms | svd_gain | outward_gain | affine_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | baseline | 0.849395 | 0.882995 | 0.935958 | 0.896214 | 0.808001 | 0.875668 | 0.800051 | 0.805538 | 0.031621 |  | 0.866702 |  |  | 0.081151 | 0.025778 | 0.081324 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss1 | 0.988057 | 0.998649 | 0.942401 | 0.950982 | 0.810203 | 0.756795 | 0.836363 | 0.870141 | 0.237805 |  | 0.619140 |  |  | 0.133060 | 0.021760 | 0.133100 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss2 | 0.975962 | 0.952580 | 0.922957 | 0.952029 | 0.820257 | 0.730904 | 0.822849 | 0.862080 | 0.064743 |  | 0.794608 |  |  | 0.137337 | 0.020550 | 0.137406 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | loss3 | 0.987770 | 0.988272 | 0.798216 | 0.979924 | 0.872114 | 0.786953 | 0.967344 | 0.928505 | 0.004418 |  | 0.181968 |  |  | 0.016589 | 0.008175 | 0.016648 |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | random_clean_y | 0.042592 | 0.022033 | 0.614730 | 0.630698 | 0.700579 | 0.491433 | 0.542966 | 0.672614 | 0.006776 | 3.840000 |  | 6.244413 | 0.195138 |  |  |  |
| 0 | generalization | burgers_widevis_l3target_d31 | 10 | random_solver_y | 0.979095 | 0.950578 | 0.955135 | 0.963185 | 0.831591 | 0.745672 | 0.817886 | 0.875375 | 0.023377 | 3.840000 |  | 2.755355 | 0.086105 |  |  |  |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | baseline | 0.980358 | 0.873790 | 0.788326 | 0.877827 | 0.779377 | 0.736028 | 0.787673 | 0.836883 | 0.126244 |  | 0.337853 |  |  | 0.042825 | 0.012462 | 0.042872 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss1 | 0.978610 | 0.908526 | 0.969940 | 0.888010 | 0.786045 | 0.844904 | 0.792766 | 0.871600 | 0.125627 |  | 0.423631 |  |  | 0.058191 | 0.014403 | 0.058266 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss2 | 0.977920 | 0.984548 | 0.966141 | 0.890983 | 0.804400 | 0.893931 | 0.807050 | 0.901224 | 0.171127 |  | 0.272094 |  |  | 0.056711 | 0.009524 | 0.056730 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | loss3 | 0.989935 | 0.995194 | 0.989279 | 0.925355 | 0.886645 | 0.948508 | 0.882787 | 0.935621 | 0.091913 |  | 0.161146 |  |  | 0.041925 | 0.005645 | 0.041934 |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | random_clean_y | 0.038935 | 0.012690 | 0.529764 | 0.681260 | 0.696331 | 0.254492 | 0.504388 | 0.654332 | 0.112028 | 3.840000 |  | 7.710833 | 0.240964 |  |  |  |
| 1 | generalization | burgers_widevis_l3target_d17 | 177 | random_solver_y | 0.981552 | 0.985251 | 0.984047 | 0.908208 | 0.832032 | 0.927211 | 0.843220 | 0.901371 | 0.193977 | 3.840000 |  | 2.379187 | 0.074350 |  |  |  |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | baseline | 0.991107 | 0.764320 | 0.989835 | 0.889401 | 0.867124 | 0.922951 | 0.806958 | 0.910146 | 0.781503 |  | 0.192842 |  |  | 0.055227 | 0.005485 | 0.055236 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss1 | 0.991431 | 0.912503 | 0.818610 | 0.978139 | 0.884381 | 0.744272 | 0.912423 | 0.938961 | 0.100595 |  | 0.152518 |  |  | 0.021753 | 0.004016 | 0.021773 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss2 | 0.992033 | 0.978701 | 0.991097 | 0.965083 | 0.873905 | 0.925649 | 0.912417 | 0.934459 | 0.083099 |  | 0.130532 |  |  | 0.021382 | 0.002766 | 0.021394 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | loss3 | 0.998491 | 0.993880 | 0.826476 | 0.996809 | 0.960699 | 0.821077 | 0.989657 | 0.988250 | 0.025577 |  | 0.019026 |  |  | 0.002553 | 0.000452 | 0.002555 |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | random_clean_y | 0.262018 | 0.025946 | 0.555086 | 0.643298 | 0.711808 | 0.304763 | 0.534645 | 0.699352 | 0.674805 | 3.840000 |  | 8.915842 | 0.278620 |  |  |  |
| 2 | generalization | burgers_widevis_l3target_d05 | 45 | random_solver_y | 0.993859 | 0.983597 | 0.799644 | 0.984960 | 0.903280 | 0.752655 | 0.935556 | 0.950773 | 0.134209 | 3.840000 |  | 1.595899 | 0.049872 |  |  |  |
| 3 | train | train_original_gaussian_corr0p03 | 1 | baseline | 0.996749 | 0.987805 | 0.996354 | 0.979324 | 0.867284 | 0.994926 | 0.983702 | 0.940539 | 0.069421 |  | 0.017333 |  |  | 0.004574 | 0.000764 | 0.004575 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss1 | 0.998715 | 0.999939 | 0.998265 | 0.996477 | 0.912386 | 0.999884 | 0.999616 | 0.970787 | 0.071266 |  | 0.001464 |  |  | 0.000767 | 0.000355 | 0.000767 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss2 | 0.998415 | 0.996777 | 0.998011 | 0.995695 | 0.904271 | 0.998834 | 0.998933 | 0.966865 | 0.012282 |  | 0.010578 |  |  | 0.001287 | 0.000885 | 0.001290 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | loss3 | 0.999796 | 0.983027 | 0.999819 | 0.999080 | 0.990532 | 0.994102 | 0.996484 | 0.993521 | 0.087148 |  | 0.006961 |  |  | 0.006523 | 0.001174 | 0.006524 |
| 3 | train | train_original_gaussian_corr0p03 | 1 | random_clean_y | 0.298586 | 0.812307 | 0.403939 | 0.653852 | 0.725186 | 0.720221 | 0.811709 | 0.817657 | 0.581129 | 3.550031 |  | 8.937369 | 0.279293 |  |  |  |
| 3 | train | train_original_gaussian_corr0p03 | 1 | random_solver_y | 0.999110 | 0.999674 | 0.998576 | 0.997053 | 0.924915 | 0.999641 | 0.999270 | 0.965302 | 0.613313 | 3.840000 |  | 0.542040 | 0.016939 |  |  |  |
| 4 | train | train_original_gaussian_corr0p03 | 128 | baseline | 0.997195 | 0.995413 | 0.996759 | 0.985350 | 0.809412 | 0.982086 | 0.981368 | 0.907021 | 0.935282 |  | 0.049051 |  |  | 0.007615 | 0.005973 | 0.007619 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss1 | 0.998899 | 0.999901 | 0.999157 | 0.998081 | 0.857719 | 0.999734 | 0.999541 | 0.930905 | 0.268775 |  | 0.002038 |  |  | 0.001150 | 0.000299 | 0.001150 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss2 | 0.998641 | 0.998669 | 0.998962 | 0.997679 | 0.860827 | 0.999050 | 0.999078 | 0.953786 | 0.854019 |  | 0.007855 |  |  | 0.001453 | 0.001292 | 0.001453 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | loss3 | 0.999896 | 0.976174 | 0.999788 | 0.998703 | 0.985014 | 0.991735 | 0.994477 | 0.989495 | 0.000199 |  | 0.016229 |  |  | 0.016550 | 0.000595 | 0.016551 |
| 4 | train | train_original_gaussian_corr0p03 | 128 | random_clean_y | 0.297051 | 0.755560 | 0.332754 | 0.692689 | 0.636107 | 0.740182 | 0.847021 | 0.731943 |  |  |  |  |  |  |  |  |
| 4 | train | train_original_gaussian_corr0p03 | 128 | random_solver_y | 0.999219 | 0.997281 | 0.999261 | 0.998297 | 0.897661 | 0.998150 | 0.998604 | 0.965289 |  |  |  |  |  |  |  |  |
| 5 | test | test_original_gaussian_corr0p03 | 45 | baseline | 0.997659 | 0.996226 | 0.995674 | 0.986321 | 0.864402 | 0.992463 | 0.989011 | 0.966968 | 0.374910 |  | 0.013300 |  |  | 0.002754 | 0.000861 | 0.002756 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss1 | 0.998499 | 0.999817 | 0.998545 | 0.996815 | 0.901954 | 0.999763 | 0.999224 | 0.981303 | 0.894237 |  | 0.003562 |  |  | 0.001281 | 0.000985 | 0.001281 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss2 | 0.998518 | 0.998957 | 0.998273 | 0.996616 | 0.897576 | 0.999159 | 0.998670 | 0.985167 | 0.873543 |  | 0.009287 |  |  | 0.001390 | 0.001204 | 0.001390 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | loss3 | 0.999945 | 0.983671 | 0.999786 | 0.998992 | 0.991922 | 0.995945 | 0.996959 | 0.994889 | 0.104172 |  | 0.006200 |  |  | 0.010461 | 0.000126 | 0.010461 |
| 5 | test | test_original_gaussian_corr0p03 | 45 | random_clean_y | 0.777430 | 0.135048 | 0.693045 | 0.671422 | 0.734826 | 0.411788 | 0.555881 | 0.688320 | 0.277306 | 3.840000 |  | 9.349027 | 0.292157 |  |  |  |
| 5 | test | test_original_gaussian_corr0p03 | 45 | random_solver_y | 0.998776 | 0.999733 | 0.998539 | 0.996977 | 0.927417 | 0.999587 | 0.999016 | 0.991193 | 0.797482 | 3.840000 |  | 0.836203 | 0.026131 |  |  |  |
| 6 | test | test_original_gaussian_corr0p03 | 71 | baseline | 0.997204 | 0.973939 | 0.997045 | 0.972806 | 0.709187 | 0.987571 | 0.987594 | 0.819620 | 0.068506 |  | 0.063398 |  |  | 0.016379 | 0.002977 | 0.016383 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss1 | 0.998240 | 0.999933 | 0.998473 | 0.989947 | 0.785441 | 0.999922 | 0.998272 | 0.897777 | 0.773285 |  | 0.003699 |  |  | 0.001325 | 0.001163 | 0.001326 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss2 | 0.998026 | 0.999454 | 0.998217 | 0.988302 | 0.787250 | 0.999415 | 0.997891 | 0.933188 | 0.614808 |  | 0.006703 |  |  | 0.001453 | 0.000880 | 0.001454 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | loss3 | 0.999868 | 0.985330 | 0.999830 | 0.998983 | 0.983183 | 0.995493 | 0.996924 | 0.994712 | 0.030352 |  | 0.010482 |  |  | 0.009162 | 0.000331 | 0.009163 |
| 6 | test | test_original_gaussian_corr0p03 | 71 | random_clean_y | 0.475014 | 0.396178 | 0.525171 | 0.693992 | 0.664476 | 0.413909 | 0.621258 | 0.709424 | 0.445880 | 3.840000 |  | 10.426745 | 0.325836 |  |  |  |
| 6 | test | test_original_gaussian_corr0p03 | 71 | random_solver_y | 0.998713 | 0.997047 | 0.998821 | 0.994648 | 0.862790 | 0.999163 | 0.998473 | 0.979862 | 0.379534 | 3.840000 |  | 0.921741 | 0.028804 |  |  |  |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | baseline | 0.994730 | 0.994599 | 0.993644 | 0.977080 | 0.676656 | 0.971810 | 0.969269 | 0.730592 | 0.103911 |  | 0.088411 |  |  | 0.014117 | 0.006269 | 0.014130 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss1 | 0.996283 | 0.979082 | 0.996238 | 0.984776 | 0.773756 | 0.975817 | 0.980053 | 0.858871 | 0.049172 |  | 0.073200 |  |  | 0.008583 | 0.002185 | 0.008595 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss2 | 0.995382 | 0.978104 | 0.995619 | 0.983418 | 0.769444 | 0.974173 | 0.979909 | 0.888984 | 0.171987 |  | 0.083818 |  |  | 0.008910 | 0.002832 | 0.008927 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | loss3 | 0.999260 | 0.977316 | 0.999497 | 0.995998 | 0.935343 | 0.990242 | 0.993280 | 0.946661 | 0.191573 |  | 0.023887 |  |  | 0.009239 | 0.002848 | 0.009240 |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | random_clean_y | 0.837126 | 0.546578 | 0.663239 | 0.695408 | 0.645718 | 0.384746 | 0.494654 | 0.663815 | 0.610977 | 3.840000 |  | 9.534696 | 0.297959 |  |  |  |
| 7 | generalization | burgers_widevis_l3target_d03 | 180 | random_solver_y | 0.997492 | 0.987462 | 0.997265 | 0.986748 | 0.844344 | 0.982719 | 0.982218 | 0.933940 | 0.294608 | 3.840000 |  | 0.988298 | 0.030884 |  |  |  |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | baseline | 0.973947 | 0.008423 | 0.905730 | 0.839119 | 0.690679 | 0.541712 | 0.569953 | 0.725331 | 0.092567 |  | 0.920464 |  |  | 0.427917 | 0.014896 | 0.427945 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss1 | 0.934214 | 0.214692 | 0.898594 | 0.838951 | 0.735486 | 0.628570 | 0.565354 | 0.749320 | 0.059113 |  | 1.198986 |  |  | 0.271308 | 0.020427 | 0.271385 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss2 | 0.940574 | 0.030234 | 0.904557 | 0.814481 | 0.700647 | 0.586748 | 0.551709 | 0.742628 | 0.071554 |  | 1.264802 |  |  | 0.337400 | 0.045184 | 0.337463 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | loss3 | 0.986930 | 0.364315 | 0.952777 | 0.912418 | 0.821217 | 0.717576 | 0.822705 | 0.890607 | 0.063869 |  | 0.551498 |  |  | 0.246045 | 0.016412 | 0.246062 |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | random_clean_y | 0.212200 | 0.070928 | 0.580424 | 0.680944 | 0.647632 | 0.286442 | 0.584864 | 0.686588 | 0.304000 | 3.840000 |  | 8.245712 | 0.257678 |  |  |  |
| 8 | generalization | burgers_widevis_l3target_d08 | 178 | random_solver_y | 0.929763 | 0.012748 | 0.914460 | 0.804893 | 0.737693 | 0.593125 | 0.596866 | 0.790897 | 0.031206 | 3.840000 |  | 3.390616 | 0.105957 |  |  |  |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | baseline | 0.992711 | 0.992120 | 0.831243 | 0.852919 | 0.623187 | 0.721167 | 0.813733 | 0.679067 | 0.619944 |  | 0.331235 |  |  | 0.087729 | 0.009662 | 0.087747 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss1 | 0.996913 | 0.998319 | 0.987770 | 0.939139 | 0.757065 | 0.988962 | 0.949881 | 0.903636 | 0.368823 |  | 0.041323 |  |  | 0.003342 | 0.001378 | 0.003355 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss2 | 0.995781 | 0.999170 | 0.984588 | 0.944324 | 0.744136 | 0.983674 | 0.951109 | 0.897739 | 0.194072 |  | 0.073879 |  |  | 0.013537 | 0.002170 | 0.013543 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | loss3 | 0.999522 | 0.964910 | 0.996129 | 0.897096 | 0.914152 | 0.979734 | 0.888952 | 0.964169 | 0.060717 |  | 0.041775 |  |  | 0.022018 | 0.001830 | 0.022019 |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | random_clean_y | 0.410016 | 0.210209 | 0.491271 | 0.629510 | 0.617295 | 0.154534 | 0.459261 | 0.623551 | 0.609288 | 3.840000 |  | 13.953588 | 0.436050 |  |  |  |
| 9 | generalization | burgers_widevis_l3target_d00 | 98 | random_solver_y | 0.996728 | 0.983610 | 0.972937 | 0.902846 | 0.814902 | 0.957261 | 0.896941 | 0.903234 | 0.270790 | 3.840000 |  | 1.870967 | 0.058468 |  |  |  |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | baseline | 0.972820 | 0.954423 | 0.824335 | 0.873985 | 0.738371 | 0.744813 | 0.791581 | 0.797557 | 0.082659 |  | 0.262300 |  |  | 0.091631 | 0.008175 | 0.091641 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss1 | 0.994305 | 0.975688 | 0.830133 | 0.891976 | 0.792642 | 0.819481 | 0.874922 | 0.851035 | 0.219250 |  | 0.096843 |  |  | 0.011212 | 0.004005 | 0.011226 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss2 | 0.984667 | 0.978250 | 0.814248 | 0.881825 | 0.793275 | 0.811248 | 0.851021 | 0.873574 | 0.053497 |  | 0.086984 |  |  | 0.010446 | 0.002632 | 0.010460 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | loss3 | 0.995533 | 0.990363 | 0.997641 | 0.980770 | 0.917950 | 0.975080 | 0.969372 | 0.973927 | 0.058457 |  | 0.036433 |  |  | 0.005468 | 0.001270 | 0.005472 |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | random_clean_y | 0.147614 | 0.181115 | 0.595859 | 0.678506 | 0.696804 | 0.415025 | 0.503123 | 0.688431 | 0.030475 | 2.833438 |  | 5.103976 | 0.159499 |  |  |  |
| 10 | generalization | burgers_widevis_l3target_d10 | 39 | random_solver_y | 0.883105 | 0.883312 | 0.994014 | 0.898238 | 0.849989 | 0.988970 | 0.883977 | 0.919973 | 0.283803 | 3.840000 |  | 0.900265 | 0.028133 |  |  |  |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | baseline | 0.967793 | 0.967389 | 0.788155 | 0.855610 | 0.683309 | 0.761856 | 0.802373 | 0.787170 | 0.020189 |  | 0.279119 |  |  | 0.035558 | 0.013708 | 0.035595 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss1 | 0.961197 | 0.964101 | 0.934821 | 0.847275 | 0.658747 | 0.621796 | 0.577058 | 0.744911 | 0.090910 |  | 0.815924 |  |  | 0.176280 | 0.025549 | 0.176343 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss2 | 0.954798 | 0.991624 | 0.804606 | 0.848530 | 0.671404 | 0.646298 | 0.648920 | 0.802197 | 0.070748 |  | 0.538196 |  |  | 0.146177 | 0.019760 | 0.146205 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | loss3 | 0.967831 | 0.950874 | 0.792904 | 0.864604 | 0.824386 | 0.705910 | 0.749534 | 0.879312 | 0.220608 |  | 0.372657 |  |  | 0.102933 | 0.012857 | 0.102954 |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | random_clean_y | 0.624420 | 0.369838 | 0.476341 | 0.632985 | 0.650261 | 0.295892 | 0.450626 | 0.637138 | 0.604551 | 3.840000 |  | 10.137949 | 0.316811 |  |  |  |
| 11 | generalization | burgers_widevis_l3target_d09 | 68 | random_solver_y | 0.928190 | 0.947060 | 0.793140 | 0.841920 | 0.699273 | 0.560624 | 0.593836 | 0.780395 | 0.054722 | 3.840000 |  | 4.164994 | 0.130156 |  |  |  |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | baseline | 0.980514 | 0.342939 | 0.791629 | 0.814602 | 0.657723 | 0.378108 | 0.553857 | 0.723263 | 0.194071 |  | 1.682755 |  |  | 0.292858 | 0.183268 | 0.298315 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss1 | 0.971293 | 0.929997 | 0.789571 | 0.820564 | 0.640629 | 0.683118 | 0.735796 | 0.815269 | 0.515053 |  | 0.345292 |  |  | 0.087193 | 0.018024 | 0.087213 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss2 | 0.984538 | 0.869644 | 0.875007 | 0.821394 | 0.621053 | 0.692144 | 0.728496 | 0.810951 | 0.227337 |  | 1.021944 |  |  | 0.153389 | 0.130956 | 0.153407 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | loss3 | 0.994402 | 0.989710 | 0.858598 | 0.888118 | 0.763835 | 0.757538 | 0.814064 | 0.861467 | 0.118831 |  | 0.220032 |  |  | 0.082088 | 0.028962 | 0.082095 |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | random_clean_y | 0.167093 | 0.071802 | 0.461409 | 0.584812 | 0.585547 | 0.382950 | 0.518350 | 0.636318 | 0.561363 | 3.840000 |  | 11.295726 | 0.352991 |  |  |  |
| 12 | generalization | burgers_widevis_l3target_d19 | 148 | random_solver_y | 0.976990 | 0.622774 | 0.913277 | 0.818455 | 0.665113 | 0.682255 | 0.705372 | 0.814584 | 0.101155 | 3.840000 |  | 2.878994 | 0.089969 |  |  |  |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | baseline | 0.994360 | 0.863729 | 0.990042 | 0.862009 | 0.625737 | 0.796609 | 0.830571 | 0.722828 | 0.831771 |  | 0.789042 |  |  | 0.156040 | 0.090573 | 0.156070 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss1 | 0.996968 | 0.999380 | 0.995450 | 0.892132 | 0.665741 | 0.980827 | 0.932856 | 0.815030 | 0.132532 |  | 0.081853 |  |  | 0.014266 | 0.001984 | 0.014273 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss2 | 0.996763 | 0.978867 | 0.993325 | 0.872521 | 0.650326 | 0.934874 | 0.903647 | 0.839927 | 0.116110 |  | 0.180844 |  |  | 0.028300 | 0.004768 | 0.028318 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | loss3 | 0.999378 | 0.991055 | 0.999402 | 0.978953 | 0.916941 | 0.990079 | 0.979931 | 0.964123 | 0.000823 |  | 0.024731 |  |  | 0.006766 | 0.001513 | 0.006768 |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | random_clean_y | 0.258952 | 0.005696 | 0.429224 | 0.505330 | 0.555813 | 0.302337 | 0.397146 | 0.580192 | 0.242976 | 3.840000 |  | 9.894614 | 0.309207 |  |  |  |
| 13 | generalization | burgers_widevis_l3target_d12 | 104 | random_solver_y | 0.997286 | 0.984084 | 0.995948 | 0.902611 | 0.735709 | 0.952187 | 0.926153 | 0.888248 | 0.050199 | 3.840000 |  | 1.759798 | 0.054994 |  |  |  |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | baseline | 0.002924 | 0.005652 | 0.983625 | 0.889317 | 0.923963 | 0.960654 | 0.864724 | 0.929875 | 0.241674 |  | 0.122551 |  |  | 0.010021 | 0.005026 | 0.010054 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss1 | 0.993186 | 0.991870 | 0.974559 | 0.909924 | 0.947554 | 0.960988 | 0.896443 | 0.957524 | 0.180490 |  | 0.048396 |  |  | 0.005634 | 0.001276 | 0.005640 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss2 | 0.993017 | 0.994309 | 0.967716 | 0.905746 | 0.942007 | 0.950194 | 0.895994 | 0.954888 | 0.165750 |  | 0.053081 |  |  | 0.008157 | 0.001509 | 0.008163 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | loss3 | 0.998686 | 0.997847 | 0.997135 | 0.906657 | 0.982604 | 0.993026 | 0.901455 | 0.989331 | 0.111744 |  | 0.010665 |  |  | 0.002659 | 0.000481 | 0.002660 |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | random_clean_y | 0.266644 | 0.254325 | 0.686528 | 0.690956 | 0.833596 | 0.621182 | 0.667573 | 0.750674 | 0.467629 | 3.840000 |  | 6.064339 | 0.189511 |  |  |  |
| 14 | generalization | burgers_widevis_l3target_d15 | 113 | random_solver_y | 0.994155 | 0.990181 | 0.975904 | 0.908775 | 0.954158 | 0.962357 | 0.897635 | 0.961743 | 0.120329 | 3.840000 |  | 0.891262 | 0.027852 |  |  |  |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | baseline | 0.992817 | 0.430398 | 0.984842 | 0.943363 | 0.875509 | 0.718898 | 0.770316 | 0.842127 | 0.375289 |  | 0.316669 |  |  | 0.140363 | 0.016381 | 0.140373 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss1 | 0.994677 | 0.711791 | 0.975866 | 0.906962 | 0.887622 | 0.760149 | 0.794056 | 0.896547 | 0.112630 |  | 0.253881 |  |  | 0.076147 | 0.003992 | 0.076159 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss2 | 0.995605 | 0.782816 | 0.985364 | 0.910994 | 0.886763 | 0.802941 | 0.813320 | 0.899802 | 0.259629 |  | 0.198972 |  |  | 0.056083 | 0.004960 | 0.056095 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | loss3 | 0.997909 | 0.932458 | 0.970040 | 0.986939 | 0.886130 | 0.904555 | 0.932533 | 0.944130 | 0.089051 |  | 0.112447 |  |  | 0.018191 | 0.003471 | 0.018203 |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | random_clean_y | 0.082575 | 0.033872 | 0.591606 | 0.710252 | 0.775100 | 0.531820 | 0.644289 | 0.734251 | 0.420883 | 3.840000 |  | 6.620401 | 0.206888 |  |  |  |
| 15 | generalization | burgers_widevis_l3target_d21 | 136 | random_solver_y | 0.995112 | 0.656456 | 0.985253 | 0.931717 | 0.892273 | 0.771269 | 0.811902 | 0.890097 | 0.473669 | 3.840000 |  | 4.740608 | 0.148144 |  |  |  |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | baseline | 0.985568 | 0.961458 | 0.786634 | 0.889431 | 0.694849 | 0.682437 | 0.748162 | 0.837014 | 0.372675 |  | 0.650106 |  |  | 0.053782 | 0.035596 | 0.053874 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss1 | 0.979579 | 0.782507 | 0.769876 | 0.926596 | 0.678797 | 0.575844 | 0.772188 | 0.842961 | 0.025577 |  | 0.529017 |  |  | 0.059930 | 0.014397 | 0.060082 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss2 | 0.981937 | 0.907216 | 0.766217 | 0.860139 | 0.653118 | 0.549737 | 0.680429 | 0.821179 | 0.118125 |  | 0.681962 |  |  | 0.079265 | 0.020451 | 0.079379 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | loss3 | 0.993769 | 0.899659 | 0.797763 | 0.985903 | 0.826035 | 0.735938 | 0.929607 | 0.909007 | 0.140333 |  | 0.168646 |  |  | 0.029999 | 0.004157 | 0.030016 |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | random_clean_y | 0.370466 | 0.079674 | 0.567792 | 0.638144 | 0.654535 | 0.360644 | 0.447362 | 0.656407 | 0.286792 | 3.840000 |  | 7.679548 | 0.239986 |  |  |  |
| 16 | generalization | burgers_widevis_l3target_d34 | 192 | random_solver_y | 0.986471 | 0.939531 | 0.780038 | 0.780508 | 0.696277 | 0.601340 | 0.626345 | 0.818266 | 0.004137 | 3.840000 |  | 3.461050 | 0.108158 |  |  |  |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | baseline | 0.990202 | 0.914718 | 0.819596 | 0.876498 | 0.752419 | 0.588513 | 0.752967 | 0.830845 | 0.255256 |  | 0.531202 |  |  | 0.110932 | 0.061145 | 0.110952 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss1 | 0.990140 | 0.599370 | 0.775083 | 0.869468 | 0.754711 | 0.624083 | 0.774124 | 0.900825 | 0.100465 |  | 0.483947 |  |  | 0.134967 | 0.011513 | 0.134993 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss2 | 0.989451 | 0.860009 | 0.747731 | 0.867613 | 0.737125 | 0.623567 | 0.773036 | 0.874745 | 0.332805 |  | 0.563640 |  |  | 0.108309 | 0.016433 | 0.108351 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | loss3 | 0.995350 | 0.925471 | 0.818749 | 0.979527 | 0.881585 | 0.789553 | 0.947466 | 0.939560 | 0.216041 |  | 0.113387 |  |  | 0.024846 | 0.007204 | 0.024856 |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | random_clean_y | 0.762334 | 0.536519 | 0.457398 | 0.605228 | 0.666568 | 0.404661 | 0.502120 | 0.721094 | 0.787944 | 3.840000 |  | 9.638398 | 0.301200 |  |  |  |
| 17 | generalization | burgers_widevis_l3target_d25 | 28 | random_solver_y | 0.990953 | 0.914816 | 0.768077 | 0.871972 | 0.761754 | 0.628214 | 0.758841 | 0.891851 | 0.331830 | 3.840000 |  | 4.297007 | 0.134281 |  |  |  |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | baseline | 0.993586 | 0.412523 | 0.935112 | 0.849661 | 0.670399 | 0.665949 | 0.671013 | 0.741029 | 0.151067 |  | 0.564580 |  |  | 0.214018 | 0.059954 | 0.214037 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss1 | 0.984974 | 0.645450 | 0.936547 | 0.870669 | 0.674493 | 0.591013 | 0.691252 | 0.785367 | 0.173042 |  | 0.742128 |  |  | 0.148718 | 0.019006 | 0.148773 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss2 | 0.992072 | 0.619699 | 0.878843 | 0.868834 | 0.692459 | 0.611394 | 0.702515 | 0.795131 | 0.009838 |  | 0.632912 |  |  | 0.153711 | 0.015316 | 0.153751 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | loss3 | 0.998239 | 0.903799 | 0.986372 | 0.927640 | 0.799202 | 0.950210 | 0.901596 | 0.907973 | 0.039799 |  | 0.098199 |  |  | 0.036182 | 0.004534 | 0.036186 |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | random_clean_y | 0.156359 | 0.025980 | 0.454029 | 0.631011 | 0.609211 | 0.435727 | 0.518110 | 0.590299 | 0.012797 | 3.840000 |  | 7.691707 | 0.240366 |  |  |  |
| 18 | generalization | burgers_widevis_l3target_d35 | 63 | random_solver_y | 0.994094 | 0.680556 | 0.915765 | 0.870032 | 0.710868 | 0.659395 | 0.720002 | 0.817620 | 0.276971 | 3.840000 |  | 3.852806 | 0.120400 |  |  |  |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | baseline | 0.988974 | 0.682388 | 0.963068 | 0.805883 | 0.597993 | 0.752613 | 0.718924 | 0.690209 | 0.039806 |  | 0.729982 |  |  | 0.275932 | 0.015844 | 0.275960 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss1 | 0.994237 | 0.996719 | 0.992541 | 0.851087 | 0.683253 | 0.949025 | 0.845423 | 0.811576 | 0.358202 |  | 0.121616 |  |  | 0.026268 | 0.009760 | 0.026274 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss2 | 0.995378 | 0.917868 | 0.990554 | 0.855709 | 0.669572 | 0.927790 | 0.837757 | 0.817875 | 0.073278 |  | 0.188132 |  |  | 0.067687 | 0.004805 | 0.067695 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | loss3 | 0.998821 | 0.999142 | 0.998209 | 0.973817 | 0.817217 | 0.967915 | 0.969996 | 0.900059 | 0.020917 |  | 0.080228 |  |  | 0.038153 | 0.008677 | 0.038155 |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | random_clean_y | 0.074196 | 0.004241 | 0.547294 | 0.570019 | 0.562987 | 0.287296 | 0.484757 | 0.623941 | 0.557446 | 3.840000 |  | 14.484574 | 0.452643 |  |  |  |
| 19 | generalization | burgers_widevis_l3target_d30 | 74 | random_solver_y | 0.995307 | 0.989687 | 0.993462 | 0.882481 | 0.757240 | 0.978584 | 0.886278 | 0.861300 | 0.553110 | 3.840000 |  | 2.361242 | 0.073789 |  |  |  |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | baseline | 0.012909 | 0.008955 | 0.783862 | 0.807208 | 0.647011 | 0.447299 | 0.587762 | 0.717704 | 0.766918 |  | 1.559168 |  |  | 0.322344 | 0.203295 | 0.322388 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss1 | 0.429912 | 0.337418 | 0.799235 | 0.823909 | 0.665386 | 0.665740 | 0.784200 | 0.800056 | 0.051472 |  | 0.450641 |  |  | 0.091868 | 0.009747 | 0.091902 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss2 | 0.720347 | 0.505188 | 0.908377 | 0.812186 | 0.684815 | 0.710697 | 0.745992 | 0.829582 | 0.088108 |  | 0.506478 |  |  | 0.123873 | 0.045117 | 0.123893 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | loss3 | 0.155088 | 0.133746 | 0.797970 | 0.965865 | 0.844006 | 0.729571 | 0.932473 | 0.903078 | 0.200461 |  | 0.218394 |  |  | 0.042781 | 0.004574 | 0.042798 |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | random_clean_y | 0.144828 | 0.005482 | 0.623157 | 0.634850 | 0.599062 | 0.383442 | 0.412808 | 0.629664 | 0.546317 | 3.840000 |  | 10.264820 | 0.320776 |  |  |  |
| 20 | generalization | burgers_widevis_l3target_d27 | 77 | random_solver_y | 0.089529 | 0.071313 | 0.805025 | 0.866451 | 0.700574 | 0.683158 | 0.802326 | 0.849538 | 0.807375 | 3.840000 |  | 7.717584 | 0.241175 |  |  |  |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | baseline | 0.952214 | 0.882373 | 0.990825 | 0.898027 | 0.536851 | 0.957555 | 0.967427 | 0.626880 | 0.621649 |  | 0.178729 |  |  | 0.050090 | 0.014456 | 0.050099 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss1 | 0.994917 | 0.986751 | 0.996056 | 0.939043 | 0.582714 | 0.989210 | 0.991025 | 0.663151 | 0.194256 |  | 0.049769 |  |  | 0.006972 | 0.002253 | 0.006979 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss2 | 0.995774 | 0.982959 | 0.993703 | 0.912845 | 0.548027 | 0.988400 | 0.989883 | 0.683897 | 0.112517 |  | 0.072273 |  |  | 0.010377 | 0.002641 | 0.010386 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | loss3 | 0.999548 | 0.989081 | 0.999124 | 0.991271 | 0.868340 | 0.992388 | 0.990740 | 0.946685 | 0.189475 |  | 0.024184 |  |  | 0.007028 | 0.001244 | 0.007030 |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | random_clean_y | 0.055145 | 0.018721 | 0.493981 | 0.523891 | 0.500201 | 0.213016 | 0.384394 | 0.579259 | 0.538361 | 3.840000 |  | 12.786983 | 0.399593 |  |  |  |
| 21 | generalization | burgers_widevis_l3target_d46 | 199 | random_solver_y | 0.996805 | 0.985610 | 0.996358 | 0.960940 | 0.711717 | 0.993376 | 0.992997 | 0.859855 | 0.514644 | 3.840000 |  | 1.782283 | 0.055696 |  |  |  |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | baseline | 0.899540 | 0.609842 | 0.983457 | 0.774999 | 0.625098 | 0.854640 | 0.718203 | 0.771274 | 0.335198 |  | 0.385544 |  |  | 0.120184 | 0.014704 | 0.120202 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss1 | 0.985801 | 0.810724 | 0.977225 | 0.860004 | 0.646812 | 0.761948 | 0.804029 | 0.850867 | 0.089151 |  | 0.824777 |  |  | 0.132457 | 0.036501 | 0.132525 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss2 | 0.985757 | 0.482403 | 0.977699 | 0.865834 | 0.656378 | 0.731561 | 0.784069 | 0.872185 | 0.040464 |  | 0.680104 |  |  | 0.168694 | 0.022917 | 0.168740 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | loss3 | 0.960829 | 0.948467 | 0.996587 | 0.924476 | 0.810311 | 0.964077 | 0.927354 | 0.928368 | 0.078159 |  | 0.147622 |  |  | 0.018722 | 0.007639 | 0.018747 |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | random_clean_y | 0.016476 | 0.166270 | 0.604605 | 0.601392 | 0.583349 | 0.440887 | 0.553411 | 0.660080 | 0.095715 | 3.840000 |  | 7.321544 | 0.228798 |  |  |  |
| 22 | generalization | burgers_widevis_l3target_d41 | 97 | random_solver_y | 0.993627 | 0.701516 | 0.983548 | 0.863030 | 0.703373 | 0.804144 | 0.820767 | 0.871660 | 0.030864 | 3.840000 |  | 2.847453 | 0.088983 |  |  |  |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | baseline | 0.535970 | 0.525059 | 0.707754 | 0.876681 | 0.826505 | 0.656822 | 0.771011 | 0.882171 | 0.060937 |  | 0.309728 |  |  | 0.044320 | 0.006201 | 0.044354 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss1 | 0.322234 | 0.320306 | 0.841443 | 0.878393 | 0.839971 | 0.828248 | 0.829823 | 0.920115 | 0.005470 |  | 0.173628 |  |  | 0.015296 | 0.003875 | 0.015331 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss2 | 0.362337 | 0.360095 | 0.849601 | 0.908994 | 0.854590 | 0.823737 | 0.839638 | 0.913428 | 0.133359 |  | 0.190082 |  |  | 0.027339 | 0.004228 | 0.027360 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | loss3 | 0.992828 | 0.991011 | 0.846479 | 0.987094 | 0.874672 | 0.839638 | 0.985917 | 0.977450 | 0.071994 |  | 0.044656 |  |  | 0.004840 | 0.001100 | 0.004847 |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | random_clean_y | 0.241482 | 0.006126 | 0.445146 | 0.540276 | 0.671564 | 0.522204 | 0.567915 | 0.782104 | 0.640702 | 3.840000 |  | 8.450685 | 0.264084 |  |  |  |
| 23 | generalization | burgers_widevis_l3target_d37 | 191 | random_solver_y | 0.264567 | 0.269520 | 0.643583 | 0.866205 | 0.819097 | 0.614954 | 0.787884 | 0.881226 | 0.099057 | 3.840000 |  | 1.432936 | 0.044779 |  |  |  |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | baseline | 0.305278 | 0.287784 | 0.792960 | 0.888600 | 0.740338 | 0.640853 | 0.748694 | 0.792343 | 0.068938 |  | 0.331708 |  |  | 0.043688 | 0.009000 | 0.043744 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss1 | 0.970362 | 0.975389 | 0.909369 | 0.934831 | 0.781635 | 0.723344 | 0.807801 | 0.847930 | 0.029247 |  | 0.359089 |  |  | 0.058098 | 0.008702 | 0.058137 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss2 | 0.698184 | 0.684740 | 0.799810 | 0.879901 | 0.789942 | 0.636758 | 0.737555 | 0.845696 | 6.505e-05 |  | 0.333752 |  |  | 0.054251 | 0.009751 | 0.054285 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | loss3 | 0.992591 | 0.972301 | 0.992453 | 0.903637 | 0.811256 | 0.931664 | 0.876375 | 0.931269 | 0.069671 |  | 0.107292 |  |  | 0.016591 | 0.001968 | 0.016602 |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | random_clean_y | 0.057646 | 0.028497 | 0.714563 | 0.635303 | 0.690729 | 0.599686 | 0.512509 | 0.678240 | 0.164590 | 3.840000 |  | 5.677999 | 0.177437 |  |  |  |
| 24 | generalization | burgers_widevis_l3target_d44 | 147 | random_solver_y | 0.981225 | 0.990013 | 0.834394 | 0.981953 | 0.825515 | 0.766945 | 0.910740 | 0.888406 | 0.210118 | 3.840000 |  | 2.325785 | 0.072681 |  |  |  |

## 10. Source Tables Used

- `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_clean_metrics.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/clean_loss/per_dataset_clean_metrics.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/per_sample_25sample_svd_attack_six_models.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/biased_direction_metrics.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/random_25sample_model_solver_subspace_similarity.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/j_error_times_attack_delta.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/summary_by_model_dataset.csv`


## 11. Frobenius Norm Basis Correction

After checking code and NPZ metadata, the Frobenius basis is:

- Old four models: full SVD over dense `1024 x 1024` Jacobians; `error_fro_norm` is true full Frobenius.
- Random two models: original `fro_norm` from top-20 SVD only; true Frobenius was recomputed from the saved full Jacobian matrix entries.
- Random effective rank remains top20-only, not full effective rank.

Corrected files:

- `forensics/burgers_six_model_latest_wideparam_summary_20260613/random_models_svd_top20_vs_true_fro_from_jacobian.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_six_models_detailed_corrected_fro_basis.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_svd_attack_six_models_corrected_fro_basis.csv`

## 12. Combined A^T b and J_error Delta Table

This table combines the old-four `A^T b`-style scalar and the random-model `J_error @ delta` scalar in one place. They are not the same algebraic quantity, so the table keeps both columns and identifies the stored quantity type per model.

| model | quantity_type | n | attack_inc | err_spec | err_fro_comparable | top20_R_solver | top20_L_solver | A^T b mean | A^T b median | Jerr_delta_L2 mean | Jerr_delta_L2 median | Jerr_delta_RMS mean | Jerr_delta_RMS median | abs_cos_delta_top_errSV | blank_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | A^T b / bias_gradient_norm | 25 | 0.009164 | 2.367025 | 3.574618 | 0.731655 | 0.800560 | 0.462951 | 0.331235 |  |  |  |  | 0.304880 | random J_error_delta not computed for old-four table |
| loss1 | A^T b / bias_gradient_norm | 25 | 0.005838 | 1.702470 | 2.781142 | 0.764206 | 0.859059 | 0.315854 | 0.173628 |  |  |  |  | 0.209058 | random J_error_delta not computed for old-four table |
| loss2 | A^T b / bias_gradient_norm | 25 | 0.005623 | 1.855112 | 2.887806 | 0.760543 | 0.868047 | 0.343340 | 0.190082 |  |  |  |  | 0.196515 | random J_error_delta not computed for old-four table |
| loss3 | A^T b / bias_gradient_norm | 25 | 0.003063 | 1.271530 | 1.898529 | 0.886612 | 0.943287 | 0.111950 | 0.080228 |  |  |  |  | 0.091452 | random J_error_delta not computed for old-four table |
| random_clean_y | J_error @ adversarial_delta | 25 | 0.036601 | 3.828716 | 6.776453 | 0.656611 | 0.675828 |  |  | 9.017979 | 8.926605 | 0.281812 | 0.278956 | 0.399197 | A^T b not computed/stored for random table |
| random_solver_y | J_error @ adversarial_delta | 25 | 0.006923 | 1.735561 | 2.681731 | 0.802382 | 0.890080 |  |  | 2.528932 | 2.343514 | 0.079029 | 0.073235 | 0.277103 | A^T b not computed/stored for random table |

CSV: `forensics/burgers_six_model_latest_wideparam_summary_20260613/combined_atb_jerror_delta_six_models_25sample.csv`

## 13. Unified Bias-Gradient Table: J_error^T clean error

This is the corrected same-math table. For all six models the main scalar is `||J_error^T error||`, where `error = model(x) - solver(x)`. The old-four rows were already stored as `bias_gradient_norm`; the random rows were newly computed from the saved full `J_error` matrices and clean residuals. The `||J_error delta||` column is kept only as a separate adversarial-delta diagnostic, not as the bias-gradient quantity.

| model | n | attack_inc | clean_mse | ||error|| | ||Jerr^T error|| mean | ||Jerr^T error|| median | Jerr^T error RMS | err_spec | err_fro | top20_R_solver | top20_L_solver | abs_cos(delta,Jerr^Terror) | ||Jerr delta|| separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 25 | 0.009164 | 0.001095 | 0.914193 | 0.462951 | 0.331235 | 0.014467 | 2.367025 | 3.574618 | 0.731655 | 0.800560 | 0.793450 |  |
| loss1 | 25 | 0.005838 | 0.000823 | 0.714167 | 0.315854 | 0.173628 | 0.009870 | 1.702470 | 2.781142 | 0.764206 | 0.859059 | 0.803056 |  |
| loss2 | 25 | 0.005623 | 0.000856 | 0.742491 | 0.343340 | 0.190082 | 0.010729 | 1.855112 | 2.887806 | 0.760543 | 0.868047 | 0.798714 |  |
| loss3 | 25 | 0.003063 | 0.000244 | 0.411851 | 0.111950 | 0.080228 | 0.003498 | 1.271530 | 1.898529 | 0.886612 | 0.943287 | 0.726458 |  |
| random_clean_y | 25 | 0.036601 | 0.008996 | 2.760806 | 3.838714 | 3.508727 | 0.119960 | 3.828716 | 6.776453 | 0.656611 | 0.675828 | 0.709841 | 9.017979 |
| random_solver_y | 25 | 0.006923 | 0.000744 | 0.676364 | 0.304904 | 0.165154 | 0.009528 | 1.735561 | 2.681731 | 0.802382 | 0.890080 | 0.730414 | 2.528932 |

CSV files:

- `forensics/burgers_six_model_latest_wideparam_summary_20260613/combined_jerror_transpose_error_six_models_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_six_models_detailed_with_jerror_transpose_error.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/random_models_jerror_transpose_error_25sample.csv`
