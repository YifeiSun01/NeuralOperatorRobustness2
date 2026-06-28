# Burgers Loss3-Targeted 52-Dataset Clean Metrics - 2026-06-11

This report records clean inference metrics for 52 datasets: the original train split, the original test split, and 50 loss3-targeted generalization datasets.

No adversarial attack is included in these numbers.

## Output Files

- 52-row dataset table: `/workspace/NeuralOperatorRobustness2/forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_clean_metrics.csv`
- compact RMSE/relative-L2/reduction table: `/workspace/NeuralOperatorRobustness2/forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_compact_rmse_relative_l2_reductions.csv`
- sample-level clean metrics: `/workspace/NeuralOperatorRobustness2/forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_sample_52_clean_metrics.csv`
- compact summary JSON: `/workspace/NeuralOperatorRobustness2/forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/summary_52_clean_metrics.json`
- per-dataset Markdown appendix: `/workspace/NeuralOperatorRobustness2/docs/burgers_loss3targeted_52dataset_clean_metrics_appendix_20260611.md`

## Dataset Sources

- Train split: `/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt`
- Test split: `/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`
- Generalization root: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers`
- Generalization policy: 50 unique non-attack semantic datasets, selected from wide visible parameters while enforcing loss3 clean-inference advantage.

## Generalization Aggregate Metrics

These are clean, non-adversarial means over the 50 generalization datasets.

| model | RMSE mean | relative L2 mean | RMSE reduction vs baseline | relative L2 reduction vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.029902 | 0.057737 | 0.00% | 0.00% |
| loss1 | 0.021027 | 0.041171 | 29.68% | 28.69% |
| loss2 | 0.022292 | 0.043542 | 25.45% | 24.59% |
| loss3 | 0.012050 | 0.023644 | 59.70% | 59.05% |

Loss3 is clean-RMSE best among all four models on 50/50 generalization datasets.

## Train/Test Metrics

| dataset_group | dataset_id | family | n_samples | baseline_rmse_mean | loss1_rmse_mean | loss2_rmse_mean | loss3_rmse_mean | rmse_loss1_vs_baseline_reduction_pct | rmse_loss2_vs_baseline_reduction_pct | rmse_loss3_vs_baseline_reduction_pct | baseline_relative_l2_mean | loss1_relative_l2_mean | loss2_relative_l2_mean | loss3_relative_l2_mean | relative_l2_loss1_vs_baseline_reduction_pct | relative_l2_loss2_vs_baseline_reduction_pct | relative_l2_loss3_vs_baseline_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | burgers_original_train_gaussian_corr0p03_seed45 | train_test_gaussian | 1350 | 0.00875634 | 0.000746925 | 0.0014018 | 0.00425008 | 91.4699 | 83.991 | 51.4628 | 0.0168149 | 0.00144669 | 0.00270588 | 0.00811692 | 91.3964 | 83.9078 | 51.7277 |
| test | burgers_original_test_gaussian_corr0p03_seed45 | train_test_gaussian | 150 | 0.00923196 | 0.000836263 | 0.00149737 | 0.00437389 | 90.9417 | 83.7806 | 52.6223 | 0.0175141 | 0.00161192 | 0.00287989 | 0.00832509 | 90.7965 | 83.5568 | 52.4665 |

## Generalization Metrics

Loss3 is clean-RMSE best on 50/50 generalization datasets.

| dataset_group | dataset_id | family | n_samples | baseline_rmse_mean | loss1_rmse_mean | loss2_rmse_mean | loss3_rmse_mean | rmse_loss1_vs_baseline_reduction_pct | rmse_loss2_vs_baseline_reduction_pct | rmse_loss3_vs_baseline_reduction_pct | baseline_relative_l2_mean | loss1_relative_l2_mean | loss2_relative_l2_mean | loss3_relative_l2_mean | relative_l2_loss1_vs_baseline_reduction_pct | relative_l2_loss2_vs_baseline_reduction_pct | relative_l2_loss3_vs_baseline_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | burgers_widevis_l3target_d00 | gaussian | 200 | 0.0208882 | 0.0150907 | 0.0169971 | 0.0086141 | 27.7548 | 18.6282 | 58.7609 | 0.0408522 | 0.029363 | 0.0331167 | 0.0168573 | 28.1239 | 18.9351 | 58.7358 |
| generalization | burgers_widevis_l3target_d01 | gaussian | 200 | 0.0276721 | 0.0200337 | 0.0224588 | 0.0114116 | 27.6032 | 18.8394 | 58.7614 | 0.0543989 | 0.0390322 | 0.043805 | 0.0223268 | 28.2482 | 19.4745 | 58.9573 |
| generalization | burgers_widevis_l3target_d02 | gaussian | 200 | 0.0251009 | 0.0117648 | 0.0129618 | 0.006844 | 53.1301 | 48.3614 | 72.7341 | 0.0345749 | 0.0165728 | 0.0183078 | 0.00957028 | 52.067 | 47.0488 | 72.3201 |
| generalization | burgers_widevis_l3target_d03 | gaussian | 200 | 0.0191863 | 0.0115113 | 0.0130524 | 0.00670663 | 40.0025 | 31.9701 | 65.0447 | 0.0310602 | 0.0189165 | 0.0214788 | 0.0109981 | 39.0972 | 30.8479 | 64.5911 |
| generalization | burgers_widevis_l3target_d04 | gaussian | 200 | 0.0254725 | 0.0209863 | 0.0228983 | 0.0121777 | 17.6121 | 10.1059 | 52.193 | 0.0513556 | 0.0423071 | 0.0461838 | 0.0246716 | 17.6193 | 10.0705 | 51.9593 |
| generalization | burgers_widevis_l3target_d05 | gaussian | 200 | 0.0226314 | 0.0129631 | 0.0137023 | 0.00763588 | 42.7209 | 39.4548 | 66.2599 | 0.0313889 | 0.0183174 | 0.0193959 | 0.0107191 | 41.6437 | 38.2077 | 65.8508 |
| generalization | burgers_widevis_l3target_d06 | gaussian | 200 | 0.0167598 | 0.0109371 | 0.0125211 | 0.00648762 | 34.7417 | 25.2908 | 61.2905 | 0.0327972 | 0.0213187 | 0.0244102 | 0.0126991 | 34.9985 | 25.5723 | 61.2798 |
| generalization | burgers_widevis_l3target_d07 | gaussian | 200 | 0.0193857 | 0.0135002 | 0.0146471 | 0.00800456 | 30.3598 | 24.4436 | 58.7089 | 0.0317653 | 0.0222023 | 0.0242007 | 0.0131909 | 30.1051 | 23.8139 | 58.4738 |
| generalization | burgers_widevis_l3target_d08 | gaussian | 200 | 0.0500174 | 0.0359293 | 0.0381041 | 0.0211596 | 28.1662 | 23.8182 | 57.6955 | 0.100065 | 0.0702784 | 0.0747747 | 0.04139 | 29.7671 | 25.2738 | 58.6368 |
| generalization | burgers_widevis_l3target_d09 | gaussian | 200 | 0.0403829 | 0.0340501 | 0.0360233 | 0.0200901 | 15.682 | 10.7957 | 50.2509 | 0.0820954 | 0.0683746 | 0.0723551 | 0.0406068 | 16.7133 | 11.8646 | 50.5371 |
| generalization | burgers_widevis_l3target_d10 | gaussian | 200 | 0.020726 | 0.0159082 | 0.0174856 | 0.00964836 | 23.2452 | 15.6342 | 53.448 | 0.0415239 | 0.0318957 | 0.0351527 | 0.0194148 | 23.1872 | 15.3434 | 53.2442 |
| generalization | burgers_widevis_l3target_d11 | gaussian | 200 | 0.0141294 | 0.00811282 | 0.00930917 | 0.00496369 | 42.582 | 34.1149 | 64.8697 | 0.0254921 | 0.0146475 | 0.0168222 | 0.00895354 | 42.541 | 34.0098 | 64.8771 |
| generalization | burgers_widevis_l3target_d12 | matern | 200 | 0.0314931 | 0.0160395 | 0.0184662 | 0.0097523 | 49.0697 | 41.3643 | 69.0336 | 0.0624315 | 0.0314564 | 0.0362379 | 0.0192566 | 49.6145 | 41.9558 | 69.1556 |
| generalization | burgers_widevis_l3target_d13 | matern | 200 | 0.0246071 | 0.0192188 | 0.0204238 | 0.0114397 | 21.8975 | 17.0005 | 53.5106 | 0.0485871 | 0.0382188 | 0.0405391 | 0.0227876 | 21.3396 | 16.5641 | 53.0994 |
| generalization | burgers_widevis_l3target_d14 | matern | 200 | 0.0406989 | 0.0316088 | 0.0327957 | 0.0192811 | 22.335 | 19.4188 | 52.625 | 0.0807368 | 0.0628297 | 0.0649649 | 0.0384552 | 22.1796 | 19.535 | 52.3697 |
| generalization | burgers_widevis_l3target_d15 | matern | 200 | 0.0224166 | 0.0117303 | 0.0124414 | 0.00726109 | 47.6716 | 44.4993 | 67.6084 | 0.0310679 | 0.0164624 | 0.0175141 | 0.0101431 | 47.0116 | 43.6264 | 67.3518 |
| generalization | burgers_widevis_l3target_d16 | matern | 200 | 0.0328539 | 0.0146891 | 0.0173989 | 0.00940408 | 55.2895 | 47.0418 | 71.3761 | 0.0636105 | 0.0285655 | 0.0338427 | 0.0181474 | 55.093 | 46.797 | 71.4711 |
| generalization | burgers_widevis_l3target_d17 | matern | 200 | 0.053124 | 0.0358779 | 0.0354411 | 0.0221682 | 32.464 | 33.2861 | 58.2709 | 0.164134 | 0.103489 | 0.102434 | 0.0651341 | 36.9483 | 37.5913 | 60.3166 |
| generalization | burgers_widevis_l3target_d18 | matern | 200 | 0.0556907 | 0.024611 | 0.0268062 | 0.0159511 | 55.8076 | 51.8659 | 71.3577 | 0.0704237 | 0.0316297 | 0.0344162 | 0.0202709 | 55.0865 | 51.1297 | 71.2158 |
| generalization | burgers_widevis_l3target_d19 | matern | 200 | 0.0657738 | 0.0340247 | 0.0371968 | 0.0227439 | 48.2702 | 43.4473 | 65.421 | 0.128807 | 0.0668141 | 0.072625 | 0.0447212 | 48.1287 | 43.6173 | 65.2806 |
| generalization | burgers_widevis_l3target_d20 | matern | 200 | 0.0600265 | 0.0330162 | 0.0367483 | 0.0219962 | 44.9973 | 38.7799 | 63.3559 | 0.118415 | 0.065505 | 0.0726558 | 0.0435063 | 44.6819 | 38.6432 | 63.2596 |
| generalization | burgers_widevis_l3target_d21 | matern | 200 | 0.0319121 | 0.0178953 | 0.0182332 | 0.0122685 | 43.9233 | 42.8644 | 61.5553 | 0.0408987 | 0.0234174 | 0.0238214 | 0.0159332 | 42.743 | 41.755 | 61.0423 |
| generalization | burgers_widevis_l3target_d22 | powerlaw_fourier | 200 | 0.0200714 | 0.0149388 | 0.0159108 | 0.00857737 | 25.5717 | 20.7291 | 57.2658 | 0.0395236 | 0.0293483 | 0.0313316 | 0.0169108 | 25.7448 | 20.7268 | 57.2133 |
| generalization | burgers_widevis_l3target_d23 | powerlaw_fourier | 200 | 0.0390221 | 0.0261423 | 0.0280729 | 0.0156037 | 33.0065 | 28.0589 | 60.0131 | 0.0773083 | 0.0519303 | 0.0556855 | 0.0309996 | 32.827 | 27.9696 | 59.9014 |
| generalization | burgers_widevis_l3target_d24 | powerlaw_fourier | 200 | 0.0199316 | 0.0158 | 0.016637 | 0.00941856 | 20.729 | 16.5293 | 52.7456 | 0.0394083 | 0.0312432 | 0.0329804 | 0.0186319 | 20.7191 | 16.311 | 52.7208 |
| generalization | burgers_widevis_l3target_d25 | powerlaw_fourier | 200 | 0.0416259 | 0.032111 | 0.0335815 | 0.0187515 | 22.8581 | 19.3254 | 54.9523 | 0.0835464 | 0.0635897 | 0.0666253 | 0.0374485 | 23.887 | 20.2535 | 55.1764 |
| generalization | burgers_widevis_l3target_d26 | powerlaw_fourier | 200 | 0.0165337 | 0.0111138 | 0.012056 | 0.00674869 | 32.7811 | 27.0825 | 59.1823 | 0.0327536 | 0.0220166 | 0.0239217 | 0.0133947 | 32.7812 | 26.9647 | 59.1046 |
| generalization | burgers_widevis_l3target_d27 | powerlaw_fourier | 200 | 0.03383 | 0.0221518 | 0.022926 | 0.0135326 | 34.5202 | 32.2317 | 59.998 | 0.0660564 | 0.0427566 | 0.0443239 | 0.0262884 | 35.2726 | 32.9 | 60.2032 |
| generalization | burgers_widevis_l3target_d28 | powerlaw_fourier | 200 | 0.0192153 | 0.0137511 | 0.0149497 | 0.00835283 | 28.4365 | 22.1989 | 56.5303 | 0.0372505 | 0.0266486 | 0.0289889 | 0.0162378 | 28.461 | 22.1787 | 56.4091 |
| generalization | burgers_widevis_l3target_d29 | powerlaw_fourier | 200 | 0.0219552 | 0.0121687 | 0.0130049 | 0.00750623 | 44.5747 | 40.766 | 65.8111 | 0.0305766 | 0.0171564 | 0.0183737 | 0.0105579 | 43.8904 | 39.9094 | 65.4705 |
| generalization | burgers_widevis_l3target_d30 | powerlaw_fourier | 200 | 0.0391768 | 0.0307324 | 0.0321123 | 0.018782 | 21.5545 | 18.0323 | 52.0583 | 0.0790297 | 0.0611553 | 0.0633274 | 0.0375092 | 22.6173 | 19.8689 | 52.5378 |
| generalization | burgers_widevis_l3target_d31 | powerlaw_fourier | 200 | 0.0370013 | 0.0333889 | 0.0337367 | 0.0204489 | 9.76306 | 8.82294 | 44.7346 | 0.0729226 | 0.0662119 | 0.0667295 | 0.0407514 | 9.20249 | 8.49267 | 44.1169 |
| generalization | burgers_widevis_l3target_d32 | powerlaw_fourier | 200 | 0.0201384 | 0.01256 | 0.0132342 | 0.00773249 | 37.6317 | 34.2838 | 61.6033 | 0.0287153 | 0.0180228 | 0.0190076 | 0.0110723 | 37.2363 | 33.8069 | 61.4411 |
| generalization | burgers_widevis_l3target_d33 | powerlaw_fourier | 200 | 0.016644 | 0.0117138 | 0.0126598 | 0.00717432 | 29.6214 | 23.938 | 56.8955 | 0.0328834 | 0.0231265 | 0.0250744 | 0.0141831 | 29.671 | 23.7473 | 56.8684 |
| generalization | burgers_widevis_l3target_d34 | powerlaw_fourier | 200 | 0.0422841 | 0.0276993 | 0.0292449 | 0.0169424 | 34.4923 | 30.8372 | 59.9321 | 0.0821174 | 0.054257 | 0.0572676 | 0.0331389 | 33.9276 | 30.2614 | 59.6445 |
| generalization | burgers_widevis_l3target_d35 | powerlaw_fourier | 200 | 0.0377971 | 0.0321148 | 0.0328861 | 0.0195687 | 15.0337 | 12.993 | 48.2269 | 0.0755834 | 0.0638614 | 0.065403 | 0.0388788 | 15.5087 | 13.4691 | 48.5618 |
| generalization | burgers_widevis_l3target_d36 | sine_mixture | 200 | 0.0375488 | 0.0245485 | 0.0245218 | 0.0106424 | 34.6224 | 34.6936 | 71.6571 | 0.049647 | 0.0324902 | 0.0324518 | 0.0140812 | 34.5575 | 34.6348 | 71.6373 |
| generalization | burgers_widevis_l3target_d37 | sine_mixture | 200 | 0.0278759 | 0.0189146 | 0.0212742 | 0.00861603 | 32.147 | 23.6824 | 69.0915 | 0.0371069 | 0.0251775 | 0.0283181 | 0.0114676 | 32.1486 | 23.685 | 69.0956 |
| generalization | burgers_widevis_l3target_d38 | sine_mixture | 200 | 0.0226499 | 0.0206316 | 0.0216936 | 0.0101236 | 8.91094 | 4.22211 | 55.3041 | 0.0445982 | 0.0406199 | 0.0427178 | 0.019924 | 8.92043 | 4.21652 | 55.3256 |
| generalization | burgers_widevis_l3target_d39 | sine_mixture | 200 | 0.0230209 | 0.0219059 | 0.022153 | 0.0105043 | 4.84316 | 3.76977 | 54.3705 | 0.0458299 | 0.0436129 | 0.0440974 | 0.0208945 | 4.83736 | 3.7803 | 54.4085 |
| generalization | burgers_widevis_l3target_d40 | sine_mixture | 200 | 0.0189297 | 0.0177194 | 0.0179467 | 0.00854363 | 6.39329 | 5.19254 | 54.8664 | 0.0377055 | 0.0352959 | 0.0357438 | 0.0170045 | 6.39063 | 5.20276 | 54.9019 |
| generalization | burgers_widevis_l3target_d41 | sine_mixture | 200 | 0.0337055 | 0.0350339 | 0.0344868 | 0.0170079 | -3.94098 | -2.31778 | 49.5399 | 0.0929582 | 0.0964855 | 0.0949278 | 0.0468649 | -3.79448 | -2.11879 | 49.585 |
| generalization | burgers_widevis_l3target_d42 | sine_mixture | 200 | 0.031532 | 0.0309423 | 0.03351 | 0.0152545 | 1.87018 | -6.27293 | 51.6223 | 0.061954 | 0.0607625 | 0.0658165 | 0.0299411 | 1.9232 | -6.23445 | 51.6721 |
| generalization | burgers_widevis_l3target_d43 | sine_mixture | 200 | 0.0195099 | 0.0160159 | 0.0166358 | 0.00824367 | 17.9089 | 14.7314 | 57.7462 | 0.0384397 | 0.0315512 | 0.0327784 | 0.0162367 | 17.9202 | 14.7278 | 57.7607 |
| generalization | burgers_widevis_l3target_d44 | sine_mixture | 200 | 0.0306223 | 0.0299367 | 0.0299964 | 0.0151803 | 2.23891 | 2.04376 | 50.4271 | 0.0609887 | 0.0596258 | 0.059732 | 0.0302076 | 2.23479 | 2.06066 | 50.4703 |
| generalization | burgers_widevis_l3target_d45 | sine_mixture | 200 | 0.0111659 | 0.00815988 | 0.00861422 | 0.00426401 | 26.9212 | 22.8521 | 61.8121 | 0.022281 | 0.0164339 | 0.0173096 | 0.00874176 | 26.2424 | 22.3125 | 60.7659 |
| generalization | burgers_widevis_l3target_d46 | sine_mixture | 200 | 0.0294824 | 0.0141254 | 0.0187268 | 0.00854877 | 52.0888 | 36.4813 | 71.0038 | 0.0646295 | 0.0305784 | 0.0401177 | 0.0182951 | 52.6866 | 37.9266 | 71.6924 |
| generalization | burgers_widevis_l3target_d47 | sine_mixture | 200 | 0.02987 | 0.0292982 | 0.0275311 | 0.0160864 | 1.91411 | 7.83004 | 46.1452 | 0.0842116 | 0.0826487 | 0.0776391 | 0.0454101 | 1.856 | 7.80475 | 46.0762 |
| generalization | burgers_widevis_l3target_d48 | sawtooth | 200 | 0.0159441 | 0.0159191 | 0.0128519 | 0.00457624 | 0.156675 | 19.3938 | 71.2981 | 0.0222854 | 0.0222505 | 0.0179634 | 0.00639632 | 0.156675 | 19.3938 | 71.2981 |
| generalization | burgers_widevis_l3target_d49 | square_wave | 200 | 0.0370513 | 0.0222884 | 0.0275533 | 0.0097359 | 39.8445 | 25.6346 | 73.7232 | 0.0800706 | 0.0480706 | 0.0594117 | 0.0209648 | 39.9647 | 25.8008 | 73.8171 |

## Parameter And Characteristic Columns

The CSV includes the dataset path, display label, description, family, solver, `nx`, `nu`, `t_final`, observed `x` statistics, spectral centroid/low/mid/high fractions, total variation, flattened parameter columns, and full JSON parameters.

Key flattened parameter columns include `param_base_family`, `param_kernel`, `param_correlation_length`, `param_matern_nu`, `param_spectral_alpha`, `param_k0`, `param_frequencies`, `param_decay`, `param_frequency`, `param_duty`, `param_target_min`, and `param_target_max`.
