# Burgers p2q2 loss1/loss2/loss3 delta manifold probe

This probe tests whether loss3 perturbations are larger, peakier, or more off-manifold than loss1/loss2 perturbations.

## Main finding

The hypothesis is strongly supported in perturbation geometry. All three controlled attacks use the same RMS-L2 radius, but loss3 concentrates that radius into much larger coordinate spikes and produces more severe out-of-range attacked inputs. This explains why loss3 can hurt clean/test/generalization performance despite using the same nominal L2 budget.

## Controlled same-source probe

Same baseline model, same five train source indices `[0, 337, 674, 1012, 1349]`, same `fast_replace_l2`, same RMS-L2 epsilon `0.06`, five attack steps. Loss1 uses a tiny random start because exact zero-start loss1 has zero gradient.

| model | delta_l2_rms | delta_linf | linf_over_l2rms | delta_fft_high_freq_ratio | delta_fft_centroid | x_adv_min | x_adv_max | below0_frac | above1_frac | oob_mean | oob_max | y_adv_min | y_adv_max | y_delta_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | 0.060000 | 0.103224 | 1.720401 | 5.744e-08 | 0.005496 | -0.104694 | 1.105990 | 0.027930 | 0.024023 | 0.002640 | 0.092892 | 0.017897 | 0.916537 | 0.124038 |
| loss2 | 0.060000 | 0.096086 | 1.601425 | 3.952e-08 | 0.003627 | -0.071398 | 1.081268 | 0.022656 | 0.010547 | 0.001182 | 0.035807 | 0.024824 | 0.873866 | 0.123668 |
| loss3 | 0.060000 | 0.300767 | 5.012780 | 1.074e-05 | 0.022204 | -0.252823 | 1.230772 | 0.037109 | 0.034180 | 0.006686 | 0.198836 | -0.053326 | 0.900980 | 0.057379 |

## Training attack peakiness at selected epochs

| model | epoch | epsilon_mean | delta_l2_rms | delta_linf | linf_over_l2rms | adv_loss | clean_loss | gain | train_target_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | 1 | 0.059912 | 0.059912 | 0.105259 | 1.756888 | 0.018561 | 0.000e+00 | 0.018561 | 0.000143 |
| loss1 | 100 | 0.059890 | 0.059890 | 0.106129 | 1.772059 | 0.018917 | 0.000e+00 | 0.018917 | 1.337e-05 |
| loss1 | 200 | 0.060115 | 0.060115 | 0.106275 | 1.767864 | 0.018959 | 0.000e+00 | 0.018959 | 1.007e-05 |
| loss1 | 400 | 0.060022 | 0.060022 | 0.106125 | 1.768109 | 0.018965 | 0.000e+00 | 0.018965 | 5.661e-06 |
| loss1 | 800 | 0.059917 | 0.059917 | 0.106226 | 1.772898 | 0.018974 | 0.000e+00 | 0.018974 | 4.821e-06 |
| loss1 | 1000 | 0.060043 | 0.060043 | 0.106446 | 1.772822 | 0.018947 | 0.000e+00 | 0.018947 | 4.359e-06 |
| loss1 | 2000 | 0.059778 | 0.059778 | 0.105372 | 1.762742 | 0.018853 | 0.000e+00 | 0.018853 | 2.553e-06 |
| loss2 | 1 | 0.060125 | 0.060125 | 0.105021 | 1.746727 | 0.019406 | 8.263e-05 | 0.019323 | 0.000143 |
| loss2 | 100 | 0.059929 | 0.059929 | 0.106664 | 1.779848 | 0.019315 | 2.150e-05 | 0.019294 | 1.788e-05 |
| loss2 | 200 | 0.059834 | 0.059834 | 0.105708 | 1.766687 | 0.019034 | 1.005e-05 | 0.019024 | 1.565e-05 |
| loss2 | 400 | 0.059992 | 0.059992 | 0.106397 | 1.773519 | 0.019026 | 4.568e-06 | 0.019021 | 5.397e-06 |
| loss2 | 800 | 0.059794 | 0.059794 | 0.105948 | 1.771896 | 0.018883 | 2.981e-06 | 0.018880 | 3.038e-06 |
| loss2 | 1000 | 0.059837 | 0.059837 | 0.106229 | 1.775320 | 0.019086 | 6.015e-06 | 0.019080 | 4.897e-06 |
| loss3 | 1 | 0.060125 | 0.060125 | 0.258993 | 4.307589 | 0.001437 | 0.000338 | 0.001099 | 0.000793 |
| loss3 | 100 | 0.059929 | 0.059929 | 0.264355 | 4.411171 | 0.000484 | 0.000154 | 0.000330 | 0.000237 |
| loss3 | 200 | 0.059834 | 0.059834 | 0.263678 | 4.406816 | 0.000424 | 0.000121 | 0.000302 | 0.000210 |
| loss3 | 400 | 0.059992 | 0.059992 | 0.268308 | 4.472383 | 0.000126 | 3.489e-05 | 9.093e-05 | 6.789e-05 |
| loss3 | 800 | 0.059794 | 0.059794 | 0.386407 | 6.462356 | 5.738e-05 | 2.408e-05 | 3.330e-05 | 3.081e-05 |
| loss3 | 1000 | 0.059837 | 0.059837 | 0.378258 | 6.321519 | 6.085e-05 | 1.675e-05 | 4.409e-05 | 3.612e-05 |

## Dataset-feature summary

| label | n_samples | x_min_mean | x_max_mean | total_variation_mean_absdiff_mean | fft_high_freq_ratio_mean | fft_spectral_centroid_norm_mean | oob_mean_violation_mean | oob_max_violation_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_probe | 5 | 0.000e+00 | 1.000000 | 0.005151 | 1.929e-09 | 0.007282 | 0.000e+00 | 0.000e+00 |
| generalization | 10000 | -0.004321 | 1.000378 | 0.002473 | 8.378e-05 | 0.002641 | 0.000301 | 0.005903 |
| test | 150 | 0.000e+00 | 1.000000 | 0.004872 | 1.749e-09 | 0.006707 | 0.000e+00 | 0.000e+00 |
| train | 1350 | 0.000e+00 | 1.000000 | 0.004948 | 1.761e-09 | 0.006782 | 0.000e+00 | 0.000e+00 |
| x_adv_loss1 | 5 | -0.035967 | 1.037740 | 0.005231 | 2.984e-09 | 0.007103 | 0.002640 | 0.092892 |
| x_adv_loss2 | 5 | 0.011941 | 0.983853 | 0.005200 | 3.590e-09 | 0.007014 | 0.001182 | 0.035807 |
| x_adv_loss3 | 5 | -0.107698 | 1.107813 | 0.005984 | 5.596e-07 | 0.007903 | 0.006686 | 0.198836 |

## Nearest dataset RMS distances

Raw nearest-neighbor RMS distance is less diagnostic because every controlled attack is forced to the same RMS-L2 radius from its clean source. It does not by itself show that loss1/loss2 are closer to the generalization set. The stronger evidence is peakiness, out-of-bound amplitude, frequency/TV shift, and target-range shift.

| query_label | generalization | test | train |
| --- | --- | --- | --- |
| clean_probe | 0.207444 | 0.230120 | 0.000e+00 |
| x_adv_loss1 | 0.204836 | 0.233840 | 0.060000 |
| x_adv_loss2 | 0.207139 | 0.225430 | 0.060000 |
| x_adv_loss3 | 0.219109 | 0.243722 | 0.060000 |

## Interpretation

- Loss1/loss2 deltas have `Linf/RMS-L2` around 1.6-1.8. Loss3 is around 5.0 in the controlled probe and 4.3 at training epoch 1, rising above 6 by late epochs.
- Therefore loss3 is not using a larger L2 budget; it is using the same L2 budget in a much more concentrated, spike-like direction.
- Controlled loss3 attacked inputs go far outside the clean input range `[0,1]`: about `[-0.253, 1.231]`. Loss1 is about `[-0.105, 1.106]`; loss2 is about `[-0.071, 1.081]`.
- Controlled loss3 solver targets also leave the usual positive clean target range, with `y_adv_min` about `-0.053`. Loss1/loss2 targets stay positive in this probe.
- The raw RMS nearest-neighbor distance does not prove loss1/loss2 are simply closer to the generalization set. It suggests the more precise mechanism is: loss1/loss2 generate smoother, lower-peak, less out-of-range perturbations, while loss3 creates sharper off-manifold perturbations.

## Output files

- `controlled_probe_delta_stats.csv`
- `training_attack_delta_epoch_selected.csv`
- `nearest_dataset_rms_distance_summary.csv`
- `sample_feature_summary.csv`
- `controlled_probe_arrays.npz`
- `delta_peakiness_linf_over_l2rms_by_epoch.png`
- `controlled_probe_xadv_delta_overlay.png`
- `controlled_probe_delta_spectra.png`
- `controlled_probe_key_metric_bars.png`
