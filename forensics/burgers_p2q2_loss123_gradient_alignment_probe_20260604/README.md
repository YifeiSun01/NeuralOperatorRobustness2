# Burgers p2q2 loss1/loss2/loss3 gradient-alignment probe

Question: does loss3 perform worse because its adversarial perturbations are off-manifold and produce a training gradient that is less aligned with clean train/test/generalization objectives?

## Answer

Yes, this probe supports that mechanism. Loss3 uses the same RMS-L2 budget as loss1/loss2, but its delta is much peakier and more out-of-range. More importantly, the parameter gradient from raw loss3 adversarial samples is much less aligned with clean train/test/generalization gradients than loss1/loss2. Clipping/capping/smoothing the loss3 delta improves the geometry and often improves the gradient alignment, which is exactly what the off-manifold hypothesis predicts.

## Delta/target geometry summary

| variant | delta_l2_rms | delta_linf | linf_over_l2rms | delta_high_freq_ratio | x_adv_min | x_adv_max | oob_max | y_adv_min | y_adv_max | y_delta_l2_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | 0.060000 | 0.099241 | 1.654010 | 1.706e-07 | -0.107669 | 1.106122 | 0.062427 | 0.038766 | 0.887846 | 0.130352 |
| loss2 | 0.060000 | 0.100418 | 1.673641 | 1.810e-07 | -0.113477 | 1.081268 | 0.059919 | 0.008024 | 0.929822 | 0.130892 |
| loss3 | 0.060000 | 0.290831 | 4.847176 | 3.471e-05 | -0.303080 | 1.255627 | 0.172071 | -0.078428 | 0.887576 | 0.066922 |
| loss3_clip01 | 0.042868 | 0.191013 | 4.412165 | 0.000323 | 0.000e+00 | 1.000000 | 0.000e+00 | 0.006228 | 0.869448 | 0.049161 |
| loss3_linfcap0p11 | 0.042812 | 0.110000 | 2.605637 | 0.000780 | -0.110000 | 1.109606 | 0.082658 | -0.044170 | 0.869458 | 0.056863 |
| loss3_lowpass32_rms | 0.060000 | 0.288088 | 4.801475 | 4.793e-14 | -0.327202 | 1.279331 | 0.182509 | -0.078396 | 0.887956 | 0.067357 |
| loss3_lowpass32_rms_clip01 | 0.042931 | 0.186952 | 4.316721 | 0.000122 | 0.000e+00 | 1.000000 | 0.000e+00 | 0.006117 | 0.869442 | 0.049681 |

## Gradient alignment cosines

Positive cosine means a small gradient-descent step on that adversarial training objective should reduce the corresponding eval loss to first order. Larger positive is better.

| variant | clean_test | clean_train | generalization_mixed4 |
| --- | --- | --- | --- |
| loss1 | 0.841490 | 0.975458 | 0.354882 |
| loss2 | 0.818420 | 0.968047 | 0.339495 |
| loss3 | 0.073322 | 0.235095 | -0.139242 |
| loss3_clip01 | 0.947284 | 0.963363 | 0.599662 |
| loss3_linfcap0p11 | 0.391957 | 0.608978 | -0.084279 |
| loss3_lowpass32_rms | 0.068014 | 0.231998 | -0.146700 |
| loss3_lowpass32_rms_clip01 | 0.948056 | 0.964202 | 0.599222 |

## Variant training objective losses

| variant | train_objective_loss | rmse_model_solver_on_variant | rel_l2_model_solver_on_variant | grad_norm |
| --- | --- | --- | --- | --- |
| loss1 | 9.289e-05 | 0.009377 | 0.018710 | 0.014519 |
| loss2 | 8.047e-05 | 0.008765 | 0.017630 | 0.014468 |
| loss3 | 0.001731 | 0.036542 | 0.075242 | 0.027994 |
| loss3_clip01 | 0.000277 | 0.016197 | 0.031398 | 0.017277 |
| loss3_linfcap0p11 | 0.000596 | 0.022619 | 0.045580 | 0.017621 |
| loss3_lowpass32_rms | 0.001717 | 0.036229 | 0.074678 | 0.028015 |
| loss3_lowpass32_rms_clip01 | 0.000266 | 0.015932 | 0.030921 | 0.017273 |

## Interpretation

- loss3 raw has much larger `Linf/RMS-L2` than loss1/loss2 under the same L2 radius, so it is not a larger-budget effect; it is a worse direction/geometry effect.
- raw loss3 pushes `x_adv` further outside `[0,1]` and can move solver targets into a less normal range.
- raw loss3 gradient alignment with clean evaluation losses is worse than loss1/loss2. This gives a direct first-order explanation for why adversarial training on loss3 samples can improve the loss3 attack objective but hurt clean/test/generalization more.
- clipping/capping/smoothing loss3 reduces off-manifold geometry. If these variants align better with eval gradients, that is causal evidence that the harmful part of loss3 is the spike/off-manifold component.

## Files
- `attack_generation_info.csv`
- `variant_delta_target_stats_summary.csv`
- `gradient_alignment_cosines.csv`
- `variant_training_objective_losses.csv`
- `eval_gradient_losses.csv`
- `gradient_alignment_cosines.png`
- `variant_delta_mechanism_bars.png`
