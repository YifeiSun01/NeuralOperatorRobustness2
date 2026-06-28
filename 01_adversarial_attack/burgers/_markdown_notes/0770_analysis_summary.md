# 50-step input-similarity probe analysis

## What was tested

Every variant starts from the same Burgers FNO baseline and uses the same 50 attack-batch schedule. `loss3_clip01`, `loss3_lowpass`, and `loss3_lowpass_clip01` first generate the same kind of raw loss3 attack, then modify the attacked input used for training and recompute `solver(x_used)`.

## Main conclusion

The 50-step result supports the off-manifold hypothesis, but with a nuance: **range clipping is the strongest early fix, while peakiness/smoothness also matters over 50 steps.** Raw loss3 remains worst. Clipping raw loss3 to `[0,1]` removes the epoch-1 jump, but clip-only is not the best long-run 50-step variant. Lowpass+clip is best, which means the harmful component is not just the loss3 name; it is the geometry of the generated `x_adv`: large out-of-range values plus a peakier delta shape.

## Generalization trajectory


| variant | step | gen RMSE | gen rel L2 | score |
| --- | --- | --- | --- | --- |
| loss1_raw | 0.000000 | 0.017932 | 0.031019 | 96.992499 |
| loss1_raw | 1.000000 | 0.016311 | 0.028179 | 97.259723 |
| loss1_raw | 10.000000 | 0.009541 | 0.016532 | 98.373816 |
| loss1_raw | 25.000000 | 0.009407 | 0.016281 | 98.398105 |
| loss1_raw | 50.000000 | 0.009523 | 0.016470 | 98.379723 |
| loss3_raw | 0.000000 | 0.017932 | 0.031019 | 96.992499 |
| loss3_raw | 1.000000 | 0.050429 | 0.086783 | 92.022606 |
| loss3_raw | 10.000000 | 0.039986 | 0.068779 | 93.576695 |
| loss3_raw | 25.000000 | 0.030109 | 0.051833 | 95.076746 |
| loss3_raw | 50.000000 | 0.030958 | 0.053344 | 94.939312 |
| loss3_clip01 | 0.000000 | 0.017932 | 0.031019 | 96.992499 |
| loss3_clip01 | 1.000000 | 0.012560 | 0.021764 | 97.870543 |
| loss3_clip01 | 10.000000 | 0.014661 | 0.025388 | 97.524388 |
| loss3_clip01 | 25.000000 | 0.011999 | 0.020836 | 97.959827 |
| loss3_clip01 | 50.000000 | 0.017244 | 0.029722 | 97.115502 |
| loss3_lowpass | 0.000000 | 0.017932 | 0.031019 | 96.992499 |
| loss3_lowpass | 1.000000 | 0.044759 | 0.077065 | 92.849795 |
| loss3_lowpass | 10.000000 | 0.023860 | 0.041234 | 96.040215 |
| loss3_lowpass | 25.000000 | 0.022698 | 0.039208 | 96.228703 |
| loss3_lowpass | 50.000000 | 0.013058 | 0.022585 | 97.791756 |
| loss3_lowpass_clip01 | 0.000000 | 0.017932 | 0.031019 | 96.992499 |
| loss3_lowpass_clip01 | 1.000000 | 0.010221 | 0.017748 | 98.256863 |
| loss3_lowpass_clip01 | 10.000000 | 0.010127 | 0.017536 | 98.277191 |
| loss3_lowpass_clip01 | 25.000000 | 0.008370 | 0.014541 | 98.567555 |
| loss3_lowpass_clip01 | 50.000000 | 0.008096 | 0.014040 | 98.616021 |

## Final step 50: loss and input geometry


| variant | gen RMSE | used x min | used x max | used OOB mean | delta Linf | Linf/RMS | nearest RMS to gen | max cosine to gen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 0.009523 | -0.161144 | 1.150800 | 0.002741 | 0.107192 | 1.764398 | 0.262319 | 0.904290 |
| loss2_raw | 0.017677 | -0.148129 | 1.130181 | 0.002180 | 0.109007 | 1.794423 | 0.262868 | 0.902421 |
| loss3_raw | 0.030958 | -0.424790 | 1.348697 | 0.003574 | 0.271184 | 4.446360 | 0.272716 | 0.895653 |
| loss3_clip01 | 0.017244 | 0.000000 | 1.000000 | 0.000000 | 0.220776 | 4.403685 | 0.265980 | 0.900452 |
| loss3_lowpass | 0.013058 | -0.369841 | 1.297184 | 0.004304 | 0.208869 | 3.427496 | 0.271333 | 0.896362 |
| loss3_lowpass_clip01 | 0.008096 | 0.000000 | 1.000000 | 0.000000 | 0.185207 | 3.900976 | 0.265837 | 0.898919 |

## Mean geometry over 50 steps


| variant | mean OOB | mean delta Linf | mean Linf/RMS | mean x min | mean x max | mean raw x min | mean raw x max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 0.002631 | 0.106429 | 1.769006 | -0.158452 | 1.156130 | -0.158452 | 1.156130 |
| loss2_raw | 0.002629 | 0.106925 | 1.776882 | -0.163500 | 1.150583 | -0.163500 | 1.150583 |
| loss3_raw | 0.003395 | 0.257001 | 4.264567 | -0.347729 | 1.338770 | -0.347729 | 1.338770 |
| loss3_clip01 | 0.000000 | 0.210240 | 4.471055 | 0.000000 | 1.000000 | -0.355741 | 1.361630 |
| loss3_lowpass | 0.004358 | 0.212762 | 3.530524 | -0.340528 | 1.345472 | -0.367049 | 1.379713 |
| loss3_lowpass_clip01 | 0.000000 | 0.179289 | 3.896905 | 0.000000 | 1.000000 | -0.369048 | 1.369043 |

## Correlation check

Across all variant-step points except baseline:


| metric | Pearson corr with gen RMSE |
| --- | --- |
| corr_gen_rmse_vs_used_oob_mean | 0.483848 |
| corr_gen_rmse_vs_used_linf_over_l2rms | 0.442194 |
| corr_gen_rmse_vs_used_delta_linf | 0.693245 |

## Interpretation

- Step 1 shows the causal jump clearly: raw loss3 sends generalization RMSE from `0.017932` to `0.050429`; clip sends it to `0.012560`; lowpass+clip sends it to `0.010221`.
- Raw loss3 has much larger pointwise peaks: mean 50-step `Linf/RMS` is `4.265`, compared with `1.769` for loss1 and `1.777` for loss2.
- Raw loss3 also leaves the normal range much farther: mean used x range is roughly `[-0.348, 1.339]`; loss1/loss2 are around `[-0.16, 1.15]`; clipped variants are exactly `[0,1]`.
- Pure lowpass without clip is mixed: it still has off-range inputs, but it lowers peakiness and improves final gen RMSE from raw loss3 `0.030958` to `0.013058`.
- Pure clip removes off-range and removes the first-step jump, but by step 50 it drifts to gen RMSE `0.017244`; lowpass+clip stays best at `0.008096`.
- Nearest-RMS/cosine-to-generalization differences exist but are modest. The stronger explanatory variables here are pointwise range and delta peakiness, especially `delta Linf`.

## Files added by this analysis

- `compact_step_summary.csv`
- `mean_attack_geometry_by_variant.csv`
- `correlation_summary.csv`
