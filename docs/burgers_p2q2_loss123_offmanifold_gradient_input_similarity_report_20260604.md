# Burgers p2q2 Loss1/Loss2/Loss3 Off-Manifold Mechanism Report

Date: 2026-06-04

This report records the diagnostic experiments used to explain why, on the current Burgers p2q2 benchmark, loss1/loss2 adversarial training can outperform raw loss3 adversarial training even when loss3 is the stronger attack objective.

The short version is:

- Raw loss3 does not merely make a larger RMS-L2 perturbation. It uses the same RMS-L2 budget but concentrates it into much larger coordinate spikes.
- Those spikes push the attacked input `x_adv` farther outside the normal clean input range `[0,1]`.
- The parameter gradient induced by raw loss3 adversarial samples is poorly aligned with the gradients that would reduce clean train/test/generalization loss; for generalization it is negative in the quick gradient-alignment probe.
- Clipping raw loss3 `x_adv` back to `[0,1]` removes the epoch-1 jump. Over 50 attack-batch steps, lowpass+clip is best, so both range and peakiness matter.

## Artifacts

Main local result directories:

- `forensics/burgers_p2q2_loss123_gradient_alignment_probe_20260604`
- `forensics/burgers_p2q2_epoch1_jump_causal_probe_20260604`
- `forensics/burgers_p2q2_loss123_delta_manifold_probe_20260603`
- `forensics/burgers_p2q2_loss123_50step_input_similarity_probe_20260604`

New reusable script:

- `tools/probe_burgers_p2q2_loss123_50step_input_similarity.py`

The 50-step run contains the full per-step/per-dataset CSV files:

- `eval_metrics.csv`
- `eval_split_summary.csv`
- `step_attack_geometry.csv`
- `x_similarity_by_split.csv`
- `optimizer_microsteps.csv`
- `compact_step_summary.csv`
- `mean_attack_geometry_by_variant.csv`
- `correlation_summary.csv`

## Mathematical Definition Of The Gradient-Alignment Test

Let the trained neural operator be

$$
f_\theta(x),
$$

and let the numerical solver be

$$
S(x).
$$

For an adversarial variant, the attack constructs

$$
x_{\mathrm{adv}} = x + \delta.
$$

The optimizer training target is always recomputed with the solver:

$$
y_{\mathrm{adv}} = S(x_{\mathrm{adv}}).
$$

The adversarial training loss for that batch is

$$
L_{\mathrm{adv}}(\theta)
= \frac{1}{B}\sum_{i=1}^{B}
\left\| f_\theta(x_{\mathrm{adv},i}) - S(x_{\mathrm{adv},i}) \right\|_2^2.
$$

The parameter gradient induced by that adversarial batch is

$$
g_{\mathrm{adv}}
= \nabla_\theta L_{\mathrm{adv}}(\theta).
$$

For an evaluation split, such as clean train, clean test, or generated generalization, define

$$
L_{E}(\theta)
= \frac{1}{|E|}\sum_{(x_j,y_j)\in E}
\left\| f_\theta(x_j) - y_j \right\|_2^2,
$$

and its parameter gradient

$$
g_E = \nabla_\theta L_E(\theta).
$$

The diagnostic computes the cosine similarity between these two parameter gradients:

$$
\operatorname{cos}(g_{\mathrm{adv}}, g_E)
= \frac{\langle g_{\mathrm{adv}}, g_E \rangle}
{\|g_{\mathrm{adv}}\|_2\,\|g_E\|_2}.
$$

A gradient-descent optimizer update on the adversarial batch is

$$
\theta^+ = \theta - \eta g_{\mathrm{adv}}.
$$

The first-order change in eval loss is

$$
L_E(\theta^+)
\approx L_E(\theta)
- \eta \langle g_E, g_{\mathrm{adv}} \rangle.
$$

Therefore:

| cosine sign | meaning |
| --- | --- |
| positive | training on this adversarial batch should reduce the eval loss to first order |
| near zero | training on this adversarial batch is almost orthogonal to that eval objective |
| negative | training on this adversarial batch should increase the eval loss to first order |

Important: this is a gradient in model-parameter space, not a PDE spatial gradient and not a gradient with respect to `delta`.

Implementation sketch:

```python
pred = model(x_adv)
loss_adv = mse(pred, solver(x_adv))
loss_adv.backward()
g_adv = concatenate_parameter_grads(model)

pred = model(x_eval)
loss_eval = mse(pred, y_eval)
loss_eval.backward()
g_eval = concatenate_parameter_grads(model)

cosine = dot(g_adv, g_eval) / (norm(g_adv) * norm(g_eval))
```

Caveat: the quick diagnostic flattened complex FNO spectral gradients with `.float()`, which can discard imaginary components. The direct epoch-1 and 50-step replay experiments below do not depend on this cosine approximation.

## Attack Objectives

For Burgers, the three attack objectives are attack-generation choices. Optimizer training still uses solver labels at the attacked input.

| objective | attack target used while generating `delta` | solver in attack forward | solver backward |
| --- | --- | ---: | ---: |
| loss1 | `model(x_clean).detach()` | no | no |
| loss2 | `solver(x_clean).detach()` | yes | no |
| loss3 | `solver(x_adv)` | yes | yes |

Training target after attack:

$$
\text{train on } f_\theta(x_{\mathrm{adv}}) \to S(x_{\mathrm{adv}}).
$$

For clipped/lowpass variants, the transformed input is used and the solver target is recomputed:

$$
x_{\mathrm{used}} = T(x_{\mathrm{adv}}),
\qquad
\text{train on } f_\theta(x_{\mathrm{used}}) \to S(x_{\mathrm{used}}).
$$

## Gradient-Alignment Probe

Positive cosine means a small gradient-descent step on that adversarial batch should reduce the corresponding eval loss to first order. Negative cosine means it should push that eval loss upward.

| variant | clean train | clean test | generalization |
| --- | ---: | ---: | ---: |
| loss1 | 0.975458 | 0.841490 | 0.354882 |
| loss2 | 0.968047 | 0.818420 | 0.339495 |
| loss3 raw | 0.235095 | 0.073322 | -0.139242 |
| loss3 clip to `[0,1]` | 0.963363 | 0.947284 | 0.599662 |
| loss3 Linf cap 0.11 | 0.608978 | 0.391957 | -0.084279 |
| loss3 lowpass32 RMS, no clip | 0.231998 | 0.068014 | -0.146700 |
| loss3 lowpass32 RMS + clip | 0.964202 | 0.948056 | 0.599222 |

Interpretation:

- Raw loss3 is badly aligned with clean train/test and negatively aligned with generalization.
- Clipping raw loss3 `x_adv` back into `[0,1]` changes generalization cosine from `-0.139242` to `0.599662`.
- Lowpass without clipping does not fix the negative generalization alignment, so the issue is not only high frequency. Range/off-manifold excursion is critical.

## Gradient-Alignment Variant Geometry

| variant | delta RMS-L2 | delta Linf | Linf/RMS | high-frequency ratio | x_adv min | x_adv max | OOB max | y_adv min | y_adv max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0.060000 | 0.099241 | 1.654010 | 1.706e-07 | -0.107669 | 1.106122 | 0.062427 | 0.038766 | 0.887846 |
| loss2 | 0.060000 | 0.100418 | 1.673641 | 1.810e-07 | -0.113477 | 1.081268 | 0.059919 | 0.008024 | 0.929822 |
| loss3 raw | 0.060000 | 0.290831 | 4.847176 | 3.471e-05 | -0.303080 | 1.255627 | 0.172071 | -0.078428 | 0.887576 |
| loss3 clip01 | 0.042868 | 0.191013 | 4.412165 | 3.230e-04 | 0.000000 | 1.000000 | 0.000000 | 0.006228 | 0.869448 |
| loss3 Linf cap 0.11 | 0.042812 | 0.110000 | 2.605637 | 7.800e-04 | -0.110000 | 1.109606 | 0.082658 | -0.044170 | 0.869458 |
| loss3 lowpass32 RMS | 0.060000 | 0.288088 | 4.801475 | 4.793e-14 | -0.327202 | 1.279331 | 0.182509 | -0.078396 | 0.887956 |
| loss3 lowpass32 RMS + clip | 0.042931 | 0.186952 | 4.316721 | 1.220e-04 | 0.000000 | 1.000000 | 0.000000 | 0.006117 | 0.869442 |

## Variant Training Objective Losses

| variant | training objective loss | RMSE model-vs-solver on variant | relative L2 model-vs-solver on variant | grad norm |
| --- | ---: | ---: | ---: | ---: |
| loss1 | 9.289e-05 | 0.009377 | 0.018710 | 0.014519 |
| loss2 | 8.047e-05 | 0.008765 | 0.017630 | 0.014468 |
| loss3 raw | 0.001731 | 0.036542 | 0.075242 | 0.027994 |
| loss3 clip01 | 0.000277 | 0.016197 | 0.031398 | 0.017277 |
| loss3 Linf cap 0.11 | 0.000596 | 0.022619 | 0.045580 | 0.017621 |
| loss3 lowpass32 RMS | 0.001717 | 0.036229 | 0.074678 | 0.028015 |
| loss3 lowpass32 RMS + clip | 0.000266 | 0.015932 | 0.030921 | 0.017273 |

## Delta Manifold Probe

Controlled same-source probe: same baseline model, same five train source indices `[0, 337, 674, 1012, 1349]`, same `fast_replace_l2`, same RMS-L2 epsilon `0.06`, five attack steps.

| model | delta RMS-L2 | delta Linf | Linf/RMS | high-frequency ratio | spectral centroid | x_adv min | x_adv max | below0 frac | above1 frac | OOB mean | OOB max | y_adv min | y_adv max | y delta RMS-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0.060000 | 0.103224 | 1.720401 | 5.744e-08 | 0.005496 | -0.104694 | 1.105990 | 0.027930 | 0.024023 | 0.002640 | 0.092892 | 0.017897 | 0.916537 | 0.124038 |
| loss2 | 0.060000 | 0.096086 | 1.601425 | 3.952e-08 | 0.003627 | -0.071398 | 1.081268 | 0.022656 | 0.010547 | 0.001182 | 0.035807 | 0.024824 | 0.873866 | 0.123668 |
| loss3 | 0.060000 | 0.300767 | 5.012780 | 1.074e-05 | 0.022204 | -0.252823 | 1.230772 | 0.037109 | 0.034180 | 0.006686 | 0.198836 | -0.053326 | 0.900980 | 0.057379 |

Nearest-neighbor RMS distance by itself was less diagnostic because all controlled attacks are forced to the same RMS-L2 radius. The stronger evidence is peakiness, out-of-bound amplitude, frequency/TV shift, and solver-target range shift.

| query label | nearest generalization RMS | nearest test RMS | nearest train RMS |
| --- | ---: | ---: | ---: |
| clean probe | 0.207444 | 0.230120 | 0.000000 |
| x_adv loss1 | 0.204836 | 0.233840 | 0.060000 |
| x_adv loss2 | 0.207139 | 0.225430 | 0.060000 |
| x_adv loss3 | 0.219109 | 0.243722 | 0.060000 |

## Epoch-1 Replay Causal Probe

This replay starts each variant from the same baseline and trains for one epoch. It directly tests whether the observed loss3 jump is caused by the optimizer update on raw loss3 adversarial samples.

### Before/After Epoch 1

| variant | split | RMSE before | relative L2 before | RMSE after | relative L2 after | RMSE delta | relative L2 delta | RMSE ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | generalization | 0.017073 | 0.029876 | 0.009742 | 0.017167 | -0.007331 | -0.012709 | 0.570625 |
| loss2 | generalization | 0.017073 | 0.029876 | 0.014452 | 0.025100 | -0.002621 | -0.004776 | 0.846477 |
| loss3 raw | generalization | 0.017073 | 0.029876 | 0.022311 | 0.038836 | 0.005238 | 0.008960 | 1.306785 |
| loss3 clip01 | generalization | 0.017073 | 0.029876 | 0.010765 | 0.018916 | -0.006308 | -0.010960 | 0.630511 |
| loss1 | test | 0.009232 | 0.017514 | 0.006093 | 0.011651 | -0.003139 | -0.005863 | 0.660038 |
| loss2 | test | 0.009232 | 0.017514 | 0.006520 | 0.012517 | -0.002712 | -0.004997 | 0.706241 |
| loss3 raw | test | 0.009232 | 0.017514 | 0.016414 | 0.030846 | 0.007182 | 0.013332 | 1.777903 |
| loss3 clip01 | test | 0.009232 | 0.017514 | 0.007603 | 0.014622 | -0.001629 | -0.002892 | 0.823578 |
| loss1 | train | 0.008741 | 0.016745 | 0.005631 | 0.010806 | -0.003109 | -0.005939 | 0.644283 |
| loss2 | train | 0.008741 | 0.016745 | 0.006087 | 0.011730 | -0.002653 | -0.005015 | 0.696420 |
| loss3 raw | train | 0.008741 | 0.016745 | 0.015289 | 0.028784 | 0.006549 | 0.012039 | 1.749236 |
| loss3 clip01 | train | 0.008741 | 0.016745 | 0.007935 | 0.015256 | -0.000805 | -0.001489 | 0.907866 |

### Epoch-1 Attack Geometry

| variant | delta RMS-L2 | delta Linf | Linf/RMS | x_adv min | x_adv max | OOB mean | OOB max | y_adv min | y_adv max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0.059912 | 0.105390 | 1.757806 | -0.164897 | 1.159173 | 0.002389 | 0.160763 | -0.045551 | 1.040876 |
| loss2 | 0.060145 | 0.105708 | 1.755212 | -0.169783 | 1.146660 | 0.002264 | 0.154169 | -0.027364 | 1.031709 |
| loss3 raw | 0.060145 | 0.259881 | 4.309852 | -0.347389 | 1.379393 | 0.004287 | 0.353879 | -0.141717 | 1.129097 |
| loss3 clip01 | 0.045998 | 0.209142 | 4.567167 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000693 | 0.997556 |

Epoch-1 result: raw loss3 immediately worsens train/test/generalization. Clipping `x_adv` back to `[0,1]` removes the jump.

## 50-Step Input-Similarity Replay

This run uses 50 attack-batch training steps per variant. One step means one attacked batch plus all optimizer microbatches for that attacked batch. Every variant starts from the same baseline and uses the same batch schedule.

Variants:

- `loss1_raw`: attack objective loss1, raw training input.
- `loss2_raw`: attack objective loss2, raw training input.
- `loss3_raw`: attack objective loss3, raw training input.
- `loss3_clip01`: raw loss3 attack, then clip used input to `[0,1]` and recompute solver label.
- `loss3_lowpass`: raw loss3 attack, then lowpass delta and recompute solver label, no clip.
- `loss3_lowpass_clip01`: raw loss3 attack, lowpass delta, clip to `[0,1]`, and recompute solver label.

### Generalization Trajectory

| variant | step | generalization RMSE | generalization relative L2 | score |
| --- | ---: | ---: | ---: | ---: |
| loss1_raw | 0 | 0.017932 | 0.031019 | 96.992499 |
| loss1_raw | 1 | 0.016311 | 0.028179 | 97.259723 |
| loss1_raw | 10 | 0.009541 | 0.016532 | 98.373816 |
| loss1_raw | 25 | 0.009407 | 0.016281 | 98.398105 |
| loss1_raw | 50 | 0.009523 | 0.016470 | 98.379723 |
| loss2_raw | 0 | 0.017932 | 0.031019 | 96.992499 |
| loss2_raw | 1 | 0.026024 | 0.044764 | 95.721484 |
| loss2_raw | 10 | 0.009955 | 0.017254 | 98.304043 |
| loss2_raw | 25 | 0.009458 | 0.016359 | 98.390551 |
| loss2_raw | 50 | 0.017677 | 0.030421 | 97.049601 |
| loss3_raw | 0 | 0.017932 | 0.031019 | 96.992499 |
| loss3_raw | 1 | 0.050429 | 0.086783 | 92.022606 |
| loss3_raw | 10 | 0.039986 | 0.068779 | 93.576695 |
| loss3_raw | 25 | 0.030109 | 0.051833 | 95.076746 |
| loss3_raw | 50 | 0.030958 | 0.053344 | 94.939312 |
| loss3_clip01 | 0 | 0.017932 | 0.031019 | 96.992499 |
| loss3_clip01 | 1 | 0.012560 | 0.021764 | 97.870543 |
| loss3_clip01 | 10 | 0.014661 | 0.025388 | 97.524388 |
| loss3_clip01 | 25 | 0.011999 | 0.020836 | 97.959827 |
| loss3_clip01 | 50 | 0.017244 | 0.029722 | 97.115502 |
| loss3_lowpass | 0 | 0.017932 | 0.031019 | 96.992499 |
| loss3_lowpass | 1 | 0.044759 | 0.077065 | 92.849795 |
| loss3_lowpass | 10 | 0.023860 | 0.041234 | 96.040215 |
| loss3_lowpass | 25 | 0.022698 | 0.039208 | 96.228703 |
| loss3_lowpass | 50 | 0.013058 | 0.022585 | 97.791756 |
| loss3_lowpass_clip01 | 0 | 0.017932 | 0.031019 | 96.992499 |
| loss3_lowpass_clip01 | 1 | 0.010221 | 0.017748 | 98.256863 |
| loss3_lowpass_clip01 | 10 | 0.010127 | 0.017536 | 98.277191 |
| loss3_lowpass_clip01 | 25 | 0.008370 | 0.014541 | 98.567555 |
| loss3_lowpass_clip01 | 50 | 0.008096 | 0.014040 | 98.616021 |

### Final Step 50: Prediction Loss And Input Geometry

| variant | gen RMSE | used x min | used x max | used OOB mean | delta Linf | Linf/RMS | nearest RMS to gen | max cosine to gen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1_raw | 0.009523 | -0.161144 | 1.150800 | 0.002741 | 0.107192 | 1.764398 | 0.262319 | 0.904290 |
| loss2_raw | 0.017677 | -0.148129 | 1.130181 | 0.002180 | 0.109007 | 1.794423 | 0.262868 | 0.902421 |
| loss3_raw | 0.030958 | -0.424790 | 1.348697 | 0.003574 | 0.271184 | 4.446360 | 0.272716 | 0.895653 |
| loss3_clip01 | 0.017244 | 0.000000 | 1.000000 | 0.000000 | 0.220776 | 4.403685 | 0.265980 | 0.900452 |
| loss3_lowpass | 0.013058 | -0.369841 | 1.297184 | 0.004304 | 0.208869 | 3.427496 | 0.271333 | 0.896362 |
| loss3_lowpass_clip01 | 0.008096 | 0.000000 | 1.000000 | 0.000000 | 0.185207 | 3.900976 | 0.265837 | 0.898919 |

### Mean Geometry Over 50 Steps

| variant | mean OOB | mean delta Linf | mean Linf/RMS | mean x min | mean x max | mean raw x min | mean raw x max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1_raw | 0.002631 | 0.106429 | 1.769006 | -0.158452 | 1.156130 | -0.158452 | 1.156130 |
| loss2_raw | 0.002629 | 0.106925 | 1.776882 | -0.163500 | 1.150583 | -0.163500 | 1.150583 |
| loss3_raw | 0.003395 | 0.257001 | 4.264567 | -0.347729 | 1.338770 | -0.347729 | 1.338770 |
| loss3_clip01 | 0.000000 | 0.210240 | 4.471055 | 0.000000 | 1.000000 | -0.355741 | 1.361630 |
| loss3_lowpass | 0.004358 | 0.212762 | 3.530524 | -0.340528 | 1.345472 | -0.367049 | 1.379713 |
| loss3_lowpass_clip01 | 0.000000 | 0.179289 | 3.896905 | 0.000000 | 1.000000 | -0.369048 | 1.369043 |

### Correlation Check

Across all variant-step points except baseline:

| metric | Pearson correlation with generalization RMSE |
| --- | ---: |
| used OOB mean | 0.483848 |
| used Linf/RMS | 0.442194 |
| used delta Linf | 0.693245 |

The strongest of these simple geometry correlations is `delta Linf`, which supports the peakiness explanation.

## Consolidated Interpretation

1. Raw loss3 creates an adversarial input distribution that is much more peak-like and more off-range than loss1/loss2 under the same RMS-L2 budget.
2. In the quick gradient-alignment test, raw loss3's parameter gradient is negatively aligned with the generalization gradient. That means a small training step on raw loss3 samples is predicted, to first order, to worsen generalization loss.
3. The direct epoch-1 replay confirms the sign: raw loss3 increases train/test/generalization RMSE immediately, while loss1/loss2 reduce it.
4. Clipping raw loss3 `x_adv` back to `[0,1]` removes the epoch-1 jump and turns the generalization gradient cosine from negative to strongly positive.
5. Lowpass without clipping does not fix the gradient-alignment problem in the quick cosine probe, but over 50 steps it partially helps, suggesting shape/peakiness matters too.
6. Lowpass+clip is the strongest 50-step variant. This supports the refined mechanism: the harmful part of raw loss3 is not simply the name loss3, but the geometry of its `x_adv`: large pointwise peaks plus off-range/off-manifold values.
7. Nearest-neighbor RMS and input cosine to generated generalization data are not by themselves enough to explain the whole effect. The stronger signals are `delta Linf`, `Linf/RMS`, and input range violation.

## Practical Recommendation

For a clean follow-up experiment, compare these variants over a longer horizon:

- loss1 raw
- loss2 raw
- loss3 raw
- loss3 clipped to `[0,1]`
- loss3 lowpass+clip

The key metrics to track should be:

- train/test/generalization RMSE and relative L2
- `delta_l2_rms`
- `delta_linf`
- `Linf/RMS`
- `x_adv_min`, `x_adv_max`, OOB mean/max
- gradient-alignment cosine with clean train/test/generalization, computed with complex gradients split into real and imaginary parts
- model-vs-solver Jacobian error spectral norm and singular-vector similarity at selected checkpoints
