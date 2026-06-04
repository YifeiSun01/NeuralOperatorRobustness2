# Burgers p2q2 loss1/loss2/loss3 50-step input-similarity replay

This probe replays early adversarial training from the same retained Burgers FNO baseline.
It runs `50` attack-batch training steps per variant. One step means one attacked batch plus all optimizer microbatches for that attacked batch.

Variants:

- `loss1_raw`: attack objective `loss1`, transform `raw`
- `loss2_raw`: attack objective `loss2`, transform `raw`
- `loss3_raw`: attack objective `loss3`, transform `raw`
- `loss3_clip01`: attack objective `loss3`, transform `clip01`
- `loss3_lowpass`: attack objective `loss3`, transform `lowpass`
- `loss3_lowpass_clip01`: attack objective `loss3`, transform `lowpass_clip01`

The raw attack input is always recorded.  For transformed loss3 variants, the actual used training input is also recorded after clipping/lowpass and after recomputing `solver(x_used)`.

## Final prediction loss by split

| variant | step | split | RMSE | relative L2 | score |
| --- | --- | --- | --- | --- | --- |
| loss1_raw | 50.000000 | train | 0.005151 | 0.009688 | 99.040513 |
| loss1_raw | 50.000000 | test | 0.005721 | 0.010644 | 98.946795 |
| loss1_raw | 50.000000 | generalization | 0.009523 | 0.016470 | 98.379723 |
| loss2_raw | 50.000000 | train | 0.007824 | 0.014715 | 98.549806 |
| loss2_raw | 50.000000 | test | 0.008195 | 0.015246 | 98.498267 |
| loss2_raw | 50.000000 | generalization | 0.017677 | 0.030421 | 97.049601 |
| loss3_clip01 | 50.000000 | train | 0.009662 | 0.018173 | 98.215144 |
| loss3_clip01 | 50.000000 | test | 0.009722 | 0.018087 | 98.223456 |
| loss3_clip01 | 50.000000 | generalization | 0.017244 | 0.029722 | 97.115502 |
| loss3_lowpass | 50.000000 | train | 0.006434 | 0.012102 | 98.804242 |
| loss3_lowpass | 50.000000 | test | 0.007120 | 0.013245 | 98.692798 |
| loss3_lowpass | 50.000000 | generalization | 0.013058 | 0.022585 | 97.791756 |
| loss3_lowpass_clip01 | 50.000000 | train | 0.004172 | 0.007846 | 99.221498 |
| loss3_lowpass_clip01 | 50.000000 | test | 0.004643 | 0.008638 | 99.143620 |
| loss3_lowpass_clip01 | 50.000000 | generalization | 0.008096 | 0.014040 | 98.616021 |
| loss3_raw | 50.000000 | train | 0.016808 | 0.031613 | 96.935588 |
| loss3_raw | 50.000000 | test | 0.017048 | 0.031716 | 96.925869 |
| loss3_raw | 50.000000 | generalization | 0.030958 | 0.053344 | 94.939312 |

## Final attack/input geometry

| variant | used delta RMS | used delta Linf | used Linf/RMS | used x min | used x max | used OOB mean | raw x min | raw x max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 0.060744 | 0.107192 | 1.764398 | -0.161144 | 1.150800 | 0.002741 | -0.161144 | 1.150800 |
| loss2_raw | 0.060744 | 0.109007 | 1.794423 | -0.148129 | 1.130181 | 0.002180 | -0.148129 | 1.130181 |
| loss3_clip01 | 0.050282 | 0.220776 | 4.403685 | 0.000000 | 1.000000 | 0.000000 | -0.361071 | 1.340487 |
| loss3_lowpass | 0.060744 | 0.208869 | 3.427496 | -0.369841 | 1.297184 | 0.004304 | -0.390137 | 1.305699 |
| loss3_lowpass_clip01 | 0.048735 | 0.185207 | 3.900976 | 0.000000 | 1.000000 | 0.000000 | -0.449298 | 1.375652 |
| loss3_raw | 0.060744 | 0.271184 | 4.446360 | -0.424790 | 1.348697 | 0.003574 | -0.424790 | 1.348697 |

## Final used-input similarity to generalization

| variant | step | nearest RMS | max cosine | angle deg | spectrum cosine | spectrum L1 | below ref min frac | above ref max frac | outside ref range mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 50.000000 | 0.262319 | 0.904290 | 24.970522 | 0.682331 | 0.599786 | 0.000000 | 0.000000 | 0.000000 |
| loss2_raw | 50.000000 | 0.262868 | 0.902421 | 25.250612 | 0.637589 | 0.621885 | 0.000000 | 0.000000 | 0.000000 |
| loss3_clip01 | 50.000000 | 0.265980 | 0.900452 | 25.505772 | 0.651643 | 0.620120 | 0.000000 | 0.000000 | 0.000000 |
| loss3_lowpass | 50.000000 | 0.271333 | 0.896362 | 26.020088 | 0.654792 | 0.618223 | 0.000228 | 0.000000 | 0.000007 |
| loss3_lowpass_clip01 | 50.000000 | 0.265837 | 0.898919 | 25.697350 | 0.656188 | 0.616837 | 0.000000 | 0.000000 | 0.000000 |
| loss3_raw | 50.000000 | 0.272716 | 0.895653 | 26.137003 | 0.666187 | 0.614123 | 0.000407 | 0.000065 | 0.000015 |

## Output files

- `eval_metrics.csv`: per-dataset prediction metrics at baseline and after every step.
- `eval_split_summary.csv`: train/test/generalization/ALL split averages at baseline and after every step.
- `step_attack_geometry.csv`: raw and used attack geometry each step.
- `x_similarity_by_split.csv`: raw/used input similarity to train/test/generalization input distributions.
- `optimizer_microsteps.csv`: optimizer microbatch losses and gradient norms.
- `config.json`: exact run configuration.
