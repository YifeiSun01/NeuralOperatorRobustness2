# Burgers p2q2 50-step gradient-alignment trajectory

This report records per-step parameter-gradient cosine similarity during the same 50 attack-batch replay used by the input-similarity probe.
The cosine is computed before the optimizer update at each step.
Complex FNO gradients are flattened by concatenating real and imaginary parts.

## Mean cosine over 50 steps

| variant | clean_train | clean_test | generalization_mixed50 |
| --- | --- | --- | --- |
| loss1_raw | 0.937120 | 0.915230 | 0.310066 |
| loss2_raw | 0.924793 | 0.856638 | 0.442757 |
| loss3_raw | 0.940144 | 0.826327 | 0.711823 |
| loss3_clip01 |  |  |  |
| loss3_lowpass |  |  |  |
| loss3_lowpass_clip01 |  |  |  |

## Selected steps

| variant | step | eval | cosine | adv loss | eval loss | adv grad norm | eval grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 1.000000 | clean_train | 0.968554 | 0.000207 | 0.000093 | 0.018846 | 0.021224 |
| loss1_raw | 1.000000 | clean_test | 0.978412 | 0.000207 | 0.000093 | 0.018846 | 0.020572 |
| loss1_raw | 1.000000 | generalization_mixed50 | 0.828362 | 0.000207 | 0.001816 | 0.018846 | 0.037415 |
| loss1_raw | 10.000000 | clean_train | 0.988344 | 0.000054 | 0.000038 | 0.009554 | 0.012670 |
| loss1_raw | 10.000000 | clean_test | 0.983035 | 0.000054 | 0.000043 | 0.009554 | 0.009485 |
| loss1_raw | 10.000000 | generalization_mixed50 | 0.936222 | 0.000054 | 0.000936 | 0.009554 | 0.033630 |
| loss2_raw | 1.000000 | clean_train | 0.829667 | 0.000133 | 0.000093 | 0.014194 | 0.021224 |
| loss2_raw | 1.000000 | clean_test | 0.873744 | 0.000133 | 0.000093 | 0.014194 | 0.020572 |
| loss2_raw | 1.000000 | generalization_mixed50 | 0.527888 | 0.000133 | 0.001816 | 0.014194 | 0.037415 |
| loss2_raw | 10.000000 | clean_train | 0.993929 | 0.000063 | 0.000052 | 0.010568 | 0.016034 |
| loss2_raw | 10.000000 | clean_test | 0.991472 | 0.000063 | 0.000048 | 0.010568 | 0.011704 |
| loss2_raw | 10.000000 | generalization_mixed50 | 0.937828 | 0.000063 | 0.000937 | 0.010568 | 0.036490 |
| loss3_raw | 1.000000 | clean_train | 0.812215 | 0.001766 | 0.000093 | 0.033337 | 0.021224 |
| loss3_raw | 1.000000 | clean_test | 0.768663 | 0.001766 | 0.000093 | 0.033337 | 0.020572 |
| loss3_raw | 1.000000 | generalization_mixed50 | 0.997186 | 0.001766 | 0.001816 | 0.033337 | 0.037415 |
| loss3_raw | 10.000000 | clean_train | 0.967398 | 0.001046 | 0.000345 | 0.029054 | 0.020819 |
| loss3_raw | 10.000000 | clean_test | 0.962252 | 0.001046 | 0.000365 | 0.029054 | 0.016509 |
| loss3_raw | 10.000000 | generalization_mixed50 | 0.914415 | 0.001046 | 0.000679 | 0.029054 | 0.033544 |

## Files

- `gradient_alignment_by_step.csv`
- `gradient_alignment_mean_by_variant.csv`
- `gradient_probe_attack_geometry.csv`
- `optimizer_microsteps.csv`
