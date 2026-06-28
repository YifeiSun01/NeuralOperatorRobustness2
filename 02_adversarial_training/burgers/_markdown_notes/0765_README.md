# Burgers p2q2 50-step gradient-alignment trajectory

This report records per-step parameter-gradient cosine similarity during the same 50 attack-batch replay used by the input-similarity probe.
The cosine is computed before the optimizer update at each step.
Complex FNO gradients are flattened by concatenating real and imaginary parts.

## Mean cosine over 50 steps

| variant | clean_train | clean_test | generalization_mixed4 |
| --- | --- | --- | --- |
| loss1_raw | 0.900233 | 0.906246 | 0.033863 |
| loss2_raw | 0.899806 | 0.854951 | 0.119059 |
| loss3_raw | 0.935371 | 0.849992 | 0.276633 |
| loss3_clip01 |  |  |  |
| loss3_lowpass |  |  |  |
| loss3_lowpass_clip01 |  |  |  |

## Selected steps

| variant | step | eval | cosine | adv loss | eval loss | adv grad norm | eval grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 1.000000 | clean_train | 0.968554 | 0.000207 | 0.000093 | 0.018846 | 0.021224 |
| loss1_raw | 1.000000 | clean_test | 0.978412 | 0.000207 | 0.000093 | 0.018846 | 0.020572 |
| loss1_raw | 1.000000 | generalization_mixed4 | 0.115906 | 0.000207 | 0.013548 | 0.018846 | 0.125060 |
| loss1_raw | 10.000000 | clean_train | 0.937802 | 0.000054 | 0.000036 | 0.006079 | 0.009783 |
| loss1_raw | 10.000000 | clean_test | 0.952391 | 0.000054 | 0.000039 | 0.006079 | 0.006388 |
| loss1_raw | 10.000000 | generalization_mixed4 | -0.044845 | 0.000054 | 0.010066 | 0.006079 | 0.096107 |
| loss1_raw | 25.000000 | clean_train | 0.749181 | 0.000039 | 0.000029 | 0.002571 | 0.004043 |
| loss1_raw | 25.000000 | clean_test | 0.932092 | 0.000039 | 0.000036 | 0.002571 | 0.003058 |
| loss1_raw | 25.000000 | generalization_mixed4 | 0.337330 | 0.000039 | 0.008847 | 0.002571 | 0.096553 |
| loss1_raw | 50.000000 | clean_train | 0.988130 | 0.000033 | 0.000027 | 0.005339 | 0.005582 |
| loss1_raw | 50.000000 | clean_test | 0.923762 | 0.000033 | 0.000027 | 0.005339 | 0.003440 |
| loss1_raw | 50.000000 | generalization_mixed4 | 0.032435 | 0.000033 | 0.007769 | 0.005339 | 0.087556 |
| loss2_raw | 1.000000 | clean_train | 0.829667 | 0.000133 | 0.000093 | 0.014194 | 0.021224 |
| loss2_raw | 1.000000 | clean_test | 0.873744 | 0.000133 | 0.000093 | 0.014194 | 0.020572 |
| loss2_raw | 1.000000 | generalization_mixed4 | 0.040681 | 0.000133 | 0.013548 | 0.014194 | 0.125060 |
| loss2_raw | 10.000000 | clean_train | 0.994147 | 0.000062 | 0.000051 | 0.010474 | 0.015903 |
| loss2_raw | 10.000000 | clean_test | 0.991728 | 0.000062 | 0.000048 | 0.010474 | 0.011575 |
| loss2_raw | 10.000000 | generalization_mixed4 | 0.289876 | 0.000062 | 0.010040 | 0.010474 | 0.098903 |
| loss2_raw | 25.000000 | clean_train | 0.986491 | 0.000056 | 0.000041 | 0.006826 | 0.008377 |
| loss2_raw | 25.000000 | clean_test | 0.977126 | 0.000056 | 0.000039 | 0.006826 | 0.004792 |
| loss2_raw | 25.000000 | generalization_mixed4 | 0.276335 | 0.000056 | 0.008565 | 0.006826 | 0.094046 |
| loss2_raw | 50.000000 | clean_train | 0.996196 | 0.000031 | 0.000028 | 0.006574 | 0.008161 |
| loss2_raw | 50.000000 | clean_test | 0.984969 | 0.000031 | 0.000025 | 0.006574 | 0.004998 |
| loss2_raw | 50.000000 | generalization_mixed4 | 0.218753 | 0.000031 | 0.007879 | 0.006574 | 0.091197 |
| loss3_raw | 1.000000 | clean_train | 0.812215 | 0.001766 | 0.000093 | 0.033337 | 0.021224 |
| loss3_raw | 1.000000 | clean_test | 0.768663 | 0.001766 | 0.000093 | 0.033337 | 0.020572 |
| loss3_raw | 1.000000 | generalization_mixed4 | 0.507348 | 0.001766 | 0.013548 | 0.033337 | 0.125060 |
| loss3_raw | 10.000000 | clean_train | 0.977912 | 0.001117 | 0.000414 | 0.032299 | 0.023194 |
| loss3_raw | 10.000000 | clean_test | 0.977483 | 0.001117 | 0.000448 | 0.032299 | 0.018294 |
| loss3_raw | 10.000000 | generalization_mixed4 | 0.514226 | 0.001117 | 0.006328 | 0.032299 | 0.075749 |
| loss3_raw | 25.000000 | clean_train | 0.985660 | 0.000628 | 0.000236 | 0.053198 | 0.029949 |
| loss3_raw | 25.000000 | clean_test | 0.989072 | 0.000628 | 0.000233 | 0.053198 | 0.036427 |
| loss3_raw | 25.000000 | generalization_mixed4 | -0.031567 | 0.000628 | 0.005660 | 0.053198 | 0.064358 |
| loss3_raw | 50.000000 | clean_train | 0.957231 | 0.000729 | 0.000225 | 0.025525 | 0.023509 |
| loss3_raw | 50.000000 | clean_test | 0.951364 | 0.000729 | 0.000235 | 0.025525 | 0.022996 |
| loss3_raw | 50.000000 | generalization_mixed4 | 0.469251 | 0.000729 | 0.005084 | 0.025525 | 0.063931 |

## Files

- `gradient_alignment_by_step.csv`
- `gradient_alignment_mean_by_variant.csv`
- `gradient_probe_attack_geometry.csv`
- `optimizer_microsteps.csv`
