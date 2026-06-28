# Burgers p2q2 50-step gradient-alignment trajectory

This report records per-step parameter-gradient cosine similarity during the same 50 attack-batch replay used by the input-similarity probe.
The cosine is computed before the optimizer update at each step.
Complex FNO gradients are flattened by concatenating real and imaginary parts.

## Mean cosine over 50 steps

| variant | clean_train | clean_test | generalization_mixed50 |
| --- | --- | --- | --- |
| loss1_raw | 0.918757 | 0.922245 | 0.235689 |
| loss2_raw | 0.853998 | 0.903673 | 0.166701 |
| loss3_raw | 0.948527 | 0.885902 | 0.735448 |
| loss3_clip01 |  |  |  |
| loss3_lowpass |  |  |  |
| loss3_lowpass_clip01 |  |  |  |

## Selected steps

| variant | step | eval | cosine | adv loss | eval loss | adv grad norm | eval grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 1.000000 | clean_train | 0.968554 | 0.000207 | 0.000093 | 0.018846 | 0.021224 |
| loss1_raw | 1.000000 | clean_test | 0.978412 | 0.000207 | 0.000093 | 0.018846 | 0.020572 |
| loss1_raw | 1.000000 | generalization_mixed50 | 0.828362 | 0.000207 | 0.001816 | 0.018846 | 0.037415 |
| loss1_raw | 10.000000 | clean_train | 0.988762 | 0.000059 | 0.000044 | 0.011575 | 0.015538 |
| loss1_raw | 10.000000 | clean_test | 0.986361 | 0.000059 | 0.000043 | 0.011575 | 0.010889 |
| loss1_raw | 10.000000 | generalization_mixed50 | 0.933273 | 0.000059 | 0.000952 | 0.011575 | 0.035454 |
| loss1_raw | 25.000000 | clean_train | 0.997277 | 0.000044 | 0.000041 | 0.010203 | 0.012547 |
| loss1_raw | 25.000000 | clean_test | 0.983973 | 0.000044 | 0.000040 | 0.010203 | 0.008632 |
| loss1_raw | 25.000000 | generalization_mixed50 | 0.881940 | 0.000044 | 0.000640 | 0.010203 | 0.030306 |
| loss1_raw | 50.000000 | clean_train | 0.929273 | 0.000030 | 0.000022 | 0.002922 | 0.002105 |
| loss1_raw | 50.000000 | clean_test | 0.877030 | 0.000030 | 0.000024 | 0.002922 | 0.003568 |
| loss1_raw | 50.000000 | generalization_mixed50 | -0.115070 | 0.000030 | 0.000539 | 0.002922 | 0.019902 |
| loss2_raw | 1.000000 | clean_train | 0.829667 | 0.000133 | 0.000093 | 0.014194 | 0.021224 |
| loss2_raw | 1.000000 | clean_test | 0.873744 | 0.000133 | 0.000093 | 0.014194 | 0.020572 |
| loss2_raw | 1.000000 | generalization_mixed50 | 0.527888 | 0.000133 | 0.001816 | 0.014194 | 0.037415 |
| loss2_raw | 10.000000 | clean_train | 0.994289 | 0.000062 | 0.000052 | 0.010378 | 0.015816 |
| loss2_raw | 10.000000 | clean_test | 0.991894 | 0.000062 | 0.000047 | 0.010378 | 0.011512 |
| loss2_raw | 10.000000 | generalization_mixed50 | 0.933555 | 0.000062 | 0.000933 | 0.010378 | 0.036255 |
| loss2_raw | 25.000000 | clean_train | 0.994790 | 0.000053 | 0.000041 | 0.008542 | 0.010266 |
| loss2_raw | 25.000000 | clean_test | 0.990784 | 0.000053 | 0.000040 | 0.008542 | 0.006416 |
| loss2_raw | 25.000000 | generalization_mixed50 | 0.838924 | 0.000053 | 0.000630 | 0.008542 | 0.029311 |
| loss2_raw | 50.000000 | clean_train | 0.413452 | 0.000027 | 0.000024 | 0.001703 | 0.001269 |
| loss2_raw | 50.000000 | clean_test | 0.925362 | 0.000027 | 0.000025 | 0.001703 | 0.003213 |
| loss2_raw | 50.000000 | generalization_mixed50 | -0.337336 | 0.000027 | 0.000526 | 0.001703 | 0.021449 |
| loss3_raw | 1.000000 | clean_train | 0.812215 | 0.001766 | 0.000093 | 0.033337 | 0.021224 |
| loss3_raw | 1.000000 | clean_test | 0.768663 | 0.001766 | 0.000093 | 0.033337 | 0.020572 |
| loss3_raw | 1.000000 | generalization_mixed50 | 0.997186 | 0.001766 | 0.001816 | 0.033337 | 0.037415 |
| loss3_raw | 10.000000 | clean_train | 0.988604 | 0.001075 | 0.000396 | 0.043642 | 0.029556 |
| loss3_raw | 10.000000 | clean_test | 0.978821 | 0.001075 | 0.000417 | 0.043642 | 0.025222 |
| loss3_raw | 10.000000 | generalization_mixed50 | 0.964874 | 0.001075 | 0.000704 | 0.043642 | 0.041805 |
| loss3_raw | 25.000000 | clean_train | 0.944539 | 0.000829 | 0.000290 | 0.019126 | 0.012352 |
| loss3_raw | 25.000000 | clean_test | 0.864372 | 0.000829 | 0.000264 | 0.019126 | 0.018642 |
| loss3_raw | 25.000000 | generalization_mixed50 | 0.334435 | 0.000829 | 0.000714 | 0.019126 | 0.016318 |
| loss3_raw | 50.000000 | clean_train | 0.956764 | 0.000739 | 0.000222 | 0.025104 | 0.022625 |
| loss3_raw | 50.000000 | clean_test | 0.947532 | 0.000739 | 0.000232 | 0.025104 | 0.024317 |
| loss3_raw | 50.000000 | generalization_mixed50 | 0.910746 | 0.000739 | 0.000460 | 0.025104 | 0.031154 |

## Files

- `gradient_alignment_by_step.csv`
- `gradient_alignment_mean_by_variant.csv`
- `gradient_probe_attack_geometry.csv`
- `optimizer_microsteps.csv`
