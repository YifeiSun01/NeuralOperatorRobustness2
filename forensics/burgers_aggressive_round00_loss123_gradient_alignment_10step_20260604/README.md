# Burgers p2q2 50-step gradient-alignment trajectory

This report records per-step parameter-gradient cosine similarity during the same 50 attack-batch replay used by the input-similarity probe.
The cosine is computed before the optimizer update at each step.
Complex FNO gradients are flattened by concatenating real and imaginary parts.

## Mean cosine over 50 steps

| variant | clean_train | clean_test | generalization_mixed50 |
| --- | --- | --- | --- |
| loss1_raw | 0.939881 | 0.904997 | -0.015529 |
| loss2_raw | 0.922909 | 0.855203 | 0.037390 |
| loss3_raw | 0.920543 | 0.787590 | 0.034210 |
| loss3_clip01 |  |  |  |
| loss3_lowpass |  |  |  |
| loss3_lowpass_clip01 |  |  |  |

## Selected steps

| variant | step | eval | cosine | adv loss | eval loss | adv grad norm | eval grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 1.000000 | clean_train | 0.968554 | 0.000207 | 0.000093 | 0.018846 | 0.021224 |
| loss1_raw | 1.000000 | clean_test | 0.978412 | 0.000207 | 0.000093 | 0.018846 | 0.020572 |
| loss1_raw | 1.000000 | generalization_mixed50 | -0.026838 | 0.000207 | 0.211211 | 0.018846 | 2.649172 |
| loss1_raw | 10.000000 | clean_train | 0.987378 | 0.000055 | 0.000041 | 0.009521 | 0.012748 |
| loss1_raw | 10.000000 | clean_test | 0.978782 | 0.000055 | 0.000041 | 0.009521 | 0.008545 |
| loss1_raw | 10.000000 | generalization_mixed50 | -0.026008 | 0.000055 | 0.200322 | 0.009521 | 2.583281 |
| loss2_raw | 1.000000 | clean_train | 0.829667 | 0.000133 | 0.000093 | 0.014194 | 0.021224 |
| loss2_raw | 1.000000 | clean_test | 0.873744 | 0.000133 | 0.000093 | 0.014194 | 0.020572 |
| loss2_raw | 1.000000 | generalization_mixed50 | 0.020045 | 0.000133 | 0.211211 | 0.014194 | 2.649172 |
| loss2_raw | 10.000000 | clean_train | 0.994141 | 0.000062 | 0.000051 | 0.010230 | 0.015693 |
| loss2_raw | 10.000000 | clean_test | 0.991886 | 0.000062 | 0.000047 | 0.010230 | 0.011423 |
| loss2_raw | 10.000000 | generalization_mixed50 | -0.016721 | 0.000062 | 0.197919 | 0.010230 | 2.606192 |
| loss3_raw | 1.000000 | clean_train | 0.812215 | 0.001766 | 0.000093 | 0.033337 | 0.021224 |
| loss3_raw | 1.000000 | clean_test | 0.768663 | 0.001766 | 0.000093 | 0.033337 | 0.020572 |
| loss3_raw | 1.000000 | generalization_mixed50 | 0.183047 | 0.001766 | 0.211211 | 0.033337 | 2.649172 |
| loss3_raw | 10.000000 | clean_train | 0.979132 | 0.001119 | 0.000415 | 0.032676 | 0.023101 |
| loss3_raw | 10.000000 | clean_test | 0.976000 | 0.001119 | 0.000451 | 0.032676 | 0.018531 |
| loss3_raw | 10.000000 | generalization_mixed50 | 0.031179 | 0.001119 | 0.163976 | 0.032676 | 2.044113 |

## Files

- `gradient_alignment_by_step.csv`
- `gradient_alignment_mean_by_variant.csv`
- `gradient_probe_attack_geometry.csv`
- `optimizer_microsteps.csv`
