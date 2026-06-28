# Burgers p2q2 50-step gradient-alignment trajectory

This report records per-step parameter-gradient cosine similarity during the same 50 attack-batch replay used by the input-similarity probe.
The cosine is computed before the optimizer update at each step.
Complex FNO gradients are flattened by concatenating real and imaginary parts.

## Mean cosine over 50 steps

| variant | clean_train | clean_test | generalization_mixed50 |
| --- | --- | --- | --- |
| loss1_raw | 0.937148 | 0.913948 | 0.093088 |
| loss2_raw | 0.952633 | 0.953660 | 0.091060 |
| loss3_raw | 0.961161 | 0.950632 | 0.231641 |
| loss3_clip01 |  |  |  |
| loss3_lowpass |  |  |  |
| loss3_lowpass_clip01 |  |  |  |

## Selected steps

| variant | step | eval | cosine | adv loss | eval loss | adv grad norm | eval grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 1.000000 | clean_train | 0.978106 | 0.000216 | 0.000080 | 0.018887 | 0.019716 |
| loss1_raw | 1.000000 | clean_test | 0.981288 | 0.000216 | 0.000088 | 0.018887 | 0.019692 |
| loss1_raw | 1.000000 | generalization_mixed50 | 0.428938 | 0.000216 | 0.007402 | 0.018887 | 0.071390 |
| loss1_raw | 10.000000 | clean_train | 0.949237 | 0.000057 | 0.000030 | 0.007295 | 0.006370 |
| loss1_raw | 10.000000 | clean_test | 0.911590 | 0.000057 | 0.000034 | 0.007295 | 0.005379 |
| loss1_raw | 10.000000 | generalization_mixed50 | 0.613956 | 0.000057 | 0.005291 | 0.007295 | 0.062804 |
| loss2_raw | 1.000000 | clean_train | 0.854370 | 0.000143 | 0.000080 | 0.014226 | 0.019716 |
| loss2_raw | 1.000000 | clean_test | 0.872354 | 0.000143 | 0.000088 | 0.014226 | 0.019692 |
| loss2_raw | 1.000000 | generalization_mixed50 | 0.201419 | 0.000143 | 0.007402 | 0.014226 | 0.071390 |
| loss2_raw | 10.000000 | clean_train | 0.995121 | 0.000076 | 0.000043 | 0.011480 | 0.011218 |
| loss2_raw | 10.000000 | clean_test | 0.996490 | 0.000076 | 0.000053 | 0.011480 | 0.011504 |
| loss2_raw | 10.000000 | generalization_mixed50 | -0.669901 | 0.000076 | 0.004682 | 0.011480 | 0.053568 |
| loss3_raw | 1.000000 | clean_train | 0.777052 | 0.001839 | 0.000080 | 0.034734 | 0.019716 |
| loss3_raw | 1.000000 | clean_test | 0.754715 | 0.001839 | 0.000088 | 0.034734 | 0.019692 |
| loss3_raw | 1.000000 | generalization_mixed50 | 0.811973 | 0.001839 | 0.007402 | 0.034734 | 0.071390 |
| loss3_raw | 10.000000 | clean_train | 0.976311 | 0.000396 | 0.000085 | 0.014402 | 0.009688 |
| loss3_raw | 10.000000 | clean_test | 0.973894 | 0.000396 | 0.000105 | 0.014402 | 0.011798 |
| loss3_raw | 10.000000 | generalization_mixed50 | -0.448022 | 0.000396 | 0.002901 | 0.014402 | 0.045568 |

## Files

- `gradient_alignment_by_step.csv`
- `gradient_alignment_mean_by_variant.csv`
- `gradient_probe_attack_geometry.csv`
- `optimizer_microsteps.csv`
