# Burgers p2q2 50-step gradient-alignment trajectory

This report records per-step parameter-gradient cosine similarity during the same 50 attack-batch replay used by the input-similarity probe.
The cosine is computed before the optimizer update at each step.
Complex FNO gradients are flattened by concatenating real and imaginary parts.

## Mean cosine over 50 steps

| variant | clean_train | clean_test | generalization_mixed4 |
| --- | --- | --- | --- |
| loss1_raw | 0.904479 | 0.893988 | 0.588985 |
| loss2_raw | 0.873912 | 0.877328 | 0.693592 |
| loss3_raw | 0.948028 | 0.873137 | 0.242348 |
| loss3_clip01 | 0.881226 | 0.828796 | 0.520322 |
| loss3_lowpass | 0.907376 | 0.736374 | 0.472702 |
| loss3_lowpass_clip01 | 0.896101 | 0.852410 | 0.568615 |

## Selected steps

| variant | step | eval | cosine | adv loss | eval loss | adv grad norm | eval grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1_raw | 1.000000 | clean_train | 0.962474 | 0.000206 | 0.000093 | 0.018495 | 0.021224 |
| loss1_raw | 1.000000 | clean_test | 0.974860 | 0.000206 | 0.000093 | 0.018495 | 0.020572 |
| loss1_raw | 1.000000 | generalization_mixed4 | 0.866072 | 0.000206 | 0.000331 | 0.018495 | 0.027294 |
| loss1_raw | 10.000000 | clean_train | 0.857277 | 0.000049 | 0.000035 | 0.003052 | 0.006216 |
| loss1_raw | 10.000000 | clean_test | 0.890155 | 0.000049 | 0.000043 | 0.003052 | 0.003925 |
| loss1_raw | 10.000000 | generalization_mixed4 | -0.401344 | 0.000049 | 0.000117 | 0.003052 | 0.017348 |
| loss1_raw | 25.000000 | clean_train | 0.994787 | 0.000042 | 0.000033 | 0.008687 | 0.010330 |
| loss1_raw | 25.000000 | clean_test | 0.989143 | 0.000042 | 0.000033 | 0.008687 | 0.008028 |
| loss1_raw | 25.000000 | generalization_mixed4 | 0.878706 | 0.000042 | 0.000088 | 0.008687 | 0.008998 |
| loss1_raw | 50.000000 | clean_train | 0.946019 | 0.000030 | 0.000025 | 0.002797 | 0.002789 |
| loss1_raw | 50.000000 | clean_test | 0.360605 | 0.000030 | 0.000026 | 0.002797 | 0.002885 |
| loss1_raw | 50.000000 | generalization_mixed4 | 0.386690 | 0.000030 | 0.000076 | 0.002797 | 0.007618 |
| loss2_raw | 1.000000 | clean_train | 0.829423 | 0.000133 | 0.000093 | 0.014188 | 0.021224 |
| loss2_raw | 1.000000 | clean_test | 0.873462 | 0.000133 | 0.000093 | 0.014188 | 0.020572 |
| loss2_raw | 1.000000 | generalization_mixed4 | 0.608182 | 0.000133 | 0.000331 | 0.014188 | 0.027294 |
| loss2_raw | 10.000000 | clean_train | 0.980731 | 0.000056 | 0.000045 | 0.006676 | 0.010963 |
| loss2_raw | 10.000000 | clean_test | 0.979463 | 0.000056 | 0.000041 | 0.006676 | 0.006637 |
| loss2_raw | 10.000000 | generalization_mixed4 | 0.943428 | 0.000056 | 0.000112 | 0.006676 | 0.020957 |
| loss2_raw | 25.000000 | clean_train | 0.973733 | 0.000053 | 0.000039 | 0.005165 | 0.006865 |
| loss2_raw | 25.000000 | clean_test | 0.962333 | 0.000053 | 0.000038 | 0.005165 | 0.003553 |
| loss2_raw | 25.000000 | generalization_mixed4 | 0.909302 | 0.000053 | 0.000105 | 0.005165 | 0.009692 |
| loss2_raw | 50.000000 | clean_train | 0.996713 | 0.000036 | 0.000031 | 0.009640 | 0.009058 |
| loss2_raw | 50.000000 | clean_test | 0.989390 | 0.000036 | 0.000028 | 0.009640 | 0.005940 |
| loss2_raw | 50.000000 | generalization_mixed4 | 0.962086 | 0.000036 | 0.000113 | 0.009640 | 0.027295 |
| loss3_raw | 1.000000 | clean_train | 0.812068 | 0.001768 | 0.000093 | 0.033308 | 0.021224 |
| loss3_raw | 1.000000 | clean_test | 0.768501 | 0.001768 | 0.000093 | 0.033308 | 0.020572 |
| loss3_raw | 1.000000 | generalization_mixed4 | 0.930254 | 0.001768 | 0.000331 | 0.033308 | 0.027294 |
| loss3_raw | 10.000000 | clean_train | 0.865138 | 0.001087 | 0.000387 | 0.017321 | 0.010536 |
| loss3_raw | 10.000000 | clean_test | 0.902228 | 0.001087 | 0.000425 | 0.017321 | 0.009986 |
| loss3_raw | 10.000000 | generalization_mixed4 | 0.147082 | 0.001087 | 0.001515 | 0.017321 | 0.113167 |
| loss3_raw | 25.000000 | clean_train | 0.981766 | 0.000736 | 0.000281 | 0.026312 | 0.015129 |
| loss3_raw | 25.000000 | clean_test | 0.965173 | 0.000736 | 0.000256 | 0.026312 | 0.024615 |
| loss3_raw | 25.000000 | generalization_mixed4 | -0.710233 | 0.000736 | 0.000754 | 0.026312 | 0.071088 |
| loss3_raw | 50.000000 | clean_train | 0.991243 | 0.000537 | 0.000175 | 0.068332 | 0.037245 |
| loss3_raw | 50.000000 | clean_test | 0.981900 | 0.000537 | 0.000153 | 0.068332 | 0.031386 |
| loss3_raw | 50.000000 | generalization_mixed4 | 0.881232 | 0.000537 | 0.000288 | 0.068332 | 0.031870 |
| loss3_clip01 | 1.000000 | clean_train | 0.973754 | 0.000306 | 0.000093 | 0.025550 | 0.021224 |
| loss3_clip01 | 1.000000 | clean_test | 0.955552 | 0.000306 | 0.000093 | 0.025550 | 0.020572 |
| loss3_clip01 | 1.000000 | generalization_mixed4 | 0.934359 | 0.000306 | 0.000331 | 0.025550 | 0.027294 |
| loss3_clip01 | 10.000000 | clean_train | 0.966295 | 0.000326 | 0.000136 | 0.016601 | 0.009984 |
| loss3_clip01 | 10.000000 | clean_test | 0.870822 | 0.000326 | 0.000131 | 0.016601 | 0.007551 |
| loss3_clip01 | 10.000000 | generalization_mixed4 | -0.575529 | 0.000326 | 0.000261 | 0.016601 | 0.019867 |
| loss3_clip01 | 25.000000 | clean_train | 0.697346 | 0.000289 | 0.000107 | 0.008102 | 0.004040 |
| loss3_clip01 | 25.000000 | clean_test | 0.962022 | 0.000289 | 0.000109 | 0.008102 | 0.004889 |
| loss3_clip01 | 25.000000 | generalization_mixed4 | 0.420151 | 0.000289 | 0.000148 | 0.008102 | 0.008053 |
| loss3_clip01 | 50.000000 | clean_train | 0.968355 | 0.000298 | 0.000120 | 0.010270 | 0.006961 |
| loss3_clip01 | 50.000000 | clean_test | 0.823546 | 0.000298 | 0.000083 | 0.010270 | 0.003928 |
| loss3_clip01 | 50.000000 | generalization_mixed4 | 0.811395 | 0.000298 | 0.000189 | 0.010270 | 0.012696 |
| loss3_lowpass | 1.000000 | clean_train | 0.818924 | 0.001627 | 0.000093 | 0.031680 | 0.021224 |
| loss3_lowpass | 1.000000 | clean_test | 0.779331 | 0.001627 | 0.000093 | 0.031680 | 0.020572 |
| loss3_lowpass | 1.000000 | generalization_mixed4 | 0.932541 | 0.001627 | 0.000331 | 0.031680 | 0.027294 |
| loss3_lowpass | 10.000000 | clean_train | 0.813887 | 0.000414 | 0.000147 | 0.011639 | 0.006907 |
| loss3_lowpass | 10.000000 | clean_test | 0.885913 | 0.000414 | 0.000182 | 0.011639 | 0.010364 |
| loss3_lowpass | 10.000000 | generalization_mixed4 | 0.765424 | 0.000414 | 0.000472 | 0.011639 | 0.061118 |
| loss3_lowpass | 25.000000 | clean_train | 0.977290 | 0.000351 | 0.000130 | 0.025999 | 0.013058 |
| loss3_lowpass | 25.000000 | clean_test | 0.963975 | 0.000351 | 0.000107 | 0.025999 | 0.010033 |
| loss3_lowpass | 25.000000 | generalization_mixed4 | 0.936271 | 0.000351 | 0.000370 | 0.025999 | 0.049672 |
| loss3_lowpass | 50.000000 | clean_train | 0.974088 | 0.000169 | 0.000088 | 0.020805 | 0.008847 |
| loss3_lowpass | 50.000000 | clean_test | 0.996128 | 0.000169 | 0.000098 | 0.020805 | 0.017325 |
| loss3_lowpass | 50.000000 | generalization_mixed4 | 0.929137 | 0.000169 | 0.000320 | 0.020805 | 0.033053 |
| loss3_lowpass_clip01 | 1.000000 | clean_train | 0.977032 | 0.000223 | 0.000093 | 0.025285 | 0.021224 |
| loss3_lowpass_clip01 | 1.000000 | clean_test | 0.961651 | 0.000223 | 0.000093 | 0.025285 | 0.020572 |
| loss3_lowpass_clip01 | 1.000000 | generalization_mixed4 | 0.931890 | 0.000223 | 0.000331 | 0.025285 | 0.027294 |
| loss3_lowpass_clip01 | 10.000000 | clean_train | 0.816324 | 0.000044 | 0.000030 | 0.003967 | 0.002844 |
| loss3_lowpass_clip01 | 10.000000 | clean_test | 0.926409 | 0.000044 | 0.000033 | 0.003967 | 0.005003 |
| loss3_lowpass_clip01 | 10.000000 | generalization_mixed4 | 0.945477 | 0.000044 | 0.000116 | 0.003967 | 0.009991 |
| loss3_lowpass_clip01 | 25.000000 | clean_train | 0.952734 | 0.000034 | 0.000024 | 0.003183 | 0.003141 |
| loss3_lowpass_clip01 | 25.000000 | clean_test | 0.407518 | 0.000034 | 0.000030 | 0.003183 | 0.003363 |
| loss3_lowpass_clip01 | 25.000000 | generalization_mixed4 | -0.461261 | 0.000034 | 0.000099 | 0.003183 | 0.014101 |
| loss3_lowpass_clip01 | 50.000000 | clean_train | 0.954283 | 0.000030 | 0.000019 | 0.004665 | 0.005189 |
| loss3_lowpass_clip01 | 50.000000 | clean_test | 0.875204 | 0.000030 | 0.000019 | 0.004665 | 0.003405 |
| loss3_lowpass_clip01 | 50.000000 | generalization_mixed4 | -0.331577 | 0.000030 | 0.000076 | 0.004665 | 0.005438 |

## Files

- `gradient_alignment_by_step.csv`
- `gradient_alignment_mean_by_variant.csv`
- `gradient_probe_attack_geometry.csv`
- `optimizer_microsteps.csv`
