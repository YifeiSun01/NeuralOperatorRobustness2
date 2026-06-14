# Burgers Strict Latest Full-52 Attack Table, 20260614

Generated: 2026-06-14T14:33:09+00:00

This report combines the current latest old4 widevis full-52 20-step attack with the existing current random_clean_y/random_solver_y full-52 20-step attack. No training or new attack is run by this builder.

## Main Result

Loss3 beats random_solver_y on 52/52 dataset rows for attack loss increase.

| model | datasets | sample_count_sum | initial_loss_mean | final_loss_mean | attack_loss_increase_mean | attack_loss_increase_median | final_delta_rms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 52 | 1.020e+04 | 0.0011785 | 0.0118478 | 0.0106693 | 0.00841407 | 0.12 |
| loss1 | 52 | 1.020e+04 | 0.000562629 | 0.00715168 | 0.00658905 | 0.00566032 | 0.12 |
| loss2 | 52 | 1.020e+04 | 0.000623494 | 0.00722935 | 0.00660586 | 0.00571937 | 0.12 |
| loss3 | 52 | 1.020e+04 | 0.000181205 | 0.00400196 | 0.00382075 | 0.00298269 | 0.119982 |
| random_clean_y | 52 | 1.020e+04 | 0.00985418 | 0.0470111 | 0.0371569 | 0.035876 | 0.117134 |
| random_solver_y | 52 | 1.020e+04 | 0.000488038 | 0.00865523 | 0.00816719 | 0.00656109 | 0.119998 |

## Best Model Counts

| best_model | dataset_count |
| --- | --- |
| loss3 | 52 |

## Largest loss3-vs-random_solver_y Margins

| dataset_order | split | display_label | random_dataset_id | loss3_attack_loss_increase_mean | random_solver_y_attack_loss_increase_mean | loss3_minus_random_solver_y_attack_increase | best_model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 21 | generalization | Matern GRF c=0.04, nu=2.5; range [0,1.5] | burgers_widevis_l3target_d18 | 0.00858821 | 0.0189325 | -0.0103443 | loss3 |
| 22 | generalization | Matern GRF c=0.08, nu=2.5; range [-0.5,1.5] | burgers_widevis_l3target_d19 | 0.0130877 | 0.0227505 | -0.00966283 | loss3 |
| 23 | generalization | Matern GRF c=0.055, nu=4; range [-0.5,1.5] | burgers_widevis_l3target_d20 | 0.0112372 | 0.0199618 | -0.00872463 | loss3 |
| 20 | generalization | Matern GRF c=0.04, nu=2.5; range [-0.65,1.35] | burgers_widevis_l3target_d17 | 0.00817109 | 0.0156803 | -0.00750923 | loss3 |
| 11 | generalization | Gaussian GRF corr=0.012; range [-0.5,1.5] | burgers_widevis_l3target_d08 | 0.00781494 | 0.0151217 | -0.00730674 | loss3 |
| 37 | generalization | Power-law Fourier alpha=1.8, k0=12; range [-0.5,1.5] | burgers_widevis_l3target_d34 | 0.00642571 | 0.0128669 | -0.0064412 | loss3 |
| 28 | generalization | Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5] | burgers_widevis_l3target_d25 | 0.00588332 | 0.0119129 | -0.00602955 | loss3 |
| 26 | generalization | Power-law Fourier alpha=1.5, k0=10; range [-0.5,1.5] | burgers_widevis_l3target_d23 | 0.00571172 | 0.011636 | -0.00592427 | loss3 |
| 19 | generalization | Matern GRF c=0.08, nu=2.5; range [-0.3,1.3] | burgers_widevis_l3target_d16 | 0.00469417 | 0.010416 | -0.00572182 | loss3 |
| 12 | generalization | Gaussian GRF corr=0.009; range [-0.5,1.5] | burgers_widevis_l3target_d09 | 0.00620012 | 0.0118784 | -0.00567827 | loss3 |
| 30 | generalization | Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5] | burgers_widevis_l3target_d27 | 0.00434952 | 0.0099795 | -0.00562998 | loss3 |
| 38 | generalization | Power-law Fourier alpha=3, k0=24; range [-0.5,1.5] | burgers_widevis_l3target_d35 | 0.00554663 | 0.0111656 | -0.00561892 | loss3 |

## Outputs

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/attack_52dataset_six_models_strict_latest_widevis_long.csv`: 312 rows
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/attack_52dataset_six_models_strict_latest_widevis_wide.csv`: 52 rows
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/attack_52dataset_six_models_strict_latest_widevis_model_summary.csv`: 6 rows
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/strict_latest_attack52_20260614/strict_latest_attack52_winners_by_dataset.csv`: 52 rows
