# Burgers loss3-aligned round01 epoch and wall-clock RMSE report

Date: 2026-06-04

This report records the RMSE and relative L2 values for the latest Burgers `round01` adversarial-training runs:

- `adversarial_training_runs/burgers_loss3_aligned_round01_loss1_1000ep_20260604`
- `adversarial_training_runs/burgers_loss3_aligned_round01_loss2_500ep_20260604`
- `adversarial_training_runs/burgers_loss3_aligned_round01_loss3_500ep_20260604`

The values below are read from each run's `burgers/eval_split_summary.csv`.
The cumulative wall-clock time is reconstructed from each run's logged `step_wall_sec` plus `eval_wall_sec`.

## Same-epoch comparison

| model | epoch | cumulative hours | train RMSE | train rel L2 | test RMSE | test rel L2 | generalization RMSE | generalization rel L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss1 | 0 | 0.000185 | 0.008984 | 0.016898 | 0.009544 | 0.017755 | 0.042543 | 0.078831 |
| loss1 | 1 | 0.001695 | 0.006533 | 0.012287 | 0.007099 | 0.013206 | 0.032537 | 0.060290 |
| loss1 | 10 | 0.014076 | 0.004850 | 0.009122 | 0.005388 | 0.010024 | 0.025598 | 0.047427 |
| loss1 | 50 | 0.071408 | 0.003731 | 0.007017 | 0.004204 | 0.007822 | 0.017559 | 0.032534 |
| loss1 | 100 | 0.143531 | 0.002965 | 0.005576 | 0.003303 | 0.006145 | 0.014980 | 0.027756 |
| loss1 | 200 | 0.289101 | 0.003464 | 0.006515 | 0.003695 | 0.006874 | 0.013957 | 0.025860 |
| loss1 | 300 | 0.425568 | 0.002678 | 0.005037 | 0.002880 | 0.005358 | 0.011098 | 0.020570 |
| loss1 | 400 | 0.562445 | 0.001972 | 0.003709 | 0.002188 | 0.004070 | 0.011243 | 0.020838 |
| loss1 | 500 | 0.699963 | 0.003023 | 0.005686 | 0.003225 | 0.006000 | 0.011799 | 0.021866 |
| loss1 | 1000 | 1.391400 | 0.001969 | 0.003703 | 0.002133 | 0.003968 | 0.009127 | 0.016920 |
| loss2 | 0 | 0.000176 | 0.008984 | 0.016898 | 0.009544 | 0.017755 | 0.042543 | 0.078831 |
| loss2 | 1 | 0.006253 | 0.008534 | 0.016052 | 0.009030 | 0.016799 | 0.032240 | 0.059738 |
| loss2 | 10 | 0.058778 | 0.004895 | 0.009207 | 0.005384 | 0.010016 | 0.024478 | 0.045356 |
| loss2 | 50 | 0.299683 | 0.004601 | 0.008653 | 0.004982 | 0.009268 | 0.018975 | 0.035154 |
| loss2 | 100 | 0.613332 | 0.004060 | 0.007636 | 0.004288 | 0.007978 | 0.016102 | 0.029839 |
| loss2 | 200 | 1.246580 | 0.003566 | 0.006707 | 0.003803 | 0.007076 | 0.013698 | 0.025382 |
| loss2 | 300 | 1.843897 | 0.002618 | 0.004924 | 0.002749 | 0.005114 | 0.010994 | 0.020378 |
| loss2 | 400 | 2.442427 | 0.001940 | 0.003648 | 0.002112 | 0.003930 | 0.011635 | 0.021564 |
| loss2 | 500 | 3.081246 | 0.002529 | 0.004756 | 0.002668 | 0.004964 | 0.012053 | 0.022336 |
| loss3 | 0 | 0.000182 | 0.008984 | 0.016898 | 0.009544 | 0.017755 | 0.042543 | 0.078831 |
| loss3 | 1 | 0.013104 | 0.019666 | 0.036989 | 0.020366 | 0.037888 | 0.021868 | 0.040534 |
| loss3 | 10 | 0.125346 | 0.017140 | 0.032237 | 0.017109 | 0.031830 | 0.026456 | 0.049029 |
| loss3 | 50 | 0.641416 | 0.014389 | 0.027063 | 0.014473 | 0.026927 | 0.020469 | 0.037940 |
| loss3 | 100 | 1.284390 | 0.013374 | 0.025154 | 0.013483 | 0.025084 | 0.019168 | 0.035531 |
| loss3 | 200 | 2.545056 | 0.012341 | 0.023212 | 0.012222 | 0.022739 | 0.017401 | 0.032256 |
| loss3 | 300 | 3.823325 | 0.007943 | 0.014939 | 0.008188 | 0.015232 | 0.012494 | 0.023148 |
| loss3 | 400 | 5.125143 | 0.006631 | 0.012471 | 0.007080 | 0.013171 | 0.010497 | 0.019452 |
| loss3 | 500 | 6.428146 | 0.006416 | 0.012067 | 0.006601 | 0.012280 | 0.009939 | 0.018412 |

## Focused 100-400 epoch comparison

| model | epoch | cumulative hours | train RMSE | test RMSE | generalization RMSE |
|---|---:|---:|---:|---:|---:|
| loss1 | 100 | 0.143531 | 0.002965 | 0.003303 | 0.014980 |
| loss1 | 200 | 0.289101 | 0.003464 | 0.003695 | 0.013957 |
| loss1 | 300 | 0.425568 | 0.002678 | 0.002880 | 0.011098 |
| loss1 | 400 | 0.562445 | 0.001972 | 0.002188 | 0.011243 |
| loss2 | 100 | 0.613332 | 0.004060 | 0.004288 | 0.016102 |
| loss2 | 200 | 1.246580 | 0.003566 | 0.003803 | 0.013698 |
| loss2 | 300 | 1.843897 | 0.002618 | 0.002749 | 0.010994 |
| loss2 | 400 | 2.442427 | 0.001940 | 0.002112 | 0.011635 |
| loss3 | 100 | 1.284390 | 0.013374 | 0.013483 | 0.019168 |
| loss3 | 200 | 2.545056 | 0.012341 | 0.012222 | 0.017401 |
| loss3 | 300 | 3.823325 | 0.007943 | 0.008188 | 0.012494 |
| loss3 | 400 | 5.125143 | 0.006631 | 0.007080 | 0.010497 |

## Same wall-clock comparison

For each target wall-clock time, the selected epoch is the latest logged evaluation epoch at or before that time.
For loss1 at 2 hours, the run had already ended at epoch 1000, so the final epoch is reported.

| target hours | model | selected epoch | actual hours | status | train RMSE | train rel L2 | test RMSE | test rel L2 | generalization RMSE | generalization rel L2 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0 | loss1 | 722 | 0.999150 | at_or_before_target | 0.003503 | 0.006589 | 0.003601 | 0.006699 | 0.011532 | 0.021371 |
| 1.0 | loss2 | 162 | 0.995520 | at_or_before_target | 0.003109 | 0.005848 | 0.003380 | 0.006288 | 0.013947 | 0.025842 |
| 1.0 | loss3 | 77 | 0.997906 | at_or_before_target | 0.015293 | 0.028765 | 0.015252 | 0.028374 | 0.020218 | 0.037471 |
| 2.0 | loss1 | 1000 | 1.391400 | run_ended_before_target | 0.001969 | 0.003703 | 0.002133 | 0.003968 | 0.009127 | 0.016920 |
| 2.0 | loss2 | 326 | 1.996943 | at_or_before_target | 0.002111 | 0.003970 | 0.002317 | 0.004310 | 0.011160 | 0.020685 |
| 2.0 | loss3 | 156 | 1.988972 | at_or_before_target | 0.011927 | 0.022434 | 0.011784 | 0.021923 | 0.018042 | 0.033443 |

## Final logged wall time

| model | final epoch | cumulative hours | train RMSE | test RMSE | generalization RMSE |
|---|---:|---:|---:|---:|---:|
| loss1 | 1000 | 1.391400 | 0.001969 | 0.002133 | 0.009127 |
| loss2 | 500 | 3.081246 | 0.002529 | 0.002668 | 0.012053 |
| loss3 | 500 | 6.428146 | 0.006416 | 0.006601 | 0.009939 |

## Interpretation

- Same epoch: loss3 eventually approaches the generalization RMSE of loss1/loss2 by epochs 300-500, but it requires much more wall-clock time.
- Same wall-clock: loss1 and loss2 are much stronger than loss3 at both 1 hour and 2 hours.
- Loss3 is directionally useful on the loss3-aligned aggressive generalization set, but its wall-clock efficiency is poor.
- Loss1 remains the best clean train/test method and is still highly competitive on aggressive generalization under equal-time comparison.

