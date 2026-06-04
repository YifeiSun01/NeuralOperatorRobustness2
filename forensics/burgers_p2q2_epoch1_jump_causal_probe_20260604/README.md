# Burgers p2q2 epoch-1 jump causal probe

This experiment replays one adversarial-training epoch from the same baseline model to test why loss1/loss2 improve immediately while loss3 jumps upward at epoch 1.

Four variants start from the same baseline: `loss1`, `loss2`, raw `loss3`, and `loss3_clip01` where the raw loss3 attacked input is clipped back to `[0,1]` and the solver target is recomputed before optimizer updates.

## Before/after epoch 1 evaluation

| variant | split | rmse_before | rel_before | rmse_after | rel_after | rmse_delta | rel_delta | rmse_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | generalization | 0.017073 | 0.029876 | 0.009742 | 0.017167 | -0.007331 | -0.012709 | 0.570625 |
| loss2 | generalization | 0.017073 | 0.029876 | 0.014452 | 0.025100 | -0.002621 | -0.004776 | 0.846477 |
| loss3_clip01 | generalization | 0.017073 | 0.029876 | 0.010765 | 0.018916 | -0.006308 | -0.010960 | 0.630511 |
| loss3_raw | generalization | 0.017073 | 0.029876 | 0.022311 | 0.038836 | 0.005238 | 0.008960 | 1.306785 |
| loss1 | test | 0.009232 | 0.017514 | 0.006093 | 0.011651 | -0.003139 | -0.005863 | 0.660038 |
| loss2 | test | 0.009232 | 0.017514 | 0.006520 | 0.012517 | -0.002712 | -0.004997 | 0.706241 |
| loss3_clip01 | test | 0.009232 | 0.017514 | 0.007603 | 0.014622 | -0.001629 | -0.002892 | 0.823578 |
| loss3_raw | test | 0.009232 | 0.017514 | 0.016414 | 0.030846 | 0.007182 | 0.013332 | 1.777903 |
| loss1 | train | 0.008741 | 0.016745 | 0.005631 | 0.010806 | -0.003109 | -0.005939 | 0.644283 |
| loss2 | train | 0.008741 | 0.016745 | 0.006087 | 0.011730 | -0.002653 | -0.005015 | 0.696420 |
| loss3_clip01 | train | 0.008741 | 0.016745 | 0.007935 | 0.015256 | -0.000805 | -0.001489 | 0.907866 |
| loss3_raw | train | 0.008741 | 0.016745 | 0.015289 | 0.028784 | 0.006549 | 0.012039 | 1.749236 |

## Epoch 1 attack geometry

| variant | delta_l2_rms | delta_linf | linf_over_l2rms | x_adv_min | x_adv_max | oob_mean | oob_max | y_adv_min | y_adv_max | attack_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | 0.059912 | 0.105390 | 1.757806 | -0.164897 | 1.159173 | 0.002389 | 0.160763 | -0.045551 | 1.040876 | 3.843688 |
| loss2 | 0.060145 | 0.105708 | 1.755212 | -0.169783 | 1.146660 | 0.002264 | 0.154169 | -0.027364 | 1.031709 | 18.030 |
| loss3_clip01 | 0.045998 | 0.209142 | 4.567167 | 0.000e+00 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000693 | 0.997556 | 42.952 |
| loss3_raw | 0.060145 | 0.259881 | 4.309852 | -0.347389 | 1.379393 | 0.004287 | 0.353879 | -0.141717 | 1.129097 | 42.399 |

## Interpretation

- If raw loss3 increases clean/test/generalization RMSE after only one replayed epoch while loss1/loss2 decrease it, the epoch-1 jump is caused by the optimizer update on raw loss3 adversarial samples, not by evaluation bookkeeping.
- If `loss3_clip01` removes or strongly reduces that jump, the cause is the off-manifold/out-of-range component of loss3 `x_adv`.
- The attack geometry table checks whether the harmful variant also has much larger `delta Linf/RMS-L2` and larger out-of-bound violation under the same RMS-L2 budget.

## Files
- `epoch1_replay_eval_split_summary.csv`
- `epoch1_before_after_delta.csv`
- `epoch1_replay_attack_batch_metrics.csv`
- `epoch1_attack_geometry_summary.csv`
- `epoch1_rmse_delta_by_variant.png`
- `epoch1_attack_geometry_by_variant.png`
