# NS2D eps160 alpha50 Three-Target Completion Status - 2026-05-24

Status: completed.

## Observed Evidence

- Process table: no `attack_ns2d_recurrent_core4.py`, `run_ns2d_eps160_alpha50`, or `eps160_alpha50` background process was found.
- GPU status: NVIDIA A100-SXM4-80GB, `1 MiB / 81920 MiB`, `0%` utilization.
- Output root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_023355_UTC/eps160_alpha50/`.
- All three evidenced targets have `summary.json`, `final_state_outputs.npz`, `final_state_metrics.csv`, `per_step_metrics.csv`, `per_sample_step_metrics.csv`, and `step_sample_trace.npz`.
- Each `per_step_metrics.csv` has `101` rows with last `k = 100`.
- Each `final_state_metrics.csv` has `10` rows.

## Target Summary

| target | method | runtime seconds | clean true_loss mean | adv true_loss mean | per-step rows | final-state rows | source |
|---|---|---:|---:|---:|---:|---:|---|
| loss1/all_w | steepest_add | 937.710 | 68.4911 | 336.555 | 101 | 10 | `eps160_alpha50/mode_wwwwwwwwww_p2_q2_20260524_092837_UTC/batch_0000_0009/loss1/steepest_add/summary.json` |
| loss2/all_a_target_w | steepest_add | 399.655 | 68.4911 | 285.897 | 101 | 10 | `eps160_alpha50/mode_aaaaaaaaaw_p2_q2_20260524_094426_UTC/batch_0000_0009/loss2/steepest_add/summary.json` |
| loss3/all_a_target_w | steepest_add | 1550.860 | 68.4911 | 369.371 | 101 | 10 | `eps160_alpha50/mode_aaaaaaaaaw_p2_q2_20260524_073831_UTC/batch_0000_0009/loss3/steepest_add/summary.json` |

## Inference

The eps160/alpha50 `steepest_add` runs that are locally evidenced are complete for `loss1/all_w`, `loss2/all_a_target_w`, and `loss3/all_a_target_w`. `loss3/all_a_target_w` has the highest final all-W true-loss mean among these three completed targets.

## Remaining Work

- These newly completed loss1/loss2 outputs still need to be synced into `/workspace/NeuralOperatorRobustness2_gitclean` and the eps160 figures should be regenerated if the user wants the `eps160_alpha50` panel to include all three rows.
