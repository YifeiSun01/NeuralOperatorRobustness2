# Burgers Round03 Long Training 10-Minute Monitor

Date: 2026-06-05 UTC.

## Scope

Observed the already-running round03 long-training driver for 10 minutes after the user requested a short health check. I did not start a duplicate training job because `burgers_loss3_selective_round03_loss1_1000ep_long_20260605` was already active.

## Source Evidence

- Monitor log: `forensics/burgers_loss3_selective_round03_long_training_monitor_20260605/loss1_10min_monitor_20260605_054458_UTC.log`
- Active run: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605`
- Eval source: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/eval_split_summary.csv`
- Checkpoint source: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints.csv`

## Observed Result

- Monitor window: 20 samples, roughly one sample every 30 seconds.
- Start of monitor: sample 1 at 2026-06-05T05:44:58Z, latest eval epoch 786.
- End of monitor: sample 20 at 2026-06-05T05:54:29Z, latest eval epoch 901.
- Process stayed alive throughout the window: `tools/adversarial_training.py` remained active under `tools/run_burgers_loss3_selective_round03_long_training_20260605.sh`.
- GPU path stayed active on Tesla V100-SXM2-32GB. Observed memory was stable at about 4896 MiB / 32768 MiB; utilization varied across train/eval/attack phases, reaching 99% in the final sample.
- Eval rows continued to be appended; epoch advanced from 786 to 901.
- Invalid-value fractions in train/test/generalization rows stayed at 0.0 across the observed samples.
- Checkpoint writing worked. Observed checkpoints included periodic epoch 800 and epoch 900 checkpoints, plus an earlier wall-clock checkpoint at epoch 715 for the 3600-second target.

## Latest Observed Metrics

Observed from sample 20 / epoch 901:

| split | RMSE mean | relative L2 mean | invalid fraction mean |
|---|---:|---:|---:|
| train | 0.00165605 | 0.0031148 | 0.0 |
| test | 0.0017759 | 0.0033039 | 0.0 |
| generalization generated50 | 0.0371724 | 0.0666681 | 0.0 |

Observed checkpoint from sample 20:

```text
adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints/burgers_epoch900_step002700.pt
```

## Conclusion

Observed evidence from the 10-minute monitor shows no anomaly: the process stayed alive, GPU execution was active, metrics continued writing, invalid fractions remained zero, memory was stable, and checkpoint saving worked.

This is only a health check for the running `loss1` stage. It is not a final comparison between `loss1`, `loss2`, and `loss3`; those conclusions require the full long-training and posthoc stages to finish.
