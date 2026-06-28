# Burgers P2Q2 Loss1/Loss2 Checkpoint Policy

Date: 2026-06-02

## Current Active Run: V5

The current user-corrected checkpoint policy is:

```text
loss1: save only epoch 1000 and epoch 2000
loss2: save only epoch 900 and epoch 1000
```

No 200/400/600 periodic model checkpoints should be saved in the active V5 runs.

## Loss3 Reference Times

Reference loss3 run:

```text
adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601
```

Computed from `burgers/train_steps.csv` plus `burgers/evaluation_passes.csv`:

```text
loss3 1/5: epoch 200, 10686.52828713262 sec, 2.968480079759 h
loss3 2/5: epoch 400, 21172.02130720811 sec, 5.881117029780 h
loss3 4/5: epoch 800, 41713.82667753671 sec, 11.587174077094 h
loss3 5/5: epoch1000, 51932.967016712995 sec, 14.425824171309 h
```

## Active V5 Runs

Runner script:

```text
tools/run_burgers_p2q2_loss12_time_matched_loss3_20260602.sh
```

Active tmux session:

```text
burgers_loss12_tm_20260602
```

Loss1 active run:

```text
adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602
```

Loss1 command uses:

```text
--training-epochs 2000
--checkpoint-every-epochs 1000
```

Therefore expected checkpoint files are:

```text
burgers_epoch1000_step003000.pt
burgers_epoch2000_step006000.pt
```

Loss2 queued run, after loss1 finishes:

```text
adversarial_training_runs/burgers_p2q2_loss2_save900_1000_v5_20260602
```

Loss2 command uses:

```text
--training-epochs 1000
--checkpoint-every-epochs 900
```

Therefore expected checkpoint files are:

```text
burgers_epoch900_step002700.pt
burgers_epoch1000_step003000.pt
```

## Note On Postprocessing

The V5 runner skips checkpoint-series Jacobian/SVD plots because those scripts expect 200/400/600/800/1000 checkpoints. Basic plots, validation, and epoch-1000 attack GIF diagnostics are still enabled.

## Monitoring

```bash
tmux attach -t burgers_loss12_tm_20260602
tail -f adversarial_training_runs/burgers_p2q2_loss12_time_matched_loss3_20260602_logs/burgers_p2q2_loss1_save1000_2000_v5_20260602.log
tail -f adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602/pipeline_logs/training.log
```
