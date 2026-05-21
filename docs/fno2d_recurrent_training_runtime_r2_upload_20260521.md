# 2D NS Recurrent FNO2d Training Runtime And R2 Upload Workflow - 2026-05-21

Status: code workflow updated; no new training run was executed by this documentation change.

## Observed Evidence

- Source files changed: `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` and `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`.
- The working tree was first reset to GitHub `vast-ai` commit `d107e33c2740fef80609ac1385f20ddbc52fafa8` before editing.
- Syntax checks passed with `python3 -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` and `bash -n 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`.

## Implemented Behavior

- Each epoch now records wall time, total elapsed time, mean epoch time, recent-window mean epoch time, ETA, and estimated total runtime in `train_log.csv`, `progress.jsonl`, `progress_latest.json`, console output, and the legacy text log.
- Train/test regression metrics are tracked as relative L2 sum, relative L2 mean, MSE, and `score_1_minus_relative_l2`. This score is an accuracy-like regression score, not classification accuracy.
- With `--r2-upload`, the trainer validates `rclone`, R2 credentials, and bucket access before starting GPU training.
- During training, small log/metric files are synced to R2 every `--r2-sync-every-epochs` epochs. Checkpoints are not copied during this periodic sync.
- After successful training, the whole output directory is uploaded to R2, including model `.pth`, checkpoint `.pt`, config, GPU verification, dataset info, logs, and final metrics.
- R2 credentials are read only from environment variables: `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` or the AWS-compatible equivalents. Secrets are not written to repository files.

## Default R2 Destination

The wrapper defaults to:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/saved_models/2D/<run-name>/`

## Direct Launch Command

Set credentials in the environment, then run:

```bash
R2_ACCESS_KEY_ID='<access-key-id>' R2_SECRET_ACCESS_KEY='<secret-access-key>' EPOCHS=500 BATCH_SIZE=4 EVAL_BATCH_SIZE=4 EVAL_EVERY=1 PROGRESS_EVERY=10 R2_UPLOAD=1 ./2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh
```

## Remaining Work

- Run the full GPU training job on the intended V100 machine.
- After epoch 1 completes, use the printed `avg_epoch`, `recent_avg_epoch`, `eta`, and `est_total` fields to estimate the full run duration.
- Verify the final R2 destination contains the completed model directory after training finishes.

## Disable Periodic Evaluation And Checkpoint Defaults - 2026-05-21 09:33 UTC

Observed evidence:
- Active PID 58013 still has original command-line arguments `--eval-every 10` and `--save-every 25`; running processes do not pick up changed defaults.
- Epoch 20 took `99.673s` with scheduled test evaluation, while nearby non-evaluation epochs were about `84s`.
- Current checkpoint directory contains `best.pt` at about `2.7G`.
- Future defaults were changed to `EVAL_EVERY=0` and `SAVE_EVERY=0` in the recurrent FNO2d wrapper and CLI.
- `SAVE_EVERY=0` now disables in-loop `latest.pt` checkpoint saves.
- `EVAL_EVERY=0` now disables periodic and final test evaluation; final results use the last epoch train metrics and NaN test metrics.
- Syntax validation passed for both the Python trainer and shell wrapper.

Inference:
- Future 500-epoch runs will only save final artifacts and upload at the end when `R2_UPLOAD=1`; no test-eval-based `best.pt` will be produced when evaluation is disabled.
- Applying this to the already-running process requires stopping and restarting because the old command-line arguments are fixed for that process.

## Epoch-25 Checkpoint Save Overhead Measurement - 2026-05-21 09:41 UTC

Observed evidence:
- `checkpoints/latest.pt` appeared at epoch 25 with size about `2.7G`; `checkpoints/best.pt` also exists at about `2.7G`.
- Checkpoint directory total size became about `5.3G`.
- Epoch 25 logged `seconds=84.0115s`, `elapsed_total_seconds=2147.7380s`; this log row is written before checkpoint saving.
- Epoch 26 logged `seconds=84.0250s`, `elapsed_total_seconds=2234.5126s`.
- Inferred checkpoint/save gap: about `2.75s`.

Inference:
- The observed latest-checkpoint write took only a few seconds on this machine, but it is still avoidable repeated multi-GB disk I/O.
- Future runs should use `SAVE_EVERY=0`; that default has already been changed.

## Active 500-Epoch Runtime Projection After 28 Epochs - 2026-05-21 09:43 UTC

Observed evidence:
- Completed epoch 28 with `elapsed_total_seconds=2402.501s` (`40m03s`).
- Recent stable non-evaluation epoch mean: about `83.982s`.
- Evaluation overhead: about `15.468s` over a normal epoch.
- Epoch-25 checkpoint overhead: about `2.75s`.

Inference:
- Remaining training time after epoch 28 is about `11h14m-11h16m`.
- Projected total training time is about `11h54m-11h56m`.
- Projected training completion from `2026-05-21 09:43:11 UTC` is about `2026-05-21 20:57-20:59 UTC`, before final R2 upload variability.

## Active Run Health Check At Epoch 86 - 2026-05-21 11:05 UTC

Observed evidence:
- Active process PID `58013` had `ps` elapsed time `02:03:43`, state `Rl+`.
- `progress_latest.json` showed epoch `86/500`, batch `40/72`, completed train batches `6160/36000`, run elapsed `7329.845s`, and train-batch ETA `9h51m47s`.
- Latest completed epoch in `train_log.csv` was epoch 85 with elapsed total `2h01m23s`; recent non-evaluation epochs remained around `83.9-84.0s`.
- Epoch 80 scheduled evaluation completed with test relative L2 mean `0.146012` and test MSE `0.0452241`.
- `nvidia-smi` reported `81121 MiB / 81920 MiB` used and `100%` GPU utilization.

Inference:
- No evidence of failure, OOM, stall, or runaway memory growth was observed.
- Approximate finish remains around `2026-05-21 20:57 UTC`, before final R2 upload variability.

## Active Run Progress Check At Epoch 449 - 2026-05-21 19:43 UTC

Observed evidence:
- Active process PID `58013` had `ps` elapsed time `10:41:51`.
- `progress_latest.json` showed epoch `449/500`, batch `30/72`, completed train batches `32286/36000`, run elapsed `38412.897s`, and train-batch ETA `1h13m39s`.
- Current total train-batch completion was `89.6833%`.
- Latest completed epoch in `train_log.csv` was epoch 448 with elapsed total `10h39m38s`; recent non-evaluation epochs remained around `83.94-83.96s`.
- `nvidia-smi` reported `81121 MiB / 81920 MiB` used and `100%` GPU utilization.

Inference:
- No evidence of failure, OOM, stall, or runaway memory growth was observed.
- The run remains on track to finish training around `2026-05-21 20:57 UTC`, before final checkpoint/R2 upload variability.

## Active Run Latest Train/Test Loss Check - 2026-05-21 20:04 UTC

Observed evidence:
- `progress_latest.json` showed epoch `463/500`, batch `50/72`, completed train batches `33314/36000`, run elapsed `39650.645s`, and train-batch ETA `53m17s`.
- Latest completed epoch in `train_log.csv` was epoch 462.
- Epoch 462 train relative L2 mean was `0.05333483872206315`; train MSE was `0.0057479435699465484`.
- Latest completed test evaluation was epoch 460.
- Epoch 460 test relative L2 mean was `0.08675776571035385`; test MSE was `0.0179227876663208`.
- Current in-progress batch loss was `0.9232521057128906` at epoch 463 batch 50.

Inference:
- Test metrics update only every 10 epochs in this active old-argument run; NaN test fields on non-evaluation epochs are expected.

## Active Run Loss Trend Check - 2026-05-21 20:06 UTC

Observed evidence:
- Latest completed epoch was epoch 464; active progress was epoch `465/500`, batch `20/72`.
- Train relative L2 mean decreased from `0.610978` at epoch 1 to `0.053272` at epoch 464, about `91.28%` lower.
- Train MSE decreased from `0.791622` at epoch 1 to `0.005734` at epoch 464.
- Test relative L2 mean decreased from `0.430839` at epoch 1 to `0.086758` at epoch 460, about `79.86%` lower.
- Test MSE decreased from `0.365911` at epoch 1 to `0.017923` at epoch 460.
- Recent test relative L2 means from epoch 410 to 460 were `0.087922`, `0.087647`, `0.087481`, `0.087220`, `0.086976`, `0.086758`.

Inference:
- The train and test curves are clearly decreasing.
- Late-stage improvement is slower and close to plateau, but the latest evaluated test loss remains the best observed test relative L2 so far.
