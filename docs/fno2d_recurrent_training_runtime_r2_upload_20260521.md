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
