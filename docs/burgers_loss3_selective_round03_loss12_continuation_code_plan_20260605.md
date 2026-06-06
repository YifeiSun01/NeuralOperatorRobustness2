# Burgers Round03 Loss1/Loss2 Continuation Code Plan

Date: 2026-06-05 UTC.

## Status

Code is prepared only. Continuation training has not been started, per the instruction to wait until the active Jacobian/SVD job finishes.

Active SVD status observed during code preparation: `svd_final_rep20_top100` was still running, around `sample_007/019` in `adversarial_training_runs/burgers_loss3_selective_round03_full_pipeline_20260605_logs/svd_final_rep20_top100.log`.

## Continuation Training Script

Script:

- `tools/run_burgers_loss3_selective_round03_loss12_continuation_20260605.sh`

Purpose:

- Continue loss1 from existing epoch 1000 checkpoint to epoch 3000.
- Continue loss2 from existing epoch 500 checkpoint to epoch 1000.
- Do not retrain from scratch.

Initial checkpoints:

- loss1: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`
- loss2: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt`

Continuation run names:

- loss1: `burgers_loss3_selective_round03_loss1_continue1000to3000_20260605`
- loss2: `burgers_loss3_selective_round03_loss2_continue500to1000_20260605`

Important encoded parameters:

- loss1 uses `--epochs 2000 --resume-epoch-offset 1000 --resume-global-step-offset 3000`.
- loss2 uses `--epochs 500 --resume-epoch-offset 500 --resume-global-step-offset 1500`.
- The data, attack, p2q2, batch, epsilon, jitter, solver-label, adv-only, and evaluation settings are kept the same as the previous round03 long run.
- Checkpoints are saved every 100 epochs plus continuation-local wall-clock targets `0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0` hours.

Guard:

- The script refuses to start unless `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_jacobian_svd_summary.csv` exists.
- This prevents accidentally starting continuation training while the current Jacobian/SVD job is still active.
- The guard can be overridden only by setting `ALLOW_BEFORE_SVD_COMPLETE=1`, which should not be used for the planned run.

GPU preflight:

- The script records GPU preflight under `forensics/burgers_loss3_selective_round03_loss12_continuation_gpu_preflight_20260605/`.
- It checks CUDA, PyTorch arch list including `sm_70`, and JAX GPU backend before training.

Optimizer-state caveat:

- Existing saved checkpoints contain model weights/config but not AdamW optimizer state.
- Therefore continuation loads the model weights exactly, resumes epoch/global-step numbering, but AdamW optimizer moments restart. This is a model-weight continuation, not a full optimizer-state continuation.

## Extended Dense Plot Script

Script:

- `tools/plot_burgers_round03_loss12_continuation_dense_comparison.py`

Purpose after continuation finishes:

- Merge loss1 original `0-1000` with loss1 continuation `1001-3000`.
- Merge loss2 original `0-500` with loss2 continuation `501-1000`.
- Keep loss3 original `0-500` for comparison.
- Add the original final wall time to continuation-local wall time so wall-clock curves continue instead of restarting.

Output directory after posthoc plotting:

- `visualizations/burgers_loss3_selective_round03_loss12_continuation_dense_20260605/`

Report after plotting:

- `docs/burgers_loss3_selective_round03_loss12_continuation_plot_report_20260605.md`

Plot families:

- same-epoch RMSE, every 1 epoch, train/test/generalization panels.
- same-epoch relative L2, every 1 epoch, train/test/generalization panels.
- wall-clock RMSE, every 1 epoch, train/test/generalization panels.
- wall-clock relative L2, every 1 epoch, train/test/generalization panels.
- same set for every 5 epochs.

Wall-clock construction:

- Original runs use reconstructed per-epoch wall time from `train_steps.csv` plus `evaluation_passes.csv`, with checkpoint epochs pinned to `checkpoints.csv` `wall_elapsed_seconds`.
- Continuation runs use the same local reconstruction, then add the corresponding original final wall time.
- The resume-initial evaluation row is excluded from continuation-local elapsed time, because `adversarial_training.py` starts `train_start_wall` after that initial checkpoint evaluation.

## Static Checks Already Performed

Observed checks:

- `bash -n tools/run_burgers_loss3_selective_round03_loss12_continuation_20260605.sh` passed.
- `python -m py_compile tools/plot_burgers_round03_loss12_continuation_dense_comparison.py tools/plot_burgers_round03_dense_training_comparison.py` passed.
- `tools/plot_burgers_round03_loss12_continuation_dense_comparison.py --check-inputs` reports the expected missing continuation run outputs, because training has intentionally not started yet.

## Planned Run Order After SVD Finishes

1. Wait for the active Jacobian/SVD posthoc pipeline to complete.
2. Run `tools/run_burgers_loss3_selective_round03_loss12_continuation_20260605.sh`.
3. After both continuation `summary.json` files exist, run:

```bash
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python   tools/plot_burgers_round03_loss12_continuation_dense_comparison.py
```

4. Interpret the extended wall-clock curves, especially whether loss1/loss2 generalization remain flat after their previous stopping points and whether loss3 still keeps its generated-generalization advantage after longer baseline training.
