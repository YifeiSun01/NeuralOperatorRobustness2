# Burgers Round03 Loss1/Loss2/Loss3 Continuation Code Plan

Date: 2026-06-05

Status: code prepared only. Continuation training has not been started because the active final Jacobian/SVD job is still running.

## User Target

Continue the existing round03 long-training runs instead of starting new runs:

| objective | current checkpoint | continuation target | extra local epochs |
| --- | --- | --- | ---: |
| loss1 | epoch 1000, step 3000 | epoch 3000 | 2000 |
| loss2 | epoch 500, step 1500 | epoch 1000 | 500 |
| loss3 | epoch 500, step 1500 | epoch 1000 | 500 |

## Source Script

New script:

- `tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh`

Continuation run names:

- loss1: `burgers_loss3_selective_round03_loss1_continue1000to3000_20260605`
- loss2: `burgers_loss3_selective_round03_loss2_continue500to1000_20260605`
- loss3: `burgers_loss3_selective_round03_loss3_continue500to1000_20260605`

Checkpoint inputs:

- loss1: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`
- loss2: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt`
- loss3: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt`

## Guard

The script refuses to start unless this final SVD summary exists:

- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_jacobian_svd_summary.csv`

This preserves the user request to wait for the active Jacobian/SVD work before starting more training. The guard can only be overridden intentionally with:

```bash
ALLOW_BEFORE_SVD_COMPLETE=1 tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh
```

## Settings Preserved

Observed from the new script:

- Same round03 generated generalization root: `generalization_datasets_burgers_loss3_selective_search/round_03`
- Same `adv-only` solver-label training mode.
- Same Burgers p2q2 `fast_replace_l2` attack.
- Same attack steps, batch sizes, epsilon fraction, jitter, alpha ratio, random-start fraction, and full train/test/generated50 evaluation settings as the prior long run continuation plan.
- GPU preflight is recorded under `forensics/burgers_loss3_selective_round03_loss123_continuation_gpu_preflight_20260605/`.

Important caveat: the currently available checkpoints store model weights/config, but not AdamW optimizer state. Continuation therefore resumes model weights and epoch/global-step numbering while AdamW moments restart.

## Plotting After Continuation

New plotting script:

- `tools/plot_burgers_round03_loss123_continuation_dense_comparison.py`

It merges:

- loss1 original epoch 0-1000 plus continuation epoch 1001-3000.
- loss2 original epoch 0-500 plus continuation epoch 501-1000.
- loss3 original epoch 0-500 plus continuation epoch 501-1000.

The wall-clock axis adds each original run's final wall time to its continuation-local wall time, so the curves continue instead of restarting.

Expected output directory after continuation:

- `visualizations/burgers_loss3_selective_round03_loss123_continuation_dense_20260605/`

Expected generated report after continuation:

- `docs/burgers_loss3_selective_round03_loss123_continuation_plot_report_20260605.md`

The plot script emits only train/test/generalization three-panel figures for:

- RMSE and relative L2.
- Same-epoch and cumulative wall-clock x axes.
- Every-1-epoch and every-5-epoch sampling.

## Verification

Observed static checks after code preparation:

- `bash -n tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh` passed.
- `python -m py_compile tools/plot_burgers_round03_loss123_continuation_dense_comparison.py` passed.
- `tools/plot_burgers_round03_loss123_continuation_dense_comparison.py --check-inputs` reports the expected missing continuation outputs because training has intentionally not started yet.

## Next Steps

1. Wait for the active final Jacobian/SVD stage to finish and generate `round03_long_final_jacobian_svd_summary.csv`.
2. Run `tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh`.
3. After training completes, generate merged dense plots:

```bash
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python \
  tools/plot_burgers_round03_loss123_continuation_dense_comparison.py
```

4. Update the round03 wall-clock interpretation and ledger using the new continuation evidence.
