# Burgers Round03 Same-Wall SVD Cancelled

Date: 2026-06-05T18:26:20Z

## User Direction

The user requested that the same-wall/time-matched Jacobian/SVD jobs should not be run. Only the basic final checkpoint SVD should remain:

- loss1 final checkpoint: epoch 1000.
- loss2 final checkpoint: epoch 500.
- loss3 final checkpoint: epoch 500.
- Evaluation samples: train/test/generalization from the round03 generated-generalization setting.

## Actions Taken

Observed active process before cancellation:

- Parent pipeline: `tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py --stages summarize,plots,gradient,svd-final,svd-wall --skip-existing`.
- Active child SVD: `tools/compare_burgers_round01_final_jacobian_svd.py` with output root `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605`.

Action taken:

- Sent `SIGTERM` only to the parent pipeline process so it cannot launch queued `svd-wall` work after the current final SVD exits.
- Did not kill the active final SVD child process.

Observed after cancellation:

- The watcher/pipeline exited with rc `143` at `2026-06-05T18:25:28Z`.
- The final SVD process is still running as an orphaned child of PID 1.
- No same-wall SVD log files were created.
- No same-wall SVD output directories matching `forensics/burgers_loss3_selective_round03_long_samewall*` were present.

## Source Updates

Updated source so future default/basic posthoc runs do not include same-wall SVD:

- `tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py`: `all` now expands only through `svd-final`; `svd-wall` remains available only if explicitly requested.
- `tools/watch_burgers_round03_posthoc_after_loss3_20260605.sh`: stage list changed from `summarize,plots,gradient,svd-final,svd-wall` to `summarize,plots,gradient,svd-final`.

Static checks:

- `python -m py_compile tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py` passed.
- `bash -n tools/watch_burgers_round03_posthoc_after_loss3_20260605.sh` passed.

## Current Status

The only active Jacobian/SVD process is the final/basic SVD for loss1 epoch1000, loss2 epoch500, and loss3 epoch500. Same-wall/time-matched SVD is cancelled.
