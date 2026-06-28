# Burgers Round03 Post-Continuation Final-SVD Automation - 2026-06-06

Status: implemented and launched. The continuation driver now automatically runs final-checkpoint Jacobian/SVD after loss1/loss2/loss3 continuation training completes.

## Code Change

Updated script:

- `tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh`

New behavior:

- The script still skips completed continuation runs instead of retraining them.
- After loss1, loss2, and loss3 continuation are complete, it calls a new `run_post_continuation_final_svd` stage by default.
- The stage can be disabled only for explicit debugging with `RUN_POST_CONTINUATION_FINAL_SVD=0`.
- If the post-continuation final SVD summary already exists, the stage skips it.
- If the SVD output directory exists but the summary is incomplete, the SVD tool is allowed to continue with `--reuse-existing` behavior.
- The stage performs a fresh GPU sanity check and refuses to continue if CUDA, `sm_70`, or JAX GPU are unavailable.

## SVD Inputs

Continuation-final checkpoints:

- loss1 epoch3000: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605/burgers/checkpoints/burgers_epoch3000_step009000.pt`
- loss2 epoch1000: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue500to1000_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`
- loss3 epoch1000: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue500to1000_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`

The post-continuation SVD reuses the fixed sample manifest from the pre-continuation long final SVD:

- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_sample_manifest.csv`

This keeps the train/test/generalization sample points matched between pre-continuation and post-continuation final SVD.

## Output

Post-continuation final SVD output directory:

- `forensics/burgers_loss3_selective_round03_loss123_continuation_final_jacobian_svd_rep20_top100_20260606`

Output prefix:

- `round03_loss123_continuation_final`

Log:

- `adversarial_training_runs/burgers_loss3_selective_round03_loss123_continuation_20260605_logs/svd_final_post_continuation_rep20_top100.log`

The SVD command computes solver/model/error Jacobians, top singular values, spectral/Frobenius/effective-rank summaries, rankwise singular-vector similarity to solver, top-k singular-subspace similarity to solver, and model-minus-solver error spectral norm shrinkage.

## Launch Evidence

Launched in tmux session:

- `round03_post_continuation_final_svd`

Observed start evidence:

- Launcher skipped all three completed continuation runs.
- Driver log recorded `start post-continuation final Jacobian/SVD` at `2026-06-06T14:20:31Z`.
- SVD log recorded GPU preflight with PyTorch `2.8.0+cu126`, CUDA `12.6`, Tesla V100-SXM2-32GB, compute capability `[7, 0]`, arch list including `sm_70`, JAX `0.10.0`, JAX backend `gpu`, and JAX device `cuda:0`.
- SVD log then entered `[device] cuda` and began `sample_000`.

## Current Interpretation

The automation bug has been fixed and the missing post-continuation final SVD is now running. The final Jacobian/SVD results are not complete until the output directory contains:

- `round03_loss123_continuation_final_jacobian_svd_summary.csv`
- `round03_loss123_continuation_final_jacobian_svd_summary.md`

Same-wall/time-matched SVD remains cancelled. This run is final-checkpoint SVD only.


## 2026-06-06 Stop And Default-Off Update

The post-continuation final SVD was stopped at the user's request.

Clarification:

- The launched job was not retraining.
- The continuation driver skipped all three completed continuation training runs.
- The launched job was only `tools/compare_burgers_round01_final_jacobian_svd.py` on the existing continuation-final checkpoints.

Observed stop evidence:

- `pgrep -af 'compare_burgers_round01_final_jacobian_svd|round03_loss123_continuation_final'` returned no process after stopping.
- `tmux list-sessions` no longer showed `round03_post_continuation_final_svd`.
- `nvidia-smi` showed `0MiB / 32768MiB` used and no running processes.

The continuation script has been changed so post-continuation final SVD is now opt-in, not automatic:

```bash
RUN_POST_CONTINUATION_FINAL_SVD=1 bash tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh
```

Default behavior was verified by running the script without that variable. It skipped all completed continuation runs and printed:

```text
skip post-continuation final SVD because RUN_POST_CONTINUATION_FINAL_SVD=0
```

Partial output status:

- Output directory exists: `forensics/burgers_loss3_selective_round03_loss123_continuation_final_jacobian_svd_rep20_top100_20260606`
- Completed final summary does not exist.
- Partial files exist for `sample_000` solver and baseline only.
- These partial files were not deleted.
