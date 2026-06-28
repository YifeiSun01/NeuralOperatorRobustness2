# Burgers Round03 Jacobian/SVD Runtime Status

Date: 2026-06-05T17:52:39Z

## Observed Status

Observed from `ps -eo pid,ppid,etime,stat,cmd --sort=pid`:

- The posthoc watcher is still running: `tools/watch_burgers_round03_posthoc_after_loss3_20260605.sh`.
- The full pipeline is still running with stages `summarize,plots,gradient,svd-final,svd-wall`.
- The active child process is `tools/compare_burgers_round01_final_jacobian_svd.py` for `svd_final_rep20_top100`.
- The active process elapsed time was about `1:47:47` at inspection.

Observed from `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/`:

- Final SVD has partial outputs only: `round03_long_final_jacobian_svd_summary.partial.csv`, `round03_long_final_top_singular_values_long.partial.csv`, `round03_long_final_solver_similarity_rankwise.partial.csv`, `round03_long_final_solver_similarity_subspaces.partial.csv`, and `round03_long_final_runtime.partial.csv`.
- Final SVD has no completed `round03_long_final_jacobian_svd_summary.csv` yet.
- Sample directories exist through `sample_010`; `sample_010` is in progress.

Observed from `adversarial_training_runs/burgers_loss3_selective_round03_full_pipeline_20260605_logs/svd_final_rep20_top100.log`:

- The job is at `[sample] 010/019`.
- The current sample is `split=generalization dataset=burgers_loss3_selective_r03_d24 idx=43`.
- The slow section is solver Jacobian evaluation, with `round03_long_final_solver_sample10` observed at row `384/1024`.
- Recent completed samples took roughly 10 to 11 minutes per sample directory.

Observed from `tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py`:

- Final SVD uses `train=5`, `test=5`, `generalization=10`, total 20 samples, top-k 100.
- After final SVD, the pipeline queues two same-wall SVD jobs.
- Each same-wall SVD uses `train=2`, `test=2`, `generalization=6`, total 10 samples, top-k 50.

## Runtime Estimate

Inference from the observed per-sample pace:

- Remaining final SVD time: about 1.5 to 2 hours from this inspection point.
- Remaining same-wall SVD time after final finishes: about 1.5 to 2 hours total for the two queued same-wall jobs.
- Total remaining Jacobian/SVD posthoc time: about 3 to 4 hours, assuming no GPU stalls, no filesystem stalls, and no numerical failure.

## Training Status

No continuation training was started during this inspection. The prepared loss1/loss2/loss3 continuation code remains staged only.

## Update - 2026-06-05T18:41:21Z

Observed from `ps -eo pid,ppid,etime,stat,cmd --sort=pid`:

- The only active round03 Jacobian/SVD process is the final/basic SVD child process `tools/compare_burgers_round01_final_jacobian_svd.py`.
- The process is running as PPID 1 after the same-wall parent pipeline was cancelled.
- No same-wall/time-matched SVD process is active.

Observed from `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/`:

- Partial output CSVs are still being updated.
- The completed final summary `round03_long_final_jacobian_svd_summary.csv` does not exist yet.
- Sample directories exist through `sample_015`.

Observed from `adversarial_training_runs/burgers_loss3_selective_round03_full_pipeline_20260605_logs/svd_final_rep20_top100.log`:

- The job is at `[sample] 015/019`.
- The slow solver Jacobian section for sample15 reached row `1024/1024` with elapsed about `276.0s`.
- The current sample is past the slowest section; remaining model/error Jacobian rows for the sample are comparatively fast.

Inference from recent sample directory timestamps and log pace:

- Remaining final SVD time is roughly `45-60 minutes` from this inspection point.
- Expected completion is around `2026-06-05T19:25:00Z` to `2026-06-05T19:45:00Z`, assuming no GPU/filesystem stall.
- Same-wall/time-matched SVD remains cancelled and should not add more runtime.

## Completed - 2026-06-05T20:19:34Z

Observed from process table:

- No active `tools/compare_burgers_round01_final_jacobian_svd.py` process remains.

Observed from `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/`:

- Final outputs exist, including `round03_long_final_jacobian_svd_summary.csv`, `round03_long_final_jacobian_svd_summary.md`, `round03_long_final_top_singular_values_long.csv`, `round03_long_final_solver_similarity_rankwise.csv`, `round03_long_final_solver_similarity_subspaces.csv`, `round03_long_final_jacobian_spectral_norm_aggregate.csv`, `round03_long_final_error_spectral_norm_aggregate.csv`, and `round03_long_final_runtime.csv`.
- Sample directories exist through `sample_019`, so all 20 requested samples completed.
- Output file line counts observed: summary CSV 181 lines, top singular values 18001 lines, rankwise similarity 16001 lines, subspace similarity 961 lines, runtime 181 lines.

Observed from `svd_final_rep20_top100.log`:

- The final sample is `[sample] 019/019`.
- The final JSON block reports `samples: 20`, models `baseline`, `loss1_epoch1000`, `loss2_epoch0500`, `loss3_epoch0500`, and `top_k: 100`.

Conclusion: the requested basic/final Jacobian/SVD is complete. Same-wall/time-matched SVD remains cancelled and did not create output directories.
