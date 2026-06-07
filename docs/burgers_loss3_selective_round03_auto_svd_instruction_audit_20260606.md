# Burgers Round03 Automatic Final-SVD Instruction Audit - 2026-06-06

Status: repository records show that the original round03 full-pipeline workflow was intended to run post-training final-checkpoint Jacobian/SVD automatically. The later continuation-training script did not carry that automatic post-continuation final-SVD stage forward.

## What Is Evidenced

Observed from `tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py`:

- The pipeline stage list includes `svd-final`.
- The file docstring describes `svd-final` as recomputing solver/model/error Jacobian SVD on final checkpoints.
- `parse_stages("all")` includes `svd-final`.
- `run_svd_final()` reads each base run's final checkpoint and invokes `tools/compare_burgers_round01_final_jacobian_svd.py`.

Observed from `tools/watch_burgers_round03_posthoc_after_loss3_20260605.sh`:

- The script comment says it waits for round03 loss3 long-training completion, then runs basic posthoc stages through final SVD.
- After detecting the loss3 summary, it runs `tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py --stages summarize,plots,gradient,svd-final --skip-existing`.

Observed from `docs/burgers_loss3_selective_round03_full_pipeline_code_plan_20260605.md`:

- The available stages include `svd-final`.
- The recorded continuation commands say that after summary exists, Jacobian/SVD diagnostics should run.
- The plan states that the SVD code recomputes `J_solver(x)`, `J_model(x)`, `J_error(x) = J_model(x) - J_solver(x)`, top singular values, spectral/Frobenius/effective-rank summaries, rankwise singular-vector similarity to solver, and top-k singular-subspace similarity to solver.
- The plan records an automatic posthoc watcher launched to run posthoc stages after loss3 completion.

## What Went Wrong For Continuation

Observed from `tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh`:

- The continuation script checks that the old final SVD summary exists before starting continuation.
- It runs loss1, loss2, and loss3 continuation training in order.
- After `continuation all done`, it exits.
- It does not run a post-continuation `svd-final` stage for loss1 epoch3000, loss2 epoch1000, or loss3 epoch1000.

## Interpretation

The prior automatic final-SVD requirement is evidenced for the original round03 full pipeline. The original automation was fulfilled for the pre-continuation final checkpoints: loss1 epoch1000, loss2 epoch500, and loss3 epoch500.

The continuation workflow should have either reused that pattern or added a new post-continuation final-SVD stage. It did not, so the continuation-final Jacobian/SVD analysis is missing locally.

Same-wall/time-matched SVD is a separate stage and remains cancelled unless explicitly restarted. This audit only concerns final-checkpoint Jacobian/SVD after continuation.
