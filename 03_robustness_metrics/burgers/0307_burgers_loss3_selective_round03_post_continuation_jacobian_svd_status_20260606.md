# Burgers Loss3-Selective Round03 Post-Continuation Jacobian/SVD Status - 2026-06-06

Status: the pre-continuation final Jacobian/SVD analysis exists; the post-continuation final checkpoint Jacobian/SVD analysis is not currently evidenced locally.

## Observed Evidence

The completed round03 long final Jacobian/SVD directory is:

- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605`

Its summary explicitly states that it was computed on:

- `loss1_epoch1000`
- `loss2_epoch0500`
- `loss3_epoch0500`

The same summary defines and reports:

```text
J_solver(x) = d solver(x) / d x
J_model(x)  = d model(x) / d x
J_error(x)  = J_model(x) - J_solver(x)
```

Available output files in that directory include:

- `round03_long_final_jacobian_svd_summary.csv`
- `round03_long_final_top_singular_values_long.csv`
- `round03_long_final_solver_similarity_rankwise.csv`
- `round03_long_final_solver_similarity_subspaces.csv`
- `round03_long_final_error_spectral_norm_aggregate.csv`
- `round03_long_final_jacobian_spectral_norm_aggregate.csv`
- `sample_*/` NPZ SVD files for solver, baseline, model, and model-minus-solver error.

The continuation-final checkpoints do exist locally:

- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605/burgers/checkpoints/burgers_epoch3000_step009000.pt`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue500to1000_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue500to1000_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`

No local `forensics/` file or directory was found matching a round03 continuation/post-continuation Jacobian/SVD analysis for `loss1_epoch3000`, `loss2_epoch1000`, or `loss3_epoch1000`.

## Interpretation

The existing Jacobian/SVD result covers the earlier long checkpoints, not the newly continued final checkpoints. It should not be described as the post-continuation Jacobian/SVD result.

The already-computed pre-continuation result does include model Jacobians, model-minus-solver error Jacobians, top singular values, singular-vector similarity to the solver, singular subspace similarity, and error spectral norm shrinkage versus the baseline model.

## Remaining Work

To answer the same forensic questions for the continuation-final models, rerun the round03 final Jacobian/SVD workflow against:

- loss1 epoch3000
- loss2 epoch1000
- loss3 epoch1000

Same-wall/time-matched SVD remains cancelled unless explicitly re-enabled; this status only concerns final-checkpoint SVD.


## 2026-06-06 Update

The missing post-continuation final SVD has now been added to the continuation script and launched in tmux session `round03_post_continuation_final_svd`.

Current output directory:

- `forensics/burgers_loss3_selective_round03_loss123_continuation_final_jacobian_svd_rep20_top100_20260606`

Current log:

- `adversarial_training_runs/burgers_loss3_selective_round03_loss123_continuation_20260605_logs/svd_final_post_continuation_rep20_top100.log`

The run was observed entering `[device] cuda` and starting `sample_000`. Final SVD conclusions should still wait for the completed summary CSV/Markdown.


## 2026-06-06 Stop Update

The briefly launched post-continuation final SVD was stopped at the user's request. It was a SVD-only job on existing checkpoints, not retraining.

The continuation script now defaults to not running post-continuation final SVD. To opt in manually later, set `RUN_POST_CONTINUATION_FINAL_SVD=1`.

The final post-continuation SVD remains incomplete; no final summary CSV/Markdown exists.
