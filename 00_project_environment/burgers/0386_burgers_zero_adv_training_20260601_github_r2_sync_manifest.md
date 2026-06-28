# Burgers Zero Adversarial Training GitHub/R2 Sync Manifest - 2026-06-01

This manifest records what should be synced for the latest Burgers visualization
and Jacobian/SVD work. It intentionally omits all credential values.

## GitHub Scope

Commit and push these lightweight/code artifacts:

```text
docs/burgers_zero_adv_training_visualization_and_jacobian_top100_20260601.md
docs/burgers_zero_adv_training_20260601_github_r2_sync_manifest.md
tools/compare_burgers_adversarial_jacobian_svd.py
```

Optionally include lightweight forensics summaries:

```text
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/README_FIXED10_TOP100_UPDATE.md
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/README_REUSE_PLAN.md
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/fixed10_from_representative20_manifest.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/runtime.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/aggregate_jacobian_svd_summary.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/jacobian_svd_summary.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/top100_singular_values_long_all7.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/top100_rankwise_left_right_vector_absdot_vs_solver.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/topk_model_vs_solver_similarity.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/topk_model_vs_solver_similarity_split_summary.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/same_point_error_spectral_norm_comparison_top100.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/same_point_error_spectral_norm_split_summary_top100.csv
```

Do not commit dense NPZ Jacobian/SVD files to GitHub unless explicitly requested.
The full result directory is about 314 MB and is better suited for R2.

## R2 Scope

Back up these complete artifacts to R2:

```text
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/
/workspace/polished_selected_download_20260601/
/workspace/polished_selected_download_20260601.zip
docs/burgers_zero_adv_training_visualization_and_jacobian_top100_20260601.md
docs/burgers_zero_adv_training_20260601_github_r2_sync_manifest.md
```

Suggested R2 prefix:

```text
neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/20260601_burgers_zero_adv_training_visualization_jacobian_top100/
```

## Notes

- The analyzed 1000-epoch checkpoint comes from `fast_replace_linf`, so the
  analysis is an L-infinity/sign-replace ablation, not the intended future
  `p=2, q=2` main experiment.
- `J_solver`, baseline Jacobians, and clean+ADV Jacobians were reused from the
  prior representative20 same-point run.
- New 1000-epoch ADV-only `J_model` and `J_error` were recomputed for the fixed
  10-point subset.
