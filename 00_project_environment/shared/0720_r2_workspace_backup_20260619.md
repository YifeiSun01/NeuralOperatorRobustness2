# R2 Workspace Backup 2026-06-19

Status: completed.

Local root:
- `/workspace/NeuralOperatorRobustness2`

R2 destination:
- `R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/`

Scope:
- Code, scripts, Markdown, LaTeX, requirements, docs.
- Data, outputs, model/checkpoint artifacts, figures, CSV/JSON/NPZ/PT/PTH files.
- New 2026-06-19 Burgers/Darcy final PNG-only plot bundles.
- Darcy/SIR20 robustness attack records and SVD/Jacobian diagnostics.

Excluded from this R2 copy:
- `.git/**`
- `adv_robust/**`
- `.venv/**`, `venv/**`, `env/**`
- Python/runtime caches.

Reason for exclusions:
- Git metadata is backed by the GitHub remote.
- The `adv_robust` virtual environment is reproducible from `requirements.txt`.
- Runtime caches are not research artifacts.

Pre-upload local payload estimate:
- Files: `59295`
- Size: `153.504 GiB`

Upload method:
- `rclone copy`, non-delete incremental upload.
- Existing remote files are preserved; matching files are skipped.

Completed upload notes:
- Main data/artifact pass uploaded or refreshed `79.535 GiB` of changed/missing
  objects under the selected R2 prefix.
- A second lightweight pass force-refreshed code, scripts, Markdown, LaTeX,
  JSON/YAML/TXT, and requirements files so same-size text edits were not missed.
- R2 returned `501 NotImplemented` for a few metadata-only update paths; those
  files were retried with S3 metadata writes disabled and uploaded successfully.

Verification:
- `outputs/final_png_only_yaxis_metric_loglog_20260619` checked with `0`
  differences and `60` matching PNG files.
- `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/svd_jacobian_vectors`
  checked with `0` differences and `325` matching NPZ files.
- `tools` checked with `0` differences and `376` matching files after excluding
  runtime `__pycache__` files.
- Remote key files confirmed present:
  `final_eval_metrics.csv`, `final_eval_metrics.json`,
  `svd_jacobian_25sample_manifest.csv`, and `svd_jacobian_metrics.csv`.
