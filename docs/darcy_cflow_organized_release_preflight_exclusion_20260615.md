# Darcy CFlow Organized Release Pre-Run Diagnostic Exclusion - 2026-06-15

## Status

Complete. The formal Darcy CFlow organized release has been cleaned so that
pre-run attack/SVD/Jacobian diagnostic artifacts are not presented as final
results.

Formal release folder:

`outputs/darcy_cflow_timematched_organized_release_20260614/`

Excluded traceability folder:

`outputs/darcy_cflow_timematched_organized_release_20260614_excluded_preflight_20260615/`

## What Was Moved Out

Moved out of the formal release:

- Pre-run attack source tables and NPZ arrays.
- Pre-run SVD/Jacobian source tables and vector arrays.
- Derived attack/SVD rankings, correlations, best-model summaries, and
  statistical appendices that mixed those diagnostics with final clean metrics.
- Polished-report attack/SVD panels and dashboards generated from those
  diagnostics.
- Stale manifests and reports that referenced the moved artifacts.

## What Remains In The Formal Release

Kept in the formal release:

- Clean 52-dataset ranked metrics:
  `data/clean_52dataset_metric_long_ranked.csv`
  and `data/cflow_clean_52dataset_metric_long_ranked.csv`.
- Final clean evaluation source tables under `data/source_tables/`.
- Main RMSE/Relative L2 curves under `figures/main_curves/`.
- Clean-only polished report figures under `figures/polished_report/`.
- Selected non-pre-run attack heatmap examples under `figures/dense_existing/`.
- Current manifest:
  `manifests/file_manifest_20260615_no_preflight.json`.

## Verification

Observed after cleanup:

- A filename scan of the formal release found no residual pre-run diagnostic
  labels.
- A Markdown/JSON/TXT content scan of the formal release found no residual
  pre-run diagnostic labels.
- `figures/polished_report/README.md` now describes only clean-evaluation
  figures.
- `reports/FINAL_RELEASE_CONTENTS_20260615.md` records the current formal
  release contents.

## Remaining Work

A complete final robustness ranking still requires a final-model attack/SVD run
using the intended final time-matched checkpoints. Until that exists locally,
the formal release should not contain attack/SVD ranking tables.
