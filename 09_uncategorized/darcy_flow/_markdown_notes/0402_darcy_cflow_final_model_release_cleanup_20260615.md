# Darcy CFlow Final-Model Release Cleanup - 2026-06-15

## Status

Complete. The formal Darcy CFlow organized release now contains final-model
clean-evaluation artifacts for exactly seven models:

- baseline
- loss1
- loss2
- loss3
- physics
- random_clean
- random_solver

## Final Model Epochs

Observed from
`outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`:

| model | epoch |
|---|---:|
| baseline | 0 |
| loss1 | 3000 |
| loss2 | 3079 |
| loss3 | 3033 |
| physics | 3121 |
| random_clean | 3500 |
| random_solver | 3500 |

Each model has clean metrics on `52` datasets: train, test, and `50`
generalization datasets.

## Formal Release Folder

`outputs/darcy_cflow_timematched_organized_release_20260614/`

Current formal-release contents:

- Clean 52-dataset ranked metrics:
  `data/clean_52dataset_metric_long_ranked.csv`
  and `data/cflow_clean_52dataset_metric_long_ranked.csv`.
- Final clean evaluation source tables under `data/source_tables/`.
- Main RMSE/Relative L2 curves under `figures/main_curves/`.
- Clean-only polished report figures under `figures/polished_report/`.
- Current manifest:
  `manifests/file_manifest_20260615_final_models_only.json`.

## Verification

Observed after cleanup:

- No local Darcy CFlow output directories matching obsolete diagnostic labels
  remain under `outputs/` at depth two.
- A filename scan of the formal release found no obsolete diagnostic labels.
- A Markdown/JSON/TXT content scan of the formal release found no obsolete
  diagnostic labels.
- The formal release model/epoch table contains exactly the intended seven
  models and final epochs listed above.
- Existing local 50-step attack heatmaps were checked and found to use earlier
  1000-1100 epoch checkpoints, so they were removed from the formal final-model
  release.

## Remaining Work

A complete final robustness ranking and final attack heatmaps still require a
50-step attack/SVD run from these same final checkpoints. Until that exists
locally, the formal release should not contain attack/SVD ranking tables or
attack heatmaps.
