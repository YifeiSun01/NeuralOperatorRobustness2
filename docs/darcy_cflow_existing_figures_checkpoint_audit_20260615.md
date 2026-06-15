# Darcy CFlow Existing Figures Checkpoint Audit - 2026-06-15

## Question

Which checkpoint epochs were used by the existing `loss3_advantage`, `extra15`,
`absolute11`, `diagnostic_existing`, and curated/existing Darcy figures?

## Observed Sources

- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260615_loss3_advantage_extra15/summary.csv`
- `analysis_outputs/darcy_seven_model_attack_heatmaps_20260615_loss3_advantage_extra15/README.md`
- `outputs/darcy_sir20_existing_curated_bundle_20260614/reports/BUNDLE_SUMMARY.md`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`

## Findings

The `loss3_advantage_extra15` attack heatmaps are **not** final 3000-3500 epoch
model results. They use this older checkpoint family:

| model | source label | checkpoint epoch |
|---|---|---:|
| baseline | `pre_adversarial_training_best_epoch500` | baseline epoch 500-era checkpoint |
| loss1 | `loss1_adversarial_training_1000c` | 1000 |
| loss2 | `loss2_adversarial_training_1000c` | 1026 |
| loss3 | `loss3_adversarial_training_1000c` | 1011 |
| physics loss | `physics_loss_adversarial_training_1000c` | 1040 |
| random clean y | `random_binary_source_fixed_clean_y_1100` | 1100 |
| random solver y | `random_binary_source_solver_recomputed_y_1100` | 1100 |

The old curated bundle path
`outputs/darcy_sir20_existing_curated_bundle_20260614/figures/attack_heatmaps_loss3_advantage/`
contains selected old seven-model attack heatmap figures. That bundle explicitly
says the random methods only existed locally to epoch 1100 at that time.

The `legacy_random_inclusive` figures in
`outputs/darcy_sir20_existing_curated_bundle_20260614/figures/legacy_random_inclusive/`
are also legacy 0-1000/1100-scale figures, not final 3000-3500 random runs.

By contrast, the current organized clean-evaluation release table
`outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`
uses the final clean-evaluation checkpoint family:

| method | epoch |
|---|---:|
| baseline | 0 |
| loss1 | 3000 |
| loss2 | 3079 |
| loss3 | 3033 |
| physics | 3121 |
| random_clean | 3500 |
| random_solver | 3500 |

Therefore the `absolute11` polished-report clean evaluation figures under
`outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/`
are clean-evaluation final-model figures, not attack heatmaps.

## Bottom Line

- `loss3_advantage_extra15` attack heatmaps: **old 1000/1100 checkpoint family**.
- `existing_curated_bundle/figures/attack_heatmaps_loss3_advantage`: **old selected attack heatmaps**, not final.
- `existing_curated_bundle/figures/legacy_random_inclusive`: **legacy 0-1000/1100**, not final.
- organized-release `absolute11`/polished clean curves: **final clean eval**, with loss1 3000, loss2 3079, loss3 3033, physics 3121, random_clean/random_solver 3500.
- The new final attack20/SVD25 run is separate and is being written to
  `outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/`.
