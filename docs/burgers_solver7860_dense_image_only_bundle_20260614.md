# Burgers Solver7860 Dense Image-Only Bundle - 2026-06-14

Status: generated the dense image-only comparison bundle for the latest random
baselines, then refreshed it with both all-model and no-random-clean variants.

Local artifacts:

- Image-only bundle:
  `visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614/`
- Full visualization mirror:
  `visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614/`
- Trace/data root:
  `forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614/`

What was recomputed:

- Recomputed only the `random_solver_y` dense attack trace from the latest
  checkpoint:
  `adversarial_training_runs/burgers_wideparam_random_field_solver_y_7860ep_continue_20260613/burgers/checkpoints/burgers_epoch7860_step007860.pt`.
- Reused existing baseline/loss1/loss2/loss3 traces from
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611`.
- Reused existing `random_clean_y` traces from
  `forensics/burgers_wideparam_loss123_randomfield_selected_worktime_round00_p2q2_six_model_visuals_20260613`.

Image contents:

- Five groups, each with one test sample and five generalization samples.
- All-model variant: six model columns, baseline, loss1, loss2, loss3, random
  clean Y e8000, and random solver Y e7860.
- No-random-clean variant: five model columns, baseline, loss1, loss2, loss3,
  and random solver Y e7860. This variant removes random clean Y so its large
  loss does not compress the other curves/panels.
- Each model cell contains final attack delta, initial condition before/after
  perturbation, model/solver output, and model-minus-solver output.
- The bottom row contains per-sample attack loss progression curves for the
  models included in that variant.
- Four PNGs per group were generated: all-model log-y, all-model linear-y,
  no-random-clean log-y, and no-random-clean linear-y, for 20 PNGs total.

Visual-sample attack loss summary across all 30 samples:

| model | initial mean | final mean | increase | n |
|---|---:|---:|---:|---:|
| baseline | 0.00128788 | 0.0152441 | 0.0139563 | 30 |
| loss1 | 0.000429735 | 0.00694397 | 0.00651423 | 30 |
| loss2 | 0.000466273 | 0.00661674 | 0.00615046 | 30 |
| loss3 | 0.000146545 | 0.00411033 | 0.00396379 | 30 |
| random_clean_y | 0.0113246 | 0.0569579 | 0.0456333 | 30 |
| random_solver_y | 0.000377704 | 0.00779751 | 0.0074198 | 30 |

R2 upload:

- Image-only bundle synced to
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614`.
- Visualization mirror synced to
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614`.
- Trace/data root synced to
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614`.

Verification:

- Local image-only bundle contains 20 PNG files and no non-PNG files.
- R2 image-only bundle `rclone size --json` returned 20 files and 53,529,981 bytes.
- R2 visualization mirror `rclone size --json` returned 20 files and 53,529,981 bytes.
- R2 trace/data `rclone size --json` returned 21 files and 169,276,310 bytes.
- R2 `lsf --recursive` includes directory markers in addition to PNGs; the
  object-count verification above is from `rclone size`.
