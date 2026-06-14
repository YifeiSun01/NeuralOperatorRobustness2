# Burgers Solver7860 Dense Image-Only Bundle - 2026-06-14

Status: generated the dense image-only comparison bundle for the latest random
baselines, refreshed it with both all-model and no-random-clean variants, then
added a `group05` panel selected specifically for the clearest `loss3`
advantage among the existing dense attack traces.

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

- Six groups, each with one test sample and five generalization samples.
  `group00` through `group04` are the original display groups. `group05` is a
  curated loss3-best group selected from the 30 already attacked samples in
  `group00` through `group04`.
- All-model variant: six model columns, baseline e500, loss1 e8000, loss2
  e2000, loss3 e1000, random clean Y e8000, and random solver Y e7860.
- No-random-clean variant: five model columns, baseline e500, loss1 e8000,
  loss2 e2000, loss3 e1000, and random solver Y e7860. This variant removes
  random clean Y so its large loss does not compress the other curves/panels.
- Loss123-only variant: three model columns, loss1 e8000, loss2 e2000, and
  loss3 e1000. This variant removes baseline, random clean Y, and random
  solver Y from both the top panels and bottom loss curves.
- Each model cell contains final attack delta, initial condition before/after
  perturbation, model/solver output, and model-minus-solver output.
- The bottom row contains per-sample attack loss progression curves for the
  models included in that variant.
- Six PNGs per group were generated: all-model log-y, all-model linear-y,
  no-random-clean log-y, no-random-clean linear-y, loss123-only log-y, and
  loss123-only linear-y, for 36 PNGs total.
- The loss123-only render was produced by
  `tools/render_burgers_solver7860_loss123_only_dense_20260614.py` from the
  existing trace NPZ files; no training or attack rerun was started.

Group05 selection:

- Selection script:
  `tools/build_burgers_solver7860_loss3_best_group05_20260614.py`.
- It did not rerun training or the old model attacks. It read the existing
  `group00` through `group04` `six_model_attack_traces.npz` files.
- A sample was eligible only if `loss3` was the strict winner for both final
  attacked MSE and attack loss increase.
- Eligible samples were ranked by the sum of `loss3`'s relative final-loss
  margin and relative loss-increase margin versus the best non-loss3 model.
- The resulting `group05` contains one test sample and five generalization
  samples. The detailed ranking is in
  `forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614/group05/source_group_loss3_advantage_ranking.csv`.

Group05 selected samples:

| new sample | source | split | descriptive dataset | best other final | loss3 final | best other increase | loss3 increase |
|---|---|---|---|---:|---:|---:|---:|
| S1 | group04/S1 | test | test Gaussian train-dist c=0.03 | 0.00159004 | 0.00124931 | 0.00158854 | 0.00122288 |
| S2 | group01/S3 | generalization | Gaussian corr=0.009; range [-0.5,1.5]; centroid=10.5; TV=0.025 | 0.0102909 | 0.00366074 | 0.00980776 | 0.00335658 |
| S3 | group02/S2 | generalization | Matern c=0.08, nu=2.5; range [-0.3,1.3]; centroid=6.0; TV=0.013 | 0.00319419 | 0.00134623 | 0.00314220 | 0.00131662 |
| S4 | group00/S6 | generalization | Sawtooth freq=2; range [0.15,1.25]; centroid=8.1; TV=0.004 | 0.00701221 | 0.00291792 | 0.00640859 | 0.00289794 |
| S5 | group00/S5 | generalization | Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5]; centroid=32.1; TV=0.064 | 0.00348057 | 0.00166014 | 0.00307695 | 0.00158509 |
| S6 | group04/S2 | generalization | Matern c=0.04, nu=2.5; range [-0.3,1.3]; centroid=11.1; TV=0.023 | 0.00660930 | 0.00350358 | 0.00627372 | 0.00338262 |

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
- The whole image-only bundle was also copied into the final solver7860 audit
  output under
  `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/figures/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614/`.

Verification:

- Local image-only bundle contains 36 PNG files and no non-PNG files.
- Local trace/data root contains 27 files.
- R2 image-only bundle `rclone size --json` returned 36 files and 89,008,328 bytes.
- R2 visualization mirror `rclone size --json` returned 36 files and 89,008,328 bytes.
- R2 trace/data `rclone size --json` returned 27 files and 203,170,665 bytes.
- The final-audit-output copy contains the same 36 PNG files. R2 verification
  for that nested final-audit folder returned 36 files and 89,008,328 bytes.
- R2 `lsf --recursive` includes directory markers in addition to PNGs; the
  object-count verification above is from `rclone size`.
- The organized release also includes the full image-only bundle as
  `outputs/burgers_solver7860_clean8000_organized_release_20260614/08_dense_image_only_bundle_full_copy/`.
