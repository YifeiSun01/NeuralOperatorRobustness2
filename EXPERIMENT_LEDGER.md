# Experiment Ledger

## 2026-05-16 Unified Eval-Metric Raw-Data Interpretation

Status: completed from the canonical 27-run saved trajectory data; no attack was rerun.

Key conclusions recorded in `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`:

- For endpoint solver-level `loss3_original`, directly optimizing the `loss3_original` family is strongest: `loss3_original_generalized_power` reaches final mean `6.8573`, followed by `loss3_original_lp_steepest_pgd` at `6.3782` and `loss3_original_pgd` at `5.3949`.
- For `loss3_increment_ratio`, the batch-mean winner is also `loss3_original_generalized_power` at `0.8199`, but per-sample comparisons are subtler: direct `loss3_increment_ratio_lp_steepest_pgd` wins more individual samples against the original-objective runs.
- For `loss1_original` and `loss2_original`, direct `loss1/loss2` original objectives remain best; `loss3_original` is not a reliable surrogate for them.
- Increment-ratio and regularized objectives are meaningful for perturbation-efficient or penalty-aware behavior, but they are not the strongest endpoint `loss3_original` attacks.

## 2026-05-16 Unified Eval-Metric Curves Shared-Y-Zero Correction

Status: completed after reviewing the first unified-evaluation replots.

Correction:

- The first three-panel replots fixed the y-axis metric within each figure, but did not enforce one global y-axis range across every figure that uses the same evaluation metric.
- The corrected output now uses `shared_y_zero`: same evaluation metric, same y-axis min/max, y-axis starts at `0`.
- The 3x3 matrix script was also corrected so all nine subplots share the same y-axis when they are evaluated by the same metric.

Corrected scripts and docs:

- `tools/plot_batch_eval_metric_three_panel_curves.py`
- `tools/plot_batch_eval_metric_matrix_curves.py`
- `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`

Corrected local outputs:

- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_three_panel_shared_y_zero/png/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves_shared_y_zero/png/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves_shared_y_zero/index_png/`

R2 upload:

- Uploaded `45` corrected PNG files, `24657968` bytes.
- R2 prefixes: `.../figures/eval_metric_three_panel_shared_y_zero/` and `.../figures/eval_metric_curves_shared_y_zero/`.

Path clarification:

- Use the complete 27-run source directory `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- The `20260516_..._local_repro` directory is only a small local reproduction of the loss3 subset, not the canonical full 27-run dataset.

## 2026-05-16 Unified Eval-Metric Three-Panel Loss Curve Replots

Status: completed from existing saved trajectory data; no attack was rerun.

Purpose:

- Replot the old three-panel `LOSS1/LOSS2/LOSS3 Objective Curves` layout so all three panels in one figure use the same evaluation metric on the y-axis.
- This fixes the ambiguity in the older figures, where each panel plotted its own optimized objective.
- The new figures keep original / increment-ratio / regularized as the three panels and PGD / LP-steepest PGD / generalized power iteration as the curves.
- Each new figure shares one y-axis min/max range across its three panels.

Data and scripts:

- Source data: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- Old script: `tools/plot_batch_three_loss_loss_only.py`.
- Existing 3x3 unified script: `tools/plot_batch_eval_metric_matrix_curves.py`.
- New three-panel unified script: `tools/plot_batch_eval_metric_three_panel_curves.py`.
- Documentation: `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`.

Outputs:

- Local PNG directory: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_three_panel/png/`.
- Generated 27 batch mean/std figures: 3 optimized-loss families times 9 shared evaluation metrics.
- R2 prefix: `s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_three_panel/png/`.
- R2 upload summary: `uploaded_files=27`, `uploaded_bytes=14315408`.

## 2026-05-16 Final RI / Ray Profile Report And R2 Backup

Status: completed and cleaned for FNO / Burgers `nu=0.001`. The final report is `docs/loss3_ray_profile_ri_final_report_fno_nu0p001_20260516.md`. The final valid data directory is `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.

Final conclusion:

- Ray / RI experiment succeeds as a local-to-global nonlinear diagnostic. Clean-point local directions win the small-radius diagnostics, but they do not remain endpoint-best after following the same fixed ray to `r=8`.
- `local_outward_growth` wins small norm-growth `100/100`, but endpoint `loss3` mean at `r=8` is `2.720` with `0/100` endpoint wins.
- `local_residual_movement` wins small residual-increment `100/100`, endpoint `loss3` mean is `4.139`, and endpoint wins are `31/100`.
- `loss3_original_final` has small local wins `0/100`, but endpoint `loss3` mean is `5.447`; it wins `58/100` among all directions and `81/100` among the three finite PGD attack objectives.

Crossover evidence:

- Mean `loss3_original_final` crosses `local_outward_growth` at approximately `r=0.823321`.
- Mean `loss3_original_final` crosses `loss3_increment_ratio_final` at approximately `r=0.929778`.
- No below-then-above crossing against `local_residual_movement` appears by `r=1` or `r=2` in the dense scan.

Final figures:

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_all_metrics_by_direction_0to8_formula_labeled_std.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to0p5_formula_labeled_std.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to1p0_formula_labeled_std.png`

R2 backup:

- `s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`
- Upload manifest: `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/r2_upload_manifest_20260516.txt`
- Manifest summary: `file_count=58`, `bytes_total=97423607`, `deleted_wrong_non_fixedsign_objects=0`.
- Full related-artifact sync manifest: `docs/r2_sync_manifest_ray_profile_20260516.md`; broader Ray-profile artifact sync uploaded `198` files and `145207187` bytes.

Bug note:

- Earlier non-fixed-sign manual PGD output is invalid. That wrapper backpropagated `-objective` and then updated `delta += alpha * grad`, which is descent for the target objective. The corrected PGD result uses `objective.backward()`, ascent update, and projection.

## 2026-05-16 Normal-Protocol Experiment 4 Batch-100 Ray Profile

Status: completed on GPU for FNO / Burgers `nu=0.001`. This is the requested old-style protocol check: no best-over-steps, no multi-restart, single initialization per attack objective, and final step direction only.

Files:

- `tools/run_loss3_ray_profile_normal_batch.py`
- `docs/loss3_ray_profile_normal_fno_nu0p001_gpu_batch100_plan_20260516.md`
- `docs/loss3_ray_profile_normal_fno_nu0p001_gpu_batch100_result_20260516.md`
- `forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/`

Run settings and hardware:

- Samples: `0..99` batch size 100.
- Endpoint radius: `epsilon=8.0`; 45 radii in the ray profile.
- Attack protocol: 50 Adam steps, learning rate `0.3`, final step only.
- Runtime: Tesla V100-SXM2-32GB, torch `2.8.0+cu126`, CUDA `12.6`, JAX backend `gpu`, `sm_70` verified.
- Runtime: 278.71 seconds.

Key result under this exact normal protocol:

- Endpoint winner counts at `r=8`: `loss3_increment_ratio_final` 47/100, `loss3_regularized_final` 43/100, `local_residual_movement` 7/100, `loss3_residual_increment_ratio_final` 2/100, `loss3_original_final` 1/100.
- Small-radius norm-growth winner: `local_outward_growth` 100/100.
- Small-radius residual-increment winner: `local_residual_movement` 100/100.
- Interpretation: batch size 100 does not make final-step `loss3_original` best under the old protocol. The ray curves clearly support the local-to-global mismatch: local directions have the steepest small-radius slope, but finite-radius optimized directions dominate at large radius.


## 2026-05-16 Corrected Experiment 4 Ray Profile Batch-20 GPU Run

Status: completed for FNO / 1D Burgers `nu=0.001` with GPU-only execution. This entry supersedes the earlier five-sample Ray Profile endpoint-winner interpretation.

Corrected files:

- `tools/run_loss3_ray_profile_corrected.py`
- `docs/loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_plan_20260516.md`
- `docs/loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_result_20260516.md`
- `docs/loss3_ray_profile_fno_nu0p001_gpu_result_20260516.md` marked historical/superseded
- `docs/loss3_original_theory_experiment_plan.md` Experiment 4 row updated

Run evidence:

- Batch samples: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 40, 47, 115`.
- Endpoint radius: `epsilon=8.0`; 41 ray radii.
- Attack steps: 100 per restart; learning rate `0.3`; update mode `raw`.
- Runtime: Tesla V100-SXM2-32GB, torch `2.8.0+cu126`, CUDA `12.6`, JAX backend `gpu`, `sm_70` verified.
- Output directory: `forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/`.

Corrected conclusion:

- Finite-radius endpoint `loss3_original` winner at `r=8`: `loss3_original_pgd_best` in 20/20 samples.
- Small-radius clean residual norm-growth winners: mostly `loss3_regularized_pgd_best` / `local_outward_growth`; these are numerically tied or near-tied because regularized PGD collapses toward the local outward-growth direction in this setting.
- Small-radius residual-increment winners: `loss3_residual_increment_ratio_pgd_best` in 18/20 samples and `local_residual_movement` in 2/20 samples.
- Interpretation: Experiment 4 is a Ray diagnostic for nonlinearity. Local/ratio directions describe the clean-point small-radius slopes, but direct `loss3_original` PGD is the fair large-radius endpoint attack. The mismatch is the intended local-to-global gap.

Raw artifacts recorded:

- `ray_profile.csv`, `ray_profile_aggregate_curves.csv`, `ray_direction_summary.csv`, `ray_direction_aggregate.csv`, `ray_winner_summary.csv`, `attack_trace.csv`, `attack_best_by_sample.csv`, `direction_alignment.csv`, `directions.npz`, `deltas.npz`, figures, and `manifest.json`.


## 2026-05-16 Experiment 4 Ray Profile Status Updated To Complete

Status: historical/superseded by the corrected batch-20 GPU run above. The raw five-sample artifacts are retained, but the endpoint-winner interpretation is not the final conclusion.

Previous status: completed for the current FNO / 1D Burgers `nu=0.001` scope. This entry
supersedes earlier notes that described Experiment 4 as not yet fully run.

Updated durable status files:

- `docs/loss3_original_theory_experiment_plan.md`
- `docs/loss3_ray_profile_fno_nu0p001_gpu_plan_20260516.md`
- `docs/loss3_ray_profile_fno_nu0p001_gpu_result_20260516.md`
- `EXPERIMENT_LEDGER.md`

Completion evidence:

- Full GPU-only ray-profile run completed on Tesla V100-SXM2-32GB with
  `torch==2.8.0+cu126`, CUDA `12.6`, JAX backend `gpu`, and required torch arch
  `sm_70` verified in the manifest.
- Samples: `0, 7, 40, 47, 115`; endpoint radius `epsilon=8.0`; 45 ray radii;
  50 Adam ascent steps per regenerated finite-radius direction.
- Direction sources: `loss3_original`, `loss3_increment_ratio`,
  `loss3_residual_increment_ratio`, `loss3_regularized`, `local_error_svd`,
  `local_outward_growth`, and `random`.
- Output directory:
  `forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/`.
- Recorded `ray_profile.csv`, `ray_profile_aggregate_curves.csv`,
  `ray_direction_summary.csv`, `ray_direction_aggregate.csv`,
  `ray_winner_summary.csv`, `attack_trace.csv`, `directions.npz`, figures, and
  `manifest.json`.

Historical conclusions from that superseded five-sample run:

- This old five-sample run was useful for raw curve inspection, but its endpoint-winner interpretation is superseded by the corrected batch-20 run above.
- The old run suggested a nonlinear local-to-global gap.
- Small-radius clean residual norm growth winner: `local_outward_growth` in 5/5
  samples.
- Small-radius residual-increment winner: `local_error_svd` in 5/5 samples.
- Finite endpoint `r=8` winner: `loss3_regularized` in 3/5 samples and
  `loss3_increment_ratio` in 2/5 samples.
- Therefore the local directions are real local diagnostics, but they are not
  the finite-radius endpoint-best attack directions.

Remaining work for Experiment 4:

- None for the current `nu=0.001` scope. Future repetitions for other
  viscosities, models, or endpoint radii should be treated as extension
  experiments, not blockers for marking Experiment 4 complete.

## 2026-05-16 Experiment 3 Small-Epsilon Sweep Status Updated To Complete

Status: completed for the current FNO / 1D Burgers `nu=0.001` scope. This entry
supersedes earlier same-day notes that described Experiment 3 as only partially
done before the GPU runs were completed.

Updated durable status files:

- `docs/loss3_original_theory_experiment_plan.md`
- `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
- `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`
- `EXPERIMENT_LEDGER.md`

Completion evidence:

- Candidate-bank small-epsilon sweep completed on GPU with epsilons
  `{1e-4, 1e-3, 1e-2, 1e-1}` for indices `0, 7, 40, 47, 115`.
- Recorded `L_f`, `L_j`, `L_e`, `G_e`, direction source counts, direction
  stability, local-reference ratios, figures, CSV tables, and a GPU manifest in
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`.
- Second version completed on GPU with projected gradient ascent on the unit
  direction `v`, recorded in
  `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`.
- The gradient version confirms that the candidate-bank directions were not just
  arbitrary picks: for small epsilon, optimized directions and values recover the
  same clean local Jacobian / outward-growth references.

Key conclusions now recorded:

- For `epsilon <= 1e-2`, the finite-difference ratios and directions are stable
  and match the clean local structure. This is the intended epsilon-refinement /
  local-convergence result.
- At `epsilon=0.1`, finite-radius drift becomes visible, especially in `L_e` and
  `G_e`, so that radius should not be described as purely local.
- `L_f` and `L_j` are much larger than `L_e`, showing that FNO and solver can be
  sensitive while still co-moving locally.
- `G_e` is much smaller than `L_e`, so residual-field movement and outward growth
  of the current clean residual are different diagnostics.
- The `L_j/idx47` direction-angle caveat is explained by a near-degenerate
  solver top-2 singular subspace (`sigma2/sigma1 = 0.977468`); top-2 subspace
  alignment is the correct diagnostic there.

Remaining work for Experiment 3:

- None for the current `nu=0.001` scope. Future repetitions for other viscosities
  or model families should be treated as extension experiments, not blockers for
  marking Experiment 3 complete.

## 2026-05-16 Loss3 Small-Epsilon Sweep Requirements And R2 Artifact Sync

Status: superseded pre-run requirements / artifact sync note. No numerical small-epsilon sweep
was run in that earlier turn, but Experiment 3 was later completed on GPU; see
`2026-05-16 Experiment 3 Small-Epsilon Sweep Status Updated To Complete`.

Source files and artifact roots:
- `docs/loss3_original_theory_experiment_plan.md`
- `docs/loss3_original_plan_r2_completion_audit_20260515.md`
- R2 prefix:
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`
- Local model checkpoint:
  `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`

Output files / synced artifacts:
- `docs/loss3_small_epsilon_sweep_requirements_r2_sync_20260516.md`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`
- `forensics/outward_growth_direction_20260515/`
- `results/burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only/`

Key settings:
- Experiment 3 target scope: FNO / 1D Burgers `nu=0.001`.
- Planned epsilon list: `{1e-4, 1e-3, 1e-2, 1e-1}`.
- Quantities still to estimate systematically: `L_f(epsilon)`,
  `L_j(epsilon)`, `L_e(epsilon)`, `G_e(epsilon)`, and cross-epsilon direction
  cosines.
- Existing local diagnostic indices from SVD/outward-growth artifacts:
  `[0, 7, 40, 47, 115]`.

Observed from local verification:
- The restored train split loads with `x=(1350, 1024)`, `y=(1350, 1024)`.
- The restored test split loads with `x=(150, 1024)`, `y=(150, 1024)`.
- The FNO checkpoint exists locally.
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/` contains
  15 raw `*_jacobian_svd.npz` files, covering FNO, solver, and error SVD arrays
  for five indices.
- `forensics/outward_growth_direction_20260515/` contains partial small-radius
  finite-difference evidence at radii `1e-4`, `1e-3`, and `1e-2`.
- `results/burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only/` contains
  older attack summaries for `Linf` epsilon `0.01` and `0.1`, but not the
  planned local operator/risk-growth sweep.

Inference:
- The required data/model/artifact prerequisites for implementing and running
  the planned Small-Epsilon Sweep are now present locally.
- Superseded: at the time of this requirements-sync note, Experiment 3 was only
  partially done. It was later completed on GPU with the systematic sweep and
  direction-stability analysis.

Remaining work from this pre-run note: superseded by the completed GPU sweep and
gradient-optimization follow-up.

## 2026-05-16 GitHub Script Sync And R2 Artifact Upload

Status: completed. Generated artifacts were uploaded to R2, and Markdown/Python files were committed and pushed to GitHub branch `vast-ai`.

Source files and artifact roots:
- Markdown / source files in the repository working tree.
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`
- `forensics/local_jacobian_frequency_20260514/`
- `forensics/outward_growth_direction_20260515/`
- `forensics/three_loss_pairwise_gradients_20260516/`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/split_summary.json`

Output locations:
- GitHub repository: `YifeiSun01/NeuralOperatorRobustness2`, branch `vast-ai`.
- GitHub commit with docs/tools: `6e4d37e` (`Record loss3 diagnostics and analysis scripts`).
- R2 prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/`.
- R2 artifact prefixes:
  - `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`
  - `forensics/local_jacobian_frequency_20260514/`
  - `forensics/outward_growth_direction_20260515/`
  - `forensics/three_loss_pairwise_gradients_20260516/`
  - `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/split_summary.json`

Observed from command output:
- R2 `forensics/` listing after upload included all four uploaded artifact directories.
- GitHub push output reported `36c5e30..6e4d37e  vast-ai -> vast-ai`.
- The staged GitHub commit contained existing Markdown/Python files only; deleted generated-result artifacts were not staged.
- No credential strings were found in the staged Markdown/Python files by the pre-commit scan.

Inference:
- GitHub now has the lightweight script/documentation update.
- R2 now has the current local generated numerical artifacts and plots under the selected machine-sync prefix.

Remaining work:
- The working tree still contains many pre-existing tracked generated artifacts marked deleted, plus non-script generated/untracked artifacts. These were intentionally not committed to GitHub.

## 2026-05-16 Loss3 Minimal Six Experiment Status Added To Plan

Status: documentation update only; no numerical experiment was run in this turn.

Source files:
- `docs/loss3_original_theory_experiment_plan.md`
- Prior completion notes in the conversation and existing audit/result docs.

Output files:
- `docs/loss3_original_theory_experiment_plan.md`

Key settings / scope:
- Current next-round scope is FNO / Burgers `nu=0.001` only.
- Broader `nu=0.01` or default-architecture sweeps are explicitly not required for the immediate next plan.

Observed from the existing records and user-confirmed notes:
- Superseded later on 2026-05-16: Experiment 3 was completed after this status
  note. Current status is Done: Experiments 1, 2, 3, and 5; Partially done:
  Experiment 6; Not yet done as a full experiment: Experiment 4.

Inference from the current status:
- Next priority should be Ray Profile / Local-to-Global Profile.
- Then complete the exact Direction Rotation Along Path experiment with recomputed `v_e*(x_t)`.
- The full Small-Epsilon Sweep was later completed and is no longer a blocker.

Remaining work:
- Run Experiment 4 ray-profile curves.
- Complete Experiment 6 with pathwise recomputed local top residual directions.
- Superseded: Experiment 3 was later completed with systematic `L_f`, `L_j`, `L_e`, `G_e`, and direction-cosine tables.

Last updated: 2026-05-16 UTC

This is the fixed entry point for experiment status. It must separate observed
evidence from inference. Chat history is not a durable experiment record.


## 2026-05-16 Chat Angle Tables Saved To Markdown

Status:

- No new numerical experiment was run.
- Appended the chat-presented numeric angle tables to
  `docs/angle_experiment_inventory_20260516.md` under `Chat-Presented Numeric Tables`.

Observed evidence summarized:

- Saved the overall 4-batch / 7-angle-definition table.
- Saved clean-point candidate direction angle means/stds.
- Saved endpoint-vs-movement loss3-path local-affine versus true nonlinear every-5-step table.
- Saved native L1/L2/L3 pairwise gradient angle summary and loss3-trajectory every-5-step table.

Inference:

- This was documentation of existing recorded data, not a new experiment.


## 2026-05-16 Angle Inventory Tables Presented In Chat

Status:

- No new numerical experiment was run.
- Presented the angle experiment inventory and headline numeric tables in the chat.

Source files:

- `docs/angle_experiment_inventory_20260516.md`
- `docs/angle_diagnostic_raw_tables_20260516.md`
- `docs/three_loss_pairwise_gradient_angles_result_20260516.md`
- `docs/outward_growth_direction_result_20260515.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source_k.csv`

Observed evidence summarized:

- Presented 4 angle diagnostic batches and 7 angle definitions.
- Parentheses in presented tables are standard deviations in degrees, not variances.
- Step-indexed values are saved every 5 PGD steps: `k = 0, 5, ..., 50`.

Inference:

- This turn is a presentation of existing recorded evidence only.


## 2026-05-16 Angle Experiment Inventory Created

Status:

- No new numerical experiment was run.
- Created `docs/angle_experiment_inventory_20260516.md` to summarize all angle-related
  diagnostics and their headline results.

Observed evidence summarized:

- Current records contain 4 angle-related diagnostic batches and 7 distinct angle
  definitions.
- The clean-point candidate-direction table has 135 CSV rows.
- The local-affine same-delta table has 165 rows and contains two angle definitions:
  endpoint-vs-movement and bias-vs-movement.
- The true nonlinear endpoint-vs-movement table has 165 rows.
- The native pairwise loss-gradient table has 165 rows and contains three angle
  definitions: L1-L2, L1-L3, and L2-L3.
- Headline native pairwise result for `k>=5`: `L1-L2 = 3.91 (7.06) deg`,
  `L1-L3 = 56.51 (18.41) deg`, `L2-L3 = 57.23 (18.41) deg`.

Inference:

- The angle diagnostics should be separated into endpoint-vs-movement/residual geometry
  diagnostics and native pairwise objective-gradient diagnostics.
- Only the native pairwise loss-gradient experiment directly proves that `loss3_original`
  has a different update direction from native `loss1_original` and `loss2_original`.


## 2026-05-16 Pairwise Gradient Angle Aggregation Clarified

Status:

- No new numerical experiment was run.
- Updated `docs/three_loss_pairwise_gradient_angles_result_20260516.md` to clarify
  saved-step cadence and mean/std aggregation.

Observed evidence summarized:

- Saved trajectory steps are every 5 PGD steps: `k = 0, 5, ..., 50`.
- Parentheses in the tables are standard deviations, not variances.
- The script computes population standard deviation with `np.std(..., ddof=0)`.
- `all k>=5` aggregates `150` rows = `5 samples x 3 delta sources x 10 saved nonzero k values`.
- Each per-trajectory `k>=5` summary aggregates `50` rows.
- A single `k=50` per-trajectory entry aggregates `5` rows, one per sample.

Inference:

- The reported angle summaries describe saved every-5-step trajectory points, not every
  internal PGD update step.


## 2026-05-16 b-plus-Adelta Versus Adelta Evidence Documented

Status:

- No new numerical experiment was run.
- Updated `docs/outward_growth_direction_result_20260515.md` with a section named
  `Where The Evidence Shows b+A delta And A delta Are Different`.

Observed evidence summarized:

- The endpoint residual objective keeps the clean residual term:
  `||f(x+delta)-j(x+delta)|| ~= ||b + A delta||`.
- The residual-movement objective subtracts the clean residual away:
  `||(f-j)(x+delta)-(f-j)(x)|| ~= ||A delta||`.
- Direction-response evidence: `error_top` top-8 has larger mismatch gain
  `0.412169` than `outward_growth` `0.368053`, but far smaller outward component
  `0.0140571` versus `0.166514`.
- Rank-1 `error` direction has mismatch gain `0.868217` but outward component
  `-0.00677836`.
- Direction-angle evidence: `outward_growth` versus `error` top-8 angle mean is
  `84.68 deg`; versus `error` rank-1 angle mean is `90.87 deg`.
- Finite-difference evidence: `outward_growth` predicted growth `0.166514` matches
  actual growth `0.167286` at `rho=1e-4` and `0.166668` at `rho=1e-3`.

Inference:

- The existing outward-growth experiment directly supports the claim that the local
  direction for `||A delta||` and the local direction for `||b + A delta||` are not the
  same.
- This should be described as endpoint residual versus residual increment, not as native
  `loss1`, because native `loss1 = ||f(x+delta)-f(x)||` linearizes with `J_f`, not
  `A = J_f - J_j`.


## 2026-05-16 Three-Loss Pairwise Gradient Angle Experiment

Status:

- Completed new GPU post-processing experiment requested by the user.
- Created `tools/analyze_three_loss_pairwise_gradients.py`.
- Created `docs/three_loss_pairwise_gradient_angles_result_20260516.md`.

Source files / inputs:

- `tools/analyze_three_loss_pairwise_gradients.py`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_000/loss1_original_pgd/config.json` and matching per-index configs.

Output files:

- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/pairwise_three_loss_gradients.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/manifest.json`
- `docs/three_loss_pairwise_gradient_angles_result_20260516.md`

Key settings:

- Device: `cuda`.
- Samples: `0, 7, 40, 47, 115`.
- Delta sources: `loss1_original_pgd`, `loss2_original_pgd`, `loss3_original_pgd`.
- Saved steps: `0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`.
- Compared native original objectives: `loss1`, `loss2`, `loss3`.

Observed evidence:

- Raw output has `165` rows = `5 samples x 3 delta sources x 11 saved steps`.
- For `k >= 5`, aggregate pairwise gradient angles are: `L1-L2 = 3.91 (7.06)` deg,
  `L1-L3 = 56.51 (18.41)` deg, `L2-L3 = 57.23 (18.41)` deg.
- For `k >= 5` along the `loss3_original_pgd` delta source, angles are:
  `L1-L2 = 9.60 (10.04)` deg, `L1-L3 = 61.30 (20.48)` deg,
  `L2-L3 = 63.05 (20.16)` deg.
- At `k=50` along the `loss3_original_pgd` delta source, mean angles are:
  `L1-L2 = 4.41 (1.54)` deg, `L1-L3 = 47.80 (20.31)` deg,
  `L2-L3 = 48.55 (20.04)` deg.

Inference:

- Native `loss1` and `loss2` gradients are usually close after nonzero attack progress.
- Native `loss3_original` gradients remain substantially different from both `loss1`
  and `loss2`, because `loss3` differentiates through the perturbed solver target
  `j(x + delta)`.
- This is the direct same-delta objective-gradient evidence that was missing from
  the earlier outward-growth experiment.

Remaining work:

- Optional: plot the three pairwise angle curves versus `k` for each delta source.


## 2026-05-16 Outward-Growth Experiment Reset Clarified

Status:

- No new numerical experiment was run.
- Updated `docs/outward_growth_direction_result_20260515.md` with a reset section
  explaining what the original outward-growth experiment actually ran.

Observed evidence summarized:

- The original outward-growth experiment used `A = J_f - J_j` and `b = f(x) - j(x)`.
- It compared the pure residual-field movement direction, which maximizes `||A v||`,
  with the clean-residual outward-growth direction `normalize(A^T b)`.
- It did not compute native pairwise gradient angles among `loss1`, `loss2`, and
  `loss3` at the same `delta_k` points.

Inference:

- The existing experiment directly supports the distinction between residual movement
  and endpoint residual growth inside the `loss3` residual geometry.
- It is conceptually related to the `loss1` versus `loss3` question, but it should not
  be described as a native `loss1` angle experiment because `loss1` uses `J_f`, not
  `J_f - J_j`.
- A separate pairwise objective-gradient diagnostic is needed to directly show that
  native `loss1_original`, `loss2_original`, and `loss3_original` have different
  local update directions.


## 2026-05-16 Angle Column Labels Reclassified As Delta Sources

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` to state explicitly that
  `loss1`, `loss2`, and `loss3` table columns are delta-source labels, not native
  angle objectives.

Observed evidence summarized:

- The true nonlinear angle formula compares two `loss3` residual-field gradients.
- The local-affine formulas use `A = J_f - J_j` and `b = f(x) - j(x)`.
- Therefore the reported `loss1` and `loss2` columns do not represent native
  `loss1` or `loss2` angle formulas; they only identify that `delta_k` came from
  the `loss1_original_pgd` or `loss2_original_pgd` saved trajectory.

Inference:

- The current angle tables should not be used to claim anything about intrinsic
  `loss1` gradient geometry, because `loss1` does not include the perturbed oracle
  `j(x + delta)`.
- A direct `loss1/loss2/loss3` gradient-angle experiment would need to compute
  `grad L1`, `grad L2`, and `grad L3` at the same fixed `delta_k` points.


## 2026-05-16 Angle Table Pairwise-Gradient Clarification

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` to clarify that the
  current endpoint-vs-movement angle tables are not pairwise `loss1/loss2/loss3`
  gradient-angle tables.

Observed evidence summarized:

- The current true nonlinear endpoint-vs-movement table compares two `loss3`-related
  gradients at saved points `z_k = x + delta_k`.
- It does not compute pairwise angles among `grad L1(delta_k)`, `grad L2(delta_k)`,
  and `grad L3(delta_k)`.

Inference:

- The existing tables support a statement about `loss3_original` endpoint-error
  geometry versus residual-movement geometry, especially along the `loss3` trajectory.
- A separate pairwise-gradient diagnostic is needed to directly claim that
  `loss1_original`, `loss2_original`, and `loss3_original` have different update
  directions at the same `delta_k`.


## 2026-05-16 Angle Table Wording Simplified

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` to remove the ambiguous
  phrase `trajectory probe` from the angle-table interpretation.

Observed evidence summarized:

- The tables report angles, not loss-growth curves.
- The `loss1`, `loss2`, and `loss3` column labels identify the trajectory that
  supplied the saved point `delta_k` where an angle was computed.
- For the true nonlinear table, the angle itself is still between two `loss3`
  residual-field gradients.

Inference:

- For the current theory claim, the `loss3` trajectory column is the cleanest one
  to emphasize. The `loss1` and `loss2` columns are secondary context only.


## 2026-05-16 Angle Table Interpretation Corrected

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` with an interpretation
  correction for the angle tables.

Source files:

- `docs/angle_diagnostic_raw_tables_20260516.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Observed evidence summarized:

- The `loss1`, `loss2`, and `loss3` columns in the step-indexed angle tables name
  the trajectory that supplied `delta_k`.
- The true nonlinear endpoint-vs-movement angle compares two gradients of the
  `loss3` residual field, even when the evaluated point came from a `loss1` or
  `loss2` trajectory.
- The local-affine endpoint-vs-movement and bias-vs-movement tables use
  `A = J_f - J_j` and `b = f(x) - j(x)`, so they are also local `loss3` residual
  geometry diagnostics evaluated along different trajectories.

Inference:

- The `loss3` columns have the cleanest interpretation for the current theoretical
  question because both the trajectory and compared gradients concern `loss3`.
- The `loss1` and `loss2` columns should be described only as angles computed at
  points reached by the `loss1` or `loss2` trajectories, not as intrinsic
  gradient-angle diagnostics for the `loss1` or `loss2` objectives.
- The bias-vs-movement table is a decomposition check inside the clean-point
  affine model, not a finite-radius attack comparison.

Remaining work:

- Optional: if intrinsic `loss1` or `loss2` gradient-angle diagnostics are needed,
  define separate endpoint/movement pairs for those objectives and compute them
  explicitly.


## 2026-05-16 Angle Mean/Std Tables Presented In Chat

Status:

- No new numerical experiment was run.
- Presented the recorded step-indexed angle mean/std tables in the chat response.

Source files:

- `docs/angle_diagnostic_raw_tables_20260516.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Observed evidence summarized:

- Three step-indexed angle families were presented: true nonlinear endpoint-vs-movement,
  local-affine endpoint-vs-movement, and local-affine bias-vs-movement.
- Tables use `mean (std)` in degrees over five samples: indices `0, 7, 40, 47, 115`.
- Saved steps are `k = 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`.

Inference:

- This was a presentation/record clarification only; no new empirical result was
  introduced.


## 2026-05-16 Angle Count And Mean/Std Tables Clarified

Status:

- Added `Count Summary And Mean/Std Tables` to
  `docs/angle_diagnostic_raw_tables_20260516.md`.
- No new numerical experiment was run for this entry.

Observed evidence summarized:

- Step-indexed angle diagnostics have three angle families: local-affine
  endpoint-vs-movement, local-affine bias-vs-movement, and true nonlinear
  endpoint-vs-movement.
- Each step-indexed family has `11` saved steps and `3` attack trajectories, giving
  `33` mean/std entries per family and `99` mean/std entries total.
- Each mean/std entry is aggregated over five samples: indices `0, 7, 40, 47, 115`.
- The clean-point `A^T b` versus top singular vector angle is not step-indexed; it is
  a local candidate-direction comparison at the clean point.

Inference:

- The clean-point outward-growth/SVD angle should not be described as a per-step
  optimizer-direction angle. It would become step-indexed only in a separate
  experiment that recomputes `A`, `b`, and SVD at each `x + delta_k`.

Remaining work:

- Optional: run a separate along-trajectory SVD/outward-growth recomputation if a
  per-step `A(z_k)^T b(z_k)` versus top singular vector angle is needed.


## 2026-05-16 Every-5-Step Angle Raw Tables Recorded

Status:

- Reran `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py` with
  `--sample-ks 0 5 10 15 20 25 30 35 40 45 50` to include `k=0`.
- Created `docs/angle_diagnostic_raw_tables_20260516.md`.
- Added `Raw Every-5-Step Angle Tables` cross-reference to
  `docs/outward_growth_direction_result_20260515.md`.

Source files / inputs:

- `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`
- Trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Existing local-affine summary:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack.csv`
- `docs/angle_diagnostic_raw_tables_20260516.md`

Observed evidence summarized:

- The true nonlinear summary now has `34` lines: header plus `3` attacks times `11`
  saved k values.
- Saved k values are `0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`.
- Angle families recorded in the Markdown: local candidate-direction angles,
  local-affine endpoint-vs-movement, local-affine bias-vs-movement, and true
  nonlinear endpoint-vs-movement.
- `k=0` is a near-zero initialization point: mean L2 delta norm is about
  `8.0046e-06`, budget ratio about `1.0006e-06`.

Inference:

- The every-5-step raw tables make explicit that the clean-point local-affine angles
  and true nonlinear same-point angles differ substantially, especially for the
  `loss3_original_pgd` path.
- `k=0` should be treated as a near-zero diagnostic, not as a stable finite update
  step, because movement-style gradients can be degenerate at exactly zero radius.

Remaining work:

- Optional: plot all three trajectory angle families versus `k`.


## 2026-05-16 Nonlinear Gradient Run Key Takeaways Recorded

Status:

- Added `Additional Key Takeaways From The Nonlinear Gradient Run` to
  `docs/outward_growth_direction_result_20260515.md`.
- No new numerical experiment was run for this entry; it summarizes existing
  nonlinear-gradient diagnostic outputs.

Observed evidence summarized:

- Source CSV:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`.
- For `loss3_original_pgd`, true nonlinear endpoint-vs-movement gradient angle
  decreases from `45.06 deg` at `k=5` to `36.36 deg` at `k=10`, `22.33 deg` at
  `k=25`, and `8.15 deg` at `k=50`.
- The corresponding `loss3_original_pgd` delta budget ratios are `0.0359`,
  `0.0924`, `0.3649`, and `0.7637`.
- The local-affine same-delta angles for the same path are much smaller:
  `21.07 deg`, `11.28 deg`, `4.22 deg`, and `1.40 deg`.
- For `loss3_original_pgd`, endpoint loss and movement loss become closer over the
  path: endpoint/movement means are `0.3438/0.1243` at `k=5`, `2.1124/2.0150` at
  `k=25`, and `4.6171/4.5728` at `k=50`.
- Movement-to-endpoint gradient norm ratio on `loss3_original_pgd` decreases from
  `2.7067` at `k=5` to `1.0486` at `k=50`.

Inference:

- The nonlinear diagnostic supports the narrative that different losses have
  substantially different local gradient geometry, especially early in optimization.
- Clean-point local-affine gradients understate the true nonlinear angle gap when
  `delta_k` is finite.
- As the residual increment grows and begins to dominate the clean residual, endpoint
  and movement objectives become more aligned, but they remain conceptually distinct.

Remaining work:

- Optional: plot angle, budget ratio, endpoint/movement loss, and gradient-norm ratio
  against `k` for the three trajectories.


## 2026-05-16 Angle Diagnostic Validity Summary Recorded

Status:

- Added `Angle Diagnostic Summary: Which Comparisons Are Valid` to
  `docs/outward_growth_direction_result_20260515.md`.
- No new numerical experiment was run for this entry; it summarizes existing angle
  diagnostics and their valid interpretation.

Observed evidence summarized:

- Local candidate-direction angles: `outward_growth` versus `error` top-8 angle mean
  `84.68 deg`, and versus `error` rank-1 angle mean `90.87 deg`.
- Same-delta local-affine gradient angles comparing
  `A^T b + A^T A delta_k` versus `A^T A delta_k`: for `loss3_original_pgd`,
  `21.07 deg` at `k=5`, `11.28 deg` at `k=10`, `4.22 deg` at `k=25`, and
  `1.40 deg` at `k=50`.
- True nonlinear same-point gradient angles comparing
  `grad_z ||f(z_k)-j(z_k)||_2` versus
  `grad_z ||(f(z_k)-j(z_k))-(f(x)-j(x))||_2`: for `loss3_original_pgd`,
  `45.06 deg` at `k=5`, `36.36 deg` at `k=10`, `22.33 deg` at `k=25`, and
  `8.15 deg` at `k=50`.

Inference:

- The first angle set is valid only for comparing local candidate diagnostics, not
  final optimizer directions or same-iterate gradients.
- The second angle set is valid inside the clean-point local-affine Taylor model but
  can understate finite-delta nonlinear effects.
- The third angle set is the preferred finite-saved-point comparison because it uses
  actual nonlinear autograd at `z_k = x + delta_k`.
- The user's concern is correct: `A^T b + A^T A delta_k` is not wrong as a local
  approximation, but it should not be used as the final finite-radius geometry when
  `delta_k` is not tiny.

Remaining work:

- Optional: add plots comparing local-affine and true nonlinear angle curves over
  `k` for each trajectory.


## 2026-05-16 Analytic-Solution Hierarchy For Delta Objectives Recorded

Status:

- Created `docs/analytic_solution_hierarchy_for_delta_objectives_20260516.md`.
- Added a cross-reference section, `Analytic-Solution Hierarchy Note`, to
  `docs/outward_growth_direction_result_20260515.md`.
- No new numerical experiment was run for this entry.

Purpose:

- Record the analytic-solution distinction requested by the user:
  pure residual movement, local affine endpoint error, and true nonlinear
  finite-radius attack are different optimization levels.

Observed record:

- The dedicated Markdown records that
  `max_{||delta|| <= epsilon} ||A delta||^2` has the simple top-right singular
  vector solution of `A` in the L2 case.
- It records that
  `max_{||delta|| <= epsilon} ||b + A delta||^2` has a KKT / implicit
  trust-region characterization, with stationarity
  `A^T A delta + A^T b = mu delta`, but is not generally a simple singular-vector
  solution.
- It records that the true nonlinear attack
  `max_{||delta|| <= epsilon} ||f(x + delta) - j(x + delta)||` has no general
  closed-form analytic solution and should be studied through iterative
  optimization, actual trajectories, final deltas, and same-point nonlinear
  gradients.

Inference:

- This entry is a theory/documentation update, not new empirical evidence.
- The recorded hierarchy clarifies when `delta` can be treated as infinitesimal:
  only in local linearization diagnostics such as `e(x+delta) approx b + A delta`.
  Finite-radius attacks require direct nonlinear evaluation or optimization.

Remaining work:

- None for this documentation request.


## 2026-05-15 True Nonlinear Same-Point Endpoint-vs-Movement Gradient Diagnostic Completed

Status:

- Added `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`.
- Ran it on existing saved FNO `nu=0.001` GPU attack trajectories; no attack rerun
  was needed.
- Added `True Nonlinear Same-Point Gradient Comparison` to
  `docs/outward_growth_direction_result_20260515.md`.
- `adv_robust/bin/python -m py_compile tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`
  passed.

Purpose:

- Address the finite-`delta` limitation of the fixed-clean-Jacobian diagnostic.
- Instead of comparing `A^T b + A^T A delta_k` with `A^T A delta_k`, directly
  differentiate the actual nonlinear losses at each saved finite point
  `z_k = x + delta_k`.

Source files / inputs:

- Script: `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`
- Result root:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/`
- Trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Configs:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss1_original_pgd/config.json`

Output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack.csv`

Key settings:

- Indices: `0, 7, 40, 47, 115`.
- Attack trajectories: `loss1_original_pgd`, `loss2_original_pgd`,
  `loss3_original_pgd`.
- Saved steps summarized: originally `k = 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`; later rerun also includes `k = 0`.
- Compared gradients:
  `grad_z ||f(z_k)-j(z_k)||_2` versus
  `grad_z ||(f(z_k)-j(z_k))-(f(x)-j(x))||_2`.

Observed evidence summarized:

- For `loss1_original_pgd`, mean nonlinear gradient angle was `13.48 deg` at
  `k=5`, `1.48 deg` at `k=25`, and `0.55 deg` at `k=50`.
- For `loss2_original_pgd`, mean nonlinear gradient angle was `37.33 deg` at
  `k=5`, `1.67 deg` at `k=25`, and `0.82 deg` at `k=50`.
- For `loss3_original_pgd`, mean nonlinear gradient angle was `45.06 deg` at
  `k=5`, `36.36 deg` at `k=10`, `22.33 deg` at `k=25`, and `8.15 deg` at
  `k=50`.
- Mean `loss3_original_pgd` delta budget ratios at those same steps were
  `0.0359`, `0.0924`, `0.3649`, and `0.7637`, respectively.

Inference:

- The user's objection was correct: the fixed-clean-point `A=J_f(x)-J_j(x)`
  diagnostic is only local-affine and should not be treated as the final answer
  when `delta_k` is finite.
- Direct nonlinear autograd shows that endpoint-error and residual-movement
  gradients can differ substantially at the same finite point, especially along
  the `loss3_original_pgd` trajectory.
- The corrected layered interpretation is: local residual movement, local
  clean-error outward growth, true finite-point nonlinear endpoint gradient, and
  final finite-radius attack direction are distinct objects.

Remaining work:

- Optional: plot the nonlinear gradient-angle curves over `k`.
- Optional: compare these exact nonlinear gradients against the actual projected
  PGD update directions after L2 projection/clipping.


## 2026-05-15 Same-Delta Local Gradient-Angle Diagnostic Completed

Status:

- Added `tools/analyze_same_delta_local_gradient_angles.py`.
- Ran the script using existing saved FNO `nu=0.001` attack trajectories and
  existing explicit error Jacobians; no GPU attack rerun and no Jacobian
  recomputation were needed.
- Added `Same-Delta Gradient Diagnostic Results` to
  `docs/outward_growth_direction_result_20260515.md`.

Source files / inputs:

- `tools/analyze_same_delta_local_gradient_angles.py`
- Trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Error Jacobians:
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/error/*_jacobian_svd.npz`
- Clean residuals:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_residual.npy`

Output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_diagnostics.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Observed evidence summarized:

- The diagnostic compares, at the same saved `delta_k`,
  `A^T b + A^T A delta_k` versus `A^T A delta_k`.
- For `loss1_original_pgd`: at `k=5`, cosine `0.999347`, angle `1.95 deg`,
  `||A^T b|| / ||A^T A delta_k|| = 0.0347`; at `k=50`, cosine `0.999915`,
  angle `0.71 deg`, ratio `0.0125`.
- For `loss2_original_pgd`: at `k=5`, cosine `0.999345`, angle `1.97 deg`,
  ratio `0.0354`; at `k=50`, cosine `0.999926`, angle `0.67 deg`, ratio
  `0.0121`.
- For `loss3_original_pgd`: at `k=5`, cosine `0.930281`, angle `21.07 deg`,
  ratio `0.8561`; at `k=10`, angle `11.28 deg`, ratio `0.3738`; at `k=25`,
  angle `4.22 deg`, ratio `0.1175`; at `k=50`, cosine `0.999478`, angle
  `1.40 deg`, ratio `0.0321`.
- `adv_robust/bin/python -m py_compile tools/analyze_same_delta_local_gradient_angles.py`
  passed.

Inference:

- This is the apples-to-apples same-current-perturbation comparison requested by
  the user.
- Along `loss1` and `loss2` trajectories, the clean-residual term is already small
  relative to `A^T A delta_k` by `k=5`, so endpoint and movement local squared
  gradients are nearly aligned.
- Along the `loss3` trajectory, the clean-residual term matters strongly early
  because the perturbation is small; its influence decays as `delta_k` grows and
  `A^T A delta_k` dominates.
- The result supports the corrected layered interpretation: `A^T b` is important
  as a zero-radius / early-step term, not as a claim about a different final
  finite-radius optimizer direction.

Remaining work:

- Optional: visualize same-delta gradient angle and norm-ratio curves over `k`.


## 2026-05-15 Outward-Growth Angle Coverage / Rerun Need Inspected

Status:

- Inspected existing outward-growth angle records and added
  `Existing Angle Records And Whether A Rerun Is Needed` to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- Existing angle table:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`.
- Existing saved attack trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`.
- Existing Jacobian source:
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/error/*_jacobian_svd.npz`.
- `all_direction_similarity_table.csv` already records `dot`, `abs_dot`,
  `angle_deg`, `subspace_projection_l2`, `max_abs_dot`, and `mean_abs_dot`
  between `outward_growth` and the top singular directions/subspaces of
  `fno`, `solver`, and `error`.
- Observed summaries from that table: `outward_growth` vs `error` top-8 angle
  mean `84.68 deg`, abs-dot mean `0.246550`; vs `error` rank-1 angle mean
  `90.87 deg`, abs-dot mean `0.125291`; `error_top8_subspace` projection mean
  `0.925912`.

Inference:

- The original outward-growth run does not need to be rerun to recover existing
  candidate-direction angle data; those records already exist.
- The original run did not record same-`delta_k` gradient comparisons such as
  `cos(A^T b + A^T A delta_k, A^T A delta_k)`,
  `||A^T b|| / ||A^T A delta_k||`, or
  `cos(A^T b, A^T A delta_k)`.
- The precise apples-to-apples optimizer-step question needs a new post-processing
  diagnostic using saved attack trajectories and existing Jacobians, not a full
  rerun of the expensive attack/Jacobian experiment.

Remaining work:

- Implement and run the same-`delta_k` gradient-angle post-processing diagnostic
  if this comparison is needed for the paper narrative.


## 2026-05-15 Outward-Growth Correction: Not Apples-to-Apples Optimizer Comparison

Status:

- Added `Correction: This Is Not An Apples-To-Apples Optimizer Comparison` to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- The existing outward-growth experiment compares `error_top`, a top singular-vector
  direction of `A=J_f-J_j`, with `outward_growth`, the clean-point first-order
  direction `A^T b` for the local endpoint objective.
- Therefore it mixes two levels: the final/global direction of the pure local
  residual-movement quadratic and the zero-radius first-step direction of the
  endpoint objective.

Inference:

- The current experiment should be interpreted only as a local diagnostic showing
  that residual movement `||A v||` and zero-radius clean-residual outward growth
  `<b/||b||, A v>` are different quantities.
- It should not be described as a fair comparison of two final optimizer directions
  or two same-iterate gradient directions.
- Apples-to-apples follow-ups are: same-`delta_k` gradient comparison
  `A^T b + A^T A delta_k` versus `A^T A delta_k`; finite-radius local affine
  optimizer comparison for `||A delta||^2` versus `||b + A delta||^2`; and true
  nonlinear optimizer comparison from matched starts.

Remaining work:

- Optional: implement the same-`delta_k` gradient diagnostic using saved attack
  trajectories.


## 2026-05-15 Outward-Growth Number-Comparison Clarification

Status:

- Added `Are These Number Comparisons Valid?` to
  `docs/outward_growth_direction_result_20260515.md`.
- Rechecked the source CSVs to clarify the exact meaning of the headline numbers.

Observed evidence summarized:

- Source aggregate table:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/aggregate_direction_response_summary.csv`.
- Source per-direction table:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_response_table.csv`.
- `error_top` aggregate row has `n_rows=40`, i.e. 5 samples times top-8
  right singular directions of `A=J_f-J_j`; it is a top-8 group mean, not a
  rank-1-only mean.
- `outward_growth` aggregate row has `n_rows=5`, i.e. one `A^T b` direction per
  sample.
- Observed top-8 group means: `error_top` mismatch gain `0.412169`, outward
  component `0.0140571`; `outward_growth` mismatch gain `0.368053`, outward
  component `0.166514`.
- Rank-1-only check from `all_direction_response_table.csv`: `error` rank-1
  mismatch gain mean `0.868217`, outward component mean `-0.00677836`; `fno`
  rank-1 mismatch gain mean `0.674167`, outward component mean `0.00326582`;
  `solver` rank-1 mismatch gain mean `0.680574`, outward component mean
  `0.00304161`.

Inference:

- It is valid to compare `mismatch_gain` values with other `mismatch_gain`
  values and `outward_component` values with other `outward_component` values.
- It is not valid to infer large outward clean-error growth from large
  `mismatch_gain` alone.
- The rank-1-only check strengthens the interpretation: the strongest residual
  movement direction has even larger `||A v||` but does not push the current
  residual outward on average.

Remaining work:

- Optional same-`delta_k` gradient comparison remains separate.


## 2026-05-15 Outward-Growth Layered Theory/Data Interpretation Added

Status:

- Added `Layered Interpretation: What Is Actually Being Compared` to
  `docs/outward_growth_direction_result_20260515.md`.
- The section consolidates the theory, formulas, observed data, current
  experiment scope, and the distinction between local candidate directions,
  same-`delta` gradients, finite-radius local affine objectives, and true
  nonlinear attack trajectories.

Observed evidence summarized:

- Source result doc: `docs/outward_growth_direction_result_20260515.md`.
- Numeric sources remain the existing outward-growth outputs:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/aggregate_direction_response_summary.csv`
  and
  `forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_summary.csv`.
- Observed data recorded in the new section include:
  `error_top` mismatch gain mean `0.412169`, `error_top` outward component mean
  `0.0140571`, `outward_growth` mismatch gain mean `0.368053`,
  `outward_growth` outward component mean `0.166514`, `fno_top` outward
  component mean `0.00218766`, `solver_top` outward component mean
  `-0.00819025`, and finite-difference checks `0.167286` at `rho=1e-4` and
  `0.166668` at `rho=1e-3` versus predicted `0.166514`.

Inference:

- The current experiment is a local candidate-direction diagnostic: it compares
  the pure residual-movement/SVD direction with the clean-point outward-growth
  direction `A^T b`.
- It is not a same-`delta_k` optimizer-gradient comparison between
  `A^T b + A^T A delta_k` and `A^T A delta_k`, and it is not a final nonlinear
  `loss3_original` optimizer-direction result.
- The layered interpretation records the correct role of each object:
  `loss3_original` for finite-radius attack, `||b + A delta||` for local affine
  endpoint approximation, `A^T b` for clean-point outward growth,
  `||A delta||` and the top right singular vector of `A` for local residual
  movement, and same-iterate gradient cosines as a separate follow-up.

Remaining work:

- Optional follow-up: compute same-trajectory gradient diagnostics from saved
  attack iterates `delta_k`, including
  `cos(A^T b + A^T A delta_k, A^T A delta_k)`,
  `||A^T b|| / ||A^T A delta_k||`, and
  `cos(A^T b, A^T A delta_k)`.


## 2026-05-15 Outward-Growth Limitation: Not Same-Delta Gradient Comparison

Status:

- Added `Important Limitation: Not A Same-Delta Gradient Comparison` to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- Inspection of `tools/analyze_outward_growth_direction.py` confirms that
  `error_top` directions are loaded from saved right singular vectors of
  `A = J_f - J_j`.
- The same script constructs `outward_growth` from the clean residual direction,
  i.e. normalized `A^T b` / clean outward-growth direction.
- The script evaluates candidate directions by metrics such as `||A v||` and
  `<b/||b||, A v>`; it does not compute same-`delta_k` gradient comparisons
  between `A^T b + A^T A delta_k` and `A^T A delta_k`.

Inference:

- The outward-growth experiment compares local mechanism directions:
  top residual-movement/SVD direction versus clean-point first-order outward
  direction.
- It should not be described as a comparison of two per-step optimizer gradients
  along the same attack trajectory.
- A direct same-`delta` follow-up would compute cosines and norm ratios for
  `A^T b + A^T A delta_k` versus `A^T A delta_k` at saved attack iterates.

Remaining work:

- Optional follow-up: use saved attack iterates to compute same-`delta_k`
  gradient comparisons for the local squared endpoint and residual-movement
  objectives.


## 2026-05-15 Outward-Growth Direction Type / Analytic Solution Clarification

Status:

- Added a section to `docs/outward_growth_direction_result_20260515.md` clarifying
  that the outward-growth experiment compares local analytic directions, not final
  nonlinear attack trajectories.

Observed evidence summarized:

- The computed `error_top` direction is the top right singular direction of
  `A = J_f - J_j`, i.e. the analytic local L2 solution of
  `max_{||v||_2=1} ||A v||_2`.
- The computed `outward_growth` direction is the normalized `A^T b` direction,
  i.e. the analytic first-order L2 solution of maximizing
  `<b/||b||, A v>` at `delta=0`.

Inference:

- The experiment compares local mechanism directions: pure residual movement
  versus current-clean-residual outward growth.
- For the affine finite-radius local endpoint problem
  `max_{||delta||_2 <= epsilon} ||b + A delta||_2^2`, one can write a KKT/eigen
  characterization using `Q=A^T A` and `c=A^T b`, but the answer is generally
  neither exactly the top singular vector nor exactly `A^T b`.
- For the true nonlinear neural-operator attack objective, there is no general
  closed-form final optimizer direction; PGD/LP-steepest are iterative methods.

Remaining work:

- None for this clarification.


## 2026-05-15 Outward-Growth Interpretation: Bias Term Versus Residual Movement

Status:

- Added a clarification section to `docs/outward_growth_direction_result_20260515.md`
  explaining why the outward-growth result supports keeping the clean residual
  bias term in `loss3_original` rather than replacing the objective by pure
  residual movement.

Observed evidence summarized:

- From the outward-growth result table, `error_top` has mismatch gain mean
  `0.412169` but outward component mean only `0.0140571`.
- `outward_growth` has mismatch gain mean `0.368053` but outward component mean
  `0.166514`.
- The finite-difference check measured `loss3_original` growth `0.167286` at
  `rho=1e-4` and `0.166668` at `rho=1e-3`, matching predicted outward growth
  `0.166514`.

Inference:

- Locally, `loss3_original(delta) ~= ||b + A delta||`, while the residual
  movement objective is `||e(x+delta)-e(x)|| ~= ||A delta||`.
- The residual movement objective removes the clean residual `b`, so its SVD/top
  singular-vector direction can maximize `||A v||` without strongly increasing
  the current clean error norm.
- The result supports using `loss3_original` as the primary regression attack
  target and using residual movement/SVD directions as local diagnostics only.
- This remains a local first-order conclusion; finite-radius `loss3_original`
  still requires iterative nonlinear optimization.

Remaining work:

- None for this clarification.


## 2026-05-15 Delta/Loss Formula Taxonomy Across Markdown

Status:

- Created `docs/delta_loss_formula_taxonomy_20260515.md` to summarize the
  attack-related formulas, losses, objective variants, and meanings of `delta`
  across repository Markdown notes.

Source files / commands:

- Markdown source discovery used `rg --files -g '*.md'`.
- Formula/loss/delta evidence search used
  `rg -n --glob '*.md' '(delta|Delta|\\delta|\\Delta|epsilon|\\epsilon|loss1|loss2|loss3|increment_ratio|regularized|A\^T|A\\delta|finite-radius|local)'`.
- Main source notes recorded in the dedicated doc include
  `three_loss_objective_experiment_plan.md`,
  `docs/loss3_original_theory_experiment_plan.md`,
  `BATCH_LOSS_ONLY_OPTIMIZATION_METHODS.md`,
  `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md`,
  `LOSS1_ZERO_DELTA_GRADIENT_CHECK.md`,
  `LP_STEEPEST_DIRECTION_CHECK.md`,
  `docs/local_jacobian_svd_direction_taxonomy_20260515.md`,
  `docs/loss_gradient_direction_vs_svd_direction_20260515.md`,
  `docs/outward_growth_direction_result_20260515.md`, and the FNO
  `nu=0.001` loss-gradient path result docs.

Output files:

- `docs/delta_loss_formula_taxonomy_20260515.md`

Observed evidence summarized:

- The Markdown notes repeatedly define the three base losses
  `loss1`, `loss2`, and `loss3`, plus objective variants `original`,
  `increment_ratio`, and `regularized`.
- The notes define local Jacobian formulas such as
  `Delta f ~= J_f delta`, `Delta j ~= J_j delta`, and
  `e(x + delta) ~= b + (J_f - J_j) delta`.
- The notes also define finite-radius attack formulas such as
  `max_{||delta||_p <= epsilon} O(delta)`, PGD / LP-steepest updates,
  ray profiles, and boundary-rescaled final deltas.
- `git status --short` currently shows many deleted tracked generated artifacts
  under `benchmark_results/`, `fno_training_runs/`, `gradient_audit/`,
  `path_audit/`, and `results/`. The taxonomy does not interpret those deleted
  artifacts as current local evidence.

Inference:

- `delta` has two different roles in the notes: an infinitesimal/local diagnostic
  variable for Jacobians, SVD directions, residual increment ratios, and
  outward-growth derivatives; and a finite adversarial perturbation for PGD,
  LP-steepest PGD, final deltas, boundary rescaling, and real `loss3_original`
  attack evaluation.
- The practical rule recorded in the doc is: formulas involving clean-input
  Jacobians or `delta -> 0` are local; formulas involving `||delta|| <= epsilon`,
  attack iterates, final deltas, or ray endpoints are finite-radius and should
  not drop higher-order/path effects.

Remaining work:

- None for this summary note. If future Markdown files introduce new objectives
  or solver-gradient conventions, update `docs/delta_loss_formula_taxonomy_20260515.md`.


## 2026-05-15 Squared-Loss Gradient Clarification Added

Status:

- Added a clarification section to `docs/outward_growth_direction_result_20260515.md`
  explaining the exact squared local endpoint-error gradient.

Inference recorded:

- For residual movement, `M(delta)=||A delta||^2` has gradient
  `2 A^T A delta`.
- For squared endpoint error, `S(delta)=||b + A delta||^2` expands exactly as
  `||b||^2 + 2 b^T A delta + delta^T A^T A delta`, with exact local-model
  gradient `2 A^T b + 2 A^T A delta`.
- `A^T b` is the first-step / infinitesimal-radius gradient at `delta=0`, not a
  full finite-radius replacement for `loss3_original`.


## 2026-05-15 Outward-Growth Metric Glossary Added

Status:

- Added a glossary for the four key outward-growth numbers to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- `0.412169`: `error_top` mismatch gain mean, i.e. mean `||A v||_2`.
- `0.368053`: `outward_growth` mismatch gain mean, i.e. mean
  `||A v_growth||_2`.
- `0.0140571`: `error_top` outward component mean, i.e. mean
  `<b/||b||, A v>`.
- `0.166514`: `outward_growth` outward component mean, i.e. mean
  `<b/||b||, A v_growth>`.

Inference:

- The first pair compares raw residual movement; the second pair compares local
  growth of the current clean error norm.


## 2026-05-15 Outward-Growth Direction Interpretation

Status:

- Interpretation of the completed FNO `nu=0.001` outward-growth experiment was
  added to `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- `outward_growth` outward component mean: `0.166514`.
- `error_top` mismatch-gain mean: `0.412169`, but outward component mean:
  `0.0140571`.
- `fno_top` outward component mean: `0.00218766`; `solver_top` outward component
  mean: `-0.00819025`; `random_best_by_outward_component` outward component
  mean: `0.0114951`.
- Finite-difference actual `loss3` growth for `outward_growth`: `0.167286` at
  `rho=1e-4` and `0.166668` at `rho=1e-3`, versus linear prediction
  `0.166514`.

Inference:

- This experiment supports a local conceptual distinction, not a finite-radius
  optimality claim. It shows that maximizing residual movement `||A v||` and
  maximizing first-order outward growth of the current error norm are different
  diagnostics.
- The completed `A^T b` row explains local outward clean-risk growth; the final
  finite-radius adversarial objective remains `loss3_original` endpoint error.


## 2026-05-15 FNO nu=0.001 Outward-Growth Direction Completed

Status:

- Completed on GPU for FNO `nu=0.001`, indices `0 7 40 47 115`.
- Dedicated result note updated: `docs/outward_growth_direction_result_20260515.md`.

Observed output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/summary.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_response_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/aggregate_direction_response_summary.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_summary.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_model_output.npy`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_solver_output.npy`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_residual.npy`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/outward_growth_direction.npy`

Observed key metrics:

- From `aggregate_direction_response_summary.csv`: `outward_growth` outward
  component mean `0.166514`, mismatch-gain mean `0.368053`, and `D_f` mean
  `0.355948`.
- From `aggregate_direction_response_summary.csv`: `error_top` mismatch-gain
  mean `0.412169` but outward component mean only `0.0140571`.
- From `aggregate_direction_response_summary.csv`: `fno_top` outward component
  mean `0.00218766`; `solver_top` outward component mean `-0.00819025`;
  `random_best_by_outward_component` outward component mean `0.0114951`.
- From `finite_difference_growth_summary.csv`: for `outward_growth`, actual
  loss3 growth mean is `0.167286` at `rho=1e-4`, `0.166668` at `rho=1e-3`, and
  `0.167548` at `rho=1e-2`, versus predicted `0.166514`.
- From `finite_difference_growth_summary.csv`: for `negative_outward_growth`,
  actual loss3 growth mean is approximately the opposite sign, e.g. `-0.165528`
  at `rho=1e-4`, versus predicted `-0.166514`.
- From `manifest.json`: all five samples completed with nondegenerate outward
  directions; per-index clean residual norms are recorded in `per_index_metadata`.

Inference:

- The missing Experiment 2 `A^T b` / `v_growth*` row is completed for FNO
  `nu=0.001`.
- The result separates residual movement from outward clean-risk growth: the
  `error_top` directions move the residual field more, but `outward_growth` is
  the direction that most increases the current clean error norm at first order.
- Small-radius finite differences validate the local linear prediction for the
  outward-growth direction.

Remaining work:

- Use these tables in the Experiment 2 writeup.
- Do not rerun FNO `nu=0.01` or DeepONet/default-net unless a later paper
  question specifically requires the broader architecture comparison.


## 2026-05-15 FNO nu=0.001 Outward-Growth Run Status Check

Status:

- User-launched production command is running on GPU.
- No process was stopped or restarted during this check.

Observed evidence:

- Running process observed via `ps`: PID `192081`, command
  `adv_robust/bin/python tools/analyze_outward_growth_direction.py ... --device cuda`.
- Output root exists: `forensics/outward_growth_direction_20260515/fno_nu0p001/`.
- `index_000/manifest.json` exists and records
  `clean_residual_norm_l2 = 0.33239443448801387`,
  `At_b_unit_norm_l2 = 0.20492601962861828`,
  `error_jacobian_consistency_l2 = 1.8676534473603818e-08`, and
  `seconds = 19.056603444973007`.
- `index_000/finite_difference_growth_table.csv` had 85 lines, matching header
  plus 84 finite-difference rows for 28 directions x 3 radii.
- `index_007/finite_difference_growth_table.csv` also had 85 lines at the time
  of the check, indicating the second sample's finite-difference table had been
  written or nearly completed.

Inference:

- The startup CUDA/JAX messages are not fatal for this run; output is being
  produced on GPU.
- The production command is progressing past index 0.

Remaining work:

- Let the command finish through indices `40`, `47`, and `115`.
- After completion, inspect `manifest.json`, aggregate CSVs, `summary.md`, and
  `docs/outward_growth_direction_result_20260515.md`, then update this ledger
  with final observed metrics and conclusions.


## 2026-05-15 FNO nu=0.001 Outward-Growth Script Prepared

Source files added/updated:

- `tools/analyze_outward_growth_direction.py`
- `docs/outward_growth_direction_experiment_plan_20260515.md`

Status:

- Script prepared for the FNO `nu=0.001` outward-growth direction experiment.
- Production five-index experiment has not been run in this turn.
- GPU smoke tests were run only to validate the script path.

Observed from GPU smoke validation:

- Command path used `--device cuda` and the script prepended the virtualenv
  `ptxas` directory before constructing the JAX solver.
- Smoke output directories: `/tmp/outward_growth_smoke` and
  `/tmp/outward_growth_fd_smoke`.
- `/tmp/outward_growth_fd_smoke/manifest.json` records source paths for the FNO
  checkpoint, Burgers test set, and index-0 Jacobian/SVD files.
- Index 0 smoke values, observed from `/tmp/outward_growth_fd_smoke/manifest.json`:
  `clean_residual_norm_l2 = 0.33239443448801387`,
  `At_b_unit_norm_l2 = 0.20492601962861828`,
  `error_jacobian_consistency_l2 = 1.8676534473603818e-08`.
- `/tmp/outward_growth_fd_smoke/all_direction_response_table.csv` includes rows
  for `fno`, `solver`, `error`, `outward_growth`,
  `negative_outward_growth`, random directions, and random-best controls.
- `/tmp/outward_growth_fd_smoke/finite_difference_growth_table.csv` includes
  actual/predicted loss3 growth, residual movement, model movement, solver
  movement, and response cosine columns.

Inference:

- The script is ready to run the requested FNO `nu=0.001` five-index
  experiment on GPU.
- The `ptxas` workaround keeps the run on GPU; it only changes PATH so JAX does
  not use the incompatible system CUDA assembler.

Remaining work:

- Run the full command over indices `0 7 40 47 115`.
- Inspect `forensics/outward_growth_direction_20260515/fno_nu0p001/summary.md`,
  CSVs, JSON manifests, and `docs/outward_growth_direction_result_20260515.md`
  after completion.
- Update this ledger with final observed metrics and conclusions after the full
  run.


## 2026-05-15 Outward-Growth Direction Experiment Plan

Plan note added:

- `docs/outward_growth_direction_experiment_plan_20260515.md`

Status:

- Experiment plan only.
- No numerical experiment was run.

Observed evidence used to create the plan:

- `docs/loss3_original_theory_experiment_plan.md` defines the missing
  `v_growth*` row in Experiment 2.
- `docs/local_jacobian_svd_direction_taxonomy_20260515.md` records the local
  `loss3_original` decomposition into an outward term `A^T b` and quadratic
  gain term `A^T A delta`.
- Existing direction-comovement tables such as
  `forensics/fno_solver_jacobian_similarity_20260514/all_direction_comovement_table.csv`
  already include model, solver, error, and random direction rows.
- Existing reusable Jacobian/SVD sources include
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`,
  `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`, and
  `forensics/deeponet_solver_jacobian_similarity_20260515/`.

Inference:

- The missing Experiment 2 component should be a focused postprocess that loads
  existing Jacobians, computes `v_growth = normalize((J_f-J_j)^T b/||b||)`, and
  adds outward-growth rows to the same response table structure.
- This should not be treated as a finite-radius attack. It is a local
  first-order diagnostic separating residual movement `||A v||` from outward
  error-norm growth `<b/||b||, A v>`.

Remaining work:

- Implement `tools/analyze_outward_growth_direction.py` or equivalent.
- Run the FNO `nu=0.001` phase first, then FNO `nu=0.01` and DeepONet
  `nu=0.01`.
- Add `docs/outward_growth_direction_result_20260515.md` after running, with
  exact source paths, output tables, observed metrics, and conclusions.


## 2026-05-15 R2 Loss3 Original Plan Completion Audit

Audit note added:

- `docs/loss3_original_plan_r2_completion_audit_20260515.md`

Status:

- Remote R2 documentation/results audit completed.
- No numerical experiment was run.
- R2 credentials were used for read-only listing / small JSON reads and were not
  written to repository files.

Observed evidence from R2:

- R2 contains `docs/main_objective_mechanism_experiment1_result_20260514.md`,
  `results/main_objective_mechanism_summary_20260514/`, and
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`,
  supporting Experiment 1.
- R2 contains local Jacobian/SVD forensics including
  `forensics/fno_solver_jacobian_similarity_20260514/`,
  `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`,
  `forensics/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`,
  and
  `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/`,
  supporting most of Experiment 2.
- R2 contains `results/three_loss_objective_round1_l2_eps8_alpha0p3/`, with the
  27-run grid for three losses x three objective variants x three optimizers.
- R2 contains six `results/three_loss_batch100_full_loss3_delta_rerun_20260514_*_final_boundary/`
  directories and many `final_delta_summary.json`,
  `final_delta_diagnostics.csv`, and `final_delta_diagnostics.npz` files, so
  Experiment 5 Boundary-Rescaled Comparison is completed on R2.
- Selected R2 JSON values for
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`:
  `loss3_original_pgd` boundary `loss3_original_mean = 5.4525`;
  `loss3_increment_ratio_pgd` boundary `loss3_original_mean = 4.2225`;
  `loss3_regularized_pgd` final `||delta||_2` mean `0.3018`, boundary
  `loss3_original_mean = 2.0995`; `loss2_increment_ratio_pgd` boundary
  `loss3_original_mean = 3.1936`; `loss2_regularized_pgd` boundary
  `loss3_original_mean = 3.6531`.

Inference:

- The R2 evidence updates the prior local-only audit: Boundary-Rescaled
  Comparison was completed historically and exists on R2, even though the
  current local working tree lacks those result directories.
- The boundary-rescaled results support the conclusion that increment-ratio and
  regularized objectives can find interior or locally efficient directions, but
  scaling those directions to the full epsilon boundary does not necessarily
  match direct `loss3_original` endpoint optimization. This supports the
  nonlinear local-to-global interpretation.
- Still not observed as completed on R2 as specified: Ray Profile /
  Local-to-Global Profile, planned Small-Epsilon Sweep with `L_f`, `L_j`,
  `L_e`, `G_e`, and exact Direction Rotation Along Path via recomputed
  `v_e*(x_t)`.

Remaining work:

- Run Ray Profile curves.
- Run the planned local Small-Epsilon Sweep.
- Run exact path direction rotation using `v_e*(x_t)`.
- Optionally add `A^T b` outward-growth direction to the local response table.
- Optionally make a compact paper-ready table from the R2 boundary JSON files.


## 2026-05-15 Loss3 Original Plan Completion Audit

Audit note added:

- `docs/loss3_original_plan_completion_audit_20260515.md`

Status:

- Documentation/results audit completed.
- No numerical experiment was run.

Observed evidence:

- `docs/main_objective_mechanism_experiment1_result_20260514.md` documents
  Experiment 1: Main Objective Comparison for FNO/Burgers `nu=0.001`, batch
  100, `epsilon=8`, `alpha=0.3`, `steps=100`, with PGD, LP-steepest PGD, and
  generalized power. It records model movement, solver movement, mismatch,
  final true error, response cosine, `D_f`, and `D_sym`.
- Current local Jacobian/SVD outputs exist under paths including
  `forensics/fno_solver_jacobian_similarity_20260514/`,
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`, and
  `forensics/deeponet_solver_jacobian_similarity_20260515/`. These support the
  local response decomposition for top model, solver, error, and random
  directions, but not the explicit outward-growth `A^T b` direction.
- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md` and
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary.md`
  provide related path-gradient evidence, but not the planned recomputation of
  `v_e*(x_t)` along the path.
- `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md` documents scripts and final-boundary
  diagnostic design, and related scripts exist under `tools/`, but no current
  local `results/three_loss_batch100_full_loss3*` directories or
  `final_delta_summary.json` files were found during this audit.
- `docs/ray_profile_markdown_lookup_20260515.md` is only a lookup note for the
  Ray Profile section, not a Ray Profile result.
- `results/burgers_loss3_clean_recomputed_summary.md` contains older
  small-epsilon attack rows, but not the planned local small-epsilon sweep over
  `{1e-4, 1e-3, 1e-2, 1e-1}` with `L_f`, `L_j`, `L_e`, `G_e`, and direction
  cosine stability.

Inference:

- Completed or strongest documented evidence: Experiment 1 and most of
  Experiment 2's SVD/Jacobian response decomposition.
- Partial or related evidence: three-loss variant/boundary diagnostics in design
  form, older small-epsilon attack summaries, and loss-gradient path analysis.
- Not completed as specified: Small-Epsilon Sweep, Ray Profile,
  Boundary-Rescaled Comparison with current local numeric outputs, and
  Direction Rotation via recomputed `v_e*(x_t)`.

Remaining work:

- Restore/fetch or rerun missing three-loss boundary outputs.
- Run Ray Profile curves.
- Run Small-Epsilon Sweep with direction stability.
- Add outward-growth `A^T b` to the local response table.
- Run the full path direction-rotation experiment.

Working-tree note:

- `git status --short` showed many deleted tracked experiment artifacts under
  paths including `benchmark_results/`, `fno_training_runs/`,
  `gradient_audit/`, `path_audit/`, and `results/`. These deleted files were
  not interpreted as current local result evidence.


## 2026-05-15 English Translation of Loss3 Original Theory Plan

Files updated:

- `docs/loss3_original_theory_experiment_plan.md`
- `EXPERIMENT_LEDGER.md`

Status:

- Documentation translation completed.
- No numerical experiment was run.

Observed evidence:

- The source Markdown file existed locally at
  `docs/loss3_original_theory_experiment_plan.md` and contained the Chinese
  theory and experiment plan for `loss3_original` as the main regression attack
  objective.
- The translated file preserves the original structure: central goal, five-step
  proof route, nine-objective table, method-objective matching, six claims, six
  proposed experiments, interpretation of representative recorded results,
  recommended paper narrative, minimal experiment set, and one-sentence summary.

Inference:

- This change is a language/clarity update only. It does not add new empirical
  evidence and does not change the planned experimental conclusions.

Remaining work:

- If the plan is executed later, record the exact scripts, output directories,
  generated numeric tables, metrics, and conclusions in a separate result note.

Working-tree note:

- Before this translation, `git status --short docs/loss3_original_theory_experiment_plan.md EXPERIMENT_LEDGER.md`
  showed `EXPERIMENT_LEDGER.md` as modified and did not show the plan file as
  modified. This translation modifies the plan file and appends this ledger
  entry.

## 2026-05-15 Ray Profile Markdown Location Lookup

Lookup note added:

- `docs/ray_profile_markdown_lookup_20260515.md`

Observed from local Markdown search:

- Markdown search for whole-word `ray` / `ray profile` / `ray experiment`
  reported Ray-related Markdown hits in
  `docs/loss3_original_theory_experiment_plan.md`.
- The main hit is `Experiment 4: Ray Profile / Local-to-Global Profile` at
  `docs/loss3_original_theory_experiment_plan.md:741`.
- The compact experiment list also names `ray profile` at
  `docs/loss3_original_theory_experiment_plan.md:994`.
- `stat` reported
  `2026-05-15 15:04:45.684671694 +0000 docs/loss3_original_theory_experiment_plan.md`;
  the file was not observed locally as a May 14 file by filesystem mtime at
  lookup time.

Inference:

- The requested Markdown file is most likely
  `docs/loss3_original_theory_experiment_plan.md`.
- The Ray-profile experiment is a planned direction/radius diagnostic, not an
  observed numerical result from this lookup.

Remaining work:

- No experiment was run in this lookup.
- If executed later, record the script, output directory, numeric tables, and
  conclusions separately.

Working-tree note:

- `git status --short` showed many deleted tracked experiment artifacts under
  paths including `benchmark_results/`, `fno_training_runs/`,
  `gradient_audit/`, `path_audit/`, and `results/`. These were not interpreted
  as current local results for the Ray-profile plan.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Figures And Target-Loss Table

Added visual and tabular summaries for the saved 50-step attack trajectories.

Files added/updated:

- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_target_loss3_table_20260515.md`
- `docs/figures/fno_nu0p001_loss_gradient_path_dashboard_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_target_vs_loss3_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_angles_by_attack_20260515.png`
- `tools/plot_loss_gradient_path_figures.py`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/target_vs_loss3_by_attack_loss_and_k.csv`

What the new table shows:

- Each row is averaged over the five initial conditions.
- For the `loss1` path, the table reports the optimized `loss1` value and the
  same-delta `loss3` value.
- For the `loss2` path, the table reports the optimized `loss2` value and the
  same-delta `loss3` value.
- For the `loss3` path, the table reports direct `loss3`.
- Early and middle steps are affected by unequal budget use: `loss1`/`loss2`
  often use much more L2 budget than `loss3`, so their same-delta `loss3` can be
  larger early.
- At `k=50`, direct `loss3` is largest on average: `4.6171` vs `4.0678` on the
  `loss1` path and `3.6982` on the `loss2` path.

## 2026-05-15 Loss-Gradient Direction Interpretation Clarification

Clarification added to:

- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_detailed_data_20260515.md`

Interpretation:

- The three losses do not form three equally different update directions.
- `loss1` and `loss2` are nearly parallel in this FNO `nu=0.001` run.
- `loss3` is the distinct direction: roughly `56-57 deg` away overall from
  `loss1/loss2`.
- The `loss3` direction becomes more similar as `delta_k` grows, decreasing from
  about `75 deg` at `k=5` to about `47 deg` at `k=50`, but remains clearly
  different.
- Reported angles are averages of per-point angles, not angles of averaged
  gradients.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Detailed Data Table

Detailed data note added:

- `docs/fno_nu0p001_loss_gradient_path_detailed_data_20260515.md`

Contents:

- overall 150-point gradient-angle/loss/gradient-norm summary;
- aggregation by optimized attack loss;
- aggregation by saved step `k`;
- aggregation by attack loss and saved step;
- aggregation by initial-condition index and attack loss;
- final `k=50` per-index table;
- full raw 150-row per-point CSV embedded directly in the Markdown file;
- pointer to the source CSV at
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/per_point_loss_gradient_angles.csv`.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Path Result

Result note added:

- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`

Postprocess script added:

- `tools/analyze_loss_gradient_path_results.py`

Analyzed run:

- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631`
- 150 points: 5 initial conditions x 3 attack losses x 10 saved steps
  (`k=5,10,...,50`).

Key result:

- `grad loss1` and `grad loss2` are nearly aligned: mean cosine `0.9904`,
  mean angle `3.91 deg`.
- `grad loss3` is substantially different from both: mean angles about
  `56.5 deg` vs `loss1` and `57.2 deg` vs `loss2`.
- The `loss3` difference is strongest early (`~75 deg` at `k=5`) and decreases
  but remains large by `k=50` (`~47 deg`).
- At final `k=50`, direct `loss3` optimization gives the largest mean final
  `loss3` (`4.617`), but not uniformly for every individual index.

## 2026-05-15 Built-In Runtime Compatibility Patch

Code updated:

- `run_three_loss_objective_attack.py`

Behavior added:

- default runtime workaround setup before model/solver construction;
- virtualenv-local Triton/NVIDIA `ptxas` is moved to the front of `PATH` when found;
- cuDNN is disabled by default for this attack script while keeping CUDA/GPU enabled;
- command-line controls are available via `--no-runtime-workarounds`,
  `--no-disable-cudnn`, and `--no-prepend-env-ptxas`;
- resolved settings are saved in `config.json` as `runtime_ptxas_dir` and
  `runtime_cudnn_enabled`.

Verification:

- `python3 -m py_compile run_three_loss_objective_attack.py`
- 1-step GPU probe with plain `python run_three_loss_objective_attack.py ...`
  succeeded without the external wrapper and saved
  `/tmp/fno_path_probe_builtin_workarounds` in about 0.86 seconds.

## 2026-05-15 Vast.ai V100 Runtime Troubleshooting

Operational note added:

- `docs/vast_v100_cuda_jax_cudnn_troubleshooting_20260515.md`

Recorded issues:

- Vast auto-tmux login can exit with `no sessions` before any experiment starts.
- Vast may auto-activate `/venv/main`, while the usable repo environment is
  `adv_robust`.
- JAX/Exponax GPU compilation can fail if it finds the system CUDA 13 `ptxas`
  before the `adv_robust` CUDA 12.4 `ptxas`.
- `run_three_loss_objective_attack.py` can fail on this V100 during FNO
  backward through cuDNN; the Jacobian scripts avoided this because they already
  disable cuDNN internally.

Stable workaround for this instance:

- activate `adv_robust`;
- use the updated `run_three_loss_objective_attack.py`, which now applies the
  ptxas PATH and no-cuDNN workarounds by default;
- the older wrapper approach is only an emergency fallback for older checkouts.

## Critical Correction

The current user question is about the FNO vs DeepONet/default-net **local
Jacobian / SVD / frequency** experiment, not the FNO-vs-DeepONet attack-ratio
tables.

Do not present the attack-ratio tables as the answer to the local-Jacobian
question. A previously created ratio-focused note was removed because it did
not answer the user's question.

## 2026-05-15 Recovery Actions

Observed from Git/GitHub:

- Branch `vast-ai` is at commit `20b689a2c385a700aa8dea91ae1b069adc6d4a77`
  (`Add loss3 mechanism diagnostics`).
- The current branch and `origin/vast-ai` point to that commit after fetch.
- `docs/`, `tools/`, `loss_attack_common.py`, DeepONet runner scripts, and
  the FNO-vs-solver forensics directory were restored from `HEAD`.
- A git-history path search did not find committed paths named
  `tools/analyze_local_jacobian_fno_deeponet.py` or
  `forensics/local_jacobian_frequency_20260514/`.

Observed from the selected R2 artifact prefix:

- The prefix contains `docs/`, `tools/`, `results/`, `forensics/`,
  `deeponet_training_runs/`, `fno_training_runs/`, datasets, and other
  artifacts.
- The selected `forensics/` prefix contains
  `fno_solver_jacobian_similarity_20260514/`.
- Searches under the selected prefix did not find
  `forensics/local_jacobian_frequency_20260514/` or
  `tools/analyze_local_jacobian_fno_deeponet.py`.
- Only small, relevant artifacts were restored/downloaded: result summaries,
  FNO-vs-solver forensics, and DeepONet checkpoints/logs for `nu=0.001` and
  `nu=0.01`.

Observed from the local Python environment:

- System `python3` does not have a usable `torch`.
- The copied `adv_robust` virtual environment has `site-packages/torch`, but
  that directory is empty; importing it yields a namespace module without
  `torch.load`.
- Therefore no new local-Jacobian recomputation has been run in this recovered
  checkout yet.


## 2026-05-15 Git Forensics For Missing Local-Jacobian Script

Observed from `origin/vast-ai`:

- Current local branch `vast-ai` tracks `origin/vast-ai` at commit
  `20b689a2c385a700aa8dea91ae1b069adc6d4a77`.
- `origin/vast-ai:tools/` contains `analyze_main_objective_mechanism.py`,
  `analyze_fno_solver_jacobian_similarity.py`, and
  `summarize_main_objective_mechanism.py`, but not
  `analyze_local_jacobian_fno_deeponet.py`.
- `git log --all --name-status -- tools` shows that commit `20b689a` added
  only those three analysis/summarization files under `tools/`.
- `git grep` in commit `20b689a` finds only indirect references to the missing
  FNO/DeepONet local-Jacobian experiment: the docs mention it, the FNO-vs-solver
  config points to `forensics/local_jacobian_frequency_20260514/...`, and the
  FNO-vs-solver script imports the missing helper.
- `git fsck --full --no-reflogs --unreachable` produced no unreachable commits
  or blobs in this checkout.

Interpretation:

- In this recovered checkout, there is no evidence that the missing helper or
  original DeepONet SVD outputs were ever committed to the visible GitHub
  branch.
- The most likely explanations are: the file/result directory existed only as
  an untracked generated artifact on the previous machine, it lived under a
  different path/name that has not been found yet, or a local commit was made on
  the previous machine but was not pushed and was not included in this recovered
  `.git` object database.

## Experiment Status

| Experiment | Status | Current evidence | What not to claim |
| --- | --- | --- | --- |
| FNO vs DeepONet/default-net local Jacobian/SVD/frequency | Partially evidenced, original outputs missing | `docs/main_objective_mechanism_experiment1_result_20260514.md` says an existing FNO/DeepONet Jacobian experiment computed `J_f`; `forensics/fno_solver_jacobian_similarity_20260514/config.json` points to `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index`; `tools/analyze_fno_solver_jacobian_similarity.py` imports the missing helper script | Do not claim the exact DeepONet high/low-frequency singular-vector conclusion from current files |
| FNO vs solver local Jacobian/SVD/frequency | Available and recorded | `docs/fno_solver_jacobian_similarity_result_20260514.md`; `forensics/fno_solver_jacobian_similarity_20260514/` | Do not confuse this with FNO vs DeepONet |
| FNO vs DeepONet/default-net attack-ratio tables | Available as old ratio evidence | `results/burgers_loss3_clean_recomputed_summary.md`; `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.md`; `.csv` | Do not use this as the Jacobian/SVD answer |
| FNO vs solver tracking-discount / mechanism diagnostics | Available and recorded | `docs/main_objective_mechanism_experiment1_result_20260514.md` and referenced mechanism summary tables | Do not use this as DeepONet evidence |

## FNO vs DeepONet Local Jacobian/SVD Evidence

Observed evidence that the experiment existed:

- `docs/main_objective_mechanism_experiment1_result_20260514.md` explicitly
  refers to an existing FNO/DeepONet Jacobian experiment.
- `forensics/fno_solver_jacobian_similarity_20260514/config.json` reuses FNO
  outputs from
  `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index`.
- `tools/analyze_fno_solver_jacobian_similarity.py` imports
  `tools.analyze_local_jacobian_fno_deeponet`, so the follow-up script depended
  on a local helper that is absent now.

Observed missing artifacts:

- `tools/analyze_local_jacobian_fno_deeponet.py`
- `forensics/local_jacobian_frequency_20260514/`
- DeepONet per-index files such as `index_*/deeponet/*jacobian_svd.npz`
- DeepONet frequency tables such as `*_top_singular_vector_metrics.csv` or
  `*_frequency_gain_by_k.csv`

Current grounded conclusion:

- The FNO-vs-DeepONet local-Jacobian experiment almost certainly existed as a
  generated/local artifact on the previous machine.
- The exact DeepONet SVD/frequency side is not currently recovered.
- The FNO side is partially recoverable from the later FNO-vs-solver forensics:
  in the recorded samples `0, 7, 40, 47, 115`, the leading FNO right singular
  vectors are low-frequency dominated.

## Restored / Downloaded Small Artifacts

Restored from git `HEAD`:

- `docs/main_objective_mechanism_experiment1_result_20260514.md`
- `docs/fno_solver_jacobian_similarity_result_20260514.md`
- `tools/analyze_fno_solver_jacobian_similarity.py`
- `forensics/fno_solver_jacobian_similarity_20260514/`
- `results/burgers_loss3_clean_recomputed_summary.md`
- `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.md`
- `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.csv`

Selected R2 downloads:

- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0.001.pt`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/config.json`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/dataset_info.json`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/summary.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/config.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/dataset_info.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/output_transform_stats.npz`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/summary.json`


## 2026-05-15 Reconstructed Missing FNO/DeepONet Local-Jacobian Helper

Status: code reconstructed and committed; one DeepONet torch import bug was
fixed during the DeepONet-vs-solver run preparation.

Added:

- `tools/analyze_local_jacobian_fno_deeponet.py`

Purpose:

- restore the helper imported by `tools/analyze_fno_solver_jacobian_similarity.py`;
- provide standalone FNO-vs-DeepONet/default-net explicit local Jacobian/SVD
  analysis;
- save per-index artifacts under
  `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/`
  with the file layout expected by the FNO-vs-solver follow-up:
  `index_*/fno/fno_index*_jacobian_svd.npz` and
  `index_*/deeponet/deeponet_index*_jacobian_svd.npz`.

Implemented interfaces used by the FNO-vs-solver script:

- `load_sample`
- `make_model`
- `compute_explicit_jacobian`
- `analyze_jacobian`
- `frequency_gains`

Verification performed in the recovered checkout:

- `python3 -m py_compile tools/analyze_local_jacobian_fno_deeponet.py tools/analyze_fno_solver_jacobian_similarity.py`
- `python3 -c "import tools.analyze_local_jacobian_fno_deeponet as m; ..."`
- a small 8x8 smoke test for `analyze_jacobian`, confirming NPZ/CSV/JSON
  outputs are written.

Not yet run in this recovered checkout:

- full 1024 x 1024 FNO-vs-DeepONet recomputation, because the copied Python
  environment initially had an empty/broken `torch` package. The environment
  was later patched enough to run the DeepONet-vs-solver Jacobian experiment
  below, but the full FNO-vs-DeepONet recomputation remains separate.

## 2026-05-15 DeepONet vs Solver Local Jacobian/SVD/Frequency

Status: completed and recorded.

Purpose:

- Recreate the FNO-vs-solver local-Jacobian comparison for DeepONet/default-net.
- Use the same five initial conditions as the FNO-vs-solver run:
  `0, 7, 40, 47, 115`.
- Use the DeepONet model setting `nu=0.01` instead of the FNO-side `nu=0.001`.

Added / generated:

- `tools/analyze_deeponet_solver_jacobian_similarity.py`
- `forensics/deeponet_solver_jacobian_similarity_20260515/`
- `docs/deeponet_solver_jacobian_similarity_result_20260515.md`

Observed setting from the run config:

- DeepONet checkpoint:
  `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- DeepONet output transform:
  `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/output_transform_stats.npz`
- Test split:
  `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/..._test.pt`
- Solver: JAX/Exponax Burgers, `nx=1024`, `nu=0.01`, `t_final=1.0`,
  `dt=0.001`, `domain=2.0`, solver dtype `float64`.

Key observed results:

- DeepONet spectral norm mean: `7.202`; solver spectral norm mean: `1.373`;
  residual/error spectral norm mean: `7.100`.
- DeepONet top-1 right singular vectors are high-frequency:
  `hi128` mean `0.761`, zero crossings mean `507.2`.
- Solver top-1 right singular vectors are low-frequency/smooth:
  `hi128` mean `1.70e-17`, zero crossings mean `0.0`.
- Error top-1 right singular vectors track DeepONet:
  `hi128` mean `0.803`, zero crossings mean `516.0`.
- DeepONet-vs-solver top-k right subspaces are nearly orthogonal:
  mean principal cosine at `k=1` is `0.125`, and at `k=8` is `0.170`.
- DeepONet-vs-error leading subspaces are almost identical:
  mean principal cosine at `k=1` is `0.985`, and at `k=8` is `0.981`.

Grounded conclusion:

- For this `nu=0.01` DeepONet/default-net model, the dominant local model
  directions are high-frequency and jagged, while the physical solver's
  dominant local directions are smooth/low-frequency.
- The dominant residual/error Jacobian is essentially the DeepONet
  high-frequency response that the solver does not share.

## 2026-05-15 Top-4 Singular Vector Shape/Fourier Comparison

Status: completed and recorded.

Purpose:

- Directly visualize the top 4 right singular vectors rather than only their
  scalar frequency metrics.
- Compare FNO `nu=0.001`, solver `nu=0.001`, DeepONet `nu=0.01`, and solver
  `nu=0.01` on the same sample indices `0, 7, 40, 47, 115`.

Added / generated:

- `tools/plot_top_singular_vector_comparison.py`
- `forensics/top_singular_vector_comparison_20260515/`
- `docs/top_singular_vector_comparison_result_20260515.md`

Intermediate local-only raw SVD artifacts generated for plotting:

- `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/`
  for FNO top singular vectors.
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/` for the
  `nu=0.001` solver top singular vectors.

Grounded conclusion:

- FNO top right singular vectors and the `nu=0.001` solver top right singular
  vectors are smooth/low-frequency.
- DeepONet `nu=0.01` top right singular vectors are visibly high-frequency and
  jagged.
- The `nu=0.01` solver top right singular vectors remain smooth/low-frequency.
- Therefore the direct plots support the numeric conclusion: FNO's dominant
  local modes resemble solver modes, while DeepONet's dominant local modes are
  high-frequency modes that the solver does not share.

## 2026-05-15 Comprehensive SVD Shape/Fourier/Angle Diagnostics

Status: completed and recorded.

Purpose:

- Produce the dense plot set requested for comparing FNO, solver, DeepONet,
  and error Jacobian SVDs.
- Plot top-8 right singular vectors for each of the five initial-condition
  indices `0, 7, 40, 47, 115`.
- Plot both per-index figures and aggregate figures averaged across the five
  indices after Fourier transform.
- Compute cross-operator pairwise vector cosines, top-k principal angles,
  singular value spectra, and within-SVD orthogonality checks.

Added / generated:

- `tools/plot_comprehensive_svd_diagnostics.py`
- `forensics/comprehensive_svd_diagnostics_20260515/`
- `docs/comprehensive_svd_diagnostics_result_20260515.md`

Generated plot families:

- per-index six-row top-8 right-singular-vector line grids;
- per-index six-row top-8 Fourier-energy grids;
- per-index pairwise right-vector cosine heatmaps;
- per-index top-k principal-angle curves;
- per-index singular-value spectra;
- per-index within-SVD orthogonality-error heatmaps;
- aggregate mean shape, FFT, cosine-heatmap, principal-angle, singular-spectrum,
  and orthogonality figures.

Key observed results:

- FNO model-vs-solver leading right-singular subspaces are close:
  `k=1` mean principal angle `4.86 deg`, `k=8` mean principal angle `7.01 deg`.
- DeepONet model-vs-solver leading right-singular subspaces are far apart:
  `k=1` mean principal angle `82.83 deg`, `k=8` mean principal angle
  `80.19 deg`.
- DeepONet model-vs-error leading right-singular subspaces are close:
  `k=1` mean principal angle `10.01 deg`.
- FNO model top-1 right singular vector is low-frequency:
  mean `hi128` `2.23e-07`, mean zero crossings `13.6`.
- DeepONet model top-1 right singular vector is high-frequency:
  mean `hi128` `0.761`, mean zero crossings `507.2`.
- DeepONet error top-1 right singular vector is also high-frequency:
  mean `hi128` `0.803`, mean zero crossings `516.0`.

Orthogonality check:

- Within a single SVD, top-8 right singular vectors are mutually orthogonal up
  to numerical error. Max off-diagonal Gram errors are about `1e-09` to
  `3e-09`.
- This is expected from SVD and is mainly a sanity check. The scientifically
  useful comparisons are cross-operator vector/subspace alignments, especially
  model-vs-solver and model-vs-error principal angles.

Grounded conclusion:

- FNO's dominant local right-singular directions resemble the solver's
  low-frequency directions.
- DeepONet's dominant local right-singular directions are high-frequency and
  resemble the DeepONet-vs-solver error directions, not the solver directions.

No-std plot variant:

- `tools/plot_comprehensive_svd_diagnostics.py` now supports
  `--no-std-shading`.
- A second copy of the plot set was generated under
  `forensics/comprehensive_svd_diagnostics_20260515_no_std/` with the same
  samples and metrics but without standard-deviation shading in aggregate line
  plots.


## 2026-05-15 FNO nu=0.01 vs Solver and DeepONet nu=0.01 SVD Diagnostics

Status: completed and recorded.

Purpose:

- Repeat the local Jacobian/SVD analysis for FNO at matched `nu=0.01`.
- Compare FNO `nu=0.01`, solver `nu=0.01`, DeepONet/default-net `nu=0.01`, and
  their error Jacobians on the same five sample indices `0, 7, 40, 47, 115`.
- Regenerate the dense line/FFT/heatmap plot set without standard-deviation
  shading.

Recovered input:

- FNO `nu=0.01` checkpoint restored from R2 selected backup:
  `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`.
- The older `tmp_old_runner_inputs_b01/...nu0.01...pth` file was not present and
  was not tracked by Git because `*.pth` is ignored.

Added / generated:

- `tools/analyze_fno_solver_jacobian_similarity.py` now supports
  `--reuse-solver` and `--reuse-solver-root`.
- `tools/plot_fno_deeponet_nu0p01_svd_diagnostics.py`
- `docs/fno_nu0p01_solver_jacobian_status_20260515.md`
- `docs/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_result_20260515.md`
- `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`
- `forensics/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`

Key observed results:

- FNO `nu=0.01` model and solver are very tightly aligned locally:
  model-vs-solver `k=1` mean principal angle `2.32 deg`, `k=8` mean principal
  angle `1.82 deg`, and model-direction response cosine mean `0.9993`.
- FNO `nu=0.01` model top-1 right singular vector is extremely low-frequency:
  mean `hi128` `2.82e-09`, mean zero crossings `0.40`.
- FNO `nu=0.01` residual/error Jacobian is small in spectral norm:
  model top-1 sigma `1.379`, solver top-1 sigma `1.373`, error top-1 sigma
  `0.0373`.
- DeepONet/default-net `nu=0.01` remains high-frequency and solver-misaligned:
  model-vs-solver `k=1` mean principal angle `82.83 deg`, model top-1 `hi128`
  `0.7614`, and model top-1 zero crossings `507.20`.
- DeepONet/default-net error remains close to the model subspace:
  model-vs-error `k=1` mean principal angle `10.01 deg`; FNO model-vs-error
  `k=1` mean principal angle is `62.53 deg`.

Grounded conclusion:

- At matched `nu=0.01`, FNO behaves even more solver-like than in the earlier
  mixed-`nu` comparison. Its model and solver Jacobians have almost the same
  leading singular values and nearly the same leading right-singular subspace.
- DeepONet/default-net is still dominated by high-frequency, solver-misaligned
  local directions, so its error directions are close to the model directions.


## 2026-05-15 Nine-Row FNO/DeepONet Local SVD Comparison

Status: completed and recorded.

Purpose:

- Put FNO `nu=0.001`, FNO `nu=0.01`, and DeepONet/default-net `nu=0.01` into
  the same figure layout.
- Use nine rows: each family has `model`, `solver`, and `error` rows.
- Generate per-index and aggregate line/FFT/heatmap/spectrum/angle plots without
  standard-deviation shading.

Added / generated:

- `tools/plot_fno001_fno01_deeponet01_ninerow_svd_diagnostics.py`
- `docs/fno001_fno01_deeponet01_ninerow_svd_diagnostics_result_20260515.md`
- `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/`

Key observed results:

- FNO `nu=0.001` model-vs-solver remains aligned: `k=1` angle `4.86 deg`,
  `k=8` angle `7.01 deg`; model top-1 `hi128` `2.231e-07`.
- FNO `nu=0.01` is even more tightly solver-aligned: `k=1` angle `2.32 deg`,
  `k=8` angle `1.82 deg`; model top-1 `hi128` `2.82e-09`.
- DeepONet/default-net `nu=0.01` remains solver-misaligned and high-frequency:
  model-vs-solver `k=1` angle `82.83 deg`, `k=8` angle `80.19 deg`, model
  top-1 `hi128` `0.7614`, and model top-1 zero crossings `507.20`.
- DeepONet/default-net error remains close to the model subspace:
  model-vs-error `k=1` angle `10.01 deg`, while FNO error directions are much
  less aligned with the FNO model directions.

Grounded conclusion:

- The nine-row figures make the contrast visually direct: both FNO variants have
  solver-like dominant local modes, while DeepONet/default-net has dominant
  high-frequency local modes that align with its error rather than the solver.

## Next Actions

1. If the exact original FNO-vs-DeepONet/default-net artifacts from the old
   Vast.ai instance are still needed, recover that full artifact directory and
   compare it against the reconstructed results.
2. For every new experiment, add a dedicated result note under `docs/`, update
   this ledger, and commit the scripts plus lightweight CSV/PNG/Markdown
   outputs. Keep large raw `.npz` artifacts local unless explicitly requested.

## 2026-05-15 Nine-Row SVD Interpretation Summary

Created `docs/nine_row_fno001_fno01_deeponet01_svd_interpretation_20260515.md` and `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/aggregate_model_vs_model_subspace_angles.csv`.  The summary consolidates FNO `nu=0.001`, FNO `nu=0.01`, and DeepONet `nu=0.01` local Jacobian SVD diagnostics: singular values, model top-8 response, high-frequency/zero-crossing metrics, model-vs-solver/model-vs-error subspace angles, model-vs-model subspace angles, and top-8 orthogonality.  Main conclusion: FNO is locally solver-like and its error Jacobian is a small residual, while DeepONet is dominated by high-frequency model directions and its error Jacobian is almost the DeepONet Jacobian itself.

## 2026-05-15 Local Jacobian/SVD Experiment Purpose

Created `docs/local_jacobian_svd_experiment_purpose_20260515.md` to connect the local Jacobian/SVD experiments to the project-level `loss3_original` narrative.  The purpose is to show that regression robustness is co-variation with the solver, not invariance of the model output; locally this means the dangerous object is `J_model - J_solver`, not `J_model` alone.  The note records the logic chain: `loss1_original` measures model movement, `loss3_original` measures perturbed-input oracle-relative error, and the Jacobian/SVD experiments explain when these directions differ.  FNO is locally solver-like, so model-sensitive directions can be co-moving false alerts; DeepONet/default-net is high-frequency and solver-misaligned, so its model-sensitive directions are much closer to error directions.
## 2026-05-15 DeepONet Loss1-vs-Loss3 Jacobian/Attack Tension

Created `docs/deeponet_loss1_vs_loss3_jacobian_attack_tension_20260515.md` plus `forensics/deeponet_loss1_vs_loss3_attack_tension_20260515/`.  Existing clean-recomputed attack-ratio tables show that DeepONet `loss3` optimization usually produces much larger final true `loss3` than `loss1` optimization, despite the local SVD fact that `J_deeponet` and `J_error` dominant subspaces are close.  Aggregated DeepONet `nu=0.01` rows: L2 median `loss3/loss1` true-loss ratio `19.07` with `loss3` wins `624/702`; Linf median ratio `84.35` with `loss3` wins `799/902`.  Conclusion: local `J_error ~= J_model` suggests a smaller conceptual gap for DeepONet than for FNO, but it does not imply finite-radius `loss1_original` and `loss3_original` attacks are equivalent.  `loss3_original` still optimizes the moving-oracle endpoint error, clean-residual outward direction, nonlinear path, and projection dynamics.


## 2026-05-15 Local Jacobian/SVD Direction Taxonomy

Created `docs/local_jacobian_svd_direction_taxonomy_20260515.md` to record the distinction between SVD directions, instantaneous gradient directions, clean-residual outward-growth directions, and finite-radius attack directions.  The note uses the existing notation `b=e(x)=f(x)-j(x)` and `A=J_e=J_f-J_j`: local squared `loss3_original` expands as `||b+A delta||^2 = ||b||^2 + 2 b^T A delta + delta^T A^T A delta`, so its gradient has both `A^T b` and `A^T A delta` terms.  Therefore the top right singular vector of `A` explains the pure residual Lipschitz / mismatch gain direction, but the actual `loss3_original` optimizer can differ because of the clean residual, moving solver target, nonlinear path effects, and projection dynamics.

## 2026-05-15 Loss Gradient Direction vs SVD Direction

Created `docs/loss_gradient_direction_vs_svd_direction_20260515.md` to record the key clarification that a top singular-vector direction is a homogeneous quadratic gain optimum / power-iteration limit, not the same as a one-step optimizer gradient at an arbitrary current perturbation. Existing local Jacobian/SVD experiments computed `v_f`, `v_j`, and `v_e`, the top right singular directions of `J_f`, `J_j`, and `J_e=J_f-J_j`; these answer whether the local linear maps have similar dominant gain directions. They do not directly answer whether `loss1`, `loss2`, and `loss3` take the same one-step gradient direction. The missing explicit diagnostics are `g1(delta_k)=J_f^T J_f delta_k`, `g2(delta_k)=J_f^T b + J_f^T J_f delta_k`, and `g3(delta_k)=J_e^T b + J_e^T J_e delta_k`, especially the clean-point outward directions `g2(0)=J_f^T b` and `g3(0)=J_e^T b`.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Path Experiment Plan

Created `docs/fno_nu0p001_loss_gradient_path_experiment_plan_20260515.md`.  This is a plan only; the experiment has not been run yet.  The proposed diagnostic uses FNO/Burgers `nu=0.001`, `epsilon=8.0`, `alpha=0.3`, `steps=100`, and indices `0, 7, 40, 47, 115`.  It will save attack trajectories with `--save_trajectory --save_every 1`, then compare one-step gradient directions of `loss1`, `loss2`, and `loss3` along the attack path.  The clean-Jacobian diagnostics are `g1(k)=J_f^T J_f delta_k`, `g2(k)=J_f^T b + J_f^T J_f delta_k`, and `g3(k)=J_e^T b + J_e^T J_e delta_k`; exact autograd gradients should also be computed at selected steps to separate local-linear behavior from nonlinear path effects.  Main outputs should include gradient cosine curves, SVD-alignment curves, linear-vs-quadratic decomposition, frequency metrics, and selected-step line plots.


## 2026-05-15 FNO Loss-Gradient Plan Amendment

Updated `docs/fno_nu0p001_loss_gradient_path_experiment_plan_20260515.md` to explicitly include the fixed-radius `b`-aware affine optimum directions that were missing from the first plan draft: `v_loss2_out=normalize(J_f^T b)`, `v_loss3_out=normalize(J_e^T b)`, `v_loss2_rho=argmax ||b+rho J_f v||`, and `v_loss3_rho=argmax ||b+rho J_e v||`.  The plan now covers three distinct objects: SVD gain directions, one-step gradients at `delta_k`, and fixed-radius local affine optimum directions.  It also adds direction-source objective evaluation so candidate directions such as `v_f`, `v_j`, and `v_e` are evaluated under `loss1/loss2/loss3` without incorrectly calling them loss-specific directions.

## 2026-05-16 Neural Operator Robustness Research Directions

Status: planning note created; no numerical experiment was run for this entry.

Created `docs/neural_operator_robustness_research_directions_20260516.md` to
organize five future work directions:

- cross-framework solver/model combinations for 1D Burgers and 2D
  Stokes/Navier-Stokes;
- comparison of attack losses with unified true solver-level evaluation;
- perturbation-size-aware attack objectives;
- PGD versus power-iteration-like optimization methods;
- structure-aware error metrics beyond pointwise `L2`.

Observed evidence:

- This entry records only the Markdown planning document created from the
  user requested research directions.

Inference:

- The suggested execution order in the note prioritizes existing 1D
  Burgers/FNO loss experiments before broader cross-framework and
  structure-aware extensions.

Remaining work:

- Turn each direction into a concrete experiment plan before running numerical
  experiments.
- Record datasets, model checkpoints, solver versions, attack hyperparameters,
  tables, figures, and conclusions for each future run.

## 2026-05-16 Loss3 Experiment 3 Small-Epsilon Sweep - FNO nu=0.001 GPU Run

Status: completed on GPU. No CPU fallback was used for the official run.

User constraint recorded:

- Official neural-operator robustness experiments must use GPU. If CUDA, JAX
  GPU, or PyTorch GPU architecture support fails, stop and repair the
  environment rather than switching to CPU.

Environment repair and persistence:

- Added the GPU-only rule to `AGENTS.md`.
- Added `docs/gpu_only_experiment_policy_20260516.md`.
- Replaced the unsupported `torch==2.11.0+cu128` wheel with
  `torch==2.8.0+cu126` in `adv_robust`; the verified arch list includes
  `sm_70` for Tesla V100.
- Regenerated `requirements.txt` with the `cu126` PyTorch wheel index and
  `torch==2.8.0+cu126`, so future environment rebuilds do not reinstall the
  broken V100-incompatible wheel.
- Verified PyTorch CUDA matmul on `Tesla V100-SXM2-32GB`; verified JAX backend
  `gpu` and JAX GPU matmul; `pip check` reported no broken requirements.

Source files and inputs:

- Experiment script: `tools/run_loss3_small_epsilon_sweep.py`.
- Plan: `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_plan_20260516.md`.
- Result: `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`.
- Dataset: `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/test.pt`.
- Model checkpoint: `fno_training_runs/burgers_nu0p001_fno1d_500/burgers_1d/checkpoints/pytorch_fno1d_500.pt`.
- Local Jacobian/SVD artifacts: `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`.
- Outward-growth artifacts: `forensics/outward_growth_direction_20260515/fno_nu0p001/`.

Key settings:

- Scope: FNO / 1D Burgers / `nu=0.001` only.
- Sample indices: `0, 7, 40, 47, 115`.
- Epsilons: `1e-4, 1e-3, 1e-2, 1e-1`.
- Direction bank: top-8 right singular directions of `J_f`, `J_j`, and
  `J_e`, both signs; outward-growth direction; 128 seeded random controls.
- Evaluation batch size: `64`.
- Runtime evidence from `manifest.json`: device `cuda:0`, GPU
  `Tesla V100-SXM2-32GB`, required arch `sm_70`, PyTorch `2.8.0+cu126`,
  JAX backend `gpu`, policy `gpu_only_no_cpu_fallback`, runtime `20.40` s.

Output files:

- Output directory:
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`.
- Tables: `candidate_metrics.csv`, `best_by_objective_epsilon_index.csv`,
  `aggregate_best_by_objective_epsilon.csv`, `local_reference_by_index.csv`,
  `local_reference_summary.csv`, `direction_stability.csv`,
  `direction_stability_summary.csv`, `best_direction_source_counts.csv`.
- Manifest: `manifest.json`.
- Figures: `figures/small_epsilon_best_values.png`,
  `figures/small_epsilon_local_reference_ratio.png`,
  `figures/small_epsilon_direction_stability_heatmap.png`,
  `figures/small_epsilon_best_direction_sources.png`.

Observed key metrics from `aggregate_best_by_objective_epsilon.csv`:

- Local reference means: `L_f=3.895`, `L_j=4.095`, `L_e=0.8682`,
  `G_e=0.1665`.
- At `epsilon=1e-4`: `L_f=3.897`, `L_j=4.094`, `L_e=0.8685`,
  `G_e=0.1658`.
- At `epsilon=1e-1`: `L_f=3.856`, `L_j=4.036`, `L_e=0.8309`,
  `G_e=0.1763`.
- Best/local ratios stay near `1` for small epsilons. At `epsilon=1e-1`,
  ratios are approximately `L_f=0.990`, `L_j=0.985`, `L_e=0.961`,
  `G_e=1.058`.
- Best direction source counts: `L_f` selected FNO directions in `20/20`
  cases; `L_j` selected solver directions in `20/20`; `L_e` selected
  residual/error directions in `20/20`; `G_e` selected outward-growth in
  `20/20`.

Observed conclusion:

- The small-epsilon candidate sweep matches the clean local Jacobian references
  for `epsilon <= 1e-2`, supporting the claim that these ratio-style metrics
  represent local structure in this FNO `nu=0.001` setting.
- Model and solver sensitivities are much larger than residual sensitivity
  (`L_f` and `L_j` around `4`, `L_e` around `0.87`), consistent with FNO
  co-moving with the solver locally.
- `G_e` is much smaller than `L_e` (`0.166` versus `0.868` locally), confirming
  that outward clean-residual growth and residual-field movement are different
  diagnostics.
- Selected directions are stable up to sign across epsilon; `G_e` keeps the same
  outward-growth sign.

Inference:

- For FNO `nu=0.001`, Experiment 3 supports using very small epsilons as a
  local-structure diagnostic, but it should not be confused with a finite-radius
  attack objective. The local residual map can move at rate `L_e` without
  increasing the clean residual norm at the same rate, which is why `G_e` is
  the stricter outward-growth diagnostic.

Remaining work:

- If this result needs to be compared against PGD endpoint attacks, run a
  separate finite-radius attack experiment under the same GPU-only rule.
- Keep generated large arrays local/R2 unless explicitly asked to commit or
  upload them.

## 2026-05-16 CUDA/GPU Wheel Compatibility Documentation

Status: documentation update; no numerical experiment was run for this entry.

Created `docs/cuda_gpu_wheel_compatibility_notes_20260516.md` and linked it
from `AGENTS.md` plus `docs/gpu_only_experiment_policy_20260516.md`. The note
records why the previous V100 run failed with `torch==2.11.0+cu128`: the wheel
could see CUDA but did not include the V100 architecture `sm_70`, causing
`no kernel image is available for execution on the device`. It also records the
second compatibility issue observed during repair: `torch==2.6.0+cu126` made
PyTorch CUDA work on V100 but pulled cuDNN `9.5.1`, while JAX `0.10.0` needed
cuDNN `9.8.0` or newer.

Observed working environment at documentation time:

- GPU: Tesla V100-SXM2-32GB, compute capability `(7, 0)` / `sm_70`.
- PyTorch: `2.8.0+cu126`; `torch.cuda.get_arch_list()` includes `sm_70`.
- JAX backend: `gpu`; JAX devices include `CudaDevice(id=0)`.

Remaining rule:

- Before any official experiment, verify the active GPU architecture, PyTorch
  arch list, real PyTorch CUDA operation, JAX GPU backend, real JAX GPU
  operation, and `pip check`. Do not use CPU fallback for official runs.

## 2026-05-16 Loss3 Small-Epsilon Result Explanation Clarification

Status: documentation clarification; no new numerical experiment was run.

Updated `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
to make the experiment logic explicit. The clarified note now states that the
run is a candidate-bank direction maximization, not PGD and not model training.
It records what was optimized (`L_f`, `L_j`, `L_e`, and `G_e` over candidate
directions), how the 178-direction bank was constructed, the GPU-only run
conditions, the 3560 FNO/solver candidate evaluations, and the key
epsilon-refinement conclusion.

Observed clarification from existing CSV files:

- For `epsilon <= 1e-2`, best/local ratios remain near `1`, so the
  finite-epsilon ratios match the clean local Jacobian references.
- At `epsilon=0.1`, values begin to drift (`L_e` about `0.961` of local
  reference and `G_e` about `1.058`), indicating finite-radius effects.
- Best direction sources remain stable in `20/20` sample-epsilon cases for each
  objective: FNO for `L_f`, solver for `L_j`, residual/error for `L_e`, and
  outward-growth for `G_e`.

Inference now stated explicitly:

- This is analogous to an epsilon-refinement/local-convergence check: once the
  perturbation radius is small enough, shrinking it further does not materially
  change the measured local direction or local ratio.

## 2026-05-16 GPU-Aware adv_robust Setup Script

Status: environment automation added; no numerical experiment was run for this
entry.

Created `tools/setup_adv_robust_gpu_env.py` as the project entry point for
creating or verifying `adv_robust`. The script detects the visible GPU compute
capability with `nvidia-smi`, creates `adv_robust` when missing, installs the
pinned `requirements.txt`, and refuses to pass unless PyTorch and JAX both run
real GPU matrix operations. It also verifies that the PyTorch wheel contains the
required `sm_*` architecture for the active GPU, currently `sm_70` on V100, and
runs `pip check`.

Verified on the current machine with:

- command: `tools/setup_adv_robust_gpu_env.py --verify-only`;
- GPU: Tesla V100-SXM2-32GB;
- required architecture: `sm_70`;
- PyTorch: `2.8.0+cu126`;
- JAX backend: `gpu`;
- result: PyTorch CUDA matmul passed, JAX GPU matmul passed, and `pip check`
  reported no broken requirements.

Updated `AGENTS.md`, `docs/cuda_gpu_wheel_compatibility_notes_20260516.md`,
`ENVIRONMENT_SETUP_REPRODUCTION_GUIDE.md`, and `ENVIRONMENT_RUN_GUIDE.md` so
future environment rebuilds use this script rather than raw manual installation.

## 2026-05-16 Loss3 Gradient Direction Optimization - FNO nu=0.001 GPU Run

Status: completed on GPU with no CPU fallback.

Purpose:

- Address the limitation of the earlier candidate-bank sweep by directly
  optimizing perturbation directions with projected gradient ascent.
- Compare gradient-optimized directions against candidate-bank selections,
  pullback eigenvectors of `J_f^T J_f`, `J_j^T J_j`, `J_e^T J_e`, and the
  clean residual outward-growth direction.

Source files and inputs:

- Script: `tools/run_loss3_gradient_direction_optimization.py`.
- Result doc:
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`.
- Output directory:
  `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`.
- Previous candidate sweep used for comparison:
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`.
- Same FNO `nu=0.001`, dataset, checkpoint, Jacobian/SVD artifacts, and
  outward-growth artifacts as the previous small-epsilon experiment.

Key settings:

- Samples: `0, 7, 40, 47, 115`.
- Epsilons: `1e-4, 1e-3, 1e-2, 1e-1`.
- Objectives: `L_f`, `L_j`, `L_e`, `G_e`.
- Optimization: Adam ascent on unit direction `v`, projected/renormalized after
  each step.
- Starts per case: analytic plus, analytic minus, and one random start.
- Steps per start: `12`; learning rate: `0.2`.
- Runtime: `1117.48` seconds.
- GPU evidence from manifest: device `cuda:0`, `Tesla V100-SXM2-32GB`, required
  arch `sm_70`, PyTorch `2.8.0+cu126`, JAX backend `gpu`.

Output files:

- `gradient_run_summary.csv`
- `gradient_trajectory.csv`
- `gradient_best_by_case.csv`
- `gradient_summary_by_objective_epsilon.csv`
- `pullback_eigen_summary.csv`
- `manifest.json`
- figures: `gradient_vs_candidate_value_ratio.png`,
  `gradient_direction_alignment.png`, `random_start_alignment.png`

Observed results:

- Gradient-optimized values nearly match candidate-bank values: aggregate
  `grad/candidate` ratios are around `0.999-1.010`.
- `L_f` optimized directions align with the FNO pullback/SVD direction with mean
  absolute cosine about `0.996-0.999`.
- `L_e` optimized directions align with the residual/error pullback/SVD
  direction for `epsilon <= 1e-2` with mean absolute cosine about `0.998-0.999`;
  at `epsilon=0.1`, the mean is about `0.973`.
- `G_e` optimized directions align with the outward-growth direction with mean
  absolute cosine about `0.999` for `epsilon <= 1e-2` and about `0.991` at
  `epsilon=0.1`.
- `L_j` optimized values match the candidate values, but direction alignment is
  weaker than for the other objectives: mean absolute cosine is about `0.982`
  for small epsilons and about `0.969` at `epsilon=0.1`; the worst case is
  index `47`, `epsilon=0.1`, cosine about `0.849` with value still `0.999` of
  candidate maximum.
- Pullback eigenvector checks: for `L_f`, `L_j`, and `L_e`, the largest
  eigenvector of `J^T J` matches the saved top right singular vector with
  absolute cosine `1.0` for all checked samples.
- Outward-growth check: `normalize(J_e^T e(x)/||e(x)||)` matches the saved
  outward-growth direction with absolute cosine `1.0`; this is not a top
  eigenvector of `J_e^T J_e`.

Conclusion:

- The second version confirms that the previous candidate-bank directions were
  not arbitrary. For small epsilons, actual gradient ascent recovers essentially
  the same directions and values.
- The correct local linear algebra is: `L_f/L_j/L_e` use pullback matrices
  `J^T J`; `G_e` uses the clean residual outward gradient `J_e^T e/||e||`.
- At `epsilon=0.1`, finite-radius drift appears, so exact direction equality
  should not be overclaimed even when the scalar objective value remains near
  the candidate maximum.


## 2026-05-16 - Experiment 4 Ray Profile PGD100 Sign-Fixed Batch100 Cross-Check

Status: completed on GPU with no CPU fallback.

Purpose:

- Re-run the Ray Profile / Local-to-Global Profile under the user's requested normal protocol: no best-over-steps, no multi-restart, batch size 100, `epsilon=8`, zero initialization, ordinary PGD, `steps=100`, `alpha=0.3`.
- Resolve the discrepancy between the earlier Ray wrapper output and the historical batch-100 loss3 result.

Key correction:

- The first hand-written PGD branch in `tools/run_loss3_ray_profile_normal_batch.py` had a sign error: it differentiated `-objective` and then updated `delta += alpha * grad`, which performs descent for the target objective.
- The corrected PGD branch now differentiates the objective directly and updates `delta += alpha * grad`, matching `tools/run_batch_three_loss_loss_only.py`.
- The earlier non-fixed-sign PGD output is superseded and should not be used as a scientific result.

Runs and artifacts:

- Corrected Ray wrapper script: `tools/run_loss3_ray_profile_normal_batch.py`.
- Corrected result doc: `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`.
- Corrected output directory: `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Historical-script cross-check output directory: `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.

Corrected Ray result:

- Endpoint `loss3_original` mean at `r=8`: `loss3_original_final=5.4469`, `loss3_increment_ratio_final=4.2215`, `loss3_regularized_final=2.0679`, `local_residual_movement=4.1391`, `local_outward_growth=2.7204`.
- Endpoint winners among all profiled directions: `loss3_original_final` 58/100, `local_residual_movement` 31/100, `loss3_increment_ratio_final` 10/100, `loss3_regularized_final` 1/100.
- Endpoint winners among only the three finite-radius PGD attack objectives: `loss3_original_final` 81/100, `loss3_increment_ratio_final` 15/100, `loss3_regularized_final` 4/100.
- Small-radius clean norm-growth winner: `local_outward_growth` 100/100.
- Small-radius residual-increment winner: `local_residual_movement` 100/100.

Historical-script cross-check:

- `loss3_original_pgd`: final delta norm mean `7.7510`, boundary `loss3_original` mean `5.4469`.
- `loss3_increment_ratio_pgd`: final delta norm mean `7.8505`, boundary `loss3_original` mean `4.2215`.
- `loss3_regularized_pgd`: final delta norm mean `0.3043`; boundary-rescaled `loss3_original` mean `2.0275`.

Conclusion:

- The user's expectation is confirmed under the historical PGD protocol: direct `loss3_original` PGD gives the strongest endpoint mean loss3.
- The earlier contradictory PGD Ray result was an implementation bug, not evidence against the historical conclusion.
- The Ray experiment still supports the intended nonlinear story: the clean local directions win tiny-radius slope diagnostics, but direct finite-radius `loss3_original` PGD gives the strongest large-radius endpoint behavior on average.


## 2026-05-16 - Unified Evaluation-Metric Loss Curve Replots

Status: completed from existing saved trajectory data. No attack was rerun.

Purpose:

- Replot the historical 27-run three-loss attack setting with a fixed y-axis evaluation metric, instead of plotting each trajectory's own optimized objective.
- This answers questions such as: among all 27 optimized trajectories, how do they compare when every curve is evaluated by `loss3_original` or `loss3_increment_ratio`?

Data source:

- Restored from R2: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- This directory contains all 27 trajectories: 3 optimized losses x 3 objective variants x 3 optimization methods.
- Each trajectory already includes all 9 recorded evaluation metrics in `loss_stats.csv` and `loss_values.npz`.

New script:

- `tools/plot_batch_eval_metric_matrix_curves.py`.
- Layout: 3-by-3 matrix with rows for optimized loss (`loss1`, `loss2`, `loss3`), columns for objective variant (`original`, `increment_ratio`, `regularized`), and method curves inside each panel.

Generated plots:

- Batch mean/std figures: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves/png/`.
- Single-index figures for dataset index `0`: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves/index_png/`.
- All nine evaluation metrics were generated, including the key loss3 views: `loss3_original_all_27_mean_std.png`, `loss3_increment_ratio_all_27_mean_std.png`, `loss3_regularized_all_27_mean_std.png`, and corresponding `index0` plots.

Documentation:

- `docs/unified_eval_metric_loss_curve_plots_20260516.md`.


## 2026-05-16 - Adam vs Ordinary PGD Loss3 Original Parameter Control

Status: completed on GPU with no CPU fallback.

Purpose:

- Test whether Adam systematically gives smaller direct `loss3_original` attack values than ordinary PGD under two additional `(epsilon, alpha)` settings.
- Keep objective fixed to direct `loss3_original`, batch fixed to 100 samples, and steps fixed to 50.

Script and outputs:

- Script: `tools/compare_adam_pgd_loss3_original_params.py`.
- Result doc: `docs/adam_vs_pgd_loss3_original_param_control_20260516.md`.
- Output directory: `forensics/adam_vs_pgd_loss3_original_params_20260516/fno_nu0p001_batch100_steps50_eps4_eps12/`.

Settings and results:

- `epsilon=4.0`, `alpha=0.15`: boundary `loss3_original` mean was Adam `1.6712` vs PGD `2.3290`; paired PGD-Adam mean difference `+0.6578`, p-value `1.487e-09`.
- `epsilon=12.0`, `alpha=0.45`: boundary `loss3_original` mean was Adam `8.1319` vs PGD `7.2865`; paired PGD-Adam mean difference `-0.8454`, p-value `2.612e-02`.

Conclusion:

- The added settings do not support a universal claim that Adam always makes direct `loss3_original` smaller.
- At smaller radius (`epsilon=4`), PGD is significantly stronger at the boundary endpoint.
- At larger radius (`epsilon=12`) and fixed 50 steps, Adam is stronger; Adam reaches the boundary immediately, while PGD's final norm mean is about `8.66` before boundary rescaling.
- The robust conclusion is that Adam changes the constrained optimizer dynamics substantially; the relative ranking depends on epsilon, alpha, step count, and boundary reach.
