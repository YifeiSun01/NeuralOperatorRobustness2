# Experiment Ledger

## 2026-05-20 Loss3 P-Not-Q Visualization Directory Location Check

Status: inspected local filesystem paths for the stopped p!=q visualization outputs; no experiment, analysis, or plotting job was launched.

Observed evidence:

- Stopped p!=q raw sweep root exists: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.
- Stopped p!=q analysis root exists: `forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520/`.
- New stopped visualization roots exist:
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520/`
- Each listed visualization root contains `manifest.json` and `figures/loss3_q_mean_curves_with_boundary_markers.png`.
- Index Markdown exists: `docs/loss3_pneq_stopped_visual_index_20260520.md`.

Inference:

- The requested new p!=q visualization directories are present locally; `pinf_q1` remains partial, and no `pinf_q2` stopped visual exists because no completed roots were present when stopped.

## 2026-05-20 Loss3 P-Not-Q Stopped Post-Processing Visuals

Status: completed post-processing and visualization from existing completed p!=q roots after stopping the overnight run; no additional experiment roots were launched.

Source files:

- Stopped p!=q sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.
- Analysis script: `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`.
- Plot script: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.

Output files:

- Analysis root: `forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520/`.
- Analysis Markdown: `docs/loss3_alpha_epsilon_core4_pneq_q_stopped_100steps_result_20260520.md`.
- Visual index: `docs/loss3_pneq_stopped_visual_index_20260520.md`.
- Visual roots:
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520/`

Observed evidence:

- Analysis skipped the interrupted `p=inf,q=1, epsilon=2.0, alpha=0.2` root because its manifest was not completed.
- Completed-setting counts in generated visual manifests: `p=1,q=2` 20, `p=1,q=inf` 20, `p=2,q=1` 20, `p=2,q=inf` 20, and `p=inf,q=1` partial 2.
- `p=inf,q=2` was not plotted because no completed roots were present when the run was stopped.

Inference:

- The stopped p!=q visual set is suitable for the four full P/Q pairs above. The `p=inf,q=1` output is only a partial preview and should be labeled as such in any interpretation.

Remaining work:

- Do not compare `p=inf,q=1` as a full alpha/epsilon sweep unless the missing settings are intentionally run later.

## 2026-05-20 Loss3 P-Not-Q Run Stopped By User

Status: stopped active overnight p!=q experiment at the user's request; no further experiment roots should be launched for this p!=q sweep in this turn.

Observed process evidence before stop:

- Active overnight driver: `bash tools/run_loss3_overnight_20260520.sh`.
- Active sweep: `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 --steps 100 --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 ...`.
- Active setting runner before stop: `p=inf,q=1, epsilon=2.0, alpha=0.2`, output root `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps2_alpha0p2_batch100_steps100_pinf_q1`.

Observed artifact status before post-processing:

- `p=1,q=2`: 20 completed roots.
- `p=1,q=inf`: 20 completed roots.
- `p=2,q=1`: 20 completed roots.
- `p=2,q=inf`: 20 completed roots.
- `p=inf,q=1`: 2 completed roots and 1 interrupted/run-started root.
- `p=inf,q=2`: not started locally in this stopped run.

Action taken:

- Sent SIGTERM to process group `63104` and verified no `loss3_overnight`, `run_loss3_alpha_epsilon_core4_sweep`, or `run_loss3_direction_proposal_ablation` processes remained.

Remaining work:

- Run post-processing only on completed roots currently present under `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.

## 2026-05-20 Loss3 P-Not-Q Path/Visualization Status Check

Status: inspected existing/local p!=q artifacts and active overnight process; no new experiment, plotting, or analysis job was launched.

Observed evidence:

- Active overnight driver is still running: `bash tools/run_loss3_overnight_20260520.sh`.
- Active p!=q sweep command is running: `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 --steps 100 --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 ...`.
- Current new p!=q raw sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.
- Completion counts observed in the new p!=q sweep root: `p=1,q=2` 20 completed roots, `p=1,q=inf` 20 completed roots, `p=2,q=1` 20 completed roots, `p=2,q=inf` 18 completed roots plus 1 `run_started` root. Planned `p=inf,q=1` and `p=inf,q=2` pairs have not appeared yet.
- The overnight script plans to generate per-P/Q visual roots after the sweep/analysis finishes: `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_100steps_20260520/`, `..._p1_qinf_...`, `..._p2_q1_...`, `..._p2_qinf_...`, `..._pinf_q1_...`, and `..._pinf_q2_...`.
- Older completed P/Q comparison figures exist under `forensics/loss3_core_per_pq_four_figures_20260518/`, with per-P/Q folders such as `p1_q2`, `p1_qinf`, `p2_q1`, `p2_qinf`, `pinf_q1`, and `pinf_q2`.

Inference:

- The new 2026-05-20 alpha/epsilon p!=q sweep is not finished yet, so its new planned visual folders have not been generated locally yet.
- The currently available P/Q visual figures are the older 2026-05-18 per-P/Q figure set, not the new alpha/epsilon p!=q overnight visual set.

Remaining work:

- Wait for the active overnight p!=q sweep to finish; then the scripted analysis and per-P/Q visual folders should be generated automatically.

## 2026-05-20 Loss3 GPI Overall Conclusion Markdown

Status: created a dedicated Markdown synthesis note from existing experiment artifacts; no experiment, plotting job, or analysis job was launched.

Output file:

- `docs/loss3_gpi_overall_conclusion_20260520.md`

Observed evidence summarized:

- 300-step p2q2 summary table: `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`.
- 300-step p2q2 visual root: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`.
- GPI early-step comparison: `forensics/loss3_gpi_early_step_comparison_20260520/`.
- Baseline trajectory/GIF source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/`.

Inference recorded:

- GPI/replacement is a strong practical optimizer for the tested `loss3` settings because it reaches the boundary immediately, quickly converges to a final-like perturbation shape, and often obtains comparable perturbations/loss much faster than additive PGD-style methods.
- The conclusion is phrased as a speed/stability/Pareto advantage, not as unconditional final-loss dominance.

Remaining work:

- Extend early-step GPI comparisons beyond the saved baseline trajectory if a paper-level claim needs broader evidence.

## 2026-05-20 Loss3 GPI Overall Interpretation Check

Status: recorded synthesis from existing generated artifacts; no experiment, plotting job, or analysis job was launched.

Source evidence:

- 300-step p2q2 summary table: `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`.
- 300-step visual root: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`.
- GPI early-step comparison: `forensics/loss3_gpi_early_step_comparison_20260520/` and `docs/loss3_gpi_early_step_comparison_20260520.md`.
- GPI trajectory/GIF source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/`.

Observed evidence:

- Replacement/GPI reaches the p-norm boundary at step 1 in the p2q2 alpha/epsilon sweeps, while raw PGD and LP-steepest additive methods take multiple steps and can be much slower depending on alpha/epsilon.
- In the saved baseline early-step comparison, selected-sample mean `cos(delta_k, delta_300)` for GPI is `0.8868` at k=5, `0.9840` at k=10, and `0.9949` at k=20.
- Final perturbation shapes are often visually and cosine-wise similar across GPI, PGD, and LP-steepest additive methods for representative samples, though at least one saved sample is an outlier where GPI and PGD/LP final directions differ substantially.
- In 300-step p2q2 runs, GPI/replacement is not always the strict largest final-loss method; `steepest_add` or raw PGD can match or slightly exceed it in final mean loss for some alpha/epsilon settings.

Inference:

- The evidence supports describing GPI/replacement as a strong practical method for this loss3 setting: it rapidly saturates the perturbation budget, quickly converges to a stable perturbation shape, and often reaches comparable final perturbations much faster than additive PGD-style methods.
- The safer conclusion is speed/stability/Pareto advantage, not unconditional final-loss dominance. For final-loss-only evaluation after long horizons, additive LP-steepest can sometimes be competitive or better.

Remaining work:

- If this becomes a paper claim, phrase it as empirical evidence over the tested alpha/epsilon and p2q2 settings, and separately report final-loss winners, boundary-arrival speed, angular motion, and perturbation smoothness/similarity.

## 2026-05-20 Loss3 GIF Trace and Early-Step Artifact Location Reply

Status: verified existing artifact paths and clarified interpretation; no experiment or plotting job was launched.

Observed evidence:

- GIF trace manifest exists: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/trajectory_condition_gifs/manifest.json`.
- GPI early-step delta grid exists: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_delta_grid_steps_1_5_10_20_100_300.png`.

Inference:

- Boundary-ratio variance difference is primarily explained by normalized p-steepest directions for `steepest_add`/replacement methods versus unnormalized raw gradient scale for `raw_add`/PGD.
- Early-step visualization is available for the baseline GIF-trace run because it saved trajectory arrays; ordinary final-only artifacts cannot support the same full condition-panel reconstruction without saved trajectories.

Remaining work:

- Use the GIF trace directory and early-step comparison directory for visual inspection; rerun/save trajectory arrays for any missing exact settings that need step-wise condition panels.

## 2026-05-20 Loss3 GPI Early-Step Perturbation Comparison 18:41 UTC

Status: generated post-processing visualizations and tables from existing trajectory artifacts; no neural-operator experiment was launched.

Source files:

- Trajectory source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/steepest_replace/trajectory_samples.npz`.
- Cross-method final comparison sources: sibling `trajectory_samples.npz` files for `raw_add`, `steepest_add`, `raw_replace`, and `steepest_replace`.
- Plotting script added: `tools/plot_loss3_gpi_early_step_comparison.py`.

Output files:

- Manifest: `forensics/loss3_gpi_early_step_comparison_20260520/manifest.json`.
- Delta grid: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_delta_grid_steps_1_5_10_20_100_300.png`.
- Loss/cosine curve: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_loss_cosine_to_final_selected_samples.png`.
- Condition panels: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_condition_panels/`.
- Similarity table: `forensics/loss3_gpi_early_step_comparison_20260520/tables/gpi_early_step_similarity.csv`.
- Result Markdown: `docs/loss3_gpi_early_step_comparison_20260520.md`.

Key settings:

- Baseline setting `epsilon=4`, `alpha=0.4`, `p=2`, `q=2`, `steps=300`.
- Method visualized: `steepest_replace` (GPI/replacement).
- Saved dataset indices: `0`, `7`, `40`, `47`.
- Plotted steps: `1`, `5`, `10`, `20`, `100`, `300`; condition panels include `1`, `5`, `10`, sample-best step, and `300`.

Observed evidence:

- Mean selected-sample `cos(delta_k, delta_300)` for GPI is `0.8868` at k=5, `0.9840` at k=10, `0.9949` at k=20, and `1.0000` at k=300.
- Mean selected-sample GPI loss is `3.4935` at k=5, `3.6642` at k=10, `3.6791` at k=20, `3.5362` at k=100, and `3.6344` at k=300.
- Dataset 40 is nonmonotone: it has higher GPI loss around k=20 than at k=100 or k=300 while remaining high-cosine to final.
- Cross-method final-delta similarity is high for datasets 7, 40, and 47, but dataset 0 is an outlier with low cosine between GPI final and PGD/LP-steepest final deltas.

Inference:

- For the saved baseline trajectory samples, GPI perturbation shape is already close to its k=300 final shape by k=10. This supports testing a 5-step/10-step GPI early-stop variant, but the evidence is currently selected-sample trajectory evidence, not a full-batch proof.
- Early stopping should be judged with both shape similarity and loss stability because the loss can fluctuate after the shape has nearly converged.

Remaining work:

- If exact early-stop performance is needed for all 100 samples and all alpha/epsilon settings, rerun or extend the sweep to save full-batch early-step deltas or compute early-stop summary metrics directly.

## 2026-05-20 Loss3 Angle Triptych Y-Axis Fix 18:34 UTC

Status: modified and regenerated visualization artifacts only; no neural-operator experiment was launched.

Source files:

- Plotting script updated: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.
- 100-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/*/per_step_metrics.csv`.
- 300-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`.

Output files:

- 100-step refreshed visual root: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/`.
- 100-step angle-available triptychs: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/dynamics_triptychs_angle_available/`.
- 300-step refreshed visual root: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`.
- 300-step angle-available triptychs: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs_angle_available/`.
- Updated visual notes: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`, `docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md`.

Observed evidence:

- Existing angle means have maximum `70.3312` degrees in both inspected 100-step and 300-step p2q2 roots.
- Existing angle mean-plus-std values have maximum `81.7749` degrees in both inspected roots.
- The previous fixed `0..180` degree y-axis compressed the angle curves unnecessarily.
- Refreshed manifests were generated at `2026-05-20T18:32:34.324100+00:00` for the 100-step visual root and `2026-05-20T18:34:08.800642+00:00` for the 300-step visual root.

Inference:

- A data-driven angle y-axis is more faithful for these figures: it preserves the actual angle range and makes PGD/LP-steepest/GPI angular-motion differences readable without clipping the observed mean +/- std band.

Remaining work:

- Use the regenerated triptychs for angular-motion interpretation; older images with fixed 0..180 y-axis should not be used for judging relative angular speed.

## 2026-05-20 Loss3 Angular-Motion Interpretation 18:31 UTC

Status: recorded interpretation from existing generated angle-dynamics figures; no new neural-operator experiment was launched.

Source files:

- Angle dynamics figures: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs_angle_available/`.
- Result Markdown updated: `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`.

Observed evidence:

- User inspection of the generated angle dynamics figures found that replacement/GPI-style methods have much larger per-step `angle(delta_k, delta_{k-1})` than raw PGD and LP-steepest additive PGD.
- Raw PGD shows the slowest angular movement; LP-steepest additive PGD rotates faster than raw PGD but still decays as the perturbation approaches the boundary.

Inference:

- Replacement/GPI should not be framed as a classical power-iteration optimizer with global guarantees for this nonquadratic neural loss.
- A better interpretation is geometric: replacement/GPI repeatedly solves a local linearized full-budget boundary-direction problem, avoiding the angular inertia of additive updates. This can explain why it reaches strong loss values quickly even when the objective is nonquadratic.
- The empirical claim should be: replacement/GPI is an aggressive boundary-direction optimizer with fast angular motion; its perturbation quality still needs to be checked with spectral/smoothness/shape metrics.

Remaining work:

- Compare angular-motion curves against high-frequency energy, derivative/TV metrics, and GIF trajectories to ensure the fast rotation does not correspond to transient spike-like perturbations.

## 2026-05-20 Loss3 Angle-Available Dynamics Triptych Fix 18:27 UTC

Status: modified and regenerated visualization artifacts only; no neural-operator experiment was launched.

Source files:

- Plotting script updated: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.
- 100-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/*/per_step_metrics.csv`.
- 300-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`.

Output files:

- 100-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`.
- 100-step angle-available triptych directory: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/dynamics_triptychs_angle_available/`.
- 100-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`.
- 300-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/manifest.json`.
- 300-step angle-available triptych directory: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs_angle_available/`.
- 300-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md`.

Observed evidence:

- The 100-step p2q2 sweep has angular metrics in 9 of 20 completed roots. The missing 11 roots are older completed settings produced before `delta_prev_angle_degrees_mean` was added.
- The explicit 100-step representative triptych settings (`epsilon,alpha` = `4,0.4`, `8,0.3`, `8,1.6`, `16,1.6`) are all among the older roots without angular metrics, so their angular panels cannot be reconstructed from local files.
- The refreshed 100-step manifest generated at `2026-05-20T18:26:38.752656+00:00` now lists 9 `angle_available_triptychs`.
- The 300-step p2q2 sweep has angular metrics in 20 of 20 completed roots. The refreshed 300-step manifest generated at `2026-05-20T18:27:35.916818+00:00` now lists 20 `angle_available_triptychs`.

Inference:

- The previous visual output hid usable angular-motion plots because only explicitly requested representative triptychs were generated, and those happened to be old no-angle settings. The new output separately plots every setting that actually has `delta_prev_angle_degrees_mean`.
- Old no-angle settings still require rerunning the experiment if angular motion is needed for those exact epsilon/alpha combinations.

Remaining work:

- Use `figures/dynamics_triptychs_angle_available/` for angular-motion review. Rerun old 100-step settings only if the exact missing epsilon/alpha angle trajectories are required.

## 2026-05-20 Loss3 Boundary-Ratio Std Interpretation 18:24 UTC

Status: inspected existing 300-step p2q2 per-step metrics and updated the dedicated result Markdown; no new neural-operator experiment was launched.

Source files:

- `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`
- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`

Observed evidence:

- For `epsilon=8, alpha=0.3`, max/mean `boundary_ratio_std` values are `raw_add=0.288/0.0588`, `steepest_add=0.0438/0.00211`, and `raw_replace=steepest_replace=~3e-08/~3e-08`.
- For `epsilon=8, alpha=0.3`, per-sample 99% boundary steps are `raw_add` mean/median/max `73.47/80/126`, `steepest_add` `29.44/29/34`, and replacement methods `1/1/1`.
- At `epsilon=8, alpha=0.3`, step 20 has `raw_add` `boundary_ratio_mean=0.310`, `boundary_ratio_std=0.194`; `steepest_add` has `boundary_ratio_mean=0.700`, `boundary_ratio_std=0.026`.
- At `epsilon=8, alpha=0.3`, step 1 has `steepest_add` `delta_step_pnorm_mean=0.3` and `delta_step_pnorm_std=1.5e-08`, consistent with the normalized p-steepest step rule.

Inference:

- The large `raw_add` boundary-ratio std is caused by sample-dependent raw-gradient scale and radial alignment, which make samples reach the epsilon boundary at very different steps.
- The smaller `steepest_add` boundary-ratio std is expected because the p-steepest direction is normalized before applying alpha, making radial budget usage much more synchronized across samples.
- Replacement/GPI boundary-ratio std is essentially floating-point noise because the replacement update enforces boundary norm from step 1 onward.

Remaining work:

- Use angular-change and post-boundary loss-gain diagnostics to study boundary movement; `boundary_ratio_std` only measures radial synchronization.

## 2026-05-20 Loss3 300-Step P2Q2 Interpretation Check 18:20 UTC

Status: inspected existing 300-step analysis tables and updated the dedicated result Markdown; no new neural-operator experiment was launched.

Source files:

- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`

Observed evidence:

- For `epsilon=2, alpha=0.4`, final mean losses are `raw_add=1.4420`, `steepest_add=1.4552`, and `raw_replace=steepest_replace=1.3651`; mean 99% boundary-hit steps are `25`, `6`, and `1` respectively.
- For `epsilon=2, alpha=0.8`, final mean losses are `raw_add=1.4551`, `steepest_add=1.4440`, and `raw_replace=steepest_replace=1.3651`; mean 99% boundary-hit steps are `13`, `3`, and `1` respectively.
- For large-epsilon examples, replacement/GPI has lower final-loss std than add methods, e.g. `epsilon=8, alpha=0.3`: replacement/GPI std `1.4882` versus `raw_add=2.4005` and `steepest_add=2.0900`; `epsilon=16, alpha=1.6`: replacement/GPI std `2.2430` versus `raw_add=4.1427` and `steepest_add=3.3813`.
- Across 20 p2q2 300-step settings, strict largest-final-mean winners are `steepest_add` in 16 settings and `raw_add` in 1 setting; replacement/GPI tie for largest final mean in 3 settings.

Inference:

- The observed 300-step evidence supports a Pareto-style conclusion: replacement/GPI is consistently fastest to the boundary and lower variance, while longer-horizon `steepest_add` can slightly exceed it in final mean loss for many settings.

Remaining work:

- Use both early-time performance and final 300-step loss when writing the final comparison; avoid claiming that GPI is always the final-loss winner.

## 2026-05-20 Loss3 Boundary-Ratio Std Visualization Fix 18:16 UTC

Status: modified and regenerated visualization artifacts only; no neural-operator experiment was launched by this fix.

Source files:

- Plotting script updated: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.
- Source 100-step sweep data: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/*/per_step_metrics.csv`.
- Source 300-step sweep data: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`.

Output files:

- 100-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`.
- 100-step added std figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/boundary_ratio_std_curves.png`.
- 100-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`.
- 300-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/manifest.json`.
- 300-step added std figure: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/boundary_ratio_std_curves.png`.
- 300-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md`.

Key settings:

- Plotted `p=2,q=2`, 20 alpha/epsilon settings, core four methods.
- The original mean boundary-ratio plot still uses true mean +/- sample std shading for every method; the new figure directly plots `boundary_ratio_std` by method and alpha/epsilon panel.

Observed evidence:

- 100-step visual manifest regenerated at `2026-05-20T18:15:51.486994+00:00` with `completed_setting_count=20` and a `boundary_std_curves` output path.
- 300-step visual manifest regenerated at `2026-05-20T18:16:42.069367+00:00` with `completed_setting_count=20` and a `boundary_std_curves` output path.
- Aggregating `boundary_ratio_std` over all 20 p2q2 100-step roots gives max/mean values: `raw_add` `0.2954/0.0751`, `raw_replace` `3.58e-08/2.86e-08`, `steepest_add` `0.0898/0.00340`, `steepest_replace` `3.58e-08/2.86e-08`.
- Aggregating `boundary_ratio_std` over all 20 p2q2 300-step roots gives max/mean values: `raw_add` `0.2954/0.0264`, `raw_replace` `3.64e-08/2.90e-08`, `steepest_add` `0.0898/0.00114`, `steepest_replace` `3.64e-08/2.90e-08`.

Inference:

- The apparent missing std shading on replacement methods is a visualization-scale issue, not missing computation: their boundary-ratio sample std is nearly zero and the true band collapses onto the mean line.
- The separate std curve figure makes this visible without artificially inflating the uncertainty band.

Remaining work:

- Use the new `boundary_ratio_std_curves.png` figures when checking whether a method has genuinely small across-sample variability versus visually hidden mean-curve shading.

## 2026-05-20 Loss3 300-Step Figure Location Check

Status: inspected generated 300-step visualization artifacts; no new experiment was launched by this check.

Observed evidence:

- 300-step visualization manifest exists and reports status `completed`, generated at `2026-05-20T10:33:34.196576+00:00`.
- Manifest source sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`.
- Manifest reports `completed_setting_count=20` and `pq_pairs_plotted=[{p: 2, q: 2}]`.
- 300-step loss figure exists: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`.
- Boundary-ratio figure exists: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/boundary_ratio_mean_curves.png`.
- Delta angular-speed figure exists: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/delta_prev_angle_degrees_mean_curves.png`.
- Dynamics triptychs exist for representative settings under `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs/`.

Inference:

- The requested 300-step loss visualization has been generated. It is scoped to `p=2,q=2` and the 20 alpha/epsilon settings.

## 2026-05-20 Loss3 Overnight Run Status Check 17:54 UTC

Status: inspected active background run; no new experiment was launched by this check.

Observed process evidence:

- Overnight driver is still running: PID `63104`, command `bash tools/run_loss3_overnight_20260520.sh`, elapsed about `15:45:48` at inspection.
- Current active stage is stage 4: strict off-diagonal `p != q` 100-step sweep, command `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 --steps 100 --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 ...`.
- Current active setting runner: PID `131603`, output root `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha0p2_batch100_steps100_p2_q1`.
- Current setting is `epsilon=8.0`, `alpha=0.2`, `p=2`, `q=1`, `steps=100`; current log has printed `[run] raw_add`.

Observed GPU evidence:

- `nvidia-smi` showed Tesla V100-SXM2-32GB, memory `9520 MiB / 32768 MiB`, GPU utilization `97%` at inspection.

Observed artifact/log evidence:

- Stage 1 `p=2,q=2` 100-step sweep/analysis/plots completed: 20 completed roots, plus one old `interrupted_not_for_analysis` root not included in analysis.
- Stage 2 `p=2,q=2` 300-step sweep completed: 20 completed roots.
- Stage 3 baseline GIF trace completed: baseline run manifest and trajectory GIF manifest are completed.
- Stage 4 strict `p != q` sweep has 50 completed roots and 1 running root out of 120 planned settings.
- Stage 4 progress by observed P/Q pair: `p=1,q=2` completed 20/20, `p=1,q=inf` completed 20/20, `p=2,q=1` completed 10/20 with the 11th running.
- Stages not yet reached in stage 4: remaining `p=2,q=1` settings, then `p=2,q=inf`, `p=inf,q=1`, and `p=inf,q=2`.
- Current logs: `logs/loss3_overnight_20260520T020904Z.log` and `logs/loss3_alpha_epsilon_core4_eps8_alpha0p2_p2_q1.log`.

Inference:

- The workflow has not finished. It is in the final major sweep stage, but the largest stage is still in progress.
- By planned setting count, completed settings are approximately 92 out of 160 experiment roots if counting stage 1/2/4 sweeps plus the baseline root; stage 4 itself is about 50/120 completed, with one active.

Remaining work:

- Continue monitoring until stage 4 completes and the automatic stage-4 analysis/visualization/similarity/post-boundary diagnostics finish.

## 2026-05-20 Loss3 Overnight Run Status Check 04:21 UTC

Status: inspected active background run; no new experiment was launched by this check.

Observed process evidence:

- Overnight driver is still running: PID `63104`, command `bash tools/run_loss3_overnight_20260520.sh`, elapsed about `02:12:24` at inspection.
- Current active stage is stage 2: `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 --steps 300 --p 2 --q 2 ...`.
- Current active setting runner: PID `74234`, output root `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/fno_nu0p001_eps2_alpha0p2_batch100_steps300_p2_q2`.
- Current setting is `epsilon=2.0`, `alpha=0.2`, `p=2`, `q=2`, `steps=300`.
- Within the current setting, `raw_add`, `raw_replace`, and `steepest_add` method summaries exist; `steepest_replace` summary is not yet present, so the setting is likely running the fourth method.

Observed GPU evidence:

- `nvidia-smi` showed Tesla V100-SXM2-32GB, memory `9522 MiB / 32768 MiB`, GPU utilization `95%` at inspection.

Observed artifact/log evidence:

- Stage 1 completed: `forensics/loss3_alpha_epsilon_core4_sweep_20260519` has `20` completed roots and no running/failed roots; stage-1 analysis, visualizations, final-delta similarity, and post-boundary diagnostics completed with `completed_setting_count=20`.
- Stage 2 status: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520` has `2` completed roots, `1` running root, and no failed roots.
- Baseline GIF trace has not started: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520` has no roots yet.
- Strict `p != q` sweep has not started: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520` has no roots yet.
- Current logs: `logs/loss3_overnight_20260520T020904Z.log` and `logs/loss3_alpha_epsilon_core4_eps2_alpha0p2_p2_q2.log`.

Inference:

- The workflow has not finished. It has completed the first major stage and is early in the second major stage.
- By major stage count: stage 1 done; stage 2 in progress; stages 3 and 4 pending.
- By 300-step p=2,q=2 stage: 2 of 20 settings completed, third setting running and near the fourth method.
- The strict off-diagonal P/Q stage is large (`120` 100-step settings) and will dominate remaining runtime after the 300-step and baseline stages complete.

Remaining work:

- Continue monitoring. If desired, inspect again after the current 300-step setting completes to refine runtime estimates.

## 2026-05-20 Loss3 Overnight Run Status Check 02:09 UTC

Status: inspected active background run; no new experiment was launched by this check.

Observed process evidence:

- Overnight driver is running: PID `63104`, command `bash tools/run_loss3_overnight_20260520.sh`, elapsed about `01:18` at inspection.
- Stage-1 sweep wrapper is running: PID `63429`, command `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_20260519 --steps 100 --p 2 --q 2 --skip-completed ...`.
- Current setting runner is running: PID `63430`, command `tools/run_loss3_direction_proposal_ablation.py`, output root `forensics/loss3_alpha_epsilon_core4_sweep_20260519/fno_nu0p001_eps1_alpha0p1_batch100_steps100_p2_q2`.
- Current method observed from log: `raw_add` for `epsilon=1.0`, `alpha=0.1`, `p=2`, `q=2`.

Observed GPU evidence:

- `nvidia-smi` showed Tesla V100-SXM2-32GB, memory `9520 MiB / 32768 MiB`, GPU utilization `86%` at inspection.
- Current setting manifest records PyTorch `2.8.0+cu126`, CUDA `12.6`, device `Tesla V100-SXM2-32GB`, compute capability `sm_70`, PyTorch arch list includes `sm_70`, and JAX backend `gpu` with device `cuda:0`.

Observed log evidence:

- Overnight log: `logs/loss3_overnight_20260520T020904Z.log`.
- Current setting log: `logs/loss3_alpha_epsilon_core4_eps1_alpha0p1_p2_q2.log`.
- Log has reached stage `1. Fill missing p=2,q=2 100-step settings` and printed `[run] raw_add` for the first new setting.

Inference:

- The overnight workflow is actively running, but tasks are sequential rather than all running in parallel. It is currently in task 1 of the requested sequence.
- Because the runner writes per-method outputs after method completion, the exact step inside `raw_add` is not visible from files yet; GPU utilization and the live Python process indicate active computation.

Remaining work:

- Continue monitoring the logs and manifests. Later stages should run automatically if this stage completes successfully.

## 2026-05-20 Loss3 Overnight Readiness Review

Status: completed readiness review; no optimizer experiment was launched.

Record file added:

- `docs/loss3_overnight_readiness_review_20260520.md`

Observed verification:

- `bash -n tools/run_loss3_overnight_20260520.sh` passed.
- `py_compile` passed for all runner/analyzer/plotter/GIF scripts used by the overnight workflow.
- Dry-run p=2,q=2 300-step plan resolved to 20 settings and the core four methods.
- Dry-run strict p!=q plan resolved to 120 settings: 6 off-diagonal P/Q pairs times 20 alpha/epsilon settings.
- Dry-run baseline GIF trace plan confirmed `trajectory_final_conditions_npz=true` and `gifs=true`.

Observed caveat:

- Existing 11 completed 100-step `p=2,q=2` roots predate the new angular-speed fields. Stage 1 uses `--skip-completed`, so only the 9 newly run 100-step roots will have those fields. The full 20-setting 300-step root will have complete angular-speed metrics for all 20 settings.

Inference:

- The current code is ready to run the requested overnight workflow and will automatically generate tables, figures, Markdown notes, delta similarity, post-boundary diagnostics, and GIF panels. Complete angular-speed analysis across all 20 p=2,q=2 settings should be taken from the fresh 300-step root unless the old 100-step roots are rerun.

Remaining work:

- Launch `bash tools/run_loss3_overnight_20260520.sh` on GPU when ready, then inspect the generated manifests and figures.

## 2026-05-20 Loss3 Delta Angular-Speed Metrics

Status: completed code/plotting update only; no optimizer experiment was launched.

Source/code files updated:

- `tools/run_loss3_direction_proposal_ablation.py`
- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`
- `tools/run_loss3_overnight_20260520.sh`

Record file updated:

- `docs/loss3_next_run_commands_20260520.md`

Prepared behavior:

- Future runs write step-to-step perturbation motion metrics to `per_step_metrics.csv` and `per_sample_step_metrics.csv`: `delta_prev_cosine`, `delta_prev_angle_degrees`, `delta_step_l2`, `delta_step_pnorm`, `delta_step_linf`, normalized step distances, and `delta_unit_direction_l2_step`.
- `delta_prev_angle_degrees` is recorded as NaN when either `delta_k` or `delta_{k-1}` has near-zero norm, avoiding a fake angle at initialization.
- Visualization now shades mean +/- std for loss and boundary-ratio curves and generates angular-speed curves when `delta_prev_angle_degrees_mean` is present.
- Representative settings now get dynamics triptychs with loss, boundary ratio, and `angle(delta_k, delta_{k-1})`, all as batch mean +/- std.

Remaining work:

- Run `bash tools/run_loss3_overnight_20260520.sh`; newly generated 300-step and off-diagonal P/Q roots will contain complete angular-speed metrics.
- Existing completed roots from before this update do not contain these fields unless rerun.
- Verification: `py_compile` passed for the modified runner/plotter/diagnostic scripts; `bash -n tools/run_loss3_overnight_20260520.sh` passed; a `/tmp` smoke test of `tools/plot_loss3_alpha_epsilon_core4_visuals.py` completed without modifying official outputs.

## 2026-05-20 Loss3 Overnight Script Preparation

Status: completed command-script preparation only; no optimizer experiment was launched.

Source/code file added:

- `tools/run_loss3_overnight_20260520.sh`

Record file updated:

- `docs/loss3_next_run_commands_20260520.md`

Prepared behavior:

- The script logs to `logs/loss3_overnight_<UTC_TIMESTAMP>.log`.
- It starts with `nvidia-smi` plus a PyTorch/JAX GPU quick check; individual experiment runners still perform the required GPU verification before official runs.
- It fills missing default `p=2,q=2` 100-step settings and refreshes analysis, visualization, final-delta similarity, and post-boundary diagnostics.
- It runs all 20 default `p=2,q=2` settings for 300 steps and automatically generates analysis, visualization, final-delta similarity, and post-boundary diagnostics.
- It runs the baseline `epsilon=4, alpha=0.4, p=2, q=2` detailed 300-step GIF trace and generates multi-panel trajectory GIFs.
- It runs strict off-diagonal `p != q` 100-step settings for `1:2`, `1:inf`, `2:1`, `2:inf`, `inf:1`, and `inf:2`, then generates per-P/Q visualizations, final-delta similarity, and post-boundary diagnostics.

Remaining work:

- The user should launch `bash tools/run_loss3_overnight_20260520.sh` in terminal when ready for the overnight GPU run.

## 2026-05-20 Loss3 Post-Boundary Mechanism Diagnostics

Status: completed post-processing analysis from existing completed `p=2,q=2` sweep artifacts; no optimizer experiment was rerun.

Source/code file added:

- `tools/analyze_loss3_post_boundary_mechanism.py`

Output / record files:

- Analysis root: `forensics/loss3_post_boundary_mechanism_20260520/`
- Per-setting diagnostics: `forensics/loss3_post_boundary_mechanism_20260520/tables/post_boundary_mechanism_by_setting.csv`
- Method rollup: `forensics/loss3_post_boundary_mechanism_20260520/tables/post_boundary_mechanism_rollup.csv`
- Result Markdown: `docs/loss3_post_boundary_mechanism_diagnostics_20260520.md`
- Main result Markdown updated: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`
- Similarity metric definitions updated: `docs/loss3_alpha_epsilon_core4_delta_similarity_20260520.md`
- Next-run command note updated: `docs/loss3_next_run_commands_20260520.md`

Observed evidence:

- The diagnostic manifest reports status `completed`, `completed_setting_count=11`, and `row_count=44` for current `p=2,q=2` artifacts.
- `raw_add` has mean 99% boundary hit step `43.1` and mean post-boundary loss gain `0.511`.
- `steepest_add` has mean 99% boundary hit step `13.45` and mean post-boundary loss gain `1.160`.
- `raw_replace` and `steepest_replace` hit the 99% boundary at step `1` and have mean post-boundary loss gain `3.028`, with mean 10-step post-boundary gain `2.917`.
- Selected trajectory diagnostics show lower boundary-hit-to-final delta cosine for replacement/GPI (`0.301`) than for `raw_add` (`0.768`) or `steepest_add` (`0.696`), consistent with larger direction refinement on the boundary.
- The `p=2` tangent-motion proxy is larger for replacement/GPI (`0.597`) than for `raw_add` (`0.215`) or `steepest_add` (`0.239`).

Inference:

- GPI/replacement's post-boundary loss growth is best interpreted as fast boundary-direction optimization after immediate budget use, not merely as early boundary arrival.
- Additive PGD-like methods may have slower post-boundary growth because they spend iterations reaching the boundary and their projected boundary updates have weaker tangent/boundary-surface motion.

Remaining work:

- Re-run the diagnostic script on the pending 20-setting 100-step sweep, the 300-step `p=2,q=2` sweep, and the detailed baseline GIF trajectory after those runs complete.

## 2026-05-20 Loss3 Next-Run Command Preparation

Status: completed code/runbook preparation only; no optimizer experiment was launched.

Source/code files updated or added:

- `tools/run_loss3_direction_proposal_ablation.py`
- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/plot_loss3_trajectory_gif_panels.py`

Record file added:

- `docs/loss3_next_run_commands_20260520.md`

Observed code changes:

- The baseline runner now has `--save-trajectory-final-conditions`, which augments selected `trajectory_samples.npz` files with `clean_initial`, `perturbed_initial`, `model_final_condition`, `solver_final_condition`, and `final_condition_residual` for every saved step.
- The alpha/epsilon sweep wrapper passes through `--save-trajectory-final-conditions` and `--no-save-delta-trajectory` when requested.
- A post-processing GIF panel script was added to render delta, perturbed initial condition, model final condition, solver final condition, and final-condition residual over optimization steps.

Prepared run order:

1. Fill the nine missing default 100-step `p=2,q=2` alpha/epsilon settings in the existing 2026-05-19 sweep root.
2. Run all 20 default `p=2,q=2` settings for 300 steps in a separate root.
3. Run the baseline `epsilon=4, alpha=0.4, p=2, q=2` detailed trajectory with GIF-ready final-condition arrays.
4. Run strict `p != q` 100-step settings for the six off-diagonal P/Q pairs.

Remaining work:

- The user should run the commands in `docs/loss3_next_run_commands_20260520.md` on GPU. After the runs complete, analyze and interpret the resulting 20-setting/300-step and non-P/Q artifacts.

## 2026-05-20 Loss3 Boundary-Marker Mechanism Interpretation

Status: completed documentation update based on existing `p=2,q=2` visualization artifacts; no optimizer experiment was rerun.

Source evidence:

- Loss curves with boundary markers: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Boundary-threshold table: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/boundary_threshold_loss_gain_summary.csv`
- Main result document updated: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`

Observed evidence:

- `raw_add` / PGD and `steepest_add` / Lp-steepest PGD have boundary-ratio markers spread across steps, showing visible radial travel toward the epsilon boundary.
- `steepest_replace` / GPI-style replacement has the `0.25`, `0.50`, `0.75`, and `0.99` mean boundary-ratio markers collapsed at the first step in the completed `p=2,q=2` settings.
- GPI/replacement still shows substantial post-boundary loss growth, so boundary arrival and loss convergence are separate phases.

Inference:

- The marker pattern supports a mechanism distinction: additive PGD-like methods spend iterations reaching the boundary, while GPI/replacement immediately uses the fixed budget and then refines the boundary direction.
- Together with the final-delta smoothness and similarity evidence, this strengthens the interpretation that `steepest_replace` / GPI is the better practical optimizer for the completed fixed-budget `p=2,q=2` loss3 setting.

Remaining work:

- Re-check the same marker-collapse pattern after the nine pending alpha/epsilon settings and any requested multi-P/Q sweeps are run.

## 2026-05-20 Loss3 Core-Four Final-Delta Similarity Analysis

Status: completed post-processing analysis from existing completed `p=2,q=2` sweep artifacts; no optimizer experiment was rerun.

Source file added:

- `tools/analyze_loss3_alpha_epsilon_core4_delta_similarity.py`

Source data:

- Completed roots under `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`, using each method's `final_deltas.npz`.

Output / record files:

- Analysis root: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/`
- Manifest: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/manifest.json`
- Pairwise summary: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/tables/final_delta_pairwise_similarity_summary.csv`
- Per-sample table: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/tables/final_delta_pairwise_similarity_per_sample.csv`
- Across-setting rollup: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/tables/final_delta_pairwise_similarity_rollup.csv`
- Result Markdown: `docs/loss3_alpha_epsilon_core4_delta_similarity_20260520.md`
- Main result Markdown updated: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`

Observed evidence:

- The analysis completed with manifest status `completed`, `completed_setting_count=11`, `p_filter=2`, `q_filter=2`.
- `raw_replace` and `steepest_replace` are identical for `p=2`: cosine `1.0000`, centered cosine `1.0000`, spectral cosine `1.0000`, relative L2 `0.0000`.
- `raw_add` and `steepest_add` are highly similar: across-setting mean cosine `0.8986`, centered cosine `0.8971`, spectral cosine `0.9454`, relative L2 `0.2627`.
- Additive methods versus replacement/GPI methods have moderate signed spatial cosine but high spectral similarity: `raw_add` vs `steepest_replace` cosine `0.5713`, spectral cosine `0.8003`; `steepest_add` vs `steepest_replace` cosine `0.6144`, spectral cosine `0.8501`.
- Existing smoothness rollup records `steepest_replace` / GPI high-frequency ratio `3.707e-09`, first-derivative L2 `0.1955`, and total variation `2.472`, lower than additive methods.

Inference:

- Final perturbations share a broad low-frequency shape and are not wildly dissimilar across methods, but they cluster by update family: additive methods together, replacement/GPI methods together.
- The evidence supports the user's visual impression that the GPI perturbation is not an abnormal high-frequency or spike-like perturbation; it is smooth by the recorded metrics and spectrally similar to the other final deltas.
- For completed `p=2,q=2` settings, GPI/replacement looks better because it combines immediate boundary use, substantial post-boundary loss gain, and smooth final perturbations.

Remaining work:

- Repeat the same similarity analysis after the nine pending alpha/epsilon settings complete, and separately for additional P/Q pairs if multi-PQ alpha/epsilon runs are launched.


## 2026-05-20 Loss3 20-Setting Plan and Boundary-Threshold Markers

Status: code/plan/visualization update completed; no new optimizer experiment was launched.

Source files updated:

- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`
- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output / record files updated:

- `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_plan.json`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/boundary_threshold_loss_gain_summary.csv`
- `docs/loss3_alpha_epsilon_core4_sweep_plan_20260519.md`
- `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`
- `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`

Observed evidence:

- Dry-run generated a 20-setting default plan for `p=2,q=2`; the additional nine settings are pending and were not run in this update.
- Existing completed alpha/epsilon visualizations remain scoped to `p=2,q=2`; refreshed manifest records `pq_pairs_plotted=[{p: 2, q: 2}]`.
- The sweep wrapper now supports `--pq-pairs` for explicit multi-PQ alpha/epsilon runs.
- The loss-curve visualization now marks first mean boundary-ratio hits at `0.25`, `0.50`, `0.75`, and `0.99`.
- The analysis script's boundary thresholds were expanded to `0.25`, `0.50`, `0.75`, `0.95`, and `0.99` for future analysis runs.

Inference:

- Current conclusions about GPI/replacement versus raw PGD are evidenced for `p=2,q=2` only in this alpha/epsilon sweep.
- The new threshold markers/table should make it easier to separate radius growth, boundary arrival, and post-boundary directional optimization.

Remaining work:

- Run the default sweep to fill the nine pending `p=2,q=2` settings, or explicitly run a larger `--pq-pairs` grid if all-PQ alpha/epsilon evidence is required.
- Re-run `tools/analyze_loss3_alpha_epsilon_core4_sweep.py` after new experiments complete.


## 2026-05-20 Loss3 Plot Layout and Boundary-Gain Interpretation Update

Status: completed full visualization layout correction, reran plots, and added boundary-hit loss-gain table; no optimizer experiment was rerun.

Source file updated:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output / record files updated:

- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/boundary_ratio_mean_curves.png`
- Heatmaps under `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/heatmap_*.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/boundary_hit_loss_gain_summary.csv`
- `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`
- `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`

Observed evidence:

- Reran the plotting script; refreshed manifest reports status `completed`, generated at `2026-05-20T01:10:40.052414+00:00`, with `completed_setting_count=11`.
- Heatmap x-axis labels were shortened, subplot/colorbar margins were increased, and representative/delta-grid panels were given explicit title and bottom-margin spacing.
- The loss curves retain `x` markers for the first step where mean `boundary_ratio >= 0.99`; delta plots do not use cross markers.
- `boundary_hit_loss_gain_summary.csv` records loss at first mean 99% boundary hit, final loss, and post-boundary loss gain.
- At `epsilon=8, alpha=0.3`, `raw_add` has no mean 99% boundary hit by step `100`, while `steepest_replace` hits at step `1` and increases mean loss from `2.720` at boundary hit to `6.805` final.

Inference:

- The visual and tabular evidence supports the user's interpretation: raw PGD/additive updates are slow partly because they spend many steps reaching the boundary.
- Replacement/GPI-style methods should be described as reaching the boundary immediately and then continuing substantial directional optimization along or near the boundary; boundary arrival is not the same as final convergence.

Remaining work:

- Use the refreshed plot set and boundary-hit loss-gain table for the written comparison.


## 2026-05-20 Loss3 Curve Plot Layout Correction

Status: completed visualization layout correction and rerun; no optimizer experiment was rerun.

Source file updated:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output files refreshed:

- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/boundary_ratio_mean_curves.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`

Observed evidence:

- Moved the curve-figure legend to a separate bottom area so it no longer overlaps the title.
- Increased subplot size and spacing.
- Removed per-subplot boundary text annotations from the loss curves; only the boundary `x` marker remains on the curve.
- Added a horizontal `0.99` threshold line and fixed y-axis range to the boundary-ratio curves.
- Reran the plotting script; refreshed manifest reports status `completed`, generated at `2026-05-20T01:06:12.923337+00:00`, with `completed_setting_count=11`.

Inference:

- The boundary-ratio and loss-curve figures should now have readable method/color legend placement and less annotation clutter.

Remaining work:

- Visually inspect the refreshed PNGs for final report use.


## 2026-05-20 Loss3 Visualization Marker Correction

Status: completed visualization correction and rerun; no optimizer experiment was rerun.

Source file updated:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output / record files refreshed:

- Visualization manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- Loss boundary-marker figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Representative sample panels: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/representative_samples/`
- Delta shape grids: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/delta_shape_grids/`
- Visualization note: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`

Observed evidence:

- Removed the black `x` marker from final-delta line plots and delta-shape grids.
- Retained `x` markers only on loss curves, where they indicate the first step with mean `boundary_ratio >= 0.99`.
- Reran the plotting script; refreshed manifest reports status `completed`, generated at `2026-05-20T01:03:29.127258+00:00`, with `completed_setting_count=11`.

Inference:

- Delta plots now show perturbation shape without a misleading marker. Boundary-arrival markers are visually reserved for loss curves only.

Remaining work:

- Use the refreshed figures for interpretation/reporting.


## 2026-05-20 Loss3 PGD Boundary Literature Note and Visualization Rerun

Status: visualization script rerun completed; literature note created from web research and local experiment context. No optimizer experiment was rerun.

Source files and sources:

- Local visualization script: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`
- Local sweep outputs: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`
- Local analysis outputs: `forensics/loss3_alpha_epsilon_core4_analysis_20260519/`
- Web sources recorded in `docs/loss3_pgd_epsilon_boundary_optimum_notes_20260520.md`, including Goodfellow et al. 2014, Madry et al. 2017, DeepFool, Boundary Attack, AutoAttack/APGD, ART docs, and Distill discussion.

Output / record files:

- Refreshed visualization manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- Refreshed main loss figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Literature note: `docs/loss3_pgd_epsilon_boundary_optimum_notes_20260520.md`

Observed evidence:

- The plotting script was rerun and completed with manifest status `completed`, `completed_setting_count=11`, and boundary marker definition `first step with boundary_ratio_mean >= 0.99`.
- The web/literature note records that fixed-budget PGD/FGSM-style loss maximization is generally expected to use the epsilon budget under local linear/nonzero-gradient assumptions.
- The note also records exceptions: general nonconvex losses may have interior stationary optima or plateaus; input box constraints, regularizers, smoothness/frequency penalties, and minimum-distortion attacks can all lead to non-boundary solutions.

Inference:

- For our loss3 fixed-budget sweep, slow boundary arrival by `raw_add` is better interpreted as an optimization-path/step-scaling issue than as evidence that the true fixed-budget optimum lies inside the epsilon ball.
- Replacement/GPI-style methods reach the boundary immediately because they match the local-linear steepest/maximization geometry more directly.
- The next diagnostic should be radial loss profiles `L(x + r u)` and boundary-rescale checks for raw-add trajectories that remain inside the ball.

Remaining work:

- Add radial-profile plots and boundary-rescale loss comparisons if we want direct evidence for whether loss3 increases monotonically along final perturbation directions.


## 2026-05-20 Loss3 Alpha/Epsilon Core-Four Visualizations

Status: completed visualization post-processing from existing sweep artifacts; no optimizer experiment was rerun.

Source files:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`
- Existing sweep outputs under `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`
- Existing analysis tables under `forensics/loss3_alpha_epsilon_core4_analysis_20260519/`

Output / record files:

- Visualization root: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/`
- Manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- Result Markdown: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`
- Main loss figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Peakiness table: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/final_delta_peakiness_summary.csv`

Key settings:

- Objective: `loss3_original`, recorded as `loss3_q`.
- Geometry: `p=2`, `q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- Completed settings visualized: `11`.
- Boundary marker definition: first step where mean `boundary_ratio >= 0.99`; this approximates `||delta||_p ~= epsilon` while avoiding exact floating-point equality.
- Representative sample panels: dataset index `40` for `(epsilon, alpha)` settings `(4,0.4)`, `(8,0.3)`, `(8,1.6)`, and `(16,1.6)`.

Observed evidence:

- The visualization script completed and wrote a manifest with status `completed`.
- Generated PNG files were readable and had nonzero dimensions, including loss curves with boundary-hit `x` markers, boundary-ratio curves, final-loss/smoothness/high-frequency heatmaps, representative clean/delta/clean+delta panels, and final-delta shape grids.
- The representative panels show clean initial condition, final delta, clean plus delta, delta spectrum, and sample-level loss curve with a boundary-hit marker.
- The peakiness table records `max(abs(delta)) / RMS(delta)`, max absolute delta, total variation, first-derivative L2, and high-frequency ratio from final deltas.

Inference:

- The marked loss curves directly separate pre-boundary growth from post-boundary optimization, which addresses the user's concern that additive methods can appear slow largely because they spend many steps reaching the epsilon boundary.
- The smoothness/peakiness figures provide a visual and numeric check for Direct-Delta-like sharp spikes or high-frequency perturbations; these should be read alongside the final-loss and boundary-arrival summaries.

Remaining work:

- Inspect the new PNG figures visually for paper/report selection.
- If a paper-facing figure set is needed, choose a smaller subset of settings and export publication-sized panels with the same boundary-marker convention.


## 2026-05-20 Loss3 Alpha/Epsilon Core-Four Sweep Completion Status

Status: completed and post-processed locally. This entry records a status/analysis check, not a new run launched by the assistant in this turn.

Source files:

- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`
- Existing runner: `tools/run_loss3_direction_proposal_ablation.py`

Output files:

- Sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`
- Sweep manifest: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_manifest.json`
- Analysis root: `forensics/loss3_alpha_epsilon_core4_analysis_20260519/`
- Result Markdown: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`

Key settings:

- Objective: `loss3_original`, recorded as `loss3_q`.
- Geometry: `p=2`, `q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- Completed setting count: `11/11`; failed setting count: `0`.
- Batch/steps: dataset indices `0..99`, `batch_size=100`, `steps=100`, seed `0`.

Observed evidence:

- `ps` showed no running sweep or ablation process at the status check.
- `nvidia-smi` showed GPU utilization `0%` and memory `0 / 32768 MiB`, so no experiment process remained active.
- `sweep_manifest.json` reported status `completed`, `completed_count=11`, `failed_count=0`, start `2026-05-19T22:53:20.841733+00:00`, finish `2026-05-20T00:18:20.508758+00:00`, and summed setting elapsed time `5099.66` seconds.
- Every completed setting contains four method summaries with `101` per-step rows per method.
- The older interrupted directory `fno_nu0p001_eps4_alpha0p3_batch100_steps100_p2_q2/` remains marked `interrupted_not_for_analysis` and was skipped by post-processing.
- Post-processing generated `44` method-summary rows, `44` boundary-arrival rows, `77` winner-summary rows, and `4` method-rollup rows.

Observed boundary-arrival headline from the generated result Markdown:

- `raw_replace` and `steepest_replace` reach `99%` boundary at step `1` for all completed settings.
- `steepest_add` reaches `99%` boundary much later, matching the planned `epsilon/alpha` scale: for example `epsilon=4, alpha=0.4` has batch-mean step `11` and slowest-sample step `14`; `epsilon=8, alpha=0.3` has batch-mean step `31` and slowest-sample step `34`.
- `raw_add` is much slower to reach the boundary: for `epsilon=4, alpha=0.4`, batch-mean step `43` and slowest-sample step `56`; for `epsilon=8, alpha=0.3`, the batch mean never reaches `99%` by step `100`, and `32/100` samples do not reach `99%` by step `100`.

Inference:

- The main run is `100%` complete with `0` estimated remaining runtime.
- The user's concern is supported by the boundary-arrival diagnostics: additive raw-gradient PGD can spend many iterations below the epsilon boundary, especially at the old `epsilon=8, alpha=0.3` setting.
- Replacement/GPI-style methods' speed advantage is tightly coupled to immediate boundary arrival; final scientific interpretation should separate boundary-arrival speed from later boundary-surface optimization.

Remaining work:

- Inspect the generated result Markdown and CSV tables for the final-loss/smoothness tradeoff narrative.
- Optionally make compact plots/tables focusing on boundary arrival versus final loss for the paper-facing summary.

## 2026-05-19 Loss3 Alpha/Epsilon Core-Four Sweep Code Prep

Status: code and command preparation completed; official numerical sweep not completed in this turn. One accidentally started setting was stopped and marked not for analysis.

Source files:

- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`
- Existing runner reused: `tools/run_loss3_direction_proposal_ablation.py`

Output / record files:

- `docs/loss3_alpha_epsilon_core4_sweep_plan_20260519.md`
- `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md` (pending; no completed numerical results yet)
- `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_plan.json`
- Interrupted partial directory: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/fno_nu0p001_eps4_alpha0p3_batch100_steps100_p2_q2/`

Key planned settings:

- Objective: `loss3_original`, recorded as `loss3_q` under `p=2,q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- New baseline reference: `epsilon=4`, `alpha=0.4`.
- Previous slow reference retained: `epsilon=8`, `alpha=0.3`.
- Curated `(epsilon, alpha)` settings: `(2,0.2)`, `(2,0.4)`, `(4,0.2)`, `(4,0.4)`, `(4,0.8)`, `(4,1.2)`, `(8,0.3)`, `(8,0.4)`, `(8,0.8)`, `(8,1.6)`, `(16,1.6)`.
- Batch/steps: dataset indices `0..99`, `batch_size=100`, `steps=100`, seed `0`.

Observed evidence:

- GPU verification passed after repairing the broken `adv_robust/bin/python3` symlink: V100 `sm_70`, PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, PyTorch arch list includes `sm_70`, JAX backend `gpu`, PyTorch and JAX GPU matmul passed, and `pip check` passed.
- The new scripts passed Python bytecode compilation.
- A dry-run plan was generated successfully for the 11 curated settings and records nominal `epsilon/alpha` boundary-reach steps for L2-steepest additive updates.
- A formal sweep command was accidentally started earlier for `epsilon=4`, `alpha=0.3`; after the user clarified to prepare code and commands only, the sweep wrapper and child runner were stopped.
- The interrupted setting has no completed root-level `per_step_metrics.csv` and its manifest is marked `interrupted_not_for_analysis`.

Inference:

- No numerical optimizer conclusion should be drawn from the interrupted partial output.
- The prepared wrapper is ready for the user to launch the requested `p=2,q=2` alpha/epsilon sweep manually.
- The central scientific diagnostic is now boundary arrival: `delta_pnorm`, `boundary_ratio`, and first step to 95%/99% boundary at batch-mean and per-sample levels.
- In `p=2`, `raw_replace` and `steepest_replace` should coincide geometrically, but both rows are kept so the custom raw-replacement method is explicitly represented.

Remaining work:

- User launches `adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py` for the curated 11-setting boundary-arrival-focused sweep.
- After completion, run `adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py`.
- Update the ledger and result Markdown with observed final-loss, boundary-arrival, growth-speed, and smoothness conclusions from completed artifacts.


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

## 2026-05-17 - adv_robust Environment And 2026-05-16 Artifact Audit

Status: completed audit on the current Vast.ai instance. No numerical experiment was rerun.

Purpose:

- Check whether the copied `adv_robust` environment is runnable and complete.
- Check whether the 2026-05-16 experiment artifacts copied onto this instance are locally complete.
- Separate observed local evidence from inference, especially for smoke runs, partial directories, and R2-backed artifacts.

Source files and inputs:

- Environment policy/setup script: `tools/setup_adv_robust_gpu_env.py`.
- Requirements file: `requirements.txt`.
- GPU compatibility note: `docs/cuda_gpu_wheel_compatibility_notes_20260516.md`.
- Local 2026-05-16 manifests under `forensics/*20260516*/`.
- Local 2026-05-16 result docs under `docs/`.
- R2 sync record: `docs/r2_sync_manifest_ray_profile_20260516.md`.

Output record:

- `docs/adv_robust_environment_and_20260516_artifact_audit_20260517.md`.

Observed environment evidence:

- The first `tools/setup_adv_robust_gpu_env.py --verify-only` check found that `adv_robust/bin/python` was unusable because `python3` and `python3.12` formed a symlink loop.
- Repaired `adv_robust/bin/python3` to point to `/usr/bin/python3.12`.
- After repair, `tools/setup_adv_robust_gpu_env.py --verify-only` passed: PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, GPU `Tesla V100-SXM2-32GB`, compute capability `(7, 0)`, required architecture `sm_70`, PyTorch arch list includes `sm_70`, PyTorch CUDA matmul returned `1.0`, JAX backend was `gpu`, JAX device was `CudaDevice(id=0)`, JAX matmul returned `1.0`, and `pip check` reported no broken requirements.
- Requirements-vs-installed check covered `79` direct requirement entries: missing requirements `[]`, pinned-version mismatches `[]`, installed distributions `85`.
- Key import smoke test passed for the major packages used by the experiments: PyTorch, JAX, NumPy/SciPy/Pandas/Matplotlib, DeepXDE, Equinox, Exponax, Torch2JAX, TensorLy, tslearn, UMAP, scikit-optimize, PhiFlow/PhiML, and related dependencies.

Observed artifact evidence:

- Found `18` `manifest.json` files under `forensics/*20260516*`.
- All `18` manifests parsed, all manifest-declared `output_files` exist, all manifest-declared `result_doc`/`plan_doc` paths exist, and all manifest-declared `source_paths` exist locally.
- `17` of the `18` manifests carry GPU runtime evidence consistent with the V100 / `sm_70` policy.
- Exception: `forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001` records `"device": "cpu"` and no `gpu_runtime`; it should be treated as an old smoke check, not an official GPU result.
- Incomplete local directory: `forensics/loss3_ray_profile_optimizer_control_20260516` contains only one `local_direction_trace.csv` file with `5` lines and no manifest/result doc; the R2 sync manifest also records it as only `1` file / `672` bytes.
- Corrected fixed-sign PGD batch-100 Ray profile directory has the manifest outputs present; `ray_profile.csv` has `31501` lines (`100 * 7 * 45 + header`), `ray_winner_summary.csv` has `101` lines, and `attack_final_by_sample.csv` has `401` lines.
- Historical-script cross-check directory `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro` contains all `9` expected `loss3_*` run directories, and each has the expected `summary.json`, `loss_stats.csv`, `loss_values.npz`, `final_delta.npz`, `final_delta_summary.json`, `final_delta_diagnostics.csv`, and `final_delta_diagnostics.npz` files.
- `docs/r2_sync_manifest_ray_profile_20260516.md` records `198` uploaded files and `145207187` bytes. This audit did not perform a live R2 network query.
- `git status --short` before this record update showed untracked experiment paths but no deleted tracked experiment artifacts.

Conclusion:

- Observed evidence supports that `adv_robust` is now runnable and complete for the repository's GPU-only experiment policy after the venv symlink repair.
- Observed evidence supports that the main 2026-05-16 GPU result directories with manifests are locally complete.
- The CPU small-epsilon smoke directory is not an official GPU result.
- The `loss3_ray_profile_optimizer_control_20260516` directory is incomplete locally and should not be interpreted as a completed experiment.

Remaining work:

- If remote backup completeness matters, perform a live R2 listing/checksum comparison against the recorded manifests.
- If saving this audit in git, explicitly stage the new doc and ledger update along with any relevant source/record files; do not rely on `git commit -am`.

## 2026-05-17 - Loss3 Original Six-Experiment Completion Audit

Status: completed audit and status-table update. No numerical experiment was rerun.

Purpose:

- Locate the user's six-experiment plan and determine whether all six planned
  experiments are completed.
- Update stale planning records if the local evidence shows later experiments
  completed the missing rows.

Source plan:

- `docs/loss3_original_theory_experiment_plan.md`, section `5.1 Current
  Completion Status for the Minimal Six Experiments`.

Audit record:

- `docs/loss3_original_six_experiment_completion_audit_20260517.md`.

Observed evidence:

- Experiment 1 is documented in
  `docs/main_objective_mechanism_experiment1_result_20260514.md`. The local
  source directory
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`
  exists with `27` loss/objective/method run directories, and the core table
  `mechanism_diagnostics/mechanism_summary.csv` has `55` lines. R2 recovery update: `results/main_objective_mechanism_summary_20260514/` has now been restored locally from R2 and contains `combined_mechanism_summary.csv`, `focused_original_objectives.csv`, and three focused-objective PNG plots.
- Experiment 2 is completed by the FNO/solver Jacobian similarity records and
  the outward-growth row in `docs/outward_growth_direction_result_20260515.md`,
  with outputs under `forensics/fno_solver_jacobian_similarity_20260514/` and
  `forensics/outward_growth_direction_20260515/fno_nu0p001/`.
- Experiment 3 is completed by the GPU small-epsilon sweep and gradient-direction
  optimization records:
  `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`,
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`,
  and the corresponding `forensics/` directories.
- Experiment 4 is completed by the corrected fixed-sign PGD100 Ray Profile in
  `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`
  and its artifact directory
  `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Experiment 5 is completed by the boundary-rescaled final-boundary diagnostics
  in the 2026-05-14 final-boundary source directory and by the 2026-05-16 local
  cross-check directory
  `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.
- Experiment 6 is completed by `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`
  and the expanded `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`.
  The expanded manifest records `status: completed`, V100 GPU runtime, and
  output files under
  `forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/`.

Changes made:

- Updated `docs/loss3_original_theory_experiment_plan.md` section 5.1 from
  `Done: 5 / Partially done: 1` to `Done: 6 / Partially done: 0` for the current
  FNO / Burgers `nu=0.001` scope.
- Added `docs/loss3_original_six_experiment_completion_audit_20260517.md` with
  observed evidence and caveats.

Conclusion:

- Observed evidence supports marking all six planned experiments complete for
  the current FNO / Burgers `nu=0.001` story.
- Remaining work is extension/presentation/full remote-backup verification, not a
  blocker for the six-experiment completion claim.

R2 follow-up in the same turn:

- Used the provided R2 S3 endpoint to restore the previously missing local
  directory `results/main_objective_mechanism_summary_20260514/`.
- Restored files: `combined_mechanism_summary.csv`,
  `focused_original_objectives.csv`, and three focused-objective PNG plots.
- Verified restored CSV line counts: `combined_mechanism_summary.csv` has `325`
  lines and `focused_original_objectives.csv` has `55` lines.
- Also checked R2 for
  `forensics/loss3_ray_profile_optimizer_control_20260516`; R2 contains only
  the same single `local_direction_trace.csv`, so that directory remains an
  incomplete/aborted trace rather than a local copy gap.

## 2026-05-17 - Six-Experiment Results And Conclusions Written Into Plan

Status: completed documentation update. No numerical experiment was rerun.

Purpose:

- Add the user's requested per-experiment explanation: what each of the six
  experiments measured, which numeric outputs matter, what conclusion each
  experiment supports, and how the six results fit together.

Files updated:

- `docs/loss3_original_theory_experiment_plan.md`:
  added section `5.2 Six-Experiment Results, Data, And Conclusions - 2026-05-17`.
- `docs/loss3_original_six_experiment_completion_audit_20260517.md`:
  appended the same detailed conclusion section for audit traceability.
- `EXPERIMENT_LEDGER.md`: this entry.

Observed source evidence used:

- Experiment 1: `docs/main_objective_mechanism_experiment1_result_20260514.md`,
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/mechanism_diagnostics/mechanism_summary.csv`,
  and the restored `results/main_objective_mechanism_summary_20260514/`.
- Experiment 2: `docs/fno_solver_jacobian_similarity_result_20260514.md`,
  `docs/outward_growth_direction_result_20260515.md`,
  `forensics/fno_solver_jacobian_similarity_20260514/`, and
  `forensics/outward_growth_direction_20260515/fno_nu0p001/`.
- Experiment 3: `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`,
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`,
  and corresponding `forensics/loss3_*_20260516/` outputs.
- Experiment 4: `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`,
  `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`,
  and the 2026-05-16 local historical-script cross-check result directory.
- Experiment 5: `docs/loss3_original_plan_r2_completion_audit_20260515.md`,
  the 2026-05-14 final-boundary source directory, and the 2026-05-16 local
  cross-check directory.
- Experiment 6: `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`,
  `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`,
  and corresponding `forensics/loss3_*rotation_path_20260516/` outputs.

Key conclusions recorded:

- Experiment 1: `loss1`/`loss2` mainly produce large model/solver co-movement;
  `loss3_original` produces larger true mismatch and endpoint error.
- Experiment 2: residual movement `v_e*` and clean residual outward growth
  `v_growth*` are different local diagnostics.
- Experiment 3: ratio diagnostics converge to local Jacobian references for
  `epsilon <= 1e-2`, while finite-radius drift appears by `epsilon=0.1`.
- Experiment 4: local tiny-radius winners are not finite-radius endpoint winners;
  direct `loss3_original` is strongest at `r=8` under the corrected protocol.
- Experiment 5: boundary-rescaling ratio/regularized directions does not make
  them beat direct `loss3_original` endpoint optimization.
- Experiment 6: residual-Jacobian directions and top-k subspaces rotate and
  steepen along the attack path, so a single clean-point linearization is not a
  full finite-radius explanation.

Conclusion:

- The plan now contains a readable per-experiment result/conclusion section, in
  addition to the completion-status table.

## 2026-05-17 - Loss3 Optimizer Direction-Proposal Ablation Plan

Status: plan written; no numerical experiment was run.

Purpose:

- Design a new experiment comparing PGD, LP-steepest PGD, and generalized power
  iteration for `loss3_original`.
- Separate the direction rule from the proposal rule so the observed fast
  convergence of generalized power can be attributed to either direction choice,
  replacement/boundary update, step size, or physical/smoothness tradeoffs.

Plan document:

- `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`.

Observed code basis:

- `tools/run_batch_three_loss_loss_only.py` currently implements:
  `pgd` as `delta <- Proj(delta + alpha * grad)`,
  `lp_steepest_pgd` as `delta <- Proj(delta + alpha * steepest_direction(grad))`,
  and `generalized_power` as `delta <- Proj(epsilon * steepest_direction(grad))`.
- `run_three_loss_objective_attack.py` documents the same PGD and LP-steepest
  update rules and also supports generalized-power variants such as
  `objective_gradient`, `pure_jvp_vjp`, and `affine_jvp_vjp`.

Proposed experiment:

- Factorial ablation over three direction rules and two proposal rules:
  raw-gradient/additive, raw-gradient/replacement, LP-steepest/additive,
  LP-steepest/replacement, generalized-power/additive, and
  generalized-power/replacement.
- Main setting: FNO / Burgers `nu=0.001`, `loss3_original`, L2 input/output,
  `epsilon=8`, batch 100, zero initialization, 100 steps, GPU only.
- Metrics: final and best-so-far `loss3_original`, boundary reach, time/steps to
  threshold, boundary-normalized per-step direction quality, direction cosines,
  final-delta pairwise cosines, projection shrink factors, and perturbation
  smoothness/physical diagnostics such as Fourier high-frequency energy, total
  variation, derivative norms, and clean/adversarial spectrum comparison.

Conclusion expected from the design:

- The experiment can tell whether generalized power is faster because it uses a
  better direction, because it replaces the perturbation directly on the
  boundary, because additive PGD is step-size limited, or because fast boundary
  directions trade off against perturbation smoothness/physical plausibility.

## 2026-05-17 - Direction-Proposal Ablation Plan Equivalence Update

Status: documentation update only. No numerical experiment was run.

Purpose:

- Clarify the user's concern that PGD, LP-steepest PGD, and generalized power
  can collapse into equivalent updates under some norm/direction/proposal
  choices.

Files updated:

- `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`.

Key additions:

- Added an explicit `Equivalence And Degeneracy Cases To Check Explicitly`
  section.
- Recorded that for L2, normalized raw-gradient updates collapse to LP-steepest
  updates: `unit_raw_add == steepest_add` when both use `g / ||g||_2`.
- Recorded that the current batch-script `generalized_power` implementation can
  collapse to LP-steepest replacement when `u_k = steepest_direction(g_k, p)`, so
  `power_replace == steepest_replace` by construction.
- Added the requirement to label power variants separately, e.g.
  `power_replace__steepest_gradient`, `power_replace__objective_gradient`,
  `power_replace__pure_jvp_vjp`, and `power_replace__affine_jvp_vjp`.
- Added required sanity outputs such as same-step direction cosines,
  final-delta cosines, max absolute delta differences, and warning flags when
  two rows are mathematically equivalent.

Conclusion:

- The planned experiment now explicitly handles equivalence/degeneracy cases and
  should first verify which rows are actually distinct before making optimizer
  superiority claims.
## 2026-05-17 - Direction-Proposal Ablation Plan P/Q Geometry Update

- Status: plan update only; no numerical run was started.
- Updated `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  to add a dedicated P/Q geometry extension for the new optimizer comparison.
- Observed from `tools/run_batch_three_loss_loss_only.py`: `--p` controls the
  perturbation projection/budget for `delta`, while `--q` controls the
  residual norm used by `loss3_original`.
- Observed from `run_three_loss_objective_attack.py`: the older single-index
  runner uses `--input_p/--output_q`, supports generalized p/q power variants,
  and documents that q-aware power uses the output dual map before mapping back
  to the p-ball.
- Observed from `three_loss_objective_experiment_plan.md`: an earlier p/q sweep
  already proposed `(2,2)`, `(inf,2)`, `(2,inf)`, `(inf,inf)`, plus sparse
  input extensions.
- Inference from the code and prior plan: the optimizer ablation should compare
  methods within each p/q pair first, and only compare across p/q pairs using
  shared auxiliary metrics such as `loss3_l2`, `loss3_linf`, common-unit delta
  norms, and smoothness diagnostics.
- Remaining work: implement the runner/analysis changes, then run GPU-verified
  smoke and official p/q sweeps.
## 2026-05-17 - Direction-Proposal Ablation Plan Autograd Gradient Clarification

- Status: plan update only; no numerical run was started.
- Updated `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  to separate exact objective-gradient computation from p-steepest direction
  construction.
- Observed from existing helpers: `tools/run_batch_three_loss_loss_only.py`
  computes the scalar loss and uses autograd for the gradient, then maps the
  gradient through `steepest_direction(grad, p_order)` for LP-steepest and
  current generalized-power rows.
- Inference from the implementation and user clarification: for ordinary PGD,
  LP-steepest PGD, and objective-gradient replacement, the q-norm residual
  gradient should come from autograd.  Only the map from `g_k` to the p-ball
  steepest direction `s_k` needs an explicit formula.
- Remaining work: when implementing the new runner, record both
  `gradient_implementation_source` and `steepest_direction_source` in the
  manifest and per-run diagnostics.
## 2026-05-17 - Direction-Proposal Ablation Plan Visualization And GIF Update

- Status: plan update only; no numerical run was started.
- Updated `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  to require visualization of final deltas, per-step delta trajectories, loss
  curves, spectra, smoothness metrics, time-space heatmaps, and GIFs.
- Observed from existing code: `loss_attack_common.py` already contains basic
  run plots, final-field plots, and GIF helper patterns;
  `run_three_loss_objective_attack.py` already supports saving `trajectory.npz`
  for per-step fields; `plot_burgers_corrected_oldstyle_5loss_gif.py` already
  renders synchronized Burgers GIF frames.
- Inference from the user request: the new optimizer ablation should save
  representative-sample trajectories and make synchronized visual comparisons,
  because final scalar metrics alone cannot show whether a method creates
  spiky/high-frequency deltas early and smooths them later.
- Remaining work: implement `--save-delta-trajectory`, trajectory NPZ outputs,
  per-step smoothness/spectrum diagnostics, static comparison grids, and GIF
  rendering in the new runner/analysis scripts.
## 2026-05-17 - Direction-Proposal Ablation Runner Implemented, Not Run

- Status: code implementation only; no numerical optimizer run was started.
- Added `tools/run_loss3_direction_proposal_ablation.py` as a single-entry
  runner/analyzer/visualizer for the loss3 direction-proposal ablation.
- Source paths changed:
  - `tools/run_loss3_direction_proposal_ablation.py`
  - `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  - `EXPERIMENT_LEDGER.md`
- Planned output paths for future runs:
  - `forensics/loss3_optimizer_direction_proposal_ablation_20260517/...`
  - method subdirectories containing `per_step_metrics.csv`,
    `per_sample_step_metrics.csv`, `final_delta_diagnostics.csv`,
    `final_deltas.npz`, `trajectory_samples.npz`, and visualization outputs.
- Key settings encoded in the script: `loss3_original`, GPU-only runtime
  evidence, `--p/--q` geometry, additive versus replacement proposals, raw /
  unit raw / LP-steepest / objective-gradient power / q-aware power directions,
  representative delta trajectories, static plots, optional GIFs, and
  post-run analysis checklist generation.
- Verification performed: `python3 -m py_compile
  tools/run_loss3_direction_proposal_ablation.py` completed successfully.
- Observed evidence: syntax check passed; no model load, solver call, CUDA
  optimization loop, or experiment result file was produced by this step.
- Inference: the code is ready for a smoke run after explicit approval, starting
  with non-contiguous dataset indices `0 7 40 47 115`, short steps, and GIFs.
- Remaining work: run GPU-verified smoke, inspect the generated figures/GIFs,
  then run the official p/q and alpha/epsilon sweeps.
## 2026-05-17 - Direction-Proposal Ablation Tiny Smoke Runtime

- Status: tiny GPU smoke completed; no scientific optimizer-quality conclusion.
- GPU verification before run: `nvidia-smi` detected Tesla V100-SXM2-32GB with
  compute capability `7.0`; `tools/setup_adv_robust_gpu_env.py --verify-only`
  passed with PyTorch `2.8.0+cu126`, CUDA `12.6`, arch list including `sm_70`,
  JAX backend `gpu`, and GPU matmul sanity checks.
- Source file: `tools/run_loss3_direction_proposal_ablation.py`.
- Output root: `forensics/loss3_optimizer_direction_proposal_ablation_20260517/smoke_tiny_runtime_p2_q2_idx0_7_steps3`.
- Result document: `docs/loss3_direction_proposal_ablation_smoke_runtime_20260517.md`.
- Key settings: FNO/Burgers `nu=0.001`, `loss3_original`, dataset indices `0`
  and `7`, methods `official3 main`, `steps=3`, `epsilon=8`, `alpha=0.3`,
  `p=2`, `q=2`, trajectory saving enabled, GIF rendering enabled.
- Observed outputs: manifest status `completed`; root CSV/NPZ outputs exist;
  66 files under `figures/`; 28 GIFs under `figures/gifs/`.
- Observed runtime: method optimizer runtime sum `32.27` seconds; whole-output
  file timestamp span including diagnostics, plots, and GIFs `74.10` seconds.
- Observed prior calibration: recent local batch100/steps100 official
  `loss3_original` runs took about `103-105` seconds per ordinary method.
- Inference: a batch100/steps100 p=2/q=2 `main` run should be planned as
  roughly `30-40` minutes without GIFs, or `50-75` minutes with full GIFs.
- Remaining work: run official p=2/q=2 main comparison without GIFs first, then
  selectively render GIFs for interesting methods/samples.
## 2026-05-17 - Direction-Proposal Ablation P2/Q2 In-Progress Runtime Status

- Status: official p=2/q=2 batch100/steps100 `main` run is still in progress.
- Status check time: `2026-05-17 22:24:07 UTC`.
- Process: PID `37171`, elapsed `13:43`, CPU `98.0%`, RSS `2280672 KiB`.
- GPU: utilization `97%`, memory `10744 / 32768 MiB`.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Log path: `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p2_q2_batch100.log`.
- Observed progress: 7 of 11 methods completed; current method is
  `power_add__pure_jvp_vjp`; three methods remain after it.
- Observed completed method runtimes: ordinary/objective-gradient methods are
  stable at about `105.5` seconds each.
- Inference: remaining time from the check was about `13-18` minutes including
  final aggregation/static plots; estimated finish window `22:37-22:42 UTC`.
- Result/status document: `docs/loss3_direction_proposal_ablation_p2_q2_run_status_20260517.md`.
## 2026-05-17 - Direction-Proposal Ablation P2/Q2 Main Run Completed

- Status: completed; no scientific interpretation performed in this status check.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Log path: `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p2_q2_batch100.log`.
- Source file: `tools/run_loss3_direction_proposal_ablation.py`.
- Key settings: FNO/Burgers `nu=0.001`, `loss3_original`, batch `100`, start
  index `0`, `steps=100`, `epsilon=8`, `alpha=0.3`, `p=2`, `q=2`, methods
  `main`, trajectory saving enabled, GIF rendering not enabled.
- Observed from `manifest.json`: status `completed`, method count `11`, finished
  at `2026-05-17 22:38:08 UTC`; GPU runtime evidence recorded with Tesla
  V100-SXM2-32GB, PyTorch `2.8.0+cu126`, CUDA `12.6`, `sm_70` in arch list,
  and JAX backend `gpu`.
- Observed method runtime summary: runtime sum `1537.89` seconds, mean `139.81`
  seconds, median `105.70` seconds; q-aware JVP/VJP methods took about
  `199-200` seconds each.
- Observed output completeness: `per_step_metrics.csv` has `1111` data rows,
  `per_sample_step_metrics.csv` has `111100` data rows, `pq_geometry_summary.csv`
  has `11` data rows, `figures/` contains `56` files, and GIF count is `0` as
  intended.
- Result/status document updated: `docs/loss3_direction_proposal_ablation_p2_q2_run_status_20260517.md`.
- Remaining work: analyze p=2/q=2 metrics/figures and write a scientific result
  document before launching more p/q geometry runs.
## 2026-05-17 - Direction-Proposal Ablation P2/Q2 Visualization Update

- Status: visualization update for the completed p=2/q=2 run; no new optimizer
  experiment was run.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Generated additional figures from existing `trajectory_samples.npz` outputs:
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_000.png`
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_007.png`
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_040.png`
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_047.png`
  - corresponding `figures/final_delta_overlaid/final_delta_overlay_sample_*.png`
  - `figures/loss_progression_by_method/loss3_q_and_boundary_ratio_methods_p2_q2.png`
- Observed existing figures: mean loss/boundary/roughness curves, delta grids,
  and delta time-space heatmaps.
- Observed scope limitation: only `p=2,q=2` has been run; no cross-P/Q figures
  exist yet. Requested trajectory index `115` was not included in batch `0..99`,
  so trajectory visualizations exist for indices `0`, `7`, `40`, and `47`.
- Observed figure count after update: `65`; GIF count remains `0`.
- Status document updated: `docs/loss3_direction_proposal_ablation_p2_q2_run_status_20260517.md`.
## 2026-05-17 - Direction-Proposal Ablation Extended P/Q Queue Started

- Status: extended P/Q queue has started and is currently running.
- Queue shell PID: `44098`; current Python PID: `44102`.
- Current pair: `p=1`, `q=1`; current method from log: `raw_add`.
- Logs:
  - `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_pq_queue.log`
  - `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p1_q1_batch100.log`
- Output root for current pair:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1`.
- Observed GPU state at check: utilization `97%`, memory `9522 / 32768 MiB`.
- Observed log state: first pair has entered `raw_add`; no failure observed.
- Scope: this status check only verifies that the queue started normally; no P/Q
  pair has completed yet in this queue.
- Status document: `docs/loss3_direction_proposal_ablation_pq_queue_status_20260517.md`.

## 2026-05-17 - Direction-Proposal Ablation P2/Q2 Actual-Loss-Only Figure

- Status: visualization update only; no optimizer experiment was run.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source metrics: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/per_step_metrics.csv`.
- New figure:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/true_loss_progression/actual_loss3_q_mean_methods_p2_q2_clean_20260517.png`.
- Observed evidence: the new figure is drawn from `loss3_q_mean` values at the
  actual optimizer iterates `delta_k`; it does not use boundary-normalized loss
  or boundary-projected diagnostic loss.
- Preservation note: existing figures were kept in place and were not deleted
  or overwritten. Future additional plots should use new file names or new
  figure directories unless replacement is explicitly requested.
- Remaining work: use actual loss as the primary comparison in scientific
  summaries; treat boundary-normalized quantities only as optional diagnostics.

## 2026-05-18 - Direction-Proposal Ablation Figure Meaning Guide

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source run inspected: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source files inspected: `tools/run_loss3_direction_proposal_ablation.py` and `tools/run_batch_three_loss_loss_only.py`.
- Result document created: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Observed evidence: current completed figure set is for `p=2,q=2`; the main actual-loss figures are `figures/curves/loss3_q_mean_vs_step.png` and `figures/true_loss_progression/actual_loss3_q_mean_methods_p2_q2_clean_20260517.png`.
- Interpretation note: `boundary_ratio` is a constraint-radius diagnostic, not a loss; `boundary_loss3_q_mean` is a boundary-rescaled diagnostic and should not be used as the primary convergence/result curve.
- Remaining work: once additional P/Q runs complete, create corresponding actual-loss-first summaries and avoid leading with boundary-normalized diagnostic plots.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Progress Inspection

- Status: inspection/status update only; no new experiment was launched and no figures were deleted or overwritten.
- Queue shell PID observed: `44098`; current Python PID observed: `48371`.
- Observed current pair: `p=1`, `q=inf`; current log shows the run has entered `raw_add`, `unit_raw_add`, and `steepest_add`.
- Observed completed output roots:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1`, manifest status `completed`, `40` PNG figures.
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2`, manifest status `completed`, `40` PNG figures.
- Observed in-progress output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_qinf`, manifest status `run_started`, `0` PNG figures at the check.
- Result/status document updated: `docs/loss3_direction_proposal_ablation_pq_queue_status_20260517.md`.
- Figure guide updated: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Remaining work: continue monitoring the queue; only interpret P/Q pairs after their manifest reports completion and figures exist.

## 2026-05-18 - Direction-Proposal Ablation Mean/Std Clean Loss Figures

- Status: visualization update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source metrics inspected:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/per_step_metrics.csv`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1/per_step_metrics.csv`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2/per_step_metrics.csv`
- Observed evidence: `loss3_q_finite_count` is `100` for inspected per-step rows, and `per_sample_step_metrics.csv` has 100 raw-add step-0 sample positions for each completed run inspected. Existing old line plots use `loss3_q_mean` without plotting `loss3_q_std`.
- New figures generated under `figures/actual_loss_with_std_clean/` for completed runs `p=2,q=2`, `p=1,q=1`, and `p=1,q=2`:
  - `actual_loss3_q_mean_std_core3.png`
  - `actual_loss3_q_mean_std_power_replacement_variants.png`
- Interpretation note: shaded bands are `loss3_q_mean +/- loss3_q_std` across the 100 samples at each step, not uncertainty across repeated seeds.
- Result/guide document updated: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Remaining work: use the core-three shaded plots as the default first-read comparison and keep all-method plots as diagnostic ablation figures.

## 2026-05-18 - Direction-Proposal Ablation Final Delta Similarity Analysis

- Status: completed post-processing analysis; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source script added: `tools/analyze_final_delta_similarity.py`.
- Source arrays:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/final_deltas.npz`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1/final_deltas.npz`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2/final_deltas.npz`
- Outputs written under each completed run: `final_delta_similarity/20260518_final_delta_similarity/`, including per-sample pairwise CSV, summary CSV, mean cosine matrices, relative L2 matrices, and heatmap PNGs.
- Result document created: `docs/loss3_final_delta_similarity_results_20260518.md`.
- Observed evidence: `p=2,q=2` has exact final-delta equivalences among `unit_raw_add`, `steepest_add`, and `power_add__objective_gradient`, and among `raw_replace`, `steepest_replace`, and `power_replace__objective_gradient`.
- Observed evidence: `p=1,q=1` and `p=1,q=2` keep exact equivalence between `steepest_replace` and `power_replace__objective_gradient`, but `unit_raw_add` and `steepest_add` are not similar.
- Inference: for `p=2`, the L2 steepest direction equals normalized raw gradient, so several method labels are redundant. For `p=1`, the Lp geometry changes the steepest direction, so the same redundancy does not hold.
- Remaining work: repeat this analysis for later P/Q pairs once their manifests complete and `final_deltas.npz` exists.

## 2026-05-18 - Direction-Proposal Ablation Generalized Power Naming Clarification

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source files inspected: `tools/run_batch_three_loss_loss_only.py` and `tools/run_loss3_direction_proposal_ablation.py`.
- Result document created: `docs/loss3_generalized_power_naming_clarification_20260518.md`.
- Observed evidence: the previous three-method runner implements `generalized_power` as `steepest_direction(autograd(loss3_q), p)` followed by `project_delta(epsilon * direction, epsilon, p)`.
- Observed evidence: the new ablation's `power_*__jvp_vjp`, `power_*__affine_jvp_vjp`, and `power_*__generalized_pq` variants compute directions through JVP/VJP of the residual Jacobian rather than directly using the objective-gradient direction.
- Interpretation: when referring to the user's earlier three-method comparison, `generalized_power` should map to `steepest_replace` / `power_replace__objective_gradient`; JVP/VJP variants should be described separately as P-Q operator-power direction variants.
- Remaining work: future plots and summaries should use clearer labels to avoid presenting these as one identical method family.

## 2026-05-18 - Direction-Proposal Ablation 3D Delta Surface Visualizations

- Status: completed post-processing visualization; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source script added: `tools/plot_delta_3d_surfaces.py`.
- Command used: `./adv_robust/bin/python tools/plot_delta_3d_surfaces.py --space-stride 4 --step-stride 1`.
- Source arrays: completed runs' per-method `trajectory_samples.npz` files under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/`.
- Observed trajectory shape example: `k` has 101 steps from `0` to `100`; `delta` has shape `(101, 4, 1024, 1)` for trajectory dataset indices `0`, `7`, `40`, and `47`.
- Output directories: `figures/delta_surfaces_3d_20260518/` under each completed run.
- Observed output counts: `p=1,q=1` 28 PNGs, `p=1,q=2` 28 PNGs, `p=1,q=inf` 28 PNGs, `p=2,q=2` 44 PNGs; total `128` 3D surface PNGs.
- Observed skipped run: `p=2,q=1` had manifest status `run_started`, so it was not included in the completed-run visualization batch.
- Result document created: `docs/loss3_delta_3d_surface_visualizations_20260518.md`.
- Figure guide updated: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Remaining work: generate corresponding 3D surfaces for later P/Q pairs after their manifests complete.

## 2026-05-18 - Direction-Proposal Ablation Method Formulas And Equivalences

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source files inspected: `tools/run_batch_three_loss_loss_only.py`, `tools/run_loss3_direction_proposal_ablation.py`, and completed `p=2,q=2/final_deltas.npz` method order.
- Result document created: `docs/loss3_optimizer_method_formulas_and_equivalences_20260518.md`.
- Observed evidence: `p=2,q=2` method order begins with seven objective-gradient methods: `raw_add`, `unit_raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`, `power_add__objective_gradient`, and `power_replace__objective_gradient`.
- Formula conclusion: for `p=2`, `s_2(g)=g/||g||_2`, so `unit_raw_add`, `steepest_add`, and `power_add__objective_gradient` have identical updates, and `raw_replace`, `steepest_replace`, and `power_replace__objective_gradient` have identical updates.
- Observed similarity evidence: completed `p=2,q=2` final-delta analysis gives mean cosine `1.000000` and mean relative L2 `0.000000` for those duplicate-formula pairs.
- Inference: high similarity among the first seven p=2,q=2 methods is mainly due to duplicated objective-gradient formulas plus the L2 identity between normalized raw gradient and L2-steepest direction; it is not a universal claim for p=1, where `unit_raw_add` and `steepest_add` are not similar.
- Remaining work: rename/relabel future plots to collapse exact duplicate methods into representative method groups before presentation.

## 2026-05-18 - Direction-Proposal Ablation Readable Formula Document

- Status: documentation formatting update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_optimizer_method_formulas_readable_20260518.md`.
- Purpose: rewrite the optimizer formulas using readable displayed math instead of code-block-style pseudocode.
- Observed evidence preserved: the readable document states the p=2 equivalences `unit_raw_add = steepest_add = power_add__objective_gradient` and `raw_replace = steepest_replace = power_replace__objective_gradient`, and cites the final-delta similarity values showing cosine `1.000000` and relative L2 `0.000000` for duplicate-formula pairs.
- Remaining work: use the readable formula document as the default reference when explaining optimizer methods to avoid confusion from code-like formatting.

## 2026-05-18 - Direction-Proposal Ablation Plain-Language Formula Document

- Status: documentation formatting update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_optimizer_method_formulas_plain_language_20260518.md`.
- Purpose: rewrite method formulas without LaTeX math syntax or code-block formatting, using plain-language equations such as `delta_next = project(delta_current + alpha * direction)`.
- Observed evidence preserved: the document explains that for `p=2`, normalized raw gradient equals L2-steepest direction, which makes `unit_raw_add = steepest_add = power_add__objective_gradient` and `raw_replace = steepest_replace = power_replace__objective_gradient`.
- Remaining work: use this plain-language document when the rendered interface does not display LaTeX formulas cleanly.

## 2026-05-18 - Direction-Proposal Ablation P2/Q2 Method Equivalence Summary

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`.
- Related result document updated: `docs/loss3_final_delta_similarity_results_20260518.md`.
- Source numeric table: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/final_delta_similarity/20260518_final_delta_similarity/final_delta_pairwise_similarity_summary.csv`.
- Observed evidence: `unit_raw_add`, `steepest_add`, and `power_add__objective_gradient` have identical final deltas for `p=2,q=2` with pairwise cosine `1.000000` and relative L2 `0.000000`.
- Observed evidence: `raw_replace`, `steepest_replace`, and `power_replace__objective_gradient` have identical final deltas for `p=2,q=2` with pairwise cosine `1.000000` and relative L2 `0.000000`.
- Observed evidence: `raw_add` versus `steepest_add` has mean cosine `0.858049` and mean relative L2 `0.384210`, so it is similar but not identical.
- Inference: the high similarity among the first seven methods in the `p=2,q=2` heatmap is mainly due to the L2 identity between normalized raw gradient and L2-steepest direction plus duplicate objective-gradient method labels; future presentation plots should collapse exact duplicates into representative method groups.
- Remaining work: use the concise p=2/q=2 equivalence summary when explaining the crowded heatmap and when designing cleaned presentation figures.

## 2026-05-18 - Direction-Proposal Ablation P-Norm Equivalence Rules

- Status: documentation/interpretation update plus post-processing for newly completed runs; no optimizer experiment was launched and no existing figures were deleted or overwritten.
- Source script used for new post-processing: `tools/analyze_final_delta_similarity.py`.
- New similarity outputs generated for newly completed runs: `p=1,q=inf` and `p=2,q=1` under `final_delta_similarity/20260518_final_delta_similarity/`.
- Result document created: `docs/loss3_p_norm_method_equivalence_rules_20260518.md`.
- Related documents updated with links: `docs/loss3_p2_q2_method_equivalence_summary_20260518.md` and `docs/loss3_final_delta_similarity_results_20260518.md`.
- Observed evidence: for completed `p=1,q=inf`, `unit_raw_add` vs `steepest_add` has mean cosine `0.344037` and mean relative L2 `1.776665`, so they are not equivalent; `steepest_replace` vs `power_replace__objective_gradient` remains exactly equivalent with cosine `1.000000` and relative L2 `0.000000`.
- Observed evidence: for completed `p=2,q=1`, `unit_raw_add` vs `steepest_add` has mean cosine `1.000000` and relative L2 `0.000000`, consistent with the p=2 rule.
- Inference: `p=2` creates additional exact equivalences because normalized raw gradient equals L2-steepest direction. For `p!=2`, those extra equivalences disappear; remaining exact matches are mostly duplicate method labels such as `steepest_replace = power_replace__objective_gradient` or the current `generalized_pq = pure_jvp_vjp` implementation.
- Remaining work: analyze `p=2,q=inf` after its manifest completes and `final_deltas.npz` exists.

## 2026-05-18 - Direction-Proposal Ablation JVP/VJP Power Method Explanation

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source files inspected: `tools/run_loss3_direction_proposal_ablation.py`.
- Result document created: `docs/loss3_jvp_vjp_power_methods_explanation_20260518.md`.
- Observed code evidence: objective-gradient methods use `grad = autograd(loss3_q)` and then `steepest_direction(grad,p)`, while JVP/VJP power methods compute `J v`, apply a q-side map, pull back with VJP, and then use `steepest_direction(vjp,p)`.
- Observed formula distinction: objective-gradient replacement uses `J^T phi_q(r_k)`, while pure JVP/VJP power uses `J^T phi_q(J v_k)`; affine JVP/VJP power uses `J^T phi_q(r_k + rho_k J v_k)`.
- Observed similarity evidence: completed `p=2,q=2` final-delta analysis gives `steepest_replace` vs `power_replace__pure_jvp_vjp` mean cosine `0.121156` and relative L2 `1.293200`, so these are not the same method in practice.
- Inference: JVP/VJP variants are local operator-power direction methods and should be interpreted separately from the earlier objective-gradient generalized-power replacement method.
- Remaining work: use clearer labels in future figures to separate objective-gradient generalized power from JVP/VJP operator-power variants.

## 2026-05-18 - Direction-Proposal Ablation Minimal Core Question Summary

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_minimal_core_question_answer_20260518.md`.
- Source run summarized: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source tables: `per_step_metrics.csv` and `final_delta_similarity/20260518_final_delta_similarity/final_delta_pairwise_similarity_summary.csv`.
- Observed evidence: for `p=2,q=2`, `raw_replace` and `steepest_replace` have identical final deltas with mean cosine `1.000000` and mean relative L2 `0.000000`.
- Observed evidence: final actual loss at `k=100` is `6.804780` for both `raw_replace` and `steepest_replace`, compared with `5.389592` for `raw_add` and `6.377158` for `steepest_add`.
- Observed evidence: replacement methods in `p=2,q=2` did not show higher recorded roughness than additive methods; `raw_replace`/`steepest_replace` had high-frequency ratio `8.10e-10` and first-derivative L2 `0.217019`.
- Inference: the original core question is answered for `p=2`: replacing PGD's additive update by direct boundary replacement using the normalized gradient becomes the same as generalized-power-style steepest replacement. For `p!=2`, the exact `raw_replace` comparison remains to be run because the completed `pq_key` runs omitted `raw_replace`.
- Remaining work: run only the minimal method set `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace` for non-2 p values if the user wants the same conclusion across P/Q geometry.

## 2026-05-18 - Direction-Proposal Ablation JVP Bias-Term Clarification

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document updated: `docs/loss3_jvp_vjp_power_methods_explanation_20260518.md`.
- Clarification: the actual local residual model includes a bias/base residual term `b_k = r(x_k)`, so the local objective is closer to `||b_k + J_k v||_q` than `||J_k v||_q`.
- Interpretation: pure JVP/VJP uses `J_k v_k` and therefore studies local operator amplification while ignoring the current residual bias direction; affine JVP/VJP includes `r(x_k) + rho_k J_k v_k` and is closer to the biased local objective, but still differs from direct objective-gradient replacement.
- Remaining work: keep pure/affine JVP/VJP variants out of the minimal core experiment unless the research question is specifically about operator-power directions.

## 2026-05-18 - Direction-Proposal Ablation raw_replace vs steepest_replace Clarification

- Status: documentation clarification only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document updated: `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`.
- Clarification: `raw_replace` in the ablation is not ordinary additive PGD walking to the boundary; it is a direct boundary replacement using the normalized raw-gradient direction.
- Formula clarification: `raw_replace` uses normalized `g_k`, while `steepest_replace` uses `s_k`; these coincide for `p=2` because `s_k = g_k / ||g_k||_2`, but they should not be assumed equal for `p!=2`.
- Remaining work: keep future labels explicit, e.g. `unit_raw_replace` rather than `raw_replace`, to avoid this ambiguity.

## 2026-05-18 - Direction-Proposal Ablation Selected Core Findings Summary

- Status: documentation consolidation only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_selected_core_findings_summary_20260518.md`.
- Related document updated: `docs/loss3_minimal_core_question_answer_20260518.md`.
- Observed evidence consolidated: for `p=2,q=2`, `raw_replace` and `steepest_replace` have identical final deltas with mean cosine `1.000000` and mean relative L2 `0.000000`; both have final actual loss mean `6.804780`, boundary ratio `1.000000`, high-frequency ratio `8.10e-10`, and first-derivative L2 `0.217019`.
- Observed evidence consolidated: `raw_add` has final actual loss mean `5.389592`; `steepest_add` has final actual loss mean `6.377158`; `raw_add` vs `steepest_add` has mean cosine `0.858049`; `steepest_add` vs `steepest_replace` has mean cosine `0.530025`.
- Interpretation consolidated: for `p=2`, normalized raw-gradient direction equals L2-steepest direction, explaining exact equivalence groups; for `p!=2`, that equivalence should not be assumed.
- Remaining work: run the minimal four-method set `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace` for non-2 p values if the user wants the same boundary-replacement conclusion beyond `p=2`.

## 2026-05-18 - Direction-Proposal Ablation Simplified Core Figure Folder

- Status: completed post-processing visualization; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source run: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Output folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2`.
- Output contents: only PNG images; generated files are `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, and `03_final_delta_core4_selected_samples.png`.
- Methods plotted: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace` only.
- Result document created: `docs/loss3_simplified_core_visualizations_20260518.md`.
- Related summary updated: `docs/loss3_selected_core_findings_summary_20260518.md`.
- Scope note: this simplified four-method folder is currently only for `p=2,q=2`, because other completed P/Q runs omitted `raw_replace` and therefore cannot produce the same four-method comparison without a minimal additional run.
- Remaining work: run the minimal four-method set for non-2 p values before generating matching simplified folders for those P/Q cases.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:18 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `65319`, running `p=inf,q=1` with methods `pq_key`.
- Observed GPU state: utilization `97%`, memory `10740 / 32768 MiB`.
- Observed completed pairs: `p=1,q=1`, `p=1,q=2`, `p=1,q=inf`, `p=2,q=1`, `p=2,q=2`, and `p=2,q=inf` have manifest status `completed` and root `final_deltas.npz`.
- Observed current pair: `p=inf,q=1` has manifest status `run_started`; six of seven method directories have 101 per-step rows through `k=100`; `power_replace__generalized_pq` has started but had not written per-step rows at the check.
- Observed remaining queue items after current pair: `p=inf,q=2` and `p=inf,q=inf`.
- Result/status document updated: `docs/loss3_direction_proposal_ablation_pq_queue_status_20260517.md`.
- Remaining work: wait for queue completion, then run any needed post-processing/summary updates for the newly completed P/Q pairs.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:20 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `71831`, running `p=inf,q=2` with methods `pq_key`.
- Observed GPU state: utilization `99%`, memory `9522 / 32768 MiB`.
- Observed log evidence: queue log has advanced to `p=inf,q=2`; current run log shows method `raw_add` has started.
- Observed process evidence: no separate plotting/post-processing command such as `plot_delta_3d_surfaces.py` or `analyze_final_delta_similarity.py` was running at this check.
- Remaining work: wait for `p=inf,q=2` and queued `p=inf,q=inf` to finish, then run any desired post-processing/summary plots for those newly completed P/Q pairs.


## 2026-05-18 - Direction-Proposal Ablation Core-Four Data Availability Check 01:22 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Question checked: whether the same simplified four-method plots can be generated for P/Q pairs beyond `p=2,q=2`.
- Observed evidence: local completed P/Q directories other than `p=2,q=2` generally contain `raw_add`, `steepest_add`, and `steepest_replace`, but not `raw_replace`.
- Observed evidence: `p=inf,q=2` is currently still running and has only `raw_add` visible so far; `p=inf,q=inf` has not started in the queue output yet.
- Inference: same-style four-line plots for non-2 p values need a minimal additional run including `raw_replace`. For `p=2` pairs, `raw_replace` should be equivalent to `steepest_replace` by L2 geometry, but that would be an inference unless the method is actually run or explicitly duplicated and labeled as inferred.
- Remaining work: wait for the current queue to finish, then decide whether to run a small core-four补跑 for selected P/Q pairs.


## 2026-05-18 - Direction-Proposal Ablation p=2,q=2 Figure Location Lookup

- Status: inspection/path lookup only; no experiment was launched and no existing figures were deleted or overwritten.
- Simplified p=2,q=2 figure folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2`.
- Observed simplified PNGs: `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, and `03_final_delta_core4_selected_samples.png`.
- Full p=2,q=2 figure tree: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures`.
- Observed process state during lookup: background P/Q queue was still running `p=inf,q=2`; no plotting process was observed.


## 2026-05-18 - Direction-Proposal Ablation p=2,q=2 Smoothness Interpretation Figures

- Status: completed post-processing visualization and documentation; no optimizer experiment was launched and no existing figures were deleted or overwritten.
- Source run: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source numeric files: root/per-method `per_step_metrics.csv`, root `per_sample_step_metrics.csv`, and root `final_deltas.npz`.
- Output folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2`.
- Output PNGs: `01_final_smoothness_core4_mean_std_annotated.png`, `02_sample040_final_delta_with_smoothness_numbers.png`, and `03_sample040_smoothness_over_steps.png`.
- Result document created: `docs/loss3_delta_smoothness_metrics_interpretation_20260518.md`.
- Observed evidence: at final step `k=100`, `raw_replace` and `steepest_replace` have actual loss mean `6.804780`, first-derivative L2 mean `0.217019`, and total variation mean `2.685119`; `raw_add` has actual loss mean `5.389592`, first-derivative L2 mean `0.645722`, and total variation mean `8.573150`.
- Observed evidence: for dataset index `40`, final-step `raw_add` has first-derivative L2 `0.949167` and total variation `14.660805`, while `steepest_replace` has first-derivative L2 `0.250301` and total variation `2.685880`.
- Inference: the sample-40 visual impression that additive PGD creates a more jagged perturbation is supported by the derivative/variation metrics. In this observed `p=2,q=2` run, the GPI-style replacement reaches higher loss while also producing smoother final deltas by these metrics.
- Remaining work: repeat the same annotated smoothness plotting for other P/Q pairs after core-four data, especially `raw_replace`, is available.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:30 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `71831`, still running `p=inf,q=2` with methods `pq_key`.
- Observed GPU state: utilization `97%`, memory `10744 / 32768 MiB`.
- Observed process evidence: no separate plotting process was observed; current active workload is the optimizer/data run.
- Remaining work: wait for `p=inf,q=2` and then `p=inf,q=inf` to finish.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:33 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `71831`, still running `p=inf,q=2` with methods `pq_key`.
- Observed GPU state: utilization `98%`, memory `10744 / 32768 MiB`.
- Observed output status: completed pairs through `p=inf,q=1` have root `final_deltas.npz`; `p=inf,q=2` remains `run_started` and root `final_deltas.npz` does not yet exist.
- Observed current method progress for `p=inf,q=2`: `raw_add`, `unit_raw_add`, `steepest_add`, `steepest_replace`, `power_replace__objective_gradient`, and `power_replace__pure_jvp_vjp` each have 101 per-step rows through `k=100`; `power_replace__generalized_pq` has started in the log but has not written per-step rows yet.
- Inference: the queue is not finished; after the current `p=inf,q=2` run, `p=inf,q=inf` remains queued.
- Remaining work: wait for `p=inf,q=2` and `p=inf,q=inf` to complete before post-processing/plotting the new P/Q results.


## 2026-05-18 - Direction-Proposal Ablation Simplified Smoothness Figure Redraw 01:41 UTC

- Status: completed post-processing figure redraw/cleanup; no optimizer experiment was launched.
- Source run: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Redrawn output: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2/02_delta_quality_core4_metrics.png`.
- Redraw details: the figure now shows `boundary_ratio`, `high_frequency_energy_ratio`, `first_derivative_l2`, and `total_variation` as mean curves with `+/- 1` sample-standard-deviation shading over the 100 samples; it also includes final-step mean `+/-` sample std bars with labels.
- Deleted per user request as redundant: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2/02_sample040_final_delta_with_smoothness_numbers.png` and `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2/03_sample040_smoothness_over_steps.png`.
- Remaining annotated record: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2/01_final_smoothness_core4_mean_std_annotated.png`.
- Result documents updated: `docs/loss3_simplified_core_visualizations_20260518.md` and `docs/loss3_delta_smoothness_metrics_interpretation_20260518.md`.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Runtime Estimate 01:41 UTC

- Status: inspection/status update only; no new experiment was launched.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `79740`, running final queued pair `p=inf,q=inf` with methods `pq_key`.
- Observed current method progress: `raw_add`, `unit_raw_add`, and `steepest_add` are complete through `k=100`; `steepest_replace` has started; `power_replace__objective_gradient`, `power_replace__pure_jvp_vjp`, and `power_replace__generalized_pq` have not started.
- Inference from previous `p=inf` pair timings: remaining runtime from 2026-05-18 01:41 UTC is approximately 10-12 minutes, with expected finish around 2026-05-18 01:51-01:54 UTC if speed remains similar.
- Remaining work: wait for `p=inf,q=inf` to complete, then post-process/plot any newly completed P/Q results requested by the user.


## 2026-05-18 - Direction-Proposal Ablation Simplified Smoothness Figure Redraw 01:48 UTC

- Status: completed post-processing figure redraw; no optimizer experiment was launched.
- Redrawn output: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2/02_delta_quality_core4_metrics.png`.
- Redraw detail: changed the second simplified figure to a wide/flat layout. The left side now has four horizontally wide mean-curve panels with `+/- 1` sample-standard-deviation shading; the right side has compact final-step mean `+/-` sample std summaries.
- Output check: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2` still contains only `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, and `03_final_delta_core4_selected_samples.png`.
- Observed data availability: other P/Q output directories generally have `raw_add`, `steepest_add`, and `steepest_replace`, but not `raw_replace`; only `p=2,q=2` currently has all four core methods actually run.
- Inference: matching four-method simplified figures for non-2 p values require a minimal `raw_replace` run. For `p=2` pairs, `raw_replace` can be inferred from `steepest_replace` by L2 geometry, but should be labeled as inferred unless actually run.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:48 UTC

- Status: inspection/status update only; no new experiment was launched.
- Current final queued pair: `p=inf,q=inf`, manifest status `run_started`, root `final_deltas.npz` absent.
- Observed method progress: all methods through `power_replace__pure_jvp_vjp` have 101 per-step rows through `k=100`; `power_replace__generalized_pq` directory exists but has not written per-step rows yet.
- Inference: the queue is on the final method of the final queued pair; likely remaining time is a few minutes if timing follows the previous `p=inf` pairs.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Completion and Core-Four Availability 01:53 UTC

- Status: inspection/status update only; no new experiment was launched and no figures were generated or deleted.
- Observed process evidence: no active `run_loss3_direction_proposal_ablation.py` or queue shell process was observed.
- Observed log evidence: the final queued pair `p=inf,q=inf` reached `[done] ablation outputs written under ... p=inf_qinf`.
- Observed output evidence: all P/Q pairs in the queue have manifest status `completed` and root `final_deltas.npz` exists.
- Observed core-method availability: `p=2,q=2` has `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`; all other queued P/Q pairs have `raw_add`, `steepest_add`, and `steepest_replace` but are missing `raw_replace`.
- Inference: three-method simplified figures can be generated now for all completed P/Q pairs. Strict four-method simplified figures require a minimal `raw_replace`补跑 for pairs other than `p=2,q=2`. For `p=2` pairs, `raw_replace` can be inferred from `steepest_replace` by L2 geometry, but should be labeled as inferred unless actually run.
- Remaining work: decide whether to generate three-method figures now, inferred `p=2` four-method figures, or run a minimal `raw_replace`补跑 before generating strict four-method figures for all P/Q pairs.


## 2026-05-18 - Raw-Replace Backfill Runtime Check 02:04 UTC

- Status: inspection/status update only; no new experiment was launched and no figures were generated or deleted.
- Current backfill queue process: shell PID `86272`.
- Current Python process: PID `89734`, running `raw_replace` for `p=2,q=inf`.
- Observed GPU state: utilization `88-99%`, memory `9522 / 32768 MiB`.
- Observed completed backfill pairs: `p=1,q=1`, `p=1,q=2`, `p=1,q=inf`, and `p=2,q=1`, each with `final_deltas.npz` and 101 `raw_replace` per-step rows.
- Observed current pair: `p=2,q=inf` has started but had not yet written per-step rows at the check.
- Observed queued pairs after current: `p=inf,q=1`, `p=inf,q=2`, and `p=inf,q=inf`.
- Observed logging note: `logs/loss3_raw_replace_backfill_queue_20260518.log` was absent, but per-pair logs existed and were updating.
- Inference: completed backfill pairs are taking about 1.9 minutes each; from 2026-05-18 02:04 UTC, estimated remaining runtime is about 7-9 minutes, with expected finish around 2026-05-18 02:11-02:13 UTC if speed stays similar.
- Remaining work: wait for the raw-replace backfill to finish, then merge/plot strict four-method simplified figures for all P/Q pairs.


## 2026-05-18 - Raw-Replace Backfill Completion 02:17 UTC

- Status: completed inspection/status update; no new experiment was launched during this check and no figures were generated or deleted.
- Observed process evidence: no active `loss3_raw_replace_backfill`, `run_loss3_direction_proposal_ablation.py`, or `adv_robust/bin/python` experiment process was observed.
- Observed GPU state: utilization `0%`, memory `0 / 32768 MiB`.
- Observed log evidence: `logs/loss3_raw_replace_backfill_pinf_qinf_batch100.log` includes `[done] ablation outputs written under ... p_inf_qinf`.
- Observed output evidence: all eight raw-replace backfill pairs have manifest status `completed`, root `final_deltas.npz`, and `raw_replace/per_step_metrics.csv` with 101 rows through `k=100`.
- Backfill output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Inference: strict four-core-method plotting is now possible for all P/Q pairs by combining original runs with the corresponding raw-replace backfill outputs.
- Remaining work: generate the simplified four-method figure folders for each P/Q pair using the original run data plus raw-replace backfill data.


## 2026-05-18 - All-PQ Core Four Figure Set 02:17 UTC

- Status: completed post-processing visualization; no optimizer experiment was launched.
- Output folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_all_pq_four_figures_20260518`.
- Output contents: exactly four PNG files and no data files: `01_actual_loss_all_pq_core4_mean_std.png`, `02_boundary_ratio_all_pq_core4_mean_std.png`, `03_final_metrics_all_pq_core4_heatmaps.png`, and `04_sample040_final_delta_all_pq_core4.png`.
- Source data: original runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517` plus raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Methods plotted: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- P/Q pairs plotted: `p=1,q=1`, `p=1,q=2`, `p=1,q=inf`, `p=2,q=1`, `p=2,q=2`, `p=2,q=inf`, `p=inf,q=1`, `p=inf,q=2`, and `p=inf,q=inf`.
- Observed file check: the output folder contains exactly 4 files, all PNG images.
- Result document created: `docs/loss3_core_all_pq_four_figures_20260518.md`.
- Inference: the plots are strict four-core-method plots because `raw_replace` now comes from actual backfill outputs for all P/Q pairs where it was previously missing.
- Remaining work: inspect the four images and, if desired, generate per-P/Q separate simplified folders using the same merged source data.


## 2026-05-18 - Per-PQ Core Four Figure Folders 02:18 UTC

- Status: completed post-processing visualization; no optimizer experiment was launched.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_per_pq_four_figures_20260518`.
- Output structure: 9 P/Q subfolders (`p1_q1`, `p1_q2`, `p1_qinf`, `p2_q1`, `p2_q2`, `p2_qinf`, `pinf_q1`, `pinf_q2`, `pinf_qinf`).
- Output contents: each P/Q subfolder contains exactly four PNG files: `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, `03_final_delta_core4_selected_samples.png`, and `04_final_smoothness_core4_mean_std.png`.
- Observed file check: 9 subfolders, 4 PNG files per subfolder, 36 total files; all files are PNG images.
- Source data: original P/Q runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517` plus raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Methods plotted: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- Result document created: `docs/loss3_core_per_pq_four_figures_20260518.md`.
- Inference: the requested per-P/Q figure organization is now available; every P/Q pair has its own folder with the same four-figure structure as the revised `p=2,q=2` style.


## 2026-05-18 - Core Four Method/PQ Numerical Analysis

- Status: completed numerical analysis from existing outputs; no optimizer experiment was launched and no figures were generated or deleted.
- Source data: original P/Q runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517` plus raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Generated numeric tables: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_method_pq_analysis_20260518/core4_pq_method_summary.csv`, `core4_pq_winner_summary.csv`, and `core4_pq_metric_ranks.csv`.
- Result document created: `docs/loss3_core_method_pq_analysis_20260518.md`.
- Observed evidence: `p=2,q=2` is the cleanest balanced setting; `raw_replace` and `steepest_replace` tie for largest final loss, fastest early growth, and smoothest final delta metrics.
- Observed evidence: `p=1` settings show a tradeoff: `steepest_add` gives the largest final finite loss, `steepest_replace` grows fastest early, and `raw_replace` is smoothest by first-derivative L2 / total variation.
- Observed evidence: `p=inf` settings have nonfinite final losses for several additive methods and nonfinite intermediate behavior for replacement methods; `p=inf` is not recommended without fixing numerical stability.
- Inference: for a practical balance of high final loss, fast growth, and low-frequency/smooth perturbations, use `p=2,q=2` with `steepest_replace` / GPI; `p=2,q=1` is a useful secondary comparison, while `p=1` is mainly useful to illustrate the loss-smoothness tradeoff and `p=inf` is currently unsuitable.
- Remaining work: if needed, inspect the per-P/Q figure folders visually and add representative figures to a paper/report section.

## 2026-05-18 - Direction-Proposal Conclusions and 2026-05-19 Alpha Plan

- Status: completed documentation/planning update; no optimizer experiment was launched and no figures were generated or deleted.
- Result document created: `docs/loss3_direction_proposal_conclusions_and_20260519_plan.md`.
- Source evidence summarized: original P/Q runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517`, raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`, per-P/Q four-figure folders under `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_per_pq_four_figures_20260518`, and numerical analysis tables under `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_method_pq_analysis_20260518`.
- Observed evidence recorded: `p=2,q=2` is the cleanest balanced case; `p=2,q=1` is a useful secondary tradeoff case; `p=1,*` shows final-loss vs smoothness tradeoffs; `p=inf,*` has nonfinite/degenerate behavior and should not be used as a main result without stability fixes.
- Inference recorded: P/Q selection is itself part of the experiment conclusion; `p=2,q=2` with `steepest_replace` / GPI should be the primary result.
- Next experiment recorded for 2026-05-19: keep `epsilon = 8`, increase `alpha`, and test whether additive methods close the speed gap once they reach the boundary quickly.
- Proposed alpha sweep recorded: baseline `0.3`, then `0.6`, `1.0`, `2.0`, `4.0`, optionally `8.0` if additive methods still reach the boundary too slowly.
- Proposed P/Q settings recorded: primary `p=2,q=2`, secondary `p=2,q=1`, optional `p=1,q=2` or `p=1,q=1`; avoid `p=inf,*` for the first alpha sweep.

## 2026-05-18 - Direction-Proposal Interpretation Update: No Universal Best Optimizer

- Status: completed documentation/interpretation update; no optimizer experiment was launched and no figures were generated or deleted.
- Result document updated: `docs/loss3_direction_proposal_conclusions_and_20260519_plan.md`.
- Observed evidence recorded: no single optimizer wins every P/Q setting by final loss; `steepest_add` wins several high-loss cases, replacement/GPI methods often win speed, and `p=inf` settings show nonfinite/degenerate behavior.
- Observed visual concern recorded: several final deltas look spike-like or highly localized, especially in P/Q settings that encourage sparse or extreme perturbations.
- Inference recorded: the current result should be framed as a tradeoff among final loss, convergence speed, P/Q geometry, and physical plausibility, not as a universal optimizer ranking.
- Mechanistic interpretation recorded: `p=1` can encourage sparse/Dirac-like perturbations, `q=inf` can focus optimization on extreme residual points, and `p=inf` gives a very large feasible set that can produce unstable/nonphysical inputs.
- Next-step recommendation recorded: the 2026-05-19 alpha sweep should evaluate smoothness and spike behavior alongside final loss, and future experiments may need smoothness penalties, spectral low-pass parameterization, or an explicit smoothness budget.

## 2026-05-18 - GitHub and R2 Sync for Direction-Proposal Ablation Work

- Status: completed code/documentation push and R2 artifact sync.
- GitHub branch: `vast-ai`.
- Git commit pushed: `ab5902c` (`Add loss3 direction proposal ablation analysis`).
- GitHub remote: `origin` / `YifeiSun01/NeuralOperatorRobustness2`.
- Files committed to GitHub: experiment ledger, loss3 direction-proposal Markdown documents under `docs/`, and analysis/plotting/running scripts under `tools/`.
- Generated experiment artifacts were intentionally not committed to Git.
- R2 bucket/prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`.
- R2 synced paths: `docs/`, `tools/`, `EXPERIMENT_LEDGER.md`, `logs/`, and `forensics/`.
- R2 verification: key docs were present, including `loss3_direction_proposal_conclusions_and_20260519_plan.md`, `loss3_core_method_pq_analysis_20260518.md`, and `loss3_core_per_pq_four_figures_20260518.md`.
- R2 verification: key tools were present, including `run_loss3_direction_proposal_ablation.py`, `analyze_final_delta_similarity.py`, and `plot_delta_3d_surfaces.py`.
- R2 verification: `forensics/loss3_core_per_pq_four_figures_20260518` contained 36 objects, matching 9 P/Q folders times 4 PNGs.
- R2 verification: `forensics/loss3_optimizer_direction_proposal_ablation_20260517` contained 1698 objects and about 1.105 GiB.
- R2 verification: `forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518` contained 176 objects and about 127.5 MiB.
- Remaining local untracked files: generated `forensics/` artifacts remain untracked by Git by design; they are stored in R2.

## 2026-05-20 - Loss3 Clean Visualization Export and 4x5 Curve Relayout

- Status: completed post-processing visualization/export; no optimizer experiment was launched.
- Code changed: `tools/plot_loss3_alpha_epsilon_core4_visuals.py` now chooses 5 columns for 20 alpha/epsilon panels, producing a clean 4x5 layout for the multi-setting curve figures.
- Regenerated visual roots: `forensics/loss3_alpha_epsilon_core4_visuals_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520`, and `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520`.
- Regenerated/new similarity roots: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p1_q2_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_delta_similarity_p1_qinf_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_qinf_stopped_100steps_20260520`, and `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520`.
- Clean image-only export folder: `forensics/loss3_visuals_clean_export_20260520_2206`.
- Export contents observed: 363 image files total; verification found no non-image files in the export folder. The folder contains copied `.png` and `.gif` files only.
- Key source data: existing completed/partial stopped artifacts under `forensics/loss3_alpha_epsilon_core4_sweep_20260519`, `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`, `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520`, and baseline GIF-trace artifacts under `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520`.
- Observed evidence: regenerated 20-setting metric figures such as `loss3_q_mean_curves_with_boundary_markers.png` and `boundary_ratio_mean_curves.png` now have 4680x2736 pixel output, consistent with a 5-column by 4-row panel layout at the configured figure size and DPI.
- Result document created: `docs/loss3_visuals_clean_export_20260520.md`.
- Remaining work: inspect the clean export folder visually and select final figures for the report/paper; no additional optimizer run is pending from this export task.

## 2026-05-20 - Loss3 No-Std Curve Figure Regeneration

- Status: completed post-processing visualization/export; no optimizer experiment was launched.
- Reason: mean +/- std shading can expand the y-axis strongly when a method or P/Q setting has large variance or spike behavior, making the central loss/boundary/angle trajectories and boundary markers hard to read.
- Code changed: `tools/plot_loss3_alpha_epsilon_core4_visuals.py` now emits paired mean-curve outputs: a mean +/- std version and a mean-only/no-std-band version for `loss3_q_mean`, `boundary_ratio_mean`, and `delta_prev_angle_degrees_mean` when the angle metric exists. Dynamics triptychs also have paired no-std outputs.
- Regenerated visual roots: `forensics/loss3_alpha_epsilon_core4_visuals_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520`, and `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520`.
- New clean image-only export folder: `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std`.
- Export contents observed: 521 image files total, including 158 copied filenames containing `_no_std`; verification found no non-image files and no subdirectories in the export folder.
- Observed evidence: representative no-std 20-setting curve figures are readable PNGs at 4680x2736 pixels, preserving the previous 4x5 panel layout.
- Result document created: `docs/loss3_visuals_clean_export_with_no_std_20260520.md`.
- Inference: use the no-std figures to compare central optimizer trajectories and boundary-hit markers; use the std-shaded figures to inspect variability/spike behavior, especially for non-`p=2,q=2` settings.
- Remaining work: visually inspect the no-std/with-std pairs and choose which version to include in the writeup for each comparison.

## 2026-05-20 - Loss3 300-Step No-Std Export Verification

- Status: completed verification only; no optimizer experiment and no new plotting run was launched.
- Observed evidence: `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std` contains 71 copied files whose names start with `p2q2_300step_visuals`.
- Observed evidence: the 300-step source folder contains the main no-std curve outputs: `loss3_q_mean_curves_with_boundary_markers_no_std.png`, `boundary_ratio_mean_curves_no_std.png`, and `delta_prev_angle_degrees_mean_curves_no_std.png` under `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/`.
- Observed evidence: the clean export includes 300-step no-std dynamics triptychs, including `dynamics_triptychs_no_std` and `dynamics_triptychs_angle_available_no_std` copied filenames.
- Inference: the 300-step figures have already been regenerated with no-std variants and moved into the current clean export folder.

## 2026-05-20 - Loss3 Surprising Findings Synthesis

- Status: completed synthesis from existing local experiment artifacts and visual inspection notes; no optimizer experiment was launched.
- Result document created: `docs/loss3_surprising_findings_synthesis_20260520.md`.
- Source evidence referenced: p2q2 300-step analysis/visual roots, p!=q stopped 100-step analysis/visual roots, GPI early-step comparison, and clean std/no-std visual export `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std`.
- Observed evidence summarized: GPI/replacement reaches the p-norm boundary immediately but still grows loss afterward; GPI perturbation shape is close to its 300-step final shape by about 5-10 steps in the saved baseline samples; 300-step additive methods can sometimes exceed GPI final mean loss; LP-steepest additive PGD shows near-linear boundary-ratio growth while raw PGD curves/slows; p/q geometry, especially `q=inf`, can create spike-like perturbations.
- Inference recorded: the central mechanism is not just reaching the epsilon boundary, but moving directionally on the boundary. GPI/replacement is strong because it uses the full budget immediately and rotates aggressively on the boundary; LP-steepest fixes radial step normalization but remains additive; raw PGD has both radial desynchronization and angular inertia.
- Remaining work: for paper-level claims, pair no-std central trajectory plots with std-shaded variability plots, and separate p2q2 conclusions from p!=q geometry/spike conclusions.

## 2026-05-20 - Loss3 Tree-Style Figure Export

- Status: completed post-processing file organization/export; no optimizer experiment and no plotting run was launched.
- Reason: the previous clean export `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std` was flat and difficult to navigate with 521 images.
- New tree-style image export folder: `forensics/loss3_visuals_tree_export_20260520_with_no_std`.
- Result document created: `docs/loss3_visuals_tree_export_with_no_std_20260520.md`.
- Export structure: top-level folders `visuals/`, `similarity/`, `gpi_early_step_comparison/`, and `baseline_giftrace/`; visual folders are further split by `p2q2`, `pneq`, P/Q pair, step count, curve type, `with_std`/`no_std`, heatmaps, representative samples, delta grids, and dynamics triptychs.
- Export contents observed: 521 image files total and 158 filenames containing `_no_std`, matching the complete clean export count. Verification found no non-image files in the tree export folder.
- Example checked: `visuals/p2q2/300step/curves/loss_with_boundary_markers/with_std/loss3_q_mean_curves_with_boundary_markers.png` and `visuals/p2q2/300step/curves/loss_with_boundary_markers/no_std/loss3_q_mean_curves_with_boundary_markers_no_std.png` are now separated into paired folders.
- Remaining work: use the tree-style export as the main browsing folder; keep the flat export only as an archival all-images dump.

## 2026-05-20 - Loss3 Cross-PQ Surprising-Findings Validation

Status: completed post-processing analysis from existing local artifacts. No new optimizer/model experiment was launched in this step.

Source files and inputs:
- Analysis script: `tools/analyze_loss3_surprising_findings_validation.py`
- Prior synthesis: `docs/loss3_surprising_findings_synthesis_20260520.md`
- Existing p=q=2 300-step artifacts under the loss3 alpha/epsilon core4 p2q2 result directories.
- Existing p!=q 100-step stopped artifacts under the loss3 alpha/epsilon core4 pneq-q result directories.
- Existing similarity, per-step metric, peakiness, and visualization-export metadata tables.

Output files:
- Result Markdown: `docs/loss3_surprising_findings_validation_20260520.md`
- Tables directory: `forensics/loss3_surprising_findings_validation_20260520/`
- Main tables: `per_setting_method_validation.csv`, `pq_method_validation_rollup.csv`, `final_loss_winner_counts_by_pq.csv`, `angle_motion_winner_counts_by_pq.csv`, `boundary_arrival_winner_counts_by_pq.csv`, `delta_similarity_pair_rollup_by_pq.csv`, `equivalent_method_pairs_by_pq.csv`
- Exception tables: `replacement_nonpositive_post_boundary_gain_settings.csv`, `raw_vs_steepest_add_linearity_exceptions.csv`, `angle_winner_by_setting.csv`, `equivalent_method_pairs_by_setting.csv`, `near_high_cosine_method_pairs_by_setting.csv`, `low_cosine_selected_pairs_by_setting.csv`, `claim_exception_table_manifest.csv`

Key settings:
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- p=q=2 sweep uses the existing 20 alpha/epsilon settings with 300 steps.
- p!=q sweep uses existing 100-step artifacts for p/q combinations including p=1,q=2; p=1,q=inf; p=2,q=1; p=2,q=inf; and partial p=inf,q=1.

Observed from generated tables:
- The validation table contains 408 per-setting/method rows, 24 PQ/method rollup rows, and 36 PQ/pair similarity rollup rows.
- Boundary arrival is not the same as convergence: replacement methods often hit the boundary at step 1, but in p2q2, p2q1, and p2qinf they can still gain substantial loss afterward. However, this positive post-boundary gain is not universal: p1qinf has many weak or negative replacement post-boundary gain settings, and p1q2 has a small number of large-epsilon exceptions.
- GPI/replacement is consistently fast in boundary arrival and angular motion, but is not the universal 300-step final-loss winner. In p2q2 300-step data, `steepest_add` wins most final-mean-loss settings; replacement remains strongest as an early, stable, fast method.
- LP steepest additive updates usually produce a straighter, lower-variance boundary-ratio trajectory than raw PGD, while raw PGD is more curved and sample-dependent. This is a strong tendency, not a theorem; q=inf settings create many exceptions.
- Boundary-ratio standard deviation is a useful diagnostic: raw PGD generally has much larger std than LP steepest and replacement, while replacement-family std is near zero when the method stays exactly on the boundary.
- Replacement-family methods have the largest per-step delta angular motion in every checked setting. The winner is not always specifically `steepest_replace`; `raw_replace` can win or tie depending on P geometry.
- Final delta similarity is conditional. It is high in p2q2 and p2q1 for many method pairs, but p=1 and q=inf cases introduce many low-cosine or spike-like exceptions.
- P/Q geometry strongly affects perturbation realism. p2q1 looks comparatively natural in the checked artifacts; p1 and q=inf geometries can create localized spikes, especially for steepest/replacement directions.
- Exact method equivalence was observed for `raw_replace` and `steepest_replace` for all checked p=2 groups, q in {1,2,inf}. The same equivalence is not observed as a universal fact for p=1 or p=inf groups. p1q2 has a near-equivalence tendency between `raw_add` and `raw_replace`, but it is not exact for all settings.

Inference from the above evidence:
- The most robust summary is not "boundary solves the attack". It is: after the optimizer reaches the epsilon boundary, the decisive difference is how fast and how freely it can rotate direction along that boundary.
- GPI/replacement is best characterized as fast, stable, and strong early; not as an unconditional final-loss maximizer after long runs.
- Claims in a paper should be stated by geometry regime: p2q2/p2q1 support the cleanest story, while p=1 and q=inf require caveats about spikes, weaker similarity, and non-universal post-boundary gain.

Remaining work:
- If a paper statement needs full coverage for missing p=inf combinations, run those missing PQ sweeps explicitly; the current p=inf,q=1 evidence is partial.
- For publication figures, separate p2q2/p2q1 conclusions from p=1/q=inf caveat figures rather than collapsing them into one universal claim.

## 2026-05-20 - Loss3 GPI Theoretical Caveat Note

Status: completed interpretation update from existing validation results; no new experiment was run.

Updated result document:
- `docs/loss3_surprising_findings_validation_20260520.md`

Observed evidence referenced:
- GPI/replacement reaches the epsilon boundary immediately in the main p=2 regimes.
- GPI/replacement has much larger delta angular motion than raw PGD or LP-steepest additive PGD.
- Final delta shapes are often similar in p2q2/p2q1 despite different paths.
- 300-step additive methods sometimes match or exceed replacement final mean loss.

Inference recorded:
- Fast GPI behavior should be treated as a strong empirical surrogate effect, not as proof that Loss 3 is exactly a generalized-power objective.
- The likely mechanism is early alignment with a dominant local mode or local linearized/quadratic component of the loss, plus aggressive boundary-surface rotation.
- The result is theoretically suspicious enough that a paper should avoid saying GPI is the mathematically correct optimizer for Loss 3.

Remaining work:
- Add diagnostics for gradient-step cosine, local spectrum dominance, linear-model predicted gain versus actual gain, and boundary-tangent update decomposition if this mechanism needs to be defended rigorously.

## 2026-05-20 - Loss3 Surprising Findings Mathematical Interpretation

Status: completed theory-interpretation note from existing experiment summaries and method definitions; no new optimizer/model experiment was launched.

Source files and evidence:
- Method definitions inspected from `tools/run_loss3_direction_proposal_ablation.py` and `tools/run_batch_three_loss_loss_only.py`.
- Prior validation result: `docs/loss3_surprising_findings_validation_20260520.md`.
- Supporting method notes: `docs/loss3_jvp_vjp_power_methods_explanation_20260518.md` and `docs/loss3_generalized_power_naming_clarification_20260518.md`.

Output file:
- `docs/loss3_surprising_findings_math_explanation_20260520.md`

Observed evidence referenced:
- The current `steepest_replace`/`generalized_power` method uses the objective gradient's p-steepest direction plus replacement to the epsilon boundary; it is not the full JVP/VJP generalized P-Q power iteration unless explicitly using those variants.
- `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace` differ by direction normalization/steepest map and additive versus replacement proposal.
- p=2 makes raw-gradient normalization and p-steepest direction identical, explaining exact `raw_replace`/`steepest_replace` equivalence for checked p=2 groups.

Inference recorded:
- Boundary arrival is only active-constraint satisfaction, not constrained stationarity; post-boundary tangent/directional motion explains continued loss growth.
- Replacement/GPI is best understood as an aggressive local-linear full-budget surrogate optimizer, not as a guaranteed optimizer for the nonlinear Loss 3 objective.
- LP-steepest additive norm growth is straighter because its p-step size is normalized; raw PGD curves because gradient scale varies by sample and step.
- Boundary-ratio standard deviation diagnoses radial synchronization; angular-change metrics diagnose boundary direction search.
- P/Q geometry explains spike-prone settings: p=1 promotes sparse extreme points and q=inf promotes localized max-residual gradients.

Remaining work:
- To turn this interpretation into stronger evidence, measure gradient-step cosine, local Jacobian/Hessian spectral dominance, linearized predicted gain versus actual gain, and p=2 tangent-update components after boundary arrival.

## 2026-05-20 - Loss3 GPI Mechanism Hypothesis Probe

Status: completed post-processing mechanism probe from existing local artifacts; no neural-operator optimizer/model experiment was launched.

Source files and inputs:
- Probe script: `tools/probe_loss3_gpi_mechanism_hypotheses.py`
- p2q2 300-step sweep: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
- p!=q 100-step sweep: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`
- Saved trajectory roots: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/` and `forensics/loss3_optimizer_direction_proposal_ablation_20260517/`
- Direction-proposal p2q2 eps8/alpha0.3 ablation root: `forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/`

Output files:
- Result Markdown: `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`
- Output directory: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/`
- Tables: `mechanism_probe_by_setting_method.csv`, `mechanism_probe_rollup_by_pq_method.csv`, `mechanism_probe_rollup_by_method.csv`, `early_to_final_trajectory_probe.csv`, `hypothesis_tests.csv`, `qaware_power_variant_probe_p2q2_eps8_alpha0p3.csv`

Observed from generated outputs:
- The probe processed `102` setting roots, `408` setting/method rows, `441` trajectory-probe rows, and `11` q-aware/objective-gradient variant rows.
- In p2q2, replacement hits 99% boundary at mean step `1`, versus raw add `49.2` and steepest add `13.45`.
- In p2q2, replacement mean post-boundary gain is `3.412`; mean post-boundary angle is `30.68 deg`, compared with `0.246 deg` for raw add and `0.612 deg` for steepest add.
- Across all settings, the Pearson correlation between post-boundary gain and post-boundary angle motion is weak (`~0.041`), so angular motion alone is not sufficient to explain loss gain.
- In saved p2q2 trajectories, replacement becomes final-like early: baseline cosine to final is about `0.887` at step 5, `0.984` at step 10, and `0.995` at step 20; eps8/alpha0.3 direction-proposal trajectory is about `0.788`, `0.966`, and `0.988` at those steps.
- In the p2q2 eps8/alpha0.3 direction-proposal ablation, objective-gradient replacement final mean loss is `6.805`, while affine JVP/VJP replacement is `3.842` and pure JVP/VJP replacement is `2.324`.
- In p1qinf, steepest methods have peakiness around `30` and top-1 energy fraction around `0.86-0.96`, while p2q2 methods have peakiness around `3.6-4.1` and top-1 energy fraction around `0.014-0.018`.

Inference from the above evidence:
- The successful current GPI label is better interpreted as objective-gradient replacement: a local-linear full-budget surrogate plus aggressive boundary-direction search.
- It should not be described as a literal generalized P-Q power method or exact optimizer for full nonlinear Loss 3.
- p2q2-like regimes appear to have a dominant useful direction that replacement reaches within a few steps; p1/qinf regimes can create large angular motion that is not useful and often spike-like.

Remaining work:
- For stronger mechanism evidence, run a GPU diagnostic that records gradient-step cosine, local Jacobian/Hessian spectral dominance, and local-linear predicted gain versus actual gain along selected trajectories.

## 2026-05-20 - Loss3 Boundary Geometry Derivation and Validation Clarification

Status: completed explanatory/validation note from existing artifacts; no new optimizer/model experiment was launched.

Output file:
- `docs/loss3_boundary_geometry_derivation_and_validation_20260520.md`

Source evidence referenced:
- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`
- `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/`
- Prior final-delta similarity and p/q concentration metrics from existing loss3 analyses.

Observed evidence recorded:
- p2q2 replacement hits 99% boundary at mean step `1`; raw add at `49.2`; steepest add at `13.45`.
- p2q2 replacement post-boundary angular motion is about `30.68 deg`, compared with `0.246 deg` for raw add and `0.612 deg` for steepest add.
- p2q2 replacement post-boundary gain is `3.412`.
- Existing trajectory probes show p2q2 replacement reaches high cosine to final delta by steps 5-20.
- p1qinf and p1q2 concentration metrics support the spike explanation for p=1/q=inf-like geometries.

Inference recorded:
- For p=2, `(I - u u^T) grad L` is the tangent-plane projection of the gradient on the L2 boundary; nonzero tangent projection means loss can still increase while staying on the boundary.
- LP-steepest additive norm growth is only approximately linear; it depends on normalized step size plus stable positive radial alignment.
- Additive angular motion after boundary is O(alpha/epsilon), while replacement has no alpha/epsilon small factor and can rotate directly to the new selected direction.
- Existing evidence supports but does not fully prove the dominant-direction hypothesis; a true spectral-mode claim needs a GPU Jacobian/Hessian or tangent-KKT diagnostic.

Remaining work:
- Implement a GPU diagnostic for p2 tangent KKT residual `||(I-u u^T) grad L|| / ||grad L||`, local-linear predicted gain, and local spectral dominance along selected trajectories.

## 2026-05-20 - Loss3 Mechanism Probe Plain-Language Walkthrough

Status: completed documentation clarification; no new experiment or post-processing run was launched.

Updated result file:
- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`

Clarification recorded:
- The mechanism probe is a post-processing analysis of existing `per_step_metrics.csv`, `final_deltas.npz`, and `trajectory_samples.npz` artifacts, not a new neural-operator run.
- A `setting root` means one completed P/Q/epsilon/alpha/steps experiment folder; the probe read 102 such roots and 408 core-method rows.
- A `trajectory probe` row is an early-step-to-final-delta comparison from saved trajectory arrays, not a new optimization run.
- The phrase boundary-surface direction change means loss increasing after `boundary_ratio ~= 1` while delta direction/angle continues changing.

Remaining work:
- If the mechanism needs to be verified beyond saved artifacts, run the proposed GPU diagnostics for tangent KKT residual, local-linear predicted gain, and local spectral dominance.

## 2026-05-20 - Loss3 p2q2 Tangent and Radial Geometry Probe

Status: completed post-processing diagnostic from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.

Source files and inputs:
- Script: `tools/probe_loss3_p2q2_tangent_and_radial_geometry.py`
- Sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
- Input tables: per-setting `per_sample_step_metrics.csv` files containing recorded autograd-gradient cosines, direction cosines, delta norms, losses, and per-step geometry metrics.

Output files:
- Result Markdown: `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`
- Output directory: `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/`
- Tables: `tangent_kkt_by_setting_method.csv`, `tangent_kkt_rollup_by_method.csv`, `radial_growth_by_setting_method.csv`, `radial_growth_rollup_by_method.csv`, `geometry_probe_correlations.csv`

Observed from generated outputs:
- Processed 20 p2q2 setting roots, 80 tangent setting/method rows, and 40 additive radial-growth rows.
- p2 tangent-gradient residual was computed as `sqrt(1 - cos(delta, grad)^2)`, equivalent to `||(I-u u^T) grad L|| / ||grad L||` when `u=delta/||delta||_2`.
- In p2q2 rollup, replacement/GPI tangent residual at boundary hit is about `0.855`, while raw add is about `0.397` and steepest add about `0.435`.
- Last gradient-bearing step tangent residual is about `0.480` for replacement/GPI, but only about `0.039` for raw add and `0.046` for steepest add.
- Post-boundary gain correlates with tangent residual at hit across alpha/epsilon settings: raw add `r=0.816`, steepest add `r=0.702`, replacement `r=0.622`.
- Replacement post-hit/last tangent residual is not positively correlated with gain (`r=-0.082` after hit and `r=-0.112` last update), so tangent motion is opportunity/capacity, not a complete predictor of useful loss growth.
- The p2 radial-growth formula matches the recorded pre-boundary radius increments with mean absolute error around `1e-7`.
- Steepest add has direction L2 mean `1.0` with CV about `3.3e-8`; raw add direction L2 CV is about `0.621`. Actual radial-increment CV is about `0.110` for steepest add and `0.621` for raw add.

Inference from the above evidence:
- The p2 boundary-stationarity story is directly supported: replacement reaches the boundary with a large tangent gradient component and therefore still has room to improve by rotating on the boundary.
- LP-steepest's straighter norm growth is mainly explained by normalized direction size and synchronized radial increments, not by cos-theta being universally more stable than raw PGD.
- Raw PGD's curved and variable radial progress is strongly tied to raw gradient norm variability.

Remaining work:
- For a stricter final convergence claim, rerun selected trajectories with gradient computation enabled at the final saved step and optionally store full gradient vectors for direct projection plots.

## 2026-05-20 - Loss3 Mechanism Validation and Landscape Visualization Plan

Status: completed planning/analysis document; no new neural-operator experiment was launched.

Output file:
- `docs/loss3_mechanism_validation_and_landscape_plan_20260520.md`

Source context inspected:
- Existing mechanism diagnostics: `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md` and `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`.
- Existing tools for landscape/curvature/ray diagnostics: `tools/run_loss3_2d_slice_planarity.py`, `tools/run_loss3_path_directional_curvature.py`, `tools/run_loss3_ray_profile*.py`, `tools/run_loss3_jacobian_subspace_rotation_path.py`, and local Jacobian/SVD analysis tools.

Observed evidence summarized:
- Existing saved metrics can already validate p2q2 tangent residual and radial-growth mechanisms.
- Existing tools can be adapted for core4/PQ ray profiles, 2D slices, boundary arc interpolation, directional curvature, and local dominant-mode/Jacobian probes.

Inference recorded:
- The next strongest validation is not another broad alpha/epsilon sweep; it is targeted landscape probes that visualize rays, 2D planes, boundary arcs, and local curvature around method deltas.
- Dominant-direction claims require stronger evidence from ray/2D/arc/Jacobian diagnostics; current early-to-final cosine evidence is suggestive but not a spectral proof.
- p/q spike claims can be tested by comparing landscape sharpness/curvature and concentration metrics across p2q2, p2q1, p2qinf, and p1qinf.

Remaining work:
- Implement or adapt `tools/run_loss3_pq_landscape_probe.py` to read core4 outputs and evaluate rays/slices/arcs on GPU after the required GPU verification.
- Run a small pilot before any large PQ landscape sweep.

## 2026-05-20 - Loss3 p2q2 Stepwise Geometry vs Next-Step Gain Probe

Status: completed post-processing diagnostic from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.

Source files and inputs:
- Script: `tools/probe_loss3_p2q2_stepwise_gain_geometry.py`
- Sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
- Input tables: per-setting `per_sample_step_metrics.csv` files with recorded loss, delta/gradient cosines, direction cosines, projection shrink factors, and step geometry.

Output files:
- Result Markdown: `docs/loss3_p2q2_stepwise_gain_geometry_probe_20260520.md`
- Output directory: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`
- Tables: `stepwise_geometry_mean_by_setting_method_step.csv`, `stepwise_geometry_correlations_by_method_scope.csv`, `stepwise_geometry_correlations_by_setting_method_scope.csv`
- Figures: `post_boundary_grad_tangent_ratio_vs_next_loss_gain.png`, `post_boundary_direction_tangent_ratio_vs_next_loss_gain.png`, `post_boundary_linear_gain_projected_vs_next_loss_gain.png`, `post_boundary_stepwise_correlation_bars.png`

Observed from generated outputs:
- Processed 20 p2q2 setting roots and 2,400,000 sample-step pairs.
- The diagnostic compares geometry at step `k` against immediate next-step gain `loss3_q[k+1] - loss3_q[k]`.
- Post-boundary `grad_tangent_ratio` correlates with next-step gain for additive methods: `raw_add r=0.337`, `steepest_add r=0.265`.
- Post-boundary projected local-linear gain is a stronger one-step predictor: `raw_add r=0.600`, `steepest_add r=0.299`.
- Post-boundary radial/signed ratio is negatively correlated with next-step gain for additive methods: `raw_add r=-0.327`, `steepest_add r=-0.222`.
- Replacement methods have weak post-boundary one-step correlations (`linear_gain_projected r=0.097`, `grad_tangent_ratio r=0.029`), despite large hit-step-to-final gains found in the previous tangent/radial probe.

Inference from the above evidence:
- Stepwise tangent projection is related to next-step loss growth, but it is only an opportunity measure. The actual projected local-linear gain better captures whether the chosen next step uses the tangent opportunity in a useful direction.
- The earlier conclusion remains consistent: replacement/GPI reaches the boundary with large tangent residual and has room to improve, but once it is rotating aggressively on the boundary, immediate gain is not explained by tangent magnitude alone.
- For LP-steepest norm growth, the data should not be interpreted as the direction being fixed. The stronger explanation is that the direction norm is fixed/normalized, so radial increments are much less variable than raw PGD, whose raw gradient norm varies strongly.

Remaining work:
- If needed, extend the same stepwise gain diagnostic to other p/q settings and add GPU landscape probes for ray/2D/arc curvature validation.

## 2026-05-20 - Loss3 Mechanism/Landscape Existing Audit, All-p2 Tangent Extension, and Core4/PQ Landscape Pilot

Status: completed audit plus one post-processing extension and one small GPU landscape evaluation pilot.

Source files and inputs:
- Audit/result docs inspected: `docs/loss3_ray_profile_ri_final_report_fno_nu0p001_20260516.md`, `docs/loss3_2d_slice_planarity_fno_nu0p001_steps100_dense_result_20260517.md`, `docs/loss3_directional_curvature_fno_nu0p001_steps100_samples5_20260517.md`, `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`, and related forensics roots.
- Post-processing script: `tools/probe_loss3_all_p2_tangent_geometry.py`
- GPU landscape script: `tools/run_loss3_core4_pq_landscape_probe.py`
- Existing core4 inputs: p2q2 baseline giftrace root and p2q1/p2qinf/p1qinf `eps=4, alpha=0.4` roots.

Output files:
- Audit doc: `docs/loss3_mechanism_landscape_existing_vs_new_audit_20260520.md`
- All-p2 tangent doc: `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- All-p2 tangent output: `forensics/loss3_all_p2_tangent_geometry_probe_20260520/`
- Core4/PQ landscape doc: `docs/loss3_core4_pq_landscape_probe_20260520.md`
- Core4/PQ landscape output: `forensics/loss3_core4_pq_landscape_probe_20260520/`

GPU verification for landscape pilot:
- `nvidia-smi` showed Tesla V100-SXM2-32GB, driver 570.211.01, CUDA 12.8 driver capability, and no active GPU processes before the run.
- `adv_robust/bin/python` reported PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, CUDA available, device `Tesla V100-SXM2-32GB`, compute capability `(7,0)`, PyTorch arch list including `sm_70`, successful CUDA matmul, JAX backend `gpu`, and JAX device `cuda:0`.
- The landscape manifest records GPU-only runtime details and no CPU fallback.

Observed from existing audit:
- Prior ray-profile, 2D-slice, curvature, and Jacobian/SVD experiments already exist and are reusable as background evidence.
- Those old experiments are mainly old `loss3_original_pgd` path diagnostics, not current core4/PQ method comparisons, so they cannot fully substitute for the current GPI/replacement mechanism checks.

Observed from all-p2 tangent extension:
- Processed 60 p=2 setting roots and 24,000 sample/method rows.
- Replacement hits the 99% boundary at step 1 in p2q1, p2q2, and p2qinf.
- Replacement tangent residual at boundary hit is high across q: about `0.873` for p2q1, `0.855` for p2q2, and `0.938` for p2qinf.
- p2qinf is the warning case: high tangent residual and large angular motion do not translate into large post-boundary gain.

Observed from GPU landscape pilot:
- Scope: four setting roots, samples `[0, 7, 40, 47]`, row counts ray `8000`, boundary arc `756`, 2D slice `1568`, curvature `64`.
- In baseline p2q2, replacement/GPI endpoint ray loss at epsilon is `1.481` for step 1, `3.494` for step 5, `3.664` for step 10, `3.679` for step 20, and `3.634` for final. This supports quick convergence to a final-like high-loss direction by around 5-10 steps.
- p2q2 boundary arcs between replacement and additive final directions stay high-loss, with minima around `98.8%` of the weaker endpoint.
- p2q1 also looks broadly connected and replacement has strong endpoint ray loss.
- p2qinf differs: endpoint final-direction winner is steepest_add rather than replacement, and some arcs dip more.

Inference from the above evidence:
- The dominant-ridge / rapid-final-like-direction story is supported for p2q2 and partly p2q1.
- The story must be qualified for q=inf and p1/qinf: high tangent opportunity and aggressive angular motion are not sufficient without useful landscape alignment.
- A paper-level dominant-mode claim still needs a new, expensive local Jacobian/SVD probe on current core4/PQ deltas; old Jacobian/SVD artifacts are background evidence only.

Remaining work:
- Optionally scale the landscape pilot to more samples/settings.
- Run targeted local Jacobian/SVD dominant-mode diagnostics for p2q2 baseline, p2qinf warning case, and p1qinf spike case only if a stronger spectral claim is needed.

## 2026-05-21 - Loss3 Correlation and Radial-Growth Clarification

Status: completed clarification from existing generated tables; no experiment was rerun.

Source files inspected:
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/geometry_probe_correlations.csv`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/radial_growth_rollup_by_method.csv`
- `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/tables/stepwise_geometry_correlations_by_method_scope.csv`

Output file:
- `docs/loss3_correlation_and_radial_growth_clarification_20260521.md`

Observed evidence:
- Setting-level hit-to-final correlations answer whether tangent residual at first boundary hit predicts total post-boundary gain across 20 p2q2 alpha/epsilon settings: raw_add `r=0.816`, steepest_add `r=0.702`, replacement/GPI `r=0.622`.
- Stepwise post-boundary correlations answer whether the geometry at each post-boundary step predicts immediate next-step gain: raw_add tangent ratio `r=0.337`, steepest_add `r=0.265`, replacement/GPI `r=0.029`.
- Post-boundary projected local-linear gain is a better one-step predictor for additive methods: raw_add `r=0.600`, steepest_add `r=0.299`, replacement/GPI `r=0.097`.
- Radial-growth rollup shows `steepest_add` direction L2 CV is about `3.3e-8`, but cos(theta) CV is `0.478`; raw_add direction L2 CV and actual radial increment CV are both about `0.621`.

Inference:
- The high hit-to-final correlations and lower stepwise correlations are not contradictory; they answer different statistical questions.
- Fixed update norm alone does not mathematically guarantee straight norm growth. The supported statement is that normalized update length plus mostly positive radial alignment makes steepest_add radial progress much steadier in the observed p2q2 data.

## 2026-05-21 - Loss3 Key Findings Master Summary

Status: completed consolidated Markdown record; no experiment was rerun.

Source files and inputs:
- Existing result docs under `docs/`, especially `docs/loss3_surprising_findings_synthesis_20260520.md`, `docs/loss3_surprising_findings_validation_20260520.md`, `docs/loss3_boundary_geometry_derivation_and_validation_20260520.md`, `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`, `docs/loss3_p2q2_stepwise_gain_geometry_probe_20260520.md`, `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`, `docs/loss3_core4_pq_landscape_probe_20260520.md`, and `docs/loss3_correlation_and_radial_growth_clarification_20260521.md`.
- Existing forensics roots including `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/`, `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`, `forensics/loss3_all_p2_tangent_geometry_probe_20260520/`, and `forensics/loss3_core4_pq_landscape_probe_20260520/`.

Output file:
- `docs/loss3_key_findings_master_summary_20260521.md`

Observed evidence recorded:
- Boundary hit is not convergence; replacement/GPI reaches boundary early but retains large tangent residual and large post-hit angular motion.
- p2q2 setting-level tangent residual at boundary hit correlates with total post-boundary gain, while stepwise tangent residual has much weaker immediate next-step correlation.
- Objective-gradient replacement is the successful current surrogate; existing ablation does not support calling it the exact generalized-power optimizer of full Loss3.
- LP-steepest straight-ish norm growth is supported by normalized update length plus observed positive radial alignment, not by fixed direction.
- Final delta shapes are often similar in p2q2/p2q1 but this weakens in p=1 or q=inf settings.
- p/q geometry strongly affects spike/concentration behavior.

Inference:
- The consolidated paper-style interpretation is that replacement/GPI is an aggressive full-budget local surrogate that is especially effective in p2q2-like landscapes with a strong shared high-loss direction or ridge, but it should not be overclaimed as a guaranteed global optimizer of Loss3.

Remaining work:
- Optional targeted local Jacobian/SVD probes are still needed for a strong spectral dominant-mode claim.

## 2026-05-21 - Loss3 Current Core4/PQ Mechanism Validation Suite

Status: completed current core4/PQ mechanism validation with GPU landscape evaluation, trajectory-focused early-ray evaluation, targeted residual-Jacobian/SVD probe, summary figures, summary tables, and Markdown conclusions.

Source files and inputs:
- Landscape runner: `tools/run_loss3_core4_pq_landscape_probe.py`
- Jacobian/SVD runner: `tools/probe_loss3_current_core4_jacobian_svd.py`
- Summary plotter: `tools/plot_loss3_mechanism_validation_summary.py`
- Existing current core4/PQ setting roots under `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/` and `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`
- Existing tangent/stepwise post-processing roots: `forensics/loss3_all_p2_tangent_geometry_probe_20260520/` and `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`

GPU verification:
- Pre-run `nvidia-smi` showed Tesla V100-SXM2-32GB, driver 570.211.01, CUDA 12.8 driver capability, and no active GPU processes.
- `adv_robust/bin/python` reported PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, CUDA available, device `Tesla V100-SXM2-32GB`, compute capability `(7,0)`, PyTorch arch list including `sm_70`, successful CUDA matmul, JAX backend `gpu`, and JAX device `cuda:0`.
- Each GPU runner manifest records GPU-only runtime metadata and no CPU fallback.

Output files:
- Expanded landscape doc: `docs/loss3_core4_pq_landscape_probe_full_20260521.md`
- Expanded landscape output: `forensics/loss3_core4_pq_landscape_probe_full_20260521/`
- Trajectory-focused landscape doc: `docs/loss3_core4_pq_landscape_probe_trajectory_20260521.md`
- Trajectory-focused landscape output: `forensics/loss3_core4_pq_landscape_probe_trajectory_20260521/`
- Targeted Jacobian/SVD doc: `docs/loss3_current_core4_jacobian_svd_probe_20260521.md`
- Targeted Jacobian/SVD output: `forensics/loss3_current_core4_jacobian_svd_probe_20260521/`
- Final mechanism summary doc: `docs/loss3_current_mechanism_validation_summary_20260521.md`
- Final mechanism summary output: `forensics/loss3_current_mechanism_validation_summary_20260521/`
- Updated master summary: `docs/loss3_key_findings_master_summary_20260521.md`

Key settings:
- Expanded landscape: p2q2 baseline plus p2q1, p2qinf, and p1qinf baseline roots; sample indices `[0,7,20,40,47,63,80,99]`; ray/arc/2D slice/curvature evaluation.
- Trajectory-focused landscape: same four roots; sample indices `[0,7,40,47]` because only those have trajectory NPZ; early steps `[1,5,10,20]`.
- Jacobian/SVD: sample index `0`; 11 current core4/PQ states including p2q2 clean/step1/step5/step10/replacement final/steepest_add final, p2qinf clean/replacement final/steepest_add final, and p1qinf clean/replacement final.

Observed evidence:
- p2q2 steepest_replace ray endpoint ratio relative to final reaches `0.961` at step 5, `1.008` at step 10, and `1.012` at step 20.
- p2q2 boundary arcs stay near the weaker endpoint: replacement-to-additive arcs have min/weak-endpoint ratio about `0.986`.
- all-p2 tangent residual confirms replacement reaches boundary at step 1 with large tangent residual: p2q1 about `0.873`, p2q2 about `0.855`, p2qinf about `0.938`.
- p2q2 residual-Jacobian spectrum becomes more dominated after GPI moves: `sigma1/sigma2` rises from `1.148` at clean to `5.094` at replacement step 5, with top-1 energy fraction rising from `0.404` to `0.925`.
- The stronger pure-SVD claim is not supported: at p2q2 replacement final, the top residual-Jacobian right singular vector has cosine only `0.281` with the replacement final delta.
- p2qinf and p1qinf remain caveats: p2qinf has stronger geometry-dependent arc/ray differences, and p1qinf replacement final has weak spectral gap (`sigma1/sigma2 = 1.235`) and near-orthogonal top direction to replacement final delta.

Inference:
- The current data supports the local-surrogate/shared-ridge explanation: replacement/GPI rapidly reaches a high-loss boundary region and can rotate aggressively there.
- The data does not support claiming that replacement/GPI simply follows the top singular vector of the local residual Jacobian.
- The cleanest paper wording should remain conditional: objective-gradient replacement is an aggressive full-budget surrogate that works very well in p2q2-like geometry, while q=inf and p=1 geometries limit the story and can produce less stable/spikier behavior.

Remaining work:
- Optional: repeat the residual-Jacobian/SVD probe for more samples if a statistically stronger spectral statement is needed. Current SVD probe is targeted, not population-level.

## 2026-05-21 - Loss3 Hypothesis Validation Status Map

Status: completed documentation-only consolidation of which mechanism hypotheses have been verified, rejected, refined, or remain open. No new experiment was run for this map.

Source files and inputs:
- `docs/loss3_current_mechanism_validation_summary_20260521.md`
- `docs/loss3_current_core4_jacobian_svd_probe_20260521.md`
- `docs/loss3_core4_pq_landscape_probe_full_20260521.md`
- `docs/loss3_core4_pq_landscape_probe_trajectory_20260521.md`
- `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`
- `docs/loss3_p2q2_stepwise_gain_geometry_probe_20260520.md`
- `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`

Output file:
- `docs/loss3_hypothesis_validation_status_20260521.md`

Observed evidence recorded:
- Boundary hit, post-boundary tangent residual, angular motion, early high-loss ray ratios, shared ridge boundary arcs, p/q spike behavior, LP-steepest radial growth, raw PGD gradient-scale effects, and targeted residual-Jacobian/SVD mismatch were each mapped to explicit hypotheses.

Inference:
- Most of the original practical hypotheses are verified or refined, but the pure residual-Jacobian top-singular-vector explanation is rejected. The remaining strongest explanation is objective-gradient full-budget replacement as an aggressive local surrogate for an affine/nonlinear Loss3 landscape.

Remaining work:
- Population-level Jacobian/SVD and direct affine trust-region comparisons remain optional future work if a stronger mathematical claim is needed.

## 2026-05-21 - Loss3 GitHub and R2 Artifact Sync

Status: completed.

Source files and inputs:
- Local git branch `vast-ai`.
- Loss3 docs and tools under `docs/` and `tools/`.
- Generated Loss3 artifact directories under `forensics/`.

GitHub output:
- Commit `9dac8f5` with message `Add Loss3 core4 mechanism validation tooling and docs`.
- Pushed to `origin/vast-ai` at `https://github.com/YifeiSun01/NeuralOperatorRobustness2/tree/vast-ai`.

R2 output:
- Destination prefix: `s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/forensics/`.
- Uploaded generated Loss3 figures, visual exports, summaries, probe outputs, and large sweep outputs.
- Verification reported `14,253` objects and `16.710 GiB` under the remote `forensics` prefix.

Observed evidence:
- R2 remote listing showed key uploaded directories including `loss3_core4_pq_landscape_probe_full_20260521`, `loss3_core4_pq_landscape_probe_trajectory_20260521`, `loss3_current_core4_jacobian_svd_probe_20260521`, `loss3_current_mechanism_validation_summary_20260521`, `loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`, and `loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520`.

Remaining work:
- None for this sync request. Local `forensics/` remains untracked by git by design.

