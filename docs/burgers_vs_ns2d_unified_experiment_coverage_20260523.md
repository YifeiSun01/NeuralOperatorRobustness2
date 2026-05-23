# Burgers vs NS2D Unified Experiment Coverage Check

Updated: 2026-05-23 UTC

Status: local record inspection and coverage summary only. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started for this note.

## Question

Which parts of the proposed unified 1D Burgers versus 2D NS optimizer-mechanism experiment plan are already done, and which parts still need to be run?

The goal is to avoid rerunning experiments that already have evidence, while also identifying which missing pieces are required for a fair cross-PDE comparison.

## Sources Inspected

Observed from local documentation and files:

- `docs/burgers_vs_ns2d_unified_optimizer_mechanism_experiment_plan_20260523.md`
- `docs/three_loss_burgers_optimizer_findings_summary_20260521.md`
- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`
- `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`
- `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
- `docs/loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_result_20260516.md`
- `docs/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520.md`
- `docs/ns2d_optimizer_validation_offline_diagnostics_20260523.md`
- `docs/ns2d_recurrent_eps32_alpha10_baseline_overview_20260522.md`
- `docs/ns2d_recurrent_eps32_alpha10_delta_fft_analysis_20260522.md`
- `docs/ns2d_recurrent_eps32_alpha10_output_fft_dealias_analysis_20260522.md`
- `docs/ns2d_recurrent_eps32_alpha10_method_grouped_curves_20260522.md`
- `2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523/records/input_method_dirs.csv`

Important local-file caveat:

- Several Burgers Markdown records reference old `forensics/loss3_alpha_epsilon_core4_*` and `forensics/loss1_loss2_core4_*` directories that are not currently present in this local working tree.
- Therefore, some Burgers results are currently evidenced by Markdown summaries, not by locally available source CSVs. If paper-grade recomputation is needed, restore those raw artifacts from R2/git/another machine before regenerating tables.

## High-Level Answer

A large fraction of the exploratory evidence is already done. The missing part is not "everything"; the missing part is a strict, normalized, cross-PDE matched comparison.

Already strong enough to avoid rerunning immediately:

- Burgers p2q2 `loss3` core-four alpha/epsilon optimizer sweep.
- Burgers `loss1/loss2/loss3` baseline core-four curve comparison at `epsilon=4`, `alpha=0.4`.
- Burgers small-epsilon local/Jacobian diagnostic.
- Burgers ray-profile/local-to-finite-radius diagnostic.
- NS2D `eps32_alpha10` core-four baseline over `loss1`, `loss2`, and multiple `loss3` W/D/A modes.
- NS2D `eps8_alpha2p5` minimal core cases for `loss1/all_w`, `loss2/all_a_target_w`, and `loss3/all_w`.
- NS2D offline boundary-matched loss, early-to-final cosine, and rotation diagnostics for completed directories.
- NS2D `eps32_alpha10` FFT/spectral/dealiasing analysis.

Still missing for the clean cross-PDE claim:

- A unified normalized-epsilon table for both systems.
- Matched small-epsilon optimizer-ranking cases for NS2D, especially `eps4`, `eps2`, `eps1` with fixed `alpha/epsilon`.
- Boundary-arrival controlled alpha sweep, especially for NS2D.
- NS2D path replay / ray-scan diagnostics analogous to the Burgers ray-profile experiments.
- A unified cross-PDE spectral table using the same band definitions and normalization.

## Block-by-Block Coverage

### Block A. Minimal matched optimizer-ranking sweep

Plan requirement:

- systems: Burgers and NS2D;
- losses: `loss1`, `loss2`, `loss3`;
- methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`;
- `p=q=2`;
- samples: 10 first;
- normalized epsilon grid: anchor, smaller radii;
- fixed `alpha/epsilon`.

Burgers coverage:

- Done, partially matched:
  - `loss1/loss2/loss3` core-four baseline at `epsilon=4`, `alpha=0.4`, batch 100, steps 100 is documented in `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`.
  - `loss3` p2q2 core-four alpha/epsilon sweep over 20 settings is documented in `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`.
  - Original `loss1/loss2` speed lookup at `epsilon=8`, `alpha=0.3` exists, but it used `pgd`, `lp_steepest_pgd`, and `generalized_power`, not exactly all four core method labels.
- Not fully done:
  - No strict normalized-epsilon matched table against NS2D exists yet.
  - No full `loss1/loss2/loss3` core-four sweep over the same normalized epsilon radii as NS2D exists as one unified analysis product.

NS2D coverage:

- Done for `eps32_alpha10`:
  - `loss1/all_w`, `loss2/all_a_target_w`, `loss3/all_w`, `loss3/all_d_target_w`, and several mixed W/D/A modes all have four methods completed.
- Done for key `eps8_alpha2p5` minimal cases:
  - `loss1/all_w`, `loss2/all_a_target_w`, `loss3/all_w`, and `loss3/all_d_target_w` have four methods completed.
  - `loss3/w1_5_d6_9_target_w` was partial in the inspected offline record: only `raw_add` was complete.
- Not fully done:
  - Missing smaller NS2D epsilon cases such as `eps4_alpha1p25`, `eps2_alpha0p625`, `eps1_alpha0p3125`.
  - Missing normalized mapping to Burgers raw epsilons.

Conclusion for Block A:

- Do not rerun Burgers baseline or Burgers loss3 alpha/epsilon sweep immediately.
- Do run/finish the missing NS2D small-epsilon minimal cases, after computing normalized epsilon matching.

### Block B. Boundary-arrival matched sweep

Plan requirement:

- choose alpha so additive methods reach the boundary around 10, 25, and 50 steps;
- compare `loss1` and `loss3` under controlled boundary-arrival schedules.

Burgers coverage:

- Partially done:
  - Burgers `loss3` alpha/epsilon sweep includes many alpha values, and the summaries record boundary-arrival steps.
  - This can be reused to approximate boundary-arrival matching for `loss3`.
- Not fully done:
  - It was not designed explicitly as a matched 10/25/50-step boundary-arrival experiment across `loss1` and `loss3`.
  - Exact raw artifacts for some documented Burgers sweeps are not present locally.

NS2D coverage:

- Not done as a dedicated alpha sweep.
- Current NS2D evidence mainly has fixed-ratio pairs such as `32:10` and `8:2.5`.

Conclusion for Block B:

- This is still a real missing experiment, mainly for NS2D.
- Burgers may not need a new run if old alpha/epsilon artifacts are restored and sufficient, but NS2D needs targeted alpha-control runs.

### Block C. Direction stability and rotation diagnostics

Plan requirement:

- early-to-final cosine;
- `angle(delta_k, delta_{k-1})`;
- `angle(update_k, update_{k-1})`;
- `angle(delta_k, update_k)`.

Burgers coverage:

- Done in documented form:
  - Three-loss summary records GPI/replacement early-to-final behavior, including high similarity by early steps.
  - Combined curves include delta angle, direction angle, and boundary-ratio plots.
  - `loss3` GPI early-step comparison and final-delta similarity docs record direction similarity.
- Caveat:
  - Current local raw `forensics/loss3_alpha_epsilon_core4_*` CSVs are missing, so unified recomputation may require artifact restoration.

NS2D coverage:

- Done for completed `eps32_alpha10` and `eps8_alpha2p5` directories:
  - `early_to_final_cosine.csv` exists.
  - `boundary_matched_true_loss.csv` exists.
  - `gradient_rotation_summary.csv` exists.
- Caveat:
  - NS2D gradient rotation is based on saved representative sample trace (`step_sample_position=0`) for per-step arrays, not full-batch gradient traces.

Conclusion for Block C:

- Enough evidence exists for first-pass mechanism comparison.
- Missing only if we require full-batch per-step gradient rotation or a single unified cross-PDE CSV table.

### Block D. Path replay and ray-scan diagnostics

Plan requirement:

- evaluate `L(x + t delta_final)` along final directions;
- cross-evaluate final deltas under other losses;
- distinguish direction quality from path-following quality.

Burgers coverage:

- Done strongly:
  - Corrected Burgers ray-profile batch-20 experiment exists in `docs/loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_result_20260516.md`.
  - It explicitly separates local/small-radius directions from finite-radius endpoint winners.

NS2D coverage:

- Not done in the same sense.
- NS2D has final-state visualizations and FFT analysis, but not a proper ray scan along each method's final delta.

Conclusion for Block D:

- Do not rerun Burgers ray-profile first.
- NS2D ray-scan/path replay is one of the most important missing mechanism experiments.

### Block E. Spectral and spatial diagnostics

Plan requirement:

- Burgers: 1D FFT profile, high/low frequency fractions, smoothness/TV.
- NS2D: 2D FFT heatmaps, radial profiles, frequency-band metrics, model/solver/difference spectra.

Burgers coverage:

- Partially done:
  - `loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md` includes high-frequency ratio, first-derivative L2, total variation.
  - `loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520.md` includes spectral-magnitude cosine.
- Not fully done:
  - No current unified spectral table aligned to the NS2D radial-band definitions.

NS2D coverage:

- Done for `eps32_alpha10`:
  - final delta FFT;
  - model output FFT;
  - solver output FFT;
  - model-solver difference FFT;
  - no-cutoff and dealiasing analyses;
  - grouped spectral curves.
- Not fully done:
  - Same plots/metrics have not necessarily been generated for all `eps8` or future smaller epsilon runs.

Conclusion for Block E:

- NS2D eps32 spectral analysis does not need rerun now.
- Need a unified cross-PDE spectral summary table and, later, spectra for smaller NS2D epsilon runs.

### Block F. Solver-involvement ablation

Plan requirement:

- distinguish model-only, fixed solver target, moving solver target, detached solver, dictionary/A modes.

Burgers coverage:

- Burgers has analogues through `loss1`, `loss2`, and `loss3`, but it does not have the same W/D/A recurrent-frame modes.
- Existing records cover model movement, fixed/moving solver-related objectives, and local solver/model/residual Jacobian diagnostics.

NS2D coverage:

- Done for `eps32_alpha10`:
  - `loss1/all_w`;
  - `loss2/all_a_target_w`;
  - `loss3/all_w`;
  - `loss3/all_d_target_w`;
  - mixed `w1_5_d6_9_target_w`, `d1_5_w6_9_target_w`, `a1_5_d6_9_target_w`.
- Partially done for `eps8_alpha2p5`:
  - `loss1/all_w`, `loss2/all_a_target_w`, `loss3/all_w`, `loss3/all_d_target_w` complete;
  - mixed W/D mode partial in inspected offline record.

Conclusion for Block F:

- NS2D eps32 W/D/A ablation is already done enough for first interpretation.
- Do not expand all W/D/A modes at every epsilon yet. First finish minimal small-epsilon cases.

### Block G. Statistical confirmation

Plan requirement:

- stage 1: 10 samples;
- stage 2: 20 or 30 only for decisive settings;
- report mean/std/standard error and per-sample win rate.

Burgers coverage:

- Strong sample coverage already exists in several records:
  - batch 100 for major optimizer curves;
  - batch 20 for corrected ray-profile;
  - selected 5-sample local small-epsilon diagnostic.

NS2D coverage:

- Stage-1 coverage exists for 10 samples in completed attack batches.
- No 20/30-sample confirmation is evidenced for NS2D optimizer mechanism yet.

Conclusion for Block G:

- Do not increase sample count yet unless the small-epsilon and boundary-control results remain ambiguous.
- NS2D sample expansion is lower priority than small-epsilon and ray-scan experiments.

## What Is Already Done Enough To Not Rerun First

### 1D Burgers

- `loss3` p2q2 core-four alpha/epsilon sweep and method ranking summary.
- Baseline `loss1/loss2/loss3` core-four curves at `epsilon=4`, `alpha=0.4`.
- `loss1/loss2` original speed lookup showing generalized power reaches near-final values fastest.
- Small-epsilon local/Jacobian diagnostic.
- Corrected ray-profile/local-to-finite-radius diagnostic.
- Final-delta similarity and spectral/smoothness diagnostics in documented form.

Caveat: if exact source CSV regeneration is needed, restore old Burgers `forensics/loss3_alpha_epsilon_core4_*` artifacts because they are not all present locally now.

### 2D NS

- `eps32_alpha10` core-four baseline for `loss1`, `loss2`, and several `loss3` W/D/A modes.
- `eps8_alpha2p5` minimal cases for `loss1/all_w`, `loss2/all_a_target_w`, `loss3/all_w`, and `loss3/all_d_target_w`.
- Offline diagnostics for completed NS2D method directories: boundary-matched true loss, early-to-final cosine, and gradient rotation.
- `eps32_alpha10` final-delta, model-output, solver-output, and model-solver-difference FFT/dealiasing analyses.
- NS2D dictionary generation/stability and A-mode support are already established enough for attacks that use dictionary targets.

## What Still Needs To Be Done

Priority 1: make existing results comparable without new GPU runs.

- Build a unified CPU-only comparison table from existing docs/CSVs:
  - system;
  - loss role;
  - method;
  - raw epsilon;
  - normalized epsilon;
  - raw alpha;
  - alpha/epsilon;
  - final loss;
  - boundary threshold steps;
  - boundary-matched true loss;
  - early-to-final cosine;
  - rotation metrics;
  - spectral metrics.
- Restore missing Burgers raw artifacts if the unified table must be generated from source CSVs instead of Markdown summaries.

Priority 2: run missing NS2D small-epsilon minimal cases.

Recommended next NS2D cases, after current running jobs are safe/finished:

- `eps4_alpha1p25`;
- `eps2_alpha0p625`;
- `eps1_alpha0p3125`;
- optional `eps0p5_alpha0p15625`.

Minimal modes only:

- `loss1/all_w`;
- `loss2/all_a_target_w`;
- `loss3/all_w`.

Do not run all W/D/A modes for these smaller epsilons first.

Priority 3: NS2D boundary-arrival alpha control.

Run targeted alpha variations for:

- `loss1/all_w`;
- `loss3/all_w`;
- one large epsilon and one smaller epsilon.

Goal: check whether `steepest_add` wins because of path geometry rather than boundary-arrival speed.

Priority 4: NS2D path replay/ray scan.

For selected final deltas from `eps32_alpha10`:

- evaluate `L(x + t delta_final)` for `t in [0,1]`;
- compare method final directions under the same ray budget;
- cross-evaluate under loss1/loss2/loss3 if feasible.

This is the cleanest missing analogue to the Burgers ray-profile experiments.

Priority 5: unified spectral comparison.

- Convert Burgers and NS2D spectra into comparable low/mid/high normalized-frequency bands.
- Keep separate notes for 1D FFT versus 2D radial FFT because the geometry differs.

## Practical Recommendation

Do not rerun everything.

The next best sequence is:

1. Create the unified CPU-only coverage/comparison table from existing evidence.
2. Run only missing NS2D smaller-epsilon minimal cases.
3. Re-run the NS2D offline diagnostics on those new cases.
4. Add NS2D ray-scan/path-replay for the decisive large-epsilon cases.
5. Only if ambiguity remains, run a small boundary-arrival alpha-control sweep.

This gives the most direct answer to the mechanism question without wasting time on experiments already covered by the Burgers and NS2D records.
