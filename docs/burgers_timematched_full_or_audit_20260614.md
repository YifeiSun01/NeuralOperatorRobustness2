# Burgers Time-Matched Full Audit - 2026-06-14

This pass audited and packaged existing Burgers selected-worktime artifacts. It
did not rerun self-training, attacks, or SVD/Jacobian analysis.

Output bundle:

- `outputs/burgers_timematched_full_or_audit_20260614/`
- `figures/`: 12 PNG curves, limited to RMSE and relative L2.
- `data/`: copied/normalized clean, attack, robustness, SVD, work-clock, and
  curve CSV/JSON files.
- `reports/burgers_timematched_full_audit_report.md`: natural-language audit
  and conclusions.
- `manifests/audit_manifest.json`: artifact and R2-availability audit.

Observed evidence:

- Clean 52-dataset table:
  `forensics/burgers_six_model_selected_worktime_summary_20260613/clean_52dataset_six_models_selected_worktime.csv`
- Selected-worktime attack table:
  `forensics/burgers_six_model_selected_worktime_summary_20260613/attack_52dataset_six_models_selected_worktime_long.csv`
- 25-sample robustness/SVD/Jacobian table:
  `forensics/burgers_six_model_selected_worktime_summary_20260613/robustness_25sample_six_models_selected_worktime.csv`
- SVD sample manifest:
  `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/sample_manifest.csv`

Main clean-generalization result from the 52-dataset table: `loss3` has the
lowest mean clean generalization RMSE (`0.0120496`) and relative L2 (`0.0236437`).
`random_solver_y` is second among the six models on clean generalization
(`0.0170637` RMSE), while `random_clean_y` is a poor generalization control
(`0.0947992` RMSE).

Robustness/SVD result from the 25-sample joined table: `loss3` has the lowest
25-sample attack loss increase mean (`0.00306321`) and lowest error spectral
norm mean (`1.27153`). Error spectral norm correlates with attack loss increase
at Pearson `0.7216` and Spearman `0.8393`; `J_error.T @ clean_error`
(`bias_gradient_norm`) correlates at Pearson `0.8572` and Spearman `0.8928`.
The attack delta/top-error-singular-vector cosine is highest for
`random_clean_y` and lowest for `loss3`, so the single top singular direction is
not by itself a complete explanation of attack direction.

Important audit caveats:

- Strict logged work-clock was computed as `attack_wall_sec + train_wall_sec`,
  excluding evaluation/plot/upload. Under that definition, the existing endpoints
  cover `loss1=13.816h`, `loss2=9.890h`, `loss3=7.427h`,
  `random_clean_y=2.902h`, and `random_solver_y=5.690h`. The existing
  selected-worktime bundle is therefore useful locally, but it is not a strict
  equal-work-clock endpoint for every method.
- The selected-worktime 52-dataset attack CSV contains `baseline`,
  `random_clean_y`, and `random_solver_y`, but not `loss1`, `loss2`, or `loss3`.
  The six-model robustness conclusion should therefore rely on the 25-sample
  joined table unless a full selected-worktime 52-dataset attack is run later.
- R2 upload was completed after credentials were supplied through the shell
  environment. Verified prefix:
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/burgers_timematched_full_or_audit_20260614`.
  R2 listing showed 39 files plus 4 directory markers.

Determination after local and R2 audit: this is not final. It needs follow-up
work before we can claim a strict six-model equal-work-clock experiment.

Required follow-up:

- Select loss1/loss2 checkpoints by the loss3 work-clock target instead of using
  their final over-budget endpoints. From local `train_steps.csv` logs, the
  nearest epochs to the loss3 target (`7.427461557h`) are loss1 epoch `2757`
  and loss2 epoch `1499`; checkpoint candidates are loss1 epoch `2700/2800`
  and loss2 epoch `1455/1500`.
- Continue random baselines to the loss3 work-clock target. Using latest local
  per-epoch rates, `random_clean_y` needs about `13,100` additional epochs
  beyond epoch `8000`, and `random_solver_y` needs about `1,860` additional
  epochs beyond epoch `6000`.
- Run/evaluate strict selected-work-clock six-model 52-dataset clean and P2Q2
  attack tables. The current selected-worktime 52-dataset attack table is
  missing `loss1`, `loss2`, and `loss3`.
- Rebuild the summary tables and RMSE/relative-L2 figures from those strict
  selected endpoints.

SVD/Jacobian does not need to be the first expensive rerun. The existing
25-sample joined table is useful evidence, but its checkpoint/work-clock
alignment should be treated as audit evidence until strict endpoint selection is
finished.

Follow-up launch status:

- Added `tools/run_burgers_timematched_strict_random_continuations_20260614.sh`
  to run the strict random-baseline continuations from the existing optimizer
  checkpoints.
- Smoke resume passed for `random_clean_y` epoch `8000 -> 8001` and
  `random_solver_y` epoch `6000 -> 6001`; both smoke checkpoints contain
  `optimizer_state_dict`.
- Formal continuation started at `2026-06-14T00:49:40Z`, PID `86717`, log root
  `adversarial_training_runs/burgers_timematched_strict_random_continuations_20260614_logs`.
- Ten-minute monitoring was clean: solver-y reached epoch `6130`, train/eval/
  optimizer CSVs were growing, GPU memory stayed around `6056 MiB`, and no
  traceback appeared. Active tailing was then stopped so the background job can
  continue.
- User decision after launch: keep `random_clean_y` at epoch `8000`; do not
  continue it to epoch `21100`. A sentinel directory was created at
  `adversarial_training_runs/burgers_wideparam_random_field_clean_y_21100ep_continue_20260613/`
  to prevent accidental clean-y continuation.
- Added and started `tools/watch_burgers_solver7860_then_postprocess_20260614.sh`.
  It waits for `random_solver_y` epoch `7860`, then automatically runs
  clean+attack+25-sample SVD/Jacobian+postprocess evaluation for
  `random_clean_y=8000` and `random_solver_y=7860`, rebuilds selected summaries,
  regenerates audit/curve outputs with `--workclock-xmax 8.0`, and uploads those
  outputs to R2 using either process `R2_*` credentials or the temporary
  `/tmp/rclone-r2/rclone.conf` remote.
  The work-clock plots are intentionally capped near 8 hours, not at the minimum
  common logged time; shorter random-clean curves stop before the right edge.

Plot refresh after final postprocess:

- Regenerated the audit curve bundle using
  `forensics/burgers_six_model_solver7860_clean8000_summary_20260614` and
  `forensics/burgers_random_solver7860_clean8000_full_suite_20260614`.
- Replaced compact `dXX` subplot titles with descriptive dataset names from
  `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/manifest.json`.
- Renamed curve outputs and labels from work-clock to wall-clock, kept the 8h
  x-axis cap, and recorded that curves are raw evaluation points with no moving
  average or smoothing.
- Increased legend size and made curve lines partially transparent.
- Added a second figure set under `figures/no_random_clean/` that excludes
  `random_clean_y` so the other methods are not visually compressed.
- Added log-y copies under `figures/log_y/` and
  `figures/log_y/no_random_clean/`; these keep the same raw evaluation points,
  descriptive dataset titles, wall-clock 8h x-axis cap, transparency, and
  enlarged legends.
- Synced the refreshed output bundle to R2 and removed stale `workclock` files
  from the remote output prefix.

Dense image-only six-model refresh:

- Generated the missing dense six-model P2Q2 comparison bundle for
  `random_clean_y=8000` and `random_solver_y=7860`:
  `visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614/`.
- Recomputed only the latest `random_solver_y=7860` dense attack trace; reused
  existing baseline/loss1/loss2/loss3 traces and reused the existing
  `random_clean_y=8000` trace.
- Each of the five group figures has one test sample plus five generalization
  samples, six model columns, delta/initial-condition/model-vs-solver/error
  panels, and bottom-row sample-wise attack-loss progression curves.
- Generated 10 PNGs total: log-y and linear-y attack-loss progression variants
  for each of five groups.
- Dedicated note: `docs/burgers_solver7860_dense_image_only_bundle_20260614.md`.

Dense four-variant refresh:

- Re-rendered the dense comparison bundle from existing traces only; no attack
  or training was rerun.
- For each of five groups, generated four image variants: all models with log-y
  loss curves, all models with linear-y loss curves, no-random-clean with log-y
  loss curves, and no-random-clean with linear-y loss curves.
- All dense model headers include epoch counts: baseline e500, loss1 e8000,
  loss2 e2000, loss3 e1000, random clean Y e8000, and random solver Y e7860.
- Added `group05`, a curated loss3-best group selected from the existing
  `group00` through `group04` dense traces. A candidate had to have `loss3` as
  the strict winner for both final attacked MSE and attack loss increase; the
  final group keeps one test sample and five generalization samples with the
  largest relative `loss3` margins.
- The image-only bundle now contains 24 PNGs and 0 non-PNG files. R2
  `rclone size --json` verified 24 image objects and 63,769,624 bytes.
- R2 trace/data verification for the dense root returned 27 objects and
  203,143,689 bytes.
- The no-random-clean version removes `random_clean_y` from the top model
  columns and the bottom loss curves, so the other methods use y-limits not
  dominated by the random-clean failure mode.
- Added a loss123-only dense variant that removes baseline, `random_clean_y`,
  and `random_solver_y`, leaving only loss1/loss2/loss3 columns and bottom
  loss curves. It has both log-y and linear-y versions for every group.
- The image-only bundle now contains 36 PNGs and 0 non-PNG files. R2
  `rclone size --json` verified 36 image objects and 89,008,328 bytes.
- R2 trace/data verification for the dense root returned 27 objects and
  203,170,665 bytes after the summary JSON update.

Polished report refresh:

- Generated the missing polished-report visualization set for `loss1`, `loss2`,
  `loss3`, `random_clean_y=8000`, and `random_solver_y=7860`.
- Full report bundle:
  `visualizations/burgers_solver7860_clean8000_polished_reports_20260614/`.
- PNG-only bundle:
  `visualizations/burgers_solver7860_clean8000_polished_reports_image_only_bundle_20260614/`.
- Copied all polished-report PNGs into the final audit output at
  `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/figures/polished_reports/`.
- Each model has 10 PNGs: attack-loss buckets, RMSE/Relative-L2 heatmap+line,
  checkpoint RMSE/Relative-L2 heatmap+line, Delta FFT heatmap+spectra, and raw/
  MA25 high-transparency max-five RMSE/Relative-L2 line plots.
- The final audit output now contains 98 PNGs total, including 50 polished-report
  PNGs.
- Dedicated note: `docs/burgers_solver7860_polished_reports_20260614.md`.

Organized release:

- Built a non-destructive, layered release folder at
  `outputs/burgers_solver7860_clean8000_organized_release_20260614/`.
- The release gathers the final audit output, summary tables, dense six-model
  attack figures/traces, random-model full-suite outputs, polished report
  figures/data, logs, and source-code references into one reviewable tree.
- Verified from `MANIFEST.json` after adding dense `group05`: 617 files,
  1,417,514,791 bytes, 154 PNG, 85 CSV, 183 JSON, 137 NPZ, and 0 missing
  expected inputs.
- R2 verification for the organized release prefix returned 617 objects and
  1,417,514,791 bytes after the `group05` addition.
- The organized release includes the complete dense image-only bundle as
  `08_dense_image_only_bundle_full_copy/`.
- Dedicated note:
  `docs/burgers_solver7860_clean8000_organized_release_20260614.md`.

Organized release loss123-only update:

- Rebuilt the organized release after adding the loss123-only dense panels.
- Verified from `MANIFEST.json`: 646 files, 1,468,076,040 bytes, 178 PNG, 85
  CSV, 183 JSON, 137 NPZ, 35 log files, 8 Python scripts, and 0 missing
  expected inputs.
- R2 verification for this organized-release update returned 646 objects and
  1,468,076,040 bytes.

Final audit output dense-copy update:

- Copied the full dense image-only bundle into the solver7860 final audit
  output:
  `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/figures/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614/`.
- The final audit output now has 134 PNG figures, including the 36 dense
  image-only PNGs under that nested bundle.
- R2 verification for
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/burgers_timematched_solver7860_clean8000_audit_20260614`
  returned 161 objects and 576,834,254 bytes.
- R2 verification for the nested dense final-audit folder returned 36 objects
  and 89,008,328 bytes.

Final audit data completeness update:

- Added compact historical SVD25 reuse3 tables under
  `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/historical_svd_attack25_reuse3/`.
  This includes the 25-sample manifest, attack step metrics, model-pair
  reductions, cross-model error-subspace similarities, config, and runtime
  summary.
- Added biased-local-direction tables under
  `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/biased_local_direction/`.
  This includes biased direction metrics, correlations, quantile summaries,
  direction pairwise angle summaries, ATB/SVD correlation ranking, config, and
  summary JSON.
- The final audit output now contains 174 files and 577,938,944 bytes locally.
- R2 verification for the final audit output returned 174 objects and
  577,938,944 bytes.
- R2 verification for `data/historical_svd_attack25_reuse3/` returned 6 objects
  and 790,734 bytes.
- R2 verification for `data/biased_local_direction/` returned 7 objects and
  313,956 bytes.

Organized release data completeness update:

- Rebuilt the organized release after adding those two compact SVD/bias table
  families.
- Verified from `MANIFEST.json`: 661 files, 1,469,211,584 bytes, 178 PNG, 94
  CSV, 187 JSON, 137 NPZ, 37 log files, 8 Python scripts, and 0 missing
  expected inputs.
- R2 verification for the organized-release update returned 661 objects and
  1,469,211,584 bytes.

Recovered prior metrics update:

- Checked R2 and local sources for the missing metric families instead of
  rerunning compute.
- Recovered old4 52-dataset P2Q2 attack raw payloads from
  `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608/`.
- Recovered the full old4 SVD25 reuse3 payload from
  `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/`,
  expanding `data/historical_svd_attack25_reuse3/` from compact CSV/JSON into
  501 files including raw NPZ payloads, top100 singular values, SVD summaries,
  top-k subspace similarities, attack joins, and correlations.
- Copied the six-model latest/corrected metric summary bundle from
  `forensics/burgers_six_model_latest_wideparam_summary_20260613/`.
- Copied the random solver7860/clean8000 raw attack/SVD suite into
  `data/recovered_prior/random_solver7860_clean8000_full_suite_20260614/`.
- Added recovered six-model 52-dataset attack tables:
  `data/attack_52dataset_six_models_recovered_full_long.csv` and
  `data/attack_52dataset_six_models_recovered_full_wide.csv`.
- Added recovered six-model common top20 singular-value table:
  `data/singular_values_top20_six_models_recovered_long.csv`.
- Added `data/paired_tests_loss3_vs_other_models.csv` with paired t-test,
  one-sided p-values, Wilcoxon p-values, and BH-FDR q-values for clean
  generalization and 25-sample robustness metrics.
- Added explicit coverage/gap audit:
  `data/missing_metric_coverage_audit.csv` and
  `data/missing_metric_coverage_audit.json`.
- Final audit R2 verification after this recovery returned 1055 objects and
  4,004,966,533 bytes.
- Rebuilt organized release after this recovery. `MANIFEST.json` now reports
  1230 files, 4,301,766,869 bytes, 178 PNG, 155 CSV, 446 JSON, 378 NPZ, 37 log
  files, 9 Python scripts, and 0 missing expected inputs.
- Organized release R2 verification after this recovery returned 1230 objects
  and 4,301,766,869 bytes.
- Remaining known gap: old4 has top100 SVD values/subspaces, but recovered
  random `random_clean_y`/`random_solver_y` SVD only has top20. The common
  six-model singular-value table is therefore top20.
