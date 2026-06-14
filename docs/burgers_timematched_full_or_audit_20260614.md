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
