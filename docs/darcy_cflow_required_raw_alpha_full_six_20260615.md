# Darcy CFlow Required-Raw Alpha Figure Fix - 2026-06-15

Status: complete.

Target release:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/
```

Observed issue:

- The previous transparent alpha folder contained wall-axis alpha figures.
- In the source tables, `wall_seconds` is present for `loss1`, `loss2`,
  `loss3`, and `physics`, but is `NaN` for `random_clean` and
  `random_solver`.
- Therefore those wall-axis alpha figures could not show all six trained
  methods.

Fix:

- Preserved the original non-alpha raw figures in:
  `figures/diagnostic_existing/required_raw_figures_previous/`.
- Moved the old wall-axis alpha files to:
  `figures/diagnostic_existing/required_raw_figures_alpha_retired_wall_axis_20260615/`.
- Regenerated the required alpha set in:
  `figures/diagnostic_existing/required_raw_figures_alpha/`.
- The regenerated alpha figures mirror the previous raw figure set using
  epoch and work-clock axes, with `alpha=0.58` for method curves and a gray
  dashed baseline horizontal line.

Validation:

- Source tables:
  `data/source_tables/six_method_common_range_eval_split_summary.csv`
  and `data/source_tables/six_method_common_range_eval_metrics.csv`.
- Expected method curves:
  `loss1`, `loss2`, `loss3`, `Physics Loss`, `random_clean`,
  `random_solver`.
- Internal source-table key `physics` is displayed as `Physics Loss`.
- Generated PNGs: `12`.
- Retired old wall-axis alpha PNGs: `6`.
- Manifest validation result:
  `all_panels_have_six_methods = true`.
- No incomplete panels were reported.

Manifest:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/manifests/required_raw_alpha_full_six_20260615.json
```

Implementation:

```text
tools/replot_darcy_required_raw_alpha_figures_20260615.py
```
