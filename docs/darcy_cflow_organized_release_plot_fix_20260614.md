# Darcy cflow organized-release plot layout fix - 2026-06-14

Status: complete for the organized release payload.

Target directory:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/
```

Actions:

- Removed all main-curve `*_work_hours.png` files from
  `figures/main_curves/`.
- Redrew main curves so only `epoch` and `wall_hours` axes remain.
- Truncated wall-hour plots to `0-4` wall-clock hours.
- Moved legends below the plots with enough reserved bottom margin, avoiding
  title/legend collisions.
- Used transparent lines with `alpha=0.58` and baseline line alpha `0.62`.
- Preserved the old `figures/diagnostic_existing/required_raw_figures_previous/`
  images.
- Added transparent-line replacements under
  `figures/diagnostic_existing/required_raw_figures_alpha/`.
- Added compact report-style overviews under `figures/polished_report/`.

Final organized-release counts after the fix:

- Main `work_hours` figures: `0`
- Main `wall_hours` figures: `36`
- Main `epoch` figures: `36`
- Required raw alpha figures: `12`
- Polished overview figures: `6`
- Total payload files: `552`
- Total payload bytes: `1,609,334,208`

Important data note:

- `random_clean` and `random_solver` are present in epoch plots.
- The source split/eval tables do not contain `wall_seconds` for
  `random_clean` or `random_solver`; wall-hour plots therefore keep those labels
  in the legend but do not fabricate wall-hour curves for them.

Implementation:

- `tools/fix_darcy_cflow_organized_release_plots_20260614.py`

Manifest:

- `outputs/darcy_cflow_timematched_organized_release_20260614/manifests/cflow_plot_layout_fix_20260614.json`
