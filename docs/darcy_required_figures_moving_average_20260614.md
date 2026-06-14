# Darcy Required Figures Moving Average - 2026-06-14

## Status

Complete for the currently archived required-figures CSVs.

Observed source files:

- `outputs/darcy_sir20_required_figures_only_20260614/data/six_method_common_range_eval_split_summary.csv`
- `outputs/darcy_sir20_required_figures_only_20260614/data/six_method_common_range_eval_metrics.csv`

Action:

- Added `tools/build_darcy_required_figures_moving_average_20260614.py`.
- Generated a separate centered moving-average figure set using a `51` epoch
  window.
- The original unsmoothed figures were not overwritten.

Output:

- `outputs/darcy_sir20_required_figures_only_20260614/figures_moving_average_ma51/`

Notes:

- These are derived visualization figures only. The source CSV values are not
  changed.
- The current archived data have a common range of epoch `1100` and
  work-clock `1128.9048905035306` seconds. After the fresh random-source runs
  finish and the required CSVs are rebuilt to the longer range, this same script
  can regenerate matching moving-average figures for the longer epoch span.
