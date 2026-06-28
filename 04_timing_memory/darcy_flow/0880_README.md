# Epsilon Sweep Raw Attack Sources - 20260615

This directory stores raw attack outputs for the Darcy CFlow epsilon-budget
sweep used by `epsilon_sweep_summary.md`.

Included usable reruns:

| epsilon label | epsilon fraction | source |
| --- | --- | --- |
| 0p01x | 0.00025 | `eps_0p01x/data/robustness_attack_52datasets_samples.csv` + `robustness_deltas/` |
| 0p05x | 0.00125 | `eps_0p05x/data/robustness_attack_52datasets_samples.csv` + `robustness_deltas/` |
| 0p1x | 0.0025 | `eps_0p1x/data/robustness_attack_52datasets_samples.csv` + `robustness_deltas/` |
| 0p2x | 0.005 | `eps_0p2x/data/robustness_attack_52datasets_samples.csv` + `robustness_deltas/` |
| 0p5x | 0.0125 | `eps_0p5x/data/robustness_attack_52datasets_samples.csv` + `robustness_deltas/` |
| 5x | 0.125 | `eps_5x/data/robustness_attack_52datasets_samples.csv` + `robustness_deltas/` |
| 10x | 0.25 | `eps_10x/data/robustness_attack_52datasets_samples.csv` + `robustness_deltas/` |

The 1x baseline was not rerun for the final analysis. It is taken from the
existing final attack50 source table:
`source_tables/robustness_attack_52datasets_samples.csv`, filtered to the fixed
25 samples x 7 models used by the residual Jacobian table.

`logs/eps_1x_interrupted_excluded.log` is kept only to document the interrupted
accidental rerun. It is not used by any final CSV or Markdown conclusion.
