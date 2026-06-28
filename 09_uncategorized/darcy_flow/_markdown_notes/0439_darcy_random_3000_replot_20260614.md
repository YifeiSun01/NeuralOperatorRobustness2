# Darcy Random 3500 Replot Record - 2026-06-14

## Status

Running under supervisor.

Observed from local files on 2026-06-14:

- `random_clean` old formal curve data only reached epoch `1100`:
  `adversarial_training_runs/darcy_binary_random_binary_fixed_y_1100ep_full50_20260613_random_binary_source_1100/darcy/train_steps.csv`.
- `random_solver` old formal curve data only reached epoch `1100`:
  `adversarial_training_runs/darcy_binary_random_binary_solver_y_1100ep_full50_20260613_random_binary_source_1100/darcy/train_steps.csv`.
- Both old epoch-1100 checkpoints existed, but neither contained
  `optimizer_state_dict`, so they were not used for an optimizer-continuous
  1100-to-3000 continuation.

## Fresh And Continuation Runs

The fresh 0-to-3000 random-source Darcy base runs are:

- `adversarial_training_runs/darcy_binary_random_binary_fixed_y_3000ep_full50_20260614_random_binary_source_3000_supervised/`
- `adversarial_training_runs/darcy_binary_random_binary_solver_y_3000ep_full50_20260614_random_binary_source_3000_supervised/`

After those reach epoch `3000`, the watcher starts optimizer-continuous
continuation runs to epoch `3500`:

- `adversarial_training_runs/darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised/`
- `adversarial_training_runs/darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised/`

Supervisor programs:

- `darcy_random_clean_3000`
- `darcy_random_solver_3000`
- `darcy_random_clean_3500_continue`
- `darcy_random_solver_3500_continue`
- `darcy_random_3000_replot`

Logs:

- `outputs/darcy_random_3000_20260614/logs/random_clean_3000_supervisor.log`
- `outputs/darcy_random_3000_20260614/logs/random_solver_3000_supervisor.log`
- `outputs/darcy_random_3000_20260614/logs/replot_waiter.log`

Status files:

- `outputs/darcy_random_3000_20260614/replot_status.json`
- `outputs/darcy_random_3000_20260614/replot_status.md`

## Plotting

`tools/build_darcy_required_six_method_figures_20260614.py` now reads the fresh
3000-epoch random base runs plus the 3000-to-3500 continuation runs, and caps
epoch plots at epoch `3500`. After both random continuation runs finish,
`tools/wait_and_replot_darcy_random_3000_20260614.py` checks that the base and
continuation final checkpoints contain `optimizer_state_dict` and then rebuilds:

- `outputs/darcy_sir20_required_figures_only_20260614/figures/`

## Current Evidence

Observed shortly after launch:

- GPU path verified with PyTorch CUDA available on Tesla V100-SXM2-32GB.
- Supervisor reported both random runs and the replot waiter as `RUNNING`.
- Early train CSV rows were written for both random runs.
- The replot status file reported `waiting_base_3000` with nonzero epochs for
  both base runs and a final target epoch of `3500`.

## Remaining Work

- Wait for both fresh base runs to reach epoch `3000`.
- Let the continuation runs reach epoch `3500`.
- Confirm final checkpoint optimizer state is present for base and continuation
  checkpoints.
- Confirm the required figures are regenerated with epoch axes reaching `3500`.
