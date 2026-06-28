# Paper Experiment Code Repository

This branch is the clean code program for the paper experiments. It is organized
so that a reader can understand the full experimental framework, install the
environment, choose the right component, run the scripts in a sensible order, and
modify the main experiment settings without digging through the full research
workspace.

## What This Branch Contains

- Python source code for data generation, model training, adversarial attacks,
  adversarial training, robustness metrics, timing/memory probes, and mechanism
  ablations.
- Documentation explaining how the folders fit together and how to run or modify
  the code.
- No raw datasets, checkpoints, generated figures, generated tables, logs, large
  archives, or paper manuscript files.

## Top-Level Structure

```text
00_shared/
  Shared solvers, losses, data-generation scripts, model-training scripts,
  reproducibility helpers, and framework-figure utilities.

01_adversarial_attack/
  Attack experiments for Burgers, Darcy Flow, Navier-Stokes, and cross-benchmark
  epsilon/objective/optimizer sweeps.

02_adversarial_training/
  Training scripts for clean, shifted, and adversarially trained models.

03_robustness_metrics/
  Post-training robustness diagnostics such as Jacobian spectral norms, solver
  singular-value summaries, and correlation/alignment analyses.

04_timing_memory/
  Runtime, wall-clock, solver-phase, memory, and rematerialization probes.

05_mechanism_and_ablations/
  Mechanistic and ablation experiments for loss geometry, loss-gradient paths,
  landscape slices, JVP/VJP spectra, and objective comparisons.
```

## Recommended Reading Order

1. `INSTALL.md` - create the Python environment and install dependencies.
2. `RUN_ORDER.md` - understand the full experiment order from data to figures.
3. `COMMANDS.md` - copy concrete shell commands for common runs.
4. `CONFIGURATION_GUIDE.md` - learn which arguments and paths to change.
5. `COMPONENT_GUIDE.md` - see what every major folder is responsible for.
6. `PYTHON_FILE_INDEX.md` - find the purpose of each Python file.

## Fast Start

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python 00_shared/reproducibility/setup_adv_robust_gpu_env.py --verify-only
```

Then inspect the arguments for any script before running it:

```powershell
python 01_adversarial_attack/burgers/fno/objective_variants/run_three_loss_objective_attack.py --help
```

For reproducibility, each script should be run from the repository root unless a
specific script states otherwise.

## Practical Rule

If you already have trained checkpoints and saved experiment outputs, start from
the plotting, summary, or analysis scripts. If you need a full rerun, follow the
order in `RUN_ORDER.md`: environment, data, clean training, attacks,
adversarial training, metrics, timing/memory, mechanism ablations, then figures.

