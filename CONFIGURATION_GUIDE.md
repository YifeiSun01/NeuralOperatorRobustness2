# Configuration Guide

Most scripts are plain Python entry points. The safest workflow is:

1. run `python path\to\script.py --help`;
2. copy the command into a run log;
3. replace paths and budgets explicitly;
4. run a small smoke test;
5. increase the dataset size, sample count, or attack steps.

## Common Path Arguments

Typical arguments used across the codebase:

- `--output_dir` or `--out-dir`: where results, plots, CSV files, and logs go.
- `--burgers-test-path`: Burgers test dataset.
- `--burgers-torch-checkpoint`: Burgers PyTorch checkpoint.
- `--ns-test-path`: Navier-Stokes test dataset.
- `--ns-torch-checkpoint`: Navier-Stokes PyTorch checkpoint.
- dataset/checkpoint arguments with Darcy-specific names in Darcy scripts.

Keep generated outputs outside the source tree when running large jobs, for
example:

```powershell
--output_dir "D:\research_runs\paper_reproduction\burgers_loss3_attack"
```

## Attack Budget Arguments

The most important attack controls are:

- `--epsilon`: norm-ball radius / attack budget.
- `--alpha`: PGD step size.
- `--steps`: number of attack iterations.
- `--norm`: usually `l2` or `linf`.
- `--random_start`: enables random initialization when supported.
- `--seed`: makes runs repeatable when supported.

For a smoke test, use a small `--steps` value and one input index. For paper
figures, use the matched budget and step count specified by the experiment.

## Objective And Loss Arguments

Common objective controls:

- `--loss_type`: selects L1, L2, L3, or a script-specific loss name.
- `--objective_variant`: selects solver-integrated or non-solver-integrated
  variants where the script supports it.
- `--attack_method`: selects PGD, steepest PGD, power iteration, or generalized
  power methods where implemented.

Before changing a loss name, inspect the script for accepted choices. Different
experiment families use slightly different naming conventions.

## System-Specific Controls

Burgers:

- viscosity, often exposed as `--burgers-nu`;
- spatial resolution, often `--burgers-nx`;
- final time and step size, often `--burgers-t-final` and `--burgers-dt`;
- model family, usually FNO or DeepONet depending on the folder.

Darcy Flow:

- coefficient-field generation settings;
- dataset scale or coefficient-flow magnitude;
- binary C-Flow settings for the main Darcy experiments;
- physics-loss or loss-4 variants in the ablation folders.

Navier-Stokes:

- grid size, input/output frame count, viscosity, and time step;
- recurrent FNO checkpoint path;
- periodic-warp or external-forcing settings for special studies.

## Turning Expensive Jobs Into Smoke Tests

Change these first:

- sample count or dataset count;
- `--steps`;
- number of epochs;
- batch size;
- output directory;
- device, for example CPU for logic checks and GPU for final runs.

Do not start a full adversarial-training or Navier-Stokes sweep until a small run
has written the expected output files.

## Disabling External Side Effects

Some pipeline scripts contain optional upload or git-update steps. Use the
available skip flags when testing:

```powershell
--dry-run --skip-r2-upload --skip-git-push
```

If a script does not have these flags, inspect the file before running it on a
new machine.

## How To Modify A Component

Data generation:

- modify the dataset size, PDE parameter, random seed, and output path;
- keep the generated format compatible with downstream training scripts.

Model training:

- modify model width/depth/modes, optimizer settings, batch size, epochs, and
  checkpoint path;
- keep the checkpoint naming consistent with attack scripts.

Adversarial attack:

- modify `epsilon`, `alpha`, `steps`, norm, loss type, objective variant, and
  sample index;
- keep output directories separated by system, model, loss, and budget.

Adversarial training:

- modify the inner attack budget, training objective, checkpoint initialization,
  and evaluation cadence;
- record both training loss and matched post-training attack results.

Metrics/plots:

- modify input summary paths and figure export paths;
- keep style settings consistent across figures used in the paper.

