# Command Examples

Run commands from the repository root. The examples below are templates: replace
data paths, checkpoint paths, budgets, and output directories with the files on
your machine.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python 00_shared/reproducibility/setup_adv_robust_gpu_env.py --verify-only
```

## Inspect A Script Before Running It

```powershell
python path\to\script.py --help
```

Use this for every unfamiliar script. Many files in this repository are research
drivers with several required paths.

## Burgers Objective Attack

```powershell
python 01_adversarial_attack/burgers/fno/objective_variants/run_three_loss_objective_attack.py `
  --case burgers `
  --loss_type loss3 `
  --objective_variant solver_integrated `
  --attack_method pgd `
  --norm l2 `
  --epsilon 0.1 `
  --alpha 0.01 `
  --steps 100 `
  --index 0 `
  --burgers-test-path "PATH\TO\burgers_test_data.h5" `
  --burgers-torch-checkpoint "PATH\TO\burgers_fno_checkpoint.pt" `
  --output_dir "runs\burgers_loss3_attack"
```

Important controls:

- `--loss_type`: switch among the target losses used in the paper.
- `--attack_method`: choose PGD, steepest PGD, power iteration, or generalized
  power methods where supported.
- `--epsilon`, `--alpha`, `--steps`: attack radius, step size, and iteration
  count.
- `--random_start`: enable random PGD initialization where appropriate.

## Burgers Final-Figure Pipeline Smoke Test

```powershell
python 01_adversarial_attack/burgers/fno/p2q2_final_figures/run_burgers_p2q2_full_pipeline.py `
  --dry-run `
  --skip-r2-upload `
  --skip-git-push
```

Keep `--skip-r2-upload` and `--skip-git-push` on unless you intentionally want
external upload or repository updates from the script.

## Darcy Flow Attack Scripts

Start by checking the exact arguments:

```powershell
python 01_adversarial_attack/darcy_flow/binary_cflow/run_darcy_attack20_52datasets_50samples_20260612.py --help
python 01_adversarial_attack/darcy_flow/epsilon_sweeps/darcy_sir20_budget_sweep.py --help
```

Use the binary C-Flow scripts for the main Darcy attack experiments and the
epsilon-sweep scripts for budget scaling plots.

## Navier-Stokes Attack Scripts

```powershell
python 01_adversarial_attack/navier_stokes/recurrent_fno2d/attack_ns2d_recurrent_core4.py --help
python 01_adversarial_attack/navier_stokes/periodic_warp_alignment/plot_ns2d_alignment_before_after_diff_20260525.py --help
```

Navier-Stokes scripts are the most expensive. First run one sample and a small
number of steps, then scale to the paper setting.

## Adversarial Training

```powershell
python 02_adversarial_training/burgers/generalization_52_datasets/run_generalization.py --help
python 02_adversarial_training/darcy_flow/generalization_52_datasets/run_darcy_five_model_one_batch_tag_20260609.py --help
python 02_adversarial_training/navier_stokes/loss3_feasibility_probe/probe_ns2d_loss3_adversarial_training_batch_20260608.py --help
```

Training scripts usually need:

- a training dataset path;
- a validation or test dataset path;
- an output checkpoint directory;
- a loss/objective choice;
- attack budget and step settings for adversarial training.

## Robustness Metrics

```powershell
python 03_robustness_metrics/jacobian_spectral_norms/run_burgers_full1024_svd_attack_correlation_20260611.py --help
python 03_robustness_metrics/model_solver_svd_similarity/build_darcy_solver_model_subspace_similarity_20260615.py --help
python 03_robustness_metrics/correlation_and_alignment/summarize_darcy_final_correlation_alignment_20260615.py --help
```

Run these after model checkpoints and attack summaries exist.

## Timing And Memory

```powershell
python 04_timing_memory/solver_phase_probes/benchmark_darcy_vs_burgers_solver_runtime_20260611.py --help
python 04_timing_memory/adversarial_training_epoch_times/time_darcy_five_model_one_dataset_tag20_20260609.py --help
python 04_timing_memory/rematerialization_and_batch_capacity/compare_ns_batch_sizes.py --help
```

Use the same model and solver settings as the main experiments, then compare the
reported wall-clock time and memory usage.

## Mechanism And Ablation Studies

```powershell
python 05_mechanism_and_ablations/loss3_geometry/plot_loss3_mechanism_validation_summary.py --help
python 05_mechanism_and_ablations/loss_landscape/plot_loss3_alpha_epsilon_core4_visuals.py --help
python 05_mechanism_and_ablations/jvp_vjp_spectrum/probe_ns2d_loss3_exact_mechanism_20260623.py --help
```

These scripts support explanation figures rather than the main benchmark table.
