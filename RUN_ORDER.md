# Experiment Run Order

The repository is organized by experimental role. A full reproduction should be
run in the order below. For a partial reproduction, start at the stage that
matches the files you already have.

## Stage 0 - Environment

1. Create and activate the Python environment.
2. Install dependencies.
3. Run:

```powershell
python 00_shared/reproducibility/setup_adv_robust_gpu_env.py --verify-only
```

## Stage 1 - Data Generation

Generate or verify the datasets before training or attack runs.

```text
00_shared/data_generation/burgers/
00_shared/data_generation/darcy_flow/
00_shared/data_generation/navier_stokes/
```

Use these scripts when you need fresh Burgers, Darcy Flow, or Navier-Stokes
datasets. If your datasets already exist, record their paths and skip to Stage 2.

## Stage 2 - Clean Model Training

Train baseline operator models before attack and adversarial-training studies.

```text
00_shared/model_training/burgers/
00_shared/model_training/darcy_flow/
00_shared/model_training/navier_stokes/
```

The trained checkpoints become inputs to:

- attack scripts in `01_adversarial_attack/`;
- adversarial-training initializations in `02_adversarial_training/`;
- robustness metrics in `03_robustness_metrics/`.

## Stage 3 - Adversarial Attack Experiments

Run attacks after clean checkpoints and test datasets are ready.

```text
01_adversarial_attack/burgers/
01_adversarial_attack/darcy_flow/
01_adversarial_attack/navier_stokes/
01_adversarial_attack/cross_benchmark/
```

Recommended order:

1. Run a small smoke test with one sample, small `--steps`, and a temporary
   `--output_dir`.
2. Run the objective-specific attack scripts for L1/L2/L3 or the target attack
   family.
3. Run epsilon sweeps only after a single-budget run is verified.
4. Generate attack summary figures from the saved outputs.

## Stage 4 - Adversarial Training

Run adversarial-training experiments after Stage 1 and Stage 2 are stable.

```text
02_adversarial_training/burgers/
02_adversarial_training/darcy_flow/
02_adversarial_training/navier_stokes/
```

Recommended order:

1. Train clean or fixed-shift baselines.
2. Train L1/L2/L3 adversarial variants.
3. Run matched attack evaluation on the trained checkpoints.
4. Save final checkpoints and summary CSV/JSON files for later plotting.

## Stage 5 - Robustness Metrics

Use these scripts after trained models and attack outputs exist.

```text
03_robustness_metrics/jacobian_spectral_norms/
03_robustness_metrics/model_solver_svd_similarity/
03_robustness_metrics/correlation_and_alignment/
```

These scripts connect the attack results to Jacobian, singular-value, and
alignment diagnostics.

## Stage 6 - Timing And Memory

Run these after the main algorithmic scripts are working, because timing and
memory measurements depend on the same solver/training paths.

```text
04_timing_memory/solver_phase_probes/
04_timing_memory/adversarial_training_epoch_times/
04_timing_memory/rematerialization_and_batch_capacity/
04_timing_memory/runtime_monitoring/
```

Use smaller settings first, then repeat with paper-scale settings.

## Stage 7 - Mechanism And Ablations

Run mechanism and ablation studies after the main attack/training results are
available.

```text
05_mechanism_and_ablations/loss3_geometry/
05_mechanism_and_ablations/loss_landscape/
05_mechanism_and_ablations/objective_mechanism/
05_mechanism_and_ablations/loss_gradient_paths/
05_mechanism_and_ablations/jvp_vjp_spectrum/
```

These scripts explain why the main objectives behave differently and support the
paper's interpretation figures.

## Stage 8 - Figure/Table Production

Plotting scripts are spread near the experiments they summarize. As a rule:

- attack figures live under `01_adversarial_attack/...`;
- adversarial-training figures live under `02_adversarial_training/...`;
- timing figures live under `04_timing_memory/...`;
- mechanism figures live under `05_mechanism_and_ablations/...`.

Run plotting after the matching raw output files exist. Use
`PYTHON_FILE_INDEX.md` to find candidate plotting files.
