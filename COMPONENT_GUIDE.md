# Component Guide

This file explains what each major component does and how it connects to the
paper experiments.

## 00_shared

Shared code used by multiple experiment families.

### core_losses_and_solvers

Contains reusable loss and solver logic. These files define the common
mathematical pieces that attack and training scripts call into:

- solver-integrated loss construction;
- attack loss helpers;
- PDE solver wrappers;
- common adversarial-training routines.

Change this folder carefully because edits can affect Burgers, Darcy, and
Navier-Stokes experiments at the same time.

### data_generation

Creates synthetic PDE datasets:

- `burgers/`: one-dimensional Burgers data, Gaussian random fields, and batched
  generation utilities;
- `darcy_flow/`: Darcy coefficient-field and response generation;
- `navier_stokes/`: Navier-Stokes dictionary, real-initial-condition, stability,
  and repair utilities.

Run this before model training when the required datasets are not already
available.

### framework_diagrams

Creates schematic framework figures and graph visualizations. These are for
paper explanation figures, not for training models.

### model_training

Contains clean model training scripts:

- Burgers FNO and DeepONet training;
- Darcy Flow FNO2d training;
- Navier-Stokes recurrent or time-dependent FNO2d training.

Outputs from this folder are the checkpoints used by attack, adversarial
training, and metric scripts.

### reproducibility

Environment and reproducibility helpers. Start here when moving the code to a
new machine.

## 01_adversarial_attack

Attack-only experiments on fixed trained models.

### burgers

Includes Burgers FNO and DeepONet attack experiments, viscosity-specific runs,
objective-variant attacks, local SVD/Jacobian analyses, and final figure
pipelines.

Use this section for Burgers attack figures and loss-objective comparisons.

### darcy_flow

Includes binary C-Flow attack experiments, epsilon sweeps, and physics-loss
variants. Use this section for Darcy matched-budget attack studies.

### navier_stokes

Includes recurrent FNO2d attacks, periodic-warp alignment, and external-forcing
experiments. These scripts are usually more expensive than Burgers and Darcy
runs.

### cross_benchmark

Compares attack behavior across systems, objectives, optimizers, and epsilon
budgets. Use this after individual system runs are available.

## 02_adversarial_training

Training experiments that include adversarial objectives during optimization.

### burgers

Contains Burgers generalization and multi-dataset training studies, including
fixed-shift and adversarially generated perturbation settings.

### darcy_flow

Contains Darcy binary C-Flow adversarial-training scripts and related evaluation
drivers.

### navier_stokes

Contains recurrent Navier-Stokes adversarial-training scripts and post-training
evaluation helpers.

## 03_robustness_metrics

Post-training and post-attack diagnostics.

### jacobian_spectral_norms

Computes or plots Jacobian spectral-norm style quantities. These are used to
compare local sensitivity with finite-budget attack behavior.

### model_solver_svd_similarity

Computes or plots singular-value summaries for model/solver behavior.

### correlation_and_alignment

Measures how metrics align with attack losses or with each other.

## 04_timing_memory

Cost and efficiency measurements.

### solver_phase_probes

Profiles solver phases and isolates expensive solver-integrated components.

### adversarial_training_epoch_times

Compares wall-clock training time across objectives or systems.

### rematerialization_and_batch_capacity

Studies memory/runtime tradeoffs from rematerialization or checkpointing.

### runtime_monitoring

Collects runtime traces and monitoring summaries.

## 05_mechanism_and_ablations

Mechanistic explanation and ablation experiments.

### loss3_geometry

Studies the geometry of the solver-integrated L3 objective.

### loss_landscape

Builds local or two-dimensional loss-landscape visualizations.

### objective_mechanism

Compares objective mechanisms and helps explain why solver-integrated attacks
behave differently.

### loss_gradient_paths

Tracks gradient paths and perturbation evolution under different losses.

### jvp_vjp_spectrum

Uses JVP/VJP or spectral probes to study the frequency and operator structure of
the learned/adversarial perturbations.
