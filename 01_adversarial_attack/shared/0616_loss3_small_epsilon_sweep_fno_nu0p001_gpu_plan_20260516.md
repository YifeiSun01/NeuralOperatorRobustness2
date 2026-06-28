# Loss3 Experiment 3 Small-Epsilon Sweep Plan - FNO nu=0.001 GPU Run

Date: 2026-05-16

Status: plan for official GPU-only run.

## Scope

This experiment is intentionally restricted to:

- PDE: 1D Burgers
- viscosity: `nu=0.001`
- model family: FNO only
- model checkpoint: `fno_training_runs/burgers_nu0p001_fno1d_500/burgers_1d/checkpoints/pytorch_fno1d_500.pt`
- test data: `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/test.pt`

No DeepONet, no `nu=0.01`, and no CPU fallback are part of this run.

## GPU-Only Requirement

Official execution must use the `adv_robust` virtual environment and GPU.
The run is invalid if it falls back to CPU. The script refuses to start unless:

- PyTorch sees CUDA;
- the installed PyTorch wheel supports the active GPU architecture;
- a real CUDA tensor operation succeeds;
- JAX reports a GPU backend and at least one GPU device.

The current verified environment is:

- GPU: Tesla V100-SXM2-32GB
- compute capability: `sm_70`
- PyTorch: `2.8.0+cu126`
- JAX: `0.10.0` with GPU backend

## Scientific Goal

Complete Experiment 3 from the Loss3 plan: test whether small-radius ratio and
residual-ratio diagnostics represent local structure.

For each clean sample and each epsilon, evaluate candidate-bank estimates of:

- `L_f(epsilon) = max ||f(x+eps v)-f(x)|| / eps`
- `L_j(epsilon) = max ||j(x+eps v)-j(x)|| / eps`
- `L_e(epsilon) = max ||e(x+eps v)-e(x)|| / eps`, where `e=f-j`
- `G_e(epsilon) = max (||e(x+eps v)||-||e(x)||) / eps`

## Sweep Settings

- sample indices: `0, 7, 40, 47, 115`
- epsilons: `1e-4, 1e-3, 1e-2, 1e-1`
- direction bank:
  - top right singular vectors of clean `J_f`, `J_j`, and `J_e`;
  - both signs of each SVD direction;
  - clean residual outward-growth direction when available;
  - seeded random control directions.
- top SVD ranks: `8`
- random directions per sample: `128`
- evaluation batch size: `64`

## Required Outputs

The run must write:

- `candidate_metrics.csv`
- `best_by_objective_epsilon_index.csv`
- `aggregate_best_by_objective_epsilon.csv`
- `local_reference_by_index.csv`
- `local_reference_summary.csv`
- `direction_stability.csv`
- `direction_stability_summary.csv`
- `best_direction_source_counts.csv`
- `manifest.json`
- PNG visualizations under `figures/`
- a Markdown result document with observed conclusions.

## Interpretation Rules

Observed evidence must come from generated CSV/JSON/PNG files. Interpretations
must distinguish:

- model sensitivity `L_f`,
- solver sensitivity `L_j`,
- residual movement `L_e`,
- clean-residual outward growth `G_e`.

In particular, `L_e` and `G_e` are not the same object. `L_e` measures how fast
the residual field moves; `G_e` measures whether the current clean residual norm
is pushed outward.
