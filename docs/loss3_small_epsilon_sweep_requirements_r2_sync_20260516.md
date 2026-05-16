# Loss3 Small-Epsilon Sweep Requirements And R2 Sync - 2026-05-16

Status: requirements / artifact sync only. No new numerical small-epsilon sweep was run.

## Goal

This note records what is needed to complete Experiment 3 from
`docs/loss3_original_theory_experiment_plan.md`: the planned Small-Epsilon Sweep
for FNO / 1D Burgers `nu=0.001`.

The sweep should estimate, for
`epsilon in {1e-4, 1e-3, 1e-2, 1e-1}`:

- `L_f(epsilon)`: local model-output sensitivity.
- `L_j(epsilon)`: local solver/oracle sensitivity.
- `L_e(epsilon)`: local residual/error-field sensitivity.
- `G_e(epsilon)`: local clean-error norm growth.
- stability of worst directions across epsilon values.
- cross-epsilon direction-cosine tables.

## Required Inputs

Observed local / synced requirements:

- Python environment: `adv_robust`.
- Model: `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`.
- Dataset directory:
  `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/`.
- Test split:
  `dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`,
  loaded successfully with shape `x=(150, 1024)`, `y=(150, 1024)`.
- Train split:
  `dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt`,
  loaded successfully with shape `x=(1350, 1024)`, `y=(1350, 1024)`.
- Solver/oracle setting: Exponax/JAX Burgers solver, `nu=0.001`, `nx=1024`,
  `t_final=1.0`, `dt=0.001`, `domain=2.0`.

## R2 Artifacts Synced Locally

Synced from R2 prefix:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Local artifact paths:

- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`
  - 195 files, about 180 MiB.
  - Contains 15 raw `*_jacobian_svd.npz` files:
    FNO, solver, and error SVD arrays for indices `0`, `7`, `40`, `47`, `115`.
  - Useful for direction-cosine checks against clean-point `J_f`, `J_j`, and
    `J_e = J_f - J_j` top directions.
- `forensics/outward_growth_direction_20260515/`
  - 65 files.
  - Contains `directions.npz`, clean inputs/outputs/residuals, finite-difference
    growth tables, and outward-growth directions for indices `0`, `7`, `40`,
    `47`, `115`.
  - This is partial evidence for Experiment 3: it checked small finite-difference
    radii `1e-4`, `1e-3`, and `1e-2` for selected local directions, but it is
    not the full planned sweep over all `L_f`, `L_j`, `L_e`, `G_e`.
- `results/burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only/`
  - 53 files.
  - Contains older attack summaries for `Linf` epsilon `0.01` and `0.1`.
  - This is context only; it is not the planned local operator/risk-growth sweep.
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/...nu0.001.../`
  - Restored missing train/test split `.pt` files.

## What Still Needs To Be Run

The exact planned Experiment 3 is still not complete. A dedicated sweep should
be run for the current FNO / Burgers `nu=0.001` scope, preferably first on the
same local diagnostic indices `[0, 7, 40, 47, 115]`.

Expected outputs:

- per-index, per-epsilon estimates of `L_f`, `L_j`, `L_e`, and `G_e`;
- direction vectors or enough metadata to reconstruct them;
- cross-epsilon direction-cosine tables;
- aggregate summary tables;
- `manifest.json` recording model, data, solver, epsilon list, indices, and
  command used;
- a dedicated result Markdown file under `docs/`.

## Inference

The synced artifacts are sufficient prerequisites to run or implement the
planned small-epsilon sweep locally: the model checkpoint, test data, solver
environment, clean-point SVD/Jacobian directions, and partial small-radius
finite-difference evidence are now present in the workspace.

The remaining missing piece is not data; it is the actual systematic sweep run
that estimates all four quantities and records cross-epsilon direction
stability.
