# FNO nu=0.01 vs DeepONet nu=0.01 Comprehensive SVD Diagnostics (2026-05-15)

Status: completed. Aggregate plots were generated without standard-deviation
shading.

## Inputs

FNO side:

- raw Jacobians/SVDs: `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`
- model checkpoint: `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`
- `nu=0.01`, sample indices `0, 7, 40, 47, 115`

DeepONet/default-net side:

- raw Jacobians/SVDs: `forensics/deeponet_solver_jacobian_similarity_20260515/`
- model checkpoint: `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- `nu=0.01`, same sample indices

Plot script:

- `tools/plot_fno_deeponet_nu0p01_svd_diagnostics.py`

Output root:

- `forensics/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`

## Key aggregate readout

| quantity | FNO nu=0.01 | DeepONet nu=0.01 |
|---|---:|---:|
| model top-1 sigma | 1.379 | 7.202 |
| solver top-1 sigma | 1.373 | 1.373 |
| error top-1 sigma | 0.0373 | 7.100 |
| model top-1 hi128 | 2.82e-09 | 0.7614 |
| error top-1 hi128 | 0.1957 | 0.8033 |
| model top-1 zero crossings | 0.40 | 507.20 |
| error top-1 zero crossings | 14.40 | 516.00 |
| model-vs-solver k=1 mean principal angle | 2.32 deg | 82.83 deg |
| model-vs-solver k=8 mean principal angle | 1.82 deg | 80.19 deg |
| model-vs-error k=1 mean principal angle | 62.53 deg | 10.01 deg |

## Generated plot families

Per-index plots under `index_000/`, `index_007/`, `index_040/`, `index_047/`,
and `index_115/`:

- `*_sixrow_top8_right_vectors.png`
- `*_sixrow_top8_fft_energy.png`
- `*_pairwise_right_vector_cosine_heatmaps.png`
- `*_principal_angle_curves.png`
- `*_singular_value_spectra.png`
- `*_orthogonality_error_heatmaps.png`

Aggregate plots:

- `aggregate_mean_top8_fft_energy_grid.png`
- `aggregate_mean_top8_shape_grid.png`
- `aggregate_mean_pairwise_right_vector_cosine_heatmaps_top8.png`
- `aggregate_principal_angle_curves_top8.png`
- `aggregate_singular_value_spectra_rank64.png`
- `aggregate_orthogonality_error_heatmaps_top8.png`

## Interpretation

At matched `nu=0.01`, FNO's local Jacobian is highly solver-like: the top-1 and
top-8 right-singular subspaces are almost aligned with the solver, the leading
singular value matches the solver, and the leading right-singular vector is very
low-frequency.

DeepONet/default-net remains qualitatively different: its leading model
singular directions are high-frequency and nearly orthogonal to the solver's
leading directions. Its error Jacobian is nearly as large as the model Jacobian
and is close to the DeepONet model subspace, which is why DeepONet model and
error directions look similar.

Within one SVD, right singular vectors are orthonormal by construction. The
orthogonality heatmaps are a numerical sanity check; the main scientific signal
is the cross-operator subspace angle and vector-overlap plots.
