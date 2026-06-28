# Comprehensive SVD Diagnostics

This directory contains dense line/heatmap diagnostics for the local Jacobian SVDs.

Rows in the main per-index figures are:

1. FNO model `nu=0.01`
2. solver `nu=0.01`
3. FNO error `J_m - J_s`
4. DeepONet model `nu=0.01`
5. solver `nu=0.01`
6. DeepONet error `J_m - J_s`

Columns are right singular vector ranks. Right singular vectors are input perturbation directions.

Std shading in aggregate line plots: disabled.

## Key Aggregate Readout

| quantity | FNO side | DeepONet side |
|---|---:|---:|
| model top-1 sigma | 1.379 | 7.202 |
| solver top-1 sigma | 1.373 | 1.373 |
| error top-1 sigma | 0.0373 | 7.1 |
| model top-1 hi128 | 2.82e-09 | 0.7614 |
| error top-1 hi128 | 0.1957 | 0.8033 |
| model top-1 zero crossings | 0.40 | 507.20 |
| error top-1 zero crossings | 14.40 | 516.00 |
| model-vs-solver k=1 mean principal angle | 2.32 deg | 82.83 deg |
| model-vs-solver k=8 mean principal angle | 1.82 deg | 80.19 deg |
| model-vs-error k=1 mean principal angle | 62.53 deg | 10.01 deg |

## Generated Plot Families

Per index under `index_*/`:

- `*_sixrow_top8_right_vectors.png`: six-row line grid of top-8 right singular vectors with `sigma`, `hi128`, and zero crossings.
- `*_sixrow_top8_fft_energy.png`: Fourier energy spectra for the same vectors.
- `*_pairwise_right_vector_cosine_heatmaps.png`: model/solver/error pairwise absolute-cosine heatmaps.
- `*_principal_angle_curves.png`: top-k subspace principal-angle curves.
- `*_singular_value_spectra.png`: singular value spectra with spectral norms in labels.
- `*_orthogonality_error_heatmaps.png`: within-operator orthogonality sanity-check heatmaps.

Aggregate across five samples:

- `aggregate_mean_top8_fft_energy_grid.png`: mean Fourier spectrum after Fourier transform.
- `aggregate_mean_top8_shape_grid.png`: display-aligned mean vector shapes; useful visually but less invariant than spectra/subspaces.
- `aggregate_mean_pairwise_right_vector_cosine_heatmaps_top8.png`: mean cross-operator vector cosine heatmaps.
- `aggregate_principal_angle_curves_top8.png`: mean top-k principal angles.
- `aggregate_singular_value_spectra_rank64.png`: mean singular spectra.
- `aggregate_orthogonality_error_heatmaps_top8.png`: mean numerical orthogonality errors.

## Orthogonality Interpretation

Within one SVD, the right singular vectors are orthonormal by construction. Therefore, the angle between different ranks inside the same operator is expected to be 90 degrees, up to numerical error. This is mostly a sanity check, not the main scientific comparison.

The meaningful comparison is cross-operator: whether the model top-k right singular subspace aligns with the solver top-k subspace, and whether the error subspace aligns with the model. The plots show FNO model/solver subspaces are close, while DeepONet model/error subspaces are close and DeepONet model/solver subspaces are far apart.
