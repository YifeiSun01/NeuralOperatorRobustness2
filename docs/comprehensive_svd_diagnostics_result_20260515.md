# Comprehensive SVD Diagnostics Result

Date: 2026-05-15

## Question

This note records the dense visual diagnostic pass requested after the
FNO-vs-solver and DeepONet-vs-solver local-Jacobian experiments.

The comparison uses the same five sampled initial conditions:

```text
sample indices: 0, 7, 40, 47, 115
top singular vectors plotted: top 8
```

For each sample, the compared local maps are:

```text
FNO model,    solver at nu=0.001, FNO error J_m - J_s
DeepONet model, solver at nu=0.01, DeepONet error J_m - J_s
```

The plotted vectors are right singular vectors, so they are input perturbation
directions. The leading singular value is the spectral norm of the local
Jacobian.

## Inputs

```text
FNO raw SVD root:
forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/

DeepONet SVD root:
forensics/deeponet_solver_jacobian_similarity_20260515/

output root:
forensics/comprehensive_svd_diagnostics_20260515/

script:
tools/plot_comprehensive_svd_diagnostics.py
```

Regeneration command:

```bash
PYTHONPATH=adv_robust/lib/python3.12/site-packages \
python3 tools/plot_comprehensive_svd_diagnostics.py \
  --sample-indices 0 7 40 47 115 \
  --top-k 8 \
  --spectrum-max-rank 64
```

## How To Read The Figures

- `sigma`: singular value for this right singular vector. For rank 1, this is
  the local spectral norm.
- `hi128`: fraction of Fourier energy above mode 128. Larger means more
  high-frequency content.
- `zc`: zero crossings of the vector. Larger means more oscillatory/jagged.
- Principal angle: angle between two top-k right-singular-vector subspaces.
  Lower angle means the subspaces are more similar.
- Pairwise cosine heatmaps: absolute cosine between individual right singular
  vectors across two operators. Bright diagonal means rank-by-rank directions
  match well.
- Orthogonality heatmaps: within one SVD, this should be identity-like. It is a
  numerical sanity check, not the main scientific comparison.

Vector signs are arbitrary in SVD. The plotting script aligns signs only for
display, and scales line shapes by max absolute value so the mode shapes are
readable. The singular values shown in labels remain the actual values.

## Generated Plot Families

Per sample index, for example under:

```text
forensics/comprehensive_svd_diagnostics_20260515/index_000/
```

there are:

- `index_000_sixrow_top8_right_vectors.png`: six rows, top-8 right singular
  vector line plots. Rows are FNO model, solver `nu=0.001`, FNO error,
  DeepONet model, solver `nu=0.01`, DeepONet error.
- `index_000_sixrow_top8_fft_energy.png`: Fourier energy spectrum for the same
  vectors.
- `index_000_pairwise_right_vector_cosine_heatmaps.png`: model/solver/error
  pairwise cosine heatmaps.
- `index_000_principal_angle_curves.png`: top-k subspace principal-angle
  curves.
- `index_000_singular_value_spectra.png`: singular value spectra.
- `index_000_orthogonality_error_heatmaps.png`: within-SVD orthogonality
  numerical-error heatmaps.

Aggregate plots across all five samples:

- `aggregate_mean_top8_shape_grid.png`: display-aligned mean vector shapes.
- `aggregate_mean_top8_fft_energy_grid.png`: mean Fourier energy spectra after
  FFT, with sample variability bands.
- `aggregate_mean_pairwise_right_vector_cosine_heatmaps_top8.png`: mean
  cross-operator vector cosine heatmaps.
- `aggregate_principal_angle_curves_top8.png`: mean top-k principal-angle
  curves.
- `aggregate_singular_value_spectra_rank64.png`: mean singular value spectra.
- `aggregate_orthogonality_error_heatmaps_top8.png`: mean orthogonality-error
  heatmaps.

The numeric tables are stored next to the plots:

```text
summary.md
singular_vector_metrics.csv
aggregate_singular_vector_metrics.csv
pairwise_right_vector_cosines.csv
principal_angles.csv
aggregate_principal_angles.csv
orthogonality_summary.csv
aggregate_orthogonality_summary.csv
```

## Key Aggregate Results

Values below are means over the five sampled initial conditions.

| quantity | FNO side | DeepONet side |
|---|---:|---:|
| model top-1 sigma | 3.895 | 7.202 |
| solver top-1 sigma | 4.095 | 1.373 |
| error top-1 sigma | 0.868 | 7.100 |
| model top-1 `hi128` | 2.23e-07 | 0.761 |
| error top-1 `hi128` | 4.56e-05 | 0.803 |
| model top-1 zero crossings | 13.6 | 507.2 |
| error top-1 zero crossings | 16.8 | 516.0 |
| model-vs-solver k=1 mean principal angle | 4.86 deg | 82.83 deg |
| model-vs-solver k=8 mean principal angle | 7.01 deg | 80.19 deg |
| model-vs-error k=1 mean principal angle | 74.32 deg | 10.01 deg |

Interpretation:

- FNO and the `nu=0.001` solver have close leading right-singular subspaces.
  The top-1 angle is about `4.86 deg`, and top-8 is about `7.01 deg`.
- DeepONet and the `nu=0.01` solver are almost orthogonal in their leading
  right-singular subspaces. The top-1 angle is about `82.83 deg`, and top-8 is
  about `80.19 deg`.
- DeepONet and its error Jacobian are close. The top-1 model-vs-error angle is
  about `10.01 deg`, which means the dominant DeepONet error directions are
  basically the DeepONet model directions.
- The Fourier plots show the same story: FNO/solver dominant directions are
  smooth and low-frequency, while DeepONet/error dominant directions are
  high-frequency and jagged.

## Orthogonality Result

The top-8 right singular vectors inside each individual SVD are orthogonal up
to numerical precision:

| operator | max off-diagonal error |
|---|---:|
| FNO model | 2.21e-09 |
| solver `nu=0.001` | 1.94e-09 |
| FNO error | 3.33e-09 |
| DeepONet model | 2.59e-09 |
| solver `nu=0.01` | 2.30e-09 |
| DeepONet error | 2.44e-09 |

This is expected: SVD returns orthonormal right singular vectors. Therefore the
angle between two different right singular vectors from the same operator is
expected to be 90 degrees. This is useful as a sanity check that the saved SVDs
are numerically well-formed.

Scientifically, the more meaningful quantities are cross-operator angles and
subspace angles: for example FNO model vs solver, DeepONet model vs solver, and
DeepONet model vs error. Individual singular vectors can be unstable when
singular values are close, so top-k subspace angles are usually more robust
than rank-by-rank vector angles.

## No-Std Plot Variant

A second copy of the same diagnostics was generated without standard-deviation
shading in the aggregate line plots:

```text
forensics/comprehensive_svd_diagnostics_20260515_no_std/
```

It was generated with:

```bash
PYTHONPATH=adv_robust/lib/python3.12/site-packages \
python3 tools/plot_comprehensive_svd_diagnostics.py \
  --sample-indices 0 7 40 47 115 \
  --top-k 8 \
  --spectrum-max-rank 64 \
  --out-root forensics/comprehensive_svd_diagnostics_20260515_no_std \
  --no-std-shading
```

Use this directory when the shaded variability bands make the mean curves hard
to read. The numeric CSV values are the same aggregation as the shaded version;
only the aggregate line-plot rendering changes.

## Main Conclusion

The comprehensive plots support the mechanism-level interpretation:

```text
FNO:
  model directions ~= solver directions
  both are low-frequency/smooth
  error directions are not the dominant model directions

DeepONet:
  model directions are high-frequency/jagged
  solver directions are low-frequency/smooth
  error directions ~= DeepONet model directions
```

So the visually observed phenomenon is real in the recovered diagnostics:
FNO's leading local perturbation directions look like the physical solver's
local sensitive directions, while DeepONet's leading local perturbation
directions are high-frequency modes that the physical solver mostly does not
share.
