# Nine-Row SVD Diagnostics

Rows in the main line/FFT grids are:

1. FNO model `nu=0.001`
2. solver `nu=0.001`
3. FNO error `nu=0.001`
4. FNO model `nu=0.01`
5. solver `nu=0.01`
6. FNO error `nu=0.01`
7. DeepONet/default-net model `nu=0.01`
8. solver `nu=0.01`
9. DeepONet/default-net error `nu=0.01`

All aggregate line plots are drawn without standard-deviation shading.

## Key Aggregate Readout

| quantity | FNO nu=0.001 | FNO nu=0.01 | DeepONet nu=0.01 |
|---|---:|---:|---:|
| model top-1 sigma | 3.895 | 1.379 | 7.202 |
| solver top-1 sigma | 4.095 | 1.373 | 1.373 |
| error top-1 sigma | 0.8682 | 0.0373 | 7.1 |
| model top-1 hi128 | 2.231e-07 | 2.82e-09 | 0.7614 |
| error top-1 hi128 | 4.559e-05 | 0.1957 | 0.8033 |
| model top-1 zero crossings | 13.60 | 0.40 | 507.20 |
| error top-1 zero crossings | 16.80 | 14.40 | 516.00 |
| model-vs-solver k=1 angle | 4.86 deg | 2.32 deg | 82.83 deg |
| model-vs-solver k=8 angle | 7.01 deg | 1.82 deg | 80.19 deg |
| model-vs-error k=1 angle | 74.32 deg | 62.53 deg | 10.01 deg |

## Main Files

Per index:

- `index_*/index_*_ninerow_top8_right_vectors.png`
- `index_*/index_*_ninerow_top8_fft_energy.png`
- `index_*/index_*_family_right_vector_cosine_heatmaps.png`
- `index_*/index_*_principal_angle_curves.png`
- `index_*/index_*_singular_value_spectra.png`
- `index_*/index_*_orthogonality_error_heatmaps.png`

Aggregate:

- `aggregate_mean_top8_shape_grid.png`
- `aggregate_mean_top8_fft_energy_grid.png`
- `aggregate_mean_family_right_vector_cosine_heatmaps_top8.png`
- `aggregate_principal_angle_curves_top8.png`
- `aggregate_singular_value_spectra_rank64.png`
- `aggregate_orthogonality_error_heatmaps_top8.png`
