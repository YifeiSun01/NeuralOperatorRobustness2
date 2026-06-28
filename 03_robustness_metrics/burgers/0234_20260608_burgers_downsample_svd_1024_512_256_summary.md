# Burgers 1024/512/256 Downsample Jacobian SVD Summary - 2026-06-08

## Source Evidence

Observed source reports and CSVs:

- `docs/burgers_jacobian_downsample_svd_probe_20260607.md`
- `docs/burgers_round03_loss123_downsample_svd_probe_20260607.md`
- `forensics/burgers_jacobian_downsample_svd_probe_20260607/downsample_jacobian_svd_comparison.csv`
- `forensics/burgers_round03_loss123_downsample_svd_probe_20260607/downsample_jacobian_svd_comparison.csv`

Observed limitation: the expanded round03 full-Jacobian SVD source directory contains `loss1_epoch5000`, `loss2_epoch2000`, and `loss3_epoch1500`; no local `*8000*jacobian_svd*.npz` was found for loss1 epoch8000.

## Methods Compared

`block_projection`: `P^T J P`, an orthogonal projection onto adjacent block-constant subspaces.

`stride`: `J[::d, ::d]`, a point-sampled submatrix. Raw stride singular values scale roughly by the sampling density, so a scaled stride sigma ratio multiplies raw stride by `2` for 512 or `4` for 256.

## Initial 3-Sample Probe

Observed overall across 9 matrices: samples `000`, `004`, `006` and matrix kinds `solver`, `baseline`, `baseline_error`.

| coarse | method | raw sigma mean | scaled sigma mean | min rank1 R/L | min top10 R/L |
|---:|---|---:|---:|---:|---:|
| 512 | block_projection | `0.9872` | `0.9872` | `0.9971/0.9979` | `0.9999/0.9999` |
| 512 | stride | `0.5005` | `1.0009` | `0.9991/0.9995` | `0.9812/0.9839` |
| 256 | block_projection | `0.9426` | `0.9426` | `0.9533/0.9661` | `0.9971/0.9976` |
| 256 | stride | `0.2563` | `1.0253` | `0.8285/0.8757` | `0.8767/0.9010` |

Key correction from the discussion: for `baseline_error` at 256, scaled stride had a closer sigma mean than block projection in this small probe. Block projection was better for singular-vector/subspace orientation, not always for sigma1 magnitude.

For `baseline_error` at 256:

- block sigma mean: `0.8908`, about `10.9%` low;
- scaled stride sigma mean: `1.0640`, about `6.4%` high;
- block top10 R/L min: `0.9971/0.9976`;
- stride top10 R/L min: `0.8767/0.9010`.

## Expanded Round03 20-Sample Probe

Observed source: `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`.

Observed scope: 20 samples x 9 matrix labels = 180 full `1024 x 1024` Jacobian NPZ files. Labels: `solver`, `baseline_model`, `baseline_error`, `loss1_epoch5000_model`, `loss1_epoch5000_error`, `loss2_epoch2000_model`, `loss2_epoch2000_error`, `loss3_epoch1500_model`, `loss3_epoch1500_error`.

The run generated 720 comparison rows. Max CUDA allocated/reserved: `19.04 MB / 26.00 MB`.

### 256 Block Projection Summary

| model | kind | sigma mean | sigma range | min top10 R/L |
|---|---|---:|---:|---:|
| baseline | model | `0.9781` | `[0.9706, 0.9878]` | `0.9974/0.9975` |
| loss1_epoch5000 | model | `0.9722` | `[0.9500, 0.9918]` | `0.9991/0.9995` |
| loss2_epoch2000 | model | `0.9699` | `[0.9355, 0.9865]` | `0.9982/0.9990` |
| loss3_epoch1500 | model | `0.9738` | `[0.9644, 0.9872]` | `0.9991/0.9991` |
| baseline | error | `0.9361` | `[0.8342, 0.9881]` | `0.9193/0.9213` |
| loss1_epoch5000 | error | `0.9493` | `[0.8548, 0.9799]` | `0.9186/0.9194` |
| loss2_epoch2000 | error | `0.9366` | `[0.7601, 0.9792]` | `0.9487/0.9575` |
| loss3_epoch1500 | error | `0.8959` | `[0.7225, 0.9691]` | `0.9991/0.9992` |

### 256 Scaled Stride Summary

| model | kind | scaled sigma mean/range | min top10 R/L |
|---|---|---:|---:|
| baseline | model | `0.9992 [0.9619, 1.0031]` | `0.9038/0.9044` |
| loss1_epoch5000 | model | `1.0001 [0.9828, 1.0229]` | `0.9281/0.9302` |
| loss2_epoch2000 | model | `1.0040 [0.9883, 1.0384]` | `0.9919/0.9918` |
| loss3_epoch1500 | model | `0.9982 [0.9709, 1.0124]` | `0.9994/0.9994` |
| baseline | error | `1.0040 [0.8915, 1.0684]` | `0.7724/0.8168` |
| loss1_epoch5000 | error | `1.0309 [0.8024, 1.2052]` | `0.7091/0.7297` |
| loss2_epoch2000 | error | `1.0231 [0.6282, 1.1963]` | `0.6796/0.6943` |
| loss3_epoch1500 | error | `1.0108 [0.8318, 1.1695]` | `0.9648/0.9671` |

## Interpretation

Observed: solver/model Jacobians are easy to compress. Their 256 block sigma means are typically within about `2-3%` of full, and their top10 subspaces remain very close.

Observed: error Jacobians are fragile. The 256 block projection can understate error sigma by up to `27.75%` for loss3 error and `23.99%` for loss2 error. The 256 scaled stride can be off by `37.18%` for loss2 error.

Corrected inference: scaled stride can be closer for sigma magnitude after calibration, especially for some smooth-ish baseline/model cases. Block projection is the better diagnostic when singular vectors/top-k subspaces and `L2`-consistent projection geometry matter.

Practical recommendation: use `512 block_projection` for quantitative screening. Use `256 block_projection` for quick top-k subspace and rough magnitude checks with calibration. Do not use uncalibrated `256 stride` for error-Jacobian spectral-norm conclusions.
