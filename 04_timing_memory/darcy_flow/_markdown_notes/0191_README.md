# Darcy Jacobian SVD Timing Benchmark

- Model: `loss3` 1000c checkpoint.
- Sample: `darcy_binary_loss3targeted_20260611_00_matern_smooth...`, sample index `0`.
- Device: V100 GPU while stage2 training was also alive.
- Full matrix dimension: `7225 x 7225`.

| projection | dim | Jacobian sec | SVD sec | total sec | matrix MiB | sigma max | sigma error vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 7225 | 42.684 | 26.918 | 69.602 | 199.13 | 0.0023142719 |  |
| block/2 | 1764 | 13.064 | 0.554 | 13.618 | 11.87 | 0.0022706068 | 1.89% |
| block/4 | 441 | 5.561 | 0.121 | 5.682 | 0.74 | 0.0022546914 | 2.57% |

## Notes

- Block projections use an orthonormal block basis on the top-left `84 x 84` crop.
- `block/2` has dimension `1764 x 1764`; `block/4` has dimension `441 x 441`.
- Full SVD is feasible for occasional Darcy samples, but projection is much faster for sweeps over many samples/models.

## Files

- `result_loss3_full_factor1.json`
- `result_loss3_block_factor2.json`
- `result_loss3_block_factor4.json`
- `singular_values_loss3_full_factor1.npy`
- `singular_values_loss3_block_factor2.npy`
- `singular_values_loss3_block_factor4.npy`
