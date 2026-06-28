# NS Batch Size Comparison

Reference is true `batch_size=1` on the same samples.

## Setup

- split: `test`
- samples: `32`
- fixed_step: `0.005`
- t_final: `20`
- target_size: `256`
- solver_mode: `lax-map`
- source: `/workspace/NeuralOperatorRobustness2/2D_NS_FNO2d_recurrent/datasets/source_zongyi_real_initial`
- device: `NVIDIA A100-SXM4-80GB`

## Summary

| batch_size | seconds | sec/sample | global max abs | global RMSE | mean rel RMSE | max rel RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 61.9806 | 1.9369 | 0 | 0 | 0 | 0 |
| 8 | 59.0430 | 1.8451 | 0 | 0 | 0 | 0 |
| 16 | 59.0354 | 1.8449 | 0 | 0 | 0 | 0 |
| 32 | 59.0208 | 1.8444 | 0 | 0 | 0 | 0 |

## Interpretation

- `batch_size=1` is the reference, so all error columns are zero for it.
- In `solver_mode=lax-map`, `batch_size=8`, `16`, and `32` matched the
  sequential reference exactly for this N=32 A100 test.
- This mode is much slower than `solver_mode=vmap`, but it is the right choice
  when exact agreement with the sequential solver matters.
