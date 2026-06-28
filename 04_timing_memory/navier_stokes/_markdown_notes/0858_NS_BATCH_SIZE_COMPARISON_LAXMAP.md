# NS Batch Size Comparison

Reference is true `batch_size=1` on the same samples.

## Setup

- split: `test`
- samples: `16`
- fixed_step: `0.005`
- t_final: `20`
- target_size: `256`
- solver_mode: `lax-map`
- source: `/workspace/NeuralOperatorRobustness2/2D_NS_FNO2d_recurrent/datasets/source_zongyi_real_initial`
- device: `NVIDIA A100-SXM4-80GB`

## Summary

| batch_size | seconds | sec/sample | global max abs | global RMSE | mean rel RMSE | max rel RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.3903 | 2.0244 | 0 | 0 | 0 | 0 |
| 4 | 30.0502 | 1.8781 | 0 | 0 | 0 | 0 |
| 8 | 29.9194 | 1.8700 | 0 | 0 | 0 | 0 |
| 16 | 29.8359 | 1.8647 | 0 | 0 | 0 | 0 |

## Interpretation

- `batch_size=1` is the reference, so all error columns are zero for it.
- In `solver_mode=lax-map`, `batch_size=4`, `8`, and `16` matched the
  sequential reference exactly for this N=16 A100 test.
- This mode is much slower than `solver_mode=vmap`, but it is the right choice
  when exact agreement with the sequential solver matters.
