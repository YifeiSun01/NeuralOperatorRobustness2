# NS Batch Size Comparison

Reference is true `batch_size=1` on the same samples.

## Setup

- split: `test`
- samples: `32`
- fixed_step: `0.005`
- t_final: `20`
- target_size: `256`
- solver_mode: `vmap`
- source: `/workspace/NeuralOperatorRobustness2/2D_NS_FNO2d_recurrent/datasets/source_zongyi_real_initial`
- device: `NVIDIA A100-SXM4-80GB`

## Summary

| batch_size | seconds | sec/sample | global max abs | global RMSE | mean rel RMSE | max rel RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62.1721 | 1.9429 | 0 | 0 | 0 | 0 |
| 4 | 18.4272 | 0.5758 | 0.00059306622 | 1.3468389e-05 | 1.1327706e-05 | 1.9903235e-05 |
| 8 | 11.8632 | 0.3707 | 0.00059306622 | 1.3468389e-05 | 1.1327706e-05 | 1.9903235e-05 |
| 16 | 8.2250 | 0.2570 | 0.071441054 | 0.0017547501 | 0.0014792298 | 0.0024629838 |
| 32 | 6.7585 | 0.2112 | 0.071441054 | 0.0017547501 | 0.0014792298 | 0.0024629838 |

## Interpretation

- `batch_size=1` is the reference, so all error columns are zero for it.
- Small nonzero differences at larger batch sizes are floating-point differences from XLA/JAX vectorized execution, not different input data or paths.
- If exact agreement with the sequential solver matters, use the largest batch size whose error is acceptable in this table.

## Chinese Notes

这里的 reference 是真正逐条跑的 `batch_size=1`。`batch_size=4` 和
`batch_size=8` 相对逐条跑的误差很小：

- 最大逐点误差约 `5.93e-4`
- 全局 RMSE 约 `1.35e-5`
- 平均 relative RMSE 约 `1.13e-5`

`batch_size=16` 和 `batch_size=32` 明显更快，但相对逐条跑的最大逐点误差
上升到约 `0.0714`，平均 relative RMSE 约 `0.00148`。这不是输入路径错了，
也不是读取了不同 initial condition；这是 JAX/XLA 在更大 batch 下用了不同的
向量化/并行浮点执行路径。浮点加法和 FFT/非线性 PDE rollout 不是严格 bitwise
可交换的，所以更大 batch 可以带来更快速度，但不一定逐点完全复现
`batch_size=1` 的结果。

当前建议：

- 如果用快速 `vmap` 模式，并且要尽量贴近逐条 solver：用
  `--solver-batch-size 8`
- 如果只关心统计分布、能接受约 `1e-3` 量级 relative RMSE：可以测试
  `--solver-batch-size 16` 或 `32`
- 如果要和逐条跑逐点完全一致：用 `--solver-mode lax-map`，见
  `NS_BATCH_SIZE_COMPARISON_LAXMAP_N32.md`
- `--batch-size` 可以比 `--solver-batch-size` 大，它只控制每轮处理多少条并
  拷回 CPU；最终每个 split 仍然保存成一个完整 `.pt` 文件。真正影响 solver
  数值路径的是 `--solver-batch-size` 和 `--solver-mode`

这次 N=32 对照运行的 GPU 峰值约 `1.6 GiB`，远低于 A100 80GB。因此目前瓶颈
不是显存，而是你愿意接受多大的 batch 数值差异。
