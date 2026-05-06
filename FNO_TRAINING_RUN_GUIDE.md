# FNO Training Run Guide

这份说明对应当前 VastAI / 单 GPU 环境：不需要 SLURM，直接在仓库根目录用 `adv_robust` 虚拟环境和 `CUDA_VISIBLE_DEVICES=0` 跑即可。训练脚本已经准备好，但默认不会自动启动，只有你执行下面命令时才会训练。

## 数据位置

### 1D Burgers

原始完整数据在：

```text
1D_Burgers/datasets/1D/Burgers/batched_exponax/
```

已经拆好的 train/test 数据在：

```text
1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/
```

当前每个 `nu` 的拆分是：

```text
train: 1350 samples
test:  150 samples
```

可用的两个 Burgers 数据集：

```text
nu=0.0005
nu=0.001
```

### 2D Navier-Stokes

2D NS 数据已经自带 train/test，训练脚本默认读这里：

```text
2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt
2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt
```

默认训练方式是：

```text
input:  frames 0..9
target: frames 10..19
```

也就是 `--t-in 10 --t-out 10`。如果想预测到第 20 个时间点，可以改成 `--t-in 10 --t-out 11`。

## 一条命令跑完整训练

在仓库根目录运行：

```bash
cd /workspace/NeuralOperatorRobustness2
RUN_ROOT=fno_training_runs/full_fno_suite \
PROBLEMS=burgers,ns \
BURGERS_NU=0.001 \
FRAMEWORKS=pytorch,jax \
EPOCHS_1D=500 \
EPOCHS_2D=500 \
BATCH_1D=64 \
BATCH_2D=4 \
EVAL_EVERY_1D=1 \
EVAL_EVERY_2D=1 \
COMPARE_SAMPLES=5 \
MEMORY_PROFILE=true \
TIME_PROFILE=true \
MEMORY_SAMPLE_INTERVAL=1 \
MEMORY_PROFILE_DETAILED_BATCHES=1 \
MEMORY_PROFILE_DETAILED_EPOCHS=-1 \
bash tools/run_full_fno_training_suite.sh
```

这条命令会依次做三件事：

1. 确认 1D Burgers 的 train/test split 已存在。
2. 训练 1D Burgers 的 PyTorch FNO1d 和 JAX FNO1d。
3. 训练 2D NS 的 PyTorch recurrent FNO2d 和 JAX recurrent FNO2d。

如果同一次运行里包含 `pytorch,jax`，训练结束后脚本会自动做框架对比：

```text
默认取测试集前 5 条样本做同输入推理
保存 PyTorch/JAX 的预测结果
比较两边预测输出差异
比较两边参数块差异
画 PyTorch/JAX loss overlay
画同输入推理对比图
```

默认会把 JAX 的初始参数对齐到 PyTorch 的初始参数：

```text
--align-jax-init-with-pytorch
```

这样可以更公平地比较两个实现。即便初始参数对齐，训练后参数也不一定逐项完全一样，因为 PyTorch 和 JAX 的 FFT、Adam、复数梯度、浮点规约顺序可能仍有细微差别。脚本会把这些差异记录下来，而不是假设它们一定为零。

如果不想对齐初始化，可以加：

```text
--no-align-jax-init-with-pytorch
```

默认会记录显存和详细阶段耗时：

```text
每 1 秒采样一次当前进程/GPU显存曲线
每个 epoch 的前 1 个 batch 记录 forward/backward/optimizer_step 的显存细节
训练后的同输入推理记录 inference_forward 显存
同时记录这些阶段的耗时 phase_seconds
```

对应参数是：

```text
--memory-profile
--time-profile
--memory-sample-interval 1
--memory-profile-detailed-batches 1
```

如果你想跑一版尽量干净的速度，不做内部显存采样，也不拆 forward/backward 计时：

```bash
cd /workspace/NeuralOperatorRobustness2
RUN_ROOT=fno_training_runs/no_profile_500 \
BURGERS_NU=0.001 \
FRAMEWORKS=pytorch,jax \
EPOCHS_1D=500 \
EPOCHS_2D=500 \
BATCH_1D=64 \
BATCH_2D=4 \
COMPARE_SAMPLES=5 \
MEMORY_PROFILE=false \
TIME_PROFILE=false \
bash tools/run_full_fno_training_suite.sh
```

这时仍然会记录粗粒度总耗时：`metrics_summary.csv` 里的 `seconds_total`，以及每个 `losses_*.csv` 里的每个 epoch 总秒数 `seconds`。但是不会生成内部 `memory/` 目录里的显存曲线和 phase CSV；总入口也会绕过外层 `run_with_gpu_monitor.sh`，避免后台 `nvidia-smi` 采样影响速度。

如果只想记录时间，不记录显存：

```bash
MEMORY_PROFILE=false TIME_PROFILE=true bash tools/run_full_fno_training_suite.sh
```

如果只想记录显存采样和显存峰值，不额外打开 time-only 对比：

```bash
MEMORY_PROFILE=true TIME_PROFILE=false bash tools/run_full_fno_training_suite.sh
```

注意：只要你要记录 forward/backward 的显存阶段，脚本就必须在阶段边界同步一次，所以 phase CSV 里仍会带 `phase_seconds`。如果只想要整段 GPU 曲线、不想拆阶段，把 `MEMORY_PROFILE_DETAILED_BATCHES=0` 一起加上。

如果只想让 detailed profiling 发生在前几个 epoch，例如只记录前 2 个 epoch 的前 5 个 batch：

```bash
MEMORY_PROFILE_DETAILED_BATCHES=5 MEMORY_PROFILE_DETAILED_EPOCHS=2 bash tools/run_full_fno_training_suite.sh
```

## 只跑某一个问题

总入口脚本支持 `PROBLEMS`：

```bash
PROBLEMS=burgers bash tools/run_full_fno_training_suite.sh
PROBLEMS=ns bash tools/run_full_fno_training_suite.sh
PROBLEMS=burgers,ns bash tools/run_full_fno_training_suite.sh
```

这样可以把 1D Burgers 和 2D Navier-Stokes 分开跑，避免每次都串行训练两个任务。

## 2D NS 加速建议

当前默认 2D NS 配置是 `target-size=256, modes=12, width=20, T_out=10`。这是比之前 `modes=64, width=60` 小很多的模型，训练速度和显存压力都会明显低一些。A100 上可以先从 `batch-size=4` 或更大的 batch 试吞吐：

```bash
cd /workspace/NeuralOperatorRobustness2
RUN_ROOT=fno_training_runs/ns_fast_batch8_500 \
PROBLEMS=ns \
FRAMEWORKS=pytorch \
EPOCHS_2D=500 \
BATCH_2D=8 \
EVAL_BATCH_2D=8 \
EVAL_EVERY_2D=10 \
MEMORY_PROFILE=false \
TIME_PROFILE=false \
bash tools/run_full_fno_training_suite.sh
```

如果 batch 8 稳定且显存还有余量，可以试：

```bash
BATCH_2D=12 EVAL_BATCH_2D=12 PROBLEMS=ns FRAMEWORKS=pytorch MEMORY_PROFILE=false TIME_PROFILE=false bash tools/run_full_fno_training_suite.sh
```

`EVAL_EVERY_2D=10` 表示每 10 个 epoch 跑一次完整 test evaluation，并且第 1 个和最后 1 个 epoch 仍会评估。最终训练结束后脚本仍会完整计算 final train/test metrics。这个不会改训练步数，只是减少训练中间的测试集推理开销。

如果允许改模型规模，原始 FNO Navier-Stokes benchmark 常见空间分辨率是 `64x64`，FNO 也强调可以低分辨率训练、高分辨率评估。你可以用更轻的 sanity 配置先看 loss 是否正常下降：

```bash
RUN_ROOT=fno_training_runs/ns_fast_128_500 \
PROBLEMS=ns \
FRAMEWORKS=pytorch \
EPOCHS_2D=500 \
BATCH_2D=16 \
EVAL_BATCH_2D=16 \
TARGET_SIZE_2D=128 \
MODES_2D=32 \
WIDTH_2D=48 \
EVAL_EVERY_2D=10 \
MEMORY_PROFILE=false \
TIME_PROFILE=false \
bash tools/run_full_fno_training_suite.sh
```

如果想每个 batch 都记录 forward/backward 细节：

```text
--memory-profile-detailed-batches -1
```

注意：JAX 没有 PyTorch 那样的 allocator-level `allocated/reserved/peak` API，所以 JAX 的分段显存用 `nvidia-smi` 当前进程显存近似记录；PyTorch 会额外记录 `torch.cuda` 的 allocated/reserved/peak。JAX 为了拆开 forward/backward，会在被详细记录的 batch 上额外执行一次 forward 作为测量探针，所以把 detailed batches 设成 `-1` 会更慢。

`memory_summary_*.json` 里会直接汇总：

```text
sampled_whole_run_peak_gpu_memory_used_mib
sampled_whole_run_seconds
profiled_train_peak_mib
profiled_inference_peak_mib
profiled_backward_to_forward_peak_ratio
profiled_backward_to_forward_mean_time_ratio
phases.forward.max_memory_mib / mean_seconds / max_seconds / total_seconds
phases.backward.max_memory_mib / mean_seconds / max_seconds / total_seconds
phases.optimizer_step.max_memory_mib / mean_seconds / max_seconds / total_seconds
phases.inference_forward.max_memory_mib / mean_seconds / max_seconds / total_seconds
```

`sampled_whole_run_peak_*` 是整段运行按采样间隔看到的峰值；如果某个显存尖峰短于采样间隔，可能会漏掉。想看更细可以把 `--memory-sample-interval` 调小，比如 `0.2`。`profiled_*` 是详细抽查 batch 的 forward/backward/inference 结果，更适合看阶段比例和机制。

如果只想跑 PyTorch：

```bash
FRAMEWORKS=pytorch bash tools/run_full_fno_training_suite.sh
```

如果只想跑 JAX：

```bash
FRAMEWORKS=jax bash tools/run_full_fno_training_suite.sh
```

## 分开运行

### 1D Burgers

```bash
cd /workspace/NeuralOperatorRobustness2
RUN_NAME=training_fno1d_burgers \
GPU_MONITOR_INTERVAL=10 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_with_gpu_monitor.sh \
adv_robust/bin/python -u tools/train_fno1d_suite.py \
  --train-path 1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt \
  --test-path 1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt \
  --output-root fno_training_runs/full_fno_suite \
  --run-name burgers_nu0.001 \
  --frameworks pytorch,jax \
  --epochs 500 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --eval-every 1 \
  --compare-samples 5 \
  --memory-profile \
  --time-profile \
  --memory-sample-interval 1 \
  --memory-profile-detailed-batches 1 \
  --memory-profile-detailed-epochs -1
```

### 2D Navier-Stokes

```bash
cd /workspace/NeuralOperatorRobustness2
RUN_NAME=training_fno2d_ns \
GPU_MONITOR_INTERVAL=10 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_with_gpu_monitor.sh \
adv_robust/bin/python -u tools/train_fno2d_suite.py \
  --output-root fno_training_runs/full_fno_suite \
  --run-name ns_real_initial_laxmap \
  --frameworks pytorch,jax \
  --epochs 500 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --eval-every 1 \
  --compare-samples 5 \
  --memory-profile \
  --time-profile \
  --memory-sample-interval 1 \
  --memory-profile-detailed-batches 1 \
  --memory-profile-detailed-epochs -1
```

2D 数据文件比较大，脚本会把 `y` 轨迹读到 CPU 内存里。A100 显存一般先用 `--batch-size 4` 比较稳，后面可以尝试 `8` 或 `16`。

## 输出位置

总汇总 CSV：

```text
fno_training_runs/full_fno_suite/metrics_summary.csv
```

1D Burgers 输出：

```text
fno_training_runs/full_fno_suite/burgers_nu0.001/burgers_1d/
```

2D NS 输出：

```text
fno_training_runs/full_fno_suite/ns_real_initial_laxmap/ns_2d/
```

每个任务目录里会保存：

```text
config.json
dataset_info.json
results.json
losses_pytorch.csv
losses_jax.csv
plots/loss_pytorch.png
plots/loss_jax.png
checkpoints/fno*_pytorch.pt
checkpoints/fno*_jax.pkl
parameters/pytorch_parameters.npz
parameters/jax_parameters.npz
inference/pytorch_samples.npz
inference/jax_samples.npz
comparisons/framework_comparison.json
comparisons/inference_summary.json
comparisons/inference_sample_comparison.csv
comparisons/inference_compare_pytorch_jax.png
comparisons/parameter_summary.json
comparisons/parameter_block_comparison.csv
comparisons/parameter_block_max_abs.png
comparisons/loss_compare_pytorch_jax.png
memory/gpu_samples_pytorch.csv
memory/gpu_samples_jax.csv
memory/memory_phases_pytorch.csv
memory/memory_phases_jax.csv
memory/memory_summary_pytorch.json
memory/memory_summary_jax.json
memory/gpu_memory_curve_pytorch.png
memory/gpu_memory_curve_jax.png
memory/phase_memory_peaks_pytorch.png
memory/phase_memory_peaks_jax.png
```

上面 `memory/` 目录只有在 `--memory-profile` 或 `--time-profile` 至少开一个时才会生成。两个都关掉时，`results.json` 和 `metrics_summary.csv` 仍会保留总耗时和最终误差，但 profiling 相关路径会是空值。

当 `MEMORY_PROFILE=true` 通过总入口运行时，外层 GPU 监控日志保存在：

```text
run_logs/training_fno1d_burgers/
run_logs/training_fno2d_ns/
```

`metrics_summary.csv` 里每一行是一组模型结果，包含：

```text
problem, framework, run_name, train_mse, train_rmse, train_mae,
train_relative_l2, test_mse, test_rmse, test_mae, test_relative_l2,
checkpoint_path, loss_csv, plot_path, seconds_total,
memory_summary_json, memory_samples_csv, memory_phases_csv,
sampled_whole_run_peak_gpu_memory_mib, sampled_whole_run_seconds,
profiled_train_peak_mib, profiled_inference_peak_mib,
profiled_forward_peak_mib, profiled_forward_mean_seconds,
profiled_backward_peak_mib, profiled_backward_mean_seconds,
profiled_inference_forward_peak_mib, profiled_inference_forward_mean_seconds,
profiled_backward_to_forward_peak_ratio,
profiled_backward_to_forward_mean_time_ratio
```

## 相关脚本

```text
tools/split_burgers_dataset.py
tools/train_fno1d_suite.py
tools/train_fno2d_suite.py
tools/run_full_fno_training_suite.sh
tools/fno_training_common.py
```
