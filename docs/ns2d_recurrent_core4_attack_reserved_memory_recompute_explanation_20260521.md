# 2D NS Core4 Attack 显存、OOM、Checkpoint/Recompute 解释

Date: 2026-05-21

本文整理 2D Navier-Stokes recurrent FNO attack benchmark 中关于 `reserved` 显存、OOM cliff、checkpoint/remat 粒度、recomputation coverage、batch-size 与时间/显存关系的解释。本文只总结已有 probe 输出，没有新增 GPU run。

## 1. `allocated` 和 `reserved` 到底是什么意思

在 PyTorch 里通常有两个显存数字：

```text
allocated = 当前活着的 PyTorch tensor 真正在用的显存
reserved  = PyTorch 已经从 CUDA 拿到手里的缓存池显存
```

`reserved` 不是“瞎估计出来但没占用”。它是框架真的从 GPU 拿到了，别人暂时不能用。但是 `reserved` 里面不一定每个 byte 都对应一个此刻正在使用的 live tensor，其中一部分可能只是 PyTorch 缓存池，为后面复用准备。

本 attack 更麻烦，因为同一个 run 里同时有 PyTorch 和 JAX/XLA：

- PyTorch 跑 FNO recurrent model。
- JAX/XLA 跑 differentiable Navier-Stokes solver。
- PyTorch 的 `reserved` 不能完整代表 JAX/XLA 下一步还会不会申请一块很大的中间 buffer。

所以 OOM 不是只看“上一秒已经用了多少”，而是看“下一步要申请的那一整块显存能不能放下”。

## 2. 为什么 `none batch=2` 只有 10GB，但 `batch=3` 直接 OOM

已观察结果：

```text
SOLVER_REMAT=none, batch=2: pass, peak reserved ~= 10.42 GiB / 11.19 GB
SOLVER_REMAT=none, batch=3: OOM
```

这看起来反直觉，但原因是：`none` 表示我们的代码没有显式 `jax.checkpoint`。在这种情况下，JAX/XLA 反向传播时可能保留很多 solver 中间状态，或者在某一步需要一次性申请很大的 workspace/buffer。

`batch=2` 的 backward buffer plan 能塞进显存；`batch=3` 的 buffer plan 可能突然需要一块几十 GiB 级别的连续中间 buffer，于是直接 OOM。它不是从 10GB 平滑线性涨到 15GB，而是某个中间请求突然跨过了显存上限。

一句话：

```text
成功 run 的 reserved 峰值 != 下一个 batch 的最大单次申请一定能放下
```

这也是为什么 OOM 边界不是线性的。

## 3. Batch、显存、时间的总体关系

在通过区域内，checkpoint/remat 模式的显存近似线性：

```text
peak_reserved ~= base + 4.2 GiB * attack_batch_size
```

但时间增长通常是次线性的：batch 增加以后，每一步的固定开销被更多样本摊薄，所以吞吐会提高，直到撞到 OOM cliff。

例子：

```text
chunk=20, batch=12: 16.04s/update, throughput = 12 / 16.04 = 0.748 sample-updates/s
chunk=20, batch=17: 19.41s/update, throughput = 17 / 19.41 = 0.876 sample-updates/s
```

batch 从 12 增加到 17，样本数增加 `41.7%`，但 update 时间只增加约 `21%`。所以在不 OOM 的范围内，batch 越大通常越划算。

## 4. Recompute coverage 和 checkpoint 粒度不是一回事

当前 target rollout 设置：

```text
fixed_step = 0.005
每秒 200 个 solver micro-step
target_frame_index = 19
总 solver micro-step = 19 * 200 = 3800
```

`recompute coverage` 指 3800 个 solver micro-step 里，有多少被显式 `jax.checkpoint` 管住。

- `none`: 我们代码里没有显式 checkpoint，所以显式 recompute coverage 是 `0%`。
- `micro/chunk=20/chunk=100/second`: 3800 个 solver micro-step 都在 checkpoint 区域里，所以概念上是 `~100%` coverage。

但它们的 checkpoint 粒度不同：

```text
micro:      每 1 步一个 checkpoint 单位，共 3800 个小单位
chunk=20:   每 20 步一个 checkpoint 单位，共 190 个 chunk
chunk=100:  每 100 步一个 checkpoint 单位，共 38 个 chunk
second:     每 200 步一个 checkpoint 单位，共 19 个 chunk
```

可以把 3800 步想成一本 3800 页的书：

- `micro` 是每 1 页夹一个书签。
- `chunk=20` 是每 20 页夹一个书签。
- `chunk=100` 是每 100 页夹一个书签。
- `second` 是每 200 页夹一个书签。

整本书都被 checkpoint 体系覆盖，所以 coverage 都是 `~100%`。区别是书签密度，也就是反传时每次需要重算多长的一段。

这不代表 `micro` 慢 3800 倍，也不代表 `second` 慢 19 倍。JAX/XLA 会做 fusion、scan backward、buffer planning，所以真实速度必须看 benchmark。

## 5. 最终 benchmark 汇总表

| Mode | 显式 recomputation coverage | checkpoint 单位 | 3800 步里多少单位 | 最大通过 batch | 第一个失败 batch | 最大 batch 显存 | 最大 batch 一步时间 | 吞吐 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `none` | `0%` explicit | 无 checkpoint | N/A | `2` | `3` | `10.42 GiB / 11.19 GB` | `7.76s/update` at batch 2 | `0.258` | 单步快，但 batch 太小，full sweep 不划算。 |
| `micro/original` | `~100%` explicit | `1` micro-step | `3800` | `14` | `15` | `60.47 GiB / 64.93 GB` at batch 14 | `15.01s/update` measured at batch 12 | `0.799` at batch 12 | 最稳推荐 batch 12；batch 14 只是一阶 probe 通过。 |
| `chunk=20` | `~100%` explicit | `20` micro-step = `0.1s` | `190` chunks | `17` | `18` | `73.34 GiB / 78.74 GB` at batch 17 | `19.41s/update` at batch 17 | `0.876` | 当前最高吞吐，但显存很紧。 |
| `chunk=100` | `~100%` explicit | `100` micro-step = `0.5s` | `38` chunks | `15` observed | 未测 `15` 以上 | `64.89 GiB / 69.67 GB` at batch 15 | 只测 1-step：first update `23.57s` | 未排名 | 显存类似 second/chunk20，但缺 5-step 稳态测速。 |
| `second` | `~100%` explicit | `200` micro-step = `1.0s` | `19` chunks | `15` | `16` | `64.89 GiB / 69.67 GB` at batch 15 | `18.68s/update` at batch 15 | `0.803` | 折中激进配置。 |

## 6. 实际怎么选

```text
最稳 full sweep:      SOLVER_REMAT=micro, ATTACK_BATCH_SIZE=12
当前最高吞吐:         SOLVER_REMAT=chunk, SOLVER_REMAT_CHUNK_STEPS=20, ATTACK_BATCH_SIZE=17
折中配置:             SOLVER_REMAT=second, ATTACK_BATCH_SIZE=15
暂不推荐 full sweep:  SOLVER_REMAT=none，因为最大 batch 只有 2
待补测速:             chunk=100，需要 5-step steady benchmark
```

核心总结：

```text
显存：通过区域内近似线性，约 4.2 GiB / sample
OOM：边界非线性，因为 JAX/XLA 可能突然需要一块大 buffer
时间：随 batch 增长，但通常小于线性增长
吞吐：batch 越大通常越划算，直到撞 OOM cliff
```
