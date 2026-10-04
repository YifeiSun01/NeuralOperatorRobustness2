# Checkpointing、Rematerialization、Batch Throughput、Reversible Dynamics 与 Neural ODE Adjoint：完整讨论整理

> 这份笔记系统整理了本次对话中关于 checkpointing、rematerialization、GPU 显存与 batch throughput、reversible network、Neural ODE adjoint、continuous adjoint 与 discrete adjoint 的全部主要内容。  
> 不是机械逐字转彅，而是把从头到尾讨论过的事实、实验、推导、误解、修正和最终认识重新组织成一套连贯逻辑。

---

# 1. 最初的术语澄清：checkpointing 和 rematerialization

一开始我们先澄清了一个基础概念：

- **Checkpointing**：决定 forward 过程中哪些中间状态要留下。
- **Rematerialization（remat）**：backward 真正走到某一段时，把之前没有保存的中间状态重新算出来。

最简单的记忆：

- checkpointing = **少存**
- rematerialization = **重算**

假设 forward 是：

\[
x_0\to x_1\to x_2\to\cdots\to x_N.
\]

如果只保存：

\[
x_0,\ x_{100},\ x_{200},\ldots
\]

那么 backward 到 \(x_{100}\sim x_{200}\) 这段时，就从 \(x_{100}\) 再正向算：

\[
x_{101},x_{102},\ldots,x_{200}.
\]

所以 rematerialization 不是“把函数倒着算”，而是：

> 从已经保存的 checkpoint 出发，把需要的那一小段重新正向计算。

---

# 2. 你自己的实验：Burgers 和 NS 都做过 remat/checkpoint sweep

重新检查你仓库之后，我们确认：

## Burgers 测过

- `none`
- `chunk`
- `all/full`

其中实际最好的是 `chunk=50`。

## NS2D 测过

- `none`
- `micro`
- `chunk=20`
- `chunk=100`
- `second`

其中最终实践里最好的是 `chunk=20`。

所以你当时不是只测了一个 checkpoint 设置，而是做了多个 checkpoint 粒度 + 多个 batch 的系统 sweep。

---

# 3. Burgers：最干净的同 batch 显存/时间对照

你 Burgers 有一个非常干净的实验：

- batch = 384
- 3-step attack
- 只改变 checkpoint/remat

结果：

| Burgers, batch=384 | Peak allocated | Reserved | attack wall time |
|---|---:|---:|---:|
| `none` | 22.56 GiB | 23.00 GiB | 11.626 s |
| `chunk50` | 3.06 GiB | 3.51 GiB | 16.956 s |

所以显存：

\[
22.56\to3.06\text{ GiB}
\]

变成：

\[
\frac{3.06}{22.56}\approx0.136.
\]

即：

\[
\boxed{\text{显存约变成原来的 }1/7.37}
\]

而时间：

\[
11.626\to16.956\text{ s}
\]

只变成：

\[
\frac{16.956}{11.626}\approx1.46.
\]

即：

\[
\boxed{\text{时间只增加约 }46\%}
\]

这是整个讨论里第一个很重要、也很反直觉的事实：

> **显存可以减少很多倍，但运行时间并不会按同样倍数增加。**

---

# 4. Burgers：不同 checkpoint 粒度本身也会影响结果

batch = 512 时：

| remat | Peak allocated | 时间 |
|---|---:|---:|
| `none` | OOM | — |
| `all` | 25.49 GiB | 18.478 s |
| `chunk50` | 4.07 GiB | 17.124 s |

这里 `chunk50` 相比 `all`：

- 显存更低
- 时间还更短

所以不能简单说：

> checkpoint 越多越好

或：

> rematerialization 越激进越好

真实系统里还会受到：

- checkpoint 放在哪
- chunk 多大
- graph 结构
- allocator
- kernel launch
- XLA / PyTorch buffer planning

等影响。

---

# 5. Burgers：最终 throughput 最优来自更大的 batch

Burgers 3-step sweep：

| batch | remat | attack sec | samples/sec | peak allocated |
|---:|---|---:|---:|---:|
| 256 | none | 15.056 | 17.003 | 14.72 GiB |
| 384 | none | 11.626 | 33.029 | 22.56 GiB |
| 384 | chunk50 | 16.956 | 22.646 | 3.06 GiB |
| 512 | chunk50 | 17.124 | 29.899 | 4.07 GiB |
| 512 | all | 18.478 | 27.709 | 25.49 GiB |
| 768 | chunk50 | 17.941 | **42.807** | 6.10 GiB |

而：

- `none, batch=512` OOM
- `none, batch=768` OOM
- `all, batch=768` OOM

所以 Burgers 最佳实测 throughput：

\[
\boxed{\text{chunk50, batch=768}}
\]

达到：

\[
42.807\text{ samples/s}.
\]

相比 no-remat 最大成功点：

\[
33.029\text{ samples/s},
\]

提高约：

\[
1.30\times.
\]

这里体现出的核心逻辑是：

> checkpoint 本身让同 batch 更慢，但它释放了大量显存，使更大的 batch 能跑起来；更大的 batch 又提高了 GPU 的并行效率，所以总 throughput 反而更高。

---

# 6. NS2D A100：完整 checkpoint sweep

A100-SXM4-80GB 上，NS2D 设置：

- spatial grid = \(256\times256\)
- target frame = 19
- `fixed_step=0.005`
- 每秒 200 个 micro-step
- 可微 target rollout = 3800 micro-steps
- `loss3 + raw_add + all_w`
- p=q=2

原始 sweep：

| Mode | Batch | Max allocated | Max reserved | Warm update |
|---|---:|---:|---:|---:|
| none | 1 | 5.07 GiB | 5.49 GiB | — |
| none | 2 | 9.25 GiB | 10.42 GiB | 7.76 s |
| micro | 6 | 25.92 GiB | 27.44 GiB | — |
| micro | 12 | 50.95 GiB | 52.02 GiB | 15.01 s |
| micro | 14 | 59.29 GiB | 60.47 GiB | — |
| chunk20 | 12 | 50.95 GiB | 52.02 GiB | 16.04 s |
| chunk20 | 17 | 71.86 GiB | 73.34 GiB | 19.41 s |
| chunk100 | 15 | 63.51 GiB | 64.89 GiB | 未完整 steady 测速 |
| second | 12 | 50.95 GiB | 52.02 GiB | 15.88 s |
| second | 15 | 63.52 GiB | 64.89 GiB | 18.68 s |

OOM：

- `none`: batch 2 pass，batch 3 OOM
- `micro`: batch 14 pass，batch 15 OOM
- `chunk20`: batch 17 pass，batch 18 OOM
- `second`: batch 15 pass，batch 16 OOM

---

# 7. NS2D A100：最大 batch 处 throughput 反而最高

### none

\[
B=2,\quad T\approx7.76\text{ s/update}
\]

throughput：

\[
0.258\text{ samples/s}.
\]

### chunk20

\[
B=17,\quad T\approx19.41\text{ s/update}
\]

throughput：

\[
0.876\text{ samples/s}.
\]

所以：

- batch capacity：2 → 17 = **8.5×**
- update time：7.76 → 19.41 = **2.50×**
- throughput：0.258 → 0.876 = **3.40×**

这说明：

> 在你当时测试到的范围内，batch 继续增大时，总 throughput 还在提高，并没有先出现内部峰值。

也就是说你实际观察到的是：

> **最优 batch 基本落在“显存允许的最大可行 batch”附近。**

---

# 8. NS2D V100：最终实践配置仍是 chunk20

V100 32GB 上：

| remat | batch | status | peak GiB | sec/sample | samples/s |
|---|---:|---|---:|---:|---:|
| micro | 4 | pass | 24.64 | 2.844 | 0.352 |
| micro | 5 | pass | 31.38 | 2.622 | 0.381 |
| chunk20 | 5 | pass | 25.18 | 2.769 | 0.361 |
| **chunk20** | **6** | **pass** | **28.98** | **2.420** | **0.413** |
| chunk20 | 7 | OOM | 31.64 | — | — |
| second | 5 | pass | 28.18 | 2.778 | 0.360 |
| second | 6 | OOM | 27.97 | — | — |
| none | 2 | OOM | 10.89 | — | — |

所以最终：

\[
\boxed{\text{NS 最后实践使用的是 chunk20}}
\]

A100 和 V100 两轮最终结论一致：

- A100：chunk20, batch 17
- V100：chunk20, batch 6

---

# 9. checkpoint 的真正收益：不是“单个样本更快”，而是“允许更大的 batch”

同 batch 下：

- checkpoint 一般会更慢
- 因为 backward 时需要重算

但它降低了每个 sample 的 activation/trajectory memory。

于是可行 batch 变大。

如果 GPU 原本还没 compute-saturated，那么：

- batch 变大
- GPU 并行度提高
- fixed overhead 被更多样本摊薄
- FFT / pointwise / FNO / linear algebra 利用率提高

所以：

\[
\frac{\text{batch size}}{\text{batch wall time}}
\]

会提高。

因此最终 throughput 提升的直接来源是：

> **更大的 batch**

而不是 checkpoint 自己让单个样本计算变快。

---

# 10. GPU 显存占满 ≠ GPU 算力占满

这是本次讨论里一个非常重要的系统层认识。

在长 reverse-mode solver 中，显存里可能同时躺着：

- 很多