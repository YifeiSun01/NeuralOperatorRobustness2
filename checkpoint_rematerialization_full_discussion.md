# Checkpointing、Rematerialization、Batch Throughput、Reversible Dynamics 与 Neural ODE Adjoint：完整讨论整理

> 这份笔记系统整理了本次对话中关于 checkpointing、rematerialization、GPU 显存与 batch throughput、reversible network、Neural ODE adjoint、continuous adjoint 与 discrete adjoint 的全部主要内容。  
> 不是机械逐字转录，而是把从头到尾讨论过的事实、实验、推导、误解、修正和最终认识重新组织成一套连贯逻辑。

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

- 很多 timestep 的 state
- ETDRK4 stage
- FFT/IFFT 中间变量
- velocity
- gradients
- nonlinear products
- backward residuals

但是某一个具体 backward 时刻，真正参与运算的只是一小部分。

因此可能出现：

> **显存已经快满 / 已 OOM，但 GPU 的计算资源还没有真正吃满。**

这就是一种：

> **memory–compute mismatch**

checkpoint/remat 的价值，就是把：

> “现在不用、只是为了未来 backward 而长期占着显存的东西”

转换成：

> “未来真正需要时再重新算”

从而腾出显存给更多 batch。

---

# 11. 你当时没记录 GPU utilization，是一个现实遗憾

你当时记录了：

- peak allocated memory
- reserved memory
- nvidia-smi memory
- batch wall time
- sec/sample
- samples/sec
- OOM boundary

但没有系统记录：

- GPU-Util
- SM utilization
- memory bandwidth utilization
- kernel occupancy
- FP/Tensor utilization

因此现在能严谨说的是：

> **你的实验显示 memory wall 在 throughput 完全饱和之前就出现。**

因为：

- batch 越大，throughput 还在提高
- 下一档直接 OOM

但不能定量声称：

> 当时 GPU utilization 只有多少百分比

因为没有记录。

---

# 12. 简单 block checkpointing 模型

设总步骤数：

\[
N.
\]

每个 checkpoint segment 长度：

\[
L.
\]

checkpoint 数量：

\[
C\approx\frac{N}{L}.
\]

所以：

> checkpoint 越密，单次 rematerialization segment 越短。

---

# 13. 为什么 peak memory 近似是 \(L+N/L\)

某一 backward 时刻需要同时保存：

1. 所有 checkpoint boundaries：
   \[
   N/L
   \]

2. 当前正在 rematerialize 的 segment：
   \[
   L
   \]

因此：

\[
M(L)\sim L+\frac{N}{L}.
\]

所以：

- checkpoint 太少：segment 太长
- checkpoint 太多：checkpoint 本身太多

两个极端都不好。

---

# 14. 理论最省显存位置在 \(\sqrt N\)

最小化：

\[
L+\frac NL
\]

得到：

\[
L^*=\sqrt N.
\]

checkpoint 数量同样：

\[
C^*\approx\sqrt N.
\]

最小 memory proxy：

\[
2\sqrt N.
\]

例如 N=10000：

- \(L=100\)
- \(C=100\)
- memory proxy ≈ 200

相比 naive 10000：

\[
200/10000=1/50.
\]

所以简单理论里，显存降低几十倍是完全可能的。

---

# 15. 为什么总 rematerialization work 可能几乎不变

简单单层 block checkpointing 下：

- 100 个块 × 每块 100 步
- 50 个块 × 每块 200 步
- 20 个块 × 每块 500 步

backward 时所有块最终都会被重新 forward 一遍。

所以总 rematerialization work 仍大约：

\[
N
\]

个额外 forward step。

因此一个非常关键的认识是：

> **checkpoint 粒度可以大幅改变 peak memory，但不一定大幅改变总 FLOPs。**

这就是为什么：

> 显存下降很多倍，不意味着运行时间也必须增加很多倍。

---

# 16. 你的实测：backward 大约是 forward 的两倍

你自己的 full-solver low-level probe：

## Burgers

- forward = 0.1593 s
- backward = 0.2870 s

比值：

\[
1.80.
\]

## NS

- forward = 1.8599 s
- backward = 3.8295 s

比值：

\[
2.06.
\]

所以对你自己的 solver path 来说：

> backward ≈ 2 × forward

比 “forward 和 backward 差不多” 更符合实测。

---

# 17. 为什么 checkpoint 不会让完整计算翻倍

如果：

- forward ≈ F
- backward ≈ 2F

原来：

\[
F+2F=3F.
\]

checkpoint 多一个 forward-like recomputation：

\[
+F.
\]

变成：

\[
4F.
\]

相对原来：

\[
4/3\approx1.33.
\]

真实系统还有额外 overhead，所以 Burgers 实测同 batch 是：

\[
1.46\times.
\]

注意：

> 11.626 s 和 16.956 s 是整个 `attack_batch` wall time，不是纯 backward-only 时间。

---

# 18. 你当时的 checkpoint 是否接近 \(\sqrt N\) 理论点

## NS

真正的 differentiable target rollout：

\[
N=3800.
\]

所以：

\[
\sqrt{3800}\approx61.6.
\]

你实际测：

| 模式 | segment 长度 L | block 数 N/L | proxy L+N/L |
|---|---:|---:|---:|
| micro | 1 | 3800 | 3801 |
| chunk20 | 20 | 190 | 210 |
| 理论最优 | 61.6 | 61.6 | 123.3 |
| chunk100 | 100 | 38 | 138 |
| second | 200 | 19 | 219 |

所以你当时不是“远远没到理论点”。

相反：

> **你的 NS sweep 已经跨过了 \(\sqrt N\) 的理论平衡点。**

其中 `chunk100` 在简单 state-count 模型下很接近理论最小值。

但是真实 throughput 最佳仍是 `chunk20`。

这说明：

> 简单 \(L+N/L\) 模型只提供一个尺度；真实 GPU optimum 还受到 ETDRK4 stage、FFT workspace、allocator、PyTorch/XLA buffer planning 等影响。

---

# 19. Burgers 的 \(\sqrt N\) 理论点

Burgers：

- N = 1000

理论：

\[
\sqrt{1000}\approx31.6.
\]

最终实际：

\[
\text{chunk}=50.
\]

simple proxy：

\[
50+1000/50=70.
\]

理论最小：

\[
2\sqrt{1000}\approx63.25.
\]

只高约 11%。

所以 Burgers 最终 chunk50 和简单理论最优非常接近。

---

# 20. 为什么 forward-only 显存很小

假设：

\[
x_{n+1}=F_n(x_n).
\]

forward：

\[
x_0\to x_1\to\cdots\to x_N.
\]

如果只做 forward，算出 \(x_{n+1}\) 后，过去的 \(x_n\) 很多时候可以立即释放。

所以：

> forward 是一种 streaming computation

它不需要把整条 trajectory 全存下来。

---

# 21. reverse-mode 为什么突然需要大量显存

最终 loss：

\[
L=L(x_N).
\]

定义：

\[
\lambda_n=\frac{\partial L}{\partial x_n}.
\]

backward：

\[
\lambda_n
=
J_{F_n}(x_n)^T\lambda_{n+1}.
\]

关键是：

> 计算 \(J_{F_n}(x_n)^T\lambda_{n+1}\) 时，需要当初 forward 的 \(x_n\) 和相关中间变量。

于是出现一个方向冲突：

- primal/function state：正向产生
- gradient/adjoint：反向传播

所以：

> 梯度往回走时，又不断需要过去 forward 时的状态。

这就是 long reverse-mode tape 的根本原因。

---

# 22. 如果函数状态也能和梯度一起反向恢复，会怎样

理想情况：

- gradient 从终点往回
- primal state 也从终点往回恢复

那么就不需要 forward 时把全部中间状态都保存。

所以：

> **如果 primal state 能可靠地从未来 state 恢复过去 state，trajectory memory 的主要问题会大幅消失。**

这正是 RevNet 和原始 Neural ODE adjoint 背后的共同直觉。

---

# 23. primal state 是什么

Primal state 就是 forward 过程中真实的状态值。

例如：

\[
x_0,x_1,\ldots,x_N.
\]

对于 PDE solver，包括：

- 当前 physical field
- Fourier coefficient
- RK/ETDRK stage
- 其它 forward 中间量

---

# 24. adjoint state 是什么

Adjoint state 可以理解为：

> loss 对某个中间 state 的敏感度。

例如：

\[
\lambda_n=\frac{\partial L}{\partial x_n}.
\]

它在 backward 中从后往前传播。

有限维欧氏空间里，adjoint operator 对应 transpose；复数情况对应 conjugate transpose。

---

# 25. RevNet 是什么

RevNet 不是：

> 找一个特殊的可逆激活函数

而是：

> **把整个 block 结构专门设计成可逆。**

经典 coupling：

\[
y_1=x_1+F(x_2),
\]

\[
y_2=x_2+G(y_1).
\]

inverse：

\[
x_2=y_2-G(y_1),
\]

\[
x_1=y_1-F(x_2).
\]

重点是：

> \(F\) 和 \(G\) 自己甚至可以不可逆。

整个 block 仍然通过 coupling structure 保持可逆。

---

# 26. RevNet 为什么不具有完全通用性

一般 neural network 可能有：

- ReLU
- max pooling
- clipping
- projection
- reduction
- dimension reduction

很多都是 many-to-one。

例如 ReLU 输出 0 时，你无法知道原输入是 0、-1 还是 -100。

所以一般网络不能默认可逆。

RevNet 能低内存，是因为：

> 它从 architecture 设计阶段就人为加入可逆性。

---

# 27. PDE solver 更不能随意设计成 RevNet

对于 PDE：

\[
u_{n+1}=\Phi_{\Delta t}(u_n)
\]

这个 update map 受：

- PDE 物理规律
- numerical discretization

共同决定。

比如：

- Euler
- RK
- ETDRK4
- projection
- splitting
- filtering

不能只为了反向恢复 state 就任意改结构，否则会改变：

- accuracy
- stability
- conservation
- numerical solution

所以 RevNet-style exact reversibility 不是 generic PDE solver 的通用属性。

---

# 28. Neural ODE 是什么

Neural ODE 把网络隐藏状态写成：

\[
\frac{dz}{dt}=f_\theta(z,t).
\]

用 ODE solver 从：

\[
z(0)
\]

积分到：

\[
z(T).
\]

它可以理解为 continuous-depth neural network。

---

# 29. 原始 Neural ODE adjoint 的核心思路

定义 adjoint：

\[
a(t)=\frac{\partial L}{\partial z(t)}.
\]

continuous adjoint ODE 可以从终点往回积分。

但计算 adjoint 需要 forward trajectory：

\[
z(t).
\]

原始 Neural ODE 的核心办法是：

> forward 不保存整个 trajectory；backward 从终点 \(z(T)\) 反时间积分，把过去的 \(z(t)\) 重建出来，同时传播 adjoint \(a(t)\)。

所以从直觉看：

> **函数 state 和梯度 state 一起往回走。**

---

# 30. RevNet 和 Neural ODE backsolve 是不是同一种方法

不是。

共同点：

- 都不想保存完整 forward trajectory
- 都想在 backward 时恢复 primal state

区别：

## RevNet

通过人为设计的显式代数逆：

\[
z_n=F_n^{-1}(z_{n+1}).
\]

## Neural ODE backsolve

通过反时间 numerical integration：

\[
z(T)\to z(t)\to z(0).
\]

所以：

> **思想类似，但数学机制不同。**

---
# 31. 为什么一般函数不能“和梯度一样倒着传”

第一个根本问题：

> **不可逆**

如果：

\[
F(x_1)=F(x_2),\quad x_1\ne x_2,
\]

那么输出无法唯一确定输入。

这不是实现问题，是数学上就没有唯一 inverse。

---

# 32. continuous flow 可逆，不代表 numerical solver 精确可逆

即使连续 ODE flow 理论上满足：

\[
\Phi_h^{-1}=\Phi_{-h},
\]

实际电脑算的是数值近似：

\[
\Psi_h.
\]

一般没有：

\[
\Psi_{-h}=\Psi_h^{-1}.
\]

例如：

\[
\dot x=\lambda x.
\]

forward Euler：

\[
x_1=(1+\lambda h)x_0.
\]

再用 Euler 往回：

\[
\tilde x_0=(1-\lambda h)x_1.
\]

结果：

\[
\tilde x_0=(1-\lambda^2h^2)x_0.
\]

所以：

\[
\tilde x_0\ne x_0.
\]

这说明：

> **连续数学系统可逆，不等于离散 numerical algorithm 可逆。**

---

# 33. 即使 inverse 存在，也可能数值极不稳定

例如：

\[
\dot x=-\lambda x.
\]

forward 是耗散：

\[
x(T)=e^{-\lambda T}x(0).
\]

误差被压缩。

反过来：

\[
x(0)=e^{+\lambda T}x(T).
\]

微小误差会被指数放大。

所以：

> **invertible ≠ stably invertible**

而且：

> **forward 越耗散、越稳定，inverse 可能越危险。**

这对 diffusion、viscous Burgers、Navier–Stokes 尤其重要。

---

# 34. Sigmoid inverse 的类比

sigmoid 数学上可逆。

但在饱和区：

- forward slope 很小
- inverse slope 很大

因此输出中很小的数值误差，反演到输入时可以变得很大。

这个直觉和耗散动力系统的 inverse conditioning 非常类似。

---

# 35. Forward trajectory 和 reverse trajectory 为什么不同

forward 实际走：

\[
x_0,x_1,\ldots,x_N.
\]

backsolve 重建：

\[
\tilde x_N,\tilde x_{N-1},\ldots,\tilde x_0.
\]

一般：

\[
\tilde x_n\ne x_n.
\]

这不是第四个完全独立的新问题，而是前面几种机制的结果：

1. numerical integrator 不精确 time-reversible
2. floating-point / discretization error
3. inverse dynamics 可能放大这些误差
4. 更一般映射甚至根本不可逆

所以：

> trajectory mismatch 是“现象”；不可逆、数值不精确可逆、逆向不稳定是原因或放大机制。

---

# 36. 连续 loss 和离散 loss：\(L_{\mathrm{cont}}\) 与 \(L_h\)

原始连续 ODE：

\[
\dot y=f(y,\theta).
\]

理想连续解产生：

\[
L_{\mathrm{cont}}(\theta).
\]

电脑实际用 finite-step numerical solver：

\[
y_{n+1}=\Phi_h(y_n,\theta)
\]

得到：

\[
L_h(\theta).
\]

一般：

\[
L_h\ne L_{\mathrm{cont}}.
\]

---

# 37. 连续梯度和离散程序梯度也不同

一般：

\[
\nabla L_h\ne\nabla L_{\mathrm{cont}}.
\]

## Discrete adjoint / ordinary autodiff

直接对实际 numerical program 求导。

得到：

\[
\nabla L_h.
\]

它是：

> 对实际执行的离散程序的正确梯度。

## Continuous adjoint / BacksolveAdjoint

先对连续 ODE 推导 adjoint，再数值解 adjoint。

它瞄准：

\[
\nabla L_{\mathrm{cont}}
\]

的数值近似。

所以有：

> discretize then differentiate

和：

> differentiate then discretize

两个不同顺序。

它们有限步长下一般不一样。

---

# 38. 反直觉点：更“真实”的连续梯度，不一定是实际程序的正确梯度

例如：

\[
\dot y=\lambda y.
\]

连续精确解导数：

\[
e^{\lambda h}.
\]

但如果电脑 forward 用 Euler：

\[
y_1=(1+\lambda h)y_0.
\]

那实际代码 derivative 是：

\[
1+\lambda h.
\]

所以：

> “更接近真实 continuous system 的 derivative”

不是：

> “刚才这段 numerical program 的 derivative”

这就是 continuous adjoint 和 discrete adjoint 的本质差别。

---

# 39. 为什么 forward 离散，backward 也应该对离散程序求导

如果 forward 实际算的是：

\[
L_h,
\]

那么 backward 最自洽的问题应该是：

> 刚才这个 \(L_h\) 对参数的导数是多少？

也就是：

\[
\nabla L_h.
\]

不能 forward 用离散 solver，backward 却突然换成连续系统的另一个 gradient。

否则相当于：

> forward 优化一个函数，backward 使用另一个函数的梯度。

这就是 gradient inconsistency。

---

# 40. \(h\to0\) 时，continuous 和 discrete 会逐渐靠拢

如果数值方法收敛：

\[
L_h\to L_{\mathrm{cont}}.
\]

在适当条件下：

\[
\nabla L_h\to\nabla L_{\mathrm{cont}}.
\]

所以两者不是永久无关，而是在有限步长下存在差别。

---

# 41. Diffrax 为什么默认 RecursiveCheckpointAdjoint

Diffrax 当前默认：

`RecursiveCheckpointAdjoint`

它的核心：

- 对实际 numerical solution 直接求导
- recursive checkpointing 控制 memory
- 对大多数问题推荐

---

# 42. Diffrax 为什么说 BacksolveAdjoint “not recommended”

官方文档核心逻辑：

- `BacksolveAdjoint` 低 memory
- 但 computed gradients 是 approximate
- recursive checkpointing 同样可以低 memory
- 因此实践中 RecursiveCheckpointAdjoint 基本总是更值得优先

“approximate gradient”的主要含义：

## 第一层：continuous/discrete gradient mismatch

Backsolve 针对连续 adjoint。

forward 却是离散 numerical solver。

所以它不是实际离散程序的 exact derivative。

## 第二层：reverse reconstructed trajectory mismatch

它从终点逆积分恢复过去 state。

这个 reconstructed trajectory 一般不完全等于真实 forward trajectory。

如果 inverse dynamics 不稳定，误差还能被进一步放大。

---

# 43. ANODE 2019 的批评

ANODE 对原始 Neural ODE adjoint 的核心批评：

1. reverse trajectory reconstruction 可能数值不稳定
2. forward/reverse trajectory 会因离散误差不一致
3. continuous adjoint 与 actual discrete solver gradient 不一致
4. gradient error 可能很大，甚至影响训练

所以：

> 从终点一路恢复 state

没有成为 generic reverse-mode AD 的通用基础。

---

# 44. ACA 为什么使用 trajectory checkpoints

ACA 的思想：

> forward 时保存一些真实 trajectory anchor。

backward：

- 从真实 forward checkpoint 出发
- 局部重新正向算
- 再局部 backward

这其实重新回到了：

> **可信 forward anchor + local rematerialization**

---

# 45. 为什么 JAX / PyTorch 的通用方案仍然是 checkpoint/remat

因为 generic computation graph 不能保证：

- inverse 存在
- inverse 稳定
- numerical solver reversible
- continuous/discrete gradient 一致

而 checkpoint/remat 只要求：

> 原来的 forward computation 能再执行一次。

因此它非常通用。

---

# 46. 本次讨论最核心的完整逻辑链

## 第一步：forward-only

函数 state 正向传播。

算完下一步后，旧 state 很多都可以丢。

所以显存需求不高。

## 第二步：reverse-mode

gradient 反向传播。

但是每个 backward step 又需要过去 forward 时的 state。

于是这些 state 不能都丢。

## 第三步：最直接办法

把整个 forward trajectory 全存。

结果：

> long rollout → huge memory tape → OOM

## 第四步：理想替代

如果 state 本身也能和 gradient 一起从终点往回恢复，就不需要全存。

RevNet、Neural ODE backsolve 都使用过这种思想。

## 第五步：为什么它不能成为通用方法

因为存在：

1. 一般映射不可逆
2. numerical solver 不精确 time-reversible
3. inverse 可能数值极不稳定
4. reconstructed reverse trajectory 会偏离 original forward trajectory
5. continuous adjoint gradient 与实际 discrete solver gradient 不一致

## 第六步：通用可靠办法

不依赖 inverse。

只保存少量 checkpoint。

需要某段时：

> 从可信 checkpoint 再正向重算。

## 第七步：为什么值得

因为：

- peak memory 可以大幅下降
- total recomputation 通常只多约一遍 forward-like work
- runtime penalty 远小于 memory saving

## 第八步：为什么最终 throughput 还能提高

memory 降低 → batch 变大 → batch parallelism 增加。

如果 GPU compute 还没饱和：

- batch 增长快于 wall time
- samples/sec 提高

直到：

- compute saturation
- 或新的 memory wall

---

# 47. 本次讨论最令人意外的几个结论

## 结论 1

**正向传播本身不是显存爆炸的根源。**

显存爆炸来自：

> backward 要反着走，但又需要过去 forward 时的 state。

---

## 结论 2

**如果 primal state 能和 gradient 一起可靠地反向恢复，trajectory memory 问题会大幅消失。**

但一般做不到。

---

## 结论 3

**“数学上可逆”远远不够。**

还必须：

- numerical inverse 存在
- 条件数好
- reverse integration 稳定
- numerical forward/backward 真正匹配

---

## 结论 4

**forward 越耗散、越稳定，inverse reconstruction 反而可能越危险。**

这对 diffusion / viscous PDE 尤其重要。

---

## 结论 5

**一个更接近连续真实物理系统的梯度，不一定是实际 numerical program 的正确梯度。**

如果 forward 是离散的，backward 对离散程序本身求导通常更自洽。

---

## 结论 6

**Checkpointing 可以让显存下降很多倍，但时间只增加有限。**

因为它通常只是多一遍 forward-like recomputation，而不是把整个计算反复做很多遍。

---

## 结论 7

**Checkpointing 的最终系统收益主要来自更大的 batch，而不是它自己计算更快。**

它本身反而会更慢。

但是显存省下来之后：

- batch 可以变大
- GPU 并行度可以提高
- throughput 可以提高

---

## 结论 8

**显存饱和和 GPU 计算饱和完全不是一回事。**

在 reverse-mode long rollout 中，显存里很多东西只是在等未来 backward 使用。

所以：

> memory wall 可能远早于 compute saturation 到来。

---

# 48. 你自己的实验最终支持了什么

## Burgers

同 batch=384：

- 22.56 GiB → 3.06 GiB
- memory ≈ 1/7.37
- attack wall time 11.626 s → 16.956 s
- runtime ≈ 1.46×

最终：

- chunk50
- batch 768
- 42.807 samples/s

---

## NS2D A100

- no-remat max pass batch = 2
- chunk20 max pass batch = 17
- batch capacity = 8.5×
- update time = 2.5×
- throughput = 3.4×

---

## NS2D V100

最终：

- chunk20
- batch 6
- batch 7 OOM
- 0.413 samples/s

---

# 49. 最终一句话总结

整个问题最底层可以这样概括：

> **函数状态是在 forward 中正向产生的，而梯度是在 backward 中反向传播的；反向传播又依赖过去的 forward 状态。forward-only 时旧状态可以不断释放，但 reverse-mode 为了未来的 backward 必须保存它们，或者以后重新得到它们。理想情况下可以让 state 也从终点一路反向恢复，但一般映射可能不可逆，数值 solver 通常也不精确可逆，逆向过程还可能数值不稳定，而且 continuous backsolve 得到的梯度还可能与实际 discrete forward 的梯度不一致。因此通用框架最终选择 checkpoint + forward rematerialization：少量保存、需要时再正向重算，用额外计算换大量显存，并把省下的显存转化为更大的 batch 和更高的实际 throughput。**