# Checkpointing 之前的完整讨论：Multi-Fidelity、Progress/Resume、稳态与瞬态 PDE、Newton/Krylov/Pseudo-Time、时间推进与 Differentiable Solver

> 这份笔记专门补充上一份 `checkpoint_rematerialization_full_discussion.md` 没有覆盖的前半段讨论。  
> 内容从 multi-fidelity / hyperparameter optimization 开始，经过 fidelity 与 progress/resume 的区别、Freeze-Thaw、CFD 中不同 fidelity 的含义，再进入 steady/transient PDE、Newton/Newton-Krylov/fixed-point/pseudo-time、显式与隐式时间推进，最后讲到 differentiable PDE solver 为什么 forward-only 显存很低、reverse-mode 显存却会迅速爆炸。  
> 最后一节正好和 checkpoint/rematerialization 那份笔记衔接。

---

# 1. 最开始的问题：什么叫 Multi-Fidelity

我们一开始讨论的是 multi-fidelity optimization / multi-fidelity hyperparameter optimization。

核心问题是：

> “fidelity 到底是什么？为什么有的论文把训练 progress 当 fidelity，有的 CFD 论文却把粗网格、中等网格、细网格叫 fidelity？”

这里最终得到的第一个重要结论是：

> **fidelity 不是某一种固定物理量，而是一个问题依赖的抽象概念。**

它表示：

> 对同一个最终目标进行评估时，某个计算/实验到底有多接近最终、最昂贵、最高精度的 target evaluation。

所以 fidelity 可以由很多东西控制，例如：

- 训练 epoch / iteration 数量
- 训练数据量
- 空间网格分辨率
- 时间步长
- PDE 模型复杂度
- solver tolerance
- 收敛程度
- Monte Carlo sample 数
- simulation accuracy
- physical model fidelity

所以：

\[
\boxed{\text{fidelity 是“大概念”，progress 只是其中一种可能的 fidelity。}}
\]

---

# 2. Training Progress 为什么可以叫 Fidelity

在 hyperparameter optimization 里，真正想优化的是：

\[
x^*=\arg\min_x L(x,T),
\]

其中：

- \(x\)：hyperparameter configuration
- \(T\)：完整训练预算
- \(L(x,T)\)：训练到最终预算后的 validation loss

但完整训练很贵。

所以可以先看：

\[
L(x,t),\qquad t<T.
\]

例如：

- 训练 10%
- 训练 20%
- 训练 40%
- 训练 100%

如果低 progress 的性能和最终性能有一定相关性，就可以把训练 progress 当成一个 fidelity 变量：

\[
s\in[0,1].
\]

例如：

\[
g(x,0.1)
\]

表示 configuration \(x\) 在 10% 训练预算下的评估，

而：

\[
g(x,1)
\]

表示完整训练后的 target evaluation。

所以在这种 HPO 场景里：

> **training progress 就是 fidelity。**

---

# 3. \((x,0.1)\to(x,0.4)\) 到底是什么意思

这里当时有一个关键澄清。

如果 fidelity 是训练 progress：

\[
s=0.1\to0.4,
\]

它不是：

> “再增加 40%”

而是：

> “从总训练进度 10% 继续训练到总进度 40%”。

所以新增计算量是：

\[
40\%-10\%=30\%.
\]

如果模型 checkpoint 已经保存，那么：

- 10% 时暂停
- 以后 resume
- 继续到 40%

就是典型的 warm-start / resume。

这时 multi-fidelity HPO 和 Freeze-Thaw 的关系就非常近。

---

# 4. Freeze-Thaw Bayesian Optimization 的核心思想

我们后来确认，你记得的那类“训练一会儿，暂停，再去训练别的 configuration，之后再回来继续”的方法，最典型就是：

**Freeze-Thaw Bayesian Optimization**。

它的核心流程是：

1. 选一个 hyperparameter configuration
2. 训练一段时间
3. 暂停
4. 根据目前 learning curve 判断是否值得继续
5. 可以转去训练另一个 configuration
6. 以后又可以回来 resume 原来的模型

所以它并不是：

> 每次重新从头评估一个 fidelity

而是：

> **同一个 configuration 的 partial training state 可以被 freeze，然后 thaw/resume。**

因此：

\[
(x,0.1)\to(x,0.4)
\]

在这个场景里几乎就是：

> 从 10% training progress 的 checkpoint resume 到 40%。

---

# 5. 当 Fidelity = Progress 时，Multi-Fidelity 和 Resume 本质上非常接近

这是我们后来得到的一个重要认识。

如果 fidelity 变量本身就是：

\[
s=\text{training progress},
\]

而低 fidelity evaluation 可以 warm-start 到高 fidelity evaluation，

那么：

> **multi-fidelity evaluation 和 resumable computation 本质上已经高度重合。**

因为：

\[
(x,s_1)\to(x,s_2)
\]

不需要重新算 \(s_2\) 的全部成本，

只需要增加：

\[
s_2-s_1
\]

这一部分计算。

所以在这种情形下：

> “提高 fidelity”

其实就是：

> “继续算”。

---

# 6. 但 Fidelity 绝对不等于 Progress

后来你指出一个非常关键的问题：

CFD 里：

- coarse mesh
- medium mesh
- fine mesh

也叫 multi-fidelity。

可是这根本不是：

- 10%
- 20%
- 40%

这种 progress。

粗网格计算完成之后，并不能简单理解成：

> “fine grid 已经跑了 30%”。

它们是：

> **同一物理问题的不同近似模型 / 不同 discretization accuracy。**

所以 coarse→fine：

\[
\text{不是同一条 computation trajectory 上的 resume}
\]

而是：

\[
\text{不同 approximation level 之间的切换}.
\]

因此我们最终明确区分：

### Progress-style fidelity

例如：

- epochs
- solver iterations
- optimization iterations
- training time
- partial convergence

它通常具有：

> **可继续、可 resume**

的特点。

### Approximation-style fidelity

例如：

- mesh resolution
- time-step resolution
- model complexity
- physical model level
- data subset size

它不一定可以直接 resume。

所以：

\[
\boxed{\text{progress 可以是 fidelity，但 fidelity 不等于 progress。}}
\]

“progress-style fidelity”只是我们讨论时为了方便起的描述性名字，不是一个必须使用的正式术语。

---

# 7. BOCA / taKG 等 Multi-Fidelity HPO 为什么可以把很多不同东西都统一叫 Fidelity

我们讨论到 BOCA、taKG 等 multi-fidelity BO 时，一个很重要的理解是：

这些方法在高层数学上只需要：

\[
g(x,s),
\]

其中：

- \(x\)：设计变量 / hyperparameters
- \(s\)：fidelity variable

至于 \(s\) 具体是什么，可以变化。

例如：

- number of training iterations
- fraction of training data
- grid resolution
- solver tolerance
- simulation accuracy

所以 multi-fidelity optimization 在理论上可以把非常不同的“便宜近似机制”统一成：

\[
\text{低 fidelity evaluation}.
\]

这也是为什么你会感觉：

> “怎么训练 progress 也叫 fidelity，mesh 也叫 fidelity，data fraction 也叫 fidelity？”

因为它们在物理意义上确实不同，

但在 optimization 层面，都承担同一个角色：

> **便宜地获得一个和最终 expensive target 有相关性的近似信息。**

---

# 8. Hyperband / BOHB 与 Freeze-Thaw 的区别

我们还比较过几类 HPO 思路。

## Freeze-Thaw

核心：

- 一个配置训练一段
- 暂停
- 保存 checkpoint
- 以后可以 resume

重点是：

> **learning curve + continuation**

## Hyperband

核心：

- 同时启动很多 configurations
- 先给少量 resource
- 淘汰表现差的
- 把更多 resource 给表现好的

重点是：

> **resource allocation + early stopping**

## BOHB

把：

- Bayesian optimization
- Hyperband

结合起来。

所以它仍然是 resource-aware HPO，但候选配置的选择更聪明。

## Multi-fidelity BO / taKG / BOCA

更显式地建模：

\[
(x,s)
\]

并考虑：

- fidelity cost
- information value
- final high-fidelity objective

所以这些方法目标相近，但 formalization 不同。

---

# 9. Transient PDE 跑到较短物理时间，是否自动等于低 Fidelity

这个问题我们后来也讨论得很细。

假设最终 target 是：

\[
u(T)
\]

例如 \(T=40\) 秒。

你先只算：

\[
u(20)
\]

能不能说：

> “这是 50% fidelity”？

答案是：

> **不能自动这么说。**

因为：

\[
u(20)
\]

并不是：

\[
u(40)
\]

的粗糙版本。

它是：

> **另一个物理时间点的 state。**

除非你的 optimization problem 已经证明：

- 早期 trajectory
- 中间 state
- partial rollout

对最终 target 有稳定 predictive value，

否则它更适合叫：

> partial trajectory / partial computation

而不是天然叫 fidelity。

---

# 10. Transient PDE 更典型的 Fidelity 是什么

对于同一个 physical horizon：

\[
0\to T,
\]

更典型的 fidelity 控制包括：

- 粗空间网格 vs 细空间网格
- 大时间步 vs 小时间步
- 低阶 vs 高阶 discretization
- 简化物理模型 vs 完整模型
- loose tolerance vs tight tolerance

例如：

> 都算到 \(T=40\)，但一个用粗网格，一个用细网格。

这才是更经典的 multi-fidelity PDE 设置。

---

# 11. 从 Multi-Fidelity 讨论转到 PDE Solver：稳态和瞬态到底有什么区别

后来讨论进入 PDE solver 本身。

最核心的区别：

## Steady PDE

通常最后要解：

\[
G(u)=0.
\]

目标是找到一个 stationary solution：

\[
u^*.
\]

## Transient PDE

有真实 physical time：

\[
\frac{\partial u}{\partial t}=F(u,t).
\]

空间离散后，通常得到：

\[
M\dot u=F(u,t),
\]

然后按 physical time 一步一步推进：

\[
u^0\to u^1\to u^2\to\cdots.
\]

---

# 12. Steady PDE 为什么是 Root-Finding Problem

对于 steady PDE：

\[
G(u)=0.
\]

这里的 \(u\) 是整个离散场。

所以本质上是一个高维 nonlinear algebraic root problem。

当时我们纠正过一句不够准确的说法：

不是“直接 attack \(G(u)=0\)”，

而应该说：

> **直接求解 nonlinear system \(G(u)=0\)。**

---

# 13. Newton Method 在这里到底干什么

Newton iteration：

\[
J_G(u_k)\Delta u=-G(u_k),
\]

然后：

\[
u_{k+1}=u_k+\Delta u.
\]

这里：

\[
J_G(u_k)
\]

是 \(G\) 的 Jacobian。

当时我们还专门澄清：

> 对 root finding 来说，这是 residual 的一阶导数 Jacobian，不是 Hessian。

只有当：

\[
G=\nabla\Phi
\]

也就是 \(G\) 本身是某个 scalar objective 的 gradient 时，

Newton root finding 里的 Jacobian 才对应：

\[
\nabla^2\Phi
\]

也就是 Hessian。

---

# 14. Newton-Krylov 是什么关系

Newton-Krylov 不是简单地和 Newton 完全平行的另一类 root method。

更准确地说：

> Newton 负责 nonlinear outer iteration，Krylov method 通常负责解 Newton step 里面的大型 linear system。

Newton step：

\[
J_G(u_k)\Delta u=-G(u_k).
\]

如果 \(J_G\) 很大，不直接 factorize，

可以用：

- GMRES
- CG（如果条件合适）
- BiCGSTAB
- 其它 Krylov solver

求：

\[
\Delta u.
\]

所以常见 nested structure：

> nonlinear Newton iteration  
> → 每一步里面一个 Krylov linear solve

---

# 15. Fixed-Point Iteration 和 Newton 的共同结构

fixed point：

\[
u_{k+1}=H(u_k).
\]

Newton：

\[
u_{k+1}=u_k+\Delta u_k.
\]

pseudo-time：

\[
u^{n+1}=u^n+\Delta\tau\,\mathcal F(u^n).
\]

虽然更新公式不同，但它们共享一个很高层的结构：

> 从 initial guess 出发，不断迭代，试图找到 steady state/root。

你当时的直觉：

> “这些方法好像都在从一个 guess 开始不断往解走”

是对的。

不同点主要是：

- update direction
- step size
- stability
- convergence speed
- local/global behavior

---

# 16. Pseudo-Time / Pseudo-Transient Continuation 是什么

对于 steady equation：

\[
G(u)=0,
\]

可以人为引入一个 artificial time：

\[
\tau.
\]

例如构造：

\[
\frac{\partial u}{\partial\tau}=-G(u).
\]

当 artificial dynamics 达到 steady state：

\[
\frac{\partial u}{\partial\tau}=0,
\]

就得到：

\[
G(u)=0.
\]

重点：

> **这个 \(\tau\) 不是 physical time。**

它只是一个 numerical device。

所以 steady PDE 虽然本身没有真实时间，也可以通过 pseudo-time marching 找 steady solution。

---

# 17. Pseudo-Time 不等于 Gradient Descent

它形式上很像：

\[
u_{k+1}=u_k-\alpha G(u_k).
\]

所以容易让人觉得：

> “这是不是 gradient descent？”

只有当：

\[
G(u)=\nabla\Phi(u)
\]

时，

它才真正等价于某种 gradient flow / gradient descent。

一般 PDE residual：

\[
G(u)
\]

不一定来自某个 scalar energy gradient。

所以：

> pseudo-time 和 gradient descent 结构类似，但一般不是同一个算法。

---

# 18. Pseudo-Transient Continuation 为什么会接近 Newton

pseudo-transient continuation 往往构造类似：

\[
\frac{u_{k+1}-u_k}{\Delta\tau}+G(u_{k+1})=0.
\]

当：

\[
\Delta\tau
\]

比较小时，

行为更接近稳定 time marching。

当：

\[
\Delta\tau
\]

变得很大，

时间导数项相对变弱，

就越来越接近直接求：

\[
G(u)=0.
\]

所以它在某种意义上可以在：

> robust time marching

和：

> fast Newton-like convergence

之间过渡。

---

# 19. 不同 Initial Guess 是否总会到同一个 steady solution

不会。

如果 nonlinear problem 有多个 root，

不同 initial guess 可能进入不同 basin。

例如：

\[
G(u)=u-u^3.
\]

root 有：

\[
u=-1,\quad0,\quad1.
\]

不同初值可能收敛到不同 root。

所以不能简单说：

> steady solver 不管从哪里开始最后都一样。

---

# 20. Transient PDE：空间离散后是什么

一个 transient PDE：

\[
u_t=\mathcal F(u,t)
\]

空间离散后通常得到 ODE/DAE：

\[
M\dot u=F(u,t).
\]

然后用 time integrator：

\[
u^n\to u^{n+1}.
\]

这里的 \(n\) 是 physical time step。

所以 transient solver 的外层结构是：

> physical time stepping

这和 pseudo-time steady solve 最大区别在于：

> 这里的 time 是真实物理时间。

---

# 21. Explicit Time Stepping

例如 forward Euler：

\[
u^{n+1}=u^n+\Delta t\,F(u^n).
\]

右边只依赖已知：

\[
u^n.
\]

所以一次 physical time step 可以直接更新。

通常不需要每一步内部再解 nonlinear equation。

特点：

- 每步便宜
- 实现简单
- 但 stability 往往限制 \(\Delta t\)

例如 CFL constraint。

---

# 22. Implicit Time Stepping

例如 backward Euler：

\[
u^{n+1}=u^n+\Delta t\,F(u^{n+1}).
\]

这里未知 \(u^{n+1}\) 出现在右边。

所以每一个 physical step 都必须解：

\[
G_n(u^{n+1})=0.
\]

这就变成一个 nonlinear root solve。

常见结构：

> physical time step  
> → Newton iteration  
> → Krylov / GMRES linear solve

所以一个 transient implicit PDE solver 可能有三层嵌套：

1. physical time steps
2. Newton nonlinear iterations
3. Krylov linear iterations

---

# 23. “每一个 implicit time step 都等于重新做一次 steady simulation 吗？”

不完全是。

形式上每一个 implicit step 确实要解一个 root：

\[
G_n(u^{n+1})=0.
\]

但是它不是完全独立的 steady-state problem。

因为：

\[
u^n
\]

已经是一个非常好的 initial guess。

如果：

\[
\Delta t
\]

不大，

那么：

\[
u^{n+1}
\]

通常离 \(u^n\) 不远。

所以每个 physical step 的 Newton solve 往往只需要少数几次迭代。

因此：

> “4000 个 implicit time steps”

不等于：

> “4000 次完整独立 steady simulation”。

---

# 24. 40 秒、dt=0.01 为什么会很贵

如果：

\[
T=40,\qquad\Delta t=0.01,
\]

那么：

\[
N_t=4000.
\]

如果是 explicit：

- 每一步一个 explicit update
- 4000 个串行 time steps

如果是 implicit：

- 4000 个 physical steps
- 每一步可能若干 Newton
- 每个 Newton 里面若干 Krylov

所以 computational cost 可以非常大。

而且这些 time steps 有依赖：

\[
u^{n+1}\text{ depends on }u^n.
\]

所以时间方向本身通常不能全部并行。

---

# 25. Explicit vs Implicit 的核心 trade-off

## Explicit

- 单步便宜
- 不需要 nonlinear solve- stability 限制大
- 可能必须用很小 dt

## Implicit

- 单步贵
- 需要 nonlinear/linear solve
- 但允许更大的 dt
- 对 stiff problems 更稳定

所以最终总成本不能只看“每一步贵不贵”。

---

# 26. Forward-Only Transient Solver 为什么显存不随所有时间步线性增长

这是后来 checkpoint 讨论的直接前奏。

如果只做 forward：

\[
u^0\to u^1\to u^2\to\cdots\to u^{N_t}.
\]

计算：

\[
u^{n+1}
\]

以后，

如果：

\[
u^n
\]

以后不再需要，

就可以释放。

所以 forward-only solver 可以 streaming：

> 只保留当前 step、下一个 step 和少量 work arrays。

因此：

> **时间步很多不自动意味着 forward 显存巨大。**

时间会很长，

但 memory 可以比较稳定。

---

# 27. Differentiable Solver 为什么突然完全不一样

如果最终 loss：

\[
L=L(u^{N_t}),
\]

而我们需要对 initial condition、parameter 等求梯度，

reverse-mode 会从：

\[
u^{N_t}
\]

往前传播 adjoint。

抽象地：

\[
u_{n+1}=F(u_n,\theta).
\]

reverse：

\[
\lambda_n
=
\left(
\frac{\partial F(u_n,\theta)}{\partial u_n}
\right)^T
\lambda_{n+1}.
\]

这里的问题是：

> 计算这个局部 VJP/Jacobian action 往往需要 forward 当时的 \(u_n\) 和该 step 内部的中间变量。

所以 forward-only 时本来可以丢掉的东西，

为了 backward 现在不能丢。

---

# 28. Reverse-Mode 的核心显存矛盾

forward state：

\[
u_0\to u_1\to\cdots\to u_N.
\]

gradient/adjoint：

\[
\lambda_N\to\lambda_{N-1}\to\cdots\to\lambda_0.
\]

所以：

> forward state 是按正时间产生的，gradient 却按反方向消费这些 state。

这导致：

> 如果不重算，就必须把过去所有将来会需要的状态提前保存。

因此：

\[
\boxed{\text{forward-only memory 很低，reverse-mode tape memory 可以非常高。}}
\]

---

# 29. 为什么不是只存每个 time step 一个 field 就行

对于你的 ETDRK4 / spectral solver，

backward 不只可能需要：

\[
u_n.
\]

还可能需要：

- RK/ETDRK intermediate stages
- Fourier transforms
- inverse FFT results
- velocity
- derivatives
- nonlinear products
- de-aliasing intermediates
- other autodiff residuals

所以真正 reverse tape 远大于：

> “time steps × 一个 physical field”。

这也是你后来 NS solver backward memory 达到几十 GiB 的原因。

---

# 30. 你当时测过 NS solver backward memory 随时间步近似线性增长

你记录过 NS solver-only 的 scaling。

大致：

| t_final | steps | PyTorch backward memory | JAX backward memory |
|---:|---:|---:|---:|
| 2 | 400 | 2.12 GiB | 7.00 GiB |
| 5 | 1000 | 5.27 GiB | 14.50 GiB |
| 10 | 2000 | 10.53 GiB | 28.36 GiB |
| 20 | 4000 | 21.05 GiB | 56.08 GiB |

所以：

> reverse-mode solver memory 随 rollout length 近似线性增长。

而 forward-only memory 却很小。

这就是最直接的证据：

> memory explosion 来自 backward tape，而不是 forward simulation 本身。

---

# 31. 一张 256×256 float32 field 为什么理论上只有约 1 GB trajectory，但实际 tape 更大

4000 个 time steps：

\[
4000\times256\times256\times4\text{ bytes}
\]

约：

\[
1.05\text{ GB}.
\]

这只是假设：

> 每个 step 只存一张 float32 physical field。

实际上 ETDRK4 每个 step 内：

- 多个 stage state
- complex Fourier arrays
- velocity fields
- spatial derivatives
- nonlinear terms
- FFT work arrays

都会进入 autodiff tape。

所以实际可以是：

- PyTorch 约 21 GiB
- JAX 约几十 GiB

而不是理论上 1 GB。

---

# 32. 为什么 backward 不能简单理解成“把 forward dynamics 反着跑”

这是后来 Neural ODE / reversible 讨论的基础。

gradient backward：

\[
\lambda_{n+1}\to\lambda_n
\]

不是在做：

\[
u_{n+1}\to u_n
\]

这个 physical state inverse。

它是在做：

> Jacobian transpose / VJP propagation。

但这个 propagation 又需要 forward state。

所以：

> gradient 可以反着传  
> 不代表 primal state 也自然可以反着恢复

这是两件完全不同的事情。

---

# 33. 如果 State 可以反着恢复，为什么会很诱人

假设：

\[
u_{n+1}=F(u_n)
\]

而且存在一个：

\[
u_n=F^{-1}(u_{n+1}),
\]

并且这个 inverse：

- 唯一
- 稳定
- 便宜
- numerical accurate

那么 backward 时：

- gradient 往回传
- state 也跟着往回恢复

就不用保存整条 trajectory。

所以：

> “能不能让函数值也跟梯度一起反向走？”

是一个非常自然的问题。

这也正好引向后面 RevNet / Neural ODE adjoint / BacksolveAdjoint 的讨论。

---

# 34. 为什么我们最后会需要 checkpoint/rematerialization

generic computation 一般无法保证：

- state map 可逆
- inverse 稳定
- numerical forward/backward 精确互逆

所以只能在两个通用方案里选：

## 方案 A：全部保存

forward 时保留整个 tape。

优点：

- backward 快
- gradient 对实际 discrete program 一致

缺点：

- memory O(N)

## 方案 B：checkpoint + rematerialization

只保存一部分 anchor。

backward 需要时：

> 从真实 forward checkpoint 再正向计算缺失 segment。

优点：

- 通用
- 稳定
- 不需要 inverse
- memory 大幅下降

代价：

- 额外 forward-like computation

这正好接到上一份：

`checkpoint_rematerialization_full_discussion.md`

---

# 35. 从 Multi-Fidelity 到 Checkpointing：整个讨论其实有一条共同主线

回头看，前后两大部分并不是完全无关。

最早讨论 multi-fidelity 时，我们一直在问：

> 能不能用“部分计算 / 低成本近似”换取足够信息，而不必每次都付完整计算成本？

到了 solver / differentiable solver：

> 能不能只保留部分状态，在需要时再补算，而不必一直付完整 memory 成本？

两个问题都属于一种更高层的思想：

> **计算资源不是只有“全部做 / 全部存”两种极端，可以通过分层、部分计算、恢复、重算，在 cost、information、memory、time 之间做 trade-off。**

区别是：

- multi-fidelity 主要是在 evaluation quality / resource allocation 上 trade-off
- checkpoint/rematerialization 主要是在 memory / recomputation 上 trade-off

---

# 36. 本次前半段讨论最重要的几个结论

## 结论 1：Fidelity 是 problem-dependent 的，不等于 progress

progress 可以是一种 fidelity，

但 mesh、dt、model complexity、data fraction 也都可以是 fidelity。

---

## 结论 2：当 fidelity 恰好是 progress 且可以 resume 时，multi-fidelity evaluation 与 Freeze-Thaw continuation 非常接近

例如：

> 10% training → 40% training

只需要新增 30% 计算。

---

## 结论 3：CFD 粗/中/细网格不是“进度百分比”

它们是不同 discretization accuracy。

所以 fidelity 和 resume 不能混为一谈。

---

## 结论 4：Transient PDE 较短物理时间不自动等于低 fidelity

20 秒 state 不是天然就是 40 秒 state 的“50% fidelity”。

它首先是 partial trajectory。

---

## 结论 5：Steady PDE 的核心是 root solving

\[
G(u)=0.
\]

Newton、fixed-point、pseudo-time 都是在用不同 update rule 寻找 steady root。

---

## 结论 6：Newton-Krylov 通常是 nonlinear outer solve + linear inner solve

Newton 负责 nonlinear correction。

Krylov 负责求 Newton step 里的大型 linear system。

---

## 结论 7：Pseudo-time 是人为 numerical time，不是 physical time

它可以把 steady root problem 转成 artificial dynamics。

---

## 结论 8：Implicit transient PDE 每一个 physical step 都可能包含 nonlinear solve

所以可能出现：

> physical time → Newton → Krylov

三层嵌套。

---

## 结论 9：Forward-only 时间积分很长，不代表显存一定很大

因为旧 state 可以不断丢掉。

---

## 结论 10：Differentiable solver 的 memory explosion 来自 reverse-mode tape

backward 需要过去 forward states 和 stage intermediates，

所以长 trajectory 必须：

- 保存
- 或重建

这才是 checkpoint/rematerialization 讨论真正的起点。

---

# 37. 最终逻辑总结

这部分讨论从 multi-fidelity 开始，最后走到了 differentiable solver 的 memory 问题。

核心逻辑可以压成下面一条：

> **Multi-fidelity 告诉我们：昂贵计算不一定每次都要完整支付，可以先用低成本近似、partial progress 或低 fidelity 信息决定是否继续。PDE solver 讨论进一步区分了 steady root solve 与 transient physical-time integration，并说明 implicit transient solver 内部还可能嵌套 Newton/Krylov。到了 differentiable solver，真正的新问题出现了：forward-only 可以不断丢弃旧 state，所以显存不大；但 reverse-mode gradient 要从终点往回传播，却依赖过去正向计算时的 state 和 stage intermediates，于是必须把它们全部保存，或者以后重新构造。这就是长时间 differentiable PDE solver 显存爆炸的根本原因，也是 checkpoint/rematerialization 产生的直接动机。**