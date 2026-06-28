# StablePDENet, PINO, FNO Physics Loss, And Optimizer-Curve Overlap Notes

Date: 2026-05-29

## 1. StablePDENet 做了什么

StablePDENet 研究的是 neural operator 在输入函数扰动下的稳定性。它的基本任务是学习 PDE solution operator:

```math
G: a \mapsto u
```

其中 `a` 可以是 initial condition、source/forcing term、coefficient field 或 boundary condition，`u` 是对应 PDE 解。

它的核心训练目标是一个 min-max 问题:

```math
\min_\theta \max_{\|\delta\|\le \epsilon}
L_{\mathrm{phys}}\bigl(G_\theta(a+\delta)\bigr).
```

内层用 PGD 找让 physics-informed loss 最大的输入扰动，外层训练模型抵抗这个扰动。

## 2. Physics-Informed Loss 是什么

StablePDENet 的训练/attack loss 不是 solver-supervised error，而是 PDE residual loss。

例如 Poisson 方程:

```math
-\Delta u = f
```

模型输出:

```math
u_\theta = G_\theta(f)
```

physics loss 检查:

```math
-\Delta u_\theta - f
```

一般形式是:

```math
L_{\mathrm{phys}}
= L_{\mathrm{PDE}} + \lambda_{\mathrm{BC}}L_{\mathrm{BC}}
+ \lambda_{\mathrm{IC}}L_{\mathrm{IC}}.
```

它衡量的是模型输出函数是否满足 PDE、边界条件、初始条件，而不是直接比较模型输出和 numerical solver reference solution。

## 3. 它有没有用 Solver-In-The-Loop

训练和 attack 阶段没有 solver-in-the-loop。

StablePDENet attack 的是:

```math
L_{\mathrm{phys}}\bigl(G_\theta(a+\delta)\bigr)
```

而不是:

```math
\|G_\theta(a+\delta)-S(a+\delta)\|.
```

这里 `S` 是 numerical solver。

论文的理由是，如果 PGD 每一步都对 `a+\delta` 重新跑高精度 solver，成本会很高。所以它用 physics-informed loss 作为 true solution error 的 surrogate。

但是 evaluation 阶段会用 numerical solver 生成 reference solution:

```math
u_{\mathrm{ref}} = S(a), \qquad
u_{\mathrm{ref}}^{\mathrm{adv}} = S(a+\delta).
```

然后报告 relative error:

```math
\frac{\|G_\theta(a+\delta)-S(a+\delta)\|}{\|S(a+\delta)\|}.
```

所以它的 attack loss 和 evaluation metric 不是同一个东西。

## 4. 它用的模型为什么能算时间导数

StablePDENet 基于 PI-DeepONet，而不是 FNO。

DeepONet 的形式是:

```math
G_\theta(a)(y)
= \sum_{k=1}^p b_k(a)t_k(y).
```

其中:

- branch net 输入 `a`，即输入函数的 sensor values；
- trunk net 输入 `y`，即输出坐标；
- `y` 可以是 `x`、`(x,y)`、`(t,x)` 或 `(t,x,y)`。

固定 branch input `a` 后，DeepONet 就像一个 PINN-style coordinate network:

```math
y \mapsto u_a(y).
```

如果 `y=(t,x)`，就可以用 automatic differentiation 计算:

```math
u_t,\quad u_x,\quad u_{xx}.
```

因此对 heat equation、diffusion-reaction 这种 time-dependent PDE，它可以自然构造完整时空 residual。

## 5. 它做了哪些 PDE 实验

StablePDENet 实验包括:

| PDE / ODE | 是否含时间 | 输入函数/扰动对象 |
|---|---:|---|
| Parametric ODE / antiderivative | 1D variable | forcing term |
| 1D Poisson | 否 | source term |
| 2D Poisson | 否 | source term |
| Elliptic / Helmholtz | 否 | source term |
| Heat equation | 是 | initial condition |
| Inhomogeneous heat equation | 是 | source term |
| Diffusion-reaction | 是 | source term |
| Diffusion-reaction coefficient case | 是 | coefficient |
| Steady Stokes | 否 | boundary condition |

“扰动对象”就是当前 neural operator 的输入函数。模型学的是 `u0 -> u` 就扰动 `u0`；学的是 `f -> u` 就扰动 `f`；学的是 `c -> u` 就扰动 `c`；学的是 boundary condition -> solution 就扰动 boundary condition。

## 6. 稳态 PDE 和时间 PDE 的区别

稳态 PDE 没有 initial condition。它的解通常由 source term、coefficient、boundary condition 决定。

例如:

```math
-\nabla\cdot(c(x)\nabla u)=f.
```

这类任务通常是:

```math
f \mapsto u,\qquad c \mapsto u,\qquad g_{\partial\Omega}\mapsto u.
```

时间 PDE 才自然有 initial condition:

```math
u_t=N(u),\qquad u(0,x)=u_0(x).
```

但时间 PDE 也不一定只扰动 initial condition。如果 source 或 coefficient 是输入，也可以扰动 source 或 coefficient。

## 7. StablePDENet 的稳定性解释

它把稳定性理解成输入扰动下输出变化受控:

```math
\|G_\theta(a+\delta)-G_\theta(a)\|
\le C\|\delta\|.
```

对小扰动做一阶 Taylor expansion:

```math
G_\theta(a+\delta)
\approx G_\theta(a)+DG_\theta(a)[\delta].
```

所以局部最坏放大倍数由 Fréchet derivative / Jacobian 的 spectral norm 控制:

```math
\|DG_\theta(a)\|_2
= \sigma_{\max}(J).
```

Jacobian norm 越大，说明存在某个输入扰动方向会被模型强烈放大。PGD adversarial training 反复训练这些坏方向，所以经验上会降低局部敏感性。

这个不是严格全局 stability theorem，更像 Taylor expansion 层面的解释加实验验证。

## 8. Jacobian Norm 实验怎么理解

论文比较了 PIDeepONet 和 StablePDENet 的 Jacobian spectral norm。

表里的 clean / attacked 不是模型参数改变，而是在不同输入点上计算 Jacobian:

```math
\|DG_\theta(a)\|,\qquad
\|DG_\theta(a+\delta)\|.
```

attack 不改变模型参数，只改变输入点。因为模型是非线性的，不同输入点的 Jacobian norm 可以不同。

主要结论是 StablePDENet 在 clean input 和 attacked input 附近的 spectral norm 都普遍小于普通 PIDeepONet。

## 9. StablePDENet 的局限

主要局限:

- baseline 基本是 PIDeepONet vs StablePDENet，没有系统比较 TRADES、Jacobian regularization、spectral norm regularization、solver-consistent PGD 等 robust training 方法；
- attack loss 是 physics residual，但 evaluation 是 solver-reference relative error，两者相关但不等价；
- 训练阶段没有 differentiable solver，也不是 solver-consistent adversarial training；
- 对 FNO final-snapshot setting 不能直接照搬，因为 StablePDENet 依赖 DeepONet 的 coordinate-query 结构；
- PGD 的具体数值超参数在可读版本里不够清晰。

## 10. PINO 和 FNO Physics Loss

PINO 是 Physics-Informed Neural Operator。它通常用 FNO 作为模型，然后训练:

```math
L = L_{\mathrm{data}} + \lambda L_{\mathrm{PDE}}.
```

例如 Darcy flow:

```math
-\nabla\cdot(k(x)\nabla u(x))=f(x).
```

FNO 输出 grid tensor `u_theta`，再用 finite difference 或 Fourier spectral derivative 算:

```math
r_\theta(x)
=-\nabla\cdot(k(x)\nabla u_\theta(x))-f(x),
```

然后:

```math
L_{\mathrm{PDE}}=\|r_\theta\|^2.
```

PINO 和 PI-DeepONet 的区别是:

- PI-DeepONet 对 trunk coordinate `y` 做 AD；
- PINO/FNO 对 grid tensor 做离散导数或谱导数。

## 11. Time-Dependent PDE 下 FNO Physics Loss 的问题

如果 FNO 只输出 final snapshot:

```math
u_0(x)\mapsto u(T,x),
```

那么对 Burgers:

```math
u_t + uu_x - \nu u_{xx}=0
```

或 Navier-Stokes vorticity:

```math
\omega_t + u\cdot\nabla\omega - \nu\Delta\omega - f=0
```

都没法完整构造 physics loss，因为缺少 `u_t` 或 `omega_t`。

PINO 处理 time-dependent PDE 的前提一般是输出一段时空轨迹或做 one-step transition。只有 final snapshot 时，最多只能做空间约束，不能做完整 PDE residual。

## 12. Fourier Continuation 是什么

Fourier spectral derivative 利用:

```math
\frac{d}{dx} \leftrightarrow ik.
```

但 FFT 默认函数是周期的。如果非周期函数直接 FFT，会在边界拼接处产生跳跃，导致 Gibbs oscillation 和导数误差。

Fourier continuation 的思想是:

给定非周期函数:

```math
f:[a,b]\to \mathbb{R},
```

构造更大区间:

```math
[a,c],\quad c>b,
```

以及扩展函数:

```math
f_c:[a,c]\to \mathbb{R}
```

满足:

```math
f_c(x)=f(x),\quad x\in[a,b],
```

并且首尾周期匹配:

```math
f_c^{(j)}(a)=f_c^{(j)}(c),\quad j=0,\dots,q.
```

然后在扩展区间上做 Fourier/FFT 求导，最后只截回原始区间 `[a,b]`。

这解决的是非周期边界导致 Fourier derivative 不准的问题，但不能解决“只有一帧没有时间导数”的问题。

## 13. 本次从 R2 找到的 Burgers / NS2D 四方法图

本地下载目录:

```text
/workspace/NeuralOperatorRobustness2/r2_found_20260529/
```

### Burgers 四方法 grouped curve

```text
/workspace/NeuralOperatorRobustness2/r2_found_20260529/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/Burgers/optimizer_grouped_loss_curves/burgers_optimizer_grouped_target_loss_curves_eps4_alpha0p4.png
```

单方法图:

```text
/workspace/NeuralOperatorRobustness2/r2_found_20260529/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/Burgers/figures/raw_add/
/workspace/NeuralOperatorRobustness2/r2_found_20260529/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/Burgers/figures/raw_replace/
/workspace/NeuralOperatorRobustness2/r2_found_20260529/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/Burgers/figures/steepest_add/
/workspace/NeuralOperatorRobustness2/r2_found_20260529/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/Burgers/figures/steepest_replace/
```

### NS2D 四方法 grouped curve

```text
/workspace/NeuralOperatorRobustness2/r2_found_20260529/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/NS2D/optimizer_grouped_loss_curves/ns2d_optimizer_grouped_target_loss_curves_eps32_vs_eps1.png
```

每个 epsilon/alpha 单独图:

```text
/workspace/NeuralOperatorRobustness2/r2_found_20260529/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/NS2D/optimizer_grouped_loss_curves/all_eps/
```

例如:

```text
eps32_alpha10_optimizer_grouped_target_loss_curves.png
eps16_alpha5_optimizer_grouped_target_loss_curves.png
eps8_alpha2p5_optimizer_grouped_target_loss_curves.png
eps4_alpha1p25_optimizer_grouped_target_loss_curves.png
eps2_alpha0p625_optimizer_grouped_target_loss_curves.png
eps1_alpha0p3125_optimizer_grouped_target_loss_curves.png
```

## 14. 曲线重合的数值核对

### Burgers: `loss3`, `eps4_alpha0p4`, `p=2,q=2`

Per-step CSV:

```text
/workspace/NeuralOperatorRobustness2/r2_found_20260529/per_step_metrics/burgers_loss3_eps4_alpha0p4/
```

关键曲线列比较:

| Pair | `loss3_q_mean` max abs diff | 结论 |
|---|---:|---|
| raw_replace vs steepest_replace | 0.0 | 完全重合 |
| raw_add vs steepest_add | 1.0398398055 | 不重合 |
| raw_add vs raw_replace | 2.1478374894 | 不重合 |
| steepest_add vs steepest_replace | 1.7773762783 | 不重合 |

最终 `loss3_q_mean`:

| Method | Final loss3_q_mean |
|---|---:|
| raw_add | 2.9997486937 |
| raw_replace | 3.0616590244 |
| steepest_add | 3.0650239444 |
| steepest_replace | 3.0616590244 |

### NS2D: `loss3`, `eps32_alpha10`, `p=2,q=2`

Per-step CSV:

```text
/workspace/NeuralOperatorRobustness2/r2_found_20260529/per_step_metrics/ns2d_loss3_eps32_alpha10/
```

关键曲线列比较:

| Pair | `active_loss_mean` max abs diff | 结论 |
|---|---:|---|
| raw_replace vs steepest_replace | 0.0 | 完全重合 |
| raw_add vs steepest_add | 167.2400428772 | 不重合 |
| raw_add vs raw_replace | 63.4294448853 | 不重合 |
| steepest_add vs steepest_replace | 203.2203048706 | 不重合 |

最终 `active_loss_mean`:

| Method | Final active_loss_mean |
|---|---:|
| raw_add | 155.7122276306 |
| raw_replace | 106.3228977203 |
| steepest_add | 304.5929718018 |
| steepest_replace | 106.3228977203 |

### p2q2 sweep summary

对整个 `loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520` summary 检查:

```text
raw_replace vs steepest_replace:
settings = 20
diff_cells = 0
max_numeric_diff = 0.0
```

也就是说在 p=2,q=2 的这批 continuous perturbation 实验里，`raw_replace` 和 `steepest_replace` 是逐字段完全一致的。

## 15. 为什么 p=2 下 raw_replace 和 steepest_replace 会完全重合

设当前梯度是:

```math
g=\nabla_\delta L.
```

`steepest` 方向是在 `L_p` 约束下最大化线性化增益:

```math
\max_{\|d\|_p\le 1} \langle g,d\rangle.
```

一般解与 dual norm 有关。如果 `p=2`，dual norm 也是 `q=2`，最陡方向就是:

```math
d^\star = \frac{g}{\|g\|_2}.
```

而 `raw_replace` 在 p=2 replace 情况下也是把 raw gradient 归一化到 L2 ball boundary:

```math
\delta_{\mathrm{raw\_replace}}
= \epsilon \frac{g}{\|g\|_2}.
```

`steepest_replace` 是:

```math
\delta_{\mathrm{steepest\_replace}}
= \epsilon d^\star
= \epsilon \frac{g}{\|g\|_2}.
```

所以二者数学上就是同一个更新，曲线完全重合是正常的。

## 16. 为什么 add 方法没有完全重合

`replace` 每一步丢掉旧的 perturbation，只保留当前方向投影到 boundary:

```math
\delta_{k+1} = \epsilon d_k.
```

所以只要当前方向相同，结果就完全相同。

`add` 是累积更新:

```math
\delta_{k+1}
= \Pi_{\|\delta\|_p\le\epsilon}
(\delta_k+\alpha\,d_k).
```

而代码里的 `raw_add` 和 `steepest_add` 对 step direction / normalization 的处理不同，且有历史路径依赖，所以即使 `p=2`，它们不一定压成同一条曲线。数据里也确实不是同一条。

## 17. 最终结论

你记得的是对的：在 Burgers / NS2D 的 p=2,q=2 四方法图里，不是四条都不同，也不是两两都重合；主要是这一对完全重合:

```text
raw_replace == steepest_replace
```

所以图里如果 legend 有四个方法，但视觉上只有三条甚至两条明显曲线，核心原因是 replacement pair 在 p=2 下数学上同一个更新，数据也验证为完全重合。

