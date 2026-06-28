# StablePDENet / PINO / DeepONet-FNO Discussion Summary

Date: 2026-05-29

This note summarizes our discussion about **StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations**, and the related distinction between physics-informed DeepONet, FNO/PINO, solver-consistent robustness, Jacobian sensitivity, and Fourier continuation.

## 1. StablePDENet 的核心问题

StablePDENet 研究的是 neural operator 在输入函数被扰动之后是否稳定。

一般 operator learning 任务可以写成:

```math
G: a \mapsto u,
```

where `a` is an input function and `u` is the PDE/ODE solution. Depending on the PDE task, `a` can be:

- initial condition `u_0(x)`;
- source / forcing term `f(x)`;
- coefficient / parameter field `c(x)`;
- boundary condition `g`.

StablePDENet 的问题意识是: 普通 neural operator 在 clean input 上可能很准, 但在 adversarial perturbation 下可能非常不稳定。对 PDE 来说, 输入函数一变, 真实解本身也会变, 所以这里的 robustness 不是分类任务里那种“标签不变”, 而是:

```math
\|G_\theta(a+\delta)-G_\theta(a)\| \le C\|\delta\|.
```

也就是输入小变动时, 输出解的变化应该受控。

## 2. 它用的模型: PIDeepONet / PI-DeepONet

StablePDENet 的基础模型是 physics-informed DeepONet, 即 PIDeepONet。DeepONet 的基本形式是:

```math
G_\theta(a)(y)
=
\sum_{k=1}^p b_k(a)t_k(y) + b_0.
```

这里:

- branch net 输入 `a`, 即输入函数在 sensor points 上的采样;
- trunk net 输入 `y`, 即输出解的 query coordinate;
- 输出是 `u(y)`.

Shape 可以理解成:

```text
branch input: [B, m]
trunk input:  [N, d]
output:       [B, N]
```

其中:

- `B`: batch size;
- `m`: 输入函数的 sensor 数量;
- `N`: 输出 query/collocation points 数量;
- `d`: 每个 query point 的坐标维度.

如果是 1D steady PDE, `y=x`, `d=1`.

如果是 2D steady PDE, `y=(x,y)`, `d=2`.

如果是 1D time-dependent PDE, `y=(t,x)`, `d=2`.

如果是 Stokes 这类多输出问题, 输出可以是:

```text
output: [B, N, 3]
```

对应 `(u,v,p)`.

## 3. DeepONet 为什么像 PINN

DeepONet 固定 branch input `a` 后, branch output `b_k(a)` 就是一组固定系数。此时模型变成:

```math
y \mapsto G_\theta(a)(y).
```

所以对一个固定 PDE 实例, 它很像 PINN / coordinate network:

```math
(t,x) \mapsto u(t,x).
```

区别是:

- PINN 通常只学一个具体解;
- DeepONet 通过 branch input 在很多 PDE 实例之间切换, 即学一族解函数.

这也是它能自然构造 physics loss 的原因: `t,x,y` 是 trunk network 的真实输入变量, 所以可以用 automatic differentiation 算:

```math
\partial_t u_\theta,\quad
\partial_x u_\theta,\quad
\partial_{xx}u_\theta,\quad
\Delta u_\theta.
```

## 4. StablePDENet 的 physics-informed loss

StablePDENet 的 training/attack 阶段不是直接比较模型输出和 solver reference, 而是把模型输出代入 PDE, 算 physics residual。

一般形式是:

```math
\mathcal{L}_{phys}
=
\mathcal{L}_{PDE}
+ \lambda_{BC}\mathcal{L}_{BC}
+ \lambda_{IC}\mathcal{L}_{IC}.
```

这些边界/初值项形式上像 regularization, 但本质上是 PDE problem constraints。没有边界条件或初值条件, 很多 PDE 问题不唯一。

### 4.1 Poisson example

For 1D Poisson:

```math
-u''(x)=f(x),\quad u(0)=u(1)=0.
```

The operator is:

```math
G: f(x) \mapsto u(x).
```

Residual:

```math
r_\theta(x)=-u_{\theta,xx}(x)-f(x).
```

Physics loss:

```math
\mathcal{L}_{phys}
=
\frac{1}{N}\sum_i |-u_{\theta,xx}(x_i)-f(x_i)|^2
+\lambda_b\left(|u_\theta(0)|^2+|u_\theta(1)|^2\right).
```

### 4.2 Heat equation example

For heat equation:

```math
u_t-\kappa u_{xx}=0,
\quad
u(0,x)=u_0(x),
\quad
u(t,0)=u(t,1)=0.
```

The operator is:

```math
G: u_0(x) \mapsto u(t,x).
```

Here the trunk coordinate is `(t,x)`, so the model can compute `u_t` by AD:

```math
r_\theta(t,x)
=
u_{\theta,t}(t,x)-\kappa u_{\theta,xx}(t,x).
```

Physics loss:

```math
\mathcal{L}_{phys}
=
\|u_{\theta,t}-\kappa u_{\theta,xx}\|^2
+\lambda_{IC}\|u_\theta(0,x)-u_0(x)\|^2
+\lambda_{BC}\|u_\theta|_{\partial\Omega}\|^2.
```

## 5. StablePDENet 的 adversarial objective

StablePDENet 把训练写成 min-max:

```math
\min_\theta
\max_{\|\delta\|\le \epsilon}
\mathcal{L}_{phys}(G_\theta(a+\delta)).
```

Attack 阶段固定模型参数 `theta`, 用 PGD 找让 physics loss 变大的输入扰动:

```math
\delta_{k+1}
=
\Pi_{\|\delta\|\le\epsilon}
\left(
\delta_k
+\alpha\,\mathrm{sign}
\left(
\nabla_\delta \mathcal{L}_{phys}(G_\theta(a+\delta_k))
\right)
\right).
```

这里 attack 的对象不是固定的 initial condition, 而是当前 operator learning 任务里的输入函数 `a`:

- 如果任务是 `u_0 -> u(t,x)`, 就扰动 initial condition `u_0`;
- 如果任务是 `f -> u`, 就扰动 source/forcing term `f`;
- 如果任务是 `c -> u`, 就扰动 coefficient `c`;
- 如果任务是 `g -> u`, 就扰动 boundary condition `g`.

统一写法就是:

```math
a \rightarrow a+\delta.
```

## 6. 它没有 solver-in-the-loop

StablePDENet 的 attack/training 路径是:

```text
a + delta
  -> G_theta(a + delta)
  -> u_theta
  -> L_phys
```

它不是:

```math
\|G_\theta(a+\delta)-S(a+\delta)\|.
```

也就是说, training/attack 阶段没有 numerical solver, 也没有 differentiable solver-in-the-loop。

作者这么做的原因是成本: 如果每个 PGD step 都重新算一次 high-fidelity solution

```math
S(a+\delta_k),
```

成本会很高。因此 StablePDENet 用 physics-informed loss 作为 true solution error 的 surrogate。

这带来一个重要区别:

```math
\text{attack loss}
=
\mathcal{L}_{phys}(G_\theta(a+\delta)),
```

但 evaluation 用的是:

```math
\text{eval error}
=
\frac{\|G_\theta(a+\delta)-S(a+\delta)\|}{\|S(a+\delta)\|}.
```

Attack objective 和 evaluation metric 不是同一个 loss。

## 7. Evaluation 阶段怎么做

Evaluation 阶段它会用 numerical solver 生成 reference solution。

For clean input:

```math
u_{ref}=S(a).
```

For adversarial input:

```math
u_{ref}^{adv}=S(a+\delta).
```

然后报告:

```math
\frac{\|G_\theta(a)-S(a)\|}{\|S(a)\|},
```

and

```math
\frac{\|G_\theta(a+\delta)-S(a+\delta)\|}{\|S(a+\delta)\|}.
```

这里必须重新算 `S(a+delta)`, 因为 PDE 输入已经变了, 真实解也变了。

## 8. 它实验了哪些 PDE / ODE

| Section | Problem | Time-dependent? | Input function `a` | Output coordinate `y` | Perturbed object |
|---|---|---:|---|---|---|
| 4.1 | Parametric ODE / antiderivative | 1D time-like | forcing `f(x)` | `x` | forcing |
| 4.2.1 | 1D Poisson | no | source `f(x)` | `x` | source |
| 4.2.2 | 2D Poisson | no | source `f(x,y)` | `(x,y)` | source |
| 4.3 | Elliptic / Helmholtz | no | source `f(x)` | `x` | source |
| 4.4.1 | Heat equation | yes | initial condition `u_0(x)` | `(t,x)` | initial condition |
| 4.4.2 | Inhomogeneous heat | yes | source `f(x)` | `(t,x)` | source |
| 4.5.1 | Diffusion-reaction | yes | source `f(x)` | `(t,x)` | source |
| 4.5.2 | Diffusion-reaction coefficient | yes | coefficient `c(x)` | `(t,x)` | coefficient |
| 4.6 | Stokes | no | boundary condition `g` | `(x,y)` | boundary condition |

关键规律:

- steady PDE 没有 initial condition, 解主要由 source/coefficient/boundary 决定;
- time-dependent PDE 可以扰动 initial condition, 也可以扰动 source/coefficient, 取决于当前 operator learning 的输入函数是什么;
- time-dependent experiments in StablePDENet use trunk coordinates containing time, such as `(t,x)`, so they can compute time derivatives by AD.

## 9. 训练流程: normal training, adversarial training, dynamic generation

### 9.1 Baseline solution manifold 是什么意思

“baseline solution manifold” 不要理解成作者显式做了 manifold learning。它更像一种说法:

> 先通过 normal training 让模型学会 clean input distribution 下的一族 PDE 解。

如果:

```math
M=\{u=S(a):a\sim\mathcal{D}\},
```

那么 `M` 就是输入函数分布对应的一族 solution functions。Normal training 先让模型学到这个 clean solution family / clean solution distribution。

所以它更接近:

```text
先训练出一个能解正常题的 baseline model.
```

### 9.2 为什么 normal training 和 adversarial training 交替

如果一开始模型还不会解 PDE, PGD 找出的 worst-case perturbation 可能没有意义, 因为模型本来就到处错。

所以先 normal training:

```math
\min_\theta \mathcal{L}_{phys}(G_\theta(a)).
```

之后再 adversarial training:

```math
\min_\theta \mathcal{L}_{phys}(G_\theta(a+\delta^\star)).
```

交替训练的目的:

- 保持 clean input 上的 accuracy;
- 提高 adversarial input 上的 stability;
- 避免只练 adversarial examples 导致 clean performance 下降。

### 9.3 动态生成 input function 是什么意思

因为训练用的是 physics-informed loss, 不需要每个 input function 都有 solver label `S(a)`, 所以每个 training step 可以重新采样一个新的输入函数:

```math
a^{(step)}\sim\mathcal{D}.
```

比如:

- 从 Gaussian random field 采样新的 source `f(x)`;
- 从 Gaussian process 采样新的 initial condition `u_0(x)`;
- 采样新的 coefficient field `c(x)`;
- 采样新的 boundary condition.

然后模型输出 `u_theta`, 把它代入 PDE residual 即可训练。

这和固定训练集不同。固定训练集是:

```math
\{a_1,\ldots,a_N\}
```

反复训练。动态生成是每一步都从 problem distribution 取新函数。它的好处是减少对静态有限训练集的 overfit, 让模型更像是在学整个 operator distribution。

## 10. 稳定性定义和 Taylor/Jacobian 解释

StablePDENet 想要的稳定性可以写成:

```math
\|G_\theta(a+\delta)-G_\theta(a)\|\le C\|\delta\|.
```

对小扰动做一阶 Taylor expansion:

```math
G_\theta(a+\delta)
\approx
G_\theta(a)
+ DG_\theta(a)[\delta].
```

所以输出变化近似为:

```math
G_\theta(a+\delta)-G_\theta(a)
\approx
DG_\theta(a)[\delta].
```

最坏方向的局部放大倍数是:

```math
\max_{\|\delta\|=1}\|DG_\theta(a)[\delta]\|
=
\|DG_\theta(a)\|_2.
```

离散化以后 `DG_theta(a)` 就是 Jacobian matrix, `||.||_2` 是 spectral norm, 即最大奇异值。

因此:

- Jacobian spectral norm 大: 存在输入扰动方向会被输出强烈放大;
- Jacobian spectral norm 小: 模型局部更平滑、更稳定.

PGD attack 找的是让 physics loss 变大的坏方向。Adversarial training 在这些坏方向上压低 loss, 因而可能隐式降低局部 sensitivity / Jacobian norm。

但这不是严格的全局 stability theorem。它主要是 Taylor-level explanation + empirical verification。

## 11. Jacobian / Fréchet derivative norm 实验

论文额外做了 Jacobian spectral norm 实验。它比较:

```math
\|DG_{\text{PIDeepONet}}(a)\|_2,
\quad
\|DG_{\text{StablePDENet}}(a)\|_2,
```

以及在 attacked input:

```math
\|DG_{\text{PIDeepONet}}(a+\delta)\|_2,
\quad
\|DG_{\text{StablePDENet}}(a+\delta)\|_2.
```

注意: clean / attacked 不是说模型参数变了。模型参数固定, 只是计算 Jacobian 的输入点不同:

- clean: at input `a`;
- attacked: at input `a+delta`.

非线性模型在不同输入点的局部导数可以不同。

论文报告 StablePDENet 的 spectral norm 普遍更小。例如:

| Experiment | PIDeepONet original | StablePDENet original | PIDeepONet attacked | StablePDENet attacked |
|---|---:|---:|---:|---:|
| ODE | 3.966 | 1.058 | 3.940 | 1.052 |
| 1D Poisson | 27.699 | 0.293 | 27.228 | 0.296 |
| Elliptic | 5.845 | 0.806 | 5.217 | 0.806 |
| Heat IC | 14.030 | 1.916 | 12.144 | 1.900 |
| Stokes | 0.051 | 0.011 | 0.052 | 0.011 |

作者据此认为 adversarial training 降低了 Fréchet derivative norm, 从而提高了局部稳定性。

我们的理解: 这个实验证明的是经验现象, 即 StablePDENet 的局部 sensitivity 更小; 它不是严格证明所有输入扰动下都有全局稳定性保证。

## 12. StablePDENet 实验证明了什么

它主要展示了三件事。

第一, 普通 PIDeepONet 对 adversarial input 很脆弱。Clean error 可以很小, attacked error 会暴涨。

第二, StablePDENet 在 attacked input 上明显更稳。它大多能保持 clean accuracy, 同时显著降低 attacked relative error。

第三, StablePDENet 的 Jacobian spectral norm 普遍更小, 支持它的局部稳定性解释。

它没有充分证明:

- physics residual stability 一定等于 solver-consistent stability;
- 对所有 PDE 和所有输入都有全局 stability guarantee;
- 它比其他 robust training baseline 更好.

## 13. StablePDENet 的 baseline 局限

它主要比较:

```text
PIDeepONet vs StablePDENet
```

也就是:

```text
no adversarial training vs their PGD physics-loss adversarial training
```

它没有系统比较:

- FGSM adversarial training;
- TRADES;
- random noise augmentation;
- Jacobian regularization;
- spectral norm regularization;
- PINO-style physics training;
- solver-consistent adversarial training;
- output-error attack;
- differentiable-solver-in-the-loop method.

所以它证明的是“加了作者自己的 adversarial training 比普通 PIDeepONet 稳”, 但没有充分回答“它相比其他稳定化方法是不是最好”。

## 14. FNO 和 DeepONet 的根本差别

DeepONet / PIDeepONet:

```math
(a,y)\mapsto u(y).
```

FNO:

```math
a \text{ grid tensor} \mapsto u \text{ grid tensor}.
```

DeepONet 的坐标 `y` 是真实输入, 所以可以对 `y` 自动微分。

FNO 的空间/时间坐标通常只是 tensor index, 或最多作为附加 channel 输入。输出仍然是固定网格上的 tensor, 不是一个任意 query coordinate 的 function evaluator。

所以:

| Aspect | DeepONet / PIDeepONet | FNO |
|---|---|---|
| Representation | `G(a)(y)` | grid tensor to grid tensor |
| Coordinate input | explicit trunk input | usually implicit grid index |
| Fixed `a` | coordinate network `y -> u_a(y)` | still grid-to-grid map |
| AD w.r.t. coordinates | natural | not natural |
| Physics loss | PINN-style AD residual | finite difference / spectral derivative / discrete residual |
| Arbitrary query points | natural | not natural |
| Final-snapshot time PDE residual | can avoid by querying `(t,x)` | impossible if only final frame |

This is the core reason StablePDENet's physics-loss attack does not directly transfer to snapshot-based FNO.

## 15. FNO 上 physics loss 的问题

For steady PDE, FNO physics loss can be natural.

Example: Darcy flow

```math
-\nabla\cdot(k(x)\nabla u(x))=f(x).
```

FNO outputs one spatial field `u(x)`. Since there is no time derivative, one can compute:

```math
\mathcal{L}_{pde}
=
\|-\nabla\cdot(k\nabla u_\theta)-f\|^2.
```

For time-dependent PDE, final-snapshot FNO cannot compute full PDE residual.

Example: Burgers

```math
u_t+uu_x-\nu u_{xx}=0.
```

If FNO only predicts:

```math
u_0(x)\mapsto u(T,x),
```

then we can compute spatial derivatives `u_x(T,x), u_xx(T,x)`, but not `u_t(T,x)`.

Same for 2D Navier-Stokes vorticity form:

```math
\omega_t + u\cdot\nabla\omega - \nu\Delta\omega - f = 0.
```

If output is only `omega(T,x,y)`, then `omega_t` is unavailable.

Thus:

- Darcy final snapshot: physics loss is possible;
- Burgers final snapshot: full physics loss is not possible;
- NS final snapshot: full physics loss is not possible.

If FNO outputs many time frames, one can approximate `u_t` by finite difference, but only if time resolution is sufficiently dense. Sparse time frames give a crude average slope, not a reliable local time derivative.

## 16. PINO: Physics-Informed Neural Operator

PINO is the representative line for doing physics-informed training with FNO.

The loss is:

```math
\mathcal{L}
=
\mathcal{L}_{data}
+\lambda\mathcal{L}_{pde}.
```

Data loss:

```math
\mathcal{L}_{data}
=
\|G_\theta(a)-u_{ref}\|^2.
```

PDE loss:

```math
\mathcal{L}_{pde}
=
\|R(G_\theta(a))\|^2,
```

where `R` is the PDE residual computed on the FNO output tensor.

For Darcy:

```math
\mathcal{L}_{pde}
=
\|-\nabla\cdot(k(x)\nabla G_\theta(k)(x))-f(x)\|^2.
```

PINO differs from PI-DeepONet because it usually computes derivatives by:

- finite difference;
- Fourier / spectral derivative;
- hybrid exact Fourier / automatic differentiation.

It is not simply AD with respect to continuous coordinate input.

## 17. PINO 对 time-dependent PDE 怎么办

PINO can handle time-dependent PDE only when temporal information is present.

It usually needs one of:

1. Dense time sequence output:

```math
u_\theta(t_0,x),\ldots,u_\theta(t_K,x).
```

Then:

```math
u_t(t_n,x)
\approx
\frac{u_\theta(t_{n+1},x)-u_\theta(t_n,x)}{\Delta t}.
```

2. One-step transition:

```math
u^n \mapsto u^{n+1}.
```

Then use a discrete-time residual, for example:

```math
r^n
=
u_\theta^{n+1}-u^n-\Delta t\,N(u^n).
```

3. A coordinate-based model, where time is an explicit input. This is closer to DeepONet/PINN, not standard grid FNO.

If FNO only outputs final snapshot:

```math
u_0 \mapsto u(T),
```

then PINO cannot construct full time-dependent PDE residual. It can only impose partial spatial constraints, which are not equivalent to the PDE.

## 18. Fourier spectral derivative

Fourier derivative uses the fact:

```math
\frac{d}{dx}
\quad\leftrightarrow\quad
ik
```

in frequency domain.

If:

```math
\hat{u}(k)=\mathcal{F}[u](k),
```

then:

```math
\widehat{u_x}(k)=ik\hat{u}(k),
```

and:

```math
\widehat{u_{xx}}(k)=-k^2\hat{u}(k).
```

This is efficient and accurate for smooth periodic functions.

The problem: FFT assumes periodicity. If `u(0) != u(1)`, FFT sees a jump at the boundary, causing Gibbs oscillations and bad derivatives. Since differentiation multiplies high frequencies by `k`, derivative errors can become severe.

## 19. Fourier continuation

Fourier continuation solves the nonperiodic-boundary problem by extending the function to a larger domain so that the extended function is periodic.

Given:

```math
f:[a,b]\to\mathbb{R},
```

construct:

```math
f_c:[a,c]\to\mathbb{R},\quad c>b,
```

such that:

```math
f_c(x)=f(x),\quad x\in[a,b],
```

and on the extension interval `[b,c]`, fill in a continuation function so that:

```math
f_c^{(j)}(a)=f_c^{(j)}(c),
\quad j=0,\ldots,q.
```

Then `f_c` is a smooth periodic function on `[a,c]`. One can compute Fourier derivatives on `[a,c]`, then restrict the derivative back to `[a,b]`.

Workflow:

1. Start with nonperiodic samples on `[a,b]`.
2. Add continuation samples on `[b,c]`.
3. FFT the extended periodic signal.
4. Multiply Fourier coefficients by `ik` or `-k^2`.
5. Inverse FFT.
6. Keep only the derivative values on the original interval.

This is different from zero padding. Zero padding may introduce artificial jumps. Fourier continuation constructs a smooth periodic extension.

FC-PINO uses this idea to make PINO's Fourier derivative more accurate for nonperiodic functions.

Important limitation:

Fourier continuation fixes nonperiodic derivative computation. It does not create missing time information. If a model only outputs one final frame, Fourier continuation cannot recover `u_t`.

## 20. Positioning relative to our work

StablePDENet is best understood as:

```text
physics-residual adversarial training for PI-DeepONet
```

PINO is:

```text
FNO + data loss + discretized PDE residual loss
```

FC-PINO is:

```text
PINO + Fourier continuation for nonperiodic spectral derivatives
```

Our direction is different:

```text
solver-consistent robustness / worst-case operator sensitivity for FNO
```

Key contrast:

| Method | Attack/training objective | Solver in attack? | Architecture assumption | Main limitation |
|---|---|---:|---|---|
| StablePDENet | `L_phys(G(a+delta))` | no | coordinate-query PI-DeepONet | attack loss != eval solver error |
| PINO | data loss + PDE residual | usually no solver in residual | grid FNO with discrete derivatives | time residual needs time trajectory |
| FC-PINO | PINO with Fourier continuation | no | FNO/PINO with spectral derivatives | fixes boundary derivative, not missing time |
| Our solver-consistent setting | `||G(a+delta)-S(a+delta)||` or sensitivity | yes / possibly | FNO or any neural operator | solver cost, but objective matches evaluation |

The most important sentence:

> StablePDENet stabilizes physics residuals of a coordinate-based PI-DeepONet, while our setting can study solver-consistent robustness and worst-case operator sensitivity of grid-based FNOs, especially where final-snapshot time PDEs make physics residual attacks ill-defined.

## 21. Short takeaway

StablePDENet is useful related work and a potential baseline, but it is not the same problem as solver-consistent FNO robustness.

It:

- uses PGD on the input function;
- maximizes physics-informed residual loss;
- trains PI-DeepONet to reduce that adversarial residual;
- evaluates against numerical solver references only after training;
- shows attacked errors improve and Jacobian spectral norms decrease.

Its limitations:

- no solver-in-the-loop training objective;
- attack loss and evaluation metric differ;
- mainly compares against PIDeepONet, not many robust baselines;
- relies on coordinate-query DeepONet for natural AD physics residuals;
- does not directly transfer to final-snapshot FNO for Burgers/NS.

For our work, the clean contrast is:

```text
physics residual stability
vs.
solver-consistent operator robustness
```

