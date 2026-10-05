# Solver 动态迭代、反向传播与 Burgers / Navier–Stokes / Darcy 实现梳理

## 1. 这次讨论的核心问题

这次讨论围绕一个核心问题展开：

> 对 PDE solver 做可微分计算时，如果 solver 的迭代次数不是预先固定的，而是由收敛条件决定，那么反向传播到底怎么处理？

进一步又把这个问题拆到了三个具体场景：

1. **steady-state solver**
   - 例如 Darcy flow。
   - 求解过程中不断做数值迭代。
   - 直到 residual 小于给定 tolerance 才停止。
   - 不同输入可能需要不同的迭代次数。

2. **implicit time-evolution solver**
   - 例如 implicit Euler、BDF、Newton–Krylov 之类的时间推进。
   - 每一个 physical time step 内部还需要解一个隐式方程。
   - 每个 time step 内部可能需要不同数量的 Newton / Krylov iteration。
   - 所以整个 forward 计算路径长度可以依赖输入。

3. **explicit / explicit-style time stepping**
   - 例如这次仓库中实际使用的 Exponax ETDRK4 Burgers 和 Navier–Stokes solver。
   - 每一个 time step 内部执行固定数量的 stage。
   - 没有“不断迭代直到 residual 小于 epsilon”的内部收敛循环。

下面把这些概念完整整理。

---

# 2. 反向传播真正要求什么

假设一个 solver 的 forward 过程是

\[
u^{(0)}
\rightarrow
u^{(1)}
\rightarrow
u^{(2)}
\rightarrow
\cdots
\rightarrow
u^{(K)}.
\]

这里的 \(K\) 可能是固定的，也可能由运行时收敛条件决定。

对于普通的 reverse-mode automatic differentiation，如果直接沿 forward 的实际计算过程进行反向传播，那么最关键的条件是：

\[
\boxed{
\text{在 backward 真正开始时，这一次 forward 实际执行过的计算依赖关系必须已经确定。}
}
\]

也就是说，这一次究竟走过哪些节点、多少次运算、计算图具体是什么结构，在 backward 开始时必须已经成为确定的事实。

这并不要求 \(K\) 在程序开始前就知道。

例如：

\[
K = \min\{k:\|r_k\|<\varepsilon\}.
\]

forward 开始之前不知道 \(K\)。

solver 运行之后发现：

\[
K=837.
\]

那么这一次 forward 的实际路径就是

\[
u^{(0)}
\rightarrow
u^{(1)}
\rightarrow
\cdots
\rightarrow
u^{(837)}.
\]

如果采用直接 unrolled backpropagation，那么这一次 backward 就沿这条已经形成的路径往回传播。

下一次输入可能得到

\[
K=1142,
\]

那就是另外一张计算图，再对那张图做反向传播。

所以：

\[
\boxed{
\text{不同 forward 的图可以不同，但每一次 backward 所对应的那张图必须已经确定。}
}
\]

---

# 3. “shape 固定”和“iteration 数固定”是两件不同的事

这次讨论里一个很重要的区分是：

\[
\boxed{\text{状态张量 shape}}
\]

和

\[
\boxed{\text{solver 实际迭代次数}}
\]

不是同一个概念。

例如一个 steady-state solver：

\[
u^{(k+1)}=G(u^{(k)},\theta),
\]

可能：

\[
u^{(k)}\in\mathbb{R}^{256\times256}
\]

在所有 iteration 中 shape 都完全不变。

但停止次数：

\[
K
=
\min\{k:\|R(u^{(k)})\|<\varepsilon\}
\]

可以依赖输入。

例如：

\[
K_1=800,
\qquad
K_2=1037,
\qquad
K_3=421.
\]

因此完全可以出现：

\[
\boxed{\text{固定 tensor shape + 动态 iteration count}}
\]

这在数值 solver 中非常常见。

---

# 4. PyTorch 与 JAX 在动态计算图上的差异

## 4.1 PyTorch eager autograd

PyTorch eager autograd 的思路是：

> forward 实际执行了什么，就为这些实际执行的操作建立 autograd graph。

例如：

```python
u = u0

while residual(u) > tol:
    u = solver_step(u)

loss = objective(u)
loss.backward()
```

假设这一次 forward 实际循环 873 次，那么这一次产生的计算链就是：

\[
u_0
\rightarrow
u_1
\rightarrow
\cdots
\rightarrow
u_{873}
\rightarrow
L.
\]

backward 再沿着这一次已经生成好的计算图反传。

下一次 forward 可能跑 921 次，也可以重新生成另一张不同大小的图。

所以对于 PyTorch eager 模式：

\[
\boxed{
\text{iteration 数不需要在 forward 开始之前就固定。}
}
\]

但 backward 开始时，这一轮实际走过的路径已经确定。

---

## 4.2 JAX 的情况更严格

JAX 的编译和控制流约束更强。

例如：

```python
jax.lax.while_loop(...)
```

可以在 forward 中表示运行时决定的循环次数：

\[
K=\text{runtime determined}.
\]

但是这种动态 `while_loop` 不适合直接做普通 reverse-mode unrolling，因为 reverse-mode 需要处理 forward 过程中产生的中间状态，而循环长度可能没有静态上界。

另一方面：

```python
jax.lax.scan(...)
```

通常用于静态已知长度的循环。

如果：

\[
K=1000
\]

在 tracing / compilation 时已经确定，那么：

```python
jax.lax.scan(step, u0, xs=None, length=1000)
```

对应的计算长度是静态的，reverse-mode 更容易处理。

因此可以把 JAX 中的区别理解成：

\[
\boxed{
\text{dynamic while-loop}
\neq
\text{static scan}
}
\]

其中静态 `scan` 更适合直接 reverse-mode differentiation。

---

# 5. 收敛型 solver：iteration 数为什么会不确定

很多 steady-state solver 或 implicit solver 都采用如下结构：

\[
u^{(k+1)}=G(u^{(k)}).
\]

每一步之后检查 residual：

\[
r_k=R(u^{(k)}).
\]

然后：

\[
\|r_k\| < \varepsilon
\]

时停止。

因此：

\[
K
=
\min\{k:\|r_k\|<\varepsilon\}.
\]

\(K\) 是运行时才知道的。

不同输入、不同系数场、不同初始条件，甚至参数的很小变化，都可能让：

\[
K
\]

发生变化。

例如：

\[
a_1(x)\Rightarrow K=250,
\]

\[
a_2(x)\Rightarrow K=417,
\]

\[
a_3(x)\Rightarrow K=183.
\]

这里应该特别注意：

## solver 一般检查的是 residual，不是真实 solution error

通常不知道真实解 \(u^\star\)，所以 solver 很难直接判断：

\[
\|u_k-u^\star\|<\varepsilon.
\]

实际更常见的是检查：

\[
\|r_k\|
=
\|b-Au_k\|
\]

或者 nonlinear residual：

\[
\|F(u_k)\|.
\]

因此更精确的描述是：

\[
\boxed{
\text{solver 根据 residual / convergence criterion 决定是否停止。}
}
\]

---

# 6. implicit time evolution 的典型结构

对于 implicit time stepping，例如：

\[
R(u_{n+1},u_n,\theta)=0,
\]

每一个 physical time step：

\[
u_n\rightarrow u_{n+1}
\]

本身需要内部求解。

例如 Newton iteration：

\[
u_{n+1}^{(0)}
\rightarrow
u_{n+1}^{(1)}
\rightarrow
\cdots
\rightarrow
u_{n+1}^{(K_n)}.
\]

直到：

\[
\|R(u_{n+1}^{(K_n)},u_n,\theta)\|
<
\varepsilon.
\]

不同 physical time step 可以有不同的内部 iteration 数：

\[
K_1=6,
\quad
K_2=8,
\quad
K_3=9,
\quad
K_4=7,
\quad
K_5=5.
\]

那么这一轮 forward 的实际结构就是：

\[
u_0
\xrightarrow[\text{6 iterations}]{}u_1
\xrightarrow[\text{8 iterations}]{}u_2
\xrightarrow[\text{9 iterations}]{}u_3
\xrightarrow[\text{7 iterations}]{}u_4
\xrightarrow[\text{5 iterations}]{}u_5.
\]

如果使用直接 unrolled backpropagation，那么 backward 就沿这一次实际发生的整条路径反传。

下一次 forward 可能变成：

\[
(7,4,9,5,11),
\]

那就是另外一条计算路径。

所以：

\[
\boxed{
\text{每一次 forward 可以有不同的内部 iteration 数。}
}
\]

而：

\[
\boxed{
\text{每一次 backward 开始时，该次 forward 的实际路径必须已经确定。}
}
\]

---

# 7. 直接 unrolled backprop 与 implicit differentiation

这里是整个问题最重要的分叉。

## 7.1 直接 unrolled backpropagation

forward：

\[
u^{(0)}
\rightarrow
u^{(1)}
\rightarrow
\cdots
\rightarrow
u^{(K)}.
\]

backward 直接沿着：

\[
u^{(K)}
\rightarrow
u^{(K-1)}
\rightarrow
\cdots
\rightarrow
u^{(0)}
\]

求梯度。

此时 forward 的所有实际 iteration 都属于 backward 的计算历史。

所以：

\[
\boxed{
K\text{ 在 backward 开始时必须已经确定。}
}
\]

但 \(K\) 可以在 forward 运行过程中动态决定。

---

## 7.2 Implicit differentiation

如果最终解满足：

\[
F(u^\star,\theta)=0,
\]

那么可以直接对这个最终方程求导：

\[
\frac{\partial F}{\partial u}
\frac{du^\star}{d\theta}
+
\frac{\partial F}{\partial\theta}
=0.
\]

因此：

\[
\frac{du^\star}{d\theta}
=
-
\left(
\frac{\partial F}{\partial u}
\right)^{-1}
\frac{\partial F}{\partial\theta}.
\]

这样 backward 不需要沿 forward solver 的每一次内部 iteration 反向走一遍。

也就是说：

\[
\boxed{
\text{forward solver 跑了多少次 iteration，可以和 backward graph 解耦。}
}
\]

这正是这次仓库中 Darcy solver 的情况。

---

# 8. 这次仓库实际检查结果

检查的仓库：

```text
YifeiSun01/NeuralOperatorRobustness2
branch: merge-vast-ai-darcy-flow
```

主要相关路径：

```text
1D_Burgers/solvers/burgers1d_solvers.py
2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py
2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame.py
2D_Darcy_FNO2d/solvers/darcy_jax_solver.py
2D_Darcy_FNO2d/data_generation/generate_darcy_grf_batched.py
```

另外 requirements 中固定了：

```text
exponax @ git+https://github.com/YifeiSun01/modified_exponax.git@febe20b2103192654b256243eb464742616e95df
```

因此也检查了对应的 modified Exponax 实现。

---

# 9. Burgers solver 的实际结构

文件：

```text
1D_Burgers/solvers/burgers1d_solvers.py
```

实际使用的 Exponax solver 中：

```python
steps = int(t_final / step)
```

然后：

```python
slower_burgers_stepper = ex.stepper.Burgers(
    1,
    self.L,
    self.nx,
    step,
    diffusivity=self.nu,
    convection_scale=1.0,
    order=4,
    conservative=self.conservative,
)
```

再：

```python
longer_rollout_advection_stepper = ex.rollout(
    slower_burgers_stepper,
    steps,
    include_init=True,
)
```

因此 Burgers 的总 time-step 数是：

\[
N_{\text{step}}
=
\frac{T}{\Delta t}.
\]

例如：

\[
T=1,
\qquad
\Delta t=0.001,
\]

则：

\[
N_{\text{step}}=1000.
\]

这里没有：

```text
while residual > epsilon:
    iterate
```

也没有 Newton / CG 形式的内部收敛循环。

所以：

\[
\boxed{
\text{Burgers Exponax 是固定步数时间推进。}
}
\]

---

# 10. Navier–Stokes solver 的实际结构

核心 attack 实现位于：

```text
2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py
```

代码先定义：

```python
steps_per_second = int(round(1.0 / fixed_step))
```

然后每秒内部：

```python
u_next, _ = jax.lax.scan(
    micro,
    carry,
    None,
    length=steps_per_second,
)
```

最外层：

```python
_, seconds = jax.lax.scan(
    one_second,
    u,
    None,
    length=int(t_final),
)
```

因此 micro-step 总数是：

\[
N_{\text{micro}}
=
t_{\text{final}}
\times
\frac{1}{\Delta t}.
\]

例如：

\[
\Delta t=0.005,
\]

则：

\[
\frac{1}{0.005}=200
\]

个 micro-steps / second。

如果：

\[
t_{\text{final}}=19,
\]

那么：

\[
N_{\text{micro}}
=
19\times200
=
3800.
\]

这 3800 个 micro-step 在 forward 开始之前就可以确定。

所以：

\[
\boxed{
\text{当前 NS attack solver 的时间推进长度是固定的。}
}
\]

---

# 11. NS 每一个 time step 内部有没有“迭代到收敛”

没有。

当前 NS 使用：

```python
ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
    ...,
    fixed_step,
    diffusivity=nu,
    order=4,
)
```

也就是 ETDRK4。

每个 micro-step 内部执行固定数量的 Runge–Kutta stage。

不存在：

\[
\text{Newton iteration}
\]

或者：

\[
\text{CG iteration}
\]

直到 residual 小于 \(\epsilon\) 的过程。

因此：

\[
\boxed{
\text{NS 每个 time step 内部的计算结构是固定的。}
}
\]

---

# 12. NS 内部的 Poisson solve 为什么也没有动态 iteration

Navier–Stokes 这里采用的是 vorticity-streamfunction formulation。

需要从 vorticity \(\omega\) 得到 stream function \(\psi\)：

\[
\Delta\psi=\omega.
\]

看起来这是一个 elliptic PDE。

但这套 Exponax solver 在 Fourier space 中直接构造 inverse Laplacian：

```python
laplacian = build_laplace_operator(
    derivative_operator,
    order=2,
)

self.inv_laplacian = jnp.where(
    laplacian == 0,
    1.0,
    1 / laplacian,
)
```

然后：

```python
stream_function_hat = (
    self.inv_laplacian * vorticity_hat
)
```

也就是对每个 Fourier mode：

\[
\hat\psi(k)
=
\frac{\hat\omega(k)}{\widehat{\Delta}(k)}.
\]

所以这个 Poisson solve 是频域中的直接代数操作。

没有：

```text
iteration 1
iteration 2
iteration 3
...
check residual
```

因此 NS 中虽然存在“解 Poisson equation”这个数学操作，但在当前 periodic Fourier spectral 实现中，并没有 iterative linear solver。

---

# 13. Darcy solver 的实际结构

文件：

```text
2D_Darcy_FNO2d/solvers/darcy_jax_solver.py
```

PDE 是：

\[
-\nabla\cdot(a(x)\nabla u(x))=1,
\qquad
u|_{\partial\Omega}=0.
\]

离散后得到：

\[
A(a)u=b.
\]

代码使用：

```python
u_int, _ = cg(
    matvec,
    rhs,
    tol=tol,
    atol=atol,
    maxiter=maxiter,
)
```

数据生成参数默认：

```python
parser.add_argument(
    "--solver-tol",
    type=float,
    default=1e-5,
)

parser.add_argument(
    "--solver-atol",
    type=float,
    default=0.0,
)

parser.add_argument(
    "--solver-maxiter",
    type=int,
    default=None,
)
```

因此 Darcy 是典型的：

\[
u^{(0)}
\rightarrow
u^{(1)}
\rightarrow
u^{(2)}
\rightarrow
\cdots
\]

然后不断检查 residual。

大致形式是：

\[
r_k=b-Au_k.
\]

当 residual 达到 tolerance 后停止。

所以：

\[
\boxed{
K_{\text{Darcy}}
\text{ 是运行时由收敛情况决定的。}
}
\]

不同 coefficient field \(a(x)\) 可以导致不同的 \(K\)。

---

# 14. Darcy backward 没有直接沿 forward CG iterations 反传

这一点非常重要。

Darcy solver 的实现说明里明确写了：

```text
JAX differentiates this CG solve by implicit differentiation
through a second linear solve.
```

也就是说 forward 可能：

\[
K_{\text{forward}}=287.
\]

但 backward 不会把这 287 个 CG iteration 一个一个倒着走回来。

最终 forward 解满足：

\[
A(a)u=b.
\]

反向传播时可以形成伴随线性系统，例如：

\[
A^T\lambda
=
\frac{\partial L}{\partial u}.
\]

然后再用一个 linear solve 求 \(\lambda\)。

因此 backward 自己又可能有：

\[
K_{\text{backward}}=315.
\]

这两个 iteration 数甚至可以不同：

\[
K_{\text{forward}}
\neq
K_{\text{backward}}.
\]

所以当前 Darcy 实现属于：

\[
\boxed{
\text{forward iterative CG + backward implicit differentiation}
}
\]

而不是：

\[
\boxed{
\text{forward CG unroll + reverse all CG iterations}
}
\]

---

# 15. ETD 和 ETDRK4 到底是什么意思

这次还确认了一个缩写问题。

\[
\boxed{
\text{ETD = Exponential Time Differencing}
}
\]

其中：

\[
E=\text{Exponential}
\]

不是 Explicit。

ETDRK4：

\[
\boxed{
\text{Exponential Time Differencing Runge–Kutta, fourth order}
}
\]

它通常处理半线性 PDE：

\[
u_t=Lu+N(u).
\]

其中：

- \(L\)：线性部分；
- \(N(u)\)：非线性部分。

例如 Burgers：

\[
u_t
=
\nu u_{xx}
-
u u_x.
\]

可以写成：

\[
Lu=\nu u_{xx},
\]

\[
N(u)=-u u_x.
\]

---

# 16. 为什么 ETDRK4 可以认为是 explicit-style method

ETDRK4 的非线性 stage 是由已经知道的 state 直接计算出来的。

它没有要求求解：

\[
F(u_{n+1})=0.
\]

也没有：

\[
u_{n+1}^{(0)}
\rightarrow
u_{n+1}^{(1)}
\rightarrow
\cdots
\]

这样的 Newton convergence loop。

因此从“是否需要在每个 time step 内解一个隐式 nonlinear system”这个角度，它属于显式 exponential Runge–Kutta 方法。

但它和普通 explicit Euler / classical RK4 又不同，因为它对线性部分使用 exponential operator：

\[
e^{L\Delta t}.
\]

例如 Fourier space 中的 diffusion：

\[
L(k)=-\nu k^2.
\]

那么：

\[
e^{L(k)\Delta t}
=
e^{-\nu k^2\Delta t}.
\]

可以逐 Fourier mode 直接乘。

所以：

\[
\boxed{
\text{Explicit 描述 nonlinear stages 不需要 implicit solve。}
}
\]

而：

\[
\boxed{
\text{Exponential 描述 linear operator 的处理方式。}
}
\]

---

# 17. 当前 Exponax ETDRK4 一个 micro-step 的内部步骤

在 modified Exponax 中，ETDRK4 的结构是固定的。

先记：

\[
\hat u_n
\]

为当前 Fourier-space state。

## Stage 0：计算当前 nonlinear term

\[
N_0=N(\hat u_n).
\]

代码概念上是：

```python
u_nonlin_hat = self._nonlinear_fun(u_hat)
```

---

## Stage 1：第一个中间状态

\[
u_1
=
E_{1/2}u_n
+
c_1N_0,
\]

其中：

\[
E_{1/2}=e^{L\Delta t/2}.
\]

然后：

\[
N_1=N(u_1).
\]

---

## Stage 2：第二个中间状态

\[
u_2
=
E_{1/2}u_n
+
c_2N_1.
\]

然后：

\[
N_2=N(u_2).
\]

---

## Stage 3：第三个中间状态

\[
u_3
=
E_{1/2}u_1
+
c_3
\left(
2N_2-N_0
\right).
\]

然后：

\[
N_3=N(u_3).
\]

---

## Final combination

最后：

\[
u_{n+1}
=
Eu_n
+
c_4N_0
+
2c_5(N_1+N_2)
+
c_6N_3,
\]

其中：

\[
E=e^{L\Delta t}.
\]

所以一个 micro-step 可以概括成：

\[
\boxed{
u_n
\rightarrow
N_0
\rightarrow
u_1
\rightarrow
N_1
\rightarrow
u_2
\rightarrow
N_2
\rightarrow
u_3
\rightarrow
N_3
\rightarrow
u_{n+1}
}
\]

每一个 micro-step 都执行同样的结构。

这里没有任何：

\[
\text{“检查 residual，如果还没收敛就再来一次”}
\]

的环节。

---

# 18. Burgers 的 ETDRK4 结构

Burgers 方程：

\[
u_t+u u_x=\nu u_{xx}.
\]

写成：

\[
u_t=Lu+N(u),
\]

其中：

\[
L u=\nu u_{xx},
\]

\[
N(u)=-u u_x.
\]

在 Fourier space 中：

\[
\widehat{Lu}(k)
=
-\nu k^2\hat u(k).
\]

因此 linear exponential factor 是：

\[
e^{-\nu k^2\Delta t}.
\]

非线性部分通过 Fourier pseudo-spectral 方法计算。

每个 time step 都固定执行 ETDRK4 的几个 stage。

所以：

\[
\boxed{
\text{Burgers 当前 solver 没有 convergence iteration。}
}
\]

---

# 19. Navier–Stokes 的 ETDRK4 结构

当前 NS 使用 vorticity formulation。

形式上：

\[
\omega_t
=
L\omega
+
N(\omega).
\]

其中主要线性部分是：

\[
L\omega
=
\nu\Delta\omega
\]

以及代码支持的 drag。

非线性 convection：

\[
N(\omega)
=
-u\cdot\nabla\omega
+
f.
\]

每次 nonlinear evaluation 中，会从：

\[
\omega
\]

先求 stream function：

\[
\Delta\psi=\omega,
\]

再得到 velocity：

\[
u=\partial_y\psi,
\qquad
v=-\partial_x\psi,
\]

再计算：

\[
u\omega_x+v\omega_y.
\]

在当前 Fourier implementation 中，上述 Poisson inversion 是直接频域求逆，不需要 iterative convergence。

因此整个 NS micro-step 的内部操作数也是固定的。

---

# 20. 三种 solver 最终分类

| Solver | PDE 类型 | 时间推进 / 求解方式 | 内部是否迭代到 residual 收敛 | 计算长度 |
|---|---|---|---|---|
| Burgers Exponax | time evolution | ETDRK4 | 否 | 固定 \(T/\Delta t\) |
| NS Exponax | time evolution | ETDRK4 | 否 | 固定 \(T/\Delta t\) |
| NS 内部 Poisson | elliptic auxiliary solve | Fourier inverse Laplacian | 否 | 直接频域操作 |
| Darcy | steady-state elliptic | Conjugate Gradient | 是 | 输入相关，运行时决定 |

---

# 21. 最终最重要的几个结论

## 结论 1：动态 iteration 本身不是不能反向传播

可以先动态 forward：

\[
K
=
\min\{k:\|r_k\|<\varepsilon\}.
\]

forward 完成后，这一次实际计算路径已经确定。

如果框架支持动态图，就可以对这一次实际路径做 backward。

---

## 结论 2：普通 unrolled reverse-mode 要求 backward 开始时路径已确定

更精确地说：

\[
\boxed{
\text{在 backward 开始时，这一次 forward 到底执行了哪些节点必须已经确定。}
}
\]

不要求每一次 forward 都一样。

---

## 结论 3：implicit differentiation 可以绕过 forward iteration history

如果最终解满足：

\[
F(u^\star,\theta)=0,
\]

可以直接对最终方程求导。

这样 forward 跑了多少 solver iteration，不需要成为 backward 的逐步计算链。

---

## 结论 4：当前 Burgers / NS 不属于“每个 time step 内迭代到收敛”

当前实现是：

\[
\boxed{
\text{fixed-step ETDRK4}
}
\]

每一个 micro-step 内部 stage 数固定。

---

## 结论 5：当前 Darcy 才是典型的 residual-based iterative solver

Darcy：

\[
A(a)u=b
\]

通过 CG 迭代。

停止条件由 residual tolerance 控制。

所以：

\[
\boxed{
\text{Darcy 的 forward iteration 数可以依赖输入。}
}
\]

---

## 结论 6：当前 Darcy 的梯度又进一步使用 implicit differentiation

因此它甚至没有要求 backward 沿 forward CG 的所有 iteration 倒着走。

forward CG iteration 数：

\[
K_{\text{forward}}
\]

和 backward 伴随 linear solve 的 iteration 数：

\[
K_{\text{backward}}
\]

可以是不同的。

---

# 22. 对 solver progress / fidelity 问题的启示

这次讨论对后续“continuous solver progress / fidelity”问题很重要，因为不同 solver 的“进度”含义并不一样。

## 对 Darcy

比较自然的 progress 可以是：

- CG iteration 数；
- residual 大小；
- residual reduction ratio；
- 已消耗计算量；
- 达到某个 tolerance 的程度。

例如：

\[
p_k
=
-\log_{10}
\left(
\frac{\|r_k\|}{\|r_0\|}
\right).
\]

这种 progress 与“收敛程度”直接相关。

---

## 对 Burgers / NS

当前 solver 没有内部 convergence iteration。

所以更自然的 progress 是：

- 已推进到的 physical time；
- 已执行的 micro-step 数；
- 已消耗的 fixed-step computational budget；
- 或者使用不同 \(\Delta t\)、resolution、spectral cutoff 形成 fidelity。

例如：

\[
p
=
\frac{t}{T}
\]

或者：

\[
p
=
\frac{k}{K_{\max}}.
\]

这里的“solver progress”与 Darcy 的 residual-based progress 是不同概念。

---

# 23. 一句话总结

这次讨论最终可以压缩成下面三句话：

\[
\boxed{
\text{Burgers / NS：固定步长 ETDRK4，没有每步内部的收敛 iteration。}
}
\]

\[
\boxed{
\text{Darcy：CG 迭代到 residual tolerance，实际 iteration 数运行时决定。}
}
\]

\[
\boxed{
\text{直接 unrolled backprop 要沿已经确定的 forward 路径反传；implicit differentiation 可以绕开 forward iteration history。}
}
\]

---

## 安全说明

本次聊天中曾出现 GitHub Personal Access Token。该敏感凭据未写入本文件。对于已经暴露在聊天文本中的 token，应立即在 GitHub 中 revoke / rotate。
