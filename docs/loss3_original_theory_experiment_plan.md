# Loss3 Original 作为回归对抗主目标的理论与实验计划

## 0. 中心目标

本文档围绕一个核心问题组织：

> 为什么在神经算子 / 回归鲁棒性问题里，`loss3_original`
> \[
> \max_{\|\delta\|_p\le \varepsilon}
> \|f(x+\delta)-j(x+\delta)\|_q
> \]
> 才是最合适的 finite-radius regression attack objective？

这里：

- \(f\) 是模型。
- \(j\) 是真实 solver / oracle。
- \(e(x)=f(x)-j(x)\) 是误差场。
- \(b=e(x)\) 是 clean input 上已有的 residual。
- \(\Delta f=f(x+\delta)-f(x)\)。
- \(\Delta j=j(x+\delta)-j(x)\)。
- \(\Delta e=e(x+\delta)-e(x)=\Delta f-\Delta j\)。

主结论：

> 回归稳健性不是要求 \(f\) 不变，而是要求 \(f\) 跟着 \(j\) 一起正确地变。  
> 因此真正危险的方向不是 high model sensitivity direction，而是 high mismatch direction between \(f\) and \(j\)。

也就是说：

\[
\|f(x+\delta)-f(x)\|_q \text{ 大}
\]

并不推出：

\[
\|f(x+\delta)-j(x+\delta)\|_q \text{ 大}.
\]

因为真实 solver \(j\) 也会随输入扰动变化。

## 1. 整体证明路线

整条逻辑分成五步。

### Step 1: Oracle Consistency

先证明真实回归攻击必须比较 \(f(x+\delta)\) 和 \(j(x+\delta)\)。

真实目标：

\[
L_3^{orig}(\delta)
=
\|f(x+\delta)-j(x+\delta)\|_q
=
\|e(x+\delta)\|_q.
\]

反例：

如果模型完美，

\[
f=j,
\]

那么任意扰动下真实回归误差都应该为：

\[
f(x+\delta)-j(x+\delta)=0.
\]

所以合理的回归攻击 loss 应该为 0。

但：

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_q
\]

和：

\[
L_2(\delta)=\|f(x+\delta)-j(x)\|_q
\]

仍然可能很大，因为真实 PDE / solver 本身可能对输入敏感。

结论：

> `loss1` 和 `loss2` 可以产生 false alert：模型完全正确时，它们仍然会报出大的攻击效果。

### Step 2: Surrogate Loss 的偏差来源

三个基础 loss：

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_q=\|\Delta f\|_q.
\]

\[
L_2(\delta)=\|f(x+\delta)-j(x)\|_q=\|b+\Delta f\|_q.
\]

\[
L_3(\delta)=\|f(x+\delta)-j(x+\delta)\|_q
=\|b+\Delta f-\Delta j\|_q.
\]

三者的含义：

- `loss1`: 模型输出自己变了多少。
- `loss2`: 模型扰动后输出离 clean solver target 多远。
- `loss3`: 扰动输入下模型和真实 solver 的真实回归误差。

`loss1` 丢掉了 \(j\) 的响应。  
`loss2` 冻结了 oracle，丢掉了 \(j(x+\delta)-j(x)\)。  
`loss3` 才保留了 perturbed-input oracle。

结论：

> 如果 \(j(x+\delta)-j(x)\) 不可忽略，那么 `loss1` / `loss2` 不是 `loss3` 的可靠 surrogate。

### Step 3: 局部线性化解释差异

在 \(x\) 附近做局部展开：

\[
\Delta f \approx J_f(x)\delta.
\]

\[
\Delta j \approx J_j(x)\delta.
\]

\[
\Delta e
=e(x+\delta)-e(x)
\approx
(J_f(x)-J_j(x))\delta.
\]

令：

\[
A=J_f(x)-J_j(x).
\]

则：

\[
e(x+\delta)\approx b+A\delta.
\]

三个局部敏感性：

\[
L_f=\|J_f\|_{p\to q}.
\]

\[
L_j=\|J_j\|_{p\to q}.
\]

\[
L_e=\|J_f-J_j\|_{p\to q}.
\]

解释：

- \(L_f\): 模型自身对输入扰动有多敏感。
- \(L_j\): 真实 solver 自身对输入扰动有多敏感。
- \(L_e\): 模型和 solver 的局部响应差别有多大。

关键情况：

\[
L_f \text{ 大},\quad L_j \text{ 大},\quad L_e \text{ 小}.
\]

这说明 \(f\) 和 \(j\) 都变化很快，但变化方向和幅度相近，因此误差场不敏感。

结论：

> \(f\) 平滑不等于回归稳健；真正重要的是 \(f-j\) 的误差场是否稳定。

### Step 4: 区分三种不同的“变化”

在局部仿射模型：

\[
e(x+\delta)\approx b+A\delta
\]

下，有三种不同对象。

#### 4.1 Error Field Movement

\[
\frac{\|e(x+\delta)-e(x)\|_q}{\|\delta\|_p}
\approx
\frac{\|A\delta\|_q}{\|\delta\|_p}.
\]

它研究：

> 误差向量本身移动得多快？

局部极限对应：

\[
\|A\|_{p\to q}=\|J_f-J_j\|_{p\to q}.
\]

这是 residual increment ratio 的意义。

#### 4.2 Error Norm Growth

\[
\frac{\|e(x+\delta)\|_q-\|e(x)\|_q}{\|\delta\|_p}
=
\frac{\|b+A\delta\|_q-\|b\|_q}{\|\delta\|_p}.
\]

它研究：

> 当前误差范数是否被单位扰动往外推大？

当 \(q=2\) 且 \(b\neq 0\)，对二范数做一阶 Taylor 展开：

\[
\nabla_z\|z\|_2\big|_{z=b}=\frac{b}{\|b\|_2}.
\]

因此：

\[
\|b+A\delta\|_2-\|b\|_2
\approx
\left\langle
\frac{b}{\|b\|_2},
A\delta
\right\rangle.
\]

如果 \(\delta=\|\delta\|_p v\)，则：

\[
\frac{\|b+A\delta\|_2-\|b\|_2}{\|\delta\|_p}
\approx
\left\langle
\frac{b}{\|b\|_2},
Av
\right\rangle.
\]

解释：

> norm increment ratio 不看 \(A\delta\) 有多长，而看它在当前误差方向 \(b\) 上的 outward component 有多大。

所以它不是 Lipschitz norm，而是 local outward risk growth。

#### 4.3 Final Error

\[
\|e(x+\delta)\|_q=\|b+A\delta\|_q.
\]

它研究：

> 扰动后最终误差有多大？

真实 finite-radius attack 就是：

\[
\max_{\|\delta\|_p\le \varepsilon}
\|e(x+\delta)\|_q.
\]

这是 `loss3_original` 的意义。

#### 4.4 关键区分

Error field can move without increasing risk.

例如：

\[
b=(1,0),\quad A\delta=(-2,0).
\]

则：

\[
\|A\delta\|=2
\]

很大，但：

\[
\|b+A\delta\|=\|(-1,0)\|=1=\|b\|.
\]

误差向量移动很大，但误差范数没有增长。

结论：

> local Lipschitz-large direction 不一定是 adversarial-risk-increasing direction。

### Step 5: 局部和全局用路径积分连接

令路径：

\[
\gamma(t)=x+t\delta,\quad t\in[0,1].
\]

由微积分基本定理：

\[
e(x+\delta)-e(x)
=
\int_0^1
J_e(x+t\delta)\delta\,dt.
\]

所以：

\[
e(x+\delta)
=
b+
\int_0^1
J_e(x+t\delta)\delta\,dt.
\]

进一步：

\[
\|e(x+\delta)-e(x)\|_q
\le
\int_0^1
\|J_e(x+t\delta)\delta\|_q\,dt
\]

\[
\le
\|\delta\|_p
\int_0^1
\|J_e(x+t\delta)\|_{p\to q}\,dt.
\]

解释：

- 局部 Lipschitz 看的是瞬时放大率 \(J_e(x)\)。
- finite-radius attack 看的是整条路径上 \(J_e(x+t\delta)\) 的累计效果。

结论：

> local Lipschitz controls possible instantaneous growth, but does not equal finite-radius attack loss.

一个方向可以局部斜率很大，但沿路径变弯、饱和、旋转，终点误差不大。  
另一个方向初始斜率未必最大，但沿路径累计后终点最大。

## 2. 九种 Objective 的语言含义

每个基础 loss 有三种 objective variant。

### 2.1 Original Objective

\[
L(\delta).
\]

问：

> 终点值多大？

### 2.2 Increment Ratio

\[
\frac{L(\delta)-L(0)}{\|\delta\|_p+\eta}.
\]

问：

> 单位扰动带来的增长效率多大？

它适合作为 local diagnostic，不适合作为大半径 finite-radius 主攻击 loss。

### 2.3 Regularized Objective

\[
L(\delta)-C\|\delta\|_p.
\]

问：

> 增长收益减去扰动代价后是否划算？

它可能选择内部半径，不一定使用完整 \(\varepsilon\) budget。

### 2.4 3 by 3 总表

| Objective | 数学形式 | 含义 | 适合用途 |
|---|---|---|---|
| loss1_original | \(\|\Delta f\|\) | 模型输出自己变了多少 | surrogate baseline / false alert 对照 |
| loss1_increment_ratio | \(\|\Delta f\|/\|\delta\|\) | 模型自身局部敏感性 | 估计 \(J_f\) 的局部响应 |
| loss1_regularized | \(\|\Delta f\|-C\|\delta\|\) | 模型变化收益减扰动代价 | cost-aware surrogate 对照 |
| loss2_original | \(\|b+\Delta f\|\) | fixed clean target error | fixed-target surrogate 对照 |
| loss2_increment_ratio | \((\|b+\Delta f\|-\|b\|)/\|\delta\|\) | fixed-target error 的局部 outward growth | 局部诊断 |
| loss2_regularized | \(\|b+\Delta f\|-C\|\delta\|\) | fixed-target 增长收益减扰动代价 | cost-aware surrogate 对照 |
| loss3_original | \(\|b+\Delta f-\Delta j\|\) | perturbed-input regression error | 主 finite-radius attack objective |
| loss3_increment_ratio | \((\|b+\Delta f-\Delta j\|-\|b\|)/\|\delta\|\) | 真实误差范数的局部 outward growth | local diagnostic |
| loss3_regularized | \(\|b+\Delta f-\Delta j\|-C\|\delta\|\) | 真实误差收益减扰动代价 | cost-aware 对照 |

另外建议新增：

| Objective | 数学形式 | 含义 | 适合用途 |
|---|---|---|---|
| loss3_residual_increment_ratio | \(\|\Delta f-\Delta j\|/\|\delta\|\) | error field movement | \(J_f-J_j\) 局部 Lipschitz / mismatch |

## 3. Method-Objective Matching

不同优化方法适合不同数学结构。

### 3.1 Generalized Power Iteration

适合局部齐次 operator norm：

\[
\max_{\|v\|_p=1}\|Av\|_q.
\]

其中 \(A=J_f-J_j\) 或 \(J_f\)。

它适合：

- residual increment ratio；
- small-\(\varepsilon\) 局部 Lipschitz 分析；
- local mechanism。

不适合作为大半径 finite-radius attack 的唯一方法，因为全局目标非线性、非齐次、路径相关。

### 3.2 PGD / LP-Steepest PGD

适合 finite-radius 非线性攻击：

\[
\max_{\|\delta\|_p\le\varepsilon}\|e(x+\delta)\|_q.
\]

LP-steepest PGD 每一步选择：

\[
s_k=
\arg\max_{\|s\|_p\le1}
\langle \nabla_\delta L(\delta_k),s\rangle.
\]

它比普通 Euclidean gradient 更匹配 \(p\)-ball 几何，尤其当 \(p\neq 2\)。

结论：

> GPI / ratio 适合局部诊断；PGD / LP-steepest on loss3_original 适合 finite-radius attack。

## 4. 需要验证的核心 Claims

### Claim 1: loss3_original 是 oracle-consistent 的主攻击目标

只有 `loss3_original` 直接度量：

\[
f(x+\delta)-j(x+\delta).
\]

`loss1` / `loss2` 是 surrogate。

### Claim 2: loss1 / loss2 会产生 false alert

它们可能找到 high model sensitivity direction，但这不等于 high regression error。

### Claim 3: 大多数方向上 \(f\) 和 \(j\) 可能同向变化

因此 \(f\) 变化大不一定是错误，可能是真实 solver 也应该变化。

### Claim 4: 真正危险的是 high mismatch direction

危险方向满足：

\[
\Delta f-\Delta j \text{ 大}
\]

或者：

\[
\cos(\Delta f,\Delta j) \text{ 低或为负}.
\]

### Claim 5: ratio / regularized 是局部或 cost-aware 诊断，不是主全局攻击目标

它们可以解释局部机制，但不能替代：

\[
\max_{\|\delta\|\le\varepsilon}\|e(x+\delta)\|.
\]

### Claim 6: 局部最优方向不等于 finite-radius 全局终点最优方向

由路径积分和非线性路径效应解释。

## 5. 实验计划

实验目标不是堆满 27 个组合，而是每组实验服务一个 claim。

### Experiment 1: Main Objective Comparison

#### 目的

验证：

> `loss1_original` / `loss2_original` 不是 `loss3_original` 的可靠替代。

#### 比较对象

- `loss1_original`
- `loss2_original`
- `loss3_original`

#### 方法

对同一 batch、同一 \(\varepsilon\)、同一 attack method，例如 PGD / LP-steepest，分别优化三种 original objective。

#### 记录指标

对每个 final delta 记录：

\[
\|\Delta f\|_q
\]

\[
\|\Delta j\|_q
\]

\[
\|\Delta f-\Delta j\|_q
\]

\[
\|e(x+\delta)\|_q
\]

\[
\|e(x+\delta)\|_q-\|e(x)\|_q
\]

\[
\cos(\Delta f,\Delta j)
\]

tracking discount：

\[
D_f(\delta)=
\frac{\|\Delta f-\Delta j\|_q}{\|\Delta f\|_q+\eta}.
\]

对称版本：

\[
D_{sym}(\delta)=
\frac{\|\Delta f-\Delta j\|_q}
{\|\Delta f\|_q+\|\Delta j\|_q+\eta}.
\]

#### 预期现象

`loss1_original` 方向：

- \(\|\Delta f\|\) 大；
- 但 \(\|e(x+\delta)\|\) 不一定最大；
- \(\cos(\Delta f,\Delta j)\) 可能高；
- \(D_f\) 可能小。

`loss3_original` 方向：

- \(\|\Delta f\|\) 不一定最大；
- 但 \(\|e(x+\delta)\|\) 最大；
- \(\|\Delta f-\Delta j\|\) 更大；
- \(\cos(\Delta f,\Delta j)\) 更低。

#### 支撑 Claim

- Claim 1
- Claim 2
- Claim 4

### Experiment 2: Local Response Decomposition Table

#### 目的

验证局部上：

> \(L_f,L_j,L_e\) 的大小和方向都不同；高 \(L_f\) 不等于高 \(L_e\)。

#### 方向

估计以下方向：

\[
v_f^*=\arg\max_{\|v\|_p=1}\|J_fv\|_q.
\]

\[
v_j^*=\arg\max_{\|v\|_p=1}\|J_jv\|_q.
\]

\[
v_e^*=\arg\max_{\|v\|_p=1}\|(J_f-J_j)v\|_q.
\]

局部 outward growth 方向：

\[
v_{growth}^*
=
\arg\max_{\|v\|_p=1}
\left\langle
\frac{e(x)}{\|e(x)\|_2},
(J_f-J_j)v
\right\rangle
\]

仅在 \(q=2\) 且 \(e(x)\neq0\) 时使用。

再加 random directions。

#### 记录表

| direction | \(\|\Delta f\|/\epsilon\) | \(\|\Delta j\|/\epsilon\) | \(\|\Delta f-\Delta j\|/\epsilon\) | \(\cos(\Delta f,\Delta j)\) | \(D_f\) |
|---|---:|---:|---:|---:|---:|
| \(v_f^*\) | | | | | |
| \(v_j^*\) | | | | | |
| \(v_e^*\) | | | | | |
| \(v_{growth}^*\) | | | | | |
| random mean | | | | | |

#### 预期现象

如果 \(v_f^*\) 上：

- \(\|\Delta f\|/\epsilon\) 大；
- \(\|\Delta j\|/\epsilon\) 也大；
- \(\cos(\Delta f,\Delta j)\) 高；
- \(\|\Delta f-\Delta j\|/\epsilon\) 相对小；

则说明：

> loss1 的最坏方向可能是 false alert：模型变大，但 solver 也跟着变。

如果 \(v_e^*\) 或 \(v_{growth}^*\) 上：

- \(\|\Delta f-\Delta j\|/\epsilon\) 大；
- \(\cos(\Delta f,\Delta j)\) 低或负；

则说明：

> 这些方向才是真正的 high mismatch direction。

#### 支撑 Claim

- Claim 2
- Claim 3
- Claim 4

### Experiment 3: Small-Epsilon Sweep

#### 目的

验证 increment / residual ratio 是否确实刻画局部结构。

#### 半径

\[
\epsilon\in\{10^{-4},10^{-3},10^{-2},10^{-1}\}.
\]

#### 估计量

\[
L_f(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|f(x+\epsilon v)-f(x)\|_q}{\epsilon}.
\]

\[
L_j(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|j(x+\epsilon v)-j(x)\|_q}{\epsilon}.
\]

\[
L_e(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|e(x+\epsilon v)-e(x)\|_q}{\epsilon}.
\]

以及 norm growth：

\[
G_e(\epsilon)\approx
\max_{\|v\|_p=1}
\frac{\|e(x+\epsilon v)\|_q-\|e(x)\|_q}{\epsilon}.
\]

#### 记录

- 数值是否随 \(\epsilon\) 稳定。
- 最坏方向 \(v^*(\epsilon)\) 是否稳定。
- 不同 \(\epsilon\) 下方向 cosine：

\[
\cos(v^*(\epsilon_i),v^*(\epsilon_j)).
\]

#### 预期现象

如果小 \(\epsilon\) 下数值和方向稳定，则 ratio objective 确实刻画局部结构。

如果 \(\epsilon\) 变大后数值和方向漂移，则说明非线性路径效应出现。

#### 支撑 Claim

- Claim 5
- Claim 6

### Experiment 4: Ray Profile / Local-to-Global Profile

#### 目的

验证：

> 局部好方向不等于 finite-radius endpoint 好方向。

#### 方向来源

- `loss3_original` final direction
- `loss3_increment_ratio` final direction
- `loss3_residual_increment_ratio` final direction
- `loss3_regularized` final direction
- random direction

将每个 final delta 归一化：

\[
v=\frac{\delta}{\|\delta\|_p}.
\]

#### 曲线

对：

\[
r\in[0,\varepsilon]
\]

画：

\[
r\mapsto \|e(x+rv)\|_q.
\]

\[
r\mapsto
\frac{\|e(x+rv)\|_q-\|e(x)\|_q}{r+\eta}.
\]

\[
r\mapsto
\frac{\|e(x+rv)-e(x)\|_q}{r+\eta}.
\]

#### 预期现象

increment ratio 方向：

- 小 \(r\) 下增长率高；
- 大 \(r\) 下可能饱和或变弯；
- \(r=\varepsilon\) 终点不一定最大。

original direction：

- 小 \(r\) 初始斜率未必最大；
- 但 \(r=\varepsilon\) 的 \(\|e(x+rv)\|\) 最大。

#### 支撑 Claim

- Claim 5
- Claim 6

### Experiment 5: Boundary-Rescaled Comparison

#### 目的

验证：

> ratio / regularized 找到的方向，即使缩放到同样 \(\varepsilon\)，也不一定最大化 `loss3_original`。

#### 方法

对每个 final delta：

\[
v=\frac{\delta}{\|\delta\|_p}.
\]

构造：

\[
\delta_{bdry}=\varepsilon v.
\]

比较：

- final delta 上的 `loss3_original`
- boundary-rescaled delta 上的 `loss3_original`
- final delta 上的 `loss3_increment_ratio`
- boundary-rescaled delta 上的 `loss3_increment_ratio`

#### 预期现象

- regularized 的 final delta 可能很小。
- 放大到 boundary 后 loss 可能增加，但仍未必达到 original attack 方向。
- increment ratio 方向可能单位增长率高，但 endpoint loss 不最大。

#### 支撑 Claim

- Claim 5
- Claim 6

### Experiment 6: Direction Rotation Along Path

#### 目的

验证 finite-radius attack 的路径相关性。

#### 方法

取一个 original attack direction \(\delta^*\)，定义：

\[
x_t=x+t\delta^*,\quad t\in[0,1].
\]

在每个 \(x_t\) 重新估计：

\[
v_e^*(x_t)
=
\arg\max_{\|v\|_p=1}
\|J_e(x_t)v\|_q.
\]

记录：

\[
\cos(v_e^*(x_t),v_e^*(x_0)).
\]

也可以记录相邻方向：

\[
\cos(v_e^*(x_t),v_e^*(x_{t+\Delta t})).
\]

#### 预期现象

如果 cosine 明显下降或震荡，说明：

> 局部最坏方向沿路径旋转，单点局部 ratio 不能代表整个 finite-radius attack。

#### 支撑 Claim

- Claim 6

## 6. 已有结果可如何解释

已有代表性结果：

```text
loss1_increment_ratio_generalized_power:
    L1_inc = 0.913
    L3_inc = 0.157
    L3_orig = 1.556

loss3_increment_ratio_generalized_power:
    L1_inc = 0.826
    L3_inc = 0.401
    L3_orig = 3.507
```

解释：

- `loss1` 方向让模型 \(f\) 自己变化快。
- 但真实误差增长率低，说明 \(j\) 也可能同向变化。
- `loss3` 方向让模型自变化稍小，但让 \(f\) 和 \(j\) 的 mismatch 更大。

结论：

> 高模型敏感性方向不等于高回归误差增长方向。

另一个代表性结果：

```text
loss1_original_pgd:
    L1_orig ≈ 11.17
    L3_orig ≈ 4.34

loss3_original_generalized_power:
    L1_orig ≈ 8.53
    L3_orig ≈ 6.86
```

解释：

- `loss1` attack 让 \(f\) 动得更大。
- 但 `loss3` attack 让真实 oracle-relative error 更大。

结论：

> 模型输出变化大，不代表相对于真实 solver 的误差大。

## 7. 最终论文叙事建议

### 主线

```text
We aim to justify loss3_original as the primary adversarial objective for operator regression.
```

中文：

> 我们的目标不是提出一堆等价 loss，而是论证 `loss3_original` 是回归 / 神经算子有限半径对抗攻击中最直接、最正确的主目标。

### 角色分配

- `loss3_original`: 主 finite-radius regression attack objective。
- `loss1_original`, `loss2_original`: surrogate baseline，用来说明简化目标的偏差。
- `loss3_increment_ratio`: local outward risk growth diagnostic。
- `loss3_residual_increment_ratio`: local error-field Lipschitz / mismatch diagnostic。
- `loss3_regularized`: cost-aware / interior-solution 对照。
- GPI: local homogeneous operator norm 的工具。
- PGD / LP-steepest PGD: finite-radius nonlinear attack 的工具。

### 最终核心结论

1. Regression robustness is not invariance of \(f\); it is co-variation of \(f\) with \(j\).

2. High model sensitivity does not imply high regression error.

3. The dangerous directions are high-mismatch directions, not merely high-sensitivity directions.

4. Local Lipschitz / ratio objectives explain mechanisms, but finite-radius adversarial risk must be evaluated by `loss3_original`.

5. `loss1` and `loss2` can produce false alerts because they ignore or freeze the oracle response.

6. `loss3_original` directly answers the attack question:

\[
\max_{\|\delta\|_p\le\varepsilon}
\|f(x+\delta)-j(x+\delta)\|_q.
\]

## 8. Recommended Minimal Experiment Set

如果需要精简，不建议继续主打 27 个组合。推荐最小但有解释力的实验集：

1. `loss1_original` vs `loss2_original` vs `loss3_original`
   - 用于证明 surrogate bias / false alert。

2. `loss3_original` vs `loss3_increment_ratio` vs `loss3_residual_increment_ratio` vs `loss3_regularized`
   - 用于解释主攻击、局部 outward growth、局部 Lipschitz、cost-aware objective 的区别。

3. local response decomposition table
   - 用于证明 \(f\) 和 \(j\) 多数方向同向变化，危险方向是 mismatch。

4. small-\(\epsilon\) sweep
   - 用于证明 ratio objective 只在局部半径下有稳定意义。

5. ray profile
   - 用于证明局部最优方向不等于 finite-radius endpoint 最优方向。

6. boundary-rescaled comparison
   - 用于证明同一 budget 下 ratio / regularized 方向仍不一定最大化 `loss3_original`。

7. direction rotation
   - 用于证明有限半径攻击是路径相关的非线性问题。

## 9. 一句话摘要

> `loss3_original` is the primary attack loss because it is the only objective among the considered losses that directly measures perturbed-input oracle-relative regression error.  
> Other objectives are useful as surrogate baselines or local diagnostics, but they should not be interpreted as equivalent finite-radius attack objectives.

中文：

> `loss3_original` 是主攻击 loss，因为它直接度量扰动输入上的真实回归误差；其他 loss 可以作为对照或局部机制分析工具，但不能替代有限半径真实攻击目标。
