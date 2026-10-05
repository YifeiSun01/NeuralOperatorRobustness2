# Multi-Fidelity 文献的最终任务分类：Prediction、Optimization 与 Active Search

更新日期：2026-10-05

本文整理我们围绕 multi-fidelity 文献讨论后形成的统一分类。重点不是按照论文使用了 GP、Bayesian optimization、Hyperband、Benders、active learning 等算法名称分类，而是按照：

> 一篇论文最后究竟想解决什么数学任务？

在这个层次上，原来的 26 篇 multi-fidelity 文献，加上后来补充讨论的 Active Search、continuous fidelity、rare-event、pause/resume 和 nonmyopic lookahead 论文，可以把真正的最终任务压缩成三大类：

1. **Prediction / Function Learning**
2. **Optimization**
3. **Active Search / Discovery**

此外还有 survey、review、benchmark 等元工作，但它们不是第四种科学任务。

---

## 1. 最重要的区分：Task、Method 和 Multi-Fidelity 是三个不同层次

很多术语容易混在一起。

最清楚的结构是：

~~~text
第一层：Task
    最终到底想得到什么？

第二层：Method
    用什么算法完成这个任务？

第三层：Multi-Fidelity Structure
    同一个任务是否存在不同成本、不同精度、不同信息来源的评价？
~~~

因此，下面这些词不应该全部放在同一个分类层级：

- Prediction
- Active Learning
- Optimization
- Bayesian Optimization
- Ranking & Selection
- Hyperband
- Inverse Problem
- Benders
- Active Search
- Multi-Fidelity

其中：

~~~text
Prediction / Function Learning
Optimization
Active Search
~~~

是最主要的三类最终任务。

而：

~~~text
Active Learning
Bayesian Optimization
Ranking & Selection
Hyperband
Evolutionary Optimization
Benders Decomposition
~~~

更多是在描述完成任务的方法、采样机制、资源分配机制或特殊问题结构。

Multi-fidelity 则是另一条独立的轴。

---

# 2. 第一大类：Prediction / Function Learning

## 2.1 最终目标

Prediction 的目标是学习一个映射：

\[
F:x\mapsto F(x)
\]

使得对一个新的输入 \(x\)，不必重新执行昂贵的 high-fidelity evaluation，也可以预测：

\[
\hat F(x)\approx F(x).
\]

最终真正关心的是：

\[
\boxed{\text{把整个函数或大范围输入上的映射学准确}}
\]

而不是只找一个最好的输入。

---

## 2.2 普通 Prediction 和 Active Learning 为什么归在同一类？

普通 surrogate learning 可以写成：

~~~text
给定已有训练数据
→ 拟合 surrogate
→ 用 surrogate 预测新的 x
~~~

Active Learning 则是：

~~~text
已有少量数据
→ 当前 surrogate
→ 主动选择最值得获得真实结果的新 x
→ 获得新数据
→ 更新 surrogate
→ 重复
~~~

两者最后得到的东西仍然是同一种：

\[
\boxed{\text{一个尽可能准确的预测模型}}
\]

因此，在“最终任务”这一层：

\[
\boxed{
\text{Active Learning 属于 Function Learning / Prediction}
}
\]

Active Learning 的特殊点是：

> 数据不是预先固定，而是算法自己决定下一次去哪里获取。

它改变的是“怎样收集训练数据”，没有改变最终任务本身。

---

## 2.3 Multi-Fidelity Prediction

如果同一个 \(x\) 可以通过多个 fidelity 得到信息：

\[
(x,z)\mapsto y(x,z),
\]

而真正目标是预测参考 fidelity：

\[
F(x)=y(x,z^\star),
\]

那么就是 multi-fidelity prediction。

典型形式包括：

\[
f_H(x)=\rho f_L(x)+\delta(x),
\]

或者：

\[
Y(x,t)=F(x)+G(x,t),
\]

也可以直接建立：

\[
g(x,z)
\]

的联合 surrogate。

---

## 2.4 代表论文

### Kennedy & O’Hagan (2000)

**Predicting the Output from a Complex Computer Code When Fast Approximations Are Available**

任务：

\[
x\rightarrow f_H(x)
\]

利用大量 low-fidelity 与少量 high-fidelity 数据预测昂贵程序输出。

这是典型的 multi-fidelity prediction。

### Picheny & Ginsbourger (2013)

**A Nonstationary Space-Time Gaussian Process Model for Partially Converged Simulations**

任务：

\[
\text{partial solver trajectory}
\rightarrow
\text{final converged result}
\]

模型形式：

\[
Y(x,t)=F(x)+G(x,t).
\]

这里的 fidelity 是求解器计算进度，最终目标仍然是预测 \(F(x)\)。

### Multifidelity Modeling for PINNs

任务是利用便宜 PINN 解和少量昂贵 PINN 解预测新的高保真 PDE 解场。

最终任务仍属于 Function Learning。

### IFC

**Infinite-Fidelity Coregionalization for Physical Simulation**

利用连续 fidelity 建模预测不同精度下的高维物理解。

最终目标仍然是建立 surrogate。

### DMFAL

**Deep Multi-Fidelity Active Learning of High-Dimensional Outputs**

它会主动决定下一次获取：

\[
(x,z)
\]

中的哪一个数据。

但最终目标是让整个高保真物理场 surrogate 更准确。

因此：

\[
\boxed{
\text{DMFAL = Multi-Fidelity Prediction + Active Data Acquisition}
}
\]

Active Learning 在这里是数据获取方式，不单独作为第四种最终任务。

---

# 3. 第二大类：Optimization

## 3.1 最终目标

Optimization 的统一数学形式是：

\[
\boxed{
x^\star=\arg\min_{x\in\mathcal X}J(x)
}
\]

或者：

\[
\boxed{
x^\star=\arg\max_{x\in\mathcal X}J(x)
}
\]

最终只要求找到一个或少数几个最优方案。

这里不要求整个 \(J(x)\) 在所有地方都预测准确。

---

# 4. Bayesian Optimization 为什么只是 Optimization 的一种方法？

Bayesian Optimization 的最终目标仍然是：

\[
x^\star=\arg\max_x F(x)
\]

或：

\[
x^\star=\arg\min_x F(x).
\]

它的特点只是：

~~~text
已有数据
→ surrogate
→ uncertainty
→ acquisition
→ 选择下一个 x 或 (x,z)
→ 真实评价
→ 更新 surrogate
~~~

之所以使用 surrogate，是因为真实评价昂贵。

所以：

\[
\boxed{
\text{Bayesian Optimization 是 Optimization 的一种求解方法}
}
\]

而不是与 Optimization 并列的最终任务。

---

## 4.1 Multi-Fidelity Bayesian Optimization

当一次评价还可以选择 fidelity：

\[
(x,z),
\]

算法不仅决定在哪里评价 \(x\)，还要决定用多少成本、什么精度 \(z\)。

但最终目标仍然是：

\[
\boxed{
x^\star=\arg\max_x F(x)
}
\]

或相应的最小化问题。

代表论文：

- MF-GP-UCB
- BOCA
- FABOLAS
- MISO / misoKG
- MF-MES
- DNN-MFBO
- BMBO-DARN
- Multi-fidelity NAS with Knowledge Distillation

---

# 5. Optimization 中其他看起来不同的东西，其实仍然属于同一任务

## 5.1 Evolutionary Optimization

例如 Branke 等人的 partially converged simulation 工作。

最终仍然在找：

\[
x^\star=\arg\min_x J(x).
\]

区别只是使用 population、selection、mutation/crossover、partial evaluation、continue/stop，而不是 Bayesian surrogate acquisition。

---

## 5.2 Hyperband

Hyperband 的任务也是找最好的配置：

\[
x^\star=\arg\min_x L(x).
\]

它采用：

~~~text
大量候选少量资源
→ 淘汰差的
→ 给幸存者更多资源
→ 再淘汰
~~~

最终目标仍然是 Optimization。

---

# 6. Ranking & Selection 本质上也是 Optimization

假设已经有有限候选：

\[
x_1,\ldots,x_N.
\]

每个候选具有一个未知期望表现：

\[
\mu_i=E[Y_i].
\]

最终要找：

\[
\boxed{
i^\star=\arg\max_{i=1,\ldots,N}\mu_i
}
\]

这完全可以写成：

\[
\boxed{
x^\star=
\arg\max_{x\in\{x_1,\ldots,x_N\}}F(x)
}
\]

因此：

\[
\boxed{
\text{Ranking & Selection = 有限离散候选上的 Optimization}
}
\]

它之所以形成独立文献，是因为常常还有：

- simulation 是随机的；
- 同一个候选需要重复运行很多次；
- 重点是如何分配 replication budget；
- 常用 probability of correct selection 作为评价指标。

例如 EV Charging Station 论文：

~~~text
固定候选设计
+ 随机仿真
+ multi-fidelity
+ 仿真预算分配
→ 最后选择最优充电站方案
~~~

最终任务仍然属于 Optimization。

---

# 7. Inverse Problem 的 point-estimation 形式也可以归到 Optimization

假设现实观察为：

\[
y_{\mathrm{obs}},
\]

模型输出为：

\[
F(\theta).
\]

如果目标是找一个参数：

\[
\theta^\star
=
\arg\min_\theta
\left\|
F(\theta)-y_{\mathrm{obs}}
\right\|^2,
\]

定义：

\[
J(\theta)
=
\left\|
F(\theta)-y_{\mathrm{obs}}
\right\|^2,
\]

就得到：

\[
\boxed{
\theta^\star=\arg\min_\theta J(\theta)
}
\]

从优化结构来看，\(\theta\) 就是普通 optimization 里的 \(x\)。

区别只是 objective 的科学含义变成：让模型输出尽可能匹配真实 observation。

因此，在本文采用的高层分类中：

\[
\boxed{
\text{point-estimation inverse problem 可以归到 Optimization}
}
\]

需要注意，如果目标不是找单个 \(\theta^\star\)，而是推断完整 posterior：

\[
p(\theta\mid y_{\mathrm{obs}}),
\]

那么任务已经超出了单纯 optimization。

本文当前讨论的是可以写成单点最小化问题的 inverse problem。

---

# 8. Benders / Mathematical Programming 仍然属于 Optimization

Benders decomposition 可能完全没有 surrogate、GP 或 Bayesian acquisition。

它利用 optimization constraints、dual information、lower bound、cuts、relaxation。

但最终仍然是：

\[
x^\star=\arg\min_x J(x).
\]

因此：

\[
\boxed{
\text{Benders 是 Optimization 的另一类求解方法}
}
\]

它和 Bayesian Optimization 的最终任务可以相同，只是利用的问题结构不同。

---

# 9. 第三大类：Active Search / Discovery

这是与前两类真正不同的最终目标。

Active Search 不要求把整个函数学准，也不只找一个最优点。

它的目标是：

\[
\boxed{
\max_\pi
\mathbb E
\left[
\#\{x:F(x)>\tau\text{ 且已经被确认}\}
\right]
}
\]

也就是：

> 在有限预算内，尽可能多发现并确认满足条件的 positive / target / discovery。

---

## 9.1 和 Optimization 的区别

Optimization：

\[
\boxed{
\text{只关心最好的一个 }x
}
\]

例如：

\[
x^\star=\arg\max_x F(x).
\]

Active Search：

\[
\boxed{
\text{关心有多少个不同的 }x\text{ 满足阈值}
}
\]

例如：

\[
F(x)>\tau.
\]

即使已经找到全局最大值，只找到一个点仍然不能完成 Active Search 的任务。

---

## 9.2 和 Prediction 的区别

Prediction：

\[
\boxed{
\text{希望 }F(x)\text{ 在整个域内尽可能准确}
}
\]

Active Search：

\[
\boxed{
\text{只关心尽量多确认 positive points}
}
\]

一个 Active Search 方法完全可能容忍大部分负例区域预测很差，只要它仍然能高效找到更多正例。

---

## 9.3 代表论文

### MF-ASC

**Figuring out the User in a Few Steps: Bayesian Multifidelity Active Search with Cokriging**

属于：

\[
\boxed{
\text{Multi-Fidelity + Active Search}
}
\]

### MF-ENS

**Nonmyopic Multifidelity Active Search**

属于：

\[
\boxed{
\text{Multi-Fidelity + Active Search + Nonmyopic Planning}
}
\]

核心目标是在有限 budget 下确认尽可能多 positive points。

它仍然使用有限 candidate pool。

### MICRO

**MICRO: Multi-Fidelity Active Search for Severe Error Discovery**

目标同样是：

\[
\boxed{
\text{最大化 confirmed severe-error discoveries}
}
\]

也属于 Multi-Fidelity Active Search。

---

# 10. Multi-Fidelity 不是第四种任务

Multi-fidelity 描述的是：

> 同一个任务是否存在多个成本和精度不同的信息来源或计算方式？

因此 Multi-Fidelity 可以出现在任何一类任务中。

## 10.1 Prediction 可以 Multi-Fidelity

代表：

- Kennedy–O’Hagan
- Picheny–Ginsbourger
- IFC
- DMFAL

## 10.2 Optimization 可以 Multi-Fidelity

代表：

- MF-GP-UCB
- BOCA
- FABOLAS
- MISO
- MF-MES
- DNN-MFBO
- BMBO-DARN
- EV Ranking & Selection
- Multi-Fidelity Benders

## 10.3 Active Search 也可以 Multi-Fidelity

代表：

- MF-ASC
- MF-ENS
- MICRO

---

# 11. 最终统一分类

因此，在“论文最终究竟想解决什么任务”这一层，本文采用下面三大类。

| 最终任务 | 数学核心 | 最终想得到什么 |
|---|---|---|
| Prediction / Function Learning | \(\hat F(x)\approx F(x)\) | 一个能预测大量新输入的模型 |
| Optimization | \(x^\star=\arg\min/\arg\max J(x)\) | 一个最好或近似最好的输入 |
| Active Search / Discovery | \(\max \#\{x:F(x)>\tau\}\) | 尽可能多的满足条件的不同输入 |

最容易记成：

\[
\boxed{
\text{Prediction：想知道整个函数}
}
\]

\[
\boxed{
\text{Optimization：想知道哪个点最好}
}
\]

\[
\boxed{
\text{Active Search：想找到尽可能多的合格点}
}
\]

---

# 12. 各种常见术语应该放在哪一层？

~~~yaml
Prediction_Function_Learning:
  final_goal:
    - "learn F(x)"
    - "predict unseen x"
  methods_or_subsettings:
    - "ordinary surrogate modeling"
    - "active learning"
    - "multi-fidelity regression"
    - "physics-informed surrogate"

Optimization:
  final_goal:
    - "find argmin or argmax"
  methods_or_subsettings:
    - "Bayesian optimization"
    - "evolutionary optimization"
    - "Hyperband / successive halving"
    - "ranking and selection"
    - "point-estimation inverse problem"
    - "mathematical programming"
    - "Benders decomposition"

Active_Search:
  final_goal:
    - "maximize number of confirmed positives"
  methods_or_subsettings:
    - "greedy active search"
    - "nonmyopic rollout"
    - "multi-fidelity active search"
~~~

---

# 13. Survey / Review / Benchmark 不作为第四类科学任务

例如：

- Peherstorfer et al. multi-fidelity survey；
- Fernández-Godino review；
- Bischl et al. HPO survey；
- JAHS-Bench-201。

这些工作的作用是总结领域、提供 benchmark、提供数据与 surrogate、给其他算法测试。

它们本身没有必要定义一个新的最终科学任务。

---

# 14. 对当前研究方向的直接含义

当前研究问题是：

~~~yaml
design_space:
  type: "continuous"
  variable: "x"

solver:
  type: "iterative numerical solver"
  features:
    - "partial progress can be observed"
    - "computation can be paused"
    - "saved state can be resumed"

fidelity:
  meaning: "solver progress / convergence level"
  desired_model: "continuous or finely discretized"

target:
  threshold: "tau"
  positive: "F(x) > tau"

objective:
  type: "Active Search"
  goal: "under a fixed total computational budget, confirm as many different positive x as possible"
~~~

因此它属于：

\[
\boxed{
\text{Active Search / Discovery}
}
\]

而不是普通 prediction，也不是单最优点 optimization。

同时它还具有：

\[
\boxed{
\text{Multi-Fidelity}
+
\text{Continuous Design Space}
+
\text{Pause/Resume Solver}
+
\text{Incremental Cost}
}
\]

这些是任务结构和计算结构上的额外属性。

---

# 15. 与已有文献的关系

当前找到的已有工作分别覆盖这些组件：

~~~text
MF-ASC / MF-ENS / MICRO
    → Multi-Fidelity + Active Search
    → 主要仍是 finite candidate pool

BOCA / CAMERA
    → continuous x / continuous fidelity
    → 主要目标不是 Active Search discovery count

Picheny–Ginsbourger
    → partially converged numerical solver
    → 预测最终收敛结果

Freeze-Thaw Bayesian Optimization
    → pause / resume / continue
    → 最终目标是 optimization，不是 Active Search

Jiang et al. multi-step trees
    → continuous-space nonmyopic lookahead
    → 最终目标是 Bayesian optimization
~~~

截至当前检索，没有找到一篇工作完整统一：

\[
\boxed{
\text{Continuous Design}
+
\text{Multi-Fidelity}
+
\text{Active Search}
}
\]

如果进一步加入：

\[
\boxed{
\text{Pause/Resume Solver}
+
\text{Incremental Cost}
}
\]

则与当前问题的匹配会更窄。

因此，当前研究的核心位置可以概括为：

\[
\boxed{
\text{Continuous-Space Multi-Fidelity Active Search for Resumable Numerical Solvers}
}
\]

其最终目标始终是：

\[
\boxed{
\text{在有限总计算预算内确认尽可能多的不同正例}
}
\]
