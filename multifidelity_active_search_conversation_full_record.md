# 36 篇 Multi-Fidelity / Active Search 文献统一框架：完整对话整理

> 本文件按本次对话的实际推进顺序整理，从最开始讨论“36 篇文章能否统一成 design space + fidelity space + cost + performance/discrepancy 的框架”开始，一直到讨论“这 26 篇核心 multi-fidelity 论文里是否有强化学习方法”为止。
>
> 为保证可读性，重复口语、情绪词和完全重复的句子做了轻微压缩；所有技术问题、公式、纠正、分类、边界条件和最终结论均保留。

---

# 1. 起点：36 篇文章能否统一成 design + fidelity + cost + performance/discrepancy？

用户最开始的问题是：

- 之前讨论过 36 篇文章；
- 发现这些文章似乎都有一个 **design space**；
- 还有一个 **fidelity space**；
- design 与 fidelity 共同决定 cost 与 performance / accuracy / difference；
- 想逐篇明确 design、fidelity、cost、performance / discrepancy 的具体变量；
- 更重要的是：在真正 evaluation 之前，算法怎么估计 cost 与 performance / discrepancy。

一开始采用的统一形式是：

\[
x\in\mathcal X
\]

表示 design / configuration / physical input，

\[
z\in\mathcal Z
\]

表示 fidelity / resource / information source / solver progress。

一次 evaluation 给出：

\[
Y(x,z)
\]

以及：

\[
C(x,z).
\]

更一般可以写成：

\[
Y_{x,z}\sim P_{x,z},
\qquad
C_{x,z}\sim Q_{x,z}.
\]

这里 \(Y\) 可以是标量、场、标签、排序信息、统计估计量、优化下界等；\(C\) 可以是 wall-clock time、CPU/GPU time、iteration、simulation budget 等。

---

# 2. 36 篇与旧的 26 篇

之前有一份完整结构化审读文件，覆盖 **26 篇核心 multi-fidelity 文献**。后来又补进约 10 篇 Active Search、non-myopic planning、continuous fidelity、freeze-thaw 等相关工作，因此当前讨论的总集合是：

\[
26+10=36.
\]

新增代表包括 MF-ASC、MF-ENS、CAMERA、Freeze-Thaw Bayesian Optimization、MICRO、Guth / Champenois / Sapsis、Palizhati 等材料发现、Active Area Search、Gong–Pan、One-Shot Multi-Step Bayesian Optimization。

所以：

\[
\boxed{36\text{ 篇是相关文献总集合}}
\]

但并不是 36 篇全部都有独立的 fidelity space。

---

# 3. 并不是 36 篇都有 fidelity space

特别讨论了：

- Active Area Search；
- One-Shot Multi-Step Bayesian Optimization。

它们原始问题本身并不是 multi-fidelity formulation。

Active Area Search 更像：

\[
x\longrightarrow y(x),
\]

只选择：

\[
x_{t+1}\in\mathcal X.
\]

One-Shot Multi-Step BO 也是：

\[
x_t\rightarrow Y_t\rightarrow x_{t+1}\rightarrow Y_{t+1}\rightarrow\cdots
\]

但没有额外 fidelity variable \(z\)。

因此：

\[
\boxed{\text{multi-step}}
\]

表示考虑未来多个决策步骤；

而：

\[
\boxed{\text{multi-fidelity}}
\]

表示同一个 design 可以选择不同质量 / 计算成本的 evaluation。

---

# 4. 对 multi-fidelity 问题，\(X\) 和 \(Z\) 是什么？

对于真正的 multi-fidelity 方法：

\[
(x,z)\in\mathcal X\times\mathcal Z.
\]

其中：

- \(x\)：选“算谁”；
- \(z\)：选“算到什么程度 / 用哪个信息源 / 用多高精度”。

例如 solver 场景：

\[
x=\text{物理参数 / 设计参数},
\]

\[
z=\text{solver progress / tolerance / iteration level}.
\]

一个 action 可能是：

\[
(x_1,z=0.2),
\]

或：

\[
(x_2,z=0.8).
\]

---

# 5. Observation、target 与 discrepancy

定义：

\[
Y(x,z)
=
\text{在 design }x\text{、fidelity }z\text{ 下得到的 observation}.
\]

指定 target fidelity：

\[
z^\star.
\]

最终 target：

\[
T(x)=Y(x,z^\star).
\]

定义 discrepancy：

\[
\delta(x,z)=Y(x,z)-T(x),
\]

或误差大小：

\[
D(x,z)=d\!\left(Y(x,z),T(x)\right).
\]

最简单：

\[
D(x,z)=|Y(x,z)-Y(x,z^\star)|.
\]

关键点：

\[
\boxed{D=D(x,z)}
\]

一般同时依赖 \(x\) 和 \(z\)，不能只写成 \(D(z)\)。

例如同样 300 iterations：

\[
D(x_1,300)=0.005,
\]

而另一个难收敛设计可能：

\[
D(x_2,300)=0.12.
\]

---

# 6. 一度引入 \(R\)，后来决定继续只用 \(D\)

为了兼容 Benders、MISO、Branke 等并非普通数值误差的情况，一度写成：

\[
R\big(Y(x,z),T(x)\big).
\]

例如：

- 普通数值近似：absolute error；
- MISO：bias / correlation / noise；
- Branke：ranking agreement；
- Benders：bound tightness / validity。

后来为了避免符号复杂化，统一继续写：

\[
\boxed{
D(x,z)=\ell\!\left(Y(x,z),T(x)\right)
}
\]

只不过损失 \(\ell\) 不必永远是 absolute error。

---

# 7. 最准确的“母框架”

对 26 篇核心 multi-fidelity 文献，底层 evaluation structure 是：

\[
\boxed{
(x,z)
\longrightarrow
\left[
Y(x,z),\;
C(x,z)
\right]
}
\]

并存在最终 target：

\[
\boxed{T(x)}
\]

通常：

\[
T(x)=Y(x,z^\star).
\]

再定义 fidelity quality：

\[
\boxed{
D(x,z)
=
\ell\!\left(
Y(x,z),T(x)
\right).
}
\]

真实世界里 \(D\) 由 \(Y\) 和 \(T\) 定义；算法决策时，\(Y(x,z)\)、\(T(x)\)、\(C(x,z)\) 在 evaluation 前往往未知，所以真正需要的是：

\[
\widehat D(x,z),
\qquad
\widehat C(x,z),
\]

或者对应 posterior / predictive distribution。

---

# 8. \(C(x,z)\) 的来源：很多论文直接假设

现实最一般：

\[
C=C(x,z).
\]

但很多理论论文简化为：

\[
C(x,z)\approx C(z),
\]

甚至：

\[
C(z_m)=\lambda_m.
\]

cost 的来源大概分为：

1. **直接给定常数**：MF-GP-UCB、MF-MES、DNN-MFBO、BMBO-DARN、DMFAL；
2. **人工 cost function**；
3. **复杂度公式**，如 \(C(N,T)\propto N^2T\)；
4. **实际 runtime 测量**；
5. **从数据学习 cost**，FABOLAS 是典型代表。

FABOLAS 建：

\[
f(x,s)=\text{validation loss},
\]

和：

\[
c(x,s)=\text{computation time},
\]

并对：

\[
\log c(x,s)
\]

建 GP。

---

# 9. \(D(x,z)\) 的来源：比 cost 更难

真实：

\[
D(x,z)=d\!\left(Y(x,z),T(x)\right).
\]

但要知道真实 \(D\)，必须知道：

\[
T(x)=Y(x,z^\star).
\]

因此如果只有 low-fidelity 数据、没有任何 target-fidelity 数据，那么：

\[
\boxed{D(x,z)\text{ 无法凭空识别}}
\]

除非有理论 / 结构假设。

主要来源：

1. **直接假设误差上界**，如 MF-GP-UCB：
   \[
   |f^{(m)}(x)-f^{(M)}(x)|\leq\zeta_m.
   \]

2. **GP 学 discrepancy**，如：
   \[
   T(x)=\rho Y_L(x)+\delta(x).
   \]

3. **联合 GP 学 \(Y(x,z)\)**，再隐式得到 discrepancy，如 BOCA。

4. **convergence trajectory**，如：
   \[
   Y(x,t)=T(x)+G(x,t),\qquad G(x,t)\to0.
   \]

5. **solver error indicator**。

6. **数学结构直接保证**，如 Benders lower bound / valid cut。

---

# 10. GP 到底在干什么？

对 GP-based multi-fidelity 方法，可以非常朴素地理解：

\[
\boxed{
\text{已有真实散点}
\rightarrow
\text{GP surrogate}
\rightarrow
\text{预测没见过的位置}
}
\]

普通 GP：

\[
x\rightarrow f(x)
\]

是在 \(x\)-space 中插值。

Multi-fidelity GP：

\[
(x,z)\rightarrow f(x,z)
\]

是在：

\[
\mathcal X\times\mathcal Z
\]

联合空间中插值 / 平滑。

已有：

\[
\mathcal D=
\{(x_i,z_i,y_i)\}_{i=1}^n,
\]

对于新点：

\[
(x_{\rm new},z_{\rm new}),
\]

GP 给：

\[
\mu(x,z),
\qquad
\sigma^2(x,z).
\]

zero-noise deterministic GP 更接近精确 interpolation；有噪声时更准确叫 regression / smoothing。FABOLAS 有时甚至是在 \(s<1\) 的数据上向 \(s=1\) 做 extrapolation。

---

# 11. 26 篇里哪些真正“根据当前信息联合选择下一组 \((x,z)\)”？

采用严格标准：

> 算法利用当前 observations / posterior / surrogate，下一轮同时决定 design \(x\) 和 fidelity \(z\)。

筛出来 9 篇：

1. MF-GP-UCB；
2. BOCA；
3. FABOLAS；
4. DNN-MFBO；
5. BMBO-DARN；
6. DMFAL；
7. MISO / misoKG；
8. MF-MES；
9. EV AOMS。

其中前 8 篇是典型 surrogate/acquisition 型联合 \((x,z)\) 选择。AOMS 也是 design–fidelity pair allocation，但走统计 allocation / ranking-and-selection 路线。

---

# 12. 很接近但不算严格 joint \((x,z)\) acquisition 的几篇

## 12.1 Branke 2017

它会：

\[
\text{先把 }x\text{ 算到低 fidelity}
\]

然后根据 ranking-reversal risk 决定：

\[
\text{要不要继续把这个 }x\text{ 算到更高 }z.
\]

所以：

\[
\boxed{\text{adaptive }z}
\]

但不是每轮在完整：

\[
\mathcal X\times\mathcal Z
\]

里统一评分。

## 12.2 Hyperband

流程：

\[
\text{抽配置}
\rightarrow
\text{小 budget 跑}
\rightarrow
\text{看结果}
\rightarrow
\text{淘汰}
\rightarrow
\text{剩余配置增加 budget}.
\]

fidelity schedule 基本预定，不是每轮自由优化 \(z\)。

## 12.3 MF-Benders

根据当前 bound improvement 决定：

\[
z_{\rm daily}
\rightarrow
z_{\rm weekly}
\rightarrow
z_{\rm monthly}
\rightarrow
z_{\rm yearly}.
\]

\(x\) 来自 master optimization，\(z\) 是单独 level switching。

## 12.4 Forrester 2006

主要选择：

\[
x_{\rm next}
\]

然后对值得的设计做 fully converged calculation。

## 12.5 NAS with Knowledge Distillation

初始化收集两级数据后，主要通过 high-fidelity prediction / UCB 选 architecture \(x\)，随后执行较高 fidelity evaluation。

---

# 13. 这些论文的最终数学任务怎么分？

一度按“BO / HPO / NAS / ranking-and-selection / active learning”分类，但这种分类混合了数学目标、应用领域和算法名字。

更干净的标准是：

\[
\boxed{\text{最终要解的数学对象是什么？}}
\]

## 13.1 A 类：best-design identification / optimization

数学结构：

\[
\boxed{
x^\star
=
\arg\max_{x\in\mathcal X}T(x)
}
\]

或：

\[
x^\star
=
\arg\min_{x\in\mathcal X}T(x).
\]

ordinary BO、HPO、NAS、ranking-and-selection、Hyperband、MF-Benders 等都可以归入这类。

26 篇里主要包括：

\[
2,5,7,8,9,10,11,12,16,17,18,22,23,25,26.
\]

## 13.2 B 类：function learning / reconstruction

最终不是一个 \(x^\star\)，而是：

\[
\boxed{
\hat T(\cdot)\approx T(\cdot)
}
\]

目标：

\[
\min
\mathcal L(\hat T,T).
\]

包括 Kennedy–O’Hagan、Picheny、MF-PINN、DMFAL、IFC、Composite MFNN / MPINN。

其中 DMFAL 是 active acquisition，但 terminal object 仍然是整个函数 / 场。

## 13.3 C 类：cost–accuracy Pareto model design

例如 DES OMFSM：

\[
\operatorname{Pareto}
\left\{
(D(z),C(z)):z\in\mathcal Z
\right\}.
\]

## 13.4 D 类：survey / benchmark / meta

例如 Peherstorfer survey、Fernández-Godino review、HPO survey、JAHS-Bench-201。

---

# 14. 两个正交层次

以后最好始终拆成：

## 第一层：evaluation / information structure

\[
\boxed{
(x,z)
\rightarrow
\left(
Y(x,z),D(x,z),C(x,z)
\right)
}
\]

## 第二层：terminal objective

例如：

### best design

\[
\Phi(T)=\arg\max_x T(x).
\]

### function reconstruction

\[
\Phi(T)=T(\cdot).
\]

### Pareto design

\[
\Phi(D,C)=\operatorname{Pareto}(D,C).
\]

这样就不会把 BO、HPO、NAS、R&S 错误拆成不同顶层数学任务。

---

# 15. 为什么 MF-MES、FABOLAS、MISO 大多 one-step，而 MF-ENS 要 lookahead？

关键不是名字不同。

真正区别是：

\[
\boxed{
\text{当前动作的价值是否能被一个 one-step surrogate 表达}
}
\]

以及：

\[
\boxed{
\text{未来树算不算得动}
}
\]

---

# 16. MF-MES：用 information value 压缩未来价值

MF-MES acquisition：

\[
A(x,m)
=
\frac{
I\!\left(
f^\star;
Y^{(m)}(x)
\mid\mathcal D
\right)
}{
\lambda_m
}.
\]

其中：

\[
f^\star
=
\max_x f^{(M)}(x).
\]

low-fidelity query 即使没有即时最终 reward，也可能：

\[
I(f^\star;Y_L)>0.
\]

所以它把：

\[
\text{“这条 LF 信息未来可能有用”}
\]

压缩成当前一个 scalar acquisition value。

---

# 17. MF-MES 分子怎么估计？

已有：

\[
\mathcal D_t
=
\{(x_i,m_i,y_i)\}_{i=1}^t.
\]

multi-fidelity GP 对任意：

\[
(x,m)
\]

给：

\[
f^{(m)}(x)\mid\mathcal D_t
\]

以及：

\[
\mu_t^{(m)}(x),
\qquad
\sigma_t^{2(m)}(x).
\]

同时知道不同 fidelity 之间的 covariance。

但是：

\[
f^\star=\max_x f^{(M)}(x)
\]

本身也未知。

所以从当前最高 fidelity GP posterior 中采样可能的函数：

\[
f_1^{(M)}(x),\;
f_2^{(M)}(x),\ldots
\]

每条函数取最大值：

\[
f_j^\star
=
\max_x f_j^{(M)}(x).
\]

得到：

\[
F^\star=
\{f_1^\star,\ldots,f_K^\star\}.
\]

从而近似：

\[
p(f^\star\mid\mathcal D_t).
\]

Mutual information 写成：

\[
I(f^\star;f^{(m)}(x)\mid\mathcal D_t)
=
H\!\left(
f^{(m)}(x)\mid\mathcal D_t
\right)
-
\mathbb E_{f^\star}
\left[
H\!\left(
f^{(m)}(x)
\mid
f^\star,\mathcal D_t
\right)
\right].
\]

直观上：

\[
\boxed{
\text{观察前的不确定性}
-
\text{知道 optimum 后剩余的不确定性}
}
\]

---

# 18. MF-MES 分母不是预测的

MF-MES 假设：

\[
\boxed{\lambda^{(m)}\text{ 已知}}
\]

通常：

\[
\lambda^{(1)}
\leq
\lambda^{(2)}
\leq
\cdots
\leq
\lambda^{(M)}.
\]

所以：

\[
\boxed{
\text{MF-MES}
=
\frac{\text{posterior 估出来的信息价值}}
{\text{预先给定的 cost}}
}
\]

---

# 19. FABOLAS：分子预测 + 分母预测

FABOLAS 有：

\[
f(x,s)=\text{validation loss},
\]

以及：

\[
c(x,s)=\text{computation time}.
\]

对：

\[
\log c(x,s)
\]

单独建 GP。

候选：

\[
(x,s)
\]

未运行前有：

\[
\widehat c(x,s).
\]

分母近似：

\[
\widehat c(x,s)+c_{\rm overhead}.
\]

FABOLAS 分子是：

\[
\boxed{
\text{如果现在做 }(x,s)\text{，
平均而言能让我对 full-data optimum 清楚多少？}
}
\]

当前 GP 有：

\[
p(y\mid x,s,D).
\]

想象可能的：

\[
y_1,y_2,\ldots
\]

每个假想 observation 形成：

\[
D'=D\cup\{(x,s,y)\}.
\]

重新看：

\[
p_{\min}^{s=1}(x'\mid D')
\]

变化多少，然后做：

\[
\mathbb E_{y\mid x,s,D}
[
\text{information gain}
].
\]

---

# 20. MISO：one-step Knowledge Gradient

MISO / misoKG 问的是：

> 如果现在做一次 \((x,z)\) observation，然后立刻做最终决策，预期最佳决策价值能提升多少？

所以它属于：

\[
\boxed{\text{one-step optimality}}
\]

而不是完整多步 planning。

---

# 21. MF-ENS 为什么必须 non-myopic？

MF-ENS 最终 reward：

\[
\boxed{
U
=
\#\{\text{被 high-fidelity 确认的 positive}\}
}
\]

low-fidelity query 本身不产生最终 reward：

\[
r_L=0.
\]

所以 LF 的价值只能通过：

\[
L
\rightarrow
\text{获得信息}
\rightarrow
\text{更新 posterior}
\rightarrow
\text{改变未来 H query}
\rightarrow
\text{未来多发现 positive}
\]

体现。

因此它必须显式考虑：

\[
\boxed{
\text{不同未来 LF outcomes 下，后续 H 选择会怎样变化}
}
\]

---

# 22. MF-ENS 的 \(2^5=32\)、\(32\times2=64\)

如果：

\[
k=5,
\]

未来五个 low-fidelity binary labels：

\[
y_{L,i}\in\{0,1\}.
\]

未来结果组合：

\[
2^5=32.
\]

如果再对一个 binary high-fidelity outcome：

\[
y_H\in\{0,1\}
\]

做 expectation，则某些 scoring 状态下会出现：

\[
2\times2^5=64.
\]

但真正普遍的指数项是：

\[
\boxed{2^k}
\]

而不是固定 64。

---

# 23. 为什么 BO 里 multi-step lookahead 更难？

分两个问题：

## 23.1 为什么“值得”lookahead？

由 reward / terminal utility 结构决定。

如果当前动作没有 immediate reward，只通过改变未来策略有价值，non-myopic 动机更强。

## 23.2 为什么“算不动”？

由 observation space、action space、horizon 共同决定。

---

# 24. Observation space 决定 future branching

如果：

\[
Y\in\{0,1\},
\]

每一步只有：

\[
b=2
\]

个未来结果。

往前看 \(H\) 步：

\[
b^H=2^H.
\]

如果 3 类结果：

\[
3^H.
\]

如果：

\[
Y\in\mathbb R,
\]

第一层就是连续无穷分支，不能像 MF-ENS 一样精确枚举。

---

# 25. Continuous-output BO 如何做 non-myopic？

One-Shot Multi-Step Bayesian Optimization 处理：

\[
x_t
\rightarrow
Y_t
\rightarrow
x_{t+1}
\rightarrow
Y_{t+1}
\rightarrow
\cdots
\]

虽然：

\[
Y_t\in\mathbb R,
\]

但可以从 predictive distribution 采：

\[
Y_t^{(1)},\ldots,Y_t^{(M)}
\]

作为 fantasy samples。

于是 continuous outcome 被近似成 \(M\) 个 scenario branches。

如果每层 \(M\) 个 fantasy：

\[
M^H
\]

仍然指数增长。

例如：

\[
M=16,\ H=4
\Rightarrow
16^4=65536,
\]

\[
M=32,\ H=5
\Rightarrow
32^5=33,554,432.
\]

---

# 26. Action space 连续性又增加一层难度

未来树每个节点还要选：

\[
a=(x,z).
\]

如果：

\[
x\in\mathbb R^d,
\qquad
z\in[0,1],
\]

那么每一个 fantasy branch 上还要解：

\[
\arg\max_{x,z}A(x,z).
\]

所以真正复杂度来自：

\[
\boxed{
\text{outcome branching}
\times
\text{action optimization}
\times
\text{belief update}
\times
\text{horizon}
}
\]

---

# 27. Exact full-horizon non-myopic 通常极贵

完整动态规划：

\[
V(S_t,B_t)
=
\max_{a_t}
\mathbb E_{Y_t}
\left[
r_t+
V(S_{t+1},B_{t+1})
\right].
\]

再展开：

\[
\max E[\max E[\max E[\cdots]]].
\]

实践中通常做：

- short horizon；
- rollout；
- scenario sampling；
- branch-and-bound / pruning；
- receding horizon；
- learned value function / policy。

所以：

\[
\boxed{
\text{exact full-horizon non-myopic 通常不可做}
}
\]

但：

\[
\boxed{
\text{approximate non-myopic 可以做}
}
\]

---

# 28. 用户当前的 solver 问题：不是静态 \((x,z)\)，而是 stateful \((x,\Delta z)\)

用户进一步明确：

> 可以先把某个 \(x\) 算到 50%，暂停；去算其他 \(x\)；过一段时间再回过头，把原来的 50% 继续推进到 100%。

因此 fidelity 是：

\[
\boxed{
\text{stateful / resumable}
}
\]

当前状态：

\[
\boxed{
S_t
=
\{(x_i,z_i,\text{trajectory}_i)\}_{i=1}^{n_t}
}
\]

下一步动作更应该写成：

\[
\boxed{
a_t=(x_i,\Delta z_i)
}
\]

而不是重新选择绝对 \(z\)。

状态转移：

\[
z_i^{t+1}
=
z_i^t+\Delta z_i.
\]

---

# 29. 新点与旧点续算统一成两类 action

对于连续 \(\mathcal X\)，不会真的存储“所有 \(x\) 的 \(z=0\)”。

实现上有：

\[
\boxed{
\operatorname{Start}(x,\Delta z)
}
\]

对从未运行过的新 design；

以及：

\[
\boxed{
\operatorname{Resume}(i,\Delta z)
}
\]

对已有 candidate \(x_i\)。

二者可以统一成 state-dependent action space：

\[
\boxed{
\mathcal A(S_t)
}
\]

---

# 30. Cost 应该是增量 cost

普通 MFBO 常写：

\[
C(x,z).
\]

但 resumable solver 更应该关心：

\[
\boxed{
\Delta C(x,z,\Delta z)
}
\]

或：

\[
\boxed{
C(x,z\rightarrow z+\Delta z).
}
\]

如果 checkpoint 能继续：

\[
\Delta C
\approx
C(x,z+\Delta z)-C(x,z).
\]

现实还可加：

\[
C_{\rm restart}
\]

和：

\[
C_{\rm decision}.
\]

---

# 31. Discrepancy 也变成推进后的 future accuracy

当前已经在：

\[
z.
\]

继续：

\[
\Delta z
\]

以后关心：

\[
D(x,z+\Delta z).
\]

甚至：

\[
\Delta D
=
D(x,z)-D(x,z+\Delta z).
\]

因此一个局部 value/cost quantity 可以是：

\[
\frac{
\mathbb E[\Delta D]
}{
\mathbb E[\Delta C]
}.
\]

但如果最终任务是 Active Search，真正 numerator 更应该是：

\[
\boxed{
\text{对最终 confirmed-positive count 的预计增量价值}
}
\]

而不是单纯 accuracy improvement。

---

# 32. 这个 stateful structure 与 Branke、Freeze-Thaw 的关系

Branke 与用户设定的相似点：

\[
\text{先低 fidelity}
\rightarrow
\text{观察}
\rightarrow
\text{决定是否继续当前 candidate}.
\]

Freeze-Thaw 的相似点：

- 多个 configurations；
- 每个 configuration 有 partial learning curve；
- 可以启动、暂停、恢复；
- 根据 partial trajectory 预测最终结果。

因此可以类比：

\[
\boxed{
\text{learning curve}
\leftrightarrow
\text{solver convergence curve}
}
\]

---

# 33. Active Search 与普通 MFBO 的最终确认成本差别

普通 best-design MFBO 最后主要只需要一个：

\[
\hat x^\star.
\]

如果想真实确认：

\[
T(\hat x^\star),
\]

只需：

\[
(\hat x^\star,z^\star)
\]

真正跑一次 target fidelity。

但用户的 Active Search 目标是：

\[
\boxed{
\text{找到大量 positive}
}
\]

最终发现集合：

\[
\boxed{
\mathcal P_B
=
\left\{
x_i:
z_i=1,\;
Y(x_i,1)>\tau
\right\}
}
\]

目标：

\[
\boxed{
\max_\pi
\mathbb E_\pi
\left[
|\mathcal P_B|
\right]
}
\]

subject to total budget。

因此：

\[
\boxed{
\text{每一个最终计入 reward 的 positive 都必须跑到 }z=1
}
\]

这是和普通 BO 极重要的区别。

---

# 34. Low fidelity 对 Active Search 真正省的是什么？

它不能省掉 confirmed positive 本身最终必须支付的 target-fidelity cost。

真正省的是：

1. 尽早淘汰 hopeless negatives；
2. 少在明显不可能 positive 的 candidate 上浪费 full fidelity；
3. 把 completion cost 优先给最有希望的 candidate；
4. 利用 intermediate observation 改善后续搜索。

理想漏斗：

\[
N_0
\rightarrow
N_1
\rightarrow
N_2
\rightarrow
N_3
\rightarrow
N_{\rm confirmed+}
\]

其中：

\[
N_0\gg N_1\gg N_2.
\]

最终每个：

\[
x\in N_{\rm confirmed+}
\]

都真正完成：

\[
z=1.
\]

---

# 35. Active Search 的 reward 很稀疏，因此天然 non-myopic

如果只有 full-fidelity positive 才计 1 分：

\[
r_t=
\begin{cases}
1,
&
z_{\rm new}=1
\text{ 且 }
Y(x,1)>\tau,\\
0,
&
\text{其他情况}.
\end{cases}
\]

那么：

\[
0\rightarrow0.2,\quad
0.2\rightarrow0.5,\quad
0.5\rightarrow0.8
\]

reward 都是：

\[
0.
\]

只有：

\[
0.8\rightarrow1
\]

且最后 positive 才：

\[
r=1.
\]

因此中间动作的价值全来自未来：

\[
\boxed{
\text{中间 observation}
\rightarrow
\text{判断是否值得继续}
\rightarrow
\text{重新分配剩余 budget}
\rightarrow
\text{最终多确认 positive}
}
\]

这和 MF-ENS low-fidelity query 的逻辑高度相似。

---

# 36. 为什么 MF-MES / FABOLAS 看起来像 fractional knapsack？

很多方法形式上写成：

\[
\frac{\text{value}}{\text{cost}}.
\]

这看起来像 fractional knapsack：

\[
\max \frac{v_i}{c_i}.
\]

但区别关键。

Fractional knapsack 中：

\[
(v_i,c_i)
\]

是固定的，而且：

\[
v(S)=\sum_{i\in S}v_i.
\]

拿了 A 不会改变 B 的 value/cost，所以 ratio greedy 可以严格最优。

BO 中：

\[
A_t(x,m)
=
\frac{I_t(x,m)}{c_m}.
\]

选了一个点并观察：

\[
y_1,
\]

posterior 会变化，于是：

\[
I_{t+1}(x,m)
\neq
I_t(x,m).
\]

不同 query 的信息还会重叠。因此它是：

\[
\boxed{
\text{state-dependent stochastic value}
}
\]

不是 additive fractional knapsack。

---

# 37. 真正完整的问题是 Bellman / sequential policy optimization

真正有限预算最优策略应写成：

\[
\boxed{
V(D,B)
=
\max_a
\mathbb E_{y\mid a,D}
\left[
V(D\cup\{(a,y)\},B-C(a))
\right]
}
\]

或带即时 reward：

\[
V(S,B)
=
\max_a
\mathbb E
\left[
r(S,a,Y)+V(S',B-C(a))
\right].
\]

MF-MES / FABOLAS 等通常不解完整 Bellman problem，而是：

\[
\boxed{
a_t
=
\arg\max_a
\frac{\text{one-step proxy value}(a)}
{\text{cost}(a)}
}
\]

因此它们是 tractable one-step Bayesian heuristics / acquisitions，而不是 full-horizon Bellman-optimal policies。

---

# 38. 这些论文有没有严格数学证明？

不能说“基本都没有数学”。

更准确有四个理论层次：

1. acquisition 公式推导；
2. 局部优化性质；
3. regret / consistency / asymptotic optimality；
4. finite-budget Bellman optimality。

很多论文有 1–3，但几乎没有证明 4。

## 38.1 MF-GP-UCB

有：

\[
\boxed{\text{regret guarantee}}
\]

并证明 multi-fidelity 信息能把 expensive evaluation 集中到 promising region。

但没有证明：

\[
\pi_{\rm MF-GP-UCB}
=
\pi^\star_{\rm Bellman}.
\]

## 38.2 BOCA

有 formal regret analysis / theorem。

同样不是 finite-horizon Bellman optimality。

## 38.3 MISO / misoKG

有：

- one-step optimality analysis；
- asymptotically near-optimal；
- finite-domain consistency。

仍然不等于有限预算全局最优 sequential policy。

## 38.4 EV AOMS

有：

- rate-optimal allocation；
- consistency；
- asymptotic optimality；
- PCS stopping rule。

理论很强。

## 38.5 MF-MES

更偏：

- information-theoretic acquisition；
- tractable approximation；
- empirical evaluation。

原始 single-fidelity MES 有 regret bound。

但：

\[
\boxed{
\text{MES 有 regret bound}
\neq
\text{MF-MES info/cost greedy 是全局预算最优}
}
\]

## 38.6 FABOLAS

属于：

\[
\boxed{
\text{principled heuristic / Bayesian decision heuristic}
}
\]

有清晰概率模型和 information-theoretic motivation，但没有 finite-budget Bellman optimality 证明。

---

# 39. 这 26 篇里有没有强化学习？

按通常较严格的机器学习定义：

\[
\boxed{
\text{没有一篇是在用标准 RL 算法解决问题}
}
\]

不过：

- MF-GP-UCB；
- Hyperband；

属于 bandit 方法。

bandit 在非常宽泛 taxonomy 中有时被看作 RL 的特殊 / 退化情形。

但如果按通常意义 RL：

- MDP；
- state transition；
- learned policy；
- value function；
- Q-learning；
- actor–critic；
- policy gradient；

这些 26 篇没有标准 RL 方法。

大多数属于：

- Bayesian sequential decision；
- Bayesian optimization；
- bandit；
- active learning；
- ranking-and-selection；
- evolutionary optimization；
- mathematical programming；
- surrogate modeling。

---

# 40. 用户当前 solver Active Search 反而更容易写成 MDP / belief-MDP

因为它真的有：

\[
S_t
=
\{(x_i,z_i,\text{solver state}_i)\}
\]

动作：

\[
a_t=(x_i,\Delta z_i),
\]

真实状态转移：

\[
z_i
\rightarrow
z_i+\Delta z_i,
\]

并有最终累计 reward：

\[
\#\{\text{full-fidelity confirmed positives}\}.
\]

因此这个问题可以形式化为：

\[
\boxed{
\text{MDP / belief-MDP / semi-MDP}
}
\]

但：

\[
\boxed{
\text{写成 MDP}
\neq
\text{一定要用 RL 算法}
}
\]

仍然可以用 Bayesian rollout、acquisition、approximate lookahead、value approximation、heuristic scheduler。

---

# 41. 整段对话最终得到的统一认识

## 41.1 底层 evaluation 结构

\[
\boxed{
(x,z)
\longrightarrow
\left[
Y(x,z),D(x,z),C(x,z)
\right]
}
\]

其中：

\[
D(x,z)
=
\ell\!\left(
Y(x,z),T(x)
\right),
\qquad
T(x)=Y(x,z^\star).
\]

## 41.2 对算法来说真正要估计的是

\[
\boxed{
\widehat D(x,z)
}
\]

和：

\[
\boxed{
\widehat C(x,z)
}
\]

因为：

- \(D\) 决定结果有多可信；
- \(C\) 决定代价多大。

## 41.3 这些量的来源可以完全不同

### \(D\)

- 直接假设；
- theoretical bound；
- GP discrepancy；
- joint GP posterior；
- convergence curve；
- solver residual / estimator；
- neural network；
- sample statistics；
- mathematical relaxation guarantee。

### \(C\)

- 固定常数；
- 人工 cost function；
- computational complexity formula；
- measured runtime；
- GP / regression cost model。

## 41.4 多数 MFBO 的核心不是 full dynamic programming

而是：

\[
\boxed{
\text{用一个 one-step acquisition proxy，
把 future value 压缩成当前一个分数}
}
\]

例如：

\[
\text{UCB},
\quad
\text{KG},
\quad
\text{MES},
\quad
\frac{\text{information}}{\text{cost}}.
\]

## 41.5 MF-ENS 的核心不同点

MF-ENS 的 low-fidelity action 本身没有最终 reward。

所以它必须考虑：

\[
\boxed{
\text{LF observation 会怎样改变未来 HF allocation}
}
\]

这产生真正的 policy-dependent future value。

binary output 让：

\[
2^k
\]

的 future branch enumeration 仍有可能。

## 41.6 用户当前问题的核心结构

用户的 Active Search + resumable solver 最自然写成：

\[
\boxed{
S_t=
\{x_i,z_i,\text{trajectory}_i\}
}
\]

动作：

\[
\boxed{
a_t=(x_i,\Delta z_i)
}
\]

状态更新：

\[
z_i^{t+1}=z_i^t+\Delta z_i.
\]

目标：

\[
\boxed{
\max_\pi
\mathbb E_\pi
\left[
\#\{
x:
z=1,\;
Y(x,1)>\tau
\}
\right]
}
\]

subject to total budget。

每一个最终计入 reward 的 positive 都必须完成：

\[
z=1.
\]

因此：

\[
\boxed{
\text{“推进谁到 100%”本身就是核心决策}
}
\]

而不是像普通 best-design MFBO 那样，只在最后主要确认一个 \(\hat x^\star\)。

---

# 42. 一句话总结整个对话

这批 multi-fidelity / active-search / BO 文献虽然应用和算法名字很多，但可以抽象成三层：

\[
\boxed{
\textbf{第一层：evaluation}
\quad
(x,z)\rightarrow(Y,D,C)
}
\]

\[
\boxed{
\textbf{第二层：belief / surrogate}
\quad
\mathcal D_t
\rightarrow
\widehat Y,\widehat D,\widehat C
}
\]

\[
\boxed{
\textbf{第三层：decision}
\quad
\text{根据当前 belief 选择下一步 action}
}
\]

普通 MFBO 大多使用 one-step acquisition。

MF-ENS / multi-step BO 试图显式考虑未来 policy。

而用户当前研究问题进一步加入：

\[
\boxed{
\text{continuous / resumable fidelity progress}
}
\]

和：

\[
\boxed{
\text{每个 confirmed positive 都必须完成 target fidelity}
}
\]

因此最自然的决策变量不再只是静态：

\[
(x,z),
\]

而是：

\[
\boxed{
(x,\Delta z)
}
\]

整个问题天然具有：

\[
\boxed{
\text{stateful、budgeted、non-myopic、sparse-reward}
}
\]

的结构。