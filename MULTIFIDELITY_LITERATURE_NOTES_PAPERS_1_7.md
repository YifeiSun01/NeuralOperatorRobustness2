# 多保真方法文献讨论整理：从离散模型到部分收敛、连续进度与动态计算分配

> 本文整理本次对话中围绕 Paper 1–Paper 7 的全部核心讨论。重点不是逐字转录聊天，而是把前后逻辑重新组织成一套可持续使用的文献笔记：每篇论文想解决什么问题、用了什么方法、fidelity 到底是什么、是否做优化、是否可续算、是否把计算进度作为模型输入，以及 2006、2013、2016/2017 三篇部分收敛工作之间到底有什么区别。

---

## 0. 统一记号与这组文献真正关心的问题

把设计或物理输入写成

\[
x.
\]

把最终最可信的高保真输出写成

\[
F(x).
\]

传统多保真文献通常还有一个 fidelity 标记

\[
z,
\]

例如

\[
z\in\{\text{coarse},\text{fine}\},
\]

或者

\[
z\in\{\text{Tadpole},\text{VSaero}\}.
\]

在“部分收敛求解器”这一支文献里，fidelity 不再一定表示不同模型，而可以表示同一个求解器已经推进到多少步：

\[
t=n_{\text{iter}}.
\]

于是一次观测可以写成

\[
y(x,t).
\]

这组文献实际上展示了一条非常清楚的演化路线：

1. 最开始：多个完全不同的模型或网格层级之间做信息融合；
2. 然后：把部分收敛的同一个 CFD 求解器当作 low fidelity；
3. 再然后：直接把 solver progress \(t\) 放进统计模型；
4. 最后：不再只预测最终数值，而是根据中途信息动态决定某个 candidate 是否值得继续算。

这条路线和“连续、可暂停、可续算的 solver progress”问题关系非常紧密。

---

# 1. Paper 1：Kennedy–O’Hagan 多层计算机代码模型

## 1.1 论文

M. C. Kennedy and A. O’Hagan,  
**Predicting the output from a complex computer code when fast approximations are available**,  
Biometrika, 87(1), 1–13, 2000.

参考：

- https://academic.oup.com/biomet/article/87/1/1/221217
- https://www2.stat.duke.edu/courses/Spring14/sta961.01/ref/KennOHag2000.pdf

## 1.2 它想解决什么问题

有一个昂贵但更可信的计算机代码

\[
f_H(x),
\]

还有一个便宜但粗糙的代码

\[
f_L(x).
\]

真正感兴趣的是预测

\[
f_H(x)
\]

在没有跑过昂贵代码的新输入 \(x\) 上会是多少。

核心问题是：

\[
\boxed{
\text{大量便宜计算}
+
\text{少量昂贵计算}
\rightarrow
\text{预测昂贵计算的输出}
}
\]

它的核心任务是预测，不是寻找最优 \(x\)。

---

## 1.3 油藏案例

论文中的经典例子是地下油藏流动模拟。

输入是 5 个区域的孔隙率和渗透率：

\[
x=(\phi_1,k_1,\phi_2,k_2,\ldots,\phi_5,k_5)\in\mathbb{R}^{10}.
\]

每个 \(\phi_i\) 是孔隙率，每个 \(k_i\) 是渗透率。

模拟器原本产生随物理时间变化的输出，但该论文选定一口井、一个固定物理时刻，只研究一个标量压力。

因此问题可以写成

\[
x\longrightarrow P(x).
\]

两个 fidelity 是：

\[
z=\text{coarse grid},
\]

以及

\[
z=\text{fine grid}.
\]

两者计算的是同一个物理量，只是空间网格精度不同。

论文把 fine 视为更可信参考，但 fine 并不等于现实世界的绝对真值。

---

## 1.4 成本关系

论文案例中，fine simulation 显著更贵。讨论中使用的典型相对成本是：

\[
1\text{ 次 fine}\approx36\text{ 次 coarse}.
\]

因此与其只做少量 fine，不如用大量 coarse 描述整体函数形状，再用少量 fine 修正。

训练配置的经典数字是：

\[
45\text{ 个 coarse}
+
7\text{ 个 fine},
\]

另外有 135 个输入用于检验，而不是 135 轮 Bayesian optimization。

---

## 1.5 核心模型

最著名的关系是

\[
\boxed{
f_H(x)=\rho f_L(x)+\delta(x)
}
\]

其中：

- \(\rho\) 是一个缩放/回归参数；
- \(f_L(x)\) 用 Gaussian Process 建模；
- \(\delta(x)\) 也用 Gaussian Process 建模。

所以整体结构很简单：

\[
\boxed{
\text{low-fidelity GP}
+
\text{一个缩放系数}
+
\text{discrepancy GP}
}
\]

没有深度神经网络，也没有大规模 SGD。

在油藏案例里，讨论中提到估计的缩放系数大约为

\[
\rho\approx0.71.
\]

---

## 1.6 GP 在这里到底是什么

Gaussian Process / Kriging 可以直观理解为一种“概率化的平滑插值”。

已知

\[
(x_1,y_1),\ldots,(x_n,y_n),
\]

对新点 \(x_*\)，GP 给出

\[
f(x_*)\sim
\mathcal N\left(
\mu(x_*),\sigma^2(x_*)
\right).
\]

它不仅给一个预测均值，还给预测不确定性。

所以它和大型神经网络的工作方式很不一样。这个案例只有几十个数据点，本来就属于 GP 很典型的使用场景。

---

## 1.7 新输入来了以后怎么预测

新输入

\[
x_*
\]

不需要真的做 fine simulation，也可以用

\[
\boxed{
\hat f_H(x_*)
=
\hat\rho\hat f_L(x_*)
+
\hat\delta(x_*)
}
\]

得到 high-fidelity 预测和不确定性。

如果允许新点先跑一次很便宜的 coarse simulation，也可以直接用真实的 \(f_L(x_*)\) 再加 correction。

---

## 1.8 这篇是不是只有离散 fidelity

它最著名、最常用的 autoregressive 模型确实是离散层级：

\[
f_1(x),f_2(x),\ldots,f_s(x).
\]

但论文后面还提出过一个连续 complexity parameter 的想法：

\[
z(x,t),\qquad t\in[0,1],
\]

并写成类似

\[
z(x,t)
=
z_0+
\int_0^t
f_\delta(x,\tau)\,d\tau.
\]

所以 Kennedy–O’Hagan 很早就意识到 fidelity/complexity 可以被连续参数化。

不过后来最有影响、最常被引用的仍然是离散的 autoregressive co-kriging：

\[
f_H=\rho f_L+\delta.
\]

---

# 2. Paper 2：两套气动程序共同优化机翼

## 2.1 论文

Forrester, Sóbester and Keane,  
**Multi-fidelity optimization via surrogate modelling**,  
Proceedings of the Royal Society A, 2007.

参考：

- https://doi.org/10.1098/rspa.2007.1900
- https://eprints.soton.ac.uk/64698/

## 2.2 最终任务

这篇已经不是单纯预测，而是真正做工程设计优化：

\[
\boxed{
x^*=\arg\min_x f_H(x)
}
\]

目标是找到高 fidelity 气动程序认为阻力最小的机翼。

机翼设计大致使用 4 个连续变量：

\[
x=(S,AR,\Lambda,T_{\rm in}),
\]

包括机翼面积、展弦比、后掠角和内翼锥度。

---

## 2.3 两个 fidelity

低 fidelity：

\[
\text{Tadpole}
\]

是快速经验气动程序。

高 fidelity：

\[
\text{VSaero}
\]

是更详细的气动分析程序。

典型运行时间：

\[
\text{Tadpole}\approx0.6\text{ s},
\]

\[
\text{VSaero}\approx2\text{ min}.
\]

因此高低成本差距大约是两个数量级。

这里的 fidelity 很明确是两个不同模型：

\[
\boxed{
z\in\{\text{Tadpole},\text{VSaero}\}
}
\]

不能把 Tadpole 运行到一半再“接着算成” VSaero。

---

## 2.4 关系模型

仍然沿用多保真 co-kriging 思路：

\[
f_H(x)
=
\rho f_L(x)+\delta(x).
\]

大量便宜 Tadpole 数据负责给出整个设计空间的大趋势；少量 VSaero 数据负责修正低保真模型的系统偏差。

---

## 2.5 初始数据

论文中先有大量 Tadpole 点，再在其中选较少 VSaero 点，例如讨论中过：

\[
100\text{ 个 Tadpole}
+
20\text{ 个 VSaero}.
\]

然后建立 high-fidelity surrogate 的均值和不确定性：

\[
\mu_H(x),\qquad \sigma_H(x).
\]

---

## 2.6 为什么它已经是 optimization

建立 surrogate 后，它计算 Expected Improvement：

\[
EI(x).
\]

每轮选

\[
\boxed{
x_{\rm next}
=
\arg\max_x EI(x)
}
\]

再到该位置增加昂贵气动评价，更新模型，再继续搜索。

所以流程是：

\[
\text{low/high data}
\rightarrow
\text{co-kriging}
\rightarrow
EI
\rightarrow
x_{\rm next}
\rightarrow
\text{新评价}
\rightarrow
\text{更新模型}.
\]

它是真正的 surrogate-based global optimization。

---

## 2.7 它还没有做什么

它主要决定的是

\[
x_{\rm next}.
\]

它没有在每一轮统一求

\[
(x_{\rm next},z_{\rm next}),
\]

也没有让 acquisition function 自由比较：

- 这一轮只跑 Tadpole；
- 这一轮跑 VSaero；
- 或者跑一个中间 fidelity。

因此它有多 fidelity，但 fidelity 本身还不是一个完整的在线决策变量。

---

# 3. Paper 3：多保真方法总综述

## 3.1 论文

Benjamin Peherstorfer, Karen Willcox, Max Gunzburger,  
**Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization**,  
SIAM Review, 2018.

参考：

- https://epubs.siam.org/doi/10.1137/16M1082469
- https://arxiv.org/abs/1806.10761

## 3.2 它不是一个单一算法

这篇论文自己不定义统一的 \(x\)、统一的 fidelity，也没有自己的计算预算。

它做的是：

\[
\boxed{
\text{把已有 multi-fidelity 方法整理成领域地图}
}
\]

核心问题是：

\[
\boxed{
\text{高精度模型太贵，而 outer loop 需要反复调用模型时，
怎样利用便宜模型减少高精度调用？}
}
\]

---

## 3.3 low fidelity 可以从哪里来

论文覆盖的 low fidelity 来源非常宽，包括：

- 简化物理模型；
- 粗空间/时间离散；
- reduced-order model；
- data-fit surrogate；
- 不同的信息来源。

因此 multi-fidelity 并不只等于“粗网格/细网格”。

---

## 3.4 三类核心策略

综述最重要的分类之一是：

\[
\boxed{
\text{Adaptation}
,\quad
\text{Fusion}
,\quad
\text{Filtering}
}
\]

### Adaptation

偶尔调用 high fidelity，然后利用高保真信息不断修改 low-fidelity 模型。

### Fusion

把 low/high 信息联合起来得到一个估计。

Kennedy–O’Hagan 的

\[
f_H=\rho f_L+\delta
\]

就是典型 fusion。

### Filtering

先用便宜模型筛选。

明显不值得看的候选直接淘汰；只有有希望或不确定的候选才继续调用高保真模型。

---

## 3.5 三个主要 outer-loop 应用

论文主要讨论：

\[
\boxed{
\text{uncertainty propagation}
}
\]

\[
\boxed{
\text{statistical inference}
}
\]

\[
\boxed{
\text{optimization}
}
\]

三者共同痛点都是：

\[
\boxed{
\text{需要大量 repeated model evaluations}
}
\]

所以 multi-fidelity 的本质并不是某个特定 GP 模型，而是如何利用便宜信息减少昂贵模型调用。

---

# 4. Paper 4：工程多保真模型综述

## 4.1 论文

M. Giselle Fernández-Godino,  
**Review of multi-fidelity models**.

正式期刊版本发表于 2023 年；早期预印本版本更早流传。

参考：

- https://doi.org/10.3934/acse.2023015
- https://arxiv.org/abs/1609.07196

## 4.2 这篇和 Paper 3 的侧重点不同

Paper 3 更关注：

\[
\boxed{
\text{什么时候用哪个模型，以及 outer-loop model management}
}
\]

Paper 4 更关注：

\[
\boxed{
\text{低保真和高保真到底怎样在数学上连接起来}
}
\]

所以它更偏 engineering multi-fidelity modeling。

---

## 4.3 典型修正方式

### 加法修正

\[
\boxed{
\hat y_H(x)=y_L(x)+\delta(x)
}
\]

其中

\[
\delta(x)\approx y_H(x)-y_L(x).
\]

### 乘法修正

\[
\boxed{
\hat y_H(x)=\rho(x)y_L(x)
}
\]

### 综合修正

\[
\boxed{
\hat y_H(x)=\rho(x)y_L(x)+\delta(x)
}
\]

Kennedy–O’Hagan 可以看成这一类的典型形式。

### Space mapping

有时 high/low 的主要差别更像是“最优位置偏了”，于是先在输入空间学映射：

\[
M:x_H\rightarrow x_L,
\]

然后用

\[
y_L(M(x))
\]

逼近 \(y_H(x)\)。

---

## 4.4 MFSM 与 MFHM

论文还可以从另一角度区分：

### Multi-Fidelity Surrogate Model

把多个 fidelity 的数据融合成统一 surrogate：

\[
\{D_L,D_H\}\rightarrow\hat y_H(x).
\]

### Multi-Fidelity Hierarchical Model

未必建立统一 surrogate，而是按层级使用模型：

\[
\text{先 low}
\rightarrow
\text{根据结果判断}
\rightarrow
\text{必要时 high}.
\]

这种思路和 filtering / model management 很接近。

---

## 4.5 工程 low fidelity 的来源

常见来源包括：

- 网格变粗；
- 物理简化；
- 几何简化；
- 线性化；
- reduced-order model；
- partial convergence；
- 数据代理模型。

因此这篇综述已经把“部分收敛求解器”明确纳入 multi-fidelity 来源。

---

# 5. 前四篇的共同观察：经典文献主要还是离散模型/离散层级

把 Paper 1–4 放在一起，主流设定通常是：

\[
\boxed{
z\in\{1,\ldots,L\}
}
\]

每个 \(z\) 对应：

- 一套代码；
- 一个网格层级；
- 一个简化物理模型；
- 一个 ROM；
- 一个数据模型。

例如：

\[
\{\text{coarse},\text{fine}\},
\]

或者：

\[
\{\text{Tadpole},\text{VSaero}\}.
\]

这些 level 往往是离散的，而且不同模型之间一般不能续算。

这和“同一个 solver 已经算了多少”的问题有本质区别。

---

# 6. 三种 fidelity 概念必须分开

## 6.1 模型身份型 fidelity

例如：

\[
z\in\{\text{Tadpole},\text{VSaero}\}.
\]

它是离散类别。

运行 Tadpole 后不能把已有状态直接续成 VSaero。

---

## 6.2 分辨率/精度参数型 fidelity

例如：

\[
h=\text{mesh size}
\]

或者 solver tolerance \(\epsilon\)。

底层参数可以很细地变化，甚至数学上可以连续参数化，但很多经典算法仍然只取几个固定档位：

\[
h\in\{h_1,h_2,h_3\}.
\]

---

## 6.3 可续算 solver-progress fidelity

例如：

\[
t=n_{\rm iter}.
\]

同一个 \(x\) 可以从

\[
t=100
\]

继续到

\[
t=150.
\]

增量成本更像

\[
C(150)-C(100),
\]

而不是重新付一次完整评价成本。

这个结构和独立的 low/high 两套模型非常不同。

---

# 7. Paper 5：用部分收敛 CFD 优化翼型和机翼

## 7.1 论文

Forrester, Bressloff and Keane,  
**Optimization using surrogate models and partially converged computational fluid dynamics simulations**,  
Proceedings of the Royal Society A, 2006.

参考：

- https://eprints.soton.ac.uk/23873/

## 7.2 它想解决什么问题

传统 CFD 优化中，每一个候选设计都要完整跑到收敛。

但很多设计在很早的时候就已经显示出：

- 明显很差；
- 或者大致落在哪个区域；
- 或者设计空间的整体趋势已经出现。

于是作者问：

\[
\boxed{
\text{能不能用 partially converged CFD 来减少完整 CFD 的总成本？}
}
\]

---

## 7.3 fidelity 来源

这里的 fidelity 是：

\[
z=n_{\rm iter}.
\]

同一个 CFD solver、同一个网格、同一套方程，只是运行到不同内部迭代次数。

二维案例里完整计算的参考水平讨论为约

\[
1500\text{ iterations}.
\]

底层 solver progress 是：

\[
1,2,3,\ldots,1500.
\]

所以它具有：

- 有序性；
- 很多层级；
- 可续算性。

---

## 7.4 为什么说 2006 仍然“像离散 low/high”

关键不是底层 solver 有多少个 iteration，而是最终 surrogate 怎么建。

2006 会研究：

\[
\text{DoE 点数}
\times
\text{每点迭代数}
\]

之间的 trade-off，并找一个“partial CFD 已经足够有用”的迭代水平。

讨论中使用过一个具体 partial 设置：

\[
n_{\rm partial}=575
\]

作为说明。

然后大量设计统一取该 partial level：

\[
f(x_1,575),\ldots,f(x_N,575).
\]

在这个固定截面上建立 Kriging：

\[
\boxed{
\hat f_{\rm partial}(x)
}
\]

注意：

\[
575
\]

不是这个 GP 的输入变量。

GP 输入仍然只有

\[
x.
\]

---

## 7.5 两个主要 Kriging

第一个：

\[
\hat f_{\rm partial}(x)
\]

学习部分收敛 CFD 在整个设计空间中的曲面。

第二个：

\[
\hat\delta(x)
\]

学习

\[
\delta(x)
=
f_{\rm full}(x)-f_{\rm partial}(x).
\]

最后融合为

\[
\boxed{
\hat f_{\rm full}(x)
=
\hat f_{\rm partial}(x)+\hat\delta(x)
}
\]

这就是 evofusion 的核心思想之一。

---

## 7.6 它的优化循环

先用 partial CFD 建 global surrogate：

\[
\hat f_{\rm partial}(x),
\]

然后找有希望的区域。

再在 Expected Improvement 较高的位置做完整 CFD，获得新的 full data，继续修正 fused surrogate。

所以它同时在减少两个浪费：

1. 大量设计不再一开始就全部跑满；
2. 完整 CFD 更集中在 promising region。

---

## 7.7 2006 到底是不是“continuous fidelity”

严格说，真实 CFD iteration 是整数：

\[
n\in\mathbb{N}.
\]

所以实际观测本身是离散的。

更关键的是，2006 最终模型并没有直接建立

\[
GP(x,t).
\]

它主要建立：

\[
GP(x)
\]

和

\[
GP_\delta(x).
\]

因此更精确的说法是：

\[
\boxed{
\text{2006 的底层 solver progress 有很多有序、可续算的 fidelity level，
但最终 surrogate 没有把 progress 当成连续输入维度。}
}
\]

---

# 8. Paper 6：预测 S 形管道求解器的最终收敛结果

## 8.1 论文

Victor Picheny and David Ginsbourger,  
**A Nonstationary Space-Time Gaussian Process Model for Partially Converged Simulations**,  
SIAM/ASA Journal on Uncertainty Quantification, 2013.

参考：

- https://epubs.siam.org/doi/10.1137/120882834

## 8.2 核心问题

同一个 solver 在不同设计 \(x\) 上都有一条 convergence trajectory：

\[
y(x,t).
\]

真正想知道的是最终收敛值：

\[
\boxed{
F(x)=\lim_{t\rightarrow\infty}y(x,t)
}
\]

问题是：

\[
\boxed{
\text{能不能利用不同设计在不同进度上的中途数据，
直接预测 }F(x)?
}
\]

---

## 8.3 S 形管道案例

系统是二维稳态不可压湍流 S 形管道，使用 OpenFOAM simpleFoam。

输入：

\[
x=(x_1,\ldots,x_7).
\]

输出是出口流速的标准差 \(f_{\rm SD}\)，越小表示出口速度越均匀。

这里的 \(t\) 是计算进度 / solver iteration，不是物理时间。

案例中轨迹记录到约 500 步，500 步结果作为实验中的最终参考。

---

## 8.4 2013 的关键变化

它不再先选一个固定 partial level。

它直接建立：

\[
\boxed{
Y(x,t)
}
\]

也就是说 GP 输入变成

\[
\boxed{
(x,t)
}
\]

因此以下所有数据都可以进入同一个模型：

\[
(x_1,50,y_{1,50}),
\]

\[
(x_1,100,y_{1,100}),
\]

\[
(x_2,73,y_{2,73}),
\]

\[
(x_3,420,y_{3,420}).
\]

不同设计可以拥有完全不同的计算深度。

---

## 8.5 核心分解

论文写成：

\[
\boxed{
Y(x,t)=F(x)+G(x,t)
}
\]

其中：

\[
F(x)
\]

是最终收敛曲面，

\[
G(x,t)
\]

是当前还没有消失的 convergence error。

要求：

\[
\boxed{
G(x,t)\rightarrow0,
\qquad
t\rightarrow\infty.
}
\]

所以模型不仅“多塞了一个 \(t\)”，还把求解器最终收敛这一结构明确编码进 covariance。

---

## 8.6 为什么要 nonstationary GP

收敛过程前期和后期统计行为不同：

- 早期误差大，变化快；
- 后期误差小，变化慢；
- \(t\) 越大，误差方差应该越小。

因此论文设计了随 \(t\) 变化的 covariance，让

\[
\operatorname{Var}[G(x,t)]
\rightarrow0.
\]

这就是标题里的 nonstationary space-time GP。

---

## 8.7 它还是 GP，不是神经网络

本质仍然是 Gaussian Process / Kriging。

区别主要在于它学习的是：

\[
\boxed{
(x,t)\mapsto y
}
\]

而不是固定 \(t\) 后的

\[
x\mapsto y.
\]

可以把它理解为一种“对整条 convergence surface 做概率化核插值”。

---

## 8.8 固定总预算时的结果

论文用固定 iteration 总预算比较：

- 少量设计全部跑满；
- 更多设计只跑部分；
- 用 space-time GP 从 partial trajectories 推最终值。

讨论中过的实验之一：

\[
18{,}500\text{ solver steps}
\]

等价于

\[
37\times500
\]

次完整运行。

在相同预算下，space-time GP 的预测误差优于只用 37 个完整收敛点的普通 Kriging。

论文还研究了：

\[
N\times t=B
\]

这种“设计点数量 vs 每个设计计算深度”的离线预算 trade-off。

---

## 8.9 它还没有做到什么

它主要解决的是预测模型：

\[
(x,t,\text{partial observations})
\rightarrow
\hat F(x).
\]

它没有完整实现这样的在线策略：

\[
\boxed{
\text{下一份计算资源到底给哪个 }x，
\text{以及再推进多少 }\Delta t？
}
\]

所以它有 continuous-progress statistical model，但还没有完整的 sequential resource allocation policy。

---

# 9. 2006 与 2013 到底区别在哪里

这是本次讨论里最容易混淆的一点。

## 9.1 相同点

两篇底层都使用：

- 同一个 CFD solver；
- solver iteration 作为精度/进度来源；
- Kriging / Gaussian Process；
- 利用部分收敛信息降低完整 CFD 成本；
- 不使用大型神经网络。

所以如果只看“机器学习算法是什么”，两篇确实很像：

\[
\boxed{
\text{都是 GP/Kriging}
}
\]

都可以粗略理解成带 covariance 的概率化插值。

---

## 9.2 真正区别：GP 学的对象不同

### 2006

先固定一个 partial level

\[
t=t_0.
\]

然后建

\[
\boxed{
f_{t_0}(x)
}
\]

即

\[
x\rightarrow\hat f_{\rm partial}(x).
\]

再建

\[
x\rightarrow\hat\delta(x).
\]

所以模型主要生活在 \(x\)-空间里。

---

### 2013

直接建

\[
\boxed{
Y(x,t)
}
\]

所以输入是

\[
(x,t).
\]

它把整条 convergence history 统一建模。

---

## 9.3 最直观的比喻

\[
\boxed{
\text{2006：从每条收敛轨迹上截一张照片}
}
\]

\[
\boxed{
\text{2013：直接拟合整部收敛电影}
}
\]

2006 先决定“300 步已经够有用了”，然后很多设计都取 300 步数据。

2013 则允许：

\[
x_1\text{ 到100步},
\]

\[
x_2\text{ 到73步},
\]

\[
x_3\text{ 到420步},
\]

这些不整齐的数据全部统一进入同一个 GP。

---

## 9.4 为什么 2013 在不整齐计算深度下更自然

假设数据是：

\[
x_1:\quad t=100,200,300,400
\]

\[
x_2:\quad t=100,200
\]

\[
x_3:\quad t=100,500
\]

\[
x_4:\quad t=350.
\]

2006 如果固定 partial level \(t_0=300\)，就希望所有设计都有接近 300 的数据。

2013 完全不要求整齐：

\[
(x_i,t_i,y_i)
\]

只要有观测就可以进入联合模型。

因此 2013 真正新增的能力是：

\[
\boxed{
\text{任意深度的中途计算都能被统一利用}
}
\]

---

## 9.5 “连续”到底是什么意思

真实 CFD 中，两篇的 solver iteration 都是整数：

\[
t=1,2,3,\ldots
\]

所以实际观测点仍然离散。

2013 所谓连续 fidelity，核心是统计模型把

\[
t
\]

连续参数化：

\[
t\in\mathbb R_+.
\]

于是可以讨论

\[
Y(x,137.5)
\]

这种统计模型上的值，并最终取

\[
t\rightarrow\infty.
\]

所以 continuous fidelity 主要指：

\[
\boxed{
t\text{ 被作为连续坐标进入 surrogate}
}
\]

不是说 CFD 真的运行“半次 iteration”。

---

## 9.6 如果所有数据都很整齐，两篇会非常像

假设所有设计都统一做到：

\[
300\text{ iterations},
\]

少数再做到 1500。

那 2006 的

\[
GP_{300}(x)+GP_\delta(x)
\]

已经可能很好。

这种场景下，2013 的优势确实不会特别大。

它真正有价值的场景是：

\[
\boxed{
\text{不同 }x\text{ 被计算到不同深度}
}
\]

并且希望把所有中途数据都保留下来利用。

---

# 10. Paper 7：按排名翻转风险决定 CFD 要算到哪一步

## 10.1 论文

Jürgen Branke, Md. Asafuddoula, Kalyan Shankar Bhattacharjee, Tapabrata Ray,  
**Efficient Use of Partially Converged Simulations in Evolutionary Optimization**,  
IEEE Transactions on Evolutionary Computation, online 2016 / volume publication 2017.

参考：

- https://wrap.warwick.ac.uk/id/eprint/79047/

## 10.2 它的核心问题已经变了

2013 主要问：

\[
\boxed{
\text{最终收敛值是多少？}
}
\]

Paper 7 问：

\[
\boxed{
\text{为了做选择决策，这个 candidate 到底还需不需要继续算？}
}
\]

这是一个重要转折。

---

## 10.3 Toysub 潜艇问题

输入是 8 个连续几何参数：

\[
x=(Z_C,Z_V,Z_L,Z_B,d_t,l_t,n_n,l_n).
\]

最终目标是在体积约束下最小化最高 fidelity 的阻力：

\[
\boxed{
x^*
=
\arg\min_x D_{100}(x)
}
\]

原始 FLUENT 数据有 284 个潜艇设计。

每个设计记录 6 个检查点：

\[
\boxed{
5,\ 10,\ 25,\ 50,\ 75,\ 100
}
\]

所以算法允许的 fidelity 是

\[
z\in\{5,10,25,50,75,100\}.
\]

这里 fidelity 明确是离散检查点。

---

## 10.4 为什么它不追求精确预测最终阻力

进化算法的 selection 通常主要依赖排名。

如果 96 个 parent+offspring 最后保留 48 个，那么真正关键的是：

\[
\boxed{
\text{这个 candidate 最终会落在前48还是后48？}
}
\]

没有必要把每个 candidate 的最终阻力预测到很多小数位。

---

## 10.5 Probability of reversal

假设在当前 fidelity \(i\)，两个 candidate 的分数差为

\[
\Delta f_i
=
|f_i(x)-f_i(y)|.
\]

算法学习：

\[
\boxed{
P(
\text{当前排名与最终排名不一致}
\mid
\Delta f_i
)
}
\]

即排名翻转概率。

如果两个候选当前差得很远，历史上最终排名很少翻转，那么可以早停。

如果差得很小、翻转风险高，则继续提升 fidelity。

---

## 10.6 核心学习模型不是 GP，而是 logistic regression

这个地方必须和 Paper 6 分开。

Paper 7 的在线决策核心是：

\[
\boxed{
\text{logistic regression}
}
\]

对每个 fidelity level 建一个 rank-reversal model。

输入：

\[
\Delta f_i.
\]

输出：

\[
P_{\rm reversal}.
\]

所以它主要学习的是“当前分数差对应多大最终误判风险”，而不是学习最终阻力具体是多少。

---

## 10.7 那论文里的 6 个 GP 是干什么的

论文确实用了 6 个 GP，但用途不同。

因为真实 FLUENT 太慢，为了大量重复跑 evolutionary optimization benchmark，作者先用 284 个真实 CFD 数据分别拟合：

\[
GP_5(x),
GP_{10}(x),
GP_{25}(x),
GP_{50}(x),
GP_{75}(x),
GP_{100}(x).
\]

这些 GP 的任务是快速模拟不同 fidelity 的 CFD benchmark。

在线“要不要继续算”的核心模型仍是 logistic regression。

所以要分开：

\[
\boxed{
\text{GP：替代真实 FLUENT，构造可重复 benchmark}
}
\]

\[
\boxed{
\text{logistic regression：在线决定是否升级 fidelity}
}
\]

---

## 10.8 它如何做 selection

假设种群保留 \(\mu=48\) 个。

当前 fidelity 下，第 48 名给出一个生死阈值 \(T_i\)。

对 candidate \(x\)，看

\[
|f_i(x)-T_i|.
\]

如果 candidate 明显比阈值好，而且排名翻转概率很低：

\[
P_{\rm reversal}<\delta,
\]

可以直接保留，不再继续算。

如果明显差，也可以直接淘汰。

只有靠近生死线、误判风险较大的 candidate 才继续：

\[
5\rightarrow10\rightarrow25\rightarrow50\rightarrow75\rightarrow100.
\]

---

## 10.9 它真正利用了可续算性

如果一个 candidate 已经算到 10 步，现在决定提升到 25 步，只支付增量：

\[
25-10=15.
\]

所以这里的成本不是每次从头算。

这是与独立 low/high 模型很重要的区别。

---

## 10.10 它已经实现逐 candidate 的动态计算深度

不同 candidate 可以：

\[
x_1\rightarrow5\text{步就停},
\]

\[
x_2\rightarrow10\text{步},
\]

\[
x_3\rightarrow50\text{步},
\]

\[
x_4\rightarrow100\text{步}.
\]

因此它已经具备：

\[
\boxed{
\text{根据当前中途信息，为不同 }x\text{ 分配不同计算深度}
}
\]

这是 Paper 5 和 Paper 6 之后很重要的一步。

---

## 10.11 但它仍然是离散检查点

虽然底层 FLUENT 可以继续迭代，但算法只允许：

\[
\boxed{
z\in\{5,10,25,50,75,100\}.
}
\]

一次升级基本是从当前 level 走到下一个 level。

它没有把

\[
\Delta t
\]

作为连续、自由选择的动作变量。

---

# 11. Paper 5、6、7 连起来看

这三篇形成了一条非常清楚的演化路线。

## Paper 5：Forrester 2006

核心发现：

\[
\boxed{
\text{部分收敛 CFD 已经可以提供有用的设计空间信息}
}
\]

方法：

\[
\text{固定 partial level}
+
\text{full correction}
+
\text{EI optimization}.
\]

核心对象：

\[
GP(x).
\]

---

## Paper 6：Picheny & Ginsbourger 2013

核心推进：

\[
\boxed{
\text{不要把 convergence history 压成一个固定 partial level}
}
\]

直接建：

\[
\boxed{
Y(x,t)=F(x)+G(x,t)
}
\]

核心对象：

\[
GP(x,t).
\]

目标主要是：

\[
\boxed{
\text{预测最终收敛值}
}
\]

---

## Paper 7：Branke et al. 2016/2017

核心推进：

\[
\boxed{
\text{不一定需要预测最终值；
只要知道当前 selection 决策是否可靠}
}
\]

学习：

\[
\boxed{
P(\text{rank reversal})
}
\]

然后动态决定：

\[
\boxed{
\text{停还是继续到下一 fidelity}
}
\]

因此不同 candidate 可以计算到不同深度。

---

# 12. 三篇的核心对比表

| 维度 | Forrester 2006 | Picheny 2013 | Branke 2016/2017 |
|---|---|---|---|
| 底层系统 | 同一 CFD | 同一 CFD | 同一 CFD / CFD benchmark |
| fidelity 来源 | iteration 数 | computational progress | iteration 检查点 |
| 实际 fidelity | 整数 iteration | 整数观测 | 6 个离散检查点 |
| 是否可续算 | 是 | 是 | 是 |
| surrogate 是否把 \(t\) 当输入 | 否 | 是 | 否 |
| GP 结构 | \(GP(x)\) + discrepancy | \(GP(x,t)\) | 6 个 GP 主要用于 benchmark |
| 在线核心模型 | Kriging + EI | space-time GP | logistic rank-reversal model |
| 主要任务 | 优化 | 最终值预测 | 进化优化中的动态早停/升级 |
| 每个 candidate 可不同深度 | 最终方法较弱 | 数据模型允许 | 明确允许 |
| fidelity 是否在线动态决定 | 较有限 | 本文不是主要目标 | 是，但只能升级离散级别 |
| 是否直接选择连续 \(\Delta t\) | 否 | 否 | 否 |

---

# 13. “它们不都是 GP 插值吗？”——最终结论

对于 Paper 5 和 Paper 6，这个直觉是对的。

两篇都没有大型神经网络，核心工具都是：

\[
\boxed{
\text{Gaussian Process / Kriging}
}
\]

都可以理解为利用 covariance/kernel 对未知函数做概率化平滑插值。

真正区别不在“是不是 GP”，而在：

\[
\boxed{
\text{GP 到底把什么当输入、在拟合什么对象}
}
\]

2006：

\[
x
\rightarrow
f_{\rm partial}(x)
\]

以及

\[
x
\rightarrow
\delta(x).
\]

2013：

\[
(x,t)
\rightarrow
Y(x,t).
\]

所以 2013 的新东西是把整个 convergence trajectory 保留下来，并给 \(t\) 方向设计专门的非平稳 covariance，而不是换了一种完全不同的机器学习算法。

---

# 14. 离散 fidelity、连续 fidelity、可续算 progress 三件事不能混在一起

## 14.1 离散模型层级

\[
z\in\{L,H\}.
\]

例如 Tadpole / VSaero。

这是 model identity。

---

## 14.2 连续参数化 fidelity

统计模型直接使用：

\[
t\in\mathbb R_+.
\]

例如 2013：

\[
Y(x,t).
\]

即使真实 solver iteration 是整数，surrogate 仍然把 \(t\) 当连续坐标。

---

## 14.3 可续算 progress

状态已经到：

\[
(x,t_0),
\]

下一步可以继续：

\[
(x,t_0+\Delta t).
\]

新增成本是增量成本，而不是从头再跑。

Paper 5–7 都具备这种底层结构，但利用方式不同。

---

# 15. 从这组文献看，真正还可以继续往前走的问题

如果把这组工作统一成一个更一般的 sequential computation problem，状态可以写成：

\[
\{(x_i,t_i,y_i)\}_{i=1}^N.
\]

每轮不只是问：

\[
\text{下一个 }x\text{ 是什么？}
\]

还要问：

\[
\boxed{
\text{开一个新的 }x
\quad\text{还是继续已有 }x_i？
}
\]

并进一步问：

\[
\boxed{
\text{如果继续，推进多少 }\Delta t？
}
\]

例如：

\[
x_A:\ 50\rightarrow80,
\]

或者

\[
x_B:\ 300\rightarrow420,
\]

或者新开

\[
x_C:\ 0\rightarrow60.
\]

这和传统 multi-fidelity 的

\[
z\in\{L,H\}
\]

已经是不同结构。

Paper 7 已经开始做“哪个 candidate 还值得继续算”，但 fidelity 仍是预先固定的离散检查点：

\[
5,10,25,50,75,100.
\]

Paper 6 已经把 \(t\) 连续参数化进 statistical model，但没有建立完整的在线资源分配决策。

因此这组文献形成的一个很清楚的空隙是：

\[
\boxed{
\text{连续/细粒度 solver progress 建模}
+
\text{暂停/续算}
+
\text{在线决定 }x
+
\text{在线决定 }\Delta t
+
\text{统一计算预算}
}
\]

也就是说，重点不只是“预测最终结果”，还包括：

\[
\boxed{
\text{下一份计算资源应该投到哪里、投多少}
}
\]

---

# 16. 七篇文献的总体逻辑图

\[
\boxed{
\text{Kennedy--O'Hagan 2000}
}
\]

多个离散代码层级：

\[
f_H=\rho f_L+\delta
\]

解决 high-fidelity prediction。

↓

\[
\boxed{
\text{Forrester 2007}
}
\]

把 multi-fidelity GP 放进 EI：

\[
\text{prediction}
\rightarrow
\text{optimization}
\]

↓

\[
\boxed{
\text{Peherstorfer et al. 2018}
}
\]

把整个领域整理成 adaptation / fusion / filtering，并覆盖 UQ、inference、optimization。

↓

\[
\boxed{
\text{Fernández-Godino review}
}
\]

更系统整理工程里的 additive / multiplicative / comprehensive correction / space mapping。

↓

\[
\boxed{
\text{Forrester 2006}
}
\]

发现：

\[
\text{同一个 CFD 的 partial convergence}
\]

本身就能当 low fidelity。

↓

\[
\boxed{
\text{Picheny \& Ginsbourger 2013}
}
\]

不再固定一个 partial level，而是：

\[
(x,t)\rightarrow Y(x,t)
\]

统一建模整条 convergence trajectory。

↓

\[
\boxed{
\text{Branke et al. 2016/2017}
}
\]

不再执着于最终数值预测，而是根据：

\[
P(\text{rank reversal})
\]

动态决定 candidate 是否继续提升 fidelity。

---

# 17. 最重要的几条结论

1. 经典 multi-fidelity 文献的主流设定长期是少数几个离散模型或 fidelity level。

2. Kennedy–O’Hagan 最经典的形式是

   \[
   f_H=\rho f_L+\delta,
   \]

   其中 low function 和 discrepancy 都用 GP；它不是深度学习模型。

3. Forrester 2007 把这种多保真 surrogate 真正用于 EI optimization，但主要主动选择的是 \(x\)，不是每轮自由选择 fidelity。

4. Peherstorfer 2018 说明 multi-fidelity 的范围远大于 co-kriging，核心还包括 adaptation、fusion 和 filtering。

5. Fernández-Godino 的工程综述更强调 low/high 之间怎样做 additive、multiplicative、comprehensive correction 或 space mapping。

6. Forrester 2006 是一个关键转折：fidelity 不再必须来自不同 solver，可以来自同一个 CFD 的不同收敛进度。

7. 2006 的底层 progress 很细、可续算，但最终 surrogate 主要仍是固定 partial level 上的 \(GP(x)\)，不是 \(GP(x,t)\)。

8. Picheny 2013 的关键贡献是把 \(t\) 直接作为联合输入，建立

   \[
   Y(x,t)=F(x)+G(x,t)
   \]

   并让 convergence error 随 \(t\rightarrow\infty\) 消失。

9. 2006 和 2013 都主要使用 GP/Kriging；它们的区别是建模对象和可利用的数据结构，不是机器学习工具换代。

10. Branke 2016/2017 进一步转向“决策需要什么信息”：只学习排名是否会翻转，并据此动态决定某个 candidate 是否继续算。

11. Branke 已经实现了逐 candidate 的动态计算深度，但 fidelity 仍是固定离散检查点。

12. 更一般的连续 solver-progress 问题可以进一步把动作写成

   \[
   (x_{\rm next},\Delta t_{\rm next}),
   \]

   即同时决定“给谁算”和“再算多少”。

---

# 18. 参考文献与链接

1. Kennedy, M. C., O’Hagan, A.  
   **Predicting the output from a complex computer code when fast approximations are available.**  
   Biometrika, 2000.  
   https://academic.oup.com/biomet/article/87/1/1/221217  
   https://www2.stat.duke.edu/courses/Spring14/sta961.01/ref/KennOHag2000.pdf

2. Forrester, A. I. J., Sóbester, A., Keane, A. J.  
   **Multi-fidelity optimization via surrogate modelling.**  
   Proceedings of the Royal Society A, 2007.  
   https://doi.org/10.1098/rspa.2007.1900  
   https://eprints.soton.ac.uk/64698/

3. Peherstorfer, B., Willcox, K., Gunzburger, M.  
   **Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization.**  
   SIAM Review, 2018.  
   https://epubs.siam.org/doi/10.1137/16M1082469  
   https://arxiv.org/abs/1806.10761

4. Fernández-Godino, M. G.  
   **Review of multi-fidelity models.**  
   https://doi.org/10.3934/acse.2023015  
   https://arxiv.org/abs/1609.07196

5. Forrester, A. I. J., Bressloff, N. W., Keane, A. J.  
   **Optimization using surrogate models and partially converged computational fluid dynamics simulations.**  
   Proceedings of the Royal Society A, 2006.  
   https://eprints.soton.ac.uk/23873/

6. Picheny, V., Ginsbourger, D.  
   **A Nonstationary Space-Time Gaussian Process Model for Partially Converged Simulations.**  
   SIAM/ASA Journal on Uncertainty Quantification, 2013.  
   https://epubs.siam.org/doi/10.1137/120882834

7. Branke, J., Asafuddoula, M., Bhattacharjee, K. S., Ray, T.  
   **Efficient Use of Partially Converged Simulations in Evolutionary Optimization.**  
   IEEE Transactions on Evolutionary Computation, online 2016 / volume publication 2017.  
   https://wrap.warwick.ac.uk/id/eprint/79047/

---

## 19. 一句话总括

这七篇从“多个离散模型之间做融合”一路走到“同一个求解器的中途计算也能当作信息，并且可以根据中途结果决定是否继续投入计算”。最值得继续追踪的问题，是把 solver progress 进一步视为可续算、细粒度甚至连续的资源变量，并在统一预算下动态决定：

\[
\boxed{
\text{下一步算哪个 }x
\quad+\quad
\text{推进多少计算进度}
}
\]

而不是预先只给出几个固定 fidelity 档位。
