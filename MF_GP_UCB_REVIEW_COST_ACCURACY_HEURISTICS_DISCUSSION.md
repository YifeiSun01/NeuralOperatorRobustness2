# MF-GP-UCB、Multi-Fidelity Bayesian Optimization 与 Active Search 方法论讨论记录

> 整理日期：2026-10-05  
> 重点论文：Kandasamy et al. (2016), *Gaussian Process Bandit Optimisation with Multi-fidelity Evaluations*  
> 会议：NIPS 2016（现称 NeurIPS）  
> 目的：完整记录本次关于 MF-GP-UCB 的任务定义、fidelity / cost / accuracy 建模、synthetic 与真实实验、reviewer 质疑、实际实现，以及由此延伸出的对 Bayesian Optimization / Active Search 方法体系性的讨论。

---

# 1. 先把这篇论文放到 36 篇文献中的位置

之前整理的 36 篇 master list 中，Kandasamy et al. (2016) 是其中明确属于 Multi-fidelity Bayesian Optimization / GP bandit 的代表论文。

论文标题：

**Kirthevasan Kandasamy, Gautam Dasarathy, Jeff Schneider, Barnabás Póczos (2016), _Gaussian Process Bandit Optimisation with Multi-fidelity Evaluations_.**

正式论文：

- https://proceedings.neurips.cc/paper/2016/hash/605ff764c617d3cd28dbbdd72be8f9a2-Abstract.html
- PDF: https://proceedings.neurips.cc/paper_files/paper/2016/file/605ff764c617d3cd28dbbdd72be8f9a2-Paper.pdf

公开 reviews：

- https://proceedings.neurips.cc/paper_files/paper/2016/file/605ff764c617d3cd28dbbdd72be8f9a2-Reviews.html

作者公开代码：

- https://github.com/kirthevasank/mf-gp-ucb

这篇论文的基本任务不是专门的 HPO，而是一般的昂贵黑箱优化：

\[
x^\star=\arg\max_{x\in\mathcal X} f^{(M)}(x).
\]

其中 \(x\) 是 design / configuration / physical parameter，\(f^{(M)}\) 是最高 fidelity、最终真正关心的目标函数。

它加入的特殊情境是：

\[
f^{(1)}(x),f^{(2)}(x),\ldots,f^{(M)}(x)
\]

这几个不同 fidelity 的函数同时存在，低 fidelity 更便宜，但与最高 fidelity 有偏差。

所以这篇论文真正研究的是：

> 在最高 fidelity 很贵、同时存在若干便宜近似评价的情况下，如何选择下一个 design \(x\)，以及选择哪个 fidelity \(m\)，从而以更少成本找到最高 fidelity 的最优点。

---

# 2. Design space 和 fidelity space 必须分开

这篇论文中有两个完全不同的变量。

## 2.1 Design variable

\[
x\in\mathcal X\subset\mathbb R^d.
\]

这里的 \(d\) 是 design space 的维度。

例如：

- SVM：\(d=2\)，两个 hyperparameter。
- SALSA：\(d=6\)，六个 hyperparameter。
- Viola–Jones：\(d=22\)，22 个 classifier thresholds。
- Supernova：\(d=3\)，三个宇宙学参数。
- Borehole：\(d=8\)，八个物理输入。
- Hartmann-6D：\(d=6\)。

所以：

\[
d=\text{design 的变量个数}.
\]

## 2.2 Fidelity variable

论文写：

\[
m\in\{1,\ldots,M\}.
\]

这里的 \(M\) 是 fidelity 的档位数。

例如：

- SVM：\(M=2\)。
- SALSA：\(M=3\)。
- Viola–Jones：\(M=2\)。
- Supernova：\(M=3\)。
- Hartmann-3D：\(M=3\)。
- Hartmann-6D：\(M=4\)。

所以：

\[
M=\text{fidelity 档位数}.
\]

这一篇 2016 论文里的 fidelity 全都是：

\[
\boxed{\text{有限、离散、有序的 fidelity}}
\]

而不是连续 fidelity。

后来的 Kandasamy et al. (2017) BOCA 才把 fidelity 推广到连续空间。

---

# 3. 这篇论文不是专门的神经网络 HPO

这篇论文的方法本身是一般的 multi-fidelity black-box optimization。

但是它的真实实验里包含三个 HPO 任务和一个 astrophysics 参数估计任务。

## 3.1 SVM hyperparameter tuning

Design：

\[
x=(\text{kernel bandwidth}, C).
\]

所以：

\[
d=2.
\]

fidelity 不是 epoch，而是训练数据量：

\[
m=1:\ 500\text{ samples},
\]

\[
m=2:\ 2000\text{ samples}.
\]

所以：

\[
f^{(1)}(x)=\text{500 个训练样本上的 5-fold CV performance},
\]

\[
f^{(2)}(x)=\text{2000 个训练样本上的 5-fold CV performance}.
\]

最终目标是优化最高 fidelity 的 CV performance。

## 3.2 SALSA hyperparameter tuning

SALSA 是 additive-kernel regression，不是神经网络。

Design 一共有 6 个 hyperparameter：

- regularization penalty；
- kernel scale；
- 4 个 input dimensions 对应的 kernel bandwidth。

所以：

\[
d=6.
\]

fidelity 是训练数据量：

\[
2000,\quad 4000,\quad 8000.
\]

因此：

\[
M=3.
\]

依旧是 dataset size 作为 fidelity，而不是 training epoch。

## 3.3 Viola–Jones

Design 是 22 个 weak classifiers 对应的 thresholds：

\[
x=(t_1,\ldots,t_{22}),
\]

所以：

\[
d=22.
\]

fidelity 是训练图像数量：

\[
m=1:\ 300\text{ images},
\]

\[
m=2:\ 3000\text{ images}.
\]

所以：

\[
M=2.
\]

这里很容易混淆：

\[
d=22
\]

表示 22 个 optimization variables，而不是 22 个 fidelity。

## 3.4 Supernova

Design：

\[
x=(H_0,\Omega_M,\Omega_\Lambda),
\]

所以：

\[
d=3.
\]

目标是最大化与 Type Ia Supernova 数据对应的 likelihood / log-likelihood。

fidelity 是数值积分网格精度：

\[
10^2,\quad10^4,\quad10^6
\]

个网格点。

因此：

\[
M=3.
\]

这里三个宇宙学参数是连续 design variables，而 fidelity 仍然是预先规定的三个离散档位。

---

# 4. Synthetic benchmark 里的 fidelity 是怎么造出来的

Synthetic benchmark 中没有真实的粗网格、小训练集、低精度实验等物理机制。

作者先有一个最高 fidelity 函数：

\[
f^{(M)}(x),
\]

然后人为构造几个近似函数：

\[
f^{(1)}(x),f^{(2)}(x),\ldots,f^{(M-1)}(x).
\]

## 4.1 Currin

高 fidelity 是标准 Currin exponential。

低 fidelity 被定义成最高 fidelity 在几个偏移点的平均值。

作者公开代码中低 fidelity 大致是：

\[
f^{(1)}(x)
=
\frac14\sum_{j=1}^4 f^{(2)}(x+\delta_j).
\]

所以这是人为构造的相关近似函数。

公开代码直接规定：

~~~matlab
costs = (10.^(0:(numFidels-1)))';
~~~

对于 Currin：

\[
[1,10].
\]

这里的 cost 是 synthetic benchmark 的人工设定，不是这两个简单解析函数真实测出来的 CPU time ratio。

## 4.2 Borehole

高 fidelity 是标准 Borehole benchmark。

低 fidelity 通过人为修改公式中的若干系数得到。

例如公开代码里，高 fidelity 分子有：

\[
2\pi T_u(H_u-H_l),
\]

低 fidelity 改成：

\[
5T_u(H_u-H_l),
\]

分母里的常数也有所改变。

成本：

\[
[1,10].
\]

## 4.3 Hartmann

Hartmann-3D 与 Hartmann-6D 都通过修改函数内部的 \(\alpha\) 权重来构造不同 fidelity。

Hartmann-3D：

\[
d=3,\qquad M=3,
\]

成本：

\[
[1,10,100].
\]

Hartmann-6D：

\[
d=6,\qquad M=4,
\]

成本：

\[
[1,10,100,1000].
\]

所以 synthetic experiment 实际上同时人工定义了：

1. low-fidelity functions；
2. fidelity ordering；
3. fidelity costs；
4. highest fidelity target。

它主要是为了测试算法机制，而不是为了证明现实系统中 cost ratio 真的是这些数。

---

# 5. 论文的统一数学框架

最高 fidelity：

\[
f^{(M)}(x)
\]

是真正想优化的目标。

第 \(m\) 个 fidelity 的查询成本写成：

\[
\lambda^{(m)}.
\]

作者假设：

\[
\lambda^{(1)}
<
\lambda^{(2)}
<
\cdots
<
\lambda^{(M)}.
\]

对 fidelity accuracy，作者假设：

\[
\boxed{
\|f^{(M)}-f^{(m)}\|_\infty
\le
\zeta^{(m)}
}
\]

也就是：

\[
|f^{(M)}(x)-f^{(m)}(x)|
\le
\zeta^{(m)}
\qquad
\forall x.
\]

这里的 \(\zeta^{(m)}\) 不是 local prediction error，也不是一个从 \(x\) 变化的函数。

它是：

\[
\boxed{
\text{第 }m\text{ 个 fidelity 相对于最高 fidelity 的全空间 worst-case error bound}
}
\]

通常还假设：

\[
\zeta^{(1)}>
\zeta^{(2)}>
\cdots>
\zeta^{(M)}=0.
\]

---

# 6. 这篇论文真正学习了什么，哪些东西没有学习

## 6.1 它学习的是每个 fidelity 的 response surface

对于每一个 fidelity：

\[
f^{(m)}(x)
\]

作者分别建 Gaussian Process。

所以：

\[
x
\longrightarrow
f^{(m)}(x)
\]

是从观测数据学习的。

GP 给出：

\[
\mu_t^{(m)}(x),
\qquad
\sigma_t^{(m)}(x).
\]

## 6.2 它没有真正学习 cost surface

理论 cost 是：

\[
\boxed{
c(x,m)=\lambda^{(m)}
}
\]

也就是说，cost 只依赖 fidelity level，不依赖 design \(x\)。

它没有：

\[
\hat c(x,m)
\]

这样的 cost model。

例如 SVM 中，同样是 2000 个 training samples，不同 hyperparameter 组合实际训练时间完全可能不同。

理论模型不处理这种：

\[
c=c(x,m)
\]

的 design-dependent cost。

## 6.3 它没有真正学习 accuracy / error surface

真正的 local error 可以定义为：

\[
e(x,m)=|f^{(M)}(x)-f^{(m)}(x)|.
\]

一个更完整的模型可能学习：

\[
\hat e(x,m)
\]

或者：

\[
p(e\mid x,m).
\]

MF-GP-UCB 没有这么做。

它理论上只要求给定：

\[
m\longrightarrow\zeta^{(m)}.
\]

也就是每个 fidelity 一个全局最大误差上界。

所以最精确的总结是：

| 关系 | MF-GP-UCB 怎么处理 |
|---|---|
| \(x,m\to f^{(m)}(x)\) | GP 学习 |
| \(m\to cost\) | 给定 \(\lambda^{(m)}\) |
| \(x,m\to cost\) | 不建模 |
| \(m\to worst-case accuracy\) | 理论给定 \(\zeta^{(m)}\) |
| \(x,m\to local error\) | 不建模 |
| fidelity 间完整相关关系 | 不直接联合学习 |

因此可以概括为：

\[
\boxed{
\text{performance surface 学；
cost surface 不学；
accuracy/error surface 也不真正学。}
}
\]

---

# 7. MF-GP-UCB 怎么选 design

对于第 \(m\) 个 fidelity，作者定义一个对最高 fidelity 的 upper bound：

\[
\phi_t^{(m)}(x)
=
\mu_{t-1}^{(m)}(x)
+
\sqrt{\beta_t}\sigma_{t-1}^{(m)}(x)
+
\zeta^{(m)}.
\]

其中：

\[
\mu+\sqrt{\beta}\sigma
\]

是普通 GP-UCB，再加 \(\zeta^{(m)}\) 是为了覆盖 low fidelity 和 highest fidelity 之间可能存在的最大偏差。

然后取：

\[
\phi_t(x)
=
\min_m \phi_t^{(m)}(x).
\]

最后：

\[
x_t
=
\arg\max_x \phi_t(x).
\]

所以 design selection 的意义是：

> 选择一个当前仍然最可能成为最高 fidelity optimum 的位置。

---

# 8. MF-GP-UCB 怎么选 fidelity

选出 \(x_t\) 后，算法再决定 fidelity。

它从最低 fidelity 开始检查 posterior uncertainty。

如果：

\[
\sqrt{\beta_t}
\sigma_{t-1}^{(m)}(x_t)
\]

仍然大于 threshold：

\[
\gamma^{(m)},
\]

说明这个 fidelity 自己都还没有搞清楚，就先在这个便宜 fidelity 上继续 query。

如果：

\[
\sqrt{\beta_t}
\sigma_{t-1}^{(m)}(x_t)
<
\gamma^{(m)},
\]

说明继续把这个 fidelity 学得更精确的价值已经下降，于是升级到更高 fidelity。

整体直觉：

\[
\boxed{
\text{低 fidelity 大范围排除；
高 fidelity 小范围确认。}
}
\]

---

# 9. Cost 在算法里到底怎么进入

cost 不直接写成 information gain / cost 或 EI / cost。

它主要通过两种方式使用。

## 9.1 总预算

总预算：

\[
\Lambda.
\]

如果依次查询：

\[
m_1,m_2,\ldots,m_T,
\]

则：

\[
\sum_{t=1}^T\lambda^{(m_t)}
\le
\Lambda.
\]

## 9.2 Fidelity switching heuristic

公开代码中，如果算法在当前 fidelity 停留的 query 数达到大约：

\[
\frac{\lambda^{(m+1)}}{\lambda^{(m)}},
\]

仍然没有升级，就增大 \(\gamma^{(m)}\)。

代码实际使用的放大系数是 5：

~~~matlab
if gammaThreshExcCounter >= ...
          mfFunc.costs(nextFidel+1)/mfFunc.costs(nextFidel)
  params.gammas(nextFidel) = ...
    params.gammas(nextFidel) * GAMMA_INC_COEFF;
end
~~~

其中：

~~~matlab
GAMMA_INC_COEFF = 5;
~~~

所以直觉是：

> 如果已经在低 fidelity 上花掉了差不多一次更高 fidelity 的等价预算，却仍然没有升级，就提高 threshold，让升级更容易发生。

---

# 10. 理论上的 \(\zeta\) 和实际实现中的 \(\zeta\)

## 10.1 理论

Algorithm 1 假设：

\[
\zeta^{(m)}
\]

作为 problem description 的一部分已经知道。

作者自己在论文 Section 5 明确承认现实里这种情况很难成立。

所以理论要求和实际条件之间存在明显落差。

## 10.2 实际实现

作者在实践中只维护一个基础尺度 \(\zeta\)，然后构造：

\[
\zeta^{(1)}=(M-1)\zeta,
\]

\[
\zeta^{(2)}=(M-2)\zeta,
\]

一直到：

\[
\zeta^{(M-1)}=\zeta.
\]

等价写成：

\[
\zeta^{(m)}=(M-m)\zeta.
\]

这样做相当于默认相邻 fidelity 的允许差异由同一个 \(\zeta\) 控制。

---

# 11. 真实实现里 \(\zeta\) 到底怎么检查

假设算法这一步选择了较高 fidelity：

\[
m>1.
\]

首先算出：

\[
f^{(m)}(x_t).
\]

然后用低一级 fidelity 的 GP posterior mean：

\[
\mu^{(m-1)}(x_t)
\]

进行廉价预检查。

代码里：

~~~matlab
diffEst = abs(nextPtVal - funcHs{nextFidel-1}(nextPt));
~~~

也就是看：

\[
\left|
f^{(m)}(x_t)
-
\mu^{(m-1)}(x_t)
\right|.
\]

如果这个差接近或超过当前允许的相邻 fidelity gap，就真正再做一次低一级 fidelity：

\[
f^{(m-1)}(x_t).
\]

公开代码：

~~~matlab
lfVal = mfFunc.evalAtFidel(nextFidel-1, nextPt);
trueDiff = nextPtVal - lfVal;
~~~

然后检查 observed discrepancy。

如果发现当前 bound 太小，就增大它：

~~~matlab
diffZetas(nextFidel-1) = ZETA_INC_COEFF*trueDiff;
~~~

其中：

~~~matlab
ZETA_INC_COEFF = 2;
~~~

所以思想是：

\[
\boxed{
\text{先用 GP mean 做 violation screening}
\rightarrow
\text{必要时补一个 lower-fidelity query}
\rightarrow
\text{观测到更大的 discrepancy 就把全局 bound 抬高}
}
\]

这并不是学习：

\[
e(x,m)
\]

这样的 local accuracy surface。

它更像一个只增不减的全局安全 envelope。

---

# 12. 一个 SVM 例子说明 \(\zeta\) 更新

假设：

\[
f^{(1)}(x)
=
500\text{ samples CV},
\]

\[
f^{(2)}(x)
=
2000\text{ samples CV}.
\]

当前：

\[
\zeta=0.05.
\]

某个 \(x_t\) 上，高 fidelity 得到：

\[
f^{(2)}(x_t)=0.91.
\]

低 fidelity GP 预测：

\[
\mu^{(1)}(x_t)=0.83.
\]

于是：

\[
|0.91-0.83|=0.08>0.05.
\]

算法觉得可能发生 fidelity-gap violation，于是在同一个 \(x_t\) 真正做低 fidelity：

\[
f^{(1)}(x_t)=0.84.
\]

真实 observed gap：

\[
|0.91-0.84|=0.07.
\]

于是把允许 discrepancy 提高到大约：

\[
2\times0.07=0.14.
\]

之后整个 design space 都使用更宽的 fidelity-level bound。

所以它没有回答：

> 在 \(x_A\) 区域 low fidelity 很准，在 \(x_B\) 区域 low fidelity 很差时应该怎么办？

它只维护一个全局值。

---

# 13. 三个 fidelity 时为什么最低级可以用 \(2\zeta\)

以 SALSA 为例：

\[
f^{(1)}=2000\text{ samples},
\]

\[
f^{(2)}=4000\text{ samples},
\]

\[
f^{(3)}=8000\text{ samples}.
\]

如果相邻层满足：

\[
|f^{(2)}-f^{(1)}|\le\zeta,
\]

\[
|f^{(3)}-f^{(2)}|\le\zeta,
\]

则由三角不等式：

\[
|f^{(3)}-f^{(1)}|
\le
|f^{(3)}-f^{(2)}|
+
|f^{(2)}-f^{(1)}|
\le
2\zeta.
\]

所以他们构造：

\[
\zeta^{(1)}=2\zeta,\qquad
\zeta^{(2)}=\zeta.
\]

Supernova 的三级网格 fidelity 也是同样的逻辑。

---

# 14. 论文文字和公开实现的一个细节差异

论文描述 fidelity discrepancy 时是绝对值意义上的 error bound：

\[
|f^{(m)}-f^{(m-1)}|.
\]

公开代码预筛查确实用了：

~~~matlab
abs(nextPtVal - funcHs{nextFidel-1}(nextPt))
~~~

但真正补做 lower fidelity 以后，代码写的是：

~~~matlab
trueDiff = nextPtVal - lfVal;
~~~

没有再取绝对值。

后面的 violation check 也直接使用这个带符号的 trueDiff。

所以这里存在实际 implementation 与论文所写 absolute discrepancy 概念不完全一致的细节。

复现或重新设计时需要特别注意。

---

# 15. Synthetic 和 real experiments 在 cost 上的区别

## 15.1 Synthetic

synthetic benchmark 里 cost 是人为指定的。

例如：

\[
[1,10],
\]

\[
[1,10,100],
\]

\[
[1,10,100,1000].
\]

这种 cost 主要是在模拟：

> 假设 low fidelity 更便宜、高 fidelity 更贵，算法能否利用这个结构。

它不是这些解析函数真实 CPU time 的比例。

## 15.2 Real experiments

SVM、SALSA、Viola–Jones、Supernova 的最终实验图横轴使用实际 computation time / CPU time。

论文还说明把算法自身计算下一次 query 的处理时间计入总时间。

但是：

\[
\boxed{
\text{实验评价时用真实 CPU time}
}
\]

和：

\[
\boxed{
\text{算法内部学了 }c(x,m)
}
\]

是两回事。

MF-GP-UCB 并没有真正建立：

\[
\hat c(x,m).
\]

理论和主要决策框架仍然基于 fidelity-level cost：

\[
\lambda^{(m)}.
\]

---

# 16. Reviewer 6 对 Currin 的质疑

Reviewer 6 注意到 Currin 的所谓 low fidelity，是通过多次调用 high-fidelity 解析函数再平均构造的。

于是他质疑：

> 论文的 motivation 是 low fidelity computationally cheaper，但这个 synthetic low fidelity 按字面计算公式看甚至可能比 high fidelity 更贵，为什么还能叫 cheap approximation？

这个批评需要区分两种解释。

## 16.1 Abstract oracle 解释

如果把 synthetic benchmark 看成 abstract oracle，可以定义：

\[
f_L(x)
\]

为一个 oracle，并人为规定：

\[
c_L=1.
\]

即使作者在数学公式上用多个 \(f_H\) 值来定义 \(f_L\)，也不意味着现实算法必须真的运行四次昂贵 simulator。

在这种解读下，Reviewer 6 的批评不是算法逻辑上的硬伤。

## 16.2 现实机制模拟解释

如果把 synthetic benchmark 看成现实 multi-fidelity mechanism 的模拟，那 reviewer 的质疑有一定道理。

真实 motivation 通常是：

- coarse CFD 比 fine CFD 便宜；
- 小数据训练比全数据训练便宜；
- 粗积分网格比细积分网格便宜。

如果 synthetic low fidelity 自身计算结构并不更简单，那么它没有模拟出这种真实 computational mechanism。

公平结论：

\[
\boxed{
\text{作为 abstract benchmark 合法；
作为真实 cheap approximation 的证据有限。}
}
\]

---

# 17. Reviewer 1 更重要的 cost-ratio 质疑

Synthetic 中常用：

\[
1:10,
\]

\[
1:10:100,
\]

甚至：

\[
1:10:100:1000.
\]

Reviewer 1 提出：

> 如果 cost ratio 没这么夸张，例如只差 2 倍，multi-fidelity 是否仍然有优势？

例如：

\[
c_L=1,\qquad c_H=100
\]

时，low fidelity 即使只提供一点信息，也很容易显得划算。

但如果：

\[
c_L=8,\qquad c_H=10,
\]

那么 low fidelity 的经济价值可能很小，直接 high fidelity 反而更合理。

所以真正的问题是：

\[
\boxed{
\text{算法优势到底来自方法本身，
还是来自实验人为给定的有利 cost ratio？}
}
\]

最终公开论文里仍然主要使用 \(1:10:100\) 这类 spacing，没有系统补充完整的 cost-ratio sensitivity study。

---

# 18. Reviewer 对 \(\zeta\) 的质疑

理论里要求：

\[
\|f^{(M)}-f^{(m)}\|_\infty\le\zeta^{(m)}.
\]

Reviewer 5 抓住的核心问题是：

> 现实中为什么会事先知道 low fidelity 在整个 design space 上最多会错多少？

这是一个很强的假设。

真实系统里：

\[
e(x,m)
=
|f^{(M)}(x)-f^{(m)}(x)|
\]

通常会随 \(x\)、physical regime、convergence state、model configuration 改变。

一个单独的 \(\zeta^{(m)}\) 很可能非常保守，也可能无法可靠事先获得。

作者实践里用 violation-triggered heuristic 调整 \(\zeta\)，但依然没有建立 local error model。

---

# 19. Reviewer 对 cost 的质疑

Reviewer 3 问了：

- \(\lambda^{(m)}\) 怎么设置？
- 现实里如果 cost 并不知道得那么精确怎么办？
- 算法结果会依赖这些 cost，如何保证 robustness？

最终方法仍然基本把：

\[
\lambda^{(m)}
\]

作为已知 fidelity-level cost。

没有建立：

\[
c(x,m)
\]

或者：

\[
p(c\mid x,m).
\]

所以这部分没有从根本上解决。

---

# 20. Reviewer 对 cross-fidelity information sharing 的质疑

多个 reviewers 都注意到一个问题：

每个 fidelity 单独建 GP：

\[
GP_1,\quad GP_2,\quad\ldots,\quad GP_M.
\]

低 fidelity 数据不会直接更新高 fidelity GP posterior。

不同 fidelity 主要通过：

\[
\zeta^{(m)}
\]

在 UCB upper bound 层面联系起来。

Reviewer 1、2、4 都提出：

> 为什么不直接建立一个联合 multi-output GP / co-kriging / correlated model，让不同 fidelity 的 observation 在 posterior 里真正共享信息？

最终 MF-GP-UCB 没有改成这种联合模型。

这也是它与后续大量 multi-fidelity GP / autoregressive / coregionalization 方法的重要区别。

---

# 21. 2016 年 NIPS 的 rebuttal 为什么现在不在 OpenReview

这篇论文是：

\[
\boxed{\text{NIPS 2016}}
\]

当时 NeurIPS 还没有使用今天这种完整 OpenReview workflow。

NeurIPS 到 2021 年才正式把完整 reviewing workflow 搬到 OpenReview。

所以这篇论文现在能看到：

- reviewers comments；
- accepted paper；
- final version；

但是没有找到公开的 author rebuttal 正文。

Reviewer 4 在公开 review 页面明确写：

> “AFTER REBUTTAL I have read the rebuttal and kept my scores.”

因此可以确认作者当年确实提交了 rebuttal。

但目前公开网络上没有找到这篇具体论文的 rebuttal 正文。

所以不能把 final paper 中后来出现的修改或 heuristic 直接冒充成作者在 rebuttal 里逐字怎么回复。

只能说：

> 最终发表版对 reviewer 提出的这些问题采取了什么处理。

---

# 22. Reviewer 的主要问题可以归纳成几类

## 22.1 理论假设太强，实践需要 heuristic

理论中假设：

- fidelity 已经定义好；
- cost 已知；
- approximation bound 已知；
- fidelity ordering 已知；
- GP model assumptions 成立。

现实里很多都拿不到。

于是实现里出现：

- adaptive \(\zeta\)；
- adaptive \(\gamma\)；
- hyperparameter fitting；
- numerical acquisition optimization；
- various thresholds。

这形成：

\[
\boxed{
\text{理论模型很干净}
\quad\text{vs}\quad
\text{实际实现大量 heuristic}
}
\]

之间的明显断层。

## 22.2 Fidelity 定义的真实性

Synthetic fidelity 是人工构造的。

这种实验可以验证 algorithm mechanism，但不能单独证明现实系统一定存在同样优质的 low fidelity。

## 22.3 Cost 的真实性

Synthetic cost：

\[
1,10,100,\ldots
\]

是人为赋值。

所以 synthetic 实验只能证明：

> 在指定 cost structure 下算法有效。

它不能证明：

> 现实 multi-fidelity 系统通常有如此有利的 cost ratio。

## 22.4 Accuracy 的可获得性

理论需要：

\[
\zeta^{(m)}.
\]

现实往往不知道。

最终实现只能：

\[
\text{initial guess}
\rightarrow
\text{observed violation}
\rightarrow
\text{increase bound}.
\]

这不是完整 accuracy modeling。

---

# 23. 由这篇论文引出的更广泛方法论问题

连续阅读 Active Search、Bayesian Optimization、Multi-fidelity BO 后，很容易出现一种强烈感觉：

\[
\boxed{
\text{很多方法的核心框架相似，
真正的新东西经常只是局部模块。}
}
\]

常见结构：

\[
\text{probabilistic surrogate}
\rightarrow
\text{uncertainty}
\rightarrow
\text{acquisition}
\rightarrow
\text{query}
\rightarrow
\text{update}.
\]

然后不同论文在局部替换：

\[
\text{UCB}
\rightarrow
\text{EI}
\rightarrow
\text{MES}
\rightarrow
\text{KG},
\]

或者：

\[
\text{single fidelity}
\rightarrow
\text{multi-fidelity},
\]

或者：

\[
\text{myopic}
\rightarrow
\text{multi-step},
\]

再加入 batch、parallel、continuous fidelity、cost-aware、active search 等条件。

所以很多论文看起来像：

\[
\boxed{
\text{旧框架}
+
\text{一个新条件}
+
\text{一个新的 acquisition / switching rule}
+
\text{一些 theorem}
+
\text{一些 benchmark}
}
\]

这就是很容易产生“创新点很小、东拼西凑”感觉的原因之一。

---

# 24. 这种东拼西凑感从哪里来

真正完整的问题通常应该是 sequential decision under uncertainty。

理想地写：

\[
V(s)
=
\max_a
\mathbb E
[
r(s,a)+V(s')
].
\]

状态 \(s\) 应该包含：

- 已经观测的数据；
- 所有 posterior beliefs；
- 已经启动但未完成的 tasks；
- 当前 solver states；
- 剩余预算；
- cost state；
- uncertainty；
- future possible observations。

完整 Bayesian optimal policy 在一般情况下通常极难计算。

于是不同论文用不同近似：

\[
V
\rightarrow
\text{one-step acquisition},
\]

或者：

\[
V
\rightarrow
\text{two-step rollout},
\]

或者：

\[
V
\rightarrow
\text{entropy reduction},
\]

或者：

\[
V
\rightarrow
\text{UCB}.
\]

因此很多 acquisition functions 本质上可以理解成：

\[
\boxed{
\text{对同一个不可解 sequential decision problem 的不同近似}
}
\]

从这个高层看，领域不是完全没有体系。

但具体论文如果只围绕局部 acquisition 写，就容易显得非常碎。

---

# 25. Active Search 中 heuristic 很多也有计算复杂度原因

Active Search 的 Bayesian optimal policy 原则上需要考虑：

- 当前 query；
- 所有可能观察结果；
- 每种结果下未来动作；
- 再下一步的观察；
- 一直到预算耗尽。

这会形成巨大的 decision tree。

因此 ENS、MF-ENS 等方法会做 nonmyopic approximation / rollout approximation。

所以 approximation 的存在本身不等于方法没有道理。

需要区分：

\[
\boxed{
\text{哪些 approximation 是计算复杂度逼出来的，
哪些只是因为关键现实量没有建模而用 heuristic 补洞。}
}
\]

---

# 26. 一个非常有用的论文审读标准

以后看任何 Active Search / BO / Multi-fidelity 论文，可以固定问四个问题：

1. 哪些量是现实直接测出来的？
2. 哪些量是从数据学习出来的？
3. 哪些量是作者提前规定或假定的？
4. 哪些量理论上需要，但实际实现用 heuristic 顶替？

对 MF-GP-UCB：

### 实际学习

\[
f^{(m)}(x)
\]

每个 fidelity 的 response surface。

### 理论给定

\[
\lambda^{(m)}
\]

fidelity-level cost。

### 理论给定，实践 heuristic

\[
\zeta^{(m)}
\]

最高 fidelity approximation bound。

### 实践 heuristic

\[
\gamma^{(m)}
\]

fidelity switching thresholds。

### Synthetic 中人工构造

\[
f^{(m)}
\]

low-fidelity functions 本身。

### Synthetic 中人工指定

\[
1:10:100
\]

等 cost ratios。

这样拆以后，论文的结构会透明很多。

---

# 27. 对“这类论文缺少核心体系”的更精确判断

一个可以接受的高层体系是：

\[
\boxed{
\text{belief model}
+
\text{decision utility}
+
\text{sequential update}
}
\]

Bayesian Optimization：

\[
p(f\mid D_t)
\rightarrow
\text{acquisition}
\rightarrow
x_{t+1}
\rightarrow
D_{t+1}.
\]

Active Search：

\[
P(y=1\mid D_t)
\rightarrow
\text{future discoveries}
\rightarrow
x_{t+1}
\rightarrow
D_{t+1}.
\]

真正显得杂乱的是：

- surrogate 是否可信；
- acquisition 是否精确；
- cost 是否已知；
- fidelity relationship 是否已知；
- noise 是否已知；
- inference 是否可计算；
- exact dynamic programming 是否可行。

很多论文在这些地方分别用不同 heuristic。

所以“没有体系”的感觉，很多时候来自：

\[
\boxed{
\text{统一的大框架存在，
但论文创新往往只是替换其中一个局部模块。}
}
\]

---

# 28. 对当前 continuous solver / active search 研究的启示

当前研究的问题可以定义得更统一。

## 28.1 State

对已经启动的每个 design \(x_i\)，保存：

\[
(x_i,z_i,y_i(z_i),\text{solver state}_i).
\]

全局 state：

\[
s_t=
\{
x_i,
z_i,
y_i(z_i),
\text{solver state}_i,
\text{belief},
B_t
\}_{i=1}^{N_t}.
\]

其中：

- \(z_i\)：solver progress；
- \(B_t\)：剩余总预算。

## 28.2 Action

动作统一成：

\[
a_t=
\begin{cases}
\text{start a new design }x,\\
\text{continue an existing design},\\
\text{inspect/check},\\
\text{stop/drop a design}.
\end{cases}
\]

这样 Active Search、Multi-fidelity BO、Freeze-Thaw、progressive solver、early stopping 都可以落在同一 sequential resource-allocation 框架中。

## 28.3 Cost

比固定：

\[
\lambda_m
\]

更真实的是：

\[
\Delta c
=
c(
x,
z_{\mathrm{current}}
\rightarrow
z_{\mathrm{next}},
\text{saved state}
).
\]

直接使用真实：

- wall-clock time；
- checkpoint resume overhead；
- save overhead；
- I/O；
- solver-specific runtime。

这样 cost 来自真实执行，而不是人为 price tag。

## 28.4 Accuracy / final-value uncertainty

也不要只用：

\[
\zeta_m.
\]

可以从 partial trajectory 学：

\[
p(
F(x)
\mid
x,
z,
y(x,z),
\text{trajectory history}
).
\]

这样 uncertainty 是：

\[
\boxed{
\text{design-dependent}
+
\text{progress-dependent}
}
\]

而不是每个 fidelity 一个统一 worst-case error bound。

---

# 29. 更有体系的研究方向

一个更完整的问题可以写成：

\[
\boxed{
\text{continuous-space,
resumable,
cost-aware,
belief-based sequential search}
}
\]

其中同时决定：

\[
\boxed{
\text{where to evaluate}
+
\text{how far to evaluate}
+
\text{whether to continue}
+
\text{when to inspect/update}
}
\]

并且真正计入：

\[
\boxed{
\text{solver cost}
+
\text{inspection cost}
+
\text{decision/model-update cost}
}
\]

最终目标可以是：

\[
\max
\mathbb E[
\text{confirmed positive discoveries}
]
\]

或者其他明确的 end utility。

这种 formulation 的价值在于：

> 不只是再发明一个 acquisition function，而是先把以前彼此分散的 BO、Active Search、Freeze-Thaw、Multi-fidelity、progressive solver 统一成一个 sequential decision problem。

---

# 30. 最后总结

Kandasamy et al. (2016) 的真正贡献可以精确描述为：

> 在一组已经存在、已经有序、具有给定成本、并且相对于最高 fidelity 有给定误差界的多个 fidelity 下，设计一个 MF-GP-UCB 策略，并分析它如何利用便宜 fidelity 排除大片无希望区域，把昂贵最高 fidelity 查询集中在 promising region。

它真正解决的是：

\[
\boxed{
\text{在给定 multi-fidelity structure 下怎么做 query allocation}
}
\]

它没有解决：

\[
\boxed{
\text{fidelity 应该怎样从真实系统中产生}
}
\]

没有真正解决：

\[
\boxed{
(x,m)\rightarrow cost
}
\]

也没有真正解决：

\[
\boxed{
(x,m)\rightarrow accuracy/error
}
\]

理论上这两部分主要通过：

\[
\lambda^{(m)},
\qquad
\zeta^{(m)}
\]

作为 assumptions 输入。

实际实现再通过 adaptive \(\zeta\)、adaptive \(\gamma\) 等 heuristic 来补现实中的缺口。

因此对这类论文最重要的阅读方式不是只看 theorem 或最终 benchmark，而是始终区分：

\[
\boxed{
\text{什么是学出来的，
什么是测出来的，
什么是人为给定的，
什么是 heuristic。}
}
\]

这也是本次讨论最后最核心的方法论结论。

---

# 参考链接

1. Kandasamy et al. (2016), NIPS paper  
   https://proceedings.neurips.cc/paper/2016/hash/605ff764c617d3cd28dbbdd72be8f9a2-Abstract.html

2. Paper PDF  
   https://proceedings.neurips.cc/paper_files/paper/2016/file/605ff764c617d3cd28dbbdd72be8f9a2-Paper.pdf

3. Public reviews  
   https://proceedings.neurips.cc/paper_files/paper/2016/file/605ff764c617d3cd28dbbdd72be8f9a2-Reviews.html

4. Public implementation  
   https://github.com/kirthevasank/mf-gp-ucb

5. Kandasamy et al. (2017), BOCA  
   https://proceedings.mlr.press/v70/kandasamy17a.html

6. Jiang et al. (2017), Efficient Nonmyopic Active Search  
   https://proceedings.mlr.press/v70/jiang17d.html

7. Nguyen et al. (2021), Nonmyopic Multifidelity Active Search  
   https://proceedings.mlr.press/v139/nguyen21f.html
