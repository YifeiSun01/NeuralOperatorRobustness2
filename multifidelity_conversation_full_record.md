# Multi-Fidelity / Solver Fidelity 对话完整整理

> 本文件按本次对话从开头到当前为止的实际讨论顺序重建。  
> 目标是完整保存技术内容、论文、公式、概念修正和最终形成的统一框架。  
> 为避免文档被大量重复口头语淹没，重复语气词没有逐字抄录，但所有实质问题、争论点、纠正和结论都保留。

---

# 0. 这次对话最开始在问什么？

本次对话的起点，是下面这个问题：

> “你给我弄一个非常详细的这个，你上网查一下，Literature Review吧，或者什么你查一下这个 solver 这个东西的 fidelity 是怎么定义的，或者不是说 solver，而是什么东西有 fidelity，fidelity 是怎么定义的，什么算 fidelity，神经网络有 fidelity 吗，solver 有 fidelity 吗，numerical solver 有 fidelity 吗，CFD 有 fidelity 吗，它的 fidelity 指的是什么东西，数据量算 fidelity 吗，训练的时间，迭代的时间算 fidelity 吗，这个什么 mesh，什么网格，网格点数量算 fidelity 吗，计算量算 fidelity 吗，有没有一些 solver 或者一些情况下它没有 fidelity，没有 fidelity 的定义，或者是有什么情况下有 fidelity，有什么情况下没有 fidelity，那个什么计算时间算 fidelity 吗，什么 progress，什么 iteration，epoch 算 fidelity 吗，神经网络的训练 epoch 算 fidelity 吗，solver 的 iteration 算 fidelity 吗。那种 temporal evolution 的那种 solver，它有 progress 作为 fidelity 吗，就各种各样不同的情况下，它 fidelity 是怎么定义的。你给我做一个详细的调查，好吗？我要非常非常详细的调查。”

所以，这一整段对话从一开始就在追问一个核心问题：

\[
\boxed{\text{fidelity 到底是什么？}}
\]

以及：

\[
\boxed{
\text{design/search variable、fidelity variable、output/accuracy、cost/resource 到底怎么区分？}
}
\]

后面的所有讨论，基本都围绕这两个问题逐层展开。

---

# 1. 最开始形成的统一抽象

为了统一很多不同论文，我们逐渐采用了下面这套符号：

\[
x\in\mathcal X
\]

表示 design / search / problem input，

\[
z\in\mathcal Z
\]

表示 fidelity / resource / evaluation level，

然后：

\[
y=g(x,z)
\]

表示在 design \(x\)、fidelity \(z\) 下得到的输出。

指定一个 target fidelity：

\[
z_\star,
\]

最终真正关心：

\[
y^\star(x)=g(x,z_\star).
\]

于是一个 low-fidelity evaluation 的误差可以写成：

\[
d(x,z)=g(x,z)-g(x,z_\star)
\]

或：

\[
e(x,z)=|g(x,z)-g(x,z_\star)|.
\]

最一般的 cost 写成：

\[
c(x,z).
\]

很多经典文章为了简化，会写成：

\[
\lambda(z)
\]

即假设 cost 只由 fidelity 决定。

这个统一框架后来成为整理所有论文的主线：

\[
\boxed{
x
\;\;+\;\;
z
\;\longrightarrow\;
g(x,z)
}
\]

同时：

\[
\boxed{
(x,z)\longrightarrow c(x,z).
}
\]

---

# 2. fidelity 不是某个固定物理量，而是一个“角色”

整个讨论中最重要的结论之一：

\[
\boxed{
\text{fidelity 不是某个固定物理量的名字，而是一个 problem-dependent 的角色。}
}
\]

所以不同文章里，下面这些量都可能被拿来作为 fidelity：

- mesh resolution；
- mesh spacing；
- finite-element element size；
- timestep；
- residual tolerance；
- solver iteration count；
- computational time；
- training epochs；
- SGD steps；
- training data amount；
- image resolution；
- model width / depth；
- simulation horizon；
- information source ID；
- different physical models；
- reduced-order model level。

关键不在变量叫什么，而在：

> 这个量是不是被作者用来控制“同一个 target evaluation 的便宜/粗糙版本”。

---

# 3. fidelity 与 design/search variable 的边界

一个变量本身不天然属于 design 或 fidelity。

例如 neural network width：

如果问题是：

\[
w^\star=\arg\max_w \text{Accuracy}(w),
\]

那么 width 是：

\[
\boxed{w\in x}
\]

即 search / architecture variable。

但如果最终模型固定：

\[
w_\star=1024,
\]

而使用：

\[
128,256,512
\]

的较小模型，只是为了便宜地预测 width=1024 的表现，

那么 width 可以被定义成：

\[
\boxed{w\in z}
\]

即 fidelity variable。

所以判断标准不是变量的名字，而是：

\[
\boxed{
\text{最终任务是在选择这个变量，还是已经固定 target 值，仅用其他值做代理？}
}
\]

同理：

- mesh size 可以是 fidelity；
- 但如果研究目标本身是 optimize mesh distribution，它也可以是 design variable。

---

# 4. fidelity 与 cost 的边界

一开始我们讨论 wall-clock time、FLOPs、GPU-hours 是否算 fidelity。

最终得到的结论是：

## 4.1 最常见理解

\[
\boxed{
\text{fidelity variable}
\rightarrow
\text{associated cost}
}
\]

比如：

\[
z=T=\text{training epochs},
\]

cost：

\[
c(T)=\text{GPU time / FLOPs}.
\]

又比如：

\[
z=h=\text{mesh resolution},
\]

cost：

\[
c(h)=\text{CPU/GPU runtime}.
\]

所以 wall-clock / FLOPs 一般优先理解成：

\[
\boxed{\text{cost}}
\]

而不是 fidelity 本身。

## 4.2 但也可以直接把 resource budget 当 fidelity

例如直接规定：

\[
z=\text{allocated GPU seconds}
\]

或者：

\[
z=\text{computational time}.
\]

这时候同一个变量同时承担：

- resource budget；
- fidelity/progress coordinate；
- cost proxy。

Picheny & Ginsbourger 2013 就非常接近这种情况。

因此：

\[
\boxed{
\text{fidelity 与 cost 在某些论文里会重合，但并没有统一标准。}
}
\]

---

# 5. Peherstorfer, Willcox & Gunzburger 2018 SIAM Review

论文：

**Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization**

这是整个 computational multi-fidelity 文献里非常经典的综述。

## 5.1 它自己的 formalism

它写：

\[
f:\mathcal Z\to\mathcal Y.
\]

这里它的 \(z\) 不是我们后来统一写法里的 fidelity。

这里的 \(z\) 是：

\[
\boxed{\text{模型输入 / design / physical parameter}}
\]

也就是我们为了避免混淆应该改写成：

\[
x.
\]

所以更清楚的统一写法是：

\[
f_{\rm hi}(x),
\]

以及：

\[
f_{\rm lo}^{(1)}(x),
\ldots,
f_{\rm lo}^{(k)}(x).
\]

它没有一个统一显式的 continuous fidelity variable。

## 5.2 high fidelity

high-fidelity model 是：

> 对当前任务达到所需精度的 reference model。

所以：

\[
\boxed{
\text{high fidelity 是 task-relative reference，不是物理世界绝对真值。}
}
\]

例如当前论文可以规定：

\[
512^3
\]

mesh 的 CFD 为 high fidelity。

未来更高分辨率：

\[
1024^3
\]

依然可能存在。

## 5.3 low-fidelity model 三大来源

论文 Figure 4 把 low-fidelity models 分成三类。

### A. Simplified models

包括：

- simplified physics；
- coarse-grid approximation；
- early stopping criteria；
- natural problem hierarchies。

例如：

\[
\text{DNS}\rightarrow \text{LES}\rightarrow \text{RANS}
\]

或：

\[
1024^3\rightarrow 128^3
\]

或：

\[
\epsilon_{\rm residual}=10^{-8}
\rightarrow
10^{-3}.
\]

### B. Projection-based reduced-order models

例如：

- POD；
- Reduced Basis；
- Krylov；
- balanced truncation。

这里的 “reduced order” 不是 RK4 那种 numerical order。

它的 order 指的是：

\[
\boxed{\text{state-space dimension / degrees of freedom}}
\]

比如：

\[
10^6\text{ DOF}
\rightarrow
50\text{ DOF}.
\]

### C. Data-fit models

只需要 high-fidelity input-output samples：

\[
(x_i,y_i)
\]

训练一个 surrogate：

\[
\hat f(x).
\]

可以是：

- regression；
- kriging；
- Gaussian Process；
- neural network。

## 5.4 三种 model management

这篇 survey 还把 multi-fidelity method 分成：

- adaptation；
- fusion；
- filtering。

### adaptation

用少量 HF 修正 LF：

\[
f_{\rm hi}(x)\approx f_{\rm lo}(x)+\delta(x).
\]

### fusion

把 HF/LF 数据统计融合，例如 control variates、cokriging。

### filtering

先算便宜 LF：

\[
x\overset{LF}{\longrightarrow}\text{score}
\]

明显没希望就淘汰，

有希望才支付 HF。

这一点后来我们特别指出：

\[
\boxed{\text{filtering 与 active search / multi-fidelity search 很接近。}}
\]

## 5.5 heterogeneous fidelity

这篇综述还强调：

\[
f_{\rm lo}^{(1)},f_{\rm lo}^{(2)},\ldots
\]

不一定能构成：

\[
L_1<L_2<L_3<H.
\]

因为可能同时存在：

- coarse-grid model；
- ROM；
- simplified physics；
- data-fit surrogate。

它们来自完全不同的机制，所以没有一个统一 refinement parameter。

---

# 6. Forrester, Bressloff & Keane 2006：Partially Converged CFD

论文：

**Optimization Using Surrogate Models and Partially Converged Computational Fluid Dynamics Simulations**

这是我们讨论 solver progress 最重要的早期论文之一。

## 6.1 它不是 physical time evolution

这里做的是 steady CFD。

solver iteration：

\[
u^{(1)}
\rightarrow
u^{(2)}
\rightarrow
\cdots
\rightarrow
u^\star
\]

逐步逼近同一个 steady solution。

这里：

\[
u^{(k)}
\]

表示第 \(k\) 次迭代时的完整 CFD 流场状态。

可以从它计算：

\[
C_D^{(k)}
\]

阻力系数，

以及：

\[
C_L^{(k)}
\]

升力系数。

所以：

\[
u^{(k)}
\rightarrow
C_D^{(k)}.
\]

## 6.2 fully converged

论文二维例子里大约：

\[
1500\text{ iterations}
\]

被作为 practically fully converged reference。

而：

\[
200,\;250,\;300
\]

iterations 的 CFD 是 partially converged results。

## 6.3 fidelity

在我们的统一语言里：

\[
\boxed{z=k=\text{iteration count}}
\]

可以作为 solver convergence fidelity。

## 6.4 accuracy relation 怎么得到

不是提前假定。

他们直接：

1. 跑不同 design；
2. 截取不同 iteration；
3. 建 surrogate；
4. 比较 partial surrogate 与 fully converged surface 的 correlation。

甚至发现：

\[
\text{correlation}
\]

并不随 iteration 平滑单调，因为某些阶段 flow structure 变化会让 surrogate 突然变差。

## 6.5 cost

使用：

\[
\text{iterations}\times\text{number of design points}
\]

作为 computing-effort proxy，

同时报告 CPU time。

所以这里是：

\[
\boxed{
\text{accuracy relation：实测}
}
\]

\[
\boxed{
\text{cost：iteration proxy + 实测 CPU}
}
\]

---

# 7. Picheny & Ginsbourger 2013

论文：

**A Nonstationary Space-Time Gaussian Process Model for Partially Converged Simulations**

这是我们讨论最久的一篇。

## 7.1 “space-time” 的真正含义

这里：

\[
\boxed{\text{space}=\text{design parameter space}}
\]

\[
\boxed{\text{time}=\text{computational time / solver progress}}
\]

绝对不是：

\[
\Omega_{\rm physical}\times [0,T]_{\rm physical}.
\]

它做的是 steady-state OpenFOAM `simpleFoam`。

## 7.2 design variable

\[
x=(x_1,\ldots,x_7)
\]

控制 S-shaped pipe 的几何设计。

## 7.3 computational time

写：

\[
Y(x,t).
\]

其中：

\[
t=\text{computational time / iteration progress}.
\]

用我们的统一语言：

\[
\boxed{t\text{ 承担 continuous fidelity/progress coordinate 的角色}}
\]

但原文主要称 computational time，不是正式写 “fidelity variable”。

## 7.4 关键 decomposition

\[
\boxed{
Y(x,t)=F(x)+G(x,t)
}
\]

其中：

\[
F(x)=\text{fully converged response}
\]

\[
G(x,t)=\text{partial-convergence error}.
\]

要求：

\[
G(x,t)\to0
\qquad
(t\to\infty).
\]

于是：

\[
Y(x,t)\to F(x).
\]

## 7.5 nonstationary GP

普通 stationary kernel：

\[
k(t,t')=k(t-t')
\]

不适合 solver convergence。

因为：

\[
(10,20)
\]

和：

\[
(410,420)
\]

虽然差都是 10，但 early stage 和 late stage 的行为完全不同。

作者观察：

- early stage 振荡快；
- error amplitude 大；
- later stage 更平滑；
- error amplitude 变小。

所以人为规定：

\[
\sigma(t)\downarrow
\]

并：

\[
\sigma(t)\to0.
\]

还使用 nonlinear time transformation：

\[
a(t)=\frac{1}{\zeta+\eta t}
\]

让 correlation length 随 absolute \(t\) 改变。

## 7.6 什么是“假设”，什么是“从数据学”

他们预先规定的是：

- \(Y=F+G\)；
- \(G\to0\)；
- error amplitude 随 \(t\) 衰减；
- later stage smoother；
- 相近 design 的 convergence behavior 相近。

但是：

- decay rate；
- covariance length scales；
- variance parameters；

是从 partially converged simulation data 里估出来的。

所以：

\[
\boxed{
\text{结构先验 + 数据拟合参数}
}
\]

## 7.7 cost

没有另外建：

\[
c(x,t).
\]

因为：

\[
t
\]

本身就是 computational-time resource coordinate。

---

# 8. residual tolerance、iteration count、computational time

这一段讨论形成了很重要的统一认识。

对于 iterative solver：

\[
r_k(x)=\|R(u_k(x))\|.
\]

设 stopping tolerance：

\[
\epsilon.
\]

停止 iteration：

\[
K(x,\epsilon)
=
\min\{k:r_k(x)\le \epsilon\}.
\]

然后 computational time：

\[
T(x,\epsilon)
=
\sum_{j=1}^{K(x,\epsilon)}c_j(x).
\]

所以存在一条链：

\[
\boxed{
\epsilon
\rightarrow
K(x,\epsilon)
\rightarrow
T(x,\epsilon).
}
\]

## 8.1 residual tolerance

\[
\epsilon
\]

是运行前设定的：

\[
\boxed{\text{convergence requirement}}
\]

## 8.2 iteration count

\[
k
\]

可以是：

\[
\boxed{\text{allocated computation budget / progress}}
\]

## 8.3 current residual

\[
r_k
\]

是运行过程中观察到的：

\[
\boxed{\text{convergence state / diagnostic}}
\]

## 8.4 computational time

\[
t_k
\]

通常是：

\[
\boxed{\text{realized cost}}
\]

但 Picheny 这类文章也会直接把它当成 progress/fidelity coordinate。

## 8.5 三者不是独立 fidelity

它们是：

\[
\boxed{\text{同一 convergence process 的不同 parameterization / state / cost}}
\]

不能简单认为是三个互相独立的 fidelity dimensions。

## 8.6 同一 tolerance 不同 design cost 不一样

可能：

\[
K(x_A,10^{-6})=100
\]

但：

\[
K(x_B,10^{-6})=1000.
\]

所以真实 cost 最一般应该写：

\[
\boxed{c(x,z)}
\]

而不是：

\[
\lambda(z).
\]

---

# 9. Courrier, Boucard & Soulier 2014

论文：

**The Use of Partially Converged Simulations in Building Surrogate Models**

这是我们讨论：

\[
\boxed{\text{不是所有 solver progress 都适合当 fidelity}}
\]

最重要的例子。

## 9.1 classical incremental Newton

loading history：

\[
0=\lambda_0<\lambda_1<\cdots<\lambda_N.
\]

每个 load step：

\[
R_n(u_n;u_{n-1},\text{history})=0.
\]

Newton：

\[
u_n^{(0)}
\to
u_n^{(1)}
\to
\cdots
\to
u_n^\star.
\]

只有：

\[
u_n^\star
\]

可靠收敛后，才能进入下一 load step。

## 9.2 为什么 partial convergence 有问题

如果每个 step 都只做少量 iterations：

\[
u_1^{(2)}
\to
u_2^{(2)}
\to
u_3^{(2)}
\]

前一步的不收敛状态会被带入后一步。

contact/friction 里还有：

- open/closed；
- stick/slip；
- friction history；
- internal variables。

所以误差可能改变整个 loading history。

原文非常直接：

> 对 classical incremental Newton-type method，partially converged calculation 的概念在他们需要的 whole-loading-interval 意义下是 “meaningless”。

更精确地说：

> 只有最后一个 loading increment 可以 partial convergence；前面的 increment 必须可靠收敛。

## 9.3 LATIN

所以他们改用：

\[
\boxed{\text{LATIN}}
\]

即 non-incremental iterative method。

每一个 global iteration：

\[
u^{(k)}(\lambda),
\qquad
\lambda\in[0,1]
\]

都给出整个 loading path 的 approximate response。

于是：

\[
u^{(1)}(\lambda),
u^{(2)}(\lambda),
\ldots
\to
u^\star(\lambda)
\]

才构成自然的 progress/fidelity hierarchy。

## 9.4 最终结论

\[
\boxed{
\text{solver progress 是否能当 fidelity，取决于 solver architecture。}
}
\]

---

# 10. Temporal evolution 与 fidelity

我们讨论了一个重要反例：

固定：

\[
\Delta t=0.01.
\]

100 steps：

\[
u(1)
\]

500 steps：

\[
u(5)
\]

1000 steps：

\[
u(10).
\]

对于真正的 transient PDE：

\[
u_t=F(u),
\]

这些是不同 physical times 的真实状态。

所以：

\[
u(1)
\]

不是：

\[
u(10)
\]

的“不准确版本”。

因此：

\[
\boxed{
\text{physical temporal progress 一般不天然等于 fidelity progress。}
}
\]

## 10.1 什么情况下 simulation horizon 可以是 fidelity

如果目标是长期统计：

\[
J(x)=\text{long-run average}
\]

那么：

\[
\hat J_{1\text{day}},
\hat J_{7\text{day}},
\hat J_{30\text{day}}
\]

都在估同一个 \(J(x)\)。

这时候 horizon 变长可以：

- 平均掉短期波动；
- 提高统计估计稳定性。

于是 horizon 可以合理作为 fidelity。

## 10.2 EV charging station 2026

论文：

**Enhancing Electric Vehicle Charging Station Design Using Multifidelity Simulations**

它明确用：

- simulation horizon；
- time-step resolution；
- arrival segmentation；

控制 fidelity。

这里 horizon 是 statistical-performance fidelity，不是 terminal-state fidelity。

## 10.3 Planning horizon

**Multi-Fidelity Benders Decomposition for Generation, Storage, and Transmission Expansion Planning**

这里 short horizon 是对 full-horizon optimization problem 的便宜近似。

也是另一种 horizon fidelity。

## 10.4 最终判断

\[
\boxed{
\text{时间更长只有在“更完整地估计同一个 QoI”时才自然形成 fidelity。}
}
\]

如果只是：

\[
u(t_1)\rightarrow u(t_2)
\]

真实物理演化，

那通常不适合作为 fidelity。

---

# 11. Neural Network / HPO 中的 fidelity

## 11.1 最经典场景：HPO

\[
x=(\text{learning rate},\text{weight decay},\ldots)
\]

fidelity：

\[
z=(N,T)
\]

其中：

\[
N=\text{training data amount}
\]

\[
T=\text{training iterations / epochs}.
\]

target：

\[
z_\star=(N_\star,T_\star).
\]

最终：

\[
f(x)=g(N_\star,T_\star,x).
\]

便宜 evaluation：

\[
g(N,T,x)
\]

帮助预测 full-training performance。

---

# 12. Kandasamy 2016 / MF-GP-UCB：离散 hierarchy

论文：

**Gaussian Process Bandit Optimisation with Multi-Fidelity Evaluations**

有：

\[
f^{(1)},\ldots,f^{(M)}
\]

并规定：

\[
\boxed{f^{(M)}=f}
\]

即第 \(M\) fidelity 就是最终真正优化的 target function。

假设：

\[
\boxed{
\|f^{(M)}-f^{(m)}\|_\infty
\le
\zeta^{(m)}
}
\]

其中：

\[
\zeta^{(1)}>
\zeta^{(2)}>
\cdots>
\zeta^{(M)}=0.
\]

同时 cost：

\[
\lambda^{(1)}
<
\lambda^{(2)}
<
\cdots<
\lambda^{(M)}.
\]

## 12.1 \(\zeta^{(m)}\)

它是：

\[
\boxed{\text{global worst-case error upper bound}}
\]

即：

\[
\sup_x
|f^{(M)}(x)-f^{(m)}(x)|
\le
\zeta^{(m)}.
\]

它不是实际误差。

更高 fidelity 的：

\[
\zeta^{(m)}
\]

更小。

但不要求每个具体 \(x\) 上实际 error 都严格下降。

## 12.2 这篇比一般 MF 假设更强

它要求：

- discrete ordered fidelity；
- cost monotone；
- worst-case error upper bound monotone。

后来 BOCA 2017 就认为这种 global uniform bound 太强。

---

# 13. Kandasamy 2017 BOCA

论文：

**Multi-fidelity Bayesian Optimisation with Continuous Approximations**

这是整个对话中最清楚的统一 framework。

## 13.1 formalism

\[
g:\mathcal Z\times\mathcal X\to\mathbb R
\]

\[
x\in\mathcal X
\]

search/design；

\[
z\in\mathcal Z
\]

fidelity。

指定：

\[
z_\bullet
\]

target fidelity：

\[
\boxed{
f(x)=g(z_\bullet,x)
}
\]

最终 optimize：

\[
x^\star=\arg\max_x g(z_\bullet,x).
\]

## 13.2 fidelity 与 approximation

严格来说：

\[
\boxed{
\text{fidelity}=z
}
\]

而：

\[
\boxed{
\text{approximation}=g(z,\cdot)
}
\]

所以 “continuous spectrum of approximations” 是：

\[
\{g(z,\cdot):z\in\mathcal Z\}
\]

由 continuous fidelity space \(\mathcal Z\) 索引。

## 13.3 NN 例子

\[
z=(N,T)
\]

其中：

\[
N=\text{training data}
\]

\[
T=\text{training iterations}.
\]

目标：

\[
z_\bullet=(N_\bullet,T_\bullet).
\]

原文明确说：

> \(N,T\) 严格来说是离散整数，但更自然地把它们看作来自一个 continuous 2D fidelity space。

所以这是：

\[
\boxed{\text{离散执行量的连续建模}}
\]

## 13.4 target fidelity 是否一定每个分量最大

在 NN 例子里：

\[
z_\bullet=(N_\bullet,T_\bullet)
\]

确实每个分量都在最大端。

实验常统一：

\[
\mathcal Z=[0,1]^p
\]

并取：

\[
z_\bullet=\mathbf 1_p.
\]

但理论 formalism 只要求：

\[
z_\bullet\in\mathcal Z
\]

是一个指定 target point。

不要求一定是 cost 最大点。

## 13.5 “target 不必最贵”

论文说：

把：

\[
z_\bullet
\]

想成最昂贵 fidelity 很自然，

但算法和理论严格来说并不要求：

\[
z_\bullet=\arg\max_z \lambda(z).
\]

真正需要：

\[
\exists z:
\lambda(z)<\lambda(z_\bullet).
\]

即至少有一些 cheap fidelity。

## 13.6 “cheap fidelity 有信息”怎么数学化

BOCA 不再用 uniform deterministic error bound。

它对：

\[
g
\]

建 Gaussian Process：

\[
g\sim\mathcal{GP}.
\]

kernel：

\[
\kappa([z,x],[z',x'])
=
\kappa_0
\phi_Z(\|z-z'\|)
\phi_X(\|x-x'\|).
\]

其中：

\[
\phi_Z
\]

控制 fidelity direction 的 smoothness / correlation。

还定义 information gap：

\[
\xi(z)
=
\sqrt{
1-\phi_Z(\|z-z_\bullet\|)^2
}.
\]

所以这里“informative”不是纯口头概念，而是通过 GP covariance 结构表达。

## 13.7 output 与 cost 怎么得到

### output / cross-fidelity relation

从 observations：

\[
D_n=
\{(z_i,x_i,y_i)\}
\]

在线更新 GP。

kernel hyperparameters 可以通过：

- marginal likelihood；
- cross validation；

从数据估。

所以：

\[
\boxed{\text{output / fidelity relation：数据学习}}
\]

### cost

BOCA 定义：

\[
\boxed{
\lambda:\mathcal Z\to\mathbb R_+
}
\]

并假设：

\[
\boxed{\lambda(z)\text{ 已知}}
\]

即：

\[
\boxed{\text{cost 不学}}
\]

## 13.8 \(O(N^2T)\)

这个式子只是 conditional example。

原文逻辑：

> 如果某个训练算法对数据量 \(N\) 是 quadratic，对 iteration \(T\) 是 linear，那么：

\[
\lambda(N,T)=O(N^2T).
\]

这不是 neural network 普遍复杂度。

它只是为了说明：

\[
\boxed{
\text{fidelity space 可以配一个 cost function}
}
\]

真实 20 Newsgroups / SVM 实验里，他们用了近似：

\[
\lambda(N,T)=NT.
\]

天体物理例子还用了：

\[
\lambda(N,G)=NG.
\]

所以：

\[
\boxed{
\lambda(z)\text{ 的具体形式完全取决于 evaluator / algorithm。}
}
\]

---

# 14. Multi-dimensional fidelity 与 ordering

对于：

\[
z=(N,T)
\]

两个点：

\[
z_A=(1000,100)
\]

和：

\[
z_B=(5000,20)
\]

不能自然说：

\[
z_A>z_B
\]

或者：

\[
z_B>z_A.
\]

因为一个 data 更多，一个 iteration 更多。

严格数学上，多维函数依然可以有 componentwise monotonicity。

但多维 fidelity space 通常没有天然 total order。

可以有 partial order：

\[
z_A\preceq z_B
\]

若每个分量都不大于对应分量。

所以：

\[
\boxed{
\text{多维 fidelity space 一般不是传统 low→medium→high 一条数轴。}
}
\]

---

# 15. non-hierarchical multi-fidelity 与 MISO

我们区分了三个概念。

## 15.1 General fidelity space

\[
z\in\mathcal Z
\]

有明确 fidelity coordinates。

可能是多维，未必 total order。

## 15.2 Non-hierarchical multi-fidelity

有一个 HF reference，

有多个 LF：

\[
L_1,L_2,L_3
\]

但 LF 之间无法统一排序。

例如：

- fine-grid Euler；
- coarse-grid Navier–Stokes。

一个 physics 更完整，一个 grid 更细，无法说谁一定更 high fidelity。

## 15.3 Multi-Information-Source Optimization

比 MF 更宽。

有：

\[
S_1,S_2,\ldots,S_K
\]

多个 information sources，

每个：

- cost 不同；
- bias 不同；
- noise 不同。

甚至不要求预先有 fidelity ranking。

所以：

\[
\boxed{
\text{MISO 比严格 hierarchical MF 更宽。}
}
\]

---

# 16. Shibo Li 2020 / 2021 / 2022

我们讨论了施博利几篇 multi-fidelity 论文。

## 16.1 Shibo Li 2020

**Multi-Fidelity Bayesian Optimization via Deep Neural Networks**

离散 fidelity：

\[
f_1(x),\ldots,f_M(x)
\]

通常：

\[
m\uparrow
\]

对应：

- higher fidelity；
- higher cost；
- better approximation。

用 DNN 学复杂 nonlinear cross-fidelity relation。

## 16.2 Shibo Li 2021

**Batch Multi-Fidelity Bayesian Optimization with Deep Auto-Regressive Networks**

类似，但支持 batch query。

## 16.3 DMFAL 2022

**Deep Multi-Fidelity Active Learning of High-Dimensional Outputs**

每一步同时选择：

\[
x=\text{query input}
\]

和：

\[
m=\text{query fidelity}
\]

目标是最大化：

\[
\text{learning benefit}/\text{cost}.
\]

这里 deep NN 是 surrogate，

fidelity 来自 physical simulation levels。

---

# 17. Shibo Li 2022 IFC

论文：

**Infinite-Fidelity Coregionalization for Physical Simulation**

这是 continuous fidelity 的核心论文之一。

## 17.1 fidelity 是什么

这里 fidelity 主要来自 numerical simulation resolution：

- mesh spacing；
- finite-element element size。

所谓 finite-element length，指：

\[
\boxed{\text{有限元单元的特征尺寸/边长}}
\]

单元更小：

\[
\Rightarrow
\text{mesh 更细}
\Rightarrow
\text{通常更贵}
\]

## 17.2 \(m=0\)

论文说：

> without loss of generality, lowest fidelity = 0。

所以：

\[
\boxed{m=0}
\]

只是最低 fidelity 的归一化坐标原点。

不是：

- mesh=0；
- 没有 mesh；
- 没有精度。

实验中：

\[
8\times8
\]

被映射到：

\[
m=0
\]

\[
64\times64
\]

映射到：

\[
m=1.
\]

还可以 extrapolate：

\[
m>1.
\]

## 17.3 \(h(m,x)\)

这里：

\[
h(m,x)
\]

不是 mesh spacing。

它是：

\[
\boxed{\text{high-dimensional PDE output 的 low-dimensional latent representation}}
\]

## 17.4 neural ODE

论文写：

\[
\frac{\partial h(m,x)}{\partial m}
=
\phi(m,h(m,x),x).
\]

含义：

> 当前 latent representation 沿 fidelity 轴怎么变化，取决于当前 fidelity \(m\)、当前 latent state \(h\)、PDE input \(x\)。

它不是在同时 optimize \(x,m\)。

而是在学习：

\[
(x,m)\mapsto \text{solution representation}.
\]

## 17.5 为什么右边有 \(h\)

就像普通 ODE：

\[
\frac{dy}{dt}=F(t,y)
\]

下一时刻变化取决于当前 state。

这里把 time axis 换成 fidelity axis：

\[
\frac{\partial h}{\partial m}
=
\phi(m,h,x).
\]

## 17.6 与 Kandasamy 2017 的区别

Shibo Li：

\[
m\in\mathbb R
\]

主要是一维 scalar continuous fidelity。

Kandasamy：

\[
z\in\mathbb R^p
\]

允许 multi-dimensional fidelity space。

所以在 fidelity-space formalism 上，BOCA 更一般。

但 Shibo Li 更专门处理：

\[
\boxed{\text{high-dimensional physical simulation outputs}}
\]

## 17.7 是否引用 Kandasamy 2017

我们查了 NeurIPS 2022 官方 PDF：

没有引用 Kandasamy 2017 BOCA。

因此它摘要里：

> existing approaches only model finite, discrete fidelities

如果按整个 multi-fidelity literature 字面理解，是写宽了。

因为 2017 BOCA 已经明确研究 continuous fidelity spaces。

更合理的 novelty 应该限定成：

\[
\boxed{
\text{continuous/infinite-fidelity coregionalization for high-dimensional physical simulation outputs}
}
\]

所以这里存在：

\[
\boxed{\text{overly broad novelty wording / literature omission}}
\]

但不能因此说整篇方法没有创新。

---

# 18. Shibo Li 的 continuous fidelity 与 Kandasamy 的 continuous fidelity

这两篇都说 continuous fidelity，但底层量不一样。

## Shibo Li

底层 numerical control：

\[
\ell\in[\ell_{\min},\ell_{\max}]
\]

mesh / element size 本身近似连续。

所以：

\[
\boxed{\text{底层数值参数本身连续}}
\]

## Kandasamy

\[
N,T\in\mathbb N
\]

实际 query 是整数，

但作者把它们嵌入：

\[
[1,N_\bullet]\times[1,T_\bullet]
\]

连续二维空间建模。

所以：

\[
\boxed{\text{离散执行量的连续统计建模}}
\]

两种 continuous 并不完全相同。

---

# 19. NN width/depth 到底是 search 还是 fidelity

我们专门讨论了这个模糊边界。

## 普通 NAS/HPO

通常：

\[
\boxed{\text{width/depth 属于 architecture/search variables}}
\]

因为最终真的要选：

\[
\text{哪个 width / depth 最好}.
\]

## Penwarden et al. 2022 PINN

**Multifidelity Modeling for Physics-Informed Neural Networks**

这里最终科学目标是：

\[
\boxed{\text{同一个 PDE solution}}
\]

于是：

- small PINN；
- large PINN；
- loose/tight optimization termination；

被作者定义为不同 solver fidelity。

所以：

\[
\text{width/depth}
\]

在这里承担 fidelity role。

## JAHS-Bench-201

这篇更直接：

它把 design space 拆成：

\[
\text{search space}
+
\text{fidelity space}.
\]

并把：

\[
z=(N,W,R,E)
\]

定义成 fidelity：

- depth multiplier；
- width multiplier；
- resolution multiplier；
- epochs。

与此同时 cell topology 等 architecture 仍留在 search space。

所以同一篇论文里：

\[
\boxed{
\text{部分 architecture characteristics 属于 }x,
\quad
\text{另一些属于 }z.
}
\]

这说明：

\[
\boxed{
\text{search/fidelity 划分是 problem-dependent role assignment，不是 universal taxonomy。}
}
\]

---

# 20. Hyperband / FABOLAS / HPO

## Hyperband

resource/fidelity：

- iterations；
- data samples；
- features。

它不建立复杂：

\[
d(x,z)
\]

模型。

直接：

1. 给很多 configuration 少量 budget；
2. 看实际 performance；
3. 淘汰差的；
4. 给好的更多 budget。

所以：

\[
\boxed{\text{accuracy relation 不建模，直接观察}}
\]

## FABOLAS

用：

\[
s=\text{training set size}
\]

作为 fidelity。

同时建模：

\[
\boxed{\text{validation error}}
\]

和：

\[
\boxed{\text{training time}}
\]

所以它很重要：

\[
\boxed{
\text{cost 不一定要假设已知，也可以从数据学。}
}
\]

---

# 21. Neural network training convergence 与 PDE residual

PDE solver：

\[
R(u)=0.
\]

所以：

\[
\|R(u_k)\|
\]

是标准 convergence diagnostic。

NN training：

\[
\min_\theta L(\theta).
\]

实践中通常监控：

- training loss；
- validation loss；
- validation metric；
- fixed epoch budget；
- patience。

理论上 stationarity 可以看：

\[
\|\nabla_\theta L(\theta_k)\|.
\]

但普通 SGD/Adam 深度学习实践里，不常拿 gradient norm 当主要 stopping criterion。

所以对应关系只能是角色类比：

| PDE solver | NN training |
|---|---|
| iteration \(k\) | epoch / SGD step |
| residual norm | training/optimization diagnostic |
| residual tolerance | early stopping / budget criterion |
| QoI | validation metric |
| runtime | GPU time |

不能强行一一对应。

---

# 22. “accuracy” 的两种用法

我们一度把这两个说得太分开，后来修正：

## task performance

例如：

\[
g(z,x)=85\%
\]

validation accuracy。

## approximation error

\[
e(z,x)
=
|g(z,x)-g(z_\bullet,x)|.
\]

这两个不是同一个数，

但非常紧密，因为 approximation error 就是不同 fidelity 下 task performance 的差。

所以：

\[
\boxed{
\text{task performance 是原始输出；approximation accuracy 是相对 target 的误差。}
}
\]

---

# 23. BOCA 中 cost 与 response 的建模

BOCA 最关键的实施细节：

## 23.1 response

\[
g(z,x)
\]

未知。

从 observations：

\[
(z_i,x_i,y_i)
\]

在线更新 GP。

所以：

\[
\boxed{\text{output/fidelity relation 是从数据学的}}
\]

## 23.2 cost

\[
\lambda(z)
\]

被假设为：

\[
\boxed{\text{known cost function}}
\]

不学。

所以 BOCA 的简化是：

\[
\boxed{
g\text{ 未知、要学； }\lambda\text{ 已知、直接给。}
}
\]

## 23.3 真实 solver 更一般

真实 CFD 常常：

\[
c=c(x,z)
\]

因为同样 residual tolerance，

不同 design 收敛难度不一样。

所以：

\[
\boxed{
\text{BOCA 的 }\lambda(z)\text{ 是一个简化。}
}
\]

更一般的框架应该同时学：

\[
g(x,z)
\]

和：

\[
c(x,z).
\]

---

# 24. 这次对话里提到的核心论文清单

下面按时间大致列出。

1. **Kennedy & O’Hagan (2000)**  
   *Predicting the Output from a Complex Computer Code When Fast Approximations Are Available*

2. **Forrester, Bressloff & Keane (2006)**  
   *Optimization Using Surrogate Models and Partially Converged Computational Fluid Dynamics Simulations*

3. **Forrester, Sóbester & Keane (2007)**  
   *Multi-fidelity Optimization via Surrogate Modelling*

4. **Kandasamy et al. (2016 / later JAIR version)**  
   *Gaussian Process Bandit Optimisation with Multi-Fidelity Evaluations*

5. **Klein et al. (2017)**  
   *Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets (FABOLAS)*

6. **Kandasamy et al. (2017)**  
   *Multi-fidelity Bayesian Optimisation with Continuous Approximations (BOCA)*

7. **Poloczek, Wang & Frazier (2017)**  
   *Multi-Information Source Optimization*

8. **Branke et al. (2017)**  
   *Efficient Use of Partially Converged Simulations in Evolutionary Optimization*

9. **Li et al. / Hyperband (JMLR 2018)**  
   *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*

10. **Peherstorfer, Willcox & Gunzburger (2018)**  
    *Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization*

11. **Meng & Karniadakis (2020)**  
    *A Composite Neural Network that Learns from Multi-Fidelity Data*

12. **Trofimov et al. (2020)**  
    *Multi-fidelity Neural Architecture Search with Knowledge Distillation*

13. **Shibo Li et al. (2020)**  
    *Multi-Fidelity Bayesian Optimization via Deep Neural Networks*

14. **Takeno et al. (2020)**  
    *Multi-fidelity Bayesian Optimization with Max-value Entropy Search and its Parallelization*

15. **Shibo Li et al. (2021)**  
    *Batch Multi-Fidelity Bayesian Optimization with Deep Auto-Regressive Networks*

16. **Penwarden et al. (2022)**  
    *Multifidelity Modeling for Physics-Informed Neural Networks*

17. **Shibo Li et al. (2022)**  
    *Deep Multi-Fidelity Active Learning of High-Dimensional Outputs*

18. **Shibo Li et al. (2022)**  
    *Infinite-Fidelity Coregionalization for Physical Simulation*

19. **Bansal et al. (2022)**  
    *JAHS-Bench-201*

20. **Bischl et al. (2023)**  
    *Hyperparameter Optimization: Foundations, Algorithms, Best Practices, and Open Challenges*

21. **Courrier, Boucard & Soulier (2014)**  
    *The Use of Partially Converged Simulations in Building Surrogate Models*

22. **Chen et al.**  
    *Multi-Fidelity Simulation Modeling for Discrete Event Simulation: An Optimization Perspective*

23. **Enhancing Electric Vehicle Charging Station Design Using Multifidelity Simulations (2026)**

24. **Multi-Fidelity Benders Decomposition for Generation, Storage, and Transmission Expansion Planning (2026)**

25. **Fernández-Godino review of multi-fidelity models**

---

# 25. 这些论文如何处理“fidelity → target accuracy”

这是整个对话最后形成的最关键分类。

## A. 直接给 deterministic error bound

代表：

### Kandasamy MF-GP-UCB

\[
\|f^{(M)}-f^{(m)}\|_\infty
\le
\zeta^{(m)}.
\]

即：

\[
\boxed{\text{直接假设一个已知全局上界}}
\]

---

## B. 用 GP / probabilistic model 从数据学习

代表：

- Kennedy–O’Hagan；
- BOCA；
- Forrester 2007 co-kriging；
- MISO。

即：

\[
\boxed{\text{跨 fidelity relation 从 observation data 学}}
\]

---

## C. 先规定结构，再从数据拟合参数

代表：

### Picheny 2013

先规定：

\[
Y=F+G
\]

以及：

\[
G\to0
\]

和 nonstationary kernel 结构，

然后从数据拟合：

- decay；
- length scale；
- variance。

---

## D. 直接实测 correlation

代表：

### Forrester 2006

跑 partial CFD，

直接比较 partial surrogate 和 fully converged surrogate：

\[
r^2.
\]

---

## E. 用 solver 自身 convergence/error indicator

代表：

### Courrier / LATIN

solver 自带 error indicator 定义当前 accuracy level。

---

## F. Deep model 学跨 fidelity relation

代表：

- Shibo Li 2020；
- Shibo Li 2021；
- DMFAL；
- IFC；
- Meng–Karniadakis。

---

## G. 不建模，直接逐步观察

代表：

### Hyperband

直接：

- 少资源跑；
- 看结果；
- 淘汰；
- 给剩余更多资源。

---

# 26. 这些论文如何处理 cost

同样没有统一答案。

## A. cost 假设已知

代表：

- MF-GP-UCB；
- BOCA；
- 一些 MFBO；
- MISO。

\[
\lambda(z)
\]

直接给算法。

## B. cost 从数据学

代表：

### FABOLAS

同时模型：

\[
\text{validation loss}
\]

和：

\[
\text{training time}.
\]

## C. resource 自己就是 cost proxy

代表：

- Hyperband；
- Picheny；
- Branke。

## D. cost 实测

代表：

- Forrester；
- Courrier；
- PINN MF；
- JAHS；
- EV simulation。

## E. 不显式建 cost

代表：

### IFC 2022

cost 只是 motivation。

---

# 27. 最后形成的“五层”框架

我们最后发现，只分：

\[
\text{design}
-
\text{fidelity}
-
\text{cost}
\]

还不够。

更完整的是：

\[
\boxed{
\underbrace{x}_{\text{design}}
\quad
\underbrace{z}_{\text{fidelity control}}
\quad
\longrightarrow
\quad
\underbrace{s(x,z)}_{\text{solver/training state}}
\quad
\longrightarrow
\quad
\underbrace{y(x,z)}_{\text{QoI / performance}}
}
\]

同时：

\[
\boxed{
c(x,z)=\text{cost}.
}
\]

对于 iterative PDE：

\[
z=k
\]

时：

\[
s=(u_k,r_k),
\]

\[
y=Q(u_k),
\]

\[
c=t_k.
\]

如果 fidelity control 是 tolerance：

\[
z=\epsilon,
\]

则：

\[
\epsilon
\rightarrow
K(x,\epsilon)
\rightarrow
(r_K,y_K,c_K).
\]

这个框架解释了为什么：

- residual tolerance；
- iteration；
- current residual；
- wall-clock time；

高度相关，但角色不同。

---

# 28. 整段对话最后的统一结论

整个 multi-fidelity literature 没有一个统一规定说：

> 什么量必须是 fidelity。

真正稳定的结构是：

\[
\boxed{
\text{有一个 target evaluation}
}
\]

以及：

\[
\boxed{
\text{一些更便宜、对 target 有信息的 evaluations}
}
\]

然后不同文章会选择不同变量来控制这些 approximations：

- mesh；
- timestep；
- solver tolerance；
- iterations；
- computational time；
- epochs；
- data size；
- width/depth；
- horizon；
- source ID。

真正决定论文方法差异的，是下面两件事：

\[
\boxed{
\text{low fidelity 到 target 的关系怎么建模？}
}
\]

以及：

\[
\boxed{
\text{evaluation cost 怎么得到？}
}
\]

有的：

- 硬假设；
- 理论上界；
- GP；
- deep network；
- empirical correlation；
- solver error indicator；
- direct observation；
- learned cost model；
- measured runtime。

所以 fidelity 本身只是一个角色，方法真正的核心是：

\[
\boxed{
\text{如何利用 cheap information，在有限 cost 下推断/优化 target fidelity。}
}
\]

---

# 29. 一张总表

| 论文 | design/input | fidelity | target relation | cost | 任务 |
|---|---|---|---|---|---|
| Kennedy–O’Hagan 2000 | \(x\) | 离散 simulator sophistication | GP 学 | implicit relative cost | prediction |
| Forrester 2006 | geometry \(x\) | iterations | empirical correlation | iterations + CPU | optimization |
| Forrester 2007 | design \(x\) | discrete model levels | co-kriging | level cost | optimization |
| Picheny 2013 | pipe design \(x\) | computational time \(t\) | structured nonstationary GP | \(t\) | prediction / optimization support |
| Courrier 2014 | structural design | LATIN convergence level | error indicator + HF correction | measured CPU | surrogate / optimization |
| Branke 2017 | candidate \(x\) | run length | learned from partial fitness | run length | evolutionary optimization |
| Kandasamy 2016 | \(x\) | \(m=1,\dots,M\) | uniform bound \(\zeta_m\) | known \(\lambda_m\) | BO |
| BOCA 2017 | \(x\) | continuous \(z\) | GP online learning | known \(\lambda(z)\) | BO |
| FABOLAS 2017 | HPO config | dataset size | GP validation loss | GP training time | HPO |
| Hyperband 2018 | HPO config | resource budget | direct observation | resource | HPO |
| MISO 2017 | \(x\) | source ID | GP bias/noise | source cost | optimization |
| Shibo Li 2020 | \(x\) | discrete fidelity | DNN learned relation | fidelity cost | BO |
| Shibo Li 2021 | \(x\) | discrete fidelity | deep autoregressive | fidelity cost | batch BO |
| DMFAL 2022 | physical input \(x\) | simulation fidelity | deep learned | benefit/cost | active learning |
| IFC 2022 | PDE input \(x\) | continuous mesh-derived \(m\) | neural ODE / GP | no explicit cost model | surrogate |
| PINN MF 2022 | PDE input | width/depth/termination | numerical validation | measured cost | PDE surrogate |
| JAHS 2022 | search config | depth/width/resolution/epoch | empirical benchmark | empirical cost | benchmark |
| Meng–Karniadakis 2020 | \(x\) | LF/HF data source | NN learned correlation | implicit | data fusion |
| EV station 2026 | station design | horizon / timestep / segmentation | statistical fusion | per-sample cost | design selection |
| MF Benders 2026 | planning design | horizon length | relaxation / bounds | solve cost | optimization |

---

# 30. 最简洁的阅读模板

以后遇到任何 multi-fidelity 论文，直接问：

\[
\boxed{
1.\;x\text{ 是什么？}
}
\]

\[
\boxed{
2.\;z\text{ 是什么？}
}
\]

\[
\boxed{
3.\;z_\star\text{ 是什么？}
}
\]

\[
\boxed{
4.\;g(x,z)\text{ 怎么得到？}
}
\]

\[
\boxed{
5.\;g(x,z)\text{ 和 }g(x,z_\star)\text{ 的关系是怎么建模的？}
}
\]

是：

- 硬假设；
- deterministic bound；
- GP；
- DNN；
- empirical correlation；
- solver indicator；
- direct observation？

再问：

\[
\boxed{
6.\;c(x,z)\text{ 怎么得到？}
}
\]

是：

- 已知公式；
- complexity proxy；
- measured wall-clock；
- resource budget；
- learned surrogate；
- 根本不建模？

这六个问题基本能把任何 multi-fidelity 文章的真正结构看清楚。

---

# 31. 本次对话最终最重要的一句话

\[
\boxed{
\text{fidelity 不是一个固定物理量；它是“相对于某个 target evaluation，控制 cheaper approximation 的角色”。}
}
\]

而真正决定不同论文差异的，是：

\[
\boxed{
\text{fidelity–target information relation}
}
\]

和：

\[
\boxed{
\text{fidelity–cost relation}
}
\]

到底怎么定义、假设、测量或学习。
