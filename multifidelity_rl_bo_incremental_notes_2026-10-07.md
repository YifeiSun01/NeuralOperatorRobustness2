# Multi-Fidelity / Active Search / Bayesian Optimization / Reinforcement Learning：增量讨论整理

> 日期：2026-10-07  
> 范围：这是对之前已经保存的 26/36 篇 multi-fidelity 文献整理的**增量更新**。  
> 本文件只记录后续新讨论：RL 与 Bayesian decision / dynamic programming / rollout 的边界、Fang 2017 RL Active Learning、EARL-BO、连续 state/action 下的 sample complexity、36 篇论文的 decision mechanism 重新分类、cost / response surrogate 的建模方式，以及 IFC / Neural ODE 等模型在 multi-fidelity 中的角色。  
> 已经在旧文件中完整整理过的 design space / fidelity space / cost / discrepancy 基本框架不重复展开。

---

# 1. 最核心的新结论：有 Policy、Bellman、Rollout，不等于 Reinforcement Learning

后续讨论里最重要的澄清，是把下面这些概念严格区分开：

[
	ext{MDP},
qquad
	ext{belief state},
qquad
	ext{policy},
qquad
	ext{Bellman equation},
qquad
	ext{backward induction},
qquad
	ext{rollout},
qquad
	ext{dynamic programming},
qquad
	ext{reinforcement learning}.
]

它们不是同义词。

Policy 只是一个从状态到动作的映射：

[
pi:Sightarrow A
]

或者随机策略：

[
pi(amid s).
]

因此只要是 sequential decision problem，都可以有 policy。Bayesian decision theory、dynamic programming、POMDP、optimal control、reinforcement learning 全都可以使用 policy 这个词。

同理，Bellman recursion 也不是 RL 专属。若 transition 和 reward model 已知，可以直接解：

[
V^star(s)
=
max_a
mathbb E
left[
r(s,a,S')+gamma V^star(S')
ight].
]

用 backward induction、value iteration、policy iteration、tree search 或 rollout 求解，这通常属于：

[
oxed{	ext{dynamic programming / planning}}
]

而不是自动成为 reinforcement learning。

---

# 2. 真正更接近 RL 的分界：Decision Function 是不是从 Experience 中学出来

一个非常有用的实用分界是：

[
oxed{
	ext{是否从 }(s_t,a_t,r_t,s_{t+1})	ext{ 经验中更新一个可复用的 decision function}
}
]

典型需要学习的对象包括：

[
V_	heta(s),
qquad
Q_	heta(s,a),
qquad
pi_	heta(amid s),
]

或者在 model-based RL 中学习：

[
hat P_	heta(s',rmid s,a).
]

关键并不是“有没有 (Q)”或者“有没有 (pi)”这个符号，而是这些未知函数是不是通过 interaction / logged trajectories 被估计出来。

例如 tabular Q-learning：

[
Q(s_t,a_t)
leftarrow
Q(s_t,a_t)
+
alpha
left[
r_t+gammamax_{a'}Q(s_{t+1},a')
-
Q(s_t,a_t)
ight].
]

这里没有给定最优标签：

[
a_t^star.
]

RL 只看到：

[
(s_t,a_t,r_t,s_{t+1}),
]

然后通过行为产生的长期后果，逐渐判断哪些 action 更好。

所以 RL 不要求事先知道 optimal action，也不要求训练数据本身来自 optimal trajectories。

---

# 3. Supervised Learning、Imitation Learning 与 RL 的区别

如果数据直接给出：

[
(s,a^star),
]

也就是告诉模型：

> 在这个 state 下，正确 action 就是这个。

然后训练：

[
pi_	heta(s)approx a^star,
]

这更接近：

[
oxed{	ext{supervised learning / behavior cloning / imitation learning}}
]

而典型 RL 没有这个 action label。

RL 数据更像：

[
(s,a,r,s').
]

它通过 reward consequences 自己判断：

[
Q(s,a)
]

应该变高还是变低。

---

# 4. “学 Value”到底是什么意思

定义 return：

[
G_t
=
r_t+gamma r_{t+1}+gamma^2r_{t+2}+cdots.
]

State value：

[
V^pi(s)
=
mathbb E_pi[G_tmid S_t=s].
]

含义：

> 现在处于 (s)，以后按 policy (pi) 行动，长期累计收益大概是多少？

Action value：

[
Q^pi(s,a)
=
mathbb E_pi[G_tmid S_t=s,A_t=a].
]

含义：

> 当前 state 是 (s)，现在先做 (a)，以后按 policy 行动，最终累计收益大概是多少？

在 resumable solver Active Search 中，可以理解为：

[
Q(S,(x_i,Delta z))
]

表示：

> 当前所有候选都有各自 solver progress 的情况下，现在把候选 (x_i) 再推进 (Delta z)，未来继续合理分配预算以后，最终预计能确认多少 positive？

Value function 的意义，就是把很长的未来：

[
a_t
ightarrow
Y_t
ightarrow
a_{t+1}
ightarrow
Y_{t+1}
ightarrowcdots
]

压缩成一个可估计的长期价值。

---

# 5. MF-ENS 为什么有 Bellman / Policy / Rollout，但仍然通常不算 RL

*Nonmyopic Multifidelity Active Search* 的结构可以写成：

[
D_t
ightarrow
a_t
ightarrow
y_t
ightarrow
D_{t+1}.
]

它确实：

- 使用 Bayesian decision theory；
- 定义 Bayesian optimal policy；
- 使用 Bellman-style backward induction；
- 实际提出 approximate dynamic programming 下的 two-stage rollout policy。

但是它依赖已有 probabilistic classifier，例如：

[
P(y_H=1mid x,D),
qquad
P(y_L=1mid x,D),
]

然后在当前 state 下在线计算未来可能结果。

其核心是：

[
oxed{
	ext{current belief model}
ightarrow
	ext{online rollout / backward induction}
ightarrow
	ext{current action}
}
]

而不是：

[
oxed{
	ext{many past episodes}
ightarrow
Q_	heta/V_	heta/pi_	heta
ightarrow
	ext{reusable learned policy}
}
]

所以 MF-ENS 更准确的分类是：

[
oxed{
	ext{Bayesian sequential decision}
+
	ext{approximate dynamic programming / rollout}
}
]

而不是标准意义上的 reinforcement learning。

MF-ENS 的 32 / 64 个未来分支，属于当前 state 下“现算”的 planning，而不是把很多过去 planning 结果学成一个可复用的 policy。

---

# 6. Planning 与 RL 的关键对照

可以用下表概括：

| 项目 | Bayesian Planning / DP / Rollout | Reinforcement Learning |
|---|---|---|
| 有 state | 可以有 | 有 |
| 有 action | 有 | 有 |
| 有 policy | 有 | 有 |
| 有 Bellman equation | 可以有 | 经常有 |
| 有 long-term reward | 可以有 | 经常有 |
| 有 rollout | 可以有 | 也可以有 |
| 必须神经网络 | 否 | 否 |
| policy/value 来源 | 当前利用 model 在线计算 | 从 experience 学 |
| 是否跨 episode 积累 decision knowledge | 通常不需要 | 通常需要 |
| 决策时是否一定不展开树 | 否 | 否 |

特别注意：

[
oxed{
	ext{RL 可以和 tree search 同时存在}
}
]

例如 AlphaZero 就是 learned policy/value + MCTS。

所以“展开树就不是 RL”这个判断不成立。真正的区别是有没有从 experience 中学习可复用的 policy/value/model。

---

# 7. RL 不等于 Deep RL，也不要求 Neural Network

小 MDP 可以直接存一张：

[
Q(s,a)
]

表格。

因此：

[
oxed{	ext{RL}
eq	ext{Deep RL}}
]

当 state/action 是大规模连续空间时，tabular 方法才失去现实意义，需要 function approximation：

[
Q_	heta(s,a),
qquad
V_	heta(s),
qquad
pi_	heta(amid s).
]

函数逼近器可以是：

- neural network；
- linear basis；
- kernel；
- tree；
- GP；
- 其他参数化模型。

现代连续控制通常使用神经网络，只是因为它在高维连续空间里比较方便。

---

# 8. 连续 State + 连续 Action 为什么明显更难

如果：

[
sinmathbb R^{d_s},
qquad
ainmathbb R^{d_a},
]

你不可能把所有 ((s,a)) 枚举成表格。

必须依赖函数泛化：

[
Q_	heta(s,a)
]

或者：

[
a=pi_	heta(s).
]

连续 state 本身已经要求 function approximation，但如果 action 还是有限集合，例如：

[
ain{0,1},
]

DQN 还可以直接输出两个 Q-value：

[
Q_	heta(s,0),
qquad
Q_	heta(s,1).
]

一旦 action 也连续：

[
ainmathbb R^{d_a},
]

就不能枚举：

[
argmax_a Q(s,a).
]

常见做法是再训练 actor：

[
a=pi_phi(s).
]

PPO、SAC、TD3、DDPG 等 continuous-control RL 都属于这一方向。

因此：

[
oxed{
	ext{continuous state + continuous action}
}
]

通常比：

[
oxed{
	ext{continuous state + finite action}
}
]

更难、更依赖大量覆盖良好的 experience。

---

# 9. Fang et al. 2017：RL Active Learning 的关键作用

论文：

**Learning how to Active Learn: A Deep Reinforcement Learning Approach**

它解决的 outer problem 和传统 Active Learning 一样：

> 在 annotation budget 下选择哪些数据标注，使最终 supervised classifier 尽可能准确。

传统 Active Learning 会手工规定：

[
	ext{score}(x)=	ext{uncertainty}(x)
]

或者 expected error reduction，然后：

[
x^star=argmax_x 	ext{score}(x).
]

Fang 的核心改变是：

[
oxed{
	ext{让 learned policy 取代固定 query heuristic}
}
]

---

# 10. Fang 的 State 是连续的，但 Action 被刻意简化为二元

Fang 的 state 是连续向量，包含：

- 当前 candidate sentence 的内容 representation；
- 当前 NER 模型的 predictive marginals；
- 当前模型 confidence。

因此可以写成：

[
s_iinmathbb R^d.
]

但 action 极其简单：

[
a_iin{0,1}.
]

其中：

[
a_i=1
]

表示标注当前句子；

[
a_i=0
]

表示跳过。

论文原文确实明确采用 streaming assumption，并说这样“permits simpler learning”。

需要严谨区分：

- “streaming setup 让 learning simpler”是论文明确说的；
- “pool-based 会造成更大 action space”是根据其 MDP formulation 的直接解释，不是逐字原文。

如果是 pool-based Active Learning，当前需要从：

[
{x_1,ldots,x_N}
]

中选择一个：

[
a_t=x_i,
]

则 action cardinality 可能是几千、几万，并且还随 pool 缩小而变化。

Fang 用 streaming 把每一步简化成：

[
oxed{
	ext{当前这一个 candidate：要 / 不要？}
}
]

---

# 11. Fang 的 Reward 是怎么得到的

Reward 定义为 held-out performance 的变化：

[
r_i
=
F1(phi_i)-F1(phi_{i-1}).
]

其中：

- (phi_{i-1})：加入当前句子之前的 NER 模型；
- (phi_i)：若 action=1，则把当前句子加入 labelled set 后更新得到的模型；
- F1 在固定的 dev / held-out labelled set 上计算。

关键点：

[
oxed{
	ext{它不做两个反事实分支同时比较}
}
]

它不会同时训练：

[
	ext{加 }x_i
]

和：

[
	ext{不加 }x_i
]

两个模型。

它只执行当前实际 action。

如果：

[
a_i=1,
]

则：

[
D_l
leftarrow
D_lcup{(x_i,y_i)},
]

更新：

[
phi_i=operatorname{Update}(D_l),
]

然后计算新的 F1。

如果：

[
a_i=0,
]

模型不更新，reward 通常就是 0。

RL 通过不同 episodes 的：

[
(s,a,r,s')
]

经验，逐渐估计：

[
Q_	heta(s,0),
qquad
Q_	heta(s,1).
]

---

# 12. Fang 真的是 10,000 Episodes，而且计算量很大

论文实验设置：

[
N=10{,}000	ext{ episodes},
]

annotation budget：

[
B=200	ext{ sentences per episode}.
]

所以如果每个 episode 用满 200 个 accepted annotations，数量级是：

[
10{,}000	imes200
=
2{,}000{,}000
]

次 accepted-label model updates。

这不等于 200 万个 unique sentences。

它反复复用一个 fully labeled source-language corpus：

1. shuffle corpus；
2. 暂时隐藏 labels；
3. RL policy 逐句决定是否 query；
4. query 时 reveal 已存在的 label；
5. 更新 classifier；
6. 在 held-out set 测 F1；
7. 计算 reward；
8. 存入 replay memory；
9. 下一个 episode 重置并重新玩。

而且 skip action 也产生 transition，所以真实：

[
(s,a,r,s')
]

数量可能超过 200 万。

---

# 13. Fang 官方代码确认：每接受一个句子，会重新拟合当前 CRF 并测 Dev F1

官方代码中主要的 CRF implementation 逻辑是：

[
a_i=1
]

以后：

[
D_l^{(j)}
=
D_l^{(j-1)}
cup
{(x_j,y_j)}.
]

然后调用：

[
	ext{CRF.train}(D_l^{(j)})
]

再调用：

[
	ext{CRF.test}(D_{m dev})
]

得到 F1。

CRF 训练代码每次新建一个 'pycrfsuite.Trainer'，把当前已选的全部 labelled sentences 加进去，并设置最多 50 optimization iterations。

所以对 CRF 主实现而言，确实近似：

[
oxed{
	ext{每新增一个 accepted sentence}
ightarrow
	ext{重新拟合当前 CRF}
ightarrow
	ext{重新跑 dev set}
}
]

这意味着数量级上约：

[
oxed{
2	ext{ million CRF fits}
+
2	ext{ million dev evaluations}
}
]

当然，这里的 CRF 非常轻量，而且每个 episode 最多只积累 200 个句子，因此单次 training 比现代 Transformer / PDE solver 小很多。

论文没有报告明确的：

- GPU 型号；
- CPU 型号；
- RAM；
- wall-clock training time。

所以不能从论文中给出准确训练小时数。

---

# 14. CRF 是什么

CRF：

[
oxed{	ext{Conditional Random Field}}
]

中文常译“条件随机场”。

它与 Random Forest 完全不同。

- C = Conditional；
- R = Random；
- F = Field。

Fang 主要用 linear-chain CRF 做 NER sequence labeling。

输入：

[
x=(x_1,ldots,x_T)
]

是一串 tokens；

输出：

[
y=(y_1,ldots,y_T)
]

是一串标签，例如：

[
(	ext{PER},O,O,	ext{LOC},	ext{LOC}).
]

典型 score：

[
operatorname{score}(x,y)
=
sum_t w^	op f(x,t,y_t)
+
sum_t A_{y_{t-1},y_t}.
]

然后：

[
P(ymid x)
=
rac{
exp(operatorname{score}(x,y))
}{
sum_{y'}exp(operatorname{score}(x,y'))
}.
]

“Field”只是概率图模型里的术语，不代表“大模型”。

Fang 官方代码的 feature 包括：

- lower-case word；
- suffix；
- upper/title/digit；
- 前后词等。

所以它是一个轻量的传统 sequence model，而不是现代巨型 neural model。

---

# 15. Fang 为什么能承受这种重训练

它能够做 10,000 episodes，依赖几个非常有利的条件：

[
oxed{
	ext{已有 fully-labeled corpus}
}
]

所以每次 query 只是 reveal label，不需要真人重新标注。

其次：

[
oxed{
	ext{inner model 是小型 linear-chain CRF}
}
]

不是训练一次就需要几十分钟或几个小时的深度网络。

它本质是在：

[
oxed{
	ext{用大量计算}
ightarrow
	ext{换取未来更少的真实 annotation}
}
]

这也是它与 expensive numerical solver 场景差别最大的地方。

---

# 16. 为什么 Fang 的具体方法不能直接解决连续 Design Optimization

Fang 的 candidate (x_i) 已经存在于 corpus 中，而且对应真实 label (y_i) 也早已存在，只是模拟时隐藏。

所以：

[
oxed{
	ext{agent 只能在已有 candidate stream 中选择}
}
]

如果真实 design space 是：

[
xin[0,1]^d,
]

RL actor 可能输出：

[
x=(0.137,0.926,ldots),
]

这个点数据库里根本不存在。

这时必须回答：

[
oxed{
	ext{这个新 }x	ext{ 的 transition 和 reward 从哪里来？}
}
]

只有三种基本来源：

1. 真正运行昂贵 solver；
2. 从历史数据库 / offline trajectories 获取；
3. 用 surrogate / simulator 产生虚拟结果。

如果没有便宜 simulator 或历史数据，连续设计空间中的直接 RL 会迅速碰到 sample complexity 问题。

---

# 17. BO 与 RL 的典型使用情境存在天然张力

经典 Bayesian Optimization 的出发点通常是：

[
oxed{
	ext{一次真实 evaluation 非常昂贵}
}
]

例如 CFD、材料实验、hyperparameter training、实验室测试。

因此只能承受：

[
20,;50,;100
]

这种数量级的真实 evaluations。

BO 的哲学是：

[
oxed{
	ext{真实 interaction 很贵，所以每一个 observation 都要尽量榨干}
}
]

典型 Deep RL 则更习惯：

[
oxed{
	ext{可以获得大量 interaction，于是用大量 experience 学 reusable policy}
}
]

所以：

[
oxed{
	ext{BO：few expensive interactions}
}
]

[
oxed{
	ext{Deep RL：many relatively cheap interactions}
}
]

两者不是理论上的互斥方法，但典型 cost regime 确实有很强的张力。

---

# 18. 为什么 RL 对“重复出现的任务”更有价值

如果只解决一个昂贵优化实例：

[
f(x)
]

一次，花巨大代价训练：

[
pi_	heta
]

可能根本不划算。

如果未来有：

[
mathcal T_1,ldots,mathcal T_N
]

很多相似 sequential decision tasks，则训练成本可以摊薄：

[
C_{m per-task}
=
rac{C_{m train}}{N}
+
C_{m inference}.
]

所以 RL 更有吸引力的场景通常是：

[
oxed{
	ext{同一类决策问题会被重复执行很多次}
}
]

而 BO 更常见于：

[
oxed{
	ext{one-off expensive optimization}
}
]

---

# 19. EARL-BO：BO 与 RL 的真正结合方式

论文：

**EARL-BO: Reinforcement Learning for Multi-Step Lookahead, High-Dimensional Bayesian Optimization**

它最值得注意的地方是：

[
oxed{
	ext{它不是预训练一次 RL policy，然后整个 BO 都免费使用}
}
]

更准确的流程是：

[
D_k
ightarrow
	ext{fit GP}_k
ightarrow
	ext{在当前 GP 虚拟环境中训练 RL}
ightarrow
x_{k+1}
ightarrow
f_{m true}(x_{k+1})
ightarrow
D_{k+1}.
]

下一次：

[
D_{k+1}
ightarrow
	ext{fit GP}_{k+1}
ightarrow
	ext{再做一轮 RL training}
ightarrow
x_{k+2}.
]

所以：

[
oxed{
	ext{RL training 本身就是每个 BO decision 的在线计算过程的一部分}
}
]

它更像：

[
oxed{
	ext{把显式 scenario-tree optimization}
ightarrow
	ext{换成当前 GP 世界里的 virtual RL training}
}
]

---

# 20. EARL-BO 中的 400 不是 400 个真实 BO Points

Algorithm 1 中：

[
	ext{max episodes}=4000
]

其中前：

[
400
]

个是 off-policy virtual episodes。

这 400 个并不是：

- 400 个真实 black-box evaluations；
- 400 个真实 BO iterations。

而是**当前一个 BO iteration 内部**的虚拟 episodes。

流程：

[
D_k
ightarrow
GP_k
]

然后前 400 episodes：

[
a_t=	ext{TuRBO acquisition action}
]

但 observation 仍然来自：

[
	ilde ysim GP_k.
]

这些虚拟 trajectories 用来 warm-start actor/critic。

之后：

[
e=401,ldots,4000
]

主要切到 PPO on-policy learning。

最后才让训练好的 current actor 选：

[
x_{k+1}
]

进行一次真实 black-box evaluation。

---

# 21. TuRBO 在 EARL-BO 中扮演什么角色

TuRBO 是传统 BO heuristic / acquisition-based policy。

在 EARL-BO 中，TuRBO 被当成早期 teacher / warm-start generator：

[
oxed{
	ext{TuRBO virtual trajectories}
ightarrow
	ext{initialize actor / critic}
}
]

避免 PPO 一开始完全随机乱撞。

之后 PPO 自己在 GP virtual environment 中继续改进 policy。

因此可以理解为：

[
oxed{
	ext{BO heuristic warm start}
ightarrow
	ext{RL refinement}
}
]

---

# 22. GP-UCB 与 EARL-BO 的本质区别：GP 扮演的角色不同

传统 GP-UCB：

[
D_k
ightarrow
GP_k
ightarrow
mu_k(x),sigma_k(x)
]

然后直接规定：

[
A_{m UCB}(x)
=
mu_k(x)+etasigma_k(x).
]

最后：

[
x_{k+1}
=
argmax_x A_{m UCB}(x).
]

这里：

[
oxed{
	ext{GP 直接服务于一个 hand-designed acquisition}
}
]

EARL-BO 也先 fit GP，但接下来：

[
oxed{
	ext{GP 被当成 virtual world model}
}
]

RL actor 选：

[
x_{t+1},
]

然后 GP posterior 提供 fantasy observation：

[
	ilde y_{t+1}
sim
p(f(x_{t+1})mid D_t).
]

再形成虚拟下一状态：

[
D_{t+1}
=
D_tcup{(x_{t+1},	ilde y_{t+1})}.
]

重复有限 horizon：

[
D_t
ightarrow
x_{t+1}
ightarrow
	ilde y_{t+1}
ightarrow
D_{t+1}
ightarrowcdots.
]

所以：

[
oxed{
	ext{GP-UCB：GP + 人工规定的一步 rule}
}
]

[
oxed{
	ext{EARL-BO：GP + virtual trajectories + learned multi-step policy}
}
]

---

# 23. EARL-BO 不是“便宜方法”，它只是比显式 Multi-Step Rollout 更可扩展

论文报告 8D Ackley 上，计算下一 BO query 的平均决策时间大致是：

| 方法 | 每个 BO iteration 的 decision time |
|---|---:|
| EI | 0.28 s |
| TuRBO | 0.27 s |
| SAASBO | 168.6 s |
| 3-step Rollout EI | (>3600) s |
| EARL-BO 3-step | 840 s |
| EARL-BO 5-step | 1075 s |

所以：

[
0.28	ext{s}
quad	ext{vs}quad
840	ext{s}
]

差约 3000 倍。

EARL-BO 的卖点不是：

[
oxed{	ext{比 myopic BO 快}}
]

而是：

[
oxed{
	ext{比真正的显式 multi-step rollout 更可承受}
}
]

也就是说两者都在支付“考虑未来”的计算费，只是计算组织方式不同。

---

# 24. EARL-BO 的计算资源

论文实验环境报告：

[
16	ext{ CPUs}
]

和最多：

[
100	ext{ GB memory}
]

每个 experiment 平均约：

[
25	ext{ hours}.
]

需要注意：

- 100 GB 是 maximum memory allocation，不等于实际一直占满；
- 论文没有说 sequential decision 天生不需要 GPU；
- 作者明确提到 GPU acceleration 可能进一步加速；
- 这里 NN 本身不大，计算还包含 GP fitting/sampling、PPO、多 episodes、多个重复实验等。

所以不能得出：

[
	ext{“RL 不需要 GPU”}
]

这种结论。

---

# 25. GP 当 Virtual Environment 的最大风险：Model Bias / Planning Delusion

EARL-BO 的核心假设是：

[
fmid D_k
sim
GP	ext{ posterior}.
]

然后从 posterior sampling：

[
	ilde f^{(j)}
sim
p(fmid D_k)
]

生成虚拟世界。

Bayesian 逻辑上这是合理的：当前 posterior 表示你认为可能的真实函数。

但真实函数：

[
f_{m true}
]

不一定真的来自：

[
GP	ext{ posterior}.
]

如果 GP 错：

[
oxed{
	ext{RL 会非常认真地学会如何在一个错误世界里做最优决策}
}
]

更严重的是 multi-step 会复合误差：

[
	ext{GP error}
ightarrow
	ilde D_{t+1}	ext{ 错}
ightarrow
	ext{下一步预测继续错}
ightarrow
cdots.
]

EARL-BO 讨论这种现象时使用：

[
oxed{	ext{planning delusion}}
]

这个概念。

---

# 26. 19D HPO：50 Initial Points 与 5-Point Ablation

主实验中的 19D HPO 使用：

[
oxed{50	ext{ initial points}}
]

作者另做 ablation：

[
oxed{19D:;50ightarrow5}
]

也就是只给 5 个 initial observations。

这时：

[
	ext{19D space}
+
	ext{5 observations}
]

极度稀疏。

结果：

[
	ext{poor GP}
ightarrow
	ext{posterior samples variation large}
ightarrow
	ext{multi-step planning suboptimal}.
]

所以“性能受到明显影响”指的是：

[
oxed{	ext{19D + 5 initial points}}
]

而不是 50-point 主设置。

这个实验非常直接地说明：

[
oxed{
	ext{RL 的 planning quality 被 GP world model 的质量限制}
}
]

RL 无法凭空纠正一个它从来没有真实访问权限的错误 virtual world。

---

# 27. 对 Resumable Solver Active Search 的意义

当前 proposed state：

[
S_t
=
{
(x_i,z_i,	ext{trajectory}_i,	ext{observations}_i)
}_{i=1}^{n_t}.
]

action：

[
a_t=(x_i,Delta z_i)
]

或者还允许：

[
a_t=(x,Delta z)
]

开始一个新连续 design。

reward 如果只在 full-fidelity positive 时给：

[
r_t
=
mathbf 1
{
z_i=1,;Y(x_i,1)>	au
},
]

则环境同时具有：

[
oxed{
	ext{high-dimensional / variable-size state}
}
]

[
oxed{
	ext{continuous or hybrid action}
}
]

[
oxed{
	ext{sparse delayed reward}
}
]

[
oxed{
	ext{expensive real interaction}
}
]

这是非常不友好的 model-free deep RL 组合。

因此当前阶段更自然的路线是：

[
oxed{
	ext{probabilistic surrogate}
+
	ext{incremental cost model}
+
	ext{short-horizon rollout / approximate DP}
}
]

先把世界模型和 decision structure 搞清楚，再积累 trajectories。

---

# 28. 什么时候 RL 才开始更值得

如果未来有：

- 大量历史 solver trajectories；
- 可信 surrogate simulator；
- 便宜 synthetic tasks；
- 同一个 policy 会在很多相似任务上重复使用；

那么可以考虑：

[
oxed{
	ext{surrogate environment}
ightarrow
	ext{virtual RL interaction}
}
]

或者：

[
oxed{
	ext{expensive rollout planner}
ightarrow
	ext{high-quality trajectories}
ightarrow
	ext{policy distillation / imitation}
}
]

再进一步做 model-based RL。

直接让 PPO / SAC 在真实昂贵 PDE solver 上从零探索，当前并不是优先选择。

---

# 29. RL 适合什么问题，不适合什么问题

更适合 RL 的条件：

- action 会真正影响 future state；
- long-horizon consequences 很强；
- 同类决策任务反复出现；
- 有大量 cheap simulator / history；
- reward 定义稳定；
- policy/value 能跨任务复用。

不友好的条件：

- 只解决一次；
- 每次 interaction 极贵；
- 没有 historical trajectories；
- 没有可信 simulator；
- state/action 高维连续；
- sparse / delayed reward；
- 不能允许大量探索失败；
- 环境变化太快，旧 experience 不可复用。

因此当前 solver 问题：

[
oxed{
	ext{很适合写成 sequential decision / belief-MDP}
}
]

但：

[
oxed{
	ext{不适合直接在真实环境上做 model-free deep RL}
}
]

---

# 30. 36 篇文献重新按 Decision Mechanism 分类

对之前 26 篇核心 + 后续约 10 篇扩展文献重新检查以后，可以得到一个更明确的结论。

最常见的 action mechanism 是：

[
oxed{
	ext{hand-designed / analytically specified acquisition, allocation, selection rule}
}
]

典型包括：

- UCB；
- EI；
- LCB；
- MES；
- KG；
- information gain / cost；
- ranking-reversal probability；
- promotion / elimination；
- resource allocation；
- fidelity-switching rule。

这些规则有的有 regret bound 或信息论推导，因此“heuristic”不能理解成“拍脑袋”，更准确是：

[
oxed{
	ext{theory-motivated hand-designed decision rule}
}
]

---

# 31. 36 篇里真正 Explicit Multi-Step / Rollout 的很少

重新核对后，明确属于 multi-step / non-myopic rollout / scenario-tree 的主要是：

[
oxed{
	ext{MF-ENS}
}
]

[
oxed{
	ext{MICRO}
}
]

[
oxed{
	ext{One-Shot Multi-Step BO}
}
]

MISO / misoKG 和 Active Area Search 更接近：

[
oxed{	ext{one-step lookahead}}
]

而不是完整 multi-stage rollout。

因此整体判断是：

[
oxed{
	ext{绝大多数 sequential papers 仍然使用 current-state acquisition/allocation rule}
}
]

而不是显式求 finite-horizon optimal policy。

---

# 32. 36 篇中没有标准 RL Decision-Policy Learning

在这套文献里，没有发现核心方法属于：

[
(s,a,r,s')
ightarrow
Q_	heta/V_	heta/pi_	heta
]

这种：

- Q-learning；
- DQN；
- PPO；
- actor-critic；
- policy gradient；

式的 RL policy learning。

所以：

[
oxed{
	ext{RL decision-policy learning = 0 / 36}
}
]

这不代表整个 BO / Active Learning / Active Search 领域没有 RL。

领域外扩展反例包括：

- Fang 2017 RL Active Learning；
- EARL-BO 2025 RL for multi-step BO。

只是它们不属于此前那 36 篇固定文献集合。

---

# 33. 这批论文如何预测 Cost：大部分根本没有学 (C(x,z))

之前逐篇核过的 26 篇核心文献里，绝大多数 cost 处理属于：

[
oxed{
	ext{given / fixed / measured}
}
]

常见形式：

[
C(z_m)=lambda_m.
]

例如：

- MF-GP-UCB：每 fidelity cost 已知常数；
- BOCA：给定 (lambda(z))；
- DNN-MFBO：每级 (lambda_m)；
- BMBO-DARN：每级常数；
- DMFAL：每级常数；
- MF-MES：每级常数；
- Hyperband：直接设定 resource budget；
- NAS-KD：固定训练 protocol / 记录耗时；
- Multi-fidelity PINN：实测训练时间；
- Benders：不训练 runtime predictor。

因此：

[
oxed{
	ext{cost 通常是 decision rule 的输入，而不是需要学习的未知函数}
}
]

少数明显例外：

### FABOLAS

一个 GP 学 validation loss：

[
f(x,s)
]

另一个 GP 学：

[
log C(x,s).
]

### JAHS-Bench-201

用大量历史数据分别训练 XGBoost 预测：

- accuracy；
- loss；
- runtime 等。

所以：

[
oxed{
	ext{learned cost model 是少数，不是主流}
}
]

---

# 34. 这批论文如何预测结果 (Y(x,z))：主流是低样本 Surrogate

大量论文使用：

[
oxed{
	ext{GP / kriging / co-kriging}
}
]

直观上就是：

> 已知少量真实点，根据 kernel / correlation 假设，对没算过的地方做 regression/interpolation/extrapolation，并给 predictive uncertainty。

GP 输出：

[
mu(x,z),
qquad
sigma^2(x,z).
]

其中：

[
mu
]

是预测均值；

[
sigma^2
]

是模型不确定性。

特别注意：

[
oxed{
sigma(x,z)

eq
|Y(x,z)-Y(x,z^star)|
}
]

predictive uncertainty 不等于真实 fidelity error。

---

# 35. Kennedy–O'Hagan / Co-Kriging 的典型结构

经典关系：

[
f_H(x)
=
ho f_L(x)+delta(x),
]

其中：

[
f_Lsim GP,
qquad
deltasim GP.
]

给定：

- 较多 low-fidelity samples；
- 较少 high-fidelity samples；

预测：

[
mu_H(x^star),
qquad
sigma_H^2(x^star).
]

口语化地说，可以理解成：

[
oxed{
	ext{已有几个点}
ightarrow
	ext{按照相关性把未知区域补出来}
}
]

这是 small-data surrogate modeling，而不是 large-scale representation learning。

---

# 36. 结果 Surrogate 中比较复杂的几个例外

## 36.1 DNN-MFBO

目标仍然是：

[
(x,m)ightarrow y_m(x).
]

第一级：

[
f_1(x)=NN_1(x).
]

更高级：

[
f_m(x)
=
NN_m(x,f_{m-1}(x)).
]

作用：

[
oxed{
	ext{让 NN 学 nonlinear cross-fidelity relationship}
}
]

而不是固定：

[
f_H=ho f_L+delta.
]

这里 neural network 只是 response surrogate，不是 RL policy。

---

## 36.2 Bayesian Neural Network 为什么在 BO 中有意义

BO 不仅需要预测：

[
hat y
]

还很依赖 uncertainty：

[
sigma(x).
]

普通 deterministic NN 不天然给 posterior variance。

Bayesian NN 让 weight 带 posterior：

[
p(wmid D),
]

通过不同权重样本得到：

[
mu(x),
qquad
sigma^2(x).
]

因此其目的可以概括为：

[
oxed{
	ext{NN 的 nonlinear capacity}
+
	ext{Bayesian uncertainty}
}
]

---

## 36.3 BMBO-DARN

Deep Auto-Regressive Network 中的 autoregressive 主要是：

[
oxed{
	ext{across fidelity levels}
}
]

不是语言模型时间序列生成。

较高级 fidelity 网络读取：

[
x,
f_1(x),
ldots,
f_{m-1}(x).
]

例如：

[
f_3(x)
=
NN_3(x,f_1(x),f_2(x)).
]

目的仍然是预测新 ((x,m)) 的 performance。

---

## 36.4 DMFAL

DMFAL 处理高维 field output：

[
Y(x)inmathbb R^p,
qquad
p	ext{ 很大}.
]

先生成低维 latent：

[
h_m(x)inmathbb R^r,
qquad
rll p,
]

再：

[
y_m(x)approx A_mh_m(x).
]

所以 NN 的作用是：

[
oxed{
	ext{高维 field}
ightarrow
	ext{低维 latent representation}
ightarrow
	ext{预测完整 field}
}
]

不是学 action policy。

---

# 37. IFC：Infinite-Fidelity Coregionalization 到底做什么

IFC 全名：

[
oxed{
	ext{Infinite-Fidelity Coregionalization}
}
]

它主要是：

[
oxed{
	ext{continuous-fidelity surrogate modeling}
}
]

不是 BO decision algorithm，也不是 RL。

设：

[
x
=
	ext{PDE 参数 / 边界条件 / 初始条件},
]

[
z
=
	ext{continuous fidelity，例如网格精度},
]

输出：

[
Y(x,z)
=
	ext{高维物理解场}.
]

它想做到：

[
oxed{
(x,z^star)
ightarrow
hat Y(x,z^star)
}
]

即使：

[
z^star
]

在训练数据中从来没出现过。

---

# 38. IFC 中 Neural ODE 的真正作用

IFC 先把完整物理解压到 latent：

[
h(x,z).
]

然后不为每个 fidelity 单独建一个离散模型，而是直接学习 latent 随 fidelity 怎么连续变化：

[
oxed{
rac{partial h(x,z)}{partial z}
=
NN(z,h,x)
}
]

这里 Neural ODE 中的“时间变量”不是物理时间，而是：

[
oxed{z=	ext{fidelity}}
]

给定一个起始 fidelity：

[
z_0,
]

可积分到任意新 fidelity：

[
h(x,z)
=
h(x,z_0)
+
int_{z_0}^{z}
NN(s,h,x),ds.
]

再通过：

[
B(z)
]

把 latent 恢复为完整场：

[
hat Y(x,z)
=
B(z)h(x,z).
]

所以：

[
oxed{
	ext{Neural ODE 在 IFC 中是 fidelity-direction dynamics model}
}
]

不是 RL transition model，也不是物理时间积分器。

---

# 39. IFC 的两个版本

### IFC-GPODE

latent 随 fidelity 的变化用 Neural ODE；

basis / decoder：

[
B(z)
]

用 GP 建模。

### IFC-ODE(^2)

连：

[
B(z)
]

本身随 fidelity 的变化也用另一个 Neural ODE 表示。

所以 IFC 的核心思想是：

[
oxed{
	ext{把离散 fidelity levels 连成一个 continuous latent trajectory}
}
]

---

# 40. Meng–Karniadakis Composite Multi-Fidelity NN

先拟合 low fidelity：

[
x
ightarrow
hat y_L(x).
]

然后将：

[
(x,hat y_L(x))
]

输入 high-fidelity correction network。

高保真关系分成：

[
F_{m linear}
]

和：

[
F_{m nonlinear},
]

组合成：

[
hat y_H
=
alpha F_{m linear}
+
(1-alpha)F_{m nonlinear}.
]

它和 KOH：

[
f_H=ho f_L+delta
]

思想上相似，只是把 cross-fidelity correction 从固定线性结构推广成 NN 学出的非线性关系。

---

# 41. JAHS-Bench 中 XGBoost Surrogate 的作用

真实训练网络很贵。

输入：

[
x=
	ext{architecture + hyperparameters}
]

fidelity：

[
z=
	ext{epoch / width / resolution / resource}
]

真实输出：

[
Y(x,z)=	ext{accuracy, loss, runtime}.
]

作者先离线收集大量真实训练记录，然后训练：

[
oxed{
	ext{XGBoost}(x,z)
ightarrow
widehat{	ext{accuracy}},
widehat{	ext{loss}},
widehat{	ext{runtime}}
}
]

以后 benchmark algorithm query 一个配置时，不需要真实训练网络，XGBoost 直接作为 surrogate 返回结果。

因此这里 XGBoost 可以看成：

[
oxed{
	ext{cheap fake environment for expensive HPO}
}
]

---

# 42. Branke 的 Logistic Regression 不是预测最终值，而是预测 Ranking Reversal

Branke 关注：

> 当前低 fidelity 上 A 比 B 好，完全收敛以后这个排序会不会反过来？

定义当前差值：

[
Delta
=
|f_z(x_A)-f_z(x_B)|.
]

用 logistic regression 预测：

[
P(	ext{ranking reversal}midDelta,z).
]

训练标签来自过去已经跑到最高 fidelity 的候选：

- low-fidelity 排名；
- final 排名；
- 是否发生 reversal。

所以它学的是：

[
oxed{
	ext{当前排序未来会翻车的概率}
}
]

不是完整预测：

[
(x,z)ightarrow f_H(x).
]

---

# 43. Palizhati 中的 SVR / KNN / RF / GPR

这些模型都只是不同的 surrogate：

[
xightarrowhat y.
]

可以选择：

- SVR；
- KNN；
- Random Forest；
- GPR。

然后外层再用：

- (epsilon)-greedy；
- GP-LCB；

等 selection rule 选择下一实验。

所以总体结构仍然是：

[
oxed{
	ext{surrogate}
+
	ext{hand-designed acquisition}
}
]

---

# 44. 最终统一认识：这批论文真正“学”的主要是 World，而不是 Policy

绝大多数模型学习的是：

[
oxed{
(x,z)ightarrow Y
}
]

或者：

[
oxed{
(x,z)ightarrow D
}
]

或者少数：

[
oxed{
(x,z)ightarrow C.
}
]

这可以叫：

[
oxed{	ext{learn the response / world}}
]

但到了：

> 现在到底做哪个 action？

大部分仍然采用：

[
oxed{
	ext{UCB / EI / MES / KG / LCB / information gain / allocation rule}
}
]

即：

[
oxed{
	ext{hand-design how to act}
}
]

而不是：

[
oxed{
S_t
ightarrow
pi_	heta
ightarrow
A_t.
}
]

所以整个文献群最准确的高层概括是：

[
oxed{
	ext{Learn the world with small-data surrogates,
then hand-design or online-plan the action.}
}
]

而标准 RL 更接近：

[
oxed{
	ext{Learn how to act from sequential experience.}
}
]

---

# 45. 对当前项目的最重要方法论启发

对于 continuous design + resumable solver progress 的新问题，当前最现实的第一阶段并不是直接从真实 solver 上训练 PPO/SAC，而是：

[
oxed{
	ext{先建立可靠的 world model}
}
]

包括：

[
hat Y,
qquad
hat C,
qquad
hat D,
qquad
hat P(S'|S,a).
]

然后先比较：

1. hand-designed myopic acquisition；
2. short-horizon online rollout；
3. approximate dynamic programming；

积累真实 trajectories。

如果以后有足够多历史数据或可信 virtual solver，再考虑：

[
oxed{
	ext{model-based RL / offline RL / policy distillation}
}
]

这样可以把高成本 online planning 摊销成可复用 learned policy。

---

# 46. 一句话总结整个新增讨论

[
oxed{
egin{aligned}
&	ext{这批 multi-fidelity 文献大多用小数据 surrogate 来估计世界；}\
&	ext{cost 常常直接给定或简单实测；}\
&	ext{action 绝大多数由 hand-designed acquisition/allocation rule 决定；}\
&	ext{显式 multi-step rollout 很少；}\
&	ext{此前固定的 36 篇中没有标准 RL policy learning；}\
&	ext{真正 RL 的关键是从 sequential experience 中学习可复用的 policy/value/model；}\
&	ext{而昂贵 continuous solver 环境最大的障碍正是缺少足够便宜的 interaction。}
end{aligned}
}
