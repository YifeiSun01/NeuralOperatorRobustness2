# MF-ENS 的准确性、成本与 Lookahead：问题澄清记录

本文记录关于 **Nonmyopic Multifidelity Active Search（MF-ENS）** 的几项关键澄清，并将它与此前分析的 26 篇 multi-fidelity 论文进行比较。

本文集中回答以下问题：

1. 一个低保真度查询返回 `0` 或 `1`，这个答案到底有多准确？
2. MF-ENS 实际预测了什么？
3. MF-ENS 为什么没有单独学习运行成本？
4. MF-ENS 为什么要进行 lookahead 和树展开？
5. 前面 26 篇论文中有没有类似的多步树展开？
6. 连续的 design space、fidelity space 和连续输出能不能进行 lookahead？

---

## 一、必须分开的三个问题

在一个 multi-fidelity 序贯搜索算法中，至少有三个相互独立的模块：

```yaml
预测模型:
  问题: "如果评价某个输入 x，会观察到什么结果？"

成本模型:
  问题: "评价这个 x，并算到 fidelity z，需要花多少时间或计算量？"

决策规划:
  问题: "考虑当前信息和剩余预算，下一步应该评价哪个 x、使用哪个 z？"
```

它们之间的关系是：

\[
\boxed{
\text{结果预测和成本预测}
\longrightarrow
\text{为候选动作提供信息}
\longrightarrow
\text{决策方法选择下一步动作}
}
\]

Lookahead 属于第三个模块。它不会自动学出某个 fidelity 有多准确，也不会自动学出运行成本。它必须使用前两个模块提供的概率和成本信息。

---

## 二、“这个点是 1 的概率”与“低保真答案正确的概率”不是同一个量

MF-ENS 中有两个标签：

\[
y_H(x)\in\{0,1\},\qquad y_L(x)\in\{0,1\}.
\]

其中：

```yaml
y_H(x): "高保真 oracle 给出的标签；论文把它当作参考答案"
y_L(x): "低保真 oracle 给出的便宜但可能出错的标签"
```

下面两个概率含义完全不同。

### 量 A：这个点是真正正例的概率

\[
P\bigl(y_H(x)=1\mid D\bigr).
\]

它回答：

> 根据已经收集的数据，这个候选点经过高保真检查以后，成为真正正例的概率有多大？

例如：

\[
P(y_H(x)=1\mid D)=0.8
\]

表示模型认为这个点有 80% 的可能性是真正的 `1`。

### 量 B：低保真答案与参考答案一致的概率

\[
P\bigl(y_L(x)=y_H(x)\mid D\bigr).
\]

它回答：

> 对这个候选点，低保真 oracle 返回的 `0/1` 有多大概率是正确的？

这个量才是本文所说的低保真 **判断准确性或可靠性**。

例如，低保真查询返回：

\[
y_L(x)=0,
\]

同时模型认为：

\[
P(y_L(x)=y_H(x)\mid D)=0.9.
\]

那么它表示：

```yaml
低保真答案为 0: true
这个 0 正确的概率: "90%"
这个点实际上为 1 的概率: "在这个简化例子中为 10%"
```

因此，不能把

\[
P(y_H(x)=1\mid D)
\]

叫作低保真准确率。

---

## 三、MF-ENS 实际怎样处理低保真答案是否准确？

MF-ENS 确实允许低保真标签出错，但它没有建立一个明确的、随输入和 fidelity 连续变化的准确率函数：

\[
A(x,z)=P\bigl(y_z(x)=y_H(x)\mid x,z,D\bigr).
\]

它采用的是更简单的处理。

```yaml
高保真 oracle:
  论文设定: "返回 exact label"
  含义: "高保真标签被当作参考真值"
  是否从数据证明其 100% 准确: false

低保真 oracle:
  论文设定: "返回 noisy label"
  是否允许出错: true
  是否显式学习 A(x,z): false

预测模型:
  类型: "修改后的 k-nearest-neighbor 概率分类器"
  预测量:
    - "P(y_H(x)=1 | D)"
    - "P(y_L(x)=1 | D)"

低保真信息权重:
  符号: "q"
  范围: "0 < q < 1"
  作用: "减弱低保真邻居标签对预测结果的影响"
  估计方法: "每轮根据已有数据进行 maximum likelihood estimation"
```

需要特别注意：

\[
\boxed{q\text{ 不是低保真准确率}}
\]

例如，\(q=0.9\) 不能直接解释成“低保真答案有 90% 的概率正确”。它只是分类器内部控制低保真标签影响强弱的权重。

论文实验中没有真正的双 fidelity 数据集，因此作者从原有真实标签出发，翻转一部分标签，人工构造低保真标签，并用不同噪声水平测试算法。这里的噪声参数用于制造实验数据，也不能与分类器学习出的 \(q\) 混为一谈。

所以，MF-ENS 对低保真可靠性的处理可以概括为：

\[
\boxed{
\text{允许低保真标签出错}
+
\text{从数据调整低保真标签在分类器中的影响}
}
\]

但是它没有完整回答：

\[
\boxed{
\text{给定任意 }x,z，\text{这次低保真判断正确的概率究竟是多少？}
}
\]

---

## 四、MF-ENS 为什么没有训练成本预测模型？

MF-ENS 直接假定：

\[
\text{一次高保真查询的时间}
=
k\times\text{一次低保真查询的时间}.
\]

因此可以写成：

```yaml
低保真查询成本: 1
高保真查询成本: k
成本是否随 x 改变: false
成本是否从历史运行数据学习: false
成本是否包含随机波动: false
```

它使用的是固定的时间比例，而不是：

\[
\widehat c(x,z)
\]

或者：

\[
p(c\mid x,z,D).
\]

论文还规定高、低保真查询可以并行运行。一次高保真查询运行期间，可以完成 \(k\) 次低保真查询。

因此，MF-ENS 不是在每一个时刻都从所有 \((x,z)\) 中完全自由地选择一个组合。它的基本安排是：

```yaml
高保真通道空闲时: "选择一个候选 x 做高保真查询"
高保真查询等待期间: "依次选择候选 x 做低保真查询"
fidelity 的时间关系: "由固定并行调度和比例 k 决定"
```

这比本文研究的可暂停、可继续的连续求解器简单很多。

---

## 五、MF-ENS 为什么要使用 Lookahead？

MF-ENS 的最终目标是：

\[
U=\text{通过高保真查询确认出来的正例数量}.
\]

只有经过高保真查询确认的 `1` 才计入最终发现数量。低保真查询即使返回 `1`，也只是提供线索，不算完成一次真正发现。

因此：

```yaml
高保真查询:
  可能立即增加发现数量: true

低保真查询:
  可能立即增加发现数量: false
  可能帮助后续选择更好的高保真候选: true
```

低保真查询的价值沿着下面的路径产生：

\[
\text{现在进行一次便宜查询}
\longrightarrow
\text{观察低保真标签}
\longrightarrow
\text{更新候选点为真正正例的概率}
\longrightarrow
\text{重新安排后续高保真查询}
\longrightarrow
\text{增加最终确认的正例数}.
\]

Lookahead 的作用，就是估计这条未来路径的价值。

它回答的问题是：

> 如果我现在查询这个点，看到不同结果以后，我将怎样改变后续查询？这些改变最终预计能多找到多少个真正正例？

Lookahead 不负责直接回答：

```yaml
低保真标签有多准确: "由概率模型或可靠性模型负责"
运行一次需要多久: "由成本假设或成本模型负责"
```

---

## 六、MF-ENS 的树具体展开什么？

MF-ENS 的低保真观测是二元标签：

\[
y_L\in\{0,1\}.
\]

所以一次未来低保真查询只有两个假想结果分支：

```text
现在考虑进行一次低保真查询
│
├── 假想结果为 0
│   └── 更新概率 → 重新选择后续高保真候选
│
└── 假想结果为 1
    └── 更新概率 → 重新选择后续高保真候选
```

假设未来考虑一个由 \(k\) 个低保真查询组成的 batch，则所有可能的标签组合是：

\[
Y_L\in\{0,1\}^{k},
\]

组合数量为：

\[
2^k.
\]

例如 \(k=5\) 时：

\[
2^5=32.
\]

这里的 32 表示五个二元标签的 32 种组合。它不表示有 32 个 fidelity，也不表示高保真输出有 32 个可能数值。

MF-ENS 并没有完整求解从现在一直到预算结束的所有决策。它采用的是近似的两阶段 rollout：

```yaml
当前阶段: "假想现在选择一个查询并观察其结果"
中间阶段: "假想再进行一批低保真探索查询"
最终阶段: "根据假想得到的低保真结果，选择一个高保真 greedy batch"
```

为了降低计算量，论文还使用：

```yaml
branch-and-bound pruning: "提前排除不可能成为最优动作的候选点或分支"
sampling approximation: "当 2^k 太大时，用有限个抽样组合近似全部组合"
```

---

## 七、这棵树必须依赖预测模型

假设一个候选点的当前查询结果 \(y\) 可能是 0 或 1。算法计算候选动作分数时，需要：

\[
f(x)
=
P(y=1\mid x,D)f(x\mid y=1)
+
P(y=0\mid x,D)f(x\mid y=0).
\]

其中：

```yaml
P(y=1 | x,D): "预测模型给出的分支概率"
f(x | y=1): "假如结果为 1，后续策略预计得到的收益"
f(x | y=0): "假如结果为 0，后续策略预计得到的收益"
```

所以准确的关系是：

\[
\boxed{
\text{概率预测模型}
\longrightarrow
\text{提供未来结果的概率}
\longrightarrow
\text{Lookahead 计算未来决策的期望收益}
}
\]

因此，“MF-ENS 没有 performance 预测，只展开树”这个说法不成立。它有一个较简单的概率分类模型，只是没有建立连续求解器的数值输出和连续准确率模型。

---

## 八、前面 26 篇论文有没有类似的多步树展开？

按这 26 篇论文各自提出的核心方法来看，基本没有论文把 **MF-ENS 式的多阶段、自适应、按照未来观测结果分叉的 rollout tree** 作为核心算法。

它们大致可以分成以下几类。

| 方法类型 | 代表论文 | 当前动作怎样选择 | 是否有 MF-ENS 式多阶段树 |
|---|---|---|---|
| 多保真回归或数据融合 | Kennedy–O’Hagan、PINN multifidelity、IFC | 主要预测参考 fidelity 的输出 | 否 |
| 当前 GP 分数和 fidelity 规则 | MF-GP-UCB、BOCA、DNN-MFBO | 给当前候选 \((x,z)\) 计算分数 | 否 |
| 当前查询的信息价值 | FABOLAS、MF-MES、DMFAL | 计算这一次评价预计提供的信息，再考虑成本 | 否；它们对未知量取期望，但不展开后续多步自适应动作 |
| 一步 lookahead | MISO / misoKG | 假想一次新观测，计算模型更新后最终推荐能够改善多少 | 只看一步，没有多层树 |
| 联合选择一个 batch | BMBO-DARN | 在实际标签返回前同时决定一批查询 | 否；同一 batch 内不会根据前一个假想结果重新选择下一个动作 |
| 固定资源调度 | Hyperband | 根据真实的部分训练分数淘汰配置并增加资源 | 否 |
| 直接使用优化或仿真结果 | Benders、部分离散事件仿真方法 | 根据真实求解进展或样本统计调整计算 | 通常没有概率情景树 |

### MISO 是 26 篇中最接近的例子

MISO 的 misoKG 会考虑：

> 如果现在评价信息源 \(z\) 和设计点 \(x\)，并观察到一个可能结果，更新 GP 以后，最终推荐的最好设计平均可以改善多少？

可以把它概括成：

\[
\frac{
\mathbb E_y\left[\max_{x'}\mu_{\text{更新后}}(x')\right]
-\max_{x'}\mu_{\text{当前}}(x')
}{c(x,z)}.
\]

它是假想一次未来观测，因此属于一步 lookahead。但它不会继续展开：

```text
第一次假想结果
→ 根据第一次结果选择第二次动作
→ 第二次假想结果
→ 根据第二次结果选择第三次动作
```

所以 MISO 不是 MF-ENS 那种多阶段 rollout tree。

### MF-MES、FABOLAS 和 DMFAL 为什么也不算多步树？

这类方法会计算：

\[
\frac{\text{当前这次评价预计带来的信息}}{\text{当前这次评价的成本}}.
\]

它们会对当前未知结果或者最终目标的不确定性进行积分或抽样，但没有为不同假想结果分别规划第二步、第三步动作。因此：

\[
\boxed{\text{对随机结果取期望}\neq\text{展开多阶段决策树}}
\]

### 为什么大部分论文没有展开树？

```yaml
原因 1:
  内容: "很多论文只负责预测或融合多保真输出，本身没有序贯选点任务"

原因 2:
  内容: "UCB、EI、MES 等方法可以直接为当前候选动作打分"

原因 3:
  内容: "多步展开需要反复假想观测、更新模型并重新优化动作，计算量迅速增加"

原因 4:
  内容: "许多论文把创新重点放在输出关联模型、成本处理或当前 acquisition 上，而不是长期规划"
```

MF-ENS 选择树展开，是因为它专门研究 budget-aware、nonmyopic active search，并希望表达低保真信息怎样改变后续高保真发现。它是一种规划方法，不是所有 active search 或 multi-fidelity 方法都必须采用的步骤。

---

## 九、连续空间到底能不能进行 Lookahead？

可以。

需要区分三个可能连续的对象：

```yaml
design x: "下一次在哪里评价"
fidelity z: "下一次计算到什么精度或进度"
observation y: "执行评价后会看到什么结果"
```

如果未来观测 \(y\) 是连续数值，它有无限多个可能结果，无法逐个枚举。理论上的 lookahead 使用积分：

\[
\mathbb E_y[G(y)]
=
\int G(y)p(y\mid x,z,D)\,dy.
\]

实际计算时可以从预测分布中抽取有限个假想结果：

\[
\mathbb E_y[G(y)]
\approx
\frac{1}{M}\sum_{m=1}^{M}G\bigl(y^{(m)}\bigr).
\]

这里发生的是：

```yaml
被有限近似的对象: "未来可能观察到的连续结果"
近似方式: "有限个 fantasy/scenario 样本"
是否必须把 design space 切成固定网格: false
是否必须把 fidelity space 切成固定档位: false
```

因此，可以采用：

\[
\boxed{
\text{有限个假想结果分支}
+
\text{每个分支中连续优化下一步 }(x,z)
}
\]

Jiang 等人的 *Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees* 就属于这种情况：它用 GP posterior 中有限数量的 fantasy samples 近似连续结果的积分，同时让各个分支中的下一步输入继续作为连续变量进行梯度优化。

这篇连续 lookahead 论文不是 MF-ENS，也不属于此前的 26 篇论文。它解决的是连续输入下的 Bayesian optimization，目标仍然是寻找最大值。引用它的目的只是说明：

\[
\boxed{\text{连续空间并不排除 Lookahead}}
\]

连续问题真正的困难是计算量。假设每一层抽取 \(M\) 个假想结果，向前看 \(h\) 层，分支数量大约按照：

\[
M^{h}
\]

增长。因此实际算法通常限制 lookahead 深度、减少样本、进行剪枝，或者使用更简单的最终阶段策略。

---

## 十、对本文连续可恢复求解器问题的直接含义

本文研究的问题是：

```yaml
design_space:
  变量: "连续输入 x"
  候选产生方式: "可以自由生成新的连续参数组合"

fidelity_space:
  主要变量: "数值求解进度或收敛程度 z"
  类型: "希望连续"

solver:
  性质: "同一个 x 随计算继续进行而逐步逼近参考答案"
  是否可以暂停并继续: true

最终目标:
  类型: "active search"
  内容: "在总时间内确认尽可能多的、最终结果超过阈值的不同输入"
```

这个问题至少需要下面三个模型或规则。

### 1. 最终达标概率或最终输出预测

可以直接预测最终输出：

\[
p\bigl(y_\star(x)\mid x,\text{当前部分求解轨迹},D\bigr),
\]

再计算：

\[
P\bigl(y_\star(x)>\tau\mid x,\text{当前信息},D\bigr).
\]

也可以直接训练最终二分类概率模型：

\[
P\bigl(H(x)=1\mid x,\text{当前部分求解轨迹},D\bigr).
\]

### 2. 当前 fidelity 的可靠性

需要估计：

\[
A(x,z)
=
P\bigl(H_z(x)=H_\star(x)\mid x,z,D\bigr).
\]

这个量直接对应本文所问的：

> 算到进度 \(z\) 时给出的 `0/1` 判断，究竟有多大概率与最终收敛答案一致？

对真实求解器来说，它可能随 \(x\) 和 \(z\) 一起变化，不能直接照搬 MF-ENS 的单一全局权重 \(q\)。

### 3. 继续计算的追加成本

因为求解器能够暂停并继续，更合适的成本是：

\[
\Delta c(x,z_0\rightarrow z_1\mid \text{保存状态}),
\]

而不是每次都使用从零开始计算到 \(z_1\) 的成本：

\[
c(x,z_1).
\]

该成本可以由历史运行记录拟合，也可以在求解器运行规律稳定时使用已知公式或在线估计。

### 4. 下一步决策

动作可能包括：

```yaml
动作 A: "生成一个新的连续输入 x，并从低进度开始计算"
动作 B: "把已经计算过的 x 从 z0 继续算到更高进度 z1"
动作 C: "把一个高概率候选继续算到参考精度，正式确认是否达标"
```

决策可以采用：

```yaml
轻量方案:
  内容: "只评价当前动作的信息价值、确认价值和追加成本"
  是否需要多步树: false

一步方案:
  内容: "假想当前观测后，计算下一次最佳动作会怎样变化"
  是否需要多层树: false

多步方案:
  内容: "抽样若干未来观测，展开有限深度的近似决策树"
  是否必须采用: false
```

因此，不能直接把 MF-ENS 的树移植过来。MF-ENS 假定有限候选集合、两个 fidelity、二元观测、固定成本比例和不可恢复的独立查询；本文的问题具有连续输入、连续进度、连续部分输出和可恢复计算状态。

合理的开发顺序是先建立：

\[
\boxed{
\text{最终达标概率模型}
+
\text{当前判断可靠性模型}
+
\text{追加成本模型}
}
\]

然后比较简单决策、一步 lookahead 和有限深度 lookahead 是否真的能在包含规划时间的总预算下找到更多已确认正例。

---

## 十一、本轮讨论中的修正

```yaml
修正 1:
  原来容易产生的误解: "P(y_H=1 | D) 是低保真准确率"
  正确说法: "它是候选点为真正正例的概率"

修正 2:
  原来容易产生的误解: "q 是低保真正确率"
  正确说法: "q 是低保真邻居在 kNN 分类器中的影响权重"

修正 3:
  原来容易产生的误解: "MF-ENS 没有预测模型，只做树展开"
  正确说法: "它有概率分类模型；树使用该模型提供分支概率和更新后的正例概率"

修正 4:
  原来容易产生的误解: "低保真没有即时奖励，所以 active search 必须使用多步树"
  正确说法: "多步 rollout 是一种处理未来信息价值的方法，也可以使用一步信息价值或其他近似评分"

修正 5:
  原来容易产生的误解: "连续问题必须把 design 和 fidelity 全部离散化后才能 lookahead"
  正确说法: "通常只用有限 fantasy samples 近似连续未来观测；分支中的动作仍可在连续空间优化"
```

---

## 主要原始文献

1. Quan Nguyen, Arghavan Modiri, Roman Garnett. [Nonmyopic Multifidelity Active Search](https://proceedings.mlr.press/v139/nguyen21f/nguyen21f.pdf). ICML 2021.
2. Matthias Poloczek, Jialei Wang, Peter Frazier. [Multi-Information Source Optimization](https://papers.neurips.cc/paper_files/paper/2017/file/df1f1d20ee86704251795841e6a9405a-Paper.pdf). NeurIPS 2017.
3. Shion Takeno et al. [Multi-fidelity Bayesian Optimization with Max-value Entropy Search and its Parallelization](https://proceedings.mlr.press/v119/takeno20a/takeno20a.pdf). ICML 2020.
4. Kirthevasan Kandasamy et al. [Multi-fidelity Bayesian Optimisation with Continuous Approximations](https://proceedings.mlr.press/v70/kandasamy17a/kandasamy17a.pdf). ICML 2017.
5. Shali Jiang et al. [Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees](https://proceedings.neurips.cc/paper/2020/file/d1d5923fc822531bbfd9d87d4760914b-Paper.pdf). NeurIPS 2020.

