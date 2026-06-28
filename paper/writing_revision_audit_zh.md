# main.tex 逐段语言审稿与改写建议

对象文件：`main.tex`

对照风格：近两年 AAAI AI for Science / Neural Operator / Robustness 论文的常见写法。它们通常把叙事组织成：

`real-world challenge -> existing assumption/gap -> proposed framework/component -> benchmark evidence`

你的稿子目前更像：

`definition ambiguity -> error operator -> metric/loss taxonomy -> solver-integrated attack/training -> empirical validation`

这个底层逻辑是强的，但表层语言可以更像会议论文：更早说 challenge，更少连续否定别人，更常用 assumption / limitation / component / benchmark / framework 这些路标词。

阅读说明：本文件里的 `### 建议 1/2/3` 是每个 `##` 小节内部的局部编号，会在新的小节重新开始。比如 `Abstract 建议 A2` 只指 Abstract 里的 gap 句，不是全文第二重要建议，也不是后面所有建议的总标题。定位修改时优先看 `当前：` 和 `建议：` 下面引用的原句，不要靠行号。

---

## 0. 全文级优先修改原则

### 0.1 把 “not do X” 改成 “assume Y, which becomes limiting when Z”

你现在 Related Work 和 Method 里有较多：

> These methods do not use ...

建议改成：

> These methods implicitly assume that ..., which becomes limiting when ...

这样从“否定别人”变成“识别共同假设”。AAAI 论文更喜欢后者，因为它显得你在抽象问题，而不是逐条挑错。

### 0.2 把 “The notation above separates ...” 改成 “The setting creates ...”

你现在经常从 notation 出发。建议从 problem/design choice 出发。

弱一些：

> The notation above separates three finite attack objective functions.

更强：

> The solver-surrogate setting creates three possible attack targets, depending on whether the target output is model-derived, fixed to the clean ground-truth output, or recomputed as the ground-truth output of the perturbed input.

### 0.3 把 “This is ...” 改成 “This distinction matters because ...”

你有较多 `This is ...`，像技术说明。AAAI 更常写成“为什么这个区别重要”。

弱一些：

> This is the key difference from our solver-integrated setting.

更强：

> This distinction matters because an attack can appear strong against a fixed target while failing to enlarge the true model-solver discrepancy at the perturbed input.

---

## 1. Abstract（摘要整体）

### Abstract 当前功能

摘要已经有 AAAI 的基本骨架：背景、gap、三点贡献、实验结论。问题是贡献句偏长，且核心 challenge 还不够尖锐。

### Abstract 建议 A1：第一句词汇升级

当前：

> Neural operators are commonly utilized as fast surrogates for numerical PDE solvers, mapping input functions to solution functions.

建议：

> Neural operators are commonly utilized as fast surrogates for numerical PDE solvers, mapping input functions to solution functions.

理由：

保留 `numerical PDE solvers`，避免加入 `initial conditions or coefficients` 这类不必要的例子，并统一使用 `solution functions`。

### Abstract 建议 A2：gap 句聚焦 robustness 定义创新

当前：

> However, their generalizability and robustness are not yet clearly defined in the solver-surrogate setting, which differs from traditional adversarial robustness definitions.

建议：

> However, traditional adversarial robustness definitions become misaligned in the operator-learning setting, because the ground-truth solver output changes with the perturbed PDE input.

理由：

这句可以只写 robustness，因为本文的定义创新主要集中在 solver-consistent robustness；generalizability 更适合作为研究对象和实验收益，而不是这一句里的定义 gap。也不要让 `traditional adversarial robustness` 去修饰 generalizability，因为传统 adversarial robustness 本来就不是 generalization 定义。句末只点出核心差异：the solver output changes with the perturbed input，避免在摘要第一段塞入过多机制解释。

### Abstract 建议 A3：把核心 challenge 明说

建议放在 Abstract 的 gap 句后面，也就是这句之后：

> However, their generalizability and robustness are not yet clearly defined in the solver-surrogate setting, which differs from traditional adversarial robustness definitions.

并且放在下面这句之前：

> This paper studies the generalizability and the robustness of a neural operator from a solver-integrated perspective...

可插入：

> The model should not be invariant to the perturbation, but should remain aligned with the solver response to that perturbation.

理由：

这句不需要再起一个 challenge 名字，直接说明机制即可：ground-truth solver output 会随 perturbation 改变，所以 model 不应该对 perturbation invariant，而应该和 solver response to that perturbation 对齐。

### Abstract 建议 A4：贡献句拆短

当前：

> First, we define and distinguish generalization and robustness for neural operators through an error-operator view, identifying fixed-input model-solver loss as a generalization metric and separating it from perturbation-based robustness metrics such as norm-bounded adversarial attack loss increase.

建议：

> First, we formalize generalization and robustness through a model-solver error operator. This separates fixed-input solver discrepancy from perturbation-induced error growth.

理由：

原句 37 词，承担太多任务。拆成两句后更像 AAAI abstract：动作清楚，贡献可扫读。

### Abstract 建议 A5：结果句更学术

当前：

> Experiments on representative PDE benchmarks show that this solver-integrated adversarial training clearly improves both generalizability and robustness.

建议：

> Experiments on representative PDE benchmarks show that this solver-integrated adversarial training consistently improves both generalizability and robustness.

理由：

这里只需要把 `clearly improves` 换成 `consistently improves`。不要引入 `shifted-input generalization` 或 `attack-time robustness`。

---

## 2. Introduction P1, lines 93-102

### Introduction P1 当前功能

介绍 neural operators 的背景和 practical need。写得清楚，但略泛。

### Introduction P1 建议 1：把 “Practical use” 升级为 challenge

当前：

> Practical use, however, requires the learned operator to generalize beyond the training dataset and remain close to the solver when the input function is perturbed.

建议：

> Further deployment, however, raises the solver-faithfulness challenge: the learner must generalize beyond the training data set and remain close to the solver under input perturbations.

理由：

`solver-faithfulness challenge` 是一个很好的自定义概念，能把 generalization 和 robustness 绑到一个核心问题上。

### Introduction P1 建议 2：最后一句沿用当前写法

当前：

> Recent neural operator studies have therefore begun to examine adversarial robustness and robustness-aware training ...

建议：

> Recent neural operator studies have therefore begun to examine adversarial robustness and robustness-aware training ...

理由：

这句沿用当前写法即可。它保留了 `adversarial robustness and robustness-aware training`，并自然承接前一句 practical deployment 的需求，不需要在这里额外加入 robustness target ambiguity。

---

## 3. Introduction P2, lines 104-117

### Introduction P2 当前功能

区分 generalization 和 robustness。内容重要，但太像定义说明，问题压力不够。

### Introduction P2 建议 1：第一句沿用当前写法

当前：

> Prior work has often not clearly separated the notions of generalization and robustness.

建议：

> Prior work has often not clearly separated the notions of generalization and robustness.

理由：

这句沿用当前写法即可。它直接点出本文要区分 generalization 和 robustness，不需要改成 `central ambiguity` 或额外强调算法后果；后面的正文已经会展开这些区别。

### Introduction P2 建议 2：定义句可以更平衡

当前：

> Generalization is a static definition: the norm of the output of our error operator. Robustness is defined in a dynamic way: how the output of our error operator amplifies under an input perturbation.

建议：

> In our formulation, generalization is a static solver-discrepancy quantity at a fixed input, whereas robustness is a perturbation-response quantity measuring how that discrepancy grows under admissible input changes.

理由：

`static definition` / `dynamic way` 有点口语和抽象；`static solver-discrepancy quantity` / `perturbation-response quantity` 更论文。

### Introduction P2 建议 3：Jacobian-error metric 的句子加 role

当前：

> We further develop a Jacobian-error metric, which is cheaper to compute and closely aligned with adversarial loss increase in the local robustness setting.

建议：

> This distinction also yields a lightweight Jacobian-error metric that locally predicts adversarial error growth while avoiding full finite-budget attacks.

理由：

新句把 metric 说成由前面 distinction “yield” 出来的组件，逻辑更自然。

---

## 4. Introduction P3, lines 119-133

### Introduction P3 当前功能

这是 Introduction 最强的一段：把 classification label invariance 和 PDE solver response 区分开。

### Introduction P3 建议 1：第一句沿用当前写法

当前：

> This setting also differs from traditional adversarial robustness.

建议：

> This setting also differs from traditional adversarial robustness.

理由：

这句沿用当前写法即可。它先建立和 traditional adversarial robustness 的区别，后面再解释区别来自 solver response 随输入变化。

### Introduction P3 建议 2：中间句沿用当前写法

当前：

> However, the PDE operator learning problem is not only a regression problem, but a regression problem with an existing numerical solver as a trusted oracle.

建议：

> However, the PDE operator learning problem is not only a regression problem, but a regression problem with an existing numerical solver as a trusted oracle.

理由：

这句沿用当前写法即可。`trusted oracle` 在这里可以保留，因为它直接强调 PDE operator learning 和普通 regression 的区别。

### Introduction P3 建议 3：解释 robustness 的对象

当前：

> Therefore, the model is not expected to be robust by itself; rather, the model-solver discrepancy is expected to be robust.

建议：

> Thus, robustness here does not mean that the neural operator output is invariant to input perturbations. It means that the model-solver error remains small as the input and the solver response change.

理由：

这里最容易误读的是 `robust` 的对象。传统 adversarial robustness 容易让读者想到模型输出对输入扰动不敏感；本文真正要求的是 error operator 或 model-solver discrepancy 的稳定性，即输入和 solver response 都变时，\(F_\theta(a)-G(a)\) 不应被显著放大。

---

## 5. Introduction P4, lines 135-147

### Introduction P4 当前功能

提出 solver-integrated adversarial training，并用 teacher-student metaphor。整体可用，但有些词偏宣传。

### Introduction P4 建议 1：第一句更直接

当前：

> Building on this view, we develop solver-integrated adversarial training.

建议：

> To address this solver-response alignment issue, we develop solver-integrated adversarial attacks and training.

理由：

`To address this` 是 AAAI 常用衔接。它把方法明确接到问题。

### Introduction P4 建议 2：teacher-student 句子压缩

当前：

> The method can be interpreted as a teacher-student solver distillation framework: the numerical solver acts as the teacher, while the neural operator is the student repeatedly trained on informative inputs discovered from the local model-solver discrepancy, so that the neural operator becomes an increasingly faithful behavioral twin of the solver.

建议：

> The numerical solver serves as a supervision oracle, and the neural operator is trained on discrepancy-seeking inputs generated from the current model-solver mismatch.

理由：

`faithful behavioral twin` 有点像宣传语；`solver-faithful surrogate` 可以用，但这里建议更简洁。

### Introduction P4 建议 3：实验句更具体

当前：

> Experiments on representative PDE benchmarks show that the proposed training clearly improves generalization and robustness...

建议：

> Across Burgers, Darcy Flow, and Navier-Stokes benchmarks, solver-integrated training improves shifted-distribution accuracy and reduces attack-induced model-solver error.

理由：

`generalization and robustness` 是抽象名词；新句告诉读者你具体改善了什么。

### Introduction P4 建议 4：这里缺 contribution bullet

建议在 Introduction 末尾加：

> Our contributions are:
> (i) we identify the solver-response alignment issue in PDE solver-surrogate attacks;
> (ii) we formalize generalization and robustness with a model-solver error operator and a local Jacobian-error metric;
> (iii) we propose solver-integrated attacks and adversarial training that use the solver in both the forward target and backward path;
> (iv) we evaluate the framework on Burgers, Darcy Flow, and Navier-Stokes benchmarks.

理由：

AAAI 论文非常依赖 contribution list。你的摘要有，但 Introduction 没有，扫读性吃亏。

---

## 6. Related Work P1, lines 151-165

### Related Work P1 当前功能

介绍 classical classification attacks。内容多，但句子里列方法太长。

### Related Work P1 建议 1：第一句加 limitation target

当前：

> Most classical adversarial attacks were designed for classification.

建议：

> Classical adversarial attacks primarily target classification models, where the label is assumed to remain unchanged under small input perturbations.

理由：

这句直接抓住和你论文相关的 assumption，而不是先泛泛分类。

### Related Work P1 建议 2：长列表压缩

当前一长串：

> FGSM and PGD..., JSMA..., C&W..., DeepFool..., Boundary Attack...

建议：

> These methods differ mainly in optimization access and perturbation structure: gradient-based attacks such as FGSM and PGD use white-box derivatives, decision-boundary methods search for nearby label changes, and black-box or patch attacks restrict the available queries or perturbation geometry.

理由：

你不需要在正文用这么多方法名撑满一段。AAAI Related Work 更倾向按 mechanism 分类。

### Related Work P1 建议 3：结尾更接本文

当前：

> Despite these optimization-specific differences, they usually assume that the ground-truth label does not change.

建议：

> This fixed-label assumption is precisely what breaks in PDE operator learning, where the ground-truth solution changes with the perturbed input.

理由：

这句把 related work 变成 gap framing。

---

## 7. Related Work P2, lines 167-179

### Related Work P2 当前功能

介绍 regression/time-series attacks，并指出它们没有 oracle in loop。逻辑正确，但 `do not put` 太硬。

### Related Work P2 建议 1：核心句改写

当前：

> They do not put the true input-to-output oracle inside the attack loop: the target loss does not use the ground-truth output associated with the perturbed input, and the backward propagation does not use gradient information from that ground-truth oracle.

建议：

> These attacks usually treat the target as observed, fixed, or model-derived. As a result, the input-dependent oracle is absent from both the forward target and the backward gradient.

理由：

这就是你要的 “assumes / becomes limiting” 写法。更抽象、更像教授写 Related Work。

### Related Work P2 建议 2：key difference 句更具体

当前：

> This is the key difference from our solver-integrated setting.

建议：

> This distinction becomes central in PDE operator learning because a perturbation changes not only the model input but also the correct solver response.

理由：

不要只说 key difference；说 difference 为什么重要。

### Related Work P2 建议 3：最后一句更像 taxonomy

当前：

> We then introduce a third target loss, L3, for the solver-integrated setting.

建议：

> We therefore organize attack losses into a hierarchy of solver coupling and introduce L3 as the fully solver-integrated objective.

理由：

`hierarchy of solver coupling` 是一个更有 AAAI framework 味道的词。

---

## 8. Related Work P3, lines 181-193

### Related Work P3 当前功能

讨论已有 neural operator robustness。内容重要，但有点像逐条指出对方不够。

### Related Work P3 建议 1：第一句承认贡献

当前：

> Beyond generic adversarial-robustness papers, one study focuses on robustness in operator learning and finds that neural-operator error can grow significantly when the perturbation norm is increased.

建议：

> Recent operator-learning robustness work provides important evidence that neural-operator errors can grow under norm-bounded perturbations.

理由：

先肯定已有工作，比 “one study focuses” 更大方。

### Related Work P3 建议 2：dictionary limitation 改成 assumption

当前：

> Solver outputs are used as ground-truth labels, but the solver is still not integrated into the attack loop.

建议：

> However, the solver is used to provide offline ground-truth labels rather than as an online component of the attack loop.

理由：

`offline ground-truth labels` vs `online solver component` 是很清楚的对比词。

### Related Work P3 建议 3：后半段压缩

当前：

> In the forward pass... nearest-neighbor... backward pass... solver gradient is ignored...

建议：

> Consequently, the forward target can be mismatched to the perturbed input, and the backward direction ignores how the solver response changes with that perturbation.

理由：

细节可以留，但主文要先给总结句。AAAI 写法通常先一句抽象，再一两句细节。

---

## 9. Related Work P4, lines 195-217

### Related Work P4 当前功能

physics-residual adversarial training。这个段落太长，建议拆成两段。

### Related Work P4 建议 1：第一段写能力

当前：

> A related line studies adversarial training with PDE residual/physics loss.

建议：

> A related line uses PDE residuals as adversarial or robustness-aware training signals.

后接：

> These methods are attractive because they can generate difficult points without requiring a supervised solver label at every perturbed input.

理由：

先说这类方法为什么合理，避免一上来就否定。

### Related Work P4 建议 2：限制句改成 “residual is not solver supervision”

当前：

> These methods do not use the numerical solver as the teacher in the forward and backward loop, which has two limitations in our setting.

建议：

> Their shared assumption is that residual consistency is a sufficient proxy for solver agreement. This assumption becomes limiting in our setting for two reasons.

理由：

这正是你前面问的句式：`assumption becomes limiting when/for`。

### Related Work P4 建议 3：第一点 limitation 更凝练

当前：

> First, physics loss is less informative than solver supervision...

建议：

> First, residual consistency is weaker than solver supervision: a field may satisfy local physics constraints without matching the solver trajectory used as the supervised target.

理由：

更像论文陈述，避免 “less informative” 过于口语。

### Related Work P4 建议 4：第二点 limitation 更任务化

当前：

> Second, many neural-operator setups output only the target field or final state...

建议：

> Second, many operator-learning benchmarks supervise only terminal fields, so the temporal derivatives required by residual losses are unavailable without additional rollout information.

理由：

`benchmarks supervise only terminal fields` 比 `setups output only` 更像 benchmark/task framing。

---

## 10. Related Work P5, lines 219-231

### Related Work P5 当前功能

heuristic, active, generative sampling。写得不错，但结尾可以更 framework 化。

### Related Work P5 建议 1：第一句更聚类

当前：

> Other operator-learning works use heuristic, active, or generative sampling.

建议：

> A broader set of operator-learning methods improves data coverage through heuristic, active, or generative sampling.

### Related Work P5 建议 2：结尾改成本文缺口

当前：

> In this sense, they do not fully exploit the special structure of PDE operator learning, where a numerical solver is already available as a supervision oracle.

建议：

> They therefore leave open the question we study here: how should sample selection change when a differentiable numerical solver is available as both a supervision oracle and a gradient source?

理由：

`leave open the question` 是非常会议论文的过渡。

---

## 11. Preliminaries P1, lines 236-290

### Preliminaries P1 当前功能

定义 spaces、model、solver、error operator、derivatives，并解释 model-only robustness 不合适。这个段落实际太长，建议至少拆成 3 段。

### Preliminaries P1 建议 1：开头加 purpose sentence

当前：

> Let X and Y be Hilbert function spaces.

建议在它前面加：

> We now define the solver-surrogate objects used to separate model error from solver response.

理由：

不要突然进入 Hilbert spaces。AAAI 文章即使数学化，也通常先告诉读者“下面定义是为了什么”。

### Preliminaries P1 建议 2：model/solver 句更自然

当前：

> We learn a neural operator as a surrogate and use a numerical solver as the oracle.

建议：

> The learned neural operator acts as a fast surrogate, while the numerical solver provides the ground-truth input-output map.

理由：

`oracle` 可以保留，但 `ground-truth input-output map` 更贴合你想强调的监督语境。

### Preliminaries P1 建议 3：model-only robustness 段开头更强

当前：

> This distinction is important for robustness.

建议：

> This distinction changes what robustness should mean for a PDE surrogate.

理由：

原句太弱；新句把它推到 paper thesis。

### Preliminaries P1 建议 4：关键结论句改写

当前：

> The desired local property is instead that the model and solver move together...

建议：

> The desired local property is co-movement: under small input changes, the neural operator and solver should respond similarly, even if neither map is locally flat.

理由：

`co-movement` 是一个很好的解释词，可以帮助读者理解 \(D F \approx D G\)。

---

## 12. Preliminaries P2, lines 292-321

### Preliminaries P2 当前功能

解释 infinitesimal h 和 finite delta。写得清楚，但可以更读者友好。

### Preliminaries P2 建议

当前：

> Thus h denotes a local perturbation direction, while later finite-budget attacks use a finite perturbation delta.

建议：

> We use h for local sensitivity analysis and reserve delta for finite adversarial perturbations optimized under a budget.

理由：

`reserve delta` 是标准写法；句子更简洁。

---

## 13. Preliminaries P3, lines 323-328

### Preliminaries P3 当前功能

说明 function norm 和 discretization。可以保留。

### Preliminaries P3 建议

当前：

> All perturbation budgets and errors are function norms.

建议：

> Unless otherwise stated, perturbation budgets and error magnitudes are measured as function-space norms.

理由：

更学术，也保留默认条件。

---

## 14. Preliminaries: Generalization, lines 361-372

### Preliminaries Generalization 当前功能

定义 generalization。可以略加解释。

### Preliminaries Generalization 建议

当前：

> Generalization is the static model-solver discrepancy at a fixed input function.

建议：

> We define generalization as the static model-solver discrepancy evaluated at a fixed input, possibly drawn from a shifted input distribution.

理由：

把后面 shifted dataset 早一点接进来。

---

## 15. Preliminaries: Robustness opening, lines 374-426

### Preliminaries Robustness opening 当前功能

定义 robustness 和 Jacobian-error。数学清楚，但开头可更强。

### Preliminaries Robustness opening 建议 1

当前：

> Robustness asks how the model-solver error changes when the input function is perturbed. We organize it into three local metrics.

建议：

> Robustness asks whether the model-solver error remains stable under admissible input perturbations. We organize this question into three local diagnostics.

理由：

`diagnostics` 比 `metrics` 更贴近后面 operator norm / Jacobian-error 的比较。

### Preliminaries Robustness opening 建议 2

当前：

> Before taking the first variation, define the operator-level Jacobian-error function...

建议：

> The key local quantity is the Jacobian-error vector, which couples the current error field with the derivative of the error operator.

理由：

先给直觉，再给公式。

---

## 16. Preliminaries: Jacobian operator norm, lines 428-458

### Preliminaries Jacobian operator norm 当前功能

定义 operator norm。写得比较像数学教材。

### Preliminaries Jacobian operator norm 建议

当前：

> The first robustness metric is the induced norm of the derivative of the error operator.

建议：

> The first diagnostic measures worst-case local sensitivity of the error operator.

理由：

`worst-case local sensitivity` 比 `induced norm of derivative` 更能让非数学审稿人立刻理解。

---

## 17. Preliminaries: finite attack loss, lines 460-489

### Preliminaries finite attack loss 当前功能

定义 finite-budget attack。清楚。

### Preliminaries finite attack loss 建议

当前：

> This metric directly measures the worst loss increase over a norm-bounded perturbation ball, rather than a local Jacobian/operator norm.

建议：

> Unlike the local diagnostics above, this finite-budget quantity measures the actual loss increase achieved by an adversarial perturbation within the admissible ball.

理由：

承上启下更清楚：local diagnostics vs finite-budget quantity。

---

## 18. Preliminaries: Jacobian-error metric, lines 491-498

### Preliminaries Jacobian-error metric 当前功能

短，清楚。但可以强调它为什么有用。

### Preliminaries Jacobian-error metric 建议

当前：

> It combines the derivative of the error operator with the current error field.

建议：

> It is therefore error-aligned: it favors directions that increase the current model-solver discrepancy, rather than directions that merely move the error field.

理由：

这句会帮助理解后面和 spectral norm 的区别。

---

## 19. Preliminaries: comparison with finite attack, lines 500-540

### Preliminaries comparison with finite attack 当前功能

证明 small-budget equivalence。可保留。

### Preliminaries comparison with finite attack 建议

当前：

> This comparison explains why the finite-budget attack reduces to the Jacobian-error metric in the small-budget limit.

建议：

> This local calculation links the finite attack objective to the Jacobian-error metric.

理由：

`local calculation links` 更自然；`reduces to` 对一些读者略突然。

---

## 20. Preliminaries: comparison with operator norm, lines 542-586

### Preliminaries comparison with operator norm 当前功能

非常重要，但段落 300+ 词，建议拆成 2 段：objective difference 和 consequence。

### Preliminaries comparison with operator norm 建议 1：开头更锋利

当前：

> The Jacobian-error metric differs from the Jacobian operator norm because the two metrics optimize different local objectives.

建议：

> The Jacobian-error metric and the operator norm answer different robustness questions.

理由：

这是更好的 topic sentence。

### Preliminaries comparison with operator norm 建议 2：结尾更像 takeaway

当前：

> Hence, in our solver-consistent robustness setting, the usual spectral/operator-norm metric is not the most suitable primary robustness metric.

建议：

> Thus, in solver-consistent robustness, spectral norm is useful as a sensitivity diagnostic but should not be the primary proxy for adversarial error growth.

理由：

更温和，也更准确。不是说 spectral norm 没用，而是说 role 不同。

---

## 21. Proposed Framework P1, lines 685-713

### Proposed Framework P1 当前功能

引入 L1/L2/L3。但开头从 notation 出发，建议从 design choice 出发。

### Proposed Framework P1 强烈建议替换第一句

当前：

> The notation above separates three finite attack objective functions.

建议：

> The solver-surrogate setting creates three natural attack targets, distinguished by how the target or ground-truth output is chosen.

后面可接：

> The target can be the model's clean output, the clean-input ground-truth solver output, or the ground-truth solver output recomputed at the perturbed input.

理由：

这就是 AAAI 风格：先说 design choice，再给公式。

### Proposed Framework P1 建议 2：L1/L2/L3 后加一句 taxonomy

建议在公式后加：

> These losses form a solver-coupling hierarchy: L1 is model-only, L2 uses a fixed ground-truth solver output, and L3 fully couples the attack to the solver at the perturbed input.

理由：

`solver-coupling hierarchy` 是本文非常可包装的核心概念。

---

## 22. Proposed Framework P2, lines 715-804

### Proposed Framework P2 当前功能

这是主文最大语言问题：一个段落 600+ 词，信息过载。建议拆成四段。

### Proposed Framework P2 建议拆分结构

#### 22.1 Forward-target variants

当前：

> The high-level losses above have implementation-level variants.

建议：

> In practice, these high-level losses differ along two implementation axes: whether the solver target is updated with the perturbed input, and whether gradients are allowed to pass through the solver.

理由：

一下子告诉读者后面 variant 的组织方式。

#### 22.2 Dictionary attack

当前：

> This is the L2-like structure used by dictionary-based FNO robustness evaluation...

建议：

> Dictionary targets approximate online solver supervision by retrieving a nearby precomputed solver output. This keeps the attack inexpensive, but decouples the target from the exact perturbed input and removes the solver from the backward path.

理由：

更清楚地说 tradeoff：cheap but decoupled。

#### 22.3 Stop-gradient variant

当前：

> A stop-gradient solver variant uses the current solver output in the target loss but blocks the solver branch in backpropagation.

建议：

> The stop-gradient variant updates the forward target but still removes the solver derivative from the attack direction.

理由：

更直接地突出 forward/backward distinction。

#### 22.4 Physics residual baseline

当前：

> This objective is self-supervised: it does not compare against the solver output...

建议：

> The physics-residual objective provides a self-supervised baseline: it enforces equation and boundary consistency of the predicted field without directly matching the solver output at the perturbed input.

理由：

少用 `does not`，改成正向定义。

#### 22.5 Integration hierarchy

当前：

> Overall, these attack variants increase solver integration in the order...

建议：

> Overall, these variants define an integration hierarchy from solver-free or weakly coupled objectives to fully solver-integrated attacks.

理由：

这里应该明确变成本文 framework 的概念，而不是只是列顺序。

---

## 23. Proposed Framework P3, lines 808-824

### Proposed Framework P3 当前功能

说明 optimizer 和 attack objective 分离。写得清楚。

### Proposed Framework P3 建议

当前：

> The losses above specify what is attacked; the optimizer specifies how the perturbation is updated.

建议：

> The attack loss defines the target of the attack, while the optimizer defines how the perturbation searches the admissible ball.

理由：

`what is attacked` 有点口语；`target/searches admissible ball` 更论文。

---

## 24. Proposed Framework P4, lines 868-875

### Proposed Framework P4 当前功能

定义 add vs replace。清楚。

### Proposed Framework P4 建议

当前：

> Add-style methods accumulate...

建议：

> Add-style updates perform projected incremental search, whereas replace-style updates jump directly to a boundary perturbation aligned with the current direction.

理由：

`projected incremental search` / `jump directly to boundary` 形成更强对比。

---

## 25. Proposed Framework P5, lines 877-894

### Proposed Framework P5 当前功能

解释四种 optimizer。可以保留，但 “simple optimization interpretations” 可更正式。

### Proposed Framework P5 建议

当前：

> The four cells in Table ... have simple optimization interpretations...

建议：

> The four optimizer variants correspond to standard projected-gradient or boundary-replacement updates.

理由：

更简洁，也减少 `simple`。

---

## 26. Proposed Framework P6, lines 896-913

### Proposed Framework P6 当前功能

L1 零初始化 caveat。技术上必要，但语言可以更像 “implementation detail”。

### Proposed Framework P6 建议

当前：

> A small implementation caveat follows for L1: it must not be started from the zero perturbation.

建议：

> One implementation detail is important for L1 attacks: zero initialization gives a zero local direction.

理由：

更直接说明原因。

---

## 27. Proposed Framework P7, lines 915-942

### Proposed Framework P7 当前功能

解释 frozen-Jacobian view 只是 heuristic。写得好，但可减少 “only heuristic” 的弱感。

### Proposed Framework P7 建议

当前：

> For solver-integrated losses, any frozen-Jacobian view is only heuristic...

建议：

> For solver-integrated losses, the frozen-Jacobian view is a local approximation rather than a full description of the finite nonlinear attack.

理由：

更准确，也不显得你在削弱自己的方法。

---

## 28. Proposed Framework P8, lines 946-956

### Proposed Framework P8 当前功能

描述 attack-then-train loop。可以更像 algorithm overview。

### Proposed Framework P8 建议

当前：

> The training stage wraps the attack objectives above in an online attack-then-train loop.

建议：

> Solver-integrated adversarial training turns the attack objective into an online data-generation loop.

理由：

`online data-generation loop` 很重要，因为你的方法本质上是自动样本选择。

---

## 29. Proposed Framework P9, lines 958-967

### Proposed Framework P9 当前功能

联系 standard adversarial training。可以更突出差别。

### Proposed Framework P9 建议

当前：

> This is a PDE operator-learning embodiment of standard adversarial training...

建议：

> This mirrors standard adversarial training, but replaces the fixed classification label with either an online solver output or a physics-residual signal.

理由：

承认继承，也强调区别。

---

## 30. Proposed Framework P10, lines 969-974

### Proposed Framework P10 当前功能

随机扰动 baseline。清楚。

### Proposed Framework P10 建议

当前：

> Experimentally, we also compare with two hand-designed random perturbation variants.

建议：

> To separate adversarial sample selection from generic data augmentation, we compare against two hand-designed random perturbation variants.

理由：

这句解释了为什么需要 baseline。

---

## 31. Experiments P1, lines 980-984

### Experiments P1 当前功能

介绍 benchmark。清楚。

### Experiments P1 建议

当前：

> We evaluate on the three canonical PDE operator-learning benchmarks...

建议：

> We evaluate on three canonical PDE operator-learning benchmarks spanning evolution and elliptic problems: 1D Burgers, 2D Darcy Flow, and 2D Navier-Stokes.

理由：

`spanning evolution and elliptic problems` 增加 benchmark 合理性。

---

## 32. Experiments P2-P3, lines 989-1006

### Experiments P2-P3 当前功能

实验设置。可以加 research question。

### Experiments P2-P3 建议

在 `Solver-Integrated Adversarial Attacks` 小节开头加：

> We ask whether stronger solver coupling in the attack objective produces larger true model-solver discrepancy at the attacked input.

理由：

AAAI 实验部分常用 RQ/Question 形式，让表格不只是堆结果。

---

## 33. Experiments main result, lines 1097-1105

### Experiments main result 当前功能

结果总结。很好，但可以更学术。

### Experiments main result 建议

当前：

> The main result is simple: to make the model output differ strongly from the solver output, the attack should optimize the full L3 objective directly.

建议：

> The central empirical finding is that true model-solver discrepancy is maximized most reliably when the attack objective follows the perturbed input through the solver.

理由：

`simple` 有点口语；新句强调本文主张：attack follows perturbed input through solver。

### Experiments main result 建议 2

当前：

> Thus the solver is useful both as the forward target ... and as a backward path ...

建议：

> This shows that solver integration matters in both roles: as the forward target at the perturbed input and as the backward path that shapes the perturbation direction.

理由：

更像 AAAI result interpretation。

---

## 34. Experiments spectral pattern, lines 1107-1114

### Experiments spectral pattern 当前功能

提出 second pattern。写得好。

### Experiments spectral pattern 建议

当前：

> A second pattern is spectral.

建议：

> A second pattern concerns the frequency content of the discovered perturbations.

理由：

更具体，不显突兀。

---

## 35. Experiments optimizer ablation Burgers, lines 1119-1137

### Experiments optimizer ablation Burgers 当前功能

解释 replace-style update 为什么快。写得有机制，是好段落。

### Experiments optimizer ablation Burgers 建议 1

当前：

> The Burgers optimizer ablation gives a somewhat surprising result.

建议：

> The Burgers optimizer ablation reveals a geometry-dependent effect.

理由：

`somewhat surprising` 不如 `geometry-dependent effect` 学术。

### Experiments optimizer ablation Burgers 建议 2

当前：

> The reason appears to be that Burgers has a broad high-loss region...

建议：

> The diagnostics suggest that Burgers has a broad high-loss region...

理由：

更科学、少主观。

---

## 36. Experiments optimizer ablation NS, lines 1139-1151

### Experiments optimizer ablation NS 当前功能

对比 NS。写得清楚。

### Experiments optimizer ablation NS 建议

当前：

> Navier-Stokes shows the opposite tendency.

建议：

> Recurrent Navier-Stokes exhibits the opposite loss-landscape geometry.

理由：

直接承接上一段的 geometry-dependent narrative。

---

## 37. Experiments adversarial training setup, lines 1156-1163

### Experiments adversarial training setup 当前功能

描述训练比较。清楚。

### Experiments adversarial training setup 建议

当前：

> Because the losses and random perturbation variants have different per-batch and per-epoch costs, we compare wall-clock-matched runs rather than equal epoch counts.

建议：

> Because solver integration changes per-batch cost, we compare wall-clock-matched runs rather than equal-epoch runs.

理由：

更短，更抓住原因。

---

## 38. Experiments Generalization paragraph, lines 1165-1178

### Experiments Generalization paragraph 当前功能

结果解释很具体，但句子略长。

### Experiments Generalization paragraph 建议 1

当前：

> Random solver is often the strongest non-Loss3 training variant early in wall-clock time; however, its perturbations come from a fixed hand-designed distribution...

建议：

> Random solver is often the strongest non-Loss3 variant early in wall-clock time. However, because its perturbations are drawn from a fixed hand-designed distribution, they do not adapt to the current model's weak regions and can overfit later.

理由：

拆句后更清楚。

### Experiments Generalization paragraph 建议 2

当前：

> Loss 3 instead regenerates adversarial samples from the current model-solver error.

建议：

> In contrast, Loss 3 regenerates samples from the current model-solver discrepancy, making sample selection adaptive.

理由：

强调 adaptive sample selection，这是你的方法卖点。

---

## 39. Experiments NS limitation, lines 1180-1184

### Experiments NS limitation 当前功能

说明 NS 比较不完整。诚实，但可以更积极。

### Experiments NS limitation 建议

当前：

> memory and wall-clock constraints prevented the same complete comparison.

建议：

> memory and wall-clock constraints limited the Navier-Stokes study to a smaller-scale comparison.

理由：

`prevented` 听起来像失败；`limited ... to` 更中性。

---

## 40. Experiments Robustness paragraph, lines 1186-1200

### Experiments Robustness paragraph 当前功能

总结 robustness table 和 diagnostics。清楚。

### Experiments Robustness paragraph 建议

当前：

> Loss 3 gives the smallest attack loss increase for both Burgers and Darcy Flow and the smallest Jacobian-error norm, so it is the most robust method in this comparison.

建议：

> Loss 3 gives the smallest attack loss increase and Jacobian-error norm on Burgers and Darcy Flow, making it the strongest robustness method among the compared training objectives.

理由：

更紧凑，更准确。

---

## 41. Conclusion P1, lines 1205-1218

### Conclusion P1 当前功能

总结贡献。可以更像 final claim。

### Conclusion P1 建议 1

当前：

> This paper studies the generalizability and robustness of neural operators from a solver-integrated perspective.

建议：

> This paper reframes neural-operator robustness as a solver-consistency problem under input perturbations.

理由：

`reframes` 比 `studies` 更有贡献感。

### Conclusion P1 建议 2

当前：

> We then propose a solver-integrated adversarial attack whose forward loss and backward gradient both use the PDE solver...

建议：

> We then propose a fully solver-integrated attack in which both the forward target and the backward gradient follow the perturbed input through the PDE solver.

理由：

这句是你的方法核心，应该写得更有力。

### Conclusion P1 建议 3

当前：

> clearly improves both generalizability and robustness...

建议：

> consistently improves shifted-input generalization and reduces adversarial model-solver error...

理由：

更具体、更少主观副词。

---

## 42. Conclusion P2, lines 1220-1229

### Conclusion P2 当前功能

限制和未来工作。写得好，但未来方向可以更具体。

### Conclusion P2 建议 1

当前：

> The main limitation is computational cost.

建议：

> The main limitation is the cost of differentiating through the solver.

理由：

更具体，直接点出 bottleneck。

### Conclusion P2 建议 2

当前：

> Future work can study how to reduce the time and memory cost of solver propagation.

建议：

> Future work should reduce this cost through checkpointed adjoints, surrogate-assisted solver gradients, or hierarchical solver-neural rollouts.

理由：

给出更具体的 future directions，会比 “can study how” 更成熟。

---

## 43. 附录 A：Benchmark PDEs, lines 1260-1450

### Appendix A 当前特点

这部分写得像标准技术说明，适合作为 appendix。主要问题是开头可以更明确说明它支撑主文哪个实验。

### Appendix A 建议

当前：

> We evaluate on three canonical PDE operator-learning benchmarks...

建议：

> This appendix specifies the PDE tasks, solver maps, input distributions, and supervised targets used in the main experiments.

后接原来的 benchmark 介绍。

### Appendix A 具体词汇建议

`They represent two different operator types.`  

可改成：

> Together, these benchmarks cover both time-dependent evolution operators and steady coefficient-to-solution operators.

理由：

`cover` / `operators` 更像 benchmark coverage。

---

## 44. 附录 B：Numerical Solver Details, lines 1456-1680

### Appendix B 当前特点

公式和算法细节充分。这里不需要改成 narrative。只需要每个 subsection 开头加一句 role。

### Appendix B 建议句

在 `1D Burgers` 开头加：

> We describe the differentiable solver used to generate both supervised targets and solver-integrated attack gradients.

在 `2D Darcy Flow` 开头加：

> Darcy Flow differs from Burgers and Navier-Stokes because its solver is a steady elliptic solve rather than a time rollout.

在 `2D Navier-Stokes` 开头加：

> The Navier-Stokes solver is the most memory-intensive case because the attack gradient must backpropagate through a long recurrent rollout.

---

## 45. 附录：Batch Differentiation, lines 1682-1757

### Appendix Batch Differentiation 当前特点

这段解释非常清楚，但开头像教材。

### Appendix Batch Differentiation 建议

当前：

> Let X and Y be Hilbert spaces...

建议先加：

> This section clarifies that batch attacks are vectorized collections of independent per-sample attacks, rather than coupled optimization problems across samples.

理由：

读者先知道为什么要看这个推导。

### Appendix Batch Differentiation 结尾句建议

当前：

> The benefit is execution-level amortization...

建议：

> Thus batching is an implementation acceleration, not a change in the mathematical attack objective.

理由：

这是本节 takeaway。

---

## 46. 附录：Differentiating Through the Solvers, lines 1759-1847

### Appendix Differentiating Through the Solvers 当前特点

这部分很有价值，解释 cost bottleneck。语言可更像 limitation analysis。

### Appendix Differentiating Through the Solvers 建议 1

当前：

> The PDE solvers do not introduce learned parameters, but differentiating through a time-dependent solver introduces many latent solver states.

建议：

> Although the PDE solvers have no learned parameters, differentiating through them creates a long computational tape of solver states.

理由：

`computational tape` 是关键概念。

### Appendix Differentiating Through the Solvers 建议 2

当前：

> There is also a serial time-depth cost.

建议：

> The second bottleneck is serial time depth.

理由：

更像分点分析。

---

## 47. 附录：Wall-Clock Time and GPU Memory, lines 1848-2035

### Appendix Wall-Clock Time and GPU Memory 当前特点

非常具体，但表格前的解释可以提炼成 takeaway。

### Appendix Wall-Clock Time and GPU Memory 建议

当前：

> The solver-tape discussion above explains the large cost gap...

建议：

> The timing results quantify the solver-tape bottleneck described above.

理由：

更短、更直接。

### Appendix Wall-Clock Time and GPU Memory 表 caption 建议

如果 caption 很长，建议把统计细节移到正文一句或 table note。caption 先说主旨：

> Component timing shows that solver forward/backward passes dominate full solver-integrated attacks.

---

## 48. 附录：Attack Optimizer / Binary Flip, around lines 2036-2268

### Appendix Attack Optimizer / Binary Flip 当前特点

技术定义充分。建议在每个 algorithm 前说明它回答哪个主文问题。

### Appendix Attack Optimizer / Binary Flip 建议句

> This appendix gives the optimizer details behind Section 4.2, separating the attack objective from the projected update rule.

对 binary flip：

> The Darcy attack is discrete: the optimizer ranks candidate coefficient flips rather than applying continuous norm-ball projection.

---

## 49. 附录：Random Process Identities, lines 2309-2480

### Appendix Random Process Identities 当前特点

这部分像概率/谱方法讲义。内容可以保留，但要更明显地服务于 generalization datasets。

### Appendix Random Process Identities 建议

当前：

> Let xi(x) be a second-order random field...

建议先加：

> We use spectral random-field constructions to generate controlled distribution shifts for the generalization benchmark.

理由：

否则读者会问：为什么突然讲 Wiener-Khinchin 和 Bochner？

### Appendix Random Process Identities 结尾句建议

在 operator diagonalization 后加：

> These identities justify sampling shifted input distributions by modifying spectral amplitudes while keeping the PDE solver fixed.

---

## 50. 附录：Generalization Dataset Construction, lines 2605-2683

### Appendix Generalization Dataset Construction 当前特点

很有用，但列表式参数多。建议开头先说实验设计目的。

### Appendix Generalization Dataset Construction 建议

当前：

> Across the shifted datasets, the solver, grid, and target-generation pipeline are fixed while the input distribution changes.

这是好句，可以再增强：

> Across all shifted datasets, only the input distribution changes; the PDE, grid, solver, and target-generation pipeline remain fixed.

理由：

这个实验控制变量非常重要，应该写得最清楚。

---

## 51. 附录：Metric Computation, lines 2864-3180

### Appendix Metric Computation 当前特点

清楚，适合 appendix。建议每个 metric 的第一句更像 “what it measures”。

### Appendix Metric Computation 示例修改

当前：

> The operator-norm diagnostic is the leading singular value...

建议：

> The operator-norm diagnostic measures the largest local displacement of the error field, independent of whether that displacement increases the current error magnitude.

理由：

先讲含义，再给公式。

当前：

> This is a scalar table only; it does not compare vector directions.

建议：

> These correlations compare scalar magnitudes only; directional agreement is evaluated separately below.

理由：

更正式，也更自然。

---

## 52. 附录：Singular-Function / Error-Vector Geometry, around lines 3185-3485

### Appendix Singular-Function / Error-Vector Geometry 当前特点

这是文章理论解释的重要附录。建议把它包装成 “mechanism behind the empirical pattern”。

### Appendix Singular-Function / Error-Vector Geometry 建议开头

> This appendix explains the geometric mechanism behind the empirical gap between model-only sensitivity, operator-norm diagnostics, and Loss-3 attack directions.

理由：

它能把附录从“额外推导”变成“机制解释”。

### Appendix Singular-Function / Error-Vector Geometry 词汇建议

`This is only a local linear probe`  

建议：

> This analysis is a local linear probe rather than a global characterization of the nonconvex finite-budget attack.

理由：

更正式。

---

## 53. 附录：Loss Landscape / Figure Captions, around lines 3484-4239

### Appendix Loss Landscape / Figure Captions 当前特点

caption 很长，解释很完整，但有些 caption 可以更像 figure takeaway。

### Appendix Loss Landscape / Figure Captions 建议原则

每个 figure caption 第一句写结论，不要第一句写文件/设置。

弱一些：

> Burgers Loss 3 mean curves for ...

更强：

> Replacement updates reach the perturbation boundary faster than additive updates in Burgers, explaining their early attack advantage.

然后再写设置。

### Appendix Loss Landscape / Figure Captions 具体词汇

`We clearly observe`  

建议：

> The results show

或：

> The diagnostics indicate

理由：

`clearly observe` 有主观感；`diagnostics indicate` 更审稿友好。

---

## 54. 最值得立刻改的 12 个句式

1. `commonly utilized` -> `widely used`
2. `not yet clearly defined` -> `traditional adversarial robustness definitions become misaligned in the operator-learning setting`
3. `This setting also differs` -> `This setting introduces input-dependent solver responses`
4. `trusted oracle` -> `a solver that provides input-dependent ground-truth outputs` / `supervision oracle`
5. `not expected to be robust by itself` -> `not expected to be invariant to input perturbations`
6. `Building on this view` -> `To address this solver-response alignment issue`
7. `They do not put...` -> `They treat the target as fixed or model-derived, leaving...`
8. `This is the key difference` -> `This distinction matters because...`
9. `The notation above separates...` -> `The solver-surrogate setting creates...`
10. `This objective is self-supervised: it does not...` -> `The physics-residual objective provides a self-supervised baseline: it enforces... without...`
11. `The main result is simple` -> `The central empirical finding is...`
12. `clearly improves` -> `consistently improves` / `substantially reduces`

---

## 55. 建议新增的全文关键词体系

建议在摘要、引言、方法、实验中反复使用同一套词：

- `solver-response alignment`
- `solver-faithfulness`
- `model-solver discrepancy`
- `solver-coupling hierarchy`
- `forward target`
- `backward path`
- `discrepancy-seeking samples`
- `solver-integrated training`
- `shifted-input generalization`
- `attack-time robustness`

这会让全文语言成长体系更统一：问题、方法、实验、结论都围绕同一套词展开。
