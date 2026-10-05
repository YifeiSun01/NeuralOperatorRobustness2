# Multi-fidelity 文献逐篇结构化审读：design、fidelity、performance 与 cost

> 审读日期：2026-10-05  
> 范围：原始清单中可唯一识别的 26 篇核心论文。清单第 27 项“其他 HPO / NAS 文献”是一个文献类别，不是单篇论文，故不虚构成第 27 篇。  
> 证据等级：A = 已审读全文或可检索全文；B = 正文受访问限制，依据正式摘要、作者稿信息及前后关联文献审读，所有无法核实的细节均明确保留。  

## 一、先给结论

用户提出的统一框架很有用，但它并不是这 26 篇论文共同原生采用的框架。最稳妥的统一写法是

$$
y \sim p\!\left(y\mid x,z\right), \qquad c \sim p\!\left(c\mid x,z\right),
$$

其中 $x\in\mathcal X$ 是设计、配置或物理输入，$z\in\mathcal Z$ 是信息源、数值分辨率、训练资源或求解进度，$y$ 是观测到的性能、物理量、场或优化界，$c$ 是获得该观测的成本；目标通常是了解参考保真度 $z^\star$ 下的 $y(x,z^\star)$ 或求其最优 $x$。

但需要保留四个关键例外：

1. **fidelity 不一定是一维标量。** BOCA、JAHS-Bench-201、DES 模型配置与 EV 充电站论文都有多维 fidelity。
2. **fidelity 不一定有全序。** MISO 的不同信息源是类别变量；多维 fidelity 通常最多只有偏序；DES 中的多个简化模块组合甚至可能不可比较。
3. **低保真输出不一定是“带误差的同一个答案”。** Benders 论文的低保真结果是松弛问题给出的下界与有效割；Hyperband 只把资源看成预算；Branke 主要学习排序是否会翻转。
4. **cost 往往没有被真正建模。** 许多论文只给每一级一个常数成本。明确学习 $c(x,z)$ 的代表是 FABOLAS 和 JAHS-Bench-201；MISO 允许给定的 $c(x,z)$；EV 论文直接测量 design–fidelity 对应的仿真时间。

因此，分析任何 multi-fidelity 方法时，不能只问“有几个 fidelity level”，而要问：$z$ 的语义是什么、是否能排序、怎样与 $x$ 分离、输出是不是同一种量、成本是否依赖 $x$，以及跨 fidelity 的关系到底是先验假设还是从数据中学出来的。

## 二、固定的 10 个审读问题

以下 26 篇均按同一套问题回答：

1. **研究对象与任务是什么？** 是物理仿真、求解器、HPO、NAS、PINN、离散事件仿真还是数学规划？
2. **design / input space $\mathcal X$ 是什么？** 列出具体变量，并说明连续、整数、类别、条件或结构化属性。
3. **fidelity space $\mathcal Z$ 是什么？** 到底改变了网格、迭代次数、数据量、训练轮数、模型来源还是别的东西？
4. **$\mathcal Z$ 的结构是什么？** 一维或多维；离散或连续；全序、偏序、无序类别，还是层级/异构结构？
5. **reference / target fidelity $z^\star$ 是什么？** 是否存在唯一 ground truth；它是极限、最大预算、真实系统，还是仅为更可信的信息源？
6. **输出、performance 与 accuracy 怎么定义？** 输出是标量、向量、场、验证准确率、预测误差、排序、下界还是最优性间隙？
7. **cost 怎么定义？** 是 CPU/墙钟时间、迭代数、样本数、FLOPs、预算单位还是求解时间；是否允许依赖 $x$？
8. **$x,z,y,c$ 的关系如何建模？** 是 GP/co-kriging、神经网络、显式函数、概率分布、误差界、经验回归、调度规则，还是完全不建模？
9. **哪些关系是先验假设，哪些从数据学习？** 参数如何拟合或在线更新；假设有多通用？
10. **如何选择下一次 $(x,z)$，适用性与限制是什么？** 并判断本文与统一框架是“原生匹配”“可重解释”还是“部分不适用”。

## 三、横向总表

| # | 论文/简称 | $x$ | $z$ | $z$ 类型 | 关系模型 | cost 处理 | 框架匹配 |
|---:|---|---|---|---|---|---|---|
| 1 | Kennedy–O’Hagan | 一般连续输入；油藏例为 10 维地质参数 | 代码版本 | 离散、有序、多级 | 自回归 co-kriging | 每级经验常数 | 可重解释 |
| 2 | Forrester 2007 | 4 个机翼几何量 | Tadpole / VSaero | 二级离散有序 | co-kriging | 实测运行时间 | 原生匹配 |
| 3 | Peherstorfer survey | 应用相关 | 任意低/高保真模型 | 可离散、连续、异构、无序 | 分类总结 | 应用相关 | 元框架 |
| 4 | Fernández-Godino review | 应用相关 | 简化、离散化、代理等 | 多种 | 校正/融合综述 | 应用相关 | 元框架 |
| 5 | Partially converged CFD | 翼型/机翼几何 | CFD 迭代次数 | 一维离散有序 | 部分解 + 差异代理 | 迭代数/CPU | 原生匹配 |
| 6 | Picheny–Ginsbourger | CFD 设计参数 | 计算时间 $t$ | 一维连续有序 | 非平稳时空 GP | $t$ 作成本代理 | 原生匹配 |
| 7 | Branke 2017 | Toysub 8 个几何量等 | 求解器检查点 | 一维离散有序 | 排序翻转概率 | 增量迭代成本 | 部分匹配 |
| 8 | Courrier 2014 | 结构设计参数 | LATIN 全局迭代状态 | 一维离散有序 | kriging/演化融合 | CPU 时间 | 原生匹配，证据 B |
| 9 | MF-GP-UCB | 连续黑箱输入 | 有限多个近似 | 离散、严格有序 | 独立 GP + 已知误差界 | 每级常数 | 原生匹配 |
| 10 | BOCA | 连续黑箱输入 | 连续 fidelity 向量 | 多维连续、通常偏序 | 乘积核 GP | 已知 $\lambda(z)$ | 原生匹配 |
| 11 | FABOLAS | 混合 HPO 变量 | 数据集比例 | 一维连续有序 | loss GP + log-cost GP | 学习 $c(x,z)$ | 原生匹配 |
| 12 | Hyperband | 任意超参数配置 | 资源预算 | 一维离散有序 | 无响应面；淘汰调度 | 与预算近似成正比 | 部分匹配 |
| 13 | HPO survey | 混合、条件超参数 | epoch/样本/折数等 | 方法相关 | 方法综述 | 多种 | 元框架 |
| 14 | MF-PINN | PDE 参数 | 网络容量 + 停止准则 | 组合后两级离散 | 低秩线性组合 | 实测训练时间 | 可重解释 |
| 15 | JAHS-Bench-201 | NAS 拓扑 + HPO | 深度、宽度、分辨率、epoch | 4 维离散偏序 | 每项指标的 XGBoost | 学习运行时间 | 原生匹配 |
| 16 | MF-KD NAS | NAS-Bench-201 拓扑 | KD 1 epoch / 普通训练 12 epoch | 二级离散，机制异构 | co-kriging | 实测训练时间 | 可重解释 |
| 17 | DNN-MFBO | 连续黑箱输入 | 有序信息级 | 离散多级 | 深度自回归 NN | 每级常数 | 原生匹配 |
| 18 | BMBO-DARN | 混合 HPO 输入 | 训练资源 | 离散多级有序 | 贝叶斯自回归 NN | 每级常数 | 原生匹配 |
| 19 | DMFAL | PDE/几何参数 | 网格分辨率 | 离散多级有序 | 低维潜变量自回归 NN | 每级常数 | 原生匹配 |
| 20 | IFC | PDE/几何参数 | 连续网格尺度编码 | 一维连续有序 | fidelity 方向神经 ODE | 未建模 | $y$ 匹配，$c$ 缺失 |
| 21 | Composite NN | 时空/物理输入 | LF / HF 数据源 | 二级离散有序 | 线性+非线性复合 NN | 未建模 | 数据融合型 |
| 22 | MISO | 连续决策变量 | 真值及多个信息源 | 离散类别、无需有序 | truth GP + source discrepancy GP | 给定 $c_z(x)$ | 原生匹配 |
| 23 | MF-MES | 连续黑箱输入 | 有限精度级 | 离散多级有序 | 多输出 GP | 每级常数 | 原生匹配 |
| 24 | DES OMFSM | 系统设计另行定义；模型配置本身被优化 | 模块精度、预热期、仿真预算 | 混合、多维、部分不可比 | 随机多目标仿真优化 | 实测墙钟时间 | 需分离两层变量 |
| 25 | EV 充电站 AOMS | 5 个站点设计量 | 天数、时间步、到达率分段 | 3 维配置，预设离散层级 | 多级无偏估计 + PCS 分配 | 实测 $c(x,z)$ | 原生匹配 |
| 26 | MF Benders | 设备投资二元变量 | 子问题时间跨度/边界耦合 | 4 级离散有序 | 松弛下界与有效割 | 求解时间，不学习 | 优化松弛型 |

## 四、逐篇审读

### 1. Kennedy & O’Hagan (2000), *Predicting the Output from a Complex Computer Code When Fast Approximations Are Available*

**证据：A。** [期刊页面](https://academic.oup.com/biomet/article-abstract/87/1/1/221217)；[全文镜像](https://www2.stat.duke.edu/courses/Spring14/sta961.01/ref/KennOHag2000.pdf)；本地全文： [01_kennedy_ohagan_2000.pdf](papers/01_kennedy_ohagan_2000.pdf)。

1. **研究对象与任务。** 用少量昂贵计算和较多廉价近似代码运行，预测昂贵确定性计算机代码的输出及其不确定性；论文核心是多层计算机实验的统计融合，不是成本感知的序贯优化。
2. **$\mathcal X$。** 一般形式是 $p$ 维连续代码输入。油藏案例中是 5 个地质区域各自的孔隙率和渗透率，共 10 个连续变量；输出为某一时刻的井压。
3. **$\mathcal Z$。** 不同精度的代码版本。油藏案例是粗网格与细网格两个版本；理论允许 $s$ 个层级。
4. **结构。** 有限、离散、从便宜粗糙到昂贵精确的全序层级；$z$ 本身不被当作连续数值输入，而是通过层级递推进入模型。
5. **$z^\star$。** 最昂贵的第 $s$ 层代码。它是目标计算代码而非物理世界的绝对 ground truth，因此仍可能有模型偏差。
6. **输出/accuracy。** 标量确定性代码输出；accuracy 由后验预测均值、方差及验证误差体现。案例比较了对细网格井压的预测。
7. **cost。** 只用每级相对计算开销描述；案例中细网格约需 1–3 天，约等价于 36 次粗网格运行。没有显式 $c(x,z)$，默认同一级的成本近似恒定。
8. **关系模型。** 递推式 $Z_t(x)=\rho_{t-1}Z_{t-1}(x)+\delta_t(x)$；$Z_1$ 与各层独立 discrepancy $\delta_t$ 均为 GP。常用嵌套设计使高保真采样点属于低保真采样点集合。
9. **假设与学习。** 先验假设是层间近似线性缩放、加性独立差异及平稳相关结构；尺度、均值、协方差超参数由数据估计/贝叶斯更新。线性自回归是强结构假设，不适合层间非线性或局部关系剧变的情形。
10. **选择与限制。** 论文重点是已给定运行的融合，没有联合 acquisition 选择 $(x,z)$。它可重解释为统一 $g(x,z)$，但原模型更自然的表示是“每个离散层一条函数并递推连接”，且 cost 不进入决策。

### 2. Forrester, Sóbester & Keane (2007), *Multi-fidelity Optimization via Surrogate Modelling*

**证据：A。** [DOI](https://doi.org/10.1098/rspa.2007.1900)；[可检索全文](https://www.researchgate.net/publication/228677353_Multi-fidelity_optimization_via_surrogate_modelling)。

1. **研究对象与任务。** 空气动力学机翼设计；利用廉价经验模型和昂贵面元/黏性分析，在少量高保真调用下最小化阻力。
2. **$\mathcal X$。** 案例为四个连续机翼几何变量：面积 $S$、展弦比 $AR$、后掠角 $\Lambda$、内翼锥度 $T_{in}$；固定升力条件下优化。
3. **$\mathcal Z$。** 两个求解器：廉价 Tadpole 经验阻力模型与昂贵 VSaero 势流加黏性分析。
4. **结构。** 二级、离散、按可信度和成本全序；一个 fidelity 对应一个模型，而不是连续精度旋钮。
5. **$z^\star$。** VSaero 输出作为优化目标的参考 fidelity，但它仍是数值气动模型，不是实验真值。
6. **输出/accuracy。** 标量阻力目标 $D/q$；固定升力。代理的准确性通过高保真预测误差与最终设计的高保真评价衡量。
7. **cost。** Tadpole 一次约 0.6 秒，VSaero 一次约 2 分钟；成本以实测运行时间和调用数体现，没有学习随 $x$ 变化的 $c(x,z)$。
8. **关系模型。** 与 Kennedy–O’Hagan 相同的 co-kriging：高保真函数是缩放后的低保真 GP 加独立 discrepancy GP；在该模型上计算高保真 expected improvement。
9. **假设与学习。** 加性差异、全局缩放和平稳核是先验结构；核参数及缩放系数用极大似然拟合。是否适合取决于两个气动模型的误差能否被平滑差异过程表示。
10. **选择与限制。** 初始约 100 个低保真点与其中 20 个高保真点，随后按高保真 EI 增加高保真样本。它原生符合 $x,z,y,c$ 叙述，但后续阶段并不真正联合选择 fidelity，且成本只作为实验账本而非 acquisition 分母。

### 3. Peherstorfer, Willcox & Gunzburger (2018), *Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization*

**证据：A。** [arXiv](https://arxiv.org/abs/1806.10761)；[期刊页面](https://epubs.siam.org/doi/abs/10.1137/16M1082469)；本地全文： [03_peherstorfer_et_al_2018.pdf](papers/03_peherstorfer_et_al_2018.pdf)。

1. **研究对象与任务。** 这是综述，不是单一算法；覆盖不确定性传播、统计推断和优化中的多保真方法。
2. **$\mathcal X$。** 随具体问题而变，可为随机输入、物理参数、设计变量或控制量。
3. **$\mathcal Z$。** 高保真模型 $f_{hi}$ 与一个或多个 $f_{lo}^{(i)}$；低保真来源包括简化物理、粗离散、降阶投影以及数据拟合模型。
4. **结构。** 没有统一几何；可以是离散层级、连续离散化参数、多个异构模型，且多个 LF 未必互相有序。
5. **$z^\star$。** 通常是被认为最可信的高保真模型或真实数据，但综述强调“高保真”不等于无误差真值。
6. **输出/accuracy。** 可以是标量、场、统计量、后验或优化目标；accuracy 因任务而异，包括误差范数、方差、偏差和优化遗憾。
7. **cost。** 计算时间、样本开销或资源预算，随应用定义；综述没有统一成本函数。
8. **关系模型。** 将方法分为 adaptation、fusion、filtering 等类别，包含控制变量、重要性采样、降阶模型、校正模型、co-kriging 等；不存在一条共同函数式。
9. **假设与学习。** 有的方法靠解析误差/相关性，有的从配对数据学习，有的只要求 LF 与 HF 高相关。文章的重要贡献恰是说明“便宜”与“有用”必须同时考虑，单说低精度不够。
10. **选择与限制。** 不提供统一 $(x,z)$ acquisition。用户框架适合作为二次分类语言，但不应声称该综述把所有方法都写成同一个 $g(x,z),c(x,z)$。

### 4. Fernández-Godino (2023), *Review of Multi-fidelity Models*

**证据：A。** [arXiv 版本](https://arxiv.org/abs/1609.07196)；[DOI](https://doi.org/10.3934/acse.2023015)；本地全文： [04_fernandez_godino_2023.pdf](papers/04_fernandez_godino_2023.pdf)。

1. **研究对象与任务。** 工程设计与分析中的多保真模型综述，重点是怎样构造、校正和评价融合代理。
2. **$\mathcal X$。** 应用相关的工程设计/物理输入，可连续或离散；综述本身没有固定 design space。
3. **$\mathcal Z$。** 物理简化、网格/时间步、经验公式、响应面、降阶模型等多种来源。
4. **结构。** 常见叙述是 LF/HF 两级离散，但文章覆盖多层与异构模型；不同来源并不天然形成一维连续或全序空间。
5. **$z^\star$。** 最可信仿真或试验数据；它是否为 ground truth 取决于具体研究。
6. **输出/accuracy。** 工程响应、目标或约束；评价可用 RMSE、相关性、交叉验证误差和最终优化解质量。
7. **cost。** 通常是运行时间或模型调用数，但没有统一 $c(x,z)$。
8. **关系模型。** 综述加性/乘性校正、space mapping、co-kriging、响应面融合以及模型管理策略。
9. **假设与学习。** 各方法分别假定平滑 discrepancy、比例关系、几何映射或统计相关性；系数可回归/极大似然拟合。没有一条全领域都合理的 fidelity–accuracy 定律。
10. **选择与限制。** 文章用于方法分类与选型，而非一个 acquisition 算法。用户框架是合适的审读模板，但只是对综述内容的重新组织。

### 5. Forrester, Bressloff & Keane (2006), *Optimization Using Surrogate Models and Partially Converged CFD Simulations*

**证据：A。** [DOI](https://doi.org/10.1098/rspa.2006.1679)；[可检索全文](https://www.researchgate.net/publication/228637097_Optimization_using_surrogate_models_and_partially_converged_computational_fluid_dynamics_simulations)。

1. **研究对象与任务。** 气动形状优化；把尚未完全收敛的 CFD 解当作廉价信息，而不是每个 fidelity 都换一个物理模型。
2. **$\mathcal X$。** 主要二维翼型案例用 4 个连续基函数权重 $w_2,\ldots,w_5$ 描述形状并最小化定升力阻力。三维机翼案例还含展弦比、折点、后掠、锥度、扭转及根/折点/翼尖形状系数等连续变量。
3. **$\mathcal Z$。** CFD 非线性求解的迭代次数/收敛程度；可在多个迭代检查点停止。
4. **结构。** 一维、离散、有序；理论上可取许多迭代数。不同 fidelity 是同一次求解轨迹的部分状态，能够继续计算。
5. **$z^\star$。** 充分收敛的 CFD；案例中完整运行约 1500 次迭代。
6. **输出/accuracy。** 阻力系数等标量气动响应；用部分收敛与完全收敛响应面之间的 $r^2$、预测误差和最终全收敛目标评价。
7. **cost。** 主要以迭代数乘设计点数和 CPU 时间衡量；同一迭代预算下近似认为成本不随 $x$ 变化。
8. **关系模型。** 先建部分收敛代理，再用少量完全收敛点学习差异代理，形成 $\hat f_{full}(x)=\hat f_{part}(x)+\hat f_{diff}(x)$；用 EI 推动昂贵更新。
9. **假设与学习。** 假定部分解已保留足够空间趋势、差异函数更平滑/更容易学；通过试验估计不同迭代数的相关性。目标数值不必随迭代单调，但相关性应逐渐增强。
10. **选择与限制。** 通过 correlation–cost 权衡选择部分收敛程度和样本规模，再用高保真 EI 优化。它非常接近统一框架，但 $z$ 是可续算的 solver state；若求解过程早期排名混乱，低 fidelity 会误导。

### 6. Picheny & Ginsbourger (2013), *A Nonstationary Space-Time Gaussian Process Model for Partially Converged Simulations*

**证据：B。** [正式 DOI/摘要](https://doi.org/10.1137/120882834)；[作者预印本记录](https://hal.science/hal-00579876v3)。出版正文与预印本下载均受站点访问控制，但正式摘要和已索引的作者预印本正文片段可交叉核实；未核实内容不臆测。

1. **研究对象与任务。** 预测随计算时间逐渐收敛的迭代仿真；案例是以 OpenFOAM/simpleFoam 求解的二维 S 形管道稳态、不可压湍流，把“尚未收敛”作为可统计建模的时间维度。
2. **$\mathcal X$。** 管道轮廓由固定控制点和 7 个连续几何参数 $x_1,\ldots,x_7$ 定义；其上下界依次为 $[4,11]$、$[15,45]$、$[5,20]$、$[5,11]$、$[20,60]$、$[9,60]$、$[9,60]$。为清楚展示 design–time 曲面，论文又固定其余 6 个量，只让最敏感的 $x_2$ 在区间内变化。
3. **$\mathcal Z$。** 计算时间/迭代进度 $t$，而非不同求解器标签。
4. **结构。** 模型中是一维、连续、有序的 time coordinate；底层数据实际来自整数 solver steps，因此“连续”是 GP 对离散迭代轨迹的连续化表示。
5. **$z^\star$。** 理论为极限响应 $F(x)$；案例用第 500 次 solver iteration 的值作为充分收敛参考。
6. **输出/accuracy。** 标量目标 $f_{SD}$，即出口截面两点 $P_9$ 与 $P_{10}$ 之间的流速标准差，越小表示出口速度越均匀。accuracy 是对 500-step/极限响应的预测误差与后验不确定度。
7. **cost。** $t$ 自身就是主要成本代理；没有另外学习 $c(x,t)$，通常隐含成本随运行时间增加。
8. **关系模型。** $Y(x,t)=F(x)+G(x,t)$；$F$ 是收敛极限的 GP，$G$ 是随时间衰减的非平稳 GP 误差，在 design 与 time 两个方向共同相关。
9. **假设与学习。** 结构假设包括误差幅度随时间消失、晚期轨迹比早期更平滑、邻近 $x$ 的轨迹相关；核超参数从轨迹数据估计。这比简单把迭代数当普通平稳输入更贴合收敛物理，但不适合振荡、不稳定或多吸引子的求解器。
10. **选择与限制。** 论文核心是时空预测，而非完整的成本感知联合 acquisition。它原生适合连续 $z$ 的统一框架，但 $c$ 只由 $t$ 暗示；其“连续 fidelity”来自把 solver step 连续化，而不是求解器真的接受任意实数步数。

### 7. Branke et al. (2017), *Efficient Use of Partially Converged Simulations in Evolutionary Optimization*

**证据：A。** [作者稿全文](https://wrap.warwick.ac.uk/id/eprint/79047/7/WRAP_CFD_TEC_merged_final.pdf)；本地全文： [07_branke_et_al_2017.pdf](papers/07_branke_et_al_2017.pdf)。

1. **研究对象与任务。** 在进化优化中利用可暂停、可继续的迭代仿真；核心不是预测最终数值，而是尽早判断一个候选解是否值得继续算。
2. **$\mathcal X$。** 合成例为 1–2 维；Toysub 潜艇案例为 8 个连续几何变量：$Z_C,Z_V,Z_L,Z_B$ 四个部件位置，尾部直径 $d_t$、尾部长度 $l_t$、鼻部形状系数 $n_n$、鼻部长度 $l_n$。目标最小阻力并满足体积约束。
3. **$\mathcal Z$。** 求解器的离散迭代检查点。Toysub 使用 FLUENT 的 5、10、25、50、75、100 次迭代。
4. **结构。** 一维、离散、全序且可续算；成本是从当前检查点继续到下一检查点的增量。
5. **$z^\star$。** 100 次迭代的最终评价，在论文实验中被当作参考结果。
6. **输出/accuracy。** 输出为候选设计的适应度/阻力。算法真正关心的是两个个体在当前 fidelity 的排序到最终是否翻转，而不是低保真数值的 RMSE。
7. **cost。** 用累计迭代次数衡量，Toysub 六级成本即 5、10、25、50、75、100；默认与 $x$ 无关并可增量续算。
8. **关系模型。** 对每一级，用已经完整评估过的个体对训练 logistic regression，估计给定当前适应度差时最终排名翻转的概率。
9. **假设与学习。** 不要求部分收敛数值单调，却假定“当前差距”对最终排名可靠性有预测力，且历史比较对新个体可迁移；回归参数在线更新。若轨迹强烈交叉且当前差距无信息，模型会学到该级不值得使用。
10. **选择与限制。** 新子代先低成本评价，若误排风险仍高则逐级提升，直到能作选择或达到最高级。统一框架只能部分描述它：$y(x,z)$ 存在，但决策变量实际依赖一对候选的排序风险，且可续算结构至关重要。

### 8. Courrier, Boucard & Soulier (2014), *The Use of Partially Converged Simulations in Building Surrogate Models*

**证据：B。** [DOI/摘要](https://doi.org/10.1016/j.advengsoft.2013.09.008)。正文获取受限；以下仅保留摘要与可交叉核实内容，具体算例设计变量未作推测。

1. **研究对象与任务。** 结构装配、接触/摩擦问题中的昂贵非线性仿真；利用部分收敛解建立工程代理。论文有一个 academic case 和一个类 industrial case，目标均涉及最小化某个响应函数的范数。
2. **$\mathcal X$。** 记为 $x\in D\subset\mathbb R^p$ 的结构设计参数。现有证据不足以逐一核实算例的参数名称。
3. **$\mathcal Z$。** LATIN 求解器的全局迭代状态/收敛误差水平；该算法的全局迭代覆盖整个加载历程，故比逐载荷步的 Newton 中间状态更适合作统一 fidelity。实验直接比较若干离散 error levels。
4. **结构。** 一维、离散、有序、可续算；每个迭代检查点对应同一结构问题的近似解。
5. **$z^\star$。** 达到收敛准则的 LATIN 解。
6. **输出/accuracy。** 标量目标 $F(x)$/结构响应范数；用部分收敛 metamodel 与 reference metamodel 的 determination coefficient、目标区定位能力及高保真校正后的表现衡量。
7. **cost。** 以 CPU/墙钟时间计算；论文报告利用部分收敛结果可获得约数量级的节省，但没有通用 $c(x,z)$ 回归。
8. **关系模型。** 以部分收敛样本建 kriging 类代理，并用 evofusion correction 以少量完整解更新。论文测试两种策略：global enrichment 建完整精确代理，以及用 weighted expected improvement 定位全局最小区域。
9. **假设与学习。** 假定迭代推进使近似解的趋势逐步接近完整解，残差能由平滑代理学习；相关和校正由配对数据估计。结构响应在接触状态切换处可能不平滑，是主要风险。
10. **选择与限制。** 实验总结约 10 个点/维、误差水平约 $7\times10^{-2}$ 至 $5\times10^{-2}$ 时可取得较好成本收益，并报告寻找最优区域约十倍节省；这些是案例经验而非通用定律。框架原生适用，但当前证据仍不足以断言具体 $p$、各 design 变量名称/范围及全部预算。

### 9. Kandasamy et al. (2016), *Gaussian Process Bandit Optimisation with Multi-fidelity Evaluations*

**证据：A。** [NeurIPS 全文](https://papers.neurips.cc/paper_files/paper/2016/file/605ff764c617d3cd28dbbdd72be8f9a2-Paper.pdf)；本地全文： [09_kandasamy_et_al_2016.pdf](papers/09_kandasamy_et_al_2016.pdf)。

1. **研究对象与任务。** 在有限总成本下优化昂贵黑箱函数，用多个便宜但有偏的近似函数辅助。
2. **$\mathcal X$。** 理论上 $\mathcal X=[0,r]^d$ 连续紧集。实验包括 SVM 的带宽/软间隔两个超参数、SALSA 的 6 个超参数、Viola–Jones 的 22 个阈值和超新星宇宙学的 $H_0,\Omega_M,\Omega_\Lambda$。
3. **$\mathcal Z$。** $M$ 个近似函数 $f^{(1)},\ldots,f^{(M)}$。实例通过训练数据量或数值积分网格改变：SVM 500/2000 样本，SALSA 2k/4k/8k，Viola–Jones 300/3000 图像，超新星积分网格 $10^2/10^4/10^6$。
4. **结构。** 有限、离散、严格有序；层级越高误差上界越小而成本越大。
5. **$z^\star$。** 第 $M$ 级函数 $f^{(M)}$，即目标黑箱。
6. **输出/accuracy。** 带噪标量 $y=f^{(m)}(x)+\epsilon$；performance 是目标级函数值和累计成本下的优化 regret。跨级 accuracy 由上界 $\zeta^{(m)}$ 描述。
7. **cost。** 每级给定固定成本 $\lambda^{(m)}$，满足递增关系；理论不允许同级成本随 $x$ 变化，实验另报告 CPU 时间。
8. **关系模型。** 每个 fidelity 使用独立 GP；不同层之间不共享协方差。联系仅来自确定性界 $\lVert f^{(M)}-f^{(m)}\rVert_\infty\le\zeta^{(m)}$，并把每层 UCB 加上该 bias 界。
9. **假设与学习。** GP 核内参数可由数据设定/估计，但关键 $\zeta^{(m)}$ 在理论中已知，且成本、误差严格排序；实践中作者给出适应性启发式估计。全域 sup-norm 界通常很难可信地预先获得。
10. **选择与限制。** 先以各层 UCB 的最小包络选择 $x$，再选最低一个仍具有足够后验不确定性的 fidelity，否则升级。它原生符合统一框架，但其跨 fidelity 模型其实不是联合 $g(x,z)$；强误差界承担了关联作用。

### 10. Kandasamy et al. (2017), *Multi-fidelity Bayesian Optimisation with Continuous Approximations*（BOCA）

**证据：A。** [PMLR 全文](https://proceedings.mlr.press/v70/kandasamy17a/kandasamy17a.pdf)；本地全文： [10_kandasamy_et_al_2017_boca.pdf](papers/10_kandasamy_et_al_2017_boca.pdf)。

1. **研究对象与任务。** 用可连续调节的一组近似来优化目标 fidelity，而不是预先只给几个离散模型。
2. **$\mathcal X$。** $\mathcal X=[0,1]^d$ 上的连续设计变量。应用包含 SVM 两个超参数及超新星模型的 $H_0,\Omega_M,\Omega_\Lambda$。
3. **$\mathcal Z$。** 连续 fidelity 向量 $z\in[0,1]^p$。SVM 用训练样本量 $N$ 与迭代次数 $T$；超新星用数据量 $N$ 与积分网格大小 $G$。
4. **结构。** 多维连续空间，通常只有分量意义上的偏序，不必存在单一“第 1、2、3 级”。实现时 $N,T,G$ 本为整数，却用归一化连续变量近似。
5. **$z^\star$。** 约定 $z^\star=\mathbf 1$，目标为 $f(x)=g(z^\star,x)$；它被假定最贵、最精确。
6. **输出/accuracy。** 标量带噪函数值；performance 用目标 fidelity 上的优化 regret。一个 fidelity 对目标的潜在偏差由核诱导的 information gap $\xi(z)$ 衡量。
7. **cost。** 给定已知 $\lambda(z)$；示例使用 $O(N^2T)$、$NT$ 或 $NG$ 型成本。理论主要写成只依赖 $z$，没有学习 $x$ 相关运行时间。
8. **关系模型。** 对联合函数 $g(z,x)$ 建 GP，核通常分解为 fidelity 核与 design 核的乘积；连续核使邻近 $z$ 共享信息。
9. **假设与学习。** 先验假设是在 design 和 fidelity 两个方向平滑，且 $z^\star$ 周围的协方差能刻画近似质量；核超参数以边际似然等方法估计。若 fidelity 变量导致相变或类别机制切换，连续核会失真。
10. **选择与限制。** 先按目标 fidelity 的 UCB 选 $x$，再根据信息差、后验不确定度和成本阈值选 $z$。这是用户所设多维连续 $x,z,y,c$ 框架的典型原生例子，但 cost 没有随 $x$ 学习，且多维 $z$ 不是全序。

### 11. Klein et al. (2017), *Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets*（FABOLAS）

**证据：A。** [PMLR 全文](https://proceedings.mlr.press/v54/klein17a/klein17a.pdf)；本地全文： [11_klein_et_al_2017_fabolas.pdf](papers/11_klein_et_al_2017_fabolas.pdf)。

1. **研究对象与任务。** 大数据集上的超参数优化，用小数据子集快速推断全数据集上的最优配置。
2. **$\mathcal X$。** 混合 HPO 空间。例如 SVM 的 $C$ 与核宽度，CNN 的学习率、batch size、层宽/单元数，以及 ResNet 的学习率、L2、momentum 和学习率下降设置。
3. **$\mathcal Z$。** 数据子集相对大小 $s=N_{sub}/N$。
4. **结构。** 一维、连续、有序，$s\in(0,1]$；实际运行只能取有限个样本数，但 GP 与 acquisition 把它连续处理。
5. **$z^\star$。** $s=1$，即完整训练集。
6. **输出/accuracy。** 验证 loss $f(x,s)$；目标是完整数据集上的最小 loss。这里“accuracy”既可指分类准确率，也可由 loss 表示，算法直接建模的是 loss。
7. **cost。** 单独对 log runtime 建 GP，得到随超参数和子集大小变化的 $c(x,s)$；这正面处理了相同数据量下不同配置运行时间不同的问题。
8. **关系模型。** loss 用 design 核与固定 fidelity 基函数 $\phi_f(s)=(1,(1-s)^2)$ 的组合 GP；log-cost 用 $\phi_c(s)=(1,s)$ 的组合 GP。
9. **假设与学习。** 基函数编码“接近全数据时 loss 趋于全数据目标、成本随样本增大”的归纳偏置；系数、核参数由观测学习。若学习曲线非平滑、子集改变类别分布或训练失败，偏置可能不成立。
10. **选择与限制。** acquisition 是单位预测成本可获得的、关于全数据最优点的熵减少，并加上 BO 开销；直接联合选 $(x,s)$。在 26 篇中，它是最完整学习 $y(x,z)$ 与 $c(x,z)$ 的代表之一。

### 12. Li et al. (2018), *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*

**证据：A。** [JMLR 全文](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf)；本地全文： [12_li_et_al_2018_hyperband.pdf](papers/12_li_et_al_2018_hyperband.pdf)。常被引用为 2017 会议工作，正式 JMLR 卷期为 2018。

1. **研究对象与任务。** 在 HPO 中自动平衡“试很多配置但每个只给少量资源”和“试少量配置但训练充分”。
2. **$\mathcal X$。** 任意超参数配置/arm，可含连续、整数、类别、条件和结构变量；Hyperband 本身不规定具体编码。
3. **$\mathcal Z$。** 单一资源预算 $r$，可代表 epoch、优化迭代、样本数或特征数。
4. **结构。** 概念上一维有序；算法只访问按 $\eta$ 几何增长的有限离散预算，如 $R\eta^{-s}$，所以实际是离散层级。
5. **$z^\star$。** 最大允许资源 $R$，不是保证得到真实泛化性能的绝对 ground truth。
6. **输出/accuracy。** 某配置在资源 $r$ 下的验证 loss/accuracy；算法比较排名，不拟合“距全预算答案的误差”。最终 performance 是给定预算下找到的配置质量。
7. **cost。** 以资源单位计，基本假定评价成本与 $r$ 成正比；同一 $r$ 下不同配置的运行时间差异通常被忽略。
8. **关系模型。** 没有 GP、概率分布或显式 $g(x,r)$。Successive Halving 在每轮保留表现最好的 $1/\eta$ 配置并给它们更多资源；Hyperband 枚举不同初始数量/预算的 brackets。
9. **假设与学习。** 不要求性能随 $r$ 单调，也不学习跨 fidelity 映射；隐含假设是早期表现有足够排序信息，且资源是可比较成本。错误的早期排序会淘汰后期优胜者。
10. **选择与限制。** $(x,r)$ 由 bracket 日程与淘汰规则决定，不由响应面 acquisition 决定。它只部分符合统一框架：可以定义 $y(x,r)$，但方法刻意不建模其函数关系和 cost 曲面。

### 13. Bischl et al. (2023), *Hyperparameter Optimization: Foundations, Algorithms, Best Practices, and Open Challenges*

**证据：A。** [开放全文](https://epub.ub.uni-muenchen.de/108819/1/WIREs_Data_Min___Knowl_-_2023_-_Bischl_-_Hyperparameter_optimization__Foundations__algorithms__best_practices__and_open.pdf)；本地全文： [13_bischl_et_al_2023_hpo.pdf](papers/13_bischl_et_al_2023_hpo.pdf)。

1. **研究对象与任务。** HPO 总体综述，包含问题定义、随机/网格搜索、BO、多保真调度、评估协议和开放问题；不是一项单独 MF 算法。
2. **$\mathcal X$。** 连续、整数、类别、条件和层级超参数，可包含算法选择，因而常是混合且非矩形的搜索空间。
3. **$\mathcal Z$。** 章节中讨论 epoch、优化迭代、数据子集、交叉验证折数、特征数等资源型 fidelity。
4. **结构。** 视算法而定，常是单一有序资源，但可多维、离散或连续；不同资源定义不能自动互换。
5. **$z^\star$。** 通常是完整训练资源和正式重采样评估；泛化到未知数据仍不是确定 ground truth。
6. **输出/accuracy。** 验证/重采样风险、多指标性能和不确定度；综述强调避免 test-set 泄漏及正确估计泛化误差。
7. **cost。** 墙钟时间、CPU/GPU、FLOPs、内存、能耗或统一资源单位；没有单一通用成本模型。
8. **关系模型。** 总结 SHA、Hyperband、BOHB、学习曲线外推和多任务/多保真 BO 等，各自使用调度、密度估计、GP 或曲线模型。
9. **假设与学习。** 各方法不同：有的只依赖早期排序，有的拟合 $p(y\mid x,z)$，有的假定可扩展学习曲线；文章不宣称一种关系在所有学习器上成立。
10. **选择与限制。** 它提供方法选型与实践检查表，而不是统一 acquisition。用户框架可用作综述的二次结构化，但不能把文中所有 HPO 方法都称作同一个 multi-fidelity 模型。

### 14. Penwarden et al. (2022), *Multifidelity Modeling for Physics-Informed Neural Networks (PINNs)*

**证据：A。** [作者版全文](https://users.cs.utah.edu/~kirby/Publications/Kirby-153.pdf)；本地全文： [14_penwarden_et_al_2022_pinn.pdf](papers/14_penwarden_et_al_2022_pinn.pdf)。

1. **研究对象与任务。** 对参数化 PDE 的 PINN 解做多保真代理；用许多小网络/较松优化得到的解与少量大网络/严格优化得到的解重构高保真解流形。
2. **$\mathcal X$。** 这里更准确叫物理参数空间 $p$：Burgers 方程改变黏性；非线性热方程使用 $\lambda,k$；Allen–Cahn 使用 $\lambda$；扩散–反应使用 $\lambda,k$。输出还以空间/时间坐标展开成解场，但选样是在参数 $p$ 上。
3. **$\mathcal Z$。** PINN 深度、每层宽度及 L-BFGS 最大迭代/停止容差的组合。激活函数、collocation 策略及前置 Adam 500 epochs 固定。
4. **结构。** 原始 fidelity 因素是多维混合变量，但论文把预定组合封装成 LF/HF 两个离散、有序标签，并没有在连续架构空间里优化 $z$。
5. **$z^\star$。** 指定的高保真 PINN 配置。例中低/高网络包括 2×5 对 5×10、5×5 对 8×10、5×10 对 8×20 等；高保真 L-BFGS 上限更大、容差更严。它仍不是解析 PDE 真解。
6. **输出/accuracy。** 每个物理参数下的高维时空/空间解快照；accuracy 用相对场误差，并与直接高保真 PINN 和低保真结果比较。
7. **cost。** 实测 PINN 训练秒数；网络容量和更严格优化导致平均成本差异。没有 $c(p,z)$ 模型，也不将成本放入 acquisition。
8. **关系模型。** 先在约 50 个参数点生成 LF 快照，用 Gram 矩阵的 pivoted Cholesky 选约 10 个代表点训练 HF；再用 LF 空间中的投影系数线性组合相应 HF 快照。这是低秩几何映射，不是 GP。
9. **假设与学习。** 假设 LF 与 HF 解流形具有相似的低秩结构，LF 选出的基点也能张成 HF 流形。论文明确承认增加宽度/深度并不理论保证单调提高 PINN 精度，随机优化会产生坏运行。
10. **选择与限制。** 选择的是哪些参数点升级为 HF，而不是任意 $(p,z)$；fidelity 组合由作者预先固定。统一框架可重解释，但“神经网络大小越大 fidelity 越高”不是普适定律，必须用验证误差与成本实证确认。

### 15. Bansal et al. (2022), *JAHS-Bench-201: A Foundation for Research on Joint Architecture and Hyperparameter Search*

**证据：A。** [NeurIPS Datasets and Benchmarks 全文](https://proceedings.neurips.cc/paper_files/paper/2022/file/fd78f2f65881c1c7ce47e26b040cf48f-Paper-Datasets_and_Benchmarks.pdf)；本地全文： [15_bansal_et_al_2022_jahs.pdf](papers/15_bansal_et_al_2022_jahs.pdf)。

1. **研究对象与任务。** 为联合 NAS 与 HPO 提供可查询的多保真 benchmark/surrogate；它本身不规定优化算法。
2. **$\mathcal X$。** 6 条 cell DAG 边各选 {skip、zero、$1\times1$ conv、$3\times3$ conv、$3\times3$ avg-pool}；再加 3 类激活函数、连续学习率 $[10^{-3},1]$、连续 weight decay $[10^{-5},10^{-2}]$、二元 trivial augmentation。按论文的角色划分，共 10 个 architecture/HPO 维度。
3. **$\mathcal Z$。** 4 个 fidelity：depth multiplier $N\in\{1,3,5\}$、width $W\in\{4,8,16\}$、图像分辨率倍率 $R\in\{0.25,0.5,1\}$、epoch $E\in\{1,\ldots,200\}$。
4. **结构。** 4 维有限离散乘积空间；每个分量大体有序，但组合之间只有偏序，例如更宽但更少 epoch 的两个点未必可比。
5. **$z^\star$。** 常用最大配置 $(5,16,1,200)$ 作为目标，但 benchmark 允许研究者自行定义目标与预算；不是所有查询都必须归结为一个唯一 $z^\star$。
6. **输出/accuracy。** 约 20 项指标，包括 train/validation/test loss 与 top-1 accuracy、每 epoch/累计 duration、FLOPs、latency、模型大小等。优化通常最大化验证准确率并可同时考虑延迟。
7. **cost。** duration、FLOPs、latency、模型大小均被记录/代理预测，所以成本显式依赖架构超参数与 fidelity，即 $c(x,z)$。
8. **关系模型。** 大规模真实训练数据上，为各项 metric 分别拟合 XGBoost surrogate；给定联合 $(x,z)$ 返回预测性能和成本。没有先验规定某个解析跨 fidelity 公式。
9. **假设与学习。** 主要假设是树模型能在采样分布内泛化；关系由数据学习。深度、宽度被作者归为 fidelity 是 benchmark 的角色选择，它们在别的 NAS 研究里也完全可以属于 architecture design。
10. **选择与限制。** benchmark 不负责选下一点，BO、Hyperband 等算法在其上决策。它非常适合用户框架，尤其说明 $z$ 可多维、离散、仅偏序，且 $y$ 与 $c$ 都依赖 $x,z$；限制是 surrogate 误差和固定数据/训练管线的域外有效性。

### 16. Trofimov et al. (2020/2021), *Multi-fidelity Neural Architecture Search with Knowledge Distillation*

**证据：A。** [arXiv 全文](https://arxiv.org/pdf/2006.08341)；本地全文： [16_trofimov_et_al_2020_nas_kd.pdf](papers/16_trofimov_et_al_2020_nas_kd.pdf)。

1. **研究对象与任务。** 在 NAS-Bench-201 架构空间中，用知识蒸馏增强极短训练的相关性，辅助搜索更好的网络结构。
2. **$\mathcal X$。** NAS-Bench-201 cell 的 6 条有向边，每条边从 5 种操作选择，以 one-hot/图结构编码；共有 15,625 个离散架构。
3. **$\mathcal Z$。** LF 为训练 $E_1=1$ epoch 且采用 NST 知识蒸馏、由 ResNet teacher 引导；HF 为训练 $E_2=12$ epochs 的常规分类训练。
4. **结构。** 二级离散并被当作有序，但两个层级不仅资源量不同，训练损失和 teacher 机制也不同，因此严格说是两个异构信息源。
5. **$z^\star$。** 算法中的目标函数是 12-epoch HF 验证结果；NAS-Bench-201 的完整约 200-epoch记录用于外部评价，不能与算法内部的 $z^\star$ 混为一谈。
6. **输出/accuracy。** 各架构的验证准确率；跨级质量还用 Kendall 排名相关等检验。最终报告候选架构在 benchmark 中的测试准确率。
7. **cost。** 实测训练时间和总预算；KD 的单 epoch 也有 teacher 特征计算，优化后约为普通 epoch 的 1.5 倍。没有学习 $c(x,z)$。
8. **关系模型。** co-kriging/自回归 GP：$y_H(x)=\rho y_L(x)+\delta(x)$，以边操作编码为输入，参数通过似然拟合。
9. **假设与学习。** 假定 KD 后的 1-epoch分数与 12-epoch分数有稳定、可由线性缩放加平滑差异表示的关系；相关强度从配对架构学习。不同架构对 teacher 的受益差异可能破坏该关系。
10. **选择与限制。** warm-up 约含 100 个 LF、20 个 HF；随后按 HF-UCB 选择架构并主要查询 HF，并非每轮自适应选择 fidelity。故它可纳入统一框架，但“fidelity”混合了训练资源和训练机制，且后期不是完整联合 $(x,z)$ BO。

### 17. Li et al. (2020), *Multi-Fidelity Bayesian Optimization via Deep Neural Networks*

**证据：A。** [NeurIPS 全文](https://proceedings.neurips.cc/paper/2020/file/60e1deb043af37db5ea4ce9ae8d2c9ea-Paper.pdf)；本地全文： [17_li_et_al_2020_dnn_mfbo.pdf](papers/17_li_et_al_2020_dnn_mfbo.pdf)。

1. **研究对象与任务。** 用深度非线性跨 fidelity 模型替代线性 co-kriging，并用信息论 acquisition 优化最高 fidelity 黑箱。
2. **$\mathcal X$。** 理论为连续黑箱输入。振动板例用杨氏模量、泊松比、密度；热传导例用椭圆热源/区域的两个半轴半径和旋转角。
3. **$\mathcal Z$。** $M$ 个离散 fidelity。工程例中由粗/细有限元网格区分；合成实验常用 2 或 3 级。
4. **结构。** 一维标签、有限、离散、严格有序；网络按相邻层级递推。
5. **$z^\star$。** 第 $M$ 级，即最细网格或最高精度仿真。
6. **输出/accuracy。** 标量函数；振动板为第 4 振型频率，热传导为达到 $70^\circ$C 的时间。优化 performance 是目标 fidelity 的最优值/简单 regret。
7. **cost。** 每级已知固定 $\lambda_m$，合成设置常用 $(1,10,100)$ 或 $(1,10)$；不允许成本随 $x$ 变化。
8. **关系模型。** 对 $m>1$，输入连接 $[x,f_{m-1}(x)]$，经神经特征 $\phi_{\theta_m}$ 后线性输出；层层组成非线性自回归网络。输出层权重作贝叶斯处理，其他参数作为点估计/超参数，用随机变分推断训练。
9. **假设与学习。** 假定存在按有序相邻层传递的低维非线性关系；关系形状从配对/非配对数据学习。深网比线性 co-kriging 灵活，但校准受近似后验、数据量和网络过拟合影响。
10. **选择与限制。** acquisition 最大化候选观测与目标最大值之间的 mutual information，再除以 $\lambda_m$，联合选 $x,m$。框架原生匹配，但 fidelity 必须预先全序、成本为级别常数，不能直接处理 MISO 式无序信息源。

### 18. Li, Kirby & Zhe (2021), *Batch Multi-Fidelity Bayesian Optimization with Deep Auto-Regressive Networks*

**证据：A。** [NeurIPS 全文](https://proceedings.neurips.cc/paper/2021/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf)；本地全文： [18_li_et_al_2021_bmbo_darn.pdf](papers/18_li_et_al_2021_bmbo_darn.pdf)。

1. **研究对象与任务。** 把深度自回归 multi-fidelity BO 扩展到 batch/并行评价，重点处理一批点之间的信息冗余。
2. **$\mathcal X$。** 混合 HPO 空间：CNN 的层数、通道、全连接深度/宽度、pool/dropout；LDA 参数；XGBoost 参数；PINN 的网络/优化相关参数。
3. **$\mathcal Z$。** 训练或求解资源级。例如 XGBoost 树数 2/10/100；PINN L-BFGS 迭代 10/100/50,000；其他实验也通常为三级资源。
4. **结构。** 有限、离散、全序；各层按相邻或累计的自回归顺序连接。
5. **$z^\star$。** 最大训练/求解预算的第 $M$ 级。
6. **输出/accuracy。** 标量验证性能/损失；目标是在最高级找到最优超参数。batch performance 用累计成本与最高 fidelity 优化结果评价。
7. **cost。** 每级给定平均成本 $\lambda_m$，常设相对比例如 1:10:50；不单独学习架构依赖的运行时间。
8. **关系模型。** 深度贝叶斯自回归网络，第 $m$ 级以 $[x,f_1(x),\ldots,f_{m-1}(x)]$ 为输入，权重用 HMC 采样；比仅接前一级的链式关系更丰富。
9. **假设与学习。** 假定 fidelity 可全序，且高层可由 $x$ 与所有低层输出的非线性函数表示；网络权重后验由数据学习。HMC 提高不确定性表达，但计算昂贵且仍受模型结构假设影响。
10. **选择与限制。** batch acquisition 用一批候选观测对最高 fidelity 最大值的 mutual information 除以总成本，并显式惩罚批内冗余，交替优化连续 $x$ 和离散 $m$。原生匹配统一框架，但 $c$ 被简化为 fidelity 常数。

### 19. Li et al. (2022), *Deep Multi-Fidelity Active Learning of High-Dimensional Outputs*（DMFAL）

**证据：A。** [PMLR 全文](https://proceedings.mlr.press/v151/li22b/li22b.pdf)；本地全文： [19_li_et_al_2022_dmfal.pdf](papers/19_li_et_al_2022_dmfal.pdf)。

1. **研究对象与任务。** 对 PDE、拓扑优化和 CFD 的高维输出场做主动多保真学习；目标是全域高保真 surrogate accuracy，不是最大化一个标量目标。
2. **$\mathcal X$。** Burgers 例为黏性；Poisson 为 4 个边界常数加 $\beta$；热方程为左右热流加扩散率 $\alpha$；拓扑优化为载荷位置与角度；CFD 为 4 个边界切向速度加 Reynolds 数。
3. **$\mathcal Z$。** 网格分辨率级。PDE 常用 $16^2/32^2/64^2$，拓扑/CFD 等用 $50^2/75^2$ 等两个或多个网格。
4. **结构。** 2–3 个离散、有序层级；不同级的输出维数 $d_m$ 也可能不同，通常随网格加密而增大。
5. **$z^\star$。** 训练体系内最细网格；某些测试另用 $128^2$ 数值解作更高精度参考，因此要区分“模型目标级”和“外部验证真值”。
6. **输出/accuracy。** $y_m(x)\in\mathbb R^{d_m}$ 的高维解场、密度布局或时间序列速度场；用目标级 normalized RMSE 等衡量。
7. **cost。** 各级给定固定的平均归一化运行成本，如 $(1,3,10)$；不拟合随 $x$ 变化的求解难度。
8. **关系模型。** 每级神经网络把输入映射到低维潜变量 $h_m$，再以基矩阵 $A_mh_m$ 生成高维输出；高一级网络接收 $x$ 与前一级潜变量。输出层权重作变分贝叶斯推断。
9. **假设与学习。** 假定各级高维场存在低秩潜表示，并能按有序网格在潜空间自回归；潜映射与基从数据学习。网格间拓扑变化、激波/不连续和极少 HF 数据会削弱假设。
10. **选择与限制。** acquisition 以单位成本下候选高维观测对最高 fidelity 输出的 mutual information 选择 $(x,m)$；这是 active learning，不是寻找 $\arg\max_x$。统一框架原生适用，但 $y$ 必须允许为向量/场，不能把 performance 只理解为一个标量 accuracy。

### 20. Li et al. (2022), *Infinite-Fidelity Coregionalization for Physical Simulation*（IFC）

**证据：A。** [arXiv 全文](https://arxiv.org/pdf/2207.00678)；本地全文： [20_li_et_al_2022_ifc.pdf](papers/20_li_et_al_2022_ifc.pdf)。

1. **研究对象与任务。** 学习可在连续、甚至训练范围外 fidelity 上生成高维物理解的 surrogate，解决“只在少数预设网格级预测”的限制。
2. **$\mathcal X$。** PDE 参数、边界条件或几何/载荷变量；实验沿用 Poisson、热、Burgers、拓扑优化与 CFD 类输入。
3. **$\mathcal Z$。** 用连续标量 $m$ 编码网格分辨率/离散化级别；例如 $m=0,1,2.14$ 可对应 $8^2,64^2,128^2$ 等网格。
4. **结构。** 一维、连续、有序，并允许在已观察 fidelity 之间插值及向更高 fidelity 外推；物理网格本身离散，连续性是模型化编码。
5. **$z^\star$。** 方法没有必然唯一的固定目标；实验可把最细训练网格或额外 $128^2$ 数值解当参考，并测试未见过的更高 fidelity。
6. **输出/accuracy。** 随网格维度变化的高维解场；用 normalized RMSE 及插值/外推误差评价。
7. **cost。** 动机上高分辨率更贵，但方法没有建立 $c(x,m)$，也没有成本感知 acquisition；训练该复杂模型本身可能更慢。
8. **关系模型。** 潜态沿 fidelity 方向满足神经 ODE：$\partial h/\partial m=\phi(m,h,x)$，$h(0,x)=\beta(x)$；输出 $y=B(m)h(m,x)+\epsilon$。$B(m)$ 可由 GP 建模（IFC-GPODE）或另一 ODE 建模（IFC-ODE2）。
9. **假设与学习。** 核心先验是潜表示随连续 fidelity 平滑演化，可由 ODE 流描述；网络、基和概率参数以变分方法从多网格数据学习。若不同网格改变解的拓扑或数值格式，连续 ODE 假设未必合理。
10. **选择与限制。** 本文只学习关系，不主动选择下一 $(x,m)$，并把 active learning 留作后续工作。它对 $y(x,z)$ 的连续多保真框架高度匹配，但用户提出的 cost 部分在文中缺失。

### 21. Meng & Karniadakis (2020), *A Composite Neural Network that Learns from Multi-Fidelity Data*

**证据：A。** [arXiv 全文](https://arxiv.org/pdf/1903.00104)；本地全文： [21_meng_karniadakis_2020.pdf](papers/21_meng_karniadakis_2020.pdf)。

1. **研究对象与任务。** 从大量 LF 数据和少量 HF 数据学习 HF 函数；还扩展为 physics-informed 版本，用 PDE 约束帮助稀疏 HF 数据融合。
2. **$\mathcal X$。** 可为普通函数输入、空间–时间坐标和物理参数。示例从合成多保真函数到非饱和流、反应输运等 PDE 正/反问题。
3. **$\mathcal Z$。** 两个数据来源标签 LF/HF。fidelity 不是神经网络本身的大小；网络是学习两源关系的工具。
4. **结构。** 二级、离散、被假定有序；标签没有连续距离或中间级。
5. **$z^\star$。** HF 数据/高保真解；在 physics-informed 例中 PDE 与边界/初值还提供额外约束，但 HF 仍可能含噪或模型误差。
6. **输出/accuracy。** 标量函数或物理解场，以及反问题中的物理参数（如水力参数、反应率/反应阶）；用相对 $L_2$ 等预测误差评价。
7. **cost。** 只作定性设定：LF 多且便宜，HF 少且贵；没有数值 $c(x,z)$、预算优化或运行时间模型。
8. **关系模型。** 先用网络拟合 $y_L(x)$；HF 由一个线性分支和一个非线性分支共同学习 $x,y_L(x)\mapsto y_H(x)$，再用可训练权重 $\alpha$ 混合。physics-informed 版本把 PDE 残差加入损失。
9. **假设与学习。** 假设 LF 对 HF 有可学习的线性或非线性信息；$alpha$、网络权重及物理参数从数据和物理残差共同训练。关系很灵活，但 deterministic 训练的预测不确定性与 cost–benefit 校准有限。
10. **选择与限制。** 训练数据位置通常预先给定，不用 acquisition 选择 $(x,z)$。它是数据融合型 multi-fidelity：$y(x,z)$ 框架适用，$c$ 和主动决策部分不适用。

### 22. Poloczek, Wang & Frazier (2017), *Multi-Information Source Optimization*

**证据：A。** [NeurIPS 全文](https://papers.neurips.cc/paper_files/paper/2017/file/df1f1d20ee86704251795841e6a9405a-Paper.pdf)；本地全文： [22_poloczek_wang_frazier_2017.pdf](papers/22_poloczek_wang_frazier_2017.pdf)。

1. **研究对象与任务。** 利用多个有偏、带噪、成本不同的信息源优化一个真实目标；它有意放宽传统“一级比一级精确”的 fidelity 概念。
2. **$\mathcal X$。** 紧致连续设计域 $D\subset\mathbb R^d$。实验有 Rosenbrock、MNIST HPO 的 4 个超参数，以及 assemble-to-order 库存系统的 8 个目标库存量。
3. **$\mathcal Z$。** 信息源索引 $\ell\in\{0,\ldots,M\}$；$ell=0$ 为 truth，其余可能是不同数据集、近似器或仿真模型，而非同一旋钮的多个档位。
4. **结构。** 离散类别、无需有序。USPS 辅助 MNIST 的例子尤其说明“便宜来源”与目标来源不同，不应硬解释为连续低分辨率。
5. **$z^\star$。** source 0 的真实目标 $g(x)$。
6. **输出/accuracy。** $Y(\ell,x)\sim\mathcal N(f(\ell,x),\lambda_\ell(x))$ 的带噪标量；每个辅助源相对 truth 的偏差由 discrepancy 表示。performance 是在 truth 上的最优解质量。
7. **cost。** 允许已知、连续且依赖位置的 $c_\ell(x)$；噪声方差 $\lambda_\ell(x)$ 也可依赖 $x$。这是少数在数学形式上明确容纳同一 source 对不同 design 成本不同的论文。
8. **关系模型。** truth 建 GP；每个源的 discrepancy $\delta_\ell(x)=f(\ell,x)-g(x)$ 建独立 GP，共同得到跨 source 协方差。超参数由历史或观测数据的边际似然估计。
9. **假设与学习。** 假定源偏差是可平滑建模且源间在给定 truth 后按所设结构独立；偏差大小和相关长度由数据学。它不要求误差或成本按 source index 单调，通用性强于 ordinal MF，但依赖正确的 discrepancy kernel。
10. **选择与限制。** misoKG 最大化一次 $(\ell,x)$ 观测对 truth 后验最优值的期望提升，再除以 $c_\ell(x)$；最终推荐 truth 后验最优设计。它是用户一般化 $x,z,y,c$ 框架的强匹配例，但 $z$ 应称 categorical information source，不能称连续/有序 fidelity。

### 23. Takeno et al. (2020), *Multi-fidelity Bayesian Optimization with Max-value Entropy Search and its Parallelization*

**证据：A。** [PMLR 全文](https://proceedings.mlr.press/v119/takeno20a/takeno20a.pdf)；本地全文： [23_takeno_et_al_2020.pdf](papers/23_takeno_et_al_2020.pdf)。

1. **研究对象与任务。** 用 max-value entropy search 做成本敏感 multi-fidelity BO，并扩展为有 pending evaluations 的并行/异步情形。
2. **$\mathcal X$。** 连续黑箱设计域。材料案例用两个材料参数，目标使模拟析出物形状接近显微图像。
3. **$\mathcal Z$。** $m\in[M]$ 的有限精度级，材料例有 3 个 accuracy levels。
4. **结构。** **离散、有限、有序**。原清单中“也可以有 continuous fidelity features”并不是这篇论文本身的设定；该文并行部分也明确处理 discrete-fidelity MFBO。
5. **$z^\star$。** 第 $M$ 级目标函数 $f^{(M)}$。
6. **输出/accuracy。** 带噪标量黑箱；材料例为模拟形状与显微图像之间的 discrepancy。性能以目标级最优值/简单 regret 评价。
7. **cost。** 每级给定固定 $\lambda_m$，材料例相对成本为 5、10、60；不依赖 $x$，也不从数据学习。
8. **关系模型。** acquisition 只要求联合输出服从多输出高斯模型；实验采用线性模型/共区域化（SLFM）型 kernel 与 ARD 空间核来共享 fidelity 信息。
9. **假设与学习。** 高斯联合模型和核结构是先验，跨级相关、长度尺度等由数据估计。它比固定线性缩放灵活，但仍依赖 GP 校准和离散预定义层级。
10. **选择与限制。** MF-MES 最大化候选 $f^{(m)}(x)$ 与目标最大值 $f_\star$ 的 mutual information，再除以 $\lambda_m$；并行版本对未完成查询条件化。框架原生匹配，但不覆盖连续 $z$ 或 $x$ 相关成本。

### 24. Chen et al. (2022/2023), *Multi-Fidelity Simulation Modeling for Discrete Event Simulation: An Optimization Perspective*

**证据：A。** [IEEE 页面](https://ieeexplore.ieee.org/document/9775096/)；[可检索作者全文](https://www.researchgate.net/publication/360649380_Multi-Fidelity_Simulation_Modeling_for_Discrete_Event_Simulation_An_Optimization_Perspective)。

1. **研究对象与任务。** 不是接受预先给好的 fidelity，而是自动构造离散事件仿真（DES）的低保真模型；把仿真速度与相对已验证高保真模型的失真做成双目标优化。
2. **$\mathcal X$。** 必须分两层。构造模型时，论文把**模型配置**记作决策向量：若干 operating unit 的 coarse/detailed 二元开关、连续 warm-up time 和离散 simulation budget。急诊室案例中为 8 个配置因素：仿真长度、warm-up 比例及 6 个 operating-unit 精细/粗化开关。下游真正的系统设计另有 12 个整数决策，表示预算约束下各类医疗人员数量。
3. **$\mathcal Z$。** 在用户统一语言下，上述“模型配置向量”应被重新命名为 $z$：模块精度开关、warm-up 和仿真长度/预算共同定义一个仿真模型。
4. **结构。** 多维混合空间：连续、离散整数和二元变量并存。多个配置通常不可形成天然全序；论文事后才按 speed–distortion 把选中模型称为 high/medium/low。
5. **$z^\star$。** 已验证的详细 DES 模型配置 $h$，作为参考而非现实系统的绝对真值。
6. **输出/accuracy。** 双目标：$f_1$ 是随机墙钟仿真时间的期望；$f_2$ 是总 distortion。设计失真用配置与 $h$ 的距离；第 $k$ 个输出的测量失真为 $(E[G_k]-g_k^0)^2+Var(G_k)=E[(G_k-g_k^0)^2]$，再跨输出加权。急诊室关注患者 length of stay。
7. **cost。** 直接测量实际 computer-clock time 并用重复仿真估计期望，因此依赖整个模型配置；未建立解析或监督学习式 $c(x_{system},z)$。
8. **关系模型。** 通过重复仿真样本均值估计时间与失真，再用 MOSO-HV 多目标随机优化寻找 Pareto 配置；并不是先拟合 GP 型 fidelity response surface。选出的模型随后可放入 multi-model optimization。
9. **假设与学习。** 假定参考模型及参考输出可信，重复仿真满足期望/方差估计条件，且配置距离有意义；时间与输出失真主要由 simulation observations 估计。设计距离的权重是人为定义，未必等价于行为误差。
10. **选择与限制。** 演化多目标搜索提出配置，仿真预算分配估计 Pareto 前沿。本文最重要的概念教训是必须把系统 design $x$ 与 simulator configuration $z$ 分开；原文两个优化层都叫 design，会导致符号混淆。它不要求 fidelity 一维或有序。

### 25. Li et al. (2026), *Enhancing Electric Vehicle Charging Station Design Using Multifidelity Simulations*

**证据：A。** [期刊 DOI 页面](https://doi.org/10.1177/10591478261468125)；[可检索接受稿全文](https://journals.sagepub.com/history/3a5d88ae-75aa-48ea-998a-cf12001d3d21/10591478261468125.38916447.pdf)。

1. **研究对象与任务。** 在随机到达、充电、光伏和电池运行条件下，从有限个 EV 充电站方案中选出期望经济表现最好的设计；用多保真仿真和固定预算 allocation 提高正确选择概率。
2. **$\mathcal X$。** 站点设计含充电枪数量 $n_c$（整数）、PV 面积 $A_{pv}$、电池容量 $E_b$、电池充放电功率 $P_b$、逆变器容量 $P_{inv}$（连续/工程离散变量）。算法在候选设计集合上做 ranking-and-selection。
3. **$\mathcal Z$。** $z=(T_{days},\Delta t,K_{arr})$：仿真天数、时间分辨率、车辆到达率的分段数。
4. **结构。** 原始上是 3 维混合 fidelity 配置；论文从中预定义有限个嵌套层级并给出高低顺序。单个分量通常越精细越贵，但不同分量一升一降时只形成偏序，论文的总序依赖所选配置表。
5. **$z^\star$。** 最长 horizon、最细时间步、最精细到达过程的最高 fidelity 仿真，被当作最终站点性能参考。
6. **输出/accuracy。** 随机的总日成本/收益指标：摊销投资与运维成本、SLA 违约惩罚以及运营利润等组合。跨 fidelity accuracy 不是点对点确定误差，而是均值、方差及相邻层输出差的统计性质。
7. **cost。** 每次仿真的实测成本 $c_{i,q}$ 可依赖设计 $i$ 和 fidelity $q$；推导可容纳这种依赖，实验中常用同一级的平均时间并要求低级更便宜。这比仅设 $\lambda_q$ 更接近真实 $c(x,z)$。
8. **关系模型。** 利用层级间差值构造 telescoping 的多级无偏估计量，在线估计各设计、各 fidelity 的均值、方差和相邻差异；不是 GP，也不假定一个确定性函数拟合所有级。
9. **假设与学习。** 假定嵌套层级可耦合、差值方差随 fidelity 提高而适合分配仿真复制，且样本统计足以近似 PCS；均值/方差/成本由仿真数据更新。若低高层排序或耦合很差，节省会消失。
10. **选择与限制。** AOMS 在固定总预算下，针对最大化 probability of correct selection 自适应选择下一设计–fidelity 复制。它原生匹配 $x,z,y,c$，但任务是有限候选的统计选择，不是连续 surrogate optimization；$z$ 的有序性来自人为挑选的配置链。

### 26. Mitchell et al. (2026), *Multi-Fidelity Benders Decomposition for Generation, Storage, and Transmission Expansion Planning*

**证据：A。** [期刊 DOI](https://doi.org/10.1016/j.segan.2026.102487)；[作者全文](https://optimization-online.org/wp-content/uploads/2026/04/Multi_Fidelity_Benders_Decomposition_Paper.pdf)；本地全文： [26_mitchell_et_al_2026_benders.pdf](papers/26_mitchell_et_al_2026_benders.pdf)。

1. **研究对象与任务。** 发电、储能和输电扩建规划（GSTEP）的混合整数线性优化；利用多个时间尺度的 Benders 子问题加速收敛，同时保留最终最优性证明。
2. **$\mathcal X$。** master 中每个候选资产的二元投资变量 $x_i$，覆盖输电线路、常规/可再生发电与储能。运营子问题还含发电、潮流、相角、储能充放电与 state of charge、未满足负荷等连续变量，但它们是 recourse，不是外层 design。
3. **$\mathcal Z$。** 子问题时间聚合/跨度与储能边界耦合强度，组成 Daily、Weekly、Monthly、Yearly 四档。低级子问题共同覆盖全年，但放松较长时间段之间的 storage state-of-charge 边界连接。
4. **结构。** 四级离散、按子问题跨度和紧度有序；不是“截短同一时间序列后预测全年输出”，而是不同强度的优化松弛。
5. **$z^\star$。** Yearly 完整时间耦合子问题，恢复原 GSTEP recourse；算法最终必须调用它才能证明对原问题最优。
6. **输出/accuracy。** 外层目标是投资成本加运营成本；子问题输出是 recourse cost、对偶信息、下界和 Benders cuts。所谓 accuracy 是 bound tightness 与 master optimality gap，而非相对 ground truth 的普通预测误差。
7. **cost。** 子问题实际求解时间/总体 wall time；低级有多个较小独立问题、可快速并行，完整 yearly 更贵。算法不学习 $c(x,z)$，实际求解成本虽会受 incumbent $x$ 影响，但决策规则没有显式利用该曲面。
8. **关系模型。** 关系来自数学规划松弛：低 fidelity 删除/放松 storage 边界耦合，产生对原问题有效但较松的下界与 cuts；不存在 GP、神经网络或统计误差回归。
9. **假设与学习。** 有效性由线性规划对偶和松弛关系解析保证，而不是从数据学习；关键条件是低 fidelity cuts 对高 fidelity 原问题仍有效。只有何时切换 fidelity 的策略依赖观察到的 Benders 收敛行为。
10. **选择与限制。** 自适应 MFBD 在当前 fidelity 的 bound improvement 变慢/停滞时提升到更长时间尺度，并保留已有 cuts；最终完整 fidelity 保证最优性。可写成 $g(x,z)$，但会掩盖本质：LF 给的是有方向保证的优化下界，不是带未知误差的同一性能测量。

## 五、从 26 篇反推出来的统一分类

### 5.1 fidelity 变量的六种不同语义

1. **模型/信息源标签：** Kennedy–O’Hagan、Forrester 2007、Meng–Karniadakis、MISO。这里一个 fidelity 往往对应一个模型；MISO 中标签甚至无序。
2. **数值离散分辨率：** DNN-MFBO、DMFAL、IFC。网格通常有序，但 IFC 才真正把它连续化。
3. **求解进度：** Forrester 2006、Picheny、Branke、Courrier。它们共享可暂停/继续的迭代轨迹，增量成本很重要。
4. **机器学习资源：** FABOLAS、Hyperband、JAHS、MF-KD、BMBO-DARN。数据量、epoch、树数和网络规模并不具有统一的误差单调性。
5. **仿真设置向量：** DES 与 EV station。多个设置共同决定速度和偏差，天然可能只有偏序或完全不可比。
6. **优化松弛强度：** MF Benders。低 fidelity 是松弛、下界和有效割，具有数学方向保证，不能用普通回归误差概括。

### 5.2 跨 fidelity 关系的七类建模方式

| 关系类型 | 代表论文 | 哪部分是先验 | 哪部分由数据学习 |
|---|---|---|---|
| 线性自回归 GP / co-kriging | 1, 2, 16 | $y_H=\rho y_L+\delta$、GP 平滑性 | $\rho$、核与 discrepancy |
| 非平稳收敛过程 GP | 6 | 误差随时间衰减、时空核结构 | 核参数、轨迹形状 |
| 误差界而非相关模型 | 9 | 已知 uniform bias bound | 各级 GP；实践可估界 |
| 联合连续 $z$ 的 GP | 10, 11 | kernel/basis 与连续平滑性 | 核参数、loss/cost 曲面 |
| 深度自回归/潜变量 | 17–20 | 有序链、低维潜结构或 ODE 连续性 | 神经映射、基和后验 |
| 不拟合函数关系 | 7, 12, 24–26 | 排序/调度/统计估计/松弛逻辑 | 翻转率、均值方差或切换状态 |
| 无序 source discrepancy | 22 | truth + source-specific discrepancy GP | 每个源的 bias/noise 结构 |

### 5.3 cost 建模强弱

- **显式学习 $c(x,z)$：** FABOLAS；JAHS-Bench-201 的运行时间 surrogate。
- **形式上允许或直接测量 $x$ 相关成本：** MISO 的已知 $c_z(x)$；EV station 的 $c_{i,q}$；DES 的模型配置墙钟时间。
- **只给每级常数：** MF-GP-UCB、DNN-MFBO、BMBO-DARN、DMFAL、MF-MES，以及许多 co-kriging 工作。
- **资源本身充当成本：** 部分收敛 CFD、Picheny、Hyperband。
- **只有“HF 贵、LF 便宜”的定性陈述：** Composite NN、IFC。

这个分类直接回答了用户提出的疑问：**同一个 fidelity 对不同 $x$ 的 cost 和 performance 完全可能不同，但多数理论为了可分析性把成本简化成 $c(z)$ 或常数 $\lambda_z$。真正显式承认并处理 $c(x,z)$ 的论文是少数。**

## 六、建议以后固定采用的数学框架

为了同时容纳这批论文，建议不要预设 fidelity 是一个有序标量，而写成：

$$
x\in\mathcal X,\qquad z\in\mathcal Z,\qquad
Y_{x,z}\sim P_{x,z},\qquad C_{x,z}\sim Q_{x,z}.
$$

- $\mathcal X$ 可以是连续、整数、类别、条件、图结构或有限候选集。
- $\mathcal Z$ 应附带类型：连续/离散、维数、全序/偏序/无序，以及是否可续算。
- $Y_{x,z}$ 可以是标量、向量、场、排序信息、统计估计量或优化下界；不能一律叫 accuracy。
- $C_{x,z}$ 最好看成随机变量而非常数；墙钟时间尤其会受 $x$、机器、并发和求解失败影响。
- 参考对象可写为 $T(x)$。只有当存在唯一 $z^\star$ 且 $T(x)=E[Y_{x,z^\star}]$ 时，才把它简化成最高 fidelity 输出。
- fidelity 质量应单独定义成任务相关的损失 $D(P_{x,z},T(x))$，不要默认“$z$ 越大误差一定越小”。

下一次选择可以统一写成

$$
(x_{n+1},z_{n+1})
=\arg\max_{x,z}
\frac{\text{预计对最终任务的价值}(x,z)}{\text{预计成本}(x,z)},
$$

但“价值”在不同论文中分别可能是 EI、UCB、mutual information、knowledge gradient、PCS 增益、淘汰可靠性、bound improvement，不能混成一个 accuracy 指标。

## 七、最重要的判别清单

读下一篇 multi-fidelity 论文时，建议先填下列字段，再判断方法是否可比：

1. $x$ 到底是工程设计、模型超参数、物理条件，还是作者把 simulator configuration 也叫成了 design？
2. $z$ 改变的是同一模型的分辨率/资源，还是换了信息源、损失函数或物理假设？
3. $\mathcal Z$ 是几维；连续、离散还是混合；全序、偏序还是无序？
4. 是否存在唯一 $z^\star$；它是计算参考、实验真值还是仅为论文内部目标级？
5. 不同 $z$ 的输出语义、维数和噪声是否相同？
6. accuracy 是数值误差、预测校准、排序一致性、统计方差、优化 regret、PCS，还是 bound gap？
7. $c$ 是实测墙钟时间还是人为预算；是否随 $x$ 变化；能否续算并只支付增量？
8. 跨 fidelity 关系是硬编码、理论有界、从数据学习，还是根本不建模？
9. acquisition 是否真的联合选择 $(x,z)$，还是 fidelity 日程预先固定？
10. 低 fidelity 失败时，方法能否识别并停止使用它？最终是否仍有正确性/最优性保证？

## 八、证据与范围说明

- 本报告逐篇区分了论文原有符号与统一重解释；尤其没有把所有作者记作 $x$ 的量都当成工程 design。
- 第 6、8 篇因出版正文访问限制标为 B；相关小节没有补写无法核实的变量名。其余条目依据本地全文或可检索全文审读。
- “最高 fidelity”通常只是论文的参考计算，并不自动等于现实 ground truth。
- 横向比较中的“有序”指论文算法所采用的次序，不代表每个样本点的数值误差必然单调。
- 本报告分析的是方法结构与变量角色，不以论文的实验胜负替代对假设适用性的判断。
