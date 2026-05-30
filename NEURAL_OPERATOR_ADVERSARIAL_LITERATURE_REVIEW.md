# Neural Operator + Adversarial Training Literature Review

本文档记录目前查到的 **neural operator 与 adversarial training / adversarial robustness / GAN-style adversarial loss** 相关工作，并特别区分它们是否使用 numerical solver、是否使用 differentiable solver、是否属于真正的 PGD / worst-case perturbation adversarial training。

## 1. 总体判断

目前来看，**neural operator 和 adversarial training 结合的文章非常少**。

已有工作大致分成三类：

1. **Adversarial robustness evaluation**：对训练好的 FNO 做 adversarial attack，评估鲁棒性。
2. **GAN-style adversarial neural operator**：用 generator/discriminator 的 adversarial loss 学函数分布或提升生成质量。
3. **Physics-loss adversarial training**：用 PGD 找到让 PDE residual / physics-informed loss 变大的输入扰动，然后训练模型降低这个 physics loss。

但是，目前没有看到成熟工作系统性地做：

```math
\min_\theta \max_{\|\delta\| \le \epsilon}
\|G_\theta(a+\delta)-S(a+\delta)\|
```

其中 `G_theta` 是 neural operator，`S` 是 numerical PDE solver，且 `S(a+delta)` 在训练或 attack 中作为 solver-consistent reference。

也就是说，**真正 solver-consistent / differentiable-solver-in-the-loop 的 neural operator adversarial training 目前基本是空白或至少非常少见**。

---

## 2. 文章分类总表

| Paper | 类型 | 是否 Neural Operator | Adversarial 含义 | 是否 PGD / worst-case input perturbation | 是否 solver-in-the-loop | 和我们工作的关系 |
|---|---|---:|---|---:|---:|---|
| Evaluating the Adversarial Robustness for Fourier Neural Operators | Robustness evaluation / attack | 是 | 对 FNO 输入做 adversarial perturbation | 是，偏 attack/eval | solver output 用作评估 reference | 很相关，但不是 adversarial training |
| Generative Adversarial Neural Operators (GANO) | GAN-style function generation | 是 | generator vs discriminator | 否 | 否 | 相关但不是同类 |
| StablePDENet | Physics-loss adversarial training | 是 | PGD 最大化 physics-informed loss | 是 | 否，主要用 physics residual | 最接近的 baseline |
| Learning Turbulent Flows with Generative Models / adv-NO | GAN/adversarial loss for turbulent flow | 是或 operator-like surrogate | discriminator/adversarial loss 提升高频细节 | 否 | 否 | 相关但不是 robustness training |
| Active learning neural operator literature | Adaptive data selection | 是 | 不是 adversarial training | 否 | solver 常用于标注新样本 | 可作为后续扩展 |

---

## 3. Evaluating the Adversarial Robustness for Fourier Neural Operators

**Paper**: Evaluating the Adversarial Robustness for Fourier Neural Operators  
**Authors**: Abolaji D. Adesoji, Pin-Yu Chen  
**arXiv**: https://arxiv.org/abs/2204.04259  
**Date**: 2022-04-08

### 核心内容

这篇文章是目前最直接相关的 **FNO adversarial robustness evaluation** 工作。它研究：训练好的 Fourier Neural Operator 在输入函数被 adversarial perturbation 扰动以后，预测结果会不会显著变差。

论文摘要明确说，它生成 norm-bounded input perturbations，并用 FNO 输出和 PDE solver 输出之间的 MSE 来评估模型鲁棒性。

### 是否 adversarial training？

主要不是完整 adversarial training，而是 **attack / robustness evaluation**。

它更像：

```math
\text{given trained } G_\theta,
\quad
\max_{\|\delta\| \le \epsilon}
\mathcal{L}(G_\theta(a+\delta), u_{ref})
```

然后观察 FNO 的误差如何随 perturbation 变大。

### 是否使用 solver？

它使用 PDE solver output 作为 evaluation reference。也就是说，solver 主要用于评估 FNO 输出和真实 PDE 解之间的差距。

但它不是：

```math
\min_\theta \max_\delta
\|G_\theta(a+\delta)-S(a+\delta)\|
```

这种 solver-consistent adversarial training。

### 和我们工作的关系

这篇文章很重要，因为它证明：

> FNO 在 norm-bounded adversarial input perturbation 下鲁棒性会快速下降。

但它和我们的潜在贡献不同：

- 它偏 robustness evaluation；
- 我们可以做 structured OOD generalization dataset；
- 我们可以做 solver-consistent adversarial objective；
- 我们可以比较 PGD、Jacobian spectral norm、JVP/VJP power iteration、solver gap。

---

## 4. Generative Adversarial Neural Operators (GANO)

**Paper**: Generative Adversarial Neural Operators  
**Authors**: Md Ashiqur Rahman, Manuel A. Florez, Anima Anandkumar, Zachary E. Ross, Kamyar Azizzadenesheli  
**Venue**: Transactions on Machine Learning Research, 2022  
**arXiv**: https://arxiv.org/abs/2205.03017  
**PDF**: https://arxiv.org/pdf/2205.03017

### 核心内容

GANO 把 GAN 从有限维数据推广到无限维函数空间。它有两个主要模块：

- generator neural operator；
- discriminator neural functional。

generator 输入一个随机函数，比如从 Gaussian random field 采样的函数，然后输出 synthetic data function。discriminator 判断输入函数是真实数据函数还是生成函数。

### adversarial 是什么意思？

这里的 adversarial 是 **GAN / Wasserstein GAN** 意义上的 adversarial：

```math
\min_G \max_D
\mathbb{E}_{u \sim P_{data}}[D(u)]
-
\mathbb{E}_{z \sim P_Z}[D(G(z))].
```

它不是 PGD / worst-case input perturbation。

### 是否使用 PDE solver？

核心训练中 **没有使用 PDE solver**。

它不是在做：

```math
G_\theta(a+\delta) \quad \text{vs} \quad S(a+\delta).
```

它是在学习函数分布：

```math
z \sim P_Z,
\quad
u_{fake}=G_\theta(z),
\quad
u_{fake} \sim P_{data}.
```

### 实验对象

论文实验主要包括：

- controlled GRF / mixture of GRFs function data；
- InSAR volcanic deformation real-world function data。

这些实验不是 supervised PDE solution operator learning，也不是 solver-in-the-loop adversarial training。

### 和我们工作的关系

GANO 可以作为 “neural operator + adversarial objective” 的相关工作，但不能作为“别人已经做过 neural operator adversarial training robustness”的证据。

它属于：

> Neural Operator + GAN-style function distribution modeling.

而我们的方向更像：

> Neural Operator + PGD / worst-case input perturbation + solver-consistent robustness.

---

## 5. StablePDENet

**Paper**: StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations  
**Authors**: Chutian Huang, Chang Ma, Kaibo Wang, Yang Xiang  
**arXiv**: https://arxiv.org/abs/2601.06472  
**Date**: 2026-01-10

### 核心内容

StablePDENet 是目前查到的最接近我们方向的文章。它研究 neural operator 在输入扰动下的 stability，并用 adversarial training 提高模型在 perturbed input 下的表现。

它把训练写成 min-max 问题：

```math
\min_\theta \max_{\|\delta\| \le \epsilon}
\mathcal{L}_{phys}(G_\theta(a+\delta)).
```

其中：

- `G_theta` 是 neural operator；
- `a` 是输入函数；
- `delta` 是 adversarial perturbation；
- `L_phys` 是 physics-informed loss / PDE residual loss。

### physics-informed loss 是什么？

它不是直接比较预测解和 solver 解，而是把模型输出代回 PDE，检查 PDE residual。

例如 Poisson 方程：

```math
-\Delta u = f.
```

模型预测：

```math
u_\theta = G_\theta(f).
```

physics residual 是：

```math
-\Delta u_\theta - f.
```

physics loss 是：

```math
\mathcal{L}_{PDE}
=
\frac{1}{N}
\sum_i
|-\Delta u_\theta(x_i)-f(x_i)|^2.
```

如果有边界条件和初始条件，还会加上 boundary loss 和 initial condition loss。

### 是否用 solver-in-the-loop？

关键点：**它不是 differentiable solver-in-the-loop adversarial training**。

训练阶段的路径是：

```math
a+\delta
\rightarrow
G_\theta(a+\delta)
\rightarrow
u_\theta
\rightarrow
\mathcal{L}_{phys}.
```

它不是：

```math
a+\delta
\rightarrow
S(a+\delta)
\rightarrow
u_{ref}^{adv},
```

也不是优化：

```math
\|G_\theta(a+\delta)-S(a+\delta)\|.
```

所以它的 adversarial objective 是 residual-consistent，而不是 solver-consistent。

### solver 在哪里出现？

solver 主要用于最后 evaluation，生成 reference solution，然后比较模型预测误差。

训练时不用 solver 反传，也不在每个 PGD step 里面重新求 `S(a+delta)`。

### Jacobian norm 的意义

StablePDENet 比较普通模型和 adversarially trained 模型的 Jacobian spectral norm。

对于 operator：

```math
G_\theta: a \mapsto u,
```

局部线性化：

```math
G_\theta(a+\delta)
\approx
G_\theta(a) + DG_\theta(a)[\delta].
```

离散后，`DG_theta(a)` 是 Jacobian。它的最大奇异值 / spectral norm 表示模型对输入扰动的最大局部放大倍数。

StablePDENet 发现 adversarial training 后 Jacobian spectral norm 变小，说明模型局部 Lipschitz constant 降低，输入扰动不容易被放大。

### 局限

1. physics residual 不等于真实 solution error；
2. residual stability 不一定 imply solver-consistent stability；
3. 没有 differentiable solver；
4. 对 time-dependent PDE，如果模型只输出 final snapshot，就很难构造完整 `u_t` residual；
5. large-scale FNO 的 Jacobian spectral norm 显式计算很贵，更适合 matrix-free JVP/VJP。

### 和我们工作的关系

这篇文章是最直接的 baseline / related work。

它做的是：

```math
\max_\delta \mathcal{L}_{phys}(G_\theta(a+\delta)).
```

我们可以做的是：

```math
\max_\delta \|G_\theta(a+\delta)-S(a+\delta)\|,
```

或者：

```math
\max_\delta \|G_\theta(a+\delta)-G_\theta(a)\|.
```

一句话定位：

> StablePDENet focuses on physics-residual stability; our work can focus on solver-consistent robustness and worst-case operator sensitivity.

---

## 5.1 StablePDENet Attack Objective vs Evaluation Metric

A key subtlety is that StablePDENet's attack objective and evaluation metric are different. During PGD attack/training, it maximizes a physics-informed residual loss:

```math
\mathcal{L}_{phys}(G_\theta(a+\delta)).
```

During evaluation, it compares the model output against a numerical-solver reference solution:

```math
\frac{\|G_\theta(a+\delta)-S(a+\delta)\|}{\|S(a+\delta)\|}.
```

So the attack is physics-residual based, while the reported robustness error is solver-reference based. These two quantities are related but not identical. See `STABLEPDENET_ATTACK_VS_EVALUATION_NOTE.md` for the full note.

---

## 6. Learning Turbulent Flows With Generative Models / adv-NO

**Paper**: Learning Turbulent Flows with Generative Models: Super-resolution, Forecasting, and Sparse Flow Reconstruction  
**Authors**: Vivek Oommen, Siavash Khodakarami, Aniruddha Bora, Zhicheng Wang, George Em Karniadakis  
**arXiv**: https://arxiv.org/abs/2509.08752  
**Date**: 2025-09-10

### 核心内容

这篇文章研究 turbulent flow 的 generative modeling，包括：

- spatio-temporal super-resolution；
- forecasting；
- sparse flow reconstruction。

它指出 standard L2 loss 训练 neural operator 时容易 oversmooth fine-scale turbulent structures，尤其是高频小尺度结构。于是它引入 adversarially trained neural operator，也就是 adv-NO，用 adversarial / generative loss 来改善输出的高频细节和统计结构。

arXiv 摘要中提到：

- 在 Schlieren jet super-resolution 中，adv-NO 降低 energy-spectrum error，并保持 sharp gradients；
- 在 3D homogeneous isotropic turbulence 中，adv-NO 能较快预测湍流演化；
- sparse reconstruction 中使用 conditional generative model 重建 3D velocity / pressure fields。

### adversarial 是什么意思？

这里的 adversarial 也是 discriminator-based adversarial loss，用来让生成或预测的流场更像真实流场。

它主要解决：

> L2 loss 导致高频结构过平滑。

不是解决：

> 输入 initial condition / coefficient 被 worst-case PGD perturbation 后模型是否稳定。

### 是否使用 differentiable solver？

没有看到它使用 differentiable solver-in-the-loop 做 adversarial training。它主要使用已有 turbulent flow 数据做 supervised / generative training。

### 和我们工作的关系

它和我们相关在：

- neural operator / operator-like surrogate；
- physical flow data；
- adversarial loss；
- high-frequency / spectral behavior。

但它不是 PGD-style robustness training，也不是 solver-consistent adversarial training。

---

## 7. Active Learning Neural Operator

这不是 adversarial training，但和我们的 generalization datasets 很相关。

相关关键词：

- active learning neural operator；
- adaptive sampling neural operator；
- uncertainty-guided neural operator training；
- multifidelity neural operator；
- adaptive data acquisition for PDE surrogate models。

可以把我们的 pipeline 接成 active learning：

1. 在原始 train distribution 上训练 FNO；
2. 构造 structured OOD / generalization datasets；
3. 在每个数据集上评估 relative L2 / solver-consistent error；
4. 找出 hardest distributions；
5. 只在这些 hard distributions 上调用 numerical solver 生成新训练样本；
6. fine-tune 或 retrain neural operator。

所以 active learning 是后续可以接的方向。

---

## 8. 我们工作的空位

从目前查到的文章来看，比较清楚的空位是：

> Neural operator adversarial training has mostly appeared as GAN-style adversarial loss, adversarial robustness evaluation, or physics-residual adversarial training. Solver-consistent adversarial training with numerical-solver references, especially for FNOs under structured distribution shifts, remains underexplored.

我们的工作可以强调：

1. **Structured generalization distributions**：不是随机 OOD，而是 kernel、correlation length、range shift、binary morphology、spectrum 等有物理/统计含义的分布变化；
2. **Multiple PDE benchmarks**：1D Burgers、2D Darcy、2D Navier-Stokes；
3. **Solver-consistent robustness**：比较 `G_theta(a+delta)` 和 `S(a+delta)`，而不是只看 physics residual；
4. **Worst-case operator sensitivity**：用 PGD、JVP/VJP、matrix-free power iteration 估计最敏感方向；
5. **Physics residual vs solver error**：验证 residual stability 是否真的 imply solver-consistent stability。

---

## 9. 可以直接放进论文的 Related Work 段落

Prior work on adversarial robustness for neural operators remains limited. Adesoji and Chen studied adversarial robustness of Fourier Neural Operators by generating norm-bounded perturbations of PDE inputs and evaluating degradation against PDE-solver references. This line of work demonstrates the sensitivity of trained FNOs, but primarily focuses on attack and evaluation rather than solver-consistent adversarial training.

Another related direction uses adversarial objectives in a generative sense. Generative Adversarial Neural Operators extend GANs to infinite-dimensional function spaces by pairing a generator neural operator with a discriminator neural functional. Similarly, adversarially trained neural operators for turbulent-flow super-resolution and forecasting use discriminator-based losses to improve high-frequency realism and reduce spectral over-smoothing. These approaches use GAN-style adversarial losses and are distinct from PGD-style worst-case input perturbation robustness.

StablePDENet is closer to our setting, as it formulates operator learning as a min-max problem and uses PGD to find worst-case perturbations of input functions. However, its adversarial objective is a physics-informed residual loss rather than a solver-consistent discrepancy. Numerical solvers are mainly used for reference evaluation, not as differentiable solver-in-the-loop training objectives. In contrast, our work studies structured distribution shifts and solver-consistent robustness of Fourier Neural Operators across Burgers, Darcy flow, and Navier-Stokes benchmarks, enabling a direct comparison between physics-residual stability, output sensitivity, and numerical-solver-consistent error.

---

## 10. Reference List

1. Abolaji D. Adesoji, Pin-Yu Chen. **Evaluating the Adversarial Robustness for Fourier Neural Operators**. arXiv:2204.04259, 2022. https://arxiv.org/abs/2204.04259
2. Md Ashiqur Rahman, Manuel A. Florez, Anima Anandkumar, Zachary E. Ross, Kamyar Azizzadenesheli. **Generative Adversarial Neural Operators**. Transactions on Machine Learning Research, 2022. https://arxiv.org/abs/2205.03017
3. Chutian Huang, Chang Ma, Kaibo Wang, Yang Xiang. **StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations**. arXiv:2601.06472, 2026. https://arxiv.org/abs/2601.06472
4. Vivek Oommen, Siavash Khodakarami, Aniruddha Bora, Zhicheng Wang, George Em Karniadakis. **Learning Turbulent Flows with Generative Models: Super-resolution, Forecasting, and Sparse Flow Reconstruction**. arXiv:2509.08752, 2025. https://arxiv.org/abs/2509.08752

