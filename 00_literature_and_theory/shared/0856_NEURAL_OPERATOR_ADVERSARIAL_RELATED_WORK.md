# Neural Operator Adversarial Training And Related Work Notes

This file records the papers and discussion points we identified around neural operators, adversarial training, GAN-style adversarial losses, active learning, and solver-consistent robustness.

## High-Level Taxonomy

| Category | Uses neural operator? | Adversarial meaning | Solver in training? | Closest to our work? |
|---|---|---|---|---|
| GAN-style neural operator generation | Yes | Generator vs discriminator | Usually no PDE solver | Related but different |
| Adversarial robustness evaluation | Yes | Attack trained model inputs | Solver may be used for data/eval | Close for attack/eval |
| Physics-loss adversarial training | Yes | PGD maximizes PDE residual | No solver-in-the-loop | Very close baseline |
| Solver-consistent adversarial training | Yes | PGD maximizes error vs numerical solution operator | Could use solver or cached solver labels | Our target direction |
| Attack-only neural-operator digital twins | Yes | Sparse or gradient-free attacks on deployed surrogates | Solver may provide labels/evaluation references | Close for motivation, not training |
| Active learning neural operator | Yes | Not adversarial; chooses new samples | Solver used to label selected samples | Related for data selection |

The key distinction for our project is:

> GAN-style adversarial loss is not the same as PGD/worst-case input perturbation adversarial training.

And another key distinction is:

> Physics-residual stability is not necessarily solver-consistent stability.

---

## 1. Generative Adversarial Neural Operators (GANO)

**Paper**: Generative Adversarial Neural Operators  
**Authors**: Md Ashiqur Rahman, Manuel A. Florez, Anima Anandkumar, Zachary E. Ross, Kamyar Azizzadenesheli  
**Venue**: Transactions on Machine Learning Research, 2022  
**arXiv**: https://arxiv.org/abs/2205.03017  
**PDF**: https://arxiv.org/pdf/2205.03017

### What It Does

GANO extends GANs from finite-dimensional Euclidean data to infinite-dimensional function spaces. It has:

- a generator neural operator;
- a discriminator neural functional;
- an input function distribution such as a Gaussian random field (GRF);
- generated synthetic function samples;
- a Wasserstein GAN-style min-max objective.

The generator receives a random input function, often sampled from a GRF, and maps it to a synthetic output function. The discriminator receives either a real function sample or a generated function sample and outputs a scalar score.

### Does It Use A PDE Solver?

For the core adversarial training objective: **no**.

GANO does not train by repeatedly solving a PDE for perturbed inputs. It does not optimize

```math
\|G_\theta(a + \delta) - S(a + \delta)\|,
```

where `S` is a numerical PDE solver.

It instead optimizes a GAN/Wasserstein distribution-matching objective between real functions and generated functions:

```math
u_{fake} = G_\theta(z), \qquad z \sim P_Z,
```

and the discriminator tries to distinguish `u_fake` from real function data `u_real`.

### Experiments

The paper mainly studies:

- controlled function-space data where real samples are GRFs or mixtures of GRFs;
- real InSAR volcanic deformation data.

These experiments are about learning function distributions, not supervised PDE solution operators.

### Relation To Our Work

GANO is useful related work for saying adversarial objectives have been combined with neural operators. But it is **not** evidence that PDE solver-consistent adversarial training for FNOs has already been solved.

GANO category:

> Neural Operator + GAN-style adversarial generative modeling.

Our intended category:

> Neural Operator + PGD/worst-case input perturbation + solver-consistent robustness.

---

## 2. StablePDENet

**Paper**: StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations  
**Authors**: Chutian Huang, Chang Ma, Kaibo Wang, Yang Xiang  
**arXiv**: https://arxiv.org/abs/2601.06472  
**Date**: 2026-01-10

### What It Does

StablePDENet studies stability of neural operators under input perturbations. It formulates operator learning as a min-max adversarial training problem and uses PGD to find input perturbations that maximize a physics-informed loss.

A simplified objective is:

```math
\min_\theta \max_{\|\delta\| \le \epsilon}
\mathcal{L}_{phys}(G_\theta(a + \delta)).
```

Here:

- `G_theta` is the neural operator;
- `a` is an input function, such as source term, boundary condition, initial condition, or coefficient;
- `delta` is an adversarial perturbation of the input function;
- `L_phys` is a physics-informed residual loss.

### What Is The Physics-Informed Loss?

The physics loss checks whether the predicted solution satisfies the PDE and its conditions.

For a PDE

```math
\mathcal{N}(u) = f,
```

with prediction

```math
u_\theta = G_\theta(a),
```

the residual is

```math
r_\theta(x) = \mathcal{N}(u_\theta)(x) - f(x).
```

A typical physics loss is

```math
\mathcal{L}_{phys}
= \mathcal{L}_{PDE}
+ \lambda_{bc}\mathcal{L}_{BC}
+ \lambda_{ic}\mathcal{L}_{IC}.
```

For Poisson,

```math
-\Delta u = f,
```

the PDE residual loss is roughly

```math
\mathcal{L}_{PDE}
= \frac{1}{N}\sum_i |-\Delta u_\theta(x_i) - f(x_i)|^2.
```

### Does It Use A Solver In Training?

Based on the paper framing and our reading: **not as differentiable solver-in-the-loop training**.

It trains through

```math
a + \delta \rightarrow G_\theta(a + \delta) \rightarrow u_\theta
\rightarrow \mathcal{L}_{phys}.
```

It does not train through

```math
a + \delta \rightarrow S(a + \delta) \rightarrow u_{ref}^{adv},
```

and it does not optimize

```math
\|G_\theta(a + \delta) - S(a + \delta)\|.
```

The solver is mainly relevant for generating reference solutions in evaluation, not for every PGD inner step during training.

### PGD Role

PGD is used to find a perturbation `delta` that makes the physics loss large:

```math
\delta^* \approx \arg\max_{\|\delta\| \le \epsilon}
\mathcal{L}_{phys}(G_\theta(a + \delta)).
```

Then the model is trained to reduce the physics loss on these adversarial inputs.

### Jacobian / Frechet Derivative Interpretation

The learned operator is

```math
G_\theta: a \mapsto u.
```

Locally,

```math
G_\theta(a + \delta)
\approx
G_\theta(a) + DG_\theta(a)[\delta].
```

After discretization, `DG_theta(a)` is a Jacobian matrix. If its spectral norm is large, then some input perturbation direction can be strongly amplified by the neural operator. StablePDENet reports smaller Jacobian spectral norms after adversarial training, suggesting improved local stability.

### Strengths

- Directly connects operator learning stability and adversarial robustness.
- Uses PGD, a standard adversarial robustness method.
- Avoids expensive solver calls inside every PGD step.
- Uses Jacobian spectral norm as a stability diagnostic.

### Limitations

- Physics residual is not the same as true solution error.
- A small PDE residual does not always imply small solver-consistent error, especially for stiff, ill-conditioned, long-time, chaotic, or non-normal systems.
- It is closer to PIDeepONet / physics-informed operator learning than to fully supervised FNO solver-consistent adversarial training.
- For time-dependent PDEs, full physics residual requires the model to output a spacetime solution `u(t,x)`. If a model only outputs a final snapshot `u(T,x)`, then a complete time residual such as `u_t - nu Delta u` cannot be computed directly.
- Explicit Jacobian spectral norm is difficult at large FNO/2D/3D scales; matrix-free JVP/VJP power iteration is more scalable.

### Relation To Our Work

StablePDENet is a strong related-work baseline.

StablePDENet studies:

```math
\max_{\|\delta\| \le \epsilon}
\mathcal{L}_{phys}(G_\theta(a + \delta)).
```

Our direction can study:

```math
\max_{\|\delta\| \le \epsilon}
\|G_\theta(a + \delta) - S(a + \delta)\|,
```

or

```math
\max_{\|\delta\| \le \epsilon}
\|G_\theta(a + \delta) - G_\theta(a)\|,
```

or matrix-free estimates of local operator sensitivity through JVP/VJP.

A sharp positioning statement:

> StablePDENet focuses on physics-residual stability; our work can focus on solver-consistent operator robustness and worst-case operator sensitivity.

---

## 2.1 StablePDENet Attack Objective vs Evaluation Metric

Important distinction: StablePDENet attacks a physics residual, but evaluates against solver-reference solution error. The attack objective is `L_phys(G_theta(a+delta))`; the evaluation error is `||G_theta(a+delta)-S(a+delta)|| / ||S(a+delta)||`. These are related but not identical. Full note: `STABLEPDENET_ATTACK_VS_EVALUATION_NOTE.md`.

---

## 3. Evaluating Adversarial Robustness In Fourier Neural Operators

**Paper**: Evaluating Adversarial Robustness in Fourier Neural Operators  
**arXiv**: https://arxiv.org/abs/2204.04259

### What It Does

This work is directly relevant to adversarial attacks and robustness evaluation for Fourier Neural Operators. It studies how vulnerable FNO-type PDE surrogate models are under adversarial perturbations.

### Relation To Our Work

This is close to our attack/evaluation side:

- perturb neural operator inputs;
- evaluate degradation of PDE surrogate predictions;
- study FNO robustness.

However, based on our current discussion, we still need to distinguish between:

- evaluating attacks on a trained FNO;
- actually adversarially retraining the FNO;
- doing solver-consistent attack objectives where perturbed inputs also have perturbed numerical-solver references.

This paper should be cited as core neural-operator adversarial robustness related work.

---

## 4. Learning Turbulent Flows With Generative Models

**Paper**: Learning Turbulent Flows with Generative Models: Super-resolution, Forecasting, and Sparse Flow Reconstruction  
**Authors**: Vivek Oommen, Siavash Khodakarami, Aniruddha Bora, Zhicheng Wang, George Em Karniadakis  
**arXiv**: https://arxiv.org/abs/2509.08752

### What It Does

This paper combines operator learning with generative/adversarial modeling for turbulent-flow tasks such as:

- super-resolution;
- forecasting;
- sparse reconstruction.

It reports an adversarially trained neural operator, sometimes described as `adv-NO`, which uses adversarial loss to improve reconstruction of fine-scale turbulent structures and high-wavenumber content.

### Does It Match PGD Robustness Training?

Not exactly. This is closer to GAN/perceptual/adversarial-loss training for realistic outputs, not necessarily PGD-style worst-case input perturbation training.

### Relation To Our Work

Useful as related work for:

- neural operators with adversarial losses;
- improving high-frequency turbulent flow prediction;
- generative models for PDE-like physical data.

But it is not the same as solver-consistent adversarial robustness under worst-case perturbations of initial conditions or coefficients.

---

## 5. Active Learning For Neural Operators

### Search Direction

Relevant keywords:

- active learning neural operator;
- adaptive sampling neural operator;
- uncertainty-guided neural operator training;
- multifidelity neural operator;
- adaptive data acquisition for PDE surrogate models.

### Key Papers Now Synced

**Active operator learning with predictive uncertainty quantification for PDEs**
by Winovich, Daneker, Lu, and Lin (arXiv:2503.03178) uses predictive
uncertainty to select new PDE examples for labeling. The operator predicts a
mean and uncertainty, typically through a Gaussian negative-log-likelihood
training objective. Evaluation is based on clean prediction error,
uncertainty calibration, and active-learning data efficiency, not adversarial
robustness.

The solver is relevant as the expensive label generator after a sample is
selected. However, the acquisition rule is not solver-integrated: it selects
high-uncertainty inputs rather than maximizing

```math
\|G_\theta(a+\delta)-S(a+\delta)\|.
```

**Multi-Resolution Active Learning of Fourier Neural Operators** by Li, Yu,
Xing, Kirby, Narayan, and Zhe (arXiv:2309.16971) selects both input functions
and simulation resolutions. Its acquisition rule is a utility/cost score based
on probabilistic multi-resolution FNO uncertainty. It is a multi-fidelity,
solver-cost-aware active-learning method, but it is not PGD-style adversarial
attack or robustness training.

### Conceptual Relation

Active learning is not adversarial training, but it is very relevant to our generated generalization datasets.

Our pipeline identifies which distribution shifts are hard:

```math
\text{dataset family} \rightarrow \text{model error} \rightarrow \text{hard regions}
```

An active-learning extension would use that information to decide which new PDE samples to generate with a numerical solver and add to training.

Possible loop:

1. Train FNO on original data.
2. Evaluate on structured OOD/generalization datasets.
3. Rank datasets by relative L2, solver-consistent error, or sensitivity.
4. Select hard regions or high-uncertainty input distributions.
5. Generate new solver-labeled data only there.
6. Retrain or fine-tune the neural operator.

This gives a bridge between generalization evaluation and active learning.

The distinction to preserve in the paper is:

> Active-learning methods query where the model is uncertain or where the
> expected information gain per solver cost is high. Solver-integrated
> adversarial training queries or trains where the learned operator provably
> deviates from the numerical solution operator under a controlled
> perturbation budget.

---

## 5.1 Neural Operator Digital Twin Sparse Attacks

**Paper**: Adversarial Vulnerabilities in Neural Operator Digital Twins:
Gradient-Free Attacks on Nuclear Thermal-Hydraulic Surrogates  
**Authors**: Samrendra Roy, Kazuma Kobayashi, Souvik Chakraborty,
Rizwan-uddin, Syed Bahauddin Alam  
**arXiv**: https://arxiv.org/abs/2603.22525

This paper studies sparse, physically plausible, gradient-free attacks on
neural-operator digital twins. Its main role for our paper is motivation:
clean validation error can be small while sparse perturbations cause large
field-level failures. It is primarily attack and vulnerability evaluation, not
an adversarial-training method and not a solver-integrated training framework.

It also analyzes sensitivity concentration through Jacobian column norms and
an effective perturbation dimension. That diagnostic helps explain why sparse
attacks can succeed, but the attack optimizer itself is gradient-free rather
than a PGD/Jacobian attack.

---

## 6. How To Position Our Work

Our project currently has three parts:

1. **Structured OOD/generalization data** for Burgers, Darcy, and Navier-Stokes.
2. **Robustness/attack evaluation** on retained FNO checkpoints.
3. Potential **adversarial training or active retraining** using hard generated distributions.

A clean positioning is:

> Prior work has studied GAN-style adversarial neural operators and physics-residual adversarial training. In contrast, our work studies structured distribution shifts and solver-consistent robustness of trained FNOs across multiple PDE benchmarks.

A sharper claim:

> Physics-residual stability does not necessarily imply solver-consistent stability; we evaluate neural operators under structured input-distribution shifts and adversarial perturbations using solver-generated references and matrix-free sensitivity analysis.

---

## 7. Suggested Related Work Paragraph

StablePDENet introduces an adversarial training framework for physics-informed neural operators, where PGD is used to generate worst-case perturbations of input functions by maximizing a physics-informed residual loss. The model is then trained to minimize this adversarial physics loss. Unlike solver-consistent approaches, the attack and defense stages do not require recomputing high-fidelity numerical solutions for each perturbation; numerical solvers are mainly used for evaluation. Their experiments show that adversarial physics-informed training improves robustness under perturbed inputs and reduces the Jacobian spectral norm of the learned operator, suggesting enhanced local stability.

Generative Adversarial Neural Operators generalize GANs to infinite-dimensional function spaces by pairing a generator neural operator with a discriminator neural functional. This line of work uses adversarial objectives to learn distributions over functions, rather than to perform PGD-style robustness training for PDE solution operators. Similarly, recent adversarially trained neural operators for turbulent-flow super-resolution and forecasting use adversarial losses to improve realism and high-frequency recovery. These GAN-style objectives are related but distinct from worst-case input perturbation robustness.

Uncertainty-driven active operator learning is another adjacent direction.
Winovich et al. select high-uncertainty PDE instances for additional solver
evaluation, while MRA-FNO selects both input functions and solver resolutions
using a utility/cost acquisition rule. These works reduce the cost of
solver-generated datasets and improve clean generalization, but they do not
construct norm-bounded worst-case perturbations or use model-solver
discrepancy as the attack/training objective.

In contrast, our work directly studies structured out-of-distribution generalization, worst-case operator sensitivity, and solver-consistent discrepancies for trained Fourier neural operators on Burgers, Darcy flow, and Navier-Stokes benchmarks. This setup allows us to compare physics-residual stability, output sensitivity, and solver-consistent error under controlled changes in initial-condition or coefficient distributions.

---

## 8. Reference List

1. Md Ashiqur Rahman, Manuel A. Florez, Anima Anandkumar, Zachary E. Ross, Kamyar Azizzadenesheli. **Generative Adversarial Neural Operators**. Transactions on Machine Learning Research, 2022. https://arxiv.org/abs/2205.03017
2. Chutian Huang, Chang Ma, Kaibo Wang, Yang Xiang. **StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations**. arXiv:2601.06472, 2026. https://arxiv.org/abs/2601.06472
3. **Evaluating Adversarial Robustness in Fourier Neural Operators**. arXiv:2204.04259. https://arxiv.org/abs/2204.04259
4. Vivek Oommen, Siavash Khodakarami, Aniruddha Bora, Zhicheng Wang, George Em Karniadakis. **Learning Turbulent Flows with Generative Models: Super-resolution, Forecasting, and Sparse Flow Reconstruction**. arXiv:2509.08752, 2025. https://arxiv.org/abs/2509.08752
5. Samrendra Roy, Kazuma Kobayashi, Souvik Chakraborty, Rizwan-uddin, Syed Bahauddin Alam. **Adversarial Vulnerabilities in Neural Operator Digital Twins: Gradient-Free Attacks on Nuclear Thermal-Hydraulic Surrogates**. arXiv:2603.22525, 2026. https://arxiv.org/abs/2603.22525
6. Nick Winovich, Mitchell Daneker, Lu Lu, Guang Lin. **Active Operator Learning with Predictive Uncertainty Quantification for Partial Differential Equations**. arXiv:2503.03178, 2025. https://arxiv.org/abs/2503.03178
7. Shibo Li, Xin Yu, Wei Xing, Mike Kirby, Akil Narayan, Shandian Zhe. **Multi-Resolution Active Learning of Fourier Neural Operators**. arXiv:2309.16971, 2024. https://arxiv.org/abs/2309.16971

