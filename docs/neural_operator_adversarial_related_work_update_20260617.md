# Neural Operator Adversarial Related Work Update - 2026-06-17

Status: completed as a literature and local-file inspection. No numerical
experiment, adversarial attack run, training run, plotting job, GPU computation,
or artifact upload was performed.

Question:
- Identify papers related to adversarial training / adversarial attacks /
  adversarial losses for neural operators.
- Check whether the Brown / George Karniadakis group paper is already mentioned
  in the local files.
- Prepare citeable related-work material for the paper.

## Short Answer

The Brown-group paper the user referred to is most likely:

- Vivek Oommen, Siavash Khodakarami, Aniruddha Bora, Zhicheng Wang, George Em
  Karniadakis. **Learning Turbulent Flows with Generative Models:
  Super-resolution, Forecasting, and Sparse Flow Reconstruction**. arXiv
  2509.08752, 2025. https://arxiv.org/abs/2509.08752

Observed from the arXiv abstract: the paper reports an adversarially trained
neural operator, `adv-NO`, for turbulent-flow super-resolution and forecasting.
It uses adversarial/generative losses to reduce spectral oversmoothing and
recover high-frequency turbulent structures. It is related, but it is not the
same as PGD-style worst-case input perturbation robustness or solver-consistent
adversarial training.

Local-file status:

- Mentioned in `NEURAL_OPERATOR_ADVERSARIAL_RELATED_WORK.md`.
- Mentioned in `NEURAL_OPERATOR_ADVERSARIAL_LITERATURE_REVIEW.md`.
- After the 2026-06-17 sync pass, also summarized in root `main.tex` under
  the active-learning / GAN-style neural-operator related-work section.
- `main.tex` does mention a separate neural-operator attack paper:
  **Adversarial Vulnerabilities in Neural Operator Digital Twins:
  Gradient-Free Attacks on Nuclear Thermal-Hydraulic Surrogates**.

## Most Relevant Papers To Cite

### 1. Direct Neural-Operator Adversarial Robustness / Attack

1. Abolaji D. Adesoji, Pin-Yu Chen. **Evaluating the Adversarial Robustness for
   Fourier Neural Operators**. arXiv:2204.04259, 2022.
   https://arxiv.org/abs/2204.04259

   Use for: the closest early FNO adversarial robustness evaluation paper.

   Observed from arXiv: it generates norm-bounded input perturbations for FNOs
   and evaluates degradation by comparing FNO output to PDE-solver output. It is
   mainly attack/evaluation, not adversarial retraining.

2. Chutian Huang, Chang Ma, Kaibo Wang, Yang Xiang. **StablePDENet: Enhancing
   Stability of Operator Learning for Solving Differential Equations**.
   arXiv:2601.06472, 2026. https://arxiv.org/abs/2601.06472

   Use for: the closest adversarial-training baseline.

   Observed from arXiv and local notes: it formulates operator learning as a
   min-max problem against worst-case input perturbations. The adversarial loss
   is physics-informed residual loss, not solver-consistent discrepancy
   `||G_theta(a+delta)-S(a+delta)||`.

3. Samrendra Roy, Kazuma Kobayashi, Souvik Chakraborty, Rizwan-uddin, Syed
   Bahauddin Alam. **Adversarial Vulnerabilities in Neural Operator Digital
   Twins: Gradient-Free Attacks on Nuclear Thermal-Hydraulic Surrogates**.
   arXiv:2603.22525, 2026. https://arxiv.org/abs/2603.22525

   Use for: recent neural-operator digital-twin attack evidence.

   Observed from arXiv: it studies sparse, physically plausible,
   gradient-free attacks on neural operator digital twins. This is attack
   robustness rather than adversarial training, but it supports the motivation
   that neural-operator surrogates need robustness evaluation beyond clean
   validation.

### 2. Neural Operator + Adversarial / Generative Loss

4. Md Ashiqur Rahman, Manuel A. Florez, Anima Anandkumar, Zachary E. Ross,
   Kamyar Azizzadenesheli. **Generative Adversarial Neural Operators**.
   Transactions on Machine Learning Research, 2022.
   https://arxiv.org/abs/2205.03017

   Use for: neural operators with GAN-style adversarial objectives in function
   spaces.

   Observed from arXiv: GANO has a generator neural operator and discriminator
   neural functional, and learns distributions over functions. This is not
   PGD-style worst-case perturbation training.

5. Vivek Oommen, Siavash Khodakarami, Aniruddha Bora, Zhicheng Wang, George Em
   Karniadakis. **Learning Turbulent Flows with Generative Models:
   Super-resolution, Forecasting, and Sparse Flow Reconstruction**. arXiv
   2509.08752, 2025. https://arxiv.org/abs/2509.08752

   Use for: the Brown/Karniadakis `adv-NO` paper.

   Observed from arXiv: standard L2 neural operators oversmooth fine turbulent
   structures; adversarially trained neural operators reduce energy-spectrum
   error and preserve sharp gradients. This should be cited as adversarial loss
   for turbulent-flow generation/reconstruction, not as solver-consistent PGD
   robustness training.

### 3. Neural Operator Foundations

6. Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik
   Bhattacharya, Andrew Stuart, Anima Anandkumar. **Fourier Neural Operator for
   Parametric Partial Differential Equations**. arXiv:2010.08895, 2020.
   https://arxiv.org/abs/2010.08895

   Use for: FNO baseline architecture and PDE surrogate framing.

7. Lu Lu, Pengzhan Jin, George Em Karniadakis. **DeepONet: Learning nonlinear
   operators for identifying differential equations based on the universal
   approximation theorem of operators**. arXiv:1910.03193, 2019; later Nature
   Machine Intelligence version. https://arxiv.org/abs/1910.03193

   Use for: DeepONet/operator-learning background and Karniadakis group
   foundation.

8. Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede
   Liu, Kamyar Azizzadenesheli, Anima Anandkumar. **Physics-Informed Neural
   Operator for Learning Partial Differential Equations**. arXiv:2111.03794,
   2021. https://arxiv.org/abs/2111.03794

   Use for: physics-informed neural operators, data-plus-residual training.

9. Lu Lu, Xuhui Meng, Shengze Cai, Zhiping Mao, Somdatta Goswami, Zhongqiang
   Zhang, George Em Karniadakis. **A comprehensive and fair comparison of two
   neural operators (with practical extensions) based on FAIR data**.
   arXiv:2111.05512, 2021. https://arxiv.org/abs/2111.05512

   Use for: FNO/DeepONet comparison, robustness/noise sensitivity context.

### 3.1 Active Learning / Solver-Cost-Aware Operator Learning

9a. Nick Winovich, Mitchell Daneker, Lu Lu, Guang Lin. **Active Operator
    Learning with Predictive Uncertainty Quantification for Partial
    Differential Equations**. arXiv:2503.03178, 2025.
    https://arxiv.org/abs/2503.03178

    Use for: uncertainty-driven active operator learning. The solver provides
    labels for selected PDE instances, but the acquisition score is predictive
    uncertainty rather than solver-model discrepancy or PGD perturbation
    robustness.

9b. Shibo Li, Xin Yu, Wei Xing, Mike Kirby, Akil Narayan, Shandian Zhe.
    **Multi-Resolution Active Learning of Fourier Neural Operators**.
    arXiv:2309.16971, 2024. https://arxiv.org/abs/2309.16971

    Use for: multi-fidelity active learning with FNOs. The method selects both
    input functions and simulation resolutions using a utility/cost
    acquisition rule. It is solver-cost-aware data acquisition, not
    adversarial attack/training.

### 4. General Adversarial-Training Foundations

10. Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy. **Explaining and
    Harnessing Adversarial Examples**. arXiv:1412.6572, 2014.
    https://arxiv.org/abs/1412.6572

    Use for: FGSM and the general adversarial-example starting point.

11. Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras,
    Adrian Vladu. **Towards Deep Learning Models Resistant to Adversarial
    Attacks**. arXiv:1706.06083, 2017.
    https://arxiv.org/abs/1706.06083

    Use for: robust optimization / PGD adversarial training.

12. Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui,
    Michael I. Jordan. **Theoretically Principled Trade-off between Robustness
    and Accuracy**. arXiv:1901.08573, 2019.
    https://arxiv.org/abs/1901.08573

    Use for: robustness-accuracy tradeoff and TRADES.

13. Francesco Croce, Matthias Hein. **Reliable evaluation of adversarial
    robustness with an ensemble of diverse parameter-free attacks**.
    arXiv:2003.01690, 2020. https://arxiv.org/abs/2003.01690

    Use for: robust-evaluation caution, attack tuning, gradient masking.

## How To Position Our Paper

The clean related-work split should be:

1. **Neural operators as PDE solvers.** Cite FNO, DeepONet, PINO, and FNO-vs-
   DeepONet comparison.
2. **Adversarial robustness and adversarial training.** Cite Goodfellow,
   Madry, TRADES, and AutoAttack.
3. **Adversarial robustness of neural operators.** Cite Adesoji--Chen and Roy
   et al.
4. **Adversarial losses for neural operators.** Cite GANO and Karniadakis
   `adv-NO`.
5. **Uncertainty-driven active learning for neural operators.** Cite
   Winovich et al. and MRA-FNO. Distinguish solver label generation from
   solver-integrated adversarial objectives.
6. **Physics-residual adversarial training.** Cite StablePDENet as the closest
   baseline.
7. **Our gap.** Prior work either attacks FNOs, uses GAN-style adversarial
   losses, or uses physics-residual adversarial training. Our paper studies
   solver-integrated adversarial attack and training, where robustness is judged
   through the model-solver discrepancy and multiple attack losses are compared
   under the same perturbation budget.

## Draft Related-Work Paragraph

Prior work on adversarial robustness for neural operators remains limited.
Adesoji and Chen evaluated Fourier Neural Operators under norm-bounded
adversarial perturbations and measured degradation against PDE-solver
references. More recently, Roy et al. studied sparse gradient-free attacks on
neural-operator digital twins, showing that physically small input changes can
cause large surrogate-solver errors. These works motivate robustness evaluation
for scientific operator-learning models, but they primarily study attack
vulnerability rather than solver-consistent adversarial training.

Another line of work uses adversarial objectives in a generative sense.
Generative Adversarial Neural Operators extend GANs to function spaces through
a generator neural operator and a discriminator neural functional. The
Karniadakis group also introduced adversarially trained neural operators
(`adv-NO`) for turbulent-flow super-resolution and forecasting, where the
adversarial loss helps recover sharp gradients and high-frequency spectral
content. These adversarial losses are important related work, but they differ
from PGD-style worst-case input perturbation robustness.

Active-learning work addresses a different neighboring problem: reducing the
cost of solver-generated training data. Winovich et al. use predictive
uncertainty to choose PDE instances for additional solver labeling, and
MRA-FNO chooses both input functions and simulation resolutions with a
utility/cost acquisition rule. These methods improve clean generalization and
data efficiency, but they do not define adversarial perturbation budgets or use
the solver-model discrepancy as the acquisition/training objective.

StablePDENet is closest to our setting because it formulates operator learning
as a min-max problem and uses PGD to find worst-case input perturbations.
However, its inner objective maximizes a physics-informed residual loss, while
our solver-integrated setting asks whether the neural operator remains aligned
with the numerical solution operator under perturbation. This distinction lets
us compare model-only sensitivity, physics residuals, and model-solver
discrepancy under controlled adversarial budgets, and it separates clean
generalization from local and finite-budget robustness.

## Suggested BibTeX Keys

- `adesoji2022fno_adversarial`
- `huang2026stablepdenet`
- `roy2026neural_operator_digital_twin_attacks`
- `rahman2022gano`
- `oommen2025adv_no_turbulence`
- `li2020fourier_neural_operator`
- `lu2019deeponet`
- `li2021pino`
- `lu2021fair_neural_operator_comparison`
- `goodfellow2014adversarial`
- `madry2017pgd`
- `zhang2019trades`
- `croce2020autoattack`

## Local Search Notes

Observed from local `rg` searches:

- `NEURAL_OPERATOR_ADVERSARIAL_RELATED_WORK.md` mentions the Karniadakis
  turbulent-flow generative-model paper around its "Learning Turbulent Flows
  With Generative Models" section.
- `NEURAL_OPERATOR_ADVERSARIAL_LITERATURE_REVIEW.md` mentions it in the
  `Learning Turbulent Flows With Generative Models / adv-NO` section and in the
  reference list.
- Root `main.tex` now includes the 2026 neural-operator digital-twin attack
  paper, Winovich active operator learning, MRA-FNO, GANO, and the
  Karniadakis/Oommen `adv-NO` turbulence paper in the synchronized
  related-work notes.
