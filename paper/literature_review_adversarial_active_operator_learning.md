# Literature Review Map: Adversarial, Active, and Physics-Informed Operator Learning

This note organizes the related work around the paper's central question:

> How should adversariality, generalization, and robustness be defined when a neural operator is evaluated against a numerical solver under perturbed PDE inputs?

The key distinction is whether a method uses the numerical solver only as an offline label generator, or whether the solver is integrated into the attack/training objective itself.

## Main Related-Work Thesis

The related work should be organized around three partially overlapping lines:

1. **Active learning for operator learning.**
2. **Adversarial training for PINNs or neural operators.**
3. **Adversarial robustness evaluation for neural operators.**

These lines are relevant, but none of them fully addresses the setting of this paper. The key reason is that the present problem is not an ordinary regression task without a reference model. For PDE operator learning, a high-fidelity numerical solver already exists. The neural operator is therefore a student model, and the numerical solver is the teacher. The central goal is to transfer the solver's input-output behavior into a fast neural operator while preserving accuracy and robustness under input perturbations.

This teacher-student view changes how robustness should be defined. Robustness should not only mean that the neural operator output changes smoothly, nor only that a physics residual remains small. It should mean that, under the same perturbed input, the neural operator remains close to the solver:

\[
L_{\mathrm{true}}(\delta)
=
\|F_\theta(a+\delta)-G(a+\delta)\|.
\]

This is the main gap in existing work. Many papers either:

- study active learning and improve clean prediction accuracy, but do not define adversarial robustness;
- study adversarial robustness/evaluation, but do not provide a solver-integrated adversarial training method;
- use adversarial or adaptive sampling, but optimize a surrogate such as a PDE residual, predictive uncertainty, or model-only sensitivity;
- use solver outputs as offline labels, but do not put \(G(a+\delta)\) inside the attack objective;
- avoid solver gradients/backpropagation entirely, even though the solver is the true teacher in this problem;
- report prediction/generalization loss, but do not define a metric suite that separates clean generalization from local/worst-case robustness.

The proposed paper should therefore emphasize that **generalization and robustness are different quantities**. Generalization is a static out-of-sample prediction error,

\[
\|F_\theta(a)-G(a)\|,
\]

while robustness is a local or adversarial property around an input,

\[
\max_{\|\delta\|_p\le \varepsilon}
\|F_\theta(a+\delta)-G(a+\delta)\|_q.
\]

Existing papers often blur this distinction. Some show that a method lowers clean test error or improves data efficiency and then use reliability or robustness language. Others evaluate attacks but do not train against the true solver-consistent failure mode. A central contribution of this paper is to define these quantities explicitly and compare solver-consistent robustness with surrogate robustness measures.

## Explicit Limitations of the Closest Prior Work

The closest prior work should be discussed carefully because several papers look superficially similar to this paper but differ in the most important part: **how the adversarial perturbation is generated and whether the solver participates in the loss objective**.

### StablePDENet

StablePDENet is relevant because it studies stability/adversarial training for operator learning. However, its central limitation for our setting is that the adversarial objective is based on a physics/stability residual rather than a solver-consistent teacher-student loss. In other words, the method attacks a quantity of the form

\[
\|\mathcal{P}(F_\theta(a+\delta))\|,
\]

not

\[
\|F_\theta(a+\delta)-G(a+\delta)\|.
\]

Therefore, the numerical solver is not deeply integrated into the perturbation-generation objective. The solver does not appear as \(G(a+\delta)\) inside the adversarial loss, and the solver's forward or backward computation is not used as the teacher signal for generating \(\delta\). StablePDENet is thus best understood as a physics-loss adversarial/stability method, not a solver-integrated adversarial-training method.

### AT-PINN / WbAR

AT-PINN/WbAR is also residual-based. It is designed for PINNs rather than neural operators, and it searches for failure regions by maximizing the PINN's PDE residual over collocation points. Its attack/training signal is a physics loss, not a solver-supervised operator loss. It does not compare a neural operator output with a solver output at the same perturbed input, and it does not use solver forward/backward information to construct adversarial perturbations. It is therefore useful as a PINN residual-adversarial baseline, but it does not address solver-integrated robustness for neural operators.

### RAMS

RAMS is very close to the residual-adversarial line above. Conceptually, it can be viewed as combining ideas from residual-based PINN failure-region localization and physics-residual adversarial sampling for operator learning. In its PINN setting, it behaves similarly to AT-PINN/WbAR: samples are moved toward high PDE residual regions. In its physics-informed operator-learning setting, it resembles StablePDENet/PINO-style physics-residual objectives: input functions or query points are moved using a PDE residual gradient.

The subtle point is its data-driven operator-learning setting. RAMS does use solver or experimental outputs after acquisition to label newly selected input functions. However, the generation of the difficult input functions is still driven by a physics residual:

\[
\delta_{\mathrm{RAMS}}
\propto
\nabla_a
\|\mathcal{P}(F_\theta(a))\|^2.
\]

It is not driven by the solver-model discrepancy

\[
\nabla_a
\|F_\theta(a)-G(a)\|^2.
\]

Thus, even when RAMS trains the model with solver-generated labels, the sampling/perturbation-generation step itself is not solver-integrated. The solver is used as a label oracle after sample selection, not as a differentiable teacher inside the acquisition or adversarial objective. This is why RAMS should be classified as a physics-residual-driven sampling/active-learning method rather than a solver-consistent adversarial-training method.

### Adesoji and Chen: Evaluating Adversarial Robustness for FNO

Adesoji and Chen are directly relevant because they evaluate adversarial robustness for Fourier neural operators. Their work shows that FNOs can be vulnerable under norm-bounded input perturbations, and it uses solver outputs as reference targets for measuring degradation.

However, the paper is primarily an attack/evaluation study. It does not provide a full method for improving robustness through solver-integrated adversarial training. It also does not develop a systematic definition that separates:

- clean generalization;
- model-only sensitivity;
- physics residual consistency;
- true solver-consistent adversarial robustness;
- local Jacobian-based robustness of the error operator.

In particular, it does not ask which robustness metric is most appropriate for PDE operator learning, nor does it compare surrogate attack objectives against the true perturbed solver-model discrepancy. This leaves open the central question addressed here: how should robustness be defined and improved when the neural operator is supposed to approximate a numerical solver under the same perturbed input?

## Positioning by Research Line

### Active Learning

Active-learning papers ask which PDE examples should be labeled next because solver calls are expensive. They usually use uncertainty, utility/cost, or model weakness scores to select new inputs, and then call the solver only after selection to generate labels. This is useful for data efficiency and clean generalization, but it is not the same as adversarial robustness.

In active learning, the solver usually plays an **offline oracle** role:

\[
a_{\mathrm{new}} \xrightarrow{\text{solver}} G(a_{\mathrm{new}}).
\]

The solver does not usually appear inside the acquisition gradient:

\[
\nabla_a \|F_\theta(a)-G(a)\|.
\]

Therefore, active learning can improve generalization without directly improving robustness to worst-case perturbations.

### Adversarial Training

Adversarial-training papers generate difficult samples during training. In standard classification, the target is often label loss. In PINNs and physics-informed operator learning, the target is often PDE residual:

\[
\|\mathcal{P}(F_\theta(a+\delta))\|.
\]

This is a meaningful physics-informed surrogate, but it is not the same as the solver-consistent error:

\[
\|F_\theta(a+\delta)-G(a+\delta)\|.
\]

For this paper, this distinction is crucial. The solver is already available and acts as the teacher. If the attack objective ignores the teacher, then the generated perturbation may increase a surrogate loss without exposing the true discrepancy between the student model and the solver.

### Adversarial Robustness

Adversarial-robustness papers evaluate whether neural operators fail under input perturbations. This is closer to the current paper, but many such works stop at vulnerability diagnosis. They may show that FNOs or digital-twin neural operators are fragile, but they often do not:

- define separate metrics for clean generalization, local sensitivity, and adversarial solver-model error;
- compare surrogate attack losses against the true solver-consistent loss;
- integrate solver forward and solver backward into adversarial training;
- use Jacobian-error metrics to connect local theory with adversarial loss increase.

The proposed paper fills this gap by treating robustness as a property of the **error operator**

\[
E(a)=F_\theta(a)-G(a),
\]

not only as a property of \(F_\theta\) alone.

## Classification Axes

### Solver Usage Levels

| Level | Meaning | Example |
|---|---|---|
| S0 | No solver in the method objective; may use PDE residual or data distribution only. | PINN residual training, GANO |
| S1 | Solver generates offline labels or new queried labels. | supervised FNO/DeepONet, active learning, RAMS data-driven mode |
| S2 | Solver output is evaluated at the perturbed input in the objective. | \( \|F_\theta(a+\delta)-G(a+\delta)\| \) |
| S3 | Solver is differentiable and its gradient/backward pass is used to construct the perturbation. | strongest solver-integrated attack/training setting |

Most existing work is S0 or S1. The proposed paper is about the S2/S3 regime.

### Attack/Sampling Objective Types

| Objective | Formula sketch | What it finds |
|---|---|---|
| Data loss | \( \|F_\theta(a)-u^{solver}(a)\| \) | clean prediction/generalization error |
| Physics residual | \( \|\mathcal{P}(F_\theta(a))\| \) | violation of PDE equations |
| Uncertainty | \( \sigma_\theta(a) \) or predictive variance | samples where the model says it is uncertain |
| GAN/adversarial loss | discriminator/generator loss | realistic high-frequency or distributional structure |
| Model-only output change | \( \|F_\theta(a+\delta)-F_\theta(a)\| \) | sensitivity of model output |
| Fixed-label error | \( \|F_\theta(a+\delta)-G(a)\| \) | error relative to unperturbed solver output |
| Solver-consistent error | \( \|F_\theta(a+\delta)-G(a+\delta)\| \) | true perturbed model-solver discrepancy |
| Jacobian spectral norm | \( \|J_E(a)\|_{p\to q} \) | maximum local error-change amplification |
| Jacobian-error proxy | \( \|J_E(a)^* e(a)\| \), \(e(a)=F_\theta(a)-G(a)\) | local direction that increases final error norm |

## Master Comparison Table

| Work | Target model | Main problem | Sampling / attack target | Uses gradient for sampling? | Solver use | Solver gradient/backward? | Robustness or generalization? | Jacobian/local metric? | Relation to this paper |
|---|---|---|---|---|---|---|---|---|---|
| PINNs, Raissi et al. | PINN / coordinate neural field | Solve PDEs with physics constraints | Collocation points; PDE residual loss | Training uses AD through network residual, but not usually adversarial sampling | Usually no solver in loop; may use data/boundary measurements | No | Solution accuracy and physics consistency | No | Foundation for residual-based physics loss |
| PI-DeepONet, Wang et al. | DeepONet | Learn parametric PDE solution operators with physical consistency | Input function and query coordinates; PDE residual + IC/BC/data losses | AD through DeepONet output and coordinates | Optional solver labels; can also train without paired input-output observations | No solver backward | Generalization/data efficiency, physical consistency | No | Physics residual is an auxiliary loss, not solver-consistent attack |
| PINO, Li et al. | FNO / neural operator | Combine data and PDE constraints, often across resolutions | No adversarial sampling by default; PDE residual imposed on predicted solution | Uses differentiable residual of neural operator output | Solver provides coarse training data or evaluation truth | No solver backward | Generalization, zero-shot super-resolution, accuracy | No | Closest physics-informed neural-operator baseline |
| StablePDENet, Huang et al. | Neural operator | Improve stability under input perturbations | Worst-case input perturbations against a physics/stability objective | Yes, adversarial training style | Not solver-consistent; objective is physics residual/stability, not \(F-G\) at perturbed input | No solver backward as core method | Stability/robustness under perturbation, but via residual surrogate | Yes: Jacobian spectral norm / Frechet-derivative norm, but not \(J_E^*e\) | Important physics-loss adversarial baseline |
| AT-PINN / WbAR, Shi et al. | PINN | Locate and refine PINN failure regions | Move collocation points toward high PDE residual | Yes, PGD/white-box residual gradient | No solver in attack/training objective | No | PINN failure localization, accuracy improvement | No | Same residual-adversarial idea, but for PINNs |
| PIAT, Shekarpaz et al. | PINN / neural PDE solver | Physics-informed adversarial training for differential equations | Perturbations generated using network/PDE residual information | Yes | No solver-consistent objective | No | Accuracy/smoothness/robustness-like regularization for PINNs | No | PINN adversarial training, not neural-operator solver discrepancy |
| RAMS, Ouyang et al. | PINNs, PI DeepONet, data-driven DeepONet | Move samples to high-residual regions efficiently | PDE residual of model output | Yes, adversarial gradient of PDE residual | In data-driven mode, solver/experiment labels selected input functions after acquisition | No solver backward in acquisition | Data efficiency/generalization; residual failure sampling | No | Essentially physics-residual adversarial sampling; data-driven mode still uses residual for acquisition |
| Winovich et al. active operator learning | DeepONet/FNO | Use predictive uncertainty for data-efficient operator learning | Highest predictive uncertainty | No PGD-style perturbation; acquisition by UQ score | Solver generates labels for selected PDE examples | No | Generalization, uncertainty calibration, solver-query efficiency | No | Active learning baseline; not adversarial robustness |
| MRA-FNO, Li et al. | FNO | Choose input functions and resolutions under simulation cost | Utility/cost acquisition; probabilistic multi-resolution FNO | Not adversarial PGD; acquisition optimization | Solver/simulation queried for selected labels/resolutions | No | Data efficiency and clean prediction accuracy | No | Active learning and multi-fidelity baseline |
| Adesoji & Chen | FNO | Evaluate adversarial robustness of FNO | Norm-bounded input perturbations; evaluated by solver-output MSE | Yes, adversarial examples for FNO | Solver output used as reference/evaluation target, mostly offline/precomputed | No differentiable solver backward | Adversarial robustness evaluation, not training | No | Direct FNO robustness prior; does not integrate solver gradients |
| Roy et al. neural-operator digital twins | Neural operators for nuclear thermal-hydraulic surrogates | Expose sparse physically plausible adversarial vulnerabilities | Sparse boundary/input perturbations using differential evolution | No gradient; gradient-free differential evolution | Simulation/ground truth for evaluation/data | No | Robustness/attack vulnerability | Yes: effective perturbation dimension and sensitivity diagnostics | Relevant robustness attack paper; closest to local sensitivity diagnostics but not solver-integrated PGD |
| Roy et al. active learning + denoising | Neural operators | Defense through active learning and input denoising | Differential-evolution vulnerability probing, then targeted data generation | Gradient-free search | Solver/data generator labels discovered vulnerability locations | No | Robustness defense plus clean accuracy | Possibly sensitivity/vulnerability oriented, not full Jacobian-error theory | Active robustness defense; still not solver-backprop integrated |
| GANO, Rahman et al. | Neural operator generator/discriminator | Learn distributions on function spaces | GAN/Wasserstein distribution matching | Adversarial discriminator training, not input perturbation attack | No solver objective | No | Generative quality/distribution matching | No | "Adversarial" means GAN-style, not robustness |
| Oommen et al. adv-NO | Neural operators / generative turbulence models | Improve turbulent flow super-resolution, forecasting, reconstruction | GAN/discriminator-style adversarial loss for sharp/high-frequency fields | Adversarial training via discriminator, not PGD input perturbation | Training data may come from simulations/experiments; solver not in attack objective | No | Accuracy, spectrum, sharp gradients; not worst-case robustness | No | Karniadakis-related adv-NO, but generative not solver-consistent |
| Enyeart & Lin adversarial autoencoders | DeepONet / Koopman autoencoder-style operator learning | Add adversarial-autoencoder mechanisms to operator architectures | Latent/distributional adversarial autoencoder objective | GAN/AAE-style, not PGD sampling | No solver-integrated objective | No | Representation/accuracy, not robustness | No | Related by wording only; not central |
| Goodfellow/Madry/TRADES/AutoAttack | Classification/deep learning | Standard adversarial examples and robust training | Worst-case perturbations against label loss | Yes | No PDE solver | No | Classification robustness | Sometimes local robustness theory, but not operator/solver | Background motivation only |
| Sun, Towards Universal Solvers | Neural operators | PGD active learning / distillation from numerical PDE solver | Worst-case inputs under smoothness/energy constraints | Yes | Differentiable solver supervises selected samples | Yes, in the strongest teacher-student setup | OOD generalization/robustness | Not the same Jacobian-error metric framework | Closest prior/self work; current paper should sharpen metrics and solver-integrated loss definitions |

## Per-Paper Detailed Summaries

The following summaries use the same checklist for each paper:

- **Problem:** what the paper studies.
- **Model class:** PINN, DeepONet, FNO, neural operator, or generative model.
- **Sampling / perturbation:** how new samples or adversarial perturbations are generated.
- **Solver usage:** whether the numerical solver is used, and at what level.
- **Delta generation:** whether \(\delta\) is generated by PGD/gradient, residual gradient, uncertainty acquisition, random sampling, or gradient-free search.
- **Robustness/generalization:** what is evaluated.
- **Metrics:** whether the paper defines local/Jacobian robustness metrics or mostly reports prediction loss.
- **Gap relative to this paper:** what it does not solve.

### Fourier Neural Operator (Li et al.)

**Problem.** FNO proposes a neural operator architecture for learning solution operators of parametric PDEs. It maps an input function, such as an initial condition or coefficient field, to an output solution field.

**Model class.** Neural operator, specifically Fourier Neural Operator.

**Sampling / perturbation.** No adversarial sampling is proposed. Training data are sampled PDE instances, usually generated by numerical solvers.

**Solver usage.** The solver is used offline to generate supervised input-output pairs. It is not part of the training objective beyond the stored labels.

**Delta generation.** None.

**Robustness/generalization.** The focus is clean operator-learning accuracy and generalization across resolutions, including zero-shot super-resolution behavior.

**Metrics.** Prediction losses such as relative error. No adversarial robustness metric, no solver-integrated perturbation metric, and no Jacobian-error proxy.

**Gap relative to this paper.** FNO provides the base model class, but it does not define robustness, adversarial training, or solver-integrated attack objectives.

### DeepONet (Lu et al.)

**Problem.** DeepONet learns nonlinear operators between function spaces using branch and trunk networks.

**Model class.** Neural operator, especially coordinate-query operator architecture.

**Sampling / perturbation.** No adversarial or active sampling objective is central to the method.

**Solver usage.** Solver-generated data can be used as supervised labels, but the solver is not inside the training gradient.

**Delta generation.** None.

**Robustness/generalization.** General operator approximation and clean prediction accuracy.

**Metrics.** Standard prediction error; no adversarial robustness or Jacobian robustness.

**Gap relative to this paper.** DeepONet is a foundation for operator learning, but does not address solver-consistent robustness or adversarial training.

### PI-DeepONet (Wang, Wang, and Perdikaris)

**Problem.** PI-DeepONet reduces the need for paired input-output data by imposing PDE constraints on DeepONet predictions.

**Model class.** Physics-informed DeepONet.

**Sampling / perturbation.** The paper uses collocation/query points and input functions to impose residual constraints. It is not primarily an adversarial sampling paper.

**Solver usage.** Solver labels are optional. The central additional signal is the PDE residual, not the numerical solver discrepancy. The solver is not used in the residual gradient.

**Delta generation.** No PGD-style adversarial \(\delta\). Gradients are used to differentiate the neural output with respect to coordinates for the PDE residual.

**Robustness/generalization.** Generalization and data efficiency; physical consistency under PDE constraints.

**Metrics.** Prediction error and residual/constraint satisfaction. No explicit adversarial robustness metric, no solver-consistent \(G(a+\delta)\), and no Jacobian-error metric.

**Gap relative to this paper.** PI-DeepONet uses physics residuals to regularize the student model, but does not define robustness as teacher-student agreement under perturbation.

### PINO (Li et al.)

**Problem.** PINO combines supervised operator learning with PDE residual constraints, often imposing physics at a higher resolution than the training data.

**Model class.** Physics-informed neural operator, mainly based on FNO.

**Sampling / perturbation.** No adversarial sampling by default. The method imposes residual constraints on predicted solution fields.

**Solver usage.** Solver-generated data can provide coarse supervised training labels. The PDE residual is computed from the model output, not from a differentiable solver.

**Delta generation.** None as an attack method. Gradients/finite differences/Fourier derivatives may be used to compute PDE residuals.

**Robustness/generalization.** Clean prediction accuracy, data efficiency, high-resolution consistency, and zero-shot super-resolution.

**Metrics.** Relative prediction errors and physics residuals. No explicit adversarial robustness metric suite and no Jacobian-error proxy.

**Gap relative to this paper.** PINO is a strong physics-informed baseline, but it does not compare surrogate physics residuals with the true perturbed solver-model error.

### StablePDENet (Huang et al.)

**Problem.** StablePDENet aims to improve stability of operator learning under input perturbations.

**Model class.** Neural operator / physics-informed operator learning.

**Sampling / perturbation.** It formulates a min-max problem with worst-case perturbations. The adversarial signal is based on a physics/stability residual rather than the solver-model discrepancy.

**Solver usage.** The method does not deeply integrate a numerical solver into the attack objective as \(G(a+\delta)\). Its core signal is physics/stability consistency.

**Delta generation.** Gradient-based adversarial perturbations against the residual/stability objective.

**Robustness/generalization.** It explicitly discusses stability/robustness under perturbation and also cares about accuracy on normal inputs.

**Metrics.** Robust/stable prediction quality under perturbed inputs. It does not develop the proposed three-part metric suite: clean generalization, true solver-consistent adversarial loss, and local Jacobian-error proxy.

**Gap relative to this paper.** It is close in spirit because it uses adversarial training for operator stability, but the attack target is still a surrogate. It does not place the solver teacher \(G(a+\delta)\) at the center of the robustness definition.

### AT-PINN / WbAR (Shi et al.)

**Problem.** This work localizes failure regions of PINNs and refines training using white-box adversarial attacks.

**Model class.** PINN, not neural operator.

**Sampling / perturbation.** It moves or generates collocation points toward high PDE residual regions.

**Solver usage.** No solver-integrated objective. The failure signal is the PDE residual of the PINN.

**Delta generation.** PGD/white-box residual-gradient style perturbation in coordinate/collocation space.

**Robustness/generalization.** It improves PINN solution quality by focusing on failure regions. The robustness notion is residual/failure-region robustness, not solver-consistent operator robustness.

**Metrics.** Residual reduction and prediction/solution accuracy. No solver-model adversarial loss and no Jacobian-error metric.

**Gap relative to this paper.** It is a useful residual-adversarial PINN baseline, but it is not neural-operator teacher-student learning and does not use solver forward/backward during attack generation.

### PIAT (Shekarpaz et al.)

**Problem.** PIAT applies adversarial training ideas to physics-informed neural networks for solving differential equations.

**Model class.** PINN / neural PDE solver.

**Sampling / perturbation.** Adversarial samples are generated to improve the physics-informed solution behavior.

**Solver usage.** The physics law is encoded through residuals. The method is not based on a solver-consistent \(F_\theta-G\) teacher-student objective.

**Delta generation.** Gradient-based adversarial training around the network/PDE residual structure.

**Robustness/generalization.** It is closer to PINN smoothing/regularization and PDE solution quality than neural-operator robustness.

**Metrics.** PDE solution error and training performance. No solver-integrated robustness metric or Jacobian-error proxy.

**Gap relative to this paper.** PIAT supports the idea that adversarial training can help scientific ML, but it does not address neural operators with a numerical solver teacher.

### RAMS (Ouyang et al.)

**Problem.** RAMS proposes residual-based adversarial-gradient moving samples for PINNs, physics-informed operator learning, and data-driven operator learning.

**Model class.** PINNs and DeepONet/operator-learning settings.

**Sampling / perturbation.** RAMS moves samples in the direction that increases the PDE residual. In PINNs, the moved samples are collocation points. In operator learning, the moved objects can be input functions and/or query points.

**Solver usage.** In the data-driven operator-learning setting, the solver or experiment is used after acquisition to label newly selected input functions. The solver is not used to define the acquisition/attack gradient.

**Delta generation.** Residual-gradient ascent:

\[
\delta_{\mathrm{RAMS}}
\propto
\nabla_{\mathrm{sample}}
\|\mathcal{P}(F_\theta(\mathrm{sample}))\|^2.
\]

This is not a solver-consistent gradient

\[
\nabla_a \|F_\theta(a)-G(a)\|.
\]

**Robustness/generalization.** The paper mainly targets data efficiency and failure-region sampling. It is not primarily a robustness-definition paper.

**Metrics.** Prediction error and residual-based performance. No explicit comparison between physics residual and true solver-model discrepancy, and no Jacobian-error metric suite.

**Gap relative to this paper.** RAMS is important because it extends residual-gradient sampling to operator learning, but even its data-driven mode uses PDE residuals for acquisition. The solver remains a label generator, not a differentiable teacher inside the adversarial objective.

### Active Operator Learning with Predictive UQ (Winovich et al.)

**Problem.** This paper uses predictive uncertainty to make operator-learning data acquisition more efficient.

**Model class.** DeepONet and FNO with uncertainty outputs or secondary UQ networks.

**Sampling / perturbation.** It selects high-uncertainty PDE examples for new solver evaluation.

**Solver usage.** Solver calls provide labels for selected examples. The acquisition score is uncertainty, not solver-model discrepancy.

**Delta generation.** No adversarial \(\delta\). Acquisition is based on predictive uncertainty.

**Robustness/generalization.** The focus is clean generalization, uncertainty calibration, and data efficiency.

**Metrics.** MSE/MAE/relative error, uncertainty calibration, coverage, correlation between uncertainty and error, and active-learning error curves.

**Gap relative to this paper.** It is active learning, not adversarial robustness. It does not define worst-case perturbations or use solver gradients to find model failure directions.

### MRA-FNO (Li et al.)

**Problem.** MRA-FNO reduces data-generation cost by actively choosing both input functions and simulation resolutions.

**Model class.** FNO with probabilistic multi-resolution active learning.

**Sampling / perturbation.** It uses utility-cost acquisition to choose examples and resolutions.

**Solver usage.** Solver/simulation is queried for selected input-resolution pairs.

**Delta generation.** No PGD/adversarial \(\delta\). The method performs active selection, not local worst-case perturbation search.

**Robustness/generalization.** Data efficiency and clean prediction accuracy.

**Metrics.** Prediction error versus labeling cost/resolution budget.

**Gap relative to this paper.** It addresses expensive solver queries, but not solver-consistent adversarial robustness.

### Adesoji and Chen: FNO Adversarial Robustness

**Problem.** This work evaluates adversarial robustness of FNOs under norm-bounded input perturbations.

**Model class.** FNO.

**Sampling / perturbation.** It generates adversarial perturbations to FNO inputs.

**Solver usage.** Solver outputs are used as reference labels for evaluation. The solver is not deeply integrated into the attack gradient.

**Delta generation.** Gradient-based adversarial examples for the FNO setting.

**Robustness/generalization.** It is directly about adversarial robustness evaluation. It shows that FNO performance degrades under adversarial perturbations.

**Metrics.** MSE or prediction error between FNO output and PDE solver output under perturbation.

**Gap relative to this paper.** It evaluates vulnerability but does not develop solver-integrated adversarial training, does not compare \(L_1/L_2/L_3\)-style objectives, and does not introduce a local Jacobian-error robustness metric.

### Roy et al.: Adversarial Vulnerabilities in Neural Operator Digital Twins

**Problem.** This paper attacks neural-operator digital twins in nuclear/thermal-hydraulic settings.

**Model class.** Neural operators for digital-twin field reconstruction.

**Sampling / perturbation.** Sparse, physically plausible perturbations are found using differential evolution.

**Solver usage.** Ground truth/simulation data are used for evaluation, but the attack is not solver-backprop integrated.

**Delta generation.** Gradient-free differential evolution, not PGD through a solver.

**Robustness/generalization.** Strongly focused on adversarial vulnerability and robustness evaluation.

**Metrics.** Relative error degradation, attack success, anomaly-detection bypass, and Jacobian-based diagnostics such as effective perturbation dimension.

**Gap relative to this paper.** It studies robustness and even local sensitivity diagnostics, but it does not define robustness through differentiable solver-consistent teacher-student error, nor does it propose solver-integrated adversarial training.

### Roy et al.: Active Learning and Input Denoising for Robust Neural Operators

**Problem.** This paper proposes a defense combining active learning and input denoising.

**Model class.** Neural operators.

**Sampling / perturbation.** It probes model weaknesses using differential evolution and then generates targeted training data at vulnerability locations.

**Solver usage.** Solver/data generator labels the discovered vulnerability locations. The solver is not used in a differentiable attack objective.

**Delta generation.** Gradient-free attack/search.

**Robustness/generalization.** Robustness defense plus clean accuracy preservation.

**Metrics.** Combined clean and robustness error metrics.

**Gap relative to this paper.** It is relevant because it tries to improve robustness, but it does not formalize solver-consistent adversarial loss or local Jacobian-error theory.

### GANO (Rahman et al.)

**Problem.** GANO learns probability distributions over function spaces using neural operator generators and discriminator functionals.

**Model class.** Generative neural operator.

**Sampling / perturbation.** Samples are generated from a latent/input function distribution such as a Gaussian random field.

**Solver usage.** No numerical solver is integrated into the objective.

**Delta generation.** No adversarial input perturbation. "Adversarial" means GAN/discriminator training.

**Robustness/generalization.** Generative quality and distribution matching, not robustness.

**Metrics.** Distributional/generative quality measures, not solver-model prediction error.

**Gap relative to this paper.** The word "adversarial" has a different meaning. It is not about PDE solver teacher-student robustness.

### Oommen et al.: adv-NO for Turbulent Flows

**Problem.** This work uses generative/adversarial modeling to improve turbulent-flow super-resolution, forecasting, and sparse reconstruction.

**Model class.** Neural operators and generative models for turbulent flows.

**Sampling / perturbation.** No PGD-style input perturbation. The adversarial component is discriminator/generative training.

**Solver usage.** Training data may come from simulation or experiment, but the solver is not inside the adversarial objective.

**Delta generation.** None in the robustness sense.

**Robustness/generalization.** Accuracy, sharp gradients, energy spectra, forecasting, and reconstruction quality.

**Metrics.** Spectrum error, reconstruction/forecasting quality, sharpness/statistical fidelity.

**Gap relative to this paper.** It is Karniadakis-related and uses "adv-NO", but the adversarial mechanism is GAN-style, not solver-consistent worst-case robustness.

### Enyeart and Lin: Adversarial Autoencoders in Operator Learning

**Problem.** This paper studies adversarial-autoencoder additions to operator-learning architectures.

**Model class.** DeepONet / Koopman autoencoder-style operator learning.

**Sampling / perturbation.** No solver-consistent adversarial sampling.

**Solver usage.** No solver-integrated objective.

**Delta generation.** Not a PGD input-attack paper; adversariality is autoencoder/distributional.

**Robustness/generalization.** Mostly representation and operator-learning performance, not solver-consistent robustness.

**Metrics.** Architecture/performance metrics rather than robustness metrics.

**Gap relative to this paper.** Related by the word "adversarial" and operator learning, but not central for PDE robustness.

### General Adversarial Learning: Goodfellow, Madry, TRADES, AutoAttack

**Problem.** These works define and evaluate adversarial examples and adversarial training in standard machine learning, mostly classification.

**Model class.** Standard neural networks, usually classifiers.

**Sampling / perturbation.** FGSM, PGD, robust optimization, TRADES, or attack ensembles.

**Solver usage.** No PDE solver.

**Delta generation.** Gradient-based adversarial perturbations.

**Robustness/generalization.** Classification robustness and the accuracy-robustness tradeoff.

**Metrics.** Robust accuracy, attack success, certified/empirical robustness.

**Gap relative to this paper.** These works motivate adversarial training, but the PDE neural-operator setting is different because the output is continuous, the label changes with input perturbation, and a numerical solver teacher exists.

## Classical Adversarial-Attack Literature and the Fixed-Label Assumption

The general adversarial-attack literature is still important background. It explains why PGD/FGSM-style fixed-budget attacks, minimum-distance attacks, black-box attacks, and robust optimization became standard. However, most of this literature was developed for classification, especially computer vision. In classification, it is natural to assume that a small perturbation of an image does not change the true class label:

\[
y(x+\delta)=y(x).
\]

This assumption makes the standard adversarial objective reasonable:

\[
\max_{\|\delta\|\le\varepsilon}
\ell(f_\theta(x+\delta),y(x)).
\]

For PDE regression, this fixed-label assumption is generally wrong. If \(x\) is an initial condition, coefficient field, source term, or boundary condition, then perturbing \(x\) changes the physical problem. The correct oracle output is therefore not \(G(x)\), but \(G(x+\delta)\). A physically meaningful regression attack should compare

\[
F_\theta(x+\delta)
\quad\text{with}\quad
G(x+\delta),
\]

not with the clean target \(G(x)\). This is why the paper separates:

\[
L_1(\delta)=\|F_\theta(x+\delta)-F_\theta(x)\|,
\]

\[
L_2(\delta)=\|F_\theta(x+\delta)-G(x)\|,
\]

and

\[
L_3(\delta)=\|F_\theta(x+\delta)-G(x+\delta)\|.
\]

The first two objectives resemble standard classification-style or fixed-target attack logic: they either measure model output movement or keep the original target fixed. They can be useful controls, but they are not the correct oracle-relative error for a PDE regression problem. In contrast, \(L_3\) measures the perturbed-input teacher-student error.

### Classical attack papers and what they contribute

| Work | Main contribution | Objective type | Relation to \(L_1/L_2/L_3\) |
|---|---|---|---|
| Szegedy et al., Intriguing Properties | Early demonstration of adversarial examples in deep networks | Find small perturbations causing misclassification | Classification fixed-label setting; no dynamic oracle |
| Goodfellow et al., FGSM | Linearized one-step gradient attack | Fixed-budget max-loss attack | Gives the steepest-ascent intuition for \(L_\infty\) attacks |
| Madry et al., PGD | Robust optimization and multi-step first-order adversarial training | Inner maximization over an \(\varepsilon\)-ball | Motivates PGD for the inner problem, but original setting is classification |
| TRADES | Accuracy-robustness tradeoff | regularized robust objective | Useful background, but not solver-teacher regression |
| AutoAttack / APGD | Reliable attack evaluation and PGD failure modes | attack ensemble / parameter-free evaluation | Motivates careful attack evaluation and avoiding weak PGD conclusions |
| DeepFool | Minimum-distance perturbation to cross decision boundary | minimum-norm attack | Different from fixed-budget max-loss attack |
| Carlini-Wagner | Strong optimization-based targeted attacks | minimum-distortion or confidence-based attack | Classification/targeted attack; no perturbed physical oracle |
| Boundary Attack | Decision-based black-box minimum-distance attack | reduce perturbation while staying adversarial | Classification decision boundary setting |
| Papernot JSMA / black-box attacks | Saliency-map and substitute-model attacks | targeted misclassification | Shows attack transferability, still label/class based |
| One-pixel / Square / HopSkipJump | sparse, random-search, or query-efficient attacks | black-box or sparse perturbation | Useful for attack methodology, not regression-oracle semantics |
| Adversarial Patch / physical-world attacks | robust physical perturbations | targeted classifier attack under transformations | Motivates physically plausible perturbations, but label remains class target |

The key conceptual difference is:

\[
\text{classification attack: keep the semantic label fixed;}
\]

\[
\text{PDE regression attack: recompute the oracle output under the perturbed input.}
\]

If the model is perfect, \(F_\theta=G\), then a correct PDE regression attack objective should remain zero:

\[
\|F_\theta(x+\delta)-G(x+\delta)\|=0.
\]

But the surrogate objectives can still be large:

\[
\|F_\theta(x+\delta)-F_\theta(x)\|
=
\|G(x+\delta)-G(x)\|,
\]

\[
\|F_\theta(x+\delta)-G(x)\|
=
\|G(x+\delta)-G(x)\|.
\]

Thus \(L_1\) and \(L_2\) can falsely report an attack even when the neural operator exactly matches the solver. They may only be measuring ordinary physical sensitivity of the PDE solution, not model failure.

### Regression adversarial attacks and their loss types

There is a smaller but important line of adversarial work for regression. This line is useful because it recognizes that the fixed-label assumption from classification is not automatically valid for continuous-output problems. However, existing regression attacks still usually do not use a perturbed oracle \(G(x+\delta)\).

Nguyen and Raff explicitly study adversarial attacks for neural-network regression and argue that regression attacks are underexplored \citep{nguyen2018regression}. They also make the key observation that, unlike classification, regression outputs should generally change when inputs change. Their proposed defense, however, is primarily a numerical-stability regularizer:

\[
\ell(y-f_\theta(x))
+
\lambda
\mathbb{E}_{\Delta x}
\ell\big(f_\theta(x)-f_\theta(x+\Delta x)\big).
\]

This is closest to an \(L_1\)-like objective because it compares the model output at \(x+\Delta x\) with the model output at \(x\). They introduce a nearest-neighbor label-difference threshold to avoid penalizing reasonable local regression variation too strongly, but the method still does not use a separate perturbed oracle.

Other regression adversarial papers often use fixed-output regression losses. Adversarial linear regression studies objectives such as

\[
\min_w
\sum_i
\max_{\|\delta_i\|\le\varepsilon}
\big(y_i-w^\top(x_i+\delta_i)\big)^2,
\]

where \(y_i\) remains fixed under the perturbation \citep{tong2018adversarialregression,ribeiro2023linearregression}. This is \(L_2\)-like: the model output at the perturbed input is compared with an unperturbed label.

Time-series regression and forecasting attacks also often adapt image-domain methods such as FGSM, BIM, PGD, or APGD. Mode and Hoque attack multivariate time-series regression models by using the gradient of a regression cost such as MSE with respect to the input series \citep{mode2020mtsregression}. Targeted time-series forecasting attacks instead choose an adversarial output target, such as an upward/downward directional shift or an amplitude-limited target, and optimize \(L(f(x_{\mathrm{adv}}),y_0)\) \citep{govindarajulu2023targetedtsf}. Probabilistic forecasting attacks target statistics of the predicted output distribution, for example

\[
\left\|
\mathbb{E}_{f(y|x+\delta)}[\chi(Y)]
-
t_{\mathrm{adv}}
\right\|^2,
\]

or define robustness through output-distribution stability such as Wasserstein deviation \citep{dangnhu2020probforecast,yoon2022robustprobforecast,liu2022forecastingattacks}. These are targeted or distributional analogues of \(L_1/L_2\)-style reasoning, not solver-consistent PDE losses.

Dense visual regression and control-output attacks show the same pattern. Monocular depth attacks compare predicted depth under perturbation with a fixed depth map or target depth \citep{zhang2020depthattack}. Steering-angle attacks such as DeepBillboard maximize steering-angle error magnitude and duration under physical perturbations \citep{zhou2018deepbillboard}. These are meaningful regression attacks, but their reference target is still fixed or task-specific. In image sensing, that can be reasonable because the physical scene may be treated as unchanged while the sensor input is perturbed. In PDE operator learning, however, the perturbed input function defines a new physical problem, so the oracle solution must be \(G(a+\delta)\).

The overall pattern is:

\[
\text{existing regression attacks}
\approx
L_1\text{-like model stability}
\quad\text{or}\quad
L_2\text{-like fixed/targeted output error}.
\]

The solver-consistent objective

\[
L_3(\delta)=\|F_\theta(a+\delta)-G(a+\delta)\|
\]

is rare because most regression settings do not provide a differentiable oracle that can be queried at the perturbed input. PDE operator learning is special precisely because such a solver can exist.

### Classic attack objectives in more detail

Most classical attacks can be grouped by what loss they optimize.

#### 1. Fixed-budget, fixed-label loss maximization

This is the FGSM/PGD family. The attacker solves

\[
\max_{\|\delta\|_p\le \varepsilon}
\ell(f_\theta(x+\delta), y),
\]

where \(y\) is the clean label. For images, the assumption is that the semantic class should not change under a small perturbation. The common untargeted objective is to increase the cross-entropy with respect to the true label:

\[
\ell_{\mathrm{CE}}(f_\theta(x+\delta),y).
\]

For targeted attacks, the objective is usually reversed: make the classifier predict a chosen target class \(t\), for example by minimizing

\[
\ell_{\mathrm{CE}}(f_\theta(x+\delta),t)
\]

or maximizing the target logit relative to others.

FGSM uses a one-step linearized update:

\[
\delta = \varepsilon\,\mathrm{sign}(\nabla_x \ell(f_\theta(x),y))
\]

for the \(L_\infty\) case. PGD iterates this idea and projects back to the \(\varepsilon\)-ball.

Relation to this paper: this is closest to a fixed-oracle attack only if the target does not change. In PDE regression, the solver output usually changes with the perturbed input, so the analogous fixed-label loss is a surrogate, not the true error.

#### 2. Robust optimization / adversarial training

Madry-style adversarial training uses a saddle-point objective:

\[
\min_\theta
\mathbb{E}_{(x,y)}
\left[
\max_{\|\delta\|_p\le \varepsilon}
\ell(f_\theta(x+\delta),y)
\right].
\]

The inner maximization uses PGD; the outer minimization trains the model to reduce the worst-case classification loss. Again, the label \(y\) is fixed.

Relation to this paper: the structure is very relevant, but the inner loss must be changed from fixed-label classification loss to solver-consistent regression loss:

\[
\max_{\|\delta\|_p\le\varepsilon}
\|F_\theta(a+\delta)-G(a+\delta)\|.
\]

#### 3. Margin / logit-based optimization attacks

Carlini-Wagner-type attacks use a confidence or margin loss instead of ordinary cross-entropy. A simplified targeted form is:

\[
\min_\delta
\|\delta\|_p
+ c\cdot
\max\left(
\max_{i\neq t} z_i(x+\delta)-z_t(x+\delta),
-\kappa
\right),
\]

where \(z_i\) are logits and \(t\) is the target class. The goal is to find a small perturbation that makes the target class win by a margin.

Relation to this paper: this is a minimum-distortion attack, not a fixed-budget maximum-loss attack. It still relies on classification logits and a fixed target class, so it does not provide the right oracle semantics for PDE regression.

#### 4. Minimum-distance decision-boundary attacks

DeepFool, Boundary Attack, and HopSkipJump are best understood as decision-boundary attacks. Instead of asking "what is the largest loss inside an \(\varepsilon\)-ball?", they ask "how small a perturbation is enough to cross the decision boundary?"

A generic version is:

\[
\min_\delta \|\delta\|_p
\quad
\text{s.t.}
\quad
\arg\max_i f_i(x+\delta)\neq y.
\]

DeepFool approximates the classifier locally and moves toward the nearest decision boundary. Boundary Attack and HopSkipJump are black-box/decision-based variants that can work with only class decisions.

Relation to this paper: this family is useful for attack methodology and robustness evaluation, but PDE regression does not have a natural discrete decision boundary unless one defines a threshold such as "relative \(L_2\) error exceeds \(\tau\)." This is why continuous-output PDE attacks require explicit success metrics.

#### 5. Saliency/sparse-feature attacks

JSMA and one-pixel attacks search for a small number of input features/pixels whose change causes misclassification. The loss is still classification-based, but the perturbation constraint is sparse:

\[
\|\delta\|_0 \le k.
\]

Relation to this paper: sparse or structured perturbation constraints can be useful for PDE inputs, but the objective still needs to be solver-consistent. Sparse perturbation alone does not solve the fixed-label problem.

#### 6. Black-box score or decision attacks

Square Attack, Boundary Attack, HopSkipJump, and substitute-model attacks avoid white-box gradients. They use score queries, class labels, random search, decision-boundary estimation, or transfer from a surrogate model.

Typical objective:

\[
\text{find }x+\delta
\text{ such that the classifier changes decision}
\]

or

\[
\max_{\|\delta\|\le\varepsilon}
\ell_{\mathrm{blackbox}}(f(x+\delta),y).
\]

Relation to this paper: the black-box idea is relevant if the solver is non-differentiable or too expensive to backpropagate through. But if only the neural operator is attacked while the solver response is ignored, the method again risks optimizing a surrogate rather than the true teacher-student error.

#### 7. Physical-world and patch attacks

Physical-world attacks and adversarial patches optimize perturbations that remain adversarial under transformations such as printing, camera capture, scaling, rotation, or placement. A typical patch objective is targeted:

\[
\max_{\mathrm{patch}}
\mathbb{E}_{T}
\log p_\theta(t\mid A(x,\mathrm{patch},T)),
\]

where \(T\) denotes random transformations and \(A\) applies the patch to the scene.

Relation to this paper: the useful lesson is that perturbation sets should reflect realistic constraints. For PDEs, this suggests smoothness, spectral, energy, boundary-condition, or physical-feasibility constraints. But the output reference should still be \(G(a+\delta)\), not the unperturbed \(G(a)\).

### Summary of the classical assumption

The classical adversarial-attack template is:

\[
\text{perturb the input while holding the semantic target fixed.}
\]

The solver-backed PDE regression template should be:

\[
\text{perturb the input and compare the model with the solver under that same perturbation.}
\]

This is exactly why the error operator

\[
E(a)=F_\theta(a)-G(a)
\]

is the right object. Robustness is not small movement of \(F_\theta\), and not small movement of \(G\). It is stability of the difference \(F_\theta-G\).

## Detailed Notes by Group

### 1. Physics-Informed Learning: PINNs, PI-DeepONet, PINO

These methods use the PDE itself as a training signal. The typical loss is

\[
\mathcal{L}
=\mathcal{L}_{data}
+\lambda_{pde}\mathcal{L}_{pde}
+\lambda_{bc}\mathcal{L}_{bc}
+\lambda_{ic}\mathcal{L}_{ic}.
\]

Here,

\[
\mathcal{L}_{pde}
=
\|\mathcal{P}(\hat u_\theta)\|^2
\]

is a physics residual. It is not the same as a solver-model error

\[
\|F_\theta(a)-G(a)\|.
\]

For time-dependent PDEs, this distinction matters. If the model only predicts one final frame, then a strong-form residual requiring \(\partial_t u\) is not directly available. Physics-informed methods usually need either a continuous coordinate model \(\hat u(x,t)\), multiple time frames, a one-step time integrator form, or a steady-state PDE such as Darcy flow.

### 2. Residual-Adversarial PINN/Operator Sampling: AT-PINN, PIAT, RAMS, StablePDENet

These methods are adversarial in the sense that they search for samples or perturbations that make a residual large. The residual is usually

\[
\|\mathcal{P}(F_\theta(a+\delta))\|^2,
\]

not

\[
\|F_\theta(a+\delta)-G(a+\delta)\|^2.
\]

This is the main limitation from the perspective of the current paper. A large PDE residual can be useful, but it is a surrogate. It does not necessarily imply large solver-model discrepancy. Conversely, a model can have solver-model error even if the residual surrogate is small, especially when boundary/initial constraints, discretization effects, non-unique residual minimizers, or weak-form/finite-difference approximations enter.

RAMS is best understood as a PDE-residual-driven sample-moving method:

\[
\delta_{\mathrm{RAMS}}
\propto
\nabla_{\mathrm{sample}}
\|\mathcal{P}(F_\theta(\mathrm{sample}))\|^2.
\]

Its PINN, physics-informed operator learning, and data-driven operator learning variants differ mainly in the object moved and how the model is trained afterward:

| RAMS setting | Moved object | Moving objective | Training after moving | Solver role |
|---|---|---|---|---|
| PINN | collocation points \((x,t)\) | PDE residual | residual/IC/BC loss | none |
| PI operator learning | input function and/or query points | PDE residual | physics residual/IC/BC/data loss | optional/offline |
| Data-driven operator learning | input functions | PDE residual acquisition | supervised data loss | solver labels selected inputs after acquisition |

Thus, even the data-driven RAMS mode uses physics residual for acquisition. The solver is used only after selection to label the new input-output pair.

### 3. Active Learning for Operator Learning: Winovich et al., MRA-FNO, Roy et al. Defense

Active-learning methods ask which PDE inputs should be labeled next because solver-generated data are expensive.

Winovich et al. select samples by predictive uncertainty. The operator predicts both mean and uncertainty, and high-uncertainty examples are sent to the solver for labeling. This is not adversarial and not a robustness metric. Evaluation is mostly clean prediction error, uncertainty calibration, and data efficiency.

MRA-FNO selects both input functions and resolutions using a utility-cost acquisition function. It is active/multi-fidelity learning, not adversarial attack.

Roy et al.'s defense paper is closer to robustness: it probes weaknesses using differential evolution attacks, generates training data at vulnerability locations, and adds denoising. However, the search is gradient-free and does not use solver backward. It is active robustness defense, not solver-integrated PGD with \(G(a+\delta)\) inside the differentiable attack objective.

### 4. Direct Adversarial Robustness for Neural Operators

Adesoji and Chen directly evaluate FNO robustness under norm-bounded input perturbations. The key evaluation quantity is prediction error against solver outputs. This is important because it establishes that FNOs are vulnerable under adversarial perturbations.

However, the method does not develop a solver-integrated training framework, and the solver is not treated as a differentiable component inside the attack gradient. It is closer to attack/evaluation than to the proposed solver-integrated attack/training framework.

Roy et al. attack neural-operator digital twins with sparse, physically plausible perturbations and gradient-free differential evolution. This work is important because it explicitly studies robustness and also introduces Jacobian-related vulnerability diagnostics such as effective perturbation dimension. Still, it is not solver-integrated training and does not define the three-metric framework used here.

### 5. GAN-Style "Adversarial" Neural Operators

GANO and Oommen et al.'s adv-NO use "adversarial" in the GAN/discriminator sense. Their goal is distribution matching, sharper turbulent structures, better energy spectra, or realistic function generation.

They do not solve the same problem as adversarial robustness:

\[
\text{GAN/adv-NO: generated distribution} \approx \text{data distribution}.
\]

The current paper studies:

\[
\text{worst-case perturbed input } a+\delta
\quad\text{and}\quad
F_\theta(a+\delta)-G(a+\delta).
\]

Therefore these papers should be cited carefully as generative/adversarial-loss neural-operator work, not as solver-consistent robustness or adversarial training baselines.

## What Existing Work Generally Does Not Do

Across the literature above, most papers do not jointly provide:

1. A neural-operator setting with continuous/regression PDE outputs.
2. A numerical solver \(G\) evaluated at the same perturbed input \(a+\delta\).
3. An adversarial objective based on the true perturbed solver-model error:

   \[
   L_3(\delta)=\|F_\theta(a+\delta)-G(a+\delta)\|.
   \]

4. A comparison against weaker surrogate objectives:

   \[
   L_1(\delta)=\|F_\theta(a+\delta)-F_\theta(a)\|,
   \]

   \[
   L_2(\delta)=\|F_\theta(a+\delta)-G(a)\|.
   \]

5. Solver forward use and, in the strongest setting, solver backward/gradient use during attack generation.
6. A robustness metric suite that includes clean generalization, adversarial loss increase, Jacobian spectral norm, and the Jacobian-error proxy \(J_E(a)^*e(a)\).

This is the paper's main opening.

More directly, existing work often treats one of the following as sufficient:

- low clean test loss;
- low PDE residual;
- high uncertainty on difficult examples;
- successful attack-induced degradation;
- improved accuracy after active sampling.

These are useful signals, but none of them alone defines solver-consistent robustness. In the proposed setting, the relevant question is not only whether the model is accurate on unseen data, or whether it satisfies a residual equation, but whether the model remains close to the numerical solver under the same local perturbation. This motivates the paper's separation between:

| Quantity | Meaning | Typical metric |
|---|---|---|
| Clean generalization | static out-of-sample teacher-student agreement | \( \|F_\theta(a)-G(a)\| \) |
| Model sensitivity | how much the neural operator changes | \( \|F_\theta(a+\delta)-F_\theta(a)\| \) |
| Physics consistency | whether the predicted field satisfies the PDE | \( \|\mathcal{P}(F_\theta(a+\delta))\| \) |
| Solver-consistent robustness | whether the neural operator remains close to the solver under the same perturbation | \( \|F_\theta(a+\delta)-G(a+\delta)\| \) |
| Local error amplification | infinitesimal sensitivity of the error operator | \( \|J_E(a)\| \), \( \|J_E(a)^*e(a)\| \) |

The paper should argue that the last two quantities are underdeveloped in prior work.

## How Prior Work Defines Robustness and Generalization

The literature uses several different notions of robustness. These notions are
often related, but they are not the same as the solver-consistent robustness
needed in this paper.

First, classical adversarial-learning papers usually define robustness through
an attacked loss or robust risk:

\[
\mathbb{E}_{(x,y)}
\max_{\|\delta\|\le \varepsilon}
\ell(f_\theta(x+\delta),y).
\]

This is natural for image classification because the label is assumed to stay
fixed under a small perturbation. In that setting, robustness is often reported
as robust accuracy, attack success rate, or the loss/accuracy after PGD,
FGSM, C&W, AutoAttack, or another attack. Generalization is then ordinary test
accuracy or test loss on clean unseen data. These works do distinguish clean
accuracy from robust accuracy, but the distinction is built around a
fixed-label assumption.

Second, neural-operator robustness papers often report clean prediction error
and attacked prediction error. Adesoji and Chen evaluate FNOs by perturbing PDE
inputs and measuring the MSE between the perturbed FNO output and solver output.
Digital-twin attack work similarly reports a sharp increase in relative
\(L_2\) error after sparse, physically plausible perturbations. This gives a
useful empirical robustness score:

\[
\frac{\|F_\theta(a+\delta)-G(a+\delta)\|}
{\|G(a+\delta)\|}.
\]

However, this is mainly an attack/evaluation metric. It does not by itself
define a local metric explaining which infinitesimal direction causes the
solver-model error to increase.

Third, stability-oriented operator papers use local sensitivity. StablePDENet
is the clearest example: it links stability to bounded Frechet derivatives and
reports Jacobian spectral norms as a diagnostic. In our notation, this is close
to

\[
\|D F_\theta[a]\|
\quad\text{or}\quad
\|D E_\theta[a]\|.
\]

This measures the maximum possible local amplification of output or error
changes. It is a meaningful stability measure, but it maximizes the change of
the error field, not necessarily the final error norm starting from the current
error \(e_\theta(a)=F_\theta(a)-G(a)\).

Fourth, PINN and residual-adversarial papers define failure or robustness
through physics residuals. AT-PINN/WbAR, PIAT, StablePDENet, and RAMS use
objectives of the form

\[
\|\mathcal{P}(F_\theta(a+\delta))\|
\]

or the corresponding residual at collocation points. This identifies regions
where the predicted solution violates the governing equation, but the residual
is a surrogate for the solver-model discrepancy. It is not the same as
\(\|F_\theta(a+\delta)-G(a+\delta)\|\).

The missing metric in these lines of work is the local gradient of the
solver-consistent error operator:

\[
E_\theta = F_\theta-G,
\qquad
e_\theta(a)=E_\theta(a),
\qquad
D E_\theta[a]^*e_\theta(a).
\]

For the squared solver-model loss

\[
\Phi(a)=\frac12\|E_\theta(a)\|^2,
\]

the first-order variation is

\[
D\Phi[a](\eta)
=
\langle D E_\theta[a](\eta),e_\theta(a)\rangle
=
\langle \eta,D E_\theta[a]^*e_\theta(a)\rangle.
\]

Thus, for small perturbation budgets,

\[
\sup_{\|\eta\|\le\varepsilon}
\left(\Phi(a+\eta)-\Phi(a)\right)
\approx
\varepsilon\|D E_\theta[a]^*e_\theta(a)\|_*.
\]

This quantity is different from a Jacobian spectral norm. The spectral norm
asks for the direction that maximizes the change in the error field
\(D E_\theta[a](\eta)\). The Jacobian-error metric asks for the direction that
maximizes the final solver-model error increase from the current error state.
Based on the papers reviewed here, existing work has used attacked loss,
relative error under attack, PDE residuals, and Jacobian/spectral-norm
stability, but has not centered this \(D E_\theta[a]^*e_\theta(a)\) metric as
the robustness diagnostic.

## Suggested Related-Work Paragraphs

The following paragraphs are written in a paper-ready style. They are intentionally argumentative: each paragraph states what prior work has done and then identifies what remains missing.

### Direct adversarial robustness and stability for neural operators

Prior work has begun to examine the vulnerability of neural operators to adversarial perturbations. Adesoji and Chen evaluate the adversarial robustness of Fourier neural operators by perturbing PDE inputs and measuring degradation relative to solver-generated outputs. StablePDENet formulates operator learning as a stability-aware min-max problem and trains against worst-case perturbations, but its adversarial signal is based on a physics/stability residual rather than a perturbed solver-model discrepancy. Recent digital-twin work further shows that sparse, physically plausible perturbations can induce large neural-operator failures, and introduces sensitivity diagnostics for architecture-dependent vulnerability. These works motivate robustness evaluation for operator learning, but they do not fully integrate the numerical solver into the adversarial objective as \(G(a+\delta)\), nor do they use solver gradients to construct perturbations.

**Longer version.** Existing robustness studies show that neural operators can fail under adversarial or physically plausible perturbations, but they often stop at evaluation or use surrogate notions of stability. For example, FNO robustness evaluation measures how prediction error degrades under perturbed inputs, while digital-twin attacks reveal severe sparse-input vulnerabilities. These works are important because they demonstrate that clean validation error is insufficient. However, they do not provide a unified definition that separates clean generalization, local error amplification, and worst-case solver-consistent robustness. In particular, they generally do not compare the true perturbed teacher-student error \( \|F_\theta(a+\delta)-G(a+\delta)\| \) against surrogate objectives such as model-only output change, fixed-label error, or physics residuals. Our work fills this gap by defining robustness directly through the error operator \(E(a)=F_\theta(a)-G(a)\), and by studying adversarial loss increase together with Jacobian-based local metrics.

### Physics-residual adversarial sampling

A related line of work uses PDE residuals to identify failure regions or informative samples. AT-PINN/WbAR, PIAT, StablePDENet, and RAMS generate adversarial or adaptive samples by maximizing the residual of the model output under the governing PDE. RAMS extends this idea from PINN collocation points to operator-learning input functions, and in its data-driven setting uses the solver only after residual-based selection to label new input functions. These methods are useful residual-driven sampling baselines, but their acquisition objective is a physics residual surrogate. In contrast, our adversarial objective directly maximizes the solver-consistent error under the same perturbed input.

**Longer version.** Physics-informed adversarial methods typically search for samples where the learned solution violates the governing PDE. This idea is natural for PINNs and physics-informed operator learning because the PDE residual is differentiable with respect to coordinates or input functions. AT-PINN/WbAR and PIAT use this idea in PINN settings, while RAMS extends residual-gradient moving samples to both PINNs and operator learning. Nevertheless, the residual objective is not the same as solver-model discrepancy. A perturbation that maximizes \( \|\mathcal{P}(F_\theta(a+\delta))\| \) does not necessarily maximize \( \|F_\theta(a+\delta)-G(a+\delta)\| \). This distinction is especially important in data-driven operator learning, where the final training target is the solver-generated solution, but the acquisition objective may still be a PDE residual. Our work treats such residual-based methods as surrogate baselines and asks whether a solver-integrated objective produces more effective adversarial examples and more robust training.

### Active learning and uncertainty-driven operator learning

Active operator learning methods reduce the cost of solver-generated datasets by selecting which PDE instances to label. Winovich et al. use predictive uncertainty to query high-uncertainty examples for additional solver evaluation, while MRA-FNO selects both input functions and simulation resolutions through a utility-cost acquisition rule. These methods improve data efficiency and clean generalization, but they do not define worst-case perturbations or adversarial robustness. Their solver use is primarily label generation after acquisition, not differentiable solver integration inside the acquisition gradient.

**Longer version.** Active learning for neural operators recognizes that solver-generated datasets are expensive and seeks to select informative PDE instances for labeling. Uncertainty-driven active operator learning selects examples where the model predicts high uncertainty, while multi-resolution active learning selects input functions and resolutions according to utility-cost tradeoffs. These approaches are valuable for improving clean generalization per solver query, but their acquisition rules are not adversarial robustness objectives. They do not ask for the worst perturbation around a given input, and the solver is typically used only after acquisition to produce a label. In contrast, our setting treats the solver as a teacher during adversarial sample construction itself: the relevant failure mode is not high uncertainty alone, but the largest local discrepancy between the neural operator and the solver under the same perturbed input.

### Generative adversarial neural operators

Generative adversarial neural operators and turbulent-flow adv-NO models use adversarial discriminator losses to match function distributions or recover sharp high-frequency turbulent structures. This adversarial formulation is orthogonal to worst-case robustness: the discriminator shapes generated output distributions, whereas our setting studies norm-bounded perturbations of PDE inputs and the resulting solver-model discrepancy.

**Longer version.** Another line of work uses adversarial losses in the generative sense. GANO learns distributions over functions through a generator neural operator and discriminator functional, and adv-NO-style turbulent-flow models use discriminator losses to preserve sharp gradients and spectral structure. These methods are useful references because they show that adversarial objectives can improve operator-learning outputs. However, their adversarial signal is distributional rather than robustness-oriented. They do not generate norm-bounded worst-case input perturbations, do not compare \(F_\theta(a+\delta)\) with \(G(a+\delta)\), and do not study solver-integrated adversarial training.

### Classical adversarial attacks and fixed-label classification

Classical adversarial-example work was developed mainly for classification, where the true semantic label is assumed to remain unchanged under a small perturbation. FGSM, PGD adversarial training, C&W attacks, DeepFool, Boundary Attack, AutoAttack, and related black-box attacks all fit this broad paradigm: the adversary perturbs an input to change or degrade the classifier's prediction while the ground-truth label is treated as fixed. This assumption is natural for images, where a visually small perturbation should not turn a dog into a car. PDE regression is different. Perturbing an initial condition, coefficient field, or boundary condition changes the underlying physical problem, and therefore changes the correct output. A fixed-target attack \( \|F_\theta(a+\delta)-G(a)\| \) can be appropriate as a control, but it is not the true perturbed-input regression error. Our framework replaces the classification fixed-label assumption with a solver-consistent oracle assumption: the reference output must be \(G(a+\delta)\), not \(G(a)\).

### Solver-integrated teacher-student view

The central distinction of our work is that the numerical solver is not merely a data generator. It is the teacher whose behavior the neural operator should approximate under both clean and perturbed inputs. This changes the natural definitions of both generalization and robustness. Clean generalization is the static teacher-student error on unseen inputs, whereas robustness is the worst-case teacher-student error in a local perturbation set. Prior work often improves one side of this picture without completing the other: active learning improves data efficiency, residual-based adversarial sampling finds physics-violating inputs, and robustness studies expose vulnerabilities. Our framework combines these goals by using solver-consistent adversarial attacks and training, comparing surrogate losses against the true perturbed solver-model discrepancy, and introducing local Jacobian-error metrics that explain which perturbation directions increase the final error.

## One-Sentence Positioning

Existing work either evaluates neural-operator attacks, performs residual-based adversarial sampling, uses active learning to reduce solver queries, or applies GAN-style adversarial losses. The proposed work is different because it treats the numerical solver as a teacher, defines robustness through the perturbed teacher-student discrepancy \( \|F_\theta(a+\delta)-G(a+\delta)\| \), compares this true objective with surrogate losses, and studies both adversarial training and local Jacobian-based robustness metrics in the same solver-integrated framework.

## Strong Claim to Preserve

The paper's most important claim is not merely "we use adversarial training." It is:

> In PDE neural-operator learning, robustness should be defined relative to the numerical solver under the same perturbed input. Surrogate objectives such as model-only sensitivity, fixed-label error, physics residuals, or uncertainty can select difficult samples, but they do not necessarily maximize the true teacher-student discrepancy.

This claim is what separates the paper from active learning, residual-based sampling, and generic adversarial robustness evaluation.

## Source Links

- FNO: https://arxiv.org/abs/2010.08895
- DeepONet: https://arxiv.org/abs/1910.03193
- PI-DeepONet: https://arxiv.org/abs/2103.10974
- PINO: https://arxiv.org/abs/2111.03794
- PI-DeepONet / physics-informed operator review: https://arxiv.org/abs/2207.05748
- Adesoji and Chen, FNO robustness: https://arxiv.org/abs/2204.04259
- PIAT: https://arxiv.org/abs/2207.06647
- AT-PINN / WbAR: https://arxiv.org/abs/2310.11789
- RAMS: https://arxiv.org/abs/2509.01234
- StablePDENet: https://arxiv.org/abs/2601.06472
- Active operator learning with predictive UQ: https://arxiv.org/abs/2503.03178
- MRA-FNO: https://arxiv.org/abs/2309.16971
- Neural-operator digital twin attacks: https://arxiv.org/abs/2603.22525
- Active learning + denoising defense: https://arxiv.org/abs/2604.13316
- GANO: https://arxiv.org/abs/2205.03017
- Oommen et al. adv-NO / turbulent generative models: https://arxiv.org/abs/2509.08752
- Adversarial autoencoders in operator learning: https://arxiv.org/abs/2412.07811
- General adversarial examples: https://arxiv.org/abs/1412.6572
- Madry adversarial training: https://arxiv.org/abs/1706.06083
- TRADES: https://arxiv.org/abs/1901.08573
- AutoAttack: https://arxiv.org/abs/2003.01690
