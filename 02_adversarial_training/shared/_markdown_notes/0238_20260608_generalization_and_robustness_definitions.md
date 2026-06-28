# Generalization And Robustness Definitions - 2026-06-08

## Purpose

This note records the working definitions used in the Burgers and Darcy Flow robustness experiments. The key point is that **generalization** and **robustness** are not the same diagnostic.

Generalization is measured by clean, static prediction error on a chosen data distribution. Robustness is measured by sensitivity to input perturbations. In this project we use two complementary robustness views: finite-budget adversarial robustness and infinitesimal local Jacobian robustness.

## 1. Clean Generalization

For an operator model \(F_\theta\) and reference solver \(G\), clean generalization measures the ordinary prediction error without adversarial perturbation:

\[
L_{\mathrm{clean}}(x)=\|F_\theta(x)-G(x)\|.
\]

Depending on the experiment, this clean loss is reported as RMSE, relative L2, MAE, or a related absolute error metric.

Interpretation:

- It answers whether the model predicts accurately on train/test/generalization samples without attack.
- It is a static absolute loss value on a dataset.
- It can be averaged over train, test, selected generalization datasets, or any proposed out-of-distribution suite.

In the local Burgers and Darcy Flow analysis, this is the metric used for statements like:

- Darcy Flow loss3 self-training lowers selected-generalization clean RMSE/relative-L2.
- Burgers loss1/loss2/loss3 self-trained models have lower pre-attack loss than the baseline model on the 52-dataset p2q2 screen.

## 2. Finite-Budget Adversarial Robustness

Finite-budget robustness measures how large the model-solver error can become inside a constrained perturbation set:

\[
R_{p,\epsilon}(x)=\max_{\|\delta\|_p\le \epsilon}\|F_\theta(x+\delta)-G(x+\delta)\|.
\]

In practice this maximum is approximated by a finite-step adversarial attack, such as the p2q2 attack used in the Burgers experiments.

Useful reported quantities include:

- clean/pre-attack loss \(\|F_\theta(x)-G(x)\|\);
- attacked final loss \(\|F_\theta(x+\delta_{\mathrm{final}})-G(x+\delta_{\mathrm{final}})\|\);
- attack loss increase from clean to final;
- attacked final loss compared against baseline under the same attack budget;
- final perturbation spectrum, such as low/mid/high FFT fractions and spectral centroid.

This is a finite-radius local diagnostic, not an infinitesimal one. The perturbation is limited to a ball, but \(\epsilon\) is not taken to zero. Therefore finite-budget attack results can include nonlinear effects, solver/model response changes, and frequency shifts in the optimized perturbation.

In the Burgers 52-dataset p2q2 20-step screen, this definition supports the conclusion:

- self-trained loss1/loss2/loss3 models all have much lower final attacked loss than baseline on `52/52` datasets;
- loss3 has the lowest final attacked loss and the strongest high-frequency final delta;
- loss1/loss2 are also robust by final attacked loss, but their final deltas mainly shift into the mid-frequency band rather than the highest band.

## 3. Infinitesimal Local Jacobian Robustness

Jacobian robustness measures first-order sensitivity at a point. For the model-solver error map

\[
e_\theta(x)=F_\theta(x)-G(x),
\]

its local linearization is

\[
e_\theta(x+\delta)\approx e_\theta(x)+J_{e_\theta}(x)\delta.
\]

The infinitesimal worst-case amplification is controlled by the spectral norm:

\[
\|J_{e_\theta}(x)\|_2.
\]

The associated local first-order approximation is

\[
\max_{\|\delta\|_2\le\epsilon}\|J_{e_\theta}(x)\delta\|_2 = \epsilon\|J_{e_\theta}(x)\|_2.
\]

This is the true local, infinitesimal robustness metric. It only describes the behavior near \(x\) to first order.

In the local Burgers SVD work, we also separately inspect:

\[
J_{F_\theta}(x),\qquad J_G(x),\qquad J_{F_\theta}(x)-J_G(x).
\]

The error Jacobian \(J_{F_\theta}(x)-J_G(x)\) is especially relevant for model-solver robustness because it describes first-order sensitivity of the prediction error, not merely sensitivity of the model output itself.

## Relationship Between The Three Metrics

The three diagnostics answer different questions:

| Diagnostic | Object | Question answered | Locality |
|---|---|---|---|
| Clean generalization | \(\|F_\theta(x)-G(x)\|\) | Is the model accurate on this dataset without attack? | Static dataset metric |
| Finite-budget adversarial robustness | \(\max_{\|\delta\|_p\le\epsilon}\|F_\theta(x+\delta)-G(x+\delta)\|\) | How bad can the error become inside a fixed perturbation budget? | Finite-radius local |
| Jacobian spectral robustness | \(\|J_{F_\theta-G}(x)\|_2\) | How much can infinitesimal perturbations be amplified to first order? | Infinitesimal local |

Therefore, a model can have:

- good clean generalization but poor robustness if small or moderate perturbations sharply increase error;
- good finite-budget robustness even if the final adversarial perturbation has a different frequency structure;
- a small local Jacobian spectral norm near clean samples while still being vulnerable farther away inside a larger \(p,\epsilon\)-ball if nonlinear effects become important.

## Working Interpretation For Current Results

The current Burgers/Darcy experiments should be described with this language:

- **Generalization improvement** means clean absolute error decreases on a chosen evaluation distribution.
- **Finite-budget robustness improvement** means, under the same attack budget and attack protocol, final attacked loss is lower than the baseline or lower after self-training.
- **Jacobian robustness improvement** means the relevant local Jacobian, especially the model-solver error Jacobian, has smaller top singular values or more favorable singular-vector structure.

The finite-budget p2q2 attack and the Jacobian spectral norm are related but not equivalent. The Jacobian is a first-order infinitesimal proxy; the p2q2 attack is a constrained finite-radius stress test. Agreement between them is strong evidence, but disagreement is not automatically a contradiction because the finite-radius attack can leave the strictly linear regime.

## Practical Reporting Rule

When reporting future results, avoid saying only that a model is “better” without specifying the diagnostic. Use explicit phrases:

- “clean generalization RMSE decreased”;
- “final p2q2 attacked loss decreased under the same budget”;
- “top singular value of the local error Jacobian decreased”;
- “attack delta shifted toward higher FFT modes”;
- “frequency shift and robustness improvement are correlated/not correlated in this run.”

This prevents confusing clean accuracy, finite-budget robustness, and infinitesimal Jacobian stability.
