# Gaussian Processes, High-Dimensional Bayesian Optimization, Matérn Kernels, and Review Records

> Discussion notes consolidated on 2026-10-06.
>
> Scope: Gaussian-process basics relevant to the discussion; model misspecification; Matérn/RBF Fourier structure; Student-t processes; Imprecise Bayesian Optimization; and the recent high-dimensional BO line involving Hvarfner et al. (ICML 2024), Xu et al. (ICLR 2025), and Papenmeier et al. (ICML 2025), including reviewer comments and rebuttal logic.
>
> Important distinction used throughout:
>
> - A result can be empirically important without being a theorem.
> - A paper can share a high-level thesis with concurrent work while still containing an independent mechanism, theorem, or diagnostic analysis.
> - “Works well in high dimensions” is not the same claim as “general high-dimensional global optimization is easy.”

---

# 1. Gaussian-process setup and the role of the kernel

A Gaussian process is written as

\[
f \sim \operatorname{GP}(m,k),
\]

where

\[
m(x)=\mathbb E[f(x)]
\]

is the mean function and

\[
k(x,x')=\operatorname{Cov}(f(x),f(x'))
\]

is the covariance kernel.

For a finite input set

\[
X=\{x_1,\ldots,x_N\},
\]

the random vector

\[
f_X=
\begin{bmatrix}
f(x_1)\\
\vdots\\
f(x_N)
\end{bmatrix}
\]

is multivariate Gaussian:

\[
f_X\sim
\mathcal N(m_X,K_{XX}),
\]

with

\[
(m_X)_i=m(x_i),\qquad
(K_{XX})_{ij}=k(x_i,x_j).
\]

For noiseless observations \(y=f_X\), prediction at a new point \(x_*\) is

\[
f_*\mid y
\sim
\mathcal N(\mu_*,v_*),
\]

where

\[
\mu_*
=
m(x_*)
+
K_{*X}K_{XX}^{-1}(y-m_X),
\]

and

\[
v_*
=
K_{**}
-
K_{*X}K_{XX}^{-1}K_{X*}.
\]

The GP posterior mean and posterior covariance function are therefore

\[
m_{\rm post}(x)
=
m(x)
+
k(x,X)K^{-1}(y-m_X),
\]

\[
k_{\rm post}(x,x')
=
k(x,x')
-
k(x,X)K^{-1}k(X,x').
\]

The key point for Bayesian optimization is that both the prior mean and the kernel influence not only prediction but also the next evaluation point selected by the acquisition function.

---

# 2. Stationary kernels, nonstationarity, and model misspecification

For a stationary kernel,

\[
k(x,x')=C(x-x').
\]

For an isotropic stationary kernel,

\[
k(x,x')=\psi(\|x-x'\|).
\]

This means correlation depends only on displacement, or only on distance in the isotropic case.

Examples of misspecification discussed:

## 2.1 Different smoothness in different regions

A standard RBF kernel uses a global length scale:

\[
k_{\rm RBF}(x,x')
=
\sigma_f^2
\exp\left(
-\frac{\|x-x'\|^2}{2\ell^2}
\right).
\]

If the real function is very smooth in one region and rough in another, a single global \(\ell\) must compromise. A nonstationary model may instead need something like

\[
\ell=\ell(x).
\]

## 2.2 Change points

A change point is a location \(c\) at which the statistical behavior changes sharply.

For example,

\[
f(x)
=
\begin{cases}
f_1(x), & x<c,\\
f_2(x), & x\ge c.
\end{cases}
\]

The function value itself does not have to jump. The change can be in:

- the mean;
- the slope;
- the variance;
- the length scale;
- the noise level;
- the local smoothness.

For instance,

\[
\ell(x)
=
\begin{cases}
5, & x<c,\\
0.2, & x\ge c.
\end{cases}
\]

is a covariance/smoothness change point.

A change point can therefore be viewed as a particularly abrupt form of nonstationarity.

## 2.3 Signal variance versus noise variance

The RBF prefactor

\[
\sigma_f^2
\]

is the signal/process variance. It controls the amplitude of function fluctuations.

Observation noise is usually modeled separately:

\[
y_i=f(x_i)+\epsilon_i,
\]

\[
\epsilon_i\sim\mathcal N(0,\sigma_n^2).
\]

If noise variance varies with location,

\[
\epsilon(x)\sim
\mathcal N(0,\sigma_n^2(x)),
\]

the noise is heteroscedastic.

Thus

\[
\sigma_f^2
\]

and

\[
\sigma_n^2
\]

must not be confused.

## 2.4 Heavy-tailed noise

Gaussian noise assumes light tails:

\[
\epsilon\sim\mathcal N(0,\sigma_n^2).
\]

If large outliers occur much more often than a Gaussian model allows, one can instead use a heavy-tailed likelihood, e.g.

\[
\epsilon\sim t_\nu.
\]

The latent function may still be a GP:

\[
f\sim \operatorname{GP}(m,k),
\]

while the observation likelihood is Student-t. In that case the usual closed-form Gaussian posterior generally disappears and approximate inference is needed.

## 2.5 Prior mean misspecification

For

\[
f\sim \operatorname{GP}(m,k),
\]

the prior mean determines what the model predicts in regions far from observed data.

If

\[
k(x,X)\approx 0,
\]

then

\[
\mu_t(x)\approx m(x).
\]

This becomes especially important in Bayesian optimization because unexplored regions are exactly where the next decision may be made.

---

# 3. RBF and Matérn kernels in space and frequency

## 3.1 RBF / squared-exponential kernel

The squared-exponential kernel, abbreviated SE and also commonly called RBF or Gaussian kernel, is

\[
k_{\rm SE}(r)
=
\sigma_f^2
\exp\left(
-\frac{r^2}{2\ell^2}
\right),
\qquad
r=\|x-x'\|.
\]

Its spectral density is also Gaussian:

\[
S_{\rm SE}(\omega)
\propto
\exp\left(
-\frac{\ell^2\|\omega\|^2}{2}
\right).
\]

Thus Gaussian covariance is a special Fourier pair:

\[
e^{-ar^2}
\quad
\longleftrightarrow
\quad
e^{-b\omega^2}.
\]

A larger spatial length scale corresponds to a narrower frequency spectrum.

## 3.2 General Matérn kernel

The Matérn kernel is

\[
k_\nu(r)
=
\sigma_f^2
\frac{2^{1-\nu}}{\Gamma(\nu)}
\left(
\frac{\sqrt{2\nu}\,r}{\ell}
\right)^\nu
K_\nu
\left(
\frac{\sqrt{2\nu}\,r}{\ell}
\right),
\]

where \(K_\nu\) is the modified Bessel function of the second kind.

Let

\[
a=\frac{\sqrt{2\nu}}{\ell}.
\]

Then schematically,

\[
k_\nu(r)\propto (ar)^\nu K_\nu(ar).
\]

Its spectral density has the form

\[
S_\nu(\omega)
\propto
\left(
a^2+\|\omega\|^2
\right)^{-(\nu+d/2)}.
\]

At high frequency,

\[
S_\nu(\omega)
\sim
\|\omega\|^{-(2\nu+d)}.
\]

This is the power-law tail associated with Matérn kernels.

The Fourier pair is therefore

\[
\left(a^2+\|\omega\|^2\right)^{-\alpha}
\quad
\longleftrightarrow
\quad
r^{\alpha-d/2}K_{\alpha-d/2}(ar).
\]

Setting

\[
\alpha=\nu+\frac d2
\]

gives the Matérn form.

## 3.3 Half-integer Matérn kernels

For

\[
\nu=p+\frac12,
\]

the Bessel function simplifies and the covariance becomes

\[
\text{polynomial}(r)\times e^{-ar}.
\]

Examples:

\[
\nu=\frac12:
\qquad
k(r)=\sigma_f^2e^{-r/\ell}.
\]

\[
\nu=\frac32:
\qquad
k(r)=
\sigma_f^2
\left(
1+\frac{\sqrt3r}{\ell}
\right)
e^{-\sqrt3r/\ell}.
\]

\[
\nu=\frac52:
\qquad
k(r)
=
\sigma_f^2
\left(
1+\frac{\sqrt5r}{\ell}
+\frac{5r^2}{3\ell^2}
\right)
e^{-\sqrt5r/\ell}.
\]

Thus the important distinction is:

\[
\boxed{
\text{RBF spatial decay: Gaussian } e^{-cr^2}
}
\]

\[
\boxed{
\text{Matérn spatial decay: polynomial}\times e^{-cr}
}
\]

while in frequency,

\[
\boxed{
\text{RBF tail: Gaussian}
}
\]

and

\[
\boxed{
\text{Matérn tail: power law}.
}
\]

This is why Matérn processes retain more high-frequency content and are rougher than RBF GP sample functions.

As

\[
\nu\to\infty,
\]

the Matérn family approaches the squared-exponential kernel.

---

# 4. Student-t processes versus Gaussian processes

The paper discussed was:

Amar Shah, Andrew Gordon Wilson, Zoubin Ghahramani,
“Student-t Processes as Alternatives to Gaussian Processes,” AISTATS 2014.

The paper did not invent Student-t processes. Its contribution was to systematize their construction, tractability, predictive behavior, covariance uncertainty interpretation, and empirical use.

## 4.1 Multivariate Student-t parameterization used in the paper

The paper writes

\[
y\sim \operatorname{MVT}_n(\nu,\phi,K)
\]

with density

\[
p(y)
=
\frac{
\Gamma((\nu+n)/2)
}{
\Gamma(\nu/2)
[(\nu-2)\pi]^{n/2}
|K|^{1/2}
}
\left[
1+
\frac{
(y-\phi)^T K^{-1}(y-\phi)
}{
\nu-2
}
\right]^{-(\nu+n)/2}.
\]

Under this parameterization,

\[
\mathbb E[y]=\phi,
\qquad
\operatorname{Cov}(y)=K
\]

for \(\nu>2\).

This differs from the more common scale-matrix parameterization, where

\[
\operatorname{Cov}(y)
=
\frac{\nu}{\nu-2}\Sigma.
\]

Hence

\[
\Sigma=\frac{\nu-2}{\nu}K.
\]

## 4.2 TP hierarchy

The finite covariance matrix can be given an inverse-Wishart prior:

\[
C_X\sim IW_n(\nu,K_\theta).
\]

Conditionally,

\[
f\mid C
\sim
\operatorname{GP}\bigl(\phi,(\nu-2)C\bigr).
\]

Marginalizing covariance uncertainty yields a Student-t process:

\[
f\sim \operatorname{TP}(\nu,\Phi,k_\theta).
\]

The base kernel \(k_\theta\) can be RBF, Matérn, nonstationary, etc.; Student-t processes do not require a special covariance family.

## 4.3 Observation-dependent predictive covariance

For a partitioned multivariate t distribution,

\[
\beta_1
=
(y_1-\phi_1)^T K_{11}^{-1}(y_1-\phi_1)
\]

appears in the conditional covariance:

\[
y_2\mid y_1
\sim
\operatorname{MVT}
\left(
\nu+n_1,
\tilde\phi_2,
\frac{\nu+\beta_1-2}{\nu+n_1-2}\tilde K_{22}
\right).
\]

The conditional mean has the same linear form as a GP with the same fixed kernel, but the predictive covariance depends on the realized observations through \(\beta_1\).

This differs from a fixed-hyperparameter Gaussian conditional covariance, which depends only on input locations and the kernel.

---

# 5. Imprecise Bayesian Optimization (Rodemann & Augustin, 2024)

Paper:

Julian Rodemann and Thomas Augustin,
“Imprecise Bayesian Optimization,”
Knowledge-Based Systems, 2024.

The central question is:

\[
\boxed{
\text{How sensitive is BO to subjective GP prior specification?}
}
\]

The authors separate four GP-prior components:

1. mean functional form;
2. mean parameters;
3. kernel functional form;
4. kernel parameters.

Examples:

- mean form: constant, linear, quadratic;
- mean parameter: constant level, intercept, slope;
- kernel family: Gaussian/RBF, Matérn, power exponential;
- kernel parameters: signal variance, length scale, etc.

The motivation is obvious in black-box optimization: if the function is truly black-box, why should the user know the correct prior mean, kernel family, or parameter values?

---

# 6. Imprecise BO: empirical sensitivity screening

The first major stage is empirical, not theoretical.

The authors use 50 synthetic benchmark functions from the R package smoof, spanning dimensions

\[
1,2,3,4,7.
\]

For each configuration:

- initial design size:
  \[
  n_{\rm init}=10;
  \]
- independent repetitions:
  \[
  R=40;
  \]
- BO iterations after initialization:
  \[
  T=20.
  \]

They vary the four prior components and observe how much the optimization trajectory changes.

## 6.1 Mean Optimization Path

They define a mean optimization path, \(MOP_t\), based on the best function value found up to BO step \(t\), averaged across repeated runs.

They then summarize sensitivity through an accumulated difference measure that captures how far optimization paths separate when a given prior component is varied.

The aggregated relative effects reported in the discussion were:

| GP prior component | Overall relative influence |
|---|---:|
| Mean functional form | 42.49 |
| Kernel functional form | 68.20 |
| Mean parameters | 77.91 |
| Kernel parameters | 11.40 |

Thus, under their chosen benchmarks and perturbation protocol,

\[
\boxed{
\text{mean parameters were the most sensitive component}
}
\]

and

\[
\boxed{
\text{kernel family was second}.
}
\]

The authors explicitly warn that these numbers are not universal constants. In particular:

- the perturbation magnitude is subjective;
- changing a parameter by a small amount is not directly comparable with changing a functional family;
- 50 synthetic functions do not span all real-world objectives;
- interactions among prior components are not exhaustively characterized.

So the empirical conclusion is correctly read as:

\[
\boxed{
\text{“mean parameters were especially influential in this systematic screen”}
}
\]

rather than:

\[
\boxed{
\text{“mean parameters are universally more important than kernels.”}
}
\]

---

# 7. Imprecise BO: why prior mean matters so much in optimization

The GP posterior mean is

\[
\mu_t(x)
=
m(x)
+
k_t(x)^T
(K_t+\sigma_n^2I)^{-1}
(y_t-m_t).
\]

Far from observed data,

\[
k_t(x)\approx 0,
\]

so

\[
\mu_t(x)\approx m(x).
\]

Thus the prior mean directly controls how unexplored regions look.

For minimization with lower confidence bound,

\[
LCB(x)
=
\mu_t(x)-\tau_t\sigma_t(x),
\]

the mean term affects exploitation and the variance term promotes uncertainty-seeking exploration.

A high prior mean can make unseen regions look poor and push the method toward local exploitation.

A low prior mean can make unseen regions appear attractive even before observing them.

Therefore prior mean is not merely a regression parameter. It acts as an implicit search-policy parameter.

This leads to the important distinction:

\[
\boxed{
\text{best prior for global function estimation}
\neq
\text{best prior for quickly finding the optimum}.
}
\]

A biased prior can, in some problems, accelerate optimization by inducing a useful search bias.

This does not imply that “wrong priors are good.” The direction of helpful bias is generally unknown in advance and can also hide the true optimum.

---

# 8. Imprecise BO: theoretical stage

After the empirical screen identifies prior mean as especially sensitive, the paper focuses its theory on prior-mean misspecification.

The theoretical stage does not prove that mean parameters are universally the most important prior component. It answers a different question:

\[
\boxed{
\text{If the prior mean is misspecified, how badly can BO regret guarantees degrade?}
}
\]

A nonzero prior mean modifies the posterior relative to the zero-mean GP through an error term of the form

\[
\epsilon_T(x)
=
m(x)
-
k_T(x)^T
(K_T+\sigma_n^2I)^{-1}m_T.
\]

In the regret analysis, accumulated mean misspecification can introduce an extra term.

If the error remains \(O(1)\) at repeatedly sampled points, its cumulative contribution can be

\[
O(T),
\]

destroying the usual sublinear-regret guarantee.

However, if misspecification is controlled by GP posterior uncertainty, schematically

\[
|\epsilon_t(x)|
\lesssim
\sigma_t(x),
\]

then the error can be absorbed into the confidence-width term and sublinear-order guarantees can be recovered.

This motivates distinguishing:

\[
\boxed{
\text{small error relative to admitted uncertainty}
}
\]

from

\[
\boxed{
\text{large bias while the model is overconfident}.
}
\]

The latter is the dangerous regime.

---

# 9. Imprecise BO: PROBO and GLCB

The authors then avoid forcing one exact prior mean.

Instead of a single GP prior

\[
\operatorname{GP}(m,k),
\]

they consider a set of plausible priors and derive lower/upper posterior mean envelopes:

\[
\underline\mu(x),
\qquad
\overline\mu(x).
\]

The width

\[
\overline\mu(x)-\underline\mu(x)
\]

measures model-specification imprecision due to uncertainty about the prior mean.

The generalized lower confidence bound adds this as a third term:

\[
GLCB(x)
=
\hat\mu(x)
-
\tau_t\hat\sigma(x)
-
\rho
\left[
\overline\mu(x)-\underline\mu(x)
\right].
\]

Conceptually:

\[
\boxed{
\text{predicted objective}
+
\text{data uncertainty}
+
\text{model-specification uncertainty}.
}
\]

The resulting approach is PROBO: Prior-mean-RObust Bayesian Optimization.

The final experiments include a graphene-production application.

Important methodological reading:

\[
\boxed{
\text{empirical screening}
\to
\text{theory for the empirically sensitive component}
\to
\text{method design}
\to
\text{new experiments}.
}
\]

That is the actual logic of the paper.

---

# 10. Public-review availability for Imprecise Bayesian Optimization

Unlike ICLR/ICML OpenReview papers, this Knowledge-Based Systems journal article does not have public reviewer reports available in the sources checked.

What is publicly visible is that the paper underwent revision and thanks three anonymous reviewers.

Therefore it is not justified to attribute a specific concern—such as “the four-way ranking is purely empirical”—to Reviewer 1/2/3 unless an actual report is obtained.

The final paper itself already acknowledges several limitations of the sensitivity study:

- benchmark coverage is finite;
- real-world coverage is limited;
- perturbation magnitude is subjective;
- parameter-vs-functional-form comparisons should be interpreted cautiously;
- interaction effects are not exhaustively analyzed.

---

# 11. Why high-dimensional BO can fail with an SE/RBF kernel

Suppose

\[
x,x'
\sim U([0,1]^d).
\]

For one coordinate,

\[
\mathbb E[(x_j-x_j')^2]=\frac16.
\]

Therefore,

\[
\|x-x'\|^2
\approx
\frac d6,
\]

and

\[
\|x-x'\|
\sim
\sqrt d.
\]

For fixed length scale \(\ell_0\),

\[
\rho^2
=
\frac{\|x-x'\|^2}{\ell_0^2}
\approx
\frac{d}{6\ell_0^2}.
\]

For the SE kernel,

\[
k_{\rm SE}(x,x')
\sim
\exp\left(
-\frac{d}{C\ell_0^2}
\right).
\]

Thus off-diagonal covariance can collapse exponentially in \(d\).

Crucially, derivatives with respect to \(\ell\) also inherit the same exponential factor, so

\[
\frac{\partial k}{\partial\ell}
\approx 0,
\]

and hence

\[
\nabla_\ell\log p(y\mid X)\approx 0.
\]

The optimizer can become stuck close to its initialization.

This is the gradient-vanishing failure mode emphasized by Xu et al.

A natural scaling is

\[
\boxed{
\ell_0=c\sqrt d.
}
\]

Then the normalized distance remains \(O(1)\):

\[
\frac{\|x-x'\|}{\ell_0}
\sim O(1).
\]

The kernel and its derivatives remain numerically meaningful.

---

# 12. Three closely related high-dimensional BO papers

There are three distinct papers in this research line.

Important: “Xu & Zhe 2024 preprint” and the Xu et al. ICLR 2025 paper are the same work at different stages, not two separate papers.

## 12.1 Hvarfner, Hellsten, Nardi — ICML 2024

Title:

“Vanilla Bayesian Optimization Performs Great in High Dimensions”

Core thesis:

\[
\boxed{
\text{vanilla BO is much stronger in high dimensions than common wisdom suggested}
}
\]

if GP priors are scaled appropriately.

Core scaling:

\[
\boxed{
\ell\propto\sqrt D.
}
\]

Implementation:

dimension-scaled LogNormal priors on length scales, followed by MAP-style inference.

Theoretical/diagnostic emphasis:

- high-dimensional pairwise distances grow;
- correlations collapse under fixed-scale kernels;
- GP model complexity / maximal information gain changes;
- EI/search behavior can become pathological or highly local.

The central chain is approximately

\[
D\uparrow
\to
\text{distance}\uparrow
\to
\text{correlation}\downarrow
\to
\text{effective model complexity}\uparrow
\to
\text{poor BO behavior}.
\]

## 12.2 Xu, Wang, Phillips, Zhe — arXiv 2024 / ICLR 2025

Early title:

“Standard Gaussian Process Can Be Excellent for High-Dimensional Bayesian Optimization”

Final ICLR title:

“Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization”

This is the same paper line, not a fourth paper.

Core thesis overlaps strongly with Hvarfner:

\[
\boxed{
\text{standard GP/BO can be highly competitive in high dimensions}
}
\]

and again identifies the scale

\[
\boxed{
\ell\sim\sqrt d.
}
\]

Independent technical emphasis:

\[
d\uparrow
\to
\|x-x'\|\sim\sqrt d
\to
k_{\rm SE}\approx 0
\to
\frac{\partial K}{\partial\ell}\approx0
\to
\nabla_\ell\log p(y\mid X)\approx0.
\]

The paper focuses on numerical training failure from initialization-induced gradient vanishing.

Additional contributions:

- probabilistic characterization/bounds for gradient vanishing;
- comparison of SE and Matérn susceptibility;
- robust initialization:
  \[
  \ell_0=c\sqrt d;
  \]
- additional bound showing how the proposed initialization mitigates the failure mode;
- empirical comparison across synthetic and real benchmarks.

The paper was accepted as ICLR 2025 Oral.

## 12.3 Papenmeier, Poloczek, Nardi — ICML 2025

Title:

“Understanding High-Dimensional Bayesian Optimization”

This paper explicitly builds on the recent line of work showing that simple/vanilla BO can work surprisingly well in high dimensions.

It separates two optimization problems:

\[
\boxed{
\text{GP surrogate fitting}
}
\]

and

\[
\boxed{
\text{acquisition-function optimization}.
}
\]

It studies:

- GP-fitting vanishing gradients;
- MLE versus MAP length-scale estimation;
- acquisition-function vanishing gradients;
- local search behavior;
- RAASP;
- how benchmark structure can favor long length scales.

Proposed method:

\[
\boxed{
MSR = \text{MLE Scaled with RAASP}.
}
\]

Length-scale initialization:

\[
\boxed{
\ell_{\rm init}=\frac{\sqrt d}{10}.
}
\]

The paper argues that ordinary MLE can be strong if initialized appropriately, and that local acquisition search can be a major source of success in extremely high-dimensional BO.

It was accepted as ICML 2025 Poster.

---

# 13. How similar are Hvarfner 2024 and Xu ICLR 2025?

A useful decomposition is:

| Aspect | Similarity |
|---|---|
| “Vanilla/standard BO can work surprisingly well in high dimension” | Very high |
| Pairwise distance grows like \(\sqrt d\) | Essentially identical starting fact |
| Recommended length-scale scale \(\ell\propto\sqrt d\) | Very high |
| Practical high-level advice | High |
| Exact estimator/training mechanism | Moderate |
| Failure mechanism | Moderate |
| Formal theoretical analysis | Much less similar |
| Matérn-vs-SE gradient robustness | Xu emphasizes much more |
| Model complexity/MIG/search behavior | Hvarfner emphasizes much more |

A reasonable qualitative reading is:

\[
\boxed{
\text{high-level thesis: highly overlapping}
}
\]

but

\[
\boxed{
\text{technical mechanism and theory: not the same paper}.
}
\]

The key distinction:

Hvarfner:

\[
\boxed{
\text{distance}
\to
\text{correlation}
\to
\text{model complexity}
\to
\text{BO behavior}
}
\]

Xu:

\[
\boxed{
\text{distance}
\to
\text{kernel derivative}
\to
\text{marginal-likelihood gradient}
\to
\text{training failure}.
}
\]

---

# 14. ICLR 2025 review record for Xu et al.

Final public reviewer scores were:

\[
\boxed{
8,\ 8,\ 6,\ 8,\ 8
}
\]

under the ICLR scale used for that cycle.

The final decision was:

\[
\boxed{
\text{Accept (Oral)}.
}
\]

The reviews were not uniformly positive from the beginning. Publicly archived discussion/snapshots show substantial novelty and scope concerns, and some reviewers explicitly state that they raised their score after clarification.

Because different public snapshots can reflect different review revisions, exact “initial score \(\to\) final score” histories should be treated carefully unless the reviewer explicitly states the change.

---

# 15. ICLR reviewer pG21: Matérn and the “emperor has no clothes” reaction

One reviewer was extremely positive about the high-level practical implication.

They observed that papers such as TuRBO, Tree UCB, RDUCB, HeSBO, ALEBO, and SAASBO often did not include a properly tuned vanilla BO baseline.

The reviewer wrote that the paper may have revealed that “the emperor has no clothes”: a substantial thread of HDBO research might have overlooked how strong a basic Matérn GP can be.

At the same time, the reviewer raised several criticisms:

1. Too few experimental replicates; about 10 runs was not enough given large variance.
2. The claim that Matérn is preferable to SE is not new.
3. Michael Stein’s 1999 work already criticized squared-exponential covariance and advocated Matérn-type modeling.
4. Some GP packages already use initialization heuristics based on empirical pairwise distances, so constant length-scale initialization is not universal.
5. The reviewer asked for experiments using the robust initialization with Matérn as well.

The Stein point is important:

Stein’s classical criticism of squared-exponential covariance is mainly about unrealistically strong smoothness assumptions, not about high-dimensional BO gradient vanishing.

Thus:

\[
\boxed{
\text{“Matérn is often a better covariance model than SE” is old}
}
\]

while

\[
\boxed{
\text{“SE initialization causes BO-specific high-dimensional gradient collapse”}
}
\]

is the newer Xu-style claim.

---

# 16. ICLR reviewer RmNw: detailed novelty and experimental criticism

This reviewer initially argued that:

- the paper mostly studies standard GP;
- gradient vanishing is a familiar phenomenon;
- the only apparent novelty might be the proposed \(\sqrt d\) initialization;
- measuring only first-step gradient norm and before/after length-scale movement seemed insufficient;
- fixed initializations such as 5, 15, 25 were not tested;
- robust initialization sometimes made performance worse;
- some wording and organization overstated the claims.

The authors’ rebuttal made an important distinction.

They said their contribution was not “discovering gradient vanishing” in the abstract.

Their claimed novelty was:

\[
\boxed{
\text{identifying a specific BO failure mode in which commonly used length-scale initialization causes GP training to have vanishing gradient at the first optimization step}
}
\]

plus a probabilistic quantitative analysis of how the failure probability grows with \(d\).

The authors also clarified the first-step logic:

If the gradient is already effectively zero at initialization, then the parameters do not move. The next step therefore evaluates the gradient at essentially the same location, so the optimizer remains stuck.

Thus the experiment intentionally used:

1. first-step gradient norm;
2. before/after length-scale change.

The paper was not claiming to characterize every possible form of gradient vanishing later in optimization.

The authors also corrected a typo.

The original wording said performance improved across “all benchmarks.”

The intended statement was performance improved across “all failed benchmarks.”

That materially narrows the claim.

After these clarifications, the reviewer explicitly wrote that they now acknowledged the novelty and contribution and increased the score.

The reason for the score increase was therefore not that the authors invented a new contribution during rebuttal. The reviewer changed their interpretation of the scope and novelty of the existing contribution.

---

# 17. ICLR reviewer VLW7: novelty overlap with Hvarfner 2024

The core criticism was:

- Hvarfner 2024 already argued that vanilla GP BO can perform well in high dimensions;
- Hvarfner already advocated length scales on the order of
  \[
  \sqrt D;
  \]
- Xu uses
  \[
  \ell_{\rm init}\propto\sqrt D;
  \]
- the two papers therefore have strongly overlapping central theses.

The reviewer was not fully convinced that “initialization is simpler than a prior” was enough by itself to justify novelty.

The review nevertheless acknowledged an independent valuable contribution:

\[
\boxed{
\text{the analysis of vanishing gradients caused by inappropriately small, commonly used length scales}.
}
\]

The correct interpretation of the final positive evaluation is not that the overlap disappeared.

It is:

\[
\boxed{
\text{the overlap remained, but reviewers judged that Xu still contained enough independent mechanism/theory/analysis}.
}
\]

---

# 18. Reviewer discussion on concurrent work

Another reviewer opened a dedicated discussion on the impact of Hvarfner 2024 on novelty.

The key points were:

1. The two works were initially posted to arXiv within two days of one another, so there is a strong case for treating them as concurrent work.
2. Directly using a dimension-scaled initialization is meaningfully simpler than requiring a particular prior distribution, even if both favor \(\sqrt d\)-scale length scales.
3. Requiring a prior imposes additional modeling commitments and may conflict with a user’s existing prior.
4. Even more practically, simply recommending Matérn instead of SE can itself be a powerful message.

The reviewer summarized the practical implication in very strong terms: perhaps the field would have been better off trying a basic Matérn kernel before adding many complicated HDBO mechanisms.

Important correction:

This “concurrent work + initialization is simpler” argument came from another reviewer in the discussion, not directly from the authors.

Concurrent work does not automatically remove a novelty problem. It mainly changes the interpretation from “later work repeating prior art” to “independent parallel discovery.” The paper still needs enough independent contribution to be publishable.

---

# 19. ICLR reviewer kZCF: high-dimensional optimization is still hard

This reviewer raised a deeper conceptual concern:

Even if GP hyperparameter training no longer fails, general high-dimensional global optimization can remain exponentially hard.

An objective may have exponentially many local optima.

Thus a second condition is needed for the paper’s practical conclusion: relevant objectives must have exploitable structure that makes good optima identifiable with feasible sample complexity.

The reviewer asked whether

\[
\ell_0\sim\sqrt d
\]

might intentionally create model mismatch by smoothing too aggressively.

The authors clarified:

\[
\ell_0\sim\sqrt d
\]

is only an initialization, not a permanently fixed final length scale.

Training can subsequently move the length scales to fit data.

The goal is simply to avoid starting inside a gradient-dead region.

The reviewer then explicitly wrote that the explanation resolved the question and increased the score.

Thus the paper does not establish:

\[
\boxed{
\text{general high-dimensional global optimization is easy}.
}
\]

It establishes a much narrower point:

\[
\boxed{
\text{one important GP-training failure mode can be avoided}.
}
\]

---

# 20. Added Humanoid experiment in the ICLR rebuttal

A reviewer asked whether the method had been tested in dimensions in the thousands.

The authors added a Humanoid reinforcement-learning benchmark with

\[
d=6392.
\]

They reported:

- common initializations caused training failure / gradient vanishing for both SE and Matérn at this extreme dimensionality;
- their robust initialization dramatically improved both;
- performance approached Hvarfner et al.’s VBO, which uses a LogNormal prior.

This experiment is important because it also illustrates the overlap with VBO:

both methods encourage dimension-appropriate length scales, but by different mechanisms.

---

# 21. ICML 2025: Understanding High-Dimensional Bayesian Optimization

The uploaded OpenReview record gives the following final reviewer recommendations:

| Reviewer | Recommendation |
|---|---:|
| xFof | 3 / 5 — Weak Accept |
| 9KgT | 4 / 5 — Accept |
| MivE | 3 / 5 — Weak Accept |
| bU1B | 4 / 5 — Accept |

Average:

\[
\boxed{
3.5/5.
}
\]

Final decision:

\[
\boxed{
\text{Accept (Poster)}.
}
\]

The Program Chairs’ decision is especially informative:

- there is no theoretical contribution;
- the experiments are not comprehensive;
- nevertheless, all reviewers found the paper well-written and well-organized;
- the paper provides impactful insights likely to facilitate new research;
- minor concerns were addressed in rebuttal.

This is an unusually clear statement of what kind of paper it is:

\[
\boxed{
\text{empirical/diagnostic contribution rather than theorem-driven novelty}.
}
\]

---

# 22. ICML reviewer xFof: “MSR is simply a combination of previous works”

This reviewer summarized three main findings:

1. GP fitting can suffer vanishing gradients;
2. acquisition-function optimization can suffer vanishing gradients;
3. MLE and MAP exhibit different bias–variance behavior.

The reviewer wrote that all three findings are consistent with prior work.

Strengths:

- the HDBO issues are worth investigating;
- they are systematically summarized;
- the proposed method is simple and effective.

Weaknesses:

- only four benchmark problems were used;
- Lasso-DNA and Mopta08 have special structure, yet the paper did not initially compensate with a broad enough benchmark suite;
- some relevant baselines were missing;
- most importantly:

> the authors did not provide any new significant suggestions for the three problems; the proposed method is simply a combination of previous works.

The reviewer gave:

\[
\boxed{
3/5=\text{Weak Accept}.
}
\]

The authors did not deny that MSR uses existing techniques.

Their rebuttal instead argued that its practical value lies in combining them effectively into a straightforward and strong HDBO recipe.

This is a crucial reading:

\[
\boxed{
\text{the novelty claim is mainly diagnosis/synthesis, not invention of every component}.
}
\]

---

# 23. ICML reviewer 9KgT: practical value, but contribution not huge

This reviewer was broadly positive.

They liked:

- the empirical evidence;
- the combination of length-scale initialization and local acquisition optimization;
- the practical insights;
- the clarity of the paper.

But they explicitly worried about the incremental value over recent papers such as “Vanilla BO.”

Their wording was essentially:

\[
\boxed{
\text{the method is better motivated and more generally applicable, but the absolute contribution is not huge}.
}
\]

They requested:

- exact RAASP implementation details;
- acquisition-function details;
- acquisition optimization details;
- acquisition optimization budget studies;
- TuRBO comparisons.

The authors added/clarified these points during rebuttal.

The reviewer then explicitly stated:

> Thanks for the rebuttal. I hope to see all these extra details in the final version. I have raised my score.

Final score:

\[
\boxed{
4/5=\text{Accept}.
}
\]

The public record does not safely establish the exact earlier numeric score, only that it was increased.

---

# 24. ICML reviewer MivE: “technical novelty is light”

This reviewer gave one of the clearest assessments:

> The technical novelty is light, but that is not the goal here.

They praised the paper for dissecting the complex behavior of GP length-scale optimization in high-dimensional BO.

As a practitioner, the reviewer found the analysis useful and expected the community to benefit from the insights.

Final recommendation:

\[
\boxed{
3/5=\text{Weak Accept}.
}
\]

This captures the paper’s accepted positioning:

\[
\boxed{
\text{light technical novelty}
+
\text{strong explanatory/practical value}.
}
\]

---

# 25. ICML reviewer bU1B: empirical claims are too broad

This reviewer liked the empirical observations but objected to broad language such as:

- “understanding HDBO”;
- “identifying fundamental challenges.”

The criticism was that the paper relies overwhelmingly on empirical observations.

The reviewer argued that stronger “fundamental” claims would require theory such as:

- upper bounds on MLE-gradient magnitudes;
- upper bounds on acquisition-gradient magnitudes;
- quantitative analysis of RAASP’s effect on gradients;
- regret analysis with RAASP.

The authors agreed to tone down the abstract/introduction and explicitly present the work as empirical/exploratory.

Final recommendation:

\[
\boxed{
4/5=\text{Accept}.
}
\]

This reviewer viewed the paper as useful exploratory research that should motivate future theory.

---

# 26. What the ICML 2025 paper adds beyond the ICLR 2025 paper

The overlap in GP fitting is large.

Both emphasize:

\[
d\uparrow
\to
\text{distance}\uparrow
\to
\text{kernel information/gradient collapse}
\]

and dimension-aware length-scale initialization.

But the ICML paper goes further in several directions.

## 26.1 Acquisition optimization

Even with a usable GP surrogate, BO still has to solve

\[
x_{t+1}
=
\arg\max_x a(x).
\]

In thousands of dimensions, the acquisition landscape itself can have nearly flat regions and tiny gradients.

The paper studies this optimization problem directly.

## 26.2 RAASP and local search

RAASP = Random Axis-Aligned Subspace Perturbations.

Instead of treating every acquisition optimization step as a global search over all coordinates, candidate generation is biased toward local perturbations around good observations, often perturbing only a subset of coordinates.

The important conceptual conclusion is:

\[
\boxed{
\text{very-high-dimensional BO success may depend heavily on effective local search behavior}.
}
\]

A globally perfect surrogate is not always necessary.

## 26.3 MLE versus MAP

The paper compares ordinary maximum-likelihood fitting with MAP fitting under a prior.

A simplified reading is:

\[
\boxed{
\text{MLE: lower prior bias, potentially high variance}
}
\]

\[
\boxed{
\text{MAP: reduced variance, but potentially damaging prior-induced bias}.
}
\]

Their conclusion is that ordinary MLE can be highly competitive if length scales are initialized appropriately.

## 26.4 MSR

They combine:

\[
\ell_{\rm init}
=
\frac{\sqrt d}{10}
\]

with MLE and RAASP.

Thus

\[
\boxed{
MSR = \text{MLE Scaled with RAASP}.
}
\]

The method itself is intentionally simple and substantially assembled from known components.

---

# 27. Relationship among the three papers

A concise chronology:

## Paper A — Hvarfner et al., ICML 2024

Question:

\[
\text{Why is vanilla BO believed to fail in high dimensions, and can a better-scaled GP prior fix it?}
\]

Main answer:

\[
\ell \sim \sqrt d
\]

through dimension-scaled priors.

Main analysis:

model complexity / information sharing / search behavior.

## Paper B — Xu et al., arXiv 2024 / ICLR 2025

Question:

\[
\text{Why does GP hyperparameter training itself fail with standard initialization?}
\]

Main answer:

initialization-induced gradient vanishing.

Main fix:

\[
\ell_0\sim\sqrt d
\]

and/or use Matérn.

Main additional contribution:

probabilistic bounds and SE-vs-Matérn robustness analysis.

## Paper C — Papenmeier et al., ICML 2025

Question:

\[
\text{What parts of recent “simple HDBO” methods are actually responsible for their success?}
\]

Main answer:

two separate failures matter:

\[
\boxed{
\text{GP fitting}
}
\]

and

\[
\boxed{
\text{acquisition optimization}.
}
\]

Main recipe:

\[
\boxed{
\text{scaled MLE initialization + RAASP local search}.
}
\]

The three papers therefore share a strong common theme:

\[
\boxed{
\text{many apparent failures of vanilla GP BO are caused by poor scaling/optimization choices, not an absolute impossibility of GP-based HDBO}.
}
\]

---

# 28. What “concurrent work” does and does not mean

If two works are posted independently at nearly the same time, concurrent work matters for priority and attribution.

It can support the claim that one group did not simply copy the other.

But it does not imply:

\[
\boxed{
\text{concurrent overlap automatically equals sufficient novelty}.
}
\]

If two papers had the same method, theorem, and experiments, a venue could still reject one as offering insufficient incremental contribution.

In the Xu/Hvarfner case, reviewers ultimately judged that Xu contained enough independent additions:

- BO-specific first-step gradient failure mechanism;
- probabilistic analysis;
- SE versus Matérn robustness;
- simple initialization rather than a persistent prior;
- large empirical study.

Thus the overlap remained real, but it was not treated as fatal.

---

# 29. Why Matérn’s prior-history criticism does not kill Xu’s result

The fact that Stein and GP literature have long criticized SE does not invalidate Xu’s more specific claim.

Old criticism:

\[
\boxed{
\text{SE implies unrealistically extreme smoothness in many applications}.
}
\]

Xu-style criticism:

\[
\boxed{
\text{SE with common fixed length-scale initialization can numerically collapse GP hyperparameter gradients in high-dimensional BO}.
}
\]

The same kernel can be criticized for different reasons.

Therefore “Matérn is often preferable” is not novel by itself, while “this specific high-dimensional training failure arises because of initialization and has this probability scaling” can still be novel.

---

# 30. Main conceptual lessons

## 30.1 Kernel choice and kernel training are different questions

A kernel family can be expressive enough in principle but still fail in practice because hyperparameter optimization never reaches a sensible region.

Thus one must separate

\[
\boxed{
\text{model family suitability}
}
\]

from

\[
\boxed{
\text{hyperparameter optimization failure}.
}
\]

## 30.2 High-dimensional pairwise distance matters numerically

For many normalized high-dimensional domains,

\[
\|x-x'\|
\sim
\sqrt d.
\]

A length scale that is sensible in \(d=5\) need not be sensible in \(d=500\).

Dimension-independent defaults can therefore be dangerous.

## 30.3 Matérn is more robust than SE to distance growth

SE decays roughly like

\[
e^{-O(d)}
\]

under fixed-scale random high-dimensional distances.

Matérn spatial covariance decays more slowly, roughly exponential in \(\sqrt d\) times a polynomial under comparable conditions.

This is one reason its gradients survive to higher dimension.

## 30.4 A good BO surrogate is not identical to a globally accurate regression model

BO only needs a model useful for sequential decision-making.

A model can have imperfect global fit yet still locate good regions efficiently.

This connects the Imprecise BO observation to the high-dimensional BO literature:

\[
\boxed{
\text{the right model for optimization is judged by decisions, not only by predictive MSE}.
}
\]

## 30.5 Acquisition optimization is a separate bottleneck

Even if

\[
\mu_t(x),\sigma_t(x)
\]

are well-defined and useful, one still has to optimize the acquisition function.

In thousands of dimensions, this can itself be the hardest numerical problem.

RAASP/local search directly addresses this second layer.

## 30.6 “Simple method wins” does not imply “high-dimensional optimization is solved”

None of these works proves that arbitrary \(d\)-dimensional black-box functions can be globally optimized with small sample complexity.

Worst-case high-dimensional optimization remains extremely hard.

The papers show that many benchmark and real-world tasks have enough exploitable structure that vanilla-style BO can perform much better than previously believed when implementation choices are corrected.

---

# 31. Reviewer-level takeaway

The two 2025 review records are particularly informative.

## ICLR 2025 Xu et al.

Reviewers were willing to accept substantial overlap with Hvarfner because they saw independent value in:

\[
\boxed{
\text{mechanistic diagnosis + quantitative theory + Matérn/SE analysis}.
}
\]

The paper eventually became an Oral.

## ICML 2025 Papenmeier et al.

Reviewers openly acknowledged:

\[
\boxed{
\text{technical novelty is light}
}
\]

and even

\[
\boxed{
\text{MSR is mostly a combination of prior components}.
}
\]

The paper was nevertheless accepted because its empirical decomposition and practical insights were seen as valuable.

This is a useful publication lesson:

\[
\boxed{
\text{a paper can be publishable because it explains, diagnoses, and systematizes an important phenomenon, even if it does not invent every component}.
}
\]

But the claim must be calibrated correctly. Reviewers reacted negatively whenever titles/abstracts sounded more universal or “fundamental” than the evidence justified.

---

# 32. Sources discussed

## Gaussian processes / Matérn

- Rasmussen & Williams, Gaussian Processes for Machine Learning:
  https://gaussianprocess.org/gpml/
- Matérn covariance discussion:
  https://gaussianprocess.org/gpml/chapters/RW4.pdf

## Student-t processes

- Shah, Wilson, Ghahramani, “Student-t Processes as Alternatives to Gaussian Processes,” AISTATS 2014:
  https://proceedings.mlr.press/v33/shah14.html

## Imprecise Bayesian Optimization

- Rodemann & Augustin, “Imprecise Bayesian Optimization,” Knowledge-Based Systems, 2024.
- LMU open-access manuscript:
  https://epub.ub.uni-muenchen.de/120260/1/1-s2.0-S0950705124008207-main.pdf

## Hvarfner et al., ICML 2024

- “Vanilla Bayesian Optimization Performs Great in High Dimensions”:
  https://proceedings.mlr.press/v235/hvarfner24a.html

## Xu et al., ICLR 2025

- “Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization”:
  https://openreview.net/forum?id=kX8h23UG6v
- ICLR proceedings:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/ec9b2a6ad5444caeff75efaa6176b3e4-Abstract-Conference.html
- Code:
  https://github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO

## Papenmeier et al., ICML 2025

- “Understanding High-Dimensional Bayesian Optimization”:
  https://openreview.net/forum?id=1d2fpvyKvJ
- PMLR:
  https://proceedings.mlr.press/v267/papenmeier25a.html

---

# 33. Compact final map

The cleanest way to remember the recent HDBO sequence is:

\[
\boxed{
\text{Hvarfner 2024: prior scaling / model complexity}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\ell\sim\sqrt d
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Xu 2024/ICLR 2025: initialization-induced GP training gradient vanishing}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\ell_0\sim\sqrt d,\quad \text{Matérn more robust than SE}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Papenmeier ICML 2025: fitting problem + acquisition optimization problem}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{scaled MLE + RAASP = MSR}
}
\]

The common meta-message is:

\[
\boxed{
\text{before inventing a more elaborate high-dimensional BO method, check whether the standard GP is simply being initialized, regularized, or optimized badly.}
}
\]
