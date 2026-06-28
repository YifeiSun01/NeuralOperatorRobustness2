# PGD / Epsilon-Ball Boundary Optimum Notes for Loss3

Generated: 2026-05-20T00:56:26.721576+00:00

## Short Answer

For the usual fixed-budget adversarial attack

```text
max_delta L(x + delta)
s.t.      ||delta||_p <= epsilon
```

the optimizer is usually expected to end on, or very near, the epsilon-ball boundary when the loss is locally linear and the input-box constraints are not the active limitation. In that common PGD/FGSM setting, using the full perturbation budget is the natural behavior.

However, this is not a universal theorem for every neural-network loss. For a general nonconvex, nonsmooth, or saturated loss, an interior local optimum or a flat interior plateau is possible. Also, if the objective includes smoothness penalties, frequency penalties, norm penalties, or asks for the minimum adversarial perturbation rather than the maximum loss under a fixed budget, the solution may intentionally stay inside the epsilon ball.

## Observed From Sources

- Goodfellow, Shlens, and Szegedy, *Explaining and Harnessing Adversarial Examples* ([arXiv 1412.6572](https://ar5iv.labs.arxiv.org/html/1412.6572)): the FGSM derivation linearizes the cost around the current input. Under that local linear approximation, the optimal max-norm-constrained perturbation uses the epsilon budget. For `L_inf`, this gives `delta = epsilon * sign(grad_x L)`. This is the cleanest reason people expect fixed-budget attacks to land on the boundary.

- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* ([arXiv 1706.06083](https://ar5iv.labs.arxiv.org/html/1706.06083)): adversarial training is formulated as a robust-optimization saddle-point problem. The inner problem is exactly a constrained maximization over an allowed perturbation set, and PGD is treated as a strong first-order method for that inner maximization. Their experiments show PGD loss often increases and plateaus rapidly, but they do not claim a general theorem that every neural-network maximizer must be on the boundary.

- Adversarial Robustness Toolbox PGD documentation ([ART docs](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html)): PGD is described as an iterative method that projects the perturbation back to an `lp` ball of radius `eps`; `eps` is the maximum perturbation budget and `eps_step` is the per-iteration step size. This matches the practical implementation pattern used in robustness work.

- DeepFool, *A Simple and Accurate Method to Fool Deep Neural Networks* ([arXiv 1511.04599](https://ar5iv.labs.arxiv.org/html/1511.04599)): DeepFool has a different goal: find a small or near-minimal perturbation that crosses the decision boundary. This is a minimum-distortion problem, not a fixed-budget maximum-loss problem, so it is a counterexample to the idea that all adversarial methods should use the full epsilon budget.

- Brendel, Rauber, and Bethge, *Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models* ([arXiv 1712.04248](https://ar5iv.labs.arxiv.org/html/1712.04248)): Boundary Attack starts from a large adversarial perturbation and reduces its size while staying adversarial. Again, this is a minimum-distance-to-adversarial-region style objective, not a fixed-epsilon max-loss objective.

- Croce and Hein, *Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks* ([arXiv 2003.01690](https://arxiv.org/abs/2003.01690)): AutoAttack/APGD work emphasizes that PGD can fail because of suboptimal step size or objective issues. This supports the interpretation that failure to reach/use the boundary can be an optimization-path/hyperparameter issue, not necessarily evidence that the true fixed-budget maximizer is interior.

- Distill discussion, Nakkiran, *Adversarial Examples are Just Bugs, Too* ([Distill 2019](https://distill.pub/2019/advex-bugs-discussion/response-5/)): the discussion gives a standard PGD update with projection onto the epsilon ball and emphasizes that what PGD finds depends on the loss/direction chosen. This is useful context: PGD's boundary behavior is tied to the chosen objective and update rule, not an objective-independent law.

- Forum-level discussions, e.g. Reddit threads on adversarial robustness evaluation, often distinguish PGD fixed-epsilon attacks from C&W/DeepFool/FAB/minimum-norm attacks and recommend stronger suites such as AutoAttack for reliable evaluation. These discussions are practical rather than proof-level, but they reinforce that `epsilon`-bounded max-loss attacks and minimum-distortion attacks answer different questions.

## Mathematical Reasoning

Let

```text
f(delta) = L(x + delta),    feasible set B = {delta : ||delta||_p <= epsilon}.
```

If `f` is differentiable and `delta*` is a strict interior local maximizer, then the usual first-order necessary condition gives

```text
grad_delta f(delta*) = 0.
```

Therefore, if the gradient is nonzero at an interior point, that point cannot be a local maximum; one can move a little in an ascent direction while staying inside the ball. This is the main mathematical reason boundary optima are expected in ordinary attack losses.

Under a local linear model,

```text
f(delta) ~= f(0) + g^T delta,
```

with `g != 0`, the maximizer over a norm ball always uses the full norm budget:

```text
||delta*||_p = epsilon.
```

For common norms:

```text
L2:    delta* = epsilon * g / ||g||_2
Linf:  delta* = epsilon * sign(g)
```

This is exactly the FGSM / steepest-ascent intuition.

## When The Optimum Can Be Interior

Interior optima or non-boundary final iterates can happen in several cases:

- The loss has an interior stationary local maximum, i.e. `grad_delta L = 0` inside the ball.
- The loss saturates or becomes flat, so many points have essentially the same value.
- The input-domain box constraint `x + delta in [0,1]` is active before the epsilon norm is saturated.
- The objective includes a penalty such as `L(x+delta) - lambda ||delta||^2`, TV penalty, smoothness penalty, or high-frequency penalty.
- The attack objective is minimum distortion, e.g. DeepFool, C&W-style, FAB, Boundary Attack, rather than fixed-budget maximum loss.
- The optimizer fails to reach the boundary because of step-size, gradient scaling, poor direction choice, early stopping, or gradient masking/obfuscation.

## Interpretation For Our Loss3 Experiments

Observed from our local alpha/epsilon sweep artifacts:

- `raw_replace` and `steepest_replace` hit the 99% boundary marker at step `1` in the completed `p=2,q=2` sweep.
- `steepest_add` reaches the boundary later, roughly on the expected `epsilon / alpha` scale.
- `raw_add` can be much slower; for the old `epsilon=8, alpha=0.3` setting, batch-mean 99% boundary arrival was not reached by step `100`, and many samples stayed below the 99% threshold.

Inference from the literature and our diagnostics:

- The slow boundary arrival of `raw_add` should not be read as evidence that the true fixed-budget loss3 maximizer is interior.
- It is more likely an optimization-path issue: raw gradients have variable norm and may move too slowly in `L2` norm, while steepest/additive or replacement-style updates use the norm geometry more directly.
- The replacement methods are effectively solving or approximating the local linearized inner maximization on the boundary, which is why they immediately use the epsilon budget.
- A fair comparison should separate two phases: reaching the boundary and optimizing along or near the boundary.

## Recommended Diagnostic

For a final perturbation direction

```text
u = delta_final / ||delta_final||_p,
```

plot a radial profile:

```text
r in [0, epsilon]  ->  L(x + r u).
```

If this curve usually increases up to `r = epsilon`, then the evidence supports the boundary-optimum interpretation for the observed directions. If it peaks at some `r < epsilon`, then the particular loss/model/sample has an interior preferred radius.

A second useful check is boundary rescaling:

```text
delta_boundary = epsilon * delta / ||delta||_p
```

If `L(x + delta_boundary) > L(x + delta)` whenever `raw_add` remains inside the ball, then raw_add's interior point is not optimal; it simply failed to use the available budget.
