# Delta, Loss, and Formula Taxonomy Across Markdown Notes

Date: 2026-05-15

## Scope

This note summarizes the attack-related formulas, loss definitions, and meanings of
`delta` that appear across the repository Markdown notes. It is written to resolve one
specific ambiguity:

> Sometimes `delta` is treated as an infinitesimal/local direction, and sometimes it is
> a finite adversarial perturbation with real radius. These are not the same object.

Observed from repository Markdown files, the relevant material clusters around:

- `three_loss_objective_experiment_plan.md`
- `docs/loss3_original_theory_experiment_plan.md`
- `BATCH_LOSS_ONLY_OPTIMIZATION_METHODS.md`
- `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md`
- `LOSS1_ZERO_DELTA_GRADIENT_CHECK.md`
- `LP_STEEPEST_DIRECTION_CHECK.md`
- `docs/main_objective_mechanism_experiment1_result_20260514.md`
- `docs/fno_solver_jacobian_similarity_result_20260514.md`
- `docs/local_jacobian_svd_experiment_purpose_20260515.md`
- `docs/local_jacobian_svd_direction_taxonomy_20260515.md`
- `docs/outward_growth_direction_experiment_plan_20260515.md`
- `docs/outward_growth_direction_result_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_experiment_plan_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_detailed_data_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_target_loss3_table_20260515.md`
- `docs/loss_gradient_direction_vs_svd_direction_20260515.md`
- `docs/deeponet_loss1_vs_loss3_jacobian_attack_tension_20260515.md`
- `docs/deeponet_solver_jacobian_similarity_result_20260515.md`
- `docs/comprehensive_svd_diagnostics_result_20260515.md`
- generated summaries under `forensics/*/summary.md`

Some environment, training, benchmark, or runbook Markdown files contain words like
`loss`, `gradient`, or `delta` in a non-attack sense. Those are not used below as
adversarial perturbation formulas.

## Core Objects

The recurring regression-attack notation is:

\[
f = \text{learned neural operator model}
\]

\[
j \text{ or } g = \text{true solver / oracle}
\]

\[
e(x)=f(x)-j(x)
\]

\[
b=e(x)=f(x)-j(x)
\]

Here `b` is the clean residual at the unperturbed input.

The perturbation is:

\[
\delta
\]

and the perturbed input is:

\[
x+\delta.
\]

The output increments are:

\[
\Delta f = f(x+\delta)-f(x)
\]

\[
\Delta j = j(x+\delta)-j(x)
\]

\[
\Delta e = e(x+\delta)-e(x)=\Delta f-\Delta j.
\]

## The Three Base Losses

The core three-loss framework is:

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_q=\|\Delta f\|_q
\]

\[
L_2(\delta)=\|f(x+\delta)-j(x)\|_q=\|b+\Delta f\|_q
\]

\[
L_3(\delta)=\|f(x+\delta)-j(x+\delta)\|_q
=\|b+\Delta f-\Delta j\|_q
=\|e(x+\delta)\|_q.
\]

Their meanings are:

| Loss | What it measures | Main role |
|---|---|---|
| `loss1` | Model output movement only | Surrogate / false-alert baseline |
| `loss2` | Perturbed model output versus clean oracle target | Fixed-target surrogate |
| `loss3` | Perturbed model output versus perturbed oracle target | Correct regression attack loss |

Inference from the theory notes: `loss3_original` is the primary finite-radius
regression attack objective because it directly measures oracle-relative error at the
perturbed input.

## Objective Variants

For each base loss, the Markdown notes define three objective variants.

### Original

\[
O_i^{orig}(\delta)=L_i(\delta).
\]

This asks: how large is the endpoint value?

### Increment Ratio

\[
O_i^{ratio}(\delta)=\frac{L_i(\delta)-L_i(0)}{\|\delta\|_p+\eta}.
\]

This asks: how much loss increase is obtained per unit perturbation?

Important: this only has a clean local-derivative interpretation when `delta` is very
small. If the optimizer returns a finite nonzero `delta`, the ratio is no longer just a
local derivative; it is a finite secant quantity.

### Regularized

\[
O_i^{reg}(\delta)=L_i(\delta)-C\|\delta\|_p.
\]

This asks: how much loss remains after subtracting a perturbation cost?

Important: this objective may prefer an interior point with
\(\|\delta\|_p < \varepsilon\). Such a final `delta` is finite unless its norm is
verified to be extremely small.

## The Nine Named Objectives

The main 3 by 3 table is:

| Objective | Formula | Interpretation |
|---|---|---|
| `loss1_original` | \(\|\Delta f\|_q\) | Endpoint model self-movement |
| `loss1_increment_ratio` | \(\|\Delta f\|_q/(\|\delta\|_p+\eta)\) | Local/finite model sensitivity ratio |
| `loss1_regularized` | \(\|\Delta f\|_q-C\|\delta\|_p\) | Cost-aware model movement |
| `loss2_original` | \(\|b+\Delta f\|_q\) | Fixed-clean-target endpoint error |
| `loss2_increment_ratio` | \((\|b+\Delta f\|_q-\|b\|_q)/(\|\delta\|_p+\eta)\) | Fixed-target outward growth ratio |
| `loss2_regularized` | \(\|b+\Delta f\|_q-C\|\delta\|_p\) | Cost-aware fixed-target error |
| `loss3_original` | \(\|b+\Delta f-\Delta j\|_q\) | True perturbed-oracle endpoint error |
| `loss3_increment_ratio` | \((\|b+\Delta f-\Delta j\|_q-\|b\|_q)/(\|\delta\|_p+\eta)\) | True error-norm outward growth ratio |
| `loss3_regularized` | \(\|b+\Delta f-\Delta j\|_q-C\|\delta\|_p\) | Cost-aware true error |

The notes also introduce an additional diagnostic:

\[
O_{resid\ ratio}(\delta)=
\frac{\|e(x+\delta)-e(x)\|_q}{\|\delta\|_p+\eta}
=
\frac{\|\Delta f-\Delta j\|_q}{\|\delta\|_p+\eta}.
\]

This is often called `loss3_residual_increment_ratio` or residual increment ratio.
It measures movement of the error field, not necessarily growth of the error norm.

## Local Linearization Formulas

When `delta` is very small around a clean input `x`, the Markdown notes repeatedly use
Jacobian linearization:

\[
\Delta f \approx J_f(x)\delta
\]

\[
\Delta j \approx J_j(x)\delta
\]

\[
\Delta e \approx (J_f(x)-J_j(x))\delta.
\]

Define:

\[
A=J_f(x)-J_j(x).
\]

Then:

\[
e(x+\delta)\approx b+A\delta.
\]

This is a local model. It is valid only near `delta = 0`, or when finite-difference
checks show the linear approximation remains accurate at the chosen radius.

## Local Operator Norms

The local sensitivity quantities are:

\[
L_f=\|J_f\|_{p\to q}
\]

\[
L_j=\|J_j\|_{p\to q}
\]

\[
L_e=\|J_f-J_j\|_{p\to q}=\|A\|_{p\to q}.
\]

For example:

\[
v_f^*=\arg\max_{\|v\|_p=1}\|J_fv\|_q
\]

\[
v_j^*=\arg\max_{\|v\|_p=1}\|J_jv\|_q
\]

\[
v_e^*=\arg\max_{\|v\|_p=1}\|Av\|_q.
\]

Here the optimized variable is a unit direction `v`, not a finite attack `delta` by
itself. If one writes \(\delta=r v\), the local interpretation requires \(r\to0\) or a
very small tested radius.

## Error Field Movement Versus Error Norm Growth

Two formulas look similar but mean different things.

### Error Field Movement

\[
\frac{\|e(x+\delta)-e(x)\|_q}{\|\delta\|_p}
\approx
\frac{\|A\delta\|_q}{\|\delta\|_p}.
\]

This asks:

> How far does the residual vector move?

It is connected to \(\|A\|_{p\to q}\), the local mismatch/Lipschitz direction.

### Error Norm Growth

For \(q=2\), when \(b\neq0\):

\[
\|b+A\delta\|_2-\|b\|_2
\approx
\left\langle \frac{b}{\|b\|_2}, A\delta \right\rangle.
\]

If \(\delta=r v\) and \(\|v\|_2=1\), then:

\[
\frac{\|b+Arv\|_2-\|b\|_2}{r}
\approx
\left\langle \frac{b}{\|b\|_2}, Av \right\rangle.
\]

This asks:

> Does the residual move outward in the current residual direction?

Inference from the outward-growth notes: maximizing \(\|Av\|\) and maximizing the
outward component \(\langle b/\|b\|,Av\rangle\) can select very different directions.

## Squared Expansion Clarification

The same point can be seen without hiding the quadratic term. Under the local affine
model:

\[
S(\delta)=\|b+A\delta\|_2^2.
\]

Expanding gives:

\[
S(\delta)=\|b\|_2^2+2b^TA\delta+\delta^TA^TA\delta.
\]

The linear part is:

\[
2b^TA\delta.
\]

The gradient at exactly \(\delta=0\) is:

\[
\nabla_\delta S(0)=2A^Tb.
\]

So the local outward-growth direction is related to:

\[
A^Tb.
\]

But for finite nonzero `delta`, the quadratic term

\[
\delta^TA^TA\delta
\]

cannot be ignored. Its gradient is:

\[
2A^TA\delta.
\]

Therefore, \(A^Tb\) is a first-step or infinitesimal direction, not the full
finite-radius solution.

For residual movement alone:

\[
M(\delta)=\|A\delta\|_2^2=\delta^TA^TA\delta
\]

and:

\[
\nabla_\delta M(\delta)=2A^TA\delta.
\]

This explains the difference between:

- maximizing residual movement \(\|A\delta\|\), and
- maximizing endpoint residual norm \(\|b+A\delta\|\).


## Local Squared-Loss Gradient Formulas

Several Markdown notes also use squared L2 local objectives to explain why SVD
directions, clean outward-growth directions, and one-step attack gradients are not the
same thing.

For `loss1`:

\[
L_{1,sq}(\delta) \approx \|J_f\delta\|_2^2
\]

\[
\nabla_\delta L_{1,sq}(\delta)=2J_f^TJ_f\delta.
\]

At exactly \(\delta=0\):

\[
\nabla_\delta L_{1,sq}(0)=0.
\]

So `loss1` has no clean-residual outward term. Its local quadratic gain direction only
emerges after starting from a nonzero `delta` or by running a power/SVD-style iteration.

For `loss2`:

\[
L_{2,sq}(\delta) \approx \|b+J_f\delta\|_2^2
\]

\[
\nabla_\delta L_{2,sq}(\delta)=2J_f^Tb+2J_f^TJ_f\delta.
\]

At exactly \(\delta=0\):

\[
\nabla_\delta L_{2,sq}(0)=2J_f^Tb.
\]

For `loss3`:

\[
L_{3,sq}(\delta) \approx \|b+J_e\delta\|_2^2,
\quad J_e=J_f-J_j
\]

\[
\nabla_\delta L_{3,sq}(\delta)=2J_e^Tb+2J_e^TJ_e\delta.
\]

At exactly \(\delta=0\):

\[
\nabla_\delta L_{3,sq}(0)=2J_e^Tb.
\]

The corresponding one-step local gradient directions at a current attack iterate
\(\delta_k\) are:

\[
g_1(\delta_k)=J_f^TJ_f\delta_k
\]

\[
g_2(\delta_k)=J_f^Tb+J_f^TJ_f\delta_k
\]

\[
g_3(\delta_k)=J_e^Tb+J_e^TJ_e\delta_k.
\]

Here \(\delta_k\) itself may be finite. The next optimizer step can be read as an
infinitesimal step around the current finite point, but that does not make the base
perturbation \(\delta_k\) infinitesimal.

## Older Implementation Loss Names

The run manifests and older attack summaries also mention implementation-level names:

| Name | Meaning in the notes |
|---|---|
| `loss2_fixed` | Fixed clean-target variant of `loss2` |
| `loss2_dict_N200`, `loss2_dict_N2000`, `loss2_dict_N20000` | Historical dictionary/oracle approximations for fixed-target comparisons |
| `loss3_stopgrad` | Perturbed-oracle `loss3` value with solver/oracle gradient stopped or altered |
| `loss3` | Full perturbed-oracle loss when solver/oracle response is part of the objective |
| `corrected_5loss` | Older result-label family containing the historical loss variants above |

These names are not new mathematical categories beyond the `loss1` / `loss2` / `loss3`
distinction. They mainly record which implementation supplied the oracle target and
whether gradients through the solver were included.

## Finite-Radius Attack Formulas

The finite-radius attack problem is:

\[
\max_{\|\delta\|_p\le\varepsilon} O(\delta).
\]

For the true regression objective:

\[
\max_{\|\delta\|_p\le\varepsilon}
\|f(x+\delta)-j(x+\delta)\|_q.
\]

In this setting, `delta` is not infinitesimal. It is a real perturbation constrained by
the attack budget.

Examples from the notes include:

\[
\varepsilon=8,\quad \alpha=0.3,\quad steps=100
\]

and stronger settings such as:

\[
\varepsilon=32,\quad \alpha=0.6.
\]

These are not local-linear smoke-test radii for 1024-point Burgers L2 attacks.

## Epsilon Scaling Formulas

The scale-calibration notes define:

\[
per\_point\_delta\_rms = relative\_rms \cdot input\_rms\_mean.
\]

For finite \(p\):

\[
\varepsilon_p = per\_point\_delta\_rms \cdot n_{points}^{1/p}.
\]

For \(p=2\):

\[
\varepsilon = per\_point\_delta\_rms \sqrt{n_{points}}.
\]

For \(p=\infty\):

\[
\varepsilon_\infty = per\_point\_delta\_rms.
\]

Step size is chosen by:

\[
\alpha=\frac{\varepsilon}{reach\_epsilon\_steps}.
\]

This is finite-scale attack design. It should not be interpreted as an infinitesimal
Taylor expansion unless the resulting \(\varepsilon\) is very small.

## PGD and LP-Steepest Updates

The projected-gradient style update is conceptually:

\[
\delta_{k+1}=\Pi_{\|\delta\|_p\le\varepsilon}
\left(\delta_k+\alpha g_k\right).
\]

The LP-steepest update uses the steepest direction in the dual geometry:

\[
s_k=\arg\max_{\|s\|_p\le1}\langle \nabla_\delta O(\delta_k),s\rangle.
\]

Then:

\[
\delta_{k+1}=\Pi_{\|\delta\|_p\le\varepsilon}
\left(\delta_k+\alpha s_k\right).
\]

Typical special cases are:

\[
p=2:\quad s_k=\frac{g_k}{\|g_k\|_2}
\]

\[
p=\infty:\quad s_k=\operatorname{sign}(g_k)
\]

\[
p=1:\quad s_k=\operatorname{sign}(g_{k,i^*})e_{i^*},\quad
 i^*=\arg\max_i |g_{k,i}|.
\]

Here \(\delta_k\) is generally finite after the first nontrivial step. The gradient is
computed at the current finite iterate, not only at \(\delta=0\).

## Ray and Boundary-Rescaled Diagnostics

Ray-profile diagnostics use a direction:

\[
v=\frac{\delta}{\|\delta\|_p}
\]

and evaluate:

\[
r\mapsto \|e(x+rv)\|_q,\quad r\in[0,\varepsilon].
\]

They may also evaluate:

\[
r\mapsto
\frac{\|e(x+rv)\|_q-\|e(x)\|_q}{r+\eta}
\]

and:

\[
r\mapsto
\frac{\|e(x+rv)-e(x)\|_q}{r+\eta}.
\]

Only the small-\(r\) part of the ray has a local/infinitesimal interpretation. The
endpoint \(r=\varepsilon\) is finite-radius behavior.

Boundary rescaling constructs:

\[
\delta_{bdry}=\varepsilon\frac{\delta_{final}}{\|\delta_{final}\|_p}.
\]

This is explicitly a finite perturbation. It should not be analyzed by dropping higher
order terms unless separately validated.

## Path-Integral Formula

The notes connect local and finite behavior through:

\[
e(x+\delta)-e(x)=\int_0^1 J_e(x+t\delta)\delta\,dt.
\]

Thus:

\[
e(x+\delta)=b+\int_0^1 J_e(x+t\delta)\delta\,dt.
\]

And:

\[
\|e(x+\delta)-e(x)\|_q
\le
\|\delta\|_p
\int_0^1 \|J_e(x+t\delta)\|_{p\to q}\,dt.
\]

This is the cleanest explanation for why finite-radius `delta` cannot usually be
reduced to a single Jacobian at `x`: the Jacobian can rotate, saturate, or change along
the path.

## Zero-Delta and Random-Start Notes

The zero-delta checks emphasize that for `loss1`:

\[
L_1(0)=\|f(x)-f(x)\|=0.
\]

At exactly \(\delta=0\), practical autograd can return a zero gradient for norms of a
zero residual. Thus `loss1` can have a dead start.

For ratio objectives:

\[
\frac{L(\delta)-L(0)}{\|\delta\|+\eta}
\]

requires \(\eta>0\) to avoid division by zero at \(\delta=0\). Even with \(\eta>0\), a
random start can be needed to avoid a zero-gradient or nonsmooth start.

For regularized objectives:

\[
L(\delta)-C\|\delta\|
\]

there is a nonsmooth norm term at \(\delta=0\), so random starts are also used in the
experiment policy.

## When Delta Can Be Treated As Infinitesimal

`delta` can be treated as infinitesimal only in these cases:

1. Deriving Jacobian linearizations at the clean input:

   \[
   f(x+\delta)-f(x)\approx J_f(x)\delta.
   \]

2. Interpreting local operator norms:

   \[
   \|J_f\|_{p\to q},\quad \|J_j\|_{p\to q},\quad \|J_f-J_j\|_{p\to q}.
   \]

3. Interpreting residual increment ratios in the limit:

   \[
   \lim_{\|\delta\|\to0}
   \frac{\|e(x+\delta)-e(x)\|}{\|\delta\|}.
   \]

4. Interpreting outward growth at \(\delta=0\):

   \[
   \nabla_\delta \|b+A\delta\|_2^2\big|_{\delta=0}=2A^Tb.
   \]

5. Finite-difference checks with explicitly tiny radii, such as:

   \[
   \rho=10^{-4},\quad 10^{-3}.
   \]

   A larger value such as \(10^{-2}\) is still a checkable small radius, but it should not
   be assumed infinitesimal without confirming that predictions match observed values.

6. Old tiny smoke settings such as:

   \[
   \varepsilon=0.01
   \]

   for a 1024-point L2 Burgers attack, where the dense per-point RMS perturbation is:

   \[
   \frac{0.01}{\sqrt{1024}}=0.0003125.
   \]

   This is local-smoke scale, not serious finite-radius attack scale.

## When Delta Cannot Be Treated As Infinitesimal

`delta` cannot be treated as infinitesimal in these cases:

1. Main finite-radius attacks:

   \[
   \max_{\|\delta\|_p\le\varepsilon} L(\delta).
   \]

2. Any endpoint comparison at \(\|\delta\|\approx\varepsilon\), especially with budgets like:

   \[
   \varepsilon=8\quad\text{or}\quad\varepsilon=32.
   \]

3. PGD or LP-steepest iterates after nontrivial updates:

   \[
   \delta_k \neq 0.
   \]

4. Boundary-rescaled deltas:

   \[
   \delta_{bdry}=\varepsilon \delta/\|\delta\|.
   \]

5. Ray-profile endpoints:

   \[
   r=\varepsilon.
   \]

6. Regularized or increment-ratio final deltas unless their final norm is explicitly
   shown to be tiny.

7. Any claim about final adversarial risk:

   \[
   \|f(x+\delta)-j(x+\delta)\|.
   \]

8. Any setting where the solver/oracle response \(j(x+\delta)-j(x)\) is non-negligible.

In these cases, quadratic and higher-order terms, solver response, and path effects
matter.

## Conditional Cases

Some formulas sit between local and finite interpretations.

| Object | Local meaning | Finite meaning |
|---|---|---|
| `increment_ratio` | Derivative/secant near zero | A finite ratio if optimizer chooses non-small `delta` |
| `regularized` | Local cost-aware diagnostic near zero | Interior finite-radius objective if final `delta` is not tiny |
| `power_iteration` | Operator norm direction for local Jacobian | Heuristic/optimizer variant if applied to finite nonlinear objective |
| `ray profile` | Local slope near \(r=0\) | Nonlinear path/endpoint behavior near \(r=\varepsilon\) |
| `A^Tb` | Gradient at zero of squared endpoint residual | Not enough to solve finite-radius attack |
| `A^TA\delta` | Linear-model curvature/residual movement term | Matters more as \(\delta\) grows |

## Practical Rule

A simple way to decide how to read a formula is:

1. If the formula contains a Jacobian at clean input, such as \(J_f(x)\), \(J_j(x)\), or
   \(A=J_f-J_j\), then it is local unless the note explicitly validates it at a finite
   radius.

2. If the formula takes \(\|\delta\|\to0\), uses \(v\) with \(\delta=rv\), or talks about
   first-order Taylor expansion, then `delta` is infinitesimal.

3. If the formula contains \(\|\delta\|_p\le\varepsilon\), PGD steps, final delta,
   boundary rescaling, or ray endpoints, then `delta` is finite.

4. If the formula is about final attack success, use `loss3_original` at the actual
   finite `delta`. Do not replace it with local Jacobian reasoning.

5. If the formula is about mechanism diagnosis, local mismatch, or initial growth, then
   local Jacobian and infinitesimal `delta` reasoning is appropriate.

## Bottom Line

The repository uses two different meanings of `delta`:

1. Local diagnostic `delta`:

   \[
   \delta\to0,\quad \delta=rv,\quad r\text{ very small}.
   \]

   This is for Jacobians, SVD directions, residual movement, outward growth, and
   finite-difference validation.

2. Finite attack `delta`:

   \[
   \|\delta\|_p\le\varepsilon.
   \]

   This is for PGD, LP-steepest PGD, final adversarial examples, ray endpoints,
   boundary rescaling, and the real `loss3_original` attack objective.

The main conceptual warning is:

\[
\text{best local direction} \neq \text{best finite-radius attack direction}.
\]

That is exactly why the notes distinguish `loss3_original` from ratio, residual-ratio,
regularized, SVD, and outward-growth diagnostics.
