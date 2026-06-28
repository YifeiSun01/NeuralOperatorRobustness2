# Loss Objective Direction Experiments In The Error-Operator Framework

Date: 2026-06-08

This note records how the older Burgers loss1/loss2/loss3 direction experiments fit into the current error-operator robustness framework.

The short answer is:

- The local error-Jacobian spectral norm describes infinitesimal movement of the error field.
- The practical loss3/physics attack usually maximizes endpoint error energy, not only error-field movement.
- Therefore the practical PGD direction contains an extra first-order residual term. This is exactly why earlier experiments found that loss3 directions, residual-change/SVD directions, and loss1/loss2 directions are related but not identical.

## Core Notation

Let

\[
\mathcal M(a)=\text{model output},\qquad \mathcal S(a)=\text{solver/oracle output}.
\]

Define the model-solver error operator

\[
\mathcal E(a)=\mathcal M(a)-\mathcal S(a).
\]

At a clean input \(a\), write

\[
e_0=\mathcal E(a),
\qquad
A=D\mathcal E(a)=D\mathcal M(a)-D\mathcal S(a).
\]

For a small perturbation \(h\),

\[
\mathcal E(a+h)\approx e_0+Ah.
\]

In finite-dimensional Burgers notation this is the same as

\[
e(x+\delta)\approx b+(J_f-J_j)\delta,
\]

where \(b=f(x)-j(x)\) and \(A=J_f-J_j\).

## Three Quantities That Must Not Be Mixed

### 1. Error-Field Movement

\[
\|\mathcal E(a+h)-\mathcal E(a)\|_2^2
\approx
\|Ah\|_2^2.
\]

This is the quantity directly controlled by the local error Jacobian.

\[
\sup_{\|h\|_2\le \epsilon}
\|\mathcal E(a+h)-\mathcal E(a)\|_2
=
\epsilon \|A\|_{2\to2}+o(\epsilon).
\]

For \(L^2\to L^2\), \(\|A\|_{2\to2}\) is the spectral norm. Equivalently,

\[
\|A\|_{2\to2}^2=\lambda_{\max}(A^*A).
\]

This is the cleanest mathematical link between local robustness and the kernel/Jacobian spectrum.

### 2. Endpoint Error Energy

The practical loss3 or physics attack often maximizes the endpoint error itself:

\[
\|\mathcal E(a+h)\|_2^2.
\]

Using the local affine approximation:

\[
\|\mathcal E(a+h)\|_2^2
\approx
\|e_0+Ah\|_2^2
=
\|e_0\|_2^2
+2\langle e_0,Ah\rangle
+\|Ah\|_2^2.
\]

So this objective has an extra first-order residual term:

\[
A^*e_0.
\]

At very small \(\epsilon\), if \(A^*e_0\ne0\), the endpoint-energy attack direction is dominated by

\[
\frac{A^*e_0}{\|A^*e_0\|_2},
\]

not by the top right singular vector of \(A\).

Only when \(e_0=0\), or \(A^*e_0\) is negligible, or when the quadratic term dominates at finite radius, does endpoint-energy behavior become close to the pure spectral-norm direction.

### 3. True Nonlinear Finite-Radius PGD

The actual attack is usually

\[
\max_{\|h\|_p\le\epsilon}
\|\mathcal M(a+h)-\mathcal S(a+h)\|_q.
\]

For non-infinitesimal \(\epsilon\), this is not solved by one clean-point Jacobian. The trajectory can rotate because \(D\mathcal M\), \(D\mathcal S\), and \(\mathcal E\) change along \(a+t h\).

## Previous Loss Taxonomy

The older three-loss notation was:

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_q=\|\Delta f\|_q,
\]

\[
L_2(\delta)=\|f(x+\delta)-j(x)\|_q=\|b+\Delta f\|_q,
\]

\[
L_3(\delta)=\|f(x+\delta)-j(x+\delta)\|_q
=\|b+\Delta f-\Delta j\|_q
=\|e(x+\delta)\|_q.
\]

The diagnostic residual-increment quantity was:

\[
\frac{\|e(x+\delta)-e(x)\|_q}{\|\delta\|_p+\eta}
=
\frac{\|\Delta f-\Delta j\|_q}{\|\delta\|_p+\eta}.
\]

This residual-increment diagnostic is the one most directly tied to \(\|J_f-J_j\|_{p\to q}\). The original loss3 endpoint objective is different because it keeps the clean residual \(b=e(x)\).

## What Was Equivalent

Observed from `docs/loss3_p_norm_method_equivalence_rules_20260518.md`, `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`, and `docs/pq_equivalence_continuous_vs_binary_summary_20260529.md`:

- For \(p=2\), the \(L^2\)-steepest direction equals the normalized raw gradient:

\[
s_2(g)=\frac{g}{\|g\|_2}.
\]

- Therefore several optimizer labels collapsed to identical update formulas in the p2q2 runs:

\[
\text{unit\_raw\_add}
=
\text{steepest\_add}
=
\text{power\_add\_\_objective\_gradient},
\]

\[
\text{raw\_replace}
=
\text{steepest\_replace}
=
\text{power\_replace\_\_objective\_gradient}.
\]

- In the strict binary Darcy case, raw and steepest flip rankings can collapse because every legal pixel flip has the same magnitude.

These are optimizer-method equivalences. They are not saying that loss1, loss2, loss3, endpoint error, and residual-change are mathematically the same objective.

## What Was Not Equivalent

Observed from `docs/loss_gradient_direction_vs_svd_direction_20260515.md`, `docs/local_jacobian_svd_direction_taxonomy_20260515.md`, and `docs/analytic_solution_hierarchy_for_delta_objectives_20260516.md`:

For squared \(L^2\) local objectives,

\[
L_{1,sq}(\delta)\approx\|J_f\delta\|_2^2,
\qquad
\nabla L_{1,sq}=2J_f^TJ_f\delta.
\]

\[
L_{2,sq}(\delta)\approx\|b+J_f\delta\|_2^2,
\qquad
\nabla L_{2,sq}=2J_f^Tb+2J_f^TJ_f\delta.
\]

\[
L_{3,sq}(\delta)\approx\|b+(J_f-J_j)\delta\|_2^2,
\qquad
\nabla L_{3,sq}=2(J_f-J_j)^Tb+2(J_f-J_j)^T(J_f-J_j)\delta.
\]

Thus:

- The SVD direction of \(J_f-J_j\) solves the pure homogeneous problem \(\max_{\|v\|=1}\|(J_f-J_j)v\|\).
- The endpoint loss3 gradient contains the extra outward term \((J_f-J_j)^Tb\).
- The practical finite-radius PGD objective can further deviate because the local Jacobian changes along the path.

This is the same caveat as in the current error-operator framework.

## Direct Gradient-Angle Experiment

Observed from `docs/fno_nu0p001_loss_gradient_path_result_20260515.md` and `docs/three_loss_pairwise_gradient_angles_result_20260516.md`.

The experiment recomputed nonlinear autograd gradients for loss1, loss2, and loss3 at the same saved perturbation points on five Burgers/FNO initial conditions.

For nonzero saved steps \(k\ge5\), across 150 points:

| gradient pair | mean cosine | mean angle |
|---|---:|---:|
| \(g_1\) vs \(g_2\) | `0.9904` | `3.91 deg` |
| \(g_1\) vs \(g_3\) | `0.5269` | `56.51 deg` |
| \(g_2\) vs \(g_3\) | `0.5167` | `57.23 deg` |

By attack trajectory, excluding \(k=0\):

| delta source | n | L1-L2 angle | L1-L3 angle | L2-L3 angle | budget ratio |
|---|---:|---:|---:|---:|---:|
| `loss1_original_pgd` | 50 | `1.01 (0.41) deg` | `55.79 (15.27) deg` | `55.98 (15.20) deg` | `0.85 (0.23)` |
| `loss2_original_pgd` | 50 | `1.12 (0.41) deg` | `52.44 (17.99) deg` | `52.67 (17.97) deg` | `0.85 (0.23)` |
| `loss3_original_pgd` | 50 | `9.60 (10.04) deg` | `61.30 (20.48) deg` | `63.05 (20.16) deg` | `0.41 (0.31)` |

Observed conclusion:

- loss1 and loss2 were almost the same direction in this FNO/Burgers trajectory experiment.
- loss3 was a substantially different direction, usually about 50-60 degrees away from loss1/loss2.

Framework interpretation:

- loss1/loss2 mostly use the model response against a fixed clean reference.
- loss3 differentiates the moving solver target and corresponds to endpoint error energy \(\|e(x+\delta)\|\).
- The extra \(A^Tb\) term and the moving solver response explain why loss3 is not the same direction as loss1/loss2 or pure residual SVD.

## SVD Direction Versus Outward-Growth Direction

Observed from `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`.

The optimized direction experiment compared four objectives:

\[
L_f=\frac{\|f(x+\epsilon v)-f(x)\|}{\epsilon},
\]

\[
L_j=\frac{\|j(x+\epsilon v)-j(x)\|}{\epsilon},
\]

\[
L_e=\frac{\|e(x+\epsilon v)-e(x)\|}{\epsilon},
\]

\[
G_e=\frac{\|e(x+\epsilon v)\|-\|e(x)\|}{\epsilon}.
\]

The result was:

- \(L_f\), \(L_j\), and \(L_e\) recover top singular-vector directions at small \(\epsilon\).
- \(G_e\) recovers the outward-growth direction

\[
\frac{J_e^T(e/\|e\|)}{\|J_e^T(e/\|e\|)\|},
\]

not the top singular vector of \(J_e\).

This is exactly the distinction between error-field movement and endpoint error growth.

## Burgers p2q2 Self-Training Mechanism Evidence

Observed from `docs/burgers_p2q2_loss123_offmanifold_gradient_input_similarity_report_20260604.md`.

The later Burgers p2q2 self-training audit found that raw loss3 was not simply "more correct". Under the same RMS-L2 budget it produced peakier, more off-range adversarial inputs.

### Quick Parameter-Gradient Alignment

Positive cosine means a small adversarial-training update should reduce that eval loss to first order. Negative means it should increase it.

| variant | clean train | clean test | generalization |
|---|---:|---:|---:|
| loss1 | `0.975458` | `0.841490` | `0.354882` |
| loss2 | `0.968047` | `0.818420` | `0.339495` |
| loss3 raw | `0.235095` | `0.073322` | `-0.139242` |
| loss3 clip to `[0,1]` | `0.963363` | `0.947284` | `0.599662` |
| loss3 lowpass32 RMS + clip | `0.964202` | `0.948056` | `0.599222` |

### Attack Geometry

| variant | delta RMS-L2 | delta Linf | Linf/RMS | high-frequency ratio | x_adv min | x_adv max | OOB max |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss1 | `0.060000` | `0.099241` | `1.654010` | `1.706e-07` | `-0.107669` | `1.106122` | `0.062427` |
| loss2 | `0.060000` | `0.100418` | `1.673641` | `1.810e-07` | `-0.113477` | `1.081268` | `0.059919` |
| loss3 raw | `0.060000` | `0.290831` | `4.847176` | `3.471e-05` | `-0.303080` | `1.255627` | `0.172071` |

### Epoch-1 Replay

| variant | split | RMSE before | RMSE after | RMSE ratio |
|---|---|---:|---:|---:|
| loss1 | generalization | `0.017073` | `0.009742` | `0.570625` |
| loss2 | generalization | `0.017073` | `0.014452` | `0.846477` |
| loss3 raw | generalization | `0.017073` | `0.022311` | `1.306785` |
| loss3 clip01 | generalization | `0.017073` | `0.010765` | `0.630511` |

Observed conclusion:

- Raw loss3 adversarial samples caused a first-epoch clean/generalization jump.
- Clipping loss3 samples back to the clean input range removed the jump.
- Lowpass alone was not enough in the quick cosine probe; range/off-manifold violation was critical.

Framework interpretation:

- The loss3 endpoint-energy objective is theoretically targeted at true model-solver error, but at finite radius it can choose directions that are too peak-like or off-manifold.
- The parameter update induced by those samples can be poorly aligned with train/test/generalization loss, even if the attack objective is mathematically meaningful.

## Solver/Jacobian Alignment Evidence

Observed from `docs/burgers_p2q2_loss123_solver_alignment_report_20260603.md`.

For the available p2q2 checkpoints, loss1 epoch2000 had the best measured model-solver Jacobian error spectral norm:

| model | epoch | all | train | test | generalization |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | `2.377762` | `1.508001` | `0.769029` | `3.543111` |
| loss1 | 2000 | `0.558097` | `0.302427` | `0.290937` | `0.818364` |
| loss2 | 900 | `0.707897` | `0.526293` | `0.446145` | `0.921560` |
| loss3 | 800 | `0.755361` | `0.532506` | `0.457528` | `1.008207` |
| loss3 | 1000 | `0.915005` | `0.783163` | `0.646639` | `1.101456` |

Observed conclusion:

- In that benchmark, loss1 was the best measured checkpoint by prediction loss, Jacobian-error norm, and singular-vector alignment.
- This does not prove loss1 is universally better. It shows that raw loss3 can fail when the generated adversarial data are poorly matched to the intended generalization distribution.

## How This Fits The Current Paper Argument

The clean theoretical statement should be:

\[
\text{local spectral norm robustness}
\quad\Longleftrightarrow\quad
\text{infinitesimal worst-case error-field movement}.
\]

That is:

\[
\lim_{\epsilon\to0}
\frac{
\sup_{\|h\|\le\epsilon}
\|\mathcal E(a+h)-\mathcal E(a)\|
}{\epsilon}
=
\|D\mathcal E(a)\|.
\]

But the practical attack loss used in self-training is often:

\[
\|\mathcal E(a+h)\|^2,
\]

whose local expansion is:

\[
\|\mathcal E(a+h)\|^2
\approx
\|e_0\|^2
+2\langle e_0,D\mathcal E(a)h\rangle
+\|D\mathcal E(a)h\|^2.
\]

So the practical PGD robustness metric and the Jacobian spectral-norm metric agree only under extra conditions:

- \(p=q=2\) or another fixed induced-norm setting is specified;
- \(\epsilon\) is small enough for local linearization;
- \(D\mathcal E(a)^*e_0\) is zero or negligible, or the analysis explicitly removes the clean residual by studying error-field change;
- finite-radius path rotation and off-manifold effects are controlled.

The older experiments are therefore not contradictory. They show the hierarchy:

1. Pure residual movement \(\|e(a+h)-e(a)\|\) is the SVD/local-Lipschitz object.
2. Endpoint error growth \(\|e(a+h)\|\) adds \(D\mathcal E(a)^*e_0\).
3. Finite-radius PGD adds nonlinear path and data-manifold effects.
4. Self-training adds a parameter-gradient alignment question: the attacked batch must produce a model-parameter update aligned with train/test/generalization improvement.

## Evidence Status

Observed evidence comes from existing Markdown/CSV records; no new GPU experiment was run for this note.

Primary source records:

- `docs/delta_loss_formula_taxonomy_20260515.md`
- `docs/loss_gradient_direction_vs_svd_direction_20260515.md`
- `docs/local_jacobian_svd_direction_taxonomy_20260515.md`
- `docs/analytic_solution_hierarchy_for_delta_objectives_20260516.md`
- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`
- `docs/three_loss_pairwise_gradient_angles_result_20260516.md`
- `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`
- `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`
- `docs/loss3_p_norm_method_equivalence_rules_20260518.md`
- `docs/pq_equivalence_continuous_vs_binary_summary_20260529.md`
- `docs/burgers_p2q2_loss123_solver_alignment_report_20260603.md`
- `docs/burgers_p2q2_loss123_offmanifold_gradient_input_similarity_report_20260604.md`

Inference:

- The prior "some things are equivalent" result mainly referred to optimizer-update labels under \(p=2\), not to all loss objectives.
- The prior "directions are not the same" result is exactly explained by the endpoint-error residual term and finite-radius solver/model path effects.
- The current paper framing should explicitly separate local error-field Lipschitz robustness from endpoint error-energy PGD robustness, then describe conditions under which they agree.
