# Darcy CFlow Residual Correlation Interpretation - 20260615

## Scope

This note records the interpretation of the final Darcy CFlow/SIR20 robustness
correlation results. It is based on existing final CSV/MD/JSON outputs only; no
training, attack rerun, or plotting was performed.

Admissible data:
- Generalization data: `generalization_datasets_darcy_binary_loss3targeted_20260611`.
- Final attack: `attack_steps=50`.
- Final models: baseline, loss1, loss2, loss3, Physics Loss, random clean,
  random solver.

Excluded:
- `generalization_datasets_darcy_lossdrop50_selected_20260607`.
- smoke / partial_smoke / one-epoch / attack_steps=1 results.

Key source files:
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_correlations_20260615.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_lossincrease_jt_spectral_correlations_20260615.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_by_model_25samples_20260615.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_by_model.csv`

## Main Conclusion

Loss3 is still the overall strongest model on clean generalization and attack50
robustness. The surprising part is not Loss3 itself; the surprising part is the
relative behavior of the three robustness scalars:

- residual attack loss increase,
- residual JT error norm,
- residual spectral norm / residual sigma1.

The small-epsilon derivation predicts that adversarial loss increase should be
closest to the residual gradient term:

```text
J_res^T error
```

where:

```text
J_res = J_model - J_solver
error = model(x) - solver(x)
```

In the final attack50 data, however, residual sigma1 has the largest Pearson
correlation with loss increase on the all-25 x 7-model table. This does not
contradict the derivation; it means the final attack50 result is not a pure
infinitesimal, first-order regime.

## Mathematical Interpretation

For the residual loss:

```text
L(x) = ||model(x) - solver(x)||^2
```

and a small input perturbation `delta`, the local expansion is:

```text
L(x + delta) - L(x)
  ~= 2 error^T J_res delta + ||J_res delta||^2
```

Interpretation:

| term | controlled by | expected regime |
|---|---|---|
| `2 error^T J_res delta` | `J_res^T error` / residual JT norm | infinitesimal epsilon, first-order attack |
| `||J_res delta||^2` | residual singular values / residual sigma1 | finite epsilon, multi-step attack, high-gain directions |
| full attack50 increase | both terms plus nonlinear effects | current final robustness table |

So the earlier theoretical expectation remains:

**As epsilon tends to zero, adversarial loss increase should be most directly
explained by `J_res^T error`.**

The current empirical result is:

**For finite attack50, the attack can also exploit the largest residual singular
directions, so residual sigma1 can correlate as strongly as, or more strongly
than, residual JT norm in Pearson correlation.**

## Observed Residual Correlations

All 25 fixed samples x 7 models:

| pair | Pearson | Spearman |
|---|---:|---:|
| loss increase vs residual JT norm | 0.504257 | 0.502891 |
| loss increase vs residual sigma1 | 0.561659 | 0.438184 |
| loss increase vs residual error L2 | 0.534601 | 0.495813 |
| residual JT norm vs residual sigma1 | 0.909114 | 0.945159 |

Generalization 21 fixed samples x 7 models:

| pair | Pearson | Spearman |
|---|---:|---:|
| loss increase vs residual JT norm | 0.353294 | 0.209620 |
| loss increase vs residual sigma1 | 0.346862 | 0.114519 |
| loss increase vs residual error L2 | 0.345888 | 0.201369 |
| residual JT norm vs residual sigma1 | 0.918346 | 0.916724 |

Interpretation:

- On all 25 fixed samples, residual sigma1 has the highest Pearson correlation
  with loss increase.
- On all 25 fixed samples, residual JT norm has the highest Spearman rank
  correlation among the three direct loss-increase pairs.
- On generalization-only 21 samples, residual JT norm and residual sigma1 are
  close in Pearson, but both correlations are weaker than in all-25.
- residual JT norm and residual sigma1 are strongly correlated with each other,
  so they are not independent diagnostics.

## Model-Only SVD Correction

The earlier model-only SVD result should not be used to explain residual
robustness.

| quantity | operator | interpretation |
|---|---|---|
| model-only SVD | `J_model` | model's own local sensitivity |
| model-solver subspace similarity | compare `J_model` and `J_solver` top-k subspaces | whether the model's local linear structure resembles solver |
| residual SVD | `J_model - J_solver` | direct residual robustness operator |

Important correction:

**Loss3 has the largest model-only block2 top singular value, so model-only
spectral norm alone would give the wrong robustness interpretation.**

The correct residual story is:

**Loss3 is robust because its local Jacobian subspace is most similar to the
solver's and its residual operator `J_model - J_solver` is smallest.**

## Loss3 Model-Solver Subspace Result

Observed by-model model-solver top-k similarity:

| metric | Loss3 value | Loss3 rank |
|---|---:|---:|
| right/input top10 overlap | 0.748112 | 1st |
| right/input top10 angle | 41.5617 deg | 1st / lowest |
| right/input top1 abs cosine | 0.897148 | 1st |
| left/output top10 overlap | 0.855752 | 1st |
| left/output top10 angle | 28.0259 deg | 1st / lowest |
| left/output top1 abs cosine | 0.958681 | 1st |

Conclusion:

**Loss3's model Jacobian top-k singular subspaces are the closest to the solver
Jacobian top-k singular subspaces among the seven final models.**

This supports Loss3's robustness, but it is not by itself the full robustness
argument. The full argument also requires residual SVD metrics.

## Loss3 Residual Operator Result

Observed residual by-model means on the 25 fixed samples:

| residual metric | Loss3 value | interpretation |
|---|---:|---|
| residual sigma1 mean | 0.00170213 | lowest among seven models |
| residual error L2 mean | 0.0334209 | lowest among seven models |
| residual JT norm mean | 4.7257e-05 | lowest among seven models |
| attack loss increase mean | 1.7162e-06 | lowest among seven models |

Pointwise fixed-25 wins:

| metric | Loss3 wins |
|---|---:|
| residual_block2_sigma1 | 25/25 |
| residual_error_l2_norm | 16/25 |
| residual_jt_error_l2_norm | 20/25 |
| adv_loss | 21/25 |
| loss_increase | 21/25 |

Conclusion:

**Loss3 is mean-best and majority pointwise-best for attack50 robustness, but it
is not pointwise-best on every attack sample. The only 25/25 residual metric is
residual sigma1.**

## Why The Correlation Result Is Not Exactly As Expected

Original theoretical expectation:

**For tiny epsilon, adversarial loss increase should be closest to residual
`J_res^T error`, not directly to residual singular values.**

Observed final attack50 behavior:

**For the finite-step attack50 table, residual sigma1 has slightly stronger
Pearson correlation with loss increase than residual JT norm does.**

Resolution:

The final attack is not an infinitesimal perturbation diagnostic. It is a
finite-step optimization. Therefore the attack can move toward directions of
large residual operator gain. In that setting, the second-order/high-gain term
`||J_res delta||^2` can become visible, so residual sigma1 becomes predictive.

This means:

- The small-epsilon derivation is still correct for the limiting first-order
  regime.
- The attack50 experiment measures a larger, more nonlinear, finite-step regime.
- residual JT norm and residual sigma1 should be treated as complementary
  robustness diagnostics.

## Recommended Follow-Up Diagnostic

To directly test the small-epsilon derivation, run a separate diagnostic, not a
replacement for the final attack50 table:

| diagnostic | purpose |
|---|---|
| tiny epsilon sweep | check whether correlation shifts toward residual JT norm as epsilon decreases |
| attack_steps=1 diagnostic | isolate first-step gradient behavior |
| predicted first-order gain | compare `Delta L_pred ~ epsilon * ||J_res^T error||` |
| predicted second-order gain | compare `Delta L_quad ~ epsilon^2 * sigma1^2` |
| same fixed-25 samples | keep sample alignment with residual SVD/JT table |

Expected outcome if the derivation is right:

**As epsilon becomes very small and steps approach one-step/linear behavior,
loss increase should align more strongly with residual JT norm than with
residual sigma1.**

## Final Wording For The Paper/Report

Suggested precise wording:

> The residual Jacobian analysis supports Loss3 robustness. Loss3 has the
> closest model-solver local singular subspaces and the smallest residual
> operator metrics. The only unexpected feature is that, under the finite
> attack50 protocol, attack loss increase correlates slightly more strongly in
> Pearson correlation with residual spectral norm than with residual
> `J_res^T error`. This does not contradict the infinitesimal derivation:
> the derivation predicts `J_res^T error` dominance in the small-epsilon
> first-order regime, whereas attack50 is a finite-step optimization that can
> exploit high-gain residual singular directions.

