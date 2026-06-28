# Burgers Attack Delta vs Error-SVD Direction Alignment

Date: 2026-06-08

## Question

Quantify whether the saved PGD attack perturbation direction is actually the
same as the top right singular vector of the error Jacobian
`J_model - J_solver`, and explain how this relates to the negative cross term in
endpoint growth.

## Short Answer

Observed from the saved artifacts: the final attack perturbation `delta` is **not
usually aligned with the single top right singular vector**. Across all `440`
matched model/sample rows, the mean absolute cosine with the top-1 right singular
vector is `0.322`, and the median top-1 angle is `76.88` degrees.

However, `delta` is also **not random relative to the SVD subspace**. The saved
top-20 right singular subspace explains `77.69%` of the final perturbation energy
on average, and top-100 explains `88.68%`. On generalization rows only, top-20
explains `84.16%`, and top-100 explains `93.13%`.

So the right interpretation is:

- the endpoint PGD perturbation is not just the top-1 spectral direction;
- it is mostly a mixture of many high singular directions;
- the top singular value still predicts damage well because it measures the
  local amplification scale, even when the final PGD direction is distributed
  across a top-k subspace.

## Mathematical Meaning

Let:

- `E(x)` be the clean model-solver residual vector;
- `DeltaE = E(x + delta) - E(x)` be the residual movement caused by attack;
- `A = J_model - J_solver` be the local error Jacobian.

The endpoint growth decomposes as:

```text
||E(x + delta)||^2 - ||E(x)||^2 = ||DeltaE||^2 + 2 <E(x), DeltaE>
```

The cross term is:

```text
2 <E(x), DeltaE>
```

If this term is negative, the residual moved a lot but partly moved against the
old clean residual direction, so endpoint growth is smaller than residual
movement energy.

Important: the top right singular vector maximizes `||A delta||`. It does **not**
maximize the outward endpoint term `<E(x), A delta>`. Therefore the top spectral
direction can have large residual movement while still contributing a negative
or weak outward component.

## What Was Measured

Observed source artifacts:

- rows: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/attack_delta_vs_error_svd_direction_rows.csv`
- summary: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/attack_delta_vs_error_svd_direction_summary.csv`
- metric correlations: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/attack_delta_vs_error_svd_direction_metric_correlations.csv`
- manifest: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/manifest.json`
- GPU preflight: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/gpu_preflight.json`
- nvidia-smi: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/nvidia_smi.txt`
- script: `tools/compute_burgers_attack_delta_svd_direction_alignment_20260608.py`

The script matched saved `final_delta_by_sample.npz` to saved error-SVD `.npz`
files by sample id and `error_spectral_norm`. All rows matched:

```text
n_rows = 440
n_missing = 0
```

Metrics:

- `abs_cos_top1`: `abs(<delta_unit, v1>)`; absolute value is used because SVD
  vector sign is arbitrary.
- `angle_top1_abs_deg`: angle corresponding to `abs_cos_top1`.
- `topk_energy`: squared projection of unit `delta` onto the saved top-k right
  singular subspace.

This is an input-space direction audit: final attack `delta` vs right singular
vectors. It does not claim to measure output-space `DeltaE` vs left singular
vectors.

## Alignment Summary

All rows:

| data/SVD group | n | top-1 abs cos mean/median | top-1 angle mean/median | top-20 energy mean | top-100 energy mean |
|---|---:|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 120 | `0.228 / 0.184` | `76.39 / 79.40` | `0.679` | `0.863` |
| second ns50 loss1 epoch2000 | 40 | `0.359 / 0.326` | `67.55 / 71.00` | `0.786` | `0.868` |
| second ns50 loss2 epoch0900 | 40 | `0.289 / 0.242` | `72.35 / 75.97` | `0.804` | `0.883` |
| round01 aligned final loss123 | 80 | `0.347 / 0.234` | `67.93 / 76.47` | `0.800` | `0.896` |
| round03 long-final loss123 | 80 | `0.357 / 0.228` | `67.51 / 76.81` | `0.842` | `0.921` |
| round03 final-extension loss123 | 80 | `0.399 / 0.295` | `64.51 / 72.85` | `0.819` | `0.890` |

Generalization rows:

| data/SVD group | n | top-1 abs cos mean/median | top-1 angle mean/median | top-20 energy mean | top-100 energy mean |
|---|---:|---:|---:|---:|---:|
| second ns50 loss3 checkpoint series | 60 | `0.269 / 0.229` | `74.03 / 76.74` | `0.725` | `0.907` |
| second ns50 loss1 epoch2000 | 20 | `0.294 / 0.273` | `72.20 / 74.13` | `0.789` | `0.868` |
| second ns50 loss2 epoch0900 | 20 | `0.369 / 0.350` | `67.79 / 69.49` | `0.821` | `0.886` |
| round01 aligned final loss123 | 40 | `0.413 / 0.315` | `63.35 / 71.65` | `0.863` | `0.935` |
| round03 long-final loss123 | 40 | `0.385 / 0.278` | `65.55 / 73.78` | `0.937` | `0.977` |
| round03 final-extension loss123 | 40 | `0.396 / 0.281` | `64.60 / 73.68` | `0.937` | `0.973` |

Aggregate over all `440` rows:

```text
abs_cos_top1 mean/median = 0.321671 / 0.227038
angle_top1_abs_deg mean/median = 69.907779 / 76.877230
top5/top20/top100 energy mean = 0.562995 / 0.776901 / 0.886816
fraction angle > 75 deg = 0.545455
```

Aggregate over `220` generalization rows:

```text
abs_cos_top1 mean/median = 0.350457 / 0.294917
angle_top1_abs_deg mean/median = 68.095717 / 72.847274
top5/top20/top100 energy mean = 0.644626 / 0.841631 / 0.931321
fraction angle > 75 deg = 0.477273
```

## Direction Metrics vs Damage Metrics

Observed from
`forensics/burgers_attack_delta_svd_direction_alignment_20260608/attack_delta_vs_error_svd_direction_metric_correlations.csv`:

All rows:

| x | y | Pearson | Spearman |
|---|---|---:|---:|
| `top20_energy` | `endpoint_growth_mse` | `0.591` | `0.745` |
| `top20_energy` | `residual_change_mse` | `0.579` | `0.736` |
| `top20_energy` | `cross_term_mse` | `-0.204` | `-0.263` |
| `error_spectral_norm` | `endpoint_growth_mse` | `0.758` | `0.719` |
| `error_spectral_norm` | `residual_change_mse` | `0.792` | `0.723` |
| `error_spectral_norm` | `cross_term_mse` | `-0.500` | `-0.450` |

Generalization rows:

| x | y | Pearson | Spearman |
|---|---|---:|---:|
| `top20_energy` | `endpoint_growth_mse` | `0.588` | `0.787` |
| `top20_energy` | `residual_change_mse` | `0.575` | `0.771` |
| `top20_energy` | `cross_term_mse` | `-0.190` | `-0.319` |
| `error_spectral_norm` | `endpoint_growth_mse` | `0.700` | `0.666` |
| `error_spectral_norm` | `residual_change_mse` | `0.749` | `0.688` |
| `error_spectral_norm` | `cross_term_mse` | `-0.502` | `-0.592` |

## Interpretation for the Paper

Observed evidence supports this phrasing:

> The error-Jacobian spectral norm is a strong predictor of attack damage, but
> the realized PGD perturbation is not generally the single top singular vector.
> It is better described as a perturbation lying mostly in a high-singular-value
> subspace. Endpoint damage is then residual movement plus an orientation term
> relative to the clean residual; this cross term is often negative and can
> partially cancel the residual movement energy.

This also explains why switching from endpoint growth to residual-change loss
only modestly improves some correlations. The endpoint metric is not unrelated;
it is the residual movement metric plus a cross term. Since the cross term can
be negative but does not completely dominate, the two damage metrics stay close
rather than becoming totally different.

## Caveat

This audit uses existing saved attack deltas and saved SVD vectors. It does not
re-run SVD, attack, or model/solver prediction. It quantifies input-space
alignment `delta` vs right singular vectors. A stricter theorem-facing audit
would additionally save or recompute output-space `DeltaE` vectors and compare
those with left singular vectors, but that was not needed for the current
question about perturbation-direction mismatch.
