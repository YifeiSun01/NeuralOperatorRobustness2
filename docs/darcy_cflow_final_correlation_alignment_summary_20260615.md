# Darcy CFlow Final Correlation/Alignment Summary - 2026-06-15

Status: complete.

This record answers whether the final robustness scalar correlations and vector
angle/alignment diagnostics have been summarized for the seven final Darcy CFlow
models.

## Scope

Final output bundle:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/
```

This is the final seven-model attack20/SVD25 run, not smoke output.

Models:

- baseline
- loss1
- loss2
- loss3
- Physics Loss
- random clean
- random solver

The 52-dataset x seven-model x eight-metric matrix is full-size. The
SVD/Jacobian correlations and vector angles are intentionally based on the fixed
25-sample SVD/Jacobian set requested earlier: train `2`, test `2`, and
generalization `21`.

## Files

Source files:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/svd_jacobian_metrics.csv
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/svd_attack_scalar_correlations.csv
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/vector_alignment_summary_by_model.csv
```

Digest files:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/final_scalar_correlation_digest.csv
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/final_vector_alignment_digest.csv
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/final_top_scalar_correlations.csv
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/reports/final_correlation_alignment_summary.md
```

Implementation:

```text
tools/summarize_darcy_final_correlation_alignment_20260615.py
```

## Counts

- Per-sample SVD/Jacobian rows: `175` = seven models x 25 fixed samples.
- Scalar correlation rows: `522`.
- Vector alignment summary rows: `28` = seven models x four splits
  (`all`, `generalization`, `test`, `train`).
- Generalization scalar digest rows: `7`.
- Generalization vector digest rows: `7`.

## Included Scalar Diagnostics

The per-sample SVD/Jacobian table includes:

- clean loss on the SVD/Jacobian sample
- attack loss increase on the same sample
- attack relative increase on the same sample
- error vector norm
- `J^T error` norm
- top input right singular value
- block Jacobian top singular values

The scalar-correlation table summarizes same-sample correlations among these
quantities, including:

- singular value vs attack loss increase
- singular value vs attack relative increase
- `J^T error` norm vs attack loss increase
- `J^T error` norm vs attack relative increase
- singular value vs `J^T error` norm
- block top singular value vs attack loss increase

For generalization samples, the digest values are:

| method | sigma vs increase | sigma vs relative | JT norm vs increase | JT norm vs relative | sigma vs JT norm |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0365 | -0.2053 | -0.4572 | -0.7076 | 0.7104 |
| loss1 | 0.1459 | -0.1092 | -0.0154 | -0.3124 | 0.8739 |
| loss2 | -0.3289 | -0.5020 | -0.2679 | -0.5055 | 0.8445 |
| loss3 | -0.2481 | -0.2667 | -0.4035 | -0.4690 | 0.6962 |
| Physics Loss | -0.1487 | -0.3937 | -0.2918 | -0.5112 | 0.8592 |
| random clean | 0.1323 | -0.0689 | -0.0357 | -0.2689 | 0.8448 |
| random solver | 0.0442 | -0.0943 | -0.1954 | -0.3697 | 0.7932 |

The strongest generalization Pearson correlations are between singular-value
diagnostics and `J^T error` norm, not between singular values and attack loss
increase.

## Included Vector Diagnostics

The per-sample SVD/Jacobian table includes pairwise vector comparisons among:

- top input right singular vector
- `J^T error`
- attack delta

For each pair it records cosine similarity, angle in degrees, and correlation.
It also records top-k singular subspace alignment against `J^T error` and attack
delta.

Generalization mean vector alignments:

| method | singular-delta cos | singular-delta angle | JT-delta cos | JT-delta angle | singular-JT cos | singular-JT angle | top-k delta cos | top-k delta angle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0638 | 86.3179 | -0.1373 | 97.9379 | -0.6128 | 140.9357 | 0.4560 | 62.7791 |
| loss1 | 0.0227 | 88.6912 | -0.0736 | 94.2256 | -0.6055 | 138.6014 | 0.3613 | 68.7433 |
| loss2 | 0.1148 | 83.3655 | -0.1187 | 96.8435 | -0.8868 | 161.5423 | 0.3845 | 67.2952 |
| loss3 | -0.0025 | 90.0860 | 0.1349 | 82.1173 | -0.4700 | 122.7485 | 0.3471 | 69.3567 |
| Physics Loss | 0.0385 | 87.7965 | -0.0108 | 90.6028 | -0.8840 | 160.8916 | 0.3736 | 67.9983 |
| random clean | 0.0779 | 85.5120 | -0.0858 | 94.9422 | -0.7974 | 155.2176 | 0.3806 | 67.5397 |
| random solver | 0.0743 | 85.7277 | -0.0736 | 94.2167 | -0.7342 | 143.8244 | 0.3834 | 67.3950 |

## Interpretation

- Yes, the requested robustness scalar correlations and vector
  angle/cosine/correlation diagnostics have been summarized.
- The SVD/Jacobian correlation and alignment diagnostics are for the fixed
  25-sample set, as requested for runtime control.
- Single leading-vector alignments are weak: singular-vector vs attack delta and
  `J^T error` vs attack delta are mostly near orthogonal.
- Top-k singular subspace overlap with attack delta is moderate, but not close
  to one.
- The singular-value diagnostics correlate much more strongly with `J^T error`
  norm than with final attack loss increase.
