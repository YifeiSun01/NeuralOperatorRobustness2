# Darcy CFlow Model-vs-Solver Singular Subspace Similarity - 2026-06-15

Observed from:

- Existing model-Jacobian SVD vectors:
  `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_vectors/`
- New solver-Jacobian block/2 SVD vectors:
  `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/solver_block2_svd_vectors/`
- Similarity table:
  `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_25samples_7models.csv`
- By-model summary:
  `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_by_model.csv`

This uses the final seven Darcy CFlow models and the fixed 25 SVD/Jacobian
samples from the 20260611 binary loss3-targeted dataset setup. It does not use
the old `lossdrop50_selected_20260607` data.

## What Was Computed

- Existing `J_model` block/2 top-10 singular vectors were reused.
- New `J_solver` block/2 top-10 singular vectors were computed on the same 25
  samples.
- For each model/sample, right/input and left/output top-k subspace similarities
  were computed by principal angles.
- The main scalar reported below is projection Frobenius cosine for top-10
  subspaces; higher means more similar.

This table compares `J_model` against `J_solver`. It is not yet the SVD of
`J_model - J_solver`.

## By-Model Summary

| method | right top10 overlap mean | right top10 angle mean | right top1 abs cos | left top10 overlap mean | left top10 angle mean | left top1 abs cos | model/solver sigma1 ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.691233 | 47.1266 | 0.783532 | 0.828165 | 31.6912 | 0.888507 | 0.384774 |
| loss1 | 0.724635 | 43.9209 | 0.851104 | 0.846535 | 29.1698 | 0.941791 | 0.540275 |
| loss2 | 0.731854 | 43.1960 | 0.858757 | 0.849315 | 28.8507 | 0.941911 | 0.519144 |
| loss3 | 0.748112 | 41.5617 | 0.897148 | 0.855752 | 28.0259 | 0.958681 | 0.689899 |
| Physics Loss | 0.729438 | 43.5050 | 0.851315 | 0.852599 | 28.4792 | 0.944366 | 0.550961 |
| random clean | 0.725535 | 43.8452 | 0.834905 | 0.851256 | 28.5615 | 0.935917 | 0.512506 |
| random solver | 0.730106 | 43.4611 | 0.857035 | 0.850569 | 28.7407 | 0.945905 | 0.563351 |

## Interpretation

Observed:

- Loss3 has the highest right/input top-10 model-vs-solver subspace overlap:
  `0.748112`.
- Loss3 has the smallest right/input top-10 mean principal angle:
  `41.5617` degrees.
- Loss3 has the highest right/input top-1 absolute cosine:
  `0.897148`.
- Loss3 also has the highest left/output top-10 model-vs-solver subspace
  overlap: `0.855752`.
- Loss3 has the highest left/output top-1 absolute cosine: `0.958681`.

Inference:

- Yes, in the final local block/2 SVD evidence, Loss3 is the model whose local
  singular subspace is most similar to the solver's local singular subspace.
- This is different from saying Loss3 has the smallest model spectral norm. It
  does not. Loss3 is best by subspace alignment, residual/error norm, J^T-error
  norm, and attack loss increase, while its model-Jacobian spectral norm is
  relatively large.
- The missing remaining quantity is the explicit residual-Jacobian SVD of
  `J_model - J_solver`.

