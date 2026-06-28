# Darcy CFlow Model-vs-Solver Block/2 Singular Subspace Similarity

Observed from final seven-model SVD/Jacobian samples.

- Source model SVD CSV: `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv`
- Solver SVD CSV: `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/solver_block2_svd_25samples.csv`
- Similarity CSV: `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_25samples_7models.csv`
- By-model summary CSV: `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_by_model.csv`
- JAX backend: `gpu`
- Samples: `25`
- Model/sample rows: `175`

Best right top-10 subspace overlap: `loss3` mean projection Frobenius cosine `0.748112`.
Best left top-10 subspace overlap: `loss3` mean projection Frobenius cosine `0.855752`.

| method | right top10 overlap mean | right top10 angle mean | right top1 abs cos | left top10 overlap mean | left top10 angle mean | left top1 abs cos | model/solver sigma1 ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.691233 | 47.1266 | 0.783532 | 0.828165 | 31.6912 | 0.888507 | 0.384774 |
| loss1 | 0.724635 | 43.9209 | 0.851104 | 0.846535 | 29.1698 | 0.941791 | 0.540275 |
| loss2 | 0.731854 | 43.196 | 0.858757 | 0.849315 | 28.8507 | 0.941911 | 0.519144 |
| loss3 | 0.748112 | 41.5617 | 0.897148 | 0.855752 | 28.0259 | 0.958681 | 0.689899 |
| Physics Loss | 0.729438 | 43.505 | 0.851315 | 0.852599 | 28.4792 | 0.944366 | 0.550961 |
| random clean | 0.725535 | 43.8452 | 0.834905 | 0.851256 | 28.5615 | 0.935917 | 0.512506 |
| random solver | 0.730106 | 43.4611 | 0.857035 | 0.850569 | 28.7407 | 0.945905 | 0.563351 |