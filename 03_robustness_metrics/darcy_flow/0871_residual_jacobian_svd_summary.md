# Darcy CFlow Residual Jacobian SVD Summary

Operator: `J_model - J_solver` on block/2 projected input-output space.

- Rows CSV: `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_svd_25samples_7models.csv`
- Summary CSV: `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_svd_by_model.csv`
- Residual vector dir: `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_vectors`
- Solver Jacobian cache dir: `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/solver_block2_jacobians`
- JAX backend: `gpu`

| method | residual sigma1 mean | residual error norm mean | residual JT norm mean | residual top10 vs JT angle | residual top10 vs delta angle | residual JT vs delta angle |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.00246356 | 0.073329 | 0.000193139 | 13.998 | 76.0416 | 83.4014 |
| loss1 | 0.00207122 | 0.0513343 | 0.000114247 | 15.2341 | 75.2805 | 80.0533 |
| loss2 | 0.00207742 | 0.0481466 | 0.000106839 | 15.4431 | 77.059 | 81.3309 |
| loss3 | 0.00170213 | 0.0334209 | 4.72575e-05 | 20.0137 | 71.497 | 80.2697 |
| Physics Loss | 0.00204244 | 0.0508341 | 0.000112514 | 15.7262 | 76.8222 | 81.3293 |
| random clean | 0.002143 | 0.0503255 | 0.000113273 | 15.1501 | 76.2319 | 81.0739 |
| random solver | 0.00200877 | 0.0593998 | 0.000131033 | 15.642 | 76.8983 | 81.4494 |