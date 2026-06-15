# Darcy CFlow SVD/Jacobian Norm Metric Definitions - 2026-06-15

Observed from:

- `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_jacobian_25sample_7model_mean_std.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_block2_top10_singular_values_raw.csv`
- `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_block2_top10_singular_values_by_model_mean_std.csv`
- Generation code: `tools/darcy_sir20_robustness.py`

This is the final seven-model Darcy CFlow robustness/SVD run on
`generalization_datasets_darcy_binary_loss3targeted_20260611`, not the older
`lossdrop50_selected_20260607` data.

## Definitions

- `error_l2_norm`: `||model(x) - y||_2` on the flattened 85x85 Darcy output.
  This is the clean residual vector norm, not RMSE and not Relative L2.
- `jt_error_l2_norm`: `||J^T error||_2`, where `J` is the local Jacobian of the
  trained model output with respect to the input coefficient field, and
  `error = model(x) - y`. This is the input-gradient norm of
  `0.5 * ||model(x) - y||_2^2`.
- `block2_sigma1`: the exact top singular value of the block/2 projected model
  Jacobian `P_out J_model L_in`. This is the spectral norm of that projected
  model Jacobian, not the solver Jacobian.
- Important non-equivalence: `block2_sigma1` is not the spectral norm of the
  model-minus-solver error operator. It is not
  `||P_out (J_model - J_solver) L_in||_2`.
- `block2_top_singular_values_json`: the top K singular values of the same
  block/2 projected model Jacobian. In this run K = 10.
- `sigma_input_right`: a full-input-space one-step power-refined estimate that
  starts from the block/2 top right singular vector lifted back to the 85x85
  input grid. It is still a model-Jacobian estimate, not a solver-Jacobian value.
- `input_right_singular_vector`: the refined full-input direction used for
  `sigma_input_right`.
- `top_right_singular_vector_basis_full`: the top K block/2 right singular
  vectors lifted to the full 85x85 input grid.

Current coverage:

- Full attack table: 7 models x 52 datasets x 50 samples, attack steps = 50.
- SVD/Jacobian table: 7 models x 25 fixed samples.
- Exact full 85x85 model-Jacobian SVD top K for all 175 model/sample pairs is
  not present in the current final table. The current final table contains
  exact block/2 top10 singular values plus the full-input one-step estimate.
- Exact block/2 or full SVD of the residual/error Jacobian
  `J_model - J_solver` is not present in the current final table.

## Key 25-sample SVD/Jacobian Means

| model | attack loss increase mean | error L2 norm mean | J^T error L2 norm mean | full-input sigma estimate mean | block/2 spectral norm mean |
|---|---:|---:|---:|---:|---:|
| baseline | 8.945459e-06 | 8.383405e-02 | 9.292769e-05 | 1.271591e-03 | 1.249747e-03 |
| loss1 | 3.649368e-06 | 6.093169e-02 | 9.922099e-05 | 1.821415e-03 | 1.780896e-03 |
| loss2 | 3.392906e-06 | 5.767659e-02 | 9.225366e-05 | 1.750792e-03 | 1.712121e-03 |
| loss3 | 1.716178e-06 | 3.880665e-02 | 6.156802e-05 | 2.353892e-03 | 2.288896e-03 |
| Physics Loss | 3.713495e-06 | 6.103865e-02 | 1.033963e-04 | 1.855503e-03 | 1.817496e-03 |
| random clean | 3.452931e-06 | 5.932469e-02 | 9.111708e-05 | 1.724926e-03 | 1.688201e-03 |
| random solver | 4.439375e-06 | 6.997601e-02 | 1.235702e-04 | 1.898284e-03 | 1.859635e-03 |

Inference from the observed table:

- Loss3 is best on attack loss increase, residual error norm, and J^T error norm.
- Loss3 is not smallest on the singular-value/spectral-norm metrics. In fact,
  Loss3 has the largest block/2 top singular value and the largest full-input
  sigma estimate in this 25-sample table.
- Therefore the singular value alone does not explain the final robustness. The
  better explanation from the recorded metrics is that Loss3 has much smaller
  clean residual and smaller `J^T error`, so the attack objective grows less even
  though one high-gain local direction exists.
- If the scientific question is whether adversarial/self-training reduces the
  sensitivity of the model-solver residual operator, the missing quantity is
  `||J_model - J_solver||_2` on the same samples. The current `block2_sigma1`
  table cannot answer that question.

## Block/2 Top10 Singular Value Means

| model | top1 | top2 | top3 | top4 | top5 | top6 | top7 | top8 | top9 | top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.249747e-03 | 9.824296e-04 | 8.841152e-04 | 6.773464e-04 | 6.175931e-04 | 5.576520e-04 | 4.964588e-04 | 4.641704e-04 | 4.386582e-04 | 4.170655e-04 |
| loss1 | 1.780896e-03 | 1.101429e-03 | 9.764174e-04 | 7.687541e-04 | 6.904222e-04 | 6.178762e-04 | 5.375126e-04 | 5.065992e-04 | 4.764648e-04 | 4.497192e-04 |
| loss2 | 1.712121e-03 | 1.083849e-03 | 9.674163e-04 | 7.830737e-04 | 7.053185e-04 | 6.254393e-04 | 5.585695e-04 | 5.259650e-04 | 4.932760e-04 | 4.585394e-04 |
| loss3 | 2.288896e-03 | 1.304689e-03 | 1.114734e-03 | 9.204764e-04 | 8.156751e-04 | 7.144567e-04 | 6.185912e-04 | 5.789749e-04 | 5.443050e-04 | 5.053840e-04 |
| Physics Loss | 1.817496e-03 | 1.143127e-03 | 1.021387e-03 | 8.124513e-04 | 7.351632e-04 | 6.579338e-04 | 5.774547e-04 | 5.432021e-04 | 5.098521e-04 | 4.776956e-04 |
| random clean | 1.688201e-03 | 1.069810e-03 | 9.672028e-04 | 7.519530e-04 | 6.866776e-04 | 6.145188e-04 | 5.306463e-04 | 4.967945e-04 | 4.670373e-04 | 4.411617e-04 |
| random solver | 1.859635e-03 | 1.153770e-03 | 1.018591e-03 | 8.157449e-04 | 7.315738e-04 | 6.544638e-04 | 5.774218e-04 | 5.424741e-04 | 5.079971e-04 | 4.751551e-04 |
