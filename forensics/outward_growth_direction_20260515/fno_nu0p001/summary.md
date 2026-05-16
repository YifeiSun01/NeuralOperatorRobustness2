# FNO nu=0.001 Outward-Growth Direction Summary

Observed from this run:

- Output directory: `forensics/outward_growth_direction_20260515/fno_nu0p001`
- Jacobian source root: `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed`
- Indices: `[0, 7, 40, 47, 115]`
- Top-k singular directions recorded: `8`
- Random directions per sample: `64`
- Finite-difference radii: `[0.0001, 0.001, 0.01]`

This experiment adds the missing Experiment 2 row:

```text
v_growth = normalize((J_f - J_j)^T (b / ||b||_2))
```


## Conclusion

Observed from this completed FNO `nu=0.001` run:

- All five samples `0, 7, 40, 47, 115` completed and none had a degenerate
  clean-residual outward direction.
- `outward_growth` has outward component mean `0.166514`, much larger than
  `error_top` mean `0.0140571`, `fno_top` mean `0.00218766`, `solver_top` mean
  `-0.00819025`, and `random_best_by_outward_component` mean `0.0114951`.
- `error_top` has the larger mismatch-gain mean, `0.412169`, while
  `outward_growth` has mismatch-gain mean `0.368053`.
- Finite-difference checks match the linear outward prediction: for
  `outward_growth`, actual loss3 growth mean is `0.167286` at `rho=1e-4` and
  `0.166668` at `rho=1e-3`, versus predicted `0.166514`; the negative control
  gives approximately the opposite sign.

Inference from these observations:

- The missing Experiment 2 row is now completed for FNO `nu=0.001`.
- The top residual-Lipschitz direction `v_e*` and the clean-risk outward
  direction `v_growth* = normalize(A^T b/||b||)` are not the same diagnostic:
  one maximizes residual movement `||A v||`, while the other maximizes first-order
  increase of the current error norm.
- This supports the planned distinction between local mismatch movement, local
  outward risk growth, and finite-radius `loss3_original` endpoint attack.


## Plain-Language Interpretation

中文解释：这个结果主要说明三件事。

1. `error_top` / `v_e*` 和 `outward_growth` / `A^T b` 不是同一个东西。
   `error_top` 让误差场移动得最多，也就是 `||A v||` 最大；但是它不一定让
   当前误差范数 `||e(x)||` 立刻变大。这里 `error_top` 的 mismatch gain mean
   是 `0.412169`，比 `outward_growth` 的 `0.368053` 更大，但它的 outward
   component mean 只有 `0.0140571`。

2. `outward_growth` 找到的是“把当前 clean residual 往外推”的方向。它的
   outward component mean 是 `0.166514`，远大于 `fno_top`、`solver_top`、
   `error_top` 和 random-best directions。这说明 `A^T b` 这个方向确实补上了
   Experiment 2 里缺的 local outward risk growth 诊断。

3. 这个局部解释是可信的，因为 small-radius finite-difference 对上了线性预测。
   对 `outward_growth`，预测增长是 `0.166514`，实际增长在 `rho=1e-4` 是
   `0.167286`，在 `rho=1e-3` 是 `0.166668`；负方向给出相反符号。

因此，这个实验不是证明 `outward_growth` 是最终 finite-radius attack 的最优方向；
它证明的是：在局部一阶意义下，“误差场移动最大”和“当前误差范数增长最快”是两个
不同概念。这个结果支持论文主线：ratio / Lipschitz / local diagnostic 只能解释局部机制，
而真正有限半径攻击还是要看 `loss3_original` 的 endpoint error。


## Metric Glossary For The Four Key Numbers

For a unit input direction `v`, with `A = J_f - J_j` and clean residual
`b = e(x) = f(x)-j(x)`, the two reported quantities are:

\[
\text{mismatch gain}(v)=\|Av\|_2
\]

\[
\text{outward component}(v)=\left\langle \frac{b}{\|b\|_2}, Av \right\rangle
\]

The four headline numbers mean:

| number | row / column | mathematical meaning | interpretation |
|---:|---|---|---|
| `0.412169` | `error_top` mismatch gain mean | mean `||A v||_2` over the top-8 error-Jacobian singular directions across 5 samples | residual/error field movement is largest along `error_top` directions |
| `0.368053` | `outward_growth` mismatch gain mean | mean `||A v_growth||_2` across 5 samples | the outward-growth direction still moves the residual field substantially, but not as much as `error_top` |
| `0.0140571` | `error_top` outward component mean | mean `<b/||b||, A v>` over the top-8 error-Jacobian singular directions across 5 samples | `error_top` moves the residual field, but only weakly in the direction that increases the current error norm |
| `0.166514` | `outward_growth` outward component mean | mean `<b/||b||, A v_growth>` across 5 samples | `outward_growth = normalize(A^T b)` is the local direction that strongly pushes the current residual outward |

So `0.412169 > 0.368053` says `error_top` has larger raw residual movement,
but `0.166514 >> 0.0140571` says `outward_growth` has much larger local
clean-error growth.


## Squared-Loss Gradient Clarification

For the local model

\[
e(x+\delta) \approx b + A\delta,
\]

there are two related but different local objectives.

Residual movement objective:

\[
M(\delta)=\|A\delta\|_2^2=\delta^T A^T A\delta.
\]

Its gradient is

\[
\nabla_\delta M(\delta)=2A^T A\delta.
\]

This objective measures how much the error field moves, independent of the
current residual direction `b`.

Squared endpoint error objective:

\[
S(\delta)=\|b+A\delta\|_2^2.
\]

Expanding exactly gives

\[
S(\delta)=\|b\|_2^2+2b^T A\delta+\delta^T A^T A\delta.
\]

Its exact local-model gradient is

\[
\nabla_\delta S(\delta)=2A^T b+2A^T A\delta.
\]

At `delta=0`, this becomes

\[
\nabla_\delta S(0)=2A^T b.
\]

So `A^T b` is not obtained by optimizing the residual movement objective. It is
the first-step gradient of the squared endpoint error objective. Dropping the
quadratic term is only a first-order, infinitesimal-radius diagnostic. For a
finite-radius attack, the quadratic term and nonlinear higher-order terms should
not be ignored; this is why the main finite-radius objective remains
`loss3_original`.

## Direction Summary

| direction group | n | mismatch gain mean | outward component mean | D_f mean |
|---|---:|---:|---:|---:|
| fno_top | 40 | 0.337848 | 0.00218766 | 0.191833 |
| solver_top | 40 | 0.340392 | -0.00819025 | 0.192255 |
| error_top | 40 | 0.412169 | 0.0140571 | 0.329056 |
| outward_growth | 5 | 0.368053 | 0.166514 | 0.355948 |
| negative_outward_growth | 5 | 0.368053 | -0.166514 | 0.355948 |
| random | 320 | 0.0453593 | -0.000143471 | 0.258463 |
| random_best_by_outward_component | 5 | 0.047458 | 0.0114951 | 0.284052 |
| random_best_by_mismatch_gain | 5 | 0.0760439 | 0.00173444 | 0.248026 |

## Finite-Difference Summary

| direction source | rho | n | actual loss3 growth mean | predicted growth mean | residual movement mean |
|---|---:|---:|---:|---:|---:|
| error | 1.0e-04 | 40 | 0.014074 | 0.0140571 | 0.412753 |
| error | 1.0e-03 | 40 | 0.0141318 | 0.0140571 | 0.412213 |
| error | 1.0e-02 | 40 | 0.0148188 | 0.0140571 | 0.412405 |
| fno | 1.0e-04 | 40 | 0.00209312 | 0.00218766 | 0.338168 |
| fno | 1.0e-03 | 40 | 0.00224155 | 0.00218766 | 0.337836 |
| fno | 1.0e-02 | 40 | 0.00236288 | 0.00218766 | 0.337758 |
| negative_outward_growth | 1.0e-04 | 5 | -0.165528 | -0.166514 | 0.367704 |
| negative_outward_growth | 1.0e-03 | 5 | -0.166365 | -0.166514 | 0.367903 |
| negative_outward_growth | 1.0e-02 | 5 | -0.165476 | -0.166514 | 0.366964 |
| outward_growth | 1.0e-04 | 5 | 0.167286 | 0.166514 | 0.368765 |
| outward_growth | 1.0e-03 | 5 | 0.166668 | 0.166514 | 0.368181 |
| outward_growth | 1.0e-02 | 5 | 0.167548 | 0.166514 | 0.369125 |
| random_best_by_mismatch_gain | 1.0e-04 | 5 | 0.00222016 | 0.00173444 | 0.0771404 |
| random_best_by_mismatch_gain | 1.0e-03 | 5 | 0.00167079 | 0.00173444 | 0.0761168 |
| random_best_by_mismatch_gain | 1.0e-02 | 5 | 0.00177817 | 0.00173444 | 0.0760449 |
| random_best_by_outward_component | 1.0e-04 | 5 | 0.0129368 | 0.0114951 | 0.0499933 |
| random_best_by_outward_component | 1.0e-03 | 5 | 0.0115535 | 0.0114951 | 0.0475233 |
| random_best_by_outward_component | 1.0e-02 | 5 | 0.0115446 | 0.0114951 | 0.0474824 |
| solver | 1.0e-04 | 40 | -0.00801186 | -0.00819025 | 0.340758 |
| solver | 1.0e-03 | 40 | -0.00812273 | -0.00819025 | 0.340372 |
| solver | 1.0e-02 | 40 | -0.00798748 | -0.00819025 | 0.339857 |

## Files

- `config.json`
- `manifest.json`
- `all_direction_response_table.csv`
- `aggregate_direction_response_summary.csv`
- `all_direction_similarity_table.csv`
- `finite_difference_growth_table.csv`
- `finite_difference_growth_summary.csv`
- `index_*/clean_model_output.npy`
- `index_*/clean_solver_output.npy`
- `index_*/clean_residual.npy`
- `index_*/outward_growth_direction.npy`

Observed source paths:
- `/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`
- `/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_000/error/error_index0_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_000/fno/fno_index0_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_000/solver/solver_index0_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_007/error/error_index7_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_007/fno/fno_index7_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_007/solver/solver_index7_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_040/error/error_index40_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_040/fno/fno_index40_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_040/solver/solver_index40_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_047/error/error_index47_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_047/fno/fno_index47_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_047/solver/solver_index47_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_115/error/error_index115_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_115/fno/fno_index115_jacobian_svd.npz`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_115/solver/solver_index115_jacobian_svd.npz`
