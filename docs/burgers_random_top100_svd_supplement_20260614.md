# Burgers Random Top100 SVD Supplement, 20260614

This supplement fixes the previous top50/top100 coverage gap for
`random_clean_y` and `random_solver_y` by deriving top100 SVD values/vectors
from already-stored `1024 x 1024` Jacobian matrices in the completed
`burgers_random_solver7860_clean8000_full_suite_20260614` artifact.

No training, attack generation, model forward pass, or Jacobian generation was
rerun. The only computation performed here is SVD of already-saved Jacobian
matrices.

## Summary

```json
{
  "device": "cuda",
  "random_jacobian_kinds": [
    "solver",
    "random_clean_y_model",
    "random_clean_y_error",
    "random_solver_y_model",
    "random_solver_y_error"
  ],
  "samples": 25,
  "random_summary_rows": 125,
  "random_long_rows": 12500,
  "six_model_error_ranked_rows": 15000,
  "six_model_error_topk_ranked_rows": 1800,
  "six_model_subspace_ranked_rows": 1500
}
```

## Error Spectrum Top50/Top100 Model Means

| scope | metric | direction | model | n | mean | std | median | min | max | rank | is_best | best_model | runner_up_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topk | error_singular_value_top100_l2 | lower | loss3 | 25 | 1.89766 | 1.01017 | 1.78648 | 0.777422 | 4.84877 | 1 | True | loss3 | random_solver_y |
| topk | error_singular_value_top100_l2 | lower | random_solver_y | 25 | 2.51191 | 1.74928 | 2.03522 | 0.212126 | 5.84759 | 2 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_l2 | lower | loss1 | 25 | 2.75929 | 1.68388 | 2.4662 | 0.545398 | 5.76057 | 3 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_l2 | lower | loss2 | 25 | 2.8739 | 1.6395 | 2.81691 | 0.612307 | 6.13542 | 4 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_l2 | lower | baseline | 25 | 3.55296 | 1.76654 | 3.38828 | 1.01204 | 7.45834 | 5 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_l2 | lower | random_clean_y | 25 | 7.22462 | 1.15856 | 7.02007 | 5.55983 | 10.831 | 6 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_mean | lower | loss3 | 25 | 0.0652985 | 0.0309275 | 0.0657082 | 0.0234765 | 0.129817 | 1 | True | loss3 | random_solver_y |
| topk | error_singular_value_top100_mean | lower | random_solver_y | 25 | 0.0894098 | 0.0561006 | 0.0881705 | 0.0117397 | 0.181269 | 2 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_mean | lower | loss2 | 25 | 0.106333 | 0.0483232 | 0.0977075 | 0.0355461 | 0.188544 | 3 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_mean | lower | loss1 | 25 | 0.106803 | 0.0509023 | 0.0989816 | 0.0357027 | 0.188312 | 4 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_mean | lower | baseline | 25 | 0.131025 | 0.042551 | 0.133088 | 0.0614227 | 0.205836 | 5 | False | loss3 | random_solver_y |
| topk | error_singular_value_top100_mean | lower | random_clean_y | 25 | 0.2518 | 0.0360966 | 0.247053 | 0.182162 | 0.32416 | 6 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_l2 | lower | loss3 | 25 | 1.89714 | 1.01015 | 1.78598 | 0.777002 | 4.84833 | 1 | True | loss3 | random_solver_y |
| topk | error_singular_value_top50_l2 | lower | random_solver_y | 25 | 2.5116 | 1.74947 | 2.03504 | 0.21146 | 5.84747 | 2 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_l2 | lower | loss1 | 25 | 2.75388 | 1.68712 | 2.46388 | 0.529238 | 5.75795 | 3 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_l2 | lower | loss2 | 25 | 2.87092 | 1.64103 | 2.81505 | 0.606289 | 6.13399 | 4 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_l2 | lower | baseline | 25 | 3.54636 | 1.76942 | 3.38357 | 0.990825 | 7.45712 | 5 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_l2 | lower | random_clean_y | 25 | 7.2237 | 1.15843 | 7.01913 | 5.55949 | 10.8298 | 6 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_mean | lower | loss3 | 25 | 0.124608 | 0.0602225 | 0.125044 | 0.0432604 | 0.250546 | 1 | True | loss3 | random_solver_y |
| topk | error_singular_value_top50_mean | lower | random_solver_y | 25 | 0.174831 | 0.111341 | 0.173478 | 0.0209493 | 0.356956 | 2 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_mean | lower | loss1 | 25 | 0.194545 | 0.100124 | 0.182851 | 0.0561273 | 0.352434 | 3 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_mean | lower | loss2 | 25 | 0.19735 | 0.0948324 | 0.185319 | 0.0589312 | 0.358402 | 4 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_mean | lower | baseline | 25 | 0.235788 | 0.0831648 | 0.235737 | 0.100235 | 0.392633 | 5 | False | loss3 | random_solver_y |
| topk | error_singular_value_top50_mean | lower | random_clean_y | 25 | 0.488568 | 0.0689275 | 0.478877 | 0.355963 | 0.626031 | 6 | False | loss3 | random_solver_y |

## Model-Solver Top50/Top100 Subspace Model Means

| scope | metric | direction | model | n | mean | std | median | min | max | rank | is_best | best_model | runner_up_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_solver_subspace | model_solver_top100_left_subspace_mean_cos | higher | random_solver_y | 25 | 0.57089 | 0.0335362 | 0.567756 | 0.499819 | 0.653416 | 1 | True | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_left_subspace_mean_cos | higher | loss3 | 25 | 0.558577 | 0.0515534 | 0.55715 | 0.471 | 0.686113 | 2 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_left_subspace_mean_cos | higher | loss2 | 25 | 0.534828 | 0.0671309 | 0.538935 | 0.400893 | 0.672005 | 3 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_left_subspace_mean_cos | higher | loss1 | 25 | 0.523989 | 0.0686576 | 0.533462 | 0.395943 | 0.671661 | 4 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_left_subspace_mean_cos | higher | baseline | 25 | 0.491911 | 0.0840125 | 0.507534 | 0.350132 | 0.643478 | 5 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_left_subspace_mean_cos | higher | random_clean_y | 25 | 0.482822 | 0.0392171 | 0.481437 | 0.405693 | 0.553443 | 6 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_right_subspace_mean_cos | higher | random_solver_y | 25 | 0.460405 | 0.0196252 | 0.46216 | 0.422623 | 0.502361 | 1 | True | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_right_subspace_mean_cos | higher | loss3 | 25 | 0.449667 | 0.0253548 | 0.447491 | 0.393737 | 0.5037 | 2 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_right_subspace_mean_cos | higher | random_clean_y | 25 | 0.4239 | 0.0335682 | 0.429739 | 0.341655 | 0.466726 | 3 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_right_subspace_mean_cos | higher | loss2 | 25 | 0.332409 | 0.0242211 | 0.329949 | 0.278564 | 0.405282 | 4 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_right_subspace_mean_cos | higher | loss1 | 25 | 0.320621 | 0.0195554 | 0.320694 | 0.290163 | 0.370035 | 5 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top100_right_subspace_mean_cos | higher | baseline | 25 | 0.313428 | 0.0241436 | 0.305755 | 0.277408 | 0.381481 | 6 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top50_left_subspace_mean_cos | higher | random_solver_y | 25 | 0.754114 | 0.0602137 | 0.759628 | 0.631718 | 0.875367 | 1 | True | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top50_left_subspace_mean_cos | higher | loss3 | 25 | 0.733096 | 0.0434972 | 0.728917 | 0.640714 | 0.825545 | 2 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top50_left_subspace_mean_cos | higher | random_clean_y | 25 | 0.592639 | 0.0724299 | 0.594875 | 0.420532 | 0.717994 | 3 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top50_left_subspace_mean_cos | higher | loss2 | 25 | 0.518723 | 0.0462219 | 0.521475 | 0.44269 | 0.609048 | 4 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top50_left_subspace_mean_cos | higher | loss1 | 25 | 0.496381 | 0.0473654 | 0.495982 | 0.419661 | 0.581016 | 5 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top50_left_subspace_mean_cos | higher | baseline | 25 | 0.463232 | 0.0432168 | 0.451467 | 0.402908 | 0.55708 | 6 | False | random_solver_y | loss3 |
| model_solver_subspace | model_solver_top50_right_subspace_mean_cos | higher | loss3 | 25 | 0.59916 | 0.044397 | 0.596831 | 0.503577 | 0.677087 | 1 | True | loss3 | random_solver_y |
| model_solver_subspace | model_solver_top50_right_subspace_mean_cos | higher | random_solver_y | 25 | 0.589964 | 0.0376151 | 0.593786 | 0.514927 | 0.658033 | 2 | False | loss3 | random_solver_y |
| model_solver_subspace | model_solver_top50_right_subspace_mean_cos | higher | random_clean_y | 25 | 0.489881 | 0.071204 | 0.496374 | 0.329355 | 0.606187 | 3 | False | loss3 | random_solver_y |
| model_solver_subspace | model_solver_top50_right_subspace_mean_cos | higher | loss2 | 25 | 0.425659 | 0.0593034 | 0.416996 | 0.325174 | 0.551532 | 4 | False | loss3 | random_solver_y |
| model_solver_subspace | model_solver_top50_right_subspace_mean_cos | higher | loss1 | 25 | 0.411351 | 0.0542382 | 0.399106 | 0.318644 | 0.524586 | 5 | False | loss3 | random_solver_y |
| model_solver_subspace | model_solver_top50_right_subspace_mean_cos | higher | baseline | 25 | 0.392146 | 0.0593444 | 0.378855 | 0.30666 | 0.52919 | 6 | False | loss3 | random_solver_y |

## Output Files

- `data/random_top100_svd_supplement_20260614/random_top100_jacobian_svd_summary.csv`
- `data/random_top100_svd_supplement_20260614/random_top100_singular_values_long.csv`
- `data/random_top100_svd_supplement_20260614/random_top100_model_solver_subspace.csv`
- `data/random_top100_svd_supplement_20260614/six_model_error_singular_values_top100_ranked_long.csv`
- `data/random_top100_svd_supplement_20260614/six_model_error_singular_values_topk_ranked_long.csv`
- `data/random_top100_svd_supplement_20260614/six_model_top50_top100_subspace_ranked_long.csv`
- `data/random_top100_svd_supplement_20260614/six_model_error_singular_values_model_summary.csv`
- `data/random_top100_svd_supplement_20260614/six_model_top50_top100_subspace_model_summary.csv`
- `data/random_top100_svd_supplement_20260614/loss3_vs_other_top100_svd_tests.csv`
- `data/random_top100_svd_supplement_20260614/loss3_vs_other_top50_top100_subspace_tests.csv`
