# Burgers Zero-Training Jacobian/SVD Top100 Reuse Plan

This directory reuses 10 fixed input points from the completed representative20 same-point Jacobian/SVD study.

Selected source sample IDs: `0, 1, 6, 7, 10, 11, 12, 15, 18, 19`.

Reused without recomputation:

- `J_solver`
- baseline `J_model`
- baseline `J_error = J_model - J_solver`
- old clean+ADV `J_model`
- old clean+ADV `J_error`

Not reused:

- `adv_only` and `adv_only_error`, because in this new directory `adv_only` is reserved for the new 1000-epoch zero-training checkpoint.

All reused SVD files keep the dense 1024 x 1024 Jacobian and store the first 100 singular values/vectors from the old full SVD.
