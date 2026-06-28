# Burgers Jacobian Downsample SVD Comparison

This diagnostic compares saved full `1024 x 1024` Burgers Jacobians with `512 x 512` and `256 x 256` coarse proxies.

Two coarse proxies are reported: `stride` takes every d-th row/column; `block_projection` computes `P^T J P` for an orthonormal block-constant basis.

Source matrices:
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/solver/solver_index0_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/baseline/baseline_index0_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/baseline_error/baseline_error_index0_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_004/solver/solver_index4_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_004/baseline/baseline_index4_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_004/baseline_error/baseline_error_index4_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_006/solver/solver_index6_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_006/baseline/baseline_index6_jacobian_svd.npz`
- `/workspace/NeuralOperatorRobustness2/forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_006/baseline_error/baseline_error_index6_jacobian_svd.npz`

Primary CSV: `downsample_jacobian_svd_comparison.csv`.

Interpretation: high rank-1 and top-k cosine means indicate the coarse proxy preserves the corresponding full-Jacobian singular directions after projection. Low values mean the coarse proxy is not representative for that matrix/direction.
