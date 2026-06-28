# Burgers p2q2 Checkpoint-Series Jacobian/SVD Diagnostics

Object definitions:

```text
J_model(x) = d model(x) / d x
J_solver(x) = d solver(x) / d x
J_error(x) = J_model(x) - J_solver(x)
```

The fixed sample points are reused from the representative20 manifest, so every checkpoint is compared on exactly the same input points.

## Checkpoints

| label | epoch | checkpoint |
|---|---:|---|
| epoch2000 | 2000 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602/burgers/checkpoints/burgers_epoch2000_step006000.pt` |

## Error-Jacobian Spectral Norm Summary

| checkpoint | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller count |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_error | ALL | 20 | 2.37776 | 2.37776 | 1 | 1 | 20 |
| baseline_error | generalization | 10 | 3.54311 | 3.54311 | 1 | 1 | 10 |
| baseline_error | test | 4 | 0.769029 | 0.769029 | 1 | 1 | 4 |
| baseline_error | train | 6 | 1.508 | 1.508 | 1 | 1 | 6 |
| epoch2000 | ALL | 20 | 0.558097 | 2.37776 | 0.303953 | 0.305636 | 20 |
| epoch2000 | generalization | 10 | 0.818364 | 3.54311 | 0.258665 | 0.271207 | 10 |
| epoch2000 | test | 4 | 0.290937 | 0.769029 | 0.412775 | 0.339655 | 4 |
| epoch2000 | train | 6 | 0.302427 | 1.508 | 0.306884 | 0.376584 | 6 |

## Output Files

- `checkpoint_series_jacobian_svd_summary.csv`: per-sample model and error norms.
- `checkpoint_series_top_singular_values_long.csv`: top singular values for model/error Jacobians.
- `checkpoint_series_solver_similarity_rankwise.csv`: rankwise singular-value and singular-vector similarity to solver.
- `checkpoint_series_solver_similarity_subspaces.csv`: top-k left/right singular subspace similarity to solver.
- `checkpoint_series_error_aggregate.csv`: split-level error-Jacobian summary.
- `sample_*/`: top-k SVD NPZ files. Reused solver/baseline NPZ files are copied/truncated with source metadata.

## Config

```json
{
  "run_dir": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602",
  "burgers_dir": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602/burgers",
  "checkpoint_csv": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602/burgers/checkpoints.csv",
  "checkpoint_specs": [
    {
      "label": "epoch2000",
      "epoch": 2000,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602/burgers/checkpoints/burgers_epoch2000_step006000.pt"
    }
  ],
  "sample_manifest": "forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/representative20_sample_manifest.csv",
  "reuse_root": "forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601",
  "top_k": 100,
  "svd_method": "topk",
  "svd_solver": "propack",
  "baseline_checkpoint": "/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
}
```
