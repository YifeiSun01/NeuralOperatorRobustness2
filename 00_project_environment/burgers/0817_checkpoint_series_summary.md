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
| epoch0200 | 200 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch200_step000600.pt` |
| epoch0400 | 400 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch400_step001200.pt` |
| epoch0600 | 600 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch600_step001800.pt` |
| epoch0800 | 800 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch800_step002400.pt` |
| epoch1000 | 1000 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch1000_step003000.pt` |

## Error-Jacobian Spectral Norm Summary

| checkpoint | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller count |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_error | ALL | 20 | 2.37776 | 2.37776 | 1 | 1 | 20 |
| baseline_error | generalization | 10 | 3.54311 | 3.54311 | 1 | 1 | 10 |
| baseline_error | test | 4 | 0.769029 | 0.769029 | 1 | 1 | 4 |
| baseline_error | train | 6 | 1.508 | 1.508 | 1 | 1 | 6 |
| epoch0200 | ALL | 20 | 2.69865 | 2.37776 | 1.38798 | 1.38787 | 4 |
| epoch0200 | generalization | 10 | 3.9573 | 3.54311 | 1.20418 | 1.2911 | 3 |
| epoch0200 | test | 4 | 1.51415 | 0.769029 | 2.06263 | 2.04152 | 0 |
| epoch0200 | train | 6 | 1.39056 | 1.508 | 1.24456 | 1.21521 | 1 |
| epoch0400 | ALL | 20 | 2.26782 | 2.37776 | 1.03904 | 0.894444 | 12 |
| epoch0400 | generalization | 10 | 3.5039 | 3.54311 | 1.07185 | 1.01999 | 4 |
| epoch0400 | test | 4 | 1.11732 | 0.769029 | 1.36647 | 0.844357 | 3 |
| epoch0400 | train | 6 | 0.974671 | 1.508 | 0.766054 | 0.708235 | 5 |
| epoch0600 | ALL | 20 | 2.08414 | 2.37776 | 0.988585 | 0.787745 | 14 |
| epoch0600 | generalization | 10 | 3.19629 | 3.54311 | 1.07607 | 0.938719 | 6 |
| epoch0600 | test | 4 | 0.879824 | 0.769029 | 1.07775 | 0.72332 | 3 |
| epoch0600 | train | 6 | 1.03341 | 1.508 | 0.783342 | 0.780037 | 5 |
| epoch0800 | ALL | 20 | 0.755361 | 2.37776 | 0.464416 | 0.373424 | 19 |
| epoch0800 | generalization | 10 | 1.00821 | 3.54311 | 0.345177 | 0.300924 | 10 |
| epoch0800 | test | 4 | 0.457528 | 0.769029 | 0.671831 | 0.559161 | 3 |
| epoch0800 | train | 6 | 0.532506 | 1.508 | 0.524872 | 0.497223 | 6 |
| epoch1000 | ALL | 20 | 0.915005 | 2.37776 | 0.590686 | 0.447408 | 17 |
| epoch1000 | generalization | 10 | 1.10146 | 3.54311 | 0.348767 | 0.332076 | 10 |
| epoch1000 | test | 4 | 0.646639 | 0.769029 | 0.97979 | 0.681232 | 3 |
| epoch1000 | train | 6 | 0.783163 | 1.508 | 0.734483 | 0.654525 | 4 |

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
  "run_dir": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601",
  "burgers_dir": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers",
  "checkpoint_csv": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints.csv",
  "checkpoint_specs": [
    {
      "label": "epoch0200",
      "epoch": 200,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch200_step000600.pt"
    },
    {
      "label": "epoch0400",
      "epoch": 400,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch400_step001200.pt"
    },
    {
      "label": "epoch0600",
      "epoch": 600,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch600_step001800.pt"
    },
    {
      "label": "epoch0800",
      "epoch": 800,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch800_step002400.pt"
    },
    {
      "label": "epoch1000",
      "epoch": 1000,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch1000_step003000.pt"
    }
  ],
  "sample_manifest": "/workspace/NeuralOperatorRobustness2/forensics/burgers_adv_training_jacobian_svd_20260531_representative20_same_points/representative20_sample_manifest.csv",
  "reuse_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_adv_training_jacobian_svd_20260531_representative20_same_points",
  "top_k": 100,
  "svd_method": "topk",
  "svd_solver": "propack",
  "baseline_checkpoint": "/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
}
```
