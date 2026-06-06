# Burgers Loss3-Aligned Round01 Final-Model Jacobian/SVD

This run recomputes all local Jacobians on the new round01 sample points.
No old solver or baseline SVD files are reused, because the generalization X changed.

Definitions:

```text
J_solver(x) = d solver(x) / d x
J_model(x)  = d model(x) / d x
J_error(x)  = J_model(x) - J_solver(x)
```

## Models

| label | epoch | checkpoint |
|---|---:|---|
| baseline | 0 | `/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt` |
| loss1_epoch1000 | 1000 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss1_1000ep_20260604/burgers/checkpoints/burgers_epoch1000_step003000.pt` |
| loss2_epoch500 | 500 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss2_500ep_20260604/burgers/checkpoints/burgers_epoch500_step001500.pt` |
| loss3_epoch500 | 500 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss3_500ep_20260604/burgers/checkpoints/burgers_epoch500_step001500.pt` |

## Model-Minus-Solver Error Spectral Norm

| model | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller than baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ALL | 1 | 0.529917 | 0.529917 | 1 | 1 | 1 |
| baseline | train | 1 | 0.529917 | 0.529917 | 1 | 1 | 1 |
| loss1_epoch1000 | ALL | 1 | 0.220891 | 0.529917 | 0.41684 | 0.41684 | 1 |
| loss1_epoch1000 | train | 1 | 0.220891 | 0.529917 | 0.41684 | 0.41684 | 1 |
| loss2_epoch500 | ALL | 1 | 0.228012 | 0.529917 | 0.430278 | 0.430278 | 1 |
| loss2_epoch500 | train | 1 | 0.228012 | 0.529917 | 0.430278 | 0.430278 | 1 |
| loss3_epoch500 | ALL | 1 | 0.529856 | 0.529917 | 0.999884 | 0.999884 | 1 |
| loss3_epoch500 | train | 1 | 0.529856 | 0.529917 | 0.999884 | 0.999884 | 1 |

## Jacobian Spectral Norm Aggregate

| jacobian kind | model | split | n | spectral mean | spectral median | spectral max | fro mean | effective rank mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| error | baseline | ALL | 1 | 0.529917 | 0.529917 | 0.529917 | 0.944427 | 7.43534 |
| error | baseline | train | 1 | 0.529917 | 0.529917 | 0.529917 | 0.944427 | 7.43534 |
| model | baseline | ALL | 1 | 3.75982 | 3.75982 | 3.75982 | 5.71414 | 4.55292 |
| model | baseline | train | 1 | 3.75982 | 3.75982 | 3.75982 | 5.71414 | 4.55292 |
| error | loss1_epoch1000 | ALL | 1 | 0.220891 | 0.220891 | 0.220891 | 0.472336 | 8.62327 |
| error | loss1_epoch1000 | train | 1 | 0.220891 | 0.220891 | 0.220891 | 0.472336 | 8.62327 |
| model | loss1_epoch1000 | ALL | 1 | 3.83276 | 3.83276 | 3.83276 | 5.86333 | 4.55838 |
| model | loss1_epoch1000 | train | 1 | 3.83276 | 3.83276 | 3.83276 | 5.86333 | 4.55838 |
| error | loss2_epoch500 | ALL | 1 | 0.228012 | 0.228012 | 0.228012 | 0.484749 | 8.69781 |
| error | loss2_epoch500 | train | 1 | 0.228012 | 0.228012 | 0.228012 | 0.484749 | 8.69781 |
| model | loss2_epoch500 | ALL | 1 | 3.8073 | 3.8073 | 3.8073 | 5.84265 | 4.59368 |
| model | loss2_epoch500 | train | 1 | 3.8073 | 3.8073 | 3.8073 | 5.84265 | 4.59368 |
| error | loss3_epoch500 | ALL | 1 | 0.529856 | 0.529856 | 0.529856 | 0.823237 | 5.28181 |
| error | loss3_epoch500 | train | 1 | 0.529856 | 0.529856 | 0.529856 | 0.823237 | 5.28181 |
| model | loss3_epoch500 | ALL | 1 | 3.82901 | 3.82901 | 3.82901 | 5.78957 | 4.55308 |
| model | loss3_epoch500 | train | 1 | 3.82901 | 3.82901 | 3.82901 | 5.78957 | 4.55308 |
| solver | solver | ALL | 1 | 3.82878 | 3.82878 | 3.82878 | 5.85646 | 4.58325 |
| solver | solver | train | 1 | 3.82878 | 3.82878 | 3.82878 | 5.85646 | 4.58325 |

## Output Files

- `round01_sample_manifest.csv`: fixed train/test/generalization sample points.
- `round01_jacobian_svd_summary.csv`: per-sample spectral norms and top singular values.
- `round01_top_singular_values_long.csv`: long top-k singular values.
- `round01_solver_similarity_rankwise.csv`: rankwise singular-vector similarity to solver.
- `round01_solver_similarity_subspaces.csv`: top-k singular subspace similarity to solver.
- `round01_error_spectral_norm_aggregate.csv`: split-level model-minus-solver summary.
- `sample_*/`: NPZ SVD files for solver, baseline, model, and model-minus-solver error.

## Config

```json
{
  "generalization_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_loss3_aligned_search/round_01",
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_loss3_aligned_round01_final_jacobian_svd_smoke_20260604",
  "model_specs": [
    {
      "label": "baseline",
      "epoch": 0,
      "path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
    },
    {
      "label": "loss1_epoch1000",
      "epoch": 1000,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss1_1000ep_20260604/burgers/checkpoints/burgers_epoch1000_step003000.pt"
    },
    {
      "label": "loss2_epoch500",
      "epoch": 500,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss2_500ep_20260604/burgers/checkpoints/burgers_epoch500_step001500.pt"
    },
    {
      "label": "loss3_epoch500",
      "epoch": 500,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss3_500ep_20260604/burgers/checkpoints/burgers_epoch500_step001500.pt"
    }
  ],
  "sample_count": 1,
  "train_samples": 1,
  "test_samples": 0,
  "generalization_samples": 0,
  "seed": 20260604,
  "top_k": 10,
  "svd_method": "topk",
  "svd_solver": "propack",
  "reuse_existing": true,
  "burgers_nu": 0.001,
  "burgers_t_final": 1.0,
  "burgers_dt": 0.001,
  "burgers_domain": 2.0,
  "burgers_jax_solver_dtype": "float64"
}
```
