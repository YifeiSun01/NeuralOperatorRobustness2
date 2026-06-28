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
| loss1_epoch50 | 50 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_50ep_20260605/burgers/checkpoints/burgers_epoch050_step000150.pt` |
| loss2_epoch50 | 50 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_50ep_20260605/burgers/checkpoints/burgers_epoch050_step000150.pt` |
| loss3_epoch50 | 50 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_50ep_20260605/burgers/checkpoints/burgers_epoch050_step000150.pt` |

## Model-Minus-Solver Error Spectral Norm

| model | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller than baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ALL | 5 | 6.63916 | 6.63916 | 1 | 1 | 5 |
| baseline | generalization | 5 | 6.63916 | 6.63916 | 1 | 1 | 5 |
| loss1_epoch50 | ALL | 5 | 5.40732 | 6.63916 | 0.775478 | 0.77361 | 5 |
| loss1_epoch50 | generalization | 5 | 5.40732 | 6.63916 | 0.775478 | 0.77361 | 5 |
| loss2_epoch50 | ALL | 5 | 5.62003 | 6.63916 | 0.81372 | 0.794559 | 5 |
| loss2_epoch50 | generalization | 5 | 5.62003 | 6.63916 | 0.81372 | 0.794559 | 5 |
| loss3_epoch50 | ALL | 5 | 4.97973 | 6.63916 | 0.784002 | 0.757993 | 5 |
| loss3_epoch50 | generalization | 5 | 4.97973 | 6.63916 | 0.784002 | 0.757993 | 5 |

## Jacobian Spectral Norm Aggregate

| jacobian kind | model | split | n | spectral mean | spectral median | spectral max | fro mean | effective rank mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| error | baseline | ALL | 5 | 6.63916 | 6.59472 | 9.44794 | 7.03525 | 1.86025 |
| error | baseline | generalization | 5 | 6.63916 | 6.59472 | 9.44794 | 7.03525 | 1.86025 |
| model | baseline | ALL | 5 | 4.55723 | 3.98334 | 6.47952 | 6.42085 | 4.18468 |
| model | baseline | generalization | 5 | 4.55723 | 3.98334 | 6.47952 | 6.42085 | 4.18468 |
| error | loss1_epoch50 | ALL | 5 | 5.40732 | 3.98364 | 9.31621 | 5.72227 | 1.99314 |
| error | loss1_epoch50 | generalization | 5 | 5.40732 | 3.98364 | 9.31621 | 5.72227 | 1.99314 |
| model | loss1_epoch50 | ALL | 5 | 4.66081 | 4.18913 | 6.45536 | 6.57913 | 4.13201 |
| model | loss1_epoch50 | generalization | 5 | 4.66081 | 4.18913 | 6.45536 | 6.57913 | 4.13201 |
| error | loss2_epoch50 | ALL | 5 | 5.62003 | 4.29586 | 9.35348 | 5.89083 | 1.82815 |
| error | loss2_epoch50 | generalization | 5 | 5.62003 | 4.29586 | 9.35348 | 5.89083 | 1.82815 |
| model | loss2_epoch50 | ALL | 5 | 4.68179 | 4.14428 | 6.52424 | 6.6071 | 4.10672 |
| model | loss2_epoch50 | generalization | 5 | 4.68179 | 4.14428 | 6.52424 | 6.6071 | 4.10672 |
| error | loss3_epoch50 | ALL | 5 | 4.97973 | 4.96642 | 8.00675 | 5.62368 | 2.29328 |
| error | loss3_epoch50 | generalization | 5 | 4.97973 | 4.96642 | 8.00675 | 5.62368 | 2.29328 |
| model | loss3_epoch50 | ALL | 5 | 4.11698 | 3.97037 | 5.43675 | 6.02518 | 4.42461 |
| model | loss3_epoch50 | generalization | 5 | 4.11698 | 3.97037 | 5.43675 | 6.02518 | 4.42461 |
| solver | solver | ALL | 5 | 5.85644 | 6.09269 | 7.721 | 7.56123 | 3.65197 |
| solver | solver | generalization | 5 | 5.85644 | 6.09269 | 7.721 | 7.56123 | 3.65197 |

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
  "generalization_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_loss3_selective_search/round_03",
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605",
  "model_specs": [
    {
      "label": "baseline",
      "epoch": 0,
      "path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
    },
    {
      "label": "loss1_epoch50",
      "epoch": 50,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_50ep_20260605/burgers/checkpoints/burgers_epoch050_step000150.pt"
    },
    {
      "label": "loss2_epoch50",
      "epoch": 50,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_50ep_20260605/burgers/checkpoints/burgers_epoch050_step000150.pt"
    },
    {
      "label": "loss3_epoch50",
      "epoch": 50,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_50ep_20260605/burgers/checkpoints/burgers_epoch050_step000150.pt"
    }
  ],
  "sample_count": 5,
  "train_samples": 0,
  "test_samples": 0,
  "generalization_samples": 5,
  "seed": 20260605,
  "top_k": 20,
  "svd_method": "topk",
  "svd_solver": "propack",
  "reuse_existing": false,
  "burgers_nu": 0.001,
  "burgers_t_final": 1.0,
  "burgers_dt": 0.001,
  "burgers_domain": 2.0,
  "burgers_jax_solver_dtype": "float64"
}
```
