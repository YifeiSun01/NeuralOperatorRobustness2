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
| baseline | ALL | 20 | 2.89099 | 2.89099 | 1 | 1 | 20 |
| baseline | generalization | 10 | 4.54727 | 4.54727 | 1 | 1 | 10 |
| baseline | test | 5 | 1.13171 | 1.13171 | 1 | 1 | 5 |
| baseline | train | 5 | 1.3377 | 1.3377 | 1 | 1 | 5 |
| loss1_epoch1000 | ALL | 20 | 0.858957 | 2.89099 | 0.339883 | 0.319523 | 20 |
| loss1_epoch1000 | generalization | 10 | 1.3575 | 4.54727 | 0.339293 | 0.301217 | 10 |
| loss1_epoch1000 | test | 5 | 0.403351 | 1.13171 | 0.372048 | 0.388484 | 5 |
| loss1_epoch1000 | train | 5 | 0.317488 | 1.3377 | 0.3089 | 0.235256 | 5 |
| loss2_epoch500 | ALL | 20 | 1.07289 | 2.89099 | 0.3793 | 0.355006 | 20 |
| loss2_epoch500 | generalization | 10 | 1.78461 | 4.54727 | 0.408055 | 0.42237 | 10 |
| loss2_epoch500 | test | 5 | 0.36569 | 1.13171 | 0.357092 | 0.312406 | 5 |
| loss2_epoch500 | train | 5 | 0.356644 | 1.3377 | 0.343998 | 0.236877 | 5 |
| loss3_epoch500 | ALL | 20 | 1.22107 | 2.89099 | 0.584777 | 0.519203 | 17 |
| loss3_epoch500 | generalization | 10 | 1.61954 | 4.54727 | 0.353735 | 0.306315 | 10 |
| loss3_epoch500 | test | 5 | 0.683756 | 1.13171 | 0.739958 | 0.733302 | 4 |
| loss3_epoch500 | train | 5 | 0.961461 | 1.3377 | 0.891682 | 0.591177 | 3 |

## Jacobian Spectral Norm Aggregate

| jacobian kind | model | split | n | spectral mean | spectral median | spectral max | fro mean | effective rank mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| error | baseline | ALL | 20 | 2.89099 | 2.51725 | 8.04455 | 3.40823 | 5.41054 |
| error | baseline | generalization | 10 | 4.54727 | 3.2128 | 8.04455 | 5.08941 | 2.32134 |
| error | baseline | test | 5 | 1.13171 | 0.774747 | 2.61831 | 1.71327 | 9.36917 |
| error | baseline | train | 5 | 1.3377 | 1.23312 | 2.24523 | 1.74082 | 7.6303 |
| model | baseline | ALL | 20 | 4.26463 | 4.18122 | 5.96134 | 6.38838 | 4.40963 |
| model | baseline | generalization | 10 | 4.38145 | 4.39305 | 5.13272 | 6.49218 | 4.19926 |
| model | baseline | test | 5 | 3.76711 | 3.70766 | 4.68276 | 6.00412 | 5.0069 |
| model | baseline | train | 5 | 4.52853 | 3.92661 | 5.96134 | 6.56504 | 4.23309 |
| error | loss1_epoch1000 | ALL | 20 | 0.858957 | 0.670867 | 2.11164 | 1.18852 | 9.95206 |
| error | loss1_epoch1000 | generalization | 10 | 1.3575 | 1.33985 | 2.11164 | 1.68651 | 4.43381 |
| error | loss1_epoch1000 | test | 5 | 0.403351 | 0.283476 | 0.858845 | 0.726058 | 14.7186 |
| error | loss1_epoch1000 | train | 5 | 0.317488 | 0.325659 | 0.386252 | 0.655001 | 16.222 |
| model | loss1_epoch1000 | ALL | 20 | 4.5893 | 4.38625 | 6.26329 | 6.73485 | 4.25445 |
| model | loss1_epoch1000 | generalization | 10 | 4.71121 | 4.59843 | 6.25118 | 6.85323 | 4.03077 |
| model | loss1_epoch1000 | test | 5 | 4.13915 | 4.1635 | 5.53262 | 6.38465 | 4.78587 |
| model | loss1_epoch1000 | train | 5 | 4.79564 | 4.04585 | 6.26329 | 6.84831 | 4.1704 |
| error | loss2_epoch500 | ALL | 20 | 1.07289 | 0.572621 | 3.21324 | 1.41834 | 10.1841 |
| error | loss2_epoch500 | generalization | 10 | 1.78461 | 1.51442 | 3.21324 | 2.10954 | 4.62306 |
| error | loss2_epoch500 | test | 5 | 0.36569 | 0.260907 | 0.688433 | 0.720209 | 16.3686 |
| error | loss2_epoch500 | train | 5 | 0.356644 | 0.356823 | 0.45681 | 0.734073 | 15.1217 |
| model | loss2_epoch500 | ALL | 20 | 4.54534 | 4.26304 | 6.15242 | 6.69449 | 4.27994 |
| model | loss2_epoch500 | generalization | 10 | 4.64845 | 4.56053 | 6.13012 | 6.8029 | 4.0652 |
| model | loss2_epoch500 | test | 5 | 4.10619 | 4.13873 | 5.42973 | 6.34455 | 4.80163 |
| model | loss2_epoch500 | train | 5 | 4.77828 | 4.06548 | 6.15242 | 6.82759 | 4.18771 |
| error | loss3_epoch500 | ALL | 20 | 1.22107 | 0.867114 | 3.65476 | 1.49687 | 4.70983 |
| error | loss3_epoch500 | generalization | 10 | 1.61954 | 1.23131 | 3.65476 | 1.88777 | 3.91357 |
| error | loss3_epoch500 | test | 5 | 0.683756 | 0.648359 | 1.00301 | 1.01414 | 6.4664 |
| error | loss3_epoch500 | train | 5 | 0.961461 | 0.823222 | 1.5639 | 1.19782 | 4.54579 |
| model | loss3_epoch500 | ALL | 20 | 4.51333 | 4.293 | 6.08519 | 6.65313 | 4.29605 |
| model | loss3_epoch500 | generalization | 10 | 4.65351 | 4.62462 | 6.08519 | 6.76992 | 4.06566 |
| model | loss3_epoch500 | test | 5 | 4.0444 | 3.99551 | 5.31508 | 6.30745 | 4.86256 |
| model | loss3_epoch500 | train | 5 | 4.70191 | 4.01949 | 6.03887 | 6.76522 | 4.1903 |
| solver | solver | ALL | 20 | 4.72115 | 4.58903 | 6.72744 | 6.85453 | 4.26975 |
| solver | solver | generalization | 10 | 4.98565 | 5.17 | 6.72744 | 7.07876 | 3.96118 |
| solver | solver | test | 5 | 4.11256 | 4.08411 | 5.55933 | 6.38126 | 4.91801 |
| solver | solver | train | 5 | 4.80072 | 4.05059 | 6.27722 | 6.87935 | 4.23864 |

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
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604",
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
  "sample_count": 20,
  "train_samples": 5,
  "test_samples": 5,
  "generalization_samples": 10,
  "seed": 20260604,
  "top_k": 100,
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
