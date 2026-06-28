# Burgers Loss3-Selective Round03 Long Final-Model Jacobian/SVD

Observed on round03 final long-run checkpoints: loss1 epoch1000, loss2 epoch500, loss3 epoch500.
No old solver or baseline SVD files are reused, because local Jacobians depend on the input X.

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
| loss1_epoch1000 | 1000 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt` |
| loss2_epoch0500 | 500 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt` |
| loss3_epoch0500 | 500 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt` |

## Model-Minus-Solver Error Spectral Norm

| model | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller than baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ALL | 20 | 3.46793 | 3.46793 | 1 | 1 | 20 |
| baseline | generalization | 10 | 5.77462 | 5.77462 | 1 | 1 | 10 |
| baseline | test | 5 | 1.23788 | 1.23788 | 1 | 1 | 5 |
| baseline | train | 5 | 1.08462 | 1.08462 | 1 | 1 | 5 |
| loss1_epoch1000 | ALL | 20 | 1.92673 | 3.46793 | 0.503044 | 0.434241 | 19 |
| loss1_epoch1000 | generalization | 10 | 3.3956 | 5.77462 | 0.587589 | 0.612164 | 9 |
| loss1_epoch1000 | test | 5 | 0.500237 | 1.23788 | 0.447316 | 0.427384 | 5 |
| loss1_epoch1000 | train | 5 | 0.41549 | 1.08462 | 0.38968 | 0.389413 | 5 |
| loss2_epoch0500 | ALL | 20 | 2.13163 | 3.46793 | 0.492032 | 0.443601 | 19 |
| loss2_epoch0500 | generalization | 10 | 3.93652 | 5.77462 | 0.649148 | 0.677719 | 9 |
| loss2_epoch0500 | test | 5 | 0.364964 | 1.23788 | 0.35394 | 0.285685 | 5 |
| loss2_epoch0500 | train | 5 | 0.28852 | 1.08462 | 0.315894 | 0.341095 | 5 |
| loss3_epoch0500 | ALL | 20 | 1.72011 | 3.46793 | 0.596787 | 0.545066 | 18 |
| loss3_epoch0500 | generalization | 10 | 2.65625 | 5.77462 | 0.436532 | 0.316626 | 9 |
| loss3_epoch0500 | test | 5 | 0.682533 | 1.23788 | 0.622928 | 0.594781 | 5 |
| loss3_epoch0500 | train | 5 | 0.885378 | 1.08462 | 0.891157 | 0.816197 | 4 |

## Jacobian Spectral Norm Aggregate

| jacobian kind | model | split | n | spectral mean | spectral median | spectral max | fro mean | effective rank mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| error | baseline | ALL | 20 | 3.46793 | 2.49676 | 9.03586 | 4.09736 | 5.48405 |
| error | baseline | generalization | 10 | 5.77462 | 5.96187 | 9.03586 | 6.50824 | 2.2781 |
| error | baseline | test | 5 | 1.23788 | 1.2576 | 2.2048 | 1.82656 | 8.54191 |
| error | baseline | train | 5 | 1.08462 | 0.811571 | 1.94316 | 1.54641 | 8.83809 |
| model | baseline | ALL | 20 | 4.23078 | 4.07756 | 5.15305 | 6.41447 | 4.34018 |
| model | baseline | generalization | 10 | 4.17735 | 4.00394 | 5.15305 | 6.29513 | 4.42356 |
| model | baseline | test | 5 | 4.39828 | 4.65233 | 4.98151 | 6.64705 | 4.23802 |
| model | baseline | train | 5 | 4.17014 | 3.97032 | 4.8875 | 6.42058 | 4.27559 |
| error | loss1_epoch1000 | ALL | 20 | 1.92673 | 1.04548 | 6.49072 | 2.26025 | 7.73692 |
| error | loss1_epoch1000 | generalization | 10 | 3.3956 | 2.72835 | 6.49072 | 3.75188 | 2.88453 |
| error | loss1_epoch1000 | test | 5 | 0.500237 | 0.419078 | 0.901767 | 0.827004 | 11.8199 |
| error | loss1_epoch1000 | train | 5 | 0.41549 | 0.334959 | 0.69417 | 0.710214 | 13.3587 |
| model | loss1_epoch1000 | ALL | 20 | 4.63118 | 4.66424 | 6.5949 | 6.75293 | 4.05865 |
| model | loss1_epoch1000 | generalization | 10 | 4.77843 | 4.54429 | 6.5949 | 6.73216 | 3.95826 |
| model | loss1_epoch1000 | test | 5 | 4.50928 | 4.88574 | 4.9858 | 6.82708 | 4.19391 |
| model | loss1_epoch1000 | train | 5 | 4.45858 | 4.40567 | 4.92516 | 6.72033 | 4.12416 |
| error | loss2_epoch0500 | ALL | 20 | 2.13163 | 0.591832 | 7.84966 | 2.49119 | 9.83438 |
| error | loss2_epoch0500 | generalization | 10 | 3.93652 | 3.17923 | 7.84966 | 4.30733 | 3.2141 |
| error | loss2_epoch0500 | test | 5 | 0.364964 | 0.298312 | 0.634149 | 0.708153 | 16.1685 |
| error | loss2_epoch0500 | train | 5 | 0.28852 | 0.276089 | 0.375852 | 0.641949 | 16.7408 |
| model | loss2_epoch0500 | ALL | 20 | 4.6348 | 4.6511 | 6.5907 | 6.76267 | 4.06523 |
| model | loss2_epoch0500 | generalization | 10 | 4.79125 | 4.5129 | 6.5907 | 6.75056 | 3.95759 |
| model | loss2_epoch0500 | test | 5 | 4.52262 | 4.88098 | 5.03709 | 6.84044 | 4.20416 |
| model | loss2_epoch0500 | train | 5 | 4.43408 | 4.30953 | 4.91501 | 6.70912 | 4.14159 |
| error | loss3_epoch0500 | ALL | 20 | 1.72011 | 1.03415 | 5.4706 | 2.10187 | 4.73376 |
| error | loss3_epoch0500 | generalization | 10 | 2.65625 | 1.85469 | 5.4706 | 3.17453 | 3.84521 |
| error | loss3_epoch0500 | test | 5 | 0.682533 | 0.716189 | 0.850064 | 0.973508 | 6.41197 |
| error | loss3_epoch0500 | train | 5 | 0.885378 | 1.05358 | 1.28135 | 1.08491 | 4.83266 |
| model | loss3_epoch0500 | ALL | 20 | 4.57753 | 4.6394 | 5.91597 | 6.69923 | 4.08387 |
| model | loss3_epoch0500 | generalization | 10 | 4.70255 | 4.56255 | 5.91597 | 6.62627 | 3.98078 |
| model | loss3_epoch0500 | test | 5 | 4.48918 | 4.86279 | 4.91958 | 6.8266 | 4.21963 |
| model | loss3_epoch0500 | train | 5 | 4.41585 | 4.16302 | 5.06197 | 6.71779 | 4.15429 |
| solver | solver | ALL | 20 | 4.96376 | 4.89745 | 8.44865 | 7.11225 | 3.97586 |
| solver | solver | generalization | 10 | 5.4573 | 5.34365 | 8.44865 | 7.44372 | 3.70176 |
| solver | solver | test | 5 | 4.50658 | 4.88279 | 4.97575 | 6.83434 | 4.28458 |
| solver | solver | train | 5 | 4.43385 | 4.28733 | 4.93092 | 6.72722 | 4.21532 |

## Output Files

- `round03_long_final_sample_manifest.csv`: fixed train/test/generalization sample points.
- `round03_long_final_jacobian_svd_summary.csv`: per-sample spectral norms and top singular values.
- `round03_long_final_top_singular_values_long.csv`: long top-k singular values.
- `round03_long_final_solver_similarity_rankwise.csv`: rankwise singular-vector similarity to solver.
- `round03_long_final_solver_similarity_subspaces.csv`: top-k singular subspace similarity to solver.
- `round03_long_final_error_spectral_norm_aggregate.csv`: split-level model-minus-solver summary.
- `sample_*/`: NPZ SVD files for solver, baseline, model, and model-minus-solver error.

## Config

```json
{
  "generalization_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_loss3_selective_search/round_03",
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605",
  "model_specs": [
    {
      "label": "baseline",
      "epoch": 0,
      "path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
    },
    {
      "label": "loss1_epoch1000",
      "epoch": 1000,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt"
    },
    {
      "label": "loss2_epoch0500",
      "epoch": 500,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt"
    },
    {
      "label": "loss3_epoch0500",
      "epoch": 500,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt"
    }
  ],
  "sample_count": 20,
  "train_samples": 5,
  "test_samples": 5,
  "generalization_samples": 10,
  "output_prefix": "round03_long_final",
  "report_title": "Burgers Loss3-Selective Round03 Long Final-Model Jacobian/SVD",
  "report_note": "Observed on round03 final long-run checkpoints: loss1 epoch1000, loss2 epoch500, loss3 epoch500.",
  "seed": 20260605,
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
