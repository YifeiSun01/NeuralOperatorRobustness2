# Burgers Loss3-Selective Round03 Final-Extension Final-Model Jacobian/SVD

Observed on round03 final-extension checkpoints with the same fixed sample manifest used by the pre-continuation long final SVD: loss1 epoch5000, loss2 epoch2000, loss3 epoch1500.
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
| loss1_epoch5000 | 5000 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/checkpoints/burgers_epoch5000_step015000.pt` |
| loss2_epoch2000 | 2000 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt` |
| loss3_epoch1500 | 1500 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt` |

## Model-Minus-Solver Error Spectral Norm

| model | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller than baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ALL | 20 | 3.46793 | 3.46793 | 1 | 1 | 20 |
| baseline | generalization | 10 | 5.77462 | 5.77462 | 1 | 1 | 10 |
| baseline | test | 5 | 1.23788 | 1.23788 | 1 | 1 | 5 |
| baseline | train | 5 | 1.08462 | 1.08462 | 1 | 1 | 5 |
| loss1_epoch5000 | ALL | 20 | 1.50912 | 3.46793 | 0.391835 | 0.338706 | 19 |
| loss1_epoch5000 | generalization | 10 | 2.75318 | 5.77462 | 0.494448 | 0.469654 | 9 |
| loss1_epoch5000 | test | 5 | 0.268445 | 1.23788 | 0.29403 | 0.230369 | 5 |
| loss1_epoch5000 | train | 5 | 0.261678 | 1.08462 | 0.284415 | 0.309839 | 5 |
| loss2_epoch2000 | ALL | 20 | 1.80615 | 3.46793 | 0.443998 | 0.359526 | 19 |
| loss2_epoch2000 | generalization | 10 | 3.32226 | 5.77462 | 0.578548 | 0.584233 | 9 |
| loss2_epoch2000 | test | 5 | 0.29986 | 1.23788 | 0.317824 | 0.232707 | 5 |
| loss2_epoch2000 | train | 5 | 0.280222 | 1.08462 | 0.301074 | 0.327733 | 5 |
| loss3_epoch1500 | ALL | 20 | 1.67337 | 3.46793 | 0.625459 | 0.587264 | 17 |
| loss3_epoch1500 | generalization | 10 | 2.4996 | 5.77462 | 0.411547 | 0.348694 | 9 |
| loss3_epoch1500 | test | 5 | 0.772003 | 1.23788 | 0.669956 | 0.636389 | 5 |
| loss3_epoch1500 | train | 5 | 0.922264 | 1.08462 | 1.00879 | 0.928545 | 3 |

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
| error | loss1_epoch5000 | ALL | 20 | 1.50912 | 0.681129 | 5.67218 | 1.81716 | 11.3977 |
| error | loss1_epoch5000 | generalization | 10 | 2.75318 | 2.51427 | 5.67218 | 3.07039 | 2.93238 |
| error | loss1_epoch5000 | test | 5 | 0.268445 | 0.263345 | 0.289711 | 0.57451 | 20.0089 |
| error | loss1_epoch5000 | train | 5 | 0.261678 | 0.254334 | 0.318772 | 0.553367 | 19.717 |
| model | loss1_epoch5000 | ALL | 20 | 4.70948 | 4.7026 | 7.23498 | 6.87497 | 4.091 |
| model | loss1_epoch5000 | generalization | 10 | 4.95474 | 4.68852 | 7.23498 | 6.98541 | 3.98405 |
| model | loss1_epoch5000 | test | 5 | 4.50084 | 4.8862 | 4.96955 | 6.81981 | 4.23587 |
| model | loss1_epoch5000 | train | 5 | 4.42762 | 4.31758 | 4.93283 | 6.70922 | 4.16001 |
| error | loss2_epoch2000 | ALL | 20 | 1.80615 | 0.648214 | 6.27296 | 2.13661 | 9.48795 |
| error | loss2_epoch2000 | generalization | 10 | 3.32226 | 2.54486 | 6.27296 | 3.65581 | 2.65588 |
| error | loss2_epoch2000 | test | 5 | 0.29986 | 0.291085 | 0.377362 | 0.640189 | 16.293 |
| error | loss2_epoch2000 | train | 5 | 0.280222 | 0.274785 | 0.331743 | 0.594628 | 16.3471 |
| model | loss2_epoch2000 | ALL | 20 | 4.66346 | 4.65326 | 6.81081 | 6.79584 | 4.06516 |
| model | loss2_epoch2000 | generalization | 10 | 4.86046 | 4.55306 | 6.81081 | 6.8277 | 3.94699 |
| model | loss2_epoch2000 | test | 5 | 4.49992 | 4.90923 | 4.95225 | 6.81495 | 4.21742 |
| model | loss2_epoch2000 | train | 5 | 4.43299 | 4.33915 | 4.90499 | 6.71299 | 4.14925 |
| error | loss3_epoch1500 | ALL | 20 | 1.67337 | 1.04966 | 5.24068 | 1.93282 | 2.69004 |
| error | loss3_epoch1500 | generalization | 10 | 2.4996 | 1.71816 | 5.24068 | 2.88491 | 2.65779 |
| error | loss3_epoch1500 | test | 5 | 0.772003 | 0.818431 | 1.19599 | 0.915999 | 3.11683 |
| error | loss3_epoch1500 | train | 5 | 0.922264 | 0.990764 | 1.18821 | 1.04545 | 2.32777 |
| model | loss3_epoch1500 | ALL | 20 | 4.63946 | 4.68517 | 6.66516 | 6.80337 | 4.1567 |
| model | loss3_epoch1500 | generalization | 10 | 4.84566 | 4.68517 | 6.66516 | 6.87784 | 4.05766 |
| model | loss3_epoch1500 | test | 5 | 4.46522 | 4.89504 | 4.92154 | 6.78227 | 4.30079 |
| model | loss3_epoch1500 | train | 5 | 4.40129 | 4.26807 | 4.89181 | 6.67552 | 4.21069 |
| solver | solver | ALL | 20 | 4.96376 | 4.89745 | 8.44865 | 7.11225 | 3.97586 |
| solver | solver | generalization | 10 | 5.4573 | 5.34365 | 8.44865 | 7.44372 | 3.70176 |
| solver | solver | test | 5 | 4.50658 | 4.88279 | 4.97575 | 6.83434 | 4.28458 |
| solver | solver | train | 5 | 4.43385 | 4.28733 | 4.93092 | 6.72722 | 4.21532 |

## Output Files

- `round03_loss123_final_extension_sample_manifest.csv`: fixed train/test/generalization sample points.
- `round03_loss123_final_extension_jacobian_svd_summary.csv`: per-sample spectral norms and top singular values.
- `round03_loss123_final_extension_top_singular_values_long.csv`: long top-k singular values.
- `round03_loss123_final_extension_solver_similarity_rankwise.csv`: rankwise singular-vector similarity to solver.
- `round03_loss123_final_extension_solver_similarity_subspaces.csv`: top-k singular subspace similarity to solver.
- `round03_loss123_final_extension_error_spectral_norm_aggregate.csv`: split-level model-minus-solver summary.
- `sample_*/`: NPZ SVD files for solver, baseline, model, and model-minus-solver error.

## Config

```json
{
  "generalization_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_loss3_selective_search/round_03",
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606",
  "model_specs": [
    {
      "label": "baseline",
      "epoch": 0,
      "path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
    },
    {
      "label": "loss1_epoch5000",
      "epoch": 5000,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/checkpoints/burgers_epoch5000_step015000.pt"
    },
    {
      "label": "loss2_epoch2000",
      "epoch": 2000,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt"
    },
    {
      "label": "loss3_epoch1500",
      "epoch": 1500,
      "path": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt"
    }
  ],
  "sample_count": 20,
  "train_samples": 5,
  "test_samples": 5,
  "generalization_samples": 10,
  "output_prefix": "round03_loss123_final_extension",
  "report_title": "Burgers Loss3-Selective Round03 Final-Extension Final-Model Jacobian/SVD",
  "report_note": "Observed on round03 final-extension checkpoints with the same fixed sample manifest used by the pre-continuation long final SVD: loss1 epoch5000, loss2 epoch2000, loss3 epoch1500.",
  "seed": 20260606,
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
