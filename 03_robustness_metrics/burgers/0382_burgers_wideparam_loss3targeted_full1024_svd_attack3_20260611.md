# Burgers Full-1024 SVD vs 15-Step Attack Probe - 2026-06-11

This is the initial 3-generalization-sample probe. It uses full `1024 x 1024` dense Jacobians and full SVD, not block projection or any coarse proxy.

## Output Files

- output root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611`
- sample manifest: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/sample_manifest.csv`
- attack step table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/attack_step_metrics.csv`
- joined SVD/attack table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/svd_attack_joined_metrics.csv`
- correlation table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/svd_attack_correlations.csv`
- runtime JSON: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/runtime_summary.json`
- per-sample full SVD NPZ files: `sample_*/{solver,baseline_model,baseline_error,...}/*_jacobian_svd.npz`

## Runtime

- total wall seconds: `2160.404`
- attack wall seconds total: `185.452`
- jacobian+SVD wall seconds total: `1974.585`

## Samples

| sample_id | dataset_id | index | family | display_label |
|---:|---|---:|---|---|
| 0 | `burgers_widevis_l3target_d31` | 10 | `powerlaw_fourier` | Power-law Fourier alpha=4, k0=36; range [-0.5,1.5] |
| 1 | `burgers_widevis_l3target_d17` | 177 | `matern` | Matern GRF c=0.04, nu=2.5; range [-0.65,1.35] |
| 2 | `burgers_widevis_l3target_d05` | 45 | `gaussian` | Gaussian GRF corr=0.009; range [0.15,1.25] |

## Joined Metrics

| sample | model | err sigma1 | attack init MSE | attack final MSE | growth abs | growth ratio | delta RMS | err-v1 vs final delta cos |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 2.36802 | 0.0011872 | 0.0345066 | 0.0333194 | 29.0655 | 0.12 | 0.0316196 |
| 0 | loss1 | 3.02591 | 0.00119666 | 0.0120534 | 0.0108567 | 10.0725 | 0.12 | 0.237807 |
| 0 | loss2 | 3.07656 | 0.0016665 | 0.00932695 | 0.00766046 | 5.59674 | 0.12 | 0.0647416 |
| 0 | loss3 | 1.06904 | 0.000204258 | 0.00653228 | 0.00632803 | 31.9806 | 0.12 | 0.00441821 |
| 1 | baseline | 1.71212 | 0.000612469 | 0.00498496 | 0.00437249 | 8.13913 | 0.12 | 0.126244 |
| 1 | loss1 | 1.99931 | 0.000745306 | 0.00828846 | 0.00754316 | 11.1209 | 0.12 | 0.125629 |
| 1 | loss2 | 1.97707 | 0.000363448 | 0.00710781 | 0.00674436 | 19.5566 | 0.12 | 0.171126 |
| 1 | loss3 | 1.70113 | 0.000256348 | 0.00422428 | 0.00396794 | 16.4787 | 0.12 | 0.0919157 |
| 2 | baseline | 1.95388 | 0.000472753 | 0.00918172 | 0.00870896 | 19.4218 | 0.12 | 0.781514 |
| 2 | loss1 | 1.22419 | 0.000301773 | 0.00301631 | 0.00271454 | 9.9953 | 0.12 | 0.100595 |
| 2 | loss2 | 1.21643 | 0.000263318 | 0.00261659 | 0.00235327 | 9.937 | 0.12 | 0.0830966 |
| 2 | loss3 | 0.420253 | 3.57665e-05 | 0.0015039 | 0.00146813 | 42.0476 | 0.12 | 0.0255779 |

## Correlations Across 12 Model-Sample Pairs

| x | y | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| `error_spectral_norm` | `attack_loss_growth_abs` | 12 | 0.476677 | 0.846154 |
| `error_spectral_norm` | `attack_loss_growth_ratio` | 12 | -0.55152 | -0.363636 |
| `error_spectral_norm` | `attack_final_mse` | 12 | 0.512738 | 0.881119 |
| `error_spectral_norm` | `attack_log_growth` | 12 | -0.532231 | -0.363636 |
| `model_solver_top1_right_abs_cos` | `attack_loss_growth_abs` | 12 | -0.956664 | -0.699301 |
| `error_top_right_final_delta_abs_cos` | `attack_loss_growth_abs` | 12 | -0.0139494 | 0.335664 |

## Config

```json
{
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611",
  "report_md": "/workspace/NeuralOperatorRobustness2/docs/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611.md",
  "generalization_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers",
  "num_samples": 3,
  "seed": 20260611,
  "attack_steps": 15,
  "epsilon_rms": 0.12,
  "alpha_rms": 0.012,
  "device": "cuda",
  "reuse_existing": true,
  "full_jacobian_shape": [
    1024,
    1024
  ],
  "svd_method": "full_np_linalg_svd",
  "model_specs": [
    {
      "key": "baseline",
      "label": "baseline",
      "epoch": 0,
      "checkpoint": "/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
    },
    {
      "key": "loss1",
      "label": "loss1_epoch8000",
      "epoch": 8000,
      "checkpoint": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt"
    },
    {
      "key": "loss2",
      "label": "loss2_epoch2000",
      "epoch": 2000,
      "checkpoint": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt"
    },
    {
      "key": "loss3",
      "label": "loss3_epoch1500",
      "epoch": 1500,
      "checkpoint": "/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt"
    }
  ],
  "solver": {
    "burgers_nu": 0.001,
    "burgers_t_final": 1.0,
    "burgers_dt": 0.001,
    "burgers_domain": 2.0,
    "burgers_jax_solver_dtype": "float64"
  }
}
```


## Added Interpretation

This initial probe is intentionally tiny: 3 generalization samples x 4 models =
12 model-sample pairs. The correlations are therefore only a smoke-test signal,
not a final statistical claim.

Mean over the 3 sampled generalization points:

| model | mean error sigma1 | mean attack initial MSE | mean attack final MSE | mean attack growth abs | mean attack growth ratio |
|---|---:|---:|---:|---:|---:|
| baseline | 2.01134 | 7.57474e-04 | 1.62244e-02 | 1.54669e-02 | 18.8755 |
| loss1 | 2.08314 | 7.47913e-04 | 7.78605e-03 | 7.03814e-03 | 10.3962 |
| loss2 | 2.09002 | 7.64422e-04 | 6.35045e-03 | 5.58603e-03 | 11.6968 |
| loss3 | 1.06348 | 1.65457e-04 | 4.08682e-03 | 3.92137e-03 | 30.1690 |

All 12 attacks reached the requested RMS-L2 boundary: final `delta_rms = 0.12`.

Main correlation signal across the 12 aligned model-sample pairs:

- `error_spectral_norm` vs absolute attack-loss growth: Pearson `0.4767`, Spearman `0.8462`.
- `error_spectral_norm` vs final attacked MSE: Pearson `0.5127`, Spearman `0.8811`.
- `error_spectral_norm` vs multiplicative growth ratio is negative here because models with very small clean initial loss, especially loss3, can have a large ratio even when the absolute final attacked loss remains small.

Interpretation for this initial run: the full error-Jacobian spectral norm agrees
better with absolute/final adversarial damage than with multiplicative growth
ratio. That is consistent with the intuition that local linear sensitivity should
track absolute small-budget loss increase, while ratios are distorted when the
clean starting loss is close to zero.

Runtime decomposition:

| component | count | total seconds | mean seconds | max seconds |
|---|---:|---:|---:|---:|
| solver_jacobian | 3 | 886.661 | 295.554 | 322.811 |
| solver_full_svd | 3 | 211.063 | 70.354 | 86.105 |
| model_jacobian | 12 | 35.363 | 2.947 | 3.832 |
| model_full_svd | 12 | 472.726 | 39.394 | 89.557 |
| error_full_svd | 12 | 361.404 | 30.117 | 89.478 |

The solver Jacobian dominates runtime. Full `1024 x 1024` SVD was used throughout;
no block projection and no top-k SVD were used.
