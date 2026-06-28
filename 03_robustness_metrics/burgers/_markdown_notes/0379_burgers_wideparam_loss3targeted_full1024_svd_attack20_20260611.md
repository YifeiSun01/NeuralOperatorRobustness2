# Burgers Full-1024 SVD/Attack 20-Sample Audit - 2026-06-11

This file describes the fixed 20-sample runner. It uses full dense `1024 x 1024` Jacobians and full SVDs. It does not use block projection, randomized SVD, or coarse top-k SVD.

## Runtime Estimate

- prior 3-sample total: `2160.404` sec = `0.600` h
- 20-sample linear estimate: `4.001` h
- recommended planning window: `4.201` to `4.721` h
- estimated attack part: `0.343` h
- estimated Jacobian+SVD part: `3.657` h

## Output Files

- output root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611`
- report: `docs/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611.md`
- sample manifest: `sample_manifest.csv` and `sample_manifest.json`
- attack table: `attack_step_metrics.csv`
- joined table: `svd_attack_joined_metrics.csv`
- SVD summary: `svd_summary.csv`
- singular values: `singular_values_top100_long.csv`
- subspace similarities: `svd_pair_similarities_topk.csv`
- cross-model error-subspace similarities: `cross_model_error_subspace_similarities_topk.csv`
- model reductions: `model_pair_reductions.csv`
- correlations: `svd_attack_correlations.csv`
- runtime details: `runtime_components.csv` and `runtime_summary.json`
- full matrix/vector payloads: `sample_*/{solver,*_model,*_error}/*_jacobian_svd.npz`

## Fixed Samples

| id | split | dataset | index | family | label |
|---:|---|---|---:|---|---|
| 0 | `train` | `train_original_gaussian_corr0p03` | 69 | `original_gaussian` | train original Gaussian GRF corr=0.03; range approximately [0,1] |
| 1 | `train` | `train_original_gaussian_corr0p03` | 860 | `original_gaussian` | train original Gaussian GRF corr=0.03; range approximately [0,1] |
| 2 | `test` | `test_original_gaussian_corr0p03` | 17 | `original_gaussian` | test original Gaussian GRF corr=0.03; range approximately [0,1] |
| 3 | `test` | `test_original_gaussian_corr0p03` | 132 | `original_gaussian` | test original Gaussian GRF corr=0.03; range approximately [0,1] |
| 4 | `generalization` | `burgers_widevis_l3target_d03` | 27 | `gaussian` | Gaussian GRF corr=0.012; range [0,1.2] |
| 5 | `generalization` | `burgers_widevis_l3target_d07` | 163 | `gaussian` | Gaussian GRF corr=0.009; range [0,1.2] |
| 6 | `generalization` | `burgers_widevis_l3target_d12` | 113 | `matern` | Matern GRF c=0.055, nu=4; range [-0.3,1.3] |
| 7 | `generalization` | `burgers_widevis_l3target_d14` | 143 | `matern` | Matern GRF c=0.04, nu=2.5; range [-0.5,1.5] |
| 8 | `generalization` | `burgers_widevis_l3target_d16` | 80 | `matern` | Matern GRF c=0.08, nu=2.5; range [-0.3,1.3] |
| 9 | `generalization` | `burgers_widevis_l3target_d23` | 61 | `powerlaw_fourier` | Power-law Fourier alpha=1.5, k0=10; range [-0.5,1.5] |
| 10 | `generalization` | `burgers_widevis_l3target_d24` | 196 | `powerlaw_fourier` | Power-law Fourier alpha=3, k0=24; range [-0.2,1.2] |
| 11 | `generalization` | `burgers_widevis_l3target_d25` | 80 | `powerlaw_fourier` | Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5] |
| 12 | `generalization` | `burgers_widevis_l3target_d27` | 69 | `powerlaw_fourier` | Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5] |
| 13 | `generalization` | `burgers_widevis_l3target_d35` | 36 | `powerlaw_fourier` | Power-law Fourier alpha=3, k0=24; range [-0.5,1.5] |
| 14 | `generalization` | `burgers_widevis_l3target_d36` | 147 | `sine_mixture` | Sine mix f=[5, 9, 15, 23], decay=0.25; range [0,1.5] |
| 15 | `generalization` | `burgers_widevis_l3target_d37` | 194 | `sine_mixture` | Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5] |
| 16 | `generalization` | `burgers_widevis_l3target_d38` | 184 | `sine_mixture` | Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.3,1.3] |
| 17 | `generalization` | `burgers_widevis_l3target_d39` | 99 | `sine_mixture` | Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.3,1.3] |
| 18 | `generalization` | `burgers_widevis_l3target_d41` | 125 | `sine_mixture` | Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.65,1.35] |
| 19 | `generalization` | `burgers_widevis_l3target_d46` | 185 | `sine_mixture` | Sine mix f=[4, 7, 11], decay=0.35; range [-0.3,1.3] |

## Config

```json
{
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611",
  "report_md": "/workspace/NeuralOperatorRobustness2/docs/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611.md",
  "train_path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt",
  "test_path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt",
  "generalization_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers",
  "sample_manifest": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611/sample_manifest.csv",
  "sample_counts": {
    "train": 2,
    "test": 2,
    "generalization": 16,
    "total": 20
  },
  "seed": 20260611,
  "attack_steps": 15,
  "epsilon_rms": 0.12,
  "alpha_rms": 0.012,
  "top_k_values": [
    5,
    10,
    20,
    50,
    100
  ],
  "reuse_existing": true,
  "full_jacobian_shape": [
    1024,
    1024
  ],
  "svd_method": "full_np_linalg_svd",
  "uses_block_projection": false,
  "uses_randomized_svd": false,
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
