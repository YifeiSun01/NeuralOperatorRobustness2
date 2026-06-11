# Burgers Full-1024 SVD/Attack 25-Sample Reuse3 Audit - 2026-06-11

This file describes the fixed 25-sample reuse3 runner. It uses full dense `1024 x 1024` Jacobians and full SVDs. It does not use block projection, randomized SVD, or coarse top-k SVD.

## Runtime Estimate

- prior 3-sample total: `2160.404` sec = `0.600` h
- 25-sample linear estimate: `4.452` h
- recommended planning window: `4.808` to `5.564` h
- estimated attack part: `0.429` h
- estimated Jacobian+SVD part: `4.022` h

## Output Files

- output root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611`
- report: `docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md`
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
| 0 | `generalization` | `burgers_widevis_l3target_d31` | 10 | `powerlaw_fourier` | Power-law Fourier alpha=4, k0=36; range [-0.5,1.5] |
| 1 | `generalization` | `burgers_widevis_l3target_d17` | 177 | `matern` | Matern GRF c=0.04, nu=2.5; range [-0.65,1.35] |
| 2 | `generalization` | `burgers_widevis_l3target_d05` | 45 | `gaussian` | Gaussian GRF corr=0.009; range [0.15,1.25] |
| 3 | `train` | `train_original_gaussian_corr0p03` | 1 | `original_gaussian` | train original Gaussian GRF corr=0.03; range approximately [0,1] |
| 4 | `train` | `train_original_gaussian_corr0p03` | 128 | `original_gaussian` | train original Gaussian GRF corr=0.03; range approximately [0,1] |
| 5 | `test` | `test_original_gaussian_corr0p03` | 45 | `original_gaussian` | test original Gaussian GRF corr=0.03; range approximately [0,1] |
| 6 | `test` | `test_original_gaussian_corr0p03` | 71 | `original_gaussian` | test original Gaussian GRF corr=0.03; range approximately [0,1] |
| 7 | `generalization` | `burgers_widevis_l3target_d03` | 180 | `gaussian` | Gaussian GRF corr=0.012; range [0,1.2] |
| 8 | `generalization` | `burgers_widevis_l3target_d08` | 178 | `gaussian` | Gaussian GRF corr=0.012; range [-0.5,1.5] |
| 9 | `generalization` | `burgers_widevis_l3target_d00` | 98 | `gaussian` | Gaussian GRF corr=0.012; range [-0.2,1.2] |
| 10 | `generalization` | `burgers_widevis_l3target_d10` | 39 | `gaussian` | Gaussian GRF corr=0.009; range [-0.2,1.2] |
| 11 | `generalization` | `burgers_widevis_l3target_d09` | 68 | `gaussian` | Gaussian GRF corr=0.009; range [-0.5,1.5] |
| 12 | `generalization` | `burgers_widevis_l3target_d19` | 148 | `matern` | Matern GRF c=0.08, nu=2.5; range [-0.5,1.5] |
| 13 | `generalization` | `burgers_widevis_l3target_d12` | 104 | `matern` | Matern GRF c=0.055, nu=4; range [-0.3,1.3] |
| 14 | `generalization` | `burgers_widevis_l3target_d15` | 113 | `matern` | Matern GRF c=0.04, nu=2.5; range [0.15,1.25] |
| 15 | `generalization` | `burgers_widevis_l3target_d21` | 136 | `matern` | Matern GRF c=0.03, nu=1.5; range [0,1.5] |
| 16 | `generalization` | `burgers_widevis_l3target_d34` | 192 | `powerlaw_fourier` | Power-law Fourier alpha=1.8, k0=12; range [-0.5,1.5] |
| 17 | `generalization` | `burgers_widevis_l3target_d25` | 28 | `powerlaw_fourier` | Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5] |
| 18 | `generalization` | `burgers_widevis_l3target_d35` | 63 | `powerlaw_fourier` | Power-law Fourier alpha=3, k0=24; range [-0.5,1.5] |
| 19 | `generalization` | `burgers_widevis_l3target_d30` | 74 | `powerlaw_fourier` | Power-law Fourier alpha=3.5, k0=28; range [-0.5,1.5] |
| 20 | `generalization` | `burgers_widevis_l3target_d27` | 77 | `powerlaw_fourier` | Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5] |
| 21 | `generalization` | `burgers_widevis_l3target_d46` | 199 | `sine_mixture` | Sine mix f=[4, 7, 11], decay=0.35; range [-0.3,1.3] |
| 22 | `generalization` | `burgers_widevis_l3target_d41` | 97 | `sine_mixture` | Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.65,1.35] |
| 23 | `generalization` | `burgers_widevis_l3target_d37` | 191 | `sine_mixture` | Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5] |
| 24 | `generalization` | `burgers_widevis_l3target_d44` | 147 | `sine_mixture` | Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.5,1.5] |

## Config

```json
{
  "out_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611",
  "report_md": "/workspace/NeuralOperatorRobustness2/docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md",
  "train_path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt",
  "test_path": "/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt",
  "generalization_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers",
  "sample_manifest": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_manifest.csv",
  "sample_counts": {
    "prior_reused": 3,
    "new_train": 2,
    "new_test": 2,
    "new_generalization": 18,
    "train": 2,
    "test": 2,
    "generalization": 21,
    "total": 25
  },
  "prior_three_root": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611",
  "prior_three_manifest": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/sample_manifest.csv",
  "reused_prior_samples": [
    {
      "sample_id": 0,
      "source": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/sample_000",
      "destination": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_000",
      "copied": true
    },
    {
      "sample_id": 1,
      "source": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/sample_001",
      "destination": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_001",
      "copied": true
    },
    {
      "sample_id": 2,
      "source": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/sample_002",
      "destination": "/workspace/NeuralOperatorRobustness2/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_002",
      "copied": true
    }
  ],
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
