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

## Joined Metric Preview

| sample | split | model | err sigma1 | init MSE | final MSE | growth abs | residual move MSE |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `generalization` | `baseline` | 2.36802 | 0.0011872 | 0.0345065 | 0.0333193 | 0.0320522 |
| 0 | `generalization` | `loss1` | 3.02591 | 0.00119666 | 0.0120534 | 0.0108568 | 0.0117436 |
| 0 | `generalization` | `loss2` | 3.07656 | 0.0016665 | 0.00932704 | 0.00766054 | 0.00647189 |
| 0 | `generalization` | `loss3` | 1.06904 | 0.000204258 | 0.00653236 | 0.0063281 | 0.00526291 |
| 1 | `generalization` | `baseline` | 1.71212 | 0.000612469 | 0.00498492 | 0.00437245 | 0.00408637 |
| 1 | `generalization` | `loss1` | 1.99931 | 0.000745306 | 0.00828849 | 0.00754318 | 0.00731165 |
| 1 | `generalization` | `loss2` | 1.97707 | 0.000363448 | 0.00710781 | 0.00674436 | 0.00718281 |
| 1 | `generalization` | `loss3` | 1.70113 | 0.000256348 | 0.00422427 | 0.00396792 | 0.00326371 |
| 2 | `generalization` | `baseline` | 1.95388 | 0.000472753 | 0.00918159 | 0.00870884 | 0.00915512 |
| 2 | `generalization` | `loss1` | 1.22419 | 0.000301773 | 0.00301629 | 0.00271452 | 0.0026107 |
| 2 | `generalization` | `loss2` | 1.21643 | 0.000263318 | 0.00261659 | 0.00235328 | 0.00213014 |
| 2 | `generalization` | `loss3` | 0.420253 | 3.57665e-05 | 0.00150391 | 0.00146814 | 0.00145493 |
| 3 | `train` | `baseline` | 0.562423 | 4.47611e-05 | 0.00151846 | 0.0014737 | 0.00140694 |
| 3 | `train` | `loss1` | 0.230608 | 3.252e-07 | 0.000679446 | 0.00067912 | 0.000681956 |
| 3 | `train` | `loss2` | 0.298257 | 2.93096e-06 | 0.00125251 | 0.00124958 | 0.00116547 |
| 3 | `train` | `loss3` | 0.671966 | 1.95405e-05 | 0.000769335 | 0.000749794 | 0.000751142 |
| 4 | `train` | `baseline` | 0.71324 | 4.99419e-05 | 0.0135757 | 0.0135258 | 0.0137155 |
| 4 | `train` | `loss1` | 0.282315 | 6.59305e-07 | 0.00156998 | 0.00156932 | 0.0015537 |
| 4 | `train` | `loss2` | 0.311704 | 2.08392e-06 | 0.00158208 | 0.00157999 | 0.00160202 |
| 4 | `train` | `loss3` | 1.07205 | 3.34845e-05 | 0.000627432 | 0.000593947 | 0.000561184 |
| 5 | `test` | `baseline` | 0.435788 | 2.9257e-05 | 0.00235852 | 0.00232926 | 0.00228443 |
| 5 | `test` | `loss1` | 0.295706 | 7.25991e-07 | 0.00178624 | 0.00178551 | 0.00178389 |
| 5 | `test` | `loss2` | 0.303508 | 2.64979e-06 | 0.00185844 | 0.00185579 | 0.00184478 |
| 5 | `test` | `loss3` | 0.852245 | 1.80945e-05 | 0.000962928 | 0.000944833 | 0.000914046 |
| 6 | `test` | `baseline` | 1.06394 | 7.15524e-05 | 0.00252388 | 0.00245233 | 0.00217325 |
| 6 | `test` | `loss1` | 0.300514 | 3.96653e-07 | 0.000935596 | 0.0009352 | 0.00092876 |
| 6 | `test` | `loss2` | 0.315856 | 1.70869e-06 | 0.00178238 | 0.00178068 | 0.001769 |
| 6 | `test` | `loss3` | 0.797558 | 2.18305e-05 | 0.000402922 | 0.000381092 | 0.000313959 |
| 7 | `generalization` | `baseline` | 0.982439 | 0.000110802 | 0.00494148 | 0.00483068 | 0.00482224 |
| 7 | `generalization` | `loss1` | 0.771032 | 7.33523e-05 | 0.00551881 | 0.00544546 | 0.00488851 |
| 7 | `generalization` | `loss2` | 0.782679 | 8.06087e-05 | 0.00495509 | 0.00487448 | 0.00456192 |
| 7 | `generalization` | `loss3` | 0.797 | 3.39755e-05 | 0.00100135 | 0.000967372 | 0.000992384 |
| 8 | `generalization` | `baseline` | 5.45037 | 0.00442233 | 0.0118383 | 0.00741594 | 0.00521402 |
| 8 | `generalization` | `loss1` | 4.33501 | 0.00486101 | 0.0155221 | 0.010661 | 0.00831397 |
| 8 | `generalization` | `loss2` | 4.82096 | 0.00590733 | 0.0171161 | 0.0112087 | 0.00799924 |
| 8 | `generalization` | `loss3` | 4.12712 | 0.00172834 | 0.00951256 | 0.00778423 | 0.00764957 |
| 9 | `generalization` | `baseline` | 2.46394 | 0.000568285 | 0.0100154 | 0.0094471 | 0.0111431 |
| 9 | `generalization` | `loss1` | 0.478847 | 6.04757e-05 | 0.00347803 | 0.00341755 | 0.00347085 |
| 9 | `generalization` | `loss2` | 0.969504 | 9.1479e-05 | 0.00227232 | 0.00218084 | 0.00215963 |
| 9 | `generalization` | `loss3` | 1.23599 | 6.93055e-05 | 0.00123149 | 0.00116219 | 0.00109177 |

## All-Pair Correlation Preview

| x | y | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| `error_spectral_norm` | `attack_loss_growth_abs` | 100 | 0.63719 | 0.77781 |
| `error_spectral_norm` | `attack_final_mse` | 100 | 0.723957 | 0.815818 |
| `error_spectral_norm` | `residual_change_mse` | 100 | 0.607666 | 0.749259 |
| `error_spectral_norm` | `attack_loss_growth_ratio` | 100 | -0.347626 | -0.827519 |

## Runtime

- total wall seconds: `14589.046`
- attack wall seconds: `183.020`
- Jacobian+SVD wall seconds: `14401.212`

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
