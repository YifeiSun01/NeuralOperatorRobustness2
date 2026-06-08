# 2026-06-08 Experiment Summary Index

This index organizes the main conclusions, formulas, references, source files, and experimental evidence from the recent Burgers/Darcy Flow robustness discussion.

## Topic Files

- `docs/20260608_jacobian_projection_theory_and_references.md`  
  Rayleigh-Ritz/Galerkin, Davis-Kahan, Wedin, stride versus block projection, and StablePDENet reference notes.

- `docs/20260608_burgers_downsample_svd_1024_512_256_summary.md`  
  Burgers `1024 x 1024` full Jacobian versus `512/256` coarse proxy results, including baseline/loss1/loss2/loss3 model and error Jacobians.

- `docs/20260608_burgers_frequency_projection_residual_summary.md`  
  High-frequency versus block-residual analysis, FFT correlations, projection residual interpretation, and why error Jacobians are harder.

- `docs/20260608_burgers_training_attack_visual_summary.md`  
  Burgers round03 loss1/loss2/loss3 training, loss1 `5000 -> 8000`, p2q2 attack visual/metric summary, and all-dataset 20-step p2q2 attack pilot status.

- `docs/20260608_darcy_flow_self_training_summary.md`  
  Darcy Flow lossdrop50 dataset selection, loss3 500-epoch self-training result, train/test/generalization behavior, accuracy-score definition, and Jacobian feasibility.

## Detailed Existing Reports Referenced

- `docs/burgers_jacobian_downsample_svd_probe_20260607.md`
- `docs/burgers_round03_loss123_downsample_svd_probe_20260607.md`
- `docs/burgers_round03_svd_vector_frequency_proxy_interpretation_20260608.md`
- `docs/burgers_round03_p2q2_metrics_record_20260607.md`
- `docs/burgers_round03_full_p2q2_20step_pilot_launch_20260607.md`
- `docs/burgers_loss1_5000to8000_launch_20260607.md`
- `docs/burgers_loss3_selective_round03_loss1_8000_plot_report_20260607.md`
- `docs/darcy_lossdrop50_loss3_500ep_launch_20260607.md`
- `docs/darcy_candidate_generalization_gradient_screen_20260607.md`
- `docs/darcy_flow_naming_audit_20260607.md`

## Main One-Line Conclusions

Observed from saved full-SVD and downsample proxy tables: solver/model Jacobians are smooth enough that `512` and often `256` proxies are close to the full `1024` structure; model-minus-solver error Jacobians are more fragile.

Observed from projection-residual analysis: for 256 block projection, top1 block residual is around `0.02-0.03` for model Jacobians but around `0.05-0.10` for error Jacobians. This explains why error-Jacobian sigma estimates can be off by `20-40%` in worst cases.

Observed from FFT/projection correlation: higher Fourier high-frequency fraction is positively associated with larger block residual; at coarse size 256, all-matrix Spearman correlation is `0.8753` for top1 and `0.8998` for top10 weighted vectors, while error-only Spearman is `0.7936` and `0.7102`.

Observed from Burgers p2q2 six-sample attack metrics: under the same `delta_rms≈0.12` and 100 attack steps, loss3 epoch1500 has the best attacked-output stability by mean MSE/RMS, but loss1/loss2/loss3 should be compared together rather than only loss3 versus baseline.

Observed from Darcy Flow loss3 self-training: selected 50 generalization datasets improved clean generalization RMSE/relative-L2 by about `46%`, but train/test clean metrics worsened, indicating specialization to the selected generalization distribution.

Observed limitation: local full-Jacobian SVD artifacts exist for round03 `loss1_epoch5000`, `loss2_epoch2000`, and `loss3_epoch1500`. A local `loss1_epoch8000` full-Jacobian NPZ was not found, so the downsample-Jacobian comparison uses `loss1_epoch5000` for loss1.
