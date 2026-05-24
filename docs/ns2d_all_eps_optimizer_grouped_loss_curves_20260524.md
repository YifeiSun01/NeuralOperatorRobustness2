# NS2D All-Epsilon Optimizer-Grouped Loss Curves - 2026-05-24

Observed from saved `final_state_outputs.npz` and `per_step_metrics.csv` files
in the gitclean data tree. No model inference, solver rollout, attack update,
PyTorch, JAX, or GPU work was run for this plotting task.

## Outputs

- Full report:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns2d_all_eps_optimizer_grouped_loss_curves_20260524.md`
- PNG directory:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/optimizer_grouped_loss_curves_20260524/ns2d_all_eps/`
- Bundle:
  `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz`

## Generated Figures

- `eps160_alpha50_optimizer_grouped_target_loss_curves.png`
- `eps32_alpha10_optimizer_grouped_target_loss_curves.png`
- `eps16_alpha5_optimizer_grouped_target_loss_curves.png`
- `eps8_alpha2p5_optimizer_grouped_target_loss_curves.png`
- `eps4_alpha1p25_optimizer_grouped_target_loss_curves.png`
- `eps2_alpha0p625_optimizer_grouped_target_loss_curves.png`
- `eps1_alpha0p3125_optimizer_grouped_target_loss_curves.png`

## Scope

Each PNG fixes one epsilon/alpha setting. Within a PNG, each row fixes one
available target loss/mode, and the four optimizer methods are overlaid.

`eps32_alpha10` and `eps8_alpha2p5` include the extra available `loss3` mode
variants. The other historical epsilon settings include the canonical targets
visible in the organized local artifacts. The current `eps160_alpha50` figure
only includes completed targets visible in the gitclean data tree; the
background loss1/loss2 run can be synced and replotted after it finishes.
