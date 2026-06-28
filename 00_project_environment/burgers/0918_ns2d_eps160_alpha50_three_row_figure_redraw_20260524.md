# NS2D eps160 alpha50 Three-Row Figure Redraw - 2026-05-24

Status: completed.

## Observed Evidence

- Source eps160 data in the main worktree contains completed `loss1/all_w`, `loss2/all_a_target_w`, and `loss3/all_a_target_w` `steepest_add` outputs.
- Synced `loss1` and `loss2` eps160 directories into the gitclean data tree with `rsync -a --exclude step_sample_trace.npz`.
- Verified gitclean now has `summary.json`, `per_step_metrics.csv`, and `final_state_outputs.npz` for all three eps160 targets.
- Re-ran `tools/plot_ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524.py` in `/workspace/NeuralOperatorRobustness2_gitclean`; it rendered `25` NS2D PNG files.
- Re-ran `tools/plot_ns2d_all_eps_optimizer_grouped_loss_curves_20260524.py`; eps160 all-epsilon optimizer-grouped report now lists `3` targets.

## Updated Figure

- Main figure: `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps160_alpha50_steepest_add_spectrum_loss_curves_dataset0.png`
- Bundle figure: `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/NS2D/figures/eps160_alpha50_steepest_add_spectrum_loss_curves_dataset0.png`
- Updated bundle tarball: `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz`

## Figure Row Check

Observed from `row_loss_summary.csv` for `eps160_alpha50`:

| row | adv true_loss for dataset index 0 | highlighted max row |
|---|---:|---|
| Clean baseline | 34.8235 | no |
| Loss 1 / all W | 223.381 | no |
| Loss 2 / all A -> W | 312.851 | yes |
| Loss 3 / all A -> W | 233.479 | no |

## Notes

- The figure-level max row is based on the selected dataset index shown in the PNG, not the 10-sample mean from the completion summary.
- No attack optimization, model training, or solver rollout was run for this redraw; this was data sync plus offline plotting.
- The large `step_sample_trace.npz` files for loss1/loss2 were intentionally not copied into gitclean because this figure does not need them.
