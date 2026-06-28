# NS2D/Burgers Cleanstyle Line-Range Redraw - 2026-05-24

Status: completed offline redraw and repackaging.

## What changed

- Observed code change in `tools/plot_optimizer_grouped_loss_curves_20260524.py`: optimizer-grouped y-limits now use only plotted solid-line values; `fill_between` standard-deviation bands are still drawn but no longer control the y-axis range.
- Observed code change in `tools/plot_ns2d_all_eps_optimizer_grouped_loss_curves_20260524.py`: all-epsilon NS2D optimizer-grouped y-limits now use only plotted solid-line values.
- Observed from cleanstyle scripts in `/workspace/NeuralOperatorRobustness2_gitclean/tools/`: NS2D/Burgers top spectrum and loss-curve panels compute chart limits from selected/mean solid lines only; std bands are drawn after the axis range is fixed.

## Regenerated outputs

- Deleted old generated PNGs only from the targeted figure directories: `100` PNGs.
- Regenerated NS2D cleanstyle figures: `25` PNGs under `docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/`.
- Regenerated Burgers cleanstyle figures: `16` PNGs under `docs/burgers_spectrum_loss_curves_cleanstyle_20260524/figures/`.
- Regenerated optimizer-grouped figures: `2` summary PNGs plus `7` NS2D all-epsilon PNGs under `docs/optimizer_grouped_loss_curves_20260524/`.
- Rebuilt combined bundle: `docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz`.

## Verification

- Observed from PIL verification: combined bundle directory contains `50` PNGs and `0` failed image opens.
- Observed tar size: about `43.6 MB`.
- Representative Burgers PNG size: `2250 x 1968`.

## Scope notes

- No attack optimization, model training, or solver rollout was run for this redraw.
- Burgers cleanstyle redraw used existing cached evaluation outputs; the cache was present before running.
- This redraw does not add missing eps160 loss1/loss2 rows; it only redraws currently evidenced local outputs.

## Main outputs

- Bundle directory: `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/`
- Download tarball: `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz`
- NS2D figures: `/workspace/NeuralOperatorRobustness2_gitclean/docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/`
- Burgers figures: `/workspace/NeuralOperatorRobustness2_gitclean/docs/burgers_spectrum_loss_curves_cleanstyle_20260524/figures/`
