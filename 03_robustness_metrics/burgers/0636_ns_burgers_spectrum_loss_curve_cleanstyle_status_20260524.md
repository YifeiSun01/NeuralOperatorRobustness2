# NS2D And Burgers Spectrum/Loss-Curve Redraw Status - 2026-05-24

Observed from regenerated PNGs and source scripts in this working tree.

## Completed

- Redrew NS2D clean-style panels: `24` PNGs in `docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/`.
- Redrew Burgers clean-style panels: `16` PNGs in `docs/burgers_spectrum_loss_curves_cleanstyle_20260524/figures/`.
- Removed the extra explanatory notes from the figure canvas. The figures now keep only the scientific labels, parameters, dataset index, panels, and row metrics.
- Burgers loss curves now use a unified loss3/residual-L2 evaluation metric. `Loss 1`, `Loss 2`, and `Loss 3` denote the attack target used to generate the perturbation; the y-axis is the common loss3/residual metric.
- Spectrum panels use normalized frequency radius `0` to `0.25` and show log amplitude. Loss 2 and Loss 3 lines are drawn thicker in the spectrum panels; Loss 3 is drawn thickest in the loss-curve panels.
- NS2D `epsilon = 1, alpha = 0.3125` now includes the locally补跑的 `loss1`, `loss2`, and `loss3` rows.

## Evidence

- NS2D report: `docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524.md`.
- Burgers report: `docs/burgers_spectrum_loss_curves_cleanstyle_20260524.md`.
- NS2D manifest: `docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/manifest.json`.
- Burgers manifest: `docs/burgers_spectrum_loss_curves_cleanstyle_20260524/manifest.json`.
- Burgers unified step-loss cache: `docs/burgers_spectrum_loss_curves_cleanstyle_20260524/selected_step_loss3_curves_cache.npz`.
- Burgers GPU evidence: `docs/burgers_spectrum_loss_curves_cleanstyle_20260524/gpu_evidence_selected_step_loss3.json`.

## Limitation

Observed from the saved Burgers historical artifacts: full per-step `x_adv` for all dataset samples is not present for `loss1` and `loss2`. The Burgers loss-curve panel therefore shows the selected dataset index and the mean/std over the saved trajectory indexes, rather than inventing an all-dataset per-step curve.

## Bundle

- Download bundle: `docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz`.
- Bundle contents: one top-level folder with `NS2D/` and `Burgers/` subfolders, containing `41` PNGs plus regenerated reports, manifests, CSVs, and this status file. Large numeric caches are left outside the bundle.

## Epsilon 160 Update

- Added the completed NS2D `Epsilon = 160, Alpha = 50` run to the NS2D figure set.
- The added case is `steepest_add`, `loss3/all_a_target_w`, dataset index `0`, generated from `eps160_alpha50/mode_aaaaaaaaaw_p2_q2_20260524_073831_UTC`.
- New PNG: `docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps160_alpha50_steepest_add_spectrum_loss_curves_dataset0.png`.
- Bundle copy: `docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524/NS2D/figures/eps160_alpha50_steepest_add_spectrum_loss_curves_dataset0.png`.
- NS2D PNG count is now `25`; total bundle PNG count is now `41`.
