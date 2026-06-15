# Darcy Binary 20260611 Current-Root Cleanup - 2026-06-15

Status: cleanup complete for locally visible obsolete-root result artifacts.

## Current Required Root

All current Darcy/SIR20 clean-evaluation, robustness, SVD/Jacobian,
correlation, heatmap, and 52-dataset x 7-model summary work must use:

```text
generalization_datasets_darcy_binary_loss3targeted_20260611/
```

This is the binary loss3-targeted Darcy generalization root used by the retained
`required_raw_figures_previous` loss curves.

## Deleted Obsolete-Root Results

Removed local generated result bundles and reports that were tied to the older
Darcy 20260607 lossdrop/soft-field root, including:

- Final seven-model robustness and 8-metric matrix bundles under
  `outputs/darcy_cflow_final_robustness_20260615*`.
- The obsolete SIR20 full/full-probe output bundles under
  `outputs/darcy_sir20_timematched_full_20260614_full*`.
- Old soft-field time-matched training/result directories under
  `adversarial_training_runs/darcy_lossdrop50_*`.
- Old soft-field Darcy forensics under
  `forensics/darcy_lossdrop50_*` and
  `forensics/darcy_stockgeneralization_gradient_screen_20260608`.
- Old Darcy/C-flow image-only bundles under
  `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608`.
- The obsolete `final_delta_fft` addendum data, figures, and manifest from the
  organized release.
- Markdown reports that summarized the obsolete-root Darcy results.
- Cross-topic or Burgers Markdown/R2 inventory text that mentioned the old
  Darcy root was cleaned so it no longer advertises obsolete Darcy results.

The raw current 20260611 binary data and its loss-curve source tables were not
deleted.

## Current Code Guard

`tools/darcy_sir20_common.py` now defaults to the 20260611 binary root and
rejects the obsolete 20260607 lossdrop/soft-field root for the modular Darcy
SIR20 final evaluation and robustness entry points.

Verification commands completed:

```text
py_compile tools/darcy_sir20_common.py tools/darcy_sir20_evaluate.py tools/darcy_sir20_robustness.py
darcy_specs(): 52 datasets total, 50 generalization datasets, all from the 20260611 binary root
obsolete-root override: rejected with RuntimeError
```
