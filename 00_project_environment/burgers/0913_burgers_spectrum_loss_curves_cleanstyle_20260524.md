# Burgers Spectrum And Loss Curve Clean-Style Panels - 2026-05-24

Observed from saved Burgers trajectory artifacts and GPU evaluation of saved `x_adv`; no attack optimization or perturbation update was run.

Each PNG has a 2 x 2 top line-plot block. Loss 1, Loss 2, and Loss 3 name the attack target; the plotted loss curves use one common loss3/residual-L2 metric.

## Figures

| method | dataset index | clean L2 | highlighted row | highlighted residual L2 | adv L2 loss1/loss2/loss3 | figure |
|---|---|---:|---|---:|---:|---|
| raw_add | 0 | 0.333 | loss3 | 3.72 | 0.53/0.522/3.72 | [burgers_spectrum_loss_curves_raw_add_dataset000.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_add/burgers_spectrum_loss_curves_raw_add_dataset000.png) |
| raw_add | 7 | 0.297 | loss3 | 4.12 | 0.333/0.335/4.12 | [burgers_spectrum_loss_curves_raw_add_dataset007.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_add/burgers_spectrum_loss_curves_raw_add_dataset007.png) |
| raw_add | 40 | 0.268 | loss3 | 3.37 | 0.596/0.349/3.37 | [burgers_spectrum_loss_curves_raw_add_dataset040.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_add/burgers_spectrum_loss_curves_raw_add_dataset040.png) |
| raw_add | 47 | 0.291 | loss3 | 3.31 | 0.785/0.81/3.31 | [burgers_spectrum_loss_curves_raw_add_dataset047.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_add/burgers_spectrum_loss_curves_raw_add_dataset047.png) |
| raw_replace | 0 | 0.333 | loss3 | 4.32 | 0.49/0.483/4.32 | [burgers_spectrum_loss_curves_raw_replace_dataset000.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_replace/burgers_spectrum_loss_curves_raw_replace_dataset000.png) |
| raw_replace | 7 | 0.297 | loss3 | 3.91 | 0.32/0.322/3.91 | [burgers_spectrum_loss_curves_raw_replace_dataset007.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_replace/burgers_spectrum_loss_curves_raw_replace_dataset007.png) |
| raw_replace | 40 | 0.268 | loss3 | 3.18 | 0.518/0.351/3.18 | [burgers_spectrum_loss_curves_raw_replace_dataset040.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_replace/burgers_spectrum_loss_curves_raw_replace_dataset040.png) |
| raw_replace | 47 | 0.291 | loss3 | 3.13 | 0.808/0.808/3.13 | [burgers_spectrum_loss_curves_raw_replace_dataset047.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/raw_replace/burgers_spectrum_loss_curves_raw_replace_dataset047.png) |
| steepest_add | 0 | 0.333 | loss3 | 3.72 | 1.1/0.521/3.72 | [burgers_spectrum_loss_curves_steepest_add_dataset000.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_add/burgers_spectrum_loss_curves_steepest_add_dataset000.png) |
| steepest_add | 7 | 0.297 | loss3 | 4.12 | 0.333/0.335/4.12 | [burgers_spectrum_loss_curves_steepest_add_dataset007.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_add/burgers_spectrum_loss_curves_steepest_add_dataset007.png) |
| steepest_add | 40 | 0.268 | loss3 | 3.37 | 0.596/0.35/3.37 | [burgers_spectrum_loss_curves_steepest_add_dataset040.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_add/burgers_spectrum_loss_curves_steepest_add_dataset040.png) |
| steepest_add | 47 | 0.291 | loss3 | 3.31 | 0.786/0.812/3.31 | [burgers_spectrum_loss_curves_steepest_add_dataset047.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_add/burgers_spectrum_loss_curves_steepest_add_dataset047.png) |
| steepest_replace | 0 | 0.333 | loss3 | 4.32 | 0.49/0.483/4.32 | [burgers_spectrum_loss_curves_steepest_replace_dataset000.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_replace/burgers_spectrum_loss_curves_steepest_replace_dataset000.png) |
| steepest_replace | 7 | 0.297 | loss3 | 3.91 | 0.32/0.322/3.91 | [burgers_spectrum_loss_curves_steepest_replace_dataset007.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_replace/burgers_spectrum_loss_curves_steepest_replace_dataset007.png) |
| steepest_replace | 40 | 0.268 | loss3 | 3.18 | 0.518/0.351/3.18 | [burgers_spectrum_loss_curves_steepest_replace_dataset040.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_replace/burgers_spectrum_loss_curves_steepest_replace_dataset040.png) |
| steepest_replace | 47 | 0.291 | loss3 | 3.13 | 0.808/0.808/3.13 | [burgers_spectrum_loss_curves_steepest_replace_dataset047.png](burgers_spectrum_loss_curves_cleanstyle_20260524/figures/steepest_replace/burgers_spectrum_loss_curves_steepest_replace_dataset047.png) |

## Output Files

- Index CSV: `docs/burgers_spectrum_loss_curves_cleanstyle_20260524/burgers_spectrum_loss_curves_readable_index.csv`
- Output cache: `docs/burgers_spectrum_loss_curves_cleanstyle_20260524/evaluated_outputs_cache.npz`
- Source script: `tools/plot_burgers_spectrum_loss_curves_cleanstyle_20260524.py`
