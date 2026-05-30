# Generalization Relative-L2 Bar Plots V4: Original NS Style

This version redraws the relative-L2 bar plots using the visual encoding from `2D_NS_FNO2d_recurrent/eval_models/plot_results.py`.

## Encoding

- Generalization dataset color is keyed by a range/parameter label, using a `coolwarm` palette, matching the original range-palette idea.
- Hatch/texture encodes kernel or the closest available dataset-family analogue: `gaussian=-`, `rbf=.`, `rq=\`, `matern=/`, `periodic=x`.
- Test and train use the original fixed colors: test `#ff7f0e`, train `#2ca02c`.
- Rows are ordered in the original group style: generalizability first, then test, then train; within generalizability, bars are grouped by kernel and range label.

## Source Style Code

- `2D_NS_FNO2d_recurrent/eval_models/plot_results.py`
- Constants reused: `KERNEL_ORDER`, `KERNEL_HATCH`, `GROUP_ORDER`, `GROUP_BASE_COLOR`, and the range-palette logic.

## Plots

- `generalization_eval/burgers_generalization_relative_l2_loss_barplot_v4_original_ns_style.png`
- `generalization_eval/darcy_generalization_relative_l2_loss_barplot_v4_original_ns_style.png`
- `generalization_eval/ns2d_generalization_relative_l2_loss_barplot_v4_original_ns_style.png`
