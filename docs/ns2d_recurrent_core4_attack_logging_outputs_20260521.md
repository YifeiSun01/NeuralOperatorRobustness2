# NS2D Core4 Attack Logging Outputs

Date: 2026-05-21

This note records the attack logging behavior added for long NS2D Core4 attack runs. No GPU run was launched for this note.

## Required Logging

Each `loss_type / method / mode_spec / attack batch` output directory now records:

- `per_step_metrics.csv`: one row per attack step `k`, including surrogate loss, true loss, loss growth rates, and mean delta norms.
- `per_sample_step_metrics.csv`: one row per sample per step, including per-sample surrogate loss, true loss, loss growth rates, and delta norms.
- `delta_threshold_crossings.csv`: first time `delta_p / epsilon` reaches `25%`, `50%`, `75%`, and `100%`.
- `summary.json`: includes the final metrics and embeds the delta threshold crossing rows.
- `final_delta_and_metrics.npz`: always saves only the final `delta` and `x_adv` for the batch.

## True Loss And Surrogate Loss Curves

The most important curves are now explicit:

- `surrogate_loss`: the loss actually optimized for the current loss/method/mode group.
- `true_loss`: the full all-W solver evaluation `||F(x + delta) - G(x + delta)||` recorded for comparison.

By default, `--true-loss-every 1` records the true-loss curve at every attack step. This adds forward solver evaluation overhead for surrogate modes, but it preserves the main evidence curve. For `loss3/all_w`, the code reuses the active loss as the true loss to avoid duplicate solver work.

`per_step_metrics.csv` now includes surrogate-loss fields:

```text
active_loss
surrogate_loss_mean
surrogate_loss_mean_increase_from_k0
surrogate_loss_mean_ratio_to_k0
surrogate_loss_mean_delta_from_prev
surrogate_loss_mean_growth_per_second
```

`per_step_metrics.csv` also includes true-loss fields:

```text
true_loss_mean
true_loss_mean_increase_from_k0
true_loss_mean_ratio_to_k0
true_loss_mean_delta_from_prev
true_loss_mean_growth_per_second
```

`per_sample_step_metrics.csv` includes matching per-sample fields:

```text
surrogate_loss_value
surrogate_loss_increase_from_k0
surrogate_loss_ratio_to_k0
surrogate_loss_delta_from_prev
surrogate_loss_growth_per_second
true_loss
true_loss_increase_from_k0
true_loss_ratio_to_k0
true_loss_delta_from_prev
true_loss_growth_per_second
```

These fields make the true and surrogate loss curves and growth speed explicit without having to infer them later from raw loss columns.

## Delta Threshold Crossings

`delta_threshold_crossings.csv` records threshold hits for:

```text
25%, 50%, 75%, 100% of epsilon under the active p-norm
```

For the current planned attack `p=q=2`, this means L2 norm of `delta` relative to `epsilon`.

The file contains three scopes:

- `batch_mean`: first `k` when mean `delta_p / epsilon` reaches the threshold.
- `all_samples`: first `k` when every sample in the batch reaches the threshold.
- `sample`: first `k` for each individual sample.

At each threshold crossing, the CSV also records both `surrogate_loss_value` and `true_loss_value`.

## Final Delta Only

The final perturbation is always saved in:

```text
final_delta_and_metrics.npz
```

The `--save-steps` default is now empty, so the script does not save intermediate trajectory arrays by default. This avoids storing large per-step `delta/x_adv` arrays. If `--save-steps` is explicitly provided, then `trajectory_samples.npz` is written only for those requested steps.

Recommended command behavior for the user's current request:

```text
Do not pass --save-steps
```

That records all loss/delta curves as CSV and saves only final delta arrays.
