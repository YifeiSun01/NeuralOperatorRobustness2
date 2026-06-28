# Y-Axis Metric Log-Log Plots

Generated: 2026-06-19T16:50:45+00:00

Final PNG-only root:
- `outputs/final_png_only_yaxis_metric_loglog_20260619`

Y-axis metrics:
- `initial_loss`: clean loss before attack.
- `final_attack_loss`: final loss after the attack.
- `loss_increase`: `final_attack_loss - initial_loss`.
- `percent_loss_increase`: `100 * (final_attack_loss - initial_loss) / initial_loss`, computed per sample before aggregation.

For every `_with_std.png`, the y-axis range is copied from the matching no-std mean plot and is not expanded by the std band.
