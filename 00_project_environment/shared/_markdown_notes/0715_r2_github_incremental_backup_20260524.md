# R2 And GitHub Incremental Backup - 2026-05-24

Status: completed.

## R2 Backup

Remote target: `r2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`.

Upload method:

- Used `rclone copy`, not `rclone sync`; no remote deletion was requested.
- For larger directories, used `--ignore-existing` so remote objects already present were skipped.
- For explicitly changed outputs, used `copyto` so the updated file replaced the older remote object.
- Temporary rclone config files were written under `/tmp` only and removed after use.

Key remote objects verified with `rclone lsl`:

| object | observed remote size |
|---|---:|
| `docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz` | 44,412,325 bytes |
| `docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps160_alpha50_steepest_add_spectrum_loss_curves_dataset0.png` | 1,058,873 bytes |
| `docs/optimizer_grouped_loss_curves_20260524/ns2d_all_eps/eps160_alpha50_optimizer_grouped_target_loss_curves.png` | 495,810 bytes |
| `docs/ns2d_eps160_alpha50_three_row_figure_redraw_20260524.md` | 2,094 bytes |
| `tools/plot_ns2d_all_eps_optimizer_grouped_loss_curves_20260524.py` | 13,586 bytes |
| `tools/plot_optimizer_grouped_loss_curves_20260524.py` | 14,635 bytes |

Backed up categories:

- Latest cleanstyle NS2D/Burgers figure bundle tarball and directory.
- Updated eps160 three-row NS2D figure and all-epsilon optimizer-grouped figure.
- eps160/alpha50 `steepest_add` result directory from gitclean, with large `step_sample_trace.npz` not copied into gitclean earlier and therefore not part of this selected backup.
- 2026-05-24 Markdown/CSV/JSON reports from `docs`.
- Critical 2026-05-24 plotting and launcher scripts from `tools`.

## GitHub Backup

Remote target: `origin/vast-ai` at `https://github.com/YifeiSun01/NeuralOperatorRobustness2.git`.

Observed pushed commit before this record:

- `5a16119 Record eps160 analysis redraw and backup scripts`

Committed files in that push were small source/record files only:

- `EXPERIMENT_LEDGER.md`
- `docs/ns2d_all_eps_optimizer_grouped_loss_curves_20260524.md`
- `docs/ns2d_eps160_alpha50_three_row_figure_redraw_20260524.md`
- `docs/ns2d_eps160_alpha50_three_target_completion_status_20260524.md`
- `docs/ns_burgers_cleanstyle_line_range_redraw_20260524.md`
- `docs/optimizer_grouped_loss_curves_20260524.md`
- `tools/plot_ns2d_all_eps_optimizer_grouped_loss_curves_20260524.py`
- `tools/plot_optimizer_grouped_loss_curves_20260524.py`
- `tools/run_ns2d_eps160_alpha50_steepest_add_loss1_loss2_20260524.sh`
- `tools/run_ns2d_eps160_alpha50_steepest_add_loss3_aw_20260524.sh`

## Scope Notes

- The main working tree still contains many tracked `D` entries from the incomplete instance copy. Those deletions were not staged or pushed.
- Large generated figures, arrays, checkpoints, and experiment results were not committed to GitHub; they were backed up to R2 instead.
- No R2 or GitHub credentials are stored in this repository file.
