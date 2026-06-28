# Loss1/Loss2 Core-Four P2Q2 Baseline Visuals - 2026-05-21

Status: completed on GPU from a new baseline run.

## Source Data

- Output root: `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2`
- Per-method source files: `per_step_metrics.csv`, `per_sample_step_metrics.csv`, `trajectory_samples.npz`, and `final_delta.npz`.
- GPU evidence: `Tesla V100-SXM2-32GB`, capability `sm_70`, PyTorch `2.8.0+cu126`, JAX backend `gpu`.

## Settings

- FNO / 1D Burgers `nu=0.001`; batch size `100`, dataset indices `0..99`.
- `epsilon=4.0`, `alpha=0.4`, `steps=300`, `p=q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- `loss1_original` uses tiny random start `1e-6`; `loss2_original` uses zero start.

## Summary Table

| objective | method | final_optimized_loss_mean | final_fraction_of_best | step_to_95pct_best_final | final_delta_pnorm_mean | final_high_frequency_energy_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- |
| loss1_original | raw_add | 6.983 | 0.9999 | 7 | 4 | 4.052e-10 |
| loss1_original | raw_replace | 6.942 | 0.9941 | 2 | 4 | 4.094e-10 |
| loss1_original | steepest_add | 6.983 | 1 | 10 | 4 | 4.326e-10 |
| loss1_original | steepest_replace | 6.942 | 0.9941 | 2 | 4 | 4.094e-10 |
| loss2_original | raw_add | 7.076 | 0.9997 | 7 | 4 | 6.924e-10 |
| loss2_original | raw_replace | 7.045 | 0.9953 | 2 | 4 | 6.725e-10 |
| loss2_original | steepest_add | 7.078 | 1 | 10 | 4 | 7.103e-10 |
| loss2_original | steepest_replace | 7.045 | 0.9953 | 2 | 4 | 6.725e-10 |

## Figures

- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss1_original_mean_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss1_original_boundary_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss1_original_high_frequency_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss1_original_final_delta_selected_indices.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss1_original_delta_heatmap_index0.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss2_original_mean_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss2_original_boundary_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss2_original_high_frequency_ratio_vs_step.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss2_original_final_delta_selected_indices.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/loss2_original_delta_heatmap_index0.png`

## Observed Evidence

Observed from the generated CSVs and figures: replacement methods jump to the boundary immediately by construction, while additive methods grow the perturbation over many steps. For `p=q=2`, `raw_replace` and `steepest_replace` use the same normalized direction for these objectives and therefore are expected to nearly overlap.

Inference should stay within this single baseline setting unless more epsilon/alpha or P/Q combinations are run.
