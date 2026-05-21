# Loss3 Overnight Readiness Review - 2026-05-20

Status: code/runbook readiness review. No optimizer experiment was launched during this review.

## Verification Performed

Observed from local checks:

- `bash -n tools/run_loss3_overnight_20260520.sh` passed.
- `py_compile` passed for all scripts used by the overnight workflow:
  - `tools/run_loss3_direction_proposal_ablation.py`
  - `tools/run_loss3_alpha_epsilon_core4_sweep.py`
  - `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`
  - `tools/plot_loss3_alpha_epsilon_core4_visuals.py`
  - `tools/analyze_loss3_alpha_epsilon_core4_delta_similarity.py`
  - `tools/analyze_loss3_post_boundary_mechanism.py`
  - `tools/plot_loss3_trajectory_gif_panels.py`
- Dry-run plan for the 300-step `p=2,q=2` sweep resolved to `20` settings, methods `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`, steps `300`.
- Dry-run plan for strict `p != q` resolved to `120` settings: 6 off-diagonal P/Q pairs times 20 alpha/epsilon settings.
- Dry-run plan for the baseline GIF trace resolved to `epsilon=4`, `alpha=0.4`, `steps=300`, core four methods, `trajectory_npz=true`, `trajectory_final_conditions_npz=true`, and `gifs=true`.

## Requirement Coverage

| User requirement | Current code coverage | Main outputs |
|---|---|---|
| Compare four methods only: raw add, raw replace, steepest add, steepest replace | Yes. Sweep wrapper default methods are exactly these four. | Per-run `per_step_metrics.csv`, summaries, figures |
| Expand alpha/epsilon from 11 to 20 | Yes. `DEFAULT_SETTING_PAIRS` has 20 curated settings. | Sweep plan and all sweep roots |
| First fill the 9 missing `p=2,q=2`, 100-step settings | Yes. Overnight stage 1 uses existing root and `--skip-completed`, so completed roots are skipped and pending roots run. | `forensics/loss3_alpha_epsilon_core4_sweep_20260519/` |
| Run all 20 `p=2,q=2` settings to 300 steps | Yes. Overnight stage 2 uses a separate 300-step root. | `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/` |
| Baseline `epsilon=4, alpha=0.4` detailed trajectory for GIF | Yes. Overnight stage 3 runs 300 steps with selected trajectory samples and final-condition arrays. | `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/` |
| Save every selected step's delta and perturbed initial condition | Yes for selected trajectory indices. `trajectory_samples.npz` stores `delta`, `x_adv`, and `perturbed_initial`. | each method's `trajectory_samples.npz` |
| Save final condition after perturbation | Yes for baseline GIF trace when `--save-trajectory-final-conditions` is used. It stores `model_final_condition`, `solver_final_condition`, and `final_condition_residual`. | each method's `trajectory_samples.npz` plus `trajectory_samples_schema.json` |
| Run `p != q` cases for 100 steps | Yes for strict off-diagonal pairs `1:2`, `1:inf`, `2:1`, `2:inf`, `inf:1`, `inf:2`. | `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/` |
| Automatically analyze and plot after runs | Yes. Overnight script runs analysis, visualization, final-delta similarity, and post-boundary diagnostics after each major run block. | `docs/`, `forensics/*analysis*`, `forensics/*visuals*`, `forensics/*similarity*`, `forensics/*post_boundary*` |
| Mark boundary hits on loss curves | Yes. Loss curves mark first mean boundary-ratio hits at 25%, 50%, 75%, and 99%. | `loss3_q_mean_curves_with_boundary_markers.png` |
| Plot loss, epsilon/norm ratio, and angular motion | Yes for newly generated artifacts with angular fields. Visualization adds loss curves, boundary-ratio curves, angular-speed curves, and representative dynamics triptychs. | `delta_prev_angle_degrees_mean_curves.png`, `dynamics_triptychs/*.png` |
| Use batch mean with std shadow | Yes. Updated visualization shades mean +/- std for line plots where std fields exist. | loss/boundary/angle curve figures and triptychs |
| Record per-step perturbation angular change | Yes for future runs. Runner writes `delta_prev_cosine`, `delta_prev_angle_degrees`, step distances, and normalized step distances. | `per_step_metrics.csv`, `per_sample_step_metrics.csv` |
| Quantify why GPI grows after boundary | Yes. Post-boundary diagnostic script summarizes post-boundary loss gain, short-window gain, tangent-motion proxies, projection shrink, direction change, and angular metrics when present. | `post_boundary_mechanism_by_setting.csv`, `post_boundary_mechanism_rollup.csv` |
| Compare final perturbation similarity | Yes. Delta similarity script computes cosine, centered cosine, spectral magnitude cosine, relative L2, and sign agreement. | `final_delta_pairwise_similarity_*.csv`, similarity figures |

## Important Caveat

Observed issue/caveat: the old 11 completed `p=2,q=2`, 100-step roots were produced before `delta_prev_angle_degrees` and related angular-speed fields existed. Stage 1 intentionally uses `--skip-completed`, so those old 11 roots will remain missing the new angle fields. The 9 newly filled 100-step roots will have the fields, and the full 20-setting 300-step root will have complete angular metrics for all 20 settings.

Inference: for complete angular-speed analysis across all 20 `p=2,q=2` settings, use the 300-step outputs. If a complete 100-step angular-speed version of all 20 settings is needed, rerun the 100-step `p=2,q=2` sweep with `--no-skip-completed` or into a fresh root.

## Expected Main Figure Outputs

For each visualization root, expect:

- `figures/loss3_q_mean_curves_with_boundary_markers.png`
- `figures/boundary_ratio_mean_curves.png`
- `figures/delta_prev_angle_degrees_mean_curves.png` when angular fields exist
- `figures/heatmap_final_loss3_q_mean.png`
- `figures/heatmap_step_to_boundary_ratio_mean_0p99.png`
- `figures/heatmap_final_high_frequency_energy_ratio_mean_log10.png`
- `figures/heatmap_final_first_derivative_l2_mean.png`
- `figures/heatmap_final_delta_peakiness.png`
- `figures/dynamics_triptychs/dynamics_triptych_<setting>.png`
- `figures/representative_samples/*.png`
- `figures/delta_shape_grids/*.png`

For the baseline GIF trace, expect:

- Built-in delta GIFs under the run root's `figures/gifs/` when `--make-gifs` succeeds.
- Multi-panel GIFs under `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/trajectory_condition_gifs/`.

## Launch Command

```bash
cd /workspace/NeuralOperatorRobustness2
bash tools/run_loss3_overnight_20260520.sh
```

Background launch:

```bash
cd /workspace/NeuralOperatorRobustness2
mkdir -p logs
nohup bash tools/run_loss3_overnight_20260520.sh > logs/loss3_overnight_launcher_$(date -u +%Y%m%dT%H%M%SZ).out 2>&1 &
```
