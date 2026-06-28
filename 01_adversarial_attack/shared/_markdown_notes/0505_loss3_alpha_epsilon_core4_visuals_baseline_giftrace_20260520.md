# Loss3 Core-Four Alpha/Epsilon Sweep Visualizations

Generated: 2026-05-20T11:06:35.402046+00:00

## Scope

Observed from existing completed artifacts under `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`; no experiment was rerun for these plots.

Geometry note: these alpha/epsilon plots are scoped to the completed roots listed below. At generation time the completed roots are `p=2,q=2`; all-PQ alpha/epsilon comparison requires additional P/Q runs and should be plotted per P/Q pair.

Boundary markers on loss curves use the first step where mean `boundary_ratio` reaches `0.25`, `0.50`, `0.75`, and `0.99`. The `0.99` marker is the near-boundary marker for `||delta||_p ~= epsilon`, chosen to avoid floating-point exact-equality issues.

## Main Figures

- Loss curves with 25/50/75/99% boundary-hit markers: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Boundary ratio curves: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/boundary_ratio_mean_curves.png`
- Delta angular-speed curves, mean +/- std over batch: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/delta_prev_angle_degrees_mean_curves.png`
- Final loss heatmap: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/heatmap_final_loss3_q_mean.png`
- Mean 99% boundary-arrival step heatmap: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/heatmap_step_to_boundary_ratio_mean_0p99.png`
- Final high-frequency energy heatmap: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/heatmap_final_high_frequency_energy_ratio_mean_log10.png`
- Final first-derivative smoothness heatmap: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/heatmap_final_first_derivative_l2_mean.png`
- Final delta peakiness heatmap: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/heatmap_final_delta_peakiness.png`
- Final delta peakiness source table: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/tables/final_delta_peakiness_summary.csv`
- Boundary-threshold loss-gain source table: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/tables/boundary_threshold_loss_gain_summary.csv`
- 99% boundary-hit loss-gain compatibility table: `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/tables/boundary_hit_loss_gain_summary.csv`

## Representative Sample Panels

Each representative panel shows clean initial condition, final delta, clean+delta, delta spectrum, and a sample-level loss curve with a boundary-hit marker.

- `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/representative_samples/sample40_eps4_alpha0p4.png`

## Delta Shape Grids

These grids show final delta traces for the stored trajectory samples. No cross markers are used on delta plots; cross markers are reserved for boundary hits on loss curves.

- `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/delta_shape_grids/delta_grid_eps4_alpha0p4.png`

## Dynamics Triptychs

Each triptych shows batch mean +/- std for loss, boundary ratio, and per-step delta angular change. The angular panel uses `angle(delta_k, delta_{k-1})` in degrees.

- `forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520/figures/dynamics_triptychs/dynamics_triptych_eps4_alpha0p4.png`

## Inputs

- `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2` (eps=4, alpha=0.4, p=2, q=2)
