# Loss3 Next Run Commands - 2026-05-20

Status: command/code preparation only. No optimizer experiment was launched while creating this note.

Scope:

- Core methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- Default alpha/epsilon grid: 20 curated settings in `tools/run_loss3_alpha_epsilon_core4_sweep.py`.
- Baseline setting for detailed trajectory/GIF diagnostics: `epsilon=4`, `alpha=0.4`, `p=2`, `q=2`.
- The runner performs the required GPU verification at actual runtime.

## One-Command Overnight Run

This runs the full sequence below: fill missing `p=2,q=2` 100-step settings, refresh tables/plots, run all 20 `p=2,q=2` settings for 300 steps, run baseline detailed GIF trace, run strict `p != q` 100-step settings, and generate the post-processing tables/figures/Markdown files.

```bash
bash tools/run_loss3_overnight_20260520.sh
```

All terminal output is also written to `logs/loss3_overnight_<UTC_TIMESTAMP>.log`.

## Delta Angular-Speed Metrics

For runs started after this update, `per_step_metrics.csv` and `per_sample_step_metrics.csv` include the following step-to-step perturbation motion metrics:

- `delta_prev_cosine`: cosine similarity between `delta_k` and `delta_{k-1}`.
- `delta_prev_angle_degrees`: angle in degrees between `delta_k` and `delta_{k-1}`.
- `delta_step_l2`, `delta_step_pnorm`, `delta_step_linf`: actual step distance in perturbation space.
- `delta_step_l2_over_epsilon`, `delta_step_pnorm_over_epsilon`: step distance normalized by epsilon.
- `delta_unit_direction_l2_step`: movement of the unit-normalized perturbation direction.

The visualization script automatically adds:

- `figures/delta_prev_angle_degrees_mean_curves.png`
- `figures/dynamics_triptychs/dynamics_triptych_<setting>.png`

The triptych plots show batch mean +/- std for loss, boundary ratio, and `angle(delta_k, delta_{k-1})`.

Note: artifacts produced before this update do not contain the new angle fields unless they are rerun. The fresh 300-step and `p != q` overnight runs will contain them.

## 1. Fill The 9 Missing p=2,q=2 100-Step Settings

This uses the existing 2026-05-19 sweep root. Because `--skip-completed` is on, the existing 11 completed settings should be skipped and only the 9 new default settings should run.

```bash
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py \
  --base-out forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --steps 100 \
  --p 2 --q 2 \
  --skip-completed \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100
```

After it finishes, refresh the 20-setting p=2,q=2 tables and figures:

```bash
adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py \
  --base forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --out-dir forensics/loss3_alpha_epsilon_core4_analysis_20260519 \
  --doc docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md

adv_robust/bin/python tools/plot_loss3_alpha_epsilon_core4_visuals.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --analysis-root forensics/loss3_alpha_epsilon_core4_analysis_20260519 \
  --out-dir forensics/loss3_alpha_epsilon_core4_visuals_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_visuals_20260520.md \
  --p-filter 2 --q-filter 2 \
  --representative-settings 4:0.4 8:0.3 8:1.6 16:1.6 \
  --dataset-index 40 \
  --delta-grid-indices 0 7 40 47
```

## 2. Run The 20 p=2,q=2 Settings For 300 Steps

This is a separate root so the 100-step and 300-step comparisons do not overwrite each other.

```bash
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py \
  --base-out forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --steps 300 \
  --p 2 --q 2 \
  --skip-completed \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100 150 200 250 300 \
  --no-plots
```

After it finishes, generate 300-step tables, boundary/post-boundary gain plots, smoothness plots, and final-delta similarity:

```bash
adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py \
  --base forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md

adv_robust/bin/python tools/plot_loss3_alpha_epsilon_core4_visuals.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --analysis-root forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md \
  --p-filter 2 --q-filter 2 \
  --representative-settings 4:0.4 8:0.3 8:1.6 16:3.2 \
  --dataset-index 40 \
  --delta-grid-indices 0 7 40 47

adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_delta_similarity.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520.md \
  --p-filter 2 --q-filter 2
```

## Additional Post-Boundary Mechanism Diagnostics

After any sweep finishes, run this to quantify why a method keeps improving after the boundary is reached:

```bash
adv_robust/bin/python tools/analyze_loss3_post_boundary_mechanism.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --out-dir forensics/loss3_post_boundary_mechanism_20260520 \
  --doc docs/loss3_post_boundary_mechanism_diagnostics_20260520.md \
  --p-filter 2 --q-filter 2 \
  --boundary-threshold 0.99
```

For the 300-step run, use the same script but swap in the 300-step sweep root/output paths:

```bash
adv_robust/bin/python tools/analyze_loss3_post_boundary_mechanism.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_post_boundary_mechanism_p2q2_300steps_20260520 \
  --doc docs/loss3_post_boundary_mechanism_p2q2_300steps_20260520.md \
  --p-filter 2 --q-filter 2 \
  --boundary-threshold 0.99
```

This produces `post_boundary_loss_gain`, short-window post-boundary gain, direction-rotation proxies, tangent-motion proxies, projection-shrink summaries, and selected-trajectory smoothness changes.

## 3. Run Baseline eps=4,alpha=0.4 Detailed Trajectory For GIFs

This run records every selected sample at every step, including:

- `delta`
- `x_adv`
- `perturbed_initial`
- `clean_initial`
- `model_final_condition = f(x + delta)`
- `solver_final_condition = g(x + delta)`
- `final_condition_residual = f(x + delta) - g(x + delta)`

The arrays are stored in each method directory as `trajectory_samples.npz`.

```bash
adv_robust/bin/python tools/run_loss3_direction_proposal_ablation.py \
  --out-root forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2 \
  --methods raw_add raw_replace steepest_add steepest_replace \
  --batch-size 100 \
  --start-index 0 \
  --epsilon 4 \
  --alpha 0.4 \
  --steps 300 \
  --p 2 --q 2 \
  --seed 0 \
  --device cuda \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100 150 200 250 300 \
  --save-trajectory-final-conditions \
  --make-gifs
```

After it finishes, build multi-panel GIFs from the saved arrays:

```bash
adv_robust/bin/python tools/plot_loss3_trajectory_gif_panels.py \
  --root forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2 \
  --out-dir forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/trajectory_condition_gifs \
  --dataset-indices 0 7 40 47 \
  --frame-step 1 \
  --duration 0.16
```

## 4. Run Strict p != q Settings For 100 Steps

This command runs the same 20 alpha/epsilon settings for the six strict off-diagonal P/Q pairs. It excludes `2:2`, `1:1`, and `inf:inf` to keep the first non-p=q pass smaller.

```bash
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py \
  --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
  --steps 100 \
  --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 \
  --skip-completed \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100 \
  --no-plots
```

After it finishes, generate the combined table and per-P/Q visual folders:

```bash
adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py \
  --base forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_100steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_pneq_q_100steps_result_20260520.md

for pq in 1:2 1:inf 2:1 2:inf inf:1 inf:2; do
  p=${pq%:*}
  q=${pq#*:}
  tag_p=${p//./p}
  tag_q=${q//./p}
  adv_robust/bin/python tools/plot_loss3_alpha_epsilon_core4_visuals.py \
    --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
    --analysis-root forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_100steps_20260520 \
    --out-dir forensics/loss3_alpha_epsilon_core4_visuals_p${tag_p}_q${tag_q}_100steps_20260520 \
    --doc docs/loss3_alpha_epsilon_core4_visuals_p${tag_p}_q${tag_q}_100steps_20260520.md \
    --p-filter ${p} --q-filter ${q} \
    --representative-settings 4:0.4 8:0.3 8:1.6 16:1.6 \
    --dataset-index 40 \
    --delta-grid-indices 0 7 40 47
done
```

If the same-diagonal non-`2:2` cases are also needed later, add `1:1 inf:inf` to both `--pq-pairs` and the plotting loop.
