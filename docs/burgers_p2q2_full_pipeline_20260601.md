# Burgers p2q2 adversarial-training full pipeline

This note records the corrected Burgers adversarial-training pipeline prepared on 2026-06-01.

## Purpose

The previous 1000-epoch Burgers adversarial run was useful as an ablation, but its saved deltas were rectangular because it used `fast_replace_linf`.  That geometry sets coordinatewise sign-like perturbations and is not the intended p=2,q=2 setting.

The corrected pipeline pins Burgers to:

```text
attack_method = fast_replace_l2
training guard = --burgers-require-p2q2
attack geometry = p=2, q=2 RMS-L2 replace direction
training_data_mode = adv-only
label_mode = solver
epochs = 1000
checkpoint_every_epochs = 200
attack_steps = 5
batch_size = 480
optimizer_batch_size = 32
epsilon_fraction = 0.06
epsilon jitter = [0.75, 1.25]
alpha_ratio = 1.0
alpha jitter = [0.75, 1.25]
epsilon buckets = 5
fixed attack probes saved every epoch
```

The only intended conceptual difference from the previous zero/adversarial 1000-epoch run is that the perturbation is L2-like rather than coordinatewise `+epsilon/-epsilon`.

## Entry points

```text
tools/run_burgers_p2q2_full_pipeline.py
tools/run_burgers_p2q2_full_pipeline.sh
tools/validate_burgers_l2_delta_geometry.py
tools/compare_burgers_checkpoint_series_jacobian_svd.py
```

Smoke test for one epoch / one batch:

```bash
tools/run_burgers_p2q2_full_pipeline.sh --mode smoke --run-name burgers_p2q2_smoke_20260601 --skip-r2-upload --skip-git-push
```

Full run:

```bash
tools/run_burgers_p2q2_full_pipeline.sh --mode full
```

The full pipeline runs training, checks delta geometry, generates the selected polished visualizations, then computes top-100 Jacobian/SVD diagnostics on the fixed representative20 sample manifest. The p2q2 guard rejects any accidental Burgers `_linf` method before training starts.

## Delta geometry check

The validator reads `attack_probe_samples.csv` and the fixed-probe NPZ files.  A pass requires:

- metadata attack method ends in `_l2`
- metadata attack type is `continuous_l2_rms`
- saved deltas have many rounded amplitudes
- saved deltas are not dominated by a single absolute amplitude

This is meant to catch the exact failure mode that produced rectangular `Linf` sign deltas.

## Post-training figures

The pipeline reuses the existing selected-polished visualization scripts and collects the same selected figure set into:

```text
visualizations/burgers_p2q2_adv_training_20260601/polished_selected_download_burgers_p2q2_20260601/
/workspace/polished_selected_download_burgers_p2q2_20260601/
```

Selected figures include the corrected attack-loss plot, relative L2/RMSE heatmap-plus-line plots, delta FFT spectrum figure, and max-five high-transparency line panels.

## Jacobian/SVD analysis

The checkpoint-series SVD script reuses the previous representative20 fixed input points.  For each selected checkpoint epoch `200, 400, 600, 800, 1000`, it computes:

```text
J_model(x)
J_error(x) = J_model(x) - J_solver(x)
```

It reuses prior `J_solver`, `baseline`, and `baseline_error` NPZ files when available, and saves only top-100 singular triplets for new checkpoint model/error Jacobians.

Main outputs:

```text
checkpoint_series_jacobian_svd_summary.csv
checkpoint_series_top_singular_values_long.csv
checkpoint_series_solver_similarity_rankwise.csv
checkpoint_series_solver_similarity_subspaces.csv
checkpoint_series_error_aggregate.csv
checkpoint_series_summary.md
sample_*/
```

The rankwise similarity tables compare singular values and left/right singular vectors to the solver.  The subspace table compares top-k left/right singular subspaces for k in `1, 5, 10, 20, 50, 100`.

## Sync behavior

The pipeline can upload to R2 and push to GitHub, but credentials are not written into the code or this document.  R2 sync uses `tools/upload_path_to_r2_20260525.sh` and expects credentials through environment variables or an rclone config.  R2 receives the full run, visualization, and forensics directories.  GitHub receives lightweight reproducibility artifacts only: code, markdown, manifests, selected figures, and CSV/Markdown summaries.  Large checkpoint and SVD NPZ payloads stay on R2 rather than being committed to Git.
