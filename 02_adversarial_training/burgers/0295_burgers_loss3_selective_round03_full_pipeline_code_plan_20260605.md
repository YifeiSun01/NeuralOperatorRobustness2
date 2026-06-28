# Burgers Round03 Long Pipeline Code Plan

Date: 2026-06-05 UTC.

This document records the code path for the requested full Burgers round03 experiment. It is a code/pipeline record, not the final numerical conclusion. The long run and posthoc diagnostics write large local artifacts under `adversarial_training_runs/` and `forensics/`; those artifacts should not be committed to Git by default.

## Goal

Run the loss3-selective round03 generalization dataset as a full experiment:

- `loss1`: 1000 epochs.
- `loss2`: 500 epochs.
- `loss3`: 500 epochs.
- Save periodic checkpoints and explicit wall-clock checkpoints.
- Compare equal-epoch and wall-clock-aligned prediction metrics on train/test/generated50 generalization datasets.
- Record adversarial attack delta statistics.
- Replay gradient-alignment trajectory on the round03 dataset.
- Recompute local Burgers solver/model/error Jacobian SVD on the new round03 inputs; do not reuse old solver/baseline Jacobians.

## Pipeline Entry Point

Use:

```bash
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python   tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py   --stages all   --skip-existing
```

Available stages:

```text
preflight,train,summarize,gradient,svd-final,svd-wall
```

Useful continuation commands:

```bash
# After training finishes, only build prediction/wall/delta summaries.
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python   tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py   --stages summarize   --skip-existing

# After summary exists, run Jacobian/SVD diagnostics.
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python   tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py   --stages svd-final,svd-wall   --skip-existing
```

The orchestrator refuses to start duplicate training if a run directory exists without `summary.json`, because that usually means a run is already in progress or incomplete.

## Training Settings

The training stage calls `tools/adversarial_training.py` with the round03 generalization root:

```text
generalization_datasets_burgers_loss3_selective_search/round_03
```

Common settings:

```text
--tasks burgers
--device cuda
--seed 20260601
--checkpoint-every-epochs 100
--checkpoint-wall-hours 0.5,1.0,1.38,1.4,2.0,3.0,3.05,3.1,4.0,5.0,6.0,6.3
--training-data-mode adv-only
--label-mode solver
--epsilon-bucket-count 5
--attack-probe-samples 5
--attack-probe-every-n-epochs 1
--attack-probe-save-targets
--burgers-attack-method fast_replace_l2
--burgers-require-p2q2
--burgers-attack-steps 5
--burgers-batch-size 480
--burgers-optimizer-batch-size 32
--burgers-epsilon-fraction 0.06
--burgers-eps-jitter-low 0.75
--burgers-eps-jitter-high 1.25
--burgers-alpha-ratio 1.0
--burgers-alpha-jitter-low 0.75
--burgers-alpha-jitter-high 1.25
--burgers-random-start-fraction 1e-6
--eval-max-samples 0
--max-generalization-eval 50
```

Run names:

```text
adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605
adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605
adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605
```

Each run writes `eval_metrics.csv`, `eval_split_summary.csv`, `checkpoints.csv`, `attack_epoch_summary.csv`, `attack_probe_samples/*.npz`, model checkpoints, memory logs, config, and summary.

## Summary Outputs

`tools/summarize_burgers_round03_long_training.py` writes:

```text
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/epoch_aligned_split_metrics.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_checkpoint_selection.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_aligned_split_metrics.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_pairwise_ratios.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/per_dataset_generated_advantage.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/per_dataset_generated_advantage_summary.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/best_epoch_by_split.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/attack_delta_selected_epochs.csv
forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/attack_delta_summary_by_loss.csv
```

It also writes the report draft:

```text
docs/burgers_loss3_selective_round03_long_training_report_20260605.md
```

## Gradient Alignment

The pipeline stage `gradient` calls:

```text
tools/probe_burgers_p2q2_loss123_50step_gradient_alignment_trajectory.py
```

Output:

```text
forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_50step_long_pipeline_20260605/
```

It records parameter-gradient cosine alignment for `loss1_raw`, `loss2_raw`, and `loss3_raw` against clean train/test and mixed round03 generalization subsets.

## Jacobian/SVD

The updated `tools/compare_burgers_round01_final_jacobian_svd.py` now accepts:

```text
--output-prefix
--report-title
--report-note
--loss1-label/--loss2-label/--loss3-label
--loss1-epoch/--loss2-epoch/--loss3-epoch
```

This keeps the original implementation but allows round03 output filenames instead of misleading `round01_*` names.

Final checkpoint SVD output:

```text
forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/
```

Wall-clock SVD outputs:

```text
forensics/burgers_loss3_selective_round03_long_samewall_loss1_final_jacobian_svd_rep10_top50_20260605/
forensics/burgers_loss3_selective_round03_long_samewall_loss2_final_jacobian_svd_rep10_top50_20260605/
```

The SVD code recomputes:

- `J_solver(x)`
- `J_model(x)`
- `J_error(x) = J_model(x) - J_solver(x)`
- top singular values
- spectral/Frobenius/effective-rank summaries
- rankwise singular-vector similarity to solver
- top-k singular-subspace similarity to solver

## Current Status

Observed from local process status while writing this plan: the previously launched `loss1_1000ep_long` run is still in progress and has passed preflight. It should not be duplicated. Once it and the subsequent `loss2`/`loss3` runs finish, run the `summarize`, `gradient`, `svd-final`, and `svd-wall` stages.

## Automatic Posthoc Watcher

Because the currently running process was originally launched with the training-only shell `tools/run_burgers_loss3_selective_round03_long_training_20260605.sh`, it would not by itself start posthoc analysis after `loss3` completes.

To make the workflow automatic, a watcher was launched on 2026-06-05 UTC:

```text
PID: 477572
log: adversarial_training_runs/burgers_loss3_selective_round03_full_pipeline_20260605_logs/posthoc_after_loss3_watcher_20260605_150814_UTC.log
```

The watcher waits for:

```text
adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/summary.json
```

Then it runs:

```bash
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python   tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py   --stages summarize,plots,gradient,svd-final,svd-wall   --skip-existing
```

The `plots` stage now includes both per-run visualization and cross-loss comparison plots:

- per-run plots from `tools/plot_burgers_training_run_visualizations_variable_epoch.py`
- comparison plots from `tools/plot_burgers_round03_long_training_comparison.py`

Expected plot output roots:

```text
visualizations/burgers_loss3_selective_round03_loss1_3000ep_long_20260605_plots/
visualizations/burgers_loss3_selective_round03_loss2_1000ep_long_20260605_plots/
visualizations/burgers_loss3_selective_round03_loss3_1000ep_long_20260605_plots/
visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/
```

### Watcher v2 Correction

The first watcher process (`PID 477572`) exited after its first status check. A more robust detached watcher was then created and launched with `setsid`:

```text
script: tools/watch_burgers_round03_posthoc_after_loss3_20260605.sh
PID: 478981
log: adversarial_training_runs/burgers_loss3_selective_round03_full_pipeline_20260605_logs/posthoc_after_loss3_watcher_v2_20260605_150959_UTC.log
status: alive at verification; PPID 1, independent SID 478981
```

This v2 watcher waits for the same loss3 summary and runs the same automatic posthoc stages:

```bash
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python   tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py   --stages summarize,plots,gradient,svd-final,svd-wall   --skip-existing
```
