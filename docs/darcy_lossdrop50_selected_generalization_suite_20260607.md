# Darcy Flow Loss-Drop 50 Generalization Suite - 2026-06-07

## Status

Observed from local GPU run logs and CSV outputs: generated a 72-dataset Darcy Flow candidate pool from previously loss-decreasing soft-coefficient families, ran a 50-step adversarial-training gradient/loss probe, and selected 50 datasets whose individual generalization eval loss decreased.

This is still a screening-baseline result, not an official full Darcy adversarial-training result, because the official Darcy train/test/checkpoint artifacts were missing locally. The screen reused the local 50-epoch Darcy screening baseline from `2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt`.

## Source Files

- Pool generator: `tools/generate_darcy_lossdrop50_pool_20260607.py`
- Gradient/loss probe: `tools/probe_darcy_gradient_alignment_screen.py`
- Selector: `tools/select_darcy_lossdrop50_20260607.py`
- Runner: `tools/run_darcy_lossdrop50_pool_screen_20260607.sh`

## Output Files

- Pool root: `generalization_datasets_darcy_lossdrop50_pool_20260607`
- Pool manifest: `generalization_datasets_darcy_lossdrop50_pool_20260607/candidate_manifest.csv`
- Probe output: `forensics/darcy_lossdrop50_pool_gradient_screen_20260607`
- Probe summary: `forensics/darcy_lossdrop50_pool_gradient_screen_20260607/candidate_screen_summary.csv`
- Selected 50 root: `generalization_datasets_darcy_lossdrop50_selected_20260607`
- Selected manifest: `generalization_datasets_darcy_lossdrop50_selected_20260607/candidate_manifest.csv`
- Selected evidence table: `generalization_datasets_darcy_lossdrop50_selected_20260607/selection_summary.csv`
- Selected dataset files: `generalization_datasets_darcy_lossdrop50_selected_20260607/darcy/*.pt`
- GPU preflight: `forensics/darcy_lossdrop50_pool_gpu_preflight_20260607/gpu_preflight.json`
- Logs: `adversarial_training_runs/darcy_lossdrop50_pool_screen_20260607_logs`

## Run Evidence

Observed from `adversarial_training_runs/darcy_lossdrop50_pool_screen_20260607_logs/driver.log`:

- Pool generation started at `2026-06-07T17:36:55Z`.
- 50-step gradient/loss screen started at `2026-06-07T17:46:15Z`.
- Selection started at `2026-06-07T18:05:18Z`.
- Selection completed at `2026-06-07T18:05:19Z`.

Observed from `gradient_alignment_screen.log`:

- Probe reached `[darcy-screen] step 50/50`.
- Probe wrote `forensics/darcy_lossdrop50_pool_gradient_screen_20260607`.

Observed from selected output counts:

- `candidate_manifest.csv`: 51 lines, meaning 50 selected datasets plus header.
- `selection_summary.csv`: 51 lines, meaning 50 selected datasets plus header.
- `darcy/`: 50 selected `.pt` files.

## Loss-Drop Criterion

Criterion used for selection:

- `window == first50`
- `eval_name` is a pool dataset
- `eval_loss_delta < 0`

Observed pool result:

- Pool candidates screened: 72
- Pool candidates with `eval_loss_delta < 0`: 72
- Pool candidates with non-decreasing loss: 0

Observed selected-50 result from `selection_summary.csv`:

- Selected count: 50
- All selected loss deltas negative: true
- Most negative selected loss delta: `-2.3176673143628549e-07`
- Least negative selected loss delta: `-1.8023750669726724e-07`
- Mean selected loss delta: `-2.0133049086249833e-07`

## Representative Selected Rows

Top selected rows by loss decrease:

| rank | dataset | eval loss delta | cosine mean | negative cosine steps |
|---:|---|---:|---:|---:|
| 1 | `darcy_lossdrop_pool_soft_l4_h10_b8_09` | -2.3176673143628549e-07 | 0.7596156919002532 | 6/50 |
| 2 | `darcy_lossdrop_pool_soft_l4_h10_b8_16` | -2.3102153553130242e-07 | 0.7598290586471558 | 6/50 |
| 3 | `darcy_lossdrop_pool_soft_l4_h10_b8_13` | -2.2437280478015964e-07 | 0.7597865331172943 | 6/50 |
| 4 | `darcy_lossdrop_pool_soft_l4_h10_b8_03` | -2.234863529793074e-07 | 0.7594929540157318 | 6/50 |
| 5 | `darcy_lossdrop_pool_soft_l4_h10_b8_12` | -2.2296688702757211e-07 | 0.7597806656360626 | 6/50 |

Last selected rows, still loss-decreasing:

| rank | dataset | eval loss delta | cosine mean | negative cosine steps |
|---:|---|---:|---:|---:|
| 46 | `darcy_lossdrop_pool_soft_l4_h10_b10_05` | -1.8229690349850592e-07 | 0.7612640559673309 | 6/50 |
| 47 | `darcy_lossdrop_pool_soft_l4_h10_b12_07` | -1.81838411587402e-07 | 0.7613058257102966 | 6/50 |
| 48 | `darcy_lossdrop_pool_soft_l4_h10_b14_07` | -1.812791149783758e-07 | 0.7616702842712403 | 6/50 |
| 49 | `darcy_lossdrop_pool_soft_l4_h10_b14_03` | -1.8105203262545422e-07 | 0.7616308736801147 | 6/50 |
| 50 | `darcy_lossdrop_pool_soft_l4_h10_b12_10` | -1.8023750669726724e-07 | 0.7612370145320893 | 6/50 |

## Interpretation

Observed:

- The requested 50-dataset condition is satisfied for this screening setup: every selected dataset has observed negative `eval_loss_delta` after the 50-step adversarial-training probe.
- The selected datasets are concentrated in soft low-contrast families, especially `low=4`, `high=10`, `soft_beta=8/10/12/14`.
- The selected datasets have moderate cosine, roughly `0.7595` to `0.7723`, and `6/50` negative-cosine steps.

Inference:

- This suite prioritizes the user's corrected acceptance requirement: all 50 generalization datasets show loss decrease after short adversarial training.
- It does not solve the separate high-cosine requirement. Compared with high-cosine binary parameter shifts, these soft-family datasets are more reliably loss-decreasing but less direction-aligned with adversarial-sample gradients.

## Remaining Work

- Use `generalization_datasets_darcy_lossdrop50_selected_20260607` as the current loss-drop 50-dataset Darcy Flow generalization suite for follow-up screening/training.
- Restore or regenerate official Darcy train/test/checkpoint artifacts before treating this as an official Darcy adversarial-training benchmark.
- If the next target is both loss decrease and higher cosine, run another pool that locally perturbs this selected soft family while optimizing cosine as a secondary criterion.
