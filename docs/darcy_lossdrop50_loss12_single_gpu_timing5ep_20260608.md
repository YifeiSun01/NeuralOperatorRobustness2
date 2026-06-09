# Darcy Loss1/Loss2 Single-GPU 5-Epoch Timing - 2026-06-08

Observed from fresh sequential timing runs on an otherwise idle Tesla V100-SXM2-32GB. Loss1 completed before loss2 was started; they did not run concurrently.

## Source Runs

- Loss1 run: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_timing5ep_20260608`
- Loss2 run: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_timing5ep_20260608`
- Loss3 wall-time reference: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/summary.json`
- Timing estimate CSV: `forensics/darcy_lossdrop50_single_gpu_timing5ep_20260608_gpu_preflight/single_gpu_timing_estimates.csv`
- GPU records: `forensics/darcy_lossdrop50_single_gpu_timing5ep_20260608_gpu_preflight/`

## Key Results

| run | epochs | elapsed seconds | seconds / epoch | estimated epochs at loss3 wall (`4966.925741s`) |
|---|---:|---:|---:|---:|
| loss1 single-GPU 5ep | 5 | 39.317795 | 7.863559 | 631.638 |
| loss2 single-GPU 5ep | 5 | 39.615143 | 7.923029 | 626.897 |
| loss3 full reference | 500 | 4966.925741 | 9.933851 | 500.000 |

Observed from the run summaries, the direct 5-epoch total-time estimate gives approximately `632` loss1 epochs and `627` loss2 epochs for the same wall time as loss3.

The script's warmed smoke estimate is slightly faster because it amortizes the fixed baseline evaluation/startup overhead: loss1 per-epoch estimate is about `7.577s`, implying about `655` epochs at the loss3 wall time; loss2 per-epoch estimate is about `7.630s`, implying about `651` epochs.

## Inference

The earlier lower epoch counts for loss1/loss2 were caused by concurrent GPU sharing, not by loss1/loss2 being intrinsically slower than loss3. Fresh single-GPU timing confirms both loss1 and loss2 are faster per epoch than the standalone loss3 run.

## Remaining Work

Run official sequential single-GPU time-matched loss1 and loss2 jobs with `--max-wall-seconds 4966.925741452724`, then supersede the concurrency-contaminated loss1/loss2 rows.
