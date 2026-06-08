# Burgers Round03 Full P2Q2 20-Step Pilot Launch - 2026-06-07

## Status

Launched the all-dataset final-only p=2,q=2 attack pilot at `2026-06-07T21:55:38Z` in tmux session `burgers_p2q2_20step_20260607`.

This run was started to resolve the visual ambiguity between the small six-sample overlay plots and the checkpoint-style delta FFT plots. The six-sample overlay is not enough evidence to conclude that loss1/loss2/loss3 all produce high-frequency final perturbations; this pilot computes the final delta FFT over all requested datasets under one consistent attack protocol.

## Scope

- Models: baseline, loss1 epoch8000, loss2 epoch2000, loss3 epoch1500.
- Data: train first 50 samples, full test set of 150 samples, and all 50 generalization datasets with 200 samples each.
- Total per model: 10200 samples.
- Total across four models: 40800 model-sample attacks.
- Attack settings: p=2, q=2 RMS-L2; `steps=20`, `epsilon_rms=0.12`, `alpha_rms=0.012`, `batch_size=500`.

## Source Code

- Main runner: `tools/run_burgers_round03_full_p2q2_finalonly_attack.py`.
- Launcher: `tools/run_burgers_round03_full_p2q2_20step_pilot_20260607.sh`.
- Existing p2q2 attack reference: `tools/plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py`.

## Output Paths

- Result root: `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`.
- Progress: `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/progress.jsonl`.
- Logs: `adversarial_training_runs/burgers_round03_full_p2q2_20step_pilot_20260607_logs/run_20260607_215538_UTC.log`.

Expected final outputs per model:

- `losses_and_delta_rms_by_sample.npz`
- `final_delta_by_sample.npz`
- `fft_power_mean_by_dataset.npz`
- `fft_summary_by_dataset.csv`

Expected global outputs:

- `config.json`
- `manifest.json`
- `summary_by_model_dataset.csv`
- `summary.json`

## Debug Run Evidence

Before launch, a debug run completed successfully:

- Debug output: `forensics/burgers_round03_full_p2q2_debug_10samples_1step_20260607`.
- Scope: baseline only, 10 samples, 1 attack step.
- Observed initial loss mean: `9.085451893042773e-05`.
- Observed final loss mean: `0.00017895766359288245`.
- Observed final delta RMS mean: `0.012000000104308128`, matching one alpha step.
- Output files included manifest, config, summary CSV/JSON, per-sample losses/delta RMS, final delta, and FFT summaries.

## GPU Evidence

At launch, `nvidia-smi` reported Tesla V100-SXM2-32GB with 0 MiB used and no running GPU processes. The runner also performs PyTorch CUDA preflight and records it in `config.json`.

During monitoring, GPU memory was about `30084 MiB / 32768 MiB`, consistent with the prior batch-500 benchmark.

## First Progress Check - 2026-06-07T21:57:55Z

Observed tmux session: `burgers_p2q2_20step_20260607` still running.

Observed first two completed batches from `progress.jsonl`:

| Model | Batch start:end | Samples | Seconds | Initial loss mean | Final loss mean | Final delta RMS mean | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0:500 | 500 | 51.4147 | 0.0002477061 | 0.0126381973 | 0.1199999973 | 28.5961 |
| baseline | 500:1000 | 500 | 49.4566 | 0.0004226719 | 0.0288239066 | 0.1199999973 | 28.5961 |

Inference from first progress check: the run is healthy, reaches the RMS perturbation boundary by step 20, and matches the expected batch-500 runtime and memory profile.

## Planned Analysis After Completion

After the run completes, analyze `summary_by_model_dataset.csv` and per-model `fft_summary_by_dataset.csv` to answer:

- Whether final attack delta high-frequency fraction is actually higher for loss1/loss2/loss3 than baseline across all datasets.
- Whether loss3 shows a stronger high-frequency shift than loss1/loss2, matching the checkpoint-style FFT observation.
- Whether the six-sample overlay impression was representative or a visual/sampling artifact.
- How final-loss increase correlates with FFT high-band power fraction and spectral centroid.


## Status Check - 2026-06-07T22:19:52Z

Observed GPU state: `nvidia-smi` reported `30084 MiB / 32768 MiB` used, GPU utilization about `43%`, temperature `51C`, and no OOM/ECC error. The NVIDIA process table still did not list the Python PID, but `pgrep` showed the active Burgers runner PID `865719`.

Observed active sessions: `tmux list-sessions` showed `burgers_p2q2_20step_20260607` still running. No Darcy Flow training tmux session was active.

Observed Burgers progress: `progress.jsonl` had `27` completed batches out of the expected `84` model batches. Baseline completed all `10200` samples; loss1 epoch8000 had completed through sample range `2500:3000`. `summary.json` was not yet written, so the Burgers pilot was still running.

Observed latest Burgers batch: loss1 epoch8000 `2500:3000`, `54.2218s`, final loss mean `0.0028975329`, final delta RMS mean `0.1199999973`, peak allocated `28.5961 GiB`.

Inference: Burgers p2q2 pilot has not blown GPU memory and is progressing normally. At the current batch rate, expected completion is roughly `55-65` minutes after this check, plus small final FFT/write overhead.

Observed Darcy Flow status: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/summary.json` exists and reports finished UTC `20260607_200325_UTC`, `epochs=500`, final checkpoint `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/checkpoints/darcy_epoch500_step002000.pt`, total wall `4975.1246s` (`82.9187 min`). Darcy peak allocated CUDA memory was `9335.814 MB`, current allocated at finish `3626.451 MB`, reserved `11288 MB`.

Inference: Darcy Flow completed normally before this Burgers status check and did not remain active while the current Burgers p2q2 pilot was using ~30 GiB.

## Status Check And Resume Preparation - 2026-06-07T22:28:42Z

Observed GPU/process state: `nvidia-smi` reported Tesla V100-SXM2-32GB using `30084 MiB / 32768 MiB`, utilization about `34%`, temperature `53C`, and no OOM/ECC error. The NVIDIA process table did not list the Python process, but `ps -eo pid,ppid,stat,etime,cmd` showed the active runner as PID `865719`:

`adv_robust/bin/python tools/run_burgers_round03_full_p2q2_finalonly_attack.py --steps 20 --batch-size 500 --train-count 50 --run-name burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`

Observed tmux state: session `burgers_p2q2_20step_20260607` is still active.

Observed Burgers progress: `progress.jsonl` has `38` completed batch rows out of the expected `84`. Baseline completed all `10200` samples and wrote:

- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/baseline/losses_and_delta_rms_by_sample.npz`
- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/baseline/final_delta_by_sample.npz`

Observed latest batch: loss1 epoch8000 completed sample range `8000:8500`, batch time `48.2810s`, initial loss mean `2.3453937956219306e-06`, final loss mean `0.0023537492379546165`, final delta RMS mean `0.11999999731779099`, peak allocated `28.5961 GiB`.

Inference: the 20-step pilot is still progressing normally and has not exhausted GPU memory. Per-model delta files are written when each model finishes; at this check only the completed baseline model had written its final delta. Loss1, loss2, and loss3 delta files are expected only after their respective model sweeps finish.

Resume support was added after this 20-step pilot was already launched. The current in-memory 20-step job should be allowed to finish normally. After it writes `summary.json` and all four model directories contain both `final_delta_by_sample.npz` and `losses_and_delta_rms_by_sample.npz`, the prepared launchers can continue the attack from the saved 20-step deltas:

- `tools/run_burgers_round03_full_p2q2_resume_20to40_20260607.sh`
- `tools/run_burgers_round03_full_p2q2_resume_20to60_20260607.sh`

Observed implementation detail: the resume launchers refuse to run unless the 20-step source root is complete. They continue from the saved 20-step `final_delta_by_sample.npz` rather than restarting from zero perturbation, preserve the original step0 loss, record the 20-step resume-start loss/delta RMS, and recompute final loss, final delta, and FFT summaries at the requested target total step count.

## Follow-Up Status - 2026-06-07T22:33:53Z

Observed GPU state: `nvidia-smi` reported `30084 MiB / 32768 MiB`, utilization about `42%`, temperature `50C`, and no OOM/ECC error.

Observed progress: `progress.jsonl` had `43` completed batch rows. Loss1 epoch8000 completed the final range `10000:10200`; loss2 epoch2000 had started and completed `0:500`.

Observed saved outputs now include both baseline and loss1 final deltas:

- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/baseline/final_delta_by_sample.npz`
- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/loss1_epoch8000/final_delta_by_sample.npz`

Inference: per-model final delta writing is working as intended. The later 20-to-40 or 20-to-60 resume run can use the saved deltas once loss2 and loss3 also finish and the global `summary.json` is written.
