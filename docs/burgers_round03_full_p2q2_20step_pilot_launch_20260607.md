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

## Completion And Data Integrity Check - 2026-06-08

Status: the 20-step all-dataset final-only p2q2 attack output is complete locally.

Observed from `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/summary.json`:

- Attack steps: `20`.
- Batch size: `500`.
- Dataset count: `52`.
- Per-model sample count: `10200`.
- Models: `baseline`, `loss1_epoch8000`, `loss2_epoch2000`, `loss3_epoch1500`.
- GPU preflight recorded PyTorch `2.8.0+cu126`, CUDA `12.6`, Tesla V100-SXM2-32GB, compute capability `sm_70`, and architecture list containing `sm_70`.

Observed from `summary_by_model_dataset.csv`:

- Row count: `208`, exactly `4 models x 52 datasets`.
- Each model has `52` dataset rows.
- Columns include attack-before and attack-after losses: `initial_loss_mean`, `final_loss_mean`, plus `loss_ratio_final_over_initial`.
- Columns include model-solver RMS before/after attack: `initial_diff_rms_mean`, `final_diff_rms_mean`.
- Columns include final perturbation and FFT summaries: `final_delta_rms_mean`, `fft_low_1_32_frac`, `fft_mid_33_128_frac`, `fft_high_129_512_frac`, `fft_spectral_centroid`.

Mean dataset-level summary from `summary_by_model_dataset.csv`:

| model | initial loss mean | final loss mean | final/initial ratio | final delta RMS | low FFT frac | mid FFT frac | high FFT frac | spectral centroid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `0.000316` | `0.029279` | `93.944425` | `0.120000` | `0.995045` | `0.004945` | `0.000010` | `4.030162` |
| loss1_epoch8000 | `0.000003` | `0.003562` | `1233.978852` | `0.120000` | `0.911815` | `0.086072` | `0.002113` | `17.468758` |
| loss2_epoch2000 | `0.000009` | `0.004573` | `556.814909` | `0.120000` | `0.929444` | `0.070296` | `0.000260` | `13.889011` |
| loss3_epoch1500 | `0.000045` | `0.001678` | `38.345026` | `0.119703` | `0.757835` | `0.208054` | `0.034111` | `31.202970` |

Observed per-model output files:

- `baseline/losses_and_delta_rms_by_sample.npz`
- `baseline/final_delta_by_sample.npz`
- `baseline/fft_power_mean_by_dataset.npz`
- `baseline/fft_summary_by_dataset.csv`
- `loss1_epoch8000/losses_and_delta_rms_by_sample.npz`
- `loss1_epoch8000/final_delta_by_sample.npz`
- `loss1_epoch8000/fft_power_mean_by_dataset.npz`
- `loss1_epoch8000/fft_summary_by_dataset.csv`
- `loss2_epoch2000/losses_and_delta_rms_by_sample.npz`
- `loss2_epoch2000/final_delta_by_sample.npz`
- `loss2_epoch2000/fft_power_mean_by_dataset.npz`
- `loss2_epoch2000/fft_summary_by_dataset.csv`
- `loss3_epoch1500/losses_and_delta_rms_by_sample.npz`
- `loss3_epoch1500/final_delta_by_sample.npz`
- `loss3_epoch1500/fft_power_mean_by_dataset.npz`
- `loss3_epoch1500/fft_summary_by_dataset.csv`

Observed NPZ array shapes, same for all four models:

- `losses_and_delta_rms_by_sample.npz`: `initial_loss`, `initial_diff_rms`, `final_loss`, `final_diff_rms`, `final_delta_rms`, each shape `(10200,)`, dtype `float32`.
- `final_delta_by_sample.npz`: `final_delta` shape `(10200, 1024)`, dtype `float32`.
- `fft_power_mean_by_dataset.npz`: `fft_power_mean` shape `(52, 513)`, dtype `float32`; `dataset_ids` shape `(52,)`.

Inference:

- The requested four-model by 52-dataset 20-step attack dataset exists locally and contains attack-before loss, attack-after loss, final delta, per-sample final delta RMS, and per-dataset mean delta FFT spectra.
- The 20-step data is enough for the first high-frequency versus low-frequency statistical check. It already shows the expected ordering in dataset-mean spectral centroid: baseline lowest, loss1/loss2 higher, loss3 highest.
- This is not a 40-step, 60-step, or 100-step attack dataset. A search for roots matching `burgers_round03_full_p2q2_52datasets_4models_finalonly*` found only the `20step` root. The prepared resume launchers can extend the saved 20-step deltas to 40 or 60 steps if needed.

## Frequency Interpretation Check - 2026-06-08

Status: rechecked the all-dataset 20-step attack summary to decide whether the six-sample overlay impression that loss1/loss2/loss3 all produce very high-frequency deltas is representative.

Observed from `summary_by_model_dataset.csv` over all 52 datasets:

| model | low 1-32 mean | mid 33-128 mean | high 129-512 mean | spectral centroid mean | high 129-512 median |
|---|---:|---:|---:|---:|---:|
| baseline | `0.9950449` | `0.0049450` | `0.0000101` | `4.0301616` | `0.0000017` |
| loss1_epoch8000 | `0.9118145` | `0.0860724` | `0.0021130` | `17.468758` | `0.0012090` |
| loss2_epoch2000 | `0.9294445` | `0.0702958` | `0.0002598` | `13.889011` | `0.0002531` |
| loss3_epoch1500 | `0.7578354` | `0.2080541` | `0.0341106` | `31.202970` | `0.0292251` |

Observed paired against baseline, all 52 datasets:

- Spectral centroid is higher than baseline for loss1, loss2, and loss3 on `52/52` datasets.
- High-band fraction is also higher than baseline for all three on `52/52` datasets, but the absolute scale differs sharply: loss3 mean high-band fraction is about `0.0341`, loss1 about `0.00211`, loss2 about `0.000260`, and baseline about `0.000010`.
- Dataset-mean spectral centroid ordering is `baseline < loss2 < loss1 < loss3`.
- Dataset-mean high-band fraction ordering is `baseline < loss2 < loss1 << loss3`.

Inference:

- The user's heatmap impression is mostly consistent with the full-data statistics if “high frequency” means the high FFT band `129-512`: loss3 is the only model with a strong high-band shift.
- The six-sample overlay impression that loss1/loss2/loss3 all look high-frequency is too strong as a global statement. Loss1/loss2 are not baseline-identical, but most of their shift is into the mid band `33-128`, not the highest band.
- Loss2 is especially close to baseline in absolute high-band energy, though its spectral centroid is still higher because low-band mass moves into the mid band.
- Therefore the clean interpretation is: loss3 becomes strongly high-frequency under the final attack; loss1/loss2 become less purely low-frequency than baseline, but they do not show the same strong high-frequency phenomenon as loss3.

## Robust Loss Reduction Check - 2026-06-08

Status: checked whether the same 20-step p2q2 attack gives lower clean loss and lower final attacked loss after self-training on all 52 datasets.

Observed paired comparison against baseline from `summary_by_model_dataset.csv`:

| model | clean initial loss mean | clean loss ratio vs baseline | datasets with lower clean loss | attacked final loss mean | final loss ratio vs baseline | datasets with lower final loss | final diff RMS ratio vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | `0.0003164451` | `1.0000` | - | `0.0292785310` | `1.0000` | - | `1.0000` |
| loss1_epoch8000 | `0.0000033187` | `0.0101095` | `52/52` | `0.0035624034` | `0.1208829` | `52/52` | `0.3396248` |
| loss2_epoch2000 | `0.0000085542` | `0.0265781` | `52/52` | `0.0045731862` | `0.1555019` | `52/52` | `0.3828037` |
| loss3_epoch1500 | `0.0000446399` | `0.1465824` | `52/52` | `0.0016781858` | `0.0609096` | `52/52` | `0.2447011` |

Observed on the 50 generalization datasets only:

| model | generalization clean loss ratio vs baseline | lower clean loss count | generalization final attack loss ratio vs baseline | lower final loss count | spectral centroid mean | high FFT frac mean |
|---|---:|---:|---:|---:|---:|---:|
| loss1_epoch8000 | `0.0101957` | `50/50` | `0.1181292` | `50/50` | `17.357529` | `0.0021869` |
| loss2_epoch2000 | `0.0265932` | `50/50` | `0.1528716` | `50/50` | `13.589438` | `0.0002619` |
| loss3_epoch1500 | `0.1433071` | `50/50` | `0.0584845` | `50/50` | `30.669102` | `0.0335560` |

Inference:

- Under this 20-step p2q2 attack protocol, self-training clearly lowers the attacked final loss for all three trained models on every dataset row. The final attacked loss reductions versus baseline are about `87.9%` for loss1, `84.4%` for loss2, and `93.9%` for loss3 by paired mean ratio.
- Clean/pre-attack loss also drops on every dataset row. Loss1 has the lowest clean loss, loss2 second, loss3 third; but loss3 has the lowest final attack loss and lowest final model-solver RMS after attack.
- The large `loss_ratio_final_over_initial` for loss1/loss2 should not be read as worse robustness by itself. Their clean initial losses are extremely small, so the ratio from their own clean loss to their own attacked loss is large; the absolute attacked final loss is still far below baseline.
- Combining frequency and loss evidence: loss3 is both the most robust by final attack loss and the most high-frequency in final delta FFT. Loss1/loss2 are also much more robust than baseline, but their delta frequency shift is mostly mid-band rather than strongly high-band.

## Clean Loss Ratio Interpretation Correction - 2026-06-08

Status: checked why `clean loss ratio vs baseline` values such as `0.0101` looked surprisingly small.

Observed from `tools/run_burgers_round03_full_p2q2_finalonly_attack.py`:

- The attack runner computes `final_loss = (pred - solver).pow(2).mean(dim=(1, 2))`.
- The runner separately records `final_diff_rms = rms_l2_norm(pred - solver)`.
- Therefore `initial_loss_mean` and `final_loss_mean` are MSE-style squared losses, while `initial_diff_rms_mean` and `final_diff_rms_mean` are output-difference RMS amplitudes.

Observed from the generated per-dataset ratio table:

- Full CSV: `forensics/burgers_round03_full_p2q2_clean_loss_per_dataset_20260608/clean_and_attack_loss_ratios_by_dataset.csv`.
- Full Markdown table: `forensics/burgers_round03_full_p2q2_clean_loss_per_dataset_20260608/clean_and_attack_loss_ratios_by_dataset.md`.
- Dedicated docs copy: `docs/burgers_round03_clean_loss_ratio_interpretation_20260608.md`.

Mean ratio summary:

| model | clean MSE ratio mean | clean RMS ratio mean | final attack MSE ratio mean | final attack RMS ratio mean |
|---|---:|---:|---:|---:|
| loss1_epoch8000 | `0.0101095` | `0.100027` | `0.120883` | `0.339625` |
| loss2_epoch2000 | `0.0265781` | `0.162403` | `0.155502` | `0.382804` |
| loss3_epoch1500 | `0.146582` | `0.395195` | `0.0609096` | `0.244701` |

Inference:

- Saying “loss1 clean loss is 1% of baseline” is correct only for MSE-style squared loss.
- If the intuitive question is output-amplitude deviation, the relevant clean quantity is RMS ratio: loss1 is about `10%` of baseline, loss2 about `16%`, and loss3 about `40%`.
- The earlier ratio table should therefore be read as squared-loss ratios for `clean loss` and `final attack loss`, with `diff RMS ratio` giving the more visually intuitive amplitude ratio.
