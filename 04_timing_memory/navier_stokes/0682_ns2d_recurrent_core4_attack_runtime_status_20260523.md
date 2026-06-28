# NS2D Recurrent Core4 Attack Runtime Status - 2026-05-23

Status: local process/log inspection only. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started by this note.

## 2026-05-23 02:26 UTC Check

Observed from `ps`, `/tmp/ns2d_pair_outer_attack_nohup.out`, and the pair-outer output tree:

- Parent command started at `2026-05-22 05:53:12 UTC`.
- Active process at inspection: `eps8_alpha2p5 / loss3 / d1_5_w6_9_target_w`, launched at `2026-05-23 02:04:35 UTC`.
- At `2026-05-23 02:26:32 UTC`, active process elapsed time was about `22 min`.
- GPU status at inspection: `NVIDIA A100-SXM4-80GB`, about `47235 MiB / 81920 MiB` used, `99%` utilization.

Completed `eps8_alpha2p5` blocks from the nohup log:

| block | start UTC | finish UTC | duration |
|---|---|---|---:|
| `loss1/all_w` | 2026-05-22 18:02:00 | 2026-05-22 19:16:17 | 1h14m17s |
| `loss2/all_a_target_w` | 2026-05-22 19:16:17 | 2026-05-22 19:49:00 | 32m43s |
| `loss3/all_w` | 2026-05-22 19:49:00 | 2026-05-22 21:53:41 | 2h04m41s |
| `loss3/all_d_target_w` | 2026-05-22 21:53:41 | 2026-05-22 23:56:40 | 2h02m59s |
| `loss3/w1_5_d6_9_target_w` | 2026-05-22 23:56:40 | 2026-05-23 02:04:35 | 2h07m55s |

Inference from previous completed `eps8_alpha2p5` loss3 mode runtimes:

- Current `loss3/d1_5_w6_9_target_w` should take about `2h05m` total, so from the 02:26 UTC check it likely had about `1h40m` to `1h50m` remaining.
- One more `eps8_alpha2p5` block remains after it: `loss3/a1_5_d6_9_target_w`, expected around another `2h05m`.
- Estimated `eps8_alpha2p5` completion: roughly `2026-05-23 06:10-06:25 UTC`, assuming no OOM or abnormal slowdown.
- The pair-outer script will then continue to `eps16_alpha5` and `eps32_alpha15` unless stopped.

## 2026-05-23 04:02 UTC Check

Status: local process/log/file inspection only. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started by this note.

Observed from `date -u`, `ps`, `/tmp/ns2d_pair_outer_attack_nohup.out`, the `eps8_alpha2p5` output tree, and `nvidia-smi`:

- Current UTC time: `2026-05-23 04:02:39`.
- Parent pair-outer script elapsed time: about `22h09m`.
- Active attack process: `eps8_alpha2p5 / loss3 / d1_5_w6_9_target_w`, started at `2026-05-23 02:04:35 UTC`, elapsed about `1h58m`.
- In this active block, completed method outputs are visible for `raw_add`, `raw_replace`, and `steepest_add`.
- The current method is inferred to be `steepest_replace`, because no `steepest_replace` output files are visible yet and the Python process is still running.
- Latest completed method file for this block: `steepest_add/summary.json` at about `2026-05-23 03:37:00 UTC`.
- GPU status: `NVIDIA A100-SXM4-80GB`, about `47197 MiB / 81920 MiB` used, `99%` utilization.

Inference:

- Previous `eps8_alpha2p5` loss3 methods generally took about `30-36 min` per method. Since the current `steepest_replace` method had been running about `25-26 min` at the check, the active `d1_5_w6_9_target_w` block likely had about `5-15 min` remaining.
- After that, one `eps8_alpha2p5` block remains: `loss3/a1_5_d6_9_target_w`, expected around `2h05m` based on neighboring loss3 mode runtimes.
- Estimated `eps8_alpha2p5` completion from this check: roughly `2026-05-23 06:10-06:30 UTC`.
- If the user means the whole pair-outer script, it will continue after `eps8_alpha2p5` to `eps16_alpha5` and `eps32_alpha15`; based on the first two pair runtimes, those two additional pairs could add roughly another `~24 h` total.

## 2026-05-23 04:42 UTC Check

Status: local process/log/file inspection only. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started by this note.

Observed from `date -u`, `ps`, `/tmp/ns2d_pair_outer_attack_nohup.out`, the `eps8_alpha2p5` output tree, and `nvidia-smi`:

- Current UTC time: `2026-05-23 04:42:47`.
- `eps8_alpha2p5 / loss3 / d1_5_w6_9_target_w` finished at `2026-05-23 04:07:42 UTC`.
- Active process at the check: `eps8_alpha2p5 / loss3 / a1_5_d6_9_target_w`, started at `2026-05-23 04:07:42 UTC`, elapsed about `35 min`.
- In the active `a1_5_d6_9_target_w` block, `raw_add` completed with `summary.json` at about `2026-05-23 04:39:11 UTC`.
- The active method is inferred to be `raw_replace`, with `steepest_add` and `steepest_replace` still remaining afterward.
- GPU status: `NVIDIA A100-SXM4-80GB`, about `47337 MiB / 81920 MiB` used, `99%` utilization.

Inference:

- The active final `eps8_alpha2p5` block has three methods remaining. Based on neighboring method durations of roughly 30-36 minutes each, the `eps8_alpha2p5` pair likely has about `1h25m-1h45m` remaining.
- Estimated `eps8_alpha2p5` completion from this check: roughly `2026-05-23 06:10-06:30 UTC`.
- If the full pair-outer script is allowed to continue, it will next run `eps16_alpha5` and `eps32_alpha15`; those two pairs likely add roughly another `~24 h` total.

## 2026-05-23 04:59 UTC Check

Status: local process/log/file inspection only. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started by this note.

Observed from `date -u`, `ps`, `/tmp/ns2d_pair_outer_attack_nohup.out`, the `eps8_alpha2p5` output tree, and `nvidia-smi`:

- Current UTC time: `2026-05-23 04:59:14`.
- Active process at the check: `eps8_alpha2p5 / loss3 / a1_5_d6_9_target_w`, started at `2026-05-23 04:07:42 UTC`, elapsed about `51m32s`.
- In the active final `eps8_alpha2p5` block, `raw_add` completed with `summary.json` at about `2026-05-23 04:39:11 UTC`.
- No completed files for `raw_replace` were visible yet, so `raw_replace` is inferred to be active.
- Remaining methods after the active `raw_replace`: `steepest_add` and `steepest_replace`.
- GPU status: `NVIDIA A100-SXM4-80GB`, about `47337 MiB / 81920 MiB` used, `99%` utilization.

Inference:

- The active `raw_replace` method had been running about `20 min` since `raw_add` completed. Neighboring methods usually take about `30-36 min`, so `raw_replace` likely had about `10-20 min` remaining.
- The final `eps8_alpha2p5` block likely has about `1h15m-1h35m` remaining in total.
- Estimated `eps8_alpha2p5` completion from this check: roughly `2026-05-23 06:15-06:35 UTC`.
- If the full pair-outer script continues, it will next run `eps16_alpha5` and `eps32_alpha15`, likely adding roughly another `~24 h` total.

## 2026-05-23 10:21:58 UTC - Paused active NS2D attack for Git/R2 sync

Status: paused the active experiment process with `SIGSTOP`; no process was killed.

Observed active experiment before pause:

- Parent launcher: `/tmp/run_ns2d_pair_outer_attack.sh`
- Active block: `eps16_alpha5 / loss3 / all_d_target_w`
- Active Python PID: `408673`
- Wrapper PID: `408671`
- After pause, PID `408673` had process state `Tl`.
- GPU utilization dropped to `0%` after the pause; GPU memory remained allocated by the stopped process, which is expected for `SIGSTOP`.

Resume command if needed:

```bash
kill -CONT 408673
```

Observed evidence:

- `ps -o pid,ppid,stat,etimes,cmd -p 408673`
- `nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`

Remaining work:

- Push source/docs/scripts to GitHub.
- Copy large generated experiment artifacts to R2.

## 2026-05-23 10:30 UTC - GitHub/R2 Sync After Pause

Status: sync record. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started by this note.

Observed evidence:

- Paused Python PID `408673` remained in process state `Tl`.
- GPU utilization remained `0%`; GPU memory remained allocated by the stopped CUDA process.
- Source/docs/scripts were committed and pushed to GitHub branch `vast-ai` in commit `c2cc0ea`.
- Large generated experiment artifacts were copied to R2 with non-destructive `rclone copy`.
- R2 verification showed the eps8 image/GIF package, NS2D core4 attack result tree, trained model directory, dictionary directory, `docs/`, `tools/`, and `EXPERIMENT_LEDGER.md` present under the configured prefix.
- Temporary R2 config was removed after the upload/verification step.

Detailed record:

- `docs/ns2d_git_r2_sync_20260523.md`

Remaining work:

- Resume the paused attack only when wanted with `kill -CONT 408673`.

## 2026-05-23 10:34 UTC - Final Experiment Process Sweep

Status: final process sweep after GitHub/R2 sync. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started by this note.

Observed evidence:

- `pgrep -af 'attack|train_fno|run_ns2d|run_fno|python'` showed normal system services such as Jupyter, TensorBoard, supervisor, portal, and log helpers.
- The only neural-operator experiment chain visible was `/tmp/run_ns2d_pair_outer_attack.sh` with child Python PID `408673`.
- PID `408673` was already paused from the earlier `SIGSTOP` operation.

Inference:

- No additional active training, solver-generation, or attack experiment process was found that needed a separate pause action.
