# Burgers Round03 Loss1/Loss2/Loss3 Continuation Status - 2026-06-05

Status: running, not complete.

This note records the launch/monitoring status for the round03 Burgers
loss1/loss2/loss3 continuation training. It is not a final result report.

Source files and prerequisites:
- Continuation driver: `tools/run_burgers_loss3_selective_round03_loss123_continuation_20260605.sh`
- Dataset: `generalization_datasets_burgers_loss3_selective_search/round_03`
- Required completed SVD gate: `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_jacobian_svd_summary.csv`
- loss1 base checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`
- loss2 base checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt`
- loss3 base checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt`

Output paths:
- Logs: `adversarial_training_runs/burgers_loss3_selective_round03_loss123_continuation_20260605_logs/`
- GPU preflight: `forensics/burgers_loss3_selective_round03_loss123_continuation_gpu_preflight_20260605/`
- Active loss1 run: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605/`
- loss2/loss3 continuation run directories were not present locally as of `2026-06-05T21:21:27Z`; the script is expected to reach them after loss1 completes.

Key settings observed from the active training process:
- Python: `/workspace/NeuralOperatorRobustness2/adv_robust/bin/python`
- Device: `cuda`
- Seed: `20260601`
- Objective: loss1 for the active run
- Continuation offsets: `--resume-epoch-offset 1000`, `--resume-global-step-offset 3000`
- Continuation length for active run: `--epochs 2000`
- Burgers attack: `fast_replace_l2`, 5 steps, epsilon fraction `0.06`
- Batch settings: `--burgers-batch-size 480`, `--burgers-optimizer-batch-size 32`
- Evaluation: full round03 generated generalization evaluation, `--max-generalization-eval 50`

GPU path observed:
- From `forensics/burgers_loss3_selective_round03_loss123_continuation_gpu_preflight_20260605/gpu_preflight.json`: PyTorch `2.8.0+cu126`, Torch CUDA `12.6`, `torch_cuda_available=true`, device `Tesla V100-SXM2-32GB`, capability `[7, 0]`, arch list includes `sm_70`, JAX `0.10.0`, JAX backend `gpu`, JAX device `cuda:0`.
- From `nvidia-smi` snapshots during monitoring: NVIDIA driver `580.76.05`, reported CUDA `13.0`, Tesla V100-SXM2-32GB in P0, about `4896 MiB / 32768 MiB` in use, GPU util between about `48%` and `63%`, final snapshot at `2026-06-05 21:20:08 UTC` showed `59%` util and `56C`.
- `nvidia-smi` process table did not list the Python PID despite nonzero GPU memory/utilization. The active process was separately observed with `pgrep -af adv_robust` as PID `526950`, running the expected `adversarial_training.py --device cuda` command.

Observed evidence:
- `tmux list-sessions` showed `round03_loss123_continuation` already running, so no duplicate session was started.
- `driver.log` observed start line: loss1 continuation began at `2026-06-05T20:23:26Z` from the loss1 epoch1000 checkpoint.
- Per-run log observed dataset preflight success: each task has train `1`, test `1`, generalization `50`.
- During the approximately 10 minute monitoring window, `train_steps.csv` advanced from about epoch `1550`/step `4650` to epoch `1671`/step `5012`.
- The latest follow-up sample before writing this note, from `eval_split_summary.csv`, was epoch `1683`/step `5049` for the generated generalization split: RMSE mean `0.035635485363701404`, relative L2 mean `0.06389568580656536`, accuracy-score mean `94.0312386197881`. These are interim during-training metrics, not final continuation results.
- `checkpoints.csv` showed periodic checkpoints through epoch `1600`, including `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605/burgers/checkpoints/burgers_epoch1600_step004800.pt`.
- `memory.csv` final sampled row near the observation end showed epoch `1671`/step `5012`, CUDA allocated `32.94091796875` MiB, CUDA reserved `4494.0` MiB, and CUDA peak allocated `3894.95654296875` MiB.
- No `summary.json` was present for loss1, loss2, or loss3 continuation as of this note.

Inference from the observed evidence:
- The loss1 continuation is actively progressing on the GPU path and appears healthy during this monitoring window.
- The full loss1/loss2/loss3 continuation experiment is not complete. loss2 and loss3 should start only after the scripted loss1 phase completes.
- The interim loss1 metrics should not be presented as final comparative continuation results.

Remaining work:
- Leave the `round03_loss123_continuation` tmux session running.
- Monitor `adversarial_training_runs/burgers_loss3_selective_round03_loss123_continuation_20260605_logs/driver.log` and the active per-run logs until all three summaries are produced.
- After completion, generate train/test/generalization RMSE, relative L2, and wall-clock curves, including stitched curves from the original round03 long training into these continuation runs.
- Update this note and `EXPERIMENT_LEDGER.md` with final observed metrics and exact CSV/JSON sources.
- Do not rerun same-wall/time-matched SVD; that diagnostic remains cancelled.
- Keep generated CSV/NPZ/model/PNG artifacts out of Git unless explicitly requested.

## ETA Update - 2026-06-05T21:48:52Z

Status: still running, active phase remains loss1 continuation.

Observed evidence:
- `tmux list-sessions` still showed `round03_loss123_continuation`.
- `pgrep -af adv_robust` still showed PID `526950` running `/workspace/NeuralOperatorRobustness2/adv_robust/bin/python ... --device cuda ... --run-name burgers_loss3_selective_round03_loss1_continue1000to3000_20260605 ... --burgers-attack-loss-objective loss1`.
- `checkpoints.csv` showed loss1 checkpoints through epoch `2000` at wall elapsed `5056.680034907535` seconds.
- Parsed `train_steps.csv` showed latest loss1 row epoch `2021`, global step `6062`, progress fraction `0.6735555555555556`.
- Parsed `eval_split_summary.csv` showed latest generated-generalization row epoch `2020`, global step `6060`, RMSE mean `0.035468239067062715`, relative L2 mean `0.06359478850767653`, accuracy-score mean `94.05875141558727`.
- Parsed run directories showed loss1 `summary.json` absent, and loss2/loss3 continuation directories absent.

Inference from observed timing:
- Recent checkpoint timing gives about `4.956` seconds/epoch from epoch1900 to epoch2000 and about `5.062` seconds/epoch from epoch1100 to epoch2000.
- Loss1 has about `979` epochs remaining from epoch2021 to epoch3000, or about `1.35` to `1.40` hours at the observed rate.
- Loss2 and loss3 each need 500 epochs after loss1, roughly `0.70` hours each if the per-epoch runtime is similar.
- Estimated remaining training time is about `2.75` to `3.1` hours, plus small handoff/final-summary overhead. Expected completion is around `2026-06-06T00:35Z` to `2026-06-06T01:00Z` if throughput stays stable and no OOM/error occurs.

## ETA Correction - 2026-06-05T23:08:16Z

Status: previous ETA based on loss1 per-epoch timing for all objectives is superseded.

Observed evidence:
- Current continuation was still in loss1 at `2026-06-05T23:08:16Z`; `pgrep -af adv_robust` showed the active command still used `--run-name burgers_loss3_selective_round03_loss1_continue1000to3000_20260605` and `--burgers-attack-loss-objective loss1`.
- Current loss1 continuation parsed from `train_steps.csv`: epoch `2950`, global step `8850`, progress fraction `0.9833333333333333`.
- Current loss1 continuation `checkpoints.csv` showed epoch `2900` at wall elapsed `9668.578715356998` seconds, with recent periodic timing about `4.997` seconds/epoch from epoch2800 to epoch2900.
- Base loss1 1000 epoch run observed from `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/summary.json`: total wall seconds `5024.663676050492` (`83.74439460084153` minutes), about `5.02` seconds/epoch.
- Base loss2 500 epoch run observed from `adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/summary.json`: total wall seconds `10900.811602581292` (`181.68019337635488` minutes), about `21.80` seconds/epoch. `checkpoints.csv` periodic timing was about `21.915` seconds/epoch from epoch100 to epoch400.
- Base loss3 500 epoch run observed from `adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/summary.json`: total wall seconds `23889.608705461957` (`398.16014509103263` minutes), about `47.78` seconds/epoch. `checkpoints.csv` periodic timing was about `47.505` seconds/epoch from epoch100 to epoch400.
- loss2 and loss3 continuation directories were still absent at this check, so those phases had not started.

Inference from observed timing:
- The earlier ETA of roughly 42 minutes each for loss2 and loss3 was wrong because it incorrectly extrapolated loss1 timing to slower loss objectives.
- More defensible remaining-time estimate at `2026-06-05T23:08:16Z`: loss1 has roughly 50 epochs left, about 4 to 5 minutes; loss2 needs about 3.0 hours; loss3 needs about 6.6 hours; total remaining training time is roughly 9.7 to 10.2 hours plus handoff/final-summary overhead.
- Expected training completion is roughly `2026-06-06T08:50Z` to `2026-06-06T09:20Z` if throughput matches the existing round03 long runs and no error/OOM occurs. Stitched plotting and final reporting would be additional work after training completes.

## Current Result Check - 2026-06-06T00:41:44Z

Status: continuation is still running. loss1 continuation is complete; loss2 continuation is active; loss3 continuation has not started.

Observed evidence:
- `tmux list-sessions` showed `round03_loss123_continuation` still running.
- `pgrep -af adv_robust` showed active PID `579129` running `/workspace/NeuralOperatorRobustness2/adv_robust/bin/python ... --device cuda ... --run-name burgers_loss3_selective_round03_loss2_continue500to1000_20260605 ... --burgers-attack-loss-objective loss2`.
- `driver.log` showed loss1 continuation started at `2026-06-05T20:23:26Z`, finished at `2026-06-05T23:13:01Z`, and loss2 continuation started immediately at `2026-06-05T23:13:01Z`.
- `nvidia-smi` at `2026-06-06 00:41:45 UTC` showed Tesla V100-SXM2-32GB in P0, `4896 MiB / 32768 MiB` used, `58%` GPU utilization, driver `580.76.05`, reported CUDA `13.0`.

Observed metrics from exact sources:
- From `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/eval_split_summary.csv`: loss1 base epoch1000 generated-generalization RMSE mean `0.03655678255777361`, relative L2 mean `0.0655581618059065`; best generated RMSE `0.0351621095675644` at epoch965.
- From `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605/summary.json` and `burgers/eval_split_summary.csv`: loss1 continuation completed in `10171.913992851973` seconds. Final epoch3000 train/test/generated RMSE means were `0.0010929236794394932` / `0.0012285454291087751` / `0.0343238570556901`; generated relative L2 mean `0.06153625953140944`. Best generated RMSE was `0.03294495584905165` at epoch2673.
- From `adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/burgers/eval_split_summary.csv`: loss2 base epoch500 generated RMSE mean `0.03966611168744192`, relative L2 mean `0.07114700937434287`; best generated RMSE `0.03830022787028556` at epoch486.
- From `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue500to1000_20260605/burgers/eval_split_summary.csv`: active loss2 continuation latest sampled epoch742 train/test/generated RMSE means were `0.0017918159641902769` / `0.0019882556388082337` / `0.03833159909118862`; generated relative L2 mean `0.06874984641837868`. Best generated RMSE so far was `0.03656475286341791` at epoch711.
- From `adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/eval_split_summary.csv`: loss3 base epoch500 generated RMSE mean `0.023660631499673727`, relative L2 mean `0.04244147745939993`; best generated RMSE `0.023076826214804887` at epoch492.
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue500to1000_20260605/` was absent at this check, so no loss3 continuation metrics are currently evidenced.

Inference from observed evidence:
- loss1 continuation improved the final generated RMSE by about `6.11%` relative to its epoch1000 final value, and improved train/test RMSE substantially. The best loss1 generated checkpoint so far is epoch2673, not epoch3000.
- loss2 continuation is improving relative to its epoch500 baseline, but its latest epoch742 generated RMSE is worse than its current best epoch711, so it is oscillating rather than monotonically improving.
- Even after loss1 continuation and the partial loss2 continuation, the existing loss3 epoch500 model remains much better on round03 generated generalization: loss3 base final generated RMSE `0.023660631499673727` versus loss1 epoch3000 final `0.0343238570556901` and loss2 epoch742 latest `0.03833159909118862`.
- The clean train/test story remains different: loss1/loss2 have much lower train/test RMSE than loss3, while loss3 has the generated-OOD advantage.
- This is still an intermediate result. The decisive continuation comparison requires loss2 to finish and loss3 continuation to run.

ETA inference:
- loss2 continuation checkpoint timing was about `21.913` seconds/epoch from epoch600 to epoch700. At epoch742, about 258 epochs remain, or roughly `1.6` hours.
- Existing loss3 base timing was about `47.5` to `47.8` seconds/epoch, so the 500 epoch loss3 continuation should take roughly `6.6` hours if similar.
- Expected training completion remains roughly around `2026-06-06T08:50Z` to `2026-06-06T09:20Z`, plus plotting/reporting time.

## Runtime Health Check - 2026-06-06T05:11:47Z

Status: running normally. loss1 and loss2 continuation are complete; loss3 continuation is active.

Observed evidence:
- `tmux list-sessions` showed `round03_loss123_continuation` still running.
- `driver.log` showed loss1 done at `2026-06-05T23:13:01Z`, loss2 started at `2026-06-05T23:13:01Z`, loss2 done at `2026-06-06T02:16:13Z`, and loss3 started at `2026-06-06T02:16:13Z`.
- `pgrep -af adv_robust` showed PID `582426` running `/workspace/NeuralOperatorRobustness2/adv_robust/bin/python ... --device cuda ... --run-name burgers_loss3_selective_round03_loss3_continue500to1000_20260605 ... --burgers-attack-loss-objective loss3`.
- `nvidia-smi` at `2026-06-06 05:11:46 UTC` showed Tesla V100-SXM2-32GB in P0, `30732 MiB / 32768 MiB` used, `19%` GPU utilization, driver `580.76.05`, reported CUDA `13.0`. The high reserved memory is consistent with the loss3 solver-in-attack path.
- loss3 per-run log showed dataset preflight passed: train `1`, test `1`, generalization `50`.
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue500to1000_20260605/burgers/checkpoints.csv` showed checkpoints through epoch700, wall elapsed `9218.809753330424` seconds.
- `train_steps.csv` latest sampled row showed loss3 epoch728, global step2184, progress fraction `0.728`.
- `eval_split_summary.csv` latest sampled generated-generalization row showed epoch728 RMSE mean `0.022943942603149046`, relative L2 mean `0.041151765127125026`, accuracy-score mean `96.06512771211996`.
- `memory.csv` latest sampled rows showed CUDA reserved `30328.0` MiB and peak allocated `28306.25341796875` MiB.

Observed completed metrics:
- loss2 continuation has `summary.json`; total wall seconds `10989.178280254826` (`183.15297133758042` minutes).
- loss2 continuation final epoch1000 train/test/generated RMSE means: `0.00156830217012471` / `0.0016576365090477869` / `0.03650817803926622`; generated relative L2 mean `0.06547490209929319`; best generated RMSE `0.03556954401493024` at epoch948.
- loss1 continuation final epoch3000 generated RMSE mean remains `0.0343238570556901`; best generated RMSE `0.03294495584905165` at epoch2673.
- loss3 base epoch500 generated RMSE mean was `0.023660631499673727`; base best generated RMSE was `0.023076826214804887` at epoch492.
- active loss3 continuation best generated RMSE so far is `0.020467741032856916` at epoch727, relative L2 `0.03671246450588762`, accuracy-score mean `96.47682786473877`.

Inference from observed evidence:
- The run is healthy: tmux exists, CUDA process exists, GPU memory/utilization are nonzero, checkpoints/eval/train/memory CSVs are updating, and no error was observed in the inspected logs.
- loss2 completed normally and improved over its epoch500 baseline, but remains worse on generated generalization than loss1 continuation and much worse than loss3.
- loss3 continuation is already improving over the loss3 epoch500 generated-generalization baseline by the sampled epoch727/728 window; this is promising, but not final until the epoch1000 summary is written.
- Current loss3 checkpoint timing is about `45.25` seconds/epoch from epoch600 to epoch700. With about 272 epochs remaining from epoch728, estimated training completion is roughly `2026-06-06T08:35Z` to `2026-06-06T08:50Z`, plus final summary/plotting time.

## Training Completion Check - 2026-06-06T13:07:52Z

Status: training complete. Post-training stitched plots/final report are not yet recorded in this entry.

Observed evidence:
- `tmux list-sessions` no longer showed `round03_loss123_continuation`; only `ssh_tmux` remained.
- `pgrep -af adv_robust` returned no training process.
- `driver.log` showed `done burgers_loss3_selective_round03_loss3_continue500to1000_20260605` at `2026-06-06T08:39:01Z`, followed by `continuation all done`.
- `nvidia-smi` at `2026-06-06 13:07:53 UTC` showed Tesla V100-SXM2-32GB with `0 MiB / 32768 MiB` used and `0%` GPU utilization.
- All three continuation run directories had `summary.json` files.

Final continuation metrics from exact sources:
- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605/summary.json` and `burgers/eval_split_summary.csv`: wall seconds `10171.913992851973`; final epoch3000 train/test/generated RMSE `0.0010929236794394932` / `0.0012285454291087751` / `0.0343238570556901`; generated relative L2 `0.06153625953140944`; best generated RMSE `0.03294495584905165` at epoch2673.
- `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue500to1000_20260605/summary.json` and `burgers/eval_split_summary.csv`: wall seconds `10989.178280254826`; final epoch1000 train/test/generated RMSE `0.00156830217012471` / `0.0016576365090477869` / `0.03650817803926622`; generated relative L2 `0.06547490209929319`; best generated RMSE `0.03556954401493024` at epoch948.
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue500to1000_20260605/summary.json` and `burgers/eval_split_summary.csv`: wall seconds `22965.618723264895`; final epoch1000 train/test/generated RMSE `0.003190380039094097` / `0.0032434361898937398` / `0.02209842411568137`; generated relative L2 `0.03962216744527335`; best generated RMSE `0.019361087025757017` at epoch893.

Inference from observed evidence:
- The continuation training itself completed successfully for loss1, loss2, and loss3.
- Final generated-generalization ranking by RMSE is loss3 best (`0.02209842411568137`), then loss1 (`0.0343238570556901`), then loss2 (`0.03650817803926622`).
- Best-checkpoint generated-generalization ranking is also loss3 best (`0.019361087025757017` at epoch893), then loss1 (`0.03294495584905165` at epoch2673), then loss2 (`0.03556954401493024` at epoch948).
- Clean train/test ranking remains different: loss1/loss2 have lower clean RMSE than loss3, while loss3 is strongest on round03 generated-OOD generalization.

Remaining work:
- Generate stitched train/test/generalization RMSE, relative L2, and wall-clock curves connecting the base long runs to continuation runs.
- Create/update final result Markdown under `docs/` with the stitched-curve sources and final comparison.
- Keep CSV/NPZ/model/PNG artifacts out of Git unless explicitly requested.
