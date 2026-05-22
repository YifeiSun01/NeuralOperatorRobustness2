## 2026-05-22 - NS2D dictionary solver batch-size probe

- Status: ran forward-only GPU probes for true NS2D solver batch sizes; no full dictionary generation and no R2 upload were launched.
- Source used: `tools/benchmark_ns_dictionary_solver_batch_size.py` and `2D_NS_FNO2d_recurrent/data_generation/generate_ns_dictionary_batched.py`.
- Result document: `docs/ns2d_dictionary_solver_batch_probe_20260522.md`.
- Numeric artifacts: JSON files under `docs/ns2d_dictionary_solver_batch_probe_20260522/`, including `batch_10.json`, `batch_20.json`, `batch_32.json`, `batch_48.json`, `batch_256.json`, `batch_512.json`, and `batch_768.json`.
- GPU path observed: A100-SXM4-80GB; PyTorch `2.8.0+cu126`; CUDA runtime `12.6`; JAX backend `gpu`.
- Solver path observed from source: `ex.stepper._navier_stokes.NavierStokesVorticityZongyi(2, ...)`, confirming the probe used the 2D Navier-Stokes vorticity solver, not Burgers.
- Observed key timings: batch 10 `2.926s`, batch 20 `3.662s`, batch 32 `4.604s`, batch 48 `5.885s`, batch 512 `50.343s`, batch 768 `73.858s` for `dt=0.01`, `t_final=20`.
- Observed numerical stability: batch 10 and 20 were fully finite; batch 32 had `31/32` finite; batch 48 had `46/48` finite; batch 512 had `485/512` finite. Nonfinite samples need fallback to `dt=0.0005`.
- Observed memory: batch 512 used about `7.52 GiB` JAX peak in-use and `16.03 GiB` JAX peak pool; batch 768 used about `11.27 GiB` JAX peak in-use. The limiting factor in these probes was `dt=0.01` numerical stability, not A100 memory.
- Inference: a first-pass `dt=0.01` solve for 2000 samples with solver batch 512 is roughly four chunks, about `3.3` minutes by observed per-chunk time; adding fallback, CPU save, and R2 upload gives a practical planning range of about `15-30` minutes if the fallback rate stays near 5%.
- Remaining work: run the full dictionary generation/upload command when ready, preferably with `--batch-size 512 --solver-batch-size 512 --solver-mode vmap --step-options 0.01,0.0005`, then record actual generation, save, and upload timings.

## 2026-05-21 - NS2D batched dictionary generation speedup plan

- Status: added a true batched dictionary generator and updated the R2 upload command; no dataset generation was launched.
- Source added: `2D_NS_FNO2d_recurrent/data_generation/generate_ns_dictionary_batched.py`.
- Result documents updated: `docs/ns2d_recurrent_dictionary_generation_runtime_estimate_20260521.md` and `docs/ns2d_recurrent_dictionary_generation_upload_command_20260521.md`.
- Observed from legacy source: `batch_size` in `VT_NS_gen_all_frame_dict.py` only controls output batching; solver generation remains a per-sample loop.
- Observed from existing batched generation records: `solver_batch_size=8`, `solver_mode=lax-map`, and `dt=0.005` averaged about `1.99 s/sample` rollout time over 1200 samples.
- Implementation: the new script supports `--solver-batch-size` for true JAX/Exponax rollout batching, default `--solver-batch-size 20`, `--solver-mode vmap`, and `--step-options 0.01,0.0005`.
- Inference: with mostly `dt=0.01` samples on the current A100, the batched dictionary run should plausibly be closer to `45-90` minutes plus save/upload time than the legacy conservative `3-5` hour estimate; if many samples fall back to `dt=0.0005`, runtime can be longer.
- Validation: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/data_generation/generate_ns_dictionary_batched.py` passed.

## 2026-05-21 - NS2D dictionary generator dt/batch controls

- Status: updated the dictionary generator so the requested step-size fallback and output batch size can be controlled from the command; no dataset generation was launched.
- Source changed: `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.py`.
- Result documents updated: `docs/ns2d_recurrent_dictionary_generation_runtime_estimate_20260521.md` and `docs/ns2d_recurrent_dictionary_generation_upload_command_20260521.md`.
- Implementation: `NS_DICT_STEP_OPTIONS` controls the ordered solver step-size attempts; the requested command uses `0.01,0.0005`, so each sample first tries `dt=0.01` and falls back directly to `dt=0.0005` if unstable.
- Implementation: `NS_DICT_BATCH_SIZE` controls output batching; the requested command uses `2000`, which creates one output `.pt` for all samples. Source inspection shows the solver loop remains per-sample, so increasing output batch size above `N=2000` will not speed this script.
- Validation: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.py` passed.

## 2026-05-21 - NS2D recurrent dictionary generation and R2 upload command

- Status: prepared a command to generate the missing `N=2000` dictionary dataset and upload it to R2; no generation run was launched.
- Result document: `docs/ns2d_recurrent_dictionary_generation_upload_command_20260521.md`.
- Source referenced: `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.py`.
- Observed from source: the existing script already uses `batch_size=nsamples`, so output batching is already one large `N=2000` `.pt` file; the solver itself still loops per sample.
- Command behavior: verifies GPU path with `nvidia-smi`, PyTorch CUDA, and JAX GPU; preflights R2 with `rclone lsf`; generates the local dictionary if missing; uploads the expected `.pt` with `rclone copyto` using R2/S3 multipart-friendly chunk settings, then verifies the remote listing.
- Secrets policy: R2 credentials are passed via environment variables in the shell command and are not written into repository files.

## 2026-05-21 - NS2D attack conservative W/D-only batch10 command correction

- Status: corrected the current run command recommendation; no GPU run launched.
- Result documents updated: `docs/ns2d_recurrent_core4_attack_multimode_command_20260521.md` and `docs/ns2d_recurrent_core4_attack_loss_mode_mapping_20260521.md`.
- Correction: current W/D-only run should use `ATTACK_BATCH_SIZE=10` and `--indices 0,1,2,3,4,5,6,7,8,9` only; indices `10..16` are intentionally excluded.
- Dictionary state: A-mode groups remain excluded until the `N2000` dictionary file exists locally.
- Command grouping retained: `loss1/all_w`, plus `loss3` for `all_w`, `all_d_target_w`, `w1_5_d6_9_target_w`, and `d1_5_w6_9_target_w`, each with the four core methods.

## 2026-05-21 - NS2D recurrent dictionary generation runtime estimate

- Status: inspected dictionary generation source and historical logs; no dataset generation was launched.
- Source inspected: `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.py` and `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.sh`.
- Historical log inspected: `2D_NS_FNO2d_recurrent/data_generation/logs_gen/VT_NS_gen_all_frame_dict_7837625.out`.
- Result document: `docs/ns2d_recurrent_dictionary_generation_runtime_estimate_20260521.md`.
- Observed from source: the dictionary generator targets `N=2000`, `ntimepoints=21`, `256 x 256`, `nu=1e-5`, `tfinal=20`, one saved `.pt` batch, and tries solver steps `0.01`, `0.005`, `0.001`, `0.0005`, `0.0001` until stable.
- Observed from the historical B200 log: full `2000/2000` generation completed in `2:03:34`, about `3.71 s/sample`; `1875` samples used `dt=0.01`, `125` used fallback `dt=0.005`, and no failures were observed.
- Observed local state: current GPU query reported `NVIDIA A100-SXM4-80GB`; local dictionary directory exists but does not contain the full `N2000 ... all_frames.pt` dictionary file.
- Inference: on the current A100, budget roughly `3-5` hours for the script as written; if fixed `dt=0.005` is forced or many samples fall back, budget roughly `5-8` hours. Expected output size is about `11-12 GB`.

## 2026-05-21 - NS2D attack true/surrogate loss curve logging

- Status: updated attack code and documentation so the primary loss evidence contains both true and surrogate loss curves; no GPU run launched.
- Source changed: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Result document updated: `docs/ns2d_recurrent_core4_attack_logging_outputs_20260521.md`.
- Implementation: `surrogate_loss_*` fields record the active optimization objective; `true_loss_*` fields record the full all-W solver evaluation `||F(x + delta) - G(x + delta)||`; default `--true-loss-every 1` records true loss every attack step.
- Efficiency note: for `loss3/all_w`, the active loss is reused as true loss to avoid duplicate solver work; other surrogate modes pay an additional forward all-W true-loss evaluation per recorded step.
- Threshold crossing rows now include both `surrogate_loss_value` and `true_loss_value`.
- Validation: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` passed.

## 2026-05-21 - NS2D attack logging outputs for loss growth and delta thresholds

- Status: updated attack logging code and documentation; no GPU run launched.
- Source changed: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Result document: `docs/ns2d_recurrent_core4_attack_logging_outputs_20260521.md`.
- Logging added: active loss growth metrics in `per_step_metrics.csv` and `per_sample_step_metrics.csv`; `delta_threshold_crossings.csv` for first 25%, 50%, 75%, and 100% `delta_p / epsilon` crossings at batch-mean, all-samples, and per-sample scopes; threshold rows embedded in `summary.json`.
- Save behavior changed: `--save-steps` default is now empty, so intermediate trajectory arrays are not saved unless explicitly requested; `final_delta_and_metrics.npz` still always saves the final delta and final adversarial initial condition.
- Validation: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` passed.

## 2026-05-21 - NS2D attack loss/mode mapping correction

- Status: corrected the experiment plan so `loss_type` and `MODE_SPEC` are not treated as a full Cartesian product; no GPU run launched.
- Result document: `docs/ns2d_recurrent_core4_attack_loss_mode_mapping_20260521.md`.
- Correction recorded: `loss1` should run once without ADW mode sweep; `loss2` corresponds to fixed/approximate-target style groups such as `all_a_target_w`; `loss3` should sweep W/D/A target-path variants.
- Corrected full grouping has `28` method combinations (`1 loss1 group + 1 loss2 group + 5 loss3 groups`, each with 4 methods), rather than the earlier over-broad `72` combinations.
- Note added to `docs/ns2d_recurrent_core4_attack_multimode_command_20260521.md` pointing to the corrected grouped command.

## 2026-05-21 - NS2D attack multi-mode command recommendation

- Status: created a multi-mode command document for running 17 initial conditions across W/D/A mode presets, 3 losses, and 4 core methods; no GPU run launched.
- Result document: `docs/ns2d_recurrent_core4_attack_multimode_command_20260521.md`.
- Observed from source: CLI accepts one `--mode-spec` per run, so multiple ADW presets should be run with a shell loop.
- Observed local state: default A-mode dictionary file was missing locally at the time of this check, so the full ADW command includes a guard and a W/D-only fallback command is documented.
- Recommended throughput settings retained: `SOLVER_REMAT=chunk`, `SOLVER_REMAT_CHUNK_STEPS=20`, `ATTACK_BATCH_SIZE=17`.

## 2026-05-21 - NS2D attack run recommendation for maximum throughput

- Status: created a run recommendation document from existing benchmark records; no GPU run launched.
- Result document: `docs/ns2d_recurrent_core4_attack_run_recommendation_20260521.md`.
- Observed evidence summarized: `chunk=20`, batch 17 has the best observed throughput at about `0.876 sample-updates/s` with `73.34 GiB` peak reserved; `micro`, batch 12 is the safer long-run configuration at about `0.799 sample-updates/s` with more memory headroom; `second`, batch 15 is a middle ground.
- Recommended maximum-throughput parameters: `SOLVER_REMAT=chunk`, `SOLVER_REMAT_CHUNK_STEPS=20`, `ATTACK_BATCH_SIZE=17`.
- Recommended safer parameters: `SOLVER_REMAT=micro`, `ATTACK_BATCH_SIZE=12`.

## 2026-05-21 - NS2D attack reserved memory and recompute explanation doc

- Status: created consolidated Chinese Markdown explaining `reserved` memory, `allocated` memory, non-linear OOM cliff, checkpoint/remat granularity, recomputation coverage, and final batch/memory/time recommendations; no GPU run launched.
- Result document: `docs/ns2d_recurrent_core4_attack_reserved_memory_recompute_explanation_20260521.md`.
- Evidence summarized from existing benchmark outputs and source code: `none`, `micro/original`, `chunk=20`, `chunk=100`, and `second` modes; target rollout has `3800` solver micro-steps.
- Remaining work: commit and push source/record files as requested; generated large outputs remain unstaged unless explicitly requested.

## 2026-05-21 - NS2D attack final remat summary table

- Status: created final summary table for remat/checkpoint modes from existing benchmark outputs; no GPU run launched.
- Result document: `docs/ns2d_recurrent_core4_attack_remat_final_table_20260521.md`.
- Clarification recorded: explicit recompute coverage is `0%` for `none` and conceptually `~100%` of the differentiable 3800-step solver rollout for `micro`, `chunk=20`, `chunk=100`, and `second`; the modes differ by checkpoint granularity, not simple percentage slowdown.
- Observed best throughput remains `chunk=20`, batch 17, with `73.34 GiB` peak reserved and about `19.41s/update`; reliable recommendation remains `micro/original`, batch 12.

## 2026-05-21 - NS2D attack remat granularity clarification

- Status: clarified checkpoint/remat granularity and `chunk=100` status in the scaling document; no GPU run launched.
- Result document updated: `docs/ns2d_recurrent_core4_attack_remat_scaling_20260521.md`.
- Observed from source: `fixed_step=0.005` gives `200` solver micro-steps per second; target frame index `19` gives `3800` micro-steps for the target rollout.
- Clarification recorded: `chunk=20` means 190 checkpointed chunks over the 19-second target rollout, `chunk=100` means 38 chunks, and `second` means 19 chunks; these are remat/checkpoint units, not direct multiplicative slowdowns.
- Observed evidence: `chunk=100` has one-step memory probes at batch 12 and 15 but no 5-step steady timing probe, so it is not ranked as a final throughput recommendation.

## 2026-05-21 - NS2D attack remat scaling plots and summary

- Status: generated CPU-only plots and tables from existing benchmark outputs; no new GPU attack run launched.
- Result document: `docs/ns2d_recurrent_core4_attack_remat_scaling_20260521.md`.
- Numeric sources created: `docs/ns2d_recurrent_core4_attack_remat_scaling_20260521.csv` and `docs/ns2d_recurrent_core4_attack_remat_scaling_fits_20260521.csv`.
- Figures created: `docs/figures/ns2d_core4_attack_remat_memory_vs_batch_20260521.png`, `docs/figures/ns2d_core4_attack_remat_time_vs_batch_20260521.png`, and `docs/figures/ns2d_core4_attack_remat_throughput_vs_batch_20260521.png`.
- Observed evidence: checkpointed/remat passing points have nearly linear peak reserved memory, about `4.2-4.3 GiB` per added attack sample; `chunk=20`, batch 17 had the best observed throughput at about `0.876 sample-updates/s`.
- Inference: memory is close to linear only inside the passing region for a fixed remat mode; OOM boundaries are non-linear because JAX/XLA can change buffer plans and request large contiguous allocations.

## 2026-05-21 - NS2D attack GPU memory terminology clarification

- Status: clarified existing benchmark memory terminology; no GPU run launched.
- Result document updated: `docs/ns2d_recurrent_core4_attack_remat_comparison_20260521.md`.
- Observed evidence referenced: `SOLVER_REMAT=none`, batch 2 passed with about `11.19 GB` reserved, while batch 3 failed with a requested allocation around `46.91 GiB`; `chunk=20`, batch 17 passed with peak reserved `78743863296` bytes.
- Inference recorded: PyTorch `reserved` memory is allocator cache accounting, not a guarantee that the next JAX/XLA allocation can fit; OOM depends on the next contiguous/requested buffer and XLA buffer plan, and memory is not linear across remat modes or batch sizes.

## 2026-05-21 - NS2D core4 attack chunk20 batch17 steady benchmark

- Status: completed the previously missing GPU benchmark for `SOLVER_REMAT=chunk`, `SOLVER_REMAT_CHUNK_STEPS=20`, `ATTACK_BATCH_SIZE=17`, `STEPS=5`; no long/full sweep launched.
- Result document updated: `docs/ns2d_recurrent_core4_attack_remat_comparison_20260521.md`.
- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_225102_UTC`.
- GPU path verified before run: `NVIDIA A100-SXM4-80GB`; PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, compute capability `sm_80`; JAX `0.10.0` backend `gpu` with `CudaDevice(id=0)`.
- Settings: checkpoint `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`, `LOSS_TYPES=loss3`, `METHODS=raw_add`, `MODE_SPEC=all_w`, `EPSILON=32`, `ALPHA=1`, `P_ORDER=2`, `Q_ORDER=2`, target frame index `19`, fixed step `0.005`.
- Observed from `solver_rollout_trace.csv`: solver update rollout used `t_final=19`, `micro_steps_per_sample=3800`, `solver_remat=chunk`, `solver_remat_chunk_steps=20`, `x_shape=17x256x256`, `output_shape=20x17x256x256`, and `requires_grad=True`; later repeated rollout calls were cache hits.
- Observed from `per_step_metrics.csv`: first update reached `k=0` at `24.2438s`; warm update intervals were about `19.42s`, `19.41s`, `19.41s`, and `19.42s`; final no-backward evaluation interval was about `4.99s`; total 5-update method time was `106.8827s`, with summary runtime `107.4113s`.
- Observed from `batch_memory.csv`: max allocated `77158011904` bytes and max reserved `78743863296` bytes (`78.74 GB` decimal / `73.34 GiB` reserved).
- Inference: `chunk=20`, batch 17 is the best observed sample-throughput setting so far at about `17 / 19.41 = 0.88` sample-updates/s, roughly `9%` better than `micro` batch 12 or `second` batch 15; it is also close to the A100 memory ceiling and should be treated as aggressive.
- Clarification recorded: prior confusing timings came from mixing first-update/JIT timing, warm update intervals, final no-backward evaluation, and different batch sizes. The final short interval is not a real attack update.
- Remaining work: before a very long full sweep, decide whether the extra throughput is worth the narrower memory headroom; the conservative setting remains `SOLVER_REMAT=micro`, `ATTACK_BATCH_SIZE=12`.

## 2026-05-21 - NS2D core4 attack remat/checkpoint comparison summary

- Status: consolidated remat/checkpoint mode results; no new GPU run launched in this documentation step.
- Result document: `docs/ns2d_recurrent_core4_attack_remat_comparison_20260521.md`.
- Observed modes summarized: `none`, `micro`, `chunk=20`, `chunk=100`, and `second`.
- Observed conclusion: `none` is fastest per step but only reaches batch 2, so throughput is poor; `micro` reaches batch 14 and has measured warm throughput about `0.80` sample-updates/s at batch 12; `second` reaches batch 15 and has measured warm throughput about `0.80` sample-updates/s at batch 15; later completed evidence shows `chunk=20` reaches batch 17 and has measured warm throughput about `0.88` sample-updates/s at batch 17.
- Recommendation recorded: reliable setting is `SOLVER_REMAT=micro`, `ATTACK_BATCH_SIZE=12`; highest-throughput observed setting is `SOLVER_REMAT=chunk`, `SOLVER_REMAT_CHUNK_STEPS=20`, `ATTACK_BATCH_SIZE=17`, but it is close to the memory ceiling; middle-ground monitored setting is `SOLVER_REMAT=second`, `ATTACK_BATCH_SIZE=15`.

## 2026-05-21 - NS2D attack solver/remat clarification before full run

- Status: stopped additional batch/remat probing at user request; checked for running attack processes and found none beyond the status command itself.
- Observed from process check: no `attack_ns2d_recurrent_core4.py`, `run_ns2d_recurrent_core4_attack.sh`, or long `adv_robust/bin/python` attack process remained active.
- Clarification recorded: current attack commands use checkpoint `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`, the newly trained modes64/width60 final checkpoint.
- Clarification recorded: current default target is zero-based `target_frame_index=19`, so the solver uses `19 * 200 = 3800` micro-steps at `fixed_step=0.005`. This matches the training setup with `T_in=10`, `T_out=10` over frames `0..19`; if attacking physical `t=20.0` / frame index `20`, the code should be run with target frame `20` and recurrent output horizon adjusted to `T_out=11`.
- Clarification recorded: solver state resolution is `256x256` per frame, matching the dataset/model; the attack does not solve a `2560x2560` spatial grid.
- Clarification recorded: `solver_remat=second` ignores `solver_remat_chunk_steps`; the trace still showing chunk value `20` is a default field and should be treated as not applicable for `second`.

## 2026-05-21 - NS2D core4 attack memory/runtime strategy summary

- Status: consolidated existing GPU probes into a dedicated strategy document; no new GPU run launched in this step.
- Result document: `docs/ns2d_recurrent_core4_attack_memory_runtime_strategy_20260521.md`.
- Sources summarized: batch-size probe document, checkpoint-speed benchmark document, solver trace document, and `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Observed from prior probes: `batch=10`, `12`, and `14` pass for `loss3/all_w/raw_add/steps=1`; `batch=15`, `16`, and `20` fail.
- Observed from prior speed benchmark: `ATTACK_BATCH_SIZE=12`, `STEPS=5` has warm update speed about `15s/step`, estimated `100`-step runtime about `25.2 minutes` for one `loss3/all_w/raw_add` batch of 12 samples.
- Inference recorded: checkpoint/rematerialization probably reduces memory enough to allow larger batches but slows each update; the optimization target should be sample-updates per second, and the next experiment should compare current checkpointed solver mode against a no-rematerialization mode.
- Recommendation recorded: use `ATTACK_BATCH_SIZE=12` for full runs, use `14` only aggressively with monitoring, avoid `15+` for the current path.

## 2026-05-21 - NS2D core4 attack speed benchmark with checkpointed solver

- Status: completed short GPU speed benchmark for `loss3/all_w/raw_add` with checkpointed JAX solver path.
- Result document: `docs/ns2d_recurrent_core4_attack_speed_checkpoint_benchmark_20260521.md`.
- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215942_UTC`.
- Settings: `ATTACK_BATCH_SIZE=12`, `LOSS_TYPES=loss3`, `METHODS=raw_add`, `MODE_SPEC=all_w`, `STEPS=5`, `EPSILON=32`, `ALPHA=1`, `P_ORDER=2`, `Q_ORDER=2`.
- Observed from `solver_rollout_trace.csv`: update calls used `t_final=19`, `fixed_step=0.005`, `micro_steps_per_sample=3800`, `x_shape=12x256x256`, `output_shape=20x12x256x256`, and `requires_grad=True`.
- Observed from `per_step_metrics.csv`: first update completed at `20.525715954019688s`; subsequent update increments were about `15.0s`; final no-backward evaluation ended at `84.67072753596585s`.
- Observed memory: max allocated `54712062976` bytes and max reserved `55857643520` bytes.
- Inference: a 100-step `loss3/all_w/raw_add` attack for one batch of 12 samples is about `25.2` minutes on this A100 path; checkpoint/rematerialization is slower per step than the user's recalled B200 no-recompute run, but batch size 12 partly offsets throughput loss per sample.
- Post-benchmark GPU status: `nvidia-smi` returned to `0 MiB / 81920 MiB`.

## 2026-05-21 - NS2D core4 attack batch-size upper-bound probe

- Status: completed GPU batch-size probe for the `loss3/all_w/raw_add` solver-backward path.
- Result document: `docs/ns2d_recurrent_core4_attack_batch_size_probe_20260521.md`.
- GPU path verified before probes: `NVIDIA A100-SXM4-80GB`, PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, `sm_80`, JAX `0.10.0` backend `gpu`.
- Settings for all probes: `LOSS_TYPES=loss3`, `METHODS=raw_add`, `MODE_SPEC=all_w`, `STEPS=1`, `EPSILON=32`, `ALPHA=1`, `P_ORDER=2`, `Q_ORDER=2`, `SAVE_STEPS=0`.
- Observed passing batch sizes and memory:
  - `ATTACK_BATCH_SIZE=10`: output `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215124_UTC`, max allocated `45746082816` bytes, max reserved `46787461120` bytes.
  - `ATTACK_BATCH_SIZE=12`: output `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215421_UTC`, max allocated `54703543296` bytes, max reserved `55857643520` bytes.
  - `ATTACK_BATCH_SIZE=14`: output `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215525_UTC`, max allocated `63663100928` bytes, max reserved `64929923072` bytes.
- Observed failing batch sizes:
  - `ATTACK_BATCH_SIZE=15`: output root `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215632_UTC`, JAX backward/DLPack OOM while trying to allocate about `14.79 GiB`.
  - `ATTACK_BATCH_SIZE=16`: output root `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215325_UTC`, JAX backward/DLPack OOM while trying to allocate about `15.78 GiB`.
  - `ATTACK_BATCH_SIZE=20`: output root `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215231_UTC`, PyTorch recurrent FNO FFT OOM with about `79.04 GiB` in use.
- Observed solver traces for passing runs: each recorded `t_final=19`, `fixed_step=0.005`, `steps_per_second=200`, `micro_steps_per_sample=3800`, and `requires_grad=True` for the attack update rollout.
- Inference: the largest observed passing batch size for this exact one-step path is `14`; recommended full-run value is `12` for headroom.
- Post-probe GPU status: `nvidia-smi` returned to `0 MiB / 81920 MiB`.

## 2026-05-21 - NS2D core4 attack solver-rollout trace proof

- Status: added explicit solver-rollout tracing and ran one minimal GPU proof probe.
- Source changed: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Result document: `docs/ns2d_recurrent_core4_attack_solver_trace_20260521.md`.
- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_214900_UTC`.
- Settings: `ATTACK_BATCH_SIZE=1`, `LOSS_TYPES=loss3`, `METHODS=raw_add`, `MODE_SPEC=all_w`, `STEPS=1`, `EPSILON=32`, `ALPHA=1`, `p=q=2`.
- Observed from `solver_rollout_trace.csv`: three solver rollout calls were recorded. The attack update call used `t_final=19`, `fixed_step=0.005`, `steps_per_second=200`, `micro_steps_per_sample=3800`, `x_shape=1x256x256`, `output_shape=20x1x256x256`, and `requires_grad=True`.
- Observed from `summary.json`: `solver_rollout_calls=3`, method runtime `15.016598572023213` seconds.
- Observed from `batch_memory.csv`: max allocated `5442786304` bytes and max reserved `5890899968` bytes.
- Inference: the current `loss3/all_w` attack path really does invoke the differentiable NS solver to the twentieth frame and backpropagates through that solver path; low memory is consistent with JAX checkpoint/rematerialization in the solver scan.

## 2026-05-21 - NS2D core4 attack epsilon scale inspection

- Status: CPU-only source/data scale inspection; no GPU attack launched.
- Source inspected: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Dataset inspected on CPU: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`.
- Observed from code: `epsilon` is a raw batchwise L2/Lp budget over the full flattened `256x256` initial condition, not per-pixel and not normalized by grid size.
- Observed from CPU stats: test initial-condition mean L2 norm `66.26712036132812`, mean per-pixel RMS `0.25885701179504395`, mean Linf `0.7032134532928467`.
- Inference: `epsilon=32768` is about `494.48x` the mean initial-condition L2 norm; `epsilon=3276825` is about `49448.73x`. Raw-L2 first sweeps should be closer to `8, 16, 32, 64` unless an external scale conversion is explicitly added.
- Result document: `docs/ns2d_recurrent_core4_attack_epsilon_scale_20260521.md`.

## 2026-05-21 - Clarified NS2D core4 attack batch-size-6 evidence

- Status: read-only clarification from existing probe output and source code; no new attack run launched.
- Source code inspected: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`, especially `_rollout_for_modes`, `model_prediction`, and `run_one`.
- Output inspected: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_214034_UTC/summary.json` and `batch_memory.csv`.
- Observed from code: for `loss3` and `mode_spec=wwwwwwwwww`, the solver rollout requests recurrent input frames `1..9` and target frame `19`; `run_one` calls `objective.backward()` for `k < steps`.
- Observed from probe output: `dataset_indices` were `0..5`, `attack_batch_size=6`, `loss_type=loss3`, `method=raw_add`, `steps=1`, `epsilon=3276825`, `alpha=5`, `p=q=2`.
- Observed memory from `batch_memory.csv`: max allocated `27829064704` bytes and max reserved `29462888448` bytes.
- Inference: batch size 6 is evidenced for this one-step all-w loss3/raw_add probe on the A100 80GB, not a blanket guarantee for every longer/full attack configuration. Full-combo attack code runs loss/method combinations sequentially, not all simultaneously.

## 2026-05-21 - 2D NS recurrent core4 attack memory/cache optimization

- Status: source optimization completed; no full attack benchmark launched.
- Source files changed: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`, `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh`.
- Result document: `docs/ns2d_recurrent_core4_attack_memory_optimization_20260521.md`.
- Observed from source edits: JAX preallocation defaults to disabled, JAX memory fraction defaults to `0.40`, model/solver/dictionary objects are reused per run, per-step logging no longer copies full `delta/grad/direction` tensors to CPU, and dictionary lookup no longer materializes the large `[B, chunk, H*W]` diff tensor.
- Observed validation: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` passed; `bash -n 2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh` passed.
- Inference: the default path should reduce cache pressure while preserving speed better than forcing `XLA_PYTHON_CLIENT_ALLOCATOR=platform`; lowest-displayed-memory mode can be tested later with `CLEAR_JAX_CACHES_AFTER_BATCH=1` or platform allocator if needed.
- Observed post-edit GPU probe: `ATTACK_BATCH_SIZE=6`, `LOSS_TYPES=loss3`, `METHODS=raw_add`, `STEPS=1` completed successfully at `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_214034_UTC`.
- Observed from that probe: peak allocated `27829064704` bytes, peak reserved `29462888448` bytes, final in-process reserved after cache release `979369984` bytes, and post-run `nvidia-smi` returned to `0 MiB / 81920 MiB`.
- Not done: no full attack runtime/memory benchmark was launched in this step.

## 2026-05-21 - 2D NS recurrent core4 attack p2 epsilon/alpha probe

- Status: completed probe runs; no full long attack launched.
- Probe checkpoint: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`.
- Settings: `p=2`, `q=2`, `epsilon=3276825`, `alpha=5`, `mode_spec=all_w`.
- Observed: batch sizes `3`, `4`, and `6` passed for `loss3/raw_add/steps=1`.
- Observed: full-combination sanity probe passed with batch size `6`, `loss1 loss2 loss3`, and `raw_add raw_replace steepest_add steepest_replace` for `steps=1`.
- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_212630_UTC`.
- Inference: `ATTACK_BATCH_SIZE=6` is acceptable for launching the first p2/q2 run, with monitoring recommended for longer step counts.
- Result document: `docs/ns2d_recurrent_core4_attack_probe_20260521.md`.

## 2026-05-21 - Corrected FNO2d recurrent visualization format to GIF only

- Status: completed; deleted the prior PNG-based visualization outputs and regenerated final-checkpoint test visualizations as GIFs.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final`.
- Observed files: five four-panel GIFs named `sample_004_initial_solver_model_diff.gif`, `sample_025_initial_solver_model_diff.gif`, `sample_028_initial_solver_model_diff.gif`, `sample_030_initial_solver_model_diff.gif`, and `sample_033_initial_solver_model_diff.gif`.
- Observed by file search: no PNG files remain under the regenerated `fno2d_recurrent_test_predictions_20260521*` visualization outputs.
- Result document updated: `docs/fno2d_recurrent_test_prediction_visualization_20260521.md`.

## 2026-05-21 - FNO2d recurrent final checkpoint test visualizations

- Status: completed; generated visualizations for five test samples.
- Source files added: `2D_NS_FNO2d_recurrent/visualizations/visualize_recurrent_test_predictions.py`, `2D_NS_FNO2d_recurrent/visualizations/run_visualize_recurrent_test_predictions.sh`.
- Checkpoint visualized for final results: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final`.
- Samples: `4`, `25`, `28`, `30`, `33`.
- Observed metrics from visualization run: relative L2 values `0.0804970786`, `0.1197442338`, `0.0816519186`, `0.0518293418`, `0.1158824265`.
- Result document: `docs/fno2d_recurrent_test_prediction_visualization_20260521.md`.
- GPU status after visualization: observed `0 MiB` used, `81153 MiB` free.

## 2026-05-21 - FNO2d recurrent late-run monitoring near epoch 498

- Status: read-only monitoring of active training process; no model/data/GPU experiment was started.
- Observed from `progress.jsonl`: active run at epoch `498/500`, batch `10/72`, completed train batches `35794/36000` at `2026-05-21T20:53:07+00:00`.
- Observed ETA from latest progress record: `4m05s` remaining train-batch ETA; inference is about `4-5` minutes to local training completion before final evaluation/save/upload overhead.
- Observed latest completed epoch: epoch `497`, train relative L2 mean `0.05292977390081986`, train MSE `0.005659206264206897`.
- Observed latest test evaluation: epoch `490`, test relative L2 mean `0.08654208987951278`, test MSE `0.0178410891443491`, score `0.9134579101204872`; next test expected at epoch `500`.
- Result record updated: `docs/fno2d_recurrent_training_monitor_20260521.md`.

## 2026-05-21 - 2D NS recurrent FNO core4 attack code scaffold

- Status: source code added; not executed.
- Source files added: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`, `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh`.
- Result/design record: `docs/ns2d_recurrent_core4_attack_design_20260521.md`.
- Observed from source reading: the four optimizer rules are implemented in `tools/run_loss3_direction_proposal_ablation.py` as `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`; the 2D NS recurrent attack plumbing is represented by `2D_NS_FNO2d_recurrent/perturbation_methods/PGD_attack_adam_batch_adaptive.py`.
- Inference from source reading: the optimizer update rules are dimension-agnostic once tensors are flattened batchwise; the 2D-specific work is solver rollout, recurrent model input construction, dictionary lookup, and `a/d/w` gradient handling.
- Key settings encoded as defaults: `p=2`, `q=2`, `num_samples=5`, `attack_batch_size=1`, target frame index `19`, and mode preset `all_w`.
- Implementation note: `loss1` and `loss2` avoid computing the perturbed target frame and only build the recurrent FNO input; `loss3` computes `G(x+delta)` at the target frame.
- Not done: no attack run, no dictionary generation, no model/dataset loading, no GPU/CPU experiment execution, and no syntax validation command, by user request to avoid disturbing the active training run.

# Experiment Ledger

## 2026-05-20 Loss3 P-Not-Q Visualization Directory Location Check

Status: inspected local filesystem paths for the stopped p!=q visualization outputs; no experiment, analysis, or plotting job was launched.

Observed evidence:

- Stopped p!=q raw sweep root exists: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.
- Stopped p!=q analysis root exists: `forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520/`.
- New stopped visualization roots exist:
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520/`
- Each listed visualization root contains `manifest.json` and `figures/loss3_q_mean_curves_with_boundary_markers.png`.
- Index Markdown exists: `docs/loss3_pneq_stopped_visual_index_20260520.md`.

Inference:

- The requested new p!=q visualization directories are present locally; `pinf_q1` remains partial, and no `pinf_q2` stopped visual exists because no completed roots were present when stopped.

## 2026-05-20 Loss3 P-Not-Q Stopped Post-Processing Visuals

Status: completed post-processing and visualization from existing completed p!=q roots after stopping the overnight run; no additional experiment roots were launched.

Source files:

- Stopped p!=q sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.
- Analysis script: `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`.
- Plot script: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.

Output files:

- Analysis root: `forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520/`.
- Analysis Markdown: `docs/loss3_alpha_epsilon_core4_pneq_q_stopped_100steps_result_20260520.md`.
- Visual index: `docs/loss3_pneq_stopped_visual_index_20260520.md`.
- Visual roots:
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520/`

Observed evidence:

- Analysis skipped the interrupted `p=inf,q=1, epsilon=2.0, alpha=0.2` root because its manifest was not completed.
- Completed-setting counts in generated visual manifests: `p=1,q=2` 20, `p=1,q=inf` 20, `p=2,q=1` 20, `p=2,q=inf` 20, and `p=inf,q=1` partial 2.
- `p=inf,q=2` was not plotted because no completed roots were present when the run was stopped.

Inference:

- The stopped p!=q visual set is suitable for the four full P/Q pairs above. The `p=inf,q=1` output is only a partial preview and should be labeled as such in any interpretation.

Remaining work:

- Do not compare `p=inf,q=1` as a full alpha/epsilon sweep unless the missing settings are intentionally run later.

## 2026-05-20 Loss3 P-Not-Q Run Stopped By User

Status: stopped active overnight p!=q experiment at the user's request; no further experiment roots should be launched for this p!=q sweep in this turn.

Observed process evidence before stop:

- Active overnight driver: `bash tools/run_loss3_overnight_20260520.sh`.
- Active sweep: `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 --steps 100 --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 ...`.
- Active setting runner before stop: `p=inf,q=1, epsilon=2.0, alpha=0.2`, output root `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps2_alpha0p2_batch100_steps100_pinf_q1`.

Observed artifact status before post-processing:

- `p=1,q=2`: 20 completed roots.
- `p=1,q=inf`: 20 completed roots.
- `p=2,q=1`: 20 completed roots.
- `p=2,q=inf`: 20 completed roots.
- `p=inf,q=1`: 2 completed roots and 1 interrupted/run-started root.
- `p=inf,q=2`: not started locally in this stopped run.

Action taken:

- Sent SIGTERM to process group `63104` and verified no `loss3_overnight`, `run_loss3_alpha_epsilon_core4_sweep`, or `run_loss3_direction_proposal_ablation` processes remained.

Remaining work:

- Run post-processing only on completed roots currently present under `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.

## 2026-05-20 Loss3 P-Not-Q Path/Visualization Status Check

Status: inspected existing/local p!=q artifacts and active overnight process; no new experiment, plotting, or analysis job was launched.

Observed evidence:

- Active overnight driver is still running: `bash tools/run_loss3_overnight_20260520.sh`.
- Active p!=q sweep command is running: `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 --steps 100 --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 ...`.
- Current new p!=q raw sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`.
- Completion counts observed in the new p!=q sweep root: `p=1,q=2` 20 completed roots, `p=1,q=inf` 20 completed roots, `p=2,q=1` 20 completed roots, `p=2,q=inf` 18 completed roots plus 1 `run_started` root. Planned `p=inf,q=1` and `p=inf,q=2` pairs have not appeared yet.
- The overnight script plans to generate per-P/Q visual roots after the sweep/analysis finishes: `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_100steps_20260520/`, `..._p1_qinf_...`, `..._p2_q1_...`, `..._p2_qinf_...`, `..._pinf_q1_...`, and `..._pinf_q2_...`.
- Older completed P/Q comparison figures exist under `forensics/loss3_core_per_pq_four_figures_20260518/`, with per-P/Q folders such as `p1_q2`, `p1_qinf`, `p2_q1`, `p2_qinf`, `pinf_q1`, and `pinf_q2`.

Inference:

- The new 2026-05-20 alpha/epsilon p!=q sweep is not finished yet, so its new planned visual folders have not been generated locally yet.
- The currently available P/Q visual figures are the older 2026-05-18 per-P/Q figure set, not the new alpha/epsilon p!=q overnight visual set.

Remaining work:

- Wait for the active overnight p!=q sweep to finish; then the scripted analysis and per-P/Q visual folders should be generated automatically.

## 2026-05-20 Loss3 GPI Overall Conclusion Markdown

Status: created a dedicated Markdown synthesis note from existing experiment artifacts; no experiment, plotting job, or analysis job was launched.

Output file:

- `docs/loss3_gpi_overall_conclusion_20260520.md`

Observed evidence summarized:

- 300-step p2q2 summary table: `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`.
- 300-step p2q2 visual root: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`.
- GPI early-step comparison: `forensics/loss3_gpi_early_step_comparison_20260520/`.
- Baseline trajectory/GIF source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/`.

Inference recorded:

- GPI/replacement is a strong practical optimizer for the tested `loss3` settings because it reaches the boundary immediately, quickly converges to a final-like perturbation shape, and often obtains comparable perturbations/loss much faster than additive PGD-style methods.
- The conclusion is phrased as a speed/stability/Pareto advantage, not as unconditional final-loss dominance.

Remaining work:

- Extend early-step GPI comparisons beyond the saved baseline trajectory if a paper-level claim needs broader evidence.

## 2026-05-20 Loss3 GPI Overall Interpretation Check

Status: recorded synthesis from existing generated artifacts; no experiment, plotting job, or analysis job was launched.

Source evidence:

- 300-step p2q2 summary table: `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`.
- 300-step visual root: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`.
- GPI early-step comparison: `forensics/loss3_gpi_early_step_comparison_20260520/` and `docs/loss3_gpi_early_step_comparison_20260520.md`.
- GPI trajectory/GIF source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/`.

Observed evidence:

- Replacement/GPI reaches the p-norm boundary at step 1 in the p2q2 alpha/epsilon sweeps, while raw PGD and LP-steepest additive methods take multiple steps and can be much slower depending on alpha/epsilon.
- In the saved baseline early-step comparison, selected-sample mean `cos(delta_k, delta_300)` for GPI is `0.8868` at k=5, `0.9840` at k=10, and `0.9949` at k=20.
- Final perturbation shapes are often visually and cosine-wise similar across GPI, PGD, and LP-steepest additive methods for representative samples, though at least one saved sample is an outlier where GPI and PGD/LP final directions differ substantially.
- In 300-step p2q2 runs, GPI/replacement is not always the strict largest final-loss method; `steepest_add` or raw PGD can match or slightly exceed it in final mean loss for some alpha/epsilon settings.

Inference:

- The evidence supports describing GPI/replacement as a strong practical method for this loss3 setting: it rapidly saturates the perturbation budget, quickly converges to a stable perturbation shape, and often reaches comparable final perturbations much faster than additive PGD-style methods.
- The safer conclusion is speed/stability/Pareto advantage, not unconditional final-loss dominance. For final-loss-only evaluation after long horizons, additive LP-steepest can sometimes be competitive or better.

Remaining work:

- If this becomes a paper claim, phrase it as empirical evidence over the tested alpha/epsilon and p2q2 settings, and separately report final-loss winners, boundary-arrival speed, angular motion, and perturbation smoothness/similarity.

## 2026-05-20 Loss3 GIF Trace and Early-Step Artifact Location Reply

Status: verified existing artifact paths and clarified interpretation; no experiment or plotting job was launched.

Observed evidence:

- GIF trace manifest exists: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/trajectory_condition_gifs/manifest.json`.
- GPI early-step delta grid exists: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_delta_grid_steps_1_5_10_20_100_300.png`.

Inference:

- Boundary-ratio variance difference is primarily explained by normalized p-steepest directions for `steepest_add`/replacement methods versus unnormalized raw gradient scale for `raw_add`/PGD.
- Early-step visualization is available for the baseline GIF-trace run because it saved trajectory arrays; ordinary final-only artifacts cannot support the same full condition-panel reconstruction without saved trajectories.

Remaining work:

- Use the GIF trace directory and early-step comparison directory for visual inspection; rerun/save trajectory arrays for any missing exact settings that need step-wise condition panels.

## 2026-05-20 Loss3 GPI Early-Step Perturbation Comparison 18:41 UTC

Status: generated post-processing visualizations and tables from existing trajectory artifacts; no neural-operator experiment was launched.

Source files:

- Trajectory source: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/steepest_replace/trajectory_samples.npz`.
- Cross-method final comparison sources: sibling `trajectory_samples.npz` files for `raw_add`, `steepest_add`, `raw_replace`, and `steepest_replace`.
- Plotting script added: `tools/plot_loss3_gpi_early_step_comparison.py`.

Output files:

- Manifest: `forensics/loss3_gpi_early_step_comparison_20260520/manifest.json`.
- Delta grid: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_delta_grid_steps_1_5_10_20_100_300.png`.
- Loss/cosine curve: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_loss_cosine_to_final_selected_samples.png`.
- Condition panels: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_condition_panels/`.
- Similarity table: `forensics/loss3_gpi_early_step_comparison_20260520/tables/gpi_early_step_similarity.csv`.
- Result Markdown: `docs/loss3_gpi_early_step_comparison_20260520.md`.

Key settings:

- Baseline setting `epsilon=4`, `alpha=0.4`, `p=2`, `q=2`, `steps=300`.
- Method visualized: `steepest_replace` (GPI/replacement).
- Saved dataset indices: `0`, `7`, `40`, `47`.
- Plotted steps: `1`, `5`, `10`, `20`, `100`, `300`; condition panels include `1`, `5`, `10`, sample-best step, and `300`.

Observed evidence:

- Mean selected-sample `cos(delta_k, delta_300)` for GPI is `0.8868` at k=5, `0.9840` at k=10, `0.9949` at k=20, and `1.0000` at k=300.
- Mean selected-sample GPI loss is `3.4935` at k=5, `3.6642` at k=10, `3.6791` at k=20, `3.5362` at k=100, and `3.6344` at k=300.
- Dataset 40 is nonmonotone: it has higher GPI loss around k=20 than at k=100 or k=300 while remaining high-cosine to final.
- Cross-method final-delta similarity is high for datasets 7, 40, and 47, but dataset 0 is an outlier with low cosine between GPI final and PGD/LP-steepest final deltas.

Inference:

- For the saved baseline trajectory samples, GPI perturbation shape is already close to its k=300 final shape by k=10. This supports testing a 5-step/10-step GPI early-stop variant, but the evidence is currently selected-sample trajectory evidence, not a full-batch proof.
- Early stopping should be judged with both shape similarity and loss stability because the loss can fluctuate after the shape has nearly converged.

Remaining work:

- If exact early-stop performance is needed for all 100 samples and all alpha/epsilon settings, rerun or extend the sweep to save full-batch early-step deltas or compute early-stop summary metrics directly.

## 2026-05-20 Loss3 Angle Triptych Y-Axis Fix 18:34 UTC

Status: modified and regenerated visualization artifacts only; no neural-operator experiment was launched.

Source files:

- Plotting script updated: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.
- 100-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/*/per_step_metrics.csv`.
- 300-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`.

Output files:

- 100-step refreshed visual root: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/`.
- 100-step angle-available triptychs: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/dynamics_triptychs_angle_available/`.
- 300-step refreshed visual root: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`.
- 300-step angle-available triptychs: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs_angle_available/`.
- Updated visual notes: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`, `docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md`.

Observed evidence:

- Existing angle means have maximum `70.3312` degrees in both inspected 100-step and 300-step p2q2 roots.
- Existing angle mean-plus-std values have maximum `81.7749` degrees in both inspected roots.
- The previous fixed `0..180` degree y-axis compressed the angle curves unnecessarily.
- Refreshed manifests were generated at `2026-05-20T18:32:34.324100+00:00` for the 100-step visual root and `2026-05-20T18:34:08.800642+00:00` for the 300-step visual root.

Inference:

- A data-driven angle y-axis is more faithful for these figures: it preserves the actual angle range and makes PGD/LP-steepest/GPI angular-motion differences readable without clipping the observed mean +/- std band.

Remaining work:

- Use the regenerated triptychs for angular-motion interpretation; older images with fixed 0..180 y-axis should not be used for judging relative angular speed.

## 2026-05-20 Loss3 Angular-Motion Interpretation 18:31 UTC

Status: recorded interpretation from existing generated angle-dynamics figures; no new neural-operator experiment was launched.

Source files:

- Angle dynamics figures: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs_angle_available/`.
- Result Markdown updated: `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`.

Observed evidence:

- User inspection of the generated angle dynamics figures found that replacement/GPI-style methods have much larger per-step `angle(delta_k, delta_{k-1})` than raw PGD and LP-steepest additive PGD.
- Raw PGD shows the slowest angular movement; LP-steepest additive PGD rotates faster than raw PGD but still decays as the perturbation approaches the boundary.

Inference:

- Replacement/GPI should not be framed as a classical power-iteration optimizer with global guarantees for this nonquadratic neural loss.
- A better interpretation is geometric: replacement/GPI repeatedly solves a local linearized full-budget boundary-direction problem, avoiding the angular inertia of additive updates. This can explain why it reaches strong loss values quickly even when the objective is nonquadratic.
- The empirical claim should be: replacement/GPI is an aggressive boundary-direction optimizer with fast angular motion; its perturbation quality still needs to be checked with spectral/smoothness/shape metrics.

Remaining work:

- Compare angular-motion curves against high-frequency energy, derivative/TV metrics, and GIF trajectories to ensure the fast rotation does not correspond to transient spike-like perturbations.

## 2026-05-20 Loss3 Angle-Available Dynamics Triptych Fix 18:27 UTC

Status: modified and regenerated visualization artifacts only; no neural-operator experiment was launched.

Source files:

- Plotting script updated: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.
- 100-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/*/per_step_metrics.csv`.
- 300-step sweep source: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`.

Output files:

- 100-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`.
- 100-step angle-available triptych directory: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/dynamics_triptychs_angle_available/`.
- 100-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`.
- 300-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/manifest.json`.
- 300-step angle-available triptych directory: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs_angle_available/`.
- 300-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md`.

Observed evidence:

- The 100-step p2q2 sweep has angular metrics in 9 of 20 completed roots. The missing 11 roots are older completed settings produced before `delta_prev_angle_degrees_mean` was added.
- The explicit 100-step representative triptych settings (`epsilon,alpha` = `4,0.4`, `8,0.3`, `8,1.6`, `16,1.6`) are all among the older roots without angular metrics, so their angular panels cannot be reconstructed from local files.
- The refreshed 100-step manifest generated at `2026-05-20T18:26:38.752656+00:00` now lists 9 `angle_available_triptychs`.
- The 300-step p2q2 sweep has angular metrics in 20 of 20 completed roots. The refreshed 300-step manifest generated at `2026-05-20T18:27:35.916818+00:00` now lists 20 `angle_available_triptychs`.

Inference:

- The previous visual output hid usable angular-motion plots because only explicitly requested representative triptychs were generated, and those happened to be old no-angle settings. The new output separately plots every setting that actually has `delta_prev_angle_degrees_mean`.
- Old no-angle settings still require rerunning the experiment if angular motion is needed for those exact epsilon/alpha combinations.

Remaining work:

- Use `figures/dynamics_triptychs_angle_available/` for angular-motion review. Rerun old 100-step settings only if the exact missing epsilon/alpha angle trajectories are required.

## 2026-05-20 Loss3 Boundary-Ratio Std Interpretation 18:24 UTC

Status: inspected existing 300-step p2q2 per-step metrics and updated the dedicated result Markdown; no new neural-operator experiment was launched.

Source files:

- `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`
- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`

Observed evidence:

- For `epsilon=8, alpha=0.3`, max/mean `boundary_ratio_std` values are `raw_add=0.288/0.0588`, `steepest_add=0.0438/0.00211`, and `raw_replace=steepest_replace=~3e-08/~3e-08`.
- For `epsilon=8, alpha=0.3`, per-sample 99% boundary steps are `raw_add` mean/median/max `73.47/80/126`, `steepest_add` `29.44/29/34`, and replacement methods `1/1/1`.
- At `epsilon=8, alpha=0.3`, step 20 has `raw_add` `boundary_ratio_mean=0.310`, `boundary_ratio_std=0.194`; `steepest_add` has `boundary_ratio_mean=0.700`, `boundary_ratio_std=0.026`.
- At `epsilon=8, alpha=0.3`, step 1 has `steepest_add` `delta_step_pnorm_mean=0.3` and `delta_step_pnorm_std=1.5e-08`, consistent with the normalized p-steepest step rule.

Inference:

- The large `raw_add` boundary-ratio std is caused by sample-dependent raw-gradient scale and radial alignment, which make samples reach the epsilon boundary at very different steps.
- The smaller `steepest_add` boundary-ratio std is expected because the p-steepest direction is normalized before applying alpha, making radial budget usage much more synchronized across samples.
- Replacement/GPI boundary-ratio std is essentially floating-point noise because the replacement update enforces boundary norm from step 1 onward.

Remaining work:

- Use angular-change and post-boundary loss-gain diagnostics to study boundary movement; `boundary_ratio_std` only measures radial synchronization.

## 2026-05-20 Loss3 300-Step P2Q2 Interpretation Check 18:20 UTC

Status: inspected existing 300-step analysis tables and updated the dedicated result Markdown; no new neural-operator experiment was launched.

Source files:

- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`

Observed evidence:

- For `epsilon=2, alpha=0.4`, final mean losses are `raw_add=1.4420`, `steepest_add=1.4552`, and `raw_replace=steepest_replace=1.3651`; mean 99% boundary-hit steps are `25`, `6`, and `1` respectively.
- For `epsilon=2, alpha=0.8`, final mean losses are `raw_add=1.4551`, `steepest_add=1.4440`, and `raw_replace=steepest_replace=1.3651`; mean 99% boundary-hit steps are `13`, `3`, and `1` respectively.
- For large-epsilon examples, replacement/GPI has lower final-loss std than add methods, e.g. `epsilon=8, alpha=0.3`: replacement/GPI std `1.4882` versus `raw_add=2.4005` and `steepest_add=2.0900`; `epsilon=16, alpha=1.6`: replacement/GPI std `2.2430` versus `raw_add=4.1427` and `steepest_add=3.3813`.
- Across 20 p2q2 300-step settings, strict largest-final-mean winners are `steepest_add` in 16 settings and `raw_add` in 1 setting; replacement/GPI tie for largest final mean in 3 settings.

Inference:

- The observed 300-step evidence supports a Pareto-style conclusion: replacement/GPI is consistently fastest to the boundary and lower variance, while longer-horizon `steepest_add` can slightly exceed it in final mean loss for many settings.

Remaining work:

- Use both early-time performance and final 300-step loss when writing the final comparison; avoid claiming that GPI is always the final-loss winner.

## 2026-05-20 Loss3 Boundary-Ratio Std Visualization Fix 18:16 UTC

Status: modified and regenerated visualization artifacts only; no neural-operator experiment was launched by this fix.

Source files:

- Plotting script updated: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`.
- Source 100-step sweep data: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/*/per_step_metrics.csv`.
- Source 300-step sweep data: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv`.

Output files:

- 100-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`.
- 100-step added std figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/boundary_ratio_std_curves.png`.
- 100-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`.
- 300-step refreshed manifest: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/manifest.json`.
- 300-step added std figure: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/boundary_ratio_std_curves.png`.
- 300-step updated note: `docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md`.

Key settings:

- Plotted `p=2,q=2`, 20 alpha/epsilon settings, core four methods.
- The original mean boundary-ratio plot still uses true mean +/- sample std shading for every method; the new figure directly plots `boundary_ratio_std` by method and alpha/epsilon panel.

Observed evidence:

- 100-step visual manifest regenerated at `2026-05-20T18:15:51.486994+00:00` with `completed_setting_count=20` and a `boundary_std_curves` output path.
- 300-step visual manifest regenerated at `2026-05-20T18:16:42.069367+00:00` with `completed_setting_count=20` and a `boundary_std_curves` output path.
- Aggregating `boundary_ratio_std` over all 20 p2q2 100-step roots gives max/mean values: `raw_add` `0.2954/0.0751`, `raw_replace` `3.58e-08/2.86e-08`, `steepest_add` `0.0898/0.00340`, `steepest_replace` `3.58e-08/2.86e-08`.
- Aggregating `boundary_ratio_std` over all 20 p2q2 300-step roots gives max/mean values: `raw_add` `0.2954/0.0264`, `raw_replace` `3.64e-08/2.90e-08`, `steepest_add` `0.0898/0.00114`, `steepest_replace` `3.64e-08/2.90e-08`.

Inference:

- The apparent missing std shading on replacement methods is a visualization-scale issue, not missing computation: their boundary-ratio sample std is nearly zero and the true band collapses onto the mean line.
- The separate std curve figure makes this visible without artificially inflating the uncertainty band.

Remaining work:

- Use the new `boundary_ratio_std_curves.png` figures when checking whether a method has genuinely small across-sample variability versus visually hidden mean-curve shading.

## 2026-05-20 Loss3 300-Step Figure Location Check

Status: inspected generated 300-step visualization artifacts; no new experiment was launched by this check.

Observed evidence:

- 300-step visualization manifest exists and reports status `completed`, generated at `2026-05-20T10:33:34.196576+00:00`.
- Manifest source sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`.
- Manifest reports `completed_setting_count=20` and `pq_pairs_plotted=[{p: 2, q: 2}]`.
- 300-step loss figure exists: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`.
- Boundary-ratio figure exists: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/boundary_ratio_mean_curves.png`.
- Delta angular-speed figure exists: `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/delta_prev_angle_degrees_mean_curves.png`.
- Dynamics triptychs exist for representative settings under `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs/`.

Inference:

- The requested 300-step loss visualization has been generated. It is scoped to `p=2,q=2` and the 20 alpha/epsilon settings.

## 2026-05-20 Loss3 Overnight Run Status Check 17:54 UTC

Status: inspected active background run; no new experiment was launched by this check.

Observed process evidence:

- Overnight driver is still running: PID `63104`, command `bash tools/run_loss3_overnight_20260520.sh`, elapsed about `15:45:48` at inspection.
- Current active stage is stage 4: strict off-diagonal `p != q` 100-step sweep, command `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 --steps 100 --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 ...`.
- Current active setting runner: PID `131603`, output root `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps8_alpha0p2_batch100_steps100_p2_q1`.
- Current setting is `epsilon=8.0`, `alpha=0.2`, `p=2`, `q=1`, `steps=100`; current log has printed `[run] raw_add`.

Observed GPU evidence:

- `nvidia-smi` showed Tesla V100-SXM2-32GB, memory `9520 MiB / 32768 MiB`, GPU utilization `97%` at inspection.

Observed artifact/log evidence:

- Stage 1 `p=2,q=2` 100-step sweep/analysis/plots completed: 20 completed roots, plus one old `interrupted_not_for_analysis` root not included in analysis.
- Stage 2 `p=2,q=2` 300-step sweep completed: 20 completed roots.
- Stage 3 baseline GIF trace completed: baseline run manifest and trajectory GIF manifest are completed.
- Stage 4 strict `p != q` sweep has 50 completed roots and 1 running root out of 120 planned settings.
- Stage 4 progress by observed P/Q pair: `p=1,q=2` completed 20/20, `p=1,q=inf` completed 20/20, `p=2,q=1` completed 10/20 with the 11th running.
- Stages not yet reached in stage 4: remaining `p=2,q=1` settings, then `p=2,q=inf`, `p=inf,q=1`, and `p=inf,q=2`.
- Current logs: `logs/loss3_overnight_20260520T020904Z.log` and `logs/loss3_alpha_epsilon_core4_eps8_alpha0p2_p2_q1.log`.

Inference:

- The workflow has not finished. It is in the final major sweep stage, but the largest stage is still in progress.
- By planned setting count, completed settings are approximately 92 out of 160 experiment roots if counting stage 1/2/4 sweeps plus the baseline root; stage 4 itself is about 50/120 completed, with one active.

Remaining work:

- Continue monitoring until stage 4 completes and the automatic stage-4 analysis/visualization/similarity/post-boundary diagnostics finish.

## 2026-05-20 Loss3 Overnight Run Status Check 04:21 UTC

Status: inspected active background run; no new experiment was launched by this check.

Observed process evidence:

- Overnight driver is still running: PID `63104`, command `bash tools/run_loss3_overnight_20260520.sh`, elapsed about `02:12:24` at inspection.
- Current active stage is stage 2: `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 --steps 300 --p 2 --q 2 ...`.
- Current active setting runner: PID `74234`, output root `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/fno_nu0p001_eps2_alpha0p2_batch100_steps300_p2_q2`.
- Current setting is `epsilon=2.0`, `alpha=0.2`, `p=2`, `q=2`, `steps=300`.
- Within the current setting, `raw_add`, `raw_replace`, and `steepest_add` method summaries exist; `steepest_replace` summary is not yet present, so the setting is likely running the fourth method.

Observed GPU evidence:

- `nvidia-smi` showed Tesla V100-SXM2-32GB, memory `9522 MiB / 32768 MiB`, GPU utilization `95%` at inspection.

Observed artifact/log evidence:

- Stage 1 completed: `forensics/loss3_alpha_epsilon_core4_sweep_20260519` has `20` completed roots and no running/failed roots; stage-1 analysis, visualizations, final-delta similarity, and post-boundary diagnostics completed with `completed_setting_count=20`.
- Stage 2 status: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520` has `2` completed roots, `1` running root, and no failed roots.
- Baseline GIF trace has not started: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520` has no roots yet.
- Strict `p != q` sweep has not started: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520` has no roots yet.
- Current logs: `logs/loss3_overnight_20260520T020904Z.log` and `logs/loss3_alpha_epsilon_core4_eps2_alpha0p2_p2_q2.log`.

Inference:

- The workflow has not finished. It has completed the first major stage and is early in the second major stage.
- By major stage count: stage 1 done; stage 2 in progress; stages 3 and 4 pending.
- By 300-step p=2,q=2 stage: 2 of 20 settings completed, third setting running and near the fourth method.
- The strict off-diagonal P/Q stage is large (`120` 100-step settings) and will dominate remaining runtime after the 300-step and baseline stages complete.

Remaining work:

- Continue monitoring. If desired, inspect again after the current 300-step setting completes to refine runtime estimates.

## 2026-05-20 Loss3 Overnight Run Status Check 02:09 UTC

Status: inspected active background run; no new experiment was launched by this check.

Observed process evidence:

- Overnight driver is running: PID `63104`, command `bash tools/run_loss3_overnight_20260520.sh`, elapsed about `01:18` at inspection.
- Stage-1 sweep wrapper is running: PID `63429`, command `tools/run_loss3_alpha_epsilon_core4_sweep.py --base-out forensics/loss3_alpha_epsilon_core4_sweep_20260519 --steps 100 --p 2 --q 2 --skip-completed ...`.
- Current setting runner is running: PID `63430`, command `tools/run_loss3_direction_proposal_ablation.py`, output root `forensics/loss3_alpha_epsilon_core4_sweep_20260519/fno_nu0p001_eps1_alpha0p1_batch100_steps100_p2_q2`.
- Current method observed from log: `raw_add` for `epsilon=1.0`, `alpha=0.1`, `p=2`, `q=2`.

Observed GPU evidence:

- `nvidia-smi` showed Tesla V100-SXM2-32GB, memory `9520 MiB / 32768 MiB`, GPU utilization `86%` at inspection.
- Current setting manifest records PyTorch `2.8.0+cu126`, CUDA `12.6`, device `Tesla V100-SXM2-32GB`, compute capability `sm_70`, PyTorch arch list includes `sm_70`, and JAX backend `gpu` with device `cuda:0`.

Observed log evidence:

- Overnight log: `logs/loss3_overnight_20260520T020904Z.log`.
- Current setting log: `logs/loss3_alpha_epsilon_core4_eps1_alpha0p1_p2_q2.log`.
- Log has reached stage `1. Fill missing p=2,q=2 100-step settings` and printed `[run] raw_add` for the first new setting.

Inference:

- The overnight workflow is actively running, but tasks are sequential rather than all running in parallel. It is currently in task 1 of the requested sequence.
- Because the runner writes per-method outputs after method completion, the exact step inside `raw_add` is not visible from files yet; GPU utilization and the live Python process indicate active computation.

Remaining work:

- Continue monitoring the logs and manifests. Later stages should run automatically if this stage completes successfully.

## 2026-05-20 Loss3 Overnight Readiness Review

Status: completed readiness review; no optimizer experiment was launched.

Record file added:

- `docs/loss3_overnight_readiness_review_20260520.md`

Observed verification:

- `bash -n tools/run_loss3_overnight_20260520.sh` passed.
- `py_compile` passed for all runner/analyzer/plotter/GIF scripts used by the overnight workflow.
- Dry-run p=2,q=2 300-step plan resolved to 20 settings and the core four methods.
- Dry-run strict p!=q plan resolved to 120 settings: 6 off-diagonal P/Q pairs times 20 alpha/epsilon settings.
- Dry-run baseline GIF trace plan confirmed `trajectory_final_conditions_npz=true` and `gifs=true`.

Observed caveat:

- Existing 11 completed 100-step `p=2,q=2` roots predate the new angular-speed fields. Stage 1 uses `--skip-completed`, so only the 9 newly run 100-step roots will have those fields. The full 20-setting 300-step root will have complete angular-speed metrics for all 20 settings.

Inference:

- The current code is ready to run the requested overnight workflow and will automatically generate tables, figures, Markdown notes, delta similarity, post-boundary diagnostics, and GIF panels. Complete angular-speed analysis across all 20 p=2,q=2 settings should be taken from the fresh 300-step root unless the old 100-step roots are rerun.

Remaining work:

- Launch `bash tools/run_loss3_overnight_20260520.sh` on GPU when ready, then inspect the generated manifests and figures.

## 2026-05-20 Loss3 Delta Angular-Speed Metrics

Status: completed code/plotting update only; no optimizer experiment was launched.

Source/code files updated:

- `tools/run_loss3_direction_proposal_ablation.py`
- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`
- `tools/run_loss3_overnight_20260520.sh`

Record file updated:

- `docs/loss3_next_run_commands_20260520.md`

Prepared behavior:

- Future runs write step-to-step perturbation motion metrics to `per_step_metrics.csv` and `per_sample_step_metrics.csv`: `delta_prev_cosine`, `delta_prev_angle_degrees`, `delta_step_l2`, `delta_step_pnorm`, `delta_step_linf`, normalized step distances, and `delta_unit_direction_l2_step`.
- `delta_prev_angle_degrees` is recorded as NaN when either `delta_k` or `delta_{k-1}` has near-zero norm, avoiding a fake angle at initialization.
- Visualization now shades mean +/- std for loss and boundary-ratio curves and generates angular-speed curves when `delta_prev_angle_degrees_mean` is present.
- Representative settings now get dynamics triptychs with loss, boundary ratio, and `angle(delta_k, delta_{k-1})`, all as batch mean +/- std.

Remaining work:

- Run `bash tools/run_loss3_overnight_20260520.sh`; newly generated 300-step and off-diagonal P/Q roots will contain complete angular-speed metrics.
- Existing completed roots from before this update do not contain these fields unless rerun.
- Verification: `py_compile` passed for the modified runner/plotter/diagnostic scripts; `bash -n tools/run_loss3_overnight_20260520.sh` passed; a `/tmp` smoke test of `tools/plot_loss3_alpha_epsilon_core4_visuals.py` completed without modifying official outputs.

## 2026-05-20 Loss3 Overnight Script Preparation

Status: completed command-script preparation only; no optimizer experiment was launched.

Source/code file added:

- `tools/run_loss3_overnight_20260520.sh`

Record file updated:

- `docs/loss3_next_run_commands_20260520.md`

Prepared behavior:

- The script logs to `logs/loss3_overnight_<UTC_TIMESTAMP>.log`.
- It starts with `nvidia-smi` plus a PyTorch/JAX GPU quick check; individual experiment runners still perform the required GPU verification before official runs.
- It fills missing default `p=2,q=2` 100-step settings and refreshes analysis, visualization, final-delta similarity, and post-boundary diagnostics.
- It runs all 20 default `p=2,q=2` settings for 300 steps and automatically generates analysis, visualization, final-delta similarity, and post-boundary diagnostics.
- It runs the baseline `epsilon=4, alpha=0.4, p=2, q=2` detailed 300-step GIF trace and generates multi-panel trajectory GIFs.
- It runs strict off-diagonal `p != q` 100-step settings for `1:2`, `1:inf`, `2:1`, `2:inf`, `inf:1`, and `inf:2`, then generates per-P/Q visualizations, final-delta similarity, and post-boundary diagnostics.

Remaining work:

- The user should launch `bash tools/run_loss3_overnight_20260520.sh` in terminal when ready for the overnight GPU run.

## 2026-05-20 Loss3 Post-Boundary Mechanism Diagnostics

Status: completed post-processing analysis from existing completed `p=2,q=2` sweep artifacts; no optimizer experiment was rerun.

Source/code file added:

- `tools/analyze_loss3_post_boundary_mechanism.py`

Output / record files:

- Analysis root: `forensics/loss3_post_boundary_mechanism_20260520/`
- Per-setting diagnostics: `forensics/loss3_post_boundary_mechanism_20260520/tables/post_boundary_mechanism_by_setting.csv`
- Method rollup: `forensics/loss3_post_boundary_mechanism_20260520/tables/post_boundary_mechanism_rollup.csv`
- Result Markdown: `docs/loss3_post_boundary_mechanism_diagnostics_20260520.md`
- Main result Markdown updated: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`
- Similarity metric definitions updated: `docs/loss3_alpha_epsilon_core4_delta_similarity_20260520.md`
- Next-run command note updated: `docs/loss3_next_run_commands_20260520.md`

Observed evidence:

- The diagnostic manifest reports status `completed`, `completed_setting_count=11`, and `row_count=44` for current `p=2,q=2` artifacts.
- `raw_add` has mean 99% boundary hit step `43.1` and mean post-boundary loss gain `0.511`.
- `steepest_add` has mean 99% boundary hit step `13.45` and mean post-boundary loss gain `1.160`.
- `raw_replace` and `steepest_replace` hit the 99% boundary at step `1` and have mean post-boundary loss gain `3.028`, with mean 10-step post-boundary gain `2.917`.
- Selected trajectory diagnostics show lower boundary-hit-to-final delta cosine for replacement/GPI (`0.301`) than for `raw_add` (`0.768`) or `steepest_add` (`0.696`), consistent with larger direction refinement on the boundary.
- The `p=2` tangent-motion proxy is larger for replacement/GPI (`0.597`) than for `raw_add` (`0.215`) or `steepest_add` (`0.239`).

Inference:

- GPI/replacement's post-boundary loss growth is best interpreted as fast boundary-direction optimization after immediate budget use, not merely as early boundary arrival.
- Additive PGD-like methods may have slower post-boundary growth because they spend iterations reaching the boundary and their projected boundary updates have weaker tangent/boundary-surface motion.

Remaining work:

- Re-run the diagnostic script on the pending 20-setting 100-step sweep, the 300-step `p=2,q=2` sweep, and the detailed baseline GIF trajectory after those runs complete.

## 2026-05-20 Loss3 Next-Run Command Preparation

Status: completed code/runbook preparation only; no optimizer experiment was launched.

Source/code files updated or added:

- `tools/run_loss3_direction_proposal_ablation.py`
- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/plot_loss3_trajectory_gif_panels.py`

Record file added:

- `docs/loss3_next_run_commands_20260520.md`

Observed code changes:

- The baseline runner now has `--save-trajectory-final-conditions`, which augments selected `trajectory_samples.npz` files with `clean_initial`, `perturbed_initial`, `model_final_condition`, `solver_final_condition`, and `final_condition_residual` for every saved step.
- The alpha/epsilon sweep wrapper passes through `--save-trajectory-final-conditions` and `--no-save-delta-trajectory` when requested.
- A post-processing GIF panel script was added to render delta, perturbed initial condition, model final condition, solver final condition, and final-condition residual over optimization steps.

Prepared run order:

1. Fill the nine missing default 100-step `p=2,q=2` alpha/epsilon settings in the existing 2026-05-19 sweep root.
2. Run all 20 default `p=2,q=2` settings for 300 steps in a separate root.
3. Run the baseline `epsilon=4, alpha=0.4, p=2, q=2` detailed trajectory with GIF-ready final-condition arrays.
4. Run strict `p != q` 100-step settings for the six off-diagonal P/Q pairs.

Remaining work:

- The user should run the commands in `docs/loss3_next_run_commands_20260520.md` on GPU. After the runs complete, analyze and interpret the resulting 20-setting/300-step and non-P/Q artifacts.

## 2026-05-20 Loss3 Boundary-Marker Mechanism Interpretation

Status: completed documentation update based on existing `p=2,q=2` visualization artifacts; no optimizer experiment was rerun.

Source evidence:

- Loss curves with boundary markers: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Boundary-threshold table: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/boundary_threshold_loss_gain_summary.csv`
- Main result document updated: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`

Observed evidence:

- `raw_add` / PGD and `steepest_add` / Lp-steepest PGD have boundary-ratio markers spread across steps, showing visible radial travel toward the epsilon boundary.
- `steepest_replace` / GPI-style replacement has the `0.25`, `0.50`, `0.75`, and `0.99` mean boundary-ratio markers collapsed at the first step in the completed `p=2,q=2` settings.
- GPI/replacement still shows substantial post-boundary loss growth, so boundary arrival and loss convergence are separate phases.

Inference:

- The marker pattern supports a mechanism distinction: additive PGD-like methods spend iterations reaching the boundary, while GPI/replacement immediately uses the fixed budget and then refines the boundary direction.
- Together with the final-delta smoothness and similarity evidence, this strengthens the interpretation that `steepest_replace` / GPI is the better practical optimizer for the completed fixed-budget `p=2,q=2` loss3 setting.

Remaining work:

- Re-check the same marker-collapse pattern after the nine pending alpha/epsilon settings and any requested multi-P/Q sweeps are run.

## 2026-05-20 Loss3 Core-Four Final-Delta Similarity Analysis

Status: completed post-processing analysis from existing completed `p=2,q=2` sweep artifacts; no optimizer experiment was rerun.

Source file added:

- `tools/analyze_loss3_alpha_epsilon_core4_delta_similarity.py`

Source data:

- Completed roots under `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`, using each method's `final_deltas.npz`.

Output / record files:

- Analysis root: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/`
- Manifest: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/manifest.json`
- Pairwise summary: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/tables/final_delta_pairwise_similarity_summary.csv`
- Per-sample table: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/tables/final_delta_pairwise_similarity_per_sample.csv`
- Across-setting rollup: `forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520/tables/final_delta_pairwise_similarity_rollup.csv`
- Result Markdown: `docs/loss3_alpha_epsilon_core4_delta_similarity_20260520.md`
- Main result Markdown updated: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`

Observed evidence:

- The analysis completed with manifest status `completed`, `completed_setting_count=11`, `p_filter=2`, `q_filter=2`.
- `raw_replace` and `steepest_replace` are identical for `p=2`: cosine `1.0000`, centered cosine `1.0000`, spectral cosine `1.0000`, relative L2 `0.0000`.
- `raw_add` and `steepest_add` are highly similar: across-setting mean cosine `0.8986`, centered cosine `0.8971`, spectral cosine `0.9454`, relative L2 `0.2627`.
- Additive methods versus replacement/GPI methods have moderate signed spatial cosine but high spectral similarity: `raw_add` vs `steepest_replace` cosine `0.5713`, spectral cosine `0.8003`; `steepest_add` vs `steepest_replace` cosine `0.6144`, spectral cosine `0.8501`.
- Existing smoothness rollup records `steepest_replace` / GPI high-frequency ratio `3.707e-09`, first-derivative L2 `0.1955`, and total variation `2.472`, lower than additive methods.

Inference:

- Final perturbations share a broad low-frequency shape and are not wildly dissimilar across methods, but they cluster by update family: additive methods together, replacement/GPI methods together.
- The evidence supports the user's visual impression that the GPI perturbation is not an abnormal high-frequency or spike-like perturbation; it is smooth by the recorded metrics and spectrally similar to the other final deltas.
- For completed `p=2,q=2` settings, GPI/replacement looks better because it combines immediate boundary use, substantial post-boundary loss gain, and smooth final perturbations.

Remaining work:

- Repeat the same similarity analysis after the nine pending alpha/epsilon settings complete, and separately for additional P/Q pairs if multi-PQ alpha/epsilon runs are launched.


## 2026-05-20 Loss3 20-Setting Plan and Boundary-Threshold Markers

Status: code/plan/visualization update completed; no new optimizer experiment was launched.

Source files updated:

- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`
- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output / record files updated:

- `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_plan.json`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/boundary_threshold_loss_gain_summary.csv`
- `docs/loss3_alpha_epsilon_core4_sweep_plan_20260519.md`
- `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`
- `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`

Observed evidence:

- Dry-run generated a 20-setting default plan for `p=2,q=2`; the additional nine settings are pending and were not run in this update.
- Existing completed alpha/epsilon visualizations remain scoped to `p=2,q=2`; refreshed manifest records `pq_pairs_plotted=[{p: 2, q: 2}]`.
- The sweep wrapper now supports `--pq-pairs` for explicit multi-PQ alpha/epsilon runs.
- The loss-curve visualization now marks first mean boundary-ratio hits at `0.25`, `0.50`, `0.75`, and `0.99`.
- The analysis script's boundary thresholds were expanded to `0.25`, `0.50`, `0.75`, `0.95`, and `0.99` for future analysis runs.

Inference:

- Current conclusions about GPI/replacement versus raw PGD are evidenced for `p=2,q=2` only in this alpha/epsilon sweep.
- The new threshold markers/table should make it easier to separate radius growth, boundary arrival, and post-boundary directional optimization.

Remaining work:

- Run the default sweep to fill the nine pending `p=2,q=2` settings, or explicitly run a larger `--pq-pairs` grid if all-PQ alpha/epsilon evidence is required.
- Re-run `tools/analyze_loss3_alpha_epsilon_core4_sweep.py` after new experiments complete.


## 2026-05-20 Loss3 Plot Layout and Boundary-Gain Interpretation Update

Status: completed full visualization layout correction, reran plots, and added boundary-hit loss-gain table; no optimizer experiment was rerun.

Source file updated:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output / record files updated:

- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/boundary_ratio_mean_curves.png`
- Heatmaps under `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/heatmap_*.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/boundary_hit_loss_gain_summary.csv`
- `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`
- `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`

Observed evidence:

- Reran the plotting script; refreshed manifest reports status `completed`, generated at `2026-05-20T01:10:40.052414+00:00`, with `completed_setting_count=11`.
- Heatmap x-axis labels were shortened, subplot/colorbar margins were increased, and representative/delta-grid panels were given explicit title and bottom-margin spacing.
- The loss curves retain `x` markers for the first step where mean `boundary_ratio >= 0.99`; delta plots do not use cross markers.
- `boundary_hit_loss_gain_summary.csv` records loss at first mean 99% boundary hit, final loss, and post-boundary loss gain.
- At `epsilon=8, alpha=0.3`, `raw_add` has no mean 99% boundary hit by step `100`, while `steepest_replace` hits at step `1` and increases mean loss from `2.720` at boundary hit to `6.805` final.

Inference:

- The visual and tabular evidence supports the user's interpretation: raw PGD/additive updates are slow partly because they spend many steps reaching the boundary.
- Replacement/GPI-style methods should be described as reaching the boundary immediately and then continuing substantial directional optimization along or near the boundary; boundary arrival is not the same as final convergence.

Remaining work:

- Use the refreshed plot set and boundary-hit loss-gain table for the written comparison.


## 2026-05-20 Loss3 Curve Plot Layout Correction

Status: completed visualization layout correction and rerun; no optimizer experiment was rerun.

Source file updated:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output files refreshed:

- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/boundary_ratio_mean_curves.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`

Observed evidence:

- Moved the curve-figure legend to a separate bottom area so it no longer overlaps the title.
- Increased subplot size and spacing.
- Removed per-subplot boundary text annotations from the loss curves; only the boundary `x` marker remains on the curve.
- Added a horizontal `0.99` threshold line and fixed y-axis range to the boundary-ratio curves.
- Reran the plotting script; refreshed manifest reports status `completed`, generated at `2026-05-20T01:06:12.923337+00:00`, with `completed_setting_count=11`.

Inference:

- The boundary-ratio and loss-curve figures should now have readable method/color legend placement and less annotation clutter.

Remaining work:

- Visually inspect the refreshed PNGs for final report use.


## 2026-05-20 Loss3 Visualization Marker Correction

Status: completed visualization correction and rerun; no optimizer experiment was rerun.

Source file updated:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`

Output / record files refreshed:

- Visualization manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- Loss boundary-marker figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Representative sample panels: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/representative_samples/`
- Delta shape grids: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/delta_shape_grids/`
- Visualization note: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`

Observed evidence:

- Removed the black `x` marker from final-delta line plots and delta-shape grids.
- Retained `x` markers only on loss curves, where they indicate the first step with mean `boundary_ratio >= 0.99`.
- Reran the plotting script; refreshed manifest reports status `completed`, generated at `2026-05-20T01:03:29.127258+00:00`, with `completed_setting_count=11`.

Inference:

- Delta plots now show perturbation shape without a misleading marker. Boundary-arrival markers are visually reserved for loss curves only.

Remaining work:

- Use the refreshed figures for interpretation/reporting.


## 2026-05-20 Loss3 PGD Boundary Literature Note and Visualization Rerun

Status: visualization script rerun completed; literature note created from web research and local experiment context. No optimizer experiment was rerun.

Source files and sources:

- Local visualization script: `tools/plot_loss3_alpha_epsilon_core4_visuals.py`
- Local sweep outputs: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`
- Local analysis outputs: `forensics/loss3_alpha_epsilon_core4_analysis_20260519/`
- Web sources recorded in `docs/loss3_pgd_epsilon_boundary_optimum_notes_20260520.md`, including Goodfellow et al. 2014, Madry et al. 2017, DeepFool, Boundary Attack, AutoAttack/APGD, ART docs, and Distill discussion.

Output / record files:

- Refreshed visualization manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- Refreshed main loss figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Literature note: `docs/loss3_pgd_epsilon_boundary_optimum_notes_20260520.md`

Observed evidence:

- The plotting script was rerun and completed with manifest status `completed`, `completed_setting_count=11`, and boundary marker definition `first step with boundary_ratio_mean >= 0.99`.
- The web/literature note records that fixed-budget PGD/FGSM-style loss maximization is generally expected to use the epsilon budget under local linear/nonzero-gradient assumptions.
- The note also records exceptions: general nonconvex losses may have interior stationary optima or plateaus; input box constraints, regularizers, smoothness/frequency penalties, and minimum-distortion attacks can all lead to non-boundary solutions.

Inference:

- For our loss3 fixed-budget sweep, slow boundary arrival by `raw_add` is better interpreted as an optimization-path/step-scaling issue than as evidence that the true fixed-budget optimum lies inside the epsilon ball.
- Replacement/GPI-style methods reach the boundary immediately because they match the local-linear steepest/maximization geometry more directly.
- The next diagnostic should be radial loss profiles `L(x + r u)` and boundary-rescale checks for raw-add trajectories that remain inside the ball.

Remaining work:

- Add radial-profile plots and boundary-rescale loss comparisons if we want direct evidence for whether loss3 increases monotonically along final perturbation directions.


## 2026-05-20 Loss3 Alpha/Epsilon Core-Four Visualizations

Status: completed visualization post-processing from existing sweep artifacts; no optimizer experiment was rerun.

Source files:

- `tools/plot_loss3_alpha_epsilon_core4_visuals.py`
- Existing sweep outputs under `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`
- Existing analysis tables under `forensics/loss3_alpha_epsilon_core4_analysis_20260519/`

Output / record files:

- Visualization root: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/`
- Manifest: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/manifest.json`
- Result Markdown: `docs/loss3_alpha_epsilon_core4_visuals_20260520.md`
- Main loss figure: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/figures/loss3_q_mean_curves_with_boundary_markers.png`
- Peakiness table: `forensics/loss3_alpha_epsilon_core4_visuals_20260520/tables/final_delta_peakiness_summary.csv`

Key settings:

- Objective: `loss3_original`, recorded as `loss3_q`.
- Geometry: `p=2`, `q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- Completed settings visualized: `11`.
- Boundary marker definition: first step where mean `boundary_ratio >= 0.99`; this approximates `||delta||_p ~= epsilon` while avoiding exact floating-point equality.
- Representative sample panels: dataset index `40` for `(epsilon, alpha)` settings `(4,0.4)`, `(8,0.3)`, `(8,1.6)`, and `(16,1.6)`.

Observed evidence:

- The visualization script completed and wrote a manifest with status `completed`.
- Generated PNG files were readable and had nonzero dimensions, including loss curves with boundary-hit `x` markers, boundary-ratio curves, final-loss/smoothness/high-frequency heatmaps, representative clean/delta/clean+delta panels, and final-delta shape grids.
- The representative panels show clean initial condition, final delta, clean plus delta, delta spectrum, and sample-level loss curve with a boundary-hit marker.
- The peakiness table records `max(abs(delta)) / RMS(delta)`, max absolute delta, total variation, first-derivative L2, and high-frequency ratio from final deltas.

Inference:

- The marked loss curves directly separate pre-boundary growth from post-boundary optimization, which addresses the user's concern that additive methods can appear slow largely because they spend many steps reaching the epsilon boundary.
- The smoothness/peakiness figures provide a visual and numeric check for Direct-Delta-like sharp spikes or high-frequency perturbations; these should be read alongside the final-loss and boundary-arrival summaries.

Remaining work:

- Inspect the new PNG figures visually for paper/report selection.
- If a paper-facing figure set is needed, choose a smaller subset of settings and export publication-sized panels with the same boundary-marker convention.


## 2026-05-20 Loss3 Alpha/Epsilon Core-Four Sweep Completion Status

Status: completed and post-processed locally. This entry records a status/analysis check, not a new run launched by the assistant in this turn.

Source files:

- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`
- Existing runner: `tools/run_loss3_direction_proposal_ablation.py`

Output files:

- Sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`
- Sweep manifest: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_manifest.json`
- Analysis root: `forensics/loss3_alpha_epsilon_core4_analysis_20260519/`
- Result Markdown: `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`

Key settings:

- Objective: `loss3_original`, recorded as `loss3_q`.
- Geometry: `p=2`, `q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- Completed setting count: `11/11`; failed setting count: `0`.
- Batch/steps: dataset indices `0..99`, `batch_size=100`, `steps=100`, seed `0`.

Observed evidence:

- `ps` showed no running sweep or ablation process at the status check.
- `nvidia-smi` showed GPU utilization `0%` and memory `0 / 32768 MiB`, so no experiment process remained active.
- `sweep_manifest.json` reported status `completed`, `completed_count=11`, `failed_count=0`, start `2026-05-19T22:53:20.841733+00:00`, finish `2026-05-20T00:18:20.508758+00:00`, and summed setting elapsed time `5099.66` seconds.
- Every completed setting contains four method summaries with `101` per-step rows per method.
- The older interrupted directory `fno_nu0p001_eps4_alpha0p3_batch100_steps100_p2_q2/` remains marked `interrupted_not_for_analysis` and was skipped by post-processing.
- Post-processing generated `44` method-summary rows, `44` boundary-arrival rows, `77` winner-summary rows, and `4` method-rollup rows.

Observed boundary-arrival headline from the generated result Markdown:

- `raw_replace` and `steepest_replace` reach `99%` boundary at step `1` for all completed settings.
- `steepest_add` reaches `99%` boundary much later, matching the planned `epsilon/alpha` scale: for example `epsilon=4, alpha=0.4` has batch-mean step `11` and slowest-sample step `14`; `epsilon=8, alpha=0.3` has batch-mean step `31` and slowest-sample step `34`.
- `raw_add` is much slower to reach the boundary: for `epsilon=4, alpha=0.4`, batch-mean step `43` and slowest-sample step `56`; for `epsilon=8, alpha=0.3`, the batch mean never reaches `99%` by step `100`, and `32/100` samples do not reach `99%` by step `100`.

Inference:

- The main run is `100%` complete with `0` estimated remaining runtime.
- The user's concern is supported by the boundary-arrival diagnostics: additive raw-gradient PGD can spend many iterations below the epsilon boundary, especially at the old `epsilon=8, alpha=0.3` setting.
- Replacement/GPI-style methods' speed advantage is tightly coupled to immediate boundary arrival; final scientific interpretation should separate boundary-arrival speed from later boundary-surface optimization.

Remaining work:

- Inspect the generated result Markdown and CSV tables for the final-loss/smoothness tradeoff narrative.
- Optionally make compact plots/tables focusing on boundary arrival versus final loss for the paper-facing summary.

## 2026-05-19 Loss3 Alpha/Epsilon Core-Four Sweep Code Prep

Status: code and command preparation completed; official numerical sweep not completed in this turn. One accidentally started setting was stopped and marked not for analysis.

Source files:

- `tools/run_loss3_alpha_epsilon_core4_sweep.py`
- `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`
- Existing runner reused: `tools/run_loss3_direction_proposal_ablation.py`

Output / record files:

- `docs/loss3_alpha_epsilon_core4_sweep_plan_20260519.md`
- `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md` (pending; no completed numerical results yet)
- `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_plan.json`
- Interrupted partial directory: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/fno_nu0p001_eps4_alpha0p3_batch100_steps100_p2_q2/`

Key planned settings:

- Objective: `loss3_original`, recorded as `loss3_q` under `p=2,q=2`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- New baseline reference: `epsilon=4`, `alpha=0.4`.
- Previous slow reference retained: `epsilon=8`, `alpha=0.3`.
- Curated `(epsilon, alpha)` settings: `(2,0.2)`, `(2,0.4)`, `(4,0.2)`, `(4,0.4)`, `(4,0.8)`, `(4,1.2)`, `(8,0.3)`, `(8,0.4)`, `(8,0.8)`, `(8,1.6)`, `(16,1.6)`.
- Batch/steps: dataset indices `0..99`, `batch_size=100`, `steps=100`, seed `0`.

Observed evidence:

- GPU verification passed after repairing the broken `adv_robust/bin/python3` symlink: V100 `sm_70`, PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, PyTorch arch list includes `sm_70`, JAX backend `gpu`, PyTorch and JAX GPU matmul passed, and `pip check` passed.
- The new scripts passed Python bytecode compilation.
- A dry-run plan was generated successfully for the 11 curated settings and records nominal `epsilon/alpha` boundary-reach steps for L2-steepest additive updates.
- A formal sweep command was accidentally started earlier for `epsilon=4`, `alpha=0.3`; after the user clarified to prepare code and commands only, the sweep wrapper and child runner were stopped.
- The interrupted setting has no completed root-level `per_step_metrics.csv` and its manifest is marked `interrupted_not_for_analysis`.

Inference:

- No numerical optimizer conclusion should be drawn from the interrupted partial output.
- The prepared wrapper is ready for the user to launch the requested `p=2,q=2` alpha/epsilon sweep manually.
- The central scientific diagnostic is now boundary arrival: `delta_pnorm`, `boundary_ratio`, and first step to 95%/99% boundary at batch-mean and per-sample levels.
- In `p=2`, `raw_replace` and `steepest_replace` should coincide geometrically, but both rows are kept so the custom raw-replacement method is explicitly represented.

Remaining work:

- User launches `adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py` for the curated 11-setting boundary-arrival-focused sweep.
- After completion, run `adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py`.
- Update the ledger and result Markdown with observed final-loss, boundary-arrival, growth-speed, and smoothness conclusions from completed artifacts.


## 2026-05-16 Unified Eval-Metric Raw-Data Interpretation

Status: completed from the canonical 27-run saved trajectory data; no attack was rerun.

Key conclusions recorded in `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`:

- For endpoint solver-level `loss3_original`, directly optimizing the `loss3_original` family is strongest: `loss3_original_generalized_power` reaches final mean `6.8573`, followed by `loss3_original_lp_steepest_pgd` at `6.3782` and `loss3_original_pgd` at `5.3949`.
- For `loss3_increment_ratio`, the batch-mean winner is also `loss3_original_generalized_power` at `0.8199`, but per-sample comparisons are subtler: direct `loss3_increment_ratio_lp_steepest_pgd` wins more individual samples against the original-objective runs.
- For `loss1_original` and `loss2_original`, direct `loss1/loss2` original objectives remain best; `loss3_original` is not a reliable surrogate for them.
- Increment-ratio and regularized objectives are meaningful for perturbation-efficient or penalty-aware behavior, but they are not the strongest endpoint `loss3_original` attacks.

## 2026-05-16 Unified Eval-Metric Curves Shared-Y-Zero Correction

Status: completed after reviewing the first unified-evaluation replots.

Correction:

- The first three-panel replots fixed the y-axis metric within each figure, but did not enforce one global y-axis range across every figure that uses the same evaluation metric.
- The corrected output now uses `shared_y_zero`: same evaluation metric, same y-axis min/max, y-axis starts at `0`.
- The 3x3 matrix script was also corrected so all nine subplots share the same y-axis when they are evaluated by the same metric.

Corrected scripts and docs:

- `tools/plot_batch_eval_metric_three_panel_curves.py`
- `tools/plot_batch_eval_metric_matrix_curves.py`
- `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`

Corrected local outputs:

- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_three_panel_shared_y_zero/png/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves_shared_y_zero/png/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves_shared_y_zero/index_png/`

R2 upload:

- Uploaded `45` corrected PNG files, `24657968` bytes.
- R2 prefixes: `.../figures/eval_metric_three_panel_shared_y_zero/` and `.../figures/eval_metric_curves_shared_y_zero/`.

Path clarification:

- Use the complete 27-run source directory `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- The `20260516_..._local_repro` directory is only a small local reproduction of the loss3 subset, not the canonical full 27-run dataset.

## 2026-05-16 Unified Eval-Metric Three-Panel Loss Curve Replots

Status: completed from existing saved trajectory data; no attack was rerun.

Purpose:

- Replot the old three-panel `LOSS1/LOSS2/LOSS3 Objective Curves` layout so all three panels in one figure use the same evaluation metric on the y-axis.
- This fixes the ambiguity in the older figures, where each panel plotted its own optimized objective.
- The new figures keep original / increment-ratio / regularized as the three panels and PGD / LP-steepest PGD / generalized power iteration as the curves.
- Each new figure shares one y-axis min/max range across its three panels.

Data and scripts:

- Source data: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- Old script: `tools/plot_batch_three_loss_loss_only.py`.
- Existing 3x3 unified script: `tools/plot_batch_eval_metric_matrix_curves.py`.
- New three-panel unified script: `tools/plot_batch_eval_metric_three_panel_curves.py`.
- Documentation: `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`.

Outputs:

- Local PNG directory: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_three_panel/png/`.
- Generated 27 batch mean/std figures: 3 optimized-loss families times 9 shared evaluation metrics.
- R2 prefix: `s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_three_panel/png/`.
- R2 upload summary: `uploaded_files=27`, `uploaded_bytes=14315408`.

## 2026-05-16 Final RI / Ray Profile Report And R2 Backup

Status: completed and cleaned for FNO / Burgers `nu=0.001`. The final report is `docs/loss3_ray_profile_ri_final_report_fno_nu0p001_20260516.md`. The final valid data directory is `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.

Final conclusion:

- Ray / RI experiment succeeds as a local-to-global nonlinear diagnostic. Clean-point local directions win the small-radius diagnostics, but they do not remain endpoint-best after following the same fixed ray to `r=8`.
- `local_outward_growth` wins small norm-growth `100/100`, but endpoint `loss3` mean at `r=8` is `2.720` with `0/100` endpoint wins.
- `local_residual_movement` wins small residual-increment `100/100`, endpoint `loss3` mean is `4.139`, and endpoint wins are `31/100`.
- `loss3_original_final` has small local wins `0/100`, but endpoint `loss3` mean is `5.447`; it wins `58/100` among all directions and `81/100` among the three finite PGD attack objectives.

Crossover evidence:

- Mean `loss3_original_final` crosses `local_outward_growth` at approximately `r=0.823321`.
- Mean `loss3_original_final` crosses `loss3_increment_ratio_final` at approximately `r=0.929778`.
- No below-then-above crossing against `local_residual_movement` appears by `r=1` or `r=2` in the dense scan.

Final figures:

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_all_metrics_by_direction_0to8_formula_labeled_std.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to0p5_formula_labeled_std.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/figures/normal_batch100_loss3_crossover_zoom_0to1p0_formula_labeled_std.png`

R2 backup:

- `s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`
- Upload manifest: `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/r2_upload_manifest_20260516.txt`
- Manifest summary: `file_count=58`, `bytes_total=97423607`, `deleted_wrong_non_fixedsign_objects=0`.
- Full related-artifact sync manifest: `docs/r2_sync_manifest_ray_profile_20260516.md`; broader Ray-profile artifact sync uploaded `198` files and `145207187` bytes.

Bug note:

- Earlier non-fixed-sign manual PGD output is invalid. That wrapper backpropagated `-objective` and then updated `delta += alpha * grad`, which is descent for the target objective. The corrected PGD result uses `objective.backward()`, ascent update, and projection.

## 2026-05-16 Normal-Protocol Experiment 4 Batch-100 Ray Profile

Status: completed on GPU for FNO / Burgers `nu=0.001`. This is the requested old-style protocol check: no best-over-steps, no multi-restart, single initialization per attack objective, and final step direction only.

Files:

- `tools/run_loss3_ray_profile_normal_batch.py`
- `docs/loss3_ray_profile_normal_fno_nu0p001_gpu_batch100_plan_20260516.md`
- `docs/loss3_ray_profile_normal_fno_nu0p001_gpu_batch100_result_20260516.md`
- `forensics/loss3_ray_profile_normal_20260516/fno_nu0p001_gpu_v100_batch100/`

Run settings and hardware:

- Samples: `0..99` batch size 100.
- Endpoint radius: `epsilon=8.0`; 45 radii in the ray profile.
- Attack protocol: 50 Adam steps, learning rate `0.3`, final step only.
- Runtime: Tesla V100-SXM2-32GB, torch `2.8.0+cu126`, CUDA `12.6`, JAX backend `gpu`, `sm_70` verified.
- Runtime: 278.71 seconds.

Key result under this exact normal protocol:

- Endpoint winner counts at `r=8`: `loss3_increment_ratio_final` 47/100, `loss3_regularized_final` 43/100, `local_residual_movement` 7/100, `loss3_residual_increment_ratio_final` 2/100, `loss3_original_final` 1/100.
- Small-radius norm-growth winner: `local_outward_growth` 100/100.
- Small-radius residual-increment winner: `local_residual_movement` 100/100.
- Interpretation: batch size 100 does not make final-step `loss3_original` best under the old protocol. The ray curves clearly support the local-to-global mismatch: local directions have the steepest small-radius slope, but finite-radius optimized directions dominate at large radius.


## 2026-05-16 Corrected Experiment 4 Ray Profile Batch-20 GPU Run

Status: completed for FNO / 1D Burgers `nu=0.001` with GPU-only execution. This entry supersedes the earlier five-sample Ray Profile endpoint-winner interpretation.

Corrected files:

- `tools/run_loss3_ray_profile_corrected.py`
- `docs/loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_plan_20260516.md`
- `docs/loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_result_20260516.md`
- `docs/loss3_ray_profile_fno_nu0p001_gpu_result_20260516.md` marked historical/superseded
- `docs/loss3_original_theory_experiment_plan.md` Experiment 4 row updated

Run evidence:

- Batch samples: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 40, 47, 115`.
- Endpoint radius: `epsilon=8.0`; 41 ray radii.
- Attack steps: 100 per restart; learning rate `0.3`; update mode `raw`.
- Runtime: Tesla V100-SXM2-32GB, torch `2.8.0+cu126`, CUDA `12.6`, JAX backend `gpu`, `sm_70` verified.
- Output directory: `forensics/loss3_ray_profile_corrected_20260516/fno_nu0p001_gpu_v100_batch20/`.

Corrected conclusion:

- Finite-radius endpoint `loss3_original` winner at `r=8`: `loss3_original_pgd_best` in 20/20 samples.
- Small-radius clean residual norm-growth winners: mostly `loss3_regularized_pgd_best` / `local_outward_growth`; these are numerically tied or near-tied because regularized PGD collapses toward the local outward-growth direction in this setting.
- Small-radius residual-increment winners: `loss3_residual_increment_ratio_pgd_best` in 18/20 samples and `local_residual_movement` in 2/20 samples.
- Interpretation: Experiment 4 is a Ray diagnostic for nonlinearity. Local/ratio directions describe the clean-point small-radius slopes, but direct `loss3_original` PGD is the fair large-radius endpoint attack. The mismatch is the intended local-to-global gap.

Raw artifacts recorded:

- `ray_profile.csv`, `ray_profile_aggregate_curves.csv`, `ray_direction_summary.csv`, `ray_direction_aggregate.csv`, `ray_winner_summary.csv`, `attack_trace.csv`, `attack_best_by_sample.csv`, `direction_alignment.csv`, `directions.npz`, `deltas.npz`, figures, and `manifest.json`.


## 2026-05-16 Experiment 4 Ray Profile Status Updated To Complete

Status: historical/superseded by the corrected batch-20 GPU run above. The raw five-sample artifacts are retained, but the endpoint-winner interpretation is not the final conclusion.

Previous status: completed for the current FNO / 1D Burgers `nu=0.001` scope. This entry
supersedes earlier notes that described Experiment 4 as not yet fully run.

Updated durable status files:

- `docs/loss3_original_theory_experiment_plan.md`
- `docs/loss3_ray_profile_fno_nu0p001_gpu_plan_20260516.md`
- `docs/loss3_ray_profile_fno_nu0p001_gpu_result_20260516.md`
- `EXPERIMENT_LEDGER.md`

Completion evidence:

- Full GPU-only ray-profile run completed on Tesla V100-SXM2-32GB with
  `torch==2.8.0+cu126`, CUDA `12.6`, JAX backend `gpu`, and required torch arch
  `sm_70` verified in the manifest.
- Samples: `0, 7, 40, 47, 115`; endpoint radius `epsilon=8.0`; 45 ray radii;
  50 Adam ascent steps per regenerated finite-radius direction.
- Direction sources: `loss3_original`, `loss3_increment_ratio`,
  `loss3_residual_increment_ratio`, `loss3_regularized`, `local_error_svd`,
  `local_outward_growth`, and `random`.
- Output directory:
  `forensics/loss3_ray_profile_20260516/fno_nu0p001_gpu_v100/`.
- Recorded `ray_profile.csv`, `ray_profile_aggregate_curves.csv`,
  `ray_direction_summary.csv`, `ray_direction_aggregate.csv`,
  `ray_winner_summary.csv`, `attack_trace.csv`, `directions.npz`, figures, and
  `manifest.json`.

Historical conclusions from that superseded five-sample run:

- This old five-sample run was useful for raw curve inspection, but its endpoint-winner interpretation is superseded by the corrected batch-20 run above.
- The old run suggested a nonlinear local-to-global gap.
- Small-radius clean residual norm growth winner: `local_outward_growth` in 5/5
  samples.
- Small-radius residual-increment winner: `local_error_svd` in 5/5 samples.
- Finite endpoint `r=8` winner: `loss3_regularized` in 3/5 samples and
  `loss3_increment_ratio` in 2/5 samples.
- Therefore the local directions are real local diagnostics, but they are not
  the finite-radius endpoint-best attack directions.

Remaining work for Experiment 4:

- None for the current `nu=0.001` scope. Future repetitions for other
  viscosities, models, or endpoint radii should be treated as extension
  experiments, not blockers for marking Experiment 4 complete.

## 2026-05-16 Experiment 3 Small-Epsilon Sweep Status Updated To Complete

Status: completed for the current FNO / 1D Burgers `nu=0.001` scope. This entry
supersedes earlier same-day notes that described Experiment 3 as only partially
done before the GPU runs were completed.

Updated durable status files:

- `docs/loss3_original_theory_experiment_plan.md`
- `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
- `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`
- `EXPERIMENT_LEDGER.md`

Completion evidence:

- Candidate-bank small-epsilon sweep completed on GPU with epsilons
  `{1e-4, 1e-3, 1e-2, 1e-1}` for indices `0, 7, 40, 47, 115`.
- Recorded `L_f`, `L_j`, `L_e`, `G_e`, direction source counts, direction
  stability, local-reference ratios, figures, CSV tables, and a GPU manifest in
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`.
- Second version completed on GPU with projected gradient ascent on the unit
  direction `v`, recorded in
  `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`.
- The gradient version confirms that the candidate-bank directions were not just
  arbitrary picks: for small epsilon, optimized directions and values recover the
  same clean local Jacobian / outward-growth references.

Key conclusions now recorded:

- For `epsilon <= 1e-2`, the finite-difference ratios and directions are stable
  and match the clean local structure. This is the intended epsilon-refinement /
  local-convergence result.
- At `epsilon=0.1`, finite-radius drift becomes visible, especially in `L_e` and
  `G_e`, so that radius should not be described as purely local.
- `L_f` and `L_j` are much larger than `L_e`, showing that FNO and solver can be
  sensitive while still co-moving locally.
- `G_e` is much smaller than `L_e`, so residual-field movement and outward growth
  of the current clean residual are different diagnostics.
- The `L_j/idx47` direction-angle caveat is explained by a near-degenerate
  solver top-2 singular subspace (`sigma2/sigma1 = 0.977468`); top-2 subspace
  alignment is the correct diagnostic there.

Remaining work for Experiment 3:

- None for the current `nu=0.001` scope. Future repetitions for other viscosities
  or model families should be treated as extension experiments, not blockers for
  marking Experiment 3 complete.

## 2026-05-16 Loss3 Small-Epsilon Sweep Requirements And R2 Artifact Sync

Status: superseded pre-run requirements / artifact sync note. No numerical small-epsilon sweep
was run in that earlier turn, but Experiment 3 was later completed on GPU; see
`2026-05-16 Experiment 3 Small-Epsilon Sweep Status Updated To Complete`.

Source files and artifact roots:
- `docs/loss3_original_theory_experiment_plan.md`
- `docs/loss3_original_plan_r2_completion_audit_20260515.md`
- R2 prefix:
  `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`
- Local model checkpoint:
  `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`

Output files / synced artifacts:
- `docs/loss3_small_epsilon_sweep_requirements_r2_sync_20260516.md`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`
- `forensics/outward_growth_direction_20260515/`
- `results/burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only/`

Key settings:
- Experiment 3 target scope: FNO / 1D Burgers `nu=0.001`.
- Planned epsilon list: `{1e-4, 1e-3, 1e-2, 1e-1}`.
- Quantities still to estimate systematically: `L_f(epsilon)`,
  `L_j(epsilon)`, `L_e(epsilon)`, `G_e(epsilon)`, and cross-epsilon direction
  cosines.
- Existing local diagnostic indices from SVD/outward-growth artifacts:
  `[0, 7, 40, 47, 115]`.

Observed from local verification:
- The restored train split loads with `x=(1350, 1024)`, `y=(1350, 1024)`.
- The restored test split loads with `x=(150, 1024)`, `y=(150, 1024)`.
- The FNO checkpoint exists locally.
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/` contains
  15 raw `*_jacobian_svd.npz` files, covering FNO, solver, and error SVD arrays
  for five indices.
- `forensics/outward_growth_direction_20260515/` contains partial small-radius
  finite-difference evidence at radii `1e-4`, `1e-3`, and `1e-2`.
- `results/burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only/` contains
  older attack summaries for `Linf` epsilon `0.01` and `0.1`, but not the
  planned local operator/risk-growth sweep.

Inference:
- The required data/model/artifact prerequisites for implementing and running
  the planned Small-Epsilon Sweep are now present locally.
- Superseded: at the time of this requirements-sync note, Experiment 3 was only
  partially done. It was later completed on GPU with the systematic sweep and
  direction-stability analysis.

Remaining work from this pre-run note: superseded by the completed GPU sweep and
gradient-optimization follow-up.

## 2026-05-16 GitHub Script Sync And R2 Artifact Upload

Status: completed. Generated artifacts were uploaded to R2, and Markdown/Python files were committed and pushed to GitHub branch `vast-ai`.

Source files and artifact roots:
- Markdown / source files in the repository working tree.
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`
- `forensics/local_jacobian_frequency_20260514/`
- `forensics/outward_growth_direction_20260515/`
- `forensics/three_loss_pairwise_gradients_20260516/`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/split_summary.json`

Output locations:
- GitHub repository: `YifeiSun01/NeuralOperatorRobustness2`, branch `vast-ai`.
- GitHub commit with docs/tools: `6e4d37e` (`Record loss3 diagnostics and analysis scripts`).
- R2 prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/`.
- R2 artifact prefixes:
  - `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`
  - `forensics/local_jacobian_frequency_20260514/`
  - `forensics/outward_growth_direction_20260515/`
  - `forensics/three_loss_pairwise_gradients_20260516/`
  - `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/split_summary.json`

Observed from command output:
- R2 `forensics/` listing after upload included all four uploaded artifact directories.
- GitHub push output reported `36c5e30..6e4d37e  vast-ai -> vast-ai`.
- The staged GitHub commit contained existing Markdown/Python files only; deleted generated-result artifacts were not staged.
- No credential strings were found in the staged Markdown/Python files by the pre-commit scan.

Inference:
- GitHub now has the lightweight script/documentation update.
- R2 now has the current local generated numerical artifacts and plots under the selected machine-sync prefix.

Remaining work:
- The working tree still contains many pre-existing tracked generated artifacts marked deleted, plus non-script generated/untracked artifacts. These were intentionally not committed to GitHub.

## 2026-05-16 Loss3 Minimal Six Experiment Status Added To Plan

Status: documentation update only; no numerical experiment was run in this turn.

Source files:
- `docs/loss3_original_theory_experiment_plan.md`
- Prior completion notes in the conversation and existing audit/result docs.

Output files:
- `docs/loss3_original_theory_experiment_plan.md`

Key settings / scope:
- Current next-round scope is FNO / Burgers `nu=0.001` only.
- Broader `nu=0.01` or default-architecture sweeps are explicitly not required for the immediate next plan.

Observed from the existing records and user-confirmed notes:
- Superseded later on 2026-05-16: Experiment 3 was completed after this status
  note. Current status is Done: Experiments 1, 2, 3, and 5; Partially done:
  Experiment 6; Not yet done as a full experiment: Experiment 4.

Inference from the current status:
- Next priority should be Ray Profile / Local-to-Global Profile.
- Then complete the exact Direction Rotation Along Path experiment with recomputed `v_e*(x_t)`.
- The full Small-Epsilon Sweep was later completed and is no longer a blocker.

Remaining work:
- Run Experiment 4 ray-profile curves.
- Complete Experiment 6 with pathwise recomputed local top residual directions.
- Superseded: Experiment 3 was later completed with systematic `L_f`, `L_j`, `L_e`, `G_e`, and direction-cosine tables.

Last updated: 2026-05-16 UTC

This is the fixed entry point for experiment status. It must separate observed
evidence from inference. Chat history is not a durable experiment record.


## 2026-05-16 Chat Angle Tables Saved To Markdown

Status:

- No new numerical experiment was run.
- Appended the chat-presented numeric angle tables to
  `docs/angle_experiment_inventory_20260516.md` under `Chat-Presented Numeric Tables`.

Observed evidence summarized:

- Saved the overall 4-batch / 7-angle-definition table.
- Saved clean-point candidate direction angle means/stds.
- Saved endpoint-vs-movement loss3-path local-affine versus true nonlinear every-5-step table.
- Saved native L1/L2/L3 pairwise gradient angle summary and loss3-trajectory every-5-step table.

Inference:

- This was documentation of existing recorded data, not a new experiment.


## 2026-05-16 Angle Inventory Tables Presented In Chat

Status:

- No new numerical experiment was run.
- Presented the angle experiment inventory and headline numeric tables in the chat.

Source files:

- `docs/angle_experiment_inventory_20260516.md`
- `docs/angle_diagnostic_raw_tables_20260516.md`
- `docs/three_loss_pairwise_gradient_angles_result_20260516.md`
- `docs/outward_growth_direction_result_20260515.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source_k.csv`

Observed evidence summarized:

- Presented 4 angle diagnostic batches and 7 angle definitions.
- Parentheses in presented tables are standard deviations in degrees, not variances.
- Step-indexed values are saved every 5 PGD steps: `k = 0, 5, ..., 50`.

Inference:

- This turn is a presentation of existing recorded evidence only.


## 2026-05-16 Angle Experiment Inventory Created

Status:

- No new numerical experiment was run.
- Created `docs/angle_experiment_inventory_20260516.md` to summarize all angle-related
  diagnostics and their headline results.

Observed evidence summarized:

- Current records contain 4 angle-related diagnostic batches and 7 distinct angle
  definitions.
- The clean-point candidate-direction table has 135 CSV rows.
- The local-affine same-delta table has 165 rows and contains two angle definitions:
  endpoint-vs-movement and bias-vs-movement.
- The true nonlinear endpoint-vs-movement table has 165 rows.
- The native pairwise loss-gradient table has 165 rows and contains three angle
  definitions: L1-L2, L1-L3, and L2-L3.
- Headline native pairwise result for `k>=5`: `L1-L2 = 3.91 (7.06) deg`,
  `L1-L3 = 56.51 (18.41) deg`, `L2-L3 = 57.23 (18.41) deg`.

Inference:

- The angle diagnostics should be separated into endpoint-vs-movement/residual geometry
  diagnostics and native pairwise objective-gradient diagnostics.
- Only the native pairwise loss-gradient experiment directly proves that `loss3_original`
  has a different update direction from native `loss1_original` and `loss2_original`.


## 2026-05-16 Pairwise Gradient Angle Aggregation Clarified

Status:

- No new numerical experiment was run.
- Updated `docs/three_loss_pairwise_gradient_angles_result_20260516.md` to clarify
  saved-step cadence and mean/std aggregation.

Observed evidence summarized:

- Saved trajectory steps are every 5 PGD steps: `k = 0, 5, ..., 50`.
- Parentheses in the tables are standard deviations, not variances.
- The script computes population standard deviation with `np.std(..., ddof=0)`.
- `all k>=5` aggregates `150` rows = `5 samples x 3 delta sources x 10 saved nonzero k values`.
- Each per-trajectory `k>=5` summary aggregates `50` rows.
- A single `k=50` per-trajectory entry aggregates `5` rows, one per sample.

Inference:

- The reported angle summaries describe saved every-5-step trajectory points, not every
  internal PGD update step.


## 2026-05-16 b-plus-Adelta Versus Adelta Evidence Documented

Status:

- No new numerical experiment was run.
- Updated `docs/outward_growth_direction_result_20260515.md` with a section named
  `Where The Evidence Shows b+A delta And A delta Are Different`.

Observed evidence summarized:

- The endpoint residual objective keeps the clean residual term:
  `||f(x+delta)-j(x+delta)|| ~= ||b + A delta||`.
- The residual-movement objective subtracts the clean residual away:
  `||(f-j)(x+delta)-(f-j)(x)|| ~= ||A delta||`.
- Direction-response evidence: `error_top` top-8 has larger mismatch gain
  `0.412169` than `outward_growth` `0.368053`, but far smaller outward component
  `0.0140571` versus `0.166514`.
- Rank-1 `error` direction has mismatch gain `0.868217` but outward component
  `-0.00677836`.
- Direction-angle evidence: `outward_growth` versus `error` top-8 angle mean is
  `84.68 deg`; versus `error` rank-1 angle mean is `90.87 deg`.
- Finite-difference evidence: `outward_growth` predicted growth `0.166514` matches
  actual growth `0.167286` at `rho=1e-4` and `0.166668` at `rho=1e-3`.

Inference:

- The existing outward-growth experiment directly supports the claim that the local
  direction for `||A delta||` and the local direction for `||b + A delta||` are not the
  same.
- This should be described as endpoint residual versus residual increment, not as native
  `loss1`, because native `loss1 = ||f(x+delta)-f(x)||` linearizes with `J_f`, not
  `A = J_f - J_j`.


## 2026-05-16 Three-Loss Pairwise Gradient Angle Experiment

Status:

- Completed new GPU post-processing experiment requested by the user.
- Created `tools/analyze_three_loss_pairwise_gradients.py`.
- Created `docs/three_loss_pairwise_gradient_angles_result_20260516.md`.

Source files / inputs:

- `tools/analyze_three_loss_pairwise_gradients.py`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_000/loss1_original_pgd/config.json` and matching per-index configs.

Output files:

- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/pairwise_three_loss_gradients.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/manifest.json`
- `docs/three_loss_pairwise_gradient_angles_result_20260516.md`

Key settings:

- Device: `cuda`.
- Samples: `0, 7, 40, 47, 115`.
- Delta sources: `loss1_original_pgd`, `loss2_original_pgd`, `loss3_original_pgd`.
- Saved steps: `0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`.
- Compared native original objectives: `loss1`, `loss2`, `loss3`.

Observed evidence:

- Raw output has `165` rows = `5 samples x 3 delta sources x 11 saved steps`.
- For `k >= 5`, aggregate pairwise gradient angles are: `L1-L2 = 3.91 (7.06)` deg,
  `L1-L3 = 56.51 (18.41)` deg, `L2-L3 = 57.23 (18.41)` deg.
- For `k >= 5` along the `loss3_original_pgd` delta source, angles are:
  `L1-L2 = 9.60 (10.04)` deg, `L1-L3 = 61.30 (20.48)` deg,
  `L2-L3 = 63.05 (20.16)` deg.
- At `k=50` along the `loss3_original_pgd` delta source, mean angles are:
  `L1-L2 = 4.41 (1.54)` deg, `L1-L3 = 47.80 (20.31)` deg,
  `L2-L3 = 48.55 (20.04)` deg.

Inference:

- Native `loss1` and `loss2` gradients are usually close after nonzero attack progress.
- Native `loss3_original` gradients remain substantially different from both `loss1`
  and `loss2`, because `loss3` differentiates through the perturbed solver target
  `j(x + delta)`.
- This is the direct same-delta objective-gradient evidence that was missing from
  the earlier outward-growth experiment.

Remaining work:

- Optional: plot the three pairwise angle curves versus `k` for each delta source.


## 2026-05-16 Outward-Growth Experiment Reset Clarified

Status:

- No new numerical experiment was run.
- Updated `docs/outward_growth_direction_result_20260515.md` with a reset section
  explaining what the original outward-growth experiment actually ran.

Observed evidence summarized:

- The original outward-growth experiment used `A = J_f - J_j` and `b = f(x) - j(x)`.
- It compared the pure residual-field movement direction, which maximizes `||A v||`,
  with the clean-residual outward-growth direction `normalize(A^T b)`.
- It did not compute native pairwise gradient angles among `loss1`, `loss2`, and
  `loss3` at the same `delta_k` points.

Inference:

- The existing experiment directly supports the distinction between residual movement
  and endpoint residual growth inside the `loss3` residual geometry.
- It is conceptually related to the `loss1` versus `loss3` question, but it should not
  be described as a native `loss1` angle experiment because `loss1` uses `J_f`, not
  `J_f - J_j`.
- A separate pairwise objective-gradient diagnostic is needed to directly show that
  native `loss1_original`, `loss2_original`, and `loss3_original` have different
  local update directions.


## 2026-05-16 Angle Column Labels Reclassified As Delta Sources

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` to state explicitly that
  `loss1`, `loss2`, and `loss3` table columns are delta-source labels, not native
  angle objectives.

Observed evidence summarized:

- The true nonlinear angle formula compares two `loss3` residual-field gradients.
- The local-affine formulas use `A = J_f - J_j` and `b = f(x) - j(x)`.
- Therefore the reported `loss1` and `loss2` columns do not represent native
  `loss1` or `loss2` angle formulas; they only identify that `delta_k` came from
  the `loss1_original_pgd` or `loss2_original_pgd` saved trajectory.

Inference:

- The current angle tables should not be used to claim anything about intrinsic
  `loss1` gradient geometry, because `loss1` does not include the perturbed oracle
  `j(x + delta)`.
- A direct `loss1/loss2/loss3` gradient-angle experiment would need to compute
  `grad L1`, `grad L2`, and `grad L3` at the same fixed `delta_k` points.


## 2026-05-16 Angle Table Pairwise-Gradient Clarification

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` to clarify that the
  current endpoint-vs-movement angle tables are not pairwise `loss1/loss2/loss3`
  gradient-angle tables.

Observed evidence summarized:

- The current true nonlinear endpoint-vs-movement table compares two `loss3`-related
  gradients at saved points `z_k = x + delta_k`.
- It does not compute pairwise angles among `grad L1(delta_k)`, `grad L2(delta_k)`,
  and `grad L3(delta_k)`.

Inference:

- The existing tables support a statement about `loss3_original` endpoint-error
  geometry versus residual-movement geometry, especially along the `loss3` trajectory.
- A separate pairwise-gradient diagnostic is needed to directly claim that
  `loss1_original`, `loss2_original`, and `loss3_original` have different update
  directions at the same `delta_k`.


## 2026-05-16 Angle Table Wording Simplified

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` to remove the ambiguous
  phrase `trajectory probe` from the angle-table interpretation.

Observed evidence summarized:

- The tables report angles, not loss-growth curves.
- The `loss1`, `loss2`, and `loss3` column labels identify the trajectory that
  supplied the saved point `delta_k` where an angle was computed.
- For the true nonlinear table, the angle itself is still between two `loss3`
  residual-field gradients.

Inference:

- For the current theory claim, the `loss3` trajectory column is the cleanest one
  to emphasize. The `loss1` and `loss2` columns are secondary context only.


## 2026-05-16 Angle Table Interpretation Corrected

Status:

- No new numerical experiment was run.
- Updated `docs/angle_diagnostic_raw_tables_20260516.md` with an interpretation
  correction for the angle tables.

Source files:

- `docs/angle_diagnostic_raw_tables_20260516.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Observed evidence summarized:

- The `loss1`, `loss2`, and `loss3` columns in the step-indexed angle tables name
  the trajectory that supplied `delta_k`.
- The true nonlinear endpoint-vs-movement angle compares two gradients of the
  `loss3` residual field, even when the evaluated point came from a `loss1` or
  `loss2` trajectory.
- The local-affine endpoint-vs-movement and bias-vs-movement tables use
  `A = J_f - J_j` and `b = f(x) - j(x)`, so they are also local `loss3` residual
  geometry diagnostics evaluated along different trajectories.

Inference:

- The `loss3` columns have the cleanest interpretation for the current theoretical
  question because both the trajectory and compared gradients concern `loss3`.
- The `loss1` and `loss2` columns should be described only as angles computed at
  points reached by the `loss1` or `loss2` trajectories, not as intrinsic
  gradient-angle diagnostics for the `loss1` or `loss2` objectives.
- The bias-vs-movement table is a decomposition check inside the clean-point
  affine model, not a finite-radius attack comparison.

Remaining work:

- Optional: if intrinsic `loss1` or `loss2` gradient-angle diagnostics are needed,
  define separate endpoint/movement pairs for those objectives and compute them
  explicitly.


## 2026-05-16 Angle Mean/Std Tables Presented In Chat

Status:

- No new numerical experiment was run.
- Presented the recorded step-indexed angle mean/std tables in the chat response.

Source files:

- `docs/angle_diagnostic_raw_tables_20260516.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Observed evidence summarized:

- Three step-indexed angle families were presented: true nonlinear endpoint-vs-movement,
  local-affine endpoint-vs-movement, and local-affine bias-vs-movement.
- Tables use `mean (std)` in degrees over five samples: indices `0, 7, 40, 47, 115`.
- Saved steps are `k = 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`.

Inference:

- This was a presentation/record clarification only; no new empirical result was
  introduced.


## 2026-05-16 Angle Count And Mean/Std Tables Clarified

Status:

- Added `Count Summary And Mean/Std Tables` to
  `docs/angle_diagnostic_raw_tables_20260516.md`.
- No new numerical experiment was run for this entry.

Observed evidence summarized:

- Step-indexed angle diagnostics have three angle families: local-affine
  endpoint-vs-movement, local-affine bias-vs-movement, and true nonlinear
  endpoint-vs-movement.
- Each step-indexed family has `11` saved steps and `3` attack trajectories, giving
  `33` mean/std entries per family and `99` mean/std entries total.
- Each mean/std entry is aggregated over five samples: indices `0, 7, 40, 47, 115`.
- The clean-point `A^T b` versus top singular vector angle is not step-indexed; it is
  a local candidate-direction comparison at the clean point.

Inference:

- The clean-point outward-growth/SVD angle should not be described as a per-step
  optimizer-direction angle. It would become step-indexed only in a separate
  experiment that recomputes `A`, `b`, and SVD at each `x + delta_k`.

Remaining work:

- Optional: run a separate along-trajectory SVD/outward-growth recomputation if a
  per-step `A(z_k)^T b(z_k)` versus top singular vector angle is needed.


## 2026-05-16 Every-5-Step Angle Raw Tables Recorded

Status:

- Reran `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py` with
  `--sample-ks 0 5 10 15 20 25 30 35 40 45 50` to include `k=0`.
- Created `docs/angle_diagnostic_raw_tables_20260516.md`.
- Added `Raw Every-5-Step Angle Tables` cross-reference to
  `docs/outward_growth_direction_result_20260515.md`.

Source files / inputs:

- `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`
- Trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Existing local-affine summary:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack.csv`
- `docs/angle_diagnostic_raw_tables_20260516.md`

Observed evidence summarized:

- The true nonlinear summary now has `34` lines: header plus `3` attacks times `11`
  saved k values.
- Saved k values are `0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`.
- Angle families recorded in the Markdown: local candidate-direction angles,
  local-affine endpoint-vs-movement, local-affine bias-vs-movement, and true
  nonlinear endpoint-vs-movement.
- `k=0` is a near-zero initialization point: mean L2 delta norm is about
  `8.0046e-06`, budget ratio about `1.0006e-06`.

Inference:

- The every-5-step raw tables make explicit that the clean-point local-affine angles
  and true nonlinear same-point angles differ substantially, especially for the
  `loss3_original_pgd` path.
- `k=0` should be treated as a near-zero diagnostic, not as a stable finite update
  step, because movement-style gradients can be degenerate at exactly zero radius.

Remaining work:

- Optional: plot all three trajectory angle families versus `k`.


## 2026-05-16 Nonlinear Gradient Run Key Takeaways Recorded

Status:

- Added `Additional Key Takeaways From The Nonlinear Gradient Run` to
  `docs/outward_growth_direction_result_20260515.md`.
- No new numerical experiment was run for this entry; it summarizes existing
  nonlinear-gradient diagnostic outputs.

Observed evidence summarized:

- Source CSV:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`.
- For `loss3_original_pgd`, true nonlinear endpoint-vs-movement gradient angle
  decreases from `45.06 deg` at `k=5` to `36.36 deg` at `k=10`, `22.33 deg` at
  `k=25`, and `8.15 deg` at `k=50`.
- The corresponding `loss3_original_pgd` delta budget ratios are `0.0359`,
  `0.0924`, `0.3649`, and `0.7637`.
- The local-affine same-delta angles for the same path are much smaller:
  `21.07 deg`, `11.28 deg`, `4.22 deg`, and `1.40 deg`.
- For `loss3_original_pgd`, endpoint loss and movement loss become closer over the
  path: endpoint/movement means are `0.3438/0.1243` at `k=5`, `2.1124/2.0150` at
  `k=25`, and `4.6171/4.5728` at `k=50`.
- Movement-to-endpoint gradient norm ratio on `loss3_original_pgd` decreases from
  `2.7067` at `k=5` to `1.0486` at `k=50`.

Inference:

- The nonlinear diagnostic supports the narrative that different losses have
  substantially different local gradient geometry, especially early in optimization.
- Clean-point local-affine gradients understate the true nonlinear angle gap when
  `delta_k` is finite.
- As the residual increment grows and begins to dominate the clean residual, endpoint
  and movement objectives become more aligned, but they remain conceptually distinct.

Remaining work:

- Optional: plot angle, budget ratio, endpoint/movement loss, and gradient-norm ratio
  against `k` for the three trajectories.


## 2026-05-16 Angle Diagnostic Validity Summary Recorded

Status:

- Added `Angle Diagnostic Summary: Which Comparisons Are Valid` to
  `docs/outward_growth_direction_result_20260515.md`.
- No new numerical experiment was run for this entry; it summarizes existing angle
  diagnostics and their valid interpretation.

Observed evidence summarized:

- Local candidate-direction angles: `outward_growth` versus `error` top-8 angle mean
  `84.68 deg`, and versus `error` rank-1 angle mean `90.87 deg`.
- Same-delta local-affine gradient angles comparing
  `A^T b + A^T A delta_k` versus `A^T A delta_k`: for `loss3_original_pgd`,
  `21.07 deg` at `k=5`, `11.28 deg` at `k=10`, `4.22 deg` at `k=25`, and
  `1.40 deg` at `k=50`.
- True nonlinear same-point gradient angles comparing
  `grad_z ||f(z_k)-j(z_k)||_2` versus
  `grad_z ||(f(z_k)-j(z_k))-(f(x)-j(x))||_2`: for `loss3_original_pgd`,
  `45.06 deg` at `k=5`, `36.36 deg` at `k=10`, `22.33 deg` at `k=25`, and
  `8.15 deg` at `k=50`.

Inference:

- The first angle set is valid only for comparing local candidate diagnostics, not
  final optimizer directions or same-iterate gradients.
- The second angle set is valid inside the clean-point local-affine Taylor model but
  can understate finite-delta nonlinear effects.
- The third angle set is the preferred finite-saved-point comparison because it uses
  actual nonlinear autograd at `z_k = x + delta_k`.
- The user's concern is correct: `A^T b + A^T A delta_k` is not wrong as a local
  approximation, but it should not be used as the final finite-radius geometry when
  `delta_k` is not tiny.

Remaining work:

- Optional: add plots comparing local-affine and true nonlinear angle curves over
  `k` for each trajectory.


## 2026-05-16 Analytic-Solution Hierarchy For Delta Objectives Recorded

Status:

- Created `docs/analytic_solution_hierarchy_for_delta_objectives_20260516.md`.
- Added a cross-reference section, `Analytic-Solution Hierarchy Note`, to
  `docs/outward_growth_direction_result_20260515.md`.
- No new numerical experiment was run for this entry.

Purpose:

- Record the analytic-solution distinction requested by the user:
  pure residual movement, local affine endpoint error, and true nonlinear
  finite-radius attack are different optimization levels.

Observed record:

- The dedicated Markdown records that
  `max_{||delta|| <= epsilon} ||A delta||^2` has the simple top-right singular
  vector solution of `A` in the L2 case.
- It records that
  `max_{||delta|| <= epsilon} ||b + A delta||^2` has a KKT / implicit
  trust-region characterization, with stationarity
  `A^T A delta + A^T b = mu delta`, but is not generally a simple singular-vector
  solution.
- It records that the true nonlinear attack
  `max_{||delta|| <= epsilon} ||f(x + delta) - j(x + delta)||` has no general
  closed-form analytic solution and should be studied through iterative
  optimization, actual trajectories, final deltas, and same-point nonlinear
  gradients.

Inference:

- This entry is a theory/documentation update, not new empirical evidence.
- The recorded hierarchy clarifies when `delta` can be treated as infinitesimal:
  only in local linearization diagnostics such as `e(x+delta) approx b + A delta`.
  Finite-radius attacks require direct nonlinear evaluation or optimization.

Remaining work:

- None for this documentation request.


## 2026-05-15 True Nonlinear Same-Point Endpoint-vs-Movement Gradient Diagnostic Completed

Status:

- Added `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`.
- Ran it on existing saved FNO `nu=0.001` GPU attack trajectories; no attack rerun
  was needed.
- Added `True Nonlinear Same-Point Gradient Comparison` to
  `docs/outward_growth_direction_result_20260515.md`.
- `adv_robust/bin/python -m py_compile tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`
  passed.

Purpose:

- Address the finite-`delta` limitation of the fixed-clean-Jacobian diagnostic.
- Instead of comparing `A^T b + A^T A delta_k` with `A^T A delta_k`, directly
  differentiate the actual nonlinear losses at each saved finite point
  `z_k = x + delta_k`.

Source files / inputs:

- Script: `tools/analyze_true_nonlinear_endpoint_vs_movement_gradients.py`
- Result root:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/`
- Trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Configs:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss1_original_pgd/config.json`

Output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack_k.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/summary_by_attack.csv`

Key settings:

- Indices: `0, 7, 40, 47, 115`.
- Attack trajectories: `loss1_original_pgd`, `loss2_original_pgd`,
  `loss3_original_pgd`.
- Saved steps summarized: originally `k = 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`; later rerun also includes `k = 0`.
- Compared gradients:
  `grad_z ||f(z_k)-j(z_k)||_2` versus
  `grad_z ||(f(z_k)-j(z_k))-(f(x)-j(x))||_2`.

Observed evidence summarized:

- For `loss1_original_pgd`, mean nonlinear gradient angle was `13.48 deg` at
  `k=5`, `1.48 deg` at `k=25`, and `0.55 deg` at `k=50`.
- For `loss2_original_pgd`, mean nonlinear gradient angle was `37.33 deg` at
  `k=5`, `1.67 deg` at `k=25`, and `0.82 deg` at `k=50`.
- For `loss3_original_pgd`, mean nonlinear gradient angle was `45.06 deg` at
  `k=5`, `36.36 deg` at `k=10`, `22.33 deg` at `k=25`, and `8.15 deg` at
  `k=50`.
- Mean `loss3_original_pgd` delta budget ratios at those same steps were
  `0.0359`, `0.0924`, `0.3649`, and `0.7637`, respectively.

Inference:

- The user's objection was correct: the fixed-clean-point `A=J_f(x)-J_j(x)`
  diagnostic is only local-affine and should not be treated as the final answer
  when `delta_k` is finite.
- Direct nonlinear autograd shows that endpoint-error and residual-movement
  gradients can differ substantially at the same finite point, especially along
  the `loss3_original_pgd` trajectory.
- The corrected layered interpretation is: local residual movement, local
  clean-error outward growth, true finite-point nonlinear endpoint gradient, and
  final finite-radius attack direction are distinct objects.

Remaining work:

- Optional: plot the nonlinear gradient-angle curves over `k`.
- Optional: compare these exact nonlinear gradients against the actual projected
  PGD update directions after L2 projection/clipping.


## 2026-05-15 Same-Delta Local Gradient-Angle Diagnostic Completed

Status:

- Added `tools/analyze_same_delta_local_gradient_angles.py`.
- Ran the script using existing saved FNO `nu=0.001` attack trajectories and
  existing explicit error Jacobians; no GPU attack rerun and no Jacobian
  recomputation were needed.
- Added `Same-Delta Gradient Diagnostic Results` to
  `docs/outward_growth_direction_result_20260515.md`.

Source files / inputs:

- `tools/analyze_same_delta_local_gradient_angles.py`
- Trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Error Jacobians:
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/error/*_jacobian_svd.npz`
- Clean residuals:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_residual.npy`

Output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_diagnostics.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_summary_by_attack_k.csv`

Observed evidence summarized:

- The diagnostic compares, at the same saved `delta_k`,
  `A^T b + A^T A delta_k` versus `A^T A delta_k`.
- For `loss1_original_pgd`: at `k=5`, cosine `0.999347`, angle `1.95 deg`,
  `||A^T b|| / ||A^T A delta_k|| = 0.0347`; at `k=50`, cosine `0.999915`,
  angle `0.71 deg`, ratio `0.0125`.
- For `loss2_original_pgd`: at `k=5`, cosine `0.999345`, angle `1.97 deg`,
  ratio `0.0354`; at `k=50`, cosine `0.999926`, angle `0.67 deg`, ratio
  `0.0121`.
- For `loss3_original_pgd`: at `k=5`, cosine `0.930281`, angle `21.07 deg`,
  ratio `0.8561`; at `k=10`, angle `11.28 deg`, ratio `0.3738`; at `k=25`,
  angle `4.22 deg`, ratio `0.1175`; at `k=50`, cosine `0.999478`, angle
  `1.40 deg`, ratio `0.0321`.
- `adv_robust/bin/python -m py_compile tools/analyze_same_delta_local_gradient_angles.py`
  passed.

Inference:

- This is the apples-to-apples same-current-perturbation comparison requested by
  the user.
- Along `loss1` and `loss2` trajectories, the clean-residual term is already small
  relative to `A^T A delta_k` by `k=5`, so endpoint and movement local squared
  gradients are nearly aligned.
- Along the `loss3` trajectory, the clean-residual term matters strongly early
  because the perturbation is small; its influence decays as `delta_k` grows and
  `A^T A delta_k` dominates.
- The result supports the corrected layered interpretation: `A^T b` is important
  as a zero-radius / early-step term, not as a claim about a different final
  finite-radius optimizer direction.

Remaining work:

- Optional: visualize same-delta gradient angle and norm-ratio curves over `k`.


## 2026-05-15 Outward-Growth Angle Coverage / Rerun Need Inspected

Status:

- Inspected existing outward-growth angle records and added
  `Existing Angle Records And Whether A Rerun Is Needed` to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- Existing angle table:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`.
- Existing saved attack trajectories:
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`.
- Existing Jacobian source:
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/error/*_jacobian_svd.npz`.
- `all_direction_similarity_table.csv` already records `dot`, `abs_dot`,
  `angle_deg`, `subspace_projection_l2`, `max_abs_dot`, and `mean_abs_dot`
  between `outward_growth` and the top singular directions/subspaces of
  `fno`, `solver`, and `error`.
- Observed summaries from that table: `outward_growth` vs `error` top-8 angle
  mean `84.68 deg`, abs-dot mean `0.246550`; vs `error` rank-1 angle mean
  `90.87 deg`, abs-dot mean `0.125291`; `error_top8_subspace` projection mean
  `0.925912`.

Inference:

- The original outward-growth run does not need to be rerun to recover existing
  candidate-direction angle data; those records already exist.
- The original run did not record same-`delta_k` gradient comparisons such as
  `cos(A^T b + A^T A delta_k, A^T A delta_k)`,
  `||A^T b|| / ||A^T A delta_k||`, or
  `cos(A^T b, A^T A delta_k)`.
- The precise apples-to-apples optimizer-step question needs a new post-processing
  diagnostic using saved attack trajectories and existing Jacobians, not a full
  rerun of the expensive attack/Jacobian experiment.

Remaining work:

- Implement and run the same-`delta_k` gradient-angle post-processing diagnostic
  if this comparison is needed for the paper narrative.


## 2026-05-15 Outward-Growth Correction: Not Apples-to-Apples Optimizer Comparison

Status:

- Added `Correction: This Is Not An Apples-To-Apples Optimizer Comparison` to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- The existing outward-growth experiment compares `error_top`, a top singular-vector
  direction of `A=J_f-J_j`, with `outward_growth`, the clean-point first-order
  direction `A^T b` for the local endpoint objective.
- Therefore it mixes two levels: the final/global direction of the pure local
  residual-movement quadratic and the zero-radius first-step direction of the
  endpoint objective.

Inference:

- The current experiment should be interpreted only as a local diagnostic showing
  that residual movement `||A v||` and zero-radius clean-residual outward growth
  `<b/||b||, A v>` are different quantities.
- It should not be described as a fair comparison of two final optimizer directions
  or two same-iterate gradient directions.
- Apples-to-apples follow-ups are: same-`delta_k` gradient comparison
  `A^T b + A^T A delta_k` versus `A^T A delta_k`; finite-radius local affine
  optimizer comparison for `||A delta||^2` versus `||b + A delta||^2`; and true
  nonlinear optimizer comparison from matched starts.

Remaining work:

- Optional: implement the same-`delta_k` gradient diagnostic using saved attack
  trajectories.


## 2026-05-15 Outward-Growth Number-Comparison Clarification

Status:

- Added `Are These Number Comparisons Valid?` to
  `docs/outward_growth_direction_result_20260515.md`.
- Rechecked the source CSVs to clarify the exact meaning of the headline numbers.

Observed evidence summarized:

- Source aggregate table:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/aggregate_direction_response_summary.csv`.
- Source per-direction table:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_response_table.csv`.
- `error_top` aggregate row has `n_rows=40`, i.e. 5 samples times top-8
  right singular directions of `A=J_f-J_j`; it is a top-8 group mean, not a
  rank-1-only mean.
- `outward_growth` aggregate row has `n_rows=5`, i.e. one `A^T b` direction per
  sample.
- Observed top-8 group means: `error_top` mismatch gain `0.412169`, outward
  component `0.0140571`; `outward_growth` mismatch gain `0.368053`, outward
  component `0.166514`.
- Rank-1-only check from `all_direction_response_table.csv`: `error` rank-1
  mismatch gain mean `0.868217`, outward component mean `-0.00677836`; `fno`
  rank-1 mismatch gain mean `0.674167`, outward component mean `0.00326582`;
  `solver` rank-1 mismatch gain mean `0.680574`, outward component mean
  `0.00304161`.

Inference:

- It is valid to compare `mismatch_gain` values with other `mismatch_gain`
  values and `outward_component` values with other `outward_component` values.
- It is not valid to infer large outward clean-error growth from large
  `mismatch_gain` alone.
- The rank-1-only check strengthens the interpretation: the strongest residual
  movement direction has even larger `||A v||` but does not push the current
  residual outward on average.

Remaining work:

- Optional same-`delta_k` gradient comparison remains separate.


## 2026-05-15 Outward-Growth Layered Theory/Data Interpretation Added

Status:

- Added `Layered Interpretation: What Is Actually Being Compared` to
  `docs/outward_growth_direction_result_20260515.md`.
- The section consolidates the theory, formulas, observed data, current
  experiment scope, and the distinction between local candidate directions,
  same-`delta` gradients, finite-radius local affine objectives, and true
  nonlinear attack trajectories.

Observed evidence summarized:

- Source result doc: `docs/outward_growth_direction_result_20260515.md`.
- Numeric sources remain the existing outward-growth outputs:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/aggregate_direction_response_summary.csv`
  and
  `forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_summary.csv`.
- Observed data recorded in the new section include:
  `error_top` mismatch gain mean `0.412169`, `error_top` outward component mean
  `0.0140571`, `outward_growth` mismatch gain mean `0.368053`,
  `outward_growth` outward component mean `0.166514`, `fno_top` outward
  component mean `0.00218766`, `solver_top` outward component mean
  `-0.00819025`, and finite-difference checks `0.167286` at `rho=1e-4` and
  `0.166668` at `rho=1e-3` versus predicted `0.166514`.

Inference:

- The current experiment is a local candidate-direction diagnostic: it compares
  the pure residual-movement/SVD direction with the clean-point outward-growth
  direction `A^T b`.
- It is not a same-`delta_k` optimizer-gradient comparison between
  `A^T b + A^T A delta_k` and `A^T A delta_k`, and it is not a final nonlinear
  `loss3_original` optimizer-direction result.
- The layered interpretation records the correct role of each object:
  `loss3_original` for finite-radius attack, `||b + A delta||` for local affine
  endpoint approximation, `A^T b` for clean-point outward growth,
  `||A delta||` and the top right singular vector of `A` for local residual
  movement, and same-iterate gradient cosines as a separate follow-up.

Remaining work:

- Optional follow-up: compute same-trajectory gradient diagnostics from saved
  attack iterates `delta_k`, including
  `cos(A^T b + A^T A delta_k, A^T A delta_k)`,
  `||A^T b|| / ||A^T A delta_k||`, and
  `cos(A^T b, A^T A delta_k)`.


## 2026-05-15 Outward-Growth Limitation: Not Same-Delta Gradient Comparison

Status:

- Added `Important Limitation: Not A Same-Delta Gradient Comparison` to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- Inspection of `tools/analyze_outward_growth_direction.py` confirms that
  `error_top` directions are loaded from saved right singular vectors of
  `A = J_f - J_j`.
- The same script constructs `outward_growth` from the clean residual direction,
  i.e. normalized `A^T b` / clean outward-growth direction.
- The script evaluates candidate directions by metrics such as `||A v||` and
  `<b/||b||, A v>`; it does not compute same-`delta_k` gradient comparisons
  between `A^T b + A^T A delta_k` and `A^T A delta_k`.

Inference:

- The outward-growth experiment compares local mechanism directions:
  top residual-movement/SVD direction versus clean-point first-order outward
  direction.
- It should not be described as a comparison of two per-step optimizer gradients
  along the same attack trajectory.
- A direct same-`delta` follow-up would compute cosines and norm ratios for
  `A^T b + A^T A delta_k` versus `A^T A delta_k` at saved attack iterates.

Remaining work:

- Optional follow-up: use saved attack iterates to compute same-`delta_k`
  gradient comparisons for the local squared endpoint and residual-movement
  objectives.


## 2026-05-15 Outward-Growth Direction Type / Analytic Solution Clarification

Status:

- Added a section to `docs/outward_growth_direction_result_20260515.md` clarifying
  that the outward-growth experiment compares local analytic directions, not final
  nonlinear attack trajectories.

Observed evidence summarized:

- The computed `error_top` direction is the top right singular direction of
  `A = J_f - J_j`, i.e. the analytic local L2 solution of
  `max_{||v||_2=1} ||A v||_2`.
- The computed `outward_growth` direction is the normalized `A^T b` direction,
  i.e. the analytic first-order L2 solution of maximizing
  `<b/||b||, A v>` at `delta=0`.

Inference:

- The experiment compares local mechanism directions: pure residual movement
  versus current-clean-residual outward growth.
- For the affine finite-radius local endpoint problem
  `max_{||delta||_2 <= epsilon} ||b + A delta||_2^2`, one can write a KKT/eigen
  characterization using `Q=A^T A` and `c=A^T b`, but the answer is generally
  neither exactly the top singular vector nor exactly `A^T b`.
- For the true nonlinear neural-operator attack objective, there is no general
  closed-form final optimizer direction; PGD/LP-steepest are iterative methods.

Remaining work:

- None for this clarification.


## 2026-05-15 Outward-Growth Interpretation: Bias Term Versus Residual Movement

Status:

- Added a clarification section to `docs/outward_growth_direction_result_20260515.md`
  explaining why the outward-growth result supports keeping the clean residual
  bias term in `loss3_original` rather than replacing the objective by pure
  residual movement.

Observed evidence summarized:

- From the outward-growth result table, `error_top` has mismatch gain mean
  `0.412169` but outward component mean only `0.0140571`.
- `outward_growth` has mismatch gain mean `0.368053` but outward component mean
  `0.166514`.
- The finite-difference check measured `loss3_original` growth `0.167286` at
  `rho=1e-4` and `0.166668` at `rho=1e-3`, matching predicted outward growth
  `0.166514`.

Inference:

- Locally, `loss3_original(delta) ~= ||b + A delta||`, while the residual
  movement objective is `||e(x+delta)-e(x)|| ~= ||A delta||`.
- The residual movement objective removes the clean residual `b`, so its SVD/top
  singular-vector direction can maximize `||A v||` without strongly increasing
  the current clean error norm.
- The result supports using `loss3_original` as the primary regression attack
  target and using residual movement/SVD directions as local diagnostics only.
- This remains a local first-order conclusion; finite-radius `loss3_original`
  still requires iterative nonlinear optimization.

Remaining work:

- None for this clarification.


## 2026-05-15 Delta/Loss Formula Taxonomy Across Markdown

Status:

- Created `docs/delta_loss_formula_taxonomy_20260515.md` to summarize the
  attack-related formulas, losses, objective variants, and meanings of `delta`
  across repository Markdown notes.

Source files / commands:

- Markdown source discovery used `rg --files -g '*.md'`.
- Formula/loss/delta evidence search used
  `rg -n --glob '*.md' '(delta|Delta|\\delta|\\Delta|epsilon|\\epsilon|loss1|loss2|loss3|increment_ratio|regularized|A\^T|A\\delta|finite-radius|local)'`.
- Main source notes recorded in the dedicated doc include
  `three_loss_objective_experiment_plan.md`,
  `docs/loss3_original_theory_experiment_plan.md`,
  `BATCH_LOSS_ONLY_OPTIMIZATION_METHODS.md`,
  `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md`,
  `LOSS1_ZERO_DELTA_GRADIENT_CHECK.md`,
  `LP_STEEPEST_DIRECTION_CHECK.md`,
  `docs/local_jacobian_svd_direction_taxonomy_20260515.md`,
  `docs/loss_gradient_direction_vs_svd_direction_20260515.md`,
  `docs/outward_growth_direction_result_20260515.md`, and the FNO
  `nu=0.001` loss-gradient path result docs.

Output files:

- `docs/delta_loss_formula_taxonomy_20260515.md`

Observed evidence summarized:

- The Markdown notes repeatedly define the three base losses
  `loss1`, `loss2`, and `loss3`, plus objective variants `original`,
  `increment_ratio`, and `regularized`.
- The notes define local Jacobian formulas such as
  `Delta f ~= J_f delta`, `Delta j ~= J_j delta`, and
  `e(x + delta) ~= b + (J_f - J_j) delta`.
- The notes also define finite-radius attack formulas such as
  `max_{||delta||_p <= epsilon} O(delta)`, PGD / LP-steepest updates,
  ray profiles, and boundary-rescaled final deltas.
- `git status --short` currently shows many deleted tracked generated artifacts
  under `benchmark_results/`, `fno_training_runs/`, `gradient_audit/`,
  `path_audit/`, and `results/`. The taxonomy does not interpret those deleted
  artifacts as current local evidence.

Inference:

- `delta` has two different roles in the notes: an infinitesimal/local diagnostic
  variable for Jacobians, SVD directions, residual increment ratios, and
  outward-growth derivatives; and a finite adversarial perturbation for PGD,
  LP-steepest PGD, final deltas, boundary rescaling, and real `loss3_original`
  attack evaluation.
- The practical rule recorded in the doc is: formulas involving clean-input
  Jacobians or `delta -> 0` are local; formulas involving `||delta|| <= epsilon`,
  attack iterates, final deltas, or ray endpoints are finite-radius and should
  not drop higher-order/path effects.

Remaining work:

- None for this summary note. If future Markdown files introduce new objectives
  or solver-gradient conventions, update `docs/delta_loss_formula_taxonomy_20260515.md`.


## 2026-05-15 Squared-Loss Gradient Clarification Added

Status:

- Added a clarification section to `docs/outward_growth_direction_result_20260515.md`
  explaining the exact squared local endpoint-error gradient.

Inference recorded:

- For residual movement, `M(delta)=||A delta||^2` has gradient
  `2 A^T A delta`.
- For squared endpoint error, `S(delta)=||b + A delta||^2` expands exactly as
  `||b||^2 + 2 b^T A delta + delta^T A^T A delta`, with exact local-model
  gradient `2 A^T b + 2 A^T A delta`.
- `A^T b` is the first-step / infinitesimal-radius gradient at `delta=0`, not a
  full finite-radius replacement for `loss3_original`.


## 2026-05-15 Outward-Growth Metric Glossary Added

Status:

- Added a glossary for the four key outward-growth numbers to
  `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- `0.412169`: `error_top` mismatch gain mean, i.e. mean `||A v||_2`.
- `0.368053`: `outward_growth` mismatch gain mean, i.e. mean
  `||A v_growth||_2`.
- `0.0140571`: `error_top` outward component mean, i.e. mean
  `<b/||b||, A v>`.
- `0.166514`: `outward_growth` outward component mean, i.e. mean
  `<b/||b||, A v_growth>`.

Inference:

- The first pair compares raw residual movement; the second pair compares local
  growth of the current clean error norm.


## 2026-05-15 Outward-Growth Direction Interpretation

Status:

- Interpretation of the completed FNO `nu=0.001` outward-growth experiment was
  added to `docs/outward_growth_direction_result_20260515.md`.

Observed evidence summarized:

- `outward_growth` outward component mean: `0.166514`.
- `error_top` mismatch-gain mean: `0.412169`, but outward component mean:
  `0.0140571`.
- `fno_top` outward component mean: `0.00218766`; `solver_top` outward component
  mean: `-0.00819025`; `random_best_by_outward_component` outward component
  mean: `0.0114951`.
- Finite-difference actual `loss3` growth for `outward_growth`: `0.167286` at
  `rho=1e-4` and `0.166668` at `rho=1e-3`, versus linear prediction
  `0.166514`.

Inference:

- This experiment supports a local conceptual distinction, not a finite-radius
  optimality claim. It shows that maximizing residual movement `||A v||` and
  maximizing first-order outward growth of the current error norm are different
  diagnostics.
- The completed `A^T b` row explains local outward clean-risk growth; the final
  finite-radius adversarial objective remains `loss3_original` endpoint error.


## 2026-05-15 FNO nu=0.001 Outward-Growth Direction Completed

Status:

- Completed on GPU for FNO `nu=0.001`, indices `0 7 40 47 115`.
- Dedicated result note updated: `docs/outward_growth_direction_result_20260515.md`.

Observed output files:

- `forensics/outward_growth_direction_20260515/fno_nu0p001/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/summary.md`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_response_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/aggregate_direction_response_summary.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_table.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/finite_difference_growth_summary.csv`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/manifest.json`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_model_output.npy`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_solver_output.npy`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/clean_residual.npy`
- `forensics/outward_growth_direction_20260515/fno_nu0p001/index_*/outward_growth_direction.npy`

Observed key metrics:

- From `aggregate_direction_response_summary.csv`: `outward_growth` outward
  component mean `0.166514`, mismatch-gain mean `0.368053`, and `D_f` mean
  `0.355948`.
- From `aggregate_direction_response_summary.csv`: `error_top` mismatch-gain
  mean `0.412169` but outward component mean only `0.0140571`.
- From `aggregate_direction_response_summary.csv`: `fno_top` outward component
  mean `0.00218766`; `solver_top` outward component mean `-0.00819025`;
  `random_best_by_outward_component` outward component mean `0.0114951`.
- From `finite_difference_growth_summary.csv`: for `outward_growth`, actual
  loss3 growth mean is `0.167286` at `rho=1e-4`, `0.166668` at `rho=1e-3`, and
  `0.167548` at `rho=1e-2`, versus predicted `0.166514`.
- From `finite_difference_growth_summary.csv`: for `negative_outward_growth`,
  actual loss3 growth mean is approximately the opposite sign, e.g. `-0.165528`
  at `rho=1e-4`, versus predicted `-0.166514`.
- From `manifest.json`: all five samples completed with nondegenerate outward
  directions; per-index clean residual norms are recorded in `per_index_metadata`.

Inference:

- The missing Experiment 2 `A^T b` / `v_growth*` row is completed for FNO
  `nu=0.001`.
- The result separates residual movement from outward clean-risk growth: the
  `error_top` directions move the residual field more, but `outward_growth` is
  the direction that most increases the current clean error norm at first order.
- Small-radius finite differences validate the local linear prediction for the
  outward-growth direction.

Remaining work:

- Use these tables in the Experiment 2 writeup.
- Do not rerun FNO `nu=0.01` or DeepONet/default-net unless a later paper
  question specifically requires the broader architecture comparison.


## 2026-05-15 FNO nu=0.001 Outward-Growth Run Status Check

Status:

- User-launched production command is running on GPU.
- No process was stopped or restarted during this check.

Observed evidence:

- Running process observed via `ps`: PID `192081`, command
  `adv_robust/bin/python tools/analyze_outward_growth_direction.py ... --device cuda`.
- Output root exists: `forensics/outward_growth_direction_20260515/fno_nu0p001/`.
- `index_000/manifest.json` exists and records
  `clean_residual_norm_l2 = 0.33239443448801387`,
  `At_b_unit_norm_l2 = 0.20492601962861828`,
  `error_jacobian_consistency_l2 = 1.8676534473603818e-08`, and
  `seconds = 19.056603444973007`.
- `index_000/finite_difference_growth_table.csv` had 85 lines, matching header
  plus 84 finite-difference rows for 28 directions x 3 radii.
- `index_007/finite_difference_growth_table.csv` also had 85 lines at the time
  of the check, indicating the second sample's finite-difference table had been
  written or nearly completed.

Inference:

- The startup CUDA/JAX messages are not fatal for this run; output is being
  produced on GPU.
- The production command is progressing past index 0.

Remaining work:

- Let the command finish through indices `40`, `47`, and `115`.
- After completion, inspect `manifest.json`, aggregate CSVs, `summary.md`, and
  `docs/outward_growth_direction_result_20260515.md`, then update this ledger
  with final observed metrics and conclusions.


## 2026-05-15 FNO nu=0.001 Outward-Growth Script Prepared

Source files added/updated:

- `tools/analyze_outward_growth_direction.py`
- `docs/outward_growth_direction_experiment_plan_20260515.md`

Status:

- Script prepared for the FNO `nu=0.001` outward-growth direction experiment.
- Production five-index experiment has not been run in this turn.
- GPU smoke tests were run only to validate the script path.

Observed from GPU smoke validation:

- Command path used `--device cuda` and the script prepended the virtualenv
  `ptxas` directory before constructing the JAX solver.
- Smoke output directories: `/tmp/outward_growth_smoke` and
  `/tmp/outward_growth_fd_smoke`.
- `/tmp/outward_growth_fd_smoke/manifest.json` records source paths for the FNO
  checkpoint, Burgers test set, and index-0 Jacobian/SVD files.
- Index 0 smoke values, observed from `/tmp/outward_growth_fd_smoke/manifest.json`:
  `clean_residual_norm_l2 = 0.33239443448801387`,
  `At_b_unit_norm_l2 = 0.20492601962861828`,
  `error_jacobian_consistency_l2 = 1.8676534473603818e-08`.
- `/tmp/outward_growth_fd_smoke/all_direction_response_table.csv` includes rows
  for `fno`, `solver`, `error`, `outward_growth`,
  `negative_outward_growth`, random directions, and random-best controls.
- `/tmp/outward_growth_fd_smoke/finite_difference_growth_table.csv` includes
  actual/predicted loss3 growth, residual movement, model movement, solver
  movement, and response cosine columns.

Inference:

- The script is ready to run the requested FNO `nu=0.001` five-index
  experiment on GPU.
- The `ptxas` workaround keeps the run on GPU; it only changes PATH so JAX does
  not use the incompatible system CUDA assembler.

Remaining work:

- Run the full command over indices `0 7 40 47 115`.
- Inspect `forensics/outward_growth_direction_20260515/fno_nu0p001/summary.md`,
  CSVs, JSON manifests, and `docs/outward_growth_direction_result_20260515.md`
  after completion.
- Update this ledger with final observed metrics and conclusions after the full
  run.


## 2026-05-15 Outward-Growth Direction Experiment Plan

Plan note added:

- `docs/outward_growth_direction_experiment_plan_20260515.md`

Status:

- Experiment plan only.
- No numerical experiment was run.

Observed evidence used to create the plan:

- `docs/loss3_original_theory_experiment_plan.md` defines the missing
  `v_growth*` row in Experiment 2.
- `docs/local_jacobian_svd_direction_taxonomy_20260515.md` records the local
  `loss3_original` decomposition into an outward term `A^T b` and quadratic
  gain term `A^T A delta`.
- Existing direction-comovement tables such as
  `forensics/fno_solver_jacobian_similarity_20260514/all_direction_comovement_table.csv`
  already include model, solver, error, and random direction rows.
- Existing reusable Jacobian/SVD sources include
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`,
  `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`, and
  `forensics/deeponet_solver_jacobian_similarity_20260515/`.

Inference:

- The missing Experiment 2 component should be a focused postprocess that loads
  existing Jacobians, computes `v_growth = normalize((J_f-J_j)^T b/||b||)`, and
  adds outward-growth rows to the same response table structure.
- This should not be treated as a finite-radius attack. It is a local
  first-order diagnostic separating residual movement `||A v||` from outward
  error-norm growth `<b/||b||, A v>`.

Remaining work:

- Implement `tools/analyze_outward_growth_direction.py` or equivalent.
- Run the FNO `nu=0.001` phase first, then FNO `nu=0.01` and DeepONet
  `nu=0.01`.
- Add `docs/outward_growth_direction_result_20260515.md` after running, with
  exact source paths, output tables, observed metrics, and conclusions.


## 2026-05-15 R2 Loss3 Original Plan Completion Audit

Audit note added:

- `docs/loss3_original_plan_r2_completion_audit_20260515.md`

Status:

- Remote R2 documentation/results audit completed.
- No numerical experiment was run.
- R2 credentials were used for read-only listing / small JSON reads and were not
  written to repository files.

Observed evidence from R2:

- R2 contains `docs/main_objective_mechanism_experiment1_result_20260514.md`,
  `results/main_objective_mechanism_summary_20260514/`, and
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`,
  supporting Experiment 1.
- R2 contains local Jacobian/SVD forensics including
  `forensics/fno_solver_jacobian_similarity_20260514/`,
  `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`,
  `forensics/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`,
  and
  `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/`,
  supporting most of Experiment 2.
- R2 contains `results/three_loss_objective_round1_l2_eps8_alpha0p3/`, with the
  27-run grid for three losses x three objective variants x three optimizers.
- R2 contains six `results/three_loss_batch100_full_loss3_delta_rerun_20260514_*_final_boundary/`
  directories and many `final_delta_summary.json`,
  `final_delta_diagnostics.csv`, and `final_delta_diagnostics.npz` files, so
  Experiment 5 Boundary-Rescaled Comparison is completed on R2.
- Selected R2 JSON values for
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`:
  `loss3_original_pgd` boundary `loss3_original_mean = 5.4525`;
  `loss3_increment_ratio_pgd` boundary `loss3_original_mean = 4.2225`;
  `loss3_regularized_pgd` final `||delta||_2` mean `0.3018`, boundary
  `loss3_original_mean = 2.0995`; `loss2_increment_ratio_pgd` boundary
  `loss3_original_mean = 3.1936`; `loss2_regularized_pgd` boundary
  `loss3_original_mean = 3.6531`.

Inference:

- The R2 evidence updates the prior local-only audit: Boundary-Rescaled
  Comparison was completed historically and exists on R2, even though the
  current local working tree lacks those result directories.
- The boundary-rescaled results support the conclusion that increment-ratio and
  regularized objectives can find interior or locally efficient directions, but
  scaling those directions to the full epsilon boundary does not necessarily
  match direct `loss3_original` endpoint optimization. This supports the
  nonlinear local-to-global interpretation.
- Still not observed as completed on R2 as specified: Ray Profile /
  Local-to-Global Profile, planned Small-Epsilon Sweep with `L_f`, `L_j`,
  `L_e`, `G_e`, and exact Direction Rotation Along Path via recomputed
  `v_e*(x_t)`.

Remaining work:

- Run Ray Profile curves.
- Run the planned local Small-Epsilon Sweep.
- Run exact path direction rotation using `v_e*(x_t)`.
- Optionally add `A^T b` outward-growth direction to the local response table.
- Optionally make a compact paper-ready table from the R2 boundary JSON files.


## 2026-05-15 Loss3 Original Plan Completion Audit

Audit note added:

- `docs/loss3_original_plan_completion_audit_20260515.md`

Status:

- Documentation/results audit completed.
- No numerical experiment was run.

Observed evidence:

- `docs/main_objective_mechanism_experiment1_result_20260514.md` documents
  Experiment 1: Main Objective Comparison for FNO/Burgers `nu=0.001`, batch
  100, `epsilon=8`, `alpha=0.3`, `steps=100`, with PGD, LP-steepest PGD, and
  generalized power. It records model movement, solver movement, mismatch,
  final true error, response cosine, `D_f`, and `D_sym`.
- Current local Jacobian/SVD outputs exist under paths including
  `forensics/fno_solver_jacobian_similarity_20260514/`,
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`, and
  `forensics/deeponet_solver_jacobian_similarity_20260515/`. These support the
  local response decomposition for top model, solver, error, and random
  directions, but not the explicit outward-growth `A^T b` direction.
- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md` and
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary.md`
  provide related path-gradient evidence, but not the planned recomputation of
  `v_e*(x_t)` along the path.
- `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md` documents scripts and final-boundary
  diagnostic design, and related scripts exist under `tools/`, but no current
  local `results/three_loss_batch100_full_loss3*` directories or
  `final_delta_summary.json` files were found during this audit.
- `docs/ray_profile_markdown_lookup_20260515.md` is only a lookup note for the
  Ray Profile section, not a Ray Profile result.
- `results/burgers_loss3_clean_recomputed_summary.md` contains older
  small-epsilon attack rows, but not the planned local small-epsilon sweep over
  `{1e-4, 1e-3, 1e-2, 1e-1}` with `L_f`, `L_j`, `L_e`, `G_e`, and direction
  cosine stability.

Inference:

- Completed or strongest documented evidence: Experiment 1 and most of
  Experiment 2's SVD/Jacobian response decomposition.
- Partial or related evidence: three-loss variant/boundary diagnostics in design
  form, older small-epsilon attack summaries, and loss-gradient path analysis.
- Not completed as specified: Small-Epsilon Sweep, Ray Profile,
  Boundary-Rescaled Comparison with current local numeric outputs, and
  Direction Rotation via recomputed `v_e*(x_t)`.

Remaining work:

- Restore/fetch or rerun missing three-loss boundary outputs.
- Run Ray Profile curves.
- Run Small-Epsilon Sweep with direction stability.
- Add outward-growth `A^T b` to the local response table.
- Run the full path direction-rotation experiment.

Working-tree note:

- `git status --short` showed many deleted tracked experiment artifacts under
  paths including `benchmark_results/`, `fno_training_runs/`,
  `gradient_audit/`, `path_audit/`, and `results/`. These deleted files were
  not interpreted as current local result evidence.


## 2026-05-15 English Translation of Loss3 Original Theory Plan

Files updated:

- `docs/loss3_original_theory_experiment_plan.md`
- `EXPERIMENT_LEDGER.md`

Status:

- Documentation translation completed.
- No numerical experiment was run.

Observed evidence:

- The source Markdown file existed locally at
  `docs/loss3_original_theory_experiment_plan.md` and contained the Chinese
  theory and experiment plan for `loss3_original` as the main regression attack
  objective.
- The translated file preserves the original structure: central goal, five-step
  proof route, nine-objective table, method-objective matching, six claims, six
  proposed experiments, interpretation of representative recorded results,
  recommended paper narrative, minimal experiment set, and one-sentence summary.

Inference:

- This change is a language/clarity update only. It does not add new empirical
  evidence and does not change the planned experimental conclusions.

Remaining work:

- If the plan is executed later, record the exact scripts, output directories,
  generated numeric tables, metrics, and conclusions in a separate result note.

Working-tree note:

- Before this translation, `git status --short docs/loss3_original_theory_experiment_plan.md EXPERIMENT_LEDGER.md`
  showed `EXPERIMENT_LEDGER.md` as modified and did not show the plan file as
  modified. This translation modifies the plan file and appends this ledger
  entry.

## 2026-05-15 Ray Profile Markdown Location Lookup

Lookup note added:

- `docs/ray_profile_markdown_lookup_20260515.md`

Observed from local Markdown search:

- Markdown search for whole-word `ray` / `ray profile` / `ray experiment`
  reported Ray-related Markdown hits in
  `docs/loss3_original_theory_experiment_plan.md`.
- The main hit is `Experiment 4: Ray Profile / Local-to-Global Profile` at
  `docs/loss3_original_theory_experiment_plan.md:741`.
- The compact experiment list also names `ray profile` at
  `docs/loss3_original_theory_experiment_plan.md:994`.
- `stat` reported
  `2026-05-15 15:04:45.684671694 +0000 docs/loss3_original_theory_experiment_plan.md`;
  the file was not observed locally as a May 14 file by filesystem mtime at
  lookup time.

Inference:

- The requested Markdown file is most likely
  `docs/loss3_original_theory_experiment_plan.md`.
- The Ray-profile experiment is a planned direction/radius diagnostic, not an
  observed numerical result from this lookup.

Remaining work:

- No experiment was run in this lookup.
- If executed later, record the script, output directory, numeric tables, and
  conclusions separately.

Working-tree note:

- `git status --short` showed many deleted tracked experiment artifacts under
  paths including `benchmark_results/`, `fno_training_runs/`,
  `gradient_audit/`, `path_audit/`, and `results/`. These were not interpreted
  as current local results for the Ray-profile plan.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Figures And Target-Loss Table

Added visual and tabular summaries for the saved 50-step attack trajectories.

Files added/updated:

- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_target_loss3_table_20260515.md`
- `docs/figures/fno_nu0p001_loss_gradient_path_dashboard_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_target_vs_loss3_20260515.png`
- `docs/figures/fno_nu0p001_loss_gradient_path_angles_by_attack_20260515.png`
- `tools/plot_loss_gradient_path_figures.py`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/target_vs_loss3_by_attack_loss_and_k.csv`

What the new table shows:

- Each row is averaged over the five initial conditions.
- For the `loss1` path, the table reports the optimized `loss1` value and the
  same-delta `loss3` value.
- For the `loss2` path, the table reports the optimized `loss2` value and the
  same-delta `loss3` value.
- For the `loss3` path, the table reports direct `loss3`.
- Early and middle steps are affected by unequal budget use: `loss1`/`loss2`
  often use much more L2 budget than `loss3`, so their same-delta `loss3` can be
  larger early.
- At `k=50`, direct `loss3` is largest on average: `4.6171` vs `4.0678` on the
  `loss1` path and `3.6982` on the `loss2` path.

## 2026-05-15 Loss-Gradient Direction Interpretation Clarification

Clarification added to:

- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_detailed_data_20260515.md`

Interpretation:

- The three losses do not form three equally different update directions.
- `loss1` and `loss2` are nearly parallel in this FNO `nu=0.001` run.
- `loss3` is the distinct direction: roughly `56-57 deg` away overall from
  `loss1/loss2`.
- The `loss3` direction becomes more similar as `delta_k` grows, decreasing from
  about `75 deg` at `k=5` to about `47 deg` at `k=50`, but remains clearly
  different.
- Reported angles are averages of per-point angles, not angles of averaged
  gradients.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Detailed Data Table

Detailed data note added:

- `docs/fno_nu0p001_loss_gradient_path_detailed_data_20260515.md`

Contents:

- overall 150-point gradient-angle/loss/gradient-norm summary;
- aggregation by optimized attack loss;
- aggregation by saved step `k`;
- aggregation by attack loss and saved step;
- aggregation by initial-condition index and attack loss;
- final `k=50` per-index table;
- full raw 150-row per-point CSV embedded directly in the Markdown file;
- pointer to the source CSV at
  `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/per_point_loss_gradient_angles.csv`.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Path Result

Result note added:

- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`

Postprocess script added:

- `tools/analyze_loss_gradient_path_results.py`

Analyzed run:

- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631`
- 150 points: 5 initial conditions x 3 attack losses x 10 saved steps
  (`k=5,10,...,50`).

Key result:

- `grad loss1` and `grad loss2` are nearly aligned: mean cosine `0.9904`,
  mean angle `3.91 deg`.
- `grad loss3` is substantially different from both: mean angles about
  `56.5 deg` vs `loss1` and `57.2 deg` vs `loss2`.
- The `loss3` difference is strongest early (`~75 deg` at `k=5`) and decreases
  but remains large by `k=50` (`~47 deg`).
- At final `k=50`, direct `loss3` optimization gives the largest mean final
  `loss3` (`4.617`), but not uniformly for every individual index.

## 2026-05-15 Built-In Runtime Compatibility Patch

Code updated:

- `run_three_loss_objective_attack.py`

Behavior added:

- default runtime workaround setup before model/solver construction;
- virtualenv-local Triton/NVIDIA `ptxas` is moved to the front of `PATH` when found;
- cuDNN is disabled by default for this attack script while keeping CUDA/GPU enabled;
- command-line controls are available via `--no-runtime-workarounds`,
  `--no-disable-cudnn`, and `--no-prepend-env-ptxas`;
- resolved settings are saved in `config.json` as `runtime_ptxas_dir` and
  `runtime_cudnn_enabled`.

Verification:

- `python3 -m py_compile run_three_loss_objective_attack.py`
- 1-step GPU probe with plain `python run_three_loss_objective_attack.py ...`
  succeeded without the external wrapper and saved
  `/tmp/fno_path_probe_builtin_workarounds` in about 0.86 seconds.

## 2026-05-15 Vast.ai V100 Runtime Troubleshooting

Operational note added:

- `docs/vast_v100_cuda_jax_cudnn_troubleshooting_20260515.md`

Recorded issues:

- Vast auto-tmux login can exit with `no sessions` before any experiment starts.
- Vast may auto-activate `/venv/main`, while the usable repo environment is
  `adv_robust`.
- JAX/Exponax GPU compilation can fail if it finds the system CUDA 13 `ptxas`
  before the `adv_robust` CUDA 12.4 `ptxas`.
- `run_three_loss_objective_attack.py` can fail on this V100 during FNO
  backward through cuDNN; the Jacobian scripts avoided this because they already
  disable cuDNN internally.

Stable workaround for this instance:

- activate `adv_robust`;
- use the updated `run_three_loss_objective_attack.py`, which now applies the
  ptxas PATH and no-cuDNN workarounds by default;
- the older wrapper approach is only an emergency fallback for older checkouts.

## Critical Correction

The current user question is about the FNO vs DeepONet/default-net **local
Jacobian / SVD / frequency** experiment, not the FNO-vs-DeepONet attack-ratio
tables.

Do not present the attack-ratio tables as the answer to the local-Jacobian
question. A previously created ratio-focused note was removed because it did
not answer the user's question.

## 2026-05-15 Recovery Actions

Observed from Git/GitHub:

- Branch `vast-ai` is at commit `20b689a2c385a700aa8dea91ae1b069adc6d4a77`
  (`Add loss3 mechanism diagnostics`).
- The current branch and `origin/vast-ai` point to that commit after fetch.
- `docs/`, `tools/`, `loss_attack_common.py`, DeepONet runner scripts, and
  the FNO-vs-solver forensics directory were restored from `HEAD`.
- A git-history path search did not find committed paths named
  `tools/analyze_local_jacobian_fno_deeponet.py` or
  `forensics/local_jacobian_frequency_20260514/`.

Observed from the selected R2 artifact prefix:

- The prefix contains `docs/`, `tools/`, `results/`, `forensics/`,
  `deeponet_training_runs/`, `fno_training_runs/`, datasets, and other
  artifacts.
- The selected `forensics/` prefix contains
  `fno_solver_jacobian_similarity_20260514/`.
- Searches under the selected prefix did not find
  `forensics/local_jacobian_frequency_20260514/` or
  `tools/analyze_local_jacobian_fno_deeponet.py`.
- Only small, relevant artifacts were restored/downloaded: result summaries,
  FNO-vs-solver forensics, and DeepONet checkpoints/logs for `nu=0.001` and
  `nu=0.01`.

Observed from the local Python environment:

- System `python3` does not have a usable `torch`.
- The copied `adv_robust` virtual environment has `site-packages/torch`, but
  that directory is empty; importing it yields a namespace module without
  `torch.load`.
- Therefore no new local-Jacobian recomputation has been run in this recovered
  checkout yet.


## 2026-05-15 Git Forensics For Missing Local-Jacobian Script

Observed from `origin/vast-ai`:

- Current local branch `vast-ai` tracks `origin/vast-ai` at commit
  `20b689a2c385a700aa8dea91ae1b069adc6d4a77`.
- `origin/vast-ai:tools/` contains `analyze_main_objective_mechanism.py`,
  `analyze_fno_solver_jacobian_similarity.py`, and
  `summarize_main_objective_mechanism.py`, but not
  `analyze_local_jacobian_fno_deeponet.py`.
- `git log --all --name-status -- tools` shows that commit `20b689a` added
  only those three analysis/summarization files under `tools/`.
- `git grep` in commit `20b689a` finds only indirect references to the missing
  FNO/DeepONet local-Jacobian experiment: the docs mention it, the FNO-vs-solver
  config points to `forensics/local_jacobian_frequency_20260514/...`, and the
  FNO-vs-solver script imports the missing helper.
- `git fsck --full --no-reflogs --unreachable` produced no unreachable commits
  or blobs in this checkout.

Interpretation:

- In this recovered checkout, there is no evidence that the missing helper or
  original DeepONet SVD outputs were ever committed to the visible GitHub
  branch.
- The most likely explanations are: the file/result directory existed only as
  an untracked generated artifact on the previous machine, it lived under a
  different path/name that has not been found yet, or a local commit was made on
  the previous machine but was not pushed and was not included in this recovered
  `.git` object database.

## Experiment Status

| Experiment | Status | Current evidence | What not to claim |
| --- | --- | --- | --- |
| FNO vs DeepONet/default-net local Jacobian/SVD/frequency | Partially evidenced, original outputs missing | `docs/main_objective_mechanism_experiment1_result_20260514.md` says an existing FNO/DeepONet Jacobian experiment computed `J_f`; `forensics/fno_solver_jacobian_similarity_20260514/config.json` points to `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index`; `tools/analyze_fno_solver_jacobian_similarity.py` imports the missing helper script | Do not claim the exact DeepONet high/low-frequency singular-vector conclusion from current files |
| FNO vs solver local Jacobian/SVD/frequency | Available and recorded | `docs/fno_solver_jacobian_similarity_result_20260514.md`; `forensics/fno_solver_jacobian_similarity_20260514/` | Do not confuse this with FNO vs DeepONet |
| FNO vs DeepONet/default-net attack-ratio tables | Available as old ratio evidence | `results/burgers_loss3_clean_recomputed_summary.md`; `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.md`; `.csv` | Do not use this as the Jacobian/SVD answer |
| FNO vs solver tracking-discount / mechanism diagnostics | Available and recorded | `docs/main_objective_mechanism_experiment1_result_20260514.md` and referenced mechanism summary tables | Do not use this as DeepONet evidence |

## FNO vs DeepONet Local Jacobian/SVD Evidence

Observed evidence that the experiment existed:

- `docs/main_objective_mechanism_experiment1_result_20260514.md` explicitly
  refers to an existing FNO/DeepONet Jacobian experiment.
- `forensics/fno_solver_jacobian_similarity_20260514/config.json` reuses FNO
  outputs from
  `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index`.
- `tools/analyze_fno_solver_jacobian_similarity.py` imports
  `tools.analyze_local_jacobian_fno_deeponet`, so the follow-up script depended
  on a local helper that is absent now.

Observed missing artifacts:

- `tools/analyze_local_jacobian_fno_deeponet.py`
- `forensics/local_jacobian_frequency_20260514/`
- DeepONet per-index files such as `index_*/deeponet/*jacobian_svd.npz`
- DeepONet frequency tables such as `*_top_singular_vector_metrics.csv` or
  `*_frequency_gain_by_k.csv`

Current grounded conclusion:

- The FNO-vs-DeepONet local-Jacobian experiment almost certainly existed as a
  generated/local artifact on the previous machine.
- The exact DeepONet SVD/frequency side is not currently recovered.
- The FNO side is partially recoverable from the later FNO-vs-solver forensics:
  in the recorded samples `0, 7, 40, 47, 115`, the leading FNO right singular
  vectors are low-frequency dominated.

## Restored / Downloaded Small Artifacts

Restored from git `HEAD`:

- `docs/main_objective_mechanism_experiment1_result_20260514.md`
- `docs/fno_solver_jacobian_similarity_result_20260514.md`
- `tools/analyze_fno_solver_jacobian_similarity.py`
- `forensics/fno_solver_jacobian_similarity_20260514/`
- `results/burgers_loss3_clean_recomputed_summary.md`
- `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.md`
- `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.csv`

Selected R2 downloads:

- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0.001.pt`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/config.json`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/dataset_info.json`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/summary.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/config.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/dataset_info.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/output_transform_stats.npz`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/summary.json`


## 2026-05-15 Reconstructed Missing FNO/DeepONet Local-Jacobian Helper

Status: code reconstructed and committed; one DeepONet torch import bug was
fixed during the DeepONet-vs-solver run preparation.

Added:

- `tools/analyze_local_jacobian_fno_deeponet.py`

Purpose:

- restore the helper imported by `tools/analyze_fno_solver_jacobian_similarity.py`;
- provide standalone FNO-vs-DeepONet/default-net explicit local Jacobian/SVD
  analysis;
- save per-index artifacts under
  `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/`
  with the file layout expected by the FNO-vs-solver follow-up:
  `index_*/fno/fno_index*_jacobian_svd.npz` and
  `index_*/deeponet/deeponet_index*_jacobian_svd.npz`.

Implemented interfaces used by the FNO-vs-solver script:

- `load_sample`
- `make_model`
- `compute_explicit_jacobian`
- `analyze_jacobian`
- `frequency_gains`

Verification performed in the recovered checkout:

- `python3 -m py_compile tools/analyze_local_jacobian_fno_deeponet.py tools/analyze_fno_solver_jacobian_similarity.py`
- `python3 -c "import tools.analyze_local_jacobian_fno_deeponet as m; ..."`
- a small 8x8 smoke test for `analyze_jacobian`, confirming NPZ/CSV/JSON
  outputs are written.

Not yet run in this recovered checkout:

- full 1024 x 1024 FNO-vs-DeepONet recomputation, because the copied Python
  environment initially had an empty/broken `torch` package. The environment
  was later patched enough to run the DeepONet-vs-solver Jacobian experiment
  below, but the full FNO-vs-DeepONet recomputation remains separate.

## 2026-05-15 DeepONet vs Solver Local Jacobian/SVD/Frequency

Status: completed and recorded.

Purpose:

- Recreate the FNO-vs-solver local-Jacobian comparison for DeepONet/default-net.
- Use the same five initial conditions as the FNO-vs-solver run:
  `0, 7, 40, 47, 115`.
- Use the DeepONet model setting `nu=0.01` instead of the FNO-side `nu=0.001`.

Added / generated:

- `tools/analyze_deeponet_solver_jacobian_similarity.py`
- `forensics/deeponet_solver_jacobian_similarity_20260515/`
- `docs/deeponet_solver_jacobian_similarity_result_20260515.md`

Observed setting from the run config:

- DeepONet checkpoint:
  `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- DeepONet output transform:
  `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/output_transform_stats.npz`
- Test split:
  `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/..._test.pt`
- Solver: JAX/Exponax Burgers, `nx=1024`, `nu=0.01`, `t_final=1.0`,
  `dt=0.001`, `domain=2.0`, solver dtype `float64`.

Key observed results:

- DeepONet spectral norm mean: `7.202`; solver spectral norm mean: `1.373`;
  residual/error spectral norm mean: `7.100`.
- DeepONet top-1 right singular vectors are high-frequency:
  `hi128` mean `0.761`, zero crossings mean `507.2`.
- Solver top-1 right singular vectors are low-frequency/smooth:
  `hi128` mean `1.70e-17`, zero crossings mean `0.0`.
- Error top-1 right singular vectors track DeepONet:
  `hi128` mean `0.803`, zero crossings mean `516.0`.
- DeepONet-vs-solver top-k right subspaces are nearly orthogonal:
  mean principal cosine at `k=1` is `0.125`, and at `k=8` is `0.170`.
- DeepONet-vs-error leading subspaces are almost identical:
  mean principal cosine at `k=1` is `0.985`, and at `k=8` is `0.981`.

Grounded conclusion:

- For this `nu=0.01` DeepONet/default-net model, the dominant local model
  directions are high-frequency and jagged, while the physical solver's
  dominant local directions are smooth/low-frequency.
- The dominant residual/error Jacobian is essentially the DeepONet
  high-frequency response that the solver does not share.

## 2026-05-15 Top-4 Singular Vector Shape/Fourier Comparison

Status: completed and recorded.

Purpose:

- Directly visualize the top 4 right singular vectors rather than only their
  scalar frequency metrics.
- Compare FNO `nu=0.001`, solver `nu=0.001`, DeepONet `nu=0.01`, and solver
  `nu=0.01` on the same sample indices `0, 7, 40, 47, 115`.

Added / generated:

- `tools/plot_top_singular_vector_comparison.py`
- `forensics/top_singular_vector_comparison_20260515/`
- `docs/top_singular_vector_comparison_result_20260515.md`

Intermediate local-only raw SVD artifacts generated for plotting:

- `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/`
  for FNO top singular vectors.
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/` for the
  `nu=0.001` solver top singular vectors.

Grounded conclusion:

- FNO top right singular vectors and the `nu=0.001` solver top right singular
  vectors are smooth/low-frequency.
- DeepONet `nu=0.01` top right singular vectors are visibly high-frequency and
  jagged.
- The `nu=0.01` solver top right singular vectors remain smooth/low-frequency.
- Therefore the direct plots support the numeric conclusion: FNO's dominant
  local modes resemble solver modes, while DeepONet's dominant local modes are
  high-frequency modes that the solver does not share.

## 2026-05-15 Comprehensive SVD Shape/Fourier/Angle Diagnostics

Status: completed and recorded.

Purpose:

- Produce the dense plot set requested for comparing FNO, solver, DeepONet,
  and error Jacobian SVDs.
- Plot top-8 right singular vectors for each of the five initial-condition
  indices `0, 7, 40, 47, 115`.
- Plot both per-index figures and aggregate figures averaged across the five
  indices after Fourier transform.
- Compute cross-operator pairwise vector cosines, top-k principal angles,
  singular value spectra, and within-SVD orthogonality checks.

Added / generated:

- `tools/plot_comprehensive_svd_diagnostics.py`
- `forensics/comprehensive_svd_diagnostics_20260515/`
- `docs/comprehensive_svd_diagnostics_result_20260515.md`

Generated plot families:

- per-index six-row top-8 right-singular-vector line grids;
- per-index six-row top-8 Fourier-energy grids;
- per-index pairwise right-vector cosine heatmaps;
- per-index top-k principal-angle curves;
- per-index singular-value spectra;
- per-index within-SVD orthogonality-error heatmaps;
- aggregate mean shape, FFT, cosine-heatmap, principal-angle, singular-spectrum,
  and orthogonality figures.

Key observed results:

- FNO model-vs-solver leading right-singular subspaces are close:
  `k=1` mean principal angle `4.86 deg`, `k=8` mean principal angle `7.01 deg`.
- DeepONet model-vs-solver leading right-singular subspaces are far apart:
  `k=1` mean principal angle `82.83 deg`, `k=8` mean principal angle
  `80.19 deg`.
- DeepONet model-vs-error leading right-singular subspaces are close:
  `k=1` mean principal angle `10.01 deg`.
- FNO model top-1 right singular vector is low-frequency:
  mean `hi128` `2.23e-07`, mean zero crossings `13.6`.
- DeepONet model top-1 right singular vector is high-frequency:
  mean `hi128` `0.761`, mean zero crossings `507.2`.
- DeepONet error top-1 right singular vector is also high-frequency:
  mean `hi128` `0.803`, mean zero crossings `516.0`.

Orthogonality check:

- Within a single SVD, top-8 right singular vectors are mutually orthogonal up
  to numerical error. Max off-diagonal Gram errors are about `1e-09` to
  `3e-09`.
- This is expected from SVD and is mainly a sanity check. The scientifically
  useful comparisons are cross-operator vector/subspace alignments, especially
  model-vs-solver and model-vs-error principal angles.

Grounded conclusion:

- FNO's dominant local right-singular directions resemble the solver's
  low-frequency directions.
- DeepONet's dominant local right-singular directions are high-frequency and
  resemble the DeepONet-vs-solver error directions, not the solver directions.

No-std plot variant:

- `tools/plot_comprehensive_svd_diagnostics.py` now supports
  `--no-std-shading`.
- A second copy of the plot set was generated under
  `forensics/comprehensive_svd_diagnostics_20260515_no_std/` with the same
  samples and metrics but without standard-deviation shading in aggregate line
  plots.


## 2026-05-15 FNO nu=0.01 vs Solver and DeepONet nu=0.01 SVD Diagnostics

Status: completed and recorded.

Purpose:

- Repeat the local Jacobian/SVD analysis for FNO at matched `nu=0.01`.
- Compare FNO `nu=0.01`, solver `nu=0.01`, DeepONet/default-net `nu=0.01`, and
  their error Jacobians on the same five sample indices `0, 7, 40, 47, 115`.
- Regenerate the dense line/FFT/heatmap plot set without standard-deviation
  shading.

Recovered input:

- FNO `nu=0.01` checkpoint restored from R2 selected backup:
  `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`.
- The older `tmp_old_runner_inputs_b01/...nu0.01...pth` file was not present and
  was not tracked by Git because `*.pth` is ignored.

Added / generated:

- `tools/analyze_fno_solver_jacobian_similarity.py` now supports
  `--reuse-solver` and `--reuse-solver-root`.
- `tools/plot_fno_deeponet_nu0p01_svd_diagnostics.py`
- `docs/fno_nu0p01_solver_jacobian_status_20260515.md`
- `docs/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_result_20260515.md`
- `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`
- `forensics/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`

Key observed results:

- FNO `nu=0.01` model and solver are very tightly aligned locally:
  model-vs-solver `k=1` mean principal angle `2.32 deg`, `k=8` mean principal
  angle `1.82 deg`, and model-direction response cosine mean `0.9993`.
- FNO `nu=0.01` model top-1 right singular vector is extremely low-frequency:
  mean `hi128` `2.82e-09`, mean zero crossings `0.40`.
- FNO `nu=0.01` residual/error Jacobian is small in spectral norm:
  model top-1 sigma `1.379`, solver top-1 sigma `1.373`, error top-1 sigma
  `0.0373`.
- DeepONet/default-net `nu=0.01` remains high-frequency and solver-misaligned:
  model-vs-solver `k=1` mean principal angle `82.83 deg`, model top-1 `hi128`
  `0.7614`, and model top-1 zero crossings `507.20`.
- DeepONet/default-net error remains close to the model subspace:
  model-vs-error `k=1` mean principal angle `10.01 deg`; FNO model-vs-error
  `k=1` mean principal angle is `62.53 deg`.

Grounded conclusion:

- At matched `nu=0.01`, FNO behaves even more solver-like than in the earlier
  mixed-`nu` comparison. Its model and solver Jacobians have almost the same
  leading singular values and nearly the same leading right-singular subspace.
- DeepONet/default-net is still dominated by high-frequency, solver-misaligned
  local directions, so its error directions are close to the model directions.


## 2026-05-15 Nine-Row FNO/DeepONet Local SVD Comparison

Status: completed and recorded.

Purpose:

- Put FNO `nu=0.001`, FNO `nu=0.01`, and DeepONet/default-net `nu=0.01` into
  the same figure layout.
- Use nine rows: each family has `model`, `solver`, and `error` rows.
- Generate per-index and aggregate line/FFT/heatmap/spectrum/angle plots without
  standard-deviation shading.

Added / generated:

- `tools/plot_fno001_fno01_deeponet01_ninerow_svd_diagnostics.py`
- `docs/fno001_fno01_deeponet01_ninerow_svd_diagnostics_result_20260515.md`
- `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/`

Key observed results:

- FNO `nu=0.001` model-vs-solver remains aligned: `k=1` angle `4.86 deg`,
  `k=8` angle `7.01 deg`; model top-1 `hi128` `2.231e-07`.
- FNO `nu=0.01` is even more tightly solver-aligned: `k=1` angle `2.32 deg`,
  `k=8` angle `1.82 deg`; model top-1 `hi128` `2.82e-09`.
- DeepONet/default-net `nu=0.01` remains solver-misaligned and high-frequency:
  model-vs-solver `k=1` angle `82.83 deg`, `k=8` angle `80.19 deg`, model
  top-1 `hi128` `0.7614`, and model top-1 zero crossings `507.20`.
- DeepONet/default-net error remains close to the model subspace:
  model-vs-error `k=1` angle `10.01 deg`, while FNO error directions are much
  less aligned with the FNO model directions.

Grounded conclusion:

- The nine-row figures make the contrast visually direct: both FNO variants have
  solver-like dominant local modes, while DeepONet/default-net has dominant
  high-frequency local modes that align with its error rather than the solver.

## Next Actions

1. If the exact original FNO-vs-DeepONet/default-net artifacts from the old
   Vast.ai instance are still needed, recover that full artifact directory and
   compare it against the reconstructed results.
2. For every new experiment, add a dedicated result note under `docs/`, update
   this ledger, and commit the scripts plus lightweight CSV/PNG/Markdown
   outputs. Keep large raw `.npz` artifacts local unless explicitly requested.

## 2026-05-15 Nine-Row SVD Interpretation Summary

Created `docs/nine_row_fno001_fno01_deeponet01_svd_interpretation_20260515.md` and `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/aggregate_model_vs_model_subspace_angles.csv`.  The summary consolidates FNO `nu=0.001`, FNO `nu=0.01`, and DeepONet `nu=0.01` local Jacobian SVD diagnostics: singular values, model top-8 response, high-frequency/zero-crossing metrics, model-vs-solver/model-vs-error subspace angles, model-vs-model subspace angles, and top-8 orthogonality.  Main conclusion: FNO is locally solver-like and its error Jacobian is a small residual, while DeepONet is dominated by high-frequency model directions and its error Jacobian is almost the DeepONet Jacobian itself.

## 2026-05-15 Local Jacobian/SVD Experiment Purpose

Created `docs/local_jacobian_svd_experiment_purpose_20260515.md` to connect the local Jacobian/SVD experiments to the project-level `loss3_original` narrative.  The purpose is to show that regression robustness is co-variation with the solver, not invariance of the model output; locally this means the dangerous object is `J_model - J_solver`, not `J_model` alone.  The note records the logic chain: `loss1_original` measures model movement, `loss3_original` measures perturbed-input oracle-relative error, and the Jacobian/SVD experiments explain when these directions differ.  FNO is locally solver-like, so model-sensitive directions can be co-moving false alerts; DeepONet/default-net is high-frequency and solver-misaligned, so its model-sensitive directions are much closer to error directions.
## 2026-05-15 DeepONet Loss1-vs-Loss3 Jacobian/Attack Tension

Created `docs/deeponet_loss1_vs_loss3_jacobian_attack_tension_20260515.md` plus `forensics/deeponet_loss1_vs_loss3_attack_tension_20260515/`.  Existing clean-recomputed attack-ratio tables show that DeepONet `loss3` optimization usually produces much larger final true `loss3` than `loss1` optimization, despite the local SVD fact that `J_deeponet` and `J_error` dominant subspaces are close.  Aggregated DeepONet `nu=0.01` rows: L2 median `loss3/loss1` true-loss ratio `19.07` with `loss3` wins `624/702`; Linf median ratio `84.35` with `loss3` wins `799/902`.  Conclusion: local `J_error ~= J_model` suggests a smaller conceptual gap for DeepONet than for FNO, but it does not imply finite-radius `loss1_original` and `loss3_original` attacks are equivalent.  `loss3_original` still optimizes the moving-oracle endpoint error, clean-residual outward direction, nonlinear path, and projection dynamics.


## 2026-05-15 Local Jacobian/SVD Direction Taxonomy

Created `docs/local_jacobian_svd_direction_taxonomy_20260515.md` to record the distinction between SVD directions, instantaneous gradient directions, clean-residual outward-growth directions, and finite-radius attack directions.  The note uses the existing notation `b=e(x)=f(x)-j(x)` and `A=J_e=J_f-J_j`: local squared `loss3_original` expands as `||b+A delta||^2 = ||b||^2 + 2 b^T A delta + delta^T A^T A delta`, so its gradient has both `A^T b` and `A^T A delta` terms.  Therefore the top right singular vector of `A` explains the pure residual Lipschitz / mismatch gain direction, but the actual `loss3_original` optimizer can differ because of the clean residual, moving solver target, nonlinear path effects, and projection dynamics.

## 2026-05-15 Loss Gradient Direction vs SVD Direction

Created `docs/loss_gradient_direction_vs_svd_direction_20260515.md` to record the key clarification that a top singular-vector direction is a homogeneous quadratic gain optimum / power-iteration limit, not the same as a one-step optimizer gradient at an arbitrary current perturbation. Existing local Jacobian/SVD experiments computed `v_f`, `v_j`, and `v_e`, the top right singular directions of `J_f`, `J_j`, and `J_e=J_f-J_j`; these answer whether the local linear maps have similar dominant gain directions. They do not directly answer whether `loss1`, `loss2`, and `loss3` take the same one-step gradient direction. The missing explicit diagnostics are `g1(delta_k)=J_f^T J_f delta_k`, `g2(delta_k)=J_f^T b + J_f^T J_f delta_k`, and `g3(delta_k)=J_e^T b + J_e^T J_e delta_k`, especially the clean-point outward directions `g2(0)=J_f^T b` and `g3(0)=J_e^T b`.

## 2026-05-15 FNO nu=0.001 Loss-Gradient Path Experiment Plan

Created `docs/fno_nu0p001_loss_gradient_path_experiment_plan_20260515.md`.  This is a plan only; the experiment has not been run yet.  The proposed diagnostic uses FNO/Burgers `nu=0.001`, `epsilon=8.0`, `alpha=0.3`, `steps=100`, and indices `0, 7, 40, 47, 115`.  It will save attack trajectories with `--save_trajectory --save_every 1`, then compare one-step gradient directions of `loss1`, `loss2`, and `loss3` along the attack path.  The clean-Jacobian diagnostics are `g1(k)=J_f^T J_f delta_k`, `g2(k)=J_f^T b + J_f^T J_f delta_k`, and `g3(k)=J_e^T b + J_e^T J_e delta_k`; exact autograd gradients should also be computed at selected steps to separate local-linear behavior from nonlinear path effects.  Main outputs should include gradient cosine curves, SVD-alignment curves, linear-vs-quadratic decomposition, frequency metrics, and selected-step line plots.


## 2026-05-15 FNO Loss-Gradient Plan Amendment

Updated `docs/fno_nu0p001_loss_gradient_path_experiment_plan_20260515.md` to explicitly include the fixed-radius `b`-aware affine optimum directions that were missing from the first plan draft: `v_loss2_out=normalize(J_f^T b)`, `v_loss3_out=normalize(J_e^T b)`, `v_loss2_rho=argmax ||b+rho J_f v||`, and `v_loss3_rho=argmax ||b+rho J_e v||`.  The plan now covers three distinct objects: SVD gain directions, one-step gradients at `delta_k`, and fixed-radius local affine optimum directions.  It also adds direction-source objective evaluation so candidate directions such as `v_f`, `v_j`, and `v_e` are evaluated under `loss1/loss2/loss3` without incorrectly calling them loss-specific directions.

## 2026-05-16 Neural Operator Robustness Research Directions

Status: planning note created; no numerical experiment was run for this entry.

Created `docs/neural_operator_robustness_research_directions_20260516.md` to
organize five future work directions:

- cross-framework solver/model combinations for 1D Burgers and 2D
  Stokes/Navier-Stokes;
- comparison of attack losses with unified true solver-level evaluation;
- perturbation-size-aware attack objectives;
- PGD versus power-iteration-like optimization methods;
- structure-aware error metrics beyond pointwise `L2`.

Observed evidence:

- This entry records only the Markdown planning document created from the
  user requested research directions.

Inference:

- The suggested execution order in the note prioritizes existing 1D
  Burgers/FNO loss experiments before broader cross-framework and
  structure-aware extensions.

Remaining work:

- Turn each direction into a concrete experiment plan before running numerical
  experiments.
- Record datasets, model checkpoints, solver versions, attack hyperparameters,
  tables, figures, and conclusions for each future run.

## 2026-05-16 Loss3 Experiment 3 Small-Epsilon Sweep - FNO nu=0.001 GPU Run

Status: completed on GPU. No CPU fallback was used for the official run.

User constraint recorded:

- Official neural-operator robustness experiments must use GPU. If CUDA, JAX
  GPU, or PyTorch GPU architecture support fails, stop and repair the
  environment rather than switching to CPU.

Environment repair and persistence:

- Added the GPU-only rule to `AGENTS.md`.
- Added `docs/gpu_only_experiment_policy_20260516.md`.
- Replaced the unsupported `torch==2.11.0+cu128` wheel with
  `torch==2.8.0+cu126` in `adv_robust`; the verified arch list includes
  `sm_70` for Tesla V100.
- Regenerated `requirements.txt` with the `cu126` PyTorch wheel index and
  `torch==2.8.0+cu126`, so future environment rebuilds do not reinstall the
  broken V100-incompatible wheel.
- Verified PyTorch CUDA matmul on `Tesla V100-SXM2-32GB`; verified JAX backend
  `gpu` and JAX GPU matmul; `pip check` reported no broken requirements.

Source files and inputs:

- Experiment script: `tools/run_loss3_small_epsilon_sweep.py`.
- Plan: `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_plan_20260516.md`.
- Result: `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`.
- Dataset: `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/test.pt`.
- Model checkpoint: `fno_training_runs/burgers_nu0p001_fno1d_500/burgers_1d/checkpoints/pytorch_fno1d_500.pt`.
- Local Jacobian/SVD artifacts: `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`.
- Outward-growth artifacts: `forensics/outward_growth_direction_20260515/fno_nu0p001/`.

Key settings:

- Scope: FNO / 1D Burgers / `nu=0.001` only.
- Sample indices: `0, 7, 40, 47, 115`.
- Epsilons: `1e-4, 1e-3, 1e-2, 1e-1`.
- Direction bank: top-8 right singular directions of `J_f`, `J_j`, and
  `J_e`, both signs; outward-growth direction; 128 seeded random controls.
- Evaluation batch size: `64`.
- Runtime evidence from `manifest.json`: device `cuda:0`, GPU
  `Tesla V100-SXM2-32GB`, required arch `sm_70`, PyTorch `2.8.0+cu126`,
  JAX backend `gpu`, policy `gpu_only_no_cpu_fallback`, runtime `20.40` s.

Output files:

- Output directory:
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`.
- Tables: `candidate_metrics.csv`, `best_by_objective_epsilon_index.csv`,
  `aggregate_best_by_objective_epsilon.csv`, `local_reference_by_index.csv`,
  `local_reference_summary.csv`, `direction_stability.csv`,
  `direction_stability_summary.csv`, `best_direction_source_counts.csv`.
- Manifest: `manifest.json`.
- Figures: `figures/small_epsilon_best_values.png`,
  `figures/small_epsilon_local_reference_ratio.png`,
  `figures/small_epsilon_direction_stability_heatmap.png`,
  `figures/small_epsilon_best_direction_sources.png`.

Observed key metrics from `aggregate_best_by_objective_epsilon.csv`:

- Local reference means: `L_f=3.895`, `L_j=4.095`, `L_e=0.8682`,
  `G_e=0.1665`.
- At `epsilon=1e-4`: `L_f=3.897`, `L_j=4.094`, `L_e=0.8685`,
  `G_e=0.1658`.
- At `epsilon=1e-1`: `L_f=3.856`, `L_j=4.036`, `L_e=0.8309`,
  `G_e=0.1763`.
- Best/local ratios stay near `1` for small epsilons. At `epsilon=1e-1`,
  ratios are approximately `L_f=0.990`, `L_j=0.985`, `L_e=0.961`,
  `G_e=1.058`.
- Best direction source counts: `L_f` selected FNO directions in `20/20`
  cases; `L_j` selected solver directions in `20/20`; `L_e` selected
  residual/error directions in `20/20`; `G_e` selected outward-growth in
  `20/20`.

Observed conclusion:

- The small-epsilon candidate sweep matches the clean local Jacobian references
  for `epsilon <= 1e-2`, supporting the claim that these ratio-style metrics
  represent local structure in this FNO `nu=0.001` setting.
- Model and solver sensitivities are much larger than residual sensitivity
  (`L_f` and `L_j` around `4`, `L_e` around `0.87`), consistent with FNO
  co-moving with the solver locally.
- `G_e` is much smaller than `L_e` (`0.166` versus `0.868` locally), confirming
  that outward clean-residual growth and residual-field movement are different
  diagnostics.
- Selected directions are stable up to sign across epsilon; `G_e` keeps the same
  outward-growth sign.

Inference:

- For FNO `nu=0.001`, Experiment 3 supports using very small epsilons as a
  local-structure diagnostic, but it should not be confused with a finite-radius
  attack objective. The local residual map can move at rate `L_e` without
  increasing the clean residual norm at the same rate, which is why `G_e` is
  the stricter outward-growth diagnostic.

Remaining work:

- If this result needs to be compared against PGD endpoint attacks, run a
  separate finite-radius attack experiment under the same GPU-only rule.
- Keep generated large arrays local/R2 unless explicitly asked to commit or
  upload them.

## 2026-05-16 CUDA/GPU Wheel Compatibility Documentation

Status: documentation update; no numerical experiment was run for this entry.

Created `docs/cuda_gpu_wheel_compatibility_notes_20260516.md` and linked it
from `AGENTS.md` plus `docs/gpu_only_experiment_policy_20260516.md`. The note
records why the previous V100 run failed with `torch==2.11.0+cu128`: the wheel
could see CUDA but did not include the V100 architecture `sm_70`, causing
`no kernel image is available for execution on the device`. It also records the
second compatibility issue observed during repair: `torch==2.6.0+cu126` made
PyTorch CUDA work on V100 but pulled cuDNN `9.5.1`, while JAX `0.10.0` needed
cuDNN `9.8.0` or newer.

Observed working environment at documentation time:

- GPU: Tesla V100-SXM2-32GB, compute capability `(7, 0)` / `sm_70`.
- PyTorch: `2.8.0+cu126`; `torch.cuda.get_arch_list()` includes `sm_70`.
- JAX backend: `gpu`; JAX devices include `CudaDevice(id=0)`.

Remaining rule:

- Before any official experiment, verify the active GPU architecture, PyTorch
  arch list, real PyTorch CUDA operation, JAX GPU backend, real JAX GPU
  operation, and `pip check`. Do not use CPU fallback for official runs.

## 2026-05-16 Loss3 Small-Epsilon Result Explanation Clarification

Status: documentation clarification; no new numerical experiment was run.

Updated `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
to make the experiment logic explicit. The clarified note now states that the
run is a candidate-bank direction maximization, not PGD and not model training.
It records what was optimized (`L_f`, `L_j`, `L_e`, and `G_e` over candidate
directions), how the 178-direction bank was constructed, the GPU-only run
conditions, the 3560 FNO/solver candidate evaluations, and the key
epsilon-refinement conclusion.

Observed clarification from existing CSV files:

- For `epsilon <= 1e-2`, best/local ratios remain near `1`, so the
  finite-epsilon ratios match the clean local Jacobian references.
- At `epsilon=0.1`, values begin to drift (`L_e` about `0.961` of local
  reference and `G_e` about `1.058`), indicating finite-radius effects.
- Best direction sources remain stable in `20/20` sample-epsilon cases for each
  objective: FNO for `L_f`, solver for `L_j`, residual/error for `L_e`, and
  outward-growth for `G_e`.

Inference now stated explicitly:

- This is analogous to an epsilon-refinement/local-convergence check: once the
  perturbation radius is small enough, shrinking it further does not materially
  change the measured local direction or local ratio.

## 2026-05-16 GPU-Aware adv_robust Setup Script

Status: environment automation added; no numerical experiment was run for this
entry.

Created `tools/setup_adv_robust_gpu_env.py` as the project entry point for
creating or verifying `adv_robust`. The script detects the visible GPU compute
capability with `nvidia-smi`, creates `adv_robust` when missing, installs the
pinned `requirements.txt`, and refuses to pass unless PyTorch and JAX both run
real GPU matrix operations. It also verifies that the PyTorch wheel contains the
required `sm_*` architecture for the active GPU, currently `sm_70` on V100, and
runs `pip check`.

Verified on the current machine with:

- command: `tools/setup_adv_robust_gpu_env.py --verify-only`;
- GPU: Tesla V100-SXM2-32GB;
- required architecture: `sm_70`;
- PyTorch: `2.8.0+cu126`;
- JAX backend: `gpu`;
- result: PyTorch CUDA matmul passed, JAX GPU matmul passed, and `pip check`
  reported no broken requirements.

Updated `AGENTS.md`, `docs/cuda_gpu_wheel_compatibility_notes_20260516.md`,
`ENVIRONMENT_SETUP_REPRODUCTION_GUIDE.md`, and `ENVIRONMENT_RUN_GUIDE.md` so
future environment rebuilds use this script rather than raw manual installation.

## 2026-05-16 Loss3 Gradient Direction Optimization - FNO nu=0.001 GPU Run

Status: completed on GPU with no CPU fallback.

Purpose:

- Address the limitation of the earlier candidate-bank sweep by directly
  optimizing perturbation directions with projected gradient ascent.
- Compare gradient-optimized directions against candidate-bank selections,
  pullback eigenvectors of `J_f^T J_f`, `J_j^T J_j`, `J_e^T J_e`, and the
  clean residual outward-growth direction.

Source files and inputs:

- Script: `tools/run_loss3_gradient_direction_optimization.py`.
- Result doc:
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`.
- Output directory:
  `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`.
- Previous candidate sweep used for comparison:
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`.
- Same FNO `nu=0.001`, dataset, checkpoint, Jacobian/SVD artifacts, and
  outward-growth artifacts as the previous small-epsilon experiment.

Key settings:

- Samples: `0, 7, 40, 47, 115`.
- Epsilons: `1e-4, 1e-3, 1e-2, 1e-1`.
- Objectives: `L_f`, `L_j`, `L_e`, `G_e`.
- Optimization: Adam ascent on unit direction `v`, projected/renormalized after
  each step.
- Starts per case: analytic plus, analytic minus, and one random start.
- Steps per start: `12`; learning rate: `0.2`.
- Runtime: `1117.48` seconds.
- GPU evidence from manifest: device `cuda:0`, `Tesla V100-SXM2-32GB`, required
  arch `sm_70`, PyTorch `2.8.0+cu126`, JAX backend `gpu`.

Output files:

- `gradient_run_summary.csv`
- `gradient_trajectory.csv`
- `gradient_best_by_case.csv`
- `gradient_summary_by_objective_epsilon.csv`
- `pullback_eigen_summary.csv`
- `manifest.json`
- figures: `gradient_vs_candidate_value_ratio.png`,
  `gradient_direction_alignment.png`, `random_start_alignment.png`

Observed results:

- Gradient-optimized values nearly match candidate-bank values: aggregate
  `grad/candidate` ratios are around `0.999-1.010`.
- `L_f` optimized directions align with the FNO pullback/SVD direction with mean
  absolute cosine about `0.996-0.999`.
- `L_e` optimized directions align with the residual/error pullback/SVD
  direction for `epsilon <= 1e-2` with mean absolute cosine about `0.998-0.999`;
  at `epsilon=0.1`, the mean is about `0.973`.
- `G_e` optimized directions align with the outward-growth direction with mean
  absolute cosine about `0.999` for `epsilon <= 1e-2` and about `0.991` at
  `epsilon=0.1`.
- `L_j` optimized values match the candidate values, but direction alignment is
  weaker than for the other objectives: mean absolute cosine is about `0.982`
  for small epsilons and about `0.969` at `epsilon=0.1`; the worst case is
  index `47`, `epsilon=0.1`, cosine about `0.849` with value still `0.999` of
  candidate maximum.
- Pullback eigenvector checks: for `L_f`, `L_j`, and `L_e`, the largest
  eigenvector of `J^T J` matches the saved top right singular vector with
  absolute cosine `1.0` for all checked samples.
- Outward-growth check: `normalize(J_e^T e(x)/||e(x)||)` matches the saved
  outward-growth direction with absolute cosine `1.0`; this is not a top
  eigenvector of `J_e^T J_e`.

Conclusion:

- The second version confirms that the previous candidate-bank directions were
  not arbitrary. For small epsilons, actual gradient ascent recovers essentially
  the same directions and values.
- The correct local linear algebra is: `L_f/L_j/L_e` use pullback matrices
  `J^T J`; `G_e` uses the clean residual outward gradient `J_e^T e/||e||`.
- At `epsilon=0.1`, finite-radius drift appears, so exact direction equality
  should not be overclaimed even when the scalar objective value remains near
  the candidate maximum.


## 2026-05-16 - Experiment 4 Ray Profile PGD100 Sign-Fixed Batch100 Cross-Check

Status: completed on GPU with no CPU fallback.

Purpose:

- Re-run the Ray Profile / Local-to-Global Profile under the user's requested normal protocol: no best-over-steps, no multi-restart, batch size 100, `epsilon=8`, zero initialization, ordinary PGD, `steps=100`, `alpha=0.3`.
- Resolve the discrepancy between the earlier Ray wrapper output and the historical batch-100 loss3 result.

Key correction:

- The first hand-written PGD branch in `tools/run_loss3_ray_profile_normal_batch.py` had a sign error: it differentiated `-objective` and then updated `delta += alpha * grad`, which performs descent for the target objective.
- The corrected PGD branch now differentiates the objective directly and updates `delta += alpha * grad`, matching `tools/run_batch_three_loss_loss_only.py`.
- The earlier non-fixed-sign PGD output is superseded and should not be used as a scientific result.

Runs and artifacts:

- Corrected Ray wrapper script: `tools/run_loss3_ray_profile_normal_batch.py`.
- Corrected result doc: `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`.
- Corrected output directory: `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Historical-script cross-check output directory: `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.

Corrected Ray result:

- Endpoint `loss3_original` mean at `r=8`: `loss3_original_final=5.4469`, `loss3_increment_ratio_final=4.2215`, `loss3_regularized_final=2.0679`, `local_residual_movement=4.1391`, `local_outward_growth=2.7204`.
- Endpoint winners among all profiled directions: `loss3_original_final` 58/100, `local_residual_movement` 31/100, `loss3_increment_ratio_final` 10/100, `loss3_regularized_final` 1/100.
- Endpoint winners among only the three finite-radius PGD attack objectives: `loss3_original_final` 81/100, `loss3_increment_ratio_final` 15/100, `loss3_regularized_final` 4/100.
- Small-radius clean norm-growth winner: `local_outward_growth` 100/100.
- Small-radius residual-increment winner: `local_residual_movement` 100/100.

Historical-script cross-check:

- `loss3_original_pgd`: final delta norm mean `7.7510`, boundary `loss3_original` mean `5.4469`.
- `loss3_increment_ratio_pgd`: final delta norm mean `7.8505`, boundary `loss3_original` mean `4.2215`.
- `loss3_regularized_pgd`: final delta norm mean `0.3043`; boundary-rescaled `loss3_original` mean `2.0275`.

Conclusion:

- The user's expectation is confirmed under the historical PGD protocol: direct `loss3_original` PGD gives the strongest endpoint mean loss3.
- The earlier contradictory PGD Ray result was an implementation bug, not evidence against the historical conclusion.
- The Ray experiment still supports the intended nonlinear story: the clean local directions win tiny-radius slope diagnostics, but direct finite-radius `loss3_original` PGD gives the strongest large-radius endpoint behavior on average.


## 2026-05-16 - Unified Evaluation-Metric Loss Curve Replots

Status: completed from existing saved trajectory data. No attack was rerun.

Purpose:

- Replot the historical 27-run three-loss attack setting with a fixed y-axis evaluation metric, instead of plotting each trajectory's own optimized objective.
- This answers questions such as: among all 27 optimized trajectories, how do they compare when every curve is evaluated by `loss3_original` or `loss3_increment_ratio`?

Data source:

- Restored from R2: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- This directory contains all 27 trajectories: 3 optimized losses x 3 objective variants x 3 optimization methods.
- Each trajectory already includes all 9 recorded evaluation metrics in `loss_stats.csv` and `loss_values.npz`.

New script:

- `tools/plot_batch_eval_metric_matrix_curves.py`.
- Layout: 3-by-3 matrix with rows for optimized loss (`loss1`, `loss2`, `loss3`), columns for objective variant (`original`, `increment_ratio`, `regularized`), and method curves inside each panel.

Generated plots:

- Batch mean/std figures: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves/png/`.
- Single-index figures for dataset index `0`: `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves/index_png/`.
- All nine evaluation metrics were generated, including the key loss3 views: `loss3_original_all_27_mean_std.png`, `loss3_increment_ratio_all_27_mean_std.png`, `loss3_regularized_all_27_mean_std.png`, and corresponding `index0` plots.

Documentation:

- `docs/unified_eval_metric_loss_curve_plots_20260516.md`.


## 2026-05-16 - Adam vs Ordinary PGD Loss3 Original Parameter Control

Status: completed on GPU with no CPU fallback.

Purpose:

- Test whether Adam systematically gives smaller direct `loss3_original` attack values than ordinary PGD under two additional `(epsilon, alpha)` settings.
- Keep objective fixed to direct `loss3_original`, batch fixed to 100 samples, and steps fixed to 50.

Script and outputs:

- Script: `tools/compare_adam_pgd_loss3_original_params.py`.
- Result doc: `docs/adam_vs_pgd_loss3_original_param_control_20260516.md`.
- Output directory: `forensics/adam_vs_pgd_loss3_original_params_20260516/fno_nu0p001_batch100_steps50_eps4_eps12/`.

Settings and results:

- `epsilon=4.0`, `alpha=0.15`: boundary `loss3_original` mean was Adam `1.6712` vs PGD `2.3290`; paired PGD-Adam mean difference `+0.6578`, p-value `1.487e-09`.
- `epsilon=12.0`, `alpha=0.45`: boundary `loss3_original` mean was Adam `8.1319` vs PGD `7.2865`; paired PGD-Adam mean difference `-0.8454`, p-value `2.612e-02`.

Conclusion:

- The added settings do not support a universal claim that Adam always makes direct `loss3_original` smaller.
- At smaller radius (`epsilon=4`), PGD is significantly stronger at the boundary endpoint.
- At larger radius (`epsilon=12`) and fixed 50 steps, Adam is stronger; Adam reaches the boundary immediately, while PGD's final norm mean is about `8.66` before boundary rescaling.
- The robust conclusion is that Adam changes the constrained optimizer dynamics substantially; the relative ranking depends on epsilon, alpha, step count, and boundary reach.

## 2026-05-17 - adv_robust Environment And 2026-05-16 Artifact Audit

Status: completed audit on the current Vast.ai instance. No numerical experiment was rerun.

Purpose:

- Check whether the copied `adv_robust` environment is runnable and complete.
- Check whether the 2026-05-16 experiment artifacts copied onto this instance are locally complete.
- Separate observed local evidence from inference, especially for smoke runs, partial directories, and R2-backed artifacts.

Source files and inputs:

- Environment policy/setup script: `tools/setup_adv_robust_gpu_env.py`.
- Requirements file: `requirements.txt`.
- GPU compatibility note: `docs/cuda_gpu_wheel_compatibility_notes_20260516.md`.
- Local 2026-05-16 manifests under `forensics/*20260516*/`.
- Local 2026-05-16 result docs under `docs/`.
- R2 sync record: `docs/r2_sync_manifest_ray_profile_20260516.md`.

Output record:

- `docs/adv_robust_environment_and_20260516_artifact_audit_20260517.md`.

Observed environment evidence:

- The first `tools/setup_adv_robust_gpu_env.py --verify-only` check found that `adv_robust/bin/python` was unusable because `python3` and `python3.12` formed a symlink loop.
- Repaired `adv_robust/bin/python3` to point to `/usr/bin/python3.12`.
- After repair, `tools/setup_adv_robust_gpu_env.py --verify-only` passed: PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, GPU `Tesla V100-SXM2-32GB`, compute capability `(7, 0)`, required architecture `sm_70`, PyTorch arch list includes `sm_70`, PyTorch CUDA matmul returned `1.0`, JAX backend was `gpu`, JAX device was `CudaDevice(id=0)`, JAX matmul returned `1.0`, and `pip check` reported no broken requirements.
- Requirements-vs-installed check covered `79` direct requirement entries: missing requirements `[]`, pinned-version mismatches `[]`, installed distributions `85`.
- Key import smoke test passed for the major packages used by the experiments: PyTorch, JAX, NumPy/SciPy/Pandas/Matplotlib, DeepXDE, Equinox, Exponax, Torch2JAX, TensorLy, tslearn, UMAP, scikit-optimize, PhiFlow/PhiML, and related dependencies.

Observed artifact evidence:

- Found `18` `manifest.json` files under `forensics/*20260516*`.
- All `18` manifests parsed, all manifest-declared `output_files` exist, all manifest-declared `result_doc`/`plan_doc` paths exist, and all manifest-declared `source_paths` exist locally.
- `17` of the `18` manifests carry GPU runtime evidence consistent with the V100 / `sm_70` policy.
- Exception: `forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001` records `"device": "cpu"` and no `gpu_runtime`; it should be treated as an old smoke check, not an official GPU result.
- Incomplete local directory: `forensics/loss3_ray_profile_optimizer_control_20260516` contains only one `local_direction_trace.csv` file with `5` lines and no manifest/result doc; the R2 sync manifest also records it as only `1` file / `672` bytes.
- Corrected fixed-sign PGD batch-100 Ray profile directory has the manifest outputs present; `ray_profile.csv` has `31501` lines (`100 * 7 * 45 + header`), `ray_winner_summary.csv` has `101` lines, and `attack_final_by_sample.csv` has `401` lines.
- Historical-script cross-check directory `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro` contains all `9` expected `loss3_*` run directories, and each has the expected `summary.json`, `loss_stats.csv`, `loss_values.npz`, `final_delta.npz`, `final_delta_summary.json`, `final_delta_diagnostics.csv`, and `final_delta_diagnostics.npz` files.
- `docs/r2_sync_manifest_ray_profile_20260516.md` records `198` uploaded files and `145207187` bytes. This audit did not perform a live R2 network query.
- `git status --short` before this record update showed untracked experiment paths but no deleted tracked experiment artifacts.

Conclusion:

- Observed evidence supports that `adv_robust` is now runnable and complete for the repository's GPU-only experiment policy after the venv symlink repair.
- Observed evidence supports that the main 2026-05-16 GPU result directories with manifests are locally complete.
- The CPU small-epsilon smoke directory is not an official GPU result.
- The `loss3_ray_profile_optimizer_control_20260516` directory is incomplete locally and should not be interpreted as a completed experiment.

Remaining work:

- If remote backup completeness matters, perform a live R2 listing/checksum comparison against the recorded manifests.
- If saving this audit in git, explicitly stage the new doc and ledger update along with any relevant source/record files; do not rely on `git commit -am`.

## 2026-05-17 - Loss3 Original Six-Experiment Completion Audit

Status: completed audit and status-table update. No numerical experiment was rerun.

Purpose:

- Locate the user's six-experiment plan and determine whether all six planned
  experiments are completed.
- Update stale planning records if the local evidence shows later experiments
  completed the missing rows.

Source plan:

- `docs/loss3_original_theory_experiment_plan.md`, section `5.1 Current
  Completion Status for the Minimal Six Experiments`.

Audit record:

- `docs/loss3_original_six_experiment_completion_audit_20260517.md`.

Observed evidence:

- Experiment 1 is documented in
  `docs/main_objective_mechanism_experiment1_result_20260514.md`. The local
  source directory
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`
  exists with `27` loss/objective/method run directories, and the core table
  `mechanism_diagnostics/mechanism_summary.csv` has `55` lines. R2 recovery update: `results/main_objective_mechanism_summary_20260514/` has now been restored locally from R2 and contains `combined_mechanism_summary.csv`, `focused_original_objectives.csv`, and three focused-objective PNG plots.
- Experiment 2 is completed by the FNO/solver Jacobian similarity records and
  the outward-growth row in `docs/outward_growth_direction_result_20260515.md`,
  with outputs under `forensics/fno_solver_jacobian_similarity_20260514/` and
  `forensics/outward_growth_direction_20260515/fno_nu0p001/`.
- Experiment 3 is completed by the GPU small-epsilon sweep and gradient-direction
  optimization records:
  `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`,
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`,
  and the corresponding `forensics/` directories.
- Experiment 4 is completed by the corrected fixed-sign PGD100 Ray Profile in
  `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`
  and its artifact directory
  `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Experiment 5 is completed by the boundary-rescaled final-boundary diagnostics
  in the 2026-05-14 final-boundary source directory and by the 2026-05-16 local
  cross-check directory
  `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.
- Experiment 6 is completed by `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`
  and the expanded `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`.
  The expanded manifest records `status: completed`, V100 GPU runtime, and
  output files under
  `forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/`.

Changes made:

- Updated `docs/loss3_original_theory_experiment_plan.md` section 5.1 from
  `Done: 5 / Partially done: 1` to `Done: 6 / Partially done: 0` for the current
  FNO / Burgers `nu=0.001` scope.
- Added `docs/loss3_original_six_experiment_completion_audit_20260517.md` with
  observed evidence and caveats.

Conclusion:

- Observed evidence supports marking all six planned experiments complete for
  the current FNO / Burgers `nu=0.001` story.
- Remaining work is extension/presentation/full remote-backup verification, not a
  blocker for the six-experiment completion claim.

R2 follow-up in the same turn:

- Used the provided R2 S3 endpoint to restore the previously missing local
  directory `results/main_objective_mechanism_summary_20260514/`.
- Restored files: `combined_mechanism_summary.csv`,
  `focused_original_objectives.csv`, and three focused-objective PNG plots.
- Verified restored CSV line counts: `combined_mechanism_summary.csv` has `325`
  lines and `focused_original_objectives.csv` has `55` lines.
- Also checked R2 for
  `forensics/loss3_ray_profile_optimizer_control_20260516`; R2 contains only
  the same single `local_direction_trace.csv`, so that directory remains an
  incomplete/aborted trace rather than a local copy gap.

## 2026-05-17 - Six-Experiment Results And Conclusions Written Into Plan

Status: completed documentation update. No numerical experiment was rerun.

Purpose:

- Add the user's requested per-experiment explanation: what each of the six
  experiments measured, which numeric outputs matter, what conclusion each
  experiment supports, and how the six results fit together.

Files updated:

- `docs/loss3_original_theory_experiment_plan.md`:
  added section `5.2 Six-Experiment Results, Data, And Conclusions - 2026-05-17`.
- `docs/loss3_original_six_experiment_completion_audit_20260517.md`:
  appended the same detailed conclusion section for audit traceability.
- `EXPERIMENT_LEDGER.md`: this entry.

Observed source evidence used:

- Experiment 1: `docs/main_objective_mechanism_experiment1_result_20260514.md`,
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/mechanism_diagnostics/mechanism_summary.csv`,
  and the restored `results/main_objective_mechanism_summary_20260514/`.
- Experiment 2: `docs/fno_solver_jacobian_similarity_result_20260514.md`,
  `docs/outward_growth_direction_result_20260515.md`,
  `forensics/fno_solver_jacobian_similarity_20260514/`, and
  `forensics/outward_growth_direction_20260515/fno_nu0p001/`.
- Experiment 3: `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`,
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`,
  and corresponding `forensics/loss3_*_20260516/` outputs.
- Experiment 4: `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`,
  `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`,
  and the 2026-05-16 local historical-script cross-check result directory.
- Experiment 5: `docs/loss3_original_plan_r2_completion_audit_20260515.md`,
  the 2026-05-14 final-boundary source directory, and the 2026-05-16 local
  cross-check directory.
- Experiment 6: `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`,
  `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`,
  and corresponding `forensics/loss3_*rotation_path_20260516/` outputs.

Key conclusions recorded:

- Experiment 1: `loss1`/`loss2` mainly produce large model/solver co-movement;
  `loss3_original` produces larger true mismatch and endpoint error.
- Experiment 2: residual movement `v_e*` and clean residual outward growth
  `v_growth*` are different local diagnostics.
- Experiment 3: ratio diagnostics converge to local Jacobian references for
  `epsilon <= 1e-2`, while finite-radius drift appears by `epsilon=0.1`.
- Experiment 4: local tiny-radius winners are not finite-radius endpoint winners;
  direct `loss3_original` is strongest at `r=8` under the corrected protocol.
- Experiment 5: boundary-rescaling ratio/regularized directions does not make
  them beat direct `loss3_original` endpoint optimization.
- Experiment 6: residual-Jacobian directions and top-k subspaces rotate and
  steepen along the attack path, so a single clean-point linearization is not a
  full finite-radius explanation.

Conclusion:

- The plan now contains a readable per-experiment result/conclusion section, in
  addition to the completion-status table.

## 2026-05-17 - Loss3 Optimizer Direction-Proposal Ablation Plan

Status: plan written; no numerical experiment was run.

Purpose:

- Design a new experiment comparing PGD, LP-steepest PGD, and generalized power
  iteration for `loss3_original`.
- Separate the direction rule from the proposal rule so the observed fast
  convergence of generalized power can be attributed to either direction choice,
  replacement/boundary update, step size, or physical/smoothness tradeoffs.

Plan document:

- `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`.

Observed code basis:

- `tools/run_batch_three_loss_loss_only.py` currently implements:
  `pgd` as `delta <- Proj(delta + alpha * grad)`,
  `lp_steepest_pgd` as `delta <- Proj(delta + alpha * steepest_direction(grad))`,
  and `generalized_power` as `delta <- Proj(epsilon * steepest_direction(grad))`.
- `run_three_loss_objective_attack.py` documents the same PGD and LP-steepest
  update rules and also supports generalized-power variants such as
  `objective_gradient`, `pure_jvp_vjp`, and `affine_jvp_vjp`.

Proposed experiment:

- Factorial ablation over three direction rules and two proposal rules:
  raw-gradient/additive, raw-gradient/replacement, LP-steepest/additive,
  LP-steepest/replacement, generalized-power/additive, and
  generalized-power/replacement.
- Main setting: FNO / Burgers `nu=0.001`, `loss3_original`, L2 input/output,
  `epsilon=8`, batch 100, zero initialization, 100 steps, GPU only.
- Metrics: final and best-so-far `loss3_original`, boundary reach, time/steps to
  threshold, boundary-normalized per-step direction quality, direction cosines,
  final-delta pairwise cosines, projection shrink factors, and perturbation
  smoothness/physical diagnostics such as Fourier high-frequency energy, total
  variation, derivative norms, and clean/adversarial spectrum comparison.

Conclusion expected from the design:

- The experiment can tell whether generalized power is faster because it uses a
  better direction, because it replaces the perturbation directly on the
  boundary, because additive PGD is step-size limited, or because fast boundary
  directions trade off against perturbation smoothness/physical plausibility.

## 2026-05-17 - Direction-Proposal Ablation Plan Equivalence Update

Status: documentation update only. No numerical experiment was run.

Purpose:

- Clarify the user's concern that PGD, LP-steepest PGD, and generalized power
  can collapse into equivalent updates under some norm/direction/proposal
  choices.

Files updated:

- `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`.

Key additions:

- Added an explicit `Equivalence And Degeneracy Cases To Check Explicitly`
  section.
- Recorded that for L2, normalized raw-gradient updates collapse to LP-steepest
  updates: `unit_raw_add == steepest_add` when both use `g / ||g||_2`.
- Recorded that the current batch-script `generalized_power` implementation can
  collapse to LP-steepest replacement when `u_k = steepest_direction(g_k, p)`, so
  `power_replace == steepest_replace` by construction.
- Added the requirement to label power variants separately, e.g.
  `power_replace__steepest_gradient`, `power_replace__objective_gradient`,
  `power_replace__pure_jvp_vjp`, and `power_replace__affine_jvp_vjp`.
- Added required sanity outputs such as same-step direction cosines,
  final-delta cosines, max absolute delta differences, and warning flags when
  two rows are mathematically equivalent.

Conclusion:

- The planned experiment now explicitly handles equivalence/degeneracy cases and
  should first verify which rows are actually distinct before making optimizer
  superiority claims.
## 2026-05-17 - Direction-Proposal Ablation Plan P/Q Geometry Update

- Status: plan update only; no numerical run was started.
- Updated `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  to add a dedicated P/Q geometry extension for the new optimizer comparison.
- Observed from `tools/run_batch_three_loss_loss_only.py`: `--p` controls the
  perturbation projection/budget for `delta`, while `--q` controls the
  residual norm used by `loss3_original`.
- Observed from `run_three_loss_objective_attack.py`: the older single-index
  runner uses `--input_p/--output_q`, supports generalized p/q power variants,
  and documents that q-aware power uses the output dual map before mapping back
  to the p-ball.
- Observed from `three_loss_objective_experiment_plan.md`: an earlier p/q sweep
  already proposed `(2,2)`, `(inf,2)`, `(2,inf)`, `(inf,inf)`, plus sparse
  input extensions.
- Inference from the code and prior plan: the optimizer ablation should compare
  methods within each p/q pair first, and only compare across p/q pairs using
  shared auxiliary metrics such as `loss3_l2`, `loss3_linf`, common-unit delta
  norms, and smoothness diagnostics.
- Remaining work: implement the runner/analysis changes, then run GPU-verified
  smoke and official p/q sweeps.
## 2026-05-17 - Direction-Proposal Ablation Plan Autograd Gradient Clarification

- Status: plan update only; no numerical run was started.
- Updated `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  to separate exact objective-gradient computation from p-steepest direction
  construction.
- Observed from existing helpers: `tools/run_batch_three_loss_loss_only.py`
  computes the scalar loss and uses autograd for the gradient, then maps the
  gradient through `steepest_direction(grad, p_order)` for LP-steepest and
  current generalized-power rows.
- Inference from the implementation and user clarification: for ordinary PGD,
  LP-steepest PGD, and objective-gradient replacement, the q-norm residual
  gradient should come from autograd.  Only the map from `g_k` to the p-ball
  steepest direction `s_k` needs an explicit formula.
- Remaining work: when implementing the new runner, record both
  `gradient_implementation_source` and `steepest_direction_source` in the
  manifest and per-run diagnostics.
## 2026-05-17 - Direction-Proposal Ablation Plan Visualization And GIF Update

- Status: plan update only; no numerical run was started.
- Updated `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  to require visualization of final deltas, per-step delta trajectories, loss
  curves, spectra, smoothness metrics, time-space heatmaps, and GIFs.
- Observed from existing code: `loss_attack_common.py` already contains basic
  run plots, final-field plots, and GIF helper patterns;
  `run_three_loss_objective_attack.py` already supports saving `trajectory.npz`
  for per-step fields; `plot_burgers_corrected_oldstyle_5loss_gif.py` already
  renders synchronized Burgers GIF frames.
- Inference from the user request: the new optimizer ablation should save
  representative-sample trajectories and make synchronized visual comparisons,
  because final scalar metrics alone cannot show whether a method creates
  spiky/high-frequency deltas early and smooths them later.
- Remaining work: implement `--save-delta-trajectory`, trajectory NPZ outputs,
  per-step smoothness/spectrum diagnostics, static comparison grids, and GIF
  rendering in the new runner/analysis scripts.
## 2026-05-17 - Direction-Proposal Ablation Runner Implemented, Not Run

- Status: code implementation only; no numerical optimizer run was started.
- Added `tools/run_loss3_direction_proposal_ablation.py` as a single-entry
  runner/analyzer/visualizer for the loss3 direction-proposal ablation.
- Source paths changed:
  - `tools/run_loss3_direction_proposal_ablation.py`
  - `docs/loss3_optimizer_direction_proposal_ablation_plan_20260517.md`
  - `EXPERIMENT_LEDGER.md`
- Planned output paths for future runs:
  - `forensics/loss3_optimizer_direction_proposal_ablation_20260517/...`
  - method subdirectories containing `per_step_metrics.csv`,
    `per_sample_step_metrics.csv`, `final_delta_diagnostics.csv`,
    `final_deltas.npz`, `trajectory_samples.npz`, and visualization outputs.
- Key settings encoded in the script: `loss3_original`, GPU-only runtime
  evidence, `--p/--q` geometry, additive versus replacement proposals, raw /
  unit raw / LP-steepest / objective-gradient power / q-aware power directions,
  representative delta trajectories, static plots, optional GIFs, and
  post-run analysis checklist generation.
- Verification performed: `python3 -m py_compile
  tools/run_loss3_direction_proposal_ablation.py` completed successfully.
- Observed evidence: syntax check passed; no model load, solver call, CUDA
  optimization loop, or experiment result file was produced by this step.
- Inference: the code is ready for a smoke run after explicit approval, starting
  with non-contiguous dataset indices `0 7 40 47 115`, short steps, and GIFs.
- Remaining work: run GPU-verified smoke, inspect the generated figures/GIFs,
  then run the official p/q and alpha/epsilon sweeps.
## 2026-05-17 - Direction-Proposal Ablation Tiny Smoke Runtime

- Status: tiny GPU smoke completed; no scientific optimizer-quality conclusion.
- GPU verification before run: `nvidia-smi` detected Tesla V100-SXM2-32GB with
  compute capability `7.0`; `tools/setup_adv_robust_gpu_env.py --verify-only`
  passed with PyTorch `2.8.0+cu126`, CUDA `12.6`, arch list including `sm_70`,
  JAX backend `gpu`, and GPU matmul sanity checks.
- Source file: `tools/run_loss3_direction_proposal_ablation.py`.
- Output root: `forensics/loss3_optimizer_direction_proposal_ablation_20260517/smoke_tiny_runtime_p2_q2_idx0_7_steps3`.
- Result document: `docs/loss3_direction_proposal_ablation_smoke_runtime_20260517.md`.
- Key settings: FNO/Burgers `nu=0.001`, `loss3_original`, dataset indices `0`
  and `7`, methods `official3 main`, `steps=3`, `epsilon=8`, `alpha=0.3`,
  `p=2`, `q=2`, trajectory saving enabled, GIF rendering enabled.
- Observed outputs: manifest status `completed`; root CSV/NPZ outputs exist;
  66 files under `figures/`; 28 GIFs under `figures/gifs/`.
- Observed runtime: method optimizer runtime sum `32.27` seconds; whole-output
  file timestamp span including diagnostics, plots, and GIFs `74.10` seconds.
- Observed prior calibration: recent local batch100/steps100 official
  `loss3_original` runs took about `103-105` seconds per ordinary method.
- Inference: a batch100/steps100 p=2/q=2 `main` run should be planned as
  roughly `30-40` minutes without GIFs, or `50-75` minutes with full GIFs.
- Remaining work: run official p=2/q=2 main comparison without GIFs first, then
  selectively render GIFs for interesting methods/samples.
## 2026-05-17 - Direction-Proposal Ablation P2/Q2 In-Progress Runtime Status

- Status: official p=2/q=2 batch100/steps100 `main` run is still in progress.
- Status check time: `2026-05-17 22:24:07 UTC`.
- Process: PID `37171`, elapsed `13:43`, CPU `98.0%`, RSS `2280672 KiB`.
- GPU: utilization `97%`, memory `10744 / 32768 MiB`.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Log path: `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p2_q2_batch100.log`.
- Observed progress: 7 of 11 methods completed; current method is
  `power_add__pure_jvp_vjp`; three methods remain after it.
- Observed completed method runtimes: ordinary/objective-gradient methods are
  stable at about `105.5` seconds each.
- Inference: remaining time from the check was about `13-18` minutes including
  final aggregation/static plots; estimated finish window `22:37-22:42 UTC`.
- Result/status document: `docs/loss3_direction_proposal_ablation_p2_q2_run_status_20260517.md`.
## 2026-05-17 - Direction-Proposal Ablation P2/Q2 Main Run Completed

- Status: completed; no scientific interpretation performed in this status check.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Log path: `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p2_q2_batch100.log`.
- Source file: `tools/run_loss3_direction_proposal_ablation.py`.
- Key settings: FNO/Burgers `nu=0.001`, `loss3_original`, batch `100`, start
  index `0`, `steps=100`, `epsilon=8`, `alpha=0.3`, `p=2`, `q=2`, methods
  `main`, trajectory saving enabled, GIF rendering not enabled.
- Observed from `manifest.json`: status `completed`, method count `11`, finished
  at `2026-05-17 22:38:08 UTC`; GPU runtime evidence recorded with Tesla
  V100-SXM2-32GB, PyTorch `2.8.0+cu126`, CUDA `12.6`, `sm_70` in arch list,
  and JAX backend `gpu`.
- Observed method runtime summary: runtime sum `1537.89` seconds, mean `139.81`
  seconds, median `105.70` seconds; q-aware JVP/VJP methods took about
  `199-200` seconds each.
- Observed output completeness: `per_step_metrics.csv` has `1111` data rows,
  `per_sample_step_metrics.csv` has `111100` data rows, `pq_geometry_summary.csv`
  has `11` data rows, `figures/` contains `56` files, and GIF count is `0` as
  intended.
- Result/status document updated: `docs/loss3_direction_proposal_ablation_p2_q2_run_status_20260517.md`.
- Remaining work: analyze p=2/q=2 metrics/figures and write a scientific result
  document before launching more p/q geometry runs.
## 2026-05-17 - Direction-Proposal Ablation P2/Q2 Visualization Update

- Status: visualization update for the completed p=2/q=2 run; no new optimizer
  experiment was run.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Generated additional figures from existing `trajectory_samples.npz` outputs:
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_000.png`
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_007.png`
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_040.png`
  - `figures/final_input_delta_overlays/final_input_delta_overlay_sample_047.png`
  - corresponding `figures/final_delta_overlaid/final_delta_overlay_sample_*.png`
  - `figures/loss_progression_by_method/loss3_q_and_boundary_ratio_methods_p2_q2.png`
- Observed existing figures: mean loss/boundary/roughness curves, delta grids,
  and delta time-space heatmaps.
- Observed scope limitation: only `p=2,q=2` has been run; no cross-P/Q figures
  exist yet. Requested trajectory index `115` was not included in batch `0..99`,
  so trajectory visualizations exist for indices `0`, `7`, `40`, and `47`.
- Observed figure count after update: `65`; GIF count remains `0`.
- Status document updated: `docs/loss3_direction_proposal_ablation_p2_q2_run_status_20260517.md`.
## 2026-05-17 - Direction-Proposal Ablation Extended P/Q Queue Started

- Status: extended P/Q queue has started and is currently running.
- Queue shell PID: `44098`; current Python PID: `44102`.
- Current pair: `p=1`, `q=1`; current method from log: `raw_add`.
- Logs:
  - `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_pq_queue.log`
  - `/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p1_q1_batch100.log`
- Output root for current pair:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1`.
- Observed GPU state at check: utilization `97%`, memory `9522 / 32768 MiB`.
- Observed log state: first pair has entered `raw_add`; no failure observed.
- Scope: this status check only verifies that the queue started normally; no P/Q
  pair has completed yet in this queue.
- Status document: `docs/loss3_direction_proposal_ablation_pq_queue_status_20260517.md`.

## 2026-05-17 - Direction-Proposal Ablation P2/Q2 Actual-Loss-Only Figure

- Status: visualization update only; no optimizer experiment was run.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source metrics: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/per_step_metrics.csv`.
- New figure:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/true_loss_progression/actual_loss3_q_mean_methods_p2_q2_clean_20260517.png`.
- Observed evidence: the new figure is drawn from `loss3_q_mean` values at the
  actual optimizer iterates `delta_k`; it does not use boundary-normalized loss
  or boundary-projected diagnostic loss.
- Preservation note: existing figures were kept in place and were not deleted
  or overwritten. Future additional plots should use new file names or new
  figure directories unless replacement is explicitly requested.
- Remaining work: use actual loss as the primary comparison in scientific
  summaries; treat boundary-normalized quantities only as optional diagnostics.

## 2026-05-18 - Direction-Proposal Ablation Figure Meaning Guide

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source run inspected: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source files inspected: `tools/run_loss3_direction_proposal_ablation.py` and `tools/run_batch_three_loss_loss_only.py`.
- Result document created: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Observed evidence: current completed figure set is for `p=2,q=2`; the main actual-loss figures are `figures/curves/loss3_q_mean_vs_step.png` and `figures/true_loss_progression/actual_loss3_q_mean_methods_p2_q2_clean_20260517.png`.
- Interpretation note: `boundary_ratio` is a constraint-radius diagnostic, not a loss; `boundary_loss3_q_mean` is a boundary-rescaled diagnostic and should not be used as the primary convergence/result curve.
- Remaining work: once additional P/Q runs complete, create corresponding actual-loss-first summaries and avoid leading with boundary-normalized diagnostic plots.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Progress Inspection

- Status: inspection/status update only; no new experiment was launched and no figures were deleted or overwritten.
- Queue shell PID observed: `44098`; current Python PID observed: `48371`.
- Observed current pair: `p=1`, `q=inf`; current log shows the run has entered `raw_add`, `unit_raw_add`, and `steepest_add`.
- Observed completed output roots:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1`, manifest status `completed`, `40` PNG figures.
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2`, manifest status `completed`, `40` PNG figures.
- Observed in-progress output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_qinf`, manifest status `run_started`, `0` PNG figures at the check.
- Result/status document updated: `docs/loss3_direction_proposal_ablation_pq_queue_status_20260517.md`.
- Figure guide updated: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Remaining work: continue monitoring the queue; only interpret P/Q pairs after their manifest reports completion and figures exist.

## 2026-05-18 - Direction-Proposal Ablation Mean/Std Clean Loss Figures

- Status: visualization update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source metrics inspected:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/per_step_metrics.csv`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1/per_step_metrics.csv`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2/per_step_metrics.csv`
- Observed evidence: `loss3_q_finite_count` is `100` for inspected per-step rows, and `per_sample_step_metrics.csv` has 100 raw-add step-0 sample positions for each completed run inspected. Existing old line plots use `loss3_q_mean` without plotting `loss3_q_std`.
- New figures generated under `figures/actual_loss_with_std_clean/` for completed runs `p=2,q=2`, `p=1,q=1`, and `p=1,q=2`:
  - `actual_loss3_q_mean_std_core3.png`
  - `actual_loss3_q_mean_std_power_replacement_variants.png`
- Interpretation note: shaded bands are `loss3_q_mean +/- loss3_q_std` across the 100 samples at each step, not uncertainty across repeated seeds.
- Result/guide document updated: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Remaining work: use the core-three shaded plots as the default first-read comparison and keep all-method plots as diagnostic ablation figures.

## 2026-05-18 - Direction-Proposal Ablation Final Delta Similarity Analysis

- Status: completed post-processing analysis; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source script added: `tools/analyze_final_delta_similarity.py`.
- Source arrays:
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/final_deltas.npz`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1/final_deltas.npz`
  - `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2/final_deltas.npz`
- Outputs written under each completed run: `final_delta_similarity/20260518_final_delta_similarity/`, including per-sample pairwise CSV, summary CSV, mean cosine matrices, relative L2 matrices, and heatmap PNGs.
- Result document created: `docs/loss3_final_delta_similarity_results_20260518.md`.
- Observed evidence: `p=2,q=2` has exact final-delta equivalences among `unit_raw_add`, `steepest_add`, and `power_add__objective_gradient`, and among `raw_replace`, `steepest_replace`, and `power_replace__objective_gradient`.
- Observed evidence: `p=1,q=1` and `p=1,q=2` keep exact equivalence between `steepest_replace` and `power_replace__objective_gradient`, but `unit_raw_add` and `steepest_add` are not similar.
- Inference: for `p=2`, the L2 steepest direction equals normalized raw gradient, so several method labels are redundant. For `p=1`, the Lp geometry changes the steepest direction, so the same redundancy does not hold.
- Remaining work: repeat this analysis for later P/Q pairs once their manifests complete and `final_deltas.npz` exists.

## 2026-05-18 - Direction-Proposal Ablation Generalized Power Naming Clarification

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source files inspected: `tools/run_batch_three_loss_loss_only.py` and `tools/run_loss3_direction_proposal_ablation.py`.
- Result document created: `docs/loss3_generalized_power_naming_clarification_20260518.md`.
- Observed evidence: the previous three-method runner implements `generalized_power` as `steepest_direction(autograd(loss3_q), p)` followed by `project_delta(epsilon * direction, epsilon, p)`.
- Observed evidence: the new ablation's `power_*__jvp_vjp`, `power_*__affine_jvp_vjp`, and `power_*__generalized_pq` variants compute directions through JVP/VJP of the residual Jacobian rather than directly using the objective-gradient direction.
- Interpretation: when referring to the user's earlier three-method comparison, `generalized_power` should map to `steepest_replace` / `power_replace__objective_gradient`; JVP/VJP variants should be described separately as P-Q operator-power direction variants.
- Remaining work: future plots and summaries should use clearer labels to avoid presenting these as one identical method family.

## 2026-05-18 - Direction-Proposal Ablation 3D Delta Surface Visualizations

- Status: completed post-processing visualization; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source script added: `tools/plot_delta_3d_surfaces.py`.
- Command used: `./adv_robust/bin/python tools/plot_delta_3d_surfaces.py --space-stride 4 --step-stride 1`.
- Source arrays: completed runs' per-method `trajectory_samples.npz` files under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/`.
- Observed trajectory shape example: `k` has 101 steps from `0` to `100`; `delta` has shape `(101, 4, 1024, 1)` for trajectory dataset indices `0`, `7`, `40`, and `47`.
- Output directories: `figures/delta_surfaces_3d_20260518/` under each completed run.
- Observed output counts: `p=1,q=1` 28 PNGs, `p=1,q=2` 28 PNGs, `p=1,q=inf` 28 PNGs, `p=2,q=2` 44 PNGs; total `128` 3D surface PNGs.
- Observed skipped run: `p=2,q=1` had manifest status `run_started`, so it was not included in the completed-run visualization batch.
- Result document created: `docs/loss3_delta_3d_surface_visualizations_20260518.md`.
- Figure guide updated: `docs/loss3_direction_proposal_ablation_figure_guide_20260518.md`.
- Remaining work: generate corresponding 3D surfaces for later P/Q pairs after their manifests complete.

## 2026-05-18 - Direction-Proposal Ablation Method Formulas And Equivalences

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source files inspected: `tools/run_batch_three_loss_loss_only.py`, `tools/run_loss3_direction_proposal_ablation.py`, and completed `p=2,q=2/final_deltas.npz` method order.
- Result document created: `docs/loss3_optimizer_method_formulas_and_equivalences_20260518.md`.
- Observed evidence: `p=2,q=2` method order begins with seven objective-gradient methods: `raw_add`, `unit_raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`, `power_add__objective_gradient`, and `power_replace__objective_gradient`.
- Formula conclusion: for `p=2`, `s_2(g)=g/||g||_2`, so `unit_raw_add`, `steepest_add`, and `power_add__objective_gradient` have identical updates, and `raw_replace`, `steepest_replace`, and `power_replace__objective_gradient` have identical updates.
- Observed similarity evidence: completed `p=2,q=2` final-delta analysis gives mean cosine `1.000000` and mean relative L2 `0.000000` for those duplicate-formula pairs.
- Inference: high similarity among the first seven p=2,q=2 methods is mainly due to duplicated objective-gradient formulas plus the L2 identity between normalized raw gradient and L2-steepest direction; it is not a universal claim for p=1, where `unit_raw_add` and `steepest_add` are not similar.
- Remaining work: rename/relabel future plots to collapse exact duplicate methods into representative method groups before presentation.

## 2026-05-18 - Direction-Proposal Ablation Readable Formula Document

- Status: documentation formatting update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_optimizer_method_formulas_readable_20260518.md`.
- Purpose: rewrite the optimizer formulas using readable displayed math instead of code-block-style pseudocode.
- Observed evidence preserved: the readable document states the p=2 equivalences `unit_raw_add = steepest_add = power_add__objective_gradient` and `raw_replace = steepest_replace = power_replace__objective_gradient`, and cites the final-delta similarity values showing cosine `1.000000` and relative L2 `0.000000` for duplicate-formula pairs.
- Remaining work: use the readable formula document as the default reference when explaining optimizer methods to avoid confusion from code-like formatting.

## 2026-05-18 - Direction-Proposal Ablation Plain-Language Formula Document

- Status: documentation formatting update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_optimizer_method_formulas_plain_language_20260518.md`.
- Purpose: rewrite method formulas without LaTeX math syntax or code-block formatting, using plain-language equations such as `delta_next = project(delta_current + alpha * direction)`.
- Observed evidence preserved: the document explains that for `p=2`, normalized raw gradient equals L2-steepest direction, which makes `unit_raw_add = steepest_add = power_add__objective_gradient` and `raw_replace = steepest_replace = power_replace__objective_gradient`.
- Remaining work: use this plain-language document when the rendered interface does not display LaTeX formulas cleanly.

## 2026-05-18 - Direction-Proposal Ablation P2/Q2 Method Equivalence Summary

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`.
- Related result document updated: `docs/loss3_final_delta_similarity_results_20260518.md`.
- Source numeric table: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/final_delta_similarity/20260518_final_delta_similarity/final_delta_pairwise_similarity_summary.csv`.
- Observed evidence: `unit_raw_add`, `steepest_add`, and `power_add__objective_gradient` have identical final deltas for `p=2,q=2` with pairwise cosine `1.000000` and relative L2 `0.000000`.
- Observed evidence: `raw_replace`, `steepest_replace`, and `power_replace__objective_gradient` have identical final deltas for `p=2,q=2` with pairwise cosine `1.000000` and relative L2 `0.000000`.
- Observed evidence: `raw_add` versus `steepest_add` has mean cosine `0.858049` and mean relative L2 `0.384210`, so it is similar but not identical.
- Inference: the high similarity among the first seven methods in the `p=2,q=2` heatmap is mainly due to the L2 identity between normalized raw gradient and L2-steepest direction plus duplicate objective-gradient method labels; future presentation plots should collapse exact duplicates into representative method groups.
- Remaining work: use the concise p=2/q=2 equivalence summary when explaining the crowded heatmap and when designing cleaned presentation figures.

## 2026-05-18 - Direction-Proposal Ablation P-Norm Equivalence Rules

- Status: documentation/interpretation update plus post-processing for newly completed runs; no optimizer experiment was launched and no existing figures were deleted or overwritten.
- Source script used for new post-processing: `tools/analyze_final_delta_similarity.py`.
- New similarity outputs generated for newly completed runs: `p=1,q=inf` and `p=2,q=1` under `final_delta_similarity/20260518_final_delta_similarity/`.
- Result document created: `docs/loss3_p_norm_method_equivalence_rules_20260518.md`.
- Related documents updated with links: `docs/loss3_p2_q2_method_equivalence_summary_20260518.md` and `docs/loss3_final_delta_similarity_results_20260518.md`.
- Observed evidence: for completed `p=1,q=inf`, `unit_raw_add` vs `steepest_add` has mean cosine `0.344037` and mean relative L2 `1.776665`, so they are not equivalent; `steepest_replace` vs `power_replace__objective_gradient` remains exactly equivalent with cosine `1.000000` and relative L2 `0.000000`.
- Observed evidence: for completed `p=2,q=1`, `unit_raw_add` vs `steepest_add` has mean cosine `1.000000` and relative L2 `0.000000`, consistent with the p=2 rule.
- Inference: `p=2` creates additional exact equivalences because normalized raw gradient equals L2-steepest direction. For `p!=2`, those extra equivalences disappear; remaining exact matches are mostly duplicate method labels such as `steepest_replace = power_replace__objective_gradient` or the current `generalized_pq = pure_jvp_vjp` implementation.
- Remaining work: analyze `p=2,q=inf` after its manifest completes and `final_deltas.npz` exists.

## 2026-05-18 - Direction-Proposal Ablation JVP/VJP Power Method Explanation

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source files inspected: `tools/run_loss3_direction_proposal_ablation.py`.
- Result document created: `docs/loss3_jvp_vjp_power_methods_explanation_20260518.md`.
- Observed code evidence: objective-gradient methods use `grad = autograd(loss3_q)` and then `steepest_direction(grad,p)`, while JVP/VJP power methods compute `J v`, apply a q-side map, pull back with VJP, and then use `steepest_direction(vjp,p)`.
- Observed formula distinction: objective-gradient replacement uses `J^T phi_q(r_k)`, while pure JVP/VJP power uses `J^T phi_q(J v_k)`; affine JVP/VJP power uses `J^T phi_q(r_k + rho_k J v_k)`.
- Observed similarity evidence: completed `p=2,q=2` final-delta analysis gives `steepest_replace` vs `power_replace__pure_jvp_vjp` mean cosine `0.121156` and relative L2 `1.293200`, so these are not the same method in practice.
- Inference: JVP/VJP variants are local operator-power direction methods and should be interpreted separately from the earlier objective-gradient generalized-power replacement method.
- Remaining work: use clearer labels in future figures to separate objective-gradient generalized power from JVP/VJP operator-power variants.

## 2026-05-18 - Direction-Proposal Ablation Minimal Core Question Summary

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_minimal_core_question_answer_20260518.md`.
- Source run summarized: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source tables: `per_step_metrics.csv` and `final_delta_similarity/20260518_final_delta_similarity/final_delta_pairwise_similarity_summary.csv`.
- Observed evidence: for `p=2,q=2`, `raw_replace` and `steepest_replace` have identical final deltas with mean cosine `1.000000` and mean relative L2 `0.000000`.
- Observed evidence: final actual loss at `k=100` is `6.804780` for both `raw_replace` and `steepest_replace`, compared with `5.389592` for `raw_add` and `6.377158` for `steepest_add`.
- Observed evidence: replacement methods in `p=2,q=2` did not show higher recorded roughness than additive methods; `raw_replace`/`steepest_replace` had high-frequency ratio `8.10e-10` and first-derivative L2 `0.217019`.
- Inference: the original core question is answered for `p=2`: replacing PGD's additive update by direct boundary replacement using the normalized gradient becomes the same as generalized-power-style steepest replacement. For `p!=2`, the exact `raw_replace` comparison remains to be run because the completed `pq_key` runs omitted `raw_replace`.
- Remaining work: run only the minimal method set `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace` for non-2 p values if the user wants the same conclusion across P/Q geometry.

## 2026-05-18 - Direction-Proposal Ablation JVP Bias-Term Clarification

- Status: documentation/interpretation update only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document updated: `docs/loss3_jvp_vjp_power_methods_explanation_20260518.md`.
- Clarification: the actual local residual model includes a bias/base residual term `b_k = r(x_k)`, so the local objective is closer to `||b_k + J_k v||_q` than `||J_k v||_q`.
- Interpretation: pure JVP/VJP uses `J_k v_k` and therefore studies local operator amplification while ignoring the current residual bias direction; affine JVP/VJP includes `r(x_k) + rho_k J_k v_k` and is closer to the biased local objective, but still differs from direct objective-gradient replacement.
- Remaining work: keep pure/affine JVP/VJP variants out of the minimal core experiment unless the research question is specifically about operator-power directions.

## 2026-05-18 - Direction-Proposal Ablation raw_replace vs steepest_replace Clarification

- Status: documentation clarification only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document updated: `docs/loss3_p2_q2_method_equivalence_summary_20260518.md`.
- Clarification: `raw_replace` in the ablation is not ordinary additive PGD walking to the boundary; it is a direct boundary replacement using the normalized raw-gradient direction.
- Formula clarification: `raw_replace` uses normalized `g_k`, while `steepest_replace` uses `s_k`; these coincide for `p=2` because `s_k = g_k / ||g_k||_2`, but they should not be assumed equal for `p!=2`.
- Remaining work: keep future labels explicit, e.g. `unit_raw_replace` rather than `raw_replace`, to avoid this ambiguity.

## 2026-05-18 - Direction-Proposal Ablation Selected Core Findings Summary

- Status: documentation consolidation only; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Result document created: `docs/loss3_selected_core_findings_summary_20260518.md`.
- Related document updated: `docs/loss3_minimal_core_question_answer_20260518.md`.
- Observed evidence consolidated: for `p=2,q=2`, `raw_replace` and `steepest_replace` have identical final deltas with mean cosine `1.000000` and mean relative L2 `0.000000`; both have final actual loss mean `6.804780`, boundary ratio `1.000000`, high-frequency ratio `8.10e-10`, and first-derivative L2 `0.217019`.
- Observed evidence consolidated: `raw_add` has final actual loss mean `5.389592`; `steepest_add` has final actual loss mean `6.377158`; `raw_add` vs `steepest_add` has mean cosine `0.858049`; `steepest_add` vs `steepest_replace` has mean cosine `0.530025`.
- Interpretation consolidated: for `p=2`, normalized raw-gradient direction equals L2-steepest direction, explaining exact equivalence groups; for `p!=2`, that equivalence should not be assumed.
- Remaining work: run the minimal four-method set `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace` for non-2 p values if the user wants the same boundary-replacement conclusion beyond `p=2`.

## 2026-05-18 - Direction-Proposal Ablation Simplified Core Figure Folder

- Status: completed post-processing visualization; no optimizer experiment was run and no existing figures were deleted or overwritten.
- Source run: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Output folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2`.
- Output contents: only PNG images; generated files are `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, and `03_final_delta_core4_selected_samples.png`.
- Methods plotted: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace` only.
- Result document created: `docs/loss3_simplified_core_visualizations_20260518.md`.
- Related summary updated: `docs/loss3_selected_core_findings_summary_20260518.md`.
- Scope note: this simplified four-method folder is currently only for `p=2,q=2`, because other completed P/Q runs omitted `raw_replace` and therefore cannot produce the same four-method comparison without a minimal additional run.
- Remaining work: run the minimal four-method set for non-2 p values before generating matching simplified folders for those P/Q cases.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:18 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `65319`, running `p=inf,q=1` with methods `pq_key`.
- Observed GPU state: utilization `97%`, memory `10740 / 32768 MiB`.
- Observed completed pairs: `p=1,q=1`, `p=1,q=2`, `p=1,q=inf`, `p=2,q=1`, `p=2,q=2`, and `p=2,q=inf` have manifest status `completed` and root `final_deltas.npz`.
- Observed current pair: `p=inf,q=1` has manifest status `run_started`; six of seven method directories have 101 per-step rows through `k=100`; `power_replace__generalized_pq` has started but had not written per-step rows at the check.
- Observed remaining queue items after current pair: `p=inf,q=2` and `p=inf,q=inf`.
- Result/status document updated: `docs/loss3_direction_proposal_ablation_pq_queue_status_20260517.md`.
- Remaining work: wait for queue completion, then run any needed post-processing/summary updates for the newly completed P/Q pairs.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:20 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `71831`, running `p=inf,q=2` with methods `pq_key`.
- Observed GPU state: utilization `99%`, memory `9522 / 32768 MiB`.
- Observed log evidence: queue log has advanced to `p=inf,q=2`; current run log shows method `raw_add` has started.
- Observed process evidence: no separate plotting/post-processing command such as `plot_delta_3d_surfaces.py` or `analyze_final_delta_similarity.py` was running at this check.
- Remaining work: wait for `p=inf,q=2` and queued `p=inf,q=inf` to finish, then run any desired post-processing/summary plots for those newly completed P/Q pairs.


## 2026-05-18 - Direction-Proposal Ablation Core-Four Data Availability Check 01:22 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Question checked: whether the same simplified four-method plots can be generated for P/Q pairs beyond `p=2,q=2`.
- Observed evidence: local completed P/Q directories other than `p=2,q=2` generally contain `raw_add`, `steepest_add`, and `steepest_replace`, but not `raw_replace`.
- Observed evidence: `p=inf,q=2` is currently still running and has only `raw_add` visible so far; `p=inf,q=inf` has not started in the queue output yet.
- Inference: same-style four-line plots for non-2 p values need a minimal additional run including `raw_replace`. For `p=2` pairs, `raw_replace` should be equivalent to `steepest_replace` by L2 geometry, but that would be an inference unless the method is actually run or explicitly duplicated and labeled as inferred.
- Remaining work: wait for the current queue to finish, then decide whether to run a small core-four补跑 for selected P/Q pairs.


## 2026-05-18 - Direction-Proposal Ablation p=2,q=2 Figure Location Lookup

- Status: inspection/path lookup only; no experiment was launched and no existing figures were deleted or overwritten.
- Simplified p=2,q=2 figure folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2`.
- Observed simplified PNGs: `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, and `03_final_delta_core4_selected_samples.png`.
- Full p=2,q=2 figure tree: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures`.
- Observed process state during lookup: background P/Q queue was still running `p=inf,q=2`; no plotting process was observed.


## 2026-05-18 - Direction-Proposal Ablation p=2,q=2 Smoothness Interpretation Figures

- Status: completed post-processing visualization and documentation; no optimizer experiment was launched and no existing figures were deleted or overwritten.
- Source run: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Source numeric files: root/per-method `per_step_metrics.csv`, root `per_sample_step_metrics.csv`, and root `final_deltas.npz`.
- Output folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2`.
- Output PNGs: `01_final_smoothness_core4_mean_std_annotated.png`, `02_sample040_final_delta_with_smoothness_numbers.png`, and `03_sample040_smoothness_over_steps.png`.
- Result document created: `docs/loss3_delta_smoothness_metrics_interpretation_20260518.md`.
- Observed evidence: at final step `k=100`, `raw_replace` and `steepest_replace` have actual loss mean `6.804780`, first-derivative L2 mean `0.217019`, and total variation mean `2.685119`; `raw_add` has actual loss mean `5.389592`, first-derivative L2 mean `0.645722`, and total variation mean `8.573150`.
- Observed evidence: for dataset index `40`, final-step `raw_add` has first-derivative L2 `0.949167` and total variation `14.660805`, while `steepest_replace` has first-derivative L2 `0.250301` and total variation `2.685880`.
- Inference: the sample-40 visual impression that additive PGD creates a more jagged perturbation is supported by the derivative/variation metrics. In this observed `p=2,q=2` run, the GPI-style replacement reaches higher loss while also producing smoother final deltas by these metrics.
- Remaining work: repeat the same annotated smoothness plotting for other P/Q pairs after core-four data, especially `raw_replace`, is available.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:30 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `71831`, still running `p=inf,q=2` with methods `pq_key`.
- Observed GPU state: utilization `97%`, memory `10744 / 32768 MiB`.
- Observed process evidence: no separate plotting process was observed; current active workload is the optimizer/data run.
- Remaining work: wait for `p=inf,q=2` and then `p=inf,q=inf` to finish.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:33 UTC

- Status: inspection/status update only; no new experiment was launched and no existing figures were deleted or overwritten.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `71831`, still running `p=inf,q=2` with methods `pq_key`.
- Observed GPU state: utilization `98%`, memory `10744 / 32768 MiB`.
- Observed output status: completed pairs through `p=inf,q=1` have root `final_deltas.npz`; `p=inf,q=2` remains `run_started` and root `final_deltas.npz` does not yet exist.
- Observed current method progress for `p=inf,q=2`: `raw_add`, `unit_raw_add`, `steepest_add`, `steepest_replace`, `power_replace__objective_gradient`, and `power_replace__pure_jvp_vjp` each have 101 per-step rows through `k=100`; `power_replace__generalized_pq` has started in the log but has not written per-step rows yet.
- Inference: the queue is not finished; after the current `p=inf,q=2` run, `p=inf,q=inf` remains queued.
- Remaining work: wait for `p=inf,q=2` and `p=inf,q=inf` to complete before post-processing/plotting the new P/Q results.


## 2026-05-18 - Direction-Proposal Ablation Simplified Smoothness Figure Redraw 01:41 UTC

- Status: completed post-processing figure redraw/cleanup; no optimizer experiment was launched.
- Source run: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2`.
- Redrawn output: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2/02_delta_quality_core4_metrics.png`.
- Redraw details: the figure now shows `boundary_ratio`, `high_frequency_energy_ratio`, `first_derivative_l2`, and `total_variation` as mean curves with `+/- 1` sample-standard-deviation shading over the 100 samples; it also includes final-step mean `+/-` sample std bars with labels.
- Deleted per user request as redundant: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2/02_sample040_final_delta_with_smoothness_numbers.png` and `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2/03_sample040_smoothness_over_steps.png`.
- Remaining annotated record: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_smoothness_annotated_figures_20260518_p2_q2/01_final_smoothness_core4_mean_std_annotated.png`.
- Result documents updated: `docs/loss3_simplified_core_visualizations_20260518.md` and `docs/loss3_delta_smoothness_metrics_interpretation_20260518.md`.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Runtime Estimate 01:41 UTC

- Status: inspection/status update only; no new experiment was launched.
- Current background queue: still running, shell PID `44098`.
- Current experiment process: PID `79740`, running final queued pair `p=inf,q=inf` with methods `pq_key`.
- Observed current method progress: `raw_add`, `unit_raw_add`, and `steepest_add` are complete through `k=100`; `steepest_replace` has started; `power_replace__objective_gradient`, `power_replace__pure_jvp_vjp`, and `power_replace__generalized_pq` have not started.
- Inference from previous `p=inf` pair timings: remaining runtime from 2026-05-18 01:41 UTC is approximately 10-12 minutes, with expected finish around 2026-05-18 01:51-01:54 UTC if speed remains similar.
- Remaining work: wait for `p=inf,q=inf` to complete, then post-process/plot any newly completed P/Q results requested by the user.


## 2026-05-18 - Direction-Proposal Ablation Simplified Smoothness Figure Redraw 01:48 UTC

- Status: completed post-processing figure redraw; no optimizer experiment was launched.
- Redrawn output: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2/02_delta_quality_core4_metrics.png`.
- Redraw detail: changed the second simplified figure to a wide/flat layout. The left side now has four horizontally wide mean-curve panels with `+/- 1` sample-standard-deviation shading; the right side has compact final-step mean `+/-` sample std summaries.
- Output check: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_simplified_figures_20260518_p2_q2` still contains only `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, and `03_final_delta_core4_selected_samples.png`.
- Observed data availability: other P/Q output directories generally have `raw_add`, `steepest_add`, and `steepest_replace`, but not `raw_replace`; only `p=2,q=2` currently has all four core methods actually run.
- Inference: matching four-method simplified figures for non-2 p values require a minimal `raw_replace` run. For `p=2` pairs, `raw_replace` can be inferred from `steepest_replace` by L2 geometry, but should be labeled as inferred unless actually run.

## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Status Check 01:48 UTC

- Status: inspection/status update only; no new experiment was launched.
- Current final queued pair: `p=inf,q=inf`, manifest status `run_started`, root `final_deltas.npz` absent.
- Observed method progress: all methods through `power_replace__pure_jvp_vjp` have 101 per-step rows through `k=100`; `power_replace__generalized_pq` directory exists but has not written per-step rows yet.
- Inference: the queue is on the final method of the final queued pair; likely remaining time is a few minutes if timing follows the previous `p=inf` pairs.


## 2026-05-18 - Direction-Proposal Ablation P/Q Queue Completion and Core-Four Availability 01:53 UTC

- Status: inspection/status update only; no new experiment was launched and no figures were generated or deleted.
- Observed process evidence: no active `run_loss3_direction_proposal_ablation.py` or queue shell process was observed.
- Observed log evidence: the final queued pair `p=inf,q=inf` reached `[done] ablation outputs written under ... p=inf_qinf`.
- Observed output evidence: all P/Q pairs in the queue have manifest status `completed` and root `final_deltas.npz` exists.
- Observed core-method availability: `p=2,q=2` has `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`; all other queued P/Q pairs have `raw_add`, `steepest_add`, and `steepest_replace` but are missing `raw_replace`.
- Inference: three-method simplified figures can be generated now for all completed P/Q pairs. Strict four-method simplified figures require a minimal `raw_replace`补跑 for pairs other than `p=2,q=2`. For `p=2` pairs, `raw_replace` can be inferred from `steepest_replace` by L2 geometry, but should be labeled as inferred unless actually run.
- Remaining work: decide whether to generate three-method figures now, inferred `p=2` four-method figures, or run a minimal `raw_replace`补跑 before generating strict four-method figures for all P/Q pairs.


## 2026-05-18 - Raw-Replace Backfill Runtime Check 02:04 UTC

- Status: inspection/status update only; no new experiment was launched and no figures were generated or deleted.
- Current backfill queue process: shell PID `86272`.
- Current Python process: PID `89734`, running `raw_replace` for `p=2,q=inf`.
- Observed GPU state: utilization `88-99%`, memory `9522 / 32768 MiB`.
- Observed completed backfill pairs: `p=1,q=1`, `p=1,q=2`, `p=1,q=inf`, and `p=2,q=1`, each with `final_deltas.npz` and 101 `raw_replace` per-step rows.
- Observed current pair: `p=2,q=inf` has started but had not yet written per-step rows at the check.
- Observed queued pairs after current: `p=inf,q=1`, `p=inf,q=2`, and `p=inf,q=inf`.
- Observed logging note: `logs/loss3_raw_replace_backfill_queue_20260518.log` was absent, but per-pair logs existed and were updating.
- Inference: completed backfill pairs are taking about 1.9 minutes each; from 2026-05-18 02:04 UTC, estimated remaining runtime is about 7-9 minutes, with expected finish around 2026-05-18 02:11-02:13 UTC if speed stays similar.
- Remaining work: wait for the raw-replace backfill to finish, then merge/plot strict four-method simplified figures for all P/Q pairs.


## 2026-05-18 - Raw-Replace Backfill Completion 02:17 UTC

- Status: completed inspection/status update; no new experiment was launched during this check and no figures were generated or deleted.
- Observed process evidence: no active `loss3_raw_replace_backfill`, `run_loss3_direction_proposal_ablation.py`, or `adv_robust/bin/python` experiment process was observed.
- Observed GPU state: utilization `0%`, memory `0 / 32768 MiB`.
- Observed log evidence: `logs/loss3_raw_replace_backfill_pinf_qinf_batch100.log` includes `[done] ablation outputs written under ... p_inf_qinf`.
- Observed output evidence: all eight raw-replace backfill pairs have manifest status `completed`, root `final_deltas.npz`, and `raw_replace/per_step_metrics.csv` with 101 rows through `k=100`.
- Backfill output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Inference: strict four-core-method plotting is now possible for all P/Q pairs by combining original runs with the corresponding raw-replace backfill outputs.
- Remaining work: generate the simplified four-method figure folders for each P/Q pair using the original run data plus raw-replace backfill data.


## 2026-05-18 - All-PQ Core Four Figure Set 02:17 UTC

- Status: completed post-processing visualization; no optimizer experiment was launched.
- Output folder: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_all_pq_four_figures_20260518`.
- Output contents: exactly four PNG files and no data files: `01_actual_loss_all_pq_core4_mean_std.png`, `02_boundary_ratio_all_pq_core4_mean_std.png`, `03_final_metrics_all_pq_core4_heatmaps.png`, and `04_sample040_final_delta_all_pq_core4.png`.
- Source data: original runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517` plus raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Methods plotted: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- P/Q pairs plotted: `p=1,q=1`, `p=1,q=2`, `p=1,q=inf`, `p=2,q=1`, `p=2,q=2`, `p=2,q=inf`, `p=inf,q=1`, `p=inf,q=2`, and `p=inf,q=inf`.
- Observed file check: the output folder contains exactly 4 files, all PNG images.
- Result document created: `docs/loss3_core_all_pq_four_figures_20260518.md`.
- Inference: the plots are strict four-core-method plots because `raw_replace` now comes from actual backfill outputs for all P/Q pairs where it was previously missing.
- Remaining work: inspect the four images and, if desired, generate per-P/Q separate simplified folders using the same merged source data.


## 2026-05-18 - Per-PQ Core Four Figure Folders 02:18 UTC

- Status: completed post-processing visualization; no optimizer experiment was launched.
- Output root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_per_pq_four_figures_20260518`.
- Output structure: 9 P/Q subfolders (`p1_q1`, `p1_q2`, `p1_qinf`, `p2_q1`, `p2_q2`, `p2_qinf`, `pinf_q1`, `pinf_q2`, `pinf_qinf`).
- Output contents: each P/Q subfolder contains exactly four PNG files: `01_actual_loss_core4_mean_std.png`, `02_delta_quality_core4_metrics.png`, `03_final_delta_core4_selected_samples.png`, and `04_final_smoothness_core4_mean_std.png`.
- Observed file check: 9 subfolders, 4 PNG files per subfolder, 36 total files; all files are PNG images.
- Source data: original P/Q runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517` plus raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Methods plotted: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- Result document created: `docs/loss3_core_per_pq_four_figures_20260518.md`.
- Inference: the requested per-P/Q figure organization is now available; every P/Q pair has its own folder with the same four-figure structure as the revised `p=2,q=2` style.


## 2026-05-18 - Core Four Method/PQ Numerical Analysis

- Status: completed numerical analysis from existing outputs; no optimizer experiment was launched and no figures were generated or deleted.
- Source data: original P/Q runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517` plus raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`.
- Generated numeric tables: `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_method_pq_analysis_20260518/core4_pq_method_summary.csv`, `core4_pq_winner_summary.csv`, and `core4_pq_metric_ranks.csv`.
- Result document created: `docs/loss3_core_method_pq_analysis_20260518.md`.
- Observed evidence: `p=2,q=2` is the cleanest balanced setting; `raw_replace` and `steepest_replace` tie for largest final loss, fastest early growth, and smoothest final delta metrics.
- Observed evidence: `p=1` settings show a tradeoff: `steepest_add` gives the largest final finite loss, `steepest_replace` grows fastest early, and `raw_replace` is smoothest by first-derivative L2 / total variation.
- Observed evidence: `p=inf` settings have nonfinite final losses for several additive methods and nonfinite intermediate behavior for replacement methods; `p=inf` is not recommended without fixing numerical stability.
- Inference: for a practical balance of high final loss, fast growth, and low-frequency/smooth perturbations, use `p=2,q=2` with `steepest_replace` / GPI; `p=2,q=1` is a useful secondary comparison, while `p=1` is mainly useful to illustrate the loss-smoothness tradeoff and `p=inf` is currently unsuitable.
- Remaining work: if needed, inspect the per-P/Q figure folders visually and add representative figures to a paper/report section.

## 2026-05-18 - Direction-Proposal Conclusions and 2026-05-19 Alpha Plan

- Status: completed documentation/planning update; no optimizer experiment was launched and no figures were generated or deleted.
- Result document created: `docs/loss3_direction_proposal_conclusions_and_20260519_plan.md`.
- Source evidence summarized: original P/Q runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517`, raw-replace backfill runs under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`, per-P/Q four-figure folders under `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_per_pq_four_figures_20260518`, and numerical analysis tables under `/workspace/NeuralOperatorRobustness2/forensics/loss3_core_method_pq_analysis_20260518`.
- Observed evidence recorded: `p=2,q=2` is the cleanest balanced case; `p=2,q=1` is a useful secondary tradeoff case; `p=1,*` shows final-loss vs smoothness tradeoffs; `p=inf,*` has nonfinite/degenerate behavior and should not be used as a main result without stability fixes.
- Inference recorded: P/Q selection is itself part of the experiment conclusion; `p=2,q=2` with `steepest_replace` / GPI should be the primary result.
- Next experiment recorded for 2026-05-19: keep `epsilon = 8`, increase `alpha`, and test whether additive methods close the speed gap once they reach the boundary quickly.
- Proposed alpha sweep recorded: baseline `0.3`, then `0.6`, `1.0`, `2.0`, `4.0`, optionally `8.0` if additive methods still reach the boundary too slowly.
- Proposed P/Q settings recorded: primary `p=2,q=2`, secondary `p=2,q=1`, optional `p=1,q=2` or `p=1,q=1`; avoid `p=inf,*` for the first alpha sweep.

## 2026-05-18 - Direction-Proposal Interpretation Update: No Universal Best Optimizer

- Status: completed documentation/interpretation update; no optimizer experiment was launched and no figures were generated or deleted.
- Result document updated: `docs/loss3_direction_proposal_conclusions_and_20260519_plan.md`.
- Observed evidence recorded: no single optimizer wins every P/Q setting by final loss; `steepest_add` wins several high-loss cases, replacement/GPI methods often win speed, and `p=inf` settings show nonfinite/degenerate behavior.
- Observed visual concern recorded: several final deltas look spike-like or highly localized, especially in P/Q settings that encourage sparse or extreme perturbations.
- Inference recorded: the current result should be framed as a tradeoff among final loss, convergence speed, P/Q geometry, and physical plausibility, not as a universal optimizer ranking.
- Mechanistic interpretation recorded: `p=1` can encourage sparse/Dirac-like perturbations, `q=inf` can focus optimization on extreme residual points, and `p=inf` gives a very large feasible set that can produce unstable/nonphysical inputs.
- Next-step recommendation recorded: the 2026-05-19 alpha sweep should evaluate smoothness and spike behavior alongside final loss, and future experiments may need smoothness penalties, spectral low-pass parameterization, or an explicit smoothness budget.

## 2026-05-18 - GitHub and R2 Sync for Direction-Proposal Ablation Work

- Status: completed code/documentation push and R2 artifact sync.
- GitHub branch: `vast-ai`.
- Git commit pushed: `ab5902c` (`Add loss3 direction proposal ablation analysis`).
- GitHub remote: `origin` / `YifeiSun01/NeuralOperatorRobustness2`.
- Files committed to GitHub: experiment ledger, loss3 direction-proposal Markdown documents under `docs/`, and analysis/plotting/running scripts under `tools/`.
- Generated experiment artifacts were intentionally not committed to Git.
- R2 bucket/prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`.
- R2 synced paths: `docs/`, `tools/`, `EXPERIMENT_LEDGER.md`, `logs/`, and `forensics/`.
- R2 verification: key docs were present, including `loss3_direction_proposal_conclusions_and_20260519_plan.md`, `loss3_core_method_pq_analysis_20260518.md`, and `loss3_core_per_pq_four_figures_20260518.md`.
- R2 verification: key tools were present, including `run_loss3_direction_proposal_ablation.py`, `analyze_final_delta_similarity.py`, and `plot_delta_3d_surfaces.py`.
- R2 verification: `forensics/loss3_core_per_pq_four_figures_20260518` contained 36 objects, matching 9 P/Q folders times 4 PNGs.
- R2 verification: `forensics/loss3_optimizer_direction_proposal_ablation_20260517` contained 1698 objects and about 1.105 GiB.
- R2 verification: `forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518` contained 176 objects and about 127.5 MiB.
- Remaining local untracked files: generated `forensics/` artifacts remain untracked by Git by design; they are stored in R2.

## 2026-05-20 - Loss3 Clean Visualization Export and 4x5 Curve Relayout

- Status: completed post-processing visualization/export; no optimizer experiment was launched.
- Code changed: `tools/plot_loss3_alpha_epsilon_core4_visuals.py` now chooses 5 columns for 20 alpha/epsilon panels, producing a clean 4x5 layout for the multi-setting curve figures.
- Regenerated visual roots: `forensics/loss3_alpha_epsilon_core4_visuals_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520`, and `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520`.
- Regenerated/new similarity roots: `forensics/loss3_alpha_epsilon_core4_delta_similarity_p1_q2_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_delta_similarity_p1_qinf_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_qinf_stopped_100steps_20260520`, and `forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520`.
- Clean image-only export folder: `forensics/loss3_visuals_clean_export_20260520_2206`.
- Export contents observed: 363 image files total; verification found no non-image files in the export folder. The folder contains copied `.png` and `.gif` files only.
- Key source data: existing completed/partial stopped artifacts under `forensics/loss3_alpha_epsilon_core4_sweep_20260519`, `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`, `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520`, and baseline GIF-trace artifacts under `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520`.
- Observed evidence: regenerated 20-setting metric figures such as `loss3_q_mean_curves_with_boundary_markers.png` and `boundary_ratio_mean_curves.png` now have 4680x2736 pixel output, consistent with a 5-column by 4-row panel layout at the configured figure size and DPI.
- Result document created: `docs/loss3_visuals_clean_export_20260520.md`.
- Remaining work: inspect the clean export folder visually and select final figures for the report/paper; no additional optimizer run is pending from this export task.

## 2026-05-20 - Loss3 No-Std Curve Figure Regeneration

- Status: completed post-processing visualization/export; no optimizer experiment was launched.
- Reason: mean +/- std shading can expand the y-axis strongly when a method or P/Q setting has large variance or spike behavior, making the central loss/boundary/angle trajectories and boundary markers hard to read.
- Code changed: `tools/plot_loss3_alpha_epsilon_core4_visuals.py` now emits paired mean-curve outputs: a mean +/- std version and a mean-only/no-std-band version for `loss3_q_mean`, `boundary_ratio_mean`, and `delta_prev_angle_degrees_mean` when the angle metric exists. Dynamics triptychs also have paired no-std outputs.
- Regenerated visual roots: `forensics/loss3_alpha_epsilon_core4_visuals_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520`, `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520`, and `forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520`.
- New clean image-only export folder: `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std`.
- Export contents observed: 521 image files total, including 158 copied filenames containing `_no_std`; verification found no non-image files and no subdirectories in the export folder.
- Observed evidence: representative no-std 20-setting curve figures are readable PNGs at 4680x2736 pixels, preserving the previous 4x5 panel layout.
- Result document created: `docs/loss3_visuals_clean_export_with_no_std_20260520.md`.
- Inference: use the no-std figures to compare central optimizer trajectories and boundary-hit markers; use the std-shaded figures to inspect variability/spike behavior, especially for non-`p=2,q=2` settings.
- Remaining work: visually inspect the no-std/with-std pairs and choose which version to include in the writeup for each comparison.

## 2026-05-20 - Loss3 300-Step No-Std Export Verification

- Status: completed verification only; no optimizer experiment and no new plotting run was launched.
- Observed evidence: `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std` contains 71 copied files whose names start with `p2q2_300step_visuals`.
- Observed evidence: the 300-step source folder contains the main no-std curve outputs: `loss3_q_mean_curves_with_boundary_markers_no_std.png`, `boundary_ratio_mean_curves_no_std.png`, and `delta_prev_angle_degrees_mean_curves_no_std.png` under `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/`.
- Observed evidence: the clean export includes 300-step no-std dynamics triptychs, including `dynamics_triptychs_no_std` and `dynamics_triptychs_angle_available_no_std` copied filenames.
- Inference: the 300-step figures have already been regenerated with no-std variants and moved into the current clean export folder.

## 2026-05-20 - Loss3 Surprising Findings Synthesis

- Status: completed synthesis from existing local experiment artifacts and visual inspection notes; no optimizer experiment was launched.
- Result document created: `docs/loss3_surprising_findings_synthesis_20260520.md`.
- Source evidence referenced: p2q2 300-step analysis/visual roots, p!=q stopped 100-step analysis/visual roots, GPI early-step comparison, and clean std/no-std visual export `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std`.
- Observed evidence summarized: GPI/replacement reaches the p-norm boundary immediately but still grows loss afterward; GPI perturbation shape is close to its 300-step final shape by about 5-10 steps in the saved baseline samples; 300-step additive methods can sometimes exceed GPI final mean loss; LP-steepest additive PGD shows near-linear boundary-ratio growth while raw PGD curves/slows; p/q geometry, especially `q=inf`, can create spike-like perturbations.
- Inference recorded: the central mechanism is not just reaching the epsilon boundary, but moving directionally on the boundary. GPI/replacement is strong because it uses the full budget immediately and rotates aggressively on the boundary; LP-steepest fixes radial step normalization but remains additive; raw PGD has both radial desynchronization and angular inertia.
- Remaining work: for paper-level claims, pair no-std central trajectory plots with std-shaded variability plots, and separate p2q2 conclusions from p!=q geometry/spike conclusions.

## 2026-05-20 - Loss3 Tree-Style Figure Export

- Status: completed post-processing file organization/export; no optimizer experiment and no plotting run was launched.
- Reason: the previous clean export `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std` was flat and difficult to navigate with 521 images.
- New tree-style image export folder: `forensics/loss3_visuals_tree_export_20260520_with_no_std`.
- Result document created: `docs/loss3_visuals_tree_export_with_no_std_20260520.md`.
- Export structure: top-level folders `visuals/`, `similarity/`, `gpi_early_step_comparison/`, and `baseline_giftrace/`; visual folders are further split by `p2q2`, `pneq`, P/Q pair, step count, curve type, `with_std`/`no_std`, heatmaps, representative samples, delta grids, and dynamics triptychs.
- Export contents observed: 521 image files total and 158 filenames containing `_no_std`, matching the complete clean export count. Verification found no non-image files in the tree export folder.
- Example checked: `visuals/p2q2/300step/curves/loss_with_boundary_markers/with_std/loss3_q_mean_curves_with_boundary_markers.png` and `visuals/p2q2/300step/curves/loss_with_boundary_markers/no_std/loss3_q_mean_curves_with_boundary_markers_no_std.png` are now separated into paired folders.
- Remaining work: use the tree-style export as the main browsing folder; keep the flat export only as an archival all-images dump.

## 2026-05-20 - Loss3 Cross-PQ Surprising-Findings Validation

Status: completed post-processing analysis from existing local artifacts. No new optimizer/model experiment was launched in this step.

Source files and inputs:
- Analysis script: `tools/analyze_loss3_surprising_findings_validation.py`
- Prior synthesis: `docs/loss3_surprising_findings_synthesis_20260520.md`
- Existing p=q=2 300-step artifacts under the loss3 alpha/epsilon core4 p2q2 result directories.
- Existing p!=q 100-step stopped artifacts under the loss3 alpha/epsilon core4 pneq-q result directories.
- Existing similarity, per-step metric, peakiness, and visualization-export metadata tables.

Output files:
- Result Markdown: `docs/loss3_surprising_findings_validation_20260520.md`
- Tables directory: `forensics/loss3_surprising_findings_validation_20260520/`
- Main tables: `per_setting_method_validation.csv`, `pq_method_validation_rollup.csv`, `final_loss_winner_counts_by_pq.csv`, `angle_motion_winner_counts_by_pq.csv`, `boundary_arrival_winner_counts_by_pq.csv`, `delta_similarity_pair_rollup_by_pq.csv`, `equivalent_method_pairs_by_pq.csv`
- Exception tables: `replacement_nonpositive_post_boundary_gain_settings.csv`, `raw_vs_steepest_add_linearity_exceptions.csv`, `angle_winner_by_setting.csv`, `equivalent_method_pairs_by_setting.csv`, `near_high_cosine_method_pairs_by_setting.csv`, `low_cosine_selected_pairs_by_setting.csv`, `claim_exception_table_manifest.csv`

Key settings:
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- p=q=2 sweep uses the existing 20 alpha/epsilon settings with 300 steps.
- p!=q sweep uses existing 100-step artifacts for p/q combinations including p=1,q=2; p=1,q=inf; p=2,q=1; p=2,q=inf; and partial p=inf,q=1.

Observed from generated tables:
- The validation table contains 408 per-setting/method rows, 24 PQ/method rollup rows, and 36 PQ/pair similarity rollup rows.
- Boundary arrival is not the same as convergence: replacement methods often hit the boundary at step 1, but in p2q2, p2q1, and p2qinf they can still gain substantial loss afterward. However, this positive post-boundary gain is not universal: p1qinf has many weak or negative replacement post-boundary gain settings, and p1q2 has a small number of large-epsilon exceptions.
- GPI/replacement is consistently fast in boundary arrival and angular motion, but is not the universal 300-step final-loss winner. In p2q2 300-step data, `steepest_add` wins most final-mean-loss settings; replacement remains strongest as an early, stable, fast method.
- LP steepest additive updates usually produce a straighter, lower-variance boundary-ratio trajectory than raw PGD, while raw PGD is more curved and sample-dependent. This is a strong tendency, not a theorem; q=inf settings create many exceptions.
- Boundary-ratio standard deviation is a useful diagnostic: raw PGD generally has much larger std than LP steepest and replacement, while replacement-family std is near zero when the method stays exactly on the boundary.
- Replacement-family methods have the largest per-step delta angular motion in every checked setting. The winner is not always specifically `steepest_replace`; `raw_replace` can win or tie depending on P geometry.
- Final delta similarity is conditional. It is high in p2q2 and p2q1 for many method pairs, but p=1 and q=inf cases introduce many low-cosine or spike-like exceptions.
- P/Q geometry strongly affects perturbation realism. p2q1 looks comparatively natural in the checked artifacts; p1 and q=inf geometries can create localized spikes, especially for steepest/replacement directions.
- Exact method equivalence was observed for `raw_replace` and `steepest_replace` for all checked p=2 groups, q in {1,2,inf}. The same equivalence is not observed as a universal fact for p=1 or p=inf groups. p1q2 has a near-equivalence tendency between `raw_add` and `raw_replace`, but it is not exact for all settings.

Inference from the above evidence:
- The most robust summary is not "boundary solves the attack". It is: after the optimizer reaches the epsilon boundary, the decisive difference is how fast and how freely it can rotate direction along that boundary.
- GPI/replacement is best characterized as fast, stable, and strong early; not as an unconditional final-loss maximizer after long runs.
- Claims in a paper should be stated by geometry regime: p2q2/p2q1 support the cleanest story, while p=1 and q=inf require caveats about spikes, weaker similarity, and non-universal post-boundary gain.

Remaining work:
- If a paper statement needs full coverage for missing p=inf combinations, run those missing PQ sweeps explicitly; the current p=inf,q=1 evidence is partial.
- For publication figures, separate p2q2/p2q1 conclusions from p=1/q=inf caveat figures rather than collapsing them into one universal claim.

## 2026-05-20 - Loss3 GPI Theoretical Caveat Note

Status: completed interpretation update from existing validation results; no new experiment was run.

Updated result document:
- `docs/loss3_surprising_findings_validation_20260520.md`

Observed evidence referenced:
- GPI/replacement reaches the epsilon boundary immediately in the main p=2 regimes.
- GPI/replacement has much larger delta angular motion than raw PGD or LP-steepest additive PGD.
- Final delta shapes are often similar in p2q2/p2q1 despite different paths.
- 300-step additive methods sometimes match or exceed replacement final mean loss.

Inference recorded:
- Fast GPI behavior should be treated as a strong empirical surrogate effect, not as proof that Loss 3 is exactly a generalized-power objective.
- The likely mechanism is early alignment with a dominant local mode or local linearized/quadratic component of the loss, plus aggressive boundary-surface rotation.
- The result is theoretically suspicious enough that a paper should avoid saying GPI is the mathematically correct optimizer for Loss 3.

Remaining work:
- Add diagnostics for gradient-step cosine, local spectrum dominance, linear-model predicted gain versus actual gain, and boundary-tangent update decomposition if this mechanism needs to be defended rigorously.

## 2026-05-20 - Loss3 Surprising Findings Mathematical Interpretation

Status: completed theory-interpretation note from existing experiment summaries and method definitions; no new optimizer/model experiment was launched.

Source files and evidence:
- Method definitions inspected from `tools/run_loss3_direction_proposal_ablation.py` and `tools/run_batch_three_loss_loss_only.py`.
- Prior validation result: `docs/loss3_surprising_findings_validation_20260520.md`.
- Supporting method notes: `docs/loss3_jvp_vjp_power_methods_explanation_20260518.md` and `docs/loss3_generalized_power_naming_clarification_20260518.md`.

Output file:
- `docs/loss3_surprising_findings_math_explanation_20260520.md`

Observed evidence referenced:
- The current `steepest_replace`/`generalized_power` method uses the objective gradient's p-steepest direction plus replacement to the epsilon boundary; it is not the full JVP/VJP generalized P-Q power iteration unless explicitly using those variants.
- `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace` differ by direction normalization/steepest map and additive versus replacement proposal.
- p=2 makes raw-gradient normalization and p-steepest direction identical, explaining exact `raw_replace`/`steepest_replace` equivalence for checked p=2 groups.

Inference recorded:
- Boundary arrival is only active-constraint satisfaction, not constrained stationarity; post-boundary tangent/directional motion explains continued loss growth.
- Replacement/GPI is best understood as an aggressive local-linear full-budget surrogate optimizer, not as a guaranteed optimizer for the nonlinear Loss 3 objective.
- LP-steepest additive norm growth is straighter because its p-step size is normalized; raw PGD curves because gradient scale varies by sample and step.
- Boundary-ratio standard deviation diagnoses radial synchronization; angular-change metrics diagnose boundary direction search.
- P/Q geometry explains spike-prone settings: p=1 promotes sparse extreme points and q=inf promotes localized max-residual gradients.

Remaining work:
- To turn this interpretation into stronger evidence, measure gradient-step cosine, local Jacobian/Hessian spectral dominance, linearized predicted gain versus actual gain, and p=2 tangent-update components after boundary arrival.

## 2026-05-20 - Loss3 GPI Mechanism Hypothesis Probe

Status: completed post-processing mechanism probe from existing local artifacts; no neural-operator optimizer/model experiment was launched.

Source files and inputs:
- Probe script: `tools/probe_loss3_gpi_mechanism_hypotheses.py`
- p2q2 300-step sweep: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
- p!=q 100-step sweep: `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`
- Saved trajectory roots: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/` and `forensics/loss3_optimizer_direction_proposal_ablation_20260517/`
- Direction-proposal p2q2 eps8/alpha0.3 ablation root: `forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/`

Output files:
- Result Markdown: `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`
- Output directory: `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/`
- Tables: `mechanism_probe_by_setting_method.csv`, `mechanism_probe_rollup_by_pq_method.csv`, `mechanism_probe_rollup_by_method.csv`, `early_to_final_trajectory_probe.csv`, `hypothesis_tests.csv`, `qaware_power_variant_probe_p2q2_eps8_alpha0p3.csv`

Observed from generated outputs:
- The probe processed `102` setting roots, `408` setting/method rows, `441` trajectory-probe rows, and `11` q-aware/objective-gradient variant rows.
- In p2q2, replacement hits 99% boundary at mean step `1`, versus raw add `49.2` and steepest add `13.45`.
- In p2q2, replacement mean post-boundary gain is `3.412`; mean post-boundary angle is `30.68 deg`, compared with `0.246 deg` for raw add and `0.612 deg` for steepest add.
- Across all settings, the Pearson correlation between post-boundary gain and post-boundary angle motion is weak (`~0.041`), so angular motion alone is not sufficient to explain loss gain.
- In saved p2q2 trajectories, replacement becomes final-like early: baseline cosine to final is about `0.887` at step 5, `0.984` at step 10, and `0.995` at step 20; eps8/alpha0.3 direction-proposal trajectory is about `0.788`, `0.966`, and `0.988` at those steps.
- In the p2q2 eps8/alpha0.3 direction-proposal ablation, objective-gradient replacement final mean loss is `6.805`, while affine JVP/VJP replacement is `3.842` and pure JVP/VJP replacement is `2.324`.
- In p1qinf, steepest methods have peakiness around `30` and top-1 energy fraction around `0.86-0.96`, while p2q2 methods have peakiness around `3.6-4.1` and top-1 energy fraction around `0.014-0.018`.

Inference from the above evidence:
- The successful current GPI label is better interpreted as objective-gradient replacement: a local-linear full-budget surrogate plus aggressive boundary-direction search.
- It should not be described as a literal generalized P-Q power method or exact optimizer for full nonlinear Loss 3.
- p2q2-like regimes appear to have a dominant useful direction that replacement reaches within a few steps; p1/qinf regimes can create large angular motion that is not useful and often spike-like.

Remaining work:
- For stronger mechanism evidence, run a GPU diagnostic that records gradient-step cosine, local Jacobian/Hessian spectral dominance, and local-linear predicted gain versus actual gain along selected trajectories.

## 2026-05-20 - Loss3 Boundary Geometry Derivation and Validation Clarification

Status: completed explanatory/validation note from existing artifacts; no new optimizer/model experiment was launched.

Output file:
- `docs/loss3_boundary_geometry_derivation_and_validation_20260520.md`

Source evidence referenced:
- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`
- `forensics/loss3_gpi_mechanism_hypothesis_probe_20260520/tables/`
- Prior final-delta similarity and p/q concentration metrics from existing loss3 analyses.

Observed evidence recorded:
- p2q2 replacement hits 99% boundary at mean step `1`; raw add at `49.2`; steepest add at `13.45`.
- p2q2 replacement post-boundary angular motion is about `30.68 deg`, compared with `0.246 deg` for raw add and `0.612 deg` for steepest add.
- p2q2 replacement post-boundary gain is `3.412`.
- Existing trajectory probes show p2q2 replacement reaches high cosine to final delta by steps 5-20.
- p1qinf and p1q2 concentration metrics support the spike explanation for p=1/q=inf-like geometries.

Inference recorded:
- For p=2, `(I - u u^T) grad L` is the tangent-plane projection of the gradient on the L2 boundary; nonzero tangent projection means loss can still increase while staying on the boundary.
- LP-steepest additive norm growth is only approximately linear; it depends on normalized step size plus stable positive radial alignment.
- Additive angular motion after boundary is O(alpha/epsilon), while replacement has no alpha/epsilon small factor and can rotate directly to the new selected direction.
- Existing evidence supports but does not fully prove the dominant-direction hypothesis; a true spectral-mode claim needs a GPU Jacobian/Hessian or tangent-KKT diagnostic.

Remaining work:
- Implement a GPU diagnostic for p2 tangent KKT residual `||(I-u u^T) grad L|| / ||grad L||`, local-linear predicted gain, and local spectral dominance along selected trajectories.

## 2026-05-20 - Loss3 Mechanism Probe Plain-Language Walkthrough

Status: completed documentation clarification; no new experiment or post-processing run was launched.

Updated result file:
- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`

Clarification recorded:
- The mechanism probe is a post-processing analysis of existing `per_step_metrics.csv`, `final_deltas.npz`, and `trajectory_samples.npz` artifacts, not a new neural-operator run.
- A `setting root` means one completed P/Q/epsilon/alpha/steps experiment folder; the probe read 102 such roots and 408 core-method rows.
- A `trajectory probe` row is an early-step-to-final-delta comparison from saved trajectory arrays, not a new optimization run.
- The phrase boundary-surface direction change means loss increasing after `boundary_ratio ~= 1` while delta direction/angle continues changing.

Remaining work:
- If the mechanism needs to be verified beyond saved artifacts, run the proposed GPU diagnostics for tangent KKT residual, local-linear predicted gain, and local spectral dominance.

## 2026-05-20 - Loss3 p2q2 Tangent and Radial Geometry Probe

Status: completed post-processing diagnostic from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.

Source files and inputs:
- Script: `tools/probe_loss3_p2q2_tangent_and_radial_geometry.py`
- Sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
- Input tables: per-setting `per_sample_step_metrics.csv` files containing recorded autograd-gradient cosines, direction cosines, delta norms, losses, and per-step geometry metrics.

Output files:
- Result Markdown: `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`
- Output directory: `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/`
- Tables: `tangent_kkt_by_setting_method.csv`, `tangent_kkt_rollup_by_method.csv`, `radial_growth_by_setting_method.csv`, `radial_growth_rollup_by_method.csv`, `geometry_probe_correlations.csv`

Observed from generated outputs:
- Processed 20 p2q2 setting roots, 80 tangent setting/method rows, and 40 additive radial-growth rows.
- p2 tangent-gradient residual was computed as `sqrt(1 - cos(delta, grad)^2)`, equivalent to `||(I-u u^T) grad L|| / ||grad L||` when `u=delta/||delta||_2`.
- In p2q2 rollup, replacement/GPI tangent residual at boundary hit is about `0.855`, while raw add is about `0.397` and steepest add about `0.435`.
- Last gradient-bearing step tangent residual is about `0.480` for replacement/GPI, but only about `0.039` for raw add and `0.046` for steepest add.
- Post-boundary gain correlates with tangent residual at hit across alpha/epsilon settings: raw add `r=0.816`, steepest add `r=0.702`, replacement `r=0.622`.
- Replacement post-hit/last tangent residual is not positively correlated with gain (`r=-0.082` after hit and `r=-0.112` last update), so tangent motion is opportunity/capacity, not a complete predictor of useful loss growth.
- The p2 radial-growth formula matches the recorded pre-boundary radius increments with mean absolute error around `1e-7`.
- Steepest add has direction L2 mean `1.0` with CV about `3.3e-8`; raw add direction L2 CV is about `0.621`. Actual radial-increment CV is about `0.110` for steepest add and `0.621` for raw add.

Inference from the above evidence:
- The p2 boundary-stationarity story is directly supported: replacement reaches the boundary with a large tangent gradient component and therefore still has room to improve by rotating on the boundary.
- LP-steepest's straighter norm growth is mainly explained by normalized direction size and synchronized radial increments, not by cos-theta being universally more stable than raw PGD.
- Raw PGD's curved and variable radial progress is strongly tied to raw gradient norm variability.

Remaining work:
- For a stricter final convergence claim, rerun selected trajectories with gradient computation enabled at the final saved step and optionally store full gradient vectors for direct projection plots.

## 2026-05-20 - Loss3 Mechanism Validation and Landscape Visualization Plan

Status: completed planning/analysis document; no new neural-operator experiment was launched.

Output file:
- `docs/loss3_mechanism_validation_and_landscape_plan_20260520.md`

Source context inspected:
- Existing mechanism diagnostics: `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md` and `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`.
- Existing tools for landscape/curvature/ray diagnostics: `tools/run_loss3_2d_slice_planarity.py`, `tools/run_loss3_path_directional_curvature.py`, `tools/run_loss3_ray_profile*.py`, `tools/run_loss3_jacobian_subspace_rotation_path.py`, and local Jacobian/SVD analysis tools.

Observed evidence summarized:
- Existing saved metrics can already validate p2q2 tangent residual and radial-growth mechanisms.
- Existing tools can be adapted for core4/PQ ray profiles, 2D slices, boundary arc interpolation, directional curvature, and local dominant-mode/Jacobian probes.

Inference recorded:
- The next strongest validation is not another broad alpha/epsilon sweep; it is targeted landscape probes that visualize rays, 2D planes, boundary arcs, and local curvature around method deltas.
- Dominant-direction claims require stronger evidence from ray/2D/arc/Jacobian diagnostics; current early-to-final cosine evidence is suggestive but not a spectral proof.
- p/q spike claims can be tested by comparing landscape sharpness/curvature and concentration metrics across p2q2, p2q1, p2qinf, and p1qinf.

Remaining work:
- Implement or adapt `tools/run_loss3_pq_landscape_probe.py` to read core4 outputs and evaluate rays/slices/arcs on GPU after the required GPU verification.
- Run a small pilot before any large PQ landscape sweep.

## 2026-05-20 - Loss3 p2q2 Stepwise Geometry vs Next-Step Gain Probe

Status: completed post-processing diagnostic from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.

Source files and inputs:
- Script: `tools/probe_loss3_p2q2_stepwise_gain_geometry.py`
- Sweep root: `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
- Input tables: per-setting `per_sample_step_metrics.csv` files with recorded loss, delta/gradient cosines, direction cosines, projection shrink factors, and step geometry.

Output files:
- Result Markdown: `docs/loss3_p2q2_stepwise_gain_geometry_probe_20260520.md`
- Output directory: `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`
- Tables: `stepwise_geometry_mean_by_setting_method_step.csv`, `stepwise_geometry_correlations_by_method_scope.csv`, `stepwise_geometry_correlations_by_setting_method_scope.csv`
- Figures: `post_boundary_grad_tangent_ratio_vs_next_loss_gain.png`, `post_boundary_direction_tangent_ratio_vs_next_loss_gain.png`, `post_boundary_linear_gain_projected_vs_next_loss_gain.png`, `post_boundary_stepwise_correlation_bars.png`

Observed from generated outputs:
- Processed 20 p2q2 setting roots and 2,400,000 sample-step pairs.
- The diagnostic compares geometry at step `k` against immediate next-step gain `loss3_q[k+1] - loss3_q[k]`.
- Post-boundary `grad_tangent_ratio` correlates with next-step gain for additive methods: `raw_add r=0.337`, `steepest_add r=0.265`.
- Post-boundary projected local-linear gain is a stronger one-step predictor: `raw_add r=0.600`, `steepest_add r=0.299`.
- Post-boundary radial/signed ratio is negatively correlated with next-step gain for additive methods: `raw_add r=-0.327`, `steepest_add r=-0.222`.
- Replacement methods have weak post-boundary one-step correlations (`linear_gain_projected r=0.097`, `grad_tangent_ratio r=0.029`), despite large hit-step-to-final gains found in the previous tangent/radial probe.

Inference from the above evidence:
- Stepwise tangent projection is related to next-step loss growth, but it is only an opportunity measure. The actual projected local-linear gain better captures whether the chosen next step uses the tangent opportunity in a useful direction.
- The earlier conclusion remains consistent: replacement/GPI reaches the boundary with large tangent residual and has room to improve, but once it is rotating aggressively on the boundary, immediate gain is not explained by tangent magnitude alone.
- For LP-steepest norm growth, the data should not be interpreted as the direction being fixed. The stronger explanation is that the direction norm is fixed/normalized, so radial increments are much less variable than raw PGD, whose raw gradient norm varies strongly.

Remaining work:
- If needed, extend the same stepwise gain diagnostic to other p/q settings and add GPU landscape probes for ray/2D/arc curvature validation.

## 2026-05-20 - Loss3 Mechanism/Landscape Existing Audit, All-p2 Tangent Extension, and Core4/PQ Landscape Pilot

Status: completed audit plus one post-processing extension and one small GPU landscape evaluation pilot.

Source files and inputs:
- Audit/result docs inspected: `docs/loss3_ray_profile_ri_final_report_fno_nu0p001_20260516.md`, `docs/loss3_2d_slice_planarity_fno_nu0p001_steps100_dense_result_20260517.md`, `docs/loss3_directional_curvature_fno_nu0p001_steps100_samples5_20260517.md`, `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`, and related forensics roots.
- Post-processing script: `tools/probe_loss3_all_p2_tangent_geometry.py`
- GPU landscape script: `tools/run_loss3_core4_pq_landscape_probe.py`
- Existing core4 inputs: p2q2 baseline giftrace root and p2q1/p2qinf/p1qinf `eps=4, alpha=0.4` roots.

Output files:
- Audit doc: `docs/loss3_mechanism_landscape_existing_vs_new_audit_20260520.md`
- All-p2 tangent doc: `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- All-p2 tangent output: `forensics/loss3_all_p2_tangent_geometry_probe_20260520/`
- Core4/PQ landscape doc: `docs/loss3_core4_pq_landscape_probe_20260520.md`
- Core4/PQ landscape output: `forensics/loss3_core4_pq_landscape_probe_20260520/`

GPU verification for landscape pilot:
- `nvidia-smi` showed Tesla V100-SXM2-32GB, driver 570.211.01, CUDA 12.8 driver capability, and no active GPU processes before the run.
- `adv_robust/bin/python` reported PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, CUDA available, device `Tesla V100-SXM2-32GB`, compute capability `(7,0)`, PyTorch arch list including `sm_70`, successful CUDA matmul, JAX backend `gpu`, and JAX device `cuda:0`.
- The landscape manifest records GPU-only runtime details and no CPU fallback.

Observed from existing audit:
- Prior ray-profile, 2D-slice, curvature, and Jacobian/SVD experiments already exist and are reusable as background evidence.
- Those old experiments are mainly old `loss3_original_pgd` path diagnostics, not current core4/PQ method comparisons, so they cannot fully substitute for the current GPI/replacement mechanism checks.

Observed from all-p2 tangent extension:
- Processed 60 p=2 setting roots and 24,000 sample/method rows.
- Replacement hits the 99% boundary at step 1 in p2q1, p2q2, and p2qinf.
- Replacement tangent residual at boundary hit is high across q: about `0.873` for p2q1, `0.855` for p2q2, and `0.938` for p2qinf.
- p2qinf is the warning case: high tangent residual and large angular motion do not translate into large post-boundary gain.

Observed from GPU landscape pilot:
- Scope: four setting roots, samples `[0, 7, 40, 47]`, row counts ray `8000`, boundary arc `756`, 2D slice `1568`, curvature `64`.
- In baseline p2q2, replacement/GPI endpoint ray loss at epsilon is `1.481` for step 1, `3.494` for step 5, `3.664` for step 10, `3.679` for step 20, and `3.634` for final. This supports quick convergence to a final-like high-loss direction by around 5-10 steps.
- p2q2 boundary arcs between replacement and additive final directions stay high-loss, with minima around `98.8%` of the weaker endpoint.
- p2q1 also looks broadly connected and replacement has strong endpoint ray loss.
- p2qinf differs: endpoint final-direction winner is steepest_add rather than replacement, and some arcs dip more.

Inference from the above evidence:
- The dominant-ridge / rapid-final-like-direction story is supported for p2q2 and partly p2q1.
- The story must be qualified for q=inf and p1/qinf: high tangent opportunity and aggressive angular motion are not sufficient without useful landscape alignment.
- A paper-level dominant-mode claim still needs a new, expensive local Jacobian/SVD probe on current core4/PQ deltas; old Jacobian/SVD artifacts are background evidence only.

Remaining work:
- Optionally scale the landscape pilot to more samples/settings.
- Run targeted local Jacobian/SVD dominant-mode diagnostics for p2q2 baseline, p2qinf warning case, and p1qinf spike case only if a stronger spectral claim is needed.

## 2026-05-21 - Loss3 Correlation and Radial-Growth Clarification

Status: completed clarification from existing generated tables; no experiment was rerun.

Source files inspected:
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/geometry_probe_correlations.csv`
- `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/tables/radial_growth_rollup_by_method.csv`
- `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/tables/stepwise_geometry_correlations_by_method_scope.csv`

Output file:
- `docs/loss3_correlation_and_radial_growth_clarification_20260521.md`

Observed evidence:
- Setting-level hit-to-final correlations answer whether tangent residual at first boundary hit predicts total post-boundary gain across 20 p2q2 alpha/epsilon settings: raw_add `r=0.816`, steepest_add `r=0.702`, replacement/GPI `r=0.622`.
- Stepwise post-boundary correlations answer whether the geometry at each post-boundary step predicts immediate next-step gain: raw_add tangent ratio `r=0.337`, steepest_add `r=0.265`, replacement/GPI `r=0.029`.
- Post-boundary projected local-linear gain is a better one-step predictor for additive methods: raw_add `r=0.600`, steepest_add `r=0.299`, replacement/GPI `r=0.097`.
- Radial-growth rollup shows `steepest_add` direction L2 CV is about `3.3e-8`, but cos(theta) CV is `0.478`; raw_add direction L2 CV and actual radial increment CV are both about `0.621`.

Inference:
- The high hit-to-final correlations and lower stepwise correlations are not contradictory; they answer different statistical questions.
- Fixed update norm alone does not mathematically guarantee straight norm growth. The supported statement is that normalized update length plus mostly positive radial alignment makes steepest_add radial progress much steadier in the observed p2q2 data.

## 2026-05-21 - Loss3 Key Findings Master Summary

Status: completed consolidated Markdown record; no experiment was rerun.

Source files and inputs:
- Existing result docs under `docs/`, especially `docs/loss3_surprising_findings_synthesis_20260520.md`, `docs/loss3_surprising_findings_validation_20260520.md`, `docs/loss3_boundary_geometry_derivation_and_validation_20260520.md`, `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`, `docs/loss3_p2q2_stepwise_gain_geometry_probe_20260520.md`, `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`, `docs/loss3_core4_pq_landscape_probe_20260520.md`, and `docs/loss3_correlation_and_radial_growth_clarification_20260521.md`.
- Existing forensics roots including `forensics/loss3_p2q2_tangent_radial_geometry_probe_20260520/`, `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`, `forensics/loss3_all_p2_tangent_geometry_probe_20260520/`, and `forensics/loss3_core4_pq_landscape_probe_20260520/`.

Output file:
- `docs/loss3_key_findings_master_summary_20260521.md`

Observed evidence recorded:
- Boundary hit is not convergence; replacement/GPI reaches boundary early but retains large tangent residual and large post-hit angular motion.
- p2q2 setting-level tangent residual at boundary hit correlates with total post-boundary gain, while stepwise tangent residual has much weaker immediate next-step correlation.
- Objective-gradient replacement is the successful current surrogate; existing ablation does not support calling it the exact generalized-power optimizer of full Loss3.
- LP-steepest straight-ish norm growth is supported by normalized update length plus observed positive radial alignment, not by fixed direction.
- Final delta shapes are often similar in p2q2/p2q1 but this weakens in p=1 or q=inf settings.
- p/q geometry strongly affects spike/concentration behavior.

Inference:
- The consolidated paper-style interpretation is that replacement/GPI is an aggressive full-budget local surrogate that is especially effective in p2q2-like landscapes with a strong shared high-loss direction or ridge, but it should not be overclaimed as a guaranteed global optimizer of Loss3.

Remaining work:
- Optional targeted local Jacobian/SVD probes are still needed for a strong spectral dominant-mode claim.

## 2026-05-21 - Loss3 Current Core4/PQ Mechanism Validation Suite

Status: completed current core4/PQ mechanism validation with GPU landscape evaluation, trajectory-focused early-ray evaluation, targeted residual-Jacobian/SVD probe, summary figures, summary tables, and Markdown conclusions.

Source files and inputs:
- Landscape runner: `tools/run_loss3_core4_pq_landscape_probe.py`
- Jacobian/SVD runner: `tools/probe_loss3_current_core4_jacobian_svd.py`
- Summary plotter: `tools/plot_loss3_mechanism_validation_summary.py`
- Existing current core4/PQ setting roots under `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/` and `forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`
- Existing tangent/stepwise post-processing roots: `forensics/loss3_all_p2_tangent_geometry_probe_20260520/` and `forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520/`

GPU verification:
- Pre-run `nvidia-smi` showed Tesla V100-SXM2-32GB, driver 570.211.01, CUDA 12.8 driver capability, and no active GPU processes.
- `adv_robust/bin/python` reported PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, CUDA available, device `Tesla V100-SXM2-32GB`, compute capability `(7,0)`, PyTorch arch list including `sm_70`, successful CUDA matmul, JAX backend `gpu`, and JAX device `cuda:0`.
- Each GPU runner manifest records GPU-only runtime metadata and no CPU fallback.

Output files:
- Expanded landscape doc: `docs/loss3_core4_pq_landscape_probe_full_20260521.md`
- Expanded landscape output: `forensics/loss3_core4_pq_landscape_probe_full_20260521/`
- Trajectory-focused landscape doc: `docs/loss3_core4_pq_landscape_probe_trajectory_20260521.md`
- Trajectory-focused landscape output: `forensics/loss3_core4_pq_landscape_probe_trajectory_20260521/`
- Targeted Jacobian/SVD doc: `docs/loss3_current_core4_jacobian_svd_probe_20260521.md`
- Targeted Jacobian/SVD output: `forensics/loss3_current_core4_jacobian_svd_probe_20260521/`
- Final mechanism summary doc: `docs/loss3_current_mechanism_validation_summary_20260521.md`
- Final mechanism summary output: `forensics/loss3_current_mechanism_validation_summary_20260521/`
- Updated master summary: `docs/loss3_key_findings_master_summary_20260521.md`

Key settings:
- Expanded landscape: p2q2 baseline plus p2q1, p2qinf, and p1qinf baseline roots; sample indices `[0,7,20,40,47,63,80,99]`; ray/arc/2D slice/curvature evaluation.
- Trajectory-focused landscape: same four roots; sample indices `[0,7,40,47]` because only those have trajectory NPZ; early steps `[1,5,10,20]`.
- Jacobian/SVD: sample index `0`; 11 current core4/PQ states including p2q2 clean/step1/step5/step10/replacement final/steepest_add final, p2qinf clean/replacement final/steepest_add final, and p1qinf clean/replacement final.

Observed evidence:
- p2q2 steepest_replace ray endpoint ratio relative to final reaches `0.961` at step 5, `1.008` at step 10, and `1.012` at step 20.
- p2q2 boundary arcs stay near the weaker endpoint: replacement-to-additive arcs have min/weak-endpoint ratio about `0.986`.
- all-p2 tangent residual confirms replacement reaches boundary at step 1 with large tangent residual: p2q1 about `0.873`, p2q2 about `0.855`, p2qinf about `0.938`.
- p2q2 residual-Jacobian spectrum becomes more dominated after GPI moves: `sigma1/sigma2` rises from `1.148` at clean to `5.094` at replacement step 5, with top-1 energy fraction rising from `0.404` to `0.925`.
- The stronger pure-SVD claim is not supported: at p2q2 replacement final, the top residual-Jacobian right singular vector has cosine only `0.281` with the replacement final delta.
- p2qinf and p1qinf remain caveats: p2qinf has stronger geometry-dependent arc/ray differences, and p1qinf replacement final has weak spectral gap (`sigma1/sigma2 = 1.235`) and near-orthogonal top direction to replacement final delta.

Inference:
- The current data supports the local-surrogate/shared-ridge explanation: replacement/GPI rapidly reaches a high-loss boundary region and can rotate aggressively there.
- The data does not support claiming that replacement/GPI simply follows the top singular vector of the local residual Jacobian.
- The cleanest paper wording should remain conditional: objective-gradient replacement is an aggressive full-budget surrogate that works very well in p2q2-like geometry, while q=inf and p=1 geometries limit the story and can produce less stable/spikier behavior.

Remaining work:
- Optional: repeat the residual-Jacobian/SVD probe for more samples if a statistically stronger spectral statement is needed. Current SVD probe is targeted, not population-level.

## 2026-05-21 - Loss3 Hypothesis Validation Status Map

Status: completed documentation-only consolidation of which mechanism hypotheses have been verified, rejected, refined, or remain open. No new experiment was run for this map.

Source files and inputs:
- `docs/loss3_current_mechanism_validation_summary_20260521.md`
- `docs/loss3_current_core4_jacobian_svd_probe_20260521.md`
- `docs/loss3_core4_pq_landscape_probe_full_20260521.md`
- `docs/loss3_core4_pq_landscape_probe_trajectory_20260521.md`
- `docs/loss3_p2q2_tangent_radial_geometry_probe_20260520.md`
- `docs/loss3_p2q2_stepwise_gain_geometry_probe_20260520.md`
- `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`

Output file:
- `docs/loss3_hypothesis_validation_status_20260521.md`

Observed evidence recorded:
- Boundary hit, post-boundary tangent residual, angular motion, early high-loss ray ratios, shared ridge boundary arcs, p/q spike behavior, LP-steepest radial growth, raw PGD gradient-scale effects, and targeted residual-Jacobian/SVD mismatch were each mapped to explicit hypotheses.

Inference:
- Most of the original practical hypotheses are verified or refined, but the pure residual-Jacobian top-singular-vector explanation is rejected. The remaining strongest explanation is objective-gradient full-budget replacement as an aggressive local surrogate for an affine/nonlinear Loss3 landscape.

Remaining work:
- Population-level Jacobian/SVD and direct affine trust-region comparisons remain optional future work if a stronger mathematical claim is needed.

## 2026-05-21 - Loss3 GitHub and R2 Artifact Sync

Status: completed.

Source files and inputs:
- Local git branch `vast-ai`.
- Loss3 docs and tools under `docs/` and `tools/`.
- Generated Loss3 artifact directories under `forensics/`.

GitHub output:
- Commit `9dac8f5` with message `Add Loss3 core4 mechanism validation tooling and docs`.
- Pushed to `origin/vast-ai` at `https://github.com/YifeiSun01/NeuralOperatorRobustness2/tree/vast-ai`.

R2 output:
- Destination prefix: `s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/forensics/`.
- Uploaded generated Loss3 figures, visual exports, summaries, probe outputs, and large sweep outputs.
- Verification reported `14,253` objects and `16.710 GiB` under the remote `forensics` prefix.

Observed evidence:
- R2 remote listing showed key uploaded directories including `loss3_core4_pq_landscape_probe_full_20260521`, `loss3_core4_pq_landscape_probe_trajectory_20260521`, `loss3_current_core4_jacobian_svd_probe_20260521`, `loss3_current_mechanism_validation_summary_20260521`, `loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`, and `loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520`.

Remaining work:
- None for this sync request. Local `forensics/` remains untracked by git by design.

## 2026-05-21 - Loss1/Loss2 1D Burgers Optimizer-Speed Lookup

Status: completed lookup from existing local results. No attack was rerun.

Source files and inputs:
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`
- Per-run `loss_stats.csv` and `summary.json` under `loss1_*_*` and
  `loss2_*_*` attack tags.
- `docs/unified_eval_metric_loss_curve_plots_20260516.md`
- `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md`

Output record:
- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`

Key settings:
- FNO / 1D Burgers `nu=0.001`, batch size `100`, dataset indices `0..99`.
- `epsilon=8`, `alpha=0.3`, `steps=100`, `p=2`, `q=2`.
- Methods available for `loss1` and `loss2`: `pgd`, `lp_steepest_pgd`,
  `generalized_power`.

Observed evidence:
- For `loss1_original`, first step reaching 95% of the final mean was
  `pgd` step `26`, `lp_steepest_pgd` step `27`, and `generalized_power` step
  `3`. Final means were `11.1718`, `11.1493`, and `11.0143`, respectively.
- For `loss2_original`, first step reaching 95% of the final mean was
  `pgd` step `24`, `lp_steepest_pgd` step `27`, and `generalized_power` step
  `2`. Final means were `11.3734`, `11.3812`, and `11.2360`, respectively.
- Runtime recorded in `summary.json` was essentially the same across these
  methods, about `137-139` seconds per tag.
- No local result directory or documentation evidence was found for applying
  `raw_replace`, `steepest_replace`, or `power_replace` directly to `loss1` or
  `loss2`; the replacement-method evidence is for later `loss3`-focused runs.

Inference:
- For the existing `loss1_original` and `loss2_original` 1D Burgers evidence,
  `generalized_power` is faster in optimization-step convergence, while
  ordinary `pgd` and `lp_steepest_pgd` are slower but end at slightly higher
  mean optimized objective values in this setting.

Remaining work:
- Run or restore `loss1`/`loss2` replacement-method results if the intended
  comparison must include `raw_replace`, `steepest_replace`, or `power_replace`.

## 2026-05-21 - Loss1/Loss2 Core-Four P2Q2 Baseline Visual Run

Status: completed on GPU. This is a new run requested as the loss1/loss2 analogue
of the Loss3 baseline visualizations, restricted to one setting.

GPU verification:
- `nvidia-smi`: Tesla V100-SXM2-32GB, driver `570.211.01`, CUDA driver API
  `12.8`, no other GPU processes at verification time.
- PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, device `Tesla V100-SXM2-32GB`,
  compute capability `(7, 0)`, architecture list includes `sm_70`, CUDA matmul
  sanity sum `8.0`.
- JAX backend `gpu`, device `[CudaDevice(id=0)]`, JAX matmul sanity sum `8.0`.

Source files and inputs:
- New runner: `tools/run_loss1_loss2_core4_baseline_visuals.py`.
- Dataset: `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`.
- Model checkpoint: `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`.
- Baseline settings mirrored from the current Loss3 baseline visual run:
  `epsilon=4`, `alpha=0.4`, `steps=300`, `p=2`, `q=2`, batch size `100`,
  dataset indices `0..99`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- `loss1_original` used tiny random start `1e-6`; `loss2_original` used zero
  start.

Output files:
- Result doc: `docs/loss1_loss2_core4_p2q2_baseline_visuals_20260521.md`.
- Output root: `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/`.
- Numeric summary: `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/method_summary.csv`.
- Per-method sources: `per_step_metrics.csv`, `per_sample_step_metrics.csv`,
  `trajectory_samples.npz`, `final_delta.npz`, and `summary.json` under each
  `loss1_original/<method>/` and `loss2_original/<method>/` directory.
- Figures: ten PNGs under
  `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/`,
  covering objective curves, boundary-ratio curves, high-frequency-ratio curves,
  final delta line plots for selected indices, and index-0 delta heatmaps.

Observed evidence:
- Figure pixel checks confirmed the generated PNGs are nonblank.
- For `loss1_original`, final optimized-loss means were: `raw_add=6.9827`,
  `raw_replace=6.9419`, `steepest_add=6.9835`, `steepest_replace=6.9419`.
  Steps to 95% of best final were `7`, `2`, `10`, and `2`, respectively.
- For `loss2_original`, final optimized-loss means were: `raw_add=7.0759`,
  `raw_replace=7.0451`, `steepest_add=7.0783`, `steepest_replace=7.0451`.
  Steps to 95% of best final were `7`, `2`, `10`, and `2`, respectively.
- All final mean p-norms were essentially at the boundary: final
  `||delta||_2` mean about `4.0` for every objective/method.
- `raw_replace` and `steepest_replace` match numerically in this `p=2` setting,
  as expected because both use the normalized gradient direction for these
  objectives.

Inference:
- For this single baseline `p=q=2` setting, replacement methods reach the
  high-loss plateau fastest in step count, while additive methods end with a
  slightly higher final optimized loss, especially `steepest_add`.
- The conclusion should not be generalized to other P/Q, epsilon, or alpha
  settings without running those settings.

Remaining work:
- Optional: sync generated `forensics/` artifacts to R2 if these visual outputs
  should be preserved off-machine.

## 2026-05-21 - Loss1/Loss2 Core-Four Corrected 0..100 Marked/Angle Visuals

Status: completed on GPU. This corrected visual run addresses the missing pieces
from the prior simplified 0..300 visual output: it restricts the displayed range
to `k=0..100`, marks L2-budget boundary arrivals, and records step-to-step angle
diagnostics.

GPU verification:
- `nvidia-smi`: Tesla V100-SXM2-32GB, driver `570.211.01`, CUDA driver API
  `12.8`, no other GPU processes at verification time.
- PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, device `Tesla V100-SXM2-32GB`,
  compute capability `(7, 0)`, architecture list includes `sm_70`, CUDA matmul
  sanity sum `8.0`.
- JAX backend `gpu`, device `[CudaDevice(id=0)]`, JAX matmul sanity sum `8.0`.

Source files and inputs:
- Updated runner: `tools/run_loss1_loss2_core4_baseline_visuals.py`.
- Command: `adv_robust/bin/python tools/run_loss1_loss2_core4_baseline_visuals.py --steps 100 --save-every 1 --out-root forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2 --doc docs/loss1_loss2_core4_p2q2_baseline_marked_angles_0to100_20260521.md`.
- Same FNO / 1D Burgers `nu=0.001` dataset and checkpoint as the prior
  baseline visual run.
- Settings: `epsilon=4`, `alpha=0.4`, `steps=100`, `p=2`, `q=2`, batch size
  `100`, dataset indices `0..99`.
- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.

Output files:
- Result doc: `docs/loss1_loss2_core4_p2q2_baseline_marked_angles_0to100_20260521.md`.
- Output root: `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/`.
- Compact numeric tables: `method_summary.csv`, `boundary_marker_hits_0to100.csv`,
  and `angle_summary_0to100.csv`.
- Full per-step/per-sample sources under each `loss1_original/<method>/` and
  `loss2_original/<method>/`: `per_step_metrics.csv`,
  `per_sample_step_metrics.csv`, `trajectory_samples.npz`, `final_delta.npz`,
  and `summary.json`.
- Corrected figures: `figures_0to100_marked_angles/` contains boundary-marked
  loss curves, boundary-ratio curves, and angle curves for both objectives;
  the standard `figures/` directory contains the accompanying loss/delta/heatmap
  plots.

Observed evidence:
- PNG pixel checks confirmed all corrected `figures_0to100_marked_angles/*.png`
  files are nonblank.
- Every-step trajectories were saved: each `trajectory_samples.npz` has delta
  shape `(101, 4, 1024, 1)`, with saved steps `0..100`.
- `per_step_metrics.csv` now records `delta_prev_angle_degrees`,
  `direction_prev_angle_degrees`, and `delta_direction_angle_degrees`.
- Boundary markers use `25%`, `50%`, `75%`, and `99%`; `100%` is not separately
  marked. Replacement methods hit the `99%` boundary at step `1`; `raw_add`
  hits at step `8`; `steepest_add` hits at step `11` for both objectives.
- For `loss1_original`, final optimized-loss means were `raw_add=6.9773`,
  `raw_replace=6.9416`, `steepest_add=6.9767`, and
  `steepest_replace=6.9416`.
- For `loss2_original`, final optimized-loss means were `raw_add=7.0698`,
  `raw_replace=7.0452`, `steepest_add=7.0742`, and
  `steepest_replace=7.0452`.

Inference:
- This corrected run should be used for the user's requested visualization
  comparison. The earlier 0..300 run remains a valid numeric baseline but did
  not include the requested boundary-marker and angle-diagnostic visual layer.

Remaining work:
- Optional: sync the corrected `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/`
  artifact directory to R2 if it should be preserved off-machine.

## 2026-05-21 - Loss1/Loss2/Loss3 Core-Four Combined 0..100 Visuals

Status: completed from existing completed outputs. No attack, model, matrix, or
solver computation was rerun for this entry; the task was a CSV/PNG synthesis of
previous GPU-verified runs.

Source files and inputs:
- New plotting script: `tools/plot_loss1_loss2_loss3_combined_core4_0to100.py`.
- Command: `adv_robust/bin/python tools/plot_loss1_loss2_loss3_combined_core4_0to100.py`.
- Loss1/Loss2 source root: `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/`.
- Loss3 source root: `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/`, filtered to `k=0..100`.
- Settings represented: FNO / 1D Burgers `nu=0.001`, `epsilon=4`,
  `alpha=0.4`, `p=q=2`, batch size `100`, methods `raw_add`, `raw_replace`,
  `steepest_add`, and `steepest_replace`.

Output files:
- Result doc: `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`.
- Output root: `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`.
- Numeric sources: `combined_per_step_metrics_0to100.csv`,
  `combined_final_summary_0to100.csv`, and
  `combined_boundary_gain_summary_0to100.csv`.
- Figures: `combined_loss_mean_by_method_0to100.png`,
  `combined_loss_normalized_by_method_0to100.png`,
  `combined_boundary_ratio_by_method_0to100.png`,
  `combined_delta_pnorm_by_method_0to100.png`,
  `combined_delta_prev_angle_by_method_0to100.png`,
  `combined_direction_prev_angle_by_method_0to100.png`,
  `combined_delta_direction_angle_by_method_0to100.png`, and
  `combined_high_frequency_ratio_by_method_0to100.png`.

Observed evidence:
- PNG pixel checks confirmed all eight combined figures are readable and
  nonblank.
- All objectives/methods end on the `epsilon=4` L2 boundary by `k=100`; final
  mean boundary ratios are approximately `1.0` and final mean `||delta||_2` is
  approximately `4.0`.
- First mean boundary-ratio `>=0.99` and post-boundary loss gains from
  `combined_boundary_gain_summary_0to100.csv`: `raw_add` hits at `k=8` for
  loss1/loss2 with post-boundary gains `0.2101` and `0.1774`; loss3 `raw_add`
  hits at `k=43` with gain `0.2545`.
- Replacement methods hit boundary at `k=1`; post-boundary gains are `0.7203`
  for loss1, `0.6935` for loss2, and `1.478` for loss3.
- `steepest_add` hits boundary at `k=11`; post-boundary gains are `0.1851`
  for loss1, `0.1609` for loss2, and `0.5622` for loss3.
- Last-finite direction turning at the end is tiny for additive loss1/loss2
  (`<=0.05031` degrees) and larger for loss3, especially replacement methods
  (`36.62` degrees). Loss1/loss2 replacement turning is about `5.13..6.38`
  degrees.

Inference:
- Under this baseline `p=q=2` setting, the combined plots support the user's
  observation: loss1 and loss2 reach the epsilon boundary early and then gain
  relatively little objective value, while loss3 retains more post-boundary
  movement and turning, most clearly in the replacement-method panels.
- Because loss scales differ across objectives, the normalized objective plot is
  the preferred view for comparing curve shape; the raw objective plot should be
  used for within-objective magnitudes.

Remaining work:
- Optional: sync `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`
  to R2 if these combined figures should be preserved off-machine.

## 2026-05-21 - Loss1/Loss2/Loss3 Combined Figure Layout Correction

Status: completed from existing CSV outputs. No attack, model, matrix, or solver
computation was rerun.

Source files and inputs:
- Updated plotting script: `tools/plot_loss1_loss2_loss3_combined_core4_0to100.py`.
- Command: `adv_robust/bin/python tools/plot_loss1_loss2_loss3_combined_core4_0to100.py`.
- Source CSVs are the same as the combined visual entry above.

Output files:
- Updated result doc: `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`.
- Output root: `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`.
- The `*_by_method_0to100.png` figures preserve the original orientation:
  four method panels, each overlaying `loss1`, `loss2`, and `loss3`, with the
  legend moved outside the panels.
- The new `*_by_objective_0to100.png` figures use the requested orientation:
  three objective rows (`loss1`, `loss2`, `loss3`), each overlaying the four
  optimizer methods, with shared x/y ranges inside each metric figure.

Observed evidence:
- Sixteen PNG files are now present: eight metrics times two orientations.
- PNG readability checks confirmed all sixteen files are nonblank.
- The result doc now documents the naming convention for `by_method` versus
  `by_objective` layouts.

Inference:
- Use the `by_objective` files for the user's requested view of one row per
  objective with four method curves. Use the `by_method` files for the inverse
  comparison of one panel per optimizer with three objective curves.

Remaining work:
- Optional: sync the regenerated combined figure directory to R2.

## 2026-05-21 - All Requested Figures Single Folder

Status: completed. No experiment, model, matrix, or solver computation was
rerun; existing PNG figures were copied into one ordinary folder for direct
folder download. No archive or compressed file was created.

Source files and inputs:
- Combined comparison figures from
  `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`.
- Loss1/Loss2 corrected 0..100 figures from
  `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/`
  and its `figures/` directory.
- Loss1/Loss2 earlier 0..300 baseline figures from
  `forensics/loss1_loss2_core4_p2q2_baseline_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/figures/`.

Output files:
- Result doc: `docs/all_requested_figures_single_folder_20260521.md`.
- Single figure folder: `all_requested_figures_20260521/`.

Observed evidence:
- The folder contains 46 PNG files and no non-PNG files.
- Per-category counts: `combined/by_method=8`, `combined/by_objective=8`,
  `loss1_loss2_0to100` marked/angle categories total `10`,
  `loss1_loss2_0to100/standard_curves_and_delta_views=10`, and
  `loss1_loss2_0to300/standard_curves_and_delta_views=10`.
- PNG readability/nonblank check reported `46` readable PNGs and `0` blank
  files.

Inference:
- Use `all_requested_figures_20260521/` as the single folder for downloading
  the requested figures. The original experiment output paths remain unchanged.

Remaining work:
- None for the requested folder organization.

## 2026-05-21 - Combined Figures Std-Shaded Versions Added

Status: completed from existing CSV outputs. No experiment, model, matrix, or
solver computation was rerun.

Source files and inputs:
- Updated plotting script: `tools/plot_loss1_loss2_loss3_combined_core4_0to100.py`.
- Command: `adv_robust/bin/python tools/plot_loss1_loss2_loss3_combined_core4_0to100.py`.
- Same source CSV roots as the combined visual entries above.

Output files:
- Updated combined result doc:
  `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`.
- Updated single-folder record:
  `docs/all_requested_figures_single_folder_20260521.md`.
- New combined figures under
  `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/` using explicit
  `*_no_std_by_*_0to100.png` and `*_with_std_by_*_0to100.png` names.
- Updated single figure folder: `all_requested_figures_20260521/combined/`
  now has `no_std/by_method`, `no_std/by_objective`, `with_std/by_method`, and
  `with_std/by_objective` subfolders.

Observed evidence:
- Generated 32 explicit combined PNGs: eight metrics times two layouts times
  no-std/with-std versions.
- The whole `all_requested_figures_20260521/` folder now contains 62 PNGs and
  no non-PNG files.
- PNG readability/nonblank check reported 62 readable files and 0 blank files.
- The with-std plots draw mean +/- std translucent bands. Loss3 direction and
  delta-direction angle bands use the saved cosine std columns converted to an
  approximate angle std band.

Inference:
- Use `combined/with_std/...` when the batch variability band is needed, and
  `combined/no_std/...` when the mean-only curves are easier to read.

Remaining work:
- None for the requested std/no-std figure correction.

## 2026-05-21 - Figure Folder Cleanup

Status: completed. No experiment was rerun.

Source files and inputs:
- User-requested single folder: `all_requested_figures_20260521/`.
- Combined output root:
  `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`.

Output/cleanup actions:
- Kept all requested figures in `all_requested_figures_20260521/`.
- Deleted stale unlabeled combined PNGs from the combined output root; those
  were superseded by explicit `*_no_std_*` and `*_with_std_*` PNGs.
- Removed the unused empty `forensics/download_bundles/` directory.
- Left original experiment result directories, CSVs, docs, and explicit
  no-std/with-std figure outputs in place.

Observed evidence:
- `all_requested_figures_20260521/` contains 62 PNGs and no non-PNG files.
- No stale unlabeled `combined_*_by_*_0to100.png` files remain in the combined
  output root.
- `forensics/download_bundles/` no longer exists.

Inference:
- The single folder for downloading is now clean and unambiguous, while the
  experiment evidence paths needed for records remain intact.

Remaining work:
- None.

## 2026-05-21 - GPI Fast Optimizer Three-Loss Interpretation

Status: completed from existing experiment records. No experiment was rerun.

Source files and inputs:
- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`.
- `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`.
- `docs/loss3_hypothesis_validation_status_20260521.md`.
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`.

Output files:
- New interpretation doc:
  `docs/gpi_fast_optimizer_three_losses_interpretation_20260521.md`.

Observed evidence:
- For `loss1_original`, existing 1D Burgers CSV evidence gives k95 values
  `pgd=26`, `lp_steepest_pgd=27`, and `generalized_power=3`.
- For `loss2_original`, existing 1D Burgers CSV evidence gives k95 values
  `pgd=24`, `lp_steepest_pgd=27`, and `generalized_power=2`.
- For `loss3_original`, the saved generalized-power curve evaluated by
  `loss3_original` is already `6.581` at step `5`, `6.680` at step `10`, and
  `6.857` at step `100`; the final mean in that comparison is `6.8573`, larger
  than `loss3_original_lp_steepest_pgd=6.3782` and `loss3_original_pgd=5.3949`.
- Existing loss3 mechanism records show replacement/GPI has much larger angular
  motion than additive methods: p2q2 replacement/GPI post-hit angle about
  `30.68 deg` versus raw-add about `0.288 deg` and steepest-add about
  `0.617 deg` in the p2q2 rollup.

Inference:
- The locally recorded evidence supports describing generalized power / GPI-style
  replacement as the fastest early optimizer in step count across the three
  original losses, and as a method with especially fast perturbation-direction
  rotation for loss3.
- The claim should stay precise: GPI is fastest/early-strong, not guaranteed to
  be the largest final 300-step mean in every later alpha/epsilon setting.

Remaining work:
- None for this interpretation note.

## 2026-05-21 - Consolidated Three-Loss Optimizer Findings Markdown

Status: completed from existing records and generated figures. No experiment was
rerun.

Source files and inputs:
- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`.
- `docs/loss1_loss2_core4_p2q2_baseline_marked_angles_0to100_20260521.md`.
- `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`.
- `docs/gpi_fast_optimizer_three_losses_interpretation_20260521.md`.
- `docs/unified_eval_metric_three_panel_loss_curve_plots_20260516.md`.
- `docs/loss3_hypothesis_validation_status_20260521.md`.
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`.
- Organized figure folder: `all_requested_figures_20260521/`.

Output files:
- New consolidated summary:
  `docs/three_loss_burgers_optimizer_findings_summary_20260521.md`.

Observed evidence recorded:
- Generalized power / GPI reaches k95 in `3` steps for `loss1_original` and
  `2` steps for `loss2_original`, compared with `24..27` steps for additive
  PGD-style methods in the saved three-method evidence.
- `loss3_original_generalized_power` reaches `6.581` by step `5`, `6.680` by
  step `10`, and `6.857` by step `100` in the saved comparison.
- Loss1/loss2 corrected core-four baseline evidence shows boundary arrival
  followed by small post-boundary gains in the p2q2 `epsilon=4`, `alpha=0.4`
  setting.
- Loss3 mechanism evidence records much larger replacement/GPI angular motion
  than additive methods, including p2q2 replacement/GPI post-hit angle about
  `30.68 deg` versus raw-add about `0.288 deg` and steepest-add about
  `0.617 deg`.
- The organized single figure folder contains `62` PNGs and no non-PNG files.

Inference:
- The consolidated Markdown states the current evidence-backed interpretation:
  GPI-style replacement is the fastest early optimizer across the three original
  losses; loss1/loss2 largely plateau after boundary arrival; loss3 benefits
  from much more aggressive boundary-direction rotation.
- The document also records the caveat that GPI should not be claimed as an
  unconditional largest-final-loss method after long runs.

Remaining work:
- None for the consolidated Markdown.

## 2026-05-21 - Loss1/Loss2/Loss3 GitHub and R2 Sync

Status: completed code/documentation push and R2 artifact upload/update.

Source files and inputs:
- Local branch: `vast-ai`.
- GitHub remote: `origin` / `https://github.com/YifeiSun01/NeuralOperatorRobustness2.git`.
- R2 destination prefix:
  `s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/`.
- Paths uploaded/updated to R2: `docs/`, `tools/`, `EXPERIMENT_LEDGER.md`,
  `all_requested_figures_20260521/`, `results/`, and `forensics/`.

Output files:
- Sync record: `docs/loss1_loss2_loss3_r2_github_sync_20260521.md`.
- GitHub substantive commit pushed: `7c1ebd4` (`Add loss1 loss2 combined optimizer findings`).

Observed evidence:
- GitHub push updated `vast-ai` from `b529421` to `7c1ebd4`.
- R2 key docs were present after upload, including
  `three_loss_burgers_optimizer_findings_summary_20260521.md`,
  `gpi_fast_optimizer_three_losses_interpretation_20260521.md`,
  `loss1_loss2_loss3_core4_combined_0to100_20260521.md`, and
  `all_requested_figures_single_folder_20260521.md`.
- R2 key tools were present after upload:
  `plot_loss1_loss2_loss3_combined_core4_0to100.py` and
  `run_loss1_loss2_core4_baseline_visuals.py`.
- `EXPERIMENT_LEDGER.md` was present at the R2 root prefix.
- R2 `all_requested_figures_20260521/`: `62` objects, `11,019,609` bytes.
- R2 `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`:
  `36` objects, `6,221,148` bytes.
- R2 total `forensics/` prefix: `14,407` objects, `18,104,140,782` bytes.
- R2 total `results/` prefix: `41,449` objects, `1,992,387,343` bytes.
- No stale unlabeled `combined_*_by_*_0to100.png` files were observed in the
  remote combined artifact directory.

Inference:
- The requested code/record update is on GitHub, while large generated data,
  plots, and experiment artifacts are available under the R2 selected machine
  sync prefix.
- The R2 upload used non-destructive copy semantics, so remote-only historical
  objects were not deleted.

Remaining work:
- After this sync record is committed/pushed, upload `docs/` and
  `EXPERIMENT_LEDGER.md` once more so the sync record itself is also present on
  R2.


## 2026-05-21 - 2D NS Recurrent FNO2d 64/64/60 Training Entry

Status: prepared a GPU-only PyTorch recurrent FNO2d command-line trainer for
the requested `modes1=64`, `modes2=64`, `width=60` setup; full training was not
launched in this turn.

Source files inspected:
- `2D_NS_FNO2d_recurrent/models/FNO2d.py`
- `2D_NS_FNO2d_recurrent/training_models/trainFNO2d_unnormalized.py`
- `2D_NS_FNO2d_recurrent/training_models/trainFNO2d_unnormalized.sh`
- `2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py`
- `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame.py`
- Historical log:
  `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_log_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.txt`

Output files:
- `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`
- `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`
- `docs/ns2d_recurrent_fno2d_m64_w60_training_entry_20260521.md`

Key settings:
- PyTorch recurrent FNO2d.
- `modes1=64`, `modes2=64`, `width=60`, `num_layers=4`.
- Defaults: `epochs=500`, `ntrain=1000`, `ntest=100`, `batch_size=4`,
  `T_in=10`, `T_out=10`, `target_size=256`, `lr=0.001`,
  `weight_decay=0.0001`.
- Trainer refuses CPU fallback and records `nvidia-smi`, PyTorch/CUDA/device,
  compute capability, CUDA arch list, and a CUDA matmul sanity check before
  training.

Observed evidence:
- Existing `FNO2d.py` has a recurrent predictor compatible with
  `[batch, s, s, T_in] -> [batch, s, s, T_out]`.
- Historical `modes64_width60_epochs500_Tin10_T10` log records
  `235989181` trainable parameters and final legacy summed relative L2 at epoch
  499: train `66.15601575`, test `10.51145339`.
- Local scan observed `0` `.pt` files under `2D_NS_FNO2d_recurrent/datasets/`.
- Local scan observed `0` `.pth` files under
  `2D_NS_FNO2d_recurrent/saved_models/`.
- The newer real-initial generator expects
  `2D_NS_FNO2d_recurrent/datasets/source_zongyi_real_initial/`, which is not
  present locally.

Verification:
- `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`
  passed.
- `adv_robust/bin/python 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py --help`
  passed.
- `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh --help`
  passed.

Inference:
- The repo has model and generator code for the requested 2D NS recurrent FNO2d
  workflow, and now has a clean runnable training entry for the 64/64/60 model.
- A full new run still requires restoring or generating the `.pt` dataset file;
  no current local file evidences that the training data or the historical
  checkpoint is present in this working tree.

Remaining work:
- Put or generate the required `.pt` dataset locally.
- Launch
  `bash 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`
  on the GPU machine.
- After a real run, record the generated `gpu_verification.txt`,
  `train_log.csv`, checkpoints, and final metrics in this ledger and a result
  Markdown.


## 2026-05-21 - 2D NS R2 Dataset Lookup

Status: inspected the Cloudflare R2 bucket for existing 2D Navier-Stokes dataset/checkpoint artifacts; no download or training run was launched.

Source paths inspected:
- R2 bucket: `neural-operator-robustness`.
- R2 selected machine-sync prefix: `machine-sync/NeuralOperatorRobustness2-selected/`.
- Local result note updated: `docs/ns2d_recurrent_fno2d_m64_w60_training_entry_20260521.md`.

Observed evidence:
- R2 top level contained `machine-sync/` and `machine-sync/NeuralOperatorRobustness2-selected/`.
- Recursive R2 scan found `5482` data/checkpoint-suffix objects and `32` `.pt` objects, with `0` `.pth` objects.
- R2 scan found `0` data objects under `2D_NS_FNO2d_recurrent/datasets/`.
- R2 scan found `0` `.pth` checkpoint objects under `2D_NS_FNO2d_recurrent/`.
- The only R2 object matching 2D NS/FNO2d keywords with a `.pt/.pth/.npy/.npz/.mat` suffix was `machine-sync/NeuralOperatorRobustness2-selected/fno_training_runs/ns_m12_w20_profile_500/ns_real_initial_laxmap/ns_2d/checkpoints/fno2d_pytorch.pt`, size `3730297` bytes.
- R2 objects under dataset/data paths with data suffixes were all 1D Burgers `.pt` files.

Inference:
- The checked R2 selected machine-sync prefix does not currently contain the required 2D NS `.pt` dataset for training the requested recurrent FNO2d `modes1=64`, `modes2=64`, `width=60` model.
- The small `ns_m12_w20_profile_500` object is a checkpoint from a smaller training-suite run, not the requested dataset and not the target 64/64/60 model.

Remaining work:
- Generate the 2D NS dataset locally with the existing generator, restore it from another machine, or provide another R2 prefix if one exists outside the checked selected machine-sync prefix.
- After the dataset is restored, run `bash 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` and record the actual run outputs.


## 2026-05-21 - 2D NS Real-Initial Generation Runtime Check 05:11 UTC

Status: inspected the active 2D NS real-initial 256x256 generation run; no command was stopped or restarted.

Observed evidence:
- Active process: PID `302925`, command `generate_ns_real_initial_batched.py --splits test,train --target-size 256 --batch-size 32 --solver-batch-size 8 --solver-mode lax-map`.
- At `2026-05-21 05:11:25 UTC`, process elapsed time was `24:26` and GPU utilization was `98%`.
- GPU memory used was `24822 MiB / 32768 MiB`; GPU memory-bandwidth utilization was `1%`.
- Completed output present: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`, size `288362621` bytes, mtime `2026-05-21 04:48:06 UTC`.
- No train `.pt` output file was present yet; output directory size was `276M`.

Inference:
- The job is actively running on GPU and is in the train split after completing the test split.
- Because this command writes one `.pt` per split at completion, local files do not expose per-batch progress for the train split; progress is only visible in the launching terminal's tqdm output.
- Based on elapsed time after the test split and current GPU utilization, the train split likely has on the order of tens of minutes remaining, not hours, assuming no late-stage slowdown.

Remaining work:
- Continue monitoring until the train `.pt` appears.
- Start the recurrent FNO2d 64/64/60 training only after the train `.pt` is fully written.


## 2026-05-21 - 2D NS Real-Initial Generation Runtime Check 05:24 UTC

Status: inspected the active generation process again; no command was stopped or restarted.

Observed evidence:
- Active process: PID `302925`, elapsed time `38:05`, CPU time `39:37`.
- At `2026-05-21 05:24:26 UTC`, GPU utilization was `99%`, GPU memory used was `24822 MiB / 32768 MiB`, and memory-bandwidth utilization was `1%`.
- Completed output still only included the test split: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`, size `288362621` bytes.
- No train `.pt` file was present yet.

Inference:
- The job remains active on GPU and has not finished the train split.
- Since stdout/stderr are attached to the launching TTY and were not redirected to a log file, this inspection cannot see per-batch tqdm progress; only process/GPU/output-file status is visible from this session.
- The remaining time is an estimate only until the train output file appears.

Remaining work:
- Continue monitoring for the train `.pt` file and avoid launching training until it appears fully written.


## 2026-05-21 - 2D NS Real-Initial Dataset Completed And FNO2d Training Launched

Status: completed 256x256 real-initial 2D NS dataset generation and launched the recurrent FNO2d 64/64/60 PyTorch training run.

Generated dataset outputs:
- Test: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`, size `288362621` bytes, shape `x=(50,256,256)`, `y=(50,256,256,21)`.
- Train: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`, size `6632250121` bytes, shape `x=(1150,256,256)`, `y=(1150,256,256,21)`.
- Summary: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/generation_summary.json`, size `21943` bytes.

Observed generation metrics:
- `generation_summary.json` contained `38` records: test `2` batches / `50` samples, train `36` batches / `1150` samples.
- Recorded rollout seconds: test `104.30499046598561`, train `2278.595210202737`.
- Recorded upsample seconds: test `0.20982681098394096`, train `0.31126681552268565`.

Training launch:
- Dry run passed before launch and reported model parameters `118024381` by PyTorch trainable-parameter count.
- Detached training process launched as PID `314121` using `/workspace/NeuralOperatorRobustness2/adv_robust/bin/python`.
- Log file: `run_logs/train_fno2d_recurrent_m64_w60_20260521_0530.log`.
- Output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_real_initial_laxmap_v100/`.
- Training settings: `modes1=64`, `modes2=64`, `width=60`, `epochs=500`, `ntrain=1000`, `ntest=50`, `batch_size=4`, `eval_batch_size=4`, `T_in=10`, `T_out=10`.
- Initial GPU status after launch: Tesla V100-SXM2-32GB, GPU utilization `100%`, memory used `22662 MiB / 32768 MiB`, power draw `223.55 W`.

Observed evidence:
- Training log showed `Training FNO2d recurrent PyTorch`, resolved train/test paths, `device: Tesla V100-SXM2-32GB (sm_70)`, and tqdm at epoch `0/500`.
- GPU verification file is expected in the output directory from the trainer startup.

Remaining work:
- Monitor `run_logs/train_fno2d_recurrent_m64_w60_20260521_0530.log` and `nvidia-smi` for training progress and OOM risk.
- After completion, record `train_log.csv`, `results.json`, checkpoints, and final metrics.
- Keep generated large datasets/checkpoints out of git unless explicitly requested; sync to R2 if requested.


## 2026-05-21 - 2D NS FNO2d Training Status Check 05:34 UTC

Status: inspected active recurrent FNO2d 64/64/60 training after dataset generation completed; no process was stopped or restarted.

Observed evidence:
- Data generation had completed before training launch; train/test 256x256 `.pt` files were present under `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/`.
- Active training process: PID `314121`, command `train_fno2d_recurrent_cli.py --modes1 64 --modes2 64 --width 60 --epochs 500` via the wrapper settings, elapsed time `02:54`, CPU time `00:07:15`.
- At `2026-05-21 05:34:08 UTC`, GPU utilization was `91%`, GPU memory used was `22662 MiB / 32768 MiB`, and power draw was `261.45 W`.
- Log file `run_logs/train_fno2d_recurrent_m64_w60_20260521_0530.log` showed `Training FNO2d recurrent PyTorch`, device `Tesla V100-SXM2-32GB (sm_70)`, and tqdm at epoch `0/500`.
- Output directory `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_real_initial_laxmap_v100/` contained startup files including `gpu_verification.txt`, `config.json`, `dataset_info.json`, and the text training log, but no final metrics were observed.

Inference:
- The dataset generation stage is finished, but the model training stage is still running.
- It is not correct to treat the full experiment as complete until `train_log.csv`, final checkpoint/results files, and final metrics are present.

Remaining work:
- Continue monitoring PID `314121`, `run_logs/train_fno2d_recurrent_m64_w60_20260521_0530.log`, and the output directory for training progress.
- If OOM occurs, reduce `--batch-size` and `--eval-batch-size` before changing model parameters.



## 2026-05-21 - 2D NS FNO2d Progress Logging Patch

Status: modified the recurrent FNO2d trainer so future launches or restarts expose batch-level progress and ETA information.

Observed evidence:
- Active training PID `314121` was still running from the pre-patch script invocation.
- Its redirected log `run_logs/train_fno2d_recurrent_m64_w60_20260521_0530.log` showed epoch `1/500` completed in approximately `217.06s`, with tqdm estimating about `30:05:15` total at that early rate.
- The old active process only exposes epoch-level tqdm progress in the log, not fixed-interval batch progress within long epochs.

Code changes:
- `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` now has `--progress-every`, default `10`, to print flushed `[progress]` lines every N train batches.
- The trainer now writes `progress.jsonl` and `progress_latest.json` under the run output directory.
- The trainer now prints `[eval]` lines before/after evaluation and `[epoch]` lines after each epoch with train/test metrics.

Verification:
- `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` passed.
- `adv_robust/bin/python 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py --help` showed `--progress-every`.

Inference:
- The patch improves progress visibility for any new training launch or restart.
- The current PID `314121` cannot adopt this patch without a restart because it has already loaded the old Python code.

Remaining work:
- Decide whether to let PID `314121` continue with epoch-level progress only, or stop and restart the run to get batch-level progress and JSON progress files from the beginning of training.



## 2026-05-21 - 2D NS FNO2d Frame Semantics Check

Status: inspected generated dataset shape and recurrent training frame slicing.

Observed evidence:
- Train generated dataset has `x=(1150,256,256)` and `y=(1150,256,256,21)`.
- Test generated dataset has `x=(50,256,256)` and `y=(50,256,256,21)`.
- For both generated datasets, `max_abs(x - y[...,0]) = 0.0`, so `y[...,0]` is the real initial condition.
- `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` slices `sample[..., :t_in]` as input and `sample[..., t_in:t_in+t_out]` as target.
- With `t_in=10`, `t_out=10`, the training target is stored frames `10..19`, not stored frame `20`.
- `2D_NS_FNO2d_recurrent/models/FNO2d.py` `RecurrentPredictor.forward` predicts one frame, appends it, and rolls the input window repeatedly for `T_out` frames.

Inference:
- The current recurrent FNO2d training setup matches input stored frames `1..10`, supervise predicted stored frames `11..20`, and ignores stored frame `21` in one-based frame naming.
- The already-running PID `314121` was launched by the assistant after dataset completion; it was not launched by the data generation script itself.

Remaining work:
- If batch-level progress is required for the active run, stop/restart with the patched `--progress-every` trainer command.


## 2026-05-21 - 2D NS FNO2d Runtime And Hardware Utilization Check 05:43 UTC

Status: inspected active recurrent FNO2d 64/64/60 training runtime and GPU utilization.

Observed evidence:
- Active training PID `314121`, elapsed time `12:05`, CPU time `00:30:00`, process RSS about `7.84 GB`.
- GPU snapshot at `2026-05-21 05:43:20 UTC`: Tesla V100-SXM2-32GB, GPU utilization `100%`, memory utilization `82%`, memory used `22662 MiB / 32768 MiB`, power draw `222.73 W`, SM clock `1530 MHz`, memory clock `877 MHz`.
- `nvidia-smi dmon` samples showed SM utilization mostly `88-100%`, memory utilization `76-84%`, and power draw `213-262 W`.
- `train_log.csv` contained epochs 1-3 with seconds `213.07864790898748`, `211.67835453199223`, and `213.18633831501938`.
- The redirected tqdm log showed epoch `3/500` around `10:51`, consistent with about `212-217s` per epoch including evaluation.

Inference:
- The V100 is substantially occupied by the training run; this is a GPU-limited run rather than an obviously CPU-idle bottleneck.
- At the observed early rate, the 500-epoch run is roughly `29.5-30.0` hours total on the current V100 setup, assuming the per-epoch time remains stable.
- Larger/newer GPUs such as A100/B200 should reduce runtime, but exact speedup must be measured because this FNO workload includes FFT/complex spectral operations and autoregressive recurrent steps, not only dense tensor-core GEMMs.

Remaining work:
- Continue monitoring actual epoch times from `train_log.csv`.
- If restarting for progress logging, use the patched `--progress-every` command and record the new PID/log path.


## 2026-05-21 - 2D NS FNO2d V100 Speed-Up Assessment

Status: assessed whether the active V100 training run can be made substantially faster without changing the requested model/data semantics.

Observed evidence:
- Active run is using a Tesla V100-SXM2-32GB with GPU utilization around `91%` at the snapshot and previous `nvidia-smi dmon` samples mostly `88-100%` SM utilization.
- Memory used remains about `22662 MiB / 32768 MiB`, so there is memory headroom, but the GPU compute path is already highly occupied.
- First three recorded epoch times are approximately `213.08s`, `211.68s`, and `213.19s`, implying about `29.5` hours for 500 epochs if stable.
- The current process was launched without `--amp`; the trainer supports `--amp`, but this must be benchmarked because the FNO implementation uses FFT and complex spectral multiplication.

Inference:
- The current V100 run is primarily GPU-limited rather than obviously blocked by CPU or data loading.
- Increasing batch size may improve overhead modestly but is unlikely by itself to reduce `~30h` to `~10h` because SM/memory utilization is already high.
- Realistic V100-side speedups without changing the model/data are likely from `--amp`, less frequent evaluation, and perhaps a slightly larger batch if stable; a full 3x speedup on V100 is uncertain and should not be assumed without a benchmark.
- Reaching roughly 10 hours is much more plausible on a faster GPU such as A100/B200, or by reducing the training budget such as epochs/early stopping.

Remaining work:
- If the user wants to optimize on this V100, stop the current run and launch a short AMP/eval-frequency benchmark before committing to a 500-epoch run.


## 2026-05-21 - 2D NS Training Stopped And Dataset Backed Up To R2

Status: stopped the active V100 recurrent FNO2d training run and backed up the generated 256x256 real-initial 2D NS dataset to R2.

Observed evidence:
- User requested stopping the active training process before backup.
- Training process PID `314121` was terminated at approximately `2026-05-21 05:48 UTC`.
- GPU snapshot after stopping showed utilization `0%` and memory used `0 MiB / 32768 MiB`.
- R2 destination prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/`.
- R2 listing after copy showed:
  - `generation_summary.json`, size `21943` bytes.
  - `test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`, size `288362621` bytes.
  - `train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`, size `6632250121` bytes.

Inference:
- The generated real-initial 256x256 train/test dataset and generation summary are backed up on R2 under the selected machine-sync prefix.
- The stopped training run did not complete; its partial local run outputs remain local and were not treated as final training results.

Remaining work:
- Push code and experiment records to GitHub branch `vast-ai`.
- Keep large dataset/checkpoint artifacts out of git unless explicitly requested.


## 2026-05-21 - GitHub Backup Completed

Status: pushed code and experiment records to GitHub branch `vast-ai`.

Observed evidence:
- Local commit created: `674810f Add 2D NS recurrent FNO training workflow`.
- `git push origin vast-ai` completed successfully and updated remote branch `vast-ai` from `e9223ea` to `674810f`.
- Staged/pushed files were limited to source and record files: recurrent FNO2d trainer, launch wrapper, `AGENTS.md`, `EXPERIMENT_LEDGER.md`, and the 2D NS training result Markdown.
- Large generated `.pt` datasets and partial training output directories were not added to git.

Inference:
- The code/record backup is present on GitHub at branch `vast-ai` through commit `674810f`; generated datasets are backed up on R2 rather than git.


## 2026-05-21 - 2D NS Recurrent FNO2d Runtime Logging And R2 Auto-Upload Workflow

Status: modified training workflow code after syncing local tree to GitHub `vast-ai` commit `d107e33c2740fef80609ac1385f20ddbc52fafa8`; no new training run was executed in this turn.

Observed evidence:
- `git fetch --prune origin vast-ai`, `git reset --hard FETCH_HEAD`, and `git clean -fd` completed before edits; `HEAD` was `d107e33c2740fef80609ac1385f20ddbc52fafa8`.
- Source files modified: `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` and `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`.
- Result/workflow note created: `docs/fno2d_recurrent_training_runtime_r2_upload_20260521.md`.
- Validation commands passed: `python3 -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` and `bash -n 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`.

Key settings implemented:
- Wrapper defaults to the real-initial 2D NS train/test dataset paths under `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/`, with `NTRAIN=1150`, `NTEST=50`, `BATCH_SIZE=4`, `EVAL_BATCH_SIZE=4`, and `EPOCHS=500` unless overridden.
- Trainer now prints and records per-epoch wall time, mean epoch time, recent mean epoch time, ETA, estimated total runtime, train/test relative L2, train/test MSE, and `1-relative_l2` score.
- `--r2-upload` now performs R2 preflight before GPU training, syncs small metric/log files every `--r2-sync-every-epochs`, and uploads the complete output directory at training completion.
- R2 secrets are read from environment variables only and are not stored in repository files.

Observed metrics:
- No model-training metrics were produced because training was not run in this turn.

Inference:
- The next full training run will print epoch duration and ETA after the first epoch and will keep updating local and R2-synced metric records.
- Final model artifacts should land under the configured R2 prefix after training completes, provided R2 preflight and final `rclone copy` both succeed.

Remaining work:
- Launch the full GPU training command with `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` set in the environment.
- Monitor `train_log.csv`, `progress.jsonl`, `progress_latest.json`, and console output for runtime estimates and train/test metrics.
- Verify final R2 model directory after completion.


## 2026-05-21 - adv_robust GPU Environment Rebuilt

Status: rebuilt local `adv_robust` virtual environment with the repository GPU setup script; no model training was run.

Observed evidence:
- `adv_robust/bin/python` was initially unusable: `tools/setup_adv_robust_gpu_env.py --verify-only` failed with `Too many levels of symbolic links` for `adv_robust/bin/python3.12`.
- Preserved broken environment as `adv_robust_broken_20260521_venv_symlink`.
- Rebuild command used: `tools/setup_adv_robust_gpu_env.py`.
- Active GPU: `NVIDIA A100-SXM4-80GB`, compute capability `8.0`, expected PyTorch arch `sm_80`.
- Verification after rebuild: PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, `torch.cuda.is_available() = True`, torch device `NVIDIA A100-SXM4-80GB`, torch arch list includes `sm_80`, PyTorch CUDA matmul succeeded.
- JAX verification after rebuild: backend `gpu`, device `[CudaDevice(id=0)]`, JAX GPU matmul succeeded.
- `pip check` reported no broken requirements.
- Result record: `docs/adv_robust_environment_rebuild_20260521.md`.

Inference:
- `adv_robust` is now installed and verified for GPU execution on the current A100 machine.
- The previous `adv_robust` directory was a broken partial environment and should not be used.

Remaining work:
- Use `adv_robust/bin/python` for training commands.
- Continue using `tools/setup_adv_robust_gpu_env.py` for future environment rebuilds instead of raw `pip install -r requirements.txt`.


### 2026-05-21 - R2 Credential Whitespace Hardening Note

Status: small follow-up edit to the 2D NS recurrent FNO2d trainer; no training run executed.

Observed evidence:
- Updated `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` so R2 credential environment variables are stripped of accidental leading/trailing whitespace before temporary rclone config creation.
- Validation command passed: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`.

Inference:
- Commands copied with an accidental trailing space inside the quoted R2 key are less likely to fail R2 preflight, though clean environment exports without spaces remain preferred.


## 2026-05-21 - Active 2D NS Recurrent FNO2d Training First-Three-Epoch Monitor

Status: inspected active 500-epoch training run; no changes made to the running process.

Observed evidence:
- Active PID `23159` is running `adv_robust/bin/python -u 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`.
- Output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC`.
- Exact metric source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/train_log.csv`.
- Latest progress source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/progress_latest.json`.
- The run had advanced into epoch `4/500`, batch `10/288`, with run elapsed `1423.15s` when inspected after the first three epochs.
- GPU snapshot at `2026-05-21 07:36:38 UTC`: `NVIDIA A100-SXM4-80GB`, utilization `68%`, memory used `23113 MiB / 81920 MiB`, power draw `189.15 W`.
- First three epoch rows from `train_log.csv`:
  - Epoch 1: `475.2787319869967s`, train relative L2 `0.4178855663278829`, test relative L2 `0.28925264835357667`, estimated total `66h00m39s`.
  - Epoch 2: `444.8703988859779s`, train relative L2 `0.27541956862677697`, test relative L2 `0.26146268486976626`, estimated total `63h54m02s`.
  - Epoch 3: `476.0251846399624s`, train relative L2 `0.253930271708447`, test relative L2 `0.27035082519054415`, estimated total `64h38m25s`.
- Dedicated note: `docs/fno2d_recurrent_training_monitor_20260521.md`.

Inference:
- Training is actively running in the `adv_robust` environment on the A100 GPU.
- Mean runtime over the first three completed epochs is about `465.39s` (`7m45s`) per epoch.
- First-three-epoch extrapolation suggests about `64.6h` total wall time for 500 epochs, including per-epoch evaluation and R2 record sync overhead.

Remaining work:
- Continue monitoring later epoch times and final upload status.
- Verify R2 contains the final model directory after training completes.


### 2026-05-21 07:38 UTC - Active 2D NS Recurrent FNO2d Runtime Estimate Update

Status: inspected active run for a user-requested runtime estimate; no changes made to the running process.

Observed evidence:
- Active PID `23159` still running under `adv_robust/bin/python`.
- Metric source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/train_log.csv`.
- Progress source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/progress_latest.json`.
- Completed epoch times available: epoch 1 `475.2787s`, epoch 2 `444.8704s`, epoch 3 `476.0252s`; mean `465.39s` per epoch.
- Latest progress showed epoch `4/500`, batch `70/288`, run elapsed `1520.47s`, ETA `64h41m39s`.

Inference:
- With current settings, 500 epochs are estimated at about `64.6h` total (`~2.7 days`).
- Estimate includes per-epoch evaluation and per-epoch R2 record sync overhead.

Remaining work:
- Refresh estimate after more completed epochs, ideally after epoch 10-20.


### 2026-05-21 07:40 UTC - Runtime Estimate Clarification

Status: inspected active run to clarify epoch-vs-ETA confusion; no changes made to the running process.

Observed evidence:
- `train_log.csv` still had three completed epochs: `475.2787s`, `444.8704s`, and `476.0252s`.
- Latest `progress_latest.json` showed epoch `4/500`, batch `120/288`, epoch elapsed `196.27s`, epoch remaining ETA `4m35s`, and total train-batch ETA `64h45m08s`.
- GPU snapshot at `2026-05-21 07:39:58 UTC`: A100 utilization `100%`, memory utilization `58%`, memory used `23113 MiB / 81920 MiB`, power draw `188.76 W`.

Inference:
- The earlier `epoch_eta=5m22s` was remaining time within an already-started epoch, not the full epoch duration.
- Measured full epoch time remains about `7m25s-7m56s`, with first-three-epoch mean `465.39s`.
- Current 500-epoch estimate remains about `64-65h` total with eval and R2 sync enabled.

Remaining work:
- Re-estimate after 10-20 completed epochs and consider reducing eval/R2 sync frequency or testing larger batch/AMP if runtime is unacceptable.



### 2026-05-21 07:43 UTC - A100 Runtime Bottleneck Check

Status: inspected the active run and training code to explain why the A100 run is estimating much longer than expected; no changes made to the running process.

Observed evidence:
- Active PID `23159` still running `adv_robust/bin/python`.
- Metric source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/train_log.csv`.
- Progress source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/progress.jsonl`.
- Latest progress inspected showed epoch `4/500`, batch `240/288`, run elapsed `1798.87s`, and total train-batch ETA `64h40m37s`.
- GPU utilization sample from `nvidia-smi dmon -s pucm -c 8` alternated between `100%` SM utilization during training bursts and `0%` at synchronization or logging sample points; framebuffer memory stayed around `23113 MB` on an `81920 MB` A100.
- Active command uses `--batch-size 4`, `--eval-batch-size 4`, `--eval-every 1`, and `--r2-sync-every-epochs 1`.
- Current wrapper `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` defaults to `BATCH_SIZE=4`, giving `ceil(1150/4)=288` train batches per epoch.
- `2D_NS_FNO2d_recurrent/models/FNO2d.py` uses `RecurrentPredictor.forward()` with `for _ in range(0, T_out, step)`; with `T_out=10` and `step=1`, each training batch performs 10 sequential FNO model calls.
- Legacy script `2D_NS_FNO2d_recurrent/training_models/trainFNO2d_unnormalized.py` used `batch_size = 20`, `ntrain = 1000`, and about `50` train batches per epoch for that configuration.

Inference:
- I did not observe data generation, solver execution, or CPU fallback inside the active training loop.
- The current run is GPU-backed but is not configured to use most of the A100 memory; it uses about `23 GB / 80 GB`.
- The largest observed configuration difference explaining the long epoch time is `batch_size=4`: the current run does about `288` train batches per epoch, and each batch contains 10 recurrent FNO steps. A batch size closer to the legacy `20` would reduce the number of optimizer steps per epoch substantially if memory and numerics allow it.
- Evaluation every epoch and R2 record sync every epoch add overhead, but they do not by themselves explain the full slowdown.

Remaining work:
- If the current runtime is unacceptable, benchmark larger `BATCH_SIZE`/`EVAL_BATCH_SIZE` values, for example 8, 12, or 16, before restarting a long 500-epoch official run.
- Consider reducing `EVAL_EVERY` and `R2_SYNC_EVERY_EPOCHS` for long runs after confirming the monitoring cadence needed.



### 2026-05-21 07:48 UTC - Fourth Epoch Runtime Refresh

Status: inspected the active run after epoch 4 completed; no changes made to the running process.

Observed evidence:
- Metric source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/train_log.csv`.
- Epoch 4 completed in `487.73004865599796s` (`8m08s`).
- Mean epoch time after four epochs is `473.83224393974524s` and recent mean is `470.97609104223375s`.
- Epoch 4 row reported estimated total `65h24m59s`.
- Latest `progress_latest.json` after epoch 4 showed epoch `5/500`, batch `110/288`, run elapsed `2081.57s`, and total train-batch ETA `65h23m55s`.
- Current GPU snapshot showed A100 utilization `100%`, memory utilization `58%`, memory used `23113 MiB / 81920 MiB`, and power draw `189.23 W / 400 W`.

Inference:
- The fourth epoch reinforces the current estimated wall time of about `65h` with the active settings.
- The run is using the A100, but memory use remains far below the 80 GB capacity, so larger batch-size benchmarking is the most direct throughput test.



### 2026-05-21 07:50 UTC - GPU Path And Utilization Verification

Status: verified the active training run is using the A100 GPU and inspected why throughput is still poor; no changes made to the running process.

Observed evidence:
- Active PID `23159` continued running the 500-epoch recurrent FNO2d trainer with `--batch-size 4`, `--eval-batch-size 4`, `--eval-every 1`, and `--r2-sync-every-epochs 1`.
- Run GPU verification file records `device: cuda:0`, `device_name: NVIDIA A100-SXM4-80GB`, `compute_capability_tag: sm_80`, `torch_version: 2.8.0+cu126`, `torch_cuda_version: 12.6`, and CUDA sanity matmul sum `8.0`.
- Saved config records `amp: false`, `batch_size: 4`, `num_workers: 0`, `ntrain: 1150`, `ntest: 50`, `modes1: 64`, `modes2: 64`, `width: 60`, `t_out: 10`, `step: 1`.
- `nvidia-smi dmon -s pucmt -c 15` showed alternating samples including SM utilization `100`, `97`, `73`, `32`, and many `0` samples, while memory stayed around `23113 MB / 81920 MB`.
- Latest progress around `2026-05-21T07:49:53+00:00` showed epoch `5/500`, batch `190/288`, run elapsed `2211.27s`, and total train-batch ETA `65h17m43s`.

Inference:
- The run is definitely GPU-backed on the A100, not CPU fallback.
- The A100 is not continuously saturated; it bursts into GPU compute and then idles between bursts, consistent with a small batch size, host-side/DataLoader/synchronization overhead, and frequent logging/evaluation checkpoints.
- The current configuration is not a good throughput configuration for an 80 GB A100.

Remaining work:
- Stop and benchmark larger batch sizes if the goal is fastest wall-clock training; suggested starting points are batch size 8 and 12, then 16 only if memory remains safe.



### 2026-05-21 08:00 UTC - A100 Fast Batch Benchmark And Default Update

Status: stopped the slow batch-size-4 active run, benchmarked larger batch sizes on A100, and updated defaults for faster training.

Observed evidence:
- Stopped active PID `23159`; it had been running with `--batch-size 4`, `--eval-batch-size 4`, `--eval-every 1`, and `--r2-sync-every-epochs 1`.
- Last inspected slow-run progress before stopping was epoch `5/500`, batch `260/288`, with total train-batch ETA `65h13m12s`.
- Batch benchmark used the real training dataset path `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`, model `modes1=64`, `modes2=64`, `width=60`, `T_out=10`, `step=1`, and A100 GPU.
- FP32 benchmark results:
  - `batch_size=10`: mean timed batch `0.924s`, `115` batches per epoch, train-only epoch estimate `106.2s`, peak memory `46.71 GiB`.
  - `batch_size=12`: mean timed batch `1.074s`, `96` batches per epoch, train-only epoch estimate `103.1s`, peak memory `55.52 GiB`.
  - `batch_size=14`: mean timed batch `1.077s`, `83` batches per epoch, train-only epoch estimate `89.4s`, peak memory `64.33 GiB`.
  - `batch_size=16`: mean timed batch `1.165s`, `72` batches per epoch, train-only epoch estimate `83.9s`, peak memory `72.19 GiB`.
  - `batch_size=20`: failed with CUDA OOM around `78.78 GiB` in use.
- AMP benchmark failed before timing because `torch.fft` plus complex spectral weights entered a CUDA `ComplexHalf` path and raised `NotImplementedError: baddbmm_cuda not implemented for ComplexHalf`.
- Updated `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` defaults to `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `EVAL_EVERY=5`, and `R2_SYNC_EVERY_EPOCHS=0` while preserving final R2 upload when `R2_UPLOAD=1`.
- Updated `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` parser defaults to match and added a clear runtime error if `--amp` is requested.
- Validation passed: `bash -n 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` and `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`.

Inference:
- Increasing batch size is the best immediate speed lever that preserves the same model and dataset, but it changes optimizer dynamics because there are fewer optimizer updates per epoch.
- `batch_size=16` is the fastest tested no-AMP setting that fits the A100, but it uses high memory and should be monitored for OOM.
- A 10-hour 500-epoch run is not currently evidenced with this exact model/training setup; batch size 16 points to roughly 12+ hours train-only before full-run overhead, evaluation, checkpoint, and final upload.

Remaining work:
- Start the next official run with `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `EVAL_EVERY=5`, `R2_SYNC_EVERY_EPOCHS=0`, and `AMP=0`.
- If batch size 16 OOMs during a full run, retry with batch size 14; if accuracy is unstable, compare against batch size 10 or tune learning rate.



### 2026-05-21 08:10 UTC - Batch Size 18 OOM Test

Status: tested user-requested `batch_size=18` on the A100 with the real recurrent FNO2d model and dataset; no official 500-epoch training run started.

Observed evidence:
- No active 2D FNO training process was found before the test.
- Pre-test GPU snapshot showed `NVIDIA A100-SXM4-80GB`, `0 MiB / 81920 MiB` used.
- Current defaults before the test remained `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `EVAL_EVERY=10`, and `R2_SYNC_EVERY_EPOCHS=0`.
- Plain FP32 `batch_size=18` failed with CUDA OOM: requested `1.05 GiB`, GPU capacity `79.25 GiB`, process memory in use `79.12 GiB`.
- Retesting from a fresh process with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also failed with CUDA OOM: requested `1.05 GiB`, process memory in use `78.31 GiB`, free `949.88 MiB`.

Inference:
- `batch_size=18` is not reliable for this exact model/data setup on the current 80GB A100.
- `batch_size=16` remains the largest tested safe setting, while `batch_size=14` is the safer fallback.

Remaining work:
- Use `batch_size=16` for the next official run unless the user chooses a more conservative batch size.
- If further speed is required, investigate implementation-level changes rather than simply increasing batch size above 16.



### 2026-05-21 08:06 UTC - Training Process Stop Confirmation

Status: rechecked active training state after the user asked to stop the prior run and test `batch_size=18`; no official training run started.

Observed evidence:
- `pgrep -af train_fno2d_recurrent_cli.py|run_fno2d_recurrent_m64_w60.sh|trainFNO2d_unnormalized.py` returned no active matching training process.
- GPU snapshot showed `NVIDIA A100-SXM4-80GB`, utilization `0%`, and memory used `0 MiB / 81920 MiB`.
- The earlier recorded `batch_size=18` real forward/backward tests failed with CUDA OOM in both normal FP32 mode and with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Inference:
- The previous slow training run remains stopped.
- `batch_size=18` should not be used for the next official run on this exact setup; `batch_size=16` remains the largest tested safe setting.



### 2026-05-21 08:12 UTC - Batch Size 17 Test

Status: tested user-requested `batch_size=17` on the A100 with the real recurrent FNO2d model and dataset; no official 500-epoch training run started.

Observed evidence:
- No active 2D FNO training process was found before the test.
- Pre-test GPU snapshot showed `NVIDIA A100-SXM4-80GB`, `0 MiB / 81920 MiB` used.
- Plain FP32 `batch_size=17` completed batch 1 in `1.572s` with peak memory `74.82 GiB`, then failed with CUDA OOM on the next allocation; process memory in use was `78.28 GiB`.
- Retesting from a fresh process with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` completed four timed batches: `1.835s`, `1.218s`, `1.218s`, `1.216s`, peak memory `76.53 GiB`.
- The expandable-segments run reported `68` batches per epoch and a short-run train-only epoch estimate of `93.3s`; warm batches after the first were about `1.217s` each.

Inference:
- `batch_size=17` can run only with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in this short test, but it is close to the memory limit.
- `batch_size=16` remains the safer official-run setting; `batch_size=17` is a speed/memory-risk option if the user accepts OOM risk and sets the allocator environment variable before process start.

Remaining work:
- If using `batch_size=17`, launch from a fresh shell/process with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and monitor early epochs for OOM.



### 2026-05-21 08:18 UTC - AMP Compatibility Retest

Status: directly retested mixed precision compatibility on the real recurrent FNO2d model path; no official training run started.

Observed evidence:
- Test environment: `NVIDIA A100-SXM4-80GB`, PyTorch `2.8.0+cu126`, CUDA `12.6`, default CUDA autocast dtype `torch.float16`, BF16 supported by hardware.
- Test used the real recurrent FNO2d model with `modes1=64`, `modes2=64`, `width=60`, `T_out=10`, `step=1`, and a real dataset batch from `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`.
- FP32/no autocast forward-backward completed successfully: `seconds=0.682`, `pred_dtype=torch.float32`, `loss_dtype=torch.float32`, peak memory `9.74 GiB` for the small batch-size-2 test.
- FP16/default AMP autocast failed with `NotImplementedError: baddbmm_cuda not implemented for ComplexHalf` at `2D_NS_FNO2d_recurrent/models/FNO2d.py`, inside `torch.einsum("bixy,ioxy->boxy", input, weights)` after `torch.fft.rfft2` produced complex half tensors.
- BF16 autocast failed with `RuntimeError: Unsupported dtype BFloat16` at `torch.fft.rfft2(x)`.

Inference:
- Full-model AMP autocast is not compatible with the current recurrent FNO2d implementation in this PyTorch/CUDA environment.
- The existing trainer guard that rejects `--amp` remains justified for this model path.
- Any mixed precision speedup would require a selective implementation that keeps FFT/spectral complex operations in FP32/complex64 and only autocasts safe real-valued submodules, followed by numerical validation.

Remaining work:
- Continue using `AMP=0` for official runs unless a selective mixed-precision implementation is added and validated.



### 2026-05-21 08:25 UTC - Batch Size 16 Full-Epoch Stress Test

Status: ran a one-epoch stress test with `batch_size=16` on the real dataset/model path to answer whether later batches can OOM after the first batch succeeds; no checkpoint, R2 upload, or official 500-epoch run was started.

Observed evidence:
- Pre-test state had no active FNO training process and A100 memory at `0 MiB / 81920 MiB`.
- Test used the real train dataset `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`, model `modes1=64`, `modes2=64`, `width=60`, `T_out=10`, `step=1`, and FP32/no AMP.
- `batch_size=16` gives `72` batches per epoch for `1150` samples.
- The stress test completed all `72/72` batches with no CUDA OOM.
- Peak CUDA memory stayed at `72.19 GiB` after the early batches and did not continue climbing through batch 72.
- Selected progress: batch 1 peak `70.42 GiB`; batches 10, 20, 30, 40, 50, 60, 70, and 72 all reported peak `72.19 GiB`.
- Mean measured GPU compute section per batch was `1.170s`; total wall time for the full epoch script was `401.1s` (`6.69min`).

Inference:
- Unlike `batch_size=17`, `batch_size=16` did not show a second-batch OOM or later-batch memory creep during a full epoch.
- `batch_size=16` is materially safer than `batch_size=17`, although it still uses high memory.
- The gap between measured compute-section time and full wall time indicates significant host-side data loading/collation/transfer overhead that should be investigated for further speedup.

Remaining work:
- Use `batch_size=16` for the next official run if prioritizing speed without accepting the `batch_size=17` OOM risk.
- Profile DataLoader/data movement and consider larger `num_workers`, precomputed contiguous input/target tensors, or GPU-resident data strategies if further speed is required.



### 2026-05-21 08:30 UTC - Data Loading Bottleneck Analysis

Status: inspected the recurrent FNO2d trainer data path after the batch-size-16 full-epoch stress test; no official training run started and no code change made in this step.

Observed evidence:
- In `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`, `load_y_tensor()` loads the full dataset to CPU with `torch.load(..., map_location="cpu")`.
- `NSTrajectoryDataset.__getitem__()` slices each sample every time it is requested and returns `x.contiguous(), y.contiguous()`.
- `make_loader()` uses `num_workers=args.num_workers`, default `0`, and `pin_memory=True`.
- The training loop recreates the DataLoader inside every epoch with `make_loader(...)`.
- The training loop transfers each batch to GPU with `xb.to(device, non_blocking=True)` and `yb.to(device, non_blocking=True)`.
- The batch-size-16 full-epoch stress test measured about `1.170s` for the synchronized GPU compute section per batch, but `401.1s` total wall time for 72 batches.

Inference:
- The difference between about `84s` of measured compute sections and `401s` wall time is consistent with host-side overhead from per-sample CPU slicing/copying/collation, DataLoader iteration, pinned-memory handling, and CPU-to-GPU transfer waits.
- Pure PCIe bandwidth alone is unlikely to explain the full overhead because the per-epoch batch tensor volume is only several GiB; the current per-sample data path is likely the larger issue.
- Candidate optimizations are: precompute contiguous input/target tensors once, avoid recreating the DataLoader every epoch, benchmark persistent multi-worker loading, and optionally preload train/test tensors to GPU with a manual GPU batcher if memory permits.

Remaining work:
- Implement and benchmark the data path optimizations before starting another official 500-epoch run if wall-clock time is the priority.



### 2026-05-21 08:42 UTC - GPU-Resident Data Path Benchmark And Trainer Integration

Status: benchmarked optimized GPU-resident data loading, integrated it into the official trainer, and validated one official epoch with `batch_size=16`; no 500-epoch official run started.

Observed evidence:
- Pre-benchmark state had no active FNO training process and A100 memory at `0 MiB / 81920 MiB`.
- Standalone optimized benchmark precomputed train `x/y` tensors once and kept them resident on GPU, using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- GPU-resident benchmark results:
  - `batch_size=16`: completed `72/72` batches, mean batch `1.209s`, total `87.1s` (`1.45min`), peak memory `77.80 GiB`.
  - `batch_size=10`: completed `115/115` batches, mean batch `0.862s`, total `99.2s` (`1.65min`), peak memory `52.32 GiB`.
- The first official trainer integration attempt kept train and test tensors resident on GPU and failed with CUDA OOM during epoch 1 because memory was too close to the limit.
- The trainer was adjusted so only train tensors are resident on GPU; test data stays on CPU and is copied during evaluation.
- Updated `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` with `--data-residency {dataloader,gpu}`, GPU-resident train tensor materialization, manual GPU batching, and GPU-tensor final train evaluation.
- Updated `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` to default to `DATA_RESIDENCY=gpu` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Validation passed: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` and `bash -n 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`.
- Official wrapper validation command with `R2_UPLOAD=0 DATA_RESIDENCY=gpu EPOCHS=1 BATCH_SIZE=16 EVAL_BATCH_SIZE=16 EVAL_EVERY=10 PROGRESS_EVERY=20 SAVE_EVERY=999 AMP=0` completed.
- Official one-epoch output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs1_Tin10_T10_recurrent_pytorch_20260521_083756_UTC`.
- Official epoch 1 row reported `seconds=100.266` (`1m40s`) including training and epoch evaluation; complete one-epoch command total time was `2m31s` including final train/test evaluation and checkpoint save.

Inference:
- The data path optimization reduces one training epoch with batch size 16 from the previous stress-test wall time of `401.1s` (`6.69min`) to about `100s` (`1m40s`) when using the official trainer and epoch evaluation.
- `batch_size=16` with GPU-resident train data is close to the memory limit but passed the official one-epoch validation when test data was not kept resident on GPU.
- `batch_size=10` has much more memory headroom and still runs a full train-only epoch in under two minutes in the standalone optimized benchmark.
- For a 500-epoch official run with eval every 10 epochs, the observed epoch timing suggests a rough lower-bound training time near `14h` plus periodic evaluation, final evaluation/checkpointing, and final R2 upload. This is a large improvement over the earlier `~65h` estimate but is not yet evidenced as under `10h`.

Remaining work:
- Use the optimized official wrapper for the next long run if speed is the priority.
- Monitor the first several official epochs for OOM because batch size 16 remains close to the memory ceiling.
- If OOM occurs, switch to `BATCH_SIZE=10` or `14` while keeping `DATA_RESIDENCY=gpu`.



### 2026-05-21 08:48 UTC - Original Script Comparison

Status: compared the original local recurrent FNO2d training script with the newer CLI after the user questioned whether the slow timing came from deviating from the original code; no official training run started.

Observed evidence:
- Original script trainFNO2d_unnormalized.py imports the same FNO2d and RecurrentPredictor model classes from 2D_NS_FNO2d_recurrent/models/FNO2d.py.
- Original script uses hard-coded batch_size=20, ntrain=1000, ntest=100, modes=96, width=80, and dataset dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt.
- Original script precomputes x_train, y_train, x_test, and y_test once from the loaded tensor, then uses torch.utils.data.TensorDataset and DataLoader.
- The newer CLI also uses the same model classes and recurrent autoregressive loop, but was added for GPU verification, separate train/test path support, epoch timing/progress logs, R2 upload, and safer CLI configurability.
- The early slow CLI path initially used batch_size=4 and a custom NSTrajectoryDataset.__getitem__() that sliced each sample and called contiguous per sample before DataLoader collation.
- Later optimization added DATA_RESIDENCY=gpu to precompute train tensors and keep them resident on GPU, reducing official one-epoch time with batch_size=16 to about 100.266s including epoch evaluation.

Inference:
- The core model was not replaced with a different model, but the early CLI was not a line-for-line copy of the original training script.
- The original script precomputed-tensor data path should be faster than the early custom Dataset path.
- The earlier 6-8min/epoch timings were mainly caused by conservative batch size and the inefficient early data path, not by the FNO model itself.

Remaining work:
- Continue using the optimized DATA_RESIDENCY=gpu path for speed, or keep a compatibility path that mirrors the original precomputed TensorDataset approach if exact original semantics are required.



### 2026-05-21 08:55 UTC - 10-12 Hour Run Configuration

Status: configured the optimized recurrent FNO2d wrapper for a 10-12 hour target run and estimated batch-size-4 runtime; no long official run remains active from this turn.

Observed evidence:
- Optimized batch-size-16 official one-epoch validation previously reported epoch seconds `100.266s` including epoch evaluation, and standalone train-only GPU-resident timing was `87.1s` per epoch.
- Optimized batch-size-4 official one-epoch validation completed with epoch seconds `152.087s` (`2m32s`) including epoch evaluation; full one-epoch command including final evaluation/checkpoint save took `3m22s`.
- The wrapper default was changed from `EPOCHS=500` to `EPOCHS=450` in `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`.
- The CLI parser default was changed from `epochs=500` to `epochs=450` in `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`.
- Validation passed after the epoch edit: `bash -n 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` and `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`.
- Existing trainer logic saves `checkpoints/best.pt` whenever evaluated test relative L2 reaches a new finite minimum and final R2 upload copies the full output directory, including `checkpoints/best.pt`.
- A short attempt to launch a detached background training process from the tool shell did not persist in this execution environment: no active matching training process remained, no new 450-epoch output directory was created, and the temporary nohup log was empty.

Inference:
- For the optimized batch-size-16 path, `450` epochs is a reasonable 10-12 hour target: about `11h` using train-only plus periodic-eval estimate, or up to about `12.5h` using the conservative one-epoch wall time.
- `1000` epochs would likely exceed the target window substantially, roughly `24h+` under the optimized batch-size-16 timings.
- With the optimized data path, batch-size-4 is no longer the earlier `~65h/500 epochs` case, but it is still slower than batch-size-16: about `17-19h` for 450 epochs or `19-21h` for 500 epochs based on the one-epoch validation.

Remaining work:
- Start the official 450-epoch run from a persistent shell using the wrapper command, with R2 credentials in environment variables and `R2_UPLOAD=1`.
- Monitor early epochs for OOM because batch-size 16 remains close to the memory ceiling.



### 2026-05-21 09:00 UTC - Default Epochs Restored To 500 And Runtime Summary Added

Status: changed optimized recurrent FNO2d defaults back to 500 epochs and added a dedicated runtime/optimization summary Markdown.

Observed evidence:
- Updated `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` default from `EPOCHS=450` to `EPOCHS=500`.
- Updated `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` parser default from `epochs=450` to `epochs=500`.
- Added `docs/fno2d_recurrent_runtime_optimization_summary_20260521.md` summarizing GPU, runtime, memory, data path optimization, AMP status, best-checkpoint behavior, and R2 upload behavior.
- No active FNO training process was running before this edit.

Inference:
- The official default now matches the user request for 500 epochs while preserving optimized `BATCH_SIZE=16` and `DATA_RESIDENCY=gpu` behavior.

Remaining work:
- Commit and push the source and Markdown updates to the `vast-ai` branch.


### 2026-05-21 09:08 UTC - Active 500-Epoch FNO2d Run First-Three-Epoch Monitor

Status: monitored the user's active optimized recurrent FNO2d run through the first three completed epochs; training remained active after observation.

Observed evidence:
- Process observed: PID `58013`, `adv_robust/bin/python -u 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`.
- Output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC`.
- Source files involved: `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`, `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`, and model definitions under `2D_NS_FNO2d_recurrent/models/FNO2d.py`.
- Key settings observed from the running process and output files: `EPOCHS=500`, `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `EVAL_EVERY=10`, `DATA_RESIDENCY=gpu`, `AMP=0`, `R2_UPLOAD=1`, `R2_SYNC_EVERY_EPOCHS=0`.
- GPU path observed from `gpu_verification.txt`: PyTorch `2.8.0+cu126`, CUDA `12.6`, GPU `NVIDIA A100-SXM4-80GB`, compute capability `sm_80`, and a passing CUDA sanity matmul.
- GPU utilization observed with `nvidia-smi`: about `81121 MiB / 81920 MiB` in use and `100%` GPU utilization during training.
- Metrics observed from `train_log.csv`:
  - Epoch 1: `93.884s`, train relative L2 mean `0.610978`, test relative L2 mean `0.430839`, test MSE `0.365911`.
  - Epoch 2: `83.839s`, train relative L2 mean `0.347183`; no test evaluation on this epoch.
  - Epoch 3: `83.837s`, train relative L2 mean `0.279608`; no test evaluation on this epoch.
- Progress observed from `progress_latest.json`: the run was in epoch `4/500`, batch `30/72` after the first three epochs.

Inference:
- The active run is using the intended optimized GPU-resident training path, not the earlier slow `batch_size=4` CPU-collation path.
- The first three epoch timings imply about `87.19s/epoch` averaged so far; the trainer estimated total runtime `12h06m36s` after epoch 3.
- Using epochs 2-3 as the non-evaluation baseline and epoch 1 as the evaluation-overhead sample, the practical estimate is about `11h45m-12h10m` before final R2 upload variability.

Remaining work:
- Continue monitoring near epoch 10 to observe the next scheduled test evaluation.
- Watch for CUDA OOM because `BATCH_SIZE=16` intentionally uses most of the A100 80GB memory.
- Verify final checkpoint files and R2 upload after training completes.

### 2026-05-21 09:12 UTC - Active 500-Epoch FNO2d Run Six-Epoch Status Update

Status: inspected the active optimized recurrent FNO2d run again after epoch 6; training remained active.

Observed evidence:
- `train_log.csv` reached epoch 6 in output directory `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC`.
- Epoch 4: `83.826s`, train relative L2 mean `0.256466`.
- Epoch 5: `83.882s`, train relative L2 mean `0.242491`.
- Epoch 6: `83.918s`, train relative L2 mean `0.232962`.
- The trainer estimate after epoch 6 was total runtime `11h39m03s`, with `11h30m27s` remaining.
- `progress_latest.json` showed epoch `7/500`, batch `40/72`.
- `nvidia-smi` still showed about `81121 MiB / 81920 MiB` in use and `100%` GPU utilization.

Inference:
- Non-evaluation epoch timing has stabilized around `83.8-83.9s`.
- Current runtime expectation is about `11h40m-12h` before final R2 upload variability.

Remaining work:
- Check epoch 10 to measure scheduled evaluation overhead during the steady-state run.

### 2026-05-21 09:18 UTC - Active 500-Epoch FNO2d Run Epochs 7-10 Timing Update

Status: inspected the active optimized recurrent FNO2d run after epoch 10; training remained active.

Observed evidence:
- `train_log.csv` reached epoch 10 in output directory `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC`.
- Epoch 7: `83.929s`, train relative L2 mean `0.225809`; no test evaluation on this epoch.
- Epoch 8: `83.859s`, train relative L2 mean `0.218148`; no test evaluation on this epoch.
- Epoch 9: `83.889s`, train relative L2 mean `0.211692`; no test evaluation on this epoch.
- Epoch 10: `99.226s`, train relative L2 mean `0.206796`, test relative L2 mean `0.225781`, test MSE `0.102908`.
- After epoch 10, the trainer reported estimated total runtime `12h04m39s` and ETA `11h50m12s`.
- `progress_latest.json` showed epoch `11/500`, batch `30/72`.
- `nvidia-smi` still showed about `81121 MiB / 81920 MiB` allocated and `100%` GPU utilization.

Inference:
- Non-evaluation epochs are stable around `83.9s`.
- The epoch-10 scheduled test evaluation added roughly `15s` over a normal epoch.
- The run is still tracking about `12h` total before final R2 upload variability.

Remaining work:
- Continue periodic checks at later evaluation epochs and verify final R2 upload after training completes.

### 2026-05-21 09:19 UTC - Active 500-Epoch FNO2d Run Epoch 11 Quick Check

Status: inspected the active run after epoch 11; training remained active.

Observed evidence:
- Epoch 11 completed in `83.894s`, train relative L2 mean `0.203593`.
- `progress_latest.json` showed epoch `12/500`, batch `1/72`.

Inference:
- Training returned to the normal non-evaluation epoch timing after the longer scheduled epoch-10 evaluation.

Remaining work:
- Continue periodic checks at later evaluation epochs and verify final R2 upload after training completes.

### 2026-05-21 09:20 UTC - Active Run GPU Memory Headroom Check

Status: inspected current A100 memory usage and reviewed the recurrent FNO2d training loop for obvious GPU tensor retention patterns; training remained active.

Observed evidence:
- `nvidia-smi` reported `81121 MiB` used, `32 MiB` free, `81920 MiB` total on `NVIDIA A100-SXM4-80GB`, with `100%` GPU utilization.
- Per-process GPU memory query showed the active Python compute process using `81112 MiB`.
- `progress_latest.json` showed epoch `12/500` complete, elapsed `17m18s`, estimated total `12h04m38s`, and ETA `11h47m21s`.
- `train_log.csv` showed epoch 12 completed in `83.978s` with train relative L2 mean `0.196368`.
- Code inspection of `train_fno2d_recurrent_cli.py` found scalar logging via `float(...detach().cpu())`, `@torch.no_grad()` evaluation, and no obvious list/dict accumulation of GPU tensors or retained computation graphs in the epoch loop.

Inference:
- Reported memory is approximately `79.22 GiB / 80 GiB`; the run is intentionally near the memory ceiling.
- The stable high `nvidia-smi` number is consistent with PyTorch's CUDA caching allocator reserving memory after the first peak, not necessarily with live tensors growing every epoch.
- Because batch shapes and evaluation shapes are fixed and epoch 10 evaluation already completed, a gradual leak-driven OOM is not currently evidenced. The remaining risk is low but not zero because the free headroom is extremely small and allocator fragmentation or an unexpected larger temporary allocation could still OOM.

Remaining work:
- Continue monitoring memory around later evaluation/checkpoint epochs, especially epoch 20 and epoch 25.
- If OOM occurs, restart with `BATCH_SIZE=14` or `BATCH_SIZE=10` for more headroom.

### 2026-05-21 09:24 UTC - Active Run GPU Memory Trend Check

Status: sampled A100 memory repeatedly while the active recurrent FNO2d run continued; training remained active.

Observed evidence:
- Four consecutive `nvidia-smi` samples at about 10-second intervals reported `81121 MiB` used, `32 MiB` free, `81920 MiB` total, and `100%` GPU utilization.
- `progress_latest.json` showed epoch `15/500`, batch `50/72`.
- `train_log.csv` had reached epoch 14; epochs 11-14 remained normal non-evaluation epochs around `83.9-84.0s`.

Inference:
- No upward GPU-memory trend was observed across the repeated samples.
- The current behavior still looks like a stable high-water/cached allocation near the memory ceiling, not a visible per-epoch leak.
- OOM risk remains nonzero only because the free headroom is tiny, not because a rising trend has been evidenced.

Remaining work:
- Recheck memory after epoch 20 evaluation and epoch 25 checkpoint save.

### 2026-05-21 09:28 UTC - Exponax T20 Train/Test GIF Visualization

Status: generated CPU-only coolwarm heatmap GIFs for five random train samples and five random test samples; active GPU training remained running.

Observed evidence:
- Plotting command ran with `CUDA_VISIBLE_DEVICES=''` and printed `torch_cuda_available=False`.
- Source train dataset: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`.
- Source test dataset: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`.
- Observed train tensor shape: `(1150, 256, 256, 21)`.
- Observed test tensor shape: `(50, 256, 256, 21)`.
- Random seed: `20260521`.
- Selected train sample indices: `131`, `831`, `907`, `927`, `1084`.
- Selected test sample indices: `4`, `30`, `32`, `35`, `45`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926`.
- PIL validation confirmed all 10 GIFs are `256x256` with `21` frames.
- Active training progress after generation: epoch `18/500`, batch `50/72`, with A100 GPU utilization still at `100%`.

Inference:
- The selected train/test samples are in the expected `(256, 256, 21)` frame layout and were rendered without using GPU.
- The CPU visualization did not interrupt the active GPU training process.

Remaining work:
- User should visually inspect the GIFs and decide whether to generate more samples or a shared-global-scale version.

### 2026-05-21 09:33 UTC - Disable Periodic Evaluation And Checkpoint Defaults

Status: updated recurrent FNO2d training defaults to avoid periodic test evaluation and periodic checkpoint writes for future runs; the already-running PID 58013 still has its original command-line arguments.

Observed evidence:
- Active PID 58013 command line still includes `--eval-every 10` and `--save-every 25`; launched processes cannot pick up changed defaults without restart.
- Current run reached epoch 21 after completing the epoch-20 scheduled evaluation.
- `train_log.csv` showed epoch 20 took `99.673s`, while nearby non-evaluation epochs 16-19 took about `83.97-84.01s`.
- Current checkpoint directory contains `best.pt` with size about `2.7G`; no `latest.pt` had been written yet at inspection time.
- Updated `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh` defaults from `EVAL_EVERY=10` to `0`, and from `SAVE_EVERY=25` to `0`.
- Updated `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` parser defaults to `--eval-every 0` and `--save-every 0`.
- Updated trainer logic so `--save-every 0` disables in-loop `latest.pt` checkpoint saves.
- Updated trainer logic so `--eval-every 0` skips final train/test evaluation and uses last epoch training metrics plus NaN test metrics in final results.
- Validation passed: `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py` and `bash -n 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`.

Inference:
- Future wrapper runs will train through 500 epochs without scheduled test evaluation and without periodic latest checkpoints, then save final artifacts at the end.
- This removes the roughly `15-16s` scheduled evaluation overhead every 10 epochs and avoids repeated multi-GB checkpoint writes every 25 epochs.
- To apply this to the active run, the current process must be stopped and restarted with the new defaults or explicit `EVAL_EVERY=0 SAVE_EVERY=0`.

Remaining work:
- Decide whether to stop and restart the active run, accepting that the already completed epochs in the old run will not continue into the restarted run unless resume support is added.

### 2026-05-21 09:41 UTC - Epoch-25 Checkpoint Save Overhead Measurement

Status: measured the active old-argument run after its first `save_every=25` latest checkpoint write; training remained active.

Observed evidence:
- `checkpoints/latest.pt` appeared at epoch 25 with size about `2.7G`; `checkpoints/best.pt` also exists at about `2.7G`.
- Checkpoint directory total size became about `5.3G`.
- `train_log.csv` row for epoch 25 reported `seconds=84.0115s` and `elapsed_total_seconds=2147.7380s`; note this row is written before the in-loop checkpoint save.
- `train_log.csv` row for epoch 26 reported `seconds=84.0250s` and `elapsed_total_seconds=2234.5126s`.
- Inferred gap not explained by epoch 26 training time: `2234.5126 - 2147.7380 - 84.0250 = 2.75s`.

Inference:
- The first observed `latest.pt` checkpoint write cost roughly `2.7-3.0s` on this machine, not minutes.
- If repeated every 25 epochs through 500 epochs, the latest-checkpoint write overhead alone would be roughly `20 * 3s = 60s`, while the disk writes would still be unnecessary churn and temporary storage pressure.
- This measurement excludes scheduled test evaluation overhead, which was about `15-16s` on epoch 20.

Remaining work:
- Future runs should keep `SAVE_EVERY=0` and `EVAL_EVERY=0` as already changed, unless intermediate recovery checkpoints or test metrics are explicitly needed.

### 2026-05-21 09:43 UTC - Active 500-Epoch Runtime Projection After 28 Epochs

Status: projected total runtime for the active old-argument recurrent FNO2d run using completed epochs 1-28; training remained active.

Observed evidence:
- Active process elapsed from `ps`: about `41m35s`.
- `train_log.csv` completed epoch 28 with `elapsed_total_seconds=2402.501s` (`40m03s`) and epoch 28 time `83.975s`.
- Current `progress_latest.json` showed epoch `28/500`, batch `70/72`, with total train-batch ETA about `11h15m04s`.
- Stable non-evaluation epoch mean from recent completed epochs was about `83.982s`.
- Observed evaluation overhead from completed evaluation epochs was about `15.468s` over a normal epoch.
- Observed epoch-25 latest-checkpoint overhead was about `2.75s`.

Inference:
- Remaining training time after epoch 28 is about `11h13m54s` without assuming future best-checkpoint overwrites, or about `11h16m06s` if every future evaluation also overwrites `best.pt`.
- Projected total training time is about `11h54m-11h56m` from training-loop start.
- Using current time `2026-05-21 09:43:11 UTC`, projected training completion is about `2026-05-21 20:57-20:59 UTC`, before final R2 upload variability.

Remaining work:
- Continue monitoring around later evaluation/checkpoint epochs and verify final R2 upload completion separately.

### 2026-05-21 11:05 UTC - Active Run Health Check At Epoch 86

Status: inspected the active recurrent FNO2d 500-epoch run; training remained active with no observed runtime failure.

Observed evidence:
- Current UTC time: `2026-05-21T11:05:20Z`.
- Active process PID `58013` had `ps` elapsed time `02:03:43`, state `Rl+`.
- `progress_latest.json` showed epoch `86/500`, batch `40/72`, completed train batches `6160/36000`, run elapsed `7329.845s`, and train-batch ETA `9h51m47s`.
- Latest completed epoch in `train_log.csv` was epoch 85 with elapsed total `2h01m23s`; recent non-evaluation epochs remained around `83.9-84.0s`.
- Epoch 80 scheduled evaluation completed normally with test relative L2 mean `0.146012` and test MSE `0.0452241`.
- `nvidia-smi` reported `NVIDIA A100-SXM4-80GB`, `81121 MiB` used, `32 MiB` free, `81920 MiB` total, `100%` GPU utilization, `72%` memory-controller utilization, temperature `52 C`, and power draw about `218.74 W`.

Inference:
- No evidence of training failure, OOM, stalled GPU, or runaway memory growth was observed at this check.
- The active run remains on roughly the same completion track, with an approximate finish around `2026-05-21 20:57 UTC` before final R2 upload variability.

Remaining work:
- Continue periodic health checks and verify final checkpoint/R2 upload after completion.

### 2026-05-21 19:43 UTC - Active Run Progress Check At Epoch 449

Status: inspected the active recurrent FNO2d 500-epoch run near completion; training remained active.

Observed evidence:
- Current UTC time: `2026-05-21T19:43:22Z`.
- Active process PID `58013` had `ps` elapsed time `10:41:51`.
- `progress_latest.json` showed epoch `449/500`, batch `30/72`, completed train batches `32286/36000`, run elapsed `38412.897s`, and train-batch ETA `1h13m39s`.
- Current total train-batch completion was `89.6833%`.
- Latest completed epoch in `train_log.csv` was epoch 448 with elapsed total `10h39m38s`; recent non-evaluation epochs remained around `83.94-83.96s`.
- `nvidia-smi` reported `81121 MiB` used, `32 MiB` free, `81920 MiB` total, `100%` GPU utilization, temperature `57 C`, and power draw about `208.99 W`.

Inference:
- No evidence of training failure, OOM, stalled GPU, or runaway memory growth was observed.
- The active run is roughly on track to finish training around `2026-05-21 20:57 UTC`, before final checkpoint/R2 upload variability.

Remaining work:
- Monitor final epochs and verify final checkpoint plus R2 upload completion.

### 2026-05-21 20:04 UTC - Active Run Latest Train/Test Loss Check

Status: inspected the active recurrent FNO2d run metrics; training remained active.

Observed evidence:
- Current UTC time: `2026-05-21T20:04:07Z`.
- Active process PID `58013` had `ps` elapsed time `11:02:33`.
- `progress_latest.json` showed epoch `463/500`, batch `50/72`, completed train batches `33314/36000`, run elapsed `39650.645s`, and train-batch ETA `53m17s`.
- Latest completed epoch in `train_log.csv` was epoch 462.
- Epoch 462 train relative L2 mean was `0.05333483872206315`, train MSE was `0.0057479435699465484`, and train score `1 - relative_l2_mean` was `0.9466651612779369`.
- Latest completed test evaluation was epoch 460.
- Epoch 460 test relative L2 mean was `0.08675776571035385`, test MSE was `0.0179227876663208`, and test score `1 - relative_l2_mean` was `0.9132422342896461`.
- Current in-progress batch loss from `progress_latest.json` was `0.9232521057128906` at epoch 463 batch 50.

Inference:
- Test metrics are only updated on scheduled evaluation epochs; epoch 461 and 462 test fields are NaN because they were not evaluation epochs, not because the run failed.
- No evidence of training failure was observed in this metric check.

Remaining work:
- Continue monitoring final epochs and verify final checkpoint plus R2 upload completion.

### 2026-05-21 20:06 UTC - Active Run Loss Trend Check

Status: inspected `train_log.csv` for train/test loss trend in the active recurrent FNO2d run; only log files were read.

Observed evidence:
- Latest completed epoch was epoch 464; active progress was epoch `465/500`, batch `20/72`.
- Train relative L2 mean decreased from `0.610978` at epoch 1 to `0.053272` at epoch 464, a drop of about `91.28%`.
- Train MSE decreased from `0.791622` at epoch 1 to `0.005734` at epoch 464.
- Test relative L2 mean decreased from `0.430839` at epoch 1 to `0.086758` at epoch 460, a drop of about `79.86%`.
- Test MSE decreased from `0.365911` at epoch 1 to `0.017923` at epoch 460.
- Selected train relative L2 means: epoch 10 `0.206796`, epoch 100 `0.111969`, epoch 200 `0.094641`, epoch 300 `0.070758`, epoch 400 `0.056783`, epoch 450 `0.053651`, epoch 464 `0.053272`.
- Recent test relative L2 means: epoch 410 `0.087922`, epoch 420 `0.087647`, epoch 430 `0.087481`, epoch 440 `0.087220`, epoch 450 `0.086976`, epoch 460 `0.086758`.

Inference:
- The loss curves show a clear downward trend.
- The largest improvement happened early; by epochs 400-460 the test curve is improving slowly and appears close to a plateau, but the latest evaluated test loss is still the best observed test relative L2 so far.

Remaining work:
- Recheck final test evaluation and final R2 upload after training completes.

### 2026-05-22 00:33 UTC - NS2D Dictionary Batch-500 Generation And GIF Visualization

Status: completed local batch-500 dictionary generation and CPU GIF visualization.

Observed evidence:
- GPU verification before launch showed PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, `NVIDIA A100-SXM4-80GB`, compute capability `(8, 0)`, JAX backend `gpu`, and JAX device `CudaDevice(id=0)`.
- Generation command used `--nsamples 2000`, `--batch-size 500`, `--solver-batch-size 500`, `--solver-mode vmap`, and `--step-options 0.01,0.0005` from `2D_NS_FNO2d_recurrent`.
- Source log: `2D_NS_FNO2d_recurrent/data_generation/logs_gen/generate_ns_dictionary_batched_b500_20260522_001856_UTC.out`.
- The tqdm generation loop completed 4/4 batches in `11:07`, with displayed average `166.91s/batch`.
- `nvidia-smi` during generation observed about `17477 MiB / 81920 MiB` used and `99%` GPU utilization.
- Output dictionary path: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Output dictionary exact size: `11534339833` bytes, about `10.74 GiB`.
- Replacement summary JSON path: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/generation_summary_dictionary_batched.json`.
- Summary shapes: `x_shape=[2000,256,256]`, `y_shape=[2000,256,256,21]`.
- Metadata step usage: `dt=0.01` for 1859 samples and fallback `dt=0.0005` for 141 samples.
- CPU-only GIF visualization generated 5 `coolwarm` 21-frame GIFs under `2D_NS_FNO2d_recurrent/visualizations/dictionary_batched_20260522/`, using sample indices `0,499,999,1499,1999`.

Inference:
- Batch-500 true NS2D dictionary generation completed without OOM on the A100.
- The observed solver loop time was about `11m07s`, not `3-4m`; the difference is consistent with actual full generation, fallback handling, and save/metadata overhead.
- The generation script wrote the `.pt` successfully before hitting a JSON serialization bug; the script has now been fixed with `json_ready()` for future runs.
- The generated dictionary and GIFs are large/generated artifacts and were not staged for git in this turn.

Remaining work:
- Upload the dictionary to R2 if the next workflow needs remote storage.
- Point attack modes that use dictionary `A` frames at the generated `.pt` file before running the full attack sweep.

### 2026-05-22 00:41 UTC - NS2D Dictionary Stability Scan After Batch-500 Generation

Status: inspected generated dictionary stability and added stricter validation tooling.

Observed evidence:
- Source dictionary: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- CPU-only stability scan script: `2D_NS_FNO2d_recurrent/data_generation/validate_ns_dictionary_stability.py`.
- Stability report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched.json`.
- Top-outlier CSV: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_top_outliers.csv`.
- Scan results: `finite_count=2000`, `nonfinite_count=0`, and `robust_flagged_count=59` using robust multiplier `20.0`.
- Robust thresholds from the report: `max_abs_threshold=6.987165093421936`, `max_rms_threshold=2.1462059020996094`.
- Largest observed outliers include index `566` with `max_abs=237200850944.0` and index `1998` with `max_abs=193768416.0`.
- `generate_ns_dictionary_batched.py` was updated to support optional `--max-abs-threshold` and `--max-rms-threshold` so finite but oversized trajectories can be rerun with the next step option.

Inference:
- The user's concern is valid: the original finite-only filter caught NaN/Inf instability but did not catch finite blow-up trajectories.
- The current generated dictionary should not be considered fully clean for dictionary-based attack modes until the 59 robust outliers are regenerated or the full dataset is regenerated with strict thresholds.

Remaining work:
- Decide the final stability threshold policy, then either patch/regenerate flagged indices or rerun the full dictionary with strict thresholds enabled.

### 2026-05-22 00:54 UTC - In-Place Repair Of Exploded NS2D Dictionary Samples

Status: completed in-place repair of finite but exploded dictionary trajectories.

Observed evidence:
- Canonical dictionary repaired in place: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Repair script: `2D_NS_FNO2d_recurrent/data_generation/repair_ns_dictionary_unstable_samples.py`.
- Repair report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_report.json`.
- Repair criterion: non-finite samples or `max_abs > 10`.
- Targeted unstable sample count: `56`.
- Rerun step options were `0.0005,0.0001`; all 56 targeted samples passed at `dt=0.0005`, so `dt=0.0001` was not needed.
- Repair total runtime: `254.23267521499656s`; `dt=0.0005` rerun step runtime: `134.64052360795904s`.
- Repair failed count: `0`.
- Post-repair scan report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_after_inplace_repair_maxabs10.json`.
- Post-repair scan observed `finite_count=2000`, `nonfinite_indices=[]`, and zero samples flagged by the explicit `max_abs > 10` threshold.
- Post-repair maximum `max_abs` is `9.996569633483887`; post-repair maximum RMS is `1.2635504007339478`.

Inference:
- The obviously exploded finite trajectories were replaced directly inside the canonical dictionary path.
- The dictionary now satisfies the explicit large-value filter requested in this turn (`max_abs <= 10`).
- A few adaptive-IQR robust outliers remain below 10; whether to rerun those too is a separate stricter policy choice.

Remaining work:
- Upload the repaired canonical dictionary to R2 if remote attack jobs need the corrected file.

### 2026-05-22 01:04 UTC - NS2D Recurrent Train/Test Dataset Generation Time And Stability Check

Status: inspected local train/test dataset generation timing and scanned the actual `.pt` files for exploded values.

Observed evidence:
- Train dataset path: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`.
- Test dataset path: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`.
- Timing source: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/generation_summary.json`.
- Stability scan report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/training_dataset_stability_report_20260522.json`.
- Per-sample scan CSV: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/training_dataset_stability_per_sample_20260522.csv`.
- Dataset generation settings recorded in metadata: `fixed_step=0.005`, `processing_batch_size=32`, `solver_batch_size=8`, `solver_mode=lax-map`, and 21 saved frames through `tfinal=20`.
- Recorded train generation time: `2278.9064770182595s` rollout+upsample for 1150 samples, about `37m58.9s`.
- Recorded test generation time: `104.51481727696955s` rollout+upsample for 50 samples, about `1m44.5s`.
- Combined recorded train+test generation time: `2383.421294295229s`, about `39m43.4s`.
- Train stability scan observed `finite_count=1150`, `max_abs>10` count `0`, `max_abs>50` count `0`, `max_abs>100` count `0`, and `max_abs>1000` count `0`; train maximum `max_abs=3.8361892700195312`.
- Test stability scan observed `finite_count=50`, `max_abs>10` count `0`, `max_abs>50` count `0`, `max_abs>100` count `0`, and `max_abs>1000` count `0`; test maximum `max_abs=3.4770100116729736`.

Inference:
- The train/test datasets used by the recurrent FNO2d training run do not show the finite blow-up problem seen in the dictionary dataset.
- The recorded generation time is from per-batch metadata and may exclude small process startup/final filesystem overhead.
- A separate 2025 B200 `.out` log found under `data_generation/logs_gen/VT_NS_gen_all_frame_6765843.out` is canceled/stale and should not be treated as the evidence for these current local `.pt` files.

Remaining work:
- If future regenerated training data uses adaptive `dt=0.01` first, rerun this same stability scan before training.

### 2026-05-22 01:15 UTC - In-Place Repair Of NS2D Dictionary Samples With Max Abs Greater Than 5

Status: stopped the in-progress R2 upload and completed stricter in-place dictionary repair.

Observed evidence:
- The active `rclone copy` upload process was stopped before the large `.pt` upload completed.
- Canonical dictionary repaired in place: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Repair script: `2D_NS_FNO2d_recurrent/data_generation/repair_ns_dictionary_unstable_samples.py`.
- Repair report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_maxabs5_report.json`.
- Repair criterion: non-finite samples or `max_abs > 5`.
- Targeted unstable sample count: `12`.
- Targeted indices: `76, 790, 820, 845, 971, 1057, 1078, 1150, 1287, 1670, 1704, 1923`.
- Rerun step option was `0.0001`; all 12 targeted samples passed at `dt=0.0001`.
- Repair total runtime: `432.052681391011s`; `dt=0.0001` rerun step runtime: `311.43955161608756s`.
- Repair failed count: `0`.
- Post-repair scan report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_after_inplace_repair_maxabs5.json`.
- Post-repair scan observed `finite_count=2000`, `nonfinite_indices=[]`, zero samples flagged by `max_abs > 5`, and zero robust-IQR flagged samples.
- Post-repair maximum `max_abs` is `4.837541103363037`; post-repair maximum RMS is `1.2635504007339478`.

Inference:
- The canonical dictionary now matches the requested stricter large-value policy (`max_abs <= 5`).
- The interrupted R2 upload happened before this stricter repair; R2 should be updated again before remote use.

Remaining work:
- Upload the max-abs-5 repaired canonical dictionary and reports to R2 when requested.

### 2026-05-22 01:28 UTC - In-Place Repair Of NS2D Dictionary Samples With Max Abs Greater Than 4

Status: completed stricter in-place dictionary repair for samples above `max_abs=4`.

Observed evidence:
- Canonical dictionary repaired in place: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Repair report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_maxabs4_report.json`.
- Pre-repair scan found 4 samples above `max_abs > 4`: indices `263, 819, 1357, 1694`.
- Rerun step option was `0.00005`; all 4 targeted samples passed at `dt=0.00005`.
- Repair total runtime: `386.4510918520391s`; `dt=0.00005` rerun step runtime: `270.24317298003007s`.
- Repair failed count: `0`.
- Post-repair scan report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_after_inplace_repair_maxabs4.json`.
- Post-repair scan observed `finite_count=2000`, `nonfinite_indices=[]`, zero samples flagged by `max_abs > 4`, and zero robust-IQR flagged samples.
- Post-repair maximum `max_abs` is `3.980586528778076`; post-repair maximum RMS is `1.2635504007339478`.

Inference:
- The canonical dictionary now satisfies the stricter large-value policy (`max_abs <= 4`).
- This brings dictionary maxima close to the train/test dataset scale.

Remaining work:
- Upload the max-abs-4 repaired canonical dictionary and reports to R2 if remote jobs need it.

### 2026-05-22 01:39 UTC - R2 Upload Of Final Max-Abs-4 NS2D Dictionary

Status: uploaded the final max-abs-4 repaired dictionary and reports to the user's Cloudflare R2 bucket.

Observed evidence:
- R2 target directory: `r2://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/`.
- Uploaded canonical dictionary: `dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- R2-reported dictionary size: `11534341969` bytes; local size: `11534341969` bytes.
- rclone reported the large `.pt` transfer completed in `4m27.1s`.
- Uploaded final maxabs4 report files:
  - `dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_maxabs4_report.json`, size `2807` bytes.
  - `stability_report_dictionary_batched_after_inplace_repair_maxabs4.json`, size `5350` bytes.
  - `stability_report_dictionary_batched_after_inplace_repair_maxabs4_top_outliers.csv`, size `4369` bytes.
- `ps` check after upload found no remaining `rclone`, repair, or stability scan process.

Inference:
- The R2 bucket now has the final dictionary version whose local post-repair scan showed `max_abs <= 4`, `finite_count=2000`, and zero robust-IQR flagged samples.
- Older maxabs10 report files remain in the same R2 directory; the current final evidence is the maxabs4 report set.

Remaining work:
- Use the R2 canonical `.pt` path above for dictionary-based attack jobs that need `A` mode.


### 2026-05-22 02:05 UTC - Prepared Full ADW NS2D Recurrent Core4 Attack Launch Command

Status: launch command prepared; attack was not started by this record update.

Observed evidence:
- Attack script supports mode presets `all_w`, `all_d_target_w`, `all_a_target_w`, `w1_5_d6_9_target_w`, `d1_5_w6_9_target_w`, and `a1_5_d6_9_target_w` in `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Local trained checkpoint exists: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`.
- Local repaired dictionary exists: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Wrapper updated: `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh` now forwards `DICTIONARY_PATH`, `OUT_ROOT`, `TRUE_LOSS_EVERY`, and `FIXED_STEP` to the Python attack CLI.
- Dedicated launch record created: `docs/ns2d_recurrent_core4_attack_full_adw_launch_20260522.md`.

Key settings prepared:
- Indices: `0,1,2,3,4,5,6,7,8,9`.
- Attack batch size: `10`.
- Steps: `100`.
- Norms: `p=2`, `q=2`.
- Budget/step: `epsilon=32`, `alpha=1`.
- True-loss logging: `TRUE_LOSS_EVERY=1`.
- Solver rematerialization: `chunk`, `20` micro-steps per chunk.
- JAX memory: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`.
- Output root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522`.

Inference:
- Correct grouped experiment set is 7 groups and 28 method combinations, not the old full Cartesian 72-combination sweep.
- Practical runtime estimate remains broad until the first group completes: about 4 hours if the previous ~32.4 minute timing is per group including all four methods, up to about 15 hours if that timing applies per method combination.

Remaining work:
- Start the attack from the documented command if the GPU is free.
- After the first completed group, update this ledger with the actual group runtime and a refined total ETA.


### 2026-05-22 02:00 UTC - Status Check: Full ADW NS2D Attack Running

Status: attack is running; no intervention performed.

Observed evidence:
- Active Python command observed for `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Current group: `LOSS_TYPES=loss1`, `MODE_SPEC=all_w`, `ATTACK_BATCH_SIZE=10`, `STEPS=100`, `EPSILON=32`, `ALPHA=1`.
- Launch log: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/launch_20260522_015450_UTC.log`.
- Active run directory: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_wwwwwwwwww_p2_q2_20260522_015452_UTC/`.
- Manifest records JAX backend `gpu`, JAX device `cuda:0`, PyTorch CUDA available, device `NVIDIA A100-SXM4-80GB`, compute capability `sm_80`.
- Current `nvidia-smi` observed about `46851 MiB / 81920 MiB` and `99%` GPU utilization.
- No method CSV had been written yet at this check.

Inference:
- The attack has started and is working on the first group/method sequence; no completed method metrics were available yet.
- The run is on GPU, not CPU fallback.

Remaining work:
- Recheck after the first `per_step_metrics.csv` or `summary.json` appears to estimate total runtime from actual group timing.


### 2026-05-22 02:05 UTC - Runtime Estimate Update For Full ADW NS2D Attack

Status: estimate updated while first attack group was running.

Observed evidence:
- At `2026-05-22T02:04:38Z`, active process was still the first group: `LOSS_TYPES=loss1`, `MODE_SPEC=all_w`.
- Launch log first group timestamp: `2026-05-22T01:54:51Z`; elapsed time at estimate was about `9m47s`.
- No `per_step_metrics.csv` or method `summary.json` existed yet for the active run, so the first method had not completed.
- Historical local benchmark summaries for similar `chunk=20` attack settings showed about `89.6s` for `steps=5`, `batch_size=12`, or roughly `17.9s/update` before extrapolation.

Inference:
- For this full command with `TRUE_LOSS_EVERY=1`, the total is more likely in the `12-16h` range than the earlier optimistic `4h` estimate.
- Conservative broad range remains `10-18h` until the first method finishes and provides direct timing.
- Approximate finish window from launch: `2026-05-22 14:00-18:00 UTC`, broad conservative window `2026-05-22 12:00-20:00 UTC`.

Remaining work:
- Re-estimate after the first method writes `summary.json`; that will make the ETA much tighter.


### 2026-05-22 03:12 UTC - Runtime Progress Update For Full ADW NS2D Attack

Status: first group completed; second group running.

Observed evidence:
- Current time checked: `2026-05-22T03:12:13Z`.
- First group `LOSS_TYPES=loss1`, `MODE_SPEC=all_w` completed and wrote summary: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_wwwwwwwwww_p2_q2_20260522_015452_UTC/summary.json`.
- First group completed four method combinations with total runtime `4423.67s` (`73.73m`).
- Per-method runtimes in first group: `raw_add=1110.34s`, `raw_replace=1104.52s`, `steepest_add=1104.40s`, `steepest_replace=1104.40s`.
- Active process moved to second group: `LOSS_TYPES=loss2`, `MODE_SPEC=all_a_target_w`.
- Second group start logged at `2026-05-22T03:08:47Z`; elapsed at check was about `3m26s`.
- Current GPU status observed about `45029 MiB / 81920 MiB` with `99%` utilization.

Inference:
- Completed progress is `1/7` groups, equivalent to `4/28` method combinations.
- Best current point estimate is about `8h36m` total from launch, based on the first group's actual runtime.
- Remaining estimate at the check time is about `7h15m-7h30m`.
- Point finish ETA is about `2026-05-22 10:30 UTC`; practical range about `10:00-11:30 UTC`, with a conservative cushion to noon UTC if later groups slow down.

Remaining work:
- Recheck after `all_a_target_w/loss2` completes to see whether A-mode dictionary groups differ materially from the first all-W group.


### 2026-05-22 03:18 UTC - Semantic Clarification: loss1 Is Not An ADW Target-Mode Sweep

Status: code semantics inspected and clarified; no running process was changed.

Observed evidence:
- In `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`, `active_losses` sets `need_target = loss_type == "loss3"`.
- For `loss1`, `g_delta` target computation is not used; the objective is `||pred - f0||_q`.
- `mode_spec` still affects `loss1` indirectly through `model_prediction`, because the FNO input requires the first `t_in=10` frames and the first nine `mode_spec` characters choose how frames 2-10 are generated.
- Current launch already completed only one `loss1` group, using `all_w`, and is now running `loss2/all_a_target_w`.

Inference:
- The user's correction is conceptually right: ADW target modes are not a parallel sweep axis for `loss1`.
- The completed `loss1/all_w` group should be interpreted as `loss1` with canonical differentiable solver context, not as an ADW target-mode condition.
- Future result summaries should separate `loss1` from ADW target-mode comparisons and avoid implying a Cartesian loss/mode design.

Remaining work:
- When analyzing final outputs, label the first group as `loss1/canonical_solver_context` or similar.
- If desired after the running launch is finished, rename CLI/report fields to distinguish input-context mode from target mode more clearly.


### 2026-05-22 03:24 UTC - Potential Mismatch Found In NS2D Attack ADW Semantics

Status: code semantics inspected; no process was stopped by this note.

Observed evidence:
- `MODE_PRESETS` in `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` resolve to 10-character strings.
- The current implementation uses the first nine characters of `mode_spec` in `model_prediction` to build the FNO input context frames for `t_in=10`.
- The final character is only used as the target-frame policy when `need_target=True`, which is currently only for `loss3`.
- For `loss1`, `need_target=False`, so the target mode is unused.

Inference:
- If the intended ADW policies are supposed to apply to frames 11-20 of the perturbed solver/target path, the current code likely implements the wrong semantics.
- The completed `loss1/all_w` group is not an ADW target-mode experiment; it is `loss1` with canonical solver-generated input context.
- The currently running and future A/D/W groups may not answer the intended question until the code separates input-context policy from target-path policy.

Remaining work:
- Stop or let the current launch continue based on user decision.
- Refactor the attack code before treating the A/D/W results as final evidence.


### 2026-05-22 03:32 UTC - Clarified No Cartesian loss1 By ADW Sweep In Active Launch

Status: inspected active launch log and output files; no process change performed.

Observed evidence:
- Launch log contains only one `loss1` run: `LOSS_TYPES=loss1 MODE_SPEC=all_w`.
- The next launch entry is `LOSS_TYPES=loss2 MODE_SPEC=all_a_target_w`.
- Completed `loss1` summary files exist only under `mode_wwwwwwwwww.../batch_0000_0009/loss1/` for `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- Current active process is `LOSS_TYPES=loss2`, `MODE_SPEC=all_a_target_w`.

Inference:
- The active launch did not run `loss1` against all ADW modes.
- The user's conceptual correction remains important: `loss1` is a single objective and should not be interpreted as an ADW target-mode sweep.

Remaining work:
- Continue to distinguish completed `loss1` as a single canonical context baseline when analyzing results.


### 2026-05-22 03:40 UTC - Audit: loss1 Solver Usage In Active NS2D Attack Code

Status: code audited; no process change performed.

Observed evidence:
- `active_losses` computes `loss1` as `batch_norm(pred - self.f0, q_order)` and sets `need_target = loss_type == "loss3"`.
- For `loss1`, the target branch `g_delta` is not computed in the optimized objective.
- `model_prediction` still calls `_rollout_for_modes` to build the recurrent FNO input context frames; with `mode_spec=all_w`, this rolls solver frames 1 through 9.
- `run_one` calls `true_loss_all_w(x_adv)` when `TRUE_LOSS_EVERY=1` for non-`loss3/all_w` objectives, so `loss1` also incurs a no-grad target-frame solver rollout for logging at each step.

Inference:
- The `loss1` group was not duplicated across ADW modes, but it did use solver computation.
- Solver-to-frame-9 usage is part of the current definition of attacking an initial condition for a recurrent FNO that needs 10 input frames.
- Solver-to-frame-20 usage for `true_loss_all_w` in loss1 is diagnostic logging, not the optimized loss1 objective; it can be skipped in future loss1 runs if that curve is not needed.
- If loss1 is intended to be pure model-only on fixed 10-frame input, current code should be changed before using loss1 results as evidence.

Remaining work:
- Decide whether to keep collecting true-loss curves for loss1 or disable them in future runs.
- Clarify and possibly refactor the loss1 input-context definition.


### 2026-05-22 03:48 UTC - Recorded loss1 Solver Usage And Naming Clarification

Status: created a dedicated Markdown clarification for the NS2D recurrent attack `loss1` semantics.

Observed evidence:
- Dedicated Markdown: `docs/ns2d_recurrent_attack_loss1_solver_usage_clarification_20260522.md`.
- Prior code inspection found `active_losses` uses `need_target = loss_type == "loss3"`, so `loss1` does not use the solver target branch.
- Prior code inspection found `model_prediction` still builds the recurrent FNO input context frames, so `loss1` can use solver frames 2..10 under the initial-condition attack setup.
- Prior code inspection found `TRUE_LOSS_EVERY=1` calls `true_loss_all_w(x_adv)` for diagnostic true-loss logging, separate from the optimized `loss1` objective.

Inference:
- The completed `loss1/all_w` label is a naming/interpretation problem, not evidence of a full `loss1 x ADW` duplicate sweep.
- Existing `loss1` results should be labeled as `loss1/canonical_solver_context` in analysis.
- If `loss1` true-loss diagnostics are not needed in future runs, they can be disabled to reduce solver overhead.

Remaining work:
- When writing final attack analysis, use the corrected `loss1/canonical_solver_context` wording.
- Consider future code refactor separating `context_policy` and `target_path_policy`.


### 2026-05-22 03:38 UTC - Conservative Runtime Estimate Revision For Full ADW NS2D Attack

Status: ETA revised upward based on completed `loss2/all_a_target_w` method timings and expected `loss3` cost.

Observed evidence:
- Current time checked: `2026-05-22T03:37:52Z`.
- Active process was still `LOSS_TYPES=loss2`, `MODE_SPEC=all_a_target_w`.
- Completed `loss1/canonical_solver_context` group total: `73.73m`, about `18.4m/method`.
- Completed `loss2/all_a_target_w` methods: `raw_add=7.77m`, `raw_replace=7.69m`, `steepest_add=7.69m`.
- GPU status observed about `45029 MiB / 81920 MiB` with `100%` utilization.

Inference:
- `loss2/all_a_target_w` is faster than `loss1`; it is not a good proxy for the remaining `loss3` groups.
- The remaining five `loss3` groups are likely the expensive part because the perturbed target path to frame 20 participates in the objective/backward path.
- If `loss3` matches `loss1`, remaining time after `loss2` is roughly `6.1h`; if `loss3` costs `25-35m/method`, remaining time is roughly `8.3-11.7h`.
- Revised practical remaining estimate at this check: `8-12h`.
- Revised finish window: about `2026-05-22 11:30-15:30 UTC`, with optimistic lower bound around `10:00 UTC` and conservative cushion to `16:00 UTC`.

Remaining work:
- Re-estimate after the first `loss3/all_w` method finishes; that will determine whether the conservative multiplier is necessary.


### 2026-05-22 03:52 UTC - Generated Fast loss1 Final Perturbation Visualizations

Status: generated CPU-only fast visualization panels from saved loss1 arrays; no model or solver rerun was completed.

Observed evidence:
- Visualization script created: `2D_NS_FNO2d_recurrent/visualizations/plot_loss1_attack_fast_panels.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_fast_panels_20260522/`.
- Report: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_fast_panels_20260522/loss1_fast_panel_report.json`.
- Generated 12 PNGs for four methods and sample positions 0, 1, 2.
- Report shows `delta_l2=0.0` and `delta_linf=0.0` for all plotted methods/samples.
- Direct metric inspection showed all four `loss1` methods have final `delta_p_mean=0.0`, final `loss1_mean=0.0`, and `boundary_ratio_mean=0.0`.
- A CPU-only full model-output plotting attempt was stopped because it consumed many CPU cores and could slow the active attack.

Inference:
- The completed `loss1` group did not produce a nonzero perturbation.
- The plotted perturbed initial and solver final conditions are identical to the clean versions because the saved final delta is zero.
- The model-vs-solver final-output panel still needs to be generated later when it is safe to run model inference/solver computation.
- The zero delta suggests the current loss1 objective can stall at zero initialization.

Remaining work:
- After the active attack finishes or the GPU is safely available, generate full model-vs-solver final-output plots for selected loss1 samples.
- Consider nonzero random initialization or an adjusted loss1 objective if nontrivial loss1 perturbations are desired.


### 2026-05-22 03:58 UTC - Root Cause Found For Zero-Delta loss1 Attack

Status: diagnosed from completed `loss1` per-step metrics and attack source code.

Observed evidence:
- `per_step_metrics.csv` for all four `loss1` methods shows `loss1_mean=0.0`, `delta_p_mean=0.0`, `boundary_ratio_mean=0.0`, `grad_l2_mean=0.0`, and `direction_l2_mean=0.0` at every optimization step.
- `final_delta_and_metrics.npz` confirms `final_delta` is exactly zero for all four methods.
- Source code initializes `delta = torch.zeros_like(problem.x0)` in `run_one`.
- Source code computes `self.f0 = self.model_prediction(self.x0, need_target=False)[0].detach()` and `loss1 = batch_norm(pred - self.f0, q_order)`.

Inference:
- At `k=0`, `x_adv=x0`, `pred=F(x0)`, and `f0=F(x0)`, so `loss1=0` exactly.
- PyTorch returns zero gradient for the exact-zero norm residual in this implementation, producing zero direction for all four update methods.
- The zero update repeats for all 100 steps, so the completed `loss1` group is a stalled attack, not a successful perturbation.
- This is not caused by too-small `epsilon` or `alpha`; it is caused by zero initialization combined with the zero-residual `loss1` objective.

Remaining work:
- For any future `loss1` run, add random/nonzero initialization or another nonzero seed direction before the first update.
- Treat the current completed `loss1` outputs as a diagnostic of stalling, not as evidence of robustness.


### 2026-05-22 03:58 UTC - Stopped Full ADW NS2D Attack Launch

Status: stopped the active full ADW attack launch on user request.

Observed evidence:
- Active `loss3/all_w` wrapper/Python/tee processes were terminated first.
- The outer shell loop automatically launched subsequent `loss3` modes after the first termination; a short stopper loop terminated those auto-restarted groups as well.
- Final process check found no matching `attack_ns2d_recurrent_core4`, `run_ns2d_recurrent_core4_attack`, `full_adw_b10_eps32_alpha1`, or loss1 visualization process.
- Final `nvidia-smi` showed `0MiB / 81920MiB`, `0%` GPU utilization, and no running GPU processes.
- Launch log shows `loss1/all_w` and `loss2/all_a_target_w` completed before stopping.
- Launch log shows `loss3/all_w` and later loss3 modes were started but not completed.

Inference:
- The previous full ADW launch is incomplete and should not be treated as a finished experiment.
- The GPU is available for a corrected rerun.
- Completed `loss1` results should be interpreted as a stalled zero-delta diagnostic; completed `loss2/all_a_target_w` may remain usable depending on the corrected experiment design.

Remaining work:
- Patch `loss1` handling before rerunning if `loss1` is included.
- Prepare a corrected launch command that prevents outer-loop continuation problems and uses safer failure behavior, such as `set -euo pipefail` at the top-level launch script.


### 2026-05-22 04:10 UTC - Patched NS2D Attack For Epsilon/Alpha Sweeps And loss1 Random Start

Status: source patch completed; no attack run started.

Observed evidence:
- Modified `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Modified `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh`.
- Added CLI support for `--epsilons`, `--alphas`, and `--epsilon-alpha-pairs`.
- Added wrapper support for `EPSILONS`, `ALPHAS`, and `EPSILON_ALPHA_PAIRS`.
- Added default `loss1` random start with `--loss1-random-start`, `--loss1-random-start-fraction`, and `--loss1-random-start-seed`.
- Default `loss1_random_start_fraction` is `0.001`, giving initial L2 radius `0.001 * epsilon`.
- Multiple epsilon/alpha settings are written under parameter-specific directories like `eps32_alpha1` to avoid overwriting outputs.
- Checks passed: Python `py_compile`, wrapper `bash -n`, and CLI `--help` option discovery.
- Dedicated patch note: `docs/ns2d_recurrent_attack_sweep_and_loss1_random_start_patch_20260522.md`.

Inference:
- Future `loss1` runs should no longer stall at exactly zero perturbation from zero initialization.
- Explicit paired sweeps such as `EPSILON_ALPHA_PAIRS="16:0.5 32:1 64:2"` avoid accidental Cartesian explosion while still testing multiple scales.

Remaining work:
- Prepare the corrected rerun command after deciding which epsilon/alpha pairs to use.
- Consider adding a tiny smoke run only after the user approves GPU usage.

### 2026-05-22 04:20 UTC - Loss2 Fast Visualization From Completed NS2D Attack Outputs

Status: summarized existing completed `loss2` outputs; no model or solver rerun.

Observed evidence:
- Source attack outputs: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_aaaaaaaaaw_p2_q2_20260522_030849_UTC/batch_0000_0009/loss2`
- Visualization output directory: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_fast_panels_20260522/`
- Report: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_fast_panels_20260522/loss2_fast_panel_report.json`
- Dedicated note: `docs/ns2d_recurrent_loss2_fast_visualization_20260522.md`
- The completed `loss2/all_a_target_w` group produced nonzero perturbations for all four methods.
- Final mean L2 perturbation was `12.0754` for `raw_add` and about `32.0` for `raw_replace`, `steepest_add`, and `steepest_replace`.
- Final true-loss ratio was about `1.0892` for `raw_add`, `1.1877` for `raw_replace`, `1.1984` for `steepest_add`, and `1.1877` for `steepest_replace`.

Inference:
- Unlike the stalled zero-delta `loss1` run, the completed `loss2` result is meaningful for scale inspection.
- `epsilon=32` is not trivially too small, because most methods reached the L2 boundary and produced visible nonzero perturbations.
- `alpha=1` is too weak for `raw_add` over 100 steps, while replacement methods saturate the budget immediately.
- A paired epsilon/alpha sweep such as `16:0.5 32:1 64:2` is still needed before selecting final attack parameters.

Remaining work:
- Generate full model-vs-solver visualization later if exact output comparison is needed.
- Rerun the corrected attack after the loss1 random-start and epsilon/alpha sweep patch, because the earlier full ADW launch was intentionally stopped.

### 2026-05-22 04:25 UTC - Corrected Loss2 Full Final-Step Visualization

Status: completed corrected visualization for existing `loss2` outputs.

Observed evidence:
- Created `2D_NS_FNO2d_recurrent/visualizations/plot_loss2_attack_full_final_panels.py`.
- Generated 12 full final-step panels under `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/`.
- Report: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/loss2_full_final_panel_report.json`.
- Dedicated note: `docs/ns2d_recurrent_loss2_full_final_visualization_20260522.md`.
- The corrected panels include clean/perturbed initial condition, final delta, clean/perturbed FNO final output, clean/perturbed solver final output, and clean/perturbed FNO-minus-solver final differences.
- GPU path was active: PyTorch CUDA available on `NVIDIA A100-SXM4-80GB`, JAX backend `gpu`, JAX device `cuda:0`.
- Runtime was `58.28` seconds; peak PyTorch allocated memory was `1.40 GiB` and peak reserved memory was `1.70 GiB`.
- Mean over three plotted samples: `raw_add` delta L2 `13.2124`; other methods delta L2 about `32.0`.
- Mean adv FNO-solver L2 over three plotted samples: `raw_add` `51.9766`, `raw_replace` `60.2301`, `steepest_add` `51.6706`, `steepest_replace` `60.2301`.

Inference:
- The earlier fast saved-array loss2 panels were inadequate for judging final FNO-vs-solver behavior because they did not rerun final model/solver outputs.
- The corrected full panels are the right visual artifact for checking whether `epsilon=32`, `alpha=1` looks too small or too large.
- `epsilon=32` is visually meaningful; `raw_add` with `alpha=1` remains too weak, while replacement methods saturate immediately.
- A paired epsilon/alpha sweep is still needed before selecting final attack settings.

Remaining work:
- Apply the full final-step visualizer to corrected sweep outputs after rerunning the patched attack.

### 2026-05-22 04:35 UTC - Patched NS2D Attack To Record Full Final States And Sample Step Trace

Status: source patch completed; no attack experiment launched.

Observed evidence:
- Modified `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- Modified `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh`.
- Dedicated note: `docs/ns2d_recurrent_attack_output_recording_patch_20260522.md`.
- Added `all_w_final_outputs` as the canonical no-gradient final FNO/solver output path.
- Added per-method full-batch final archives: `final_state_outputs.npz` and `final_state_metrics.csv`.
- Added per-method selected-sample step trace archives: `step_sample_trace.npz` and `step_sample_trace_metrics.csv`.
- Defaults record full-batch final state for every sample and per-step trace for `sample_position=0`.
- New controls include `--record-final-state-outputs`, `--record-step-sample-outputs`, `--record-step-sample-position`, `--record-step-sample-every`, and `--record-step-sample-gradients`.
- Wrapper environment variables now expose the same controls.
- Checks passed: Python `py_compile` for the attack script and `bash -n` for the wrapper.

Inference:
- Future attack runs will preserve enough information to inspect clean vs adversarial initial conditions, final deltas, FNO final output, solver final output, and FNO-minus-solver differences.
- Output size will increase. If storage pressure becomes an issue, use `RECORD_STEP_SAMPLE_EVERY=5` or `RECORD_STEP_SAMPLE_GRADIENTS=0` while keeping full final-state output enabled.

Remaining work:
- Launch the corrected attack when the user is ready.
- Generate plots/GIFs from the new `final_state_outputs.npz` and `step_sample_trace.npz` after the run finishes.

### 2026-05-22 04:45 UTC - Updated NS2D Attack Alpha/Epsilon Recommendation

Status: tuning recommendation updated from existing `loss2` evidence; no attack run launched.

Observed evidence:
- Old completed `loss2/raw_add` setting used `epsilon=32`, `alpha=1`, and `steps=100`.
- It ended with mean `delta_p=12.0754`, giving boundary ratio `0.3774`.
- Dedicated note: `docs/ns2d_recurrent_attack_alpha_epsilon_tuning_20260522.md`.

Inference:
- `epsilon=32, alpha=1` is too slow for additive methods; it would need about `265` steps to reach the L2 boundary under the observed raw-add scaling.
- To reach the boundary by about 100 steps, `alpha / epsilon` needs about `2.65x` the old ratio.
- To reach the boundary by about 50 steps, `alpha / epsilon` needs about `5.3x` the old ratio.
- To reach the boundary by about 25 steps, `alpha / epsilon` needs about `10.6x` the old ratio.
- The prior paired sweep recommendation `16:0.5 32:1 64:2` is superseded because it preserves the old too-small ratio.

Recommendation:
- Use paired values, not a Cartesian product.
- Recommended calibration sweep: `EPSILON_ALPHA_PAIRS="8:1.25 16:2.5 32:5 32:10"`.
- Shorter first check: `EPSILON_ALPHA_PAIRS="16:2.5 32:5"`.

Remaining work:
- Run the corrected attack calibration and inspect `delta_threshold_crossings.csv`, `final_state_outputs.npz`, and `step_sample_trace.npz`.

### 2026-05-22 04:40 UTC - Regenerated Loss2 Full Final Panels With Independent Difference Ranges

Status: regenerated existing visualization outputs; no attack rerun.

Observed evidence:
- Updated `2D_NS_FNO2d_recurrent/visualizations/plot_loss2_attack_full_final_panels.py`.
- Regenerated 12 PNGs under `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/`.
- Updated report: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/loss2_full_final_panel_report.json`.
- Report now has `plot_version=independent_difference_ranges_with_loss_titles`.
- Difference panels now use independent color ranges.
- Figure title now includes per-sample clean loss, adversarial loss, loss difference, and loss ratio.
- Difference/change panel titles now include L2 values.
- Example `loss2/steepest_add`, sample position `0`, dataset index `0`: clean loss `34.82`, adversarial loss `47.04`, difference `+12.22`, ratio `1.351`.
- Visualization runtime was `62.95` seconds; post-run `nvidia-smi` showed `0MiB / 81920MiB` and no GPU process.

Inference:
- The revised figures are more useful for checking whether the attack actually increases final FNO-vs-solver loss.
- Independent difference ranges avoid hiding small FNO-vs-solver error fields behind the larger final-change ranges.

Dataset/source clarification:
- This completed attack used the test dataset path from the manifest and `indices=0,1,2,3,4,5,6,7,8,9`; it did not use training-set initial conditions.

### 2026-05-22 04:50 UTC - Estimated Boundary Steps For Recommended Alpha/Epsilon Pairs

Status: analysis from existing `loss2` threshold logs; no experiment run launched.

Observed evidence:
- Old `loss2/raw_add` with `epsilon=32`, `alpha=1`, `steps=100` reached 25% boundary at step 62 and final boundary ratio `0.3774`, implying 100% around step `265` under linear scaling.
- Old `loss2/steepest_add` reached 25% at step 9, 50% at step 20, 75% at step 31, and 100% at step 70.
- Old replacement methods were effectively at the boundary after the first update; exact 100% threshold appeared by step 10 due floating-point thresholding.
- Updated note: `docs/ns2d_recurrent_attack_alpha_epsilon_tuning_20260522.md`.

Inference:
- `8:1.25`, `16:2.5`, and `32:5` are all `5x` the old `alpha/epsilon` ratio. Raw-add should hit the boundary around step `53`, leaving about `47` projected-on-boundary steps in a 100-step attack.
- `32:10` is `10x` the old ratio. Raw-add should hit the boundary around step `27`, leaving about `73` projected-on-boundary steps.
- For steepest-add, observed-scaled crossing is about step `14` for the `5x` pairs and about step `7` for `32:10`; ideal epsilon/alpha estimates are even earlier.
- Replacement methods ignore `alpha` in their replacement step and should be treated as immediate-boundary tests.

### 2026-05-22 04:55 UTC - Tightened Alpha/Epsilon Recommendation For 50-Step Boundary Target

Status: recommendation updated from existing loss2 evidence; no experiment run launched.

Observed evidence:
- Old `loss2/raw_add` with `epsilon=32`, `alpha=1` estimates boundary crossing around step `265`.
- Earlier `5x` recommendation estimates raw-add boundary crossing around step `53`, which is borderline for a strict 50-step target.
- Updated note: `docs/ns2d_recurrent_attack_alpha_epsilon_tuning_20260522.md`.

Inference:
- Because `loss3` may optimize more slowly than `loss2`, the main run should target boundary crossing clearly before 50 steps in the loss2 calibration scale.
- Main recommended ratio is now `10x` the old `alpha/epsilon` ratio.
- Recommended main sweep: `EPSILON_ALPHA_PAIRS="8:2.5 16:5 32:10"`.
- Optional aggressive backup: add `32:15`, estimated around step `18` for raw-add under old loss2 scaling.

### 2026-05-22 05:39 UTC - Running Status Check For Aggressive NS2D Attack Launch

Status: inspected active background run; no process changes made.

Observed evidence:
- Active launch root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps_aggressive_20260522`.
- Active log: `nohup_launch_20260522_044859_UTC.log`.
- Active Python process still has `--loss-types loss1 --mode-spec all_w`.
- Run elapsed time for Python PID `302955` was about `50m36s` at inspection.
- GPU was active at `99%` utilization with about `46953 MiB / 81920 MiB` used.
- Completed files exist for `batch_0000_0009/eps8_alpha2p5/loss1/raw_add/` and `batch_0000_0009/eps8_alpha2p5/loss1/raw_replace/`, including `summary.json`, `final_state_outputs.npz`, and `step_sample_trace.npz`.

Inference:
- `loss1` is still running and has not yet advanced to `loss2`.
- The first epsilon/alpha pair has completed at least `raw_add` and `raw_replace`; subsequent `loss1` methods/pairs remain in progress.

### 2026-05-22 05:44 UTC - Running Status Check For Aggressive NS2D Attack Launch

Status: inspected active background run; no process changes made.

Observed evidence:
- Active launch root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps_aggressive_20260522`.
- Active Python process `302955` was still running `--loss-types loss1 --mode-spec all_w` after about `55m06s` elapsed.
- GPU was active at `99%` utilization with about `46953 MiB / 81920 MiB` used.
- Completed `loss1` summaries existed for `eps8_alpha2p5/raw_add` and `eps8_alpha2p5/raw_replace`.
- `eps8_alpha2p5/raw_add` runtime was `1131.10s`; final `loss1_mean=304.5169`; final `delta_p_mean=8.0`; final boundary ratio about `1.0`.
- `eps8_alpha2p5/raw_replace` runtime was `1124.85s`; final `loss1_mean=304.2200`; final `delta_p_mean=8.0`; final boundary ratio about `1.0`.

Inference:
- `loss1` has not completed yet and has not advanced to `loss2`.
- Current progress is `2 / 16` loss1 subruns completed for the four epsilon/alpha pairs and four methods.
- The active subrun is likely the next `loss1` method for `eps8_alpha2p5`, because no later epsilon/alpha directories have summaries yet.

### 2026-05-22 06:00 UTC - Stopped Wrong-Order Aggressive NS2D Attack Launch

Status: stopped the active background attack run before relaunch.

Observed evidence:
- Previous launch root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps_aggressive_20260522`.
- The active process group was terminated before this record; a follow-up `pgrep -af 'attack_ns2d_recurrent_core4|run_ns2d_recurrent_core4_attack|full_adw_b10|pair_outer'` found no real attack process, only the `pgrep` command itself.
- Follow-up `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits` reported `NVIDIA A100-SXM4-80GB, 0, 81920, 0`.
- Partial completed outputs from the stopped run remain under the old root, including `eps8_alpha2p5/loss1/raw_add` and `eps8_alpha2p5/loss1/raw_replace`.

Inference:
- The stopped run used the wrong global ordering for the user's desired comparison: it ran loss/method work before completing a baseline epsilon/alpha pair across losses.
- The relaunch command should use epsilon/alpha pair as the outer loop, with the baseline pair first, then run `loss1`, `loss2`, and the no-`A`/with-`A` `loss3` modes inside each pair.

### 2026-05-22 05:54 UTC - Pair-Outer Baseline-First NS2D Attack Launch Check

Status: inspected active background run; no process changes made.

Observed evidence:
- Launcher PID `308447` is running `/tmp/run_ns2d_pair_outer_attack.sh`.
- Active Python PID `308458` is running `attack_ns2d_recurrent_core4.py` with `--out-root 2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Active command uses `--indices 0,1,2,3,4,5,6,7,8,9`, `--attack-batch-size 10`, `--steps 100`, `--epsilon-alpha-pairs 32:10`, `--loss-types loss1`, and `--mode-spec all_w`.
- `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits` reported `NVIDIA A100-SXM4-80GB, 46851, 81920, 98`.
- Run log exists at `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/nohup_pair_outer_20260522_055312_UTC.log` and shows the first block: `pair=32:10 loss=loss1 mode=all_w`.

Inference:
- The corrected pair-outer launch has started successfully and is currently running the baseline pair first.
- No completed summary files were visible yet at this early inspection point, so step/loss progress should be read again after the first method finishes.

### 2026-05-22 07:02 UTC - Baseline-First NS2D Attack Loss1 Progress Check

Status: inspected active background run; no process changes made.

Observed evidence:
- Active Python process `308458` is still running `attack_ns2d_recurrent_core4.py` with `--loss-types loss1`, `--mode-spec all_w`, `--epsilon-alpha-pairs 32:10`, and `--attack-batch-size 10`.
- GPU query reported `NVIDIA A100-SXM4-80GB, 46953, 81920, 98`, so the attack is still actively using the GPU.
- Under `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1/`, completed `summary.json` files exist for `raw_add`, `raw_replace`, and `steepest_add`.
- No `steepest_replace/summary.json` was visible at inspection time, and the active command has not advanced to `loss2`.
- Observed method runtimes from completed summaries: `raw_add` about `18.95 min`, `raw_replace` about `18.84 min`, and `steepest_add` about `18.84 min`.
- Observed final true-loss means from completed summaries: clean baseline `68.49199676513672`; `raw_add` adversarial `93.04156494140625`; `raw_replace` adversarial `98.32636260986328`; `steepest_add` adversarial `158.28329467773438`.

Inference:
- The baseline `loss1` group is not complete yet: progress is `3 / 4` methods completed.
- The current active method is inferred to be `steepest_replace`, because it is the only configured `loss1` method without a completed summary.
- Based on the observed per-method runtime around 18.8-19.0 minutes, the remaining `loss1` method should finish soon if runtime stays similar.

### 2026-05-22 07:42 UTC - Baseline-First NS2D Attack Progress Check

Status: inspected active background run; no process changes made.

Observed evidence:
- Current launcher root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522`.
- Active Python process `313364` is running `attack_ns2d_recurrent_core4.py` with `--out-root .../eps32_alpha10`, `--loss-types loss3`, `--mode-spec all_w`, `--epsilon-alpha-pairs 32:10`, `--attack-batch-size 10`, and `--steps 100`.
- GPU query reported `NVIDIA A100-SXM4-80GB, 47235, 81920, 99`.
- The run log shows `pair=32:10 loss=loss1 mode=all_w` finished at `2026-05-22 07:08:56 UTC`, `pair=32:10 loss=loss2 mode=all_a_target_w` finished at `2026-05-22 07:41:33 UTC`, and `pair=32:10 loss=loss3 mode=all_w` started at `2026-05-22 07:41:33 UTC`.
- Completed method summaries under `eps32_alpha10`: `loss1` has 4 method summaries; `loss2` has 4 method summaries; `loss3` has 0 method summaries at inspection time.
- Observed method runtimes: `loss1` methods each about `18.84-18.95 min`; `loss2` methods each about `8.06-8.11 min`.

Inference:
- Baseline pair `32:10` has completed `loss1` and `loss2` fully and is now in the first `loss3` block (`all_w`).
- In terms of configured method summaries, the full sweep has completed `8 / 112` summaries, with the 9th in progress. Within the baseline pair, it has completed `8 / 28` summaries.
- Runtime estimates for the remaining sweep remain uncertain because `loss3` timings have not produced a completed method summary yet.

### 2026-05-22 07:50 UTC - Loss1 Saved Final-State Visualization For Baseline Pair

Status: generated CPU-only visualization from completed `loss1/all_w` saved final-state outputs; no model/solver rerun.

Observed evidence:
- Source loss root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`.
- Visualization script: `2D_NS_FNO2d_recurrent/visualizations/plot_attack_saved_final_state_panels.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522`.
- Report file: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522/loss1_saved_final_state_panel_report.json`.
- Generated `40` PNG panels, covering four methods and ten sample positions.
- The plotting command ran with `CUDA_VISIBLE_DEVICES=''` and the script imports NumPy/matplotlib only.
- Follow-up GPU query reported `NVIDIA A100-SXM4-80GB, 47235, 81920, 99`, indicating the long-running attack remained active.
- Result Markdown: `docs/ns2d_recurrent_loss1_saved_final_visualization_20260522.md`.

Observed summary metrics from the report across ten samples:
- `raw_add`: clean true loss mean `68.491997`, adversarial true loss mean `93.041568`, mean increase `24.549571`, mean ratio `1.422519`, mean delta L2 `31.999998`.
- `raw_replace`: clean true loss mean `68.491997`, adversarial true loss mean `98.326353`, mean increase `29.834356`, mean ratio `1.586603`, mean delta L2 `32.000000`.
- `steepest_add`: clean true loss mean `68.491997`, adversarial true loss mean `158.283305`, mean increase `89.791308`, mean ratio `2.568838`, mean delta L2 `31.929921`.
- `steepest_replace`: clean true loss mean `68.491997`, adversarial true loss mean `98.326353`, mean increase `29.834356`, mean ratio `1.586603`, mean delta L2 `32.000000`.

Inference:
- In this completed baseline `loss1` block, `steepest_add` shows the largest mean true-loss increase among the four methods.
- The visualizations are diagnostic snapshots of saved final states and should be compared later with the full `loss2`/`loss3` sweep outputs.

### 2026-05-22 07:56 UTC - Loss1 Loss Curves With Standard Deviation Shading

Status: generated CPU-only loss curve plots for completed baseline `loss1/all_w`; no model/solver rerun.

Observed evidence:
- Source loss root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`.
- Source files: each method's `per_step_metrics.csv`.
- Plotting script: `2D_NS_FNO2d_recurrent/visualizations/plot_attack_loss_curves.py`.
- Step-curve PNG: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_overview_with_std.png`.
- Wall-time PNG: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_wall_time_with_std.png`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_summary_with_std.csv`.
- Result Markdown: `docs/ns2d_recurrent_loss1_loss_curves_20260522.md`.
- The plotting command used `CUDA_VISIBLE_DEVICES=''`; the script imports NumPy/matplotlib only.
- Follow-up GPU query reported `NVIDIA A100-SXM4-80GB, 47235, 81920, 99`, indicating the long-running attack remained active.

Observed summary from the CSV:
- `raw_add`: boundary step `1`, final objective `445.810025 +/- 30.303622`, true loss `68.491732 -> 93.041568`, true increase `24.549836`, final boundary `1.000000`, runtime `18.601783 min`.
- `raw_replace`: boundary step `61`, final objective `401.017053 +/- 50.599833`, true loss `68.491732 -> 98.326353`, true increase `29.834621`, final boundary `1.000000`, runtime `18.519913 min`.
- `steepest_add`: boundary step `12`, final objective `519.401230 +/- 25.633891`, true loss `68.491732 -> 158.283305`, true increase `89.791573`, final boundary `0.997810`, runtime `18.525295 min`.
- `steepest_replace`: boundary step `61`, final objective `401.017053 +/- 50.599833`, true loss `68.491732 -> 98.326353`, true increase `29.834621`, final boundary `1.000000`, runtime `18.545024 min`.

Inference:
- In the completed `loss1` baseline block, `steepest_add` has the strongest true-loss growth by 100 steps.
- The standard deviation shading shows substantial sample-to-sample variability, so the final-state panels remain important for interpreting individual cases.

### 2026-05-22 08:05 UTC - Loss1 Steepest Add Versus Replace Interpretation

Status: inspected completed baseline `loss1/all_w` code and metrics; no process changes made.

Observed evidence:
- Source loss root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`.
- Source code inspected: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- For `p=2`, `steepest_direction` returns the normalized gradient; therefore `raw_replace` and `steepest_replace` are the same update direction in this run.
- Replacement uses `delta = epsilon * direction`; additive uses `delta = delta + alpha * direction` followed by projection.
- Observed k=1 metrics: `raw_replace` and `steepest_replace` both have loss1 `449.930487`, true loss `92.175734`, boundary `0.999999982`.
- Observed k=100 metrics: `steepest_add` has loss1 `519.401230`, true loss `158.283305`, boundary `0.997810042`; `steepest_replace` has loss1 `401.017053`, true loss `98.326353`, boundary `0.999999982`.
- Result Markdown: `docs/ns2d_recurrent_loss1_steepest_add_vs_replace_note_20260522.md`.

Inference:
- The fact that `steepest_add` beats `steepest_replace` in this completed block is not by itself evidence of a code bug.
- The generalized-power intuition assumes a fixed quadratic/local-linear problem; this attack uses a nonlinear recurrent FNO at finite `epsilon=32`, and the true loss also includes solver output.
- The earlier exact 100% boundary step for replacement showing as step 61 is a floating-point threshold artifact; replacement is effectively on the boundary at step 1.
- A smaller-epsilon or fixed-linearized/JVP-only diagnostic would be the cleaner test of the generalized-power expectation.

### 2026-05-22 08:12 UTC - Recorded 1D Burgers Versus 2D NS Optimizer Difference

Status: documented the interpretation that the best attack optimizer may differ between 1D Burgers and 2D NS recurrent-FNO experiments.

Observed evidence:
- Current 2D source loss root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`.
- Current 2D settings: `epsilon=32`, `alpha=10`, `p=q=2`, `loss1`, `mode=all_w`, `attack_batch_size=10`, `steps=100`.
- Current 2D result: `steepest_add` has final loss1 mean `519.401230`, final true loss mean `158.283305`, and true-loss increase `89.791573`, larger than `steepest_replace` final loss1 mean `401.017053`, final true loss mean `98.326353`, and true-loss increase `29.834621`.
- Prior 1D Burgers observation is user-reported from earlier project experiments: `steepest_replace` was usually strongest in that setting.
- New result Markdown: `docs/ns2d_recurrent_1d_vs_2d_attack_optimizer_observation_20260522.md`.

Inference:
- The current result suggests attack optimizer ranking can be PDE- and regime-dependent.
- The 2D NS recurrent setting is nonlinear and finite-epsilon; `steepest_add` may benefit from accumulating/rotating directions on the boundary, whereas replacement discards path history.
- This should be treated as a useful observation, not yet a final universal conclusion; smaller-epsilon, loss2/loss3, sample-wise, and fixed-linearized diagnostics remain useful follow-ups.

### 2026-05-22 20:32 UTC - Pair-Outer NS2D Attack Progress Check

Status: inspected active background run; no process changes made.

Observed evidence:
- Active launcher root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522`.
- Active Python PID `328653` is running `attack_ns2d_recurrent_core4.py` with `--out-root .../eps8_alpha2p5`, `--epsilon-alpha-pairs 8:2.5`, `--loss-types loss3`, `--mode-spec all_w`, `--attack-batch-size 10`, and `--steps 100`.
- Current time from `date -u`: `2026-05-22 20:32:53 UTC`.
- GPU query reported `NVIDIA A100-SXM4-80GB, 47197, 81920, 99`.
- Log evidence: `pair=32:10` completed all seven blocks and finished at `2026-05-22 18:02:00 UTC`; `pair=8:2.5 loss=loss1 mode=all_w` finished at `2026-05-22 19:16:17 UTC`; `pair=8:2.5 loss=loss2 mode=all_a_target_w` finished at `2026-05-22 19:49:00 UTC`; `pair=8:2.5 loss=loss3 mode=all_w` started at `2026-05-22 19:49:00 UTC`.
- Completed method summary count: `37 / 112` total configured method summaries.
- Completed by pair/loss: `eps32_alpha10` has `loss1=4`, `loss2=4`, `loss3=20`; `eps8_alpha2p5` has `loss1=4`, `loss2=4`, `loss3=1`.
- In the active `eps8_alpha2p5/loss3/all_w` block, only `raw_add/summary.json` exists so far; `raw_add` runtime was about `31.19 min`, with final adversarial true loss `98.74174499511719` and clean true loss `68.49199676513672`.

Inference:
- The run has completed the full baseline pair `32:10` and is partway through the second pair `8:2.5`.
- The active method is inferred to be the next method after `raw_add` in `eps8_alpha2p5/loss3/all_w`, likely `raw_replace`, because only `raw_add` has a summary in that block and the active Python command is still in `loss3/all_w`.
- By method-summary count, progress is about `33%` complete (`37 / 112`), with the 38th method currently running.
- Runtime projection remains approximate, but if later pairs behave similarly to `32:10`, the full sweep may take on the order of two days rather than one day.

### 2026-05-22 20:38 UTC - eps32 alpha10 Baseline Overview Plots

Status: generated CPU-only overview plots for the completed baseline pair `epsilon=32`, `alpha=10`; no model/solver rerun.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Plotting script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_baseline_overview.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522`.
- Generated plots: loss/objective curves with std shading, delta norm/boundary curves with std shading, sample-0 angle diagnostics, sample-0 final perturbation grid, and final metric heatmap.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_final_metric_summary.csv`.
- Result Markdown: `docs/ns2d_recurrent_eps32_alpha10_baseline_overview_20260522.md`.
- PNG size check showed generated images with dimensions: loss curves `2400x3255`, delta curves `2400x3255`, angle curves `2400x3255`, final delta grid `2400x3255`, final heatmap `2700x900`.
- Follow-up GPU query reported `NVIDIA A100-SXM4-80GB, 47197, 81920, 99`, indicating the long-running attack remained active.

Observed summary:
- `steepest_add` was best by final true-loss increase in all completed baseline blocks.
- Largest true-loss increases: `loss3/all_w` with `steepest_add` `236.100975`; `loss3/d1_5_w6_9_target_w` with `steepest_add` `220.735313`; `loss3/all_d_target_w` with `steepest_add` `147.921621`; `loss3/w1_5_d6_9_target_w` with `steepest_add` `139.366045`; `loss3/a1_5_d6_9_target_w` with `steepest_add` `108.555902`.

Inference:
- For the completed `epsilon=32`, `alpha=10` pair, `steepest_add` dominates by final true-loss increase across the observed loss/mode blocks.
- Angle curves are sample-0 diagnostics only because the attack run records per-step trace for `step_sample_position=0`; batch loss and delta curves use mean/std over the 10 attacked samples.

### 2026-05-22 20:46 UTC - eps32 alpha10 Baseline Overview v2 Plot Corrections

Status: regenerated CPU-only baseline overview plots with corrected true-loss y-axis and boundary threshold markers; no model/solver rerun.

Observed evidence:
- Updated output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2`.
- Updated loss curve PNG: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_loss_curves_with_std_shared_true_y_thresholds.png`.
- Updated delta curve PNG: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_delta_norm_boundary_with_std_thresholds.png`.
- The true-loss column now uses a shared y-axis range across all rows.
- Objective, true-loss, delta-norm, and boundary-ratio curves now mark first crossings of `25%`, `50%`, `75%`, and `100%` boundary ratio.
- PNG size check showed generated images with valid dimensions: loss curves `2400x3255`, delta curves `2400x3255`, angle curves `2400x3255`, final delta grid `2400x3255`, final heatmap `2700x900`.
- Follow-up GPU query reported `NVIDIA A100-SXM4-80GB, 47197, 81920, 99`, indicating the long-running attack remained active.

Inference:
- The v2 plots are better for comparing true loss across modes because the second column has a unified scale.
- `angle(delta, direction)` should be read as the sample-0 angle between current perturbation and current gradient-derived update direction; it is not a batch-mean angle.

### 2026-05-22 20:50 UTC - eps32 alpha10 Delta Curve raw_add Visibility Check

Status: inspected saved per-step metrics to explain why `raw_add` is not visually distinct in several delta norm/boundary plots; no process changes made.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- `raw_add` per-step CSV files exist for all completed baseline blocks.
- In `loss1/all_w` and all `loss3` modes, `raw_add` reaches about 100% boundary at step 1 and its boundary curve differs from `raw_replace`/`steepest_replace` by only about `3e-8` to `5e-8` in max absolute difference.
- In `loss2/all_a_target_w`, `raw_add` reaches about 100% boundary much later, around step 68, and its boundary curve is visibly different from replace curves.

Inference:
- `raw_add`/PGD is present in the delta norm plots, but in most blocks it is visually hidden because it overlaps almost exactly with replacement curves at `||delta||/epsilon ~= 1` from step 1 onward.
- The visible exception in `loss2` happens because `raw_add` does not immediately hit the boundary there.
- Future plots should use distinct line styles, z-order, alpha, or separate method panels to make overlapping PGD/replacement curves visible.

### 2026-05-22 20:55 UTC - eps32 alpha10 PGD Norm Overlap Versus Loss Difference Check

Status: inspected saved per-step metrics and final perturbation arrays to answer whether `raw_add`/PGD was missing or incorrectly recorded in delta norm plots; no process changes made.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Source code inspected: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` lines 132-159, 467-484, and 1223-1300.
- Formula from code: additive methods use `delta <- Proj(delta + alpha * direction)`; replacement methods use `delta <- Proj(epsilon * direction)`.
- For `p=2`, `steepest_direction(grad)` is `normalize_to_p_ball(grad, 2.0)`, so `raw_replace` and `steepest_replace` share the same direction in this run.
- Numeric check comparing `raw_add` versus `raw_replace`:
  - `loss3/all_w`: max absolute boundary-ratio difference `4.172e-08`, final true loss `155.712` versus `106.323`, final-delta cosine mean `0.078`, min `-0.275`.
  - `loss1/all_w`: max absolute boundary-ratio difference `4.172e-08`, final true loss `93.042` versus `98.326`, final objective `445.810` versus `401.017`, final-delta cosine mean `0.142`, min `-0.605`.
  - `loss3/d1_5_w6_9_target_w`: max absolute boundary-ratio difference `4.768e-08`, final true loss `171.509` versus `137.871`, final-delta cosine mean `0.300`, min `-0.130`.
  - `loss2/all_a_target_w` is the visible exception: max absolute boundary-ratio difference `9.579e-01`, because `raw_add` reaches the boundary much later than replacement.

Inference:
- The saved data support the explanation that PGD/raw_add is present in the delta norm plots but often hidden by overlapping boundary-ratio curves.
- Overlap of `||delta||/epsilon` only means the perturbations have the same norm. It does not mean the perturbation vectors are the same.
- Final perturbation cosine similarities between `raw_add` and `raw_replace` are often low, so the losses can differ substantially despite identical norm curves.

### 2026-05-22 21:00 UTC - eps32 alpha10 Perturbed Final Output Heatmaps

Status: generated CPU-only sample-0 final-output heatmaps from saved `final_state_outputs.npz`; no model/solver rerun.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2`.
- Generated PNGs: `eps32_alpha10_baseline_adv_model_final_sample0_grid.png`, `eps32_alpha10_baseline_adv_solver_final_sample0_grid.png`, `eps32_alpha10_baseline_adv_model_minus_solver_sample0_grid.png`, and `eps32_alpha10_baseline_abs_adv_model_minus_solver_sample0_grid.png`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_adv_final_output_diff_sample0_summary.csv`.
- PNG size check showed all four final-output images have valid dimensions `2520x3360`.
- Follow-up GPU query reported `NVIDIA A100-SXM4-80GB, 47197, 81920, 99`, indicating the long-running attack remained active.

Observed sample-0 top true-loss increases:
- `loss3/all_w` + `steepest_add`: adversarial true loss `263.904083`, increase `229.079956`, ratio `7.578197`.
- `loss3/d1_5_w6_9_target_w` + `steepest_add`: adversarial true loss `257.590118`, increase `222.765991`, ratio `7.396887`.
- `loss3/w1_5_d6_9_target_w` + `steepest_add`: adversarial true loss `145.218689`, increase `110.394562`, ratio `4.170060`.
- `loss3/all_d_target_w` + `steepest_add`: adversarial true loss `145.115677`, increase `110.291550`, ratio `4.167102`.
- `loss1/all_w` + `steepest_add`: adversarial true loss `133.663818`, increase `98.839691`, ratio `3.838253`.

Inference:
- The new heatmaps visualize the perturbed final FNO output, perturbed final solver output, and signed/absolute model-solver final differences for the same sample-0 perturbations as the final-delta grid.
- These are sample-0 diagnostics and should be interpreted alongside the batch mean/std curves.

### 2026-05-22 21:08 UTC - eps32 alpha10 Final Delta Fourier Analysis

Status: generated CPU-only Fourier analysis for completed baseline final perturbations; no model/solver rerun.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_delta_fft_analysis.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522`.
- Generated plots: `eps32_alpha10_final_delta_fft_log_magnitude_sample0_grid.png`, `eps32_alpha10_final_delta_radial_fft_profiles.png`, and `eps32_alpha10_final_delta_fft_band_metrics_heatmaps.png`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_fft_metrics_summary.csv`.
- Result Markdown: `docs/ns2d_recurrent_eps32_alpha10_delta_fft_analysis_20260522.md`.
- PNG size check showed valid images: FFT grid `2520x3360`, radial profiles `1500x2835`, band heatmap `3150x900`.
- Follow-up GPU query reported `NVIDIA A100-SXM4-80GB, 47197, 81920, 99`, indicating the long-running attack remained active.

Observed aggregate metrics:
- `loss1`: mean low fraction `0.999710`, mid fraction `0.000290`, high fraction `8.48e-08`, spectral centroid `0.013960`.
- `loss2`: mean low fraction `0.999977`, mid fraction `0.000021`, high fraction `2.12e-06`, spectral centroid `0.011957`.
- `loss3`: mean low fraction `0.998162`, mid fraction `0.001837`, high fraction `1.01e-06`, spectral centroid `0.019692`.

Inference:
- The user's observation is mostly supported: `loss2` is especially smooth/low-frequency, `loss1` is also low-frequency, and `loss3` shifts more energy into mid-frequency/curvier spatial structure.
- The strict high-frequency band remains tiny for all groups, so the visual "higher frequency" in `loss3` is primarily a shift toward low-mid/mid radial frequencies rather than a large Nyquist-scale high-frequency component.

### 2026-05-22 21:18 UTC - eps32 alpha10 LP Steepest PGD Final-Delta Observation

Status: recorded follow-up interpretation of the final-delta and FFT plots; no experiment rerun.

Observed evidence:
- Source final-delta plot set: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2`.
- Source FFT plot set: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522`.
- Result Markdown updated: `docs/ns2d_recurrent_eps32_alpha10_delta_fft_analysis_20260522.md`.

Inference:
- `loss3` + LP steepest PGD (`steepest_add`) produces some of the most visually distinctive final perturbation structures among the completed `epsilon=32`, `alpha=10` baseline cases, especially in `loss3/all_w` and related W-heavy settings.
- The visible structure and FFT metrics suggest that LP steepest PGD is selecting a materially different spatial perturbation direction, not simply making the same perturbation larger.
- This supports the current working conclusion that 2D recurrent NS attack behavior can differ from the earlier 1D Burgers pattern, where replacement/GPI-style updates looked more dominant.

### 2026-05-22 21:17 UTC - eps32 alpha10 Output FFT Dealiasing Analysis

Status: generated CPU-only FFT/dealiasing analysis for saved final delta, clean/adv FNO final outputs, clean/adv solver final outputs, and model-solver final differences; no model/solver rerun.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_output_fft_dealias_analysis.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_output_fft_dealias_analysis_20260522/eps32_alpha10_output_fft_dealias_metrics_summary.csv`.
- Result Markdown: `docs/ns2d_recurrent_eps32_alpha10_output_fft_dealias_analysis_20260522.md`.
- Training-data and dictionary generation both set `dealiasing_fraction=2 / 3`; active attack code rolls out solver frames for `d`/`w` modes.

Observed metrics:
- Final-delta outside-`2/3` square fractions: `loss1` `1.361e-10`, `loss2` `1.183e-06`, `loss3` `3.490e-10`.
- Final-delta outside-square cross-arm share: `loss1` `0.852`, `loss2` `0.836`, `loss3` `0.835`.
- Adv FNO final outside-square fractions are around `1.1e-06` to `1.5e-06`.
- Adv solver final outside-square fractions are around `1.8e-06` to `2.4e-06`.
- Adv model-solver difference outside-square fractions are much larger, around `7.2e-05` to `1.27e-04`, with substantially higher spectral centroids.

Inference:
- The `2/3` dealiasing fingerprint is visible in solver outputs, FNO outputs, and final perturbations.
- The trained FNO appears to mirror the solver/training-data spectral cutoff, though not as an exact hard projector.
- `loss1/all_w` can inherit the cutoff because solver-generated frames 2-10 are used as recurrent model input in `w` mode, even though the loss1 target itself is `F(x)`.
- `loss2/all_a_target_w` is smoother and lacks the same obvious cutoff-box boundary because its active objective uses dictionary/fixed-target structure rather than differentiating through a perturbed solver target.
- The model-solver difference is where mid/high-frequency mismatch is most visible.

### 2026-05-22 21:22 UTC - eps32 alpha10 Method-Grouped Loss and Spectral Curves

Status: generated CPU-only curve plots grouped by optimizer/method; no model/solver rerun.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_method_grouped_curves.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522`.
- Generated loss-curve PNGs: `eps32_alpha10_loss_curves_by_method_all_blocks_linear.png` and `eps32_alpha10_loss_curves_by_method_all_blocks_logy.png`.
- Generated spectral PNGs: `eps32_alpha10_final-delta_radial_fft_by_method_all_blocks.png`, `eps32_alpha10_adv-model-final_radial_fft_by_method_all_blocks.png`, `eps32_alpha10_adv-solver-final_radial_fft_by_method_all_blocks.png`, and `eps32_alpha10_adv-model-minus-solver_radial_fft_by_method_all_blocks.png`.
- Result Markdown: `docs/ns2d_recurrent_eps32_alpha10_method_grouped_curves_20260522.md`.

Inference:
- The method-grouped plots complement the block-grouped plots: block-grouped views compare optimizers within one loss/mode block, while method-grouped views compare all loss/mode blocks for one optimizer.
- Spectral plots mark both `rho=sqrt(2)/3` and `rho=2/3`, clarifying why a square 2/3 cutoff appears as a radial drop-off band rather than a single point.
- These plots support the existing interpretation that loss2 perturbations are smoothest/lowest-frequency, loss3 perturbations carry more mid-frequency structure, and model/solver outputs both show a 2/3-rule spectral cutoff.

### 2026-05-22 21:30 UTC - eps32 alpha10 FFT Heatmaps Without Cutoff Overlays

Status: generated CPU-only FFT heatmaps with no drawn cutoff line/box for visual inspection of the dealiasing boundary; no model/solver rerun.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_output_fft_heatmaps_no_cutoff.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522`.
- Generated no-cutoff fields: `x_clean`, `x_adv`, `final_delta`, `clean_model_final`, `clean_solver_final`, `clean_model_minus_solver`, `adv_model_final`, `adv_solver_final`, `adv_model_minus_solver`, `model_final_change`, and `solver_final_change`.
- Result Markdown: `docs/ns2d_recurrent_eps32_alpha10_fft_heatmaps_no_cutoff_20260522.md`.

Inference:
- These plots complement the previous overlayed FFT/dealiasing plots by removing all visual guide lines. If the spectral cutoff remains visible by eye, the boundary is a feature of the saved fields rather than an artifact of the overlay.

### 2026-05-22 21:36 UTC - Corrected Visual Interpretation of 2/3 Cutoff FFT Heatmaps

Status: updated records after visual inspection corrected the earlier interpretation; no code execution beyond Markdown updates.

Observed evidence:
- No-cutoff FFT heatmap directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_no_cutoff_20260522`.
- Updated notes: `docs/ns2d_recurrent_eps32_alpha10_fft_heatmaps_no_cutoff_20260522.md` and `docs/ns2d_recurrent_eps32_alpha10_output_fft_dealias_analysis_20260522.md`.

Correction:
- The clean/adversarial FNO model-output FFT heatmaps do not show a visually clear `2/3` cutoff box in the same strong way as solver outputs or solver-change outputs.
- The earlier statement that the FNO visibly learned the solver cutoff should be weakened to: model outputs have low outside-cutoff energy, but the sharp visual cutoff box is not clearly visible without overlays.
- Strong visual cutoff evidence is present in solver-related fields and in final delta fields whose attack gradients pass through differentiable solver/dealiasing paths.

Inference:
- `loss1/all_w` can show the solver cutoff fingerprint because solver-generated frames 2-10 are part of the recurrent FNO input path, even though the loss1 target is `F(x)`.
- `loss3` can show the fingerprint because target frame 20 is kept in `w` mode and differentiates through `G(x+delta)`.
- `loss2/all_a_target_w` lacks the same differentiable perturbed-solver path, helping explain why its delta lacks the clear cutoff-box structure.
- The clean initial condition appears lower-band-limited, consistent with an upsampled lower-resolution source; `x_adv = x_clean + delta` can show both the original low-frequency support and the solver-gradient-shaped cutoff structure.

### 2026-05-22 21:41 UTC - eps32 alpha10 Batch-Mean FFT Heatmaps Without Cutoff Overlays

Status: generated CPU-only 10-sample batch-mean FFT heatmaps with no drawn cutoff guide; no model/solver rerun.

Observed evidence:
- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_output_fft_heatmaps_no_cutoff_batchmean.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522`.
- Result Markdown: `docs/ns2d_recurrent_eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522.md`.
- Generated fields: `x_clean`, `x_adv`, `final_delta`, `clean_model_final`, `clean_solver_final`, `clean_model_minus_solver`, `adv_model_final`, `adv_solver_final`, `adv_model_minus_solver`, `model_final_change`, and `solver_final_change`.

Observed method:
- For each saved field, the script computes FFT per sample, takes the magnitude, averages magnitudes over the 10 saved samples, and then plots `log10(mean |FFT|)`.
- It intentionally does not average fields before FFT, avoiding phase cancellation.

Inference:
- These batch-mean no-cutoff heatmaps are better than sample-0 plots for judging whether the apparent spectral support/cutoff/directional structure is robust across the attack batch.

### 2026-05-22 21:50 UTC - eps32 alpha10 Visual/FFT Conclusions and Image-Only Bundle

Status: consolidated recent `epsilon=32`, `alpha=10` visual/FFT conclusions into a Markdown note and collected generated visual outputs into an image-only package; no experiment/model/solver rerun.

Observed evidence:
- Summary Markdown: `docs/ns2d_recurrent_eps32_alpha10_visual_fft_conclusions_20260522.md`.
- Bundle root: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522`.
- Image-only folder: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/images_only`.
- Records folder: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/records`.
- Manifest CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/records/bundle_manifest.csv`.
- Manifest JSON: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/records/bundle_manifest.json`.
- Source visualization directories: `11` directories.
- Copied image files: `134`.
- Copied record files: `33`.

File organization:
- `images_only/` contains only image extensions and intentionally excludes CSV/JSON/Markdown/TXT records.
- `records/` preserves CSV/JSON/JSONL/TXT/Markdown records, including the new consolidated conclusion note.
- Original source visualization folders were left intact to avoid deleting existing evidence.

Inference:
- This package is intended as a clean visual browsing folder plus a separate record archive for reproducibility.

### 2026-05-22 21:56 UTC - Corrected Model/Solver Cutoff and Loss2 Final-Delta Curve Interpretation

Status: inspected existing final-delta FFT metrics and radial profile behavior to clarify visual interpretation; no model/solver rerun.

Observed evidence:
- Method-grouped curves: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_method_grouped_curves_20260522/eps32_alpha10_final-delta_radial_fft_by_method_all_blocks.png`, `eps32_alpha10_adv-model-final_radial_fft_by_method_all_blocks.png`, and `eps32_alpha10_adv-solver-final_radial_fft_by_method_all_blocks.png`.
- Final-delta FFT CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_fft_metrics_summary.csv`.
- Updated summary Markdown: `docs/ns2d_recurrent_eps32_alpha10_visual_fft_conclusions_20260522.md`.

Observed metrics:
- `loss2/all_a_target_w` final delta has mean low-frequency fraction about `0.999975-0.999978` and spectral centroid about `0.0117-0.0122` across methods.
- Its first radial-profile bin contains about `0.91-0.93` of the normalized radial profile, higher than `loss1/all_w` (`0.75-0.82`) and `loss3/all_w` (`0.55-0.70`).

Correction/inference:
- Model/FNO outputs share major spectral directions with solver outputs but do not show the same visually sharp `2/3` cutoff as solver outputs.
- The solver and solver-change plots are the strongest visual evidence for the hard cutoff.
- Loss2 final-delta curves look unusually high at the low-frequency start because the perturbation is extremely concentrated in the first few radial Fourier bins. Heatmaps can make this less obvious because they use log-magnitude colormaps and percentile scaling rather than radial log-y profiles.

### 2026-05-22 22:02 UTC - Solver Radial Spectrum Bump Near 0.5 Interpretation

Status: inspected the method-grouped radial spectrum implementation and recorded interpretation of the solver bump/feature near `rho ~= 0.47-0.55`; no model/solver rerun.

Observed evidence:
- Script inspected: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_method_grouped_curves.py`.
- The plotted radial coordinate normalizes FFT radius by the diagonal Nyquist radius.
- The script marks `RADIAL_AXIS_CUTOFF = sqrt(2)/3 ~= 0.471` and `RADIAL_CORNER_CUTOFF = 2/3 ~= 0.667`.
- Radial profile uses mean FFT power per radial bin and then normalizes the binned profile.
- Updated summary Markdown: `docs/ns2d_recurrent_eps32_alpha10_visual_fft_conclusions_20260522.md`.

Inference:
- A square per-coordinate `2/3` mask becomes a radial transition band from `rho ~= 0.47` to `rho ~= 0.67`.
- In anisotropic spectra, the surviving angular sectors inside that square can produce an apparent bump or boundary near `rho ~= 0.5` before the final drop near `2/3`.
- This feature should be interpreted primarily as square-cutoff geometry plus anisotropic spectral directions, not as a separate physical cascade scale without further angular-sector diagnostics.

### 2026-05-22 22:08 UTC - Radial Frequency Normalization Note

Status: documented the radial FFT normalization explanation in the consolidated visual/FFT conclusions Markdown; no experiment/model/solver rerun.

Observed evidence:
- Updated summary Markdown: `docs/ns2d_recurrent_eps32_alpha10_visual_fft_conclusions_20260522.md`.

Recorded explanation:
- The radial plots normalize `rho` by the FFT square's diagonal corner radius `sqrt(0.5^2 + 0.5^2) = sqrt(2)/2`, so `rho=1` is the corner, not the horizontal Nyquist point.
- The per-coordinate `2/3` retained-mode cutoff gives `|kx|, |ky| <= 1/3`.
- The cutoff square intersects the axis at `rho = sqrt(2)/3 ~= 0.471` and reaches the retained square corner at `rho = 2/3 ~= 0.667`.

Inference:
- This explains why the solver radial spectrum has a transition band from about `0.471` to `0.667`, rather than a single radial cutoff point.

### 2026-05-22 22:12 UTC - Linear-Y Final-Delta Radial FFT Plot

Status: generated CPU-only linear-y final-delta radial FFT plots for `epsilon=32`, `alpha=10`; no model/solver rerun.

Observed evidence:
- Script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_final_delta_radial_linear_y.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_final_delta_radial_linear_y_20260522`.
- Generated plots: `eps32_alpha10_final-delta_radial_fft_by_method_all_blocks_linear_y.png` and `eps32_alpha10_final-delta_radial_fft_by_method_all_blocks_linear_y_lowfreq_zoom.png`.
- First-bin CSV: `eps32_alpha10_final_delta_radial_linear_y_first_bins.csv`.
- Updated summary Markdown: `docs/ns2d_recurrent_eps32_alpha10_visual_fft_conclusions_20260522.md`.

Observed first-bin values at `rho ~= 0.0052`:
- `loss2/all_a_target_w`: about `0.909-0.926` across methods.
- `loss1/all_w`: about `0.746-0.821` across methods.
- `loss3/all_w`: about `0.546-0.700` across methods.

Inference:
- Loss2's final delta is genuinely more concentrated in the lowest radial FFT bin.
- The log-y grouped plot visually compressed this separation because all these values lie near `10^0`; linear-y and low-frequency zoom make the difference clearer.

### 2026-05-22 22:18 UTC - Added Linear-Y Final-Delta Plots to Image Bundle

Status: copied the newly generated linear-y final-delta radial FFT plots into the existing image-only report package and updated package manifests; no model/solver rerun.

Observed evidence:
- Source directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_final_delta_radial_linear_y_20260522`.
- Bundle image directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/images_only/eps32_alpha10_final_delta_radial_linear_y_20260522`.
- Bundle records directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/records/eps32_alpha10_final_delta_radial_linear_y_20260522`.
- Added image files: `2`.
- Added record files: `2`.
- Current total image files in `images_only/`: `136`.
- Current total record files in `records/`: `3`.

Inference:
- The image-only package now includes the linear-y and low-frequency zoom views needed to inspect loss2's low-frequency final-delta concentration.

## 2026-05-22 22:23:23 UTC - NS2D attack optimizer behavior explanation

Status: analysis recorded; no model, solver, or GPU experiment was run for this update.

Source files / evidence:
- `docs/ns2d_recurrent_eps32_alpha10_visual_fft_conclusions_20260522.md`
- Prior 2D NS eps32/alpha10 visual/FFT artifacts under `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/`
- Prior 1D/Burgers comparison notes referenced in existing docs.

Observed evidence:
- In current 2D NS recurrent FNO attack records, `steepest_add` / LP-steepest PGD is often strongest, while replacement/GPI-style methods are not consistently best.
- The final perturbation and output FFT diagnostics show solver/dealiasing fingerprints and mode-dependent spectral behavior.

Inference:
- The 1D Burgers setting likely behaved closer to a locally quadratic/dominant-direction problem, where replacement/GPI is well matched.
- The 2D NS recurrent setting is more finite-radius, nonlinear, recurrent, solver-coupled, and frequency-filtered; additive normalized updates can preserve useful trajectory history while replacement can discard it.

Remaining work:
- Run smaller epsilon comparisons and a frozen-linearized/JVP diagnostic to test whether replacement becomes stronger when the objective is forced closer to a fixed quadratic problem.

## 2026-05-22 22:25:39 UTC - Validation plan for NS2D optimizer-geometry hypothesis

Status: analysis/plan recorded; no experiment was run and no GPU/solver/model computation was started.

Source files / evidence:
- `docs/ns2d_recurrent_eps32_alpha10_visual_fft_conclusions_20260522.md`
- Existing eps32/alpha10 attack curves, FFT plots, and visual report package.

Inference to validate:
- 1D Burgers likely favored replacement/GPI because it behaved closer to a fixed local quadratic/dominant-direction objective.
- 2D NS recurrent FNO likely favors `steepest_add` because gradients rotate across a nonlinear recurrent solver-coupled trajectory, so additive updates preserve useful perturbation history.

Planned validation:
- Frozen-linearized diagnostic, epsilon sweep, gradient-angle trajectory diagnostics, boundary-matched comparisons, solver-gradient ablations, spectral projection ablations, and true-loss vs surrogate-loss checks.

## 2026-05-22 22:29:03 UTC - 1D Burgers evidence check for NS2D optimizer hypothesis

Status: existing experiment records inspected and summarized; no new neural-operator run, GPU computation, solver call, or attack computation was started.

Source files / evidence:
- `docs/ns2d_vs_1d_burgers_optimizer_hypothesis_check_20260522.md`
- `docs/three_loss_burgers_optimizer_findings_summary_20260521.md`
- `docs/loss3_hypothesis_validation_status_20260521.md`
- `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
- `docs/loss3_path_geometry_theory_and_angle_evidence_20260517.md`
- `docs/loss3_current_mechanism_validation_summary_20260521.md`

Observed evidence:
- Prior 1D Burgers records support GPI/replacement as fastest early optimizer and strong boundary-direction method, but not as an unconditional final-loss theorem.
- Small-epsilon and path-geometry diagnostics support local-linear stability and early entry into a stable high-loss corridor in the tested Burgers setting.

Inference:
- This supports the current NS2D explanation in refined form: Burgers favored replacement because the boundary corridor quickly became stable/final-like, while NS2D likely favors `steepest_add` because the full recurrent solver-coupled objective is more path-dependent and benefits from accumulated perturbation history.

Remaining work:
- Run NS2D small-epsilon, early-to-final cosine, boundary-matched true-loss, and frozen-linearized diagnostics to test whether the same Burgers mechanisms appear or fail in 2D.

## 2026-05-22 22:31:41 UTC - Plain-language clarification on NS2D nonlinearity vs GPI intuition

Status: conceptual clarification recorded; no model, solver, GPU, plotting, or attack computation was run.

Source files / evidence:
- `docs/ns2d_vs_1d_burgers_optimizer_hypothesis_check_20260522.md`
- Current 2D NS eps32/alpha10 optimizer observations and prior 1D Burgers optimizer diagnostics.

Inference:
- The current explanation is that finite-radius nonlinearity, recurrent rollout, and solver/warm-up frame dependence can weaken replacement/GPI intuition even for `loss1`.
- A small-epsilon or frozen-linearized 2D NS diagnostic should test whether replacement/GPI becomes stronger when the problem is forced back toward a local Jacobian regime.

## 2026-05-22 22:35:25 UTC - Existing-data validation of NS2D optimizer complexity explanation

Status: existing local result summaries inspected only; no new neural-operator run, GPU computation, solver call, attack computation, or plotting was started.

Source files / evidence:
- `docs/ns2d_vs_1d_burgers_optimizer_hypothesis_check_20260522.md`
- `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/*/*/batch_0000_0009/*/*/summary.json`

Observed evidence:
- At `eps32_alpha10`, `steepest_add` strongly beats `steepest_replace` for 2D NS `loss1/all_w` and most `loss3` modes.
- At `eps8_alpha2p5`, the `loss1/all_w` gap nearly disappears and the `loss3/all_w` advantage shrinks, consistent with smaller epsilon being closer to a local-linear/replacement-friendly regime.

Inference:
- Existing 2D NS evidence supports the finite-radius/nonlinear/path-dependent explanation: more complex full objectives can favor LP-steepest additive PGD, while simpler or smaller-radius objectives look more replacement/GPI-like.
- A frozen-linearized 2D NS diagnostic remains the strongest missing confirmation.

## 2026-05-22 22:36:58 UTC - NS2D optimizer-hypothesis validation experiment design

Status: experiment design recorded; no neural-operator run, GPU computation, solver call, attack computation, or plotting was started.

Source files / evidence:
- `docs/ns2d_optimizer_hypothesis_validation_experiment_design_20260522.md`
- Existing 2D NS eps32/eps8 summaries under `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/`
- Prior 1D Burgers optimizer diagnostics in `docs/three_loss_burgers_optimizer_findings_summary_20260521.md` and related docs.

Designed experiments:
- Epsilon sweep, frozen-linearized/JVP diagnostic, early-to-final direction stability, gradient rotation diagnostics, boundary-matched true-loss comparison, W/D/A mode ablation, and hybrid add/replace probes.

Remaining work:
- Implement/run the priority experiments, starting with epsilon sweep and early-to-final direction stability because they can reuse the current attack logging structure.

## 2026-05-22 22:43:44 UTC - NS2D R2 and GitHub sync

Status: R2 uploads completed; GitHub source/record files staged for commit and push.

Source files / evidence:
- `docs/ns2d_r2_github_sync_20260522.md`
- R2 size verification from the completed `rclone size` commands.

Observed evidence:
- Visualization report package uploaded to R2 at `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522`, with `137` objects and `348.500 MiB`.
- Attack result directory uploaded to R2 at `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522`, with `411` objects and `7.679 GiB`.

Inference:
- Large generated artifacts are now backed up in R2 and should remain out of GitHub unless explicitly force-added later.

Remaining work:
- Commit and push the staged Markdown/source records to `origin/vast-ai`.
