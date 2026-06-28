# NS2D Recurrent Core4 Attack Full ADW Launch Command - 2026-05-22

## Status

Prepared for launch, not run by this documentation step.

Observed local prerequisites:

- Trained checkpoint exists locally: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`.
- Repaired dictionary exists locally: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- The repaired dictionary is the max-abs-4 version uploaded to R2 on 2026-05-22.
- Wrapper updated: `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh` now forwards `DICTIONARY_PATH`, `OUT_ROOT`, `TRUE_LOSS_EVERY`, and `FIXED_STEP`.

## Experiment Set

This is the corrected grouped experiment set, not the old full Cartesian product.

- `loss1`: `all_w` only.
- `loss2`: `all_a_target_w` only, using the dictionary for approximate intermediate frames and true solver target at frame 20.
- `loss3`: `all_w`, `all_d_target_w`, `w1_5_d6_9_target_w`, `d1_5_w6_9_target_w`, `a1_5_d6_9_target_w`.
- Four update methods for every group: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.

Total combinations: `(1 + 1 + 5) groups * 4 methods = 28 combinations`.

## Launch Command

This command runs ten test initial conditions, indices `0..9`, in one attack batch. It records the full surrogate-loss curve and full true all-W solver loss curve at every attack step with `TRUE_LOSS_EVERY=1`.

```bash
cd /workspace/NeuralOperatorRobustness2

export PYTHON_BIN=adv_robust/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

export CHECKPOINT=2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt
export DICTIONARY_PATH=2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt
export OUT_ROOT=2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522

export INDICES=0,1,2,3,4,5,6,7,8,9
export ATTACK_BATCH_SIZE=10
export STEPS=100
export EPSILON=32
export ALPHA=1
export P_ORDER=2
export Q_ORDER=2
export TRUE_LOSS_EVERY=1
export FIXED_STEP=0.005
export SOLVER_REMAT=chunk
export SOLVER_REMAT_CHUNK_STEPS=20
export DICTIONARY_CHUNK_SIZE=32
export EMPTY_TORCH_CACHE_AFTER_BATCH=1
export EMPTY_TORCH_CACHE_AFTER_METHOD=0
export CLEAR_JAX_CACHES_AFTER_BATCH=0

mkdir -p "$OUT_ROOT"
test -f "$CHECKPOINT"
test -f "$DICTIONARY_PATH"

LOG="$OUT_ROOT/launch_$(date -u +%Y%m%d_%H%M%S)_UTC.log"

echo "Logging to $LOG"
nvidia-smi | tee -a "$LOG"

run_attack_group () {
  local mode="$1"
  local loss="$2"
  echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') Running LOSS_TYPES=$loss MODE_SPEC=$mode ===" | tee -a "$LOG"
  MODE_SPEC="$mode" LOSS_TYPES="$loss" METHODS="raw_add raw_replace steepest_add steepest_replace"     ./2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh 2>&1 | tee -a "$LOG"
}

run_attack_group all_w loss1
run_attack_group all_a_target_w loss2

for MODE in   all_w   all_d_target_w   w1_5_d6_9_target_w   d1_5_w6_9_target_w   a1_5_d6_9_target_w
do
  run_attack_group "$MODE" loss3
done
```

## Expected Outputs

Each group creates one timestamped directory below `$OUT_ROOT`, with per-method subdirectories such as:

- `per_step_metrics.csv`: mean loss curves, surrogate loss curve, true solver loss curve, delta norm curve, and growth metrics.
- `per_sample_step_metrics.csv`: per-initial-condition loss curves and delta norm curves.
- `delta_threshold_crossings.csv`: first steps where `||delta||_p / epsilon` reaches 25%, 50%, 75%, and 100%.
- `final_delta_and_metrics.npz`: final delta and final adversarial initial conditions only.
- `summary.json`: final metrics and runtime.
- `solver_rollout_trace.csv`: solver call/runtime trace for that group.
- `batch_memory.csv`: CUDA memory snapshot after each attack batch.

No intermediate `trajectory_samples.npz` is written unless `SAVE_STEPS` is set.

## Runtime Estimate

Observed/inferred basis from previous attack benchmark notes:

- `ATTACK_BATCH_SIZE=10`, `SOLVER_REMAT=chunk`, `SOLVER_REMAT_CHUNK_STEPS=20`, `STEPS=100` is a conservative setting under the observed max passing batch range.
- Expensive all-W-like groups were previously estimated around `~32.4 min` per group for 100 steps and all four methods on 10 samples.
- Full set has 7 groups and 28 method combinations.

Inference:

- Conservative upper estimate: `7 groups * 32.4 min ~= 3.8 hours` if the timing is per grouped run with all four methods.
- Older rough upper estimate in prior notes was `28 * 32.4 min ~= 15.1 hours` if treating each method combination as separately that expensive. That is likely overly pessimistic if the measured 32.4 minutes already included all four methods for one group.
- Practical expected range for this command: roughly `4-15 hours`; watch the first completed group to tighten the estimate. If the first group takes about 30-35 minutes, expect roughly 4 hours total. If one method alone takes 30 minutes, expect closer to 15 hours.

## Parameter Notes

- `EPSILON=32`, `ALPHA=1`, `P_ORDER=2`, `Q_ORDER=2` are the current recommended scaled values for the attack code's whole-initial-condition L2 projection. Do not use the earlier mistaken `3276825` value.
- `TRUE_LOSS_EVERY=1` is intentionally expensive but records the true all-W solver loss at every step, which is the requested important curve.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` avoids the default large JAX preallocation. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35` leaves extra headroom compared with the wrapper default of `0.40`.
- `SOLVER_REMAT=chunk` with `20` micro-steps per rematerialized chunk is the current throughput/memory compromise.


## Runtime Status Check - 2026-05-22 02:00 UTC

Observed evidence:

- Launch log exists: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/launch_20260522_015450_UTC.log`.
- Active process observed: `adv_robust/bin/python 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` with `--loss-types loss1`, `--mode-spec all_w`, `--attack-batch-size 10`, `--steps 100`, `--epsilon 32`, `--alpha 1`.
- Active output directory observed: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_wwwwwwwwww_p2_q2_20260522_015452_UTC/`.
- `manifest.json` exists and records `jax_backend=gpu`, `jax_devices=[cuda:0]`, `torch_cuda_available=true`, `torch_device_name=NVIDIA A100-SXM4-80GB`, `torch_compute_capability=sm_80`.
- Current `nvidia-smi` observed `46851 MiB / 81920 MiB` memory use and `99%` GPU utilization.
- No `per_step_metrics.csv` had been written yet at this check, which is expected before the first method finishes because this script writes method CSV outputs after `run_one` completes.

Inference:

- The full ADW attack launch has started and is currently running the first group: `loss1` with `all_w`.
- The run is using GPU. The missing process name in the `nvidia-smi` process table appears to be a PID namespace/reporting issue, not evidence of CPU fallback, because GPU memory and utilization are high and the manifest confirms GPU backends.


## Runtime Estimate Update - 2026-05-22 02:05 UTC

Observed evidence:

- At `2026-05-22T02:04:38Z`, the run was still in the first group: `loss1` with `all_w`.
- Launch time for the first group in the log was `2026-05-22T01:54:51Z`, so elapsed time was about `9m47s`.
- No `per_step_metrics.csv` or method `summary.json` had been written yet, so the first method had not completed at this checkpoint.
- Existing historical local benchmark summaries for `loss3`, `batch_size=12`, `steps=5`, `solver_remat=chunk`, `chunk=20` show about `89.6s` for one method, roughly `17.9s/update` before extrapolation.

Inference:

- The earlier optimistic total estimate of about `4h` is likely too optimistic for this exact full command.
- A more realistic current estimate is about `25-35 minutes per method` for expensive all-W-like settings, especially because `TRUE_LOSS_EVERY=1` evaluates the true all-W solver loss curve at every step.
- With `28` method combinations, the total is likely around `12-16 hours`, with a conservative broad range of `10-18 hours` until the first completed method gives a direct measurement.
- From the launch time `2026-05-22T01:54:51Z`, this points to an approximate finish window of `2026-05-22 14:00-18:00 UTC`, with a conservative window of `2026-05-22 12:00-20:00 UTC`.


## Runtime Progress Update - 2026-05-22 03:12 UTC

Observed evidence:

- Current time checked: `2026-05-22T03:12:13Z`.
- Full launch started first group at `2026-05-22T01:54:51Z`.
- First group completed: `loss1` with `MODE_SPEC=all_w`.
- First group output: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_wwwwwwwwww_p2_q2_20260522_015452_UTC/summary.json`.
- First group summary recorded four completed method combinations with total runtime `4423.67s` (`73.73m`):
  - `raw_add`: `1110.34s` (`18.51m`).
  - `raw_replace`: `1104.52s` (`18.41m`).
  - `steepest_add`: `1104.40s` (`18.41m`).
  - `steepest_replace`: `1104.40s` (`18.41m`).
- Current active process is second group: `LOSS_TYPES=loss2`, `MODE_SPEC=all_a_target_w`.
- Second group started at `2026-05-22T03:08:47Z`, so it had been running about `3m26s` at this check.
- Current GPU status observed around `45029 MiB / 81920 MiB` and `99%` GPU utilization.

Inference:

- Completed progress: `1/7` groups, or `4/28` method combinations.
- Using the first completed group as the best current estimate, one group takes about `73.7m`.
- Full run estimate: `7 * 73.7m = 516m`, about `8h36m` total from launch.
- Remaining estimate at `2026-05-22T03:12:13Z`: about `7h15m-7h30m` if later groups are similar.
- Point ETA: about `2026-05-22 10:30 UTC`.
- Practical ETA range: about `2026-05-22 10:00-11:30 UTC`; keep a broader conservative range to noon UTC in case A/D groups or dictionary/solver behavior is slower.


## Semantic Clarification - loss1 and ADW Modes - 2026-05-22

Observed from code inspection:

- In `active_losses`, `need_target = loss_type == "loss3"`.
- Therefore, for `loss1`, the target-frame branch is not used and `g_delta` is not computed.
- `loss1` is computed as `||pred - f0||_q`, where `pred` is the model output for the perturbed initial condition and `f0` is the clean model output.
- The `mode_spec` still affects `loss1` indirectly because the recurrent FNO needs the first `t_in=10` input frames. In `model_prediction`, the first nine post-initial context frames are built according to the first nine characters of `mode_spec`.
- For the current run, `all_w` under `loss1` means differentiable solver-generated context frames for the FNO input, not a loss1 target mode.

Inference:

- The user is correct that ADW target semantics are not an independent axis for `loss1`.
- `loss1/all_w` should be interpreted as `loss1` with canonical differentiable-solver input-context generation, not as a target-mode combination.
- Future analysis should not compare `loss1` across ADW target modes. The completed first group should be labeled as `loss1/canonical_solver_context` or equivalent in analysis text.
- The current launch does not run `loss1` for all A/D/W modes; only that one canonical `loss1` group was run, then the launch moved on to `loss2/all_a_target_w` and the `loss3` mode groups.


## Potential Implementation Mismatch - ADW Frame Semantics - 2026-05-22

Observed from code inspection:

- `MODE_PRESETS` are resolved to a 10-character `mode_spec`.
- Current implementation interprets the first nine characters inside `model_prediction` as the policy for the FNO input context frames after the initial condition, i.e. frames needed to build the `t_in=10` model input.
- The last character is used only as the target-frame policy when `need_target=True`, which currently happens only for `loss3`.
- Therefore, current presets such as `w1_5_d6_9_target_w` control differentiability/dictionary usage for the model input context frames, not a full per-frame policy for frames 11-20 of the solver target trajectory.
- For `loss1`, `need_target=False`, so the target part of the preset is unused. The completed `loss1/all_w` group is best interpreted as `loss1` with canonical differentiable-solver input context.

Inference:

- This is likely a mismatch with the user's intended ADW design if the intended ADW states apply to frames 11-20 of the perturbed solver/target path.
- The currently running `loss2/all_a_target_w` and future `loss3` groups may not answer the intended ADW question, because A/D/W are attached to input-context construction rather than to the target-path frame semantics described by the user.
- Recommended action is to pause/stop the current launch before more GPU time is spent, then refactor the CLI to separate:
  - input context policy for frames 2-10, normally canonical solver `W`, and
  - target path policy for frames 11-20, used by `loss2`/`loss3` according to the intended A/D/W experiment design.

Remaining work:

- Decide whether to stop the current run.
- Refactor argument names and validation so `loss1` cannot be presented as an ADW target-mode experiment.
- Reimplement target-path ADW policies to match frames 11-20 if that is the intended experiment.


## Launch Combination Clarification - loss1 Was Not Swept Across ADW Modes - 2026-05-22

Observed evidence:

- Launch log shows only one `loss1` group: `LOSS_TYPES=loss1 MODE_SPEC=all_w`.
- After that group completed, the launch moved to `LOSS_TYPES=loss2 MODE_SPEC=all_a_target_w`.
- Current completed summary files include four `loss1` methods under only `mode_wwwwwwwwww.../loss1/`: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- No completed `loss1` summary files exist under A/D mixed mode directories in the active launch output.

Inference:

- The active launch did not run a Cartesian product of `loss1` with all ADW modes.
- The completed `loss1` work is exactly one group with four update methods.
- Conceptually, `loss1` should continue to be treated as its own single objective, not as an ADW target-mode sweep.


## loss1 Solver Usage Audit - 2026-05-22

Observed from code inspection:

- `loss1` is optimized in `active_losses` as `batch_norm(pred - self.f0, q_order)`.
- For `loss1`, `need_target=False`, so `model_prediction` does not compute the `g_delta` target branch.
- However, `model_prediction` still calls `_rollout_for_modes` to construct the first `t_in=10` model input frames. With `mode_spec=all_w`, this requires solver frames 1 through 9.
- Therefore, `loss1/all_w` performs differentiable solver rollout to frame 9 for the model input context during the attack objective/backward pass.
- Separately, because `TRUE_LOSS_EVERY=1`, `run_one` calls `true_loss_all_w(x_adv)` at every step for any non-`loss3/all_w` objective, including `loss1`. This no-grad logging rollout goes to the target frame index 19 and records the true solver loss curve.

Inference:

- The active launch did not duplicate `loss1` across ADW modes.
- The completed `loss1` group did use solver work; it was not merely a naming issue.
- Solver usage for model input context is necessary if the attack variable is only the initial condition and the recurrent FNO requires the first 10 frames as input.
- The full target-frame `true_loss_all_w` solver rollout during `loss1` is not part of the optimized loss1 objective; it is diagnostic logging requested for true-loss curves. If not needed for loss1, it can be disabled or restricted in future runs to save time.
- If the intended definition of `loss1` is pure neural-model-on-fixed-10-frame-input with no solver-derived context, then the current implementation does not match that definition and needs refactoring.


## Conservative Runtime Estimate Revision - 2026-05-22 03:38 UTC

Observed evidence:

- Current time checked: `2026-05-22T03:37:52Z`.
- Active process remained in `LOSS_TYPES=loss2`, `MODE_SPEC=all_a_target_w`.
- Completed `loss1/canonical_solver_context` group: 4 methods, total `73.73m`, about `18.4m/method`.
- Completed `loss2/all_a_target_w` methods so far:
  - `raw_add`: `466.49s` (`7.77m`).
  - `raw_replace`: `461.50s` (`7.69m`).
  - `steepest_add`: `461.39s` (`7.69m`).
- Current GPU status observed around `45029 MiB / 81920 MiB`, `100%` utilization.

Inference:

- The previous estimate based only on the first group may still be optimistic for the remaining `loss3` groups.
- `loss2/all_a_target_w` is faster than `loss1` because the active objective uses dictionary frames and does not require a differentiable solver target path; it still pays diagnostic true-loss logging because `TRUE_LOSS_EVERY=1`.
- The remaining expensive part is the five `loss3` groups. They are expected to be slower than `loss2/all_a`, and possibly slower than `loss1`, because the perturbed target path to frame 20 participates in the objective/backward path.
- Remaining work after `loss2` is approximately 20 `loss3` method combinations.
- Lower-bound remaining time if `loss3` matches `loss1`: about `20 * 18.4m = 6.1h`, plus the last `loss2` method.
- More conservative practical estimate for `loss3`: `25-35m/method`, giving about `8.3-11.7h` for the remaining 20 method combinations.
- Revised practical remaining estimate at `2026-05-22T03:38Z`: about `8-12h`.
- Revised finish window: roughly `2026-05-22 11:30-15:30 UTC`; optimistic lower bound around `10:00 UTC`, conservative cushion to `16:00 UTC`.


## Stop Event - 2026-05-22 03:58 UTC

Observed evidence:

- The running full ADW attack launch was stopped after the user requested pausing/restarting.
- Initial `SIGTERM` stopped the active `loss3/all_w` process, but the outer shell loop continued and started subsequent groups.
- A short stopper loop then terminated the auto-restarted wrapper/Python/tee processes for `all_d_target_w`, `w1_5_d6_9_target_w`, `d1_5_w6_9_target_w`, and `a1_5_d6_9_target_w`.
- Final process check found no matching `attack_ns2d_recurrent_core4`, `run_ns2d_recurrent_core4_attack`, `full_adw_b10_eps32_alpha1`, or loss1 visualization process.
- Final `nvidia-smi` showed `0MiB / 81920MiB`, `0%` GPU utilization, and no running GPU processes.
- Launch log shows completed groups:
  - `loss1` with `MODE_SPEC=all_w` completed.
  - `loss2` with `MODE_SPEC=all_a_target_w` completed.
- Launch log shows interrupted/not-completed groups:
  - `loss3` with `MODE_SPEC=all_w` was started and then stopped.
  - Remaining `loss3` groups were started by the outer loop but immediately stopped before completion.

Inference:

- The GPU is free for a corrected rerun.
- Existing completed `loss1` and `loss2/all_a_target_w` outputs remain on disk, but the current launch should be treated as stopped/incomplete.
- The next rerun should use corrected `loss1` handling, especially nonzero/random initialization if `loss1` is included.
