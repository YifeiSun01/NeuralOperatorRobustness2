# NS2D Optimizer-Hypothesis Validation Runtime Estimate

Updated: 2026-05-22 23:59:58 UTC

Status: estimate only. No new attack, solver, model, plotting, or GPU computation was started.

## Observed Timing Basis

Observed from the active pair-outer NS2D attack log on the A100-SXM4-80GB machine:

| run type | observed duration |
|---|---:|
| `loss1 / all_w` | about `75 min` |
| `loss2 / all_a_target_w` | about `33 min` |
| one `loss3` mode | about `123-125 min` |
| one full current pair with 5 loss3 modes | about `12.1 h` |

The current running script is heavier than the minimal validation plan because each epsilon/alpha pair runs:

1. `loss1/all_w`
2. `loss2/all_a_target_w`
3. `loss3/all_w`
4. `loss3/all_d_target_w`
5. `loss3/w1_5_d6_9_target_w`
6. `loss3/d1_5_w6_9_target_w`
7. `loss3/a1_5_d6_9_target_w`

## Experiment 1: Epsilon Sweep

Minimal version from the design:

- per epsilon: `loss1/all_w` + `loss2/all_a_target_w` + `loss3/all_w`
- estimated per epsilon: `75 + 33 + 124 = 232 min`, about `3.9 h`
- four epsilons: about `15.5-16 h`

Current expanded version:

- per epsilon: loss1 + loss2 + five loss3 modes
- estimated per epsilon: `75 + 33 + 5*124 = 728 min`, about `12.1 h`
- four pairs: about `48-49 h`

Observed current run:

- `eps32_alpha10` complete.
- `eps8_alpha2p5` is in progress.
- remaining from the last status check was about `30-31 h` for the full current script.
- current script does not include `eps4_alpha1p25`; it includes `eps32_alpha15` instead.

## Experiment 2: Early-To-Final Direction Stability

If the current attack already records per-step sample traces for one sample only, then a first-pass early-to-final cosine analysis can be done mostly offline from saved `step_sample_trace.npz` and final deltas.

Estimated extra runtime:

- CPU/offline plotting/tables: about `10-30 min`.
- If full-batch intermediate deltas are not saved and must be rerun for all 10 samples, then it costs about one attack run for each selected loss/mode. For the minimal three cases, about `3.9 h` per epsilon or `15.5 h` for four epsilons.

Recommended first pass:

- Use already saved step-sample traces for the representative sample.
- Do not rerun full attacks unless the one-sample evidence is unclear.

## Experiment 3: Boundary-Matched True-Loss Comparison

If per-step metrics already contain delta norm, true loss, and boundary ratio, this is offline.

Estimated extra runtime:

- CPU/offline analysis: about `10-30 min`.
- No GPU rerun needed if current metrics are sufficient.

## Experiment 4: Gradient Rotation Diagnostic

The current script has `--record-step-sample-gradients`, so one-sample gradient rotation can likely be computed offline.

Estimated extra runtime:

- one-sample offline analysis: about `20-60 min`.
- full-batch gradient rotation would require rerunning or saving much larger per-step gradient arrays; estimate roughly the same as the corresponding attack runtime, plus heavy disk output. Not recommended first.

## Experiment 5: Frozen-Linearized / Local Jacobian Diagnostic

This is the cleanest but most expensive/new implementation experiment.

Three possible versions:

1. Tiny pilot, 1 sample, one loss, one epsilon-like radius:
   - estimated implementation/debug time: `0.5-1 day`.
   - GPU runtime: probably `1-3 h`, but uncertain because JVP/VJP/Jacobian strategy matters.

2. Useful pilot, 1-2 samples, loss1 and loss3:
   - estimated implementation/debug + run: `1-2 days`.

3. Full population-level version:
   - likely multiple days and not worth doing until the pilot proves the diagnostic is informative.

## Practical Recommendation

Fastest meaningful validation path:

1. Let the current expanded epsilon run finish: remaining about `30-31 h` from the last status check.
2. Immediately do offline analyses from existing saved records:
   - early-to-final cosine for saved step sample,
   - boundary-matched true loss,
   - gradient rotation for saved step sample.
   - estimated offline time: `1-2 h` total.
3. Add a small missing `eps4_alpha1p25` minimal run only for:
   - `loss1/all_w`
   - `loss2/all_a_target_w`
   - `loss3/all_w`
   - estimated time: about `3.9 h`.
4. Only then decide whether to implement the frozen-linearized diagnostic.

## Total Estimates

Minimal evidence package after current run finishes:

- current remaining: about `30-31 h`.
- offline analyses: `1-2 h`.
- optional missing eps4 minimal run: about `4 h`.
- total from last status point: about `35-37 h` if eps4 is added.

If starting from scratch for the minimal epsilon sweep only:

- about `15.5-16 h` for four epsilons and three cases per epsilon.

If doing the full expanded current-style sweep from scratch:

- about `48-49 h` for four pairs with five loss3 modes each.

If also adding a frozen-linearized diagnostic:

- add about `1-2 days` for a useful pilot, more for a full version.


## 2026-05-23 Offline Analysis Feasibility Check

Status: local files inspected only. No Python analysis, plotting, model
inference, solver call, attack step, or GPU computation was started.

Observed from local files under:

- `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/`

Available records include:

- `per_step_metrics.csv`
- `per_sample_step_metrics.csv`
- `step_sample_trace.npz`
- `step_sample_trace_metrics.csv`
- `final_delta_and_metrics.npz`
- `final_state_metrics.csv`
- `summary.json`
- `solver_rollout_trace.csv`

Inference from these records:

- Early-to-final cosine can be computed from saved `step_sample_trace.npz`
  and final delta records, at least for the saved representative sample.
- Boundary-matched true-loss comparison can be computed from
  `per_step_metrics.csv` / `per_sample_step_metrics.csv` if the needed true
  loss and delta-norm columns are present.
- Gradient rotation diagnostics can be computed from the saved representative
  sample traces where gradient/update arrays were recorded; full-batch
  gradient rotation would require much larger saved gradient arrays or a rerun.

Expected memory:

- GPU memory: `0 GiB` if implemented as CPU-only CSV/NPZ post-processing.
- CPU RAM for CSV/table-only analysis: usually well under `1 GiB`.
- CPU RAM for one `step_sample_trace.npz` at a time: roughly tens to hundreds
  of MiB, depending on which arrays are loaded.
- Conservative safe implementation: stream/read one method directory at a time
  and close arrays promptly. This should avoid disturbing the active GPU attack.

Expected runtime:

- Per completed epsilon/mode/method set: minutes, not hours.
- For all currently completed local records: roughly `30-90 min` to compute
  tables and summary plots, assuming the columns/arrays are already present.
- With extra polished figures and Markdown interpretation: about `1-2 h`.

Operational recommendation:

- These three diagnostics are safe to run while the attack continues only if
  the script is forced to CPU and does not import/use JAX or PyTorch GPU.
- Use NumPy/Pandas/Matplotlib CPU-only post-processing and set
  `CUDA_VISIBLE_DEVICES=""` as an additional guard.
- Do not compute missing solver/model outputs during this pass; that would
  become a real GPU experiment and could interfere with the running attack.
