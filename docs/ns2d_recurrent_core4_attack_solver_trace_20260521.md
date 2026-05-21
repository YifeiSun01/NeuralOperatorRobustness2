# 2D NS Recurrent Core4 Attack Solver Trace

Observed on 2026-05-21 after adding explicit solver-rollout tracing:

- Source changed: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- New output file per run: `solver_rollout_trace.csv`.
- Trace columns include `t_final`, `fixed_step`, `steps_per_second`, `micro_steps_per_sample`, input/output shape, `requires_grad`, cache hit, source, mode spec, and required frames.

Proof probe:

- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_214900_UTC`.
- Settings: `INDICES=0`, `ATTACK_BATCH_SIZE=1`, `LOSS_TYPES=loss3`, `METHODS=raw_add`, `MODE_SPEC=all_w`, `STEPS=1`, `EPSILON=32`, `ALPHA=1`, `p=q=2`.
- Observed solver trace:
  - Call 0: `t_final=9`, `micro_steps_per_sample=1800`, `x_shape=1x256x256`, `output_shape=10x1x256x256`, `requires_grad=False`, source `model_prediction`, required frames `1..9`.
  - Call 1: `t_final=19`, `micro_steps_per_sample=3800`, `x_shape=1x256x256`, `output_shape=20x1x256x256`, `requires_grad=True`, source `model_prediction`, required frames `1..9,19`.
  - Call 2: `t_final=19`, `micro_steps_per_sample=3800`, `x_shape=1x256x256`, `output_shape=20x1x256x256`, `requires_grad=False`, source `model_prediction`, required frames `1..9,19`.
- Observed from summary: `solver_rollout_calls=3`; the method completed in `15.016598572023213` seconds.
- Observed memory: max allocated `5442786304` bytes and max reserved `5890899968` bytes for this batch-1 one-step probe.
- Observed post-run `nvidia-smi`: GPU memory returned to `0 MiB / 81920 MiB`.

Inference:

- The proof probe did invoke the differentiable NS solver to frame 19 for `loss3/all_w`, with `requires_grad=True` during the attack update.
- The lower-than-expected memory is consistent with the JAX solver using `jax.checkpoint` inside the scan, which trades recomputation for reduced stored activations.
- The earlier batch-6 probe did not include this trace because the trace was added afterward, but it used the same `loss3/all_w` code path according to its manifest and summary.
