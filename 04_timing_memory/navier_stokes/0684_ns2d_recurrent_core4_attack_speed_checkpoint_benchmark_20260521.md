# 2D NS Recurrent Core4 Attack Speed Benchmark With Solver Checkpointing

Observed on 2026-05-21 on `NVIDIA A100-SXM4-80GB`:

- PyTorch: `2.8.0+cu126`, CUDA runtime `12.6`, compute capability `sm_80`.
- JAX: `0.10.0`, backend `gpu`.
- Benchmark path: `loss3 + raw_add + MODE_SPEC=all_w + STEPS=5 + EPSILON=32 + ALPHA=1 + p=q=2`.
- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215942_UTC`.
- Batch size: `12` test initial conditions.
- Solver trace: seven rollout calls; update calls used `t_final=19`, `fixed_step=0.005`, `micro_steps_per_sample=3800`, `x_shape=12x256x256`, `output_shape=20x12x256x256`, `requires_grad=True`.

Observed timing from `per_step_metrics.csv`:

| k | seconds since method start | interpretation |
|---:|---:|---|
| 0 | `20.525715954019688` | first update, includes JAX compile/warmup for this process |
| 1 | `35.53788198099937` | update 2 complete |
| 2 | `50.5436809649691` | update 3 complete |
| 3 | `65.5657470979495` | update 4 complete |
| 4 | `80.5680645769462` | update 5 complete |
| 5 | `84.67072753596585` | final no-backward evaluation |

Derived estimates:

- First update: about `20.5s`.
- Warm update steps after compile: about `15.0s` per step for batch size `12`.
- Final no-backward evaluation: about `4.1s`.
- One `100`-step `loss3/all_w/raw_add` attack on one batch of 12 samples is estimated at about `25.2` minutes: `20.5 + 99*15.0 + 4.1` seconds.
- Memory for this benchmark: max allocated `54712062976` bytes and max reserved `55857643520` bytes.

Inference:

- JAX checkpoint/rematerialization does slow the solver path by recomputing during backward.
- Compared with the user's recalled B200 number of about `7s/step` for batch size `7`, this A100/checkpointed path is about `2.1x` slower per optimizer step, but processes `12` samples per step instead of `7`. Per sample, the observed A100/checkpointed path is roughly `15/12 = 1.25s` per sample-step versus the recalled `7/7 = 1.0s` per sample-step, so the throughput penalty is closer to about `25%` under that comparison, while using much less peak memory than the recalled B200 run.
- A full sweep over many loss/method/mode combinations can still take hours because combinations run sequentially.
