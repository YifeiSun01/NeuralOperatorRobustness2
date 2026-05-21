# 2D NS Recurrent Core4 Attack Batch-Size Probe

Observed on 2026-05-21 on `NVIDIA A100-SXM4-80GB` after GPU verification:

- PyTorch: `2.8.0+cu126`, CUDA runtime `12.6`, device compute capability `sm_80`.
- JAX: `0.10.0`, backend `gpu`, device `CudaDevice(id=0)`.
- Probe path: `loss3 + raw_add + MODE_SPEC=all_w + STEPS=1 + EPSILON=32 + ALPHA=1 + p=q=2`.
- This path uses solver rollout to target frame 19 and backward through the differentiable solver path.

Observed successful probes:

| attack batch size | output root | max allocated | max reserved | solver trace |
|---:|---|---:|---:|---|
| 10 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215124_UTC` | `45746082816` bytes | `46787461120` bytes | `10x256x256 -> 20x10x256x256`, `requires_grad=True` call present |
| 12 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215421_UTC` | `54703543296` bytes | `55857643520` bytes | `12x256x256 -> 20x12x256x256`, `requires_grad=True` call present |
| 14 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215525_UTC` | `63663100928` bytes | `64929923072` bytes | `14x256x256 -> 20x14x256x256`, `requires_grad=True` call present |

Observed failed probes:

| attack batch size | output root | failure |
|---:|---|---|
| 15 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215632_UTC` | JAX backward / DLPack path OOM while trying to allocate about `14.79 GiB`; only `manifest.json` was written. |
| 16 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215325_UTC` | JAX backward / DLPack path OOM while trying to allocate about `15.78 GiB`; only `manifest.json` was written. |
| 20 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215231_UTC` | PyTorch recurrent FNO FFT OOM at `torch.fft.rfft2`; process had about `79.04 GiB` in use and only about `201 MiB` free. |

Inference:

- For this exact one-step `loss3/all_w/raw_add` solver-backward probe, the largest observed passing attack batch size is `14`; `15` fails.
- For full attack sweeps, `ATTACK_BATCH_SIZE=12` is the recommended practical value because it leaves more headroom for allocator fragmentation, JAX/PyTorch caches, and future mode/method variations.
- `ATTACK_BATCH_SIZE=14` is usable only as an aggressive setting and should be monitored closely.
- `ATTACK_BATCH_SIZE=15` and above should be treated as unsafe for this path on the current A100 80GB setup.

Post-probe status:

- After the OOM probes exited, `nvidia-smi` showed `0 MiB / 81920 MiB` in use, so the failed processes released GPU memory.
