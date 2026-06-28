# 2D NS Recurrent Core4 Attack Probe - 2026-05-21

Observed probe settings:

- Checkpoint: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`
- Norm: `p=2`, `q=2`
- Epsilon: `3276825`
- Alpha: `5`
- Mode spec: `all_w` / `wwwwwwwwww`
- Probe loss/method for capacity: `loss3` + `raw_add` + `steps=1`

Observed results:

- Batch size `3` passed for `loss3/raw_add/steps=1`.
- Batch size `4` passed for `loss3/raw_add/steps=1`.
- Batch size `6` passed for `loss3/raw_add/steps=1`.
- Full-combination sanity probe passed with batch size `6`: `6 samples x 3 losses x 4 methods x steps=1`.
- Full-combination probe output directory: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_212630_UTC`.
- GPU after probes: observed `0 MiB` used and `81153 MiB` free.

Inference:

- `ATTACK_BATCH_SIZE=6` is usable for this initial p2/q2 setting on the current A100 after the training process released memory.
- Since only `steps=1` was probed, long runs should still be monitored, but the basic model/solver/backward path and all four optimizer methods are runnable.

## Post-Memory-Optimization Probe

Observed on 2026-05-21 after cache/memory code changes:

- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_214034_UTC`.
- Settings: `ATTACK_BATCH_SIZE=6`, `LOSS_TYPES=loss3`, `METHODS=raw_add`, `MODE_SPEC=all_w`, `STEPS=1`, `EPSILON=3276825`, `ALPHA=5`, `p=2`, `q=2`, `SAVE_STEPS=0`.
- Observed from `batch_memory.csv`: max allocated `27829064704` bytes and max reserved `29462888448` bytes.
- Observed from `summary.json`: final in-process reserved memory after cache release was `979369984` bytes.
- Observed from post-run `nvidia-smi`: GPU memory returned to `0 MiB / 81920 MiB`.
