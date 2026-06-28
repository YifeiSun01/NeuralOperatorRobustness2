# 2D NS Recurrent Core4 Attack Memory Optimization

Observed from source edits on 2026-05-21:

- `XLA_PYTHON_CLIENT_PREALLOCATE` defaults to `false` before JAX import, so JAX does not reserve most of the GPU at process start.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` now defaults to `0.40` in both the Python CLI and shell wrapper. This is mainly a safety limit if preallocation is re-enabled externally; the normal default remains lazy allocation.
- The wrapper intentionally does not set `XLA_PYTHON_CLIENT_ALLOCATOR=platform` by default. That allocator can return memory to the driver more aggressively, but it often slows repeated JAX/PyTorch work.
- The recurrent FNO model, differentiable NS rollout cache, and optional approximate dictionary are now constructed once per run and reused across attack batches. They are no longer reloaded for every batch.
- Per-step metric logging now computes `delta`, gradient, and direction norms on GPU and only transfers small metric vectors/scalars to CPU. Full `delta` and `x_adv` arrays are transferred only for requested save steps and final artifacts.
- Approximate-dictionary nearest-neighbor search now uses the identity `||q-c||^2 = ||q||^2 + ||c||^2 - 2 q.c`, avoiding the previous `[B, chunk, H*W]` temporary difference tensor.
- The DLPack bridge detaches and reuses contiguous tensors deliberately, avoiding unnecessary autograd references across the JAX boundary.
- CUDA memory snapshots are written to `manifest.json`, `summary.json`, and `batch_memory.csv` for attack outputs.
- New cache-control knobs are exposed: `--empty-torch-cache-after-batch`, `--no-empty-torch-cache-after-batch`, `--empty-torch-cache-after-method`, and `--clear-jax-caches-after-batch`.

Inference from the code changes:

- The largest practical memory/cache savings should come from avoiding repeated checkpoint/dictionary reloads and avoiding full tensor CPU transfers during every attack step.
- The default path should keep speed close to the previous probe because JAX compilation caches are retained by default and the platform allocator is not forced.
- If the user needs the lowest possible displayed GPU cache after each batch, `CLEAR_JAX_CACHES_AFTER_BATCH=1` and/or `XLA_PYTHON_CLIENT_ALLOCATOR=platform` can be tested, but those settings may increase runtime.

Validation performed:

- `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py` passed.
- `bash -n 2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh` passed.

Not run:

- No full attack benchmark was launched during this optimization step.
- No GPU solver/model workload was run for this validation step.

Post-edit GPU probe:

- Command shape: `ATTACK_BATCH_SIZE=6`, `LOSS_TYPES=loss3`, `METHODS=raw_add`, `MODE_SPEC=all_w`, `STEPS=1`, `EPSILON=3276825`, `ALPHA=5`, `p=2`, `q=2`, `SAVE_STEPS=0`.
- Output source: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_214034_UTC`.
- Observed from `manifest.json`: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`, JAX backend `gpu`, PyTorch device `NVIDIA A100-SXM4-80GB`.
- Observed from `manifest.json`: CUDA memory after shared model load was `943966720` allocated bytes and `958398464` reserved bytes.
- Observed from `batch_memory.csv`: peak allocated bytes during the one-step batch-6 probe were `27829064704`; peak reserved bytes were `29462888448`.
- Observed from `summary.json`: after batch-level cache release inside the still-running process, reserved bytes dropped to `979369984`.
- Observed from post-run `nvidia-smi`: after process exit, GPU memory usage returned to `0 MiB / 81920 MiB`.
- Observed runtime for the single `loss3/raw_add` one-step method was `19.43s`; this includes JAX compile/warm-up cost for the process.
