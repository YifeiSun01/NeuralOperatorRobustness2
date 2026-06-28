# 2D NS Recurrent FNO Core4 Attack Code

Observed from source reading only on 2026-05-21:

- `tools/run_loss3_direction_proposal_ablation.py` contains the core-four optimizer split: `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- `2D_NS_FNO2d_recurrent/perturbation_methods/PGD_attack_adam_batch_adaptive.py` contains the existing 2D NS recurrent-FNO solver bridge, dictionary approximation, and `a/d/w` mode-spec handling.
- `2D_NS_FNO2d_recurrent/models/FNO2d.py` confirms the recurrent model input shape is `(batch, H, W, 10)` and the recurrent predictor returns the next 10 frames.

Code added:

- `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`
- `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh`

Design:

- Losses are `loss1 = ||F(x+delta)-F(x)||_q`, `loss2 = ||F(x+delta)-G(x)||_q`, and `loss3 = ||F(x+delta)-G(x+delta)||_q`.
- Optimizers are `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace`.
- The mode spec is 10 characters: frames 1 through 9 for the recurrent input, plus the final target frame. `a` means dictionary approximation, `d` means solver forward with detach, and `w` means solver forward with gradient.
- The default target frame is zero-based index 19, corresponding to the twentieth frame.
- The script refuses CPU fallback for real runs.

Not run:

- No attack was executed.
- No model, dataset, or dictionary was loaded during this code-writing step.
- No GPU computation was started.

Memory/cache update on 2026-05-21:

- JAX lazy allocation is the default: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`.
- The model, differentiable rollout cache, and optional dictionary are shared once per run instead of reloaded per attack batch.
- Step metrics avoid full `delta/grad/direction` GPU-to-CPU copies; only small norm vectors/scalars are transferred each step.
- Dictionary nearest-neighbor lookup avoids the large `[B, chunk, H*W]` diff tensor by using a matrix-product distance formula.
- Detailed record: `docs/ns2d_recurrent_core4_attack_memory_optimization_20260521.md`.
