# Loss3 Direction-Proposal Ablation Smoke Runtime - 2026-05-17

Status: tiny GPU smoke completed.  This is a runtime and output-schema test, not
a scientific optimizer conclusion.

## Command

```bash
./adv_robust/bin/python tools/run_loss3_direction_proposal_ablation.py \
  --out-root forensics/loss3_optimizer_direction_proposal_ablation_20260517/smoke_tiny_runtime_p2_q2_idx0_7_steps3 \
  --dataset-indices 0 7 \
  --methods official3 main \
  --epsilon 8 \
  --alpha 0.3 \
  --steps 3 \
  --p 2 \
  --q 2 \
  --device cuda \
  --save-delta-trajectory \
  --make-gifs
```

## GPU Verification

Observed before the run:

- `nvidia-smi`: Tesla V100-SXM2-32GB, compute capability `7.0`, driver `580.142`.
- `tools/setup_adv_robust_gpu_env.py --verify-only`: PASS.
- PyTorch: `2.8.0+cu126`, CUDA `12.6`.
- Active device: Tesla V100-SXM2-32GB, compute capability `sm_70`.
- PyTorch arch list includes `sm_70`.
- JAX backend: `gpu`, devices `[CudaDevice(id=0)]`.
- PyTorch and JAX GPU matmul sanity checks passed.
- `pip check`: no broken requirements.

## Observed Outputs

Observed from `forensics/loss3_optimizer_direction_proposal_ablation_20260517/smoke_tiny_runtime_p2_q2_idx0_7_steps3`:

- `manifest.json` status: `completed`.
- Methods run: `14`.
- Samples: dataset indices `0` and `7`.
- Steps: `3`, therefore four saved states per method: `k=0,1,2,3`.
- Root files include `per_step_metrics.csv`, `per_sample_step_metrics.csv`,
  `direction_cosines.csv`, `method_pairwise_final_delta_cosines.csv`,
  `pq_geometry_summary.csv`, `cross_norm_final_metrics.csv`, `final_deltas.npz`,
  `analysis_checklist.md`, figures, and GIFs.
- Row counts:
  - `per_step_metrics.csv`: 56 data rows.
  - `per_sample_step_metrics.csv`: 112 data rows.
  - `direction_cosines.csv`: 546 data rows.
  - `method_pairwise_final_delta_cosines.csv`: 182 data rows.
- Generated files under `figures/`: 66.
- Generated GIFs: 28.

## Runtime Observations

Observed from method `summary.json` files:

| Method | Runtime seconds | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|
| `generalized_power` | 1.860 | 31.6 | 40.0 |
| `lp_steepest_pgd` | 1.829 | 31.6 | 40.0 |
| `pgd` | 2.527 | 31.6 | 40.0 |
| `power_add__affine_jvp_vjp` | 3.210 | 55.6 | 64.0 |
| `power_add__objective_gradient` | 1.830 | 31.6 | 40.0 |
| `power_add__pure_jvp_vjp` | 3.576 | 55.6 | 64.0 |
| `power_replace__affine_jvp_vjp` | 3.224 | 55.6 | 64.0 |
| `power_replace__objective_gradient` | 1.854 | 31.6 | 40.0 |
| `power_replace__pure_jvp_vjp` | 3.199 | 55.6 | 64.0 |
| `raw_add` | 1.827 | 31.6 | 40.0 |
| `raw_replace` | 1.843 | 31.6 | 40.0 |
| `steepest_add` | 1.819 | 31.6 | 40.0 |
| `steepest_replace` | 1.822 | 31.6 | 40.0 |
| `unit_raw_add` | 1.849 | 31.6 | 40.0 |

Observed aggregate runtime:

- Sum of per-method optimizer runtimes: `32.27` seconds.
- Mean per-method runtime: `2.31` seconds.
- Median per-method runtime: `1.85` seconds.
- Output file modification-time span for the whole smoke, including diagnostics,
  plots, and GIFs: `74.10` seconds.

Observed comparison with previous batch100 runs:

- Recent local FNO batch100/steps100 `loss3_original_pgd`: `104.95` seconds.
- Recent local FNO batch100/steps100 `loss3_original_lp_steepest_pgd`: `103.88` seconds.
- Recent local FNO batch100/steps100 `loss3_original_generalized_power`: `103.47` seconds.
- Older 2026-05-14 FNO batch100/steps100 official-method runtimes were around
  `283` to `285` seconds per method, so the estimate should keep a safety margin.

## Runtime Estimate

Inference from the smoke and previous batch100 records:

- Ordinary autograd-gradient methods in the new `main` preset should cost roughly
  one prior official-method runtime each, about `105` seconds per method on the
  current machine.
- q-aware JVP/VJP methods were about `1.7x` to `1.9x` slower than ordinary
  methods in the tiny smoke.  A practical estimate is about `180` to `200`
  seconds per q-aware method for batch100/steps100.
- The `main` preset has 7 ordinary/objective-gradient methods and 4 q-aware
  power methods.

Estimated p=2/q=2 official main run, batch100/steps100:

```text
7 ordinary methods * ~105 s   ~= 12.3 min
4 q-aware methods * ~190 s    ~= 12.7 min
optimizer subtotal            ~= 25 min
static diagnostics/plots       + 3-10 min
```

Recommended planning estimate without GIFs: `30-40 minutes` for one p/q pair.

GIFs are much more expensive at official scale.  The tiny smoke produced 28 GIFs
with four frames each.  A full run with 11 methods, five samples, and 101 saved
steps would produce 55 GIFs and thousands of frames.  Recommended planning
estimate with full GIFs enabled: `50-75 minutes` for one p/q pair.

For the four-pair p/q matrix `(2,2)`, `(inf,2)`, `(2,inf)`, `(inf,inf)`:

- Without GIFs: about `2-3 hours` total.
- With full GIFs for every pair: about `3.5-5 hours` total.

## Recommendation

Run the official batch100 p=2/q=2 main comparison first without GIFs, but keep
`--save-delta-trajectory` enabled.  Then generate or rerun GIFs only for selected
methods/samples after seeing which methods are scientifically interesting.

Suggested next command:

```bash
./adv_robust/bin/python tools/run_loss3_direction_proposal_ablation.py \
  --out-root forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2 \
  --batch-size 100 \
  --start-index 0 \
  --methods main \
  --epsilon 8 \
  --alpha 0.3 \
  --steps 100 \
  --p 2 \
  --q 2 \
  --device cuda \
  --trajectory-indices 0 7 40 47 \
  --save-delta-trajectory
```

Do not use this smoke to make optimizer-quality claims.  It only verifies that
the code path runs, outputs are produced, and runtime is within a manageable
range.
