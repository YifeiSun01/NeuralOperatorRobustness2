# Loss3 Mechanism/Landscape Existing-vs-New Audit - 2026-05-20

Status: completed audit plus one post-processing extension and one small GPU landscape pilot.

## What Already Existed And Can Be Reused

Observed existing ray-profile evidence:

- `docs/loss3_ray_profile_ri_final_report_fno_nu0p001_20260516.md`
- `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`

This already proves a local-to-global point for old PGD directions: clean local directions can win near radius 0 but lose at large radius, while finite-radius PGD directions win near `epsilon=8`. It is useful background, but it is not a current core4/PQ comparison.

Observed existing 2D slice/curvature evidence:

- `docs/loss3_2d_slice_planarity_fno_nu0p001_steps100_dense_result_20260517.md`
- `docs/loss3_directional_curvature_fno_nu0p001_steps100_samples5_20260517.md`
- `docs/loss3_path_linearity_curvature_raw_data_master_20260517.md`

These already show PGD-path local planarity/curvature behavior: early curvature is higher, then the path enters a lower-curvature corridor, with a moderate late-boundary rebound. This is useful mechanism background, but it follows saved `loss3_original_pgd` paths, not the current four core methods or multiple PQ geometries.

Observed existing Jacobian/SVD evidence:

- `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`
- `forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/`
- broader local-Jacobian/SVD docs from 2026-05-15.

These already show that residual-Jacobian geometry rotates and steepens along an old PGD endpoint path. It supports the idea that clean local SVD is not enough. It does not yet prove a core4/PQ dominant-mode story for GPI/replacement.

## What Was Newly Run Here

Post-processing, no GPU/model rerun:

- Script: `tools/probe_loss3_all_p2_tangent_geometry.py`
- Result: `docs/loss3_all_p2_tangent_geometry_probe_20260520.md`
- Output: `forensics/loss3_all_p2_tangent_geometry_probe_20260520/`

This extended the true p=2 tangent residual `sqrt(1 - cos(delta, grad)^2)` to `p2q1`, `p2q2`, and `p2qinf` from existing core4 CSVs.

Small GPU landscape pilot:

- Script: `tools/run_loss3_core4_pq_landscape_probe.py`
- Result: `docs/loss3_core4_pq_landscape_probe_20260520.md`
- Output: `forensics/loss3_core4_pq_landscape_probe_20260520/`
- Scope: baseline `eps=4, alpha=0.4`, sample indices `[0, 7, 40, 47]`, PQ settings `p2q2`, `p2q1`, `p2qinf`, `p1qinf`.
- Generated rows: ray `8000`, p=2 boundary arc `756`, 2D slice `1568`, finite-difference curvature `64`.

GPU path was verified before the run and recorded in the landscape probe manifest: Tesla V100-SXM2-32GB, compute capability `sm_70`, PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, PyTorch arch list includes `sm_70`, JAX backend `gpu`.

## What The New Evidence Says

Observed:

- In baseline `p2q2`, replacement/GPI early ray endpoint loss is already close to final by step 5 and essentially final-level by step 10-20.
- p2q2 boundary arcs between replacement and additive final directions stay high-loss, with minima around `98.8%` of the weaker endpoint. This supports a shared broad ridge / same high-loss region.
- p2q1 also looks broadly connected and replacement has a strong endpoint ray.
- p2qinf is qualitatively different: `steepest_add` wins endpoint ray among final directions, and replacement-to-raw/additive arcs have more noticeable dips.
- all-p2 tangent residual is high for replacement across q, but only `p2q1/p2q2` translate that cleanly into large post-boundary gain. `p2qinf` is the warning case where tangent opportunity is high but gain is weak.

Inference:

- We can reuse the older experiments as background evidence for nonlinear local-to-global behavior, local curvature changes, and Jacobian rotation.
- We cannot use the older experiments alone as proof of the current GPI/core4/PQ mechanism because they were PGD-path scoped.
- The new pilot directly supports the current mechanism for `p2q2` and partly `p2q1`: GPI/replacement rapidly reaches a final-like high-loss direction, and method final deltas are connected by a high-loss boundary ridge.
- The story must be qualified for `q=inf` and `p=1/q=inf`: aggressive boundary motion can exist without equally useful gain, and spike/concentration behavior remains a real caveat.

## Still Not Fully Done

Not yet run in this turn:

- A full local Jacobian/SVD dominant-mode recomputation at core4/GPI step 1/5/10/final for multiple PQs. Existing Jacobian/SVD evidence is old-PGD-path evidence, not current core4/PQ evidence.
- A larger-sample landscape sweep. The new landscape probe is a four-sample pilot, intentionally small.

Recommended next if paper-level proof is needed:

1. Run the same landscape probe on more samples and maybe two epsilon/alpha settings.
2. Then run the expensive local Jacobian/SVD dominant-mode probe only for selected representative cases: `p2q2` baseline, `p2qinf` warning case, and `p1qinf` spike case.
