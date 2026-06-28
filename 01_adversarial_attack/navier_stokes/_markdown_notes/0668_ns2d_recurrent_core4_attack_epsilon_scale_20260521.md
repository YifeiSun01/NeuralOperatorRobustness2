# 2D NS Recurrent Core4 Attack Epsilon Scale

Observed from source and CPU-only dataset statistics on 2026-05-21:

- Source file: `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`.
- The attack uses batchwise raw tensor norms. For `p=2`, `epsilon` is an L2 budget over the full `256x256` initial condition for each sample. It is not per-pixel and is not divided by `256` or `65536`.
- Dataset source: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`.
- CPU-only statistics for test initial conditions `y[..., 0]`, shape `(50, 256, 256)`:
  - Mean L2 norm: `66.26712036132812`.
  - Mean per-pixel RMS: `0.25885701179504395`.
  - Mean L-infinity norm: `0.7032134532928467`.
- Conversion for `256x256`: per-pixel RMS budget is `epsilon / 256`.
- Therefore `epsilon=32768` means RMS budget `128.0`, about `494.48x` the mean initial-condition L2 norm.
- `epsilon=3276825` means RMS budget `12800.09765625`, about `49448.73x` the mean initial-condition L2 norm.

Inference:

- If the intended perturbation is in the raw dataset scale, `32768` is extremely large for this code path.
- Reasonable first raw-L2 sweeps are closer to `epsilon in {8, 16, 32, 64}`. These correspond to about `12%`, `24%`, `48%`, and `97%` of the mean initial-condition L2 norm.
- For additive p=2 steepest methods, a practical first step size is `alpha ~= epsilon / 20` to `epsilon / 10`. For the shared four-method run including `raw_add`, start more conservatively with `alpha=1` when `epsilon=32`.
