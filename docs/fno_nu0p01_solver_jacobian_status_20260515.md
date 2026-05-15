# FNO nu=0.01 vs Solver Local Jacobian/SVD Status (2026-05-15)

Status: completed.

## Source of the FNO nu=0.01 checkpoint

The old `tmp_old_runner_inputs_b01/...nu0.01...pth` file is not present in this
workspace or in the selected R2 copy. It is also ignored by `.gitignore` through
`*.pth`, so it was not tracked by Git.

The recovered checkpoint used here is the manifest-recorded FNO run:

- `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`
- restored from R2 prefix `machine-sync/NeuralOperatorRobustness2-selected/`
- config: `nu=0.01`, FNO1d modes 16, width 64, 4 layers, 500 epochs, seed 1234
- train/test split: `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/...nu0.01...`

## Local Jacobian analysis

Command class:

```bash
adv_robust/bin/python tools/analyze_fno_solver_jacobian_similarity.py   --out-root forensics/fno_nu0p01_solver_jacobian_similarity_20260515   --fno-checkpoint fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt   --fno-test-path 1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45_test.pt   --burgers-nu 0.01   --no-reuse-fno   --reuse-solver   --reuse-solver-root forensics/deeponet_solver_jacobian_similarity_20260515
```

Important control: `--no-reuse-fno` was required so the script did not reuse the
old FNO `nu=0.001` raw Jacobians. The solver Jacobians were reused from the
already computed `nu=0.01` DeepONet-vs-solver run, because the solver depends on
the input and `nu`, not on the model.

Sample indices: `0, 7, 40, 47, 115`.

Output root:

- `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`

## Aggregate result

| quantity | value |
|---|---:|
| FNO model spectral norm mean | 1.379 |
| solver spectral norm mean | 1.373 |
| error spectral norm mean | 0.0373 |
| FNO top-8 direction response cosine mean | 0.9993 |
| FNO top-8 mismatch gain mean | 0.0155 |
| error-direction response cosine mean | 0.3524 |
| FNO model-vs-solver k=1 mean principal angle | 2.32 deg |
| FNO model-vs-solver k=8 mean principal angle | 1.82 deg |
| FNO model top-1 hi128 | 2.82e-09 |
| FNO model top-1 zero crossings | 0.40 |
| FNO error top-1 hi128 | 0.1957 |
| FNO error top-1 zero crossings | 14.40 |

## Interpretation

For `nu=0.01`, FNO is even more tightly aligned with the solver than the earlier
`nu=0.001` FNO side: the model and solver spectral norms are almost equal, the
top subspace principal angles are only about 1 to 2 degrees, and the response
cosine in model top-8 directions is about `0.999`.

The residual/error Jacobian is much smaller in spectral norm (`0.0373`) than the
model/solver Jacobians (`~1.37`). Its dominant direction is not the same as the
model or solver leading direction, and it carries more high-frequency content
than the FNO model direction.
