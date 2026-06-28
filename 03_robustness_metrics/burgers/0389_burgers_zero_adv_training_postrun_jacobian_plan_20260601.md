# Burgers Zero Adversarial Training Post-Run Jacobian/SVD Plan

This note records the required post-training Jacobian/SVD analysis for the current Burgers zero adversarial training run.

## Required Timing

Run this after the current 1000-epoch Burgers adversarial training finishes and the saved checkpoints are available.

Current run:

```text
burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601
```

Expected checkpoints:

```text
epoch 200
epoch 400
epoch 600
epoch 800
epoch 1000
```

Also include the original baseline model if the goal is to compare before/after adversarial training.

## Fixed Sample Rule

Use a fixed sample manifest. Do not resample separately for each model.

The same input initial conditions must be used for every checkpoint/model:

```text
same x_1 for baseline, epoch200, epoch400, epoch600, epoch800, epoch1000
same x_2 for baseline, epoch200, epoch400, epoch600, epoch800, epoch1000
...
same x_10 for baseline, epoch200, epoch400, epoch600, epoch800, epoch1000
```

This is required because the comparison is local:

```text
J_model(x)
J_solver(x)
J_error(x) = J_model(x) - J_solver(x)
```

If `x` changes between models, the Jacobian/SVD comparison is not clean.

## Sample Count

Use 10 total samples, not 20.

Recommended split:

```text
train: 2 samples
generalization: 8 samples
test: optional, only if explicitly requested later
```

The user preference is:

```text
take one or two from training,
take the rest from generalization.
```

The manifest must record:

```text
sample_id
source_split
dataset_id
dataset_path
local_index
selection_reason
```

## What To Compute For Every Fixed Sample

For each fixed input `x`, compute:

```text
J_solver(x)
J_model_baseline(x)
J_model_epoch200(x)
J_model_epoch400(x)
J_model_epoch600(x)
J_model_epoch800(x)
J_model_epoch1000(x)
```

Then compute the error Jacobians:

```text
J_error_baseline(x) = J_model_baseline(x) - J_solver(x)
J_error_epoch200(x) = J_model_epoch200(x) - J_solver(x)
J_error_epoch400(x) = J_model_epoch400(x) - J_solver(x)
J_error_epoch600(x) = J_model_epoch600(x) - J_solver(x)
J_error_epoch800(x) = J_model_epoch800(x) - J_solver(x)
J_error_epoch1000(x) = J_model_epoch1000(x) - J_solver(x)
```

## What To Save

For every sample and every object above, save the full SVD NPZ:

```text
jacobian
singular_values
left_singular_vectors
right_singular_vectors
```

Also save summary CSV files for:

```text
spectral norm
top-20 singular values
top-k singular value similarity versus solver
top-k right singular vector similarity versus solver
top-k left singular vector similarity versus solver
top-k subspace similarity versus solver
Fourier-band gain similarity versus solver
```

## Main Question

The post-run analysis should answer:

```text
As adversarial training proceeds from baseline to epoch1000,
does J_error(x) shrink on the same fixed initial conditions?
```

More specifically:

```text
1. Does ||J_error||_2 decrease checkpoint by checkpoint?
2. Does J_model remain on the solver scale, rather than collapse to zero?
3. Do singular values of J_model become closer to J_solver?
4. Do right singular vectors become closer to solver right singular vectors?
5. Do left singular vectors become closer to solver left singular vectors?
6. Does the high-frequency Fourier-gain mismatch shrink?
```

The comparison must always be same fixed input point, different model.

