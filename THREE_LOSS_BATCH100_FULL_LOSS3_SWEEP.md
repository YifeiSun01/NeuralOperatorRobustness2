# Three-Loss Batch-100 Full-Loss3 Sweep

This note documents the shell script

```text
tools/run_three_loss_batch100_full_loss3_sweep.sh
```

The script runs a batch-100 attack sweep for the three loss objectives, then
automatically regenerates the loss-curve visualizations after each parameter
setting.

## How To Run

From the repository root:

```bash
cd /workspace/NeuralOperatorRobustness2
bash tools/run_three_loss_batch100_full_loss3_sweep.sh
```

The script does not randomly choose samples. It always uses the fixed batch

\[
\text{dataset indices } 0,1,\ldots,99.
\]

This is controlled by

```bash
BATCH_SIZE=100
START_INDEX=0
```

## Main Experiment Rules

The script runs the corrected full-loss3 version.

For loss1:

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_q.
\]

Loss1 uses one tiny nonzero random initial perturbation:

\[
\delta_0\neq 0.
\]

The same random initial \(\delta_0\) is shared by all loss1 variants and all
three optimization methods. The zero-initialized loss1 runs are intentionally
not run, because loss1 gets zero gradient at \(\delta=0\) in PyTorch/autograd.

For loss2:

\[
L_2(\delta)=\|f(x+\delta)-g(x)\|_q.
\]

Here \(g(x)\) is computed once at the original input and then kept fixed. The
solver is not recomputed per step for loss2, and the solver gradient is not
used.

For loss3:

\[
L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|_q.
\]

This is the full solver-gradient version. During loss3 optimization, the solver
output \(g(x+\delta)\) is differentiated, so the local gradient contains the
effect of

\[
J_f-J_g.
\]

## Optimization Methods

For each optimized loss and objective variant, the script runs three methods:

```text
pgd
lp_steepest_pgd
generalized_power
```

The optimized objective can be one of:

```text
original
increment_ratio
regularized
```

So each parameter setting runs:

```text
3 losses x 3 objective variants x 3 methods = 27 attack tags
```

Because loss1 zero-initialized runs are not included, there are no extra loss1
zero-start tags.

## Parameter Sweep

The script runs these 8 settings:

```text
eps8_alpha0p3    epsilon=8    alpha=0.3
eps8_alpha1p5    epsilon=8    alpha=1.5
eps8_alpha3p0    epsilon=8    alpha=3.0
eps16_alpha0p3   epsilon=16   alpha=0.3
eps40_alpha0p3   epsilon=40   alpha=0.3
eps16_alpha1p5   epsilon=16   alpha=1.5
eps4_alpha0p3    epsilon=4    alpha=0.3
eps1p6_alpha0p3  epsilon=1.6  alpha=0.3
```

The default shared parameters are:

```text
batch size: 100
start index: 0
steps: 100
p: 2
q: 2
eta: 1e-6
regularization C: 1.0
seed: 0
loss1 random-start scale: 1e-6
model: FNO Burgers checkpoint
solver: JAX Burgers solver
```

## Scripts Used

The shell script calls:

```text
tools/run_and_plot_batch_three_loss_loss_only.py
```

That wrapper first calls:

```text
tools/run_batch_three_loss_loss_only.py
```

to run the attacks and save the numeric loss data.

Then it calls:

```text
tools/plot_batch_three_loss_loss_only.py
```

to generate the batch mean/std loss-curve figures.

Finally it calls:

```text
tools/plot_batch_single_index_loss_curves.py
```

to generate the dataset-index-0 single-sample loss-curve figures.

## Output Directories

For each parameter setting, outputs are written under

```text
results/three_loss_batch100_full_loss3_<setting>_final_boundary
```

For example:

```text
results/three_loss_batch100_full_loss3_eps8_alpha0p3_final_boundary
```

Each attack tag has its own subdirectory, for example:

```text
loss3_original_pgd
loss3_increment_ratio_lp_steepest_pgd
loss1_regularized_generalized_power
```

## Per-Step Loss Data

Each tag stores per-step loss data in:

```text
loss_stats.csv
loss_values.npz
summary.json
```

The per-step data records all nine evaluated objectives:

```text
loss1_original
loss1_increment_ratio
loss1_regularized
loss2_original
loss2_increment_ratio
loss2_regularized
loss3_original
loss3_increment_ratio
loss3_regularized
```

For batch curves, the plotted solid line is the mean over the 100 samples, and
the translucent band is plus/minus one standard deviation.

## Final-Only Boundary Diagnostics

For each tag, the code also saves final-only diagnostics:

```text
final_delta_diagnostics.csv
final_delta_diagnostics.npz
final_delta_summary.json
```

These files are saved only for the final attack step.

They contain:

```text
final_delta_pnorm
boundary_delta_pnorm
boundary_rescale_factor
final_<loss/objective>
boundary_<loss/objective>
```

The boundary values are computed by taking the final perturbation direction and
rescaling it to the epsilon boundary:

\[
\delta_{\mathrm{bdry}}
=
\epsilon
\frac{\delta_{\mathrm{final}}}{\|\delta_{\mathrm{final}}\|_p}.
\]

Then all nine objectives are recomputed at

\[
x+\delta_{\mathrm{bdry}}.
\]

This lets us compare the actual final perturbation with the same direction
extended to the full perturbation budget.

## Visualization Contents

The batch mean/std figures are saved under:

```text
figures/loss_curves/png
```

The dataset-index-0 figures are saved under:

```text
figures/loss_curves/index_png
```

Each figure title includes the experiment parameters:

```text
model
solver
batch size
dataset index range
steps
p, q
epsilon
alpha
eta
regularization C
loss1 initial-delta mode
```

Each subplot title also reports:

```text
final ||delta||_p
current original loss L_i at the final delta
boundary original loss L_i after rescaling delta to ||delta||_p = epsilon
```

For mean/std figures, these values are batch means. For index-0 figures, these
values are for dataset index 0 only.

## Runtime Notes

Earlier batch-100 runs on the A100 used about 2.5 GB of GPU memory and roughly
95%-97% GPU utilization. The corrected full-loss3 version may use slightly more
time and memory because loss3 now backpropagates through the JAX solver.

The shell script writes one log file per parameter setting under:

```text
logs/three_loss_batch100_full_loss3_sweep
```

The script runs settings sequentially, not in parallel.
