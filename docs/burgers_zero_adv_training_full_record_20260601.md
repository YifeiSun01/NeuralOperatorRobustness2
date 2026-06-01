# Burgers Zero Adversarial Training Full Record - 2026-06-01

This file records the main decisions, data, conclusions, and artifact locations from the recent adversarial-training discussion. It intentionally does not contain API tokens, access keys, or GitHub credentials.

## 1. Current Main Run

The active formal run is Burgers, ADV-only, random epsilon / random alpha jitter, 5 attack steps, 1000 epochs:

```text
run name:
burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601

process:
PID 2978408

task:
burgers only

training mode:
ADV-only

epochs:
1000

checkpoint interval:
every 200 epochs

attack steps:
5

batch size:
burgers batch size = 480
optimizer microbatch size = 32

epsilon jitter:
0.75 to 1.25

alpha jitter:
0.75 to 1.25

epsilon buckets:
5 buckets

device:
cuda
```

At the last status check in this record:

```text
elapsed runtime: about 9 min 19 sec
latest recorded epoch: 11 / 1000
latest recorded global step: 33
GPU: Tesla V100-SXM2-32GB
GPU memory: about 30.7 GB / 32.8 GB
GPU utilization: about 39% at the instant checked
```

This run is using almost the whole V100 32GB memory budget, without OOM so far. Earlier 512 batch-size probing OOMed, so batch size 480 is the largest stable setting we found for this exact setup.

Output directory:

```text
adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/
```

Log file:

```text
adversarial_training_runs/logs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601.out
```

## 2. What This Run Records

This run was changed to match the requested detailed tracking:

```text
Every epoch:
  evaluate all 52 datasets:
    1 train dataset
    1 test dataset
    50 generalization datasets

Every epoch:
  record train/test/generalization RMSE
  record train/test/generalization relative L2
  record per-dataset metrics and split summaries

Every epoch:
  record attack clean loss before attack
  record adversarial loss after attack
  record attack loss gain
  record relative attack gain
  record epsilon statistics
  record alpha statistics
  record delta norms and boundary ratio

Every epoch:
  save 5 fixed probe samples after attack
  save x_clean, x_adv, delta, y_clean, y_adv

Every 200 epochs:
  save model checkpoint
```

Important output files:

```text
burgers/config.json
burgers/data_range_summary.json
burgers/train_steps.csv
burgers/optimizer_steps.csv
burgers/evaluation_passes.csv
burgers/eval_metrics.csv
burgers/eval_split_summary.csv
burgers/attack_epoch_summary.csv
burgers/attack_batches.csv
burgers/attack_epsilon_bucket_summary.csv
burgers/attack_probe_epochs.csv
burgers/attack_probe_samples.csv
burgers/attack_probe_samples/*.npz
burgers/memory.csv
planned_workload_estimate.json
preflight_dataset_counts.json
run_config.json
```

The NPZ attack-probe files contain:

```text
attack_global_step
delta
epoch
epoch_end_global_step
local_batch_idx
probe_rank
source_index
task
x_adv
x_clean
y_adv
y_clean
```

The five fixed probe source indices in the current run are:

```text
0, 337, 674, 1012, 1349
```

These are fixed across epochs, so the comparison is same input point across training time. That is the right setup for checking whether the same initial condition becomes harder to attack as the model changes.

## 3. Evaluation Design

The run now does not wait 100 epochs between evaluations. It evaluates every epoch.

For each epoch it records:

```text
train:
  1 dataset, 1350 samples

test:
  1 dataset, 150 samples

generalization:
  50 datasets, 10000 total samples

ALL:
  52 datasets, 11500 total samples
```

So the final run should give 1000-point curves for each of the 52 datasets. This is exactly what is needed to distinguish true training trends from the earlier sparse 0%, 20%, 40%, 60%, 80%, 100% checkpoint artifacts.

At epoch 10, the split-level evaluation snapshot was:

```text
ALL:
  RMSE mean        0.0144931
  relative L2 mean 0.0252723

train:
  RMSE             0.0123452
  relative L2      0.0232196

test:
  RMSE             0.0127790
  relative L2      0.0237741

generalization:
  RMSE mean        0.0145704
  relative L2 mean 0.0253433
```

These early numbers are not final conclusions. They are only a sanity check that per-epoch evaluation is writing correctly.

## 4. Epsilon Five-Bucket Statistics

The current run records 5 epsilon buckets using the epsilon jitter factor:

```text
bucket 0: 0.75 to 0.85
bucket 1: 0.85 to 0.95
bucket 2: 0.95 to 1.05
bucket 3: 1.05 to 1.15
bucket 4: 1.15 to 1.25
```

For each epoch and bucket, the code records:

```text
sample_count
sample_fraction
epsilon mean/std/min/max
epsilon_jitter_factor mean/std/min/max
alpha mean/std/min/max
clean_loss_before_attack mean/std/min/max
adv_loss_after_attack mean/std/min/max
attack_loss_gain mean/std/min/max
loss_increase mean/std/min/max
relative gain mean/std/min/max
```

The reason for this bucket design is that random epsilon mixes small, medium, and large perturbation radii. A single average attack loss is still meaningful, but it means:

```text
expected attack loss under the epsilon distribution
```

It does not by itself answer what happens at a fixed epsilon. The five buckets let us ask:

```text
small epsilon: does local robustness improve?
medium epsilon: does ordinary robustness improve?
large epsilon: does broader distribution robustness improve?
```

This is the right structure for the user's hypothesis:

```text
as adversarial training proceeds, attack should become harder,
and attack gain should shrink, possibly with later perturbations
becoming more high-frequency or more complicated.
```

## 5. Attack Step Delta Comparison

A separate fixed-sample experiment compared attack steps:

```text
1, 2, 3, 5, 10
```

Directory:

```text
forensics/burgers_attack_step_delta_comparison_20260601/
```

Files:

```text
README.md
manifest.json
summary_vs_step10.csv
per_sample_vs_step10.csv
pairwise_step_similarity.csv
```

Setup:

```text
model: baseline Burgers model
device: CPU, to avoid interrupting the GPU training
attack method: fast_replace_linf
same 8 fixed train samples
same seed
same epsilon jitter values across step counts
random start: 0
reference: step 10
```

Fixed source indices:

```text
39, 341, 557, 796, 924, 1015, 1277, 1296
```

Summary versus step 10:

| steps | adv loss mean | gain mean | relative L2 diff vs step10 | cosine vs step10 | sign agreement vs step10 | high freq ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0003509740 | 0.0002763172 | 1.0971 | 0.3889 | 0.6945 | 0.03688 |
| 2 | 0.0002757090 | 0.0002010522 | 0.8309 | 0.6487 | 0.8243 | 0.03639 |
| 3 | 0.0003097637 | 0.0002351069 | 1.1123 | 0.3672 | 0.6836 | not primary |
| 5 | 0.0003051852 | 0.0002305284 | 1.1314 | 0.3518 | 0.6759 | 0.03424 |
| 10 | 0.0002744939 | 0.0001998370 | reference | reference | reference | reference |

Important interpretation:

```text
For fast_replace_linf, more attack steps do not monotonically increase attack loss.
```

This is because the method repeatedly replaces the perturbation by:

```text
delta = epsilon * sign(gradient at current x_adv)
```

It is not additive PGD accumulating small steps. The direction can jump or oscillate. Therefore step 5 is not guaranteed to look like half of step 10, and step 10 is not guaranteed to have the largest loss on a tiny sample.

Even so, step 5 was chosen for the formal run because it is a stronger and less trivial attack than 1 or 2 steps, while being much faster than 10 steps.

## 6. Runtime and Batch-Size Findings

Batch-size sweep on V100 32GB:

| Burgers batch size | status | approximate memory | interpretation |
|---:|---|---:|---|
| 256 | worked | about 15.8 GB | too conservative |
| 320 | worked | about 20.3 GB | stable |
| 384 | worked | about 23.6 GB | stable |
| 448 | worked | about 27.8 GB | stable |
| 480 | worked | about 30.4 to 30.7 GB | chosen |
| 512 | OOM | over limit | too large |

The current batch size 480 is therefore already close to the largest practical value on this GPU. Increasing it further is not realistic without OOM. Also, for 1350 train samples:

```text
batch size 480 gives 3 attack batches per epoch
```

So any larger batch below 512 would not reduce the number of attack batches per epoch much. It would only slightly change kernel efficiency.

Speed sweep:

| configuration | estimated sec / epoch | estimated 1000-epoch runtime | note |
|---|---:|---:|---|
| steps=1, opt=32 | about 18.46 sec | about 5.13 h | too weak as main attack |
| steps=2, opt=32 | about 26.06 sec | about 7.24 h | faster, but user switched to steps=5 |
| steps=2, opt=128 | about 25.64 sec | about 7.12 h | fewer optimizer updates, less comparable |
| steps=3, opt=32 | about 35.00 sec | about 9.72 h | middle |
| steps=3, opt=128 | about 32.99 sec | about 9.16 h | fewer optimizer updates |
| steps=5, opt=32 | about 51.2 sec early estimate | about 14.2 h | current formal run |

Current step=5 early epoch attack timing:

```text
epoch 1 attack: 47.93 sec
epoch 2 attack: 46.38 sec
epoch 3 attack: 46.20 sec
epoch 4 attack: 50.56 sec
epoch 5 attack: 47.90 sec
epoch 6 attack: 48.22 sec
epoch 7 attack: 50.80 sec
epoch 8 attack: 49.21 sec
epoch 9 attack: 47.00 sec
epoch 10 attack: 46.71 sec
```

The estimate of about 14 hours is consistent with these early timings. The per-epoch full evaluation costs only about 0.7 sec, so per-epoch evaluation is not the runtime bottleneck. The bottleneck is adversarial attack generation.

## 7. Why ADV-Only Is the Current Main Choice

The current main recommendation is:

```text
ADV-only + random epsilon + random alpha
```

Reason:

```text
The user's main target is to show that adversarial training makes the model
locally less vulnerable and makes the error Jacobian closer to the solver.
```

In the 20 same-point Jacobian comparison:

```text
J_error(x) = J_model(x) - J_solver(x)
```

The result was:

| split | points | baseline mean | ADV-only mean | clean+ADV mean | ADV-only improved | clean+ADV improved |
|---|---:|---:|---:|---:|---:|---:|
| train | 6 | 1.5080 | 0.4898 | 0.7549 | 6/6 | 5/6 |
| test | 4 | 0.7690 | 0.4383 | 0.7058 | 4/4 | 2/4 |
| generalization | 10 | 3.5431 | 1.4922 | 1.7005 | 10/10 | 9/10 |
| ALL | 20 | 2.3778 | 0.9807 | 1.2179 | 20/20 | 16/20 |

With standard deviations:

```text
ALL baseline J_error spectral norm:
  2.3778 +/- 1.8595

ALL ADV-only J_error spectral norm:
  0.9807 +/- 0.7248

ALL clean+ADV J_error spectral norm:
  1.2179 +/- 0.7931
```

So ADV-only gave the cleanest evidence for robust error-Jacobian contraction:

```text
ADV-only improved J_error on 20/20 same input points.
clean+ADV improved J_error on 16/20 same input points.
```

Clean+ADV may still be useful as a control group or as a clean-accuracy/robustness compromise, but it mixes two objectives:

```text
clean distribution fitting
adversarial distribution fitting
```

That makes it less clean for proving the robustness/J_error contraction story.

## 8. Random Epsilon vs Fixed Epsilon

The reason to keep epsilon random is valid:

```text
fixed epsilon:
  trains mostly one attack radius

random epsilon:
  trains a distribution of attack radii
```

Interpretation:

```text
small epsilon:
  local smoothness and local robustness

medium epsilon:
  ordinary robustness

large epsilon:
  farther perturbations and broader generalization pressure
```

Random epsilon makes the average loss mean:

```text
E_epsilon[loss_after_attack(epsilon)]
```

That average is meaningful, but it should be interpreted as an expectation over the epsilon distribution, not as a fixed-epsilon curve. That is why the new five-bucket logging is important.

## 9. Why Evaluation Loss Was Not Monotonic

The earlier 0%, 20%, 40%, 60%, 80%, 100% plots showed loss going down, then up, then down again. That was not necessarily a plotting bug.

Main reasons:

```text
1. The training objective is not a fixed supervised dataset.
   Every epoch the attack is regenerated using the current model.

2. The attacked samples change as the model changes.
   The target being optimized moves.

3. Random epsilon, random alpha, and attack randomness create noise.

4. Clean+ADV mixes clean samples and adversarial samples.
   That can create oscillation between clean accuracy and robust behavior.

5. Earlier evaluation was sparse.
   Looking only every 100 epochs can make random or oscillatory behavior look like a strange sawtooth.
```

The new run evaluates every epoch, so it should reveal whether the oscillation is real, noisy, periodic, or checkpoint-specific.

## 10. Attack Gain and Optimizer Update Loss

The correct attack-strength quantities are:

```text
clean_loss_before_attack
adv_loss_after_attack
attack_loss_gain = adv_loss_after_attack - clean_loss_before_attack
attack_loss_gain_relative = attack_loss_gain / clean_loss_before_attack
```

The quantity formerly labeled like optimizer update MSE is different. It is the microbatch loss during optimizer updates:

```text
microbatch 1:
  loss(model_0, x_adv_micro_1) -> backward -> optimizer.step

microbatch 2:
  loss(model_1, x_adv_micro_2) -> backward -> optimizer.step

microbatch 3:
  loss(model_2, x_adv_micro_3) -> backward -> optimizer.step
```

Therefore it is not the same as:

```text
loss(model_old, full_attacked_batch)
```

This explains why:

```text
adv_loss_after_attack
```

can be much larger than:

```text
microbatch loss during optimizer updates
```

They are not computed at the same model parameters, and sometimes not over exactly the same sample mix. In clean+ADV the optimizer loss also mixes clean and attacked samples, which makes the comparison even more misleading.

For studying attack strength, use:

```text
clean_loss_before_attack
adv_loss_after_attack
attack_loss_gain
attack_loss_gain_relative
epsilon-bucket attack gain
```

## 11. Jacobian / Frechet Derivative Interpretation

The StablePDENet-style analysis is about the neural operator:

```text
G_theta: a -> u
```

where:

```text
a = input function, for Burgers usually the initial condition
u = output solution function
```

The Frechet derivative:

```text
DG_theta(a)
```

describes:

```text
if the input function changes by delta a,
how much does the output solution function change?
```

After discretization:

```text
input a is an m-dimensional vector
output u is an n-dimensional vector

G_theta: R^m -> R^n

J_theta(a) = dG_theta(a) / da
```

So:

```text
G_theta(a + delta a)
approximately equals
G_theta(a) + J_theta(a) delta a
```

The spectral norm:

```text
||J_theta(a)||_2 = sigma_max(J_theta(a))
```

means:

```text
the maximum possible output amplification over all unit input perturbations
```

For our comparison, the most important object is not only:

```text
J_model
```

but:

```text
J_error = J_model - J_solver
```

because we do not want the model to become a dead flat model. We want:

```text
J_model approximately equals J_solver
```

That is why the result that `J_error` shrinks is stronger than merely saying `J_model` shrinks.

## 12. SVD Meaning

For a Jacobian matrix:

```text
J = U Sigma V^T
```

The key identity is:

```text
J v_i = sigma_i u_i
```

Meaning:

```text
v_i:
  right singular vector
  input perturbation direction
  for Burgers, an initial-condition perturbation pattern

u_i:
  left singular vector
  output response direction
  the solution-response pattern caused by v_i

sigma_i:
  singular value
  amplification factor from v_i to u_i
```

So:

```text
v_1:
  most dangerous input perturbation direction

u_1:
  main output response caused by that dangerous direction

sigma_1:
  maximum amplification
```

To say the trained model becomes more like the solver locally, we need three kinds of evidence:

```text
1. singular values closer to solver
2. right singular vectors closer to solver
3. left singular vectors closer to solver
```

When comparing singular vectors, sign does not matter:

```text
v and -v represent the same direction
```

So similarity should use:

```text
abs(dot(v_model, v_solver))
abs(dot(u_model, u_solver))
```

If several singular values are close to each other, individual vectors can rotate. Then the more stable comparison is top-k subspace similarity rather than only vector-by-vector top-1 similarity.

## 13. Same-Point Jacobian / SVD Evidence

Representative same-point experiment:

```text
20 fixed input points:
  6 train
  4 test
  10 generalization

For every same input x:
  compute J_solver(x)
  compute J_baseline_model(x)
  compute J_adv_only_model(x)
  compute J_clean_plus_adv_model(x)

Then compare:
  J_model
  J_model - J_solver
  singular values
  right singular vectors
  left singular vectors
  Fourier-band behavior
```

Directory:

```text
forensics/burgers_adv_training_jacobian_svd_20260531_representative20_same_points/
```

Main summary file:

```text
same_point_full_jacobian_similarity_summary.md
```

Key conclusion:

```text
ADV-only does not simply make the model Jacobian collapse to zero.
Instead, the model Jacobian stays on roughly the solver scale,
while the error Jacobian J_model - J_solver becomes much smaller.
```

Model Jacobian norm scale:

```text
solver:
  5.9035 +/- 1.8238

baseline model:
  5.2886 +/- 1.2904

ADV-only model:
  5.6696 +/- 1.5668

clean+ADV model:
  5.7426 +/- 1.6570
```

This supports the important interpretation:

```text
the trained model is not just flatter;
it is locally more solver-like.
```

Top-20 singular value relative RMSE versus solver:

```text
baseline:
  0.1057

ADV-only:
  0.0412

clean+ADV:
  0.0267
```

Right top-1 singular vector abs dot versus solver:

```text
baseline:
  0.9047 +/- 0.2568

ADV-only:
  0.9633 +/- 0.1575

clean+ADV:
  0.9932 +/- 0.0264
```

Right top-8 subspace mean cosine:

```text
baseline:
  0.9582

ADV-only:
  0.9905

clean+ADV:
  0.9897
```

Left top-1 singular vector abs dot:

```text
baseline:
  0.8443

ADV-only:
  0.9520

clean+ADV:
  0.9752
```

Fourier high-band relative RMSE:

```text
baseline:
  11.3704

ADV-only:
  1.5750

clean+ADV:
  1.6560
```

Interpretation:

```text
ADV-only is best for consistent J_error contraction.
clean+ADV is sometimes even better for singular-value/vector alignment,
but it is less consistent point-by-point on J_error.
```

## 14. What Fourier Gain Means

The Fourier-gain plots are not another training loss.

They are a frequency-domain diagnostic:

```text
Take a singular vector or Jacobian response.
Apply Fourier transform.
Measure how much energy or amplification appears in low, middle, or high frequencies.
```

For Burgers, this answers questions like:

```text
Does the most dangerous input direction look low-frequency or high-frequency?
Does the trained model react to high-frequency perturbations like the solver?
Does adversarial training reduce the mismatch in high-frequency response?
```

So the Fourier-gain result is part of the same "is the model locally more solver-like?" question. It helps distinguish:

```text
model became flat everywhere
```

from:

```text
model learned solver-like frequency response
```

## 15. Darcy and NS2D Status From This Discussion

Earlier Darcy full10 formal runs did not finish:

```text
full10_darcy_adv_only_20260530:
  failed

full10_darcy_clean_plus_adv_20260530:
  failed

reason:
  CUDA out of memory

old setting:
  darcy batch size 448
  optimizer batch size 32
  attack steps 10
```

Darcy is 2D, but it is a steady elliptic solve with binary coefficient-flip style attack in this code path. It is not a long time-rollout like Burgers or NS2D. That is why "2D" does not automatically mean slower than Burgers. Runtime depends on solver and attack structure, not only dimension.

NS2D had been started after the Darcy OOM queue progression, but then the user requested stopping background jobs and focusing on Burgers/Jacobian and the new Burgers zero adversarial training.

Current formal run in this record is:

```text
Burgers ADV-only step=5
```

not Darcy and not NS2D.

## 16. Visualization Terms Previously Discussed

Some plotting terms that caused confusion:

```text
sawtooth:
  jagged up-down pattern, like loss decreases then increases then decreases again

positive sawtooth:
  the upward parts of that jagged pattern

offset/base:
  the baseline portion of a stacked or change plot

scale + offset:
  rescaling a plotted quantity and then shifting it for visual separation

hatch:
  diagonal or patterned fill inside a bar/region

hatch transparent / hatch opaque:
  whether that patterned fill is faint or solid
```

For future plots, the requested style change was:

```text
base part darker
change part lighter
black arrows made lighter / semi-transparent
shared y-axis range across panels
```

## 17. Files Already Written or Relevant

Previously pushed docs:

```text
docs/adversarial_training_detailed_logging_20260531.md
docs/stablepdenet_jacobian_frechet_derivative_notes_20260531.md
docs/burgers_zero_adv_training_jacobian_svd_summary_20260531.md
```

Important pushed code:

```text
tools/adversarial_training.py
```

Important code changes:

```text
per-epoch evaluation support
attack probe sample logging
fixed probe indices
x_clean/x_adv/delta/y_clean/y_adv NPZ saving
checkpoint every N epochs
random epsilon/alpha jitter controls
five-bucket epsilon statistics
attack loss gain and relative gain logging
```

New data/forensics directory for attack-step delta comparison:

```text
forensics/burgers_attack_step_delta_comparison_20260601/
```

Current live-run data directory:

```text
adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/
```

## 18. GitHub / R2 Sync Policy

GitHub should receive:

```text
source code
shell scripts
Markdown documentation
small CSV/JSON summaries
small forensic tables
```

R2 should receive:

```text
larger generated artifacts
training run CSV/JSON snapshots
attack probe NPZ files
plots
model checkpoints
full run output directories
```

For this record, credentials are deliberately excluded from Markdown. Uploads should use the configured external secrets only in the shell environment or temporary upload config, not in committed files.

## 19. Main Conclusions

The current best experimental path is:

```text
Burgers ADV-only
random epsilon / random alpha
5 attack steps
batch size 480
1000 epochs
checkpoint every 200 epochs
evaluate every epoch on all 52 datasets
save fixed attack probes every epoch
record five epsilon buckets
```

The scientific reason is:

```text
ADV-only gave the clearest J_error contraction evidence:
20/20 same input points improved versus baseline.
```

The computational reason is:

```text
batch size 480 is near the V100 32GB limit.
step=5 is much faster than step=10 while still nontrivial.
per-epoch evaluation adds little overhead.
```

The analysis target is:

```text
not just "loss goes down",
but whether adversarial training makes attack gain shrink,
whether fixed input perturbations become harder,
whether delta becomes more high-frequency/complex,
and whether J_model becomes locally more like J_solver.
```

