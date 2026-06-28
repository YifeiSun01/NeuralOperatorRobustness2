# Burgers Standard Training vs Adversarial Training Runtime

Date: 2026-06-01

This note records the runtime comparison discussed for the 1D Burgers FNO model.
The main question was how long the ordinary Burgers training takes on the
1500-sample dataset, and how that compares with adversarial training.

## Standard Burgers FNO Training

Local run:

```text
fno_training_runs/burgers_no_profile_500_fair/burgers_nu0.001/burgers_1d/
```

Configuration:

```text
epochs: 500
train samples: 1350
test samples: 150
total dataset samples: 1500
batch size: 64
eval batch size: 128
modes: 16
width: 64
num layers: 4
nu: 0.001
learning rate: 0.001
weight decay: 0.0001
dtype: float32
```

Recorded PyTorch runtime:

```text
seconds_total: 82.03584703092929 sec
loss CSV seconds sum: 81.62666506832466 sec
```

Converted:

```text
500 epochs standard training: 82.04 sec = 1.37 min
1000 epochs standard training, linear estimate: 164.07 sec = 2.73 min
```

Source files:

```text
fno_training_runs/burgers_no_profile_500_fair/burgers_nu0.001/burgers_1d/config.json
fno_training_runs/burgers_no_profile_500_fair/burgers_nu0.001/burgers_1d/results.json
fno_training_runs/burgers_no_profile_500_fair/burgers_nu0.001/burgers_1d/losses_pytorch.csv
```

## Completed Linf Adversarial Training Reference

Completed run:

```text
adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/
```

Configuration summary:

```text
epochs: 1000
train samples: 1350
attack batch size: 480
optimizer batch size: 32
attack steps: 5
training data mode: adv-only
evaluation schedule: every epoch
evaluation pass count including baseline: 1001
global steps: 3000
optimizer steps: 43000
```

Recorded runtime:

```text
1000 epochs: 51139.143410257995 sec = 14.2053 h
500 epochs, linear estimate: 25569.571705128998 sec = 7.1027 h
```

This run used the old accidental `fast_replace_linf` geometry, so it is useful
as a timing reference but not the intended p=2,q=2 perturbation geometry.

Source file:

```text
adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/summary.json
```

## Corrected p=2,q=2 Adversarial Training Estimate

Prepared corrected pipeline:

```text
tools/run_burgers_p2q2_full_pipeline.py
tools/run_burgers_p2q2_full_pipeline.sh
```

Intended corrected configuration:

```text
attack method: fast_replace_l2
attack geometry: p=2,q=2 RMS-L2 replace direction
epochs: 1000
checkpoint every epochs: 200
train samples: 1350
attack batch size: 480
optimizer batch size: 32
attack steps: 5
epsilon fraction: 0.06
epsilon jitter: [0.75, 1.25]
alpha ratio: 1.0
alpha jitter: [0.75, 1.25]
training data mode: adv-only
label mode: solver
evaluation schedule: every epoch over train/test/50 generalization datasets
```

Smoke run:

```text
adversarial_training_runs/burgers_p2q2_smoke_20260601/
```

Smoke timing:

```text
1 attack batch elapsed: 18.367136884015054 sec
estimated per attack batch: 18.313872325001284 sec
attack part: 16.94885282800533 sec
optimizer update part: 0.5471019700635225 sec
peak CUDA allocated: 28284.599609375 MiB
```

Delta geometry check:

```text
attack_method: fast_replace_l2
attack_type: continuous_l2_rms
median rounded unique delta values: 1013
median top_abs_fraction: 0.0009765625
passed_l2_non_rectangular_check: true
```

Estimated corrected p=2,q=2 runtime:

```text
500 epochs: 27827.027604130573 sec = 7.7297 h
1000 epochs: 55653.34419206029 sec = 15.4593 h
```

Important epoch/step distinction:

```text
1 epoch = 3 attack batches, because 1350 samples / batch 480 -> 3 batches
1000 epochs = 3000 global steps
500 epochs = 1500 global steps
500 global steps = about 166.7 epochs
```

Therefore, if "500 steps" means literal `global_step = 500`, the estimate is:

```text
500 global steps / 3000 global steps * 15.4593 h = about 2.58 h
```

Source files:

```text
adversarial_training_runs/burgers_p2q2_smoke_20260601/burgers/summary.json
adversarial_training_runs/burgers_p2q2_smoke_20260601/burgers/l2_delta_geometry_check.json
adversarial_training_runs/burgers_p2q2_smoke_20260601/burgers/attack_batches.csv
```

## Runtime Comparison

Same 500 epochs:

```text
standard training: 82.04 sec = 1.37 min
corrected p2q2 adversarial training: 7.73 h = 463.78 min
ratio: about 339x slower
```

Same 1000 epochs:

```text
standard training estimate: 2.73 min
corrected p2q2 adversarial training estimate: 15.46 h = 927.56 min
ratio: about 339x slower
```

Literal 500 global adversarial steps compared to standard 500 epochs:

```text
500 global adversarial steps: about 2.58 h = 154.6 min
standard 500 epochs: 1.37 min
ratio: about 113x slower
```

## Interpretation

The ordinary Burgers FNO training is fast because each batch only needs model
forward/backward updates on fixed `(x, y)` pairs.

The adversarial training is much slower because each attack batch first runs a
solver-gradient adversarial attack.  With the current corrected p=2,q=2 setup,
each attack batch performs 5 attack steps, generates attacked samples, records
fixed probes and epsilon-bucket statistics, then runs optimizer updates.  The
attack dominates the cost; the optimizer update itself is small by comparison.

So the large time difference is expected:

```text
ordinary training: train the model on fixed samples
adversarial training: repeatedly solve a local worst-case perturbation problem,
then train the model on the attacked samples
```

