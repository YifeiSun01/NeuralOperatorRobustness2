# Burgers adversarial training, Jacobian/SVD, and zero-training run summary

Date: 2026-05-31

This note records the main conclusions from the Burgers adversarial-training discussion and the follow-up Jacobian/SVD analysis. It intentionally does not record Cloudflare, GitHub, or other credential values.

## 1. What object the Jacobian means here

For the Burgers neural operator, the input is an input function, usually the initial condition or source-like field:

```text
a
```

The output is the solution function:

```text
u
```

The neural operator is therefore:

```text
G_theta: a -> u
```

After discretization, this becomes a finite-dimensional map:

```text
G_theta: R^m -> R^n
```

The local derivative at one fixed input `a` is the Jacobian:

```text
J_model(a) = d G_theta(a) / d a
```

For the numerical PDE solver, the analogous object is:

```text
J_solver(a) = d Solver(a) / d a
```

The key mismatch object we have been using is:

```text
J_error(a) = J_model(a) - J_solver(a)
```

This is not the prediction error itself. It is the local derivative error: how differently the model and the true solver respond to a small input-function perturbation around exactly the same input point.

## 2. SVD meaning: right singular vectors, left singular vectors, singular values

For any Jacobian matrix:

```text
J = U Sigma V^T
```

The defining relation is:

```text
J v_i = sigma_i u_i
```

In our setting:

| SVD object | Meaning in the Jacobian | Physical interpretation |
|---|---|---|
| `v_i` | right singular vector | input perturbation direction |
| `u_i` | left singular vector | output response direction |
| `sigma_i` | singular value | amplification factor |

So:

```text
right singular vector v_i:
which shape of perturbation in the initial condition / input field is being tested

left singular vector u_i:
what shape of output response that input perturbation causes

singular value sigma_i:
how much that perturbation is amplified
```

The top singular triplet is the most important one:

```text
v_1:
most amplified input perturbation direction

u_1:
dominant output response caused by that perturbation

sigma_1:
maximum local amplification factor, equal to the spectral norm
```

The standard SVD interpretation is therefore exactly what we are using:

```text
right singular vector = input direction
left singular vector  = output direction
singular value        = gain / amplification
```

Two technical cautions matter:

1. The sign of a singular vector is arbitrary. `v` and `-v` represent the same direction, so comparisons should use absolute dot products such as `abs(dot(v_model, v_solver))`.
2. If several singular values are close, an individual singular vector can rotate inside the nearly degenerate subspace. In that case top-k subspace alignment is more reliable than only top-1 vector alignment.

Reference sources used for the interpretation:

- Stanford EE263 SVD notes: https://ee263.stanford.edu/lectures/svd.pdf
- R/MIT SVD documentation, especially the sign-indeterminacy point: https://web.mit.edu/r/current/lib/R/library/base/html/svd.html

## 3. What it means for the model to become more solver-like

The desired outcome is not simply:

```text
J_model becomes small
```

That could mean the model is becoming insensitive or overly flat. The desired outcome is:

```text
J_model ~= J_solver
```

Equivalently:

```text
J_error = J_model - J_solver becomes small
```

To support the claim that adversarial training makes the model more solver-like locally, we need evidence at several levels:

| Evidence type | What it checks |
|---|---|
| `||J_error||_2` decreases | model-solver derivative mismatch becomes smaller |
| singular values become closer | the model amplifies input perturbations by similar amounts as the solver |
| right singular vectors become closer | the model identifies similar dangerous input directions as the solver |
| left singular vectors become closer | the model produces similar output-response directions as the solver |
| Fourier gain becomes closer | the model responds to frequency-mode perturbations more like the solver |

## 4. Same-point Burgers Jacobian experiment

The corrected comparison used the same fixed input point for all models. It did not treat `ADV-only` or `clean+ADV` as input points. For each fixed sample `x`, the comparison was:

```text
same x:
  solver
  baseline model
  ADV-only trained model
  clean+ADV trained model
```

For every fixed `x`, we computed:

```text
J_solver(x)
J_baseline_model(x)
J_adv_only_model(x)
J_clean_plus_adv_model(x)
J_baseline_error(x)       = J_baseline_model(x) - J_solver(x)
J_adv_only_error(x)       = J_adv_only_model(x) - J_solver(x)
J_clean_plus_adv_error(x) = J_clean_plus_adv_model(x) - J_solver(x)
```

So per sample there are seven Jacobian-level objects:

```text
1. solver
2. baseline model
3. ADV-only model
4. clean+ADV model
5. baseline error
6. ADV-only error
7. clean+ADV error
```

The representative sample mix was:

```text
6 train samples
4 test samples
10 generalization samples
20 total fixed points
```

The important condition is that model comparisons are same-point comparisons:

```text
same x, different models
```

This is the correct way to interpret local Jacobian behavior.

## 5. Main numerical result: error Jacobian contraction

The spectral norm of:

```text
J_error = J_model - J_solver
```

contracted strongly after adversarial training.

| split | points | baseline error mean | ADV-only error mean | clean+ADV error mean | ADV-only lower than baseline | clean+ADV lower than baseline |
|---|---:|---:|---:|---:|---:|---:|
| train | 6 | 1.5080 | 0.4898 | 0.7549 | 6/6 | 5/6 |
| test | 4 | 0.7690 | 0.4383 | 0.7058 | 4/4 | 2/4 |
| generalization | 10 | 3.5431 | 1.4922 | 1.7005 | 10/10 | 9/10 |
| all | 20 | 2.3778 | 0.9807 | 1.2179 | 20/20 | 16/20 |

With standard deviations over the 20 fixed points:

| object | mean spectral norm | standard deviation |
|---|---:|---:|
| baseline error | 2.3778 | 1.8595 |
| ADV-only error | 0.9807 | 0.7248 |
| clean+ADV error | 1.2179 | 0.7931 |

Interpretation:

```text
ADV-only made J_error smaller on 20/20 fixed points.
clean+ADV made J_error smaller on 16/20 fixed points.
```

This supports the claim that adversarial training, especially ADV-only, reduces local model-solver derivative mismatch.

## 6. Model Jacobian itself did not simply collapse

The model Jacobian spectral norm did not simply become tiny. This is important.

Approximate all-sample means:

| object | mean spectral norm | standard deviation |
|---|---:|---:|
| solver Jacobian | 5.9035 | 1.8238 |
| baseline model Jacobian | 5.2886 | 1.2904 |
| ADV-only model Jacobian | 5.6696 | 1.5668 |
| clean+ADV model Jacobian | 5.7426 | 1.6570 |

Interpretation:

```text
The model did not merely become flat.
The better result is that J_model moved closer to J_solver.
```

This matters because a flat model could have a small Jacobian but still be physically wrong. Our evidence points more toward local derivative alignment with the solver.

## 7. Singular value and singular vector similarity

The top singular values became closer to the solver.

Top-20 singular value relative RMSE vs solver:

| model | top-20 singular value relative RMSE |
|---|---:|
| baseline | 0.1057 |
| ADV-only | 0.0412 |
| clean+ADV | 0.0267 |

The top singular vectors also became more aligned with the solver.

Right singular vector, top-1 absolute dot vs solver:

| model | mean | standard deviation |
|---|---:|---:|
| baseline | 0.9047 | 0.2568 |
| ADV-only | 0.9633 | 0.1575 |
| clean+ADV | 0.9932 | 0.0264 |

Right singular subspace, top-8 mean cosine:

| model | mean cosine |
|---|---:|
| baseline | 0.9582 |
| ADV-only | 0.9905 |
| clean+ADV | 0.9897 |

Left singular vector, top-1 absolute dot vs solver:

| model | mean |
|---|---:|
| baseline | 0.8443 |
| ADV-only | 0.9520 |
| clean+ADV | 0.9752 |

Interpretation:

```text
right singular vectors:
the trained models identify input perturbation directions more like the solver.

left singular vectors:
the trained models produce output-response directions more like the solver.

singular values:
the trained models amplify these perturbations by more solver-like amounts.
```

This is stronger than saying only the loss went down. It says the local linearized behavior became more solver-like.

## 8. Fourier gain meaning

The Fourier gain plots are not the same thing as SVD.

SVD asks:

```text
Among all possible input directions, which directions are most amplified?
```

Fourier gain asks:

```text
If I perturb the input along a fixed Fourier mode k, how much output change do I get?
```

In formula-like terms, if `v_k` is a fixed sine/cosine Fourier mode:

```text
Fourier gain at k = ||J v_k||
```

So `k=40` means Fourier frequency mode 40. It does not mean sample index 40.

Fourier gain is useful because it gives a frequency-response view:

```text
low k:
large-scale smooth perturbations

high k:
high-frequency oscillatory perturbations
```

The observed conclusion was that adversarial training made the model's frequency response closer to the solver, especially in high-frequency bands where baseline mismatch was much larger.

Example all-band/high-band relative RMSE evidence recorded earlier:

```text
high-band Fourier gain relative RMSE vs solver:
baseline   about 11.3704
ADV-only   about 1.5750
clean+ADV  about 1.6560
```

Interpretation:

```text
The baseline model had a much worse high-frequency response mismatch.
After adversarial training, the high-frequency response became much closer to the solver.
```

## 9. ADV-only vs clean+ADV

Based on the current Burgers Jacobian/SVD evidence, the recommended primary run is:

```text
ADV-only + random epsilon/alpha jitter
```

Reason:

```text
ADV-only gave the cleanest J_error contraction:
20/20 fixed points improved relative to baseline.
```

clean+ADV is still useful as a comparison or ablation, but it mixes two objectives:

```text
clean samples
attacked samples
```

This can help retain clean-distribution accuracy, but it can also dilute the pure adversarial objective. In the current Jacobian data, clean+ADV improved many metrics, but its `J_error` contraction was less consistent than ADV-only:

```text
ADV-only:   20/20 points improved
clean+ADV: 16/20 points improved
```

Practical recommendation:

| goal | preferred training data mode |
|---|---|
| prove robust/local derivative improvement | ADV-only |
| study clean accuracy vs robustness tradeoff | clean+ADV as an ablation |

## 10. Random epsilon and alpha

The current recommendation is to keep epsilon random within a controlled range instead of using one fixed epsilon.

Current Burgers setting:

```text
nominal epsilon fraction = 0.06
epsilon jitter = 0.75 to 1.25
observed epsilon range ~= 0.045 to 0.075
```

Current alpha setting:

```text
alpha ratio = 1.0
alpha jitter = 0.75 to 1.25
```

The reason is that random epsilon turns training into multi-scale adversarial training:

```text
small epsilon:
local smoothness and small-perturbation robustness

medium epsilon:
ordinary robustness

large epsilon:
stronger distribution shift and harder adversarial examples
```

If epsilon is fixed, many attacks tend to live near one fixed boundary radius. Random epsilon samples a range of perturbation radii, which is closer to training a robustness distribution.

The mean attack loss under random epsilon still has meaning, but its meaning is:

```text
expected attack loss under the epsilon distribution
```

It is not:

```text
attack loss at one fixed epsilon value
```

Therefore, the mean is useful for trend tracking, but the best analysis should also split results by epsilon bucket:

```text
small epsilon bucket
medium epsilon bucket
large epsilon bucket
```

This would answer whether adversarial training makes attacks harder at all perturbation scales, or only at large/small epsilon.

## 11. Detailed logging requirements now in the training code

The detailed adversarial training code is intended to record:

```text
evaluation every epoch
train/test/generalization metrics every epoch
attack loss before and after attack
attack gain
relative attack gain
epsilon and alpha statistics
fixed-index attack probes
x_clean, x_adv, delta
y_clean, y_adv when target saving is enabled
checkpoint every 200 epochs
```

For Burgers, the fixed probe records allow same-index tracking over training:

```text
same initial condition
same probe index
new model after each epoch
new attack result
saved delta and attacked input
```

This is meant to test the hypothesis:

```text
as adversarial training progresses,
the attack may become harder,
attack gain may shrink,
and the required perturbation pattern may become more complex or more high-frequency.
```

The current run saves probe samples every epoch.

## 12. Current Burgers zero adversarial training run

The first 1000-epoch run used default Burgers batch size:

```text
run: burgers_zero_advonly_random_jitter_1000ep_20260531
batch size: 256
optimizer microbatch size: 32
GPU memory: about 15.8 GiB used on a 32 GiB V100
estimated runtime: about 17 to 18 hours
```

Because that only used about half the GPU memory, it was stopped early and replaced with a larger-batch run.

Batch sweep results:

| Burgers batch size | result | approximate GPU memory behavior | attack throughput |
|---:|---|---|---:|
| 256 | worked | about 15.8 GiB in the original run | about 23.6 samples/s |
| 320 | worked | reserved about 20.3 GiB | about 29.1 samples/s |
| 384 | worked | reserved about 23.6 GiB | about 34.9 samples/s |
| 448 | worked | reserved about 27.8 GiB | about 38.7 samples/s |
| 480 | worked | reserved about 30.4 GiB | about 42.0 samples/s in sweep |
| 512 | failed | OOM on 32 GiB V100 | not usable |

Therefore the current formal run uses:

```text
run: burgers_zero_advonly_random_jitter_1000ep_bs480_20260531
PID at launch: 2953120
task: burgers
training data mode: adv-only
epochs: 1000
checkpoint every epochs: 200
batch size: 480
optimizer batch size: 32
eval max samples: 0, full evaluation
max generalization eval: 50
attack probe samples: 5
attack probe every N epochs: 1
attack probe save targets: enabled
alpha jitter: 0.75 to 1.25
epsilon jitter: 0.75 to 1.25
device: cuda
```

Observed early formal-run status:

```text
GPU: Tesla V100-SXM2-32GB
memory total: 32768 MiB
memory used during run: about 30730 MiB
utilization observed: about 46% to 87%
temperature observed: about 53 C
```

Early epoch examples from the formal batch-480 run:

| epoch | attack batches | attack samples | attack wall seconds | attack samples/s |
|---:|---:|---:|---:|---:|
| 8 | 3 | 1350 | 28.8647 | 46.7699 |
| 9 | 3 | 1350 | 28.8002 | 46.8747 |
| 10 | 3 | 1350 | 29.0578 | 46.4592 |
| 11 | 3 | 1350 | 28.9256 | 46.6714 |
| 12 | 3 | 1350 | 28.8025 | 46.8710 |
| 13 | 3 | 1350 | 31.2353 | 43.2203 |
| 14 | 3 | 1350 | 30.6247 | 44.0820 |
| 15 | 3 | 1350 | 29.5844 | 45.6321 |

Early estimate for the batch-480 run:

```text
warm run attack time: about 29 to 31 seconds per epoch
evaluation time: about 0.72 seconds per epoch
expected total epoch time including train/probe overhead: roughly low-to-mid 30 seconds
rough total time for 1000 epochs: about 9 to 10 hours
first checkpoint at epoch 200: roughly 1.8 to 2.1 hours after launch
```

This is substantially faster than the default batch-256 run while still staying below the 512-batch OOM point.

## 13. Practical experimental plan

Recommended main experiment:

```text
Burgers zero adversarial training
ADV-only
random epsilon jitter
random alpha jitter
1000 epochs
checkpoint every 200 epochs
full evaluation every epoch
fixed attack probes every epoch
batch size 480 on the 32 GiB V100
```

Recommended ablations after the main run:

```text
1. clean+ADV with the same random epsilon/alpha schedule
2. ADV-only with fixed epsilon/alpha
3. ADV-only with epsilon bucket analysis
```

The most important downstream plots should be:

```text
train/test/generalization Relative L2 per epoch
train/test/generalization RMSE per epoch
attack loss before attack per epoch
attack loss after attack per epoch
attack gain per epoch
relative attack gain per epoch
attack gain split by epsilon bucket
fixed-probe delta spectra over epoch
fixed-probe x_clean vs x_adv over epoch
checkpoint-level J_error spectral norm
checkpoint-level singular value/vector similarity vs solver
checkpoint-level Fourier gain similarity vs solver
```

## 14. Core conclusion

The main scientific story is:

```text
Adversarial training should not be judged only by evaluation loss.
The stronger claim is local operator behavior.
```

For Burgers, the current same-point Jacobian evidence says:

```text
ADV-only adversarial training strongly reduces J_error = J_model - J_solver.
The model Jacobian does not simply collapse to zero.
The model's singular values, dangerous input directions, and output response directions become more solver-like.
Random epsilon is conceptually appropriate because it trains robustness across a perturbation-radius distribution.
The current batch-480 zero run is the best active main run because it uses the GPU much more fully without hitting the 512-batch OOM boundary.
```

