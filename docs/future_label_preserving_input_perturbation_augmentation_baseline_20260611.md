# Future Experiment: Label-Preserving Input-Perturbation Augmentation Baseline

Date: 2026-06-11

Status: future experiment idea; not run yet.

## Core Idea

除了现在已经有的 adversarial training 对照之外，未来再加一个新的 robustness baseline：

- Burgers loss1, loss2, loss3 adversarial training.
- Darcy flow loss1, loss2, loss3 adversarial training.
- Physics / Physicx / Navier-Stokes-related loss4 adversarial training, exact naming TBD.
- New baseline: label-preserving input-perturbation augmentation.

这个新 baseline 的训练方式是：对每一个训练样本的输入 `x` 加一个很小的扰动 `delta`，但是保持输出 `y` 不变。

```text
original:  (x, y)
augmented: (x + delta, y)
```

`delta` 可以是随机扰动、高频噪声、低频平滑扰动、Fourier/spectral noise，或者几种扰动的混合。这个方法的目标不是替代真正的 adversarial training，而是作为一个简单、局部、被动的数据增强对照组。

## Naming

这个方法以后不要直接叫成和 loss1/loss2/loss3 同一类的 adversarial training。更准确的名字是：

```text
label-preserving input-perturbation augmentation baseline
```

或者：

```text
noisy-input invariant-label baseline
```

原因是它没有主动寻找 worst-case perturbation，也没有根据 PDE solver 重新计算 `x + delta` 对应的新 `y`。

## Why This Baseline Is Probably Flawed

这个方法很可能是不合理、也不够好的，主要问题是：

- PDE 里 `x` 变了以后，真实的解 `y` 通常也应该变。
- 如果强行保持 `y` 不变，训练集中会出现物理上不一致的 pair：`(x + delta, y)`。
- 它只能探索原始 train/test data 附近的局部邻域。
- 它不能主动探索非常不一样的 `x`。
- 它不能像真正的 adversarial training 那样寻找让模型最坏、最容易出问题的方向。
- 它可能让 train/test 附近的局部 robustness 变好一点，但对更远的 generalization datasets 不一定有帮助。
- 它可能不能降低 generalization loss，甚至可能因为错误 label 让 generalization 变差。

## Hypothesis

预期结果：

- 在 train/test distribution 附近，小扰动 attack 下可能会有一点稳定性提升。
- clean train/test loss 可能变化不大，或者轻微改善。
- 在更远的泛化数据集上，尤其是 kernel、range、frequency、spectrum、smoothness 都明显变化的数据集上，这个方法大概率不如真正的 loss1/loss2/loss3 adversarial training。
- 对 Burgers wide-parameter generalization datasets，预期 loss3 adversarial training 仍然应该明显强于这个 noisy-input invariant-label baseline。

## Perturbation Variants To Try

未来实现时可以试这些 `delta` 类型：

- small Gaussian noise
- smooth low-frequency noise
- high-frequency Fourier noise
- power-law spectral noise
- random phase sinusoidal noise
- mixed-frequency perturbation
- amplitude-scheduled perturbation
- multiple `delta` samples per original `x`

关键参数：

- `num_perturbations_per_sample`
- `delta_amplitude`
- `delta_family`
- `delta_frequency_band`
- `delta_smoothness`
- `clip_or_rescale_x`
- `same_y_label = true`

## Evaluation Plan

需要和这些模型一起比较：

- baseline model
- loss1 adversarial-training model
- loss2 adversarial-training model
- loss3 adversarial-training model
- label-preserving input-perturbation augmentation model

任务范围：

- Burgers
- Darcy flow
- Physics / Physicx / Navier-Stokes-related task, exact naming TBD

记录指标：

- clean RMSE
- clean relative L2
- train/test loss
- generalization-dataset loss
- adversarial attack initial loss
- adversarial attack final loss
- adversarial attack loss growth
- 15-step small-budget attack behavior when matching the existing attack setting
- Jacobian/SVD metric when feasible: `||J_model - J_solver||_2`
- correlation between attack loss growth and Jacobian/SVD spectral norm

核心问题：

- 它能不能降低 train/test 附近的小 budget attack loss growth？
- 它能不能降低 true generalization loss？
- 它在远分布泛化数据集上是不是明显弱于 loss3？
- 它是不是只带来局部 robustness，而没有真正的 broader distribution-shift robustness？

## Important Control Note

这个实验的价值在于它是一个 flawed control baseline。它可以帮助回答：简单地对 `x` 加 noise、让模型学会局部不变性，到底能不能接近真正 adversarial training 的效果。

如果以后想做更物理正确的版本，可以另开一个 solver-corrected augmentation：对 `x + delta` 重新跑 solver 得到新的 `y_delta`，训练 `(x + delta, y_delta)`。但那是另一个实验，不要和这里这个保持 `y` 不变的 baseline 混在一起。
