# NS2D Recurrent Attack Sweep And loss1 Random-Start Patch - 2026-05-22

## Status

Implemented code changes only. No GPU attack run was started for this patch.

Changed files:

```text
2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py
2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh
```

## Change 1: Multiple Epsilon/Alpha Settings

The Python CLI now supports three ways to set epsilon/alpha.

### Single Setting, Backward Compatible

```bash
--epsilon 32 --alpha 1
```

The wrapper still supports:

```bash
EPSILON=32
ALPHA=1
```

### Explicit Paired Sweep, Recommended

This runs exactly three parameter settings, not a Cartesian product:

```bash
--epsilon-alpha-pairs 16:0.5 32:1 64:2
```

Wrapper equivalent:

```bash
export EPSILON_ALPHA_PAIRS="16:0.5 32:1 64:2"
```

Use this when we want two or three controlled scale settings without accidentally multiplying the whole experiment too much.

### Cartesian Product Sweep

This runs every epsilon with every alpha:

```bash
--epsilons 16 32 64 --alphas 0.5 1 2
```

Wrapper equivalent:

```bash
export EPSILONS="16 32 64"
export ALPHAS="0.5 1 2"
```

This example is `3 * 3 = 9` parameter combinations, so use it only intentionally.

## Output Layout

For a single epsilon/alpha setting, output layout stays backward compatible:

```text
batch_0000_0009/loss1/raw_add/
```

For multiple settings, the code inserts a parameter directory so results do not overwrite each other:

```text
batch_0000_0009/eps16_alpha0p5/loss1/raw_add/
batch_0000_0009/eps32_alpha1/loss1/raw_add/
batch_0000_0009/eps64_alpha2/loss1/raw_add/
```

Each method `summary.json` records the actual `epsilon` and `alpha`.

## Change 2: loss1 Random Start

The `loss1` attack no longer starts from exactly zero by default. This fixes the previous stall:

```text
delta = 0
loss1 = ||F(x + delta) - F(x)|| = 0
grad = 0
```

New default behavior:

```text
loss1_random_start = true
loss1_random_start_fraction = 0.001
loss1_random_start_seed = 12345
```

For `epsilon=32`, the initial L2 radius is:

```text
0.001 * 32 = 0.032
```

This is intentionally tiny compared with the full L2 budget, but nonzero so `loss1` can escape the exact-zero gradient fixed point.

The same random direction is used across update methods for the same batch/mode/norm setup, which makes methods easier to compare. Different epsilon values scale the same direction to different tiny radii.

Wrapper controls:

```bash
export LOSS1_RANDOM_START=1
export LOSS1_RANDOM_START_FRACTION=0.001
export LOSS1_RANDOM_START_SEED=12345
```

To disable for an ablation:

```bash
export LOSS1_RANDOM_START=0
```

or direct CLI:

```bash
--no-loss1-random-start
```

## Checks Run

These checks were run and passed without starting an experiment:

```text
adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py
bash -n 2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh
CUDA_VISIBLE_DEVICES='' adv_robust/bin/python 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py --help
```

The help output confirms the new flags:

```text
--epsilons
--alphas
--epsilon-alpha-pairs
--loss1-random-start / --no-loss1-random-start
--loss1-random-start-fraction
--loss1-random-start-seed
```

## Recommended Next Command Pattern

For a controlled rerun with three paired scale settings:

```bash
export EPSILON_ALPHA_PAIRS="16:0.5 32:1 64:2"
export LOSS1_RANDOM_START=1
export LOSS1_RANDOM_START_FRACTION=0.001
export LOSS1_RANDOM_START_SEED=12345
```

Then run the same wrapper per loss/mode group as before. The top-level launch script should use `set -euo pipefail` so it does not continue to the next group if one group is manually stopped or fails.
