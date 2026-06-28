# NS2D Recurrent Core4 Attack Runtime Status - 2026-05-22

Updated: 2026-05-22 23:57:34 UTC

Status: runtime inspection only. No attack, solver, model, plotting, or GPU computation was started by this status check.

## Active Process

Observed process:

`adv_robust/bin/python 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py ...`

Observed GPU state from `nvidia-smi`:

- GPU: `NVIDIA A100-SXM4-80GB`
- memory used: about `47197 MiB / 81920 MiB`
- utilization: `100%`

## Current Position

The outer wrapper is:

`/tmp/run_ns2d_pair_outer_attack.sh`

Planned pair order:

1. `32:10 eps32_alpha10`
2. `8:2.5 eps8_alpha2p5`
3. `16:5 eps16_alpha5`
4. `32:15 eps32_alpha15`

Each pair runs:

1. `loss1 / all_w`
2. `loss2 / all_a_target_w`
3. `loss3 / all_w`
4. `loss3 / all_d_target_w`
5. `loss3 / w1_5_d6_9_target_w`
6. `loss3 / d1_5_w6_9_target_w`
7. `loss3 / a1_5_d6_9_target_w`

Observed current status:

- `eps32_alpha10` is complete.
- `eps8_alpha2p5` has completed:
  - `loss1 / all_w`
  - `loss2 / all_a_target_w`
  - `loss3 / all_w`
  - `loss3 / all_d_target_w`
- The active run has advanced to:
  - `eps8_alpha2p5 / loss3 / w1_5_d6_9_target_w`

## Observed Completed Durations

Observed from the nohup log:

| pair | loss | mode | duration |
|---|---|---|---:|
| `32:10` | `loss1` | `all_w` | `75.72 min` |
| `32:10` | `loss2` | `all_a_target_w` | `32.62 min` |
| `32:10` | `loss3` | `all_w` | `122.85 min` |
| `32:10` | `loss3` | `all_d_target_w` | `123.42 min` |
| `32:10` | `loss3` | `w1_5_d6_9_target_w` | `124.58 min` |
| `32:10` | `loss3` | `d1_5_w6_9_target_w` | `125.32 min` |
| `32:10` | `loss3` | `a1_5_d6_9_target_w` | `124.28 min` |
| `8:2.5` | `loss1` | `all_w` | `74.28 min` |
| `8:2.5` | `loss2` | `all_a_target_w` | `32.72 min` |
| `8:2.5` | `loss3` | `all_w` | `124.68 min` |
| `8:2.5` | `loss3` | `all_d_target_w` | `122.98 min` |

## Estimate

Inference from observed completed durations:

- One full pair is about `12.1 h`.
- Each `loss3` mode is about `2.05 h`.
- `eps8_alpha2p5` has about three `loss3` modes remaining, about `6.2 h`.
- After `eps8_alpha2p5`, two full pairs remain: `eps16_alpha5` and `eps32_alpha15`, about `24.2 h` total.

Estimated remaining time from this status check:

- about `30-31 h` total remaining.
- rough finish time: `2026-05-24 06:27 UTC`.
- same time in GMT-5: `2026-05-24 01:27 GMT-5`.

## Caveats

- This estimate assumes future loss3 modes continue around `2.05 h` each and loss1/loss2 stay close to prior timings.
- The estimate may shift if a mode has slower JAX compilation, allocator behavior, or filesystem output overhead.
