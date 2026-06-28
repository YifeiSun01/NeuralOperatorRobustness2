# Loss3 Alpha/Epsilon Core-Four Sweep Plan - 2026-05-19

Status: code and command preparation completed. The official numerical sweep has not been completed in this turn.

## Scope

Observed setup from local code and generated plan files:

- Model/task: FNO / 1D Burgers, `nu=0.001`.
- Objective: `loss3_original`, recorded by the runner as `loss3_q` and auxiliary `loss3_l2` / `loss3_linf`.
- Norm geometry: `p=2`, `q=2`.
- New baseline reference: `epsilon=4`, `alpha=0.4`.
- Previous slow reference retained for comparison: `epsilon=8`, `alpha=0.3`.
- Batch: dataset indices `0..99`, `batch_size=100`.
- Steps: `100`.
- Seed: `0`.
- Trajectory figures/data requested for dataset indices `0`, `7`, `40`, and `47`.

## Main Question

The main question is whether `generalized_power` / replacement-style updates look better mainly because they jump to the epsilon boundary immediately, while additive methods spend many iterations growing `||delta||_p` before they can optimize on the boundary.

Therefore the sweep must record not only final loss and smoothness, but also the boundary-arrival process:

- `delta_pnorm_mean` at every saved optimizer step.
- `boundary_ratio_mean = ||delta||_p / epsilon` at every saved optimizer step.
- First step where the batch mean reaches `95%` and `99%` of the boundary.
- First step where each sample reaches `95%` and `99%` of the boundary.
- The max per-sample first-hit step, which answers when the slowest reached sample got to the boundary threshold.

## Curated Epsilon/Alpha Settings

These are not a Cartesian product. They are hand-picked around the new `epsilon=4`, `alpha=0.4` baseline, with the old `epsilon=8`, `alpha=0.3` included as a slow-boundary reference.

| epsilon | alpha | nominal `epsilon/alpha` steps for L2-steepest add | Purpose |
|---:|---:|---:|---|
| `2` | `0.2` | `10` | smaller radius, same 10-step boundary target |
| `2` | `0.4` | `5` | smaller radius, faster boundary arrival |
| `4` | `0.2` | `20` | new radius, intentionally slower additive control |
| `4` | `0.4` | `10` | new baseline; reaches L2-steepest boundary in about 10 steps |
| `4` | `0.8` | `5` | new radius, faster additive reach |
| `4` | `1.2` | `3.33` | new radius, aggressive alpha stress |
| `8` | `0.3` | `26.67` | old slow baseline retained for comparison |
| `8` | `0.4` | `20` | old radius with slightly larger alpha |
| `8` | `0.8` | `10` | old radius, 10-step boundary target |
| `8` | `1.6` | `5` | old radius, fast boundary target |
| `16` | `1.6` | `10` | larger radius, 10-step boundary target |

Interpretation intent:

- `epsilon/alpha` is the nominal boundary-arrival time for `steepest_add` under `p=2`, because the L2-steepest direction has p-norm `1` before projection.
- `raw_add` does not have this exact nominal step count because its raw gradient norm changes with step and sample.
- `raw_replace` and `steepest_replace` should reach the boundary immediately after the first update if their direction is nonzero.
- Crossed controls separate radius effects from step-size effects.

## Core Methods

The comparison is restricted to four methods:

| Method | Meaning | Update rule summary |
|---|---|---|
| `raw_add` | ordinary raw-gradient PGD/ascent | `delta <- Proj(delta + alpha * grad)` |
| `raw_replace` | user-proposed normalized raw-gradient replacement | `delta <- Proj(epsilon * normalize_p(grad))` |
| `steepest_add` | Lp-steepest PGD/ascent | `delta <- Proj(delta + alpha * steepest_p(grad))` |
| `steepest_replace` | Lp-steepest replacement / GPI-style update | `delta <- Proj(epsilon * steepest_p(grad))` |

Inference from the method definitions:

- `steepest_add` is the Lp-steepest PGD row.
- `steepest_replace` is the generalized-power-iteration style boundary replacement row used for this objective-gradient comparison.
- `raw_add` is ordinary PGD/ascent using the raw autograd gradient.
- `raw_replace` is the custom baseline: boundary replacement using the p-normalized raw gradient.
- In the `p=2` setting, `raw_replace` and `steepest_replace` should be equivalent because L2-normalized raw gradient is the L2-steepest direction. They are both kept in the output so this equivalence is explicit in the tables.

## Prepared Code

Observed files prepared in this turn:

- `tools/run_loss3_alpha_epsilon_core4_sweep.py`: orchestration wrapper that calls `tools/run_loss3_direction_proposal_ablation.py` for each `(epsilon, alpha)` setting.
- `tools/analyze_loss3_alpha_epsilon_core4_sweep.py`: post-processing script that reads completed `per_step_metrics.csv` / `per_sample_step_metrics.csv` files and writes summary/winner/boundary-arrival tables plus a result Markdown.

The wrapper does not implement new optimizer math; it passes only the four requested method names into the existing GPU-only runner.

## GPU Verification Evidence

Observed before the interrupted attempt:

- `nvidia-smi`: Tesla V100-SXM2-32GB, driver `570.211.01`, CUDA capability shown by driver as `12.8`.
- `tools/setup_adv_robust_gpu_env.py --verify-only` passed after repairing the broken `adv_robust/bin/python3` symlink.
- PyTorch: `2.8.0+cu126`, CUDA runtime `12.6`.
- PyTorch device: Tesla V100-SXM2-32GB, compute capability `sm_70`.
- PyTorch arch list includes `sm_70`.
- PyTorch CUDA matmul passed.
- JAX backend: `gpu`; JAX GPU matmul passed.
- `pip check`: no broken requirements.

## Current Local Artifact State

Observed from local files:

- A dry-run generated sweep plan exists at `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_plan.json` with `11` curated settings.
- One setting directory exists from an interrupted earlier start: `forensics/loss3_alpha_epsilon_core4_sweep_20260519/fno_nu0p001_eps4_alpha0p3_batch100_steps100_p2_q2/`.
- That setting manifest is marked `interrupted_not_for_analysis`.
- No completed `per_step_metrics.csv` exists under that interrupted setting at the time this note was written.

Interpretation rule:

- Do not use the interrupted setting directory as numerical evidence.
- A valid official setting requires `manifest.json` status `completed` and root-level `per_step_metrics.csv`.

## Commands

GPU verification before running:

```bash
cd /workspace/NeuralOperatorRobustness2
nvidia-smi
python3 tools/setup_adv_robust_gpu_env.py --verify-only
```

Dry-run command to regenerate/check the curated plan without launching attacks:

```bash
cd /workspace/NeuralOperatorRobustness2
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py --dry-run
```

Official sweep command for the requested `p=2,q=2` core-four comparison:

```bash
cd /workspace/NeuralOperatorRobustness2
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py
```

Equivalent explicit command, useful if you want the settings visible in shell history:

```bash
cd /workspace/NeuralOperatorRobustness2
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py   --settings   2:0.2 2:0.4   4:0.2 4:0.4 4:0.8 4:1.2   8:0.3 8:0.4 8:0.8 8:1.6   16:1.6
```

Post-processing after the sweep finishes:

```bash
cd /workspace/NeuralOperatorRobustness2
adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py
```

Expected analysis outputs after completed runs:

- `forensics/loss3_alpha_epsilon_core4_analysis_20260519/core4_alpha_epsilon_method_summary.csv`
- `forensics/loss3_alpha_epsilon_core4_analysis_20260519/core4_boundary_arrival_summary.csv`
- `forensics/loss3_alpha_epsilon_core4_analysis_20260519/core4_alpha_epsilon_winner_summary.csv`
- `forensics/loss3_alpha_epsilon_core4_analysis_20260519/core4_alpha_epsilon_method_rollup.csv`
- `docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md`

## Remaining Work

- User launches the official sweep from the command above.
- After completion, run the analysis script.
- Inspect boundary-arrival tables first, then final loss, growth speed, and final-delta smoothness/high-frequency behavior.
- Update `EXPERIMENT_LEDGER.md` and the result Markdown with observed numerical conclusions after completed outputs exist.

## 2026-05-20 Expansion: 20 Alpha/Epsilon Settings and P/Q Scope

Status: planning/code update completed; the additional nine settings have not been run yet in this update. A dry-run regenerated `forensics/loss3_alpha_epsilon_core4_sweep_20260519/sweep_plan.json` with `20` settings for the default `p=2,q=2` run.

Important scope clarification:

- The completed alpha/epsilon sweep currently visualized is `p=2,q=2` only.
- Previous all-PQ experiments exist elsewhere, but they are not the same alpha/epsilon sweep.
- The sweep wrapper now supports `--pq-pairs`, so all-PQ alpha/epsilon runs can be launched explicitly, but they should be interpreted and plotted per P/Q pair.

The default setting list is now:

| epsilon | alpha | nominal `epsilon/alpha` steps | Status | Purpose |
|---:|---:|---:|---|---|
| `1` | `0.1` | `10` | added, pending | small-radius 10-step control |
| `1` | `0.2` | `5` | added, pending | small-radius fast control |
| `2` | `0.2` | `10` | completed | existing small-radius 10-step control |
| `2` | `0.4` | `5` | completed | existing small-radius fast control |
| `2` | `0.8` | `2.5` | added, pending | aggressive small-radius control |
| `4` | `0.2` | `20` | completed | slow additive control at new radius |
| `4` | `0.4` | `10` | completed | main baseline |
| `4` | `0.8` | `5` | completed | faster additive control |
| `4` | `1.2` | `3.33` | completed | aggressive alpha stress |
| `4` | `1.6` | `2.5` | added, pending | very fast boundary arrival at new radius |
| `8` | `0.2` | `40` | added, pending | slower-than-old-radius stress |
| `8` | `0.3` | `26.67` | completed | old slow baseline |
| `8` | `0.4` | `20` | completed | old radius, slightly larger alpha |
| `8` | `0.8` | `10` | completed | old radius, 10-step target |
| `8` | `1.6` | `5` | completed | old radius, fast target |
| `8` | `2.4` | `3.33` | added, pending | aggressive old-radius step size |
| `12` | `0.6` | `20` | added, pending | mid-large radius slow control |
| `12` | `1.2` | `10` | added, pending | mid-large radius 10-step control |
| `16` | `1.6` | `10` | completed | large radius 10-step target |
| `16` | `3.2` | `5` | added, pending | large radius fast target |

Default p=2,q=2 command to run only the missing settings uses the same command as before because `--skip-completed` defaults to true:

```bash
cd /workspace/NeuralOperatorRobustness2
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py
```

Optional multi-PQ command, if a full alpha/epsilon/PQ grid is needed, is explicit and much larger:

```bash
cd /workspace/NeuralOperatorRobustness2
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py --pq-pairs 1:1 1:2 1:inf 2:1 2:2 2:inf inf:1 inf:2 inf:inf
```

For plotting multi-PQ results, use `--p-filter` and `--q-filter` and write each P/Q pair to a separate output directory; do not mix P/Q pairs into one alpha/epsilon heatmap.

Boundary-marker visualization was also expanded: loss curves now mark first mean boundary-ratio hits at `25%`, `50%`, `75%`, and `99%`.

