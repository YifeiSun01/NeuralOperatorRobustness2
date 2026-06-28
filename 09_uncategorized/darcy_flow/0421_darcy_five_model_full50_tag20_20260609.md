# Darcy/C-flow Five-Model Full-50 20-Step Tag Result - 2026-06-09

## Status

Completed the full Darcy/C-flow generalization tag sweep requested by the user:

- 50 generalization `.pt` files
- 5 formal models: `baseline`, `loss1`, `loss2`, `loss3`, `physical_source`
- Batch size: `48`
- Attack steps: `20`
- Shared attack objective: `loss3`
- Ranking metric: ascending mean attack loss gain

## Main Result

Observed from `forensics/darcy_five_model_full50_tag20_20260609/summary_by_model.csv`:

| rank | model | samples | datasets | clean loss mean | final tag loss mean | mean loss gain | relative gain mean |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | physical_source | 2400 | 50 | 1.5452080666037796e-07 | 2.2655414384065153e-07 | 7.203333720543863e-08 | 0.48388714697166807 |
| 2 | loss2 | 2400 | 50 | 2.388385874141363e-07 | 3.4006990865679637e-07 | 1.012313212545024e-07 | 0.4850481873371427 |
| 3 | loss1 | 2400 | 50 | 2.9298726013167926e-07 | 3.9749649036302517e-07 | 1.0450923022542469e-07 | 0.3765278256293641 |
| 4 | loss3 | 2400 | 50 | 9.476862238363044e-08 | 4.0037479414500866e-07 | 3.0560617149788525e-07 | 4.059539115148639 |
| 5 | baseline | 2400 | 50 | 3.2841694900328143e-07 | 7.814819267082385e-07 | 4.530649779432849e-07 | 1.733155802086306 |

Observed least-burst / most robust model by mean loss gain: `physical_source`.

## Per-Dataset Winner Check

Observed from `forensics/darcy_five_model_full50_tag20_20260609/summary_by_dataset.csv`:

- `physical_source` won `50 / 50` datasets by lowest mean attack loss gain.
- `loss3` won `0 / 50` datasets.
- `loss3` ranked 4th on `47 / 50` datasets.
- `loss3` ranked 5th on `3 / 50` datasets.

## Source Files

- Script: `tools/run_darcy_five_model_generalization_tag_20260609.py`
- Result JSON: `forensics/darcy_five_model_full50_tag20_20260609/result.json`
- Summary by model: `forensics/darcy_five_model_full50_tag20_20260609/summary_by_model.csv`
- Summary by dataset: `forensics/darcy_five_model_full50_tag20_20260609/summary_by_dataset.csv`
- All samples: `forensics/darcy_five_model_full50_tag20_20260609/all_samples.csv`
- GPU preflight: `forensics/darcy_five_model_full50_tag20_20260609/gpu_preflight.json`

## Inference

Inference from the completed full-50 run: under 20-step binary steepest-replace tag attacks with the shared solver-consistent `loss3` objective, the `physical_source` model is the robust winner on this Darcy/C-flow generalization sweep. The `loss3` model has the lowest clean loss among the five, but its post-attack loss gain is much larger than `physical_source`, `loss2`, and `loss1`.
