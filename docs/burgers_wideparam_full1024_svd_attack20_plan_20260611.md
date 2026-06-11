# Burgers Full-1024 SVD/Attack 20-Sample Plan - 2026-06-11

This is the planned expanded audit after the initial 3-sample full-1024 probe.
It keeps the same strict rule: full dense `1024 x 1024` Jacobians and full SVDs,
with no block projection, no randomized SVD, and no coarse top-k proxy.

## Sample Design

- `2` samples from Burgers train.
- `2` samples from Burgers test.
- `16` samples from the final wide-parameter loss3-targeted generalization root.
- The `16` generalization samples are selected from `16` different datasets.
- The sample manifest is fixed and reused by all models and the solver.

Default generalization root:

`generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers`

Default output root:

`forensics/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611`

## Models

The default runner uses the same four final Burgers checkpoints:

| key | checkpoint |
|---|---|
| `baseline` | `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt` |
| `loss1` | `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt` |
| `loss2` | `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt` |
| `loss3` | `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt` |

## Runtime Estimate

The estimate is based on the completed 3-sample full-1024 probe:

- total wall time: `2160.404` seconds = `36.01` minutes.
- attack time: `185.452` seconds.
- Jacobian + SVD time: `1974.585` seconds.

Linear extrapolation to 20 samples:

- total: about `14402.7` seconds = `4.00` hours.
- attack: about `1236.3` seconds = `20.6` minutes.
- Jacobian + SVD: about `13163.9` seconds = `3.66` hours.

Planning range with overhead: about `4.2` to `4.7` hours on the same machine.
If additional model checkpoints are added, each extra model over 20 samples is
roughly another `0.5` hour, because it adds model Jacobians, model SVDs,
error-Jacobian SVDs, and a batched attack pass.

## Outputs Recorded

The runner writes:

- `sample_manifest.csv/json`: fixed train/test/generalization sample list.
- `runtime_estimate.json`: estimate based on the 3-sample measured run.
- `attack_step_metrics.csv`: every attack step for every sample and model.
- `svd_summary.csv`: solver/model/error SVD summaries.
- `singular_values_top100_long.csv`: Top-100 singular values in long form.
- `svd_pair_similarities_topk.csv`: model-vs-solver, error-vs-model, and error-vs-solver Top-K left/right subspace similarity for K in `5,10,20,50,100`.
- `cross_model_error_subspace_similarities_topk.csv`: cross-model error-Jacobian singular-vector subspace similarity.
- `svd_attack_joined_metrics.csv`: aligned model/sample rows joining SVD metrics, attack metrics, and residual-change decomposition.
- `model_pair_reductions.csv`: spectral-norm and attack-damage reductions between model pairs, including reduction versus baseline.
- `svd_attack_correlations.csv`: Pearson/Spearman correlations for spectral, vector-similarity, attack-growth, final-loss, and residual-change metrics.
- `runtime_components.csv` and `runtime_summary.json`: detailed timing.
- `sample_*/{solver,*_model,*_error}/*_jacobian_svd.npz`: full dense Jacobian, singular values, and full left/right singular vectors.

## Commands

Estimate and materialize the fixed manifest only:

```bash
/venv/adv_robust/bin/python tools/run_burgers_wideparam_full1024_svd_attack20_20260611.py --estimate-only
```

Run the full audit:

```bash
bash tools/run_burgers_wideparam_full1024_svd_attack20_20260611.sh
```

The runner is resumable with `--reuse-existing` by default. It writes partial CSVs
after each sample so an interrupted run can be inspected and resumed.
