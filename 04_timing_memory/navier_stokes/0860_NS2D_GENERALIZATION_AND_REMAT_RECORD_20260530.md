# NS2D Generalization And Remat Record

Date: 2026-05-30

This note records the current status of the NS2D dataset completion, no-zoom figures, and the 2D NS checkpoint/rematerialization throughput experiment.

## NS2D Dataset Completion

Goal: add one more NS2D generalization dataset with parameters different from the existing set, so that NS2D reaches 50 datasets total.

Completed dataset:

- Name: `ns_target_grf_alpha1p4_tau12`
- Parameters:
  - `alpha = 1.4`
  - `tau = 12.0`
  - `transform = identity`
- Parameter duplication count: `0`

Final dataset directory:

- `generalization_datasets_rmse_1p5_3x_all_ns50`

Evaluation output:

- `generalization_eval_rmse_1p5_3x_all_ns50/metrics.csv`

NS2D evaluation result:

- NS test RMSE: `0.133570927`
- New dataset RMSE: `0.236655`
- RMSE ratio: `1.771757x`
- NS2D valid datasets: `50/50`
- NS2D all datasets satisfy `1.5x-3x` test RMSE
- Solver stability:
  - max `|y| = 4.813862`
  - failed stability count: `0`

Overall status:

- Burgers: `50`
- Darcy / C-flow: `50`
- NS2D: `50`

## No-Zoom Barplot Versions

Goal: add versions of the ranked and grouped-by-generator RMSE figures without the zoom / near train-test RMSE inset.

Output directory:

- `rmse_1p5_3x_all_ns50_barplots_no_zoom`

Added files:

- `burgers_rmse_ranked_no_zoom.png`
- `burgers_rmse_grouped_by_generator_no_zoom.png`
- `darcy_rmse_ranked_no_zoom.png`
- `darcy_rmse_grouped_by_generator_no_zoom.png`
- `ns2d_rmse_ranked_no_zoom.png`
- `ns2d_rmse_grouped_by_generator_no_zoom.png`

Check:

- The figures are non-empty.
- The no-zoom versions do not include the zoom / near train-test RMSE inset text.
- The main train/test reference lines were kept for comparison.

## NS2D Checkpoint / Rematerialization Throughput Experiment

Question: for the 2D NS attack, does stronger checkpoint/rematerialization make the run more worthwhile overall because the batch size can increase more than the time per update increases?

Experiment record:

- Summary file: `adversarial_training_runs/ns2d_v100_remat_throughput_sweep_20260530/RESULT_SUMMARY.md`
- CSV: `adversarial_training_runs/ns2d_v100_remat_throughput_sweep_20260530/v100_remat_throughput_summary.csv`

Setup:

- GPU: Tesla V100-SXM2-32GB
- Attack: 2D NS recurrent core4
- Loss/method: `loss3`, `raw_add`
- Norms: `p=2`, `q=2`
- Grid/input: `256x256`
- Attack steps: `3`
- PDE micro-steps per sample/update: `3800`
- XLA settings:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`

Important OOM result:

- `solver_remat=none`, `batch_size=2`: OOM in the original sweep.
- Follow-up run with `solver_remat=none`, `batch_size=1`: also OOM.
- The batch-1 no-remat run failed during backward while trying to allocate `15.64 GiB`.

So under this exact V100 / 2D NS / full differentiable attack setup, no-remat is not a useful baseline. Rematerialization is required just to run the attack.

Passed configurations:

| remat | batch | nvidia-smi peak GiB | torch max allocated GB | warm update sec | warm samples/sec | measured samples/sec |
|---|---:|---:|---:|---:|---:|---:|
| `micro` | 4 | 24.64 | 18.89 | 11.377 | 0.352 | 0.280 |
| `micro` | 5 | 31.38 | 23.54 | 13.111 | 0.381 | 0.308 |
| `second` | 5 | 28.18 | 23.54 | 13.890 | 0.360 | 0.297 |
| `chunk` | 5 | 25.18 | 23.54 | 13.844 | 0.361 | 0.296 |
| `chunk` | 6 | 28.98 | 27.86 | 14.522 | 0.413 | 0.342 |

Failed configurations:

- `none`, `batch_size=1`: OOM
- `none`, `batch_size=2`: OOM
- `chunk`, `batch_size=7`: OOM
- `second`, `batch_size=6`: OOM
- `second`, `batch_size=7`: OOM

Best observed setting:

- `solver_remat = chunk`
- `solver_remat_chunk_steps = 20`
- `batch_size = 6`
- warm throughput: `0.413 samples/sec`
- measured end-to-end throughput: `0.342 samples/sec`

Relative to `micro`, `batch_size=4`:

| config | peak memory ratio | warm time ratio | batch ratio | warm throughput ratio |
|---|---:|---:|---:|---:|
| `chunk bs5` | 1.02x | 1.22x | 1.25x | 1.03x |
| `chunk bs6` | 1.18x | 1.28x | 1.50x | 1.18x |
| `micro bs5` | 1.27x | 1.15x | 1.25x | 1.08x |
| `second bs5` | 1.14x | 1.22x | 1.25x | 1.02x |

Interpretation:

- Yes, the memory/time/batch relationship is not linear.
- In this measured sweep, increasing batch from 4 to 6 raised warm update time by only `1.28x`, while batch size rose by `1.50x`.
- That produced a `1.18x` warm-throughput improvement.
- Therefore, for this exact experiment, `chunk` rematerialization with `batch_size=6` is the fastest measured configuration.

Important limitation:

- The statement "batch 6 is fastest" means fastest among the configurations measured here, under this exact V100, 2D NS, 3-step, full-gradient attack setup.
- It should not be read as a universal rule for every GPU, every step count, or every NS2D attack variant.
- Stronger or differently placed rematerialization is not automatically better: `chunk bs7`, `second bs6`, `second bs7`, and no-remat all OOM in this sweep.
