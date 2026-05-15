# Experiment Ledger

Last updated: 2026-05-15 UTC

This is the fixed entry point for experiment status. It must separate observed
evidence from inference. Chat history is not a durable experiment record.

## Critical Correction

The current user question is about the FNO vs DeepONet/default-net **local
Jacobian / SVD / frequency** experiment, not the FNO-vs-DeepONet attack-ratio
tables.

Do not present the attack-ratio tables as the answer to the local-Jacobian
question. A previously created ratio-focused note was removed because it did
not answer the user's question.

## 2026-05-15 Recovery Actions

Observed from Git/GitHub:

- Branch `vast-ai` is at commit `20b689a2c385a700aa8dea91ae1b069adc6d4a77`
  (`Add loss3 mechanism diagnostics`).
- The current branch and `origin/vast-ai` point to that commit after fetch.
- `docs/`, `tools/`, `loss_attack_common.py`, DeepONet runner scripts, and
  the FNO-vs-solver forensics directory were restored from `HEAD`.
- A git-history path search did not find committed paths named
  `tools/analyze_local_jacobian_fno_deeponet.py` or
  `forensics/local_jacobian_frequency_20260514/`.

Observed from the selected R2 artifact prefix:

- The prefix contains `docs/`, `tools/`, `results/`, `forensics/`,
  `deeponet_training_runs/`, `fno_training_runs/`, datasets, and other
  artifacts.
- The selected `forensics/` prefix contains
  `fno_solver_jacobian_similarity_20260514/`.
- Searches under the selected prefix did not find
  `forensics/local_jacobian_frequency_20260514/` or
  `tools/analyze_local_jacobian_fno_deeponet.py`.
- Only small, relevant artifacts were restored/downloaded: result summaries,
  FNO-vs-solver forensics, and DeepONet checkpoints/logs for `nu=0.001` and
  `nu=0.01`.

Observed from the local Python environment:

- System `python3` does not have a usable `torch`.
- The copied `adv_robust` virtual environment has `site-packages/torch`, but
  that directory is empty; importing it yields a namespace module without
  `torch.load`.
- Therefore no new local-Jacobian recomputation has been run in this recovered
  checkout yet.


## 2026-05-15 Git Forensics For Missing Local-Jacobian Script

Observed from `origin/vast-ai`:

- Current local branch `vast-ai` tracks `origin/vast-ai` at commit
  `20b689a2c385a700aa8dea91ae1b069adc6d4a77`.
- `origin/vast-ai:tools/` contains `analyze_main_objective_mechanism.py`,
  `analyze_fno_solver_jacobian_similarity.py`, and
  `summarize_main_objective_mechanism.py`, but not
  `analyze_local_jacobian_fno_deeponet.py`.
- `git log --all --name-status -- tools` shows that commit `20b689a` added
  only those three analysis/summarization files under `tools/`.
- `git grep` in commit `20b689a` finds only indirect references to the missing
  FNO/DeepONet local-Jacobian experiment: the docs mention it, the FNO-vs-solver
  config points to `forensics/local_jacobian_frequency_20260514/...`, and the
  FNO-vs-solver script imports the missing helper.
- `git fsck --full --no-reflogs --unreachable` produced no unreachable commits
  or blobs in this checkout.

Interpretation:

- In this recovered checkout, there is no evidence that the missing helper or
  original DeepONet SVD outputs were ever committed to the visible GitHub
  branch.
- The most likely explanations are: the file/result directory existed only as
  an untracked generated artifact on the previous machine, it lived under a
  different path/name that has not been found yet, or a local commit was made on
  the previous machine but was not pushed and was not included in this recovered
  `.git` object database.

## Experiment Status

| Experiment | Status | Current evidence | What not to claim |
| --- | --- | --- | --- |
| FNO vs DeepONet/default-net local Jacobian/SVD/frequency | Partially evidenced, original outputs missing | `docs/main_objective_mechanism_experiment1_result_20260514.md` says an existing FNO/DeepONet Jacobian experiment computed `J_f`; `forensics/fno_solver_jacobian_similarity_20260514/config.json` points to `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index`; `tools/analyze_fno_solver_jacobian_similarity.py` imports the missing helper script | Do not claim the exact DeepONet high/low-frequency singular-vector conclusion from current files |
| FNO vs solver local Jacobian/SVD/frequency | Available and recorded | `docs/fno_solver_jacobian_similarity_result_20260514.md`; `forensics/fno_solver_jacobian_similarity_20260514/` | Do not confuse this with FNO vs DeepONet |
| FNO vs DeepONet/default-net attack-ratio tables | Available as old ratio evidence | `results/burgers_loss3_clean_recomputed_summary.md`; `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.md`; `.csv` | Do not use this as the Jacobian/SVD answer |
| FNO vs solver tracking-discount / mechanism diagnostics | Available and recorded | `docs/main_objective_mechanism_experiment1_result_20260514.md` and referenced mechanism summary tables | Do not use this as DeepONet evidence |

## FNO vs DeepONet Local Jacobian/SVD Evidence

Observed evidence that the experiment existed:

- `docs/main_objective_mechanism_experiment1_result_20260514.md` explicitly
  refers to an existing FNO/DeepONet Jacobian experiment.
- `forensics/fno_solver_jacobian_similarity_20260514/config.json` reuses FNO
  outputs from
  `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index`.
- `tools/analyze_fno_solver_jacobian_similarity.py` imports
  `tools.analyze_local_jacobian_fno_deeponet`, so the follow-up script depended
  on a local helper that is absent now.

Observed missing artifacts:

- `tools/analyze_local_jacobian_fno_deeponet.py`
- `forensics/local_jacobian_frequency_20260514/`
- DeepONet per-index files such as `index_*/deeponet/*jacobian_svd.npz`
- DeepONet frequency tables such as `*_top_singular_vector_metrics.csv` or
  `*_frequency_gain_by_k.csv`

Current grounded conclusion:

- The FNO-vs-DeepONet local-Jacobian experiment almost certainly existed as a
  generated/local artifact on the previous machine.
- The exact DeepONet SVD/frequency side is not currently recovered.
- The FNO side is partially recoverable from the later FNO-vs-solver forensics:
  in the recorded samples `0, 7, 40, 47, 115`, the leading FNO right singular
  vectors are low-frequency dominated.

## Restored / Downloaded Small Artifacts

Restored from git `HEAD`:

- `docs/main_objective_mechanism_experiment1_result_20260514.md`
- `docs/fno_solver_jacobian_similarity_result_20260514.md`
- `tools/analyze_fno_solver_jacobian_similarity.py`
- `forensics/fno_solver_jacobian_similarity_20260514/`
- `results/burgers_loss3_clean_recomputed_summary.md`
- `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.md`
- `results/clean_recomputed_summary/method_ratio_best_vs_second_tests.csv`

Selected R2 downloads:

- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0.001.pt`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/config.json`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/dataset_info.json`
- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/training_logs/summary.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/config.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/dataset_info.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/output_transform_stats.npz`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/summary.json`


## 2026-05-15 Reconstructed Missing FNO/DeepONet Local-Jacobian Helper

Status: code reconstructed and committed; one DeepONet torch import bug was
fixed during the DeepONet-vs-solver run preparation.

Added:

- `tools/analyze_local_jacobian_fno_deeponet.py`

Purpose:

- restore the helper imported by `tools/analyze_fno_solver_jacobian_similarity.py`;
- provide standalone FNO-vs-DeepONet/default-net explicit local Jacobian/SVD
  analysis;
- save per-index artifacts under
  `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/`
  with the file layout expected by the FNO-vs-solver follow-up:
  `index_*/fno/fno_index*_jacobian_svd.npz` and
  `index_*/deeponet/deeponet_index*_jacobian_svd.npz`.

Implemented interfaces used by the FNO-vs-solver script:

- `load_sample`
- `make_model`
- `compute_explicit_jacobian`
- `analyze_jacobian`
- `frequency_gains`

Verification performed in the recovered checkout:

- `python3 -m py_compile tools/analyze_local_jacobian_fno_deeponet.py tools/analyze_fno_solver_jacobian_similarity.py`
- `python3 -c "import tools.analyze_local_jacobian_fno_deeponet as m; ..."`
- a small 8x8 smoke test for `analyze_jacobian`, confirming NPZ/CSV/JSON
  outputs are written.

Not yet run in this recovered checkout:

- full 1024 x 1024 FNO-vs-DeepONet recomputation, because the copied Python
  environment initially had an empty/broken `torch` package. The environment
  was later patched enough to run the DeepONet-vs-solver Jacobian experiment
  below, but the full FNO-vs-DeepONet recomputation remains separate.

## 2026-05-15 DeepONet vs Solver Local Jacobian/SVD/Frequency

Status: completed and recorded.

Purpose:

- Recreate the FNO-vs-solver local-Jacobian comparison for DeepONet/default-net.
- Use the same five initial conditions as the FNO-vs-solver run:
  `0, 7, 40, 47, 115`.
- Use the DeepONet model setting `nu=0.01` instead of the FNO-side `nu=0.001`.

Added / generated:

- `tools/analyze_deeponet_solver_jacobian_similarity.py`
- `forensics/deeponet_solver_jacobian_similarity_20260515/`
- `docs/deeponet_solver_jacobian_similarity_result_20260515.md`

Observed setting from the run config:

- DeepONet checkpoint:
  `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- DeepONet output transform:
  `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/output_transform_stats.npz`
- Test split:
  `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/..._test.pt`
- Solver: JAX/Exponax Burgers, `nx=1024`, `nu=0.01`, `t_final=1.0`,
  `dt=0.001`, `domain=2.0`, solver dtype `float64`.

Key observed results:

- DeepONet spectral norm mean: `7.202`; solver spectral norm mean: `1.373`;
  residual/error spectral norm mean: `7.100`.
- DeepONet top-1 right singular vectors are high-frequency:
  `hi128` mean `0.761`, zero crossings mean `507.2`.
- Solver top-1 right singular vectors are low-frequency/smooth:
  `hi128` mean `1.70e-17`, zero crossings mean `0.0`.
- Error top-1 right singular vectors track DeepONet:
  `hi128` mean `0.803`, zero crossings mean `516.0`.
- DeepONet-vs-solver top-k right subspaces are nearly orthogonal:
  mean principal cosine at `k=1` is `0.125`, and at `k=8` is `0.170`.
- DeepONet-vs-error leading subspaces are almost identical:
  mean principal cosine at `k=1` is `0.985`, and at `k=8` is `0.981`.

Grounded conclusion:

- For this `nu=0.01` DeepONet/default-net model, the dominant local model
  directions are high-frequency and jagged, while the physical solver's
  dominant local directions are smooth/low-frequency.
- The dominant residual/error Jacobian is essentially the DeepONet
  high-frequency response that the solver does not share.

## 2026-05-15 Top-4 Singular Vector Shape/Fourier Comparison

Status: completed and recorded.

Purpose:

- Directly visualize the top 4 right singular vectors rather than only their
  scalar frequency metrics.
- Compare FNO `nu=0.001`, solver `nu=0.001`, DeepONet `nu=0.01`, and solver
  `nu=0.01` on the same sample indices `0, 7, 40, 47, 115`.

Added / generated:

- `tools/plot_top_singular_vector_comparison.py`
- `forensics/top_singular_vector_comparison_20260515/`
- `docs/top_singular_vector_comparison_result_20260515.md`

Intermediate local-only raw SVD artifacts generated for plotting:

- `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/`
  for FNO top singular vectors.
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/` for the
  `nu=0.001` solver top singular vectors.

Grounded conclusion:

- FNO top right singular vectors and the `nu=0.001` solver top right singular
  vectors are smooth/low-frequency.
- DeepONet `nu=0.01` top right singular vectors are visibly high-frequency and
  jagged.
- The `nu=0.01` solver top right singular vectors remain smooth/low-frequency.
- Therefore the direct plots support the numeric conclusion: FNO's dominant
  local modes resemble solver modes, while DeepONet's dominant local modes are
  high-frequency modes that the solver does not share.

## 2026-05-15 Comprehensive SVD Shape/Fourier/Angle Diagnostics

Status: completed and recorded.

Purpose:

- Produce the dense plot set requested for comparing FNO, solver, DeepONet,
  and error Jacobian SVDs.
- Plot top-8 right singular vectors for each of the five initial-condition
  indices `0, 7, 40, 47, 115`.
- Plot both per-index figures and aggregate figures averaged across the five
  indices after Fourier transform.
- Compute cross-operator pairwise vector cosines, top-k principal angles,
  singular value spectra, and within-SVD orthogonality checks.

Added / generated:

- `tools/plot_comprehensive_svd_diagnostics.py`
- `forensics/comprehensive_svd_diagnostics_20260515/`
- `docs/comprehensive_svd_diagnostics_result_20260515.md`

Generated plot families:

- per-index six-row top-8 right-singular-vector line grids;
- per-index six-row top-8 Fourier-energy grids;
- per-index pairwise right-vector cosine heatmaps;
- per-index top-k principal-angle curves;
- per-index singular-value spectra;
- per-index within-SVD orthogonality-error heatmaps;
- aggregate mean shape, FFT, cosine-heatmap, principal-angle, singular-spectrum,
  and orthogonality figures.

Key observed results:

- FNO model-vs-solver leading right-singular subspaces are close:
  `k=1` mean principal angle `4.86 deg`, `k=8` mean principal angle `7.01 deg`.
- DeepONet model-vs-solver leading right-singular subspaces are far apart:
  `k=1` mean principal angle `82.83 deg`, `k=8` mean principal angle
  `80.19 deg`.
- DeepONet model-vs-error leading right-singular subspaces are close:
  `k=1` mean principal angle `10.01 deg`.
- FNO model top-1 right singular vector is low-frequency:
  mean `hi128` `2.23e-07`, mean zero crossings `13.6`.
- DeepONet model top-1 right singular vector is high-frequency:
  mean `hi128` `0.761`, mean zero crossings `507.2`.
- DeepONet error top-1 right singular vector is also high-frequency:
  mean `hi128` `0.803`, mean zero crossings `516.0`.

Orthogonality check:

- Within a single SVD, top-8 right singular vectors are mutually orthogonal up
  to numerical error. Max off-diagonal Gram errors are about `1e-09` to
  `3e-09`.
- This is expected from SVD and is mainly a sanity check. The scientifically
  useful comparisons are cross-operator vector/subspace alignments, especially
  model-vs-solver and model-vs-error principal angles.

Grounded conclusion:

- FNO's dominant local right-singular directions resemble the solver's
  low-frequency directions.
- DeepONet's dominant local right-singular directions are high-frequency and
  resemble the DeepONet-vs-solver error directions, not the solver directions.

No-std plot variant:

- `tools/plot_comprehensive_svd_diagnostics.py` now supports
  `--no-std-shading`.
- A second copy of the plot set was generated under
  `forensics/comprehensive_svd_diagnostics_20260515_no_std/` with the same
  samples and metrics but without standard-deviation shading in aggregate line
  plots.


## 2026-05-15 FNO nu=0.01 vs Solver and DeepONet nu=0.01 SVD Diagnostics

Status: completed and recorded.

Purpose:

- Repeat the local Jacobian/SVD analysis for FNO at matched `nu=0.01`.
- Compare FNO `nu=0.01`, solver `nu=0.01`, DeepONet/default-net `nu=0.01`, and
  their error Jacobians on the same five sample indices `0, 7, 40, 47, 115`.
- Regenerate the dense line/FFT/heatmap plot set without standard-deviation
  shading.

Recovered input:

- FNO `nu=0.01` checkpoint restored from R2 selected backup:
  `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`.
- The older `tmp_old_runner_inputs_b01/...nu0.01...pth` file was not present and
  was not tracked by Git because `*.pth` is ignored.

Added / generated:

- `tools/analyze_fno_solver_jacobian_similarity.py` now supports
  `--reuse-solver` and `--reuse-solver-root`.
- `tools/plot_fno_deeponet_nu0p01_svd_diagnostics.py`
- `docs/fno_nu0p01_solver_jacobian_status_20260515.md`
- `docs/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_result_20260515.md`
- `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`
- `forensics/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`

Key observed results:

- FNO `nu=0.01` model and solver are very tightly aligned locally:
  model-vs-solver `k=1` mean principal angle `2.32 deg`, `k=8` mean principal
  angle `1.82 deg`, and model-direction response cosine mean `0.9993`.
- FNO `nu=0.01` model top-1 right singular vector is extremely low-frequency:
  mean `hi128` `2.82e-09`, mean zero crossings `0.40`.
- FNO `nu=0.01` residual/error Jacobian is small in spectral norm:
  model top-1 sigma `1.379`, solver top-1 sigma `1.373`, error top-1 sigma
  `0.0373`.
- DeepONet/default-net `nu=0.01` remains high-frequency and solver-misaligned:
  model-vs-solver `k=1` mean principal angle `82.83 deg`, model top-1 `hi128`
  `0.7614`, and model top-1 zero crossings `507.20`.
- DeepONet/default-net error remains close to the model subspace:
  model-vs-error `k=1` mean principal angle `10.01 deg`; FNO model-vs-error
  `k=1` mean principal angle is `62.53 deg`.

Grounded conclusion:

- At matched `nu=0.01`, FNO behaves even more solver-like than in the earlier
  mixed-`nu` comparison. Its model and solver Jacobians have almost the same
  leading singular values and nearly the same leading right-singular subspace.
- DeepONet/default-net is still dominated by high-frequency, solver-misaligned
  local directions, so its error directions are close to the model directions.

## Next Actions

1. If the exact original FNO-vs-DeepONet/default-net artifacts from the old
   Vast.ai instance are still needed, recover that full artifact directory and
   compare it against the reconstructed results.
2. For every new experiment, add a dedicated result note under `docs/`, update
   this ledger, and commit the scripts plus lightweight CSV/PNG/Markdown
   outputs. Keep large raw `.npz` artifacts local unless explicitly requested.
