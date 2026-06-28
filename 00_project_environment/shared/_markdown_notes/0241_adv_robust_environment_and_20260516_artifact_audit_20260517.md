# adv_robust Environment And 2026-05-16 Artifact Audit - 2026-05-17

Status: completed audit on the current Vast.ai instance. No numerical experiment was rerun.

## Environment Check

Observed from the first `tools/setup_adv_robust_gpu_env.py --verify-only` run:

- `adv_robust/` existed, but the Python entrypoint was broken by a symlink loop: `adv_robust/bin/python -> python3`, `python3 -> python3.12`, and `python3.12 -> python3`.
- Because `adv_robust/bin/python` did not resolve to a real executable, the setup script tried to recreate the venv and failed with "Too many levels of symbolic links".

Observed repair:

- Repointed `adv_robust/bin/python3` to `/usr/bin/python3.12`.
- Re-ran `tools/setup_adv_robust_gpu_env.py --verify-only`.

Observed from the successful verification run:

- GPU: `Tesla V100-SXM2-32GB`.
- Compute capability: `7.0`; required PyTorch architecture: `sm_70`.
- Python: `/workspace/NeuralOperatorRobustness2/adv_robust/bin/python`.
- PyTorch: `2.8.0+cu126`.
- PyTorch CUDA runtime: `12.6`.
- `torch.cuda.is_available()`: `True`.
- PyTorch device: `Tesla V100-SXM2-32GB`.
- PyTorch supported architectures: `['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']`.
- PyTorch CUDA matrix multiply returned `1.0`.
- JAX backend: `gpu`.
- JAX devices: `[CudaDevice(id=0)]`.
- JAX GPU matrix multiply returned `1.0`.
- `pip check`: `No broken requirements found.`

Observed from `requirements.txt` versus installed distributions:

- Direct requirement entries checked: `79`.
- Missing direct requirements: `[]`.
- Version mismatches among pinned `==` requirements: `[]`.
- Installed distributions: `85`.
- Extra installed packages include R2/object-store support dependencies such as `boto3`, `botocore`, `s3transfer`, `jmespath`, and `urllib3`.

Observed import smoke test:

- Key imports passed for `torch`, `jax`, `numpy`, `scipy`, `pandas`, `matplotlib`, `h5py`, `sklearn`, `numba`, `deepxde`, `equinox`, `exponax`, `torch2jax`, `tensorly`, `tslearn`, `umap`, `skopt`, `graphviz`, `phi`, and `phiml`.

Inference from the evidence above:

- The copied `adv_robust` environment was not runnable before the symlink repair.
- After the symlink repair, the environment is installed sufficiently for this repository's GPU-only experiments: PyTorch and JAX both execute real GPU operations on the V100, the PyTorch wheel includes `sm_70`, direct requirements are present, and dependency metadata is consistent.

## 2026-05-16 Artifact Check

Observed from local manifest audit:

- Found `18` `manifest.json` files under `forensics/*20260516*`.
- All `18` manifests parsed successfully.
- All manifest-declared `output_files` exist locally.
- All manifest-declared `result_doc` and `plan_doc` paths exist locally.
- All manifest-declared `source_paths` exist locally.
- `17` of the `18` manifests include GPU runtime evidence consistent with the V100 / `sm_70` policy.

Important exception:

- `forensics/loss3_small_epsilon_sweep_20260516/smoke_fno_nu0p001/manifest.json` records `"device": "cpu"` and has no `gpu_runtime` block.
- That directory should be treated as an old smoke check, not as an official GPU experiment result.
- It is superseded by GPU-recorded artifacts such as `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/` and `forensics/loss3_small_epsilon_sweep_20260516/gpu_guard_probe/`.

Observed incomplete scratch directory:

- `forensics/loss3_ray_profile_optimizer_control_20260516/` contains only one file: `fno_nu0p001_gpu_v100_batch100_eps4_alpha0p15_steps50_local_pgd/local_direction_trace.csv`.
- That CSV has `5` lines including the header.
- `docs/r2_sync_manifest_ray_profile_20260516.md` also records this directory as only `1` file and `672` bytes.
- No local `manifest.json` or result doc was found for this directory.

Inference from that evidence:

- `forensics/loss3_ray_profile_optimizer_control_20260516/` is not a complete experiment result. Treat it as an incomplete/aborted trace unless later R2 or another machine provides additional artifacts.

Observed from the corrected fixed-sign PGD batch-100 Ray profile:

- Directory: `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Manifest-declared outputs are present.
- Local file inventory found `59` files including the local R2 upload manifest.
- `r2_upload_manifest_20260516.txt` records `file_count=58` and `bytes_total=97423607`; the upload manifest itself accounts for the local inventory being one file larger than the recorded upload count.
- `ray_profile.csv` has `31501` lines, matching `100 samples * 7 directions * 45 radii + header`.
- `ray_winner_summary.csv` has `101` lines, matching `100 samples + header`.
- `attack_final_by_sample.csv` has `401` lines, matching `100 samples * 4 attack specs + header`.

Observed from the historical-script cross-check directory:

- Directory: `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.
- Found all `9` expected `loss3_*` run directories: `loss3_original`, `loss3_increment_ratio`, and `loss3_regularized`, each with `pgd`, `lp_steepest_pgd`, and `generalized_power` variants.
- For each of the `9` run directories, the expected `summary.json`, `loss_stats.csv`, `loss_values.npz`, `final_delta.npz`, `final_delta_summary.json`, `final_delta_diagnostics.csv`, and `final_delta_diagnostics.npz` files are present.

Observed from `docs/r2_sync_manifest_ray_profile_20260516.md`:

- The recorded R2 sync uploaded `198` files and `145207187` bytes under the documented R2 prefix.
- This audit did not perform a live R2 network verification; it only checked local files and the recorded local upload manifests.

Observed from `git status --short` before this record update:

- The affected experiment paths are untracked in the working tree.
- No deleted tracked experiment artifacts were shown.

## Conclusion

Observed evidence supports this current local conclusion:

- The copied package environment is now runnable after repairing the venv Python symlink.
- The main 2026-05-16 GPU experiment artifacts with manifests are locally complete according to their own manifests and source-path records.
- The corrected fixed-sign PGD batch-100 Ray profile and the historical cross-check directory are locally complete by row-count and expected-file checks.

Remaining caveats:

- The CPU `smoke_fno_nu0p001` directory should not be used as an official GPU result.
- The `loss3_ray_profile_optimizer_control_20260516` directory is incomplete locally and should not be interpreted as a completed experiment.
- R2 remote completeness was not re-queried live in this audit.
