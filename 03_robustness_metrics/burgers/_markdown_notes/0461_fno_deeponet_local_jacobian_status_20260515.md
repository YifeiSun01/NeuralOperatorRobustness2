# FNO vs DeepONet Local Jacobian/SVD Status

Date recorded: 2026-05-15 UTC

This note corrects a previous confusion: this is not the FNO-vs-DeepONet
attack-ratio experiment. The experiment in question is the local Jacobian/SVD
frequency experiment comparing FNO and DeepONet/default-net around selected
initial conditions.

## Observed Evidence

Observed from git commit `20b689a2c385a700aa8dea91ae1b069adc6d4a77`:

- `docs/main_objective_mechanism_experiment1_result_20260514.md` says:
  "The existing FNO/DeepONet Jacobian experiment already computed `J_f` for
  FNO; this follow-up adds the solver Jacobian `J_j` and the residual Jacobian
  `J_e`."
- `forensics/fno_solver_jacobian_similarity_20260514/config.json` has:
  `reuse_fno_root =
  /workspace/NeuralOperatorRobustness2/forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index`.
- `tools/analyze_fno_solver_jacobian_similarity.py` imports
  `tools.analyze_local_jacobian_fno_deeponet` and reuses FNO SVD files named
  like `index_*/fno/fno_index*_jacobian_svd.npz`.

Observed from the current working tree:

- `forensics/local_jacobian_frequency_20260514/` is not present.
- `tools/analyze_local_jacobian_fno_deeponet.py` is not present.
- A git-history filename search did not find committed paths named
  `local_jacobian_frequency_20260514` or
  `analyze_local_jacobian_fno_deeponet.py`.

Observed from the selected R2 artifact prefix:

- `forensics/fno_solver_jacobian_similarity_20260514/` exists.
- Direct checks/searches did not find
  `forensics/local_jacobian_frequency_20260514/`.
- Direct checks/searches did not find
  `tools/analyze_local_jacobian_fno_deeponet.py`.
- DeepONet checkpoints/logs for `nu=0.001` and `nu=0.01` were available and
  selectively downloaded, but those are model artifacts, not the old SVD
  result tables.

Observed from the current Python environment:

- System `python3` cannot import usable `torch`.
- The copied `adv_robust` environment contains an empty
  `site-packages/torch/` directory, so `torch.load` is not available.
- No new FNO-vs-DeepONet Jacobian recomputation has been run in this recovered
  checkout yet.

## What Can Be Concluded

The FNO-vs-DeepONet local Jacobian experiment almost certainly existed as a
local/generated artifact on the previous machine, because the later
FNO-vs-solver Jacobian experiment explicitly reused its FNO SVD outputs.

However, the original FNO-vs-DeepONet result directory and script are not
currently available in this checkout, and they do not appear to have been
committed under those names or included in the selected R2 prefix.

Therefore, the exact DeepONet singular-vector frequency metrics cannot be
claimed from the current local evidence.

## Related Available Evidence

The committed FNO-vs-solver follow-up does contain FNO local-Jacobian SVD
frequency metrics for sample indices `0, 7, 40, 47, 115`. In those committed
outputs, the leading FNO right singular vectors are low-frequency dominated:

- `top1_right_hi128` is around `1e-6` or smaller for the sampled FNO Jacobians.
- `top8_right_zero_crossings_mean` is around `12--13.5` for the sampled FNO
  Jacobians.

This supports a statement about the FNO side in the later FNO-vs-solver
analysis, but it does not recover the missing DeepONet side.

Important boundary:

- The statement "FNO is low-frequency while DeepONet is high-frequency" is not
  currently verified from recovered DeepONet SVD files.
- If that statement is true, it still needs either the original
  `deeponet/*jacobian_svd.npz` artifacts or a fresh rerun.


## Reconstructed Code

Added on 2026-05-15 UTC:

- `tools/analyze_local_jacobian_fno_deeponet.py`

This reconstructs the missing helper expected by
`tools/analyze_fno_solver_jacobian_similarity.py`. It can also be run directly
to recompute the FNO-vs-DeepONet local Jacobian/SVD/frequency experiment.

Default output root:

```text
forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/
```

Current verification:

- Syntax compile passes for the reconstructed helper and the FNO-vs-solver
  script.
- Importing the helper exposes the required functions.
- A small synthetic SVD smoke test writes NPZ/CSV/JSON outputs.

The full recovered experiment still needs a working PyTorch/DeepXDE environment
before rerunning the 1024 x 1024 Jacobian computations.

## Missing Files To Recover

Look for these on the old Vast.ai instance, R2, or any copied artifact bundle:

- `forensics/local_jacobian_frequency_20260514/`
- `forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/`
- `tools/analyze_local_jacobian_fno_deeponet.py`
- per-index files under that root, especially:
  - `index_*/fno/*jacobian_svd.npz`
  - `index_*/deeponet/*jacobian_svd.npz`
  - `*_top_singular_vector_metrics.csv`
  - `*_frequency_gain_by_k.csv`
  - any `summary.md`

## Current Answer To The User Question

Was there an FNO-vs-DeepONet local Jacobian/SVD/frequency experiment?

Yes, there is strong indirect evidence that it existed.

Can the exact conclusion "one model's perturbation is high-frequency and the
other is low-frequency" be verified from the current files?

Not yet. The original DeepONet Jacobian/SVD frequency outputs are missing from
the current checkout and the selected R2 prefix.
