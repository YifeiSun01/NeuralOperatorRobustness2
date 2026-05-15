# Vast.ai V100 CUDA/JAX/cuDNN Troubleshooting

Date: 2026-05-15 UTC

This note records the environment/runtime failures encountered while launching
the FNO `nu=0.001` loss-gradient trajectory experiment on the Vast.ai instance.
It is meant to be the first place to check if a future run appears to exit
immediately, JAX reports a `ptxas` error, or PyTorch fails during FNO backward.

## Context

Experiment being launched:

- Case: 1D Burgers
- Model: FNO, `nu=0.001`
- Attack/objectives: `loss1`, `loss2`, `loss3`
- Indices: `0, 7, 40, 47, 115`
- Attack length: 50 PGD steps
- Saved trajectory cadence: every 5 steps
- Virtual environment that should be used:
  `/workspace/NeuralOperatorRobustness2/adv_robust`

The stable run tag that started successfully after the fixes was:

```text
results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631
logs/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631
```

At the time it was checked, `index=0` had completed all three losses and the
run was continuing on later indices.


## Code Compatibility Update

`run_three_loss_objective_attack.py` now has built-in runtime workarounds for
this issue. By default it will:

- look for a virtualenv-local Triton/NVIDIA `ptxas` and move that directory to
  the front of `PATH`;
- set `torch.backends.cudnn.enabled = False` before constructing the model and
  solver objects;
- record the resolved runtime settings in each run's `config.json` via
  `runtime_ptxas_dir` and `runtime_cudnn_enabled`.

The compatibility controls are:

```bash
--runtime-workarounds / --no-runtime-workarounds
--disable-cudnn / --no-disable-cudnn
--prepend-env-ptxas / --no-prepend-env-ptxas
```

For this Vast V100 instance, the recommended default is to leave these
workarounds enabled. On a cleaner CUDA stack, such as a different GPU image,
`--no-runtime-workarounds` can be used to restore the raw PyTorch/JAX behavior.

Verified probe after this code change:

```bash
source adv_robust/bin/activate
python run_three_loss_objective_attack.py \
  --case burgers \
  --loss_type loss1 \
  --objective_variant original \
  --attack_method pgd \
  --norm 2 \
  --input_p 2 \
  --output_q 2 \
  --epsilon 8.0 \
  --alpha 0.3 \
  --steps 1 \
  --index 0 \
  --burgers-nu 0.001 \
  --burgers-t-final 1.0 \
  --burgers-dt 0.001 \
  --burgers-domain 2.0 \
  --random_start \
  --random_start_scale 1e-6 \
  --seed 0 \
  --save_trajectory \
  --save_every 1 \
  --output_dir /tmp/fno_path_probe_builtin_workarounds \
  --no-progress
```

Observed success:

```text
[runtime] using virtualenv ptxas dir first: .../adv_robust/lib/python3.12/site-packages/triton/backends/nvidia/bin
[runtime] torch.backends.cudnn.enabled=False
[done] saved /tmp/fno_path_probe_builtin_workarounds in 0.86s
```

## Problem 1: SSH Login Exited Immediately

Observed symptom:

```text
no sessions
[exited]
Connection to ... closed.
```

Cause:

- Vast.ai injects an auto-tmux block into `/root/.bashrc`.
- That block tries to attach to a tmux session on SSH login.
- When no attachable session existed, the login shell exited before the user
  reached an interactive prompt.

This was not caused by JAX, PyTorch, or the experiment command. The experiment
had not started yet.

Fix:

```bash
ssh -T vast-ai-20260515 'touch ~/.no_auto_tmux && ls -la ~/.no_auto_tmux && echo OK_DISABLED_AUTO_TMUX'
ssh -t vast-ai-20260515
```

Then create or attach tmux manually:

```bash
tmux new -A -s fno_grad
```

If a session already exists:

```bash
ssh -t vast-ai-20260515 'tmux attach -t fno_grad'
```

If a tmux pane exits too fast to read, capture its recent output:

```bash
ssh -T vast-ai-20260515 'tmux capture-pane -t fno_grad -p -S -100'
```

## Problem 2: Wrong Python Environment

Observed symptom:

```text
/venv/main/bin/python
ModuleNotFoundError: No module named 'jax'
```

Cause:

- Vast automatically activated `/venv/main`.
- The repo-specific virtual environment was actually:
  `/workspace/NeuralOperatorRobustness2/adv_robust`.
- `adv_robust` has the required stack:
  `torch 2.6.0+cu124`, `jax 0.10.0`, and `numpy 2.4.4`.

Important correction:

- The machine did have JAX.
- The active `/venv/main` environment did not.

Fix:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate

which python
python -c "import torch, jax, numpy; print('ok'); print('torch', torch.__version__); print('jax', jax.__version__); print('jax devices', jax.devices())"
```

Expected Python:

```text
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python
```

Do not blindly run `pip install -r requirements.txt` to fix this. The
requirements file can reinstall PyTorch wheels that are not appropriate for
this V100 instance. The correct fix here was to activate `adv_robust`.

## Problem 3: JAX GPU `ptxas` Failure

Observed error:

```text
jax.errors.JaxRuntimeError: UNIMPLEMENTED: /usr/local/cuda/bin/ptxas ptxas too old. Falling back to the driver to compile.
```

Observed machine/toolchain:

- GPU: Tesla V100-SXM2-32GB
- Driver-reported CUDA: 13.0
- System `ptxas`:

```text
/usr/local/cuda/bin/ptxas
Cuda compilation tools, release 13.0, V13.0.88
```

Environment-local `ptxas`:

```text
/workspace/NeuralOperatorRobustness2/adv_robust/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
Cuda compilation tools, release 12.4, V12.4.99
```

Cause:

- JAX compiles parts of the Exponax/Burgers solver for GPU execution.
- During compilation it invokes `ptxas`, the NVIDIA GPU assembler/compiler
  backend.
- The shell found the system CUDA 13 `ptxas` first.
- That toolchain did not work for this JAX + V100 / compute capability 7.0
  combination.

Fix:

Put the `adv_robust` CUDA 12.4 `ptxas` directory before `/usr/local/cuda/bin`
in `PATH`:

```bash
export PATH=/workspace/NeuralOperatorRobustness2/adv_robust/lib/python3.12/site-packages/triton/backends/nvidia/bin:$PATH
```

Minimal verification:

```bash
source adv_robust/bin/activate
export PATH=/workspace/NeuralOperatorRobustness2/adv_robust/lib/python3.12/site-packages/triton/backends/nvidia/bin:$PATH

python - <<'PY2'
import jax
import jax.numpy as jnp
print("devices", jax.devices())
x = jnp.arange(16.0)
print(float((x * x).sum()))
print(jnp.fft.rfftfreq(1024, 1 / 1024).shape)
PY2
```

Expected result:

```text
devices [CudaDevice(id=0)]
1240.0
(513,)
```

## Problem 4: PyTorch/cuDNN FNO Backward Failure

Observed errors:

```text
RuntimeError: GET was unable to find an engine to execute this computation
```

and after disabling the cuDNN v8 API:

```text
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
```

Cause:

- The attack script runs FNO on GPU and calls `loss.backward()`.
- PyTorch attempts to use cuDNN for convolution backward.
- On this V100 + current PyTorch/CUDA/cuDNN combination, cuDNN could not find a
  valid convolution backward algorithm.

Important distinction:

- cuDNN is not mathematically required for the experiment.
- Disabling cuDNN still uses GPU/CUDA.
- It only changes the low-level implementation chosen for convolution
  forward/backward.
- It may be slower than cuDNN, but it is much faster and more practical than
  moving the full experiment to CPU.

Fix:

Run the attack script through a tiny Python wrapper that disables cuDNN before
loading and executing `run_three_loss_objective_attack.py`:

```bash
run_attack_gpu_nocudnn() {
  python - "$@" <<'PY2'
import runpy
import sys
import torch

torch.backends.cudnn.enabled = False
sys.argv = ["run_three_loss_objective_attack.py", *sys.argv[1:]]
runpy.run_path("run_three_loss_objective_attack.py", run_name="__main__")
PY2
}
```

Minimal verification that GPU/no-cuDNN works:

```bash
run_attack_gpu_nocudnn \
  --case burgers \
  --loss_type loss1 \
  --objective_variant original \
  --attack_method pgd \
  --norm 2 \
  --input_p 2 \
  --output_q 2 \
  --epsilon 8.0 \
  --alpha 0.3 \
  --steps 1 \
  --index 0 \
  --burgers-nu 0.001 \
  --burgers-t-final 1.0 \
  --burgers-dt 0.001 \
  --burgers-domain 2.0 \
  --random_start \
  --random_start_scale 1e-6 \
  --seed 0 \
  --save_trajectory \
  --save_every 1 \
  --output_dir /tmp/fno_path_probe_gpu_nocudnn \
  --no-progress
```

Observed successful probe:

```text
[run-start] ... output_dir=/tmp/fno_path_probe_gpu_nocudnn
[done] saved /tmp/fno_path_probe_gpu_nocudnn in 0.87s
```

## Why Previous Jacobian Experiments Did Not Hit This

The earlier local-Jacobian scripts already disabled cuDNN internally.

Observed code:

```text
tools/analyze_fno_solver_jacobian_similarity.py
tools/analyze_deeponet_solver_jacobian_similarity.py
tools/analyze_local_jacobian_fno_deeponet.py
```

These scripts contain:

```python
torch.backends.cudnn.enabled = False
```

Therefore previous Jacobian/SVD experiments could use GPU while avoiding the
cuDNN convolution-backward failure.

The attack script was different:

```text
run_three_loss_objective_attack.py
```

It selected `cuda` by default when available, but did not disable cuDNN. That is
why the trajectory experiment failed until the no-cuDNN wrapper was used.

Another difference:

- Some earlier FNO-vs-solver Jacobian runs reused already-computed FNO
  Jacobians from `forensics/local_jacobian_frequency_20260514/...`, so not every
  later comparison recomputed every model Jacobian from scratch.
- When Jacobians were recomputed, those scripts still had the cuDNN workaround.

## Stable Launch Pattern For This Vast.ai Instance

Use this setup before launching the FNO `nu=0.001` trajectory experiment:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate
python run_three_loss_objective_attack.py ...
```

The script now applies the ptxas PATH and no-cuDNN workarounds by default. The old `run_attack_gpu_nocudnn` wrapper is no longer required for this script, but the wrapper remains a useful emergency fallback for older checkouts.

## Interpretation For Future Runs

If the terminal exits immediately before any experiment output appears:

- suspect Vast auto-tmux first;
- check `~/.no_auto_tmux`;
- attach to an explicit tmux session and capture the pane output.

If `import jax` fails:

- check `which python`;
- activate `adv_robust`;
- do not assume the machine lacks JAX.

If JAX reports `ptxas`:

- check which `ptxas` is first in `PATH`;
- put the `adv_robust` CUDA 12.4 `ptxas` before system CUDA 13.

If PyTorch fails during `loss.backward()` with cuDNN/engine messages:

- disable cuDNN before importing/running the attack script;
- keep GPU enabled;
- do not switch the full experiment to CPU unless no GPU workaround remains.

## Commands Used To Check Status

Check whether the experiment is currently running:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | rg 'run_three_loss_objective_attack|python - --case burgers'
nvidia-smi
```

Check recent tmux output:

```bash
tmux capture-pane -t fno_grad -p -S -160
```

Check completed summaries:

```bash
find results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631 \
  -maxdepth 3 \
  -name summary.json \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
  | sort
```

Check logs:

```bash
find logs/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631 \
  -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' \
  | sort
```
