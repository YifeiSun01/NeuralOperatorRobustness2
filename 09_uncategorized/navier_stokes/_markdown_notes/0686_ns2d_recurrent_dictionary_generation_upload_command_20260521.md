# NS2D recurrent dictionary generation and R2 upload command - 2026-05-21

This note records the command for generating the missing `N=2000` dictionary dataset and uploading it to Cloudflare R2. No generation run was launched while writing this note.

## Observed source behavior

- Fast generator: `2D_NS_FNO2d_recurrent/data_generation/generate_ns_dictionary_batched.py`
- Legacy generator: `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.py`
- The fast generator creates the same expected `N=2000`, `tfinal=20`, `ntimepoints=21`, `nu=1e-5` dictionary filename.
- `--batch-size` controls the outer processing chunk.
- `--solver-batch-size` controls true JAX/Exponax solver batching. This is the knob that can speed generation up.
- The generated command now uses `--batch-size 512 --solver-batch-size 512 --solver-mode vmap` on the A100. This is based on the 2026-05-22 forward solver probe; reduce to `256`, `128`, or `20` if a full production run shows memory or stability issues beyond the expected fallback samples.
- The generated command uses `--step-options 0.01,0.0005`, so each sample first tries fast `dt=0.01` and jumps directly to safer `dt=0.0005` if unstable.
- Expected local output when run from `2D_NS_FNO2d_recurrent/`:
  `datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`

## R2 target

```text
r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt
```

## Runtime estimate

From the historical legacy-script B200 log, `N=2000` completed in `2:03:34` with most samples using `dt=0.01`. Existing batched-generation records with `solver_batch_size=8` and `dt=0.005` averaged about `1.99 s/sample` for rollout. Inference from the 2026-05-22 solver-batch probe: with `solver_batch_size=512`, the first `dt=0.01` pass for 2000 samples is roughly four chunks, about `3-4` minutes of solver time. Including fallback samples at `dt=0.0005`, CPU save, and R2 upload, plan roughly `15-30` minutes if the fallback rate stays near the observed 5%.

## Command

The command intentionally keeps secrets out of repository files. Provide R2 credentials through environment variables before running it.

```bash
cd /workspace/NeuralOperatorRobustness2/2D_NS_FNO2d_recurrent

set -euo pipefail

export R2_ACCESS_KEY_ID='<paste-access-key-id>'
export R2_SECRET_ACCESS_KEY='<paste-secret-access-key>'

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RCLONE_CONFIG_R2_AUTO_TYPE=s3
export RCLONE_CONFIG_R2_AUTO_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_AUTO_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_AUTO_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_AUTO_ENDPOINT='https://606bf6c862a4e8f63dabb6243dce2df7.r2.cloudflarestorage.com'
export RCLONE_CONFIG_R2_AUTO_ACL=private

OUT_FILE='datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt'
R2_OUT='r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt'
LOG_DIR='data_generation/logs_gen'
LOG_FILE="$LOG_DIR/VT_NS_gen_all_frame_dict_$(date -u +%Y%m%d_%H%M%S)_UTC.out"

mkdir -p "$LOG_DIR" datasets/exponax_datasets/t20/dictionary

nvidia-smi
../adv_robust/bin/python - <<'PYVERIFY'
import jax, torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('PyTorch CUDA is not available; stop before generation.')
print('torch_device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
print('jax_backend', jax.default_backend(), 'jax_devices', jax.devices())
if jax.default_backend() != 'gpu':
    raise SystemExit('JAX GPU backend is not active; stop before generation.')
PYVERIFY

rclone lsf 'r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/' >/dev/null

if [ ! -f "$OUT_FILE" ]; then
  ../adv_robust/bin/python -u data_generation/generate_ns_dictionary_batched.py \
    --nsamples 2000 \
    --batch-size 512 \
    --solver-batch-size 512 \
    --solver-mode vmap \
    --step-options 0.01,0.0005 \
    2>&1 | tee "$LOG_FILE"
else
  echo "Dictionary already exists locally: $OUT_FILE"
fi

if [ ! -f "$OUT_FILE" ]; then
  echo "Dictionary generation did not produce expected file: $OUT_FILE" >&2
  exit 1
fi

ls -lh "$OUT_FILE"
rclone copyto "$OUT_FILE" "$R2_OUT" --progress --transfers 1 --s3-upload-concurrency 8 --s3-chunk-size 128M
rclone lsl "$R2_OUT"
rclone lsf 'r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/' --files-only | grep -F "$(basename "$OUT_FILE")"
```
