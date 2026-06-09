#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-/workspace/NeuralOperatorRobustness2/adv_robust/bin/python}"
SVD_OUT="forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608"
ATTACK_OUT="forensics/burgers_first_master_finalmodels_svd20_p2q2_attack_correlation_20260608"
LOG_DIR="run_logs/burgers_first_master_svd20_attack_correlation_20260608"
PREFIX="first_master_finalmodels"

BASELINE="1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
LOSS1="adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt"
LOSS2="adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt"
LOSS3="adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt"

mkdir -p "$SVD_OUT" "$ATTACK_OUT" "$LOG_DIR"

nvidia-smi | tee "$SVD_OUT/nvidia_smi_preflight.txt"
"$PY" - <<'PYGPU' | tee "$SVD_OUT/gpu_preflight.json"
import json
import torch

info = {
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required; refusing CPU fallback")
info.update({
    "torch_device_name": torch.cuda.get_device_name(0),
    "torch_device_capability": list(torch.cuda.get_device_capability(0)),
    "torch_cuda_arch_list": torch.cuda.get_arch_list(),
})
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit("PyTorch wheel does not report sm_70 support; refusing official run")
x = torch.ones((128, 128), device="cuda")
y = x @ x
torch.cuda.synchronize()
info["torch_cuda_matmul_0_0"] = float(y[0, 0].item())
try:
    import jax
    import jax.numpy as jnp
    info["jax_version"] = jax.__version__
    info["jax_backend"] = jax.default_backend()
    info["jax_devices"] = [str(d) for d in jax.devices()]
    if jax.default_backend() != "gpu":
        raise SystemExit("JAX GPU backend is required for this solver/SVD run; refusing CPU fallback")
    z = jnp.ones((128, 128), dtype=jnp.float32)
    w = z @ z
    info["jax_gpu_matmul_0_0"] = float(w[0, 0])
except Exception as exc:
    info["jax_error"] = repr(exc)
    raise
print(json.dumps(info, indent=2))
PYGPU

"$PY" tools/compare_burgers_round01_final_jacobian_svd.py \
  --generalization-root generalization_datasets \
  --out-root "$SVD_OUT" \
  --output-prefix "$PREFIX" \
  --report-title "Burgers First Master Generalization Final-Model Jacobian/SVD" \
  --report-note "Observed no first-root SVD artifact locally or in selected R2; this run recomputes first master semantic root sample Jacobians before attack correlation." \
  --baseline-checkpoint "$BASELINE" \
  --loss1-checkpoint "$LOSS1" \
  --loss1-label loss1_epoch8000 \
  --loss1-epoch 8000 \
  --loss2-checkpoint "$LOSS2" \
  --loss2-label loss2_epoch2000 \
  --loss2-epoch 2000 \
  --loss3-checkpoint "$LOSS3" \
  --loss3-label loss3_epoch1500 \
  --loss3-epoch 1500 \
  --train-samples 6 \
  --test-samples 4 \
  --generalization-samples 10 \
  --seed 20260608 \
  --device cuda \
  --top-k 100 \
  --svd-method topk \
  --svd-solver propack \
  2>&1 | tee "$LOG_DIR/svd.log"

"$PY" tools/run_burgers_svd20_p2q2_attack_correlation_20260608.py \
  --svd-manifest "$SVD_OUT/${PREFIX}_sample_manifest.csv" \
  --svd-summary "$SVD_OUT/${PREFIX}_jacobian_svd_summary.csv" \
  --out-dir "$ATTACK_OUT" \
  --steps 20 \
  --batch-size 20 \
  --model-spec "baseline=$BASELINE" \
  --model-spec "loss1_epoch8000=$LOSS1" \
  --model-spec "loss2_epoch2000=$LOSS2" \
  --model-spec "loss3_epoch1500=$LOSS3" \
  2>&1 | tee "$LOG_DIR/attack_correlation.log"
