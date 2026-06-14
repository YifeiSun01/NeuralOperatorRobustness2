import json, os, sys, time
import jax, jax.numpy as jnp
import torch
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "jax_version": jax.__version__,
    "jax_backend": jax.default_backend(),
    "jax_devices": [str(d) for d in jax.devices()],
    "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
payload["device_name"] = torch.cuda.get_device_name(0)
payload["device_capability"] = torch.cuda.get_device_capability(0)
payload["torch_arch_list"] = torch.cuda.get_arch_list()
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit(f"sm_70 missing from torch arch list: {torch.cuda.get_arch_list()}")
if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX backend is not gpu: {jax.default_backend()}")
x = torch.randn(64, 64, device="cuda")
payload["torch_probe_sum"] = float((x @ x).sum().detach().cpu())
payload["jax_probe_sum"] = float(jnp.sum(jnp.ones((64, 64)) @ jnp.ones((64, 64))).block_until_ready())
print(json.dumps(payload, indent=2))
