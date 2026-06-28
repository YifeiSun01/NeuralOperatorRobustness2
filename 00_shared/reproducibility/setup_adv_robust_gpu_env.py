#!/usr/bin/env python3
"""Create or verify the adv_robust GPU environment.

This is the project entry point for rebuilding the local virtual environment.
It is intentionally strict: if PyTorch or JAX cannot execute on GPU, the script
fails instead of allowing a CPU fallback.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = PROJECT_ROOT / "adv_robust"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
REQUIRED_TORCH_LINE = "torch==2.8.0+cu126"
REQUIRED_TORCH_INDEX = "--extra-index-url https://download.pytorch.org/whl/cu126"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def capture(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    print("[check] " + " ".join(cmd), flush=True)
    return subprocess.check_output(cmd, cwd=PROJECT_ROOT, env=env, text=True, stderr=subprocess.STDOUT)


def expected_arch_from_nvidia_smi() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is not available; cannot verify the GPU before environment setup.")
    out = capture(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    first = next((line.strip() for line in out.splitlines() if line.strip()), "")
    if not first or "," not in first:
        raise RuntimeError(f"Could not parse nvidia-smi compute capability output: {out!r}")
    name, cap = [part.strip() for part in first.split(",", 1)]
    match = re.match(r"^(\d+)\.(\d+)$", cap)
    if not match:
        raise RuntimeError(f"Could not parse GPU compute capability {cap!r} from nvidia-smi output {out!r}")
    arch = f"sm_{match.group(1)}{match.group(2)}"
    print(f"[gpu] detected {name}, compute capability {cap}, required PyTorch arch {arch}", flush=True)
    return name, arch


def check_requirements_policy(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []
    if REQUIRED_TORCH_INDEX not in text:
        missing.append(REQUIRED_TORCH_INDEX)
    if REQUIRED_TORCH_LINE not in text:
        missing.append(REQUIRED_TORCH_LINE)
    if missing:
        raise RuntimeError(
            "requirements.txt does not contain the verified V100-compatible PyTorch pins. "
            "Missing: " + ", ".join(missing)
        )


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def ensure_venv(venv: Path, python_executable: str) -> Path:
    py = venv_python(venv)
    if py.exists():
        print(f"[venv] using existing virtualenv: {venv}", flush=True)
        return py
    python_path = shutil.which(python_executable)
    if python_path is None:
        raise RuntimeError(f"Could not find {python_executable!r}. Install Python 3.12 or pass --python.")
    print(f"[venv] creating {venv} with {python_path}", flush=True)
    run([python_path, "-m", "venv", str(venv)])
    if not py.exists():
        raise RuntimeError(f"Virtualenv creation did not produce {py}")
    return py


def install_requirements(py: Path, requirements: Path) -> None:
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(py), "-m", "pip", "install", "-r", str(requirements)])


def verification_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def verify_gpu_runtime(py: Path, expected_arch: str) -> None:
    code = r"""
import os
import sys
import torch
import jax
import jax.numpy as jnp

print("python", sys.executable)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("torch_cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")

device = torch.device("cuda:0")
name = torch.cuda.get_device_name(device)
cap = torch.cuda.get_device_capability(device)
required_arch = f"sm_{cap[0]}{cap[1]}"
expected_arch = os.environ.get("EXPECTED_TORCH_ARCH")
arch_list = list(torch.cuda.get_arch_list())
print("torch_device", name)
print("torch_capability", cap)
print("torch_required_arch", required_arch)
print("torch_expected_arch", expected_arch)
print("torch_arch_list", arch_list)
if expected_arch and required_arch != expected_arch:
    raise SystemExit(f"PyTorch active arch {required_arch} does not match nvidia-smi expected arch {expected_arch}")
if required_arch not in arch_list:
    raise SystemExit(f"PyTorch wheel does not include {required_arch}; arch list is {arch_list}")

x = torch.eye(32, device=device, dtype=torch.float32)
y = x @ x
torch.cuda.synchronize(device)
print("torch_cuda_matmul", float(y[0, 0].detach().cpu()))

backend = jax.default_backend()
devices = jax.devices()
gpu_devices = jax.devices("gpu")
print("jax_backend", backend)
print("jax_devices", devices)
print("jax_gpu_devices", gpu_devices)
if backend != "gpu" or not gpu_devices:
    raise SystemExit(f"JAX is not on GPU: backend={backend}, devices={devices}")

jx = jnp.eye(16)
jy = jx @ jx
jy.block_until_ready()
print("jax_gpu_matmul", float(jy[0, 0]))
print("gpu_runtime_verification", "PASS")
"""
    env = verification_env()
    env["EXPECTED_TORCH_ARCH"] = expected_arch
    run([str(py), "-c", code], env=env)
    run([str(py), "-m", "pip", "check"], env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--verify-only", action="store_true", help="Skip install and only run strict GPU checks.")
    parser.add_argument(
        "--skip-requirements-policy-check",
        action="store_true",
        help="Allow requirements.txt to deviate from the verified torch==2.8.0+cu126 policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.requirements.is_absolute():
        args.requirements = PROJECT_ROOT / args.requirements
    if not args.venv.is_absolute():
        args.venv = PROJECT_ROOT / args.venv

    _, expected_arch = expected_arch_from_nvidia_smi()
    if not args.skip_requirements_policy_check:
        check_requirements_policy(args.requirements)

    py = ensure_venv(args.venv, args.python)
    if not args.verify_only:
        install_requirements(py, args.requirements)
    verify_gpu_runtime(py, expected_arch)
    print("[ok] adv_robust GPU environment is installed and verified. No CPU fallback was used.", flush=True)


if __name__ == "__main__":
    main()
