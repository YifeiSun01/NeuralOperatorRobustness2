"""JAX utilities for the FNO Darcy-flow benchmark.

The original FNO data generator solves

    -div(a(x) grad u(x)) = 1,   u|boundary = 0

with a second-order finite-difference discretization.  This module keeps the
same finite-difference spirit but exposes the solve as a differentiable,
matrix-free JAX CG solve so it can be used inside attacks.
"""

from __future__ import annotations

import os
import sysconfig
from functools import partial
from pathlib import Path
from typing import Callable


def _configure_jax_cuda_toolchain() -> None:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return
    cuda_nvcc = Path(purelib) / "nvidia" / "cuda_nvcc"
    if not cuda_nvcc.exists():
        return
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_nvcc}")
    bin_dir = str(cuda_nvcc / "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


_configure_jax_cuda_toolchain()

import jax
import jax.numpy as jnp
from jax.scipy.fft import idctn
from jax.scipy.sparse.linalg import cg


Array = jax.Array


def binary_from_latent(latent: Array, low: float = 3.0, high: float = 12.0) -> Array:
    """Threshold a latent Gaussian random field into the two-phase coefficient."""

    return jnp.where(latent >= 0.0, high, low)


def soft_binary_from_latent(
    latent: Array,
    low: float = 3.0,
    high: float = 12.0,
    beta: float = 8.0,
) -> Array:
    """Differentiable relaxation of the hard Darcy coefficient threshold."""

    return low + (high - low) * jax.nn.sigmoid(beta * latent)


def make_grf_batch_fn(
    *,
    nx: int,
    alpha: float = 2.0,
    tau: float = 3.0,
    low: float = 3.0,
    high: float = 12.0,
    dtype=jnp.float32,
    binary: bool = True,
    beta: float = 8.0,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Return a jitted sampler for the Neumann-Laplacian GRF used by FNO Darcy.

    The sampler accepts a base PRNG key and a vector of integer sample indices.
    Folding the key with the index keeps samples deterministic even if the
    generation batch size changes.
    """

    k1, k2 = jnp.meshgrid(jnp.arange(nx), jnp.arange(nx), indexing="ij")
    spectrum = tau ** (alpha - 1.0) * (
        jnp.pi**2 * (k1.astype(dtype) ** 2 + k2.astype(dtype) ** 2) + tau**2
    ) ** (-alpha / 2.0)
    spectrum = spectrum.astype(dtype)

    @jax.jit
    def sample(base_key: Array, sample_indices: Array) -> tuple[Array, Array]:
        sample_indices = sample_indices.astype(jnp.uint32)

        def one(index):
            key = jax.random.fold_in(base_key, index)
            xi = jax.random.normal(key, (nx, nx), dtype=dtype)
            coeffs = nx * spectrum * xi
            coeffs = coeffs.at[0, 0].set(0.0)
            latent = idctn(coeffs, type=2, axes=(-2, -1), norm="ortho")
            if binary:
                a = binary_from_latent(latent, low=low, high=high)
            else:
                a = soft_binary_from_latent(latent, low=low, high=high, beta=beta)
            return latent.astype(dtype), a.astype(dtype)

        latent_batch, a_batch = jax.vmap(one)(sample_indices)
        return latent_batch, a_batch

    return sample


def _darcy_matvec_one(a: Array, u_int: Array) -> Array:
    """Apply the finite-difference Darcy operator to one interior solution."""

    n = a.shape[0]
    h = 1.0 / (n - 1.0)
    u = jnp.pad(u_int, ((1, 1), (1, 1)), mode="constant")
    center = u[1:-1, 1:-1]

    a_center = a[1:-1, 1:-1]
    a_e = 0.5 * (a_center + a[2:, 1:-1])
    a_w = 0.5 * (a_center + a[:-2, 1:-1])
    a_n = 0.5 * (a_center + a[1:-1, 2:])
    a_s = 0.5 * (a_center + a[1:-1, :-2])

    out = (
        (a_e + a_w + a_n + a_s) * center
        - a_e * u[2:, 1:-1]
        - a_w * u[:-2, 1:-1]
        - a_n * u[1:-1, 2:]
        - a_s * u[1:-1, :-2]
    )
    return out / (h * h)


def _solve_one(
    a: Array,
    *,
    forcing_value: float,
    tol: float,
    atol: float,
    maxiter: int | None,
) -> Array:
    rhs = jnp.ones((a.shape[0] - 2, a.shape[1] - 2), dtype=a.dtype) * forcing_value

    def matvec(u_int):
        return _darcy_matvec_one(a, u_int)

    u_int, _ = cg(matvec, rhs, tol=tol, atol=atol, maxiter=maxiter)
    return jnp.pad(u_int, ((1, 1), (1, 1)), mode="constant")


def make_darcy_solve_fn(
    *,
    forcing_value: float = 1.0,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> Callable[[Array], Array]:
    """Return a jitted batched Darcy solver.

    JAX differentiates this CG solve by implicit differentiation through a
    second linear solve, which is the right primitive for attacks that need
    gradients of ``solver(A)`` with respect to the coefficient field ``A``.
    """

    @partial(jax.jit, donate_argnums=())
    def solve_batch(a_batch: Array) -> Array:
        a_batch = jnp.asarray(a_batch)
        return jax.vmap(
            partial(
                _solve_one,
                forcing_value=forcing_value,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
            )
        )(a_batch)

    return solve_batch


def residual_norm(a_batch: Array, u_batch: Array, forcing_value: float = 1.0) -> Array:
    """Return per-sample interior residual norms for diagnostics."""

    def one(a, u):
        rhs = jnp.ones_like(u[1:-1, 1:-1]) * forcing_value
        res = _darcy_matvec_one(a, u[1:-1, 1:-1]) - rhs
        return jnp.linalg.norm(res.reshape(-1)) / jnp.maximum(jnp.linalg.norm(rhs.reshape(-1)), 1e-12)

    return jax.vmap(one)(a_batch, u_batch)
