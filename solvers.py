# torch_spectral_solvers.py

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _complex_roots_of_unity(
    num_circle_points: int,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Match modified Exponax's ETDRK contour points.
    theta = 2.0 * math.pi * (torch.arange(1, num_circle_points + 1, device=device, dtype=dtype) - 0.5)
    theta = theta / num_circle_points
    return radius * torch.exp(1j * theta)


def _broadcast_linear_operator(linear_operator: torch.Tensor, num_circle_points: int) -> torch.Tensor:
    # Add one trailing contour dimension.
    return linear_operator[..., None].to(torch.complex64 if linear_operator.dtype == torch.complex64 else torch.complex128)


class ETDRK4:
    def __init__(
        self,
        dt: float,
        linear_operator: torch.Tensor,
        nonlinear_fun,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.dt = float(dt)
        self.linear_operator = linear_operator
        self.nonlinear_fun = nonlinear_fun

        device = linear_operator.device
        real_dtype = torch.float32 if linear_operator.dtype == torch.complex64 else torch.float64

        L = linear_operator
        hL = self.dt * L

        self.exp_term = torch.exp(hL)
        self.half_exp_term = torch.exp(0.5 * hL)

        roots = _complex_roots_of_unity(
            num_circle_points=num_circle_points,
            radius=circle_radius,
            device=device,
            dtype=real_dtype,
        )

        LR = _broadcast_linear_operator(hL, num_circle_points) + roots

        self.coef_1 = self.dt * torch.mean((torch.exp(LR / 2.0) - 1.0) / LR, dim=-1).real

        self.coef_2 = self.coef_1
        self.coef_3 = self.coef_1

        self.coef_4 = self.dt * torch.mean(
            (-4.0 - LR + torch.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3,
            dim=-1,
        ).real

        self.coef_5 = self.dt * torch.mean(
            (2.0 + LR + torch.exp(LR) * (-2.0 + LR)) / LR**3,
            dim=-1,
        ).real

        self.coef_6 = self.dt * torch.mean(
            (-4.0 - 3.0 * LR - LR**2 + torch.exp(LR) * (4.0 - LR)) / LR**3,
            dim=-1,
        ).real

    def step_fourier(self, u_hat: torch.Tensor) -> torch.Tensor:
        # Cox-Matthews ETDRK4 step in Fourier space.
        u_nonlin_hat = self.nonlinear_fun(u_hat)

        u_stage_1_hat = self.half_exp_term * u_hat + self.coef_1 * u_nonlin_hat
        u_stage_1_nonlin_hat = self.nonlinear_fun(u_stage_1_hat)

        u_stage_2_hat = self.half_exp_term * u_hat + self.coef_2 * u_stage_1_nonlin_hat
        u_stage_2_nonlin_hat = self.nonlinear_fun(u_stage_2_hat)

        u_stage_3_hat = self.half_exp_term * u_stage_1_hat + self.coef_3 * (
            2.0 * u_stage_2_nonlin_hat - u_nonlin_hat
        )
        u_stage_3_nonlin_hat = self.nonlinear_fun(u_stage_3_hat)

        u_next_hat = (
            self.exp_term * u_hat
            + self.coef_4 * u_nonlin_hat
            + self.coef_5 * 2.0 * (u_stage_1_nonlin_hat + u_stage_2_nonlin_hat)
            + self.coef_6 * u_stage_3_nonlin_hat
        )

        return u_next_hat


def build_1d_wavenumbers(
    num_points: int,
    domain_extent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # rfftfreq returns cycles per unit length. Multiply by 2*pi for angular wavenumbers.
    dx = domain_extent / num_points
    return 2.0 * math.pi * torch.fft.rfftfreq(num_points, d=dx, device=device, dtype=dtype)


def build_2d_wavenumbers(
    num_points: int,
    domain_extent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Full FFT frequencies on the first spatial axis, real FFT frequencies on the second.
    dx = domain_extent / num_points
    kx = 2.0 * math.pi * torch.fft.fftfreq(num_points, d=dx, device=device, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.rfftfreq(num_points, d=dx, device=device, dtype=dtype)
    return kx[:, None], ky[None, :]


def dealias_mask_1d(
    num_points: int,
    dealiasing_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    # Keep low Fourier modes according to the given fraction.
    mode_ids = torch.arange(num_points // 2 + 1, device=device)
    cutoff = int(math.floor(dealiasing_fraction * (num_points // 2)))
    return mode_ids <= cutoff


def dealias_mask_2d(
    num_points: int,
    dealiasing_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    # 2D low-pass mask for rfft2 output.
    kx_ids = torch.fft.fftfreq(num_points, d=1.0, device=device) * num_points
    ky_ids = torch.fft.rfftfreq(num_points, d=1.0, device=device) * num_points

    cutoff = dealiasing_fraction * (num_points / 2.0)

    mask_x = torch.abs(kx_ids[:, None]) <= cutoff
    mask_y = torch.abs(ky_ids[None, :]) <= cutoff

    return mask_x & mask_y


@dataclass
class Burgers1DETDRK4:
    num_points: int
    domain_extent: float
    dt: float
    diffusivity: float
    convection_scale: float = 1.0
    conservative: bool = False
    dealiasing_fraction: float = 2.0 / 3.0
    num_circle_points: int = 16
    circle_radius: float = 1.0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128

        k = build_1d_wavenumbers(
            num_points=self.num_points,
            domain_extent=self.domain_extent,
            device=self.device,
            dtype=self.dtype,
        )

        self.derivative_operator = (1j * k).to(complex_dtype)
        self.laplace_operator = -(k**2).to(complex_dtype)

        self.linear_operator = self.diffusivity * self.laplace_operator

        self.mask = dealias_mask_1d(
            num_points=self.num_points,
            dealiasing_fraction=self.dealiasing_fraction,
            device=self.device,
        )

        self.integrator = ETDRK4(
            dt=self.dt,
            linear_operator=self.linear_operator,
            nonlinear_fun=self.nonlinear_fun,
            num_circle_points=self.num_circle_points,
            circle_radius=self.circle_radius,
        )

    def fft(self, u: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(u, dim=-1)

    def ifft(self, u_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft(u_hat, n=self.num_points, dim=-1)

    def dealias(self, u_hat: torch.Tensor) -> torch.Tensor:
        return u_hat * self.mask

    def nonlinear_fun(self, u_hat: torch.Tensor) -> torch.Tensor:
        u_hat_dealiased = self.dealias(u_hat)

        if self.conservative:
            u = self.ifft(u_hat_dealiased)
            flux = 0.5 * u**2
            flux_hat = self.fft(flux)
            return -self.convection_scale * self.derivative_operator * self.dealias(flux_hat)

        u = self.ifft(u_hat_dealiased)
        u_x = self.ifft(self.derivative_operator * u_hat_dealiased)
        convection = u * u_x
        return -self.convection_scale * self.fft(convection)

    def step(self, u: torch.Tensor) -> torch.Tensor:
        # Input shape: (N,) or (B, N). Output shape is the same.
        input_was_unbatched = u.ndim == 1

        if input_was_unbatched:
            u = u[None, :]

        u = u.to(device=self.device, dtype=self.dtype)
        u_hat = self.fft(u)
        u_next_hat = self.integrator.step_fourier(u_hat)
        u_next = self.ifft(u_next_hat)

        if input_was_unbatched:
            u_next = u_next[0]

        return u_next

    def repeat(self, u0: torch.Tensor, steps: int) -> torch.Tensor:
        u = u0
        for _ in range(int(steps)):
            u = self.step(u)
        return u


def solve_burgers_final_batch(
    u0_batch: torch.Tensor,
    t_final: float,
    dt: float = 0.001,
    domain_extent: float = 2.0,
    diffusivity: float = 1e-3,
    convection_scale: float = 1.0,
    conservative: bool = False,
    dealiasing_fraction: float = 2.0 / 3.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # Input shape: (B, N). Output shape: (B, N).
    if u0_batch.ndim != 2:
        raise ValueError(f"Expected u0_batch shape (B, N), got {tuple(u0_batch.shape)}")

    num_points = u0_batch.shape[-1]
    steps = int(round(t_final / dt))

    solver = Burgers1DETDRK4(
        num_points=num_points,
        domain_extent=domain_extent,
        dt=dt,
        diffusivity=diffusivity,
        convection_scale=convection_scale,
        conservative=conservative,
        dealiasing_fraction=dealiasing_fraction,
        device=device,
        dtype=dtype,
    )

    return solver.repeat(u0_batch, steps)


@dataclass
class NavierStokesVorticity2DZongyiETDRK4:
    num_points: int
    domain_extent: float
    dt: float
    diffusivity: float
    drag: float = 0.0
    vorticity_convection_scale: float = 1.0
    injection_scale: float = 1.0
    dealiasing_fraction: float = 2.0 / 3.0
    num_circle_points: int = 16
    circle_radius: float = 1.0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128

        kx, ky = build_2d_wavenumbers(
            num_points=self.num_points,
            domain_extent=self.domain_extent,
            device=self.device,
            dtype=self.dtype,
        )

        self.derivative_x = (1j * kx).to(complex_dtype)
        self.derivative_y = (1j * ky).to(complex_dtype)

        laplace = -(kx**2 + ky**2)
        self.laplace_operator = laplace.to(complex_dtype)

        self.linear_operator = self.diffusivity * self.laplace_operator + self.drag

        self.inv_laplacian = torch.zeros_like(self.laplace_operator)
        nonzero = self.laplace_operator != 0
        self.inv_laplacian[nonzero] = 1.0 / self.laplace_operator[nonzero]

        self.mask = dealias_mask_2d(
            num_points=self.num_points,
            dealiasing_fraction=self.dealiasing_fraction,
            device=self.device,
        )

        self.injection = self.build_zongyi_forcing().to(complex_dtype)

        self.integrator = ETDRK4(
            dt=self.dt,
            linear_operator=self.linear_operator,
            nonlinear_fun=self.nonlinear_fun,
            num_circle_points=self.num_circle_points,
            circle_radius=self.circle_radius,
        )

    def fft(self, u: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(u, dim=(-2, -1))

    def ifft(self, u_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(u_hat, s=(self.num_points, self.num_points), dim=(-2, -1))

    def dealias(self, u_hat: torch.Tensor) -> torch.Tensor:
        return u_hat * self.mask

    def build_zongyi_forcing(self) -> torch.Tensor:
        # Build forcing on [0, 1), matching the modified Exponax helper.
        t = torch.linspace(
            0.0,
            1.0,
            self.num_points + 1,
            device=self.device,
            dtype=self.dtype,
        )[:-1]

        x, y = torch.meshgrid(t, t, indexing="ij")
        f_physical = 0.1 * (
            torch.sin(2.0 * math.pi * (x + y))
            + torch.cos(2.0 * math.pi * (x + y))
        )

        return self.injection_scale * torch.fft.rfft2(f_physical, dim=(-2, -1))

    def nonlinear_fun(self, omega_hat: torch.Tensor) -> torch.Tensor:
        omega_hat_dealiased = self.dealias(omega_hat)

        stream_function_hat = self.inv_laplacian * omega_hat_dealiased

        velocity_x_hat = self.derivative_y * stream_function_hat
        velocity_y_hat = -self.derivative_x * stream_function_hat

        omega_x_hat = self.derivative_x * omega_hat_dealiased
        omega_y_hat = self.derivative_y * omega_hat_dealiased

        velocity_x = self.ifft(self.dealias(velocity_x_hat))
        velocity_y = self.ifft(self.dealias(velocity_y_hat))

        omega_x = self.ifft(self.dealias(omega_x_hat))
        omega_y = self.ifft(self.dealias(omega_y_hat))

        convection = velocity_x * omega_x + velocity_y * omega_y
        convection_hat = self.fft(convection)

        return -self.vorticity_convection_scale * convection_hat + self.injection

    def step(self, omega: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, N) or (B, N, N). Output shape is the same.
        input_was_unbatched = omega.ndim == 2

        if input_was_unbatched:
            omega = omega[None, :, :]

        omega = omega.to(device=self.device, dtype=self.dtype)

        omega_hat = self.fft(omega)
        omega_next_hat = self.integrator.step_fourier(omega_hat)
        omega_next = self.ifft(omega_next_hat)

        if input_was_unbatched:
            omega_next = omega_next[0]

        return omega_next

    def repeat(self, omega0: torch.Tensor, steps: int) -> torch.Tensor:
        omega = omega0
        for _ in range(int(steps)):
            omega = self.step(omega)
        return omega


def solve_ns_zongyi_rollout_batch(
    u0_batch: torch.Tensor,
    t_final: int = 20,
    fixed_step: float = 0.005,
    domain_extent: float = 1.0,
    diffusivity: float = 1e-5,
    drag: float = 0.0,
    vorticity_convection_scale: float = 1.0,
    injection_scale: float = 1.0,
    dealiasing_fraction: float = 2.0 / 3.0,
    apply_exponax_orientation_transform: bool = True,
    return_time_last: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # Input shape: (B, N, N).
    # If return_time_last=True, output shape: (B, N, N, t_final + 1).
    # If return_time_last=False, output shape: (B, t_final + 1, N, N).
    if u0_batch.ndim != 3:
        raise ValueError(f"Expected u0_batch shape (B, N, N), got {tuple(u0_batch.shape)}")

    num_points = u0_batch.shape[-1]

    if u0_batch.shape[-2] != num_points:
        raise ValueError(f"Expected square input, got {tuple(u0_batch.shape)}")

    if abs(round(1.0 / fixed_step) - (1.0 / fixed_step)) > 1e-12:
        raise ValueError("fixed_step must divide one second exactly.")

    steps_per_second = int(round(1.0 / fixed_step))

    solver = NavierStokesVorticity2DZongyiETDRK4(
        num_points=num_points,
        domain_extent=domain_extent,
        dt=fixed_step,
        diffusivity=diffusivity,
        drag=drag,
        vorticity_convection_scale=vorticity_convection_scale,
        injection_scale=injection_scale,
        dealiasing_fraction=dealiasing_fraction,
        num_circle_points=16,
        device=device,
        dtype=dtype,
    )

    u = u0_batch.to(device=solver.device, dtype=dtype)

    if apply_exponax_orientation_transform:
        u = torch.rot90(torch.flip(u, dims=(-2,)), k=3, dims=(-2, -1))

    frames = [u]

    for _ in range(int(t_final)):
        for _ in range(steps_per_second):
            u = solver.step(u)

        frames.append(u)

    seq = torch.stack(frames, dim=1)

    if apply_exponax_orientation_transform:
        # The modified Exponax Zongyi generator swaps the two spatial axes
        # before saving sparse integer-second frames.
        seq = torch.swapaxes(seq, -1, -2)

    if return_time_last:
        seq = seq.permute(0, 2, 3, 1).contiguous()

    return seq


def spectral_upsample_2d(
    x: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    # Input shape: (B, H, W). Output shape: (B, target_size, target_size).
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B, H, W), got {tuple(x.shape)}")

    batch_size, old_h, old_w = x.shape

    if old_h != old_w:
        raise ValueError(f"Expected square input, got {tuple(x.shape)}")

    if target_size < old_h:
        raise ValueError("target_size must be greater than or equal to the current size.")

    x_hat = torch.fft.rfft2(x, dim=(-2, -1))

    new_hat = torch.zeros(
        batch_size,
        target_size,
        target_size // 2 + 1,
        dtype=x_hat.dtype,
        device=x.device,
    )

    old_half = old_h // 2
    new_hat[:, :old_half, : old_w // 2 + 1] = x_hat[:, :old_half, :]
    new_hat[:, -old_half:, : old_w // 2 + 1] = x_hat[:, -old_half:, :]

    scale = (target_size * target_size) / (old_h * old_w)
    y = torch.fft.irfft2(new_hat, s=(target_size, target_size), dim=(-2, -1))

    return scale * y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Burgers example.
    burgers_x = torch.randn(8, 1024, device=device)
    burgers_y = solve_burgers_final_batch(
        burgers_x,
        t_final=1.0,
        dt=0.001,
        domain_extent=2.0,
        diffusivity=1e-3,
        conservative=False,
        device=device,
    )
    print("Burgers output:", burgers_y.shape)

    # 2D NS example.
    ns_x = torch.randn(2, 256, 256, device=device)
    ns_y = solve_ns_zongyi_rollout_batch(
        ns_x,
        t_final=20,
        fixed_step=0.005,
        domain_extent=1.0,
        diffusivity=1e-5,
        return_time_last=True,
        device=device,
    )
    print("NS output:", ns_y.shape)
