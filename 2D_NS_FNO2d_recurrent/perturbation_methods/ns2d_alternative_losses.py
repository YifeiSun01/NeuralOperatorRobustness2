"""Alternative differentiable final-state metrics for NS2D loss3 attacks.

The attack driver keeps the existing loss3/AW data path:

    FNO final state vs. target solver final state

This module only changes the metric used to compare those two final-state
fields.  Alignment metrics estimate a bounded transform on detached final
fields, then apply the fixed transform to the live tensors so gradients still
flow to the attack input without retaining an expensive registration graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


LOSS3_METRIC_CHOICES = (
    "qnorm",
    "dists",
    "ms_ssim",
    "scattering2d",
    "affine_dists",
    "local_warp_dists",
    "homography_dists",
    "tps_dists",
    "elastic_dists",
    "svf_dists",
)


def _batch_flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _batch_norm(x: torch.Tensor, order: float) -> torch.Tensor:
    flat = _batch_flat(x)
    if math.isinf(float(order)):
        return flat.abs().amax(dim=1)
    return torch.linalg.vector_norm(flat, ord=float(order), dim=1)


def _as_image(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [B,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def _repeat_to_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)


def _normalize_pair(pred: torch.Tensor, target: torch.Tensor, mode: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    pred_img = _as_image(pred)
    target_img = _as_image(target)
    if mode == "none":
        return pred_img, target_img
    if mode != "pair_minmax_detached":
        raise ValueError(f"Unknown loss3 image normalization mode: {mode}")
    both = torch.cat([pred_img, target_img], dim=1)
    reduce_dims = tuple(range(1, both.ndim))
    lo = both.amin(dim=reduce_dims, keepdim=True).detach()
    hi = both.amax(dim=reduce_dims, keepdim=True).detach()
    scale = (hi - lo).clamp_min(float(eps))
    return (pred_img - lo) / scale, (target_img - lo) / scale


def _require_kornia_transform():
    try:
        import kornia.geometry.transform as KGT
    except ImportError as exc:
        raise ImportError(
            "Official warp metrics require Kornia. Install it with `pip install kornia`."
        ) from exc
    return KGT


def _warp_affine_kornia(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    KGT = _require_kornia_transform()
    return KGT.warp_affine(
        x,
        matrix,
        dsize=(x.shape[-2], x.shape[-1]),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _warp_perspective_kornia(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    KGT = _require_kornia_transform()
    return KGT.warp_perspective(
        x,
        matrix,
        dsize=(x.shape[-2], x.shape[-1]),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _warp_tps_kornia(x: torch.Tensor, points_src: torch.Tensor, points_dst: torch.Tensor) -> torch.Tensor:
    KGT = _require_kornia_transform()
    # warp_image_tps samples input coordinates for each output coordinate, so use
    # the reverse transform, following Kornia's documented TPS example.
    kernel_weights, affine_weights = KGT.get_tps_transform(points_dst, points_src)
    return KGT.warp_image_tps(
        x,
        points_src,
        kernel_weights,
        affine_weights,
        align_corners=False,
        padding_mode="border",
    )


def _warp_elastic_kornia(
    x: torch.Tensor,
    noise: torch.Tensor,
    kernel_size: int,
    smooth_passes: int,
) -> torch.Tensor:
    KGT = _require_kornia_transform()
    kernel = max(3, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    sigma_value = max(1.0, (kernel / 3.0) * max(1, int(smooth_passes)))
    return KGT.elastic_transform2d(
        x,
        noise,
        kernel_size=(kernel, kernel),
        sigma=(sigma_value, sigma_value),
        alpha=(1.0, 1.0),
        align_corners=False,
        mode="bilinear",
        padding_mode="border",
    )


def _per_sample_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).flatten(1).square().mean(dim=1)


def _smoothness_per_sample(disp: torch.Tensor) -> torch.Tensor:
    dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
    dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    return dx.square().flatten(1).mean(dim=1) + dy.square().flatten(1).mean(dim=1)




def _normalized_control_grid(batch_size: int, grid_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx, yy], dim=-1).reshape(1, grid_size * grid_size, 2)
    return points.repeat(batch_size, 1, 1)


def _detach_diag(diag: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in diag.items():
        if torch.is_tensor(value):
            out[key] = value.detach()
        else:
            out[key] = value
    return out


@dataclass
class MetricResult:
    values: torch.Tensor
    diagnostics: dict[str, Any]


class _DISTSAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            import piq
        except ImportError as exc:
            raise ImportError("loss3 metric 'dists' requires `pip install piq`.") from exc
        self.loss_none = piq.DISTS(reduction="none")
        self.loss_mean = piq.DISTS(reduction="mean")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x3 = _repeat_to_three_channels(x)
        y3 = _repeat_to_three_channels(y)
        values = self.loss_none(x3, y3)
        if torch.is_tensor(values) and values.ndim > 0 and values.shape[0] == x3.shape[0]:
            return values.reshape(x3.shape[0], -1).mean(dim=1)
        if x3.shape[0] == 1:
            return values.reshape(1)
        per_sample = []
        for idx in range(x3.shape[0]):
            per_sample.append(self.loss_mean(x3[idx : idx + 1], y3[idx : idx + 1]).reshape(()))
        return torch.stack(per_sample)


class _MSSSIMAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from pytorch_msssim import ms_ssim
        except ImportError as exc:
            raise ImportError("loss3 metric 'ms_ssim' requires `pip install pytorch-msssim`.") from exc
        self.ms_ssim = ms_ssim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        score = self.ms_ssim(x, y, data_range=1.0, size_average=False)
        return 1.0 - score.reshape(x.shape[0], -1).mean(dim=1)


class _Scattering2DAdapter(nn.Module):
    def __init__(self, j: int) -> None:
        super().__init__()
        self.j = int(j)
        self._shape: tuple[int, int] | None = None
        self._scattering: Any = None

    def _get_scattering(self, height: int, width: int, device: torch.device):
        if self._scattering is None or self._shape != (height, width):
            try:
                from kymatio.torch import Scattering2D
            except ImportError as exc:
                raise ImportError("loss3 metric 'scattering2d' requires `pip install kymatio`.") from exc
            self._shape = (height, width)
            self._scattering = Scattering2D(J=self.j, shape=(height, width)).to(device)
        return self._scattering

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = x.shape
        scattering = self._get_scattering(height, width, x.device)
        losses = []
        for channel in range(channels):
            fx = scattering(x[:, channel])
            fy = scattering(y[:, channel])
            losses.append((fx - fy).reshape(x.shape[0], -1).square().mean(dim=1))
        return torch.stack(losses, dim=0).mean(dim=0)


class Loss3MetricComputer(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.name = str(args.loss3_metric)
        self.q_order = float(args.q_order)
        self.normalization = str(args.loss3_image_normalization)
        self.eps = float(args.loss3_metric_eps)
        self.align_objective = str(args.loss3_align_objective)

        self.dists: _DISTSAdapter | None = None
        self.ms_ssim: _MSSSIMAdapter | None = None
        self.scattering: _Scattering2DAdapter | None = None
        self._monai_warp_layer: Any = None
        self._monai_dvf2ddf_layer: Any = None
        self._monai_dvf2ddf_steps: int | None = None

        self.affine_inner_steps = int(args.loss3_affine_inner_steps)
        self.affine_lr = float(args.loss3_affine_lr)
        self.affine_max_shift_ratio = float(args.loss3_affine_max_shift_ratio)
        self.affine_max_angle = float(args.loss3_affine_max_angle_deg) * math.pi / 180.0
        self.affine_max_log_scale = float(args.loss3_affine_max_log_scale)
        self.affine_reg_weight = float(args.loss3_affine_reg_weight)

        self.local_grid_size = int(args.loss3_local_grid_size)
        self.local_inner_steps = int(args.loss3_local_inner_steps)
        self.local_lr = float(args.loss3_local_lr)
        self.local_max_disp_ratio = float(args.loss3_local_max_disp_ratio)
        self.local_mag_weight = float(args.loss3_local_mag_weight)
        self.local_smooth_weight = float(args.loss3_local_smooth_weight)

        self.homography_inner_steps = int(args.loss3_homography_inner_steps)
        self.homography_lr = float(args.loss3_homography_lr)
        self.homography_max_corner_ratio = float(args.loss3_homography_max_corner_ratio)
        self.homography_reg_weight = float(args.loss3_homography_reg_weight)

        self.tps_grid_size = int(args.loss3_tps_grid_size)
        self.tps_inner_steps = int(args.loss3_tps_inner_steps)
        self.tps_lr = float(args.loss3_tps_lr)
        self.tps_max_disp_ratio = float(args.loss3_tps_max_disp_ratio)
        self.tps_offset_weight = float(args.loss3_tps_offset_weight)
        self.tps_smooth_weight = float(args.loss3_tps_smooth_weight)

        self.elastic_grid_size = int(args.loss3_elastic_grid_size)
        self.elastic_inner_steps = int(args.loss3_elastic_inner_steps)
        self.elastic_lr = float(args.loss3_elastic_lr)
        self.elastic_max_disp_ratio = float(args.loss3_elastic_max_disp_ratio)
        self.elastic_smooth_kernel = int(args.loss3_elastic_smooth_kernel)
        self.elastic_smooth_passes = int(args.loss3_elastic_smooth_passes)
        self.elastic_mag_weight = float(args.loss3_elastic_mag_weight)
        self.elastic_smooth_weight = float(args.loss3_elastic_smooth_weight)

        self.svf_grid_size = int(args.loss3_svf_grid_size)
        self.svf_inner_steps = int(args.loss3_svf_inner_steps)
        self.svf_lr = float(args.loss3_svf_lr)
        self.svf_max_vel_ratio = float(args.loss3_svf_max_vel_ratio)
        self.svf_int_steps = int(args.loss3_svf_int_steps)
        self.svf_mag_weight = float(args.loss3_svf_mag_weight)
        self.svf_smooth_weight = float(args.loss3_svf_smooth_weight)

        dists_metrics = {"dists", "affine_dists", "local_warp_dists", "homography_dists", "tps_dists", "elastic_dists", "svf_dists"}
        if self.name in dists_metrics or self.align_objective == "dists":
            self.dists = _DISTSAdapter()
        if self.name == "ms_ssim":
            self.ms_ssim = _MSSSIMAdapter()
        if self.name == "scattering2d":
            self.scattering = _Scattering2DAdapter(args.loss3_scattering_j)

    def _normalized_images(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _normalize_pair(pred, target, self.normalization, self.eps)

    def _dists_values(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.dists is None:
            self.dists = _DISTSAdapter().to(x.device)
        return self.dists(x, y)

    def _align_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.align_objective == "l2":
            return _per_sample_mse(x, y).mean()
        if self.align_objective == "dists":
            return self._dists_values(x, y).mean()
        raise ValueError(f"Unknown alignment objective: {self.align_objective}")

    def _ddf_yx_pixels_from_xy_normalized(self, disp_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
        ddf = torch.empty_like(disp_xy)
        ddf[:, 0] = disp_xy[:, 1] * ((height - 1) / 2.0)
        ddf[:, 1] = disp_xy[:, 0] * ((width - 1) / 2.0)
        return ddf

    def _xy_normalized_from_ddf_yx_pixels(self, ddf: torch.Tensor, height: int, width: int) -> torch.Tensor:
        disp_xy = torch.empty_like(ddf)
        disp_xy[:, 0] = ddf[:, 1] * (2.0 / max(1, width - 1))
        disp_xy[:, 1] = ddf[:, 0] * (2.0 / max(1, height - 1))
        return disp_xy

    def _warp_dense_monai_xy(self, image: torch.Tensor, disp_xy: torch.Tensor) -> torch.Tensor:
        try:
            from monai.networks.blocks import Warp
        except ImportError as exc:
            raise ImportError(
                "Official dense warp metrics require MONAI. Install it with `pip install monai`."
            ) from exc
        if self._monai_warp_layer is None:
            self._monai_warp_layer = Warp(mode="bilinear", padding_mode="border")
        self._monai_warp_layer = self._monai_warp_layer.to(device=image.device)
        ddf = self._ddf_yx_pixels_from_xy_normalized(disp_xy, image.shape[-2], image.shape[-1])
        return self._monai_warp_layer(image, ddf)

    def _integrate_velocity_monai_xy(self, velocity_xy: torch.Tensor) -> torch.Tensor:
        try:
            from monai.networks.blocks import DVF2DDF
        except ImportError as exc:
            raise ImportError(
                "Official SVF integration requires MONAI. Install it with `pip install monai`."
            ) from exc
        if self._monai_dvf2ddf_layer is None or self._monai_dvf2ddf_steps != self.svf_int_steps:
            self._monai_dvf2ddf_layer = DVF2DDF(
                num_steps=self.svf_int_steps,
                mode="bilinear",
                padding_mode="border",
            )
            self._monai_dvf2ddf_steps = self.svf_int_steps
        self._monai_dvf2ddf_layer = self._monai_dvf2ddf_layer.to(device=velocity_xy.device)
        height, width = velocity_xy.shape[-2:]
        dvf = self._ddf_yx_pixels_from_xy_normalized(velocity_xy, height, width)
        ddf = self._monai_dvf2ddf_layer(dvf)
        return self._xy_normalized_from_ddf_yx_pixels(ddf, height, width)

    def _affine_matrix(self, raw: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        KGT = _require_kornia_transform()
        tx = torch.tanh(raw[:, 0]) * self.affine_max_shift_ratio * 2.0
        ty = torch.tanh(raw[:, 1]) * self.affine_max_shift_ratio * 2.0
        angle = torch.tanh(raw[:, 2]) * self.affine_max_angle
        log_scale = torch.tanh(raw[:, 3]) * self.affine_max_log_scale
        scale = torch.exp(log_scale)
        center = torch.empty(raw.shape[0], 2, device=raw.device, dtype=raw.dtype)
        center[:, 0] = (width - 1) / 2.0
        center[:, 1] = (height - 1) / 2.0
        scale_xy = torch.stack([scale, scale], dim=1)
        matrix = KGT.get_rotation_matrix2d(center, angle * (180.0 / math.pi), scale_xy)
        matrix[:, 0, 2] += tx * ((width - 1) / 2.0)
        matrix[:, 1, 2] += ty * ((height - 1) / 2.0)

        diagnostics = {
            "translation_ratio": torch.sqrt((tx / 2.0).square() + (ty / 2.0).square()),
            "rotation_abs_deg": angle.abs() * (180.0 / math.pi),
            "scale_deviation": (scale - 1.0).abs(),
            "affine_reg": tx.square() + ty.square() + angle.square() + log_scale.square(),
            "official_warp_backend": "kornia.warp_affine",
        }
        return matrix, diagnostics

    def _estimate_affine(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_detached = pred_img.detach()
        target_detached = target_img.detach()
        _, _, height, width = pred_img.shape
        raw = torch.zeros(pred_img.shape[0], 4, device=pred_img.device, dtype=pred_img.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=self.affine_lr)
        for _ in range(max(0, self.affine_inner_steps)):
            matrix, diag = self._affine_matrix(raw, height, width)
            warped = _warp_affine_kornia(pred_detached, matrix)
            loss = self._align_loss(warped, target_detached) + self.affine_reg_weight * diag["affine_reg"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw.nan_to_num_(nan=0.0, posinf=6.0, neginf=-6.0)
                raw.clamp_(-6.0, 6.0)
        matrix, diag = self._affine_matrix(raw.detach(), height, width)
        return matrix.detach(), _detach_diag(diag)

    def _estimate_local_disp(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_detached = pred_img.detach()
        target_detached = target_img.detach()
        batch, _, height, width = pred_img.shape
        raw = torch.zeros(
            batch,
            2,
            self.local_grid_size,
            self.local_grid_size,
            device=pred_img.device,
            dtype=pred_img.dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([raw], lr=self.local_lr)
        for _ in range(max(0, self.local_inner_steps)):
            disp_low = torch.tanh(raw) * self.local_max_disp_ratio * 2.0
            disp = F.interpolate(disp_low, size=(height, width), mode="bilinear", align_corners=False)
            warped = self._warp_dense_monai_xy(pred_detached, disp)
            mag = disp.square().flatten(1).mean(dim=1)
            smooth = _smoothness_per_sample(disp)
            loss = self._align_loss(warped, target_detached) + self.local_mag_weight * mag.mean() + self.local_smooth_weight * smooth.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw.nan_to_num_(nan=0.0, posinf=6.0, neginf=-6.0)
                raw.clamp_(-6.0, 6.0)

        disp_low = torch.tanh(raw.detach()) * self.local_max_disp_ratio * 2.0
        disp = F.interpolate(disp_low, size=(height, width), mode="bilinear", align_corners=False).detach()
        disp_vectors = disp.permute(0, 2, 3, 1)
        disp_norm = torch.linalg.vector_norm(disp_vectors, ord=2, dim=-1)
        diagnostics = {
            "displacement_norm": disp.square().flatten(1).mean(dim=1).sqrt(),
            "max_displacement": disp_norm.flatten(1).amax(dim=1),
            "smoothness": _smoothness_per_sample(disp),
            "official_warp_backend": "monai.networks.blocks.Warp",
        }
        return disp, _detach_diag(diagnostics)

    def _homography_matrix_from_raw(self, raw: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        KGT = _require_kornia_transform()
        batch = raw.shape[0]
        src = torch.tensor(
            [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]],
            device=raw.device,
            dtype=raw.dtype,
        ).unsqueeze(0).repeat(batch, 1, 1)
        raw_offsets = torch.tanh(raw)
        offsets = torch.empty_like(raw_offsets)
        offsets[..., 0] = raw_offsets[..., 0] * self.homography_max_corner_ratio * float(width)
        offsets[..., 1] = raw_offsets[..., 1] * self.homography_max_corner_ratio * float(height)
        dst = src + offsets
        matrix = KGT.get_perspective_transform(src, dst)
        corner_norm = torch.sqrt((offsets[..., 0] / float(width)).square() + (offsets[..., 1] / float(height)).square())
        diag = {
            "corner_displacement_mean": corner_norm.mean(dim=1),
            "corner_displacement_max": corner_norm.amax(dim=1),
            "homography_reg": corner_norm.square().mean(dim=1),
            "official_warp_backend": "kornia.get_perspective_transform+warp_perspective",
        }
        return matrix, diag

    def _estimate_homography(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_detached = pred_img.detach()
        target_detached = target_img.detach()
        batch, _, height, width = pred_img.shape
        raw = torch.zeros(batch, 4, 2, device=pred_img.device, dtype=pred_img.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=self.homography_lr)
        for _ in range(max(0, self.homography_inner_steps)):
            matrix, diag = self._homography_matrix_from_raw(raw, height, width)
            warped = _warp_perspective_kornia(pred_detached, matrix)
            loss = self._align_loss(warped, target_detached) + self.homography_reg_weight * diag["homography_reg"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw.nan_to_num_(nan=0.0, posinf=6.0, neginf=-6.0)
                raw.clamp_(-6.0, 6.0)
        matrix, diag = self._homography_matrix_from_raw(raw.detach(), height, width)
        return matrix.detach(), _detach_diag(diag)

    def _tps_points_from_raw(self, raw: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = raw.shape[0]
        points_src = _normalized_control_grid(batch, self.tps_grid_size, raw.device, raw.dtype)
        offsets_grid = torch.tanh(raw) * self.tps_max_disp_ratio * 2.0
        offsets = offsets_grid.permute(0, 2, 3, 1).reshape(batch, -1, 2)
        points_dst = points_src + offsets
        offset_norm = torch.linalg.vector_norm(offsets, ord=2, dim=-1)
        diag = {
            "control_displacement_mean": offset_norm.mean(dim=1),
            "control_displacement_max": offset_norm.amax(dim=1),
            "tps_offset_reg": offsets_grid.square().flatten(1).mean(dim=1),
            "tps_smoothness": _smoothness_per_sample(offsets_grid),
            "official_warp_backend": "kornia.get_tps_transform+warp_image_tps",
        }
        return points_src, points_dst, diag

    def _estimate_tps(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        pred_detached = pred_img.detach()
        target_detached = target_img.detach()
        batch, _, height, width = pred_img.shape
        raw = torch.zeros(batch, 2, self.tps_grid_size, self.tps_grid_size, device=pred_img.device, dtype=pred_img.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=self.tps_lr)
        for _ in range(max(0, self.tps_inner_steps)):
            points_src, points_dst, diag = self._tps_points_from_raw(raw, height, width)
            warped = _warp_tps_kornia(pred_detached, points_src, points_dst)
            loss = (
                self._align_loss(warped, target_detached)
                + self.tps_offset_weight * diag["tps_offset_reg"].mean()
                + self.tps_smooth_weight * diag["tps_smoothness"].mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw.nan_to_num_(nan=0.0, posinf=6.0, neginf=-6.0)
                raw.clamp_(-6.0, 6.0)
        points_src, points_dst, diag = self._tps_points_from_raw(raw.detach(), height, width)
        return points_src.detach(), points_dst.detach(), _detach_diag(diag)

    def _elastic_noise_from_raw(self, raw: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        noise_low = torch.tanh(raw) * self.elastic_max_disp_ratio * 2.0
        noise = F.interpolate(noise_low, size=(height, width), mode="bilinear", align_corners=False)
        noise_vectors = noise.permute(0, 2, 3, 1)
        noise_norm = torch.linalg.vector_norm(noise_vectors, ord=2, dim=-1)
        elastic_mag_reg = noise.square().flatten(1).mean(dim=1)
        diag = {
            "elastic_mag_reg": elastic_mag_reg,
            "elastic_displacement_norm": elastic_mag_reg.clamp_min(1e-12).sqrt(),
            "elastic_max_displacement": noise_norm.flatten(1).amax(dim=1),
            "elastic_smoothness": _smoothness_per_sample(noise),
            "official_warp_backend": "kornia.elastic_transform2d",
        }
        return noise, diag

    def _estimate_elastic_disp(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_detached = pred_img.detach()
        target_detached = target_img.detach()
        batch, _, height, width = pred_img.shape
        raw = torch.zeros(batch, 2, self.elastic_grid_size, self.elastic_grid_size, device=pred_img.device, dtype=pred_img.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=self.elastic_lr)
        for _ in range(max(0, self.elastic_inner_steps)):
            noise, diag = self._elastic_noise_from_raw(raw, height, width)
            warped = _warp_elastic_kornia(pred_detached, noise, self.elastic_smooth_kernel, self.elastic_smooth_passes)
            loss = (
                self._align_loss(warped, target_detached)
                + self.elastic_mag_weight * diag["elastic_mag_reg"].mean()
                + self.elastic_smooth_weight * diag["elastic_smoothness"].mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw.nan_to_num_(nan=0.0, posinf=6.0, neginf=-6.0)
                raw.clamp_(-6.0, 6.0)
        noise, diag = self._elastic_noise_from_raw(raw.detach(), height, width)
        return noise.detach(), _detach_diag(diag)

    def _svf_disp_from_raw(self, raw: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        velocity_low = torch.tanh(raw) * self.svf_max_vel_ratio * 2.0
        velocity = F.interpolate(velocity_low, size=(height, width), mode="bilinear", align_corners=False)
        ddf = self._integrate_velocity_monai_xy(velocity)
        ddf_vectors = ddf.permute(0, 2, 3, 1)
        ddf_norm = torch.linalg.vector_norm(ddf_vectors, ord=2, dim=-1)
        velocity_mag_reg = velocity.square().flatten(1).mean(dim=1)
        svf_disp_mag_reg = ddf.square().flatten(1).mean(dim=1)
        diag = {
            "velocity_mag_reg": velocity_mag_reg,
            "velocity_norm": velocity_mag_reg.clamp_min(1e-12).sqrt(),
            "velocity_smoothness": _smoothness_per_sample(velocity),
            "svf_displacement_mag_reg": svf_disp_mag_reg,
            "svf_displacement_norm": svf_disp_mag_reg.clamp_min(1e-12).sqrt(),
            "svf_max_displacement": ddf_norm.flatten(1).amax(dim=1),
            "official_warp_backend": "monai.DVF2DDF+monai.Warp",
        }
        return ddf, diag

    def _estimate_svf_disp(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_detached = pred_img.detach()
        target_detached = target_img.detach()
        batch, _, height, width = pred_img.shape
        raw = torch.zeros(batch, 2, self.svf_grid_size, self.svf_grid_size, device=pred_img.device, dtype=pred_img.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=self.svf_lr)
        for _ in range(max(0, self.svf_inner_steps)):
            ddf, diag = self._svf_disp_from_raw(raw, height, width)
            warped = self._warp_dense_monai_xy(pred_detached, ddf)
            loss = (
                self._align_loss(warped, target_detached)
                + self.svf_mag_weight * diag["velocity_mag_reg"].mean()
                + self.svf_smooth_weight * diag["velocity_smoothness"].mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw.nan_to_num_(nan=0.0, posinf=6.0, neginf=-6.0)
                raw.clamp_(-6.0, 6.0)
        ddf, diag = self._svf_disp_from_raw(raw.detach(), height, width)
        return ddf.detach(), _detach_diag(diag)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> MetricResult:
        if self.name == "qnorm":
            values = _batch_norm(pred - target, self.q_order)
            return MetricResult(values=values, diagnostics={"name": self.name, "direction": "larger_is_more_dissimilar"})

        pred_img, target_img = self._normalized_images(pred, target)

        if self.name == "dists":
            values = self._dists_values(pred_img, target_img)
            return MetricResult(values=values, diagnostics={"name": self.name, "direction": "larger_is_more_dissimilar"})

        if self.name == "ms_ssim":
            if self.ms_ssim is None:
                self.ms_ssim = _MSSSIMAdapter().to(pred_img.device)
            values = self.ms_ssim(pred_img, target_img)
            return MetricResult(values=values, diagnostics={"name": self.name, "direction": "larger_is_more_dissimilar"})

        if self.name == "scattering2d":
            if self.scattering is None:
                self.scattering = _Scattering2DAdapter(j=3).to(pred_img.device)
            values = self.scattering(pred_img, target_img)
            return MetricResult(values=values, diagnostics={"name": self.name, "direction": "larger_is_more_dissimilar", "scattering_j": self.scattering.j})

        if self.name == "affine_dists":
            matrix, diag = self._estimate_affine(pred_img, target_img)
            warped = _warp_affine_kornia(pred_img, matrix.to(device=pred_img.device, dtype=pred_img.dtype))
            content = self._dists_values(warped, target_img)
            reg = diag["affine_reg"].to(device=content.device, dtype=content.dtype)
            values = content + self.affine_reg_weight * reg
            diag.update({"name": self.name, "direction": "larger_is_more_dissimilar", "content_loss": content.detach(), "align_objective": self.align_objective})
            return MetricResult(values=values, diagnostics=diag)

        if self.name == "local_warp_dists":
            disp, diag = self._estimate_local_disp(pred_img, target_img)
            warped = self._warp_dense_monai_xy(pred_img, disp.to(device=pred_img.device, dtype=pred_img.dtype))
            content = self._dists_values(warped, target_img)
            mag = diag["displacement_norm"].to(device=content.device, dtype=content.dtype).square()
            smooth = diag["smoothness"].to(device=content.device, dtype=content.dtype)
            values = content + self.local_mag_weight * mag + self.local_smooth_weight * smooth
            diag.update({"name": self.name, "direction": "larger_is_more_dissimilar", "content_loss": content.detach(), "align_objective": self.align_objective})
            return MetricResult(values=values, diagnostics=diag)


        if self.name == "homography_dists":
            matrix, diag = self._estimate_homography(pred_img, target_img)
            warped = _warp_perspective_kornia(pred_img, matrix.to(device=pred_img.device, dtype=pred_img.dtype))
            content = self._dists_values(warped, target_img)
            reg = diag["homography_reg"].to(device=content.device, dtype=content.dtype)
            values = content + self.homography_reg_weight * reg
            diag.update({"name": self.name, "direction": "larger_is_more_dissimilar", "content_loss": content.detach(), "align_objective": self.align_objective})
            return MetricResult(values=values, diagnostics=diag)

        if self.name == "tps_dists":
            points_src, points_dst, diag = self._estimate_tps(pred_img, target_img)
            warped = _warp_tps_kornia(
                pred_img,
                points_src.to(device=pred_img.device, dtype=pred_img.dtype),
                points_dst.to(device=pred_img.device, dtype=pred_img.dtype),
            )
            content = self._dists_values(warped, target_img)
            offset = diag["tps_offset_reg"].to(device=content.device, dtype=content.dtype)
            smooth = diag["tps_smoothness"].to(device=content.device, dtype=content.dtype)
            values = content + self.tps_offset_weight * offset + self.tps_smooth_weight * smooth
            diag.update({"name": self.name, "direction": "larger_is_more_dissimilar", "content_loss": content.detach(), "align_objective": self.align_objective})
            return MetricResult(values=values, diagnostics=diag)

        if self.name == "elastic_dists":
            noise, diag = self._estimate_elastic_disp(pred_img, target_img)
            warped = _warp_elastic_kornia(
                pred_img,
                noise.to(device=pred_img.device, dtype=pred_img.dtype),
                self.elastic_smooth_kernel,
                self.elastic_smooth_passes,
            )
            content = self._dists_values(warped, target_img)
            mag = diag["elastic_mag_reg"].to(device=content.device, dtype=content.dtype)
            smooth = diag["elastic_smoothness"].to(device=content.device, dtype=content.dtype)
            values = content + self.elastic_mag_weight * mag + self.elastic_smooth_weight * smooth
            diag.update({"name": self.name, "direction": "larger_is_more_dissimilar", "content_loss": content.detach(), "align_objective": self.align_objective})
            return MetricResult(values=values, diagnostics=diag)

        if self.name == "svf_dists":
            ddf, diag = self._estimate_svf_disp(pred_img, target_img)
            warped = self._warp_dense_monai_xy(pred_img, ddf.to(device=pred_img.device, dtype=pred_img.dtype))
            content = self._dists_values(warped, target_img)
            mag = diag["velocity_mag_reg"].to(device=content.device, dtype=content.dtype)
            smooth = diag["velocity_smoothness"].to(device=content.device, dtype=content.dtype)
            values = content + self.svf_mag_weight * mag + self.svf_smooth_weight * smooth
            diag.update({"name": self.name, "direction": "larger_is_more_dissimilar", "content_loss": content.detach(), "align_objective": self.align_objective, "svf_int_steps": self.svf_int_steps})
            return MetricResult(values=values, diagnostics=diag)

        raise ValueError(f"Unknown loss3 metric: {self.name}")


def build_loss3_metric(args: Any) -> Loss3MetricComputer:
    if str(args.loss3_metric) not in LOSS3_METRIC_CHOICES:
        raise ValueError(f"Unknown loss3 metric {args.loss3_metric!r}; expected one of {LOSS3_METRIC_CHOICES}")
    return Loss3MetricComputer(args)
