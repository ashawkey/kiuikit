"""
Self-contained single-image camera intrinsics estimation (pinhole) via GeoCalib.

Original project: GeoCalib (`https://github.com/cvg/GeoCalib`)
License: Apache-2.0.

Public API:
    - `GeoCalib`: load model weights and predict (fx, fy, cx, cy) from a single RGB image
      assuming pinhole camera model.
"""

from __future__ import annotations

import logging
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

logger = logging.getLogger(__name__)


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    return deg / 180 * torch.pi


def fov2focal(fov: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return size / 2 / torch.tan(fov / 2)


def focal2fov(focal: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return 2 * torch.arctan(size / (2 * focal))


def rad2rotmat(roll: torch.Tensor, pitch: torch.Tensor, yaw: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert (batched) roll, pitch, yaw angles (radians) to rotation matrix."""
    if yaw is None:
        yaw = roll.new_zeros(roll.shape)

    Rx = pitch.new_zeros(pitch.shape + (3, 3))
    Rx[..., 0, 0] = 1
    Rx[..., 1, 1] = torch.cos(pitch)
    Rx[..., 1, 2] = torch.sin(pitch)
    Rx[..., 2, 1] = -torch.sin(pitch)
    Rx[..., 2, 2] = torch.cos(pitch)

    Ry = yaw.new_zeros(yaw.shape + (3, 3))
    Ry[..., 0, 0] = torch.cos(yaw)
    Ry[..., 0, 2] = -torch.sin(yaw)
    Ry[..., 1, 1] = 1
    Ry[..., 2, 0] = torch.sin(yaw)
    Ry[..., 2, 2] = torch.cos(yaw)

    Rz = roll.new_zeros(roll.shape + (3, 3))
    Rz[..., 0, 0] = torch.cos(roll)
    Rz[..., 0, 1] = torch.sin(roll)
    Rz[..., 1, 0] = -torch.sin(roll)
    Rz[..., 1, 1] = torch.cos(roll)
    Rz[..., 2, 2] = 1

    return Rz @ Rx @ Ry


def _center_crop_to_multiple(
    img: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-crop a BCHW tensor to the largest size divisible by `multiple`.

    Returns:
      - cropped image
      - crop_pad tensor (dw, dh) where values are <= 0 (new_size - old_size)
    """
    if img.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(img.shape)}")
    h, w = img.shape[-2:]
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    if new_h == h and new_w == w:
        crop_pad = img.new_zeros((2,))
        return img, crop_pad

    top = (h - new_h) // 2
    left = (w - new_w) // 2
    img_c = img[..., top : top + new_h, left : left + new_w]
    crop_pad = img.new_tensor([float(new_w - w), float(new_h - h)])
    return img_c, crop_pad


def _compute_resize_hw(h: int, w: int, resize: Optional[Union[int, Tuple[int, int]]], side: str) -> Optional[Tuple[int, int]]:
    if resize is None:
        return None
    if isinstance(resize, tuple):
        if len(resize) != 2:
            raise ValueError(f"resize must be int, None, or (h,w); got {resize}")
        return int(resize[0]), int(resize[1])

    side_size = int(resize)
    aspect_ratio = w / h
    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
    return (
        (side_size, int(side_size * aspect_ratio))
        if side == "vert" or (side != "horz" and (side == "short") ^ (aspect_ratio < 1.0))
        else (int(side_size / aspect_ratio), side_size)
    )


def preprocess_image(
    img_bchw: torch.Tensor,
    *,
    resize: Optional[Union[int, Tuple[int, int]]] = 320,
    edge_divisible_by: Optional[int] = 32,
    side: str = "short",
    interpolation: str = "bilinear",
    align_corners=None,
    antialias: bool = True,
    square_crop: bool = False,
) -> dict[str, Any]:
    """Preprocess for GeoCalib inference.

    Args:
      img_bchw: float tensor (B,C,H,W) in [0,1]
    Returns:
      dict with keys: image, scales, crop_pad (optional), image_size, original_image_size, transform
    """
    if img_bchw.ndim != 4:
        raise ValueError(f"Expected batched image (B,C,H,W), got {tuple(img_bchw.shape)}")
    img = img_bchw
    h, w = img.shape[-2:]

    if square_crop:
        min_size = min(h, w)
        top = (h - min_size) // 2
        left = (w - min_size) // 2
        img = img[..., top : top + min_size, left : left + min_size]

    new_size = _compute_resize_hw(int(img.shape[-2]), int(img.shape[-1]), resize, side)
    if new_size is not None:
        kwargs = {}
        if interpolation in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = align_corners
            if "antialias" in F.interpolate.__code__.co_varnames:
                kwargs["antialias"] = antialias
        img = F.interpolate(img, size=new_size, mode=interpolation, **kwargs)

    # scale maps original (w,h) -> resized/cropped (w',h') for intrinsics adjustment
    scale = torch.tensor([img.shape[-1] / w, img.shape[-2] / h], device=img.device, dtype=img.dtype)
    T = np.diag([float(scale[0].cpu()), float(scale[1].cpu()), 1.0])

    data: dict[str, Any] = {
        "scales": scale,
        "image_size": np.array([img.shape[-1], img.shape[-2]]),
        "transform": T,
        "original_image_size": np.array([w, h]),
    }

    if edge_divisible_by is not None:
        img, crop_pad = _center_crop_to_multiple(img, int(edge_divisible_by))
        data["crop_pad"] = crop_pad
        data["image_size"] = np.array([img.shape[-1], img.shape[-2]])

    data["image"] = img
    return data


class EuclideanManifold:
    @staticmethod
    def J_plus(x: torch.Tensor) -> torch.Tensor:
        return torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)

    @staticmethod
    def plus(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return x + delta


class SphericalManifold:
    @staticmethod
    def householder_vector(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.sum(x[..., :-1] ** 2, -1)
        xpiv = x[..., -1]
        norm = torch.norm(x, dim=-1)
        if torch.any(sigma < 1e-7):
            sigma = torch.where(sigma < 1e-7, sigma + 1e-7, sigma)
            logger.warning("sigma < 1e-7")
        vpiv = torch.where(xpiv < 0, xpiv - norm, -sigma / (xpiv + norm))
        beta = 2 * vpiv**2 / (sigma + vpiv**2)
        v = torch.cat([x[..., :-1] / vpiv[..., None], torch.ones_like(vpiv)[..., None]], -1)
        return v, beta

    @staticmethod
    def apply_householder(y: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return y - v * (beta * torch.einsum("...i,...i->...", v, y))[..., None]

    @classmethod
    def J_plus(cls, x: torch.Tensor) -> torch.Tensor:
        v, beta = cls.householder_vector(x)
        H = -torch.einsum("..., ...k, ...l->...kl", beta, v, v)
        H = H + torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)
        return H[..., :-1]

    @classmethod
    def plus(cls, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        nx = torch.norm(x, dim=-1, keepdim=True)
        nd = torch.norm(delta, dim=-1, keepdim=True)
        nd_ = torch.where(nd < eps, nd + eps, nd)
        sinc = torch.where(nd < eps, nd.new_ones(nd.shape), torch.sin(nd_) / nd_)
        exp_delta = torch.cat([sinc * delta, torch.cos(nd)], -1)
        v, beta = cls.householder_vector(x)
        return nx * cls.apply_householder(exp_delta, v, beta)


@torch.jit.script
def J_vecnorm(vec: torch.Tensor) -> torch.Tensor:
    D = vec.shape[-1]
    norm_x = torch.norm(vec, dim=-1, keepdim=True).unsqueeze(-1)
    if (norm_x == 0).any():
        norm_x = norm_x + 1e-6
    xxT = torch.einsum("...i,...j->...ij", vec, vec)
    identity = torch.eye(D, device=vec.device, dtype=vec.dtype)
    return identity / norm_x - (xxT / norm_x**3)


@torch.jit.script
def J_up_projection(uv: torch.Tensor, abc: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
    if wrt == "uv":
        c = abc[..., 2][..., None, None, None]
        return -c * torch.eye(2, device=uv.device, dtype=uv.dtype).expand(uv.shape[:-1] + (2, 2))
    if wrt == "abc":
        J = uv.new_zeros(uv.shape[:-1] + (2, 3))
        J[..., 0, 0] = 1
        J[..., 1, 1] = 1
        J[..., 0, 2] = -uv[..., 0]
        J[..., 1, 2] = -uv[..., 1]
        return J
    raise ValueError(f"Unknown wrt: {wrt}")


class Gravity:
    """Gravity direction stored as a unit 3D vector (batched)."""

    eps = 1e-4

    def __init__(self, data: torch.Tensor) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if data.ndim == 1:
            data = data[None]
        if data.ndim != 2 or data.shape[-1] != 3:
            raise ValueError(f"Expected (B,3) gravity, got {tuple(data.shape)}")
        self.vec3d = F.normalize(data.float(), dim=-1)

    @property
    def device(self) -> torch.device:
        return self.vec3d.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vec3d.dtype

    @property
    def batch(self) -> int:
        return int(self.vec3d.shape[0])

    def to(self, *args, **kwargs) -> "Gravity":
        return Gravity(self.vec3d.to(*args, **kwargs))

    def float(self) -> "Gravity":
        return Gravity(self.vec3d.float())

    def new_tensor(self, *args, **kwargs) -> torch.Tensor:
        return self.vec3d.new_tensor(*args, **kwargs)

    def new_zeros(self, *args, **kwargs) -> torch.Tensor:
        return self.vec3d.new_zeros(*args, **kwargs)

    def new_ones(self, *args, **kwargs) -> torch.Tensor:
        return self.vec3d.new_ones(*args, **kwargs)

    @classmethod
    def from_rp(cls, roll: torch.Tensor, pitch: torch.Tensor) -> "Gravity":
        roll = torch.as_tensor(roll)
        pitch = torch.as_tensor(pitch)
        sr, cr = torch.sin(roll), torch.cos(roll)
        sp, cp = torch.sin(pitch), torch.cos(pitch)
        return cls(torch.stack([-sr * cp, -cr * cp, sp], dim=-1))

    @property
    def x(self) -> torch.Tensor:
        return self.vec3d[..., 0]

    @property
    def y(self) -> torch.Tensor:
        return self.vec3d[..., 1]

    @property
    def z(self) -> torch.Tensor:
        return self.vec3d[..., 2]

    @property
    def roll(self) -> torch.Tensor:
        roll = torch.asin(-self.x / (torch.sqrt(1 - self.z**2) + self.eps))
        offset = -torch.pi * torch.sign(self.x)
        return torch.where(self.y < 0, roll, -roll + offset)

    def J_roll(self) -> torch.Tensor:
        cp = torch.cos(self.pitch)
        cr, sr = torch.cos(self.roll), torch.sin(self.roll)
        Jr = self.new_zeros((self.batch, 3))
        Jr[..., 0] = -cr * cp
        Jr[..., 1] = sr * cp
        return Jr

    @property
    def pitch(self) -> torch.Tensor:
        return torch.asin(self.z)

    def J_pitch(self) -> torch.Tensor:
        cp, sp = torch.cos(self.pitch), torch.sin(self.pitch)
        cr, sr = torch.cos(self.roll), torch.sin(self.roll)
        Jp = self.new_zeros((self.batch, 3))
        Jp[..., 0] = sr * sp
        Jp[..., 1] = cr * sp
        Jp[..., 2] = cp
        return Jp

    @property
    def rp(self) -> torch.Tensor:
        return torch.stack([self.roll, self.pitch], dim=-1)

    def J_rp(self) -> torch.Tensor:
        return torch.stack([self.J_roll(), self.J_pitch()], dim=-1)

    @property
    def R(self) -> torch.Tensor:
        return rad2rotmat(roll=self.roll, pitch=self.pitch)

    def update(self, delta: torch.Tensor, spherical: bool = False) -> "Gravity":
        if spherical:
            return Gravity(SphericalManifold.plus(self.vec3d, delta))
        rp = EuclideanManifold.plus(self.rp, delta)
        return Gravity.from_rp(rp[..., 0], rp[..., 1])


@dataclass
class PinholeCamera:
    """Pinhole camera model used by the GeoCalib LM optimizer.

    Tensor layout: (B,6) = [w, h, fx, fy, cx, cy]
    """

    eps = 1e-3

    data: torch.Tensor

    @classmethod
    def from_dict(cls, param_dict: Dict[str, torch.Tensor]) -> "PinholeCamera":
        # Accept scalars/ndarrays and convert to tensors.
        param_dict = {k: torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in param_dict.items()}

        h, w = param_dict["height"], param_dict["width"]
        cx, cy = param_dict.get("cx", w / 2), param_dict.get("cy", h / 2)

        if "f" in param_dict:
            f = param_dict["f"]
        elif "vfov" in param_dict:
            vfov = param_dict["vfov"]
            f = fov2focal(vfov, h)
        else:
            raise ValueError("Focal length or vertical field of view must be provided.")

        fx, fy = f, f
        if "scales" in param_dict:
            fx = fx * param_dict["scales"][..., 0] / param_dict["scales"][..., 1]

        params = torch.stack([w, h, fx, fy, cx, cy], dim=-1).float()
        if params.ndim == 1:
            params = params[None]
        return cls(params)

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def shape(self) -> torch.Size:
        return self.data.shape[:-1]

    def to(self, *args, **kwargs) -> "PinholeCamera":
        return PinholeCamera(self.data.to(*args, **kwargs))

    def float(self) -> "PinholeCamera":
        return PinholeCamera(self.data.float())

    def new_tensor(self, *args, **kwargs) -> torch.Tensor:
        return self.data.new_tensor(*args, **kwargs)

    def new_zeros(self, *args, **kwargs) -> torch.Tensor:
        return self.data.new_zeros(*args, **kwargs)

    def new_ones(self, *args, **kwargs) -> torch.Tensor:
        return self.data.new_ones(*args, **kwargs)

    @property
    def size(self) -> torch.Tensor:
        return self.data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        return self.data[..., 2:4]

    @property
    def vfov(self) -> torch.Tensor:
        return focal2fov(self.f[..., 1], self.size[..., 1])

    @property
    def c(self) -> torch.Tensor:
        return self.data[..., 4:6]

    @property
    def K(self) -> torch.Tensor:
        K = self.new_zeros(self.shape + (3, 3))
        K[..., 0, 0] = self.f[..., 0]
        K[..., 1, 1] = self.f[..., 1]
        K[..., 0, 2] = self.c[..., 0]
        K[..., 1, 2] = self.c[..., 1]
        K[..., 2, 2] = 1
        return K

    def update_focal(self, delta: torch.Tensor, as_log: bool = False):
        f = torch.exp(torch.log(self.f) + delta) if as_log else self.f + delta
        min_f = fov2focal(self.new_ones(self.shape[0]) * deg2rad(torch.tensor(150.0, device=self.device)), self.size[..., 1])
        max_f = fov2focal(self.new_ones(self.shape[0]) * deg2rad(torch.tensor(5.0, device=self.device)), self.size[..., 1])
        min_f = min_f.unsqueeze(-1).expand(-1, 2)
        max_f = max_f.unsqueeze(-1).expand(-1, 2)
        f = f.clamp(min=min_f, max=max_f)
        fx = f[..., 1] * self.f[..., 0] / self.f[..., 1]
        f = torch.stack([fx, f[..., 1]], -1)
        return PinholeCamera(torch.cat([self.size, f, self.c], -1))

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        scales = (scales, scales) if isinstance(scales, (int, float)) else scales
        s = scales if isinstance(scales, torch.Tensor) else self.new_tensor(scales)
        return PinholeCamera(torch.cat([self.size * s, self.f * s, self.c * s], -1))

    def crop(self, pad: Tuple[float]):
        pad = pad if isinstance(pad, torch.Tensor) else self.new_tensor(pad)
        size = self.size + pad.to(self.size)
        c = self.c + pad.to(self.c) / 2
        return PinholeCamera(torch.cat([size, self.f, c], -1))

    def undo_scale_crop(self, data: Dict[str, torch.Tensor]):
        camera = self.crop(-data["crop_pad"]) if "crop_pad" in data else self
        return camera.scale(1.0 / data["scales"])

    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates into normalized 2D coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / (self.f.unsqueeze(-2))

    def J_normalize(self, p2d: torch.Tensor, wrt: str = "f") -> torch.Tensor:
        """Jacobian of normalize wrt focal or points."""
        if wrt == "f":
            J_f = -(p2d - self.c.unsqueeze(-2)) / ((self.f.unsqueeze(-2)) ** 2)
            return torch.diag_embed(J_f)  # (..., N, 2, 2)
        raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    def pixel_coordinates(self) -> torch.Tensor:
        """Pixel coordinates as (B, h*w, 2)."""
        if self.size.ndim != 2 or self.size.shape[0] < 1:
            raise ValueError(f"Expected batched camera size, got {tuple(self.size.shape)}")
        if not torch.all(self.size == self.size[0:1]).item():
            raise ValueError("Batched pixel grid with varying image sizes is not supported.")
        w, h = self.size[0].unbind(-1)
        h, w = int(round(float(h.item()))), int(round(float(w.item())))
        x = torch.arange(0, w, dtype=self.dtype, device=self.device)
        y = torch.arange(0, h, dtype=self.dtype, device=self.device)
        x, y = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack((x, y), dim=-1).reshape(-1, 2)
        return xy.unsqueeze(0).expand(self.shape[0], -1, -1)

    def pixel_bearing_many(self, p3d: torch.Tensor) -> torch.Tensor:
        return F.normalize(p3d, dim=-1)

    def image2world(self, p2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p2d = self.normalize(p2d)
        valid = p2d.new_ones((p2d.shape[0], 1)).bool()
        ones = p2d.new_ones(p2d.shape[:-1] + (1,))
        p3d = torch.cat([p2d, ones], -1)
        return p3d, valid

    def J_image2world(self, p2d: torch.Tensor, wrt: str = "f") -> torch.Tensor:
        if wrt == "f":
            return self.J_normalize(p2d, wrt)
        raise ValueError(f"Unknown wrt: {wrt}")


# Perspective field functions

def get_up_field(camera: PinholeCamera, gravity: Gravity, normalize: bool = True) -> torch.Tensor:
    w, h = camera.size[0].unbind(-1)
    h, w = int(round(float(h.item()))), int(round(float(w.item())))

    uv = camera.normalize(camera.pixel_coordinates())
    abc = gravity.vec3d
    projected_up2d = abc[..., None, :2] - abc[..., 2, None, None] * uv
    if normalize:
        projected_up2d = F.normalize(projected_up2d, dim=-1)
    return projected_up2d.reshape(camera.shape[0], h, w, 2)


def J_up_field(camera: PinholeCamera, gravity: Gravity, spherical: bool = False, log_focal: bool = False) -> torch.Tensor:
    w, h = camera.size[0].unbind(-1)
    h, w = int(round(float(h.item()))), int(round(float(w.item())))

    xy = camera.pixel_coordinates()
    uv = camera.normalize(xy)
    projected_up2d = gravity.vec3d[..., None, :2] - gravity.vec3d[..., 2, None, None] * uv

    J = []
    J_norm2proj = J_vecnorm(get_up_field(camera, gravity, normalize=False).reshape(camera.shape[0], -1, 2))

    # gravity jacobian
    J_proj2abc = J_up_projection(uv, gravity.vec3d, wrt="abc")
    J_abc2delta = SphericalManifold.J_plus(gravity.vec3d) if spherical else gravity.J_rp()
    J_proj2delta = torch.einsum("...Nij,...jk->...Nik", J_proj2abc, J_abc2delta)
    J_up2delta = torch.einsum("...Nij,...Njk->...Nik", J_norm2proj, J_proj2delta)
    J.append(J_up2delta)

    # focal jacobian
    J_proj2uv = J_up_projection(uv, gravity.vec3d, wrt="uv")
    J_uv2f = camera.J_normalize(xy)  # wrt="f"
    if log_focal:
        J_uv2f = J_uv2f * camera.f[..., None, None, :]
    J_uv2f = J_uv2f.sum(-1)
    J_proj2f = torch.einsum("...ij,...j->...i", J_proj2uv, J_uv2f)
    J_up2f = torch.einsum("...Nij,...Nj->...Ni", J_norm2proj, J_proj2f)[..., None]
    J.append(J_up2f)

    n_params = sum(j.shape[-1] for j in J)
    return torch.cat(J, axis=-1).reshape(camera.shape[0], h, w, 2, n_params)


def get_latitude_field(camera: PinholeCamera, gravity: Gravity) -> torch.Tensor:
    w, h = camera.size[0].unbind(-1)
    h, w = int(round(float(h.item()))), int(round(float(w.item())))

    uv1, _ = camera.image2world(camera.pixel_coordinates())
    rays = camera.pixel_bearing_many(uv1)
    lat = torch.einsum("...Nj,...j->...N", rays, gravity.vec3d)
    eps = 1e-6
    lat_asin = torch.asin(lat.clamp(min=-1 + eps, max=1 - eps))
    return lat_asin.reshape(camera.shape[0], h, w, 1)


def J_latitude_field(camera: PinholeCamera, gravity: Gravity, spherical: bool = False, log_focal: bool = False) -> torch.Tensor:
    w, h = camera.size[0].unbind(-1)
    h, w = int(round(float(h.item()))), int(round(float(w.item())))

    xy = camera.pixel_coordinates()
    uv1, _ = camera.image2world(xy)
    uv1_norm = camera.pixel_bearing_many(uv1)

    J = []
    J_norm2w_to_img = J_vecnorm(uv1)[..., :2]

    # gravity jacobian
    J_delta = SphericalManifold.J_plus(gravity.vec3d) if spherical else gravity.J_rp()
    J_delta = torch.einsum("...Ni,...ij->...Nj", uv1_norm, J_delta)
    J.append(J_delta)

    # focal jacobian
    J_w_to_img2f = camera.J_image2world(xy, "f")
    if log_focal:
        J_w_to_img2f = J_w_to_img2f * camera.f[..., None, None, :]
    J_w_to_img2f = J_w_to_img2f.sum(-1)
    J_norm2f = torch.einsum("...Nij,...Nj->...Ni", J_norm2w_to_img, J_w_to_img2f)
    J_f = torch.einsum("...Ni,...i->...N", J_norm2f, gravity.vec3d).unsqueeze(-1)
    J.append(J_f)

    n_params = sum(j.shape[-1] for j in J)
    return torch.cat(J, axis=-1).reshape(camera.shape[0], h, w, 1, n_params)


def get_perspective_field(
    camera: PinholeCamera,
    gravity: Gravity,
    use_up: bool = True,
    use_latitude: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert use_up or use_latitude
    w, h = camera.size[0].unbind(-1)
    h, w = int(round(float(h.item()))), int(round(float(w.item())))

    if use_up:
        up = get_up_field(camera, gravity).permute(0, 3, 1, 2)
    else:
        up = camera.new_zeros((camera.shape[0], 2, h, w))

    if use_latitude:
        lat = get_latitude_field(camera, gravity).permute(0, 3, 1, 2)
    else:
        lat = camera.new_zeros((camera.shape[0], 1, h, w))

    return up, lat


def J_perspective_field(
    camera: PinholeCamera,
    gravity: Gravity,
    use_up: bool = True,
    use_latitude: bool = True,
    spherical: bool = False,
    log_focal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert use_up or use_latitude
    w, h = camera.size[0].unbind(-1)
    h, w = int(round(float(h.item()))), int(round(float(w.item())))

    if use_up:
        J_up = J_up_field(camera, gravity, spherical, log_focal)
    else:
        J_up = camera.new_zeros((camera.shape[0], h, w, 2, 3))

    if use_latitude:
        J_lat = J_latitude_field(camera, gravity, spherical, log_focal)
    else:
        J_lat = camera.new_zeros((camera.shape[0], h, w, 1, 3))

    return J_up, J_lat


# Network modules


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        use_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, unit2only: bool = False, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        if not unit2only:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        if self.upsample:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        return output


class NMF2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.S, self.D, self.R = 1, 512, 64
        self.train_steps = 6
        self.eval_steps = 7
        self.inv_t = 1

    def _build_bases(self, B: int, S: int, D: int, R: int, device: str = "cpu") -> torch.Tensor:
        bases = torch.rand((B * S, D, R), device=device)
        return F.normalize(bases, dim=1)

    def local_step(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        return coef * numerator / (denominator + 1e-6)

    def local_inference(self, x: torch.Tensor, bases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        bases = self._build_bases(B, self.S, D, self.R, device=str(x.device))
        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))
        return x.view(B, C, H, W)


class Hamburger(nn.Module):
    def __init__(self, ham_channels: int = 512):
        super().__init__()
        self.ham_in = ConvModule(ham_channels, ham_channels, 1)
        self.ham = NMF2D()
        self.ham_out = ConvModule(ham_channels, ham_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=False)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=False)
        return ham


class LightHamHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_index = [0, 1, 2, 3]
        self.in_channels = [64, 128, 320, 512]
        self.out_channels = 64
        self.ham_channels = 512
        self.align_corners = False
        self.squeeze = ConvModule(sum(self.in_channels), self.ham_channels, 1)
        self.hamburger = Hamburger(self.ham_channels)
        self.align = ConvModule(self.ham_channels, self.out_channels, 1)
        self.linear_pred_uncertainty = nn.Sequential(
            ConvModule(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=1),
        )
        self.out_conv = ConvModule(self.out_channels, self.out_channels, 3, padding=1, bias=False)
        self.ll_fusion = FeatureFusionBlock(self.out_channels, upsample=False)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = [features["hl"][i] for i in self.in_index]
        inputs = [
            F.interpolate(level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            for level in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        feats = self.align(x)
        assert "ll" in features, "Low-level features are required for this model"
        feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = self.out_conv(feats)
        feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = self.ll_fusion(feats, features["ll"].clone())
        uncertainty = self.linear_pred_uncertainty(feats).squeeze(1)
        return feats, uncertainty


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, act_layer: nn.Module = nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1[..., None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[..., None, None] * self.mlp(self.norm2(x)))
        return x.view(B, C, N).permute(0, 2, 1)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.embed_dims = [64, 128, 320, 512]
        self.mlp_ratios = [8, 8, 4, 4]
        self.drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.depths = [3, 3, 12, 3]
        self.num_stages = 4

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = StemConv(3, self.embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=self.embed_dims[i - 1],
                    embed_dim=self.embed_dims[i],
                )
            block = nn.ModuleList(
                [
                    Block(dim=self.embed_dims[i], mlp_ratio=self.mlp_ratios[i], drop=self.drop_rate)
                    for _ in range(self.depths[i])
                ]
            )
            norm = nn.LayerNorm(self.embed_dims[i])
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, data):
        x = data["image"][:, [2, 1, 0], :, :] * 255.0
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return {"features": outs}


# LM optimizer


def get_trivial_estimation(data: Dict[str, torch.Tensor]) -> Tuple[PinholeCamera, Gravity]:
    """Get initial camera for optimization with roll=0, pitch=0, vfov from a heuristic focal."""
    ref = data.get("up_field", data["latitude_field"]).detach()
    h, w = ref.shape[-2:]
    batch_h = ref.new_ones((ref.shape[0],)) * h
    batch_w = ref.new_ones((ref.shape[0],)) * w
    init_r = ref.new_zeros((ref.shape[0],))
    init_p = ref.new_zeros((ref.shape[0],))
    focal = 0.7 * torch.max(batch_h, batch_w)
    init_vfov = focal2fov(focal, batch_h)
    params = {"width": batch_w, "height": batch_h, "vfov": init_vfov}
    params |= {"scales": data["scales"]} if "scales" in data else {}
    camera = PinholeCamera.from_dict(params).float().to(ref.device)
    gravity = Gravity.from_rp(init_r, init_p).float().to(ref.device)
    return camera, gravity


def scaled_loss(x: torch.Tensor, fn: Callable, a: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a2 = a**2
    loss, loss_d1, loss_d2 = fn(x / a2)
    return loss * a2, loss_d1, loss_d2 / a2


def huber_loss(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = x <= 1
    sx = torch.sqrt(x + 1e-8)
    isx = torch.max(sx.new_tensor(torch.finfo(torch.float).eps), 1 / sx)
    loss = torch.where(mask, x, 2 * sx - 1)
    loss_d1 = torch.where(mask, torch.ones_like(x), isx)
    loss_d2 = torch.where(mask, torch.zeros_like(x), -isx / (2 * x))
    return loss, loss_d1, loss_d2


def early_stop(new_cost: torch.Tensor, prev_cost: torch.Tensor, atol: float, rtol: float) -> bool:
    return torch.allclose(new_cost, prev_cost, atol=atol, rtol=rtol)


def update_lambda(
    lamb: torch.Tensor,
    prev_cost: torch.Tensor,
    new_cost: torch.Tensor,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e2,
) -> torch.Tensor:
    new_lamb = lamb * torch.where(new_cost > prev_cost, 10, 0.1)
    return torch.clamp(new_lamb, lambda_min, lambda_max)


def optimizer_step(G: torch.Tensor, H: torch.Tensor, lambda_: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diag = H.diagonal(dim1=-2, dim2=-1)
    diag = diag * lambda_.unsqueeze(-1)
    H = H + diag.clamp(min=eps).diag_embed()
    H_, G_ = H.cpu(), G.cpu()
    try:
        U = torch.linalg.cholesky(H_)
    except RuntimeError:
        logger.warning("Cholesky decomposition failed. Stopping.")
        delta = H.new_zeros((H.shape[0], H.shape[-1]))
    else:
        delta = torch.cholesky_solve(G_[..., None], U)[..., 0]
    return delta.to(H.device)


class LMOptimizer(nn.Module):
    """Levenberg-Marquardt optimizer for camera calibration (GeoCalib)."""

    default_conf = {
        "num_steps": 30,
        "lambda_": 0.1,
        "fix_lambda": False,
        "early_stop": True,
        "atol": 1e-8,
        "rtol": 1e-8,
        "use_spherical_manifold": True,
        "use_log_focal": True,
        "up_loss_fn_scale": 1e-2,
        "lat_loss_fn_scale": 1e-2,
        "verbose": False,
    }

    def __init__(self, conf: Dict[str, Any]):
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.num_steps = conf.num_steps
        self.setup_optimization_and_priors()

    def setup_optimization_and_priors(self, data: Dict[str, torch.Tensor] = None) -> None:
        if data is None:
            data = {}
        self.estimate_gravity = True
        self.estimate_focal = True
        self.gravity_delta_dims = (0, 1) if self.estimate_gravity else (-1,)
        self.focal_delta_dims = (max(self.gravity_delta_dims) + 1,) if self.estimate_focal else (-1,)

    def calculate_residuals(self, camera: PinholeCamera, gravity: Gravity, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        perspective_up, perspective_lat = get_perspective_field(camera, gravity)
        perspective_lat = torch.sin(perspective_lat)
        residuals: dict[str, torch.Tensor] = {}
        if "up_field" in data:
            up_residual = (data["up_field"] - perspective_up).permute(0, 2, 3, 1)
            residuals["up_residual"] = up_residual.reshape(up_residual.shape[0], -1, 2)
        if "latitude_field" in data:
            target_lat = torch.sin(data["latitude_field"])
            lat_residual = (target_lat - perspective_lat).permute(0, 2, 3, 1)
            residuals["latitude_residual"] = lat_residual.reshape(lat_residual.shape[0], -1, 1)
        return residuals

    def calculate_costs(
        self, residuals: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        costs, weights = {}, {}
        if "up_residual" in residuals:
            up_cost = (residuals["up_residual"] ** 2).sum(dim=-1)
            up_cost, up_weight, _ = scaled_loss(up_cost, huber_loss, self.conf.up_loss_fn_scale)
            if "up_confidence" in data:
                up_conf = data["up_confidence"].reshape(up_weight.shape[0], -1)
                up_weight = up_weight * up_conf
                up_cost = up_cost * up_conf
            costs["up_cost"] = up_cost
            weights["up_weights"] = up_weight
        if "latitude_residual" in residuals:
            lat_cost = (residuals["latitude_residual"] ** 2).sum(dim=-1)
            lat_cost, lat_weight, _ = scaled_loss(lat_cost, huber_loss, self.conf.lat_loss_fn_scale)
            if "latitude_confidence" in data:
                lat_conf = data["latitude_confidence"].reshape(lat_weight.shape[0], -1)
                lat_weight = lat_weight * lat_conf
                lat_cost = lat_cost * lat_conf
            costs["latitude_cost"] = lat_cost
            weights["latitude_weights"] = lat_weight
        return costs, weights

    def calculate_gradient_and_hessian(
        self, J: torch.Tensor, residuals: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dims = ()
        if self.estimate_gravity:
            dims = (0, 1)
        if self.estimate_focal:
            dims += (2,)
        assert dims
        J = J[..., dims]
        Grad = torch.einsum("...Njk,...Nj->...Nk", J, residuals)
        Grad = weights[..., None] * Grad
        Grad = Grad.sum(-2)
        Hess = torch.einsum("...Njk,...Njl->...Nkl", J, J)
        Hess = weights[..., None, None] * Hess
        Hess = Hess.sum(-3)
        return Grad, Hess

    def setup_system(
        self,
        camera: PinholeCamera,
        gravity: Gravity,
        residuals: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
        as_rpf: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        J_up, J_lat = J_perspective_field(
            camera,
            gravity,
            spherical=self.conf.use_spherical_manifold and not as_rpf,
            log_focal=self.conf.use_log_focal and not as_rpf,
        )
        J_up = J_up.reshape(J_up.shape[0], -1, J_up.shape[-2], J_up.shape[-1])
        J_lat = J_lat.reshape(J_lat.shape[0], -1, J_lat.shape[-2], J_lat.shape[-1])

        n_params = 2 * self.estimate_gravity + self.estimate_focal
        Grad = J_up.new_zeros(J_up.shape[0], n_params)
        Hess = J_up.new_zeros(J_up.shape[0], n_params, n_params)

        if "up_residual" in residuals:
            Up_Grad, Up_Hess = self.calculate_gradient_and_hessian(J_up, residuals["up_residual"], weights["up_weights"])
            Grad = Grad + Up_Grad
            Hess = Hess + Up_Hess
        if "latitude_residual" in residuals:
            Lat_Grad, Lat_Hess = self.calculate_gradient_and_hessian(J_lat, residuals["latitude_residual"], weights["latitude_weights"])
            Grad = Grad + Lat_Grad
            Hess = Hess + Lat_Hess
        return Grad, Hess

    def update_estimate(self, camera: PinholeCamera, gravity: Gravity, delta: torch.Tensor) -> Tuple[PinholeCamera, Gravity]:
        delta_gravity = delta[..., self.gravity_delta_dims] if self.estimate_gravity else delta.new_zeros(delta.shape[:-1] + (2,))
        new_gravity = gravity.update(delta_gravity, spherical=self.conf.use_spherical_manifold)
        delta_f = delta[..., self.focal_delta_dims] if self.estimate_focal else delta.new_zeros(delta.shape[:-1] + (1,))
        new_camera = camera.update_focal(delta_f, as_log=self.conf.use_log_focal)
        return new_camera, new_gravity

    def optimize(self, data: Dict[str, torch.Tensor], camera_opt: PinholeCamera, gravity_opt: Gravity):
        key = list(data.keys())[0]
        B = data[key].shape[0]
        lamb = data[key].new_ones(B) * self.conf.lambda_
        infos = {"stop_at": self.num_steps}
        prev_cost = None
        for i in range(self.num_steps):
            errors = self.calculate_residuals(camera_opt, gravity_opt, data)
            costs, weights = self.calculate_costs(errors, data)
            cost = sum(c.mean(-1) for c in costs.values())
            if prev_cost is None:
                prev_cost = cost
            Grad, Hess = self.setup_system(camera_opt, gravity_opt, errors, weights)
            delta = optimizer_step(Grad, Hess, lamb)
            camera_new, gravity_new = self.update_estimate(camera_opt, gravity_opt, delta)
            errors_new = self.calculate_residuals(camera_new, gravity_new, data)
            costs_new, _ = self.calculate_costs(errors_new, data)
            new_cost = sum(c.mean(-1) for c in costs_new.values())
            if not self.conf.fix_lambda:
                lamb = update_lambda(lamb, prev_cost, new_cost)
            camera_opt, gravity_opt = camera_new, gravity_new
            if early_stop(new_cost, prev_cost, atol=self.conf.atol, rtol=self.conf.rtol) and self.conf.early_stop:
                infos["stop_at"] = i + 1
                break
            prev_cost = new_cost
        infos["stop_at"] = camera_opt.new_ones(camera_opt.shape[0]) * infos["stop_at"]
        return camera_opt, gravity_opt, infos

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_init, gravity_init = get_trivial_estimation(data)
        self.setup_optimization_and_priors(data)
        camera_opt, gravity_opt, infos = self.optimize(data, camera_init, gravity_init)
        return {"camera": camera_opt, "gravity": gravity_opt, **infos}


class LowLevelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channel = 3
        self.feat_dim = 64
        self.conv1 = ConvModule(self.in_channel, self.feat_dim, kernel_size=3, padding=1)
        self.conv2 = ConvModule(self.feat_dim, self.feat_dim, kernel_size=3, padding=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data["image"]
        assert x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0, "Image size must be multiple of 32."
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        return {"features": c2}


class UpDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = LightHamHead()
        self.linear_pred_up = nn.Conv2d(self.decoder.out_channels, 2, kernel_size=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, log_confidence = self.decoder(data["features"])
        up = self.linear_pred_up(x)
        return {"up_field": F.normalize(up, dim=1), "up_confidence": torch.sigmoid(log_confidence)}


class LatitudeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = LightHamHead()
        self.linear_pred_latitude = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, log_confidence = self.decoder(data["features"])
        eps = 1e-5
        lat = torch.tanh(self.linear_pred_latitude(x))
        lat = torch.asin(torch.clamp(lat, -1 + eps, 1 - eps))
        return {"latitude_field": lat, "latitude_confidence": torch.sigmoid(log_confidence)}


class PerspectiveDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_head = UpDecoder()
        self.latitude_head = LatitudeDecoder()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.up_head(data) | self.latitude_head(data)


class GeoCalibNet(nn.Module):
    def __init__(self, **optimizer_options):
        super().__init__()
        self.backbone = MSCAN()
        self.ll_enc = LowLevelEncoder()
        self.perspective_decoder = PerspectiveDecoder()
        self.optimizer = LMOptimizer({**optimizer_options})

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = {"hl": self.backbone(data)["features"], "ll": self.ll_enc(data)["features"]}
        out = self.perspective_decoder({"features": features})
        out |= {k: data[k] for k in ["image", "scales"] if k in data}
        out |= self.optimizer(out)
        return out

    def flexible_load(self, state_dict: Dict[str, torch.Tensor]) -> None:
        dict_params = set(state_dict.keys())
        model_params = set(map(lambda n: n[0], self.named_parameters()))
        if dict_params == model_params:
            self.load_state_dict(state_dict, strict=True)
            return
        if len(dict_params & model_params) == 0:
            strip_prefix = lambda x: ".".join(x.split(".")[:1] + x.split(".")[2:])
            state_dict = {strip_prefix(n): p for n, p in state_dict.items()}
            dict_params = set(state_dict.keys())
            if len(dict_params & model_params) == 0:
                raise ValueError("Could not load checkpoint; no matching parameters.")
        self.load_state_dict(state_dict, strict=False)


class GeoCalib(nn.Module):
    """GeoCalib intrinsics estimator (pinhole).

    Main usage:
      - Call the module with an RGB image tensor in shape (H, W, 3), float in [0,1].
        It returns (fx, fy, cx, cy) for a pinhole camera with centered principal point.

    Advanced usage:
      - Call `calibrate(img_bchw, ...)` to get the full camera/gravity outputs.
    """

    def __init__(self, weights: str = "pinhole", device: Optional[str] = None):
        super().__init__()
        if weights in {"pinhole"}: # only support pinhole
            url = f"https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-{weights}.tar"
            model_dir = f"{torch.hub.get_dir()}/geocalib"
            state_dict = torch.hub.load_state_dict_from_url(url, model_dir, map_location="cpu", file_name=f"{weights}.tar")
        elif Path(weights).exists():
            state_dict = torch.load(weights, map_location="cpu")
        else:
            raise ValueError(f"Invalid weights: {weights}")

        self.model = GeoCalibNet()
        self.model.flexible_load(state_dict["model"])
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    @torch.no_grad()
    def calibrate(
        self,
        img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        assert img.ndim == 4 and img.shape[0] == 1

        img_data = preprocess_image(img, resize=320, edge_divisible_by=32)
        out = self.model(img_data)
        camera, gravity = out["camera"], out["gravity"]

        # Post-process: undo resize/crop scaling to express intrinsics in original pixel coordinates.
        camera = camera.undo_scale_crop(img_data)
        inverse_scales = 1.0 / img_data["scales"]
        zero = camera.new_zeros(camera.f.shape[0])
        out["focal_uncertainty"] = out.get("focal_uncertainty", zero) * inverse_scales[1]

        # Return FoV in radians for convenience/testing.
        vfov = camera.vfov  # (B,)
        hfov = 2.0 * torch.atan(camera.size[..., 0] / (2.0 * camera.f[..., 0]))

        return {
            "camera": camera,
            "gravity": gravity,
            "hfov": hfov,
            "vfov": vfov,
            **{k: out[k] for k in out.keys() if "uncertainty" in k},
        }

    @torch.no_grad()
    def forward(
        self, rgb: torch.Tensor, simple: bool = False
    ) -> Tuple[float, float, float, float, float, float]:
        """Run GeoCalib on a single RGB image.

        Args:
          rgb: (H,W,3) float tensor in [0,1].
          simple: If True, returns a simplified pinhole intrinsics tuple
            `(fx, fy, cx, cy, hfov, vfov)` where `fx == fy` and `(cx, cy)` is centered.
            If False, returns `(fx, fy, cx, cy, hfov, vfov)` extracted from the full
            `calibrate()` output (i.e. using the optimized camera intrinsics directly;
            no `fx=fy` or centered principal point assumptions).

        Returns:
          (fx, fy, cx, cy, hfov, vfov) as Python floats (FoVs in radians).
        """
        if not isinstance(rgb, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor (H,W,3) float in [0,1], got {type(rgb)}")
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected rgb shape (H,W,3), got {tuple(rgb.shape)}")
        if not rgb.is_floating_point():
            raise TypeError(f"Expected floating dtype for rgb in [0,1], got {rgb.dtype}")
        mn = float(rgb.min().item())
        mx = float(rgb.max().item())
        if mn < -1e-4 or mx > 1.0001:
            raise ValueError(f"Expected rgb values in [0,1], got min={mn}, max={mx}")

        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        img_chw = rgb.permute(2, 0, 1).contiguous().to(self.device)
        res = self.calibrate(img_chw[None])

        if not simple:
            camera = res["camera"]
            fx = float(camera.f[0, 0].item())
            fy = float(camera.f[0, 1].item())
            cx = float(camera.c[0, 0].item())
            cy = float(camera.c[0, 1].item())
            hfov = float(res["hfov"][0].item()) if "hfov" in res else float(2.0 * math.atan(w / (2.0 * fx)))
            vfov = float(res["vfov"][0].item()) if "vfov" in res else float(2.0 * math.atan(h / (2.0 * fy)))
            return fx, fy, cx, cy, hfov, vfov
        else:
            # derive intrinsics from predicted vertical FoV, so fx = fy
            vfov = float(res["vfov"][0].item())
            fx = fy = float(h / (2.0 * math.tan(vfov / 2.0)))
            cx = float(w / 2.0)
            cy = float(h / 2.0)
            hfov = float(2.0 * math.atan(w / (2.0 * fx)))
            return fx, fy, cx, cy, hfov, vfov


def main(argv: Optional[list[str]] = None) -> int:

    import kiui

    parser = argparse.ArgumentParser(
        prog="python -m kiui.geocalib",
        description="Estimate pinhole camera intrinsics (fx, fy, cx, cy) from a single RGB image via GeoCalib.",
    )
    parser.add_argument("image", type=str, help="Path to an input image.")
    parser.add_argument("--weights", type=str, default="pinhole", help="Checkpoint name or local path. Default: pinhole")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g., cpu, cuda, cuda:0). Default: auto (cuda if available else cpu).",
    )
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    args = parser.parse_args(argv)

    rgb = kiui.read_image(args.image, mode="torch", order="RGB")
    kiui.lo(rgb)

    model = GeoCalib(weights=args.weights, device=args.device)
    with torch.no_grad():
        fx, fy, cx, cy, hfov, vfov = model(rgb)

    if args.json:
        print(json.dumps({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "hfov": hfov, "vfov": vfov}))
    else:
        print(
            f"fx={fx:.6f} fy={fy:.6f} cx={cx:.6f} cy={cy:.6f} "
            f"hfov={math.degrees(hfov):.3f} vfov={math.degrees(vfov):.3f}"
        )
    return 0


if __name__ == "__main__":
    main()
