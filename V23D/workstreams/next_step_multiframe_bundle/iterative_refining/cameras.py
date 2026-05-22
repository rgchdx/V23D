from __future__ import annotations

from dataclasses import dataclass
import math

import torch

@dataclass
class WeakPerspectiveCamera:
    s: torch.Tensor
    tx: torch.Tensor
    ty: torch.Tensor



@dataclass
class OrbitCameraSpec:
    yaw_deg: float
    pitch_deg: float = 0.0
    radius: float = 2.5
    focal: float = 1.0


def project_points_weakpersp(points_3d: torch.Tensor, camera: WeakPerspectiveCamera) -> torch.Tensor:
    # Project the 3D points under a simple weak-perspective camera using `s`, `tx`, `ty`.
    # This is for the camera projection loss that we use in the loss.py synthetic_camera_error() function,
    # which compares the camera parameters implied by the current 3D joints to those of a synthetic "orbit" camera.
    # xy is the 2D part of the input 3D points. ... is the batch and joint dims, and :2 takes the x and y components.
    xy = points_3d[..., :2]
    # Reshape the camera parameters to be broadcastable with the points.
    s = camera.s.reshape(*camera.s.shape, *([1] * (xy.ndim - camera.s.ndim)))
    tx = camera.tx.reshape(*camera.tx.shape, *([1] * (xy.ndim - camera.tx.ndim)))
    ty = camera.ty.reshape(*camera.ty.shape, *([1] * (xy.ndim - camera.ty.ndim)))
    # Apply the weak perspective projection: scale the xy by s, then add the translations.
    out = xy * s.unsqueeze(-1)
    out[..., 0] += tx
    out[..., 1] += ty
    return out


def _rotation_y(yaw_deg: float, device=None, dtype=None) -> torch.Tensor:
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], device=device, dtype=dtype)


def _rotation_x(pitch_deg: float, device=None, dtype=None) -> torch.Tensor:
    a = math.radians(pitch_deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], device=device, dtype=dtype)


def build_orbit_cameras(
    n_views: int,
    radius: float = 2.5,
    pitch_deg: float = 0.0,
    focal: float = 1.0,
    device=None,
    dtype=None,
) -> list[dict[str, torch.Tensor | float]]:
    # This creates a list of camera specifications for an "orbit" sequence, where the camera rotates around the subject at a 
    # fixed radius and pitch, with the yaw changing to cover 360 degrees across the n_views. 
    # Output is a list of dicts, each containing the camera parameters and the corresponding rotation matrix R and translation t.
    cams = []
    for i in range(n_views):
        yaw_deg = (360.0 * i) / max(n_views, 1)
        R = _rotation_x(pitch_deg, device=device, dtype=dtype) @ _rotation_y(yaw_deg, device=device, dtype=dtype)
        cam_pos = torch.tensor([math.sin(math.radians(yaw_deg)) * radius, 0.0, math.cos(math.radians(yaw_deg)) * radius], device=device, dtype=dtype)
        t = -(R @ cam_pos)
        cams.append({
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "radius": radius,
            "focal": focal,
            "R": R,
            "t": t,
        })
    return cams


# Simple perspective projection function for testing the camera projection error loss. Not used in the main pipeline,
# but this could enable projection loss if used.
def perspective_project(points_3d: torch.Tensor, R: torch.Tensor, t: torch.Tensor, focal: float | torch.Tensor, cx: float = 0.0, cy: float = 0.0) -> torch.Tensor:
    pts_cam = points_3d @ R.T + t
    z = pts_cam[..., 2:3].clamp(min=1e-6)
    f = torch.as_tensor(focal, device=points_3d.device, dtype=points_3d.dtype)
    x = f * pts_cam[..., 0:1] / z + cx
    y = f * pts_cam[..., 1:2] / z + cy
    return torch.cat([x, y], dim=-1)
