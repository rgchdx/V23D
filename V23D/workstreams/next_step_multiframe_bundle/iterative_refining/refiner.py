from __future__ import annotations

from dataclasses import dataclass

import torch

from .cameras import WeakPerspectiveCamera, build_orbit_cameras, perspective_project, project_points_weakpersp
from .losses import (
    body_proportion_loss,
    camera_parameter_error_loss,
    joint_2d_loss,
    joint_3d_loss,
    pose_prior_loss,
    procrustes_aligned_2d_joint_loss,
    procrustes_aligned_3d_joint_loss,
    projection_error_loss,
    shape_prior_loss,
    sjc_style_multiview_consistency_loss,
)


@dataclass
class IterativeRefinementConfig:
    w_pose: float = 0.2
    w_shape: float = 0.1
    w_proj: float = 1.0
    w_joint_2d: float = 1.0
    w_joint_3d: float = 0.5
    w_proc_2d: float = 0.25
    w_proc_3d: float = 0.5
    w_body_prop: float = 0.2
    w_camera: float = 0.1
    w_sjc: float = 0.1


class IterativeRefinementLossBuilder:
    def __init__(self, config: IterativeRefinementConfig | None = None):
        self.cfg = config or IterativeRefinementConfig()

    def build(
        self,
        joints_3d: torch.Tensor,
        joints_2d_target: torch.Tensor,
        pose: torch.Tensor,
        betas: torch.Tensor,
        weak_camera: WeakPerspectiveCamera,
        template_joints_3d: torch.Tensor | None = None,
        target_joints_3d: torch.Tensor | None = None,
        conf_2d: torch.Tensor | None = None,
        score_gradients: torch.Tensor | None = None,
        camera_targets: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] = (None, None, None),
    ) -> dict[str, torch.Tensor]:
        pred_2d = project_points_weakpersp(joints_3d, weak_camera)
        losses = {
            "pose": self.cfg.w_pose * pose_prior_loss(pose),
            "shape": self.cfg.w_shape * shape_prior_loss(betas),
            "projection": self.cfg.w_proj * projection_error_loss(pred_2d, joints_2d_target, conf_2d),
            "joint_2d": self.cfg.w_joint_2d * joint_2d_loss(pred_2d, joints_2d_target, conf_2d),
            "camera": self.cfg.w_camera * camera_parameter_error_loss(weak_camera, *camera_targets),
        }
        if target_joints_3d is not None:
            losses["joint_3d"] = self.cfg.w_joint_3d * joint_3d_loss(joints_3d, target_joints_3d)
            losses["proc_3d"] = self.cfg.w_proc_3d * procrustes_aligned_3d_joint_loss(joints_3d, target_joints_3d)
        else:
            zero = joints_3d.new_tensor(0.0)
            losses["joint_3d"] = zero
            losses["proc_3d"] = zero
        losses["proc_2d"] = self.cfg.w_proc_2d * procrustes_aligned_2d_joint_loss(pred_2d, joints_2d_target, conf_2d)
        if template_joints_3d is not None:
            losses["body_prop"] = self.cfg.w_body_prop * body_proportion_loss(joints_3d, template_joints_3d)
        else:
            losses["body_prop"] = joints_3d.new_tensor(0.0)
        losses["sjc"] = self.cfg.w_sjc * sjc_style_multiview_consistency_loss(pred_2d, target_views=joints_2d_target, score_gradients=score_gradients)
        losses["total"] = sum(losses.values())
        return losses

    def synthetic_camera_error(self, joints_3d: torch.Tensor, n_views: int = 8) -> dict[str, torch.Tensor]:
        cams = build_orbit_cameras(n_views, device=joints_3d.device, dtype=joints_3d.dtype)
        projs = []
        for cam in cams:
            projs.append(perspective_project(joints_3d, cam["R"], cam["t"], cam["focal"]))
        proj_stack = torch.stack(projs, dim=0)
        pair_err = torch.abs(proj_stack[1:] - proj_stack[:-1]).mean()
        return {
            "camera_multiview_variation": pair_err,
            "num_virtual_cams": torch.tensor(float(n_views), device=joints_3d.device),
        }
