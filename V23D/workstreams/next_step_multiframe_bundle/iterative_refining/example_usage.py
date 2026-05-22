from __future__ import annotations

import torch

from .cameras import WeakPerspectiveCamera
from .refiner import IterativeRefinementConfig, IterativeRefinementLossBuilder


def demo():
    B, J = 4, 24
    joints_3d = torch.randn(B, J, 3)
    joints_2d = torch.randn(B, J, 2)
    template = torch.randn(B, J, 3)
    target_3d = torch.randn(B, J, 3)
    pose = torch.randn(B, 72)
    betas = torch.randn(1, 10)
    conf = torch.ones(B, J)
    cam = WeakPerspectiveCamera(
        s=torch.ones(B),
        tx=torch.zeros(B),
        ty=torch.zeros(B),
    )

    builder = IterativeRefinementLossBuilder(IterativeRefinementConfig())
    losses = builder.build(
        joints_3d=joints_3d,
        joints_2d_target=joints_2d,
        pose=pose,
        betas=betas,
        weak_camera=cam,
        template_joints_3d=template,
        target_joints_3d=target_3d,
        conf_2d=conf,
    )
    for k, v in losses.items():
        print(k, float(v.detach().cpu()))
    print(builder.synthetic_camera_error(joints_3d))


if __name__ == "__main__":
    demo()
