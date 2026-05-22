from __future__ import annotations

import torch
import torch.nn.functional as F

from .cameras import WeakPerspectiveCamera, project_points_weakpersp


# Safe mean that returns 0 for empty tensors, and supports optional weighting.
def _safe_mean(x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    if w is None:
        return x.mean() if x.numel() > 0 else x.new_tensor(0.0)
    denom = w.sum().clamp(min=1e-6)
    return (x * w).sum() / denom


# Robust loss function that behaves like L1 for large residuals, but is smooth near zero.
def charbonnier(residual: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(residual * residual + eps * eps)


# Pose prior loss takes the body pose parameters (72D) and optionally the global orientation (3D), and penalizes deviation from zero.
def pose_prior_loss(body_pose: torch.Tensor, global_orient: torch.Tensor | None = None, weight_body: float = 1.0, weight_global: float = 0.1) -> torch.Tensor:
    loss = weight_body * (body_pose ** 2).mean()
    if global_orient is not None:
        loss = loss + weight_global * (global_orient ** 2).mean()
    return loss


# Shape prior loss takes the shape parameters (betas) and penalizes deviation from zero, encouraging a more "average" body shape.
def shape_prior_loss(betas: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return weight * (betas ** 2).mean()



# The projection error loss compares the 2D projection of the predicted 3D joints of the target with the target 2D joints,
# optionally weighted by a confidence map, and optionally using a robust loss.
def projection_error_loss(pred_2d: torch.Tensor, target_2d: torch.Tensor, conf: torch.Tensor | None = None, robust: bool = True) -> torch.Tensor:
    diff = pred_2d - target_2d
    err = torch.linalg.norm(diff, dim=-1)
    if robust:
        err = charbonnier(err, eps=2.0)
    return _safe_mean(err, conf)



# A joint consistency loss that encourages the predicted 3D joints to be consistent with the 2D joints from MediaPipe outputs.
def joint_2d_loss(pred_2d: torch.Tensor, target_2d: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    return projection_error_loss(pred_2d, target_2d, conf=conf, robust=True)



# A 3D joint loss that encourages the predicted 3D joints to be close to the target 3D joints, optionally with root alignment and 
# confidence weigthting. This is used for the 3D joint supervision loss.
def joint_3d_loss(pred_3d: torch.Tensor, target_3d: torch.Tensor, conf: torch.Tensor | None = None, root_align: bool = True) -> torch.Tensor:
    if root_align:
        pred_3d = pred_3d - pred_3d[..., :1, :]
        target_3d = target_3d - target_3d[..., :1, :]
    err = torch.linalg.norm(pred_3d - target_3d, dim=-1)
    return _safe_mean(err, conf)



# A simple joint consistency loss that compares joint locations in the predicted with the target jonits.
def simple_joint_consistency_loss(joints_a: torch.Tensor, joints_b: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return weight * torch.linalg.norm(joints_a - joints_b, dim=-1).mean()




# This is a procrustes alignment loss. 
# It first computes a similarity Procrustes alignment of the predicted joints to the target joints, and then computes the mean 
# distance between the aligned joints and the target joints. This is useful for evaluating the 3D joint error up to a similarity 
# transoform, which is common in human pose estimation when the scale and global position may be ambiguous.
def _procrustes_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Similarity Procrustes alignment for batched joint sets: (..., J, D)."""
    pred_mean = pred.mean(dim=-2, keepdim=True)
    tgt_mean = target.mean(dim=-2, keepdim=True)
    X = pred - pred_mean
    Y = target - tgt_mean
    Xn = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    Yn = Y / Y.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    H = Xn.transpose(-2, -1) @ Yn
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)
    if pred.shape[-1] == 3:
        det = torch.det(R)
        eye = torch.eye(3, device=pred.device, dtype=pred.dtype).expand(R.shape).clone()
        eye[..., -1, -1] = torch.sign(det)
        R = Vh.transpose(-2, -1) @ eye @ U.transpose(-2, -1)
    scale = (Yn * (Xn @ R.transpose(-2, -1))).sum(dim=(-2, -1), keepdim=True)
    aligned = scale.unsqueeze(-1) * (Xn @ R.transpose(-2, -1)) + tgt_mean
    return aligned



# A loss that compares the camera parameters implied by the current 3D joints to those of a synthetic "orbit" camera.
def procrustes_aligned_3d_joint_loss(pred_3d: torch.Tensor, target_3d: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    aligned = _procrustes_align(pred_3d, target_3d)
    err = torch.linalg.norm(aligned - target_3d, dim=-1)
    return _safe_mean(err, conf)



# A 2D version of the procrustes-aligned joint loss, which can be used to evaluate the 2D joint error up to a similarity transform. 
# This is less common but could be useful for evaluating the 2D reprojection error while ignoring scale and translation differences.
def procrustes_aligned_2d_joint_loss(pred_2d: torch.Tensor, target_2d: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    aligned = _procrustes_align(pred_2d, target_2d)
    err = torch.linalg.norm(aligned - target_2d, dim=-1)
    return _safe_mean(err, conf)



# This is a placeholder for a more complex multiview consistency loss inspired by SJC, 
# which could combine view consistency with optional score-gradient alignment. The current implementation is a simple 
#combination of a reconstruction loss and a score-gradient alignment loss, but this could be extended with more 
# sophisticated terms or weighting strategies. The idea is to encourage the predicted views to be consistent with the target views,
# while also aligning with any available score gradients that indicate where the model should focus its attention.
def _bone_lengths(joints: torch.Tensor, edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.stack([torch.linalg.norm(joints[..., a, :] - joints[..., b, :], dim=-1) for a, b in edges], dim=-1)




# This is a placeholder for a more complex multiview consistency loss inspired by SJC,
# which could combine view consistency with optional score-gradient alignment. The current implementation is a simple
# combination of a reconstruction loss and a score-gradient alignment loss, but this could be extended with more
# sophisticated terms or weighting strategies. The idea is to encourage the predicted views to be consistent with
def body_proportion_loss(pred_3d: torch.Tensor, template_3d: torch.Tensor, edges: list[tuple[int, int]] | None = None) -> torch.Tensor:
    """Penalize implausible bone-length ratios relative to a template body."""
    if edges is None:
        edges = [(1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20), (17, 19), (19, 21), (1, 16), (2, 17)]
    n_joints = min(pred_3d.shape[-2], template_3d.shape[-2])
    edges = [(a, b) for a, b in edges if a < n_joints and b < n_joints]
    if not edges:
        return pred_3d.new_tensor(0.0)
    pred_len = _bone_lengths(pred_3d, edges)
    tmpl_len = _bone_lengths(template_3d, edges)
    pred_len = pred_len / pred_len.mean(dim=-1, keepdim=True).clamp(min=1e-6)
    tmpl_len = tmpl_len / tmpl_len.mean(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.abs(pred_len - tmpl_len).mean()



# This is a simple camera parameter loss that pushes weak perspective camera parameters to the target.
def camera_parameter_error_loss(camera: WeakPerspectiveCamera, target_s: torch.Tensor | None = None, target_tx: torch.Tensor | None = None, target_ty: torch.Tensor | None = None) -> torch.Tensor:
    loss = camera.s.new_tensor(0.0)
    if target_s is not None:
        loss = loss + torch.abs(camera.s - target_s).mean()
    if target_tx is not None:
        loss = loss + torch.abs(camera.tx - target_tx).mean()
    if target_ty is not None:
        loss = loss + torch.abs(camera.ty - target_ty).mean()
    return loss


def perspective_mask_distance_loss(
    projected_xy: torch.Tensor,
    distance_transform: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample an outside-mask distance transform at projected pixel coordinates.

    `projected_xy` is in pixel coordinates `(x, y)`.
    `distance_transform` is `(H, W)` where 0 means inside/on-mask and larger
    values mean farther outside the person mask.
    """
    if projected_xy.numel() == 0:
        return distance_transform.new_tensor(0.0)
    if valid is not None:
        projected_xy = projected_xy[valid]
    if projected_xy.numel() == 0:
        return distance_transform.new_tensor(0.0)

    h, w = distance_transform.shape[-2:]
    x = projected_xy[:, 0].clamp(0, max(w - 1, 0))
    y = projected_xy[:, 1].clamp(0, max(h - 1, 0))
    xn = (x / max(w - 1, 1)) * 2.0 - 1.0
    yn = (y / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([xn, yn], dim=-1).view(1, 1, -1, 2)
    dt4 = distance_transform.view(1, 1, h, w)
    vals = F.grid_sample(dt4, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return vals.mean()


def projected_bbox_loss(
    projected_xy: torch.Tensor,
    valid: torch.Tensor,
    target_bbox_xyxy: torch.Tensor,
    image_wh: tuple[int, int],
) -> torch.Tensor:
    """Compare projected bbox to target mask bbox in normalized image coords."""
    if projected_xy.numel() == 0 or valid.numel() == 0 or not torch.any(valid):
        return projected_xy.new_tensor(0.0)
    pts = projected_xy[valid]
    pred_xyxy = torch.stack([
        pts[:, 0].min(), pts[:, 1].min(),
        pts[:, 0].max(), pts[:, 1].max(),
    ])
    w_img, h_img = float(image_wh[0]), float(image_wh[1])
    norm = projected_xy.new_tensor([w_img, h_img, w_img, h_img]).clamp(min=1.0)
    return torch.abs((pred_xyxy - target_bbox_xyxy) / norm).mean()


def mesh_anchor_loss(verts: torch.Tensor, reference_verts: torch.Tensor) -> torch.Tensor:
    """Keep refined vertices close to the starting mesh."""
    return torch.abs(verts - reference_verts).mean()


def mesh_edge_length_loss(
    verts: torch.Tensor,
    edge_index: torch.Tensor,
    reference_edge_lengths: torch.Tensor,
) -> torch.Tensor:
    """Preserve original edge lengths to avoid foldovers/stretching."""
    if edge_index.numel() == 0:
        return verts.new_tensor(0.0)
    va = verts[edge_index[:, 0]]
    vb = verts[edge_index[:, 1]]
    lengths = torch.linalg.norm(va - vb, dim=-1)
    return torch.abs(lengths - reference_edge_lengths).mean()


def mesh_laplacian_loss(offsets: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Smooth neighboring offsets so refinement stays coherent."""
    if edge_index.numel() == 0:
        return offsets.new_tensor(0.0)
    oa = offsets[edge_index[:, 0]]
    ob = offsets[edge_index[:, 1]]
    return torch.abs(oa - ob).mean()


def sjc_style_multiview_consistency_loss(pred_views: torch.Tensor, target_views: torch.Tensor | None = None, score_gradients: torch.Tensor | None = None, weight_score: float = 1.0, weight_recon: float = 1.0) -> torch.Tensor:
    """Simple SJC-style placeholder: combine view consistency with optional score-gradient alignment."""
    loss = pred_views.new_tensor(0.0)
    if target_views is not None:
        loss = loss + weight_recon * torch.abs(pred_views - target_views).mean()
    if score_gradients is not None:
        loss = loss + weight_score * (pred_views * score_gradients).mean().abs()
    return loss



# A ROMP stype loss. This combines a reprojection loss on the 2D joints with pose and shape priors, and an optional temporal 
# consistency term that encourages smoothness across frames. This is a simple implementation inspired by ROMP, 
# but it could be extended with additional terms or weighting strategies as needed.
def romp_style_multiframe_loss(
    joints_3d: torch.Tensor,
    joints_2d_target: torch.Tensor,
    camera: WeakPerspectiveCamera,
    pose: torch.Tensor,
    betas: torch.Tensor,
    conf_2d: torch.Tensor | None = None,
    temporal_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    pred_2d = project_points_weakpersp(joints_3d, camera)
    losses = {
        "proj_2d": joint_2d_loss(pred_2d, joints_2d_target, conf_2d),
        "pose_prior": pose_prior_loss(pose),
        "shape_prior": shape_prior_loss(betas),
    }
    if joints_3d.shape[0] > 1:
        losses["temporal"] = temporal_weight * torch.abs(joints_3d[1:] - joints_3d[:-1]).mean()
    else:
        losses["temporal"] = joints_3d.new_tensor(0.0)
    losses["total"] = sum(losses.values())
    return losses
