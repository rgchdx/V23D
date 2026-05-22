"""
fit_smpl_visible_rigid.py
=========================
Alternative joint-fitting pipeline:

1) Per-frame pose estimation from MediaPipe 2D joints (VISIBLE ONLY)
   with weak-perspective projection.
2) Per-frame rigid registration (root-orient delta + scale + translation)
   using only visible joints and known COLMAP intrinsics/extrinsics.
3) Export 24-joint SMPL skeleton per frame.

This is designed for cases where global multi-frame optimization drifts and
2D joint alignment on individual frames is poor.

Usage
-----
python fit_smpl_visible_rigid.py \
    --colmap-dir E:/V23D_Data/colmap_rerun/sparse/1 \
    --landmarks-json E:/V23D_Data/smpl_out/landmarks_mediapipe.json \
    --out-dir E:/V23D_Data/smpl_visible_rigid \
    --smpl-model E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

from src.recon.smpl_fitter import (
    SMPL,
    _build_K,
    _precompute_silhouette_dt,
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
    MP_TO_SMPL_PAIRS,
)
from src.pose.extract_mediapipe import extract_joint_observations, load_landmarks_json


STAGE1_REPROJ_JOINTS = {1, 2, 4, 5, 7, 8, 16, 17}
STAGE2_REPROJ_JOINTS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17}
STAGE2_ACTIVE_POSE_JOINTS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17}
STAGE3_ACTIVE_POSE_JOINTS = set(range(1, 24))


def _split_stage_iters(total: int, s1: int, s2: int, s3: int) -> tuple[int, int, int]:
    if s1 > 0 or s2 > 0 or s3 > 0:
        vals = [max(0, int(s1)), max(0, int(s2)), max(0, int(s3))]
        if sum(vals) == 0:
            return 30, 60, max(60, total)
        return tuple(vals)
    total = max(int(total), 30)
    a = max(20, total // 5)
    b = max(30, total // 3)
    c = max(20, total - a - b)
    return a, b, c


def _select_obs_subset(
    obs_xy: np.ndarray,
    obs_w: np.ndarray,
    smpl_idx: list[int],
    allowed_joints: set[int] | None,
    min_count: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]] | tuple[None, None, list[int]]:
    if len(smpl_idx) == 0:
        return None, None, []
    if allowed_joints is None:
        keep = list(range(len(smpl_idx)))
    else:
        keep = [i for i, sj in enumerate(smpl_idx) if sj in allowed_joints]
        if len(keep) < min_count:
            keep = list(range(len(smpl_idx)))
    if not keep:
        return None, None, []
    xy = torch.from_numpy(obs_xy[keep]).float()
    w = torch.from_numpy(obs_w[keep]).float()
    sidx = [smpl_idx[i] for i in keep]
    return xy, w, sidx


def _mask_pose_grads(pose: nn.Parameter, active_pose_joints: set[int]):
    if pose.grad is None:
        return
    mask = torch.zeros(pose.shape[-1], dtype=torch.bool, device=pose.grad.device)
    for joint_idx in active_pose_joints:
        if 0 <= joint_idx < 24:
            mask[joint_idx * 3:(joint_idx + 1) * 3] = True
    mask[:3] = False
    pose.grad[:, ~mask] = 0.0


def _expand_kinematic_chain(smpl_ids: list[int], parents: torch.Tensor) -> set[int]:
    out: set[int] = set()
    parent_np = parents.detach().cpu().numpy().astype(np.int64)
    for joint_idx in smpl_ids:
        j = int(joint_idx)
        out.add(j)
        while j > 0:
            p = int(parent_np[j - 1])
            out.add(p)
            j = p
    out.discard(0)
    return out


def _precompute_mask_stats(
    frame_names: list[str],
    masks_dir: Path,
    img_hw: tuple[int, int],
) -> dict[str, dict[str, np.ndarray | float] | None]:
    stats: dict[str, dict[str, np.ndarray | float] | None] = {}
    H, W = img_hw
    for name in frame_names:
        stem = Path(name).stem
        mask_arr = None
        for ext in (".png", ".jpg", ".jpeg"):
            mp = masks_dir / (stem + ext)
            if mp.exists():
                mask_arr = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                break
        if mask_arr is None:
            stats[name] = None
            continue
        if mask_arr.shape[:2] != (H, W):
            mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)
        fg = (mask_arr > 127).astype(np.uint8)
        ys, xs = np.where(fg > 0)
        if len(xs) == 0:
            stats[name] = None
            continue
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        cx, cy = float(xs.mean()), float(ys.mean())
        area = float(len(xs))
        stats[name] = {
            "bbox": np.array([x0, y0, x1, y1], dtype=np.float32),
            "centroid": np.array([cx, cy], dtype=np.float32),
            "area": area,
        }
    return stats


def _silhouette_dt_loss(
    verts_world: torch.Tensor,
    proj_mat: torch.Tensor,
    dt_tensor: torch.Tensor | None,
    vert_idx: np.ndarray | None,
) -> torch.Tensor:
    if dt_tensor is None or vert_idx is None or len(vert_idx) == 0:
        return verts_world.new_tensor(0.0)
    H_dt, W_dt = dt_tensor.shape
    v3 = verts_world[vert_idx]
    v4 = torch.cat([v3, torch.ones(v3.shape[0], 1, device=v3.device, dtype=v3.dtype)], dim=1)
    proj = (proj_mat @ v4.T).T
    z = proj[:, 2]
    valid = z > 0.01
    if not torch.any(valid):
        return verts_world.new_tensor(0.0)
    xy = proj[valid, :2] / z[valid, None].clamp(min=0.01)
    xn = (xy[:, 0] / max(W_dt - 1, 1)) * 2.0 - 1.0
    yn = (xy[:, 1] / max(H_dt - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([xn, yn], dim=1).view(1, 1, -1, 2)
    vals = F.grid_sample(
        dt_tensor.view(1, 1, H_dt, W_dt),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(-1)
    return vals.mean()


def _silhouette_bbox_loss(
    verts_world: torch.Tensor,
    proj_mat: torch.Tensor,
    mask_stats: dict[str, np.ndarray | float] | None,
    vert_idx: np.ndarray | None,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    if mask_stats is None or vert_idx is None or len(vert_idx) == 0:
        return verts_world.new_tensor(0.0)
    H, W = image_hw
    v3 = verts_world[vert_idx]
    v4 = torch.cat([v3, torch.ones(v3.shape[0], 1, device=v3.device, dtype=v3.dtype)], dim=1)
    proj = (proj_mat @ v4.T).T
    z = proj[:, 2]
    valid = z > 0.01
    if not torch.any(valid):
        return verts_world.new_tensor(0.0)
    xy = proj[valid, :2] / z[valid, None].clamp(min=0.01)
    pred_min = xy.min(dim=0).values
    pred_max = xy.max(dim=0).values
    pred_ctr = xy.mean(dim=0)
    pred_area = torch.clamp(pred_max[0] - pred_min[0], min=1.0) * torch.clamp(pred_max[1] - pred_min[1], min=1.0)

    tgt_bbox = torch.from_numpy(np.asarray(mask_stats["bbox"], dtype=np.float32)).to(verts_world.device)
    tgt_ctr = torch.from_numpy(np.asarray(mask_stats["centroid"], dtype=np.float32)).to(verts_world.device)
    tgt_area = torch.tensor(float(mask_stats["area"]), dtype=verts_world.dtype, device=verts_world.device)

    norm_bbox = torch.tensor([W, H, W, H], dtype=verts_world.dtype, device=verts_world.device)
    norm_ctr = torch.tensor([W, H], dtype=verts_world.dtype, device=verts_world.device)
    bbox_loss = torch.abs(torch.cat([pred_min, pred_max]) - tgt_bbox).div(norm_bbox).mean()
    ctr_loss = torch.abs(pred_ctr - tgt_ctr).div(norm_ctr).mean()
    area_loss = torch.abs(torch.log(pred_area.clamp(min=1.0)) - torch.log(tgt_area.clamp(min=1.0)))
    return bbox_loss + ctr_loss + 0.25 * area_loss


def _weakpersp_init(obs_xy: np.ndarray, cam_K: np.ndarray, smpl_j_xy: np.ndarray) -> tuple[float, float, float]:
    """Init weak-perspective params (s, tx, ty)."""
    f = float((cam_K[0, 0] + cam_K[1, 1]) * 0.5)
    # scale from shoulder span if available, else bbox ratio
    s = 180.0
    tx = float(np.nanmean(obs_xy[:, 0]))
    ty = float(np.nanmean(obs_xy[:, 1]))
    if len(obs_xy) >= 2:
        obs_span = np.linalg.norm(obs_xy.max(0) - obs_xy.min(0))
        mdl_span = np.linalg.norm(smpl_j_xy.max(0) - smpl_j_xy.min(0))
        if mdl_span > 1e-5:
            s = float(obs_span / mdl_span)
    return s, tx, ty


def _depth_init_from_shoulders(obs: np.ndarray, cam_K: np.ndarray, smpl_shoulders_w: float) -> float:
    """Perspective depth init from pixel shoulder width."""
    f = float((cam_K[0, 0] + cam_K[1, 1]) * 0.5)
    depth = f * smpl_shoulders_w / 60.0
    # MP shoulders: 11,12
    l = obs[11, :2]
    r = obs[12, :2]
    if not (np.isnan(l).any() or np.isnan(r).any()):
        px = float(np.linalg.norm(l - r))
        if px > 5.0:
            depth = f * smpl_shoulders_w / px
    return max(depth, 0.2)


def fit_frame_visible_rigid(
    smpl: SMPL,
    betas: torch.Tensor,
    cam_K: np.ndarray,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
    obs_lms: np.ndarray,
    device: torch.device,
    init_pose: np.ndarray | None = None,
    init_trans: np.ndarray | None = None,
    init_scale: float | None = None,
    min_visibility: float = 0.4,
    dt_tensor: torch.Tensor | None = None,
    mask_stats: dict[str, np.ndarray | float] | None = None,
    sil_vert_idx: np.ndarray | None = None,
    n_iters_stage1: int = 40,
    n_iters_stage2: int = 60,
    n_iters_stage3: int = 80,
    lambda_sil: float = 0.0,
    lambda_sil_bbox: float = 0.0,
    lambda_root_prior: float = 0.02,
    lambda_scale_prior: float = 0.02,
    lambda_pose_stage2: float = 0.25,
    lambda_pose_stage3: float = 0.10,
    freeze_scale_after_stage1: bool = True,
    n_iters: int = 180,
) -> dict | None:
    """
    Fit one frame using visible joints only.
    Returns dict with pose, scale, trans, joints24_world and reproj error.
    """
    # Build correspondences
    obs = extract_joint_observations(obs_lms, MP_TO_SMPL_PAIRS, min_visibility=min_visibility)
    obs_pts = obs["xy"]
    obs_w = obs["conf"]
    smpl_idx = obs["smpl_idx"].tolist()
    mp_idx = obs["mp_idx"].tolist()

    if len(obs_pts) < 6:
        return None

    obs_np = np.asarray(obs_pts, np.float32)
    obs_w_np = np.asarray(obs_w, np.float32)

    K_t = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    R_t = torch.from_numpy(cam_R.astype(np.float32)).to(device)
    t_t = torch.from_numpy(cam_t.astype(np.float32)).to(device)
    proj_mat = K_t @ torch.cat([R_t, t_t.unsqueeze(1)], dim=1)
    image_hw = (int(round(float(cam_K[1, 2] * 2.0))), int(round(float(cam_K[0, 2] * 2.0))))

    # Rest joints for init
    with torch.no_grad():
        _, j0 = smpl(betas, torch.zeros(1, 72, device=device), torch.zeros(1, 3, device=device))
    j0_np = j0[0].detach().cpu().numpy()

    # Single-stage robust perspective fitting (visible joints only)
    # Optional initialization from previous/global fit.
    if init_pose is not None:
        pose0 = torch.from_numpy(init_pose.astype(np.float32)).to(device).unsqueeze(0)
    else:
        pose0 = torch.zeros(1, 72, device=device)
    pose = nn.Parameter(pose0.clone())
    root_delta = nn.Parameter(torch.zeros(1, 3, device=device))
    if init_scale is not None:
        log_scale = nn.Parameter(torch.tensor([np.log(max(float(init_scale), 1e-5))], device=device, dtype=torch.float32))
    else:
        log_scale = nn.Parameter(torch.zeros(1, device=device))

    # init trans from shoulder depth + image centroid
    smpl_shoulders_w = float(np.linalg.norm(j0_np[16] - j0_np[17]))
    depth0 = _depth_init_from_shoulders(obs_lms, cam_K, smpl_shoulders_w)
    centroid = np.nanmean(obs_np, axis=0)
    K_inv = np.linalg.inv(cam_K)
    ray = K_inv @ np.array([centroid[0], centroid[1], 1.0], dtype=np.float32)
    ray = ray / max(ray[2], 1e-6)
    p_cam = ray * depth0
    p_world = cam_R.T @ (p_cam - cam_t)
    if init_trans is not None:
        trans0 = np.asarray(init_trans, dtype=np.float32)
        trans = nn.Parameter(torch.from_numpy(trans0).to(device).unsqueeze(0))
    else:
        trans = nn.Parameter(torch.from_numpy(p_world.astype(np.float32)).to(device).unsqueeze(0))

    s1_iters, s2_iters, s3_iters = _split_stage_iters(n_iters, n_iters_stage1, n_iters_stage2, n_iters_stage3)

    def _forward_world():
        p = pose.clone()
        p[:, :3] = p[:, :3] + root_delta
        v, j = smpl(betas, p, torch.zeros(1, 3, device=device))
        s = torch.exp(log_scale)
        vw = v * s.view(1, 1, 1) + trans.unsqueeze(1)
        jw = j * s.view(1, 1, 1) + trans.unsqueeze(1)
        return p, s, vw, jw

    def _reproj_loss(jw: torch.Tensor, obs_xy_t: torch.Tensor, obs_w_t: torch.Tensor, smpl_ids: list[int]) -> torch.Tensor:
        j_sel = jw[0, smpl_ids]
        j4 = torch.cat([j_sel, torch.ones(j_sel.shape[0], 1, device=device)], dim=1)
        pr = (proj_mat @ j4.T).T
        xy = pr[:, :2] / pr[:, 2:3].clamp(min=0.01)
        diff = xy - obs_xy_t.to(device)
        return (torch.sqrt((diff ** 2).sum(dim=1) + 4.0) * obs_w_t.to(device)).mean()

    obs1_xy, obs1_w, sidx1 = _select_obs_subset(obs_np, obs_w_np, smpl_idx, STAGE1_REPROJ_JOINTS, min_count=4)
    obs2_xy, obs2_w, sidx2 = _select_obs_subset(obs_np, obs_w_np, smpl_idx, STAGE2_REPROJ_JOINTS, min_count=6)
    obs3_xy, obs3_w, sidx3 = _select_obs_subset(obs_np, obs_w_np, smpl_idx, None, min_count=6)
    observed_chain = _expand_kinematic_chain(smpl_idx, smpl.parents)
    stage2_active = STAGE2_ACTIVE_POSE_JOINTS.intersection(observed_chain) if observed_chain else STAGE2_ACTIVE_POSE_JOINTS
    stage3_active = observed_chain.intersection(STAGE3_ACTIVE_POSE_JOINTS) if observed_chain else STAGE3_ACTIVE_POSE_JOINTS
    if not stage2_active:
        stage2_active = STAGE2_ACTIVE_POSE_JOINTS
    if not stage3_active:
        stage3_active = stage2_active

    def _run_stage(
        n_stage_iters: int,
        obs_xy_t: torch.Tensor | None,
        obs_w_t: torch.Tensor | None,
        smpl_ids_stage: list[int],
        active_pose_joints: set[int],
        lambda_pose_prior: float,
        use_silhouette: bool,
        freeze_scale: bool,
        lr: float,
    ):
        if n_stage_iters <= 0 or obs_xy_t is None or len(smpl_ids_stage) == 0:
            return
        opt = torch.optim.Adam([pose, root_delta, log_scale, trans], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_stage_iters, eta_min=max(lr * 0.1, 1e-4))
        for _ in range(n_stage_iters):
            opt.zero_grad()
            p, _, vw, jw = _forward_world()
            reproj = _reproj_loss(jw, obs_xy_t, obs_w_t, smpl_ids_stage)
            sil = _silhouette_dt_loss(vw[0], proj_mat, dt_tensor, sil_vert_idx) if use_silhouette else jw.new_tensor(0.0)
            sil_bbox = _silhouette_bbox_loss(vw[0], proj_mat, mask_stats, sil_vert_idx, image_hw) if use_silhouette else jw.new_tensor(0.0)
            reg_root = (root_delta ** 2).mean() * lambda_root_prior
            reg_scale = (log_scale ** 2).mean() * lambda_scale_prior
            reg_pose = (p[:, 3:] ** 2).mean() * lambda_pose_prior + (p[:, :3] ** 2).mean() * 0.01
            loss = reproj + lambda_sil * sil + lambda_sil_bbox * sil_bbox + reg_root + reg_scale + reg_pose
            loss.backward()
            _mask_pose_grads(pose, active_pose_joints)
            if freeze_scale and log_scale.grad is not None:
                log_scale.grad.zero_()
            opt.step()
            sched.step()

            _run_stage(s1_iters, obs1_xy, obs1_w, sidx1, set(), lambda_pose_prior=0.0, use_silhouette=False, freeze_scale=False, lr=0.03)
            _run_stage(s2_iters, obs2_xy, obs2_w, sidx2, stage2_active, lambda_pose_prior=lambda_pose_stage2, use_silhouette=lambda_sil > 0.0 or lambda_sil_bbox > 0.0, freeze_scale=freeze_scale_after_stage1, lr=0.015)
            _run_stage(s3_iters, obs3_xy, obs3_w, sidx3, stage3_active, lambda_pose_prior=lambda_pose_stage3, use_silhouette=lambda_sil > 0.0 or lambda_sil_bbox > 0.0, freeze_scale=freeze_scale_after_stage1, lr=0.01)

    # Final
    with torch.no_grad():
        p, s, _, jw = _forward_world()
        j_sel = jw[0, smpl_idx]
        j4 = torch.cat([j_sel, torch.ones(j_sel.shape[0], 1, device=device)], dim=1)
        pr = (proj_mat @ j4.T).T
        xy = pr[:, :2] / pr[:, 2:3].clamp(min=0.01)
        err = torch.sqrt(((xy - torch.from_numpy(obs_np).to(device)) ** 2).sum(dim=1)).mean().item()

    return {
        "pose": p[0].detach().cpu().numpy(),
        "scale": float(torch.exp(log_scale).item()),
        "trans": trans[0].detach().cpu().numpy(),
        "joints24_world": jw[0].detach().cpu().numpy(),
        "reproj_px": float(err),
        "n_detected_joints": int(len(mp_idx)),
        "detected_mp_indices": mp_idx,
        "detected_smpl_indices": smpl_idx,
    }


def main():
    ap = argparse.ArgumentParser(description="Visible-joint-only rigid registration SMPL fitter")
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--landmarks-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smpl-model", default=(
        r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models"
        r"\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    ))
    ap.add_argument("--betas-npy", default=None, help="Optional existing betas.npy")
    ap.add_argument("--masks-dir", default=None, help="Optional binary masks directory for silhouette DT loss")
    ap.add_argument("--init-poses-json", default=None,
                    help="Optional poses_per_frame.json from previous fit")
    ap.add_argument("--init-trans-npy", default=None,
                    help="Optional trans_per_frame.npy from previous fit")
    ap.add_argument("--init-scale-npy", default=None,
                    help="Optional scale.npy from previous fit")
    ap.add_argument("--n-betas", type=int, default=10)
    ap.add_argument("--min-visibility", type=float, default=0.4,
                    help="Only use MediaPipe joints with visibility >= this threshold")
    ap.add_argument("--n-iters-stage1", type=int, default=0, help="Stage 1 iters: global orientation/translation only")
    ap.add_argument("--n-iters-stage2", type=int, default=0, help="Stage 2 iters: torso/hips/legs pose only")
    ap.add_argument("--n-iters-stage3", type=int, default=0, help="Stage 3 iters: full body pose")
    ap.add_argument("--n-iters", type=int, default=180,
                    help="Per-frame optimization iterations")
    ap.add_argument("--lambda-sil", type=float, default=0.01, help="Chamfer-style silhouette DT weight")
    ap.add_argument("--lambda-sil-bbox", type=float, default=0.20, help="Projected mesh bbox/centroid silhouette weight")
    ap.add_argument("--lambda-root-prior", type=float, default=0.02)
    ap.add_argument("--lambda-scale-prior", type=float, default=0.02)
    ap.add_argument("--lambda-pose-stage2", type=float, default=0.25)
    ap.add_argument("--lambda-pose-stage3", type=float, default=0.10)
    ap.add_argument("--freeze-scale-after-stage1", action="store_true", default=True)
    ap.add_argument("--n-sil-verts", type=int, default=768, help="Sampled mesh vertices for silhouette DT loss")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = _read_colmap_cameras_txt(Path(args.colmap_dir) / "cameras.txt")
    images = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")

    lm_dict = load_landmarks_json(args.landmarks_json)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    smpl = SMPL(args.smpl_model, n_betas=args.n_betas).to(dev)

    # Use existing betas if provided (better shape prior)
    if args.betas_npy and Path(args.betas_npy).exists():
        b_np = np.load(args.betas_npy).astype(np.float32)
        if len(b_np) != args.n_betas:
            b_np = b_np[: args.n_betas]
        betas = torch.from_numpy(b_np).to(dev).unsqueeze(0)
        print(f"Loaded betas from {args.betas_npy}")
    else:
        betas = torch.zeros(1, args.n_betas, device=dev)
        print("No betas input -> using zeros")

    init_poses = {}
    init_trans = {}
    init_scale = None
    if args.init_poses_json and Path(args.init_poses_json).exists():
        init_poses = json.loads(Path(args.init_poses_json).read_text())
        print(f"Loaded init poses: {args.init_poses_json}")
    if args.init_trans_npy and Path(args.init_trans_npy).exists():
        t_obj = np.load(args.init_trans_npy, allow_pickle=True)
        try:
            init_trans = t_obj.item() if hasattr(t_obj, "item") else {}
        except Exception:
            init_trans = {}
        print(f"Loaded init trans: {args.init_trans_npy}")
    if args.init_scale_npy and Path(args.init_scale_npy).exists():
        s_obj = np.load(args.init_scale_npy)
        init_scale = float(np.ravel(s_obj)[0])
        print(f"Loaded init scale: {init_scale:.4f}")

    names = sorted([n for n in lm_dict.keys() if n in images])
    if args.max_frames > 0:
        names = names[: args.max_frames]

    dt_maps = {}
    mask_stats = {}
    sil_vert_idx = None
    if args.masks_dir and Path(args.masks_dir).exists() and names:
        first_info = images[names[0]]
        cam_def = cams[first_info["cam_id"]]
        dt_np = _precompute_silhouette_dt(names, Path(args.masks_dir), (cam_def["h"], cam_def["w"]))
        mask_stats = _precompute_mask_stats(names, Path(args.masks_dir), (cam_def["h"], cam_def["w"]))
        dt_maps = {
            k: torch.from_numpy(v.astype(np.float32)).to(dev) if v is not None else None
            for k, v in dt_np.items()
        }
        rng = np.random.default_rng(0)
        sil_vert_idx = rng.choice(smpl.n_verts, size=min(args.n_sil_verts, smpl.n_verts), replace=False)

    results = {}
    errs = []
    prev_pose = None
    prev_trans = None

    for i, name in enumerate(names):
        if i % 20 == 0:
            print(f"[{i}/{len(names)}] {name}")
        lms = lm_dict[name]
        if lms is None:
            continue
        info = images[name]
        K = _build_K(cams[info["cam_id"]]).astype(np.float32)

        pose_init = None
        trans_init = None
        if name in init_poses:
            pose_init = np.asarray(init_poses[name], dtype=np.float32)
        elif prev_pose is not None:
            pose_init = prev_pose

        if name in init_trans:
            trans_init = np.asarray(init_trans[name], dtype=np.float32)
        elif prev_trans is not None:
            trans_init = prev_trans

        out = fit_frame_visible_rigid(
            smpl=smpl,
            betas=betas,
            cam_K=K,
            cam_R=info["R"],
            cam_t=info["t"],
            obs_lms=lms,
            device=dev,
            init_pose=pose_init,
            init_trans=trans_init,
            init_scale=init_scale,
            min_visibility=args.min_visibility,
            dt_tensor=dt_maps.get(name),
            mask_stats=mask_stats.get(name),
            sil_vert_idx=sil_vert_idx,
            n_iters_stage1=args.n_iters_stage1,
            n_iters_stage2=args.n_iters_stage2,
            n_iters_stage3=args.n_iters_stage3,
            lambda_sil=args.lambda_sil,
            lambda_sil_bbox=args.lambda_sil_bbox,
            lambda_root_prior=args.lambda_root_prior,
            lambda_scale_prior=args.lambda_scale_prior,
            lambda_pose_stage2=args.lambda_pose_stage2,
            lambda_pose_stage3=args.lambda_pose_stage3,
            freeze_scale_after_stage1=args.freeze_scale_after_stage1,
            n_iters=args.n_iters,
        )
        if out is None:
            continue
        results[name] = {
            "pose": out["pose"].tolist(),
            "scale": out["scale"],
            "trans": out["trans"].tolist(),
            "reproj_px": out["reproj_px"],
            "joints24_world": out["joints24_world"].tolist(),
            "n_detected_joints": out["n_detected_joints"],
            "detected_mp_indices": out["detected_mp_indices"],
            "detected_smpl_indices": out["detected_smpl_indices"],
        }
        errs.append(out["reproj_px"])
        prev_pose = np.asarray(out["pose"], dtype=np.float32)
        prev_trans = np.asarray(out["trans"], dtype=np.float32)

    (out_dir / "visible_rigid_fit.json").write_text(json.dumps(results, indent=2))
    poses_out = {k: v["pose"] for k, v in results.items()}
    trans_out = {k: v["trans"] for k, v in results.items()}
    scales_out = {k: v["scale"] for k, v in results.items()}
    (out_dir / "poses_per_frame.json").write_text(json.dumps(poses_out, indent=2))
    np.save(str(out_dir / "trans_per_frame.npy"), trans_out)
    (out_dir / "scale_per_frame.json").write_text(json.dumps(scales_out, indent=2))

    if scales_out:
        np.save(str(out_dir / "scale.npy"), np.array([
            float(np.median(np.array(list(scales_out.values()), dtype=np.float32)))
        ], dtype=np.float32))
    if errs:
        print(f"Done: {len(results)} frames, reproj mean={np.mean(errs):.2f}px, median={np.median(errs):.2f}px")
    print(f"Saved -> {out_dir / 'visible_rigid_fit.json'}")


if __name__ == "__main__":
    main()
