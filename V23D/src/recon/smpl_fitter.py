"""
Multi-frame SMPL fitting from 2D MediaPipe landmarks + COLMAP cameras.

SMPL model: v1.1.0 pkl (neutral / male / female), 24 joints, 6890 verts.

Strategy
--------
• Use a small evenly-spaced subset of frames (default 40) so the fit is
  constrained by a diverse set of views without being dominated by
  redundant consecutive frames.
• 3-stage optimisation:
    Stage 0 — translation only                  (shape/pose frozen at rest)
    Stage 1 — global orient + trans             (shape frozen)
    Stage 2 — shape β + per-frame θ + trans     (full, heavy regularisation)
• Strong regularisation keeps betas small and pose angles small.
• No silhouette loss — reprojection across 40 diverse views is sufficient.

MediaPipe 33-pt → SMPL 24-joint fitting mapping:
    Use only anatomically reliable correspondences for optimisation.
    We intentionally exclude NOSE and HEEL landmarks because they do not align
    cleanly with SMPL skeletal joints and tend to drag the fit toward the wrong
    body parts.

    MP 11  (LEFT_SHOULDER) → SMPL 16
    MP 12  (RIGHT_SHOULDER)→ SMPL 17
    MP 13  (LEFT_ELBOW)    → SMPL 18
    MP 14  (RIGHT_ELBOW)   → SMPL 19
    MP 15  (LEFT_WRIST)    → SMPL 20
    MP 16  (RIGHT_WRIST)   → SMPL 21
    MP 23  (LEFT_HIP)      → SMPL  1
    MP 24  (RIGHT_HIP)     → SMPL  2
    MP 25  (LEFT_KNEE)     → SMPL  4
    MP 26  (RIGHT_KNEE)    → SMPL  5
    MP 27  (LEFT_ANKLE)    → SMPL  7
    MP 28  (RIGHT_ANKLE)   → SMPL  8
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
# COLMAP I/O helpers
# ══════════════════════════════════════════════════════════════════════

def _read_colmap_cameras_txt(path: Path) -> dict[int, dict]:
    cams = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cam_id = int(parts[0])
        w, h   = int(parts[2]), int(parts[3])
        params = list(map(float, parts[4:]))
        cams[cam_id] = dict(model=parts[1], w=w, h=h, params=params)
    return cams


def _read_colmap_images_txt(path: Path) -> dict[str, dict]:
    images: dict[str, dict] = {}
    lines = [l for l in path.read_text().splitlines()
             if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 9:
            i += 1
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz      = map(float, parts[5:8])
        cam_id          = int(parts[8])
        name            = parts[9]
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        images[name] = dict(R=R, t=t, cam_id=cam_id)
        i += 2
    return images


def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
        [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
        [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ], dtype=np.float64)


def _build_K(cam: dict) -> np.ndarray:
    p = cam["params"]
    w, h = cam["w"], cam["h"]
    if cam["model"] in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL", "PINHOLE"):
        f, cx, cy = p[0], p[1], p[2]
    elif cam["model"] == "OPENCV":
        f, cx, cy = p[0], p[2], p[3]
    else:
        f, cx, cy = p[0], w/2.0, h/2.0
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = cx;  K[1, 2] = cy
    return K


# ══════════════════════════════════════════════════════════════════════
# MediaPipe → SMPL joint mapping  (15 correspondences)
# ══════════════════════════════════════════════════════════════════════

MP_TO_SMPL_PAIRS: list[tuple[int, int]] = [
    (11, 16),   # l_shoulder  → left_shoulder
    (12, 17),   # r_shoulder  → right_shoulder
    (13, 18),   # l_elbow     → left_elbow
    (14, 19),   # r_elbow     → right_elbow
    (15, 20),   # l_wrist     → left_wrist
    (16, 21),   # r_wrist     → right_wrist
    (23,  1),   # l_hip       → left_hip
    (24,  2),   # r_hip       → right_hip
    (25,  4),   # l_knee      → left_knee
    (26,  5),   # r_knee      → right_knee
    (27,  7),   # l_ankle     → left_ankle
    (28,  8),   # r_ankle     → right_ankle
]

TRACKED_MP_INDICES: list[int] = sorted({mp for mp, _ in MP_TO_SMPL_PAIRS})
MP_TO_SMPL:         dict[int, int] = dict(MP_TO_SMPL_PAIRS)


# ══════════════════════════════════════════════════════════════════════
# DLT multi-view triangulation  (no body model needed)
# ══════════════════════════════════════════════════════════════════════

def triangulate_joints(
    lm_dict:    dict[str, Optional[np.ndarray]],
    images:     dict[str, dict],
    cam_K:      np.ndarray,
    mp_indices: list[int],
    min_views:  int = 4,
) -> np.ndarray:
    """Returns float64 array (len(mp_indices), 3) — NaN where underdetermined."""
    n = len(mp_indices)
    joints3d = np.full((n, 3), np.nan, dtype=np.float64)

    for ji, mp_idx in enumerate(mp_indices):
        rows = []
        for name, lms in lm_dict.items():
            if lms is None or name not in images:
                continue
            x2d = lms[mp_idx, :2]
            if np.any(np.isnan(x2d)):
                continue
            R, t = images[name]["R"], images[name]["t"]
            P = cam_K @ np.hstack([R, t[:, None]])
            x, y = float(x2d[0]), float(x2d[1])
            rows.append(x * P[2] - P[0])
            rows.append(y * P[2] - P[1])
        if len(rows) < min_views * 2:
            continue
        A = np.stack(rows)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-10:
            continue
        joints3d[ji] = X[:3] / X[3]
    return joints3d


# ══════════════════════════════════════════════════════════════════════
# SMPL v1.1 loader  (works without chumpy)
# ══════════════════════════════════════════════════════════════════════

def _load_smpl_pkl(model_path: Path) -> dict:
    """Load SMPL v1.0/1.1 pkl, returning plain numpy arrays."""
    import pickle

    if "chumpy" not in sys.modules:
        _chumpy = types.ModuleType("chumpy")
        class _Ch:
            def __init__(self, *a, **kw): pass
            def __setstate__(self, s): self.__dict__.update(s)
        _chumpy.Ch = _Ch
        _chumpy.array = lambda *a, **k: a[0] if a else None
        sys.modules["chumpy"]    = _chumpy
        sys.modules["chumpy.ch"] = _chumpy

    with open(model_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    def _to_numpy(v) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v.astype(np.float32)
        if sp.issparse(v):
            return v.toarray().astype(np.float32)
        for attr in ("x", "r", "_r"):
            a = getattr(v, attr, None)
            if a is not None and isinstance(a, np.ndarray):
                return a.astype(np.float32)
        return v

    return {k: _to_numpy(v) for k, v in raw.items()}


# ══════════════════════════════════════════════════════════════════════
# SMPL PyTorch module  (24 joints, 6890 verts)
# ══════════════════════════════════════════════════════════════════════

class SMPL(nn.Module):
    """
    Differentiable SMPL forward pass.

    Parameters
    ----------
    model_path : SMPL v1.0 or v1.1 .pkl file
    n_betas    : shape coefficients to use (≤10 recommended; max 300 for v1.1)
    """

    def __init__(self, model_path: str | Path, n_betas: int = 10):
        super().__init__()
        dd = _load_smpl_pkl(Path(model_path))

        def _t(x):
            return torch.from_numpy(np.asarray(x, dtype=np.float32))

        sd = dd["shapedirs"]            # (6890, 3, K) where K ≥ n_betas
        sd = sd[:, :, :n_betas]

        self.register_buffer("v_template",  _t(dd["v_template"]))   # (6890, 3)
        self.register_buffer("shapedirs",   _t(sd))                  # (6890, 3, n_betas)
        self.register_buffer("posedirs",    _t(dd["posedirs"]))      # (6890, 3, 207)
        self.register_buffer("J_regressor", _t(dd["J_regressor"]))   # (24, 6890)
        self.register_buffer("weights",     _t(dd["weights"]))       # (6890, 24)

        kintree = np.array(dd["kintree_table"], dtype=np.int64)
        self.register_buffer("parents",
                             torch.from_numpy(kintree[0, 1:]))        # (23,)
        self.register_buffer("faces_buf",
                             torch.from_numpy(
                                 np.array(dd["f"], dtype=np.int64)))  # (13776,3)
        self.n_verts = int(self.v_template.shape[0])
        self.n_betas = n_betas

    @property
    def faces(self) -> np.ndarray:
        return self.faces_buf.cpu().numpy()

    def forward(
        self,
        betas: torch.Tensor,   # (B, n_betas)
        pose:  torch.Tensor,   # (B, 72)  axis-angle, 24 joints × 3
        trans: torch.Tensor,   # (B, 3)
        scale: torch.Tensor | None = None,   # (B,) or (B,1), optional global scale
    ):
        """Returns verts (B, 6890, 3) and joints (B, 24, 3)."""
        B = betas.shape[0]

        # 1. Shape
        v_shaped = self.v_template + torch.einsum(
            "vci,bi->bvc", self.shapedirs, betas)                   # (B,6890,3)

        # 2. Joints in rest pose
        J = torch.einsum("jv,bvd->bjd", self.J_regressor, v_shaped) # (B,24,3)

        # 3. Pose blend shapes
        rot_mats  = _batch_rodrigues(pose.reshape(-1, 3)).reshape(B, 24, 3, 3)
        I3        = torch.eye(3, device=pose.device, dtype=pose.dtype)
        pose_feat = (rot_mats[:, 1:] - I3).reshape(B, -1)           # (B,207)
        v_posed   = v_shaped + torch.einsum(
            "vcd,bd->bvc",
            self.posedirs.reshape(self.n_verts, 3, 207),
            pose_feat)                                               # (B,6890,3)

        # 4. LBS
        A = _lbs_global_transforms(J, rot_mats, self.parents)        # (B,24,4,4)
        # weights: (6890,24), A: (B,24,4,4) → T: (B,6890,4,4)
        T = torch.einsum("vj,bjxy->bvxy", self.weights, A)           # (B,6890,4,4)

        v_homo = torch.cat(
            [v_posed, torch.ones(B, self.n_verts, 1, device=pose.device)], -1)
        verts  = (T * v_homo.unsqueeze(-1)).sum(-2)[..., :3]         # (B,6890,3)

        if scale is not None:
            if scale.ndim == 1:
                scale = scale[:, None]
            verts = verts * scale.unsqueeze(-1)

        verts  = verts + trans.unsqueeze(1)

        joints = torch.einsum("jv,bvd->bjd", self.J_regressor, verts) # (B,24,3)
        return verts, joints


def _batch_rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle → (..., 3, 3) rotation matrices."""
    angle = rvec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = rvec / angle
    ca    = torch.cos(angle).unsqueeze(-1)
    sa    = torch.sin(angle).unsqueeze(-1)
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    O  = torch.zeros_like(ax)
    K  = torch.stack([O, -az, ay, az, O, -ax, -ay, ax, O], -1
                     ).reshape(*rvec.shape[:-1], 3, 3)
    I  = torch.eye(3, device=rvec.device, dtype=rvec.dtype
                   ).expand(*rvec.shape[:-1], 3, 3)
    return ca*I + sa*K + (1-ca)*torch.einsum("...i,...j->...ij", axis, axis)


def _lbs_global_transforms(
    J:        torch.Tensor,   # (B, 24, 3)
    rot_mats: torch.Tensor,   # (B, 24, 3, 3)
    parents:  torch.Tensor,   # (23,) int64
) -> torch.Tensor:            # (B, 24, 4, 4)
    B = J.shape[0]
    J_rel = J.clone()
    J_rel[:, 1:] = J[:, 1:] - J[:, parents]

    Ts = torch.zeros(B, 24, 4, 4, device=J.device, dtype=J.dtype)
    Ts[:, :, :3, :3] = rot_mats
    Ts[:, :, :3,  3] = J_rel
    Ts[:, :,  3,  3] = 1.0

    G = [Ts[:, 0]]
    for i in range(1, 24):
        G.append(G[parents[i-1]] @ Ts[:, i])
    G = torch.stack(G, 1)                                # (B,24,4,4)

    # subtract rest-pose joints
    Jw = torch.zeros(B, 24, 4, 1, device=J.device, dtype=J.dtype)
    Jw[:, :, :3, 0] = J
    Jw[:, :,  3, 0] = 1.0
    packed = -(G @ Jw)[..., :3, 0]                       # (B,24,3)

    rest = torch.zeros_like(G)
    rest[:, :, :3, :3] = torch.eye(3, device=J.device, dtype=J.dtype)
    rest[:, :, :3,  3] = packed
    rest[:, :,  3,  3] = 1.0
    return G @ rest


# ══════════════════════════════════════════════════════════════════════
# Frame selection — evenly spaced, quality-filtered
# ══════════════════════════════════════════════════════════════════════

def select_frames(
    lm_dict:     dict[str, Optional[np.ndarray]],
    images:      dict[str, dict],
    n_frames:    int = 40,
    min_visible: int = 8,
) -> list[str]:
    """Pick n_frames evenly spaced from frames that have ≥ min_visible joints."""
    candidates = []
    for name in sorted(lm_dict.keys()):
        lms = lm_dict[name]
        if lms is None or name not in images:
            continue
        n_vis = sum(
            1 for mp_idx in TRACKED_MP_INDICES
            if not np.any(np.isnan(lms[mp_idx, :2]))
        )
        if n_vis >= min_visible:
            candidates.append(name)

    if not candidates:
        raise RuntimeError("No frames with enough visible landmarks + COLMAP pose.")

    step     = max(1, len(candidates) // n_frames)
    selected = candidates[::step][:n_frames]
    print(f"Frame selection: {len(candidates)} eligible -> {len(selected)} used")
    return selected


# ══════════════════════════════════════════════════════════════════════
# Per-frame init: weak-perspective depth from shoulder width
# ══════════════════════════════════════════════════════════════════════

def _init_trans_per_frame_2d(
    lm_dict:           dict[str, Optional[np.ndarray]],
    images:            dict[str, dict],
    cam_K:             np.ndarray,
    frame_names:       list[str],
    smpl_shoulder_w:   float,   # SMPL rest-pose shoulder width (world units)
    smpl_pelvis_offset: np.ndarray,  # (3,) offset from observed centroid to pelvis
) -> np.ndarray:                 # (B, 3) world-space trans per frame
    """
    For each frame:
      1. Estimate camera-space depth from observed shoulder width (weak-perspective)
      2. Back-project 2D centroid of upper-body landmarks to 3D camera space
      3. Convert to world-space translation
    This completely avoids DLT, which fails for near-parallel rays in orbiting setups.
    """
    f     = (cam_K[0, 0] + cam_K[1, 1]) / 2.0
    K_inv = np.linalg.inv(cam_K)
    trans_list = []

    for name in frame_names:
        lms  = lm_dict.get(name)
        info = images[name]
        R    = info["R"]
        t    = info["t"]

        # -- depth from shoulder width ------------------------------------
        depth_cam = None
        if lms is not None:
            for (mi_l, mi_r), smpl_w in [((11, 12), smpl_shoulder_w),
                                          ((23, 24), smpl_shoulder_w * 0.8)]:
                if mi_l < lms.shape[0] and mi_r < lms.shape[0]:
                    lp, rp = lms[mi_l, :2], lms[mi_r, :2]
                    if not (np.isnan(lp).any() or np.isnan(rp).any()):
                        px_dist = np.linalg.norm(lp - rp)
                        if px_dist > 5.0:
                            depth_cam = f * smpl_w / px_dist
                            break
        if depth_cam is None:
            depth_cam = f * smpl_shoulder_w / 60.0  # assume ~60px shoulders

        # -- 2D centroid of visible upper-body joints ----------------------
        UPPER_MP = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        pts2d = []
        if lms is not None:
            for mi in UPPER_MP:
                if mi < lms.shape[0] and not np.isnan(lms[mi, :2]).any():
                    pts2d.append(lms[mi, :2])
        if not pts2d:
            pts2d = [[cam_K[0, 2], cam_K[1, 2]]]
        cx2d, cy2d = np.mean(pts2d, axis=0)

        # -- back-project to 3D camera space ------------------------------
        d_cam = K_inv @ np.array([cx2d, cy2d, 1.0], dtype=np.float64)
        d_cam = d_cam / d_cam[2]          # z-normalise
        pos_cam = d_cam * depth_cam       # pelvis in camera space

        # -- to world space: v_world = R^T (v_cam - t) -------------------
        pos_world = R.T @ (pos_cam - t)
        # Adjust: the 2D centroid is upper body, not pelvis.
        # smpl_pelvis_offset corrects from chest/shoulder centroid to pelvis.
        pos_world = pos_world + smpl_pelvis_offset
        trans_list.append(pos_world.astype(np.float32))

    return np.stack(trans_list)   # (B, 3)


# ══════════════════════════════════════════════════════════════════════
# Silhouette helpers
# ══════════════════════════════════════════════════════════════════════

def _precompute_silhouette_dt(
    frame_names: list[str],
    masks_dir:   Path,
    img_hw:      tuple[int, int],
) -> dict[str, np.ndarray | None]:
    """
    For each frame load its binary mask and compute the distance-transform
    of the *background* (i.e. DT is 0 inside the person, positive outside).
    Returns {name: dt_array (H,W) float32} or None if mask missing.
    """
    from scipy.ndimage import distance_transform_edt
    import cv2
    dt_map = {}
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
            dt_map[name] = None
            continue
        # Resize to camera resolution if needed
        if mask_arr.shape[:2] != (H, W):
            mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)
        fg = (mask_arr > 127).astype(np.uint8)
        # DT of background: 0 inside person, distance outside
        bg = 1 - fg
        dt = distance_transform_edt(bg).astype(np.float32)
        dt_map[name] = dt
    n_ok = sum(v is not None for v in dt_map.values())
    print(f"  Silhouette DTs loaded: {n_ok}/{len(frame_names)} frames")
    return dt_map


def _precompute_mask_bbox_stats(
    frame_names: list[str],
    masks_dir: Path,
    img_hw: tuple[int, int],
) -> dict[str, dict[str, float] | None]:
    import cv2

    H, W = img_hw
    out: dict[str, dict[str, float] | None] = {}
    for name in frame_names:
        stem = Path(name).stem
        mask_arr = None
        for ext in (".png", ".jpg", ".jpeg"):
            mp = masks_dir / (stem + ext)
            if mp.exists():
                mask_arr = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                break
        if mask_arr is None:
            out[name] = None
            continue
        if mask_arr.shape[:2] != (H, W):
            mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)
        fg = mask_arr > 127
        ys, xs = np.where(fg)
        if len(xs) == 0:
            out[name] = None
            continue
        w_box = float(xs.max() - xs.min() + 1)
        h_box = float(ys.max() - ys.min() + 1)
        out[name] = {
            "x0": float(xs.min()),
            "y0": float(ys.min()),
            "x1": float(xs.max()),
            "y1": float(ys.max()),
            "w": w_box,
            "h": h_box,
            "area": float(fg.sum()),
        }
    return out


# ══════════════════════════════════════════════════════════════════════
# Main fitting function
# ══════════════════════════════════════════════════════════════════════

def fit_smpl_multiview(
    smpl_model_path: str | Path,
    lm_dict:         dict[str, Optional[np.ndarray]],
    images:          dict[str, dict],
    cam_K:           np.ndarray,
    masks_dir:       Path | None = None,   # NEW: for silhouette loss
    # frame sampling
    n_frames:        int   = 40,
    min_visible:     int   = 8,
    # model
    n_betas:         int   = 10,
    # stage iteration counts
    n_iters_s0:      int   = 200,   # translation only
    n_iters_s1:      int   = 200,   # + global orient
    n_iters_s2:      int   = 600,   # + full shape + per-frame pose
    lr:              float = 3e-3,
    # loss weights
    lambda_2d:       float = 1.0,
    lambda_sil:      float = 0.008,  # silhouette / outside-mask penalty
    lambda_beta:     float = 8.0,
    lambda_pose:     float = 1.0,
    device:          str   = "cuda",
) -> dict:
    """
    Fit SMPL to 2D landmark observations across multiple views.
    Initialises depth per-frame from weak-perspective shoulder width (no DLT).
    Optionally adds silhouette mask loss if masks_dir is provided.

    Returns dict: betas, scale, poses, trans_per_frame, joints3d, verts, faces
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    smpl = SMPL(smpl_model_path, n_betas=n_betas).to(dev)

    # ── Select frames ─────────────────────────────────────────────────
    frame_names = select_frames(lm_dict, images, n_frames, min_visible)
    B = len(frame_names)

    # ── Get SMPL rest-pose body measurements for init ─────────────────
    with torch.no_grad():
        _, rest_j = smpl(
            torch.zeros(1, n_betas, device=dev),
            torch.zeros(1, 72,     device=dev),
            torch.zeros(1, 3,      device=dev))
        rj = rest_j[0].cpu().numpy()   # (24, 3)

    smpl_shoulder_w = float(np.linalg.norm(rj[16] - rj[17]))  # L/R shoulder
    print(f"  SMPL shoulder width (rest pose): {smpl_shoulder_w:.4f} world units")

    # The 2D centroid of upper-body MP landmarks maps roughly to the
    # chest/shoulder area.  Shift it down toward the pelvis (joint 0).
    # In SMPL rest pose, chest centroid (mean of shoulders+neck) is above pelvis.
    chest_center = (rj[16] + rj[17] + rj[12]) / 3.0  # shoulders + neck
    pelvis_offset = rj[0] - chest_center   # vector: chest->pelvis in SMPL space
    # We'll scale this by the COLMAP units (estimated from shoulder width in pixels)
    # so keep it as a unit-normalised direction; scale will be applied during init.
    pelvis_offset_dir = pelvis_offset  # already in SMPL units
    print(f"  Chest->pelvis offset (SMPL): {pelvis_offset_dir.round(3)}")

    # ── Per-frame init using weak-perspective from image landmarks ─────
    print("Initialising per-frame trans from 2D landmarks (no DLT)...")
    trans_init_np = _init_trans_per_frame_2d(
        lm_dict, images, cam_K, frame_names,
        smpl_shoulder_w, pelvis_offset_dir.astype(np.float64))
    print(f"  Init trans[0] = {trans_init_np[0].round(3)}")
    print(f"  Init trans range: {trans_init_np.min(0).round(2)} .. {trans_init_np.max(0).round(2)}")

    # ── Load silhouette distance transforms ───────────────────────────
    # Infer image size from camera params
    sample_info  = list(images.values())[0]
    # cam_K[1,2]*2 ≈ H; cam_K[0,2]*2 ≈ W
    H_img = int(round(cam_K[1, 2] * 2))
    W_img = int(round(cam_K[0, 2] * 2))
    dt_maps_np: dict[str, np.ndarray | None] = {}
    bbox_stats_np: dict[str, dict[str, float] | None] = {n: None for n in frame_names}
    if masks_dir is not None and Path(masks_dir).exists():
        dt_maps_np = _precompute_silhouette_dt(
            frame_names, Path(masks_dir), (H_img, W_img))
        bbox_stats_np = _precompute_mask_bbox_stats(
            frame_names, Path(masks_dir), (H_img, W_img))
    else:
        print("  No masks_dir -> silhouette loss disabled")
        dt_maps_np = {n: None for n in frame_names}

    # Convert DT maps to GPU tensors
    dt_tensors: list[torch.Tensor | None] = []
    for name in frame_names:
        dt = dt_maps_np.get(name)
        if dt is not None:
            dt_tensors.append(torch.from_numpy(dt).to(dev))
        else:
            dt_tensors.append(None)

    # Sample a subset of mesh vertex indices for silhouette loss (for speed)
    n_sil_verts = 512
    rng = np.random.default_rng(0)
    sil_vert_idx = rng.choice(smpl.n_verts, n_sil_verts, replace=False).tolist()

    # ── Build per-frame observation tensors ───────────────────────────
    K_t = torch.from_numpy(cam_K).float().to(dev)
    R_tensors  = []
    t_tensors  = []
    proj_mats  = []
    obs_pts    = []   # list of (k, 2) tensors
    obs_sidxs  = []   # list of [SMPL joint index, …]

    for name in frame_names:
        lms  = lm_dict[name]
        info = images[name]
        R_t  = torch.from_numpy(info["R"]).float().to(dev)
        tv   = torch.from_numpy(info["t"]).float().to(dev)
        R_tensors.append(R_t)
        t_tensors.append(tv)
        P    = K_t @ torch.cat([R_t, tv.unsqueeze(1)], dim=1)  # (3,4)
        proj_mats.append(P)

        pts, sidxs = [], []
        if lms is not None:
            for mp_idx, smpl_j in MP_TO_SMPL_PAIRS:
                x, y = float(lms[mp_idx, 0]), float(lms[mp_idx, 1])
                if not (np.isnan(x) or np.isnan(y)):
                    pts.append([x, y])
                    sidxs.append(smpl_j)
        obs_pts.append(
            torch.tensor(pts, dtype=torch.float32, device=dev) if pts else None)
        obs_sidxs.append(sidxs)

    # ── Parameters ────────────────────────────────────────────────────
    betas = nn.Parameter(torch.zeros(1, n_betas, device=dev))
    poses = nn.Parameter(torch.zeros(B, 72,      device=dev))
    trans = nn.Parameter(
        torch.from_numpy(trans_init_np).float().to(dev))  # (B, 3)

    # ── Loss helpers ──────────────────────────────────────────────────
    def _reproj_loss(verts_b: torch.Tensor, joints_b: torch.Tensor) -> torch.Tensor:
        """Mean squared 2D joint reprojection error."""
        loss = torch.zeros(1, device=dev)
        cnt  = 0
        for fi in range(B):
            if obs_pts[fi] is None or len(obs_pts[fi]) == 0:
                continue
            sidx = obs_sidxs[fi]
            k    = min(len(sidx), obs_pts[fi].shape[0])
            J3d  = joints_b[fi, sidx[:k]]
            J4d  = torch.cat([J3d, torch.ones(k, 1, device=dev)], 1)
            proj = (proj_mats[fi] @ J4d.T).T
            xy   = proj[:, :2] / proj[:, 2:3].clamp(min=0.01)
            diff = xy - obs_pts[fi][:k]
            loss = loss + (diff ** 2).sum()
            cnt  += k
        return loss / max(cnt, 1)

    def _sil_loss(verts_b: torch.Tensor) -> torch.Tensor:
        """Chamfer-style silhouette loss: penalize projected verts outside the mask."""
        loss = torch.zeros(1, device=dev)
        cnt  = 0
        for fi in range(B):
            dt = dt_tensors[fi]
            if dt is None:
                continue
            H_dt, W_dt = dt.shape
            V3  = verts_b[fi, sil_vert_idx]                      # (n_sv, 3)
            V4  = torch.cat([V3, torch.ones(len(sil_vert_idx), 1, device=dev)], 1)
            proj = (proj_mats[fi] @ V4.T).T                       # (n_sv, 3)
            xy   = proj[:, :2] / proj[:, 2:3].clamp(min=0.01)    # (n_sv, 2)
            # Normalise pixel coords to [-1, 1] for grid_sample
            xn = (xy[:, 0] / (W_dt - 1)) * 2.0 - 1.0
            yn = (xy[:, 1] / (H_dt - 1)) * 2.0 - 1.0
            grid = torch.stack([xn, yn], dim=1).unsqueeze(0).unsqueeze(0)  # (1,1,n_sv,2)
            dt4  = dt.unsqueeze(0).unsqueeze(0)                   # (1,1,H,W)
            vals = torch.nn.functional.grid_sample(
                dt4, grid, mode="bilinear", padding_mode="border", align_corners=True)
            loss = loss + vals.squeeze().sum()
            cnt  += len(sil_vert_idx)
        return loss / max(cnt, 1)

    def _bbox_loss(verts_b: torch.Tensor) -> torch.Tensor:
        """Penalize mismatch between projected mesh bbox and mask bbox/area."""
        loss = torch.zeros(1, device=dev)
        cnt = 0
        norm_xy = torch.tensor([W_img, H_img], device=dev, dtype=torch.float32)
        for fi in range(B):
            st = bbox_stats_np.get(frame_names[fi])
            if st is None:
                continue
            V3 = verts_b[fi, sil_vert_idx]
            V4 = torch.cat([V3, torch.ones(len(sil_vert_idx), 1, device=dev)], 1)
            proj = (proj_mats[fi] @ V4.T).T
            z = proj[:, 2]
            valid = z > 1e-3
            if not torch.any(valid):
                continue
            xy = proj[valid, :2] / z[valid, None].clamp(min=0.01)
            pred_min = xy.min(dim=0).values
            pred_max = xy.max(dim=0).values
            pred_wh = (pred_max - pred_min).clamp(min=1.0)
            pred_area = pred_wh[0] * pred_wh[1]

            tgt_min = torch.tensor([st["x0"], st["y0"]], device=dev)
            tgt_max = torch.tensor([st["x1"], st["y1"]], device=dev)
            tgt_wh = torch.tensor([st["w"], st["h"]], device=dev)
            tgt_area = torch.tensor(st["area"], device=dev)

            l_box = (torch.abs(pred_min - tgt_min) / norm_xy).mean() + (torch.abs(pred_max - tgt_max) / norm_xy).mean()
            l_wh = (torch.abs(pred_wh - tgt_wh) / norm_xy).mean()
            l_area = torch.abs(torch.log(pred_area.clamp(min=1.0)) - torch.log(tgt_area.clamp(min=1.0)))
            loss = loss + l_box + 0.75 * l_wh + 0.25 * l_area
            cnt += 1
        return loss / max(cnt, 1)

    def _forward():
        return smpl(betas.expand(B, -1), poses, trans)

    # ══ Stage 0: translation only (fix depth & XY) ════════════════════
    print(f"\nStage 0 ({n_iters_s0} iters) — translation only")
    opt0 = torch.optim.Adam([trans], lr=lr * 2)
    for it in range(n_iters_s0):
        opt0.zero_grad()
        vb, jb = _forward()
        loss = lambda_2d * _reproj_loss(vb, jb)
        loss.backward()
        opt0.step()
        if it % 50 == 0:
            print(f"  [{it:3d}] reproj={loss.item():.4f}")

    # ══ Stage 1: + global orientation (root pose only) ════════════════
    print(f"\nStage 1 ({n_iters_s1} iters) — + global orient")
    opt1 = torch.optim.Adam([trans, poses], lr=lr)
    for it in range(n_iters_s1):
        opt1.zero_grad()
        vb, jb   = _forward()
        loss_2d  = _reproj_loss(vb, jb)
        loss_sil = _sil_loss(vb)
        loss_box = _bbox_loss(vb)
        loss_body = (poses[:, 3:] ** 2).mean()
        loss = (
            lambda_2d * loss_2d
            + lambda_sil * (loss_sil + 0.6 * loss_box)
            + lambda_pose * 10.0 * loss_body
        )
        loss.backward()
        if poses.grad is not None:
            poses.grad[:, 3:] = 0.0   # only update root (0:3)
        opt1.step()
        if it % 50 == 0:
            print(f"  [{it:3d}] reproj={loss_2d.item():.4f}  sil={loss_sil.item():.4f} box={loss_box.item():.4f}")

    # ══ Stage 2: full (shape + pose + trans) ══════════════════════════
    print(f"\nStage 2 ({n_iters_s2} iters) — full: shape + pose + trans")
    opt2  = torch.optim.Adam([betas, poses, trans], lr=lr * 0.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, n_iters_s2, eta_min=lr * 0.02)
    for it in range(n_iters_s2):
        opt2.zero_grad()
        vb, jb    = _forward()
        loss_2d   = _reproj_loss(vb, jb)
        loss_sil  = _sil_loss(vb)
        loss_box  = _bbox_loss(vb)
        loss_beta = (betas ** 2).mean()
        loss_pose = (poses[:, 3:] ** 2).mean()
        loss_temp = (torch.abs(poses[1:] - poses[:-1]).mean() + 0.5 * torch.abs(trans[1:] - trans[:-1]).mean()) if B > 1 else torch.zeros(1, device=dev)
        loss_beta_bound = torch.relu(torch.abs(betas) - 3.0).mean()
        loss_pose_bound = torch.relu(torch.abs(poses[:, 3:]) - 1.8).mean()
        loss = (lambda_2d   * loss_2d
              + lambda_sil  * (loss_sil + 0.8 * loss_box)
              + lambda_beta * loss_beta
              + lambda_pose * (loss_pose + 0.25 * loss_temp)
              + 4.0 * loss_beta_bound
              + 1.5 * loss_pose_bound)
        loss.backward()
        opt2.step()
        sched.step()
        if it % 100 == 0:
            print(f"  [{it:3d}] reproj={loss_2d.item():.4f}  "
                  f"sil={loss_sil.item():.4f}  "
                                    f"box={loss_box.item():.4f}  "
                                    f"beta={loss_beta.item():.4f}  pose={loss_pose.item():.4f}  temp={loss_temp.item():.4f}")

    # ── Extract results ────────────────────────────────────────────────
    with torch.no_grad():
        betas_np   = betas[0].cpu().numpy()
        mean_pose  = poses.mean(0, keepdim=True)
        mean_trans = trans.mean(0, keepdim=True)
        v_out, j_out = smpl(betas.expand(1, -1), mean_pose, mean_trans)
        verts_np   = v_out[0].cpu().numpy()
        joints_np  = j_out[0].cpu().numpy()
        poses_np   = {frame_names[i]: poses[i].cpu().numpy() for i in range(B)}
        trans_np   = {frame_names[i]: trans[i].cpu().numpy() for i in range(B)}

    print(f"\nFinal beta = {betas_np.round(3)}")
    return dict(
        betas           = betas_np,
        scale           = 1.0,
        poses           = poses_np,
        trans_per_frame = trans_np,
        joints3d        = joints_np,
        verts           = verts_np,
        faces           = smpl.faces,
    )


# ══════════════════════════════════════════════════════════════════════
# OBJ export
# ══════════════════════════════════════════════════════════════════════

def save_skeleton_obj(
    joints3d:   np.ndarray,
    mp_indices: list[int],
    out_path:   Path,
):
    """Write sparse skeleton OBJ from DLT-triangulated joints."""
    idx_map = {}
    lines = ["# Sparse skeleton (DLT triangulation)\n"]
    vi = 1
    for ji, mp_idx in enumerate(mp_indices):
        pt = joints3d[ji]
        if np.any(np.isnan(pt)):
            continue
        lines.append(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
        idx_map[mp_idx] = vi
        vi += 1

    BONE_PAIRS = [
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (11, 23), (12, 24), (0, 11), (0, 12),
    ]
    for a, b in BONE_PAIRS:
        if a in idx_map and b in idx_map:
            lines.append(f"l {idx_map[a]} {idx_map[b]}\n")

    out_path.write_text("".join(lines))
    print(f"Saved skeleton -> {out_path}  ({vi-1} joints)")


def save_mesh_obj(verts: np.ndarray, faces: np.ndarray, out_path: Path):
    """Write SMPL mesh as OBJ (6890 verts, 13776 faces)."""
    lines = ["# SMPL mesh\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    out_path.write_text("".join(lines))
    print(f"Saved mesh -> {out_path}  ({len(verts)} verts, {len(faces)} faces)")