"""
fit_smpl_to_3d_mesh.py
========================
Fit SMPL body model to a reconstructed 3D mesh (PLY) using:
  • Chamfer distance:  SMPL surface <-> target mesh (bidirectional)
  • 2D reprojection:   MediaPipe landmarks projected from COLMAP cameras
  • Iterative refinement across 4 stages (coarse → fine)

Iterative stages
-----------------
  S0  global scale + translation               (shape/pose frozen)
  S1  + root orientation                       (shape frozen)
  S2  + shape betas + global scale             (pose fixed to neutral)
  S3  + per-frame joint pose                   (full, cosine-annealed LR)
  S4  per-frame pose refinement loops          (iterate S3 N times with NN refresh)

The key insight: using the 3D mesh as a direct geometric target is far more
stable than using DLT-triangulated joints from an orbiting camera setup.
The Chamfer loss naturally handles scale, orientation and shape simultaneously.

Usage
------
python fit_smpl_to_3d_mesh.py \
    --mesh          E:/V23D_Data/human_mesh_v6.ply \
    --colmap-dir    E:/V23D_Data/colmap_rerun/sparse/1 \
    --frames-dir    E:/V23D_Data/frames \
    --masks-dir     E:/V23D_Data/masks_rerun \
    --out-dir       E:/V23D_Data/smpl_v3 \
    --n-iters       2000 \
    --refine-iters  3
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

# ── project imports ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.recon.smpl_fitter import (
    SMPL,
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
    _build_K,
    MP_TO_SMPL_PAIRS,
    save_mesh_obj,
)
from src.pose.extract_mediapipe import (
    extract_landmarks_dir,
    load_landmarks_json,
    TRACKED_MP_INDICES,
)


# ══════════════════════════════════════════════════════════════════════════════
# Mesh loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load PLY → (verts, faces, colors).  Returns float32 verts (N,3)."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()
    verts  = np.asarray(mesh.vertices,       dtype=np.float32)
    faces  = np.asarray(mesh.triangles,      dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors,  dtype=np.float32) if mesh.has_vertex_colors() else None
    print(f"  Loaded mesh: {len(verts)} verts, {len(faces)} faces  [{path.name}]")
    return verts, faces, colors


def sample_mesh_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    n_samples: int = 20_000,
    seed: int = 0,
) -> np.ndarray:
    """Uniformly sample points on the mesh surface. Returns (n_samples, 3)."""
    rng = np.random.default_rng(seed)
    # face areas
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.maximum(areas, 1e-12)
    probs = areas / areas.sum()
    fi = rng.choice(len(faces), size=n_samples, p=probs)
    # barycentric
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    sqr1 = np.sqrt(r1)
    u = 1.0 - sqr1
    v = sqr1 * (1.0 - r2)
    w = sqr1 * r2
    pts = (u[:, None] * verts[faces[fi, 0]] +
           v[:, None] * verts[faces[fi, 1]] +
           w[:, None] * verts[faces[fi, 2]])
    return pts.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Rigid align SMPL to target (Procrustes)
# ══════════════════════════════════════════════════════════════════════════════

def procrustes_align(
    src: np.ndarray,  # (N,3) SMPL verts
    tgt: np.ndarray,  # (M,3) target mesh verts
    allow_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Find (scale, R, t) such that scale * R @ src.T + t[:,None] ≈ tgt.
    Uses a robust centroid + KDTree-based NN approach.
    Returns (scale, R, t) all as float64.
    """
    # Rough bounding-box align first (handles large translation offsets)
    src_c  = src.mean(0)
    tgt_c  = tgt.mean(0)

    # Scale from bounding-box diagonal ratio
    src_d  = np.linalg.norm(src.max(0) - src.min(0))
    tgt_d  = np.linalg.norm(tgt.max(0) - tgt.min(0))
    scale  = float(tgt_d / src_d) if src_d > 1e-6 else 1.0

    t = tgt_c - scale * src_c
    return scale, np.eye(3, dtype=np.float64), t.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Chamfer helpers (scipy KDTree, not differentiable, used for NN lookup)
# ══════════════════════════════════════════════════════════════════════════════

def chamfer_nn_targets(
    smpl_verts_np: np.ndarray,   # (6890, 3)  detached numpy
    tgt_pts:       np.ndarray,   # (M, 3)     target surface points
    tgt_kdtree:    cKDTree,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bidirectional NN:
      smpl_nn : (6890, 3) nearest target point for each SMPL vert
      tgt_nn  : (M, 3)    nearest SMPL vert for each target point
    """
    _, smpl_ii = tgt_kdtree.query(smpl_verts_np, workers=-1)   # (6890,)
    smpl_nn    = tgt_pts[smpl_ii]

    smpl_kdtree = cKDTree(smpl_verts_np)
    _, tgt_ii   = smpl_kdtree.query(tgt_pts, workers=-1)        # (M,)
    tgt_nn      = smpl_verts_np[tgt_ii]

    return smpl_nn.astype(np.float32), tgt_nn.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Frame selection & 2-D observation build
# ══════════════════════════════════════════════════════════════════════════════

def build_2d_observations(
    lm_dict:     dict,
    images:      dict,
    cam_K:       np.ndarray,
    n_frames:    int,
    min_visible: int,
    device,
):
    """Select frames and build projection matrices + 2D observations."""
    # quality filter
    candidates = []
    for name in sorted(lm_dict.keys()):
        lms = lm_dict.get(name)
        if lms is None or name not in images:
            continue
        nv = sum(1 for mi in TRACKED_MP_INDICES if not np.isnan(lms[mi, :2]).any())
        if nv >= min_visible:
            candidates.append(name)
    step   = max(1, len(candidates) // n_frames)
    frames = candidates[::step][:n_frames]
    print(f"  Using {len(frames)}/{len(candidates)} eligible frames for 2D loss")

    K_t = torch.from_numpy(cam_K.astype(np.float32)).to(device)

    proj_mats, obs_pts, obs_sidxs = [], [], []
    R_list, t_list = [], []

    for name in frames:
        info = images[name]
        R = torch.from_numpy(info["R"].astype(np.float32)).to(device)
        t = torch.from_numpy(info["t"].astype(np.float32)).to(device)
        R_list.append(R); t_list.append(t)
        P = K_t @ torch.cat([R, t.unsqueeze(1)], 1)
        proj_mats.append(P)

        lms  = lm_dict[name]
        pts, sidxs = [], []
        if lms is not None:
            for mi, sj in MP_TO_SMPL_PAIRS:
                x, y = float(lms[mi, 0]), float(lms[mi, 1])
                if not (np.isnan(x) or np.isnan(y)):
                    pts.append([x, y])
                    sidxs.append(sj)
        obs_pts.append(
            torch.tensor(pts, dtype=torch.float32, device=device) if pts else None)
        obs_sidxs.append(sidxs)

    return frames, proj_mats, obs_pts, obs_sidxs, R_list, t_list


# ══════════════════════════════════════════════════════════════════════════════
# Main fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_smpl_to_mesh(
    smpl_model_path: Path,
    target_mesh_path: Path,
    lm_dict:          dict,
    images:           dict,
    cam_K:            np.ndarray,
    masks_dir:        Path | None = None,
    # sampling
    n_surface_pts:    int   = 15_000,
    n_frames:         int   = 40,
    min_visible:      int   = 8,
    # model
    n_betas:          int   = 10,
    # iterations
    n_iters_s0:       int   = 300,
    n_iters_s1:       int   = 300,
    n_iters_s2:       int   = 400,
    n_iters_s3:       int   = 800,
    n_refine_loops:   int   = 3,    # repeat S3 with NN refresh
    lr:               float = 3e-3,
    # loss weights
    lambda_chamfer:   float = 10.0,
    lambda_2d:        float = 1.0,
    lambda_beta:      float = 5.0,
    lambda_pose:      float = 0.5,
    device:           str   = "cuda",
) -> dict:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # ── Load target mesh ─────────────────────────────────────────────
    tgt_verts, tgt_faces, _ = load_ply_mesh(target_mesh_path)
    print(f"  Sampling {n_surface_pts} surface points from target...")
    tgt_pts = sample_mesh_surface(tgt_verts, tgt_faces, n_surface_pts)
    tgt_kdtree = cKDTree(tgt_pts)
    tgt_pts_t  = torch.from_numpy(tgt_pts).to(dev)

    # ── Load SMPL ────────────────────────────────────────────────────
    smpl = SMPL(smpl_model_path, n_betas=n_betas).to(dev)

    # ── SMPL rest-pose measurements ──────────────────────────────────
    with torch.no_grad():
        _, rj = smpl(torch.zeros(1, n_betas, device=dev),
                     torch.zeros(1, 72, device=dev),
                     torch.zeros(1, 3,  device=dev))
    rj_np = rj[0].cpu().numpy()

    # ── Procrustes init: align SMPL verts to target ──────────────────
    smpl_rest_np, _ = [x.cpu().numpy() for x in smpl(
        torch.zeros(1, n_betas, device=dev),
        torch.zeros(1, 72, device=dev),
        torch.zeros(1, 3,  device=dev)
    )[:2]]
    smpl_rest_np = smpl_rest_np[0]

    init_scale, init_R, init_t = procrustes_align(smpl_rest_np, tgt_verts)
    print(f"  Procrustes init: scale={init_scale:.4f}  t={init_t.round(3)}")

    # ── 2-D observations ─────────────────────────────────────────────
    frames, proj_mats, obs_pts, obs_sidxs, R_list, t_list = build_2d_observations(
        lm_dict, images, cam_K, n_frames, min_visible, dev)
    B = len(frames)

    # ── Silhouette DT ─────────────────────────────────────────────────
    from src.recon.smpl_fitter import _precompute_silhouette_dt
    H_img = int(round(cam_K[1, 2] * 2))
    W_img = int(round(cam_K[0, 2] * 2))
    dt_tensors: list[torch.Tensor | None] = []
    if masks_dir is not None and masks_dir.exists():
        from src.recon.smpl_fitter import _precompute_silhouette_dt
        dt_np = _precompute_silhouette_dt(frames, masks_dir, (H_img, W_img))
        for name in frames:
            dt = dt_np.get(name)
            dt_tensors.append(torch.from_numpy(dt).to(dev) if dt is not None else None)
    else:
        dt_tensors = [None] * B

    n_sil = 512
    sil_idx = np.random.default_rng(0).choice(smpl.n_verts, n_sil, replace=False).tolist()

    # ── Parameters ───────────────────────────────────────────────────
    log_scale = nn.Parameter(torch.tensor(np.log(init_scale), dtype=torch.float32, device=dev))
    trans_g   = nn.Parameter(torch.tensor(init_t,             dtype=torch.float32, device=dev))
    betas     = nn.Parameter(torch.zeros(1, n_betas,           device=dev))
    # per-frame pose — root orient only first
    poses     = nn.Parameter(torch.zeros(B, 72,                device=dev))

    # ── Cached NN targets (refreshed each refine loop) ───────────────
    smpl_nn_t  = torch.zeros(smpl.n_verts, 3, device=dev)
    tgt_nn_t   = torch.zeros(n_surface_pts, 3, device=dev)

    def _refresh_nn(verts_np: np.ndarray):
        smpl_nn, tgt_nn = chamfer_nn_targets(verts_np, tgt_pts, tgt_kdtree)
        smpl_nn_t.copy_(torch.from_numpy(smpl_nn))
        tgt_nn_t.copy_(torch.from_numpy(tgt_nn))

    # ── Forward (single global pose, shape, scale, trans) ────────────
    # During S0/S1/S2 we use a single global pose applied to all frames.
    # During S3 each frame has its own pose.
    def _forward_global():
        """Single global pose (mean of poses param)."""
        s = torch.exp(log_scale)
        tg = trans_g.unsqueeze(0)   # (1, 3)
        mean_pose = poses.mean(0, keepdim=True)
        vb, jb = smpl(betas.expand(1, -1), mean_pose, tg / s)
        vb = vb * s + tg - (tg / s) * s   # apply scale+trans differently:
        # Actually: place model at trans_g with scale
        vb2, jb2 = _forward_with_params(betas, mean_pose.expand(1, -1),
                                        trans_g.unsqueeze(0), torch.exp(log_scale))
        return vb2, jb2

    def _forward_with_params(b, p, tr, s):
        """verts = scale * SMPL(b,p,0) + tr"""
        vb, jb = smpl(b, p, torch.zeros_like(tr))
        vb = vb * s + tr.unsqueeze(1)
        jb = jb * s + tr.unsqueeze(1)
        return vb, jb

    def _forward_per_frame():
        s  = torch.exp(log_scale)
        tr = trans_g.unsqueeze(0).expand(B, -1)   # broadcast global trans
        return _forward_with_params(betas.expand(B, -1), poses, tr, s)

    # ── Chamfer loss ─────────────────────────────────────────────────
    def _chamfer(verts_single: torch.Tensor) -> torch.Tensor:
        """verts_single: (N_v, 3)"""
        # S->T  (SMPL verts to nearest target point)
        loss_st = ((verts_single - smpl_nn_t) ** 2).sum(-1).mean()
        # T->S  (target surface pts to nearest SMPL vert — use fixed targets)
        loss_ts = ((tgt_pts_t - tgt_nn_t) ** 2).sum(-1).mean()
        return loss_st + loss_ts

    # ── Reprojection loss ─────────────────────────────────────────────
    def _reproj(joints_b: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=dev)
        cnt  = 0
        for fi in range(B):
            if obs_pts[fi] is None or len(obs_pts[fi]) == 0:
                continue
            sidx = obs_sidxs[fi]
            k = min(len(sidx), obs_pts[fi].shape[0])
            J3  = joints_b[fi, sidx[:k]]
            J4  = torch.cat([J3, torch.ones(k, 1, device=dev)], 1)
            prj = (proj_mats[fi] @ J4.T).T
            xy  = prj[:, :2] / prj[:, 2:3].clamp(min=0.01)
            loss = loss + ((xy - obs_pts[fi][:k]) ** 2).sum()
            cnt += k
        return loss / max(cnt, 1)

    # ── Silhouette loss ───────────────────────────────────────────────
    def _sil(verts_b: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=dev)
        cnt  = 0
        for fi in range(B):
            dt = dt_tensors[fi]
            if dt is None:
                continue
            H_dt, W_dt = dt.shape
            V3  = verts_b[fi, sil_idx]
            V4  = torch.cat([V3, torch.ones(n_sil, 1, device=dev)], 1)
            prj = (proj_mats[fi] @ V4.T).T
            xy  = prj[:, :2] / prj[:, 2:3].clamp(min=0.01)
            xn  = (xy[:, 0] / (W_dt - 1)) * 2.0 - 1.0
            yn  = (xy[:, 1] / (H_dt - 1)) * 2.0 - 1.0
            grid = torch.stack([xn, yn], 1).unsqueeze(0).unsqueeze(0)
            vals = torch.nn.functional.grid_sample(
                dt.unsqueeze(0).unsqueeze(0), grid,
                mode="bilinear", padding_mode="border", align_corners=True)
            loss = loss + vals.squeeze().sum()
            cnt += n_sil
        return loss / max(cnt, 1)

    def _reg():
        return lambda_beta * (betas ** 2).mean() + lambda_pose * (poses[:, 3:] ** 2).mean()

    # ══════════════════════════════════════════════════════════════════
    # Initial NN targets (using rest-pose scaled to target)
    print("\nComputing initial NN correspondences...")
    with torch.no_grad():
        vb0, _ = _forward_per_frame()
    _refresh_nn(vb0[0].cpu().numpy())

    # ══ Stage 0: scale + translation ═══════════════════════════════════
    print(f"\nStage 0 ({n_iters_s0} iters) — scale + translation")
    opt0 = torch.optim.Adam([log_scale, trans_g], lr=lr * 2)
    for it in range(n_iters_s0):
        opt0.zero_grad()
        vb, jb = _forward_per_frame()
        loss_c = _chamfer(vb[0])
        loss_r = _reproj(jb)
        loss   = lambda_chamfer * loss_c + lambda_2d * loss_r
        loss.backward()
        opt0.step()
        if it % 100 == 0:
            print(f"  [{it:3d}] chamfer={loss_c.item():.5f}  reproj={loss_r.item():.4f}"
                  f"  scale={torch.exp(log_scale).item():.4f}")
        if it % 50 == 0:  # refresh NNs periodically
            with torch.no_grad():
                vb_np = vb[0].detach().cpu().numpy()
            _refresh_nn(vb_np)

    # ══ Stage 1: + root orientation ════════════════════════════════════
    print(f"\nStage 1 ({n_iters_s1} iters) — + root orientation")
    opt1 = torch.optim.Adam([log_scale, trans_g, poses], lr=lr)
    for it in range(n_iters_s1):
        opt1.zero_grad()
        vb, jb = _forward_per_frame()
        loss_c = _chamfer(vb[0])
        loss_r = _reproj(jb)
        loss_p = (poses[:, 3:] ** 2).mean()
        loss   = lambda_chamfer * loss_c + lambda_2d * loss_r + lambda_pose * 10.0 * loss_p
        loss.backward()
        if poses.grad is not None:
            poses.grad[:, 3:].zero_()   # only root 0:3
        opt1.step()
        if it % 100 == 0:
            print(f"  [{it:3d}] chamfer={loss_c.item():.5f}  reproj={loss_r.item():.4f}")
        if it % 50 == 0:
            with torch.no_grad():
                vb_np = vb[0].detach().cpu().numpy()
            _refresh_nn(vb_np)

    # ══ Stage 2: + shape betas ══════════════════════════════════════════
    print(f"\nStage 2 ({n_iters_s2} iters) — + shape betas")
    opt2 = torch.optim.Adam([log_scale, trans_g, betas, poses], lr=lr * 0.7)
    for it in range(n_iters_s2):
        opt2.zero_grad()
        vb, jb = _forward_per_frame()
        loss_c = _chamfer(vb[0])
        loss_r = _reproj(jb)
        loss_b = (betas ** 2).mean()
        loss_p = (poses[:, 3:] ** 2).mean()
        loss   = (lambda_chamfer * loss_c + lambda_2d * loss_r
                + lambda_beta * loss_b + lambda_pose * 5.0 * loss_p)
        loss.backward()
        if poses.grad is not None:
            poses.grad[:, 3:].zero_()   # still only root during shape stage
        opt2.step()
        if it % 100 == 0:
            print(f"  [{it:3d}] chamfer={loss_c.item():.5f}  reproj={loss_r.item():.4f}"
                  f"  beta={loss_b.item():.4f}")
        if it % 100 == 0:
            with torch.no_grad():
                vb_np = vb[0].detach().cpu().numpy()
            _refresh_nn(vb_np)

    # ══ Stage 3: full pose + iterative NN refinement ═══════════════════
    print(f"\nStage 3 ({n_iters_s3} iters × {n_refine_loops} loops) — full pose + iterative refinement")
    for loop in range(n_refine_loops):
        print(f"\n  --- Refinement loop {loop+1}/{n_refine_loops} ---")
        # Refresh NNs
        with torch.no_grad():
            vb_np = vb[0].detach().cpu().numpy()
        _refresh_nn(vb_np)

        opt3 = torch.optim.Adam([log_scale, trans_g, betas, poses], lr=lr * 0.3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt3, n_iters_s3, eta_min=lr * 0.01)
        lambda_sil = 0.003 * (loop + 1)  # increase sil weight each loop

        for it in range(n_iters_s3):
            opt3.zero_grad()
            vb, jb = _forward_per_frame()
            loss_c = _chamfer(vb[0])
            loss_r = _reproj(jb)
            loss_s = _sil(vb)
            loss_b = (betas ** 2).mean()
            loss_p = (poses[:, 3:] ** 2).mean()
            loss   = (lambda_chamfer * loss_c
                    + lambda_2d      * loss_r
                    + lambda_sil     * loss_s
                    + lambda_beta    * loss_b
                    + lambda_pose    * loss_p)
            loss.backward()
            opt3.step()
            sched.step()
            if it % 200 == 0:
                print(f"    [{it:3d}] chamfer={loss_c.item():.5f}  reproj={loss_r.item():.4f}"
                      f"  sil={loss_s.item():.4f}  beta={loss_b.item():.4f}")
            if it % 100 == 0:
                with torch.no_grad():
                    vb_np = vb[0].detach().cpu().numpy()
                _refresh_nn(vb_np)

    # ── Extract results ────────────────────────────────────────────────
    with torch.no_grad():
        scale_val  = float(torch.exp(log_scale).item())
        betas_np   = betas[0].cpu().numpy()
        trans_np   = trans_g.cpu().numpy()
        mean_pose  = poses.mean(0, keepdim=True)

        # Canonical mesh: mean shape + rest pose + global scale
        v_can, _ = _forward_with_params(
            betas, torch.zeros(1, 72, device=dev),
            trans_g.unsqueeze(0), torch.exp(log_scale))
        verts_canon = v_can[0].cpu().numpy()

        # Per-frame
        v_all, j_all = _forward_per_frame()
        poses_dict = {frames[i]: poses[i].cpu().numpy() for i in range(B)}
        trans_dict = {frames[i]: trans_np for i in range(B)}   # global trans shared

    print(f"\nFinal: scale={scale_val:.4f}  beta={betas_np.round(3)}")
    return dict(
        betas           = betas_np,
        scale           = scale_val,
        trans           = trans_np,
        poses           = poses_dict,
        trans_per_frame = trans_dict,
        joints3d        = j_all[0].cpu().numpy(),
        verts           = verts_canon,
        faces           = smpl.faces,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Fit SMPL to reconstructed 3D mesh.")
    ap.add_argument("--mesh",          required=True,
                    help="Reconstructed PLY mesh (e.g. human_mesh_v6.ply)")
    ap.add_argument("--colmap-dir",    required=True,
                    help="COLMAP sparse dir (cameras.txt + images.txt)")
    ap.add_argument("--frames-dir",    required=True,
                    help="Frames directory (JPG/PNG)")
    ap.add_argument("--masks-dir",     default=None)
    ap.add_argument("--out-dir",       required=True)
    ap.add_argument("--landmarks-json",default=None,
                    help="Pre-computed MediaPipe landmarks JSON")
    _DEFAULT_SMPL = (
        r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models"
        r"\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    )
    ap.add_argument("--smpl-model",    default=_DEFAULT_SMPL)
    ap.add_argument("--n-surface-pts", type=int, default=15000)
    ap.add_argument("--n-frames",      type=int, default=40)
    ap.add_argument("--n-betas",       type=int, default=10)
    ap.add_argument("--n-iters",       type=int, default=1800,
                    help="Iterations per stage (split 300/300/400/800 by default)")
    ap.add_argument("--refine-loops",  type=int, default=3,
                    help="Number of iterative NN-refresh refinement loops in S3")
    ap.add_argument("--lr",            type=float, default=3e-3)
    ap.add_argument("--device",        default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load COLMAP
    cams_txt   = Path(args.colmap_dir) / "cameras.txt"
    images_txt = Path(args.colmap_dir) / "images.txt"
    cameras    = _read_colmap_cameras_txt(cams_txt)
    images     = _read_colmap_images_txt(images_txt)
    cam_K      = _build_K(cameras[list(cameras.keys())[0]])
    print(f"COLMAP: {len(images)} images  K=f{cam_K[0,0]:.1f}")

    # Load landmarks
    lm_path = (Path(args.landmarks_json)
                if args.landmarks_json and Path(args.landmarks_json).exists()
                else out_dir / "landmarks_mediapipe.json")
    if lm_path.exists():
        print(f"Loading cached landmarks: {lm_path}")
        lm_dict = load_landmarks_json(lm_path)
    else:
        print(f"Extracting MediaPipe landmarks -> {lm_path}")
        lm_dict = extract_landmarks_dir(Path(args.frames_dir), lm_path)

    # Distribute iters across stages (s0=17%, s1=17%, s2=22%, s3=44%)
    n  = args.n_iters
    s0 = max(50,  int(n * 0.17))
    s1 = max(50,  int(n * 0.17))
    s2 = max(100, int(n * 0.22))
    s3 = max(100, n - s0 - s1 - s2)

    result = fit_smpl_to_mesh(
        smpl_model_path  = Path(args.smpl_model),
        target_mesh_path = Path(args.mesh),
        lm_dict          = lm_dict,
        images           = images,
        cam_K            = cam_K,
        masks_dir        = Path(args.masks_dir) if args.masks_dir else None,
        n_surface_pts    = args.n_surface_pts,
        n_frames         = args.n_frames,
        n_betas          = args.n_betas,
        n_iters_s0       = s0,
        n_iters_s1       = s1,
        n_iters_s2       = s2,
        n_iters_s3       = s3,
        n_refine_loops   = args.refine_loops,
        lr               = args.lr,
        device           = args.device,
    )

    # Save outputs
    save_mesh_obj(result["verts"], result["faces"], out_dir / "smpl_canonical.obj")
    np.save(str(out_dir / "betas.npy"),          result["betas"])
    np.save(str(out_dir / "scale.npy"),          np.array([result["scale"]], dtype=np.float32))
    np.save(str(out_dir / "trans_global.npy"),   result["trans"])
    (out_dir / "poses_per_frame.json").write_text(
        json.dumps({k: v.tolist() for k, v in result["poses"].items()}, indent=2))
    np.save(str(out_dir / "trans_per_frame.npy"),
            {k: v.tolist() for k, v in result["trans_per_frame"].items()})

    print(f"\nDone. Outputs in {out_dir}")
    print(f"  smpl_canonical.obj  betas.npy  scale.npy  trans_global.npy")


if __name__ == "__main__":
    main()
