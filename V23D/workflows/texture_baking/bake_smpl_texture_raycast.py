"""
bake_smpl_texture_raycast.py
==============================
Bake a texture onto a fitted SMPL mesh using per-camera depth-buffer
visibility testing (software rasterisation + raycast-style occlusion check).

Pipeline
---------
1.  Load SMPL (betas, scale, trans_global) → canonical mesh
2.  UV-unwrap with xatlas → texel <=> triangle mapping
3.  For each texel: compute its 3D world position from barycentric coords
4.  For each camera:
      a.  Render a depth buffer of the SMPL mesh (Z-buffer rasterisation)
      b.  Project each texel's 3D point into the camera image plane
      c.  Compare z-depth against depth buffer → visibility
      d.  Weight by face normal · view direction (front-face favour)
5.  Blend visible camera samples (weighted average)
6.  Dilate atlas (pull-push) to fill seam/border gaps
7.  Save OBJ + MTL + PNG  (and optionally ROMP NPZ)

Dependencies
-----------
    xatlas  trimesh  cv2  PIL  numpy  scipy  torch (only for SMPL forward)
    open3d  (only for loading COLMAP cameras via shared code)

Usage
------
python bake_smpl_texture_raycast.py \
    --out-dir     E:/V23D_Data/smpl_v3 \
    --frames-dir  E:/V23D_Data/frames \
    --colmap-dir  E:/V23D_Data/colmap_rerun/sparse/1 \
    --masks-dir   E:/V23D_Data/masks_rerun \
    --tex-size    2048 \
    --blend-n     5

The script expects --out-dir to contain:
    smpl_canonical.obj  (or loaded directly from SMPL+betas)
    betas.npy
    scale.npy
    trans_global.npy   (or falls back to trans_per_frame mean)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
import xatlas

# ── project imports ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from src.recon.smpl_fitter import (
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
    _build_K,
)


# ══════════════════════════════════════════════════════════════════════════════
# COLMAP camera helpers (re-used from bake_texture.py pattern)
# ══════════════════════════════════════════════════════════════════════════════

def _quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
        [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
        [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])


def _build_cameras(cameras, images, frames_dir):
    """Build list of camera dicts with K, R, t, img_path."""
    frames_dir = Path(frames_dir)
    cams = []
    for info in images.values():
        cam_id  = info["cam_id"]
        cam_def = cameras[cam_id]
        K       = _build_K(cam_def).astype(np.float32)
        R       = info["R"].astype(np.float32)
        t       = info["t"].astype(np.float32)
        W, H    = cam_def["w"], cam_def["h"]
        # Find frame file
        name     = info.get("name", "")
        img_path = frames_dir / name
        if not img_path.exists():
            stem = Path(name).stem
            for ext in (".jpg", ".jpeg", ".png"):
                p = frames_dir / (stem + ext)
                if p.exists():
                    img_path = p
                    break
        cams.append(dict(K=K, R=R, t=t, W=W, H=H, img_path=img_path, name=name))
    return cams


# ══════════════════════════════════════════════════════════════════════════════
# Mesh loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_obj(path: Path):
    """Minimal OBJ reader. Returns verts (N,3), faces (F,3) as int64."""
    verts, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(tri)
    return np.array(verts, np.float32), np.array(faces, np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# UV unwrap
# ══════════════════════════════════════════════════════════════════════════════

def unwrap_uv(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    xatlas UV unwrap. Returns:
        uv_verts  (N_uv, 2)  float32  in [0,1]
        uv_faces  (F, 3)     int64    indices into uv_verts
    """
    print("  UV-unwrapping with xatlas...", end="", flush=True)
    vmapping, indices, uvs = xatlas.parametrize(verts, faces.astype(np.uint32))
    print(f"  done — {len(uvs)} UV verts, {len(indices)} faces")
    return uvs.astype(np.float32), indices.astype(np.int64), vmapping.astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# Z-buffer rasteriser (software, per-camera)
# ══════════════════════════════════════════════════════════════════════════════

def _rasterise_depth(
    verts:  np.ndarray,   # (N, 3) world-space
    faces:  np.ndarray,   # (F, 3)
    K:      np.ndarray,   # (3, 3)
    R:      np.ndarray,   # (3, 3)
    t:      np.ndarray,   # (3,)
    W: int,
    H: int,
    near: float = 0.01,
) -> np.ndarray:
    """
    Render Z-buffer (depth in camera space). Returns float32 (H, W).
    Pixels with no geometry get +inf.
    Fully vectorised numpy scan-line rasteriser.
    """
    # Transform to camera space
    verts_cam = (R @ verts.T).T + t          # (N, 3)
    depth_buf = np.full((H, W), np.inf, dtype=np.float32)

    f  = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Project all verts to pixel
    z  = verts_cam[:, 2]
    z_safe = np.where(z > near, z, 1.0)
    px = (f * verts_cam[:, 0] / z_safe + cx).astype(np.float32)
    py = (f * verts_cam[:, 1] / z_safe + cy).astype(np.float32)

    # Face vertex z-depths and pixel positions
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    z0 = z[i0]; z1 = z[i1]; z2 = z[i2]
    x0 = px[i0]; x1 = px[i1]; x2 = px[i2]
    y0 = py[i0]; y1 = py[i1]; y2 = py[i2]

    # Only process faces fully in front of camera
    front = (z0 > near) & (z1 > near) & (z2 > near)

    for fi in np.where(front)[0]:
        _x0, _y0 = float(x0[fi]), float(y0[fi])
        _x1, _y1 = float(x1[fi]), float(y1[fi])
        _x2, _y2 = float(x2[fi]), float(y2[fi])
        _z0, _z1, _z2 = float(z0[fi]), float(z1[fi]), float(z2[fi])

        xmin = max(0, int(min(_x0, _x1, _x2)))
        xmax = min(W - 1, int(max(_x0, _x1, _x2)) + 1)
        ymin = max(0, int(min(_y0, _y1, _y2)))
        ymax = min(H - 1, int(max(_y0, _y1, _y2)) + 1)
        if xmin > xmax or ymin > ymax:
            continue

        area = (_x1 - _x0) * (_y2 - _y0) - (_x2 - _x0) * (_y1 - _y0)
        if abs(area) < 1e-6:
            continue
        inv_a = 1.0 / area

        gx, gy = np.meshgrid(
            np.arange(xmin, xmax + 1, dtype=np.float32),
            np.arange(ymin, ymax + 1, dtype=np.float32))

        w0 = ((_x1 - _x0) * (gy - _y0) - (gx - _x0) * (_y1 - _y0)) * inv_a  # w2
        w1 = ((_x2 - _x0) * (gy - _y0) - (gx - _x0) * (_y2 - _y0))           # ← wrong sign
        # Correct barycentric via edge functions
        w0 = ((gx - _x1) * (_y2 - _y1) - (gy - _y1) * (_x2 - _x1)) * inv_a
        w1 = ((gx - _x2) * (_y0 - _y2) - (gy - _y2) * (_x0 - _x2)) * inv_a
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z_interp = (w0 * _z0 + w1 * _z1 + w2 * _z2).astype(np.float32)
        gy_i = gy.astype(np.int32)
        gx_i = gx.astype(np.int32)
        rows = gy_i[inside]
        cols = gx_i[inside]
        zvals = z_interp[inside]

        # Scatter min — use a flattened index
        flat = rows * W + cols
        order = np.argsort(flat)
        flat_s  = flat[order]
        zvals_s = zvals[order]
        rows_s  = rows[order]
        cols_s  = cols[order]
        # For duplicate flat indices keep only minimum z
        _, first = np.unique(flat_s, return_index=True)
        for idx in first:
            r_, c_, zv_ = int(rows_s[idx]), int(cols_s[idx]), float(zvals_s[idx])
            if zv_ < depth_buf[r_, c_]:
                depth_buf[r_, c_] = zv_

    return depth_buf


def _project_points(
    pts:  np.ndarray,   # (N, 3)
    K:    np.ndarray,
    R:    np.ndarray,
    t:    np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns pixel coords (N, 2) and z-depths (N,)."""
    cam = (R @ pts.T).T + t
    z   = cam[:, 2]
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    u = f * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = f * cam[:, 1] / np.maximum(z, 1e-6) + cy
    return np.stack([u, v], axis=1), z


# ══════════════════════════════════════════════════════════════════════════════
# Face normals
# ══════════════════════════════════════════════════════════════════════════════

def _compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n  = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(mag, 1e-8)


def _compute_vertex_normals(verts, faces, face_normals):
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], face_normals)
    mag = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(mag, 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# Texel sampling
# ══════════════════════════════════════════════════════════════════════════════

def _generate_texel_positions(
    uv_verts:  np.ndarray,  # (N_uv, 2) float32
    uv_faces:  np.ndarray,  # (F, 3) int64
    vmapping:  np.ndarray,  # (N_uv,) → original vert index
    verts:     np.ndarray,  # (N_orig, 3)
    normals:   np.ndarray,  # (N_orig, 3) vertex normals
    tex_size:  int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterise the UV atlas and compute for each texel:
      - 3D world position
      - 3D surface normal
      - face index
    Returns:
      valid_mask  (H, W)  bool
      texel_pos   (H, W, 3) float32   NaN where no triangle
      texel_nrm   (H, W, 3) float32
      texel_fidx  (H, W)    int32    -1 where no triangle
    """
    H = W = tex_size
    texel_pos  = np.full((H, W, 3), np.nan, dtype=np.float32)
    texel_nrm  = np.full((H, W, 3), np.nan, dtype=np.float32)
    texel_fidx = np.full((H, W),     -1,    dtype=np.int32)

    print(f"  Rasterising {len(uv_faces)} UV faces into {H}x{W} atlas...", flush=True)

    for fi in range(len(uv_faces)):
        i0, i1, i2 = uv_faces[fi]
        # UV coords scaled to pixel space
        u0, v0 = uv_verts[i0] * (tex_size - 1)
        u1, v1 = uv_verts[i1] * (tex_size - 1)
        u2, v2 = uv_verts[i2] * (tex_size - 1)
        # (u=column, v=row) — xatlas uses V=down convention
        c0, r0 = float(u0), float(v0)
        c1, r1 = float(u1), float(v1)
        c2, r2 = float(u2), float(v2)

        rmin = max(0, int(min(r0, r1, r2)))
        rmax = min(H - 1, int(max(r0, r1, r2)) + 1)
        cmin = max(0, int(min(c0, c1, c2)))
        cmax = min(W - 1, int(max(c0, c1, c2)) + 1)
        if rmin > rmax or cmin > cmax:
            continue

        area = (c1 - c0) * (r2 - r0) - (c2 - c0) * (r1 - r0)
        if abs(area) < 1e-6:
            continue
        inv_a = 1.0 / area

        rr = np.arange(rmin, rmax + 1)
        cc = np.arange(cmin, cmax + 1)
        gc, gr = np.meshgrid(cc, rr)
        gcf = gc.astype(np.float32)
        grf = gr.astype(np.float32)

        # Barycentric weights
        w1 = ((c1 - c0) * (grf - r0) - (gcf - c0) * (r1 - r0)) * inv_a  # w2
        w2 = ((c2 - c0) * (grf - r0) - (gcf - c0) * (r2 - r0))           # not quite...
        # Correct formula:
        w0 = ((gcf - c1) * (r2 - r1) - (grf - r1) * (c2 - c1)) * inv_a
        w1 = ((gcf - c2) * (r0 - r2) - (grf - r2) * (c0 - c2)) * inv_a
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not inside.any():
            continue

        # 3D positions
        ov0 = vmapping[i0]; ov1 = vmapping[i1]; ov2 = vmapping[i2]
        p3d = (w0[inside, None] * verts[ov0]
             + w1[inside, None] * verts[ov1]
             + w2[inside, None] * verts[ov2])
        n3d = (w0[inside, None] * normals[ov0]
             + w1[inside, None] * normals[ov1]
             + w2[inside, None] * normals[ov2])
        nmag = np.linalg.norm(n3d, axis=1, keepdims=True)
        n3d  = n3d / np.maximum(nmag, 1e-8)

        rows_in = gr[inside]
        cols_in = gc[inside]
        texel_pos [rows_in, cols_in] = p3d
        texel_nrm [rows_in, cols_in] = n3d
        texel_fidx[rows_in, cols_in] = fi

        if fi % 2000 == 0:
            print(f"    face {fi}/{len(uv_faces)}", end="\r", flush=True)

    print()
    valid_mask = texel_fidx >= 0
    print(f"  Valid texels: {valid_mask.sum()}")
    return valid_mask, texel_pos, texel_nrm, texel_fidx


# ══════════════════════════════════════════════════════════════════════════════
# Per-camera bake
# ══════════════════════════════════════════════════════════════════════════════

def bake_texture(
    verts:       np.ndarray,        # (N, 3) SMPL canonical
    faces:       np.ndarray,        # (F, 3)
    cameras:     list[dict],
    tex_size:    int    = 2048,
    blend_n:     int    = 5,        # top-N cameras per texel
    depth_tol:   float  = 0.02,     # depth tolerance for visibility (metres-ish)
    masks_dir:   Path | None = None,
    n_cam_limit: int    = 0,        # 0 = all cameras
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        texture_rgb   (H, W, 3)  uint8
        texel_weight  (H, W)     float32  (coverage map)
    """
    import math

    # Face normals
    face_normals = _compute_face_normals(verts, faces)
    vert_normals = _compute_vertex_normals(verts, faces, face_normals)

    # UV unwrap
    uv_verts, uv_faces, vmapping = unwrap_uv(verts, faces)

    # Texel 3D positions
    valid_mask, texel_pos, texel_nrm, texel_fidx = _generate_texel_positions(
        uv_verts, uv_faces, vmapping, verts, vert_normals, tex_size)

    H = W = tex_size
    valid_yx = np.argwhere(valid_mask)          # (N_valid, 2)
    valid_pts = texel_pos[valid_mask]           # (N_valid, 3)
    valid_nrm = texel_nrm[valid_mask]           # (N_valid, 3)
    N_valid = len(valid_pts)
    print(f"  Baking {N_valid} valid texels from {len(cameras)} cameras")

    # Accumulate colors and weights
    accum_rgb = np.zeros((N_valid, 3), dtype=np.float64)
    accum_w   = np.zeros(N_valid,     dtype=np.float64)

    # Per-texel top-N tracking
    # top_colors: (N_valid, blend_n, 3), top_weights: (N_valid, blend_n)
    top_colors  = np.zeros((N_valid, blend_n, 3), dtype=np.float32)
    top_weights = np.full((N_valid, blend_n), -np.inf, dtype=np.float32)

    cam_list = cameras
    if n_cam_limit > 0:
        step = max(1, len(cameras) // n_cam_limit)
        cam_list = cameras[::step][:n_cam_limit]
    print(f"  Using {len(cam_list)} cameras for bake")

    for ci, cam in enumerate(cam_list):
        K, R, t, cW, cH = cam["K"], cam["R"], cam["t"], cam["W"], cam["H"]
        img_path = cam["img_path"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if img.shape[0] != cH or img.shape[1] != cW:
            img = cv2.resize(img, (cW, cH))

        # Optional mask
        mask = None
        if masks_dir is not None:
            stem = Path(cam["name"]).stem
            for ext in (".png", ".jpg", ".jpeg"):
                mp = masks_dir / (stem + ext)
                if mp.exists():
                    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        mask = (cv2.resize(m, (cW, cH)) > 127)
                    break

        # Render depth buffer
        depth_buf = _rasterise_depth(verts, faces, K, R, t, cW, cH)

        # Project all valid texel positions into this camera
        proj_uv, proj_z = _project_points(valid_pts, K, R, t)

        # Camera-space depth of texel
        texel_z = proj_z

        # Pixel coords (float)
        pu = proj_uv[:, 0]
        pv = proj_uv[:, 1]

        # Visibility: texel is visible if:
        #   1. inside image
        #   2. z > 0
        #   3. |proj_z - depth_buf[v, u]| < depth_tol
        in_bounds = ((pu >= 0) & (pu < cW - 1) &
                     (pv >= 0) & (pv < cH - 1) &
                     (texel_z > 0.01))

        pi_u = pu[in_bounds].astype(np.int32).clip(0, cW - 1)
        pi_v = pv[in_bounds].astype(np.int32).clip(0, cH - 1)

        buf_z  = depth_buf[pi_v, pi_u]
        vis    = np.abs(texel_z[in_bounds] - buf_z) < depth_tol

        # Face-normal visibility weight
        cam_pos_world = -(R.T @ t)
        view_dir = cam_pos_world - valid_pts[in_bounds]
        view_dir = view_dir / np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-8)
        dot = (valid_nrm[in_bounds] * view_dir).sum(axis=1)
        weight = dot.clip(0) * vis

        # Mask from segmentation
        if mask is not None:
            in_mask = mask[pi_v, pi_u]
            weight = weight * in_mask

        # Bilinear sample color
        pu_f = pu[in_bounds]
        pv_f = pv[in_bounds]
        pu0 = pu_f.astype(np.int32).clip(0, cW - 2)
        pv0 = pv_f.astype(np.int32).clip(0, cH - 2)
        du = (pu_f - pu0).clip(0, 1)
        dv = (pv_f - pv0).clip(0, 1)
        c00 = img[pv0,     pu0    ]
        c10 = img[pv0 + 1, pu0    ]
        c01 = img[pv0,     pu0 + 1]
        c11 = img[pv0 + 1, pu0 + 1]
        color = ((1 - dv[:, None]) * (1 - du[:, None]) * c00 +
                 (    dv[:, None]) * (1 - du[:, None]) * c10 +
                 (1 - dv[:, None]) * (    du[:, None]) * c01 +
                 (    dv[:, None]) * (    du[:, None]) * c11)

        # Update top-N per texel — vectorised
        valid_in_idx = np.where(in_bounds)[0]
        good = weight > 0
        if good.any():
            vii  = valid_in_idx[good]
            wi   = weight[good]
            ci_  = color[good]
            # For each (texel, cam) pair: replace slot if wi > current min
            # Batch: find minimum slot for each relevant texel
            min_slots = np.argmin(top_weights[vii], axis=1)       # (K,)
            cur_min_w = top_weights[vii, min_slots]               # (K,)
            update    = wi > cur_min_w                             # (K,) bool
            if update.any():
                upd_vii   = vii[update]
                upd_slots = min_slots[update]
                top_weights[upd_vii, upd_slots] = wi[update]
                top_colors [upd_vii, upd_slots] = ci_[update]

        if (ci + 1) % 20 == 0 or ci == len(cam_list) - 1:
            print(f"    Camera {ci+1}/{len(cam_list)}", end="\r", flush=True)

    print()

    # Blend top-N colors per texel
    w_sum = np.maximum(top_weights, 0).sum(axis=1, keepdims=True)   # (N_valid, 1)
    has_w = (w_sum > 0).squeeze()
    blended = np.zeros((N_valid, 3), dtype=np.float32)
    blended[has_w] = (
        (np.maximum(top_weights[has_w], 0)[:, :, None] * top_colors[has_w]).sum(axis=1)
        / w_sum[has_w]
    )

    # Write into atlas
    texture = np.zeros((H, W, 3), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.float32)
    texture[valid_yx[:, 0], valid_yx[:, 1]] = blended
    coverage[valid_yx[:, 0], valid_yx[:, 1]] = has_w.astype(np.float32)

    # Fill uncovered valid texels with nearest neighbour (seam dilation)
    texture_uint8 = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
    texture_uint8 = _dilate_atlas(texture_uint8, valid_mask, n_iters=8)

    return texture_uint8, coverage


# ══════════════════════════════════════════════════════════════════════════════
# Atlas dilation (pull-push / nearest-neighbor fill)
# ══════════════════════════════════════════════════════════════════════════════

def _dilate_atlas(
    tex: np.ndarray,       # (H, W, 3) uint8
    valid: np.ndarray,     # (H, W) bool
    n_iters: int = 8,
) -> np.ndarray:
    """Iteratively dilate valid pixels into their empty neighbours."""
    out = tex.copy()
    mask = valid.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(n_iters):
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel)
        new_px = (dilated_mask > 0) & (~mask)
        if not new_px.any():
            break
        # For new pixels, average valid neighbours
        for c in range(3):
            blurred = cv2.blur(out[:, :, c].astype(np.float32),
                               (3, 3)) / np.maximum(
                               cv2.blur(mask.astype(np.float32), (3, 3)), 1e-6)
            out[:, :, c][new_px] = np.clip(blurred[new_px], 0, 255).astype(np.uint8)
        mask = dilated_mask > 0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# OBJ + MTL export
# ══════════════════════════════════════════════════════════════════════════════

def export_obj_with_uv(
    verts:     np.ndarray,   # (N, 3)
    faces:     np.ndarray,   # (F, 3) original face indices
    uv_verts:  np.ndarray,   # (N_uv, 2)
    uv_faces:  np.ndarray,   # (F, 3) uv indices
    vmapping:  np.ndarray,   # (N_uv,) → original vert
    tex_name:  str,
    out_path:  Path,
):
    mtl_name = out_path.stem + ".mtl"
    lines = [f"mtllib {mtl_name}\n", "usemtl material0\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for uv in uv_verts:
        lines.append(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")  # flip V
    for fi in range(len(faces)):
        f3 = faces[fi]
        uv3 = uv_faces[fi]
        # OBJ: v/vt indices (1-based)
        lines.append(f"f "
                     f"{f3[0]+1}/{uv3[0]+1} "
                     f"{f3[1]+1}/{uv3[1]+1} "
                     f"{f3[2]+1}/{uv3[2]+1}\n")
    out_path.write_text("".join(lines))

    mtl_path = out_path.parent / mtl_name
    mtl_path.write_text(
        f"newmtl material0\n"
        f"Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\n"
        f"map_Kd {tex_name}\n"
    )
    print(f"  Saved OBJ: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",     required=True,
                    help="Directory with betas.npy, scale.npy, trans_global.npy, smpl_canonical.obj")
    ap.add_argument("--frames-dir",  required=True)
    ap.add_argument("--colmap-dir",  required=True,
                    help="COLMAP sparse dir (cameras.txt + images.txt)")
    ap.add_argument("--masks-dir",   default=None)
    ap.add_argument("--tex-size",    type=int, default=2048)
    ap.add_argument("--blend-n",     type=int, default=5,
                    help="Top-N cameras to blend per texel")
    ap.add_argument("--depth-tol",   type=float, default=0.025,
                    help="Z-depth tolerance for visibility check (scene units)")
    ap.add_argument("--n-cams",      type=int, default=0,
                    help="Limit cameras used for baking (0=all, evenly spaced)")
    ap.add_argument("--output",      default=None,
                    help="Output directory (defaults to out-dir/textured)")
    _DEFAULT_SMPL = (
        r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models"
        r"\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    )
    ap.add_argument("--smpl-model",  default=_DEFAULT_SMPL)
    args = ap.parse_args()

    out_dir  = Path(args.out_dir)
    tex_out  = Path(args.output) if args.output else out_dir / "textured"
    tex_out.mkdir(parents=True, exist_ok=True)

    # ── Load SMPL canonical mesh ─────────────────────────────────────
    canon_obj = out_dir / "smpl_canonical.obj"
    if canon_obj.exists():
        print(f"Loading canonical mesh from {canon_obj}")
        verts, faces = _load_obj(canon_obj)
    else:
        # Re-run SMPL forward from betas+scale+trans
        print("smpl_canonical.obj not found — running SMPL forward from saved params...")
        import torch
        from src.recon.smpl_fitter import SMPL
        betas_np = np.load(str(out_dir / "betas.npy"))
        scale_v  = float(np.load(str(out_dir / "scale.npy"))[0])
        trans_v  = np.load(str(out_dir / "trans_global.npy"))
        smpl = SMPL(args.smpl_model, n_betas=len(betas_np))
        dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        smpl = smpl.to(dev)
        with torch.no_grad():
            b = torch.from_numpy(betas_np).float().unsqueeze(0).to(dev)
            p = torch.zeros(1, 72, device=dev)
            t = torch.zeros(1, 3,  device=dev)
            vb, _ = smpl(b, p, t)
        verts = (vb[0].cpu().numpy() * scale_v + trans_v).astype(np.float32)
        faces = smpl.faces

    print(f"Mesh: {len(verts)} verts, {len(faces)} faces")

    # ── Load COLMAP cameras ──────────────────────────────────────────
    cams_txt   = Path(args.colmap_dir) / "cameras.txt"
    images_txt = Path(args.colmap_dir) / "images.txt"
    cameras_def = _read_colmap_cameras_txt(cams_txt)
    images_def  = _read_colmap_images_txt(images_txt)
    cam_list = _build_cameras(cameras_def, images_def, args.frames_dir)
    print(f"Cameras loaded: {len(cam_list)}")

    # ── Bake ─────────────────────────────────────────────────────────
    texture_rgb, coverage = bake_texture(
        verts      = verts,
        faces      = faces,
        cameras    = cam_list,
        tex_size   = args.tex_size,
        blend_n    = args.blend_n,
        depth_tol  = args.depth_tol,
        masks_dir  = Path(args.masks_dir) if args.masks_dir else None,
        n_cam_limit= args.n_cams,
    )

    # ── Save texture ─────────────────────────────────────────────────
    tex_name = "smpl_texture.png"
    tex_path = tex_out / tex_name
    Image.fromarray(texture_rgb).save(str(tex_path))
    print(f"  Texture saved: {tex_path}")

    # ── Re-run UV unwrap to get uv_verts/uv_faces for OBJ export ─────
    uv_verts, uv_faces, vmapping = unwrap_uv(verts, faces)
    obj_path = tex_out / "smpl_textured.obj"
    export_obj_with_uv(verts, faces, uv_verts, uv_faces, vmapping,
                       tex_name, obj_path)

    print(f"\nDone. Output: {tex_out}")
    print(f"  smpl_textured.obj  smpl_textured.mtl  {tex_name}")


if __name__ == "__main__":
    main()
