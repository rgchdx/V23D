from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(r"C:/V23D/V23D")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflows.debug_visualization.export_front_back_part_images import (  # noqa: E402
    FaceDetectorYuNet,
    HybridPartDetector,
    PART_FACE,
    PART_LARM,
    PART_RARM,
    PART_TORSO,
    PART_LLEG,
    PART_RLEG,
    YoloPosePartDetector,
    _smpl_vertex_parts,
)


OBJ_PATH = Path(r"E:/smpl_textured_from_splat.obj")
PLY_FALLBACK_PATH = Path(r"E:/V23D_Data/per_part_splat_refetch1/smpl_textured_from_splat.ply")
SMPL_PKL_PATH = Path(r"E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
YOLO_MODEL_PATH = Path(r"C:/V23D/V23D/yolov8x-pose.pt")
FACE_CACHE_DIR = Path(r"C:/V23D/output/face_cache")

FRONT_REF_IMG = Path(r"E:/zero123_dataset/humans_train/person_017/frame_000/reference.png")
BACK_REF_IMG = Path(r"E:/zero123_dataset/humans_train/person_017/frame_165/reference.png")
EXTERNAL_BACK_CHEST_RED_OVERLAY = Path(r"\\students\student-n-r\rgdix\My Documents\My Pictures\Screenshots\Screenshot 2026-05-26 114642.png")

# Chest detector mode: "yolo" or "mediapipe" (single detector only)
CHEST_DETECTOR_MODE = "yolo"

# Texture toggles
APPLY_FACE_HAIR_TEXTURE = False
APPLY_ARM_TEXTURE = True

OUT_PLY = Path(r"E:/smpl_textured_face_body_dualview.ply")
OUT_PREVIEW = Path(r"E:/smpl_textured_face_body_dualview.png")
OUT_FRONT_FACE_DET = Path(r"E:/reference_front_face_detection.png")
OUT_FRONT_BODY = Path(r"E:/reference_front_body_regions.png")
OUT_BACK_BODY = Path(r"E:/reference_back_body_regions.png")
OUT_FRONT_ARM_DET = Path(r"E:/reference_front_arms_detected.png")
OUT_BACK_ARM_DET = Path(r"E:/reference_back_arms_detected.png")
OUT_FRONT_ANN = Path(r"E:/smpl_mesh_annotations_front.png")
OUT_BACK_ANN = Path(r"E:/smpl_mesh_annotations_back.png")
OUT_FRONT_ARM_ANN = Path(r"E:/smpl_mesh_arms_front.png")
OUT_BACK_ARM_ANN = Path(r"E:/smpl_mesh_arms_back.png")
OUT_FRONT_ARM_SPLAT = Path(r"E:/smpl_arms_splat_overlay_front.png")
OUT_BACK_ARM_SPLAT = Path(r"E:/smpl_arms_splat_overlay_back.png")
OUT_FRONT_CHEST_SPLAT = Path(r"E:/smpl_chest_splat_overlay_front.png")
OUT_BACK_CHEST_SPLAT = Path(r"E:/smpl_chest_splat_overlay_back.png")
OUT_BACK_CHEST_MASK_USED = Path(r"E:/reference_back_chest_mask_used.png")

REGION_BGR = {
    "face": (0, 220, 255),
    "chest": (0, 210, 0),
    "left_arm": (255, 200, 0),
    "right_arm": (0, 140, 255),
    "hair": (180, 0, 180),
}


def _load_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(OBJ_PATH), enable_post_processing=True)
    if not mesh.has_triangles():
        raise RuntimeError(f"Could not load mesh: {OBJ_PATH}")
    if not mesh.has_vertex_colors() and PLY_FALLBACK_PATH.exists():
        base = o3d.io.read_triangle_mesh(str(PLY_FALLBACK_PATH), enable_post_processing=True)
        if base.has_vertex_colors() and len(base.vertices) == len(mesh.vertices):
            mesh.vertex_colors = base.vertex_colors
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color((0.62, 0.62, 0.62))
    mesh.compute_vertex_normals()
    return mesh


def _read_rgb_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = img[:, :, :3]
    if alpha is not None:
        mask = (alpha > 10).astype(np.uint8) * 255
    else:
        corners = np.concatenate(
            [
                img[:20, :20].reshape(-1, 3),
                img[:20, -20:].reshape(-1, 3),
                img[-20:, :20].reshape(-1, 3),
                img[-20:, -20:].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(corners, axis=0)
        diff = np.abs(img.astype(np.float32) - bg[None, None, :]).sum(axis=2)
        mask = (diff > 28.0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb, mask


def _parts_for_mesh(verts: np.ndarray) -> np.ndarray:
    base_parts = _smpl_vertex_parts(SMPL_PKL_PATH)
    if len(base_parts) == len(verts):
        return base_parts
    base_mesh = o3d.io.read_triangle_mesh(str(PLY_FALLBACK_PATH), enable_post_processing=True)
    base_verts = np.asarray(base_mesh.vertices)
    if len(base_verts) != len(base_parts):
        raise RuntimeError("Fallback mesh/part mapping mismatch")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(base_verts)
        _dist, nn = tree.query(verts, k=1)
    except Exception:
        nn = []
        for v in verts:
            d2 = np.sum((base_verts - v[None, :]) ** 2, axis=1)
            nn.append(int(np.argmin(d2)))
        nn = np.asarray(nn, dtype=np.int32)
    return base_parts[np.asarray(nn, dtype=np.int32)]


def _project(verts: np.ndarray, width: int, height: int, mirrored: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = verts.copy()
    if mirrored:
        v[:, 0] *= -1.0
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    sx = (width - 40) / max(float(x.max() - x.min()), 1e-6)
    sy = (height - 40) / max(float(y.max() - y.min()), 1e-6)
    s = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + width * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + height * 0.5).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height) & np.isfinite(z)
    return px, py, inside


def _direct_sample(img_rgb: np.ndarray, mask: np.ndarray, qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    qx = np.clip(qxy[:, 0].astype(np.int32), 0, w - 1)
    qy = np.clip(qxy[:, 1].astype(np.int32), 0, h - 1)
    direct = mask[qy, qx] > 0
    colors = np.zeros((len(qxy), 3), dtype=np.float32)
    if np.any(direct):
        colors[direct] = img_rgb[qy[direct], qx[direct]].astype(np.float32) / 255.0
    lum_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lum = np.zeros((len(qxy),), dtype=np.float32)
    lum[direct] = lum_img[qy[direct], qx[direct]]
    return colors, direct, lum


def _nearest_sample(img_rgb: np.ndarray, mask: np.ndarray, qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.zeros((len(qxy), 3), dtype=np.float32), np.zeros((len(qxy),), dtype=bool), np.zeros((len(qxy),), dtype=np.float32)
    xy = coords[:, [1, 0]].astype(np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        dist, idx = tree.query(qxy.astype(np.float32), k=1)
    except Exception:
        dist = []
        idx = []
        for q in qxy.astype(np.float32):
            d2 = np.sum((xy - q[None, :]) ** 2, axis=1)
            ii = int(np.argmin(d2))
            idx.append(ii)
            dist.append(float(np.sqrt(d2[ii])))
        idx = np.asarray(idx, dtype=np.int32)
        dist = np.asarray(dist, dtype=np.float32)
    nearest = coords[np.asarray(idx).reshape(-1)]
    colors = img_rgb[nearest[:, 0], nearest[:, 1]].astype(np.float32) / 255.0
    lum_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lum = lum_img[nearest[:, 0], nearest[:, 1]]
    valid = np.asarray(dist) <= 42.0
    return colors, valid, lum


def _shade(colors: np.ndarray, normals: np.ndarray, luminance: np.ndarray, strength: float = 0.25, lum_boost: float = 0.50) -> np.ndarray:
    light_dir = np.array([0.18, -0.26, 0.94], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    ndotl = np.clip(normals @ light_dir, 0.0, 1.0)
    base = colors * ((1.0 - strength) + strength * ndotl[:, None] + 0.2)
    valid_l = luminance > 0
    if np.any(valid_l):
        mu = float(np.mean(luminance[valid_l]))
        sigma = max(float(np.std(luminance[valid_l])), 0.06)
        rel = np.clip((luminance - mu) / (2.2 * sigma), -1.0, 1.0)
        base = np.clip(base * (1.0 + lum_boost * rel[:, None]), 0.0, 1.0)
    return np.clip(base, 0.0, 1.0)


def _kmeans_spread_fill(
    colors: np.ndarray,
    base_colors: np.ndarray,
    verts: np.ndarray,
    fill_region_mask: np.ndarray,
    changed_thresh: float = 0.045,
    k: int = 6,
) -> np.ndarray:
    """Fill untextured (gap/white-ish) vertices using k-nearest neighbors.

    For each untextured target vertex, finds k nearest textured vertices in 3D and
    assigns a distance-weighted average of their colors.
    """
    out = colors.copy()
    changed = fill_region_mask & (np.linalg.norm(colors - base_colors, axis=1) > changed_thresh)
    targets = fill_region_mask & (~changed)
    if not np.any(changed) or not np.any(targets):
        return out

    src_idx = np.where(changed)[0]
    dst_idx = np.where(targets)[0]
    src_xyz = verts[src_idx].astype(np.float32)
    dst_xyz = verts[dst_idx].astype(np.float32)
    src_col = np.clip(colors[src_idx].astype(np.float32), 0.0, 1.0)

    kn = int(np.clip(k, 1, max(1, min(32, len(src_col)))))

    # k-NN lookup in 3D vertex space.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(src_xyz)
        d, nn = tree.query(dst_xyz, k=kn)
        d = np.asarray(d, dtype=np.float32)
        nn = np.asarray(nn, dtype=np.int32)
    except Exception:
        d = []
        nn = []
        for p in dst_xyz:
            d2 = np.sum((src_xyz - p[None, :]) ** 2, axis=1)
            order = np.argsort(d2)[:kn]
            nn.append(order)
            d.append(np.sqrt(d2[order]))
        d = np.asarray(d, dtype=np.float32)
        nn = np.asarray(nn, dtype=np.int32)

    if kn == 1:
        # Shape normalization for single-neighbor case.
        if nn.ndim == 1:
            nn = nn[:, None]
        if d.ndim == 1:
            d = d[:, None]

    neigh_col = src_col[nn]  # (N,kn,3)
    w = 1.0 / np.maximum(d, 1e-6)
    w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-6)
    filled = np.sum(neigh_col * w[:, :, None], axis=1)
    filled = np.clip(filled, 0.0, 1.0)
    out[dst_idx] = filled
    return out


def _iterative_knn_fill_white(
    verts: np.ndarray,
    colors: np.ndarray,
    passes: int = 8,
    k: int = 8,
    white_mean_thr: float = 0.90,
    white_chroma_thr: float = 0.08,
) -> np.ndarray:
    """Iteratively fill white/untextured mesh vertices using k-nearest colored neighbors.

    This runs pass-by-pass over ALL vertices so small white triangulation islands are
    progressively absorbed by nearby textured colors.
    """
    out = np.clip(colors.copy().astype(np.float32), 0.0, 1.0)
    vv = verts.astype(np.float32)

    for _ in range(int(max(1, passes))):
        cmean = np.mean(out, axis=1)
        cch = np.max(out, axis=1) - np.min(out, axis=1)
        white = (cmean >= float(white_mean_thr)) & (cch <= float(white_chroma_thr))
        if not np.any(white):
            break

        seed = ~white
        if not np.any(seed):
            break

        src_idx = np.where(seed)[0]
        dst_idx = np.where(white)[0]
        src_xyz = vv[src_idx]
        dst_xyz = vv[dst_idx]
        src_col = out[src_idx]

        kn = int(np.clip(k, 1, max(1, min(32, len(src_idx)))))

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(src_xyz)
            d, nn = tree.query(dst_xyz, k=kn)
            d = np.asarray(d, dtype=np.float32)
            nn = np.asarray(nn, dtype=np.int32)
        except Exception:
            d = []
            nn = []
            for p in dst_xyz:
                d2 = np.sum((src_xyz - p[None, :]) ** 2, axis=1)
                order = np.argsort(d2)[:kn]
                nn.append(order)
                d.append(np.sqrt(d2[order]))
            d = np.asarray(d, dtype=np.float32)
            nn = np.asarray(nn, dtype=np.int32)

        if kn == 1:
            if nn.ndim == 1:
                nn = nn[:, None]
            if d.ndim == 1:
                d = d[:, None]

        neigh_col = src_col[nn]
        w = 1.0 / np.maximum(d, 1e-6)
        w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-6)
        fill = np.sum(neigh_col * w[:, :, None], axis=1)
        out[dst_idx] = np.clip(fill, 0.0, 1.0)

    return out


def _arm_stretch_sample(
    arm_verts: np.ndarray,
    arm_normals: np.ndarray,
    img_rgb: np.ndarray,
    arm_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stretch-map detected arm pixels onto arm vertices.

    Returns:
      shaded_colors: (N,3) in [0,1]
      src_xy: (N,2) sampled source pixel coordinates (x,y)
      valid: (N,) bool
    """
    n = len(arm_verts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    pix_yx = np.argwhere(arm_mask > 0)
    if len(pix_yx) == 0:
        return np.zeros((n, 3), dtype=np.float32), np.zeros((n, 2), dtype=np.int32), np.zeros((n,), dtype=bool)

    # Remove near-white background leakage from detector masks.
    pix_yx_orig = pix_yx.copy()
    pix_rgb = img_rgb[pix_yx[:, 0], pix_yx[:, 1]].astype(np.float32)
    keep = np.mean(pix_rgb, axis=1) < 245.0
    if np.sum(keep) > 128:
        pix_yx = pix_yx[keep]
    if len(pix_yx) == 0:
        pix_yx = pix_yx_orig

    # 2D arm orientation in image (detected arm region).
    xy = pix_yx[:, [1, 0]].astype(np.float32)
    xy_c = xy.mean(axis=0)
    _, _, vt2 = np.linalg.svd(xy - xy_c[None, :], full_matrices=False)
    img_u_axis = vt2[0]
    img_v_axis = vt2[1]
    uv_pix = np.stack([
        (xy - xy_c[None, :]) @ img_u_axis,
        (xy - xy_c[None, :]) @ img_v_axis,
    ], axis=1)
    umin, umax = float(uv_pix[:, 0].min()), float(uv_pix[:, 0].max())
    vmin, vmax = float(uv_pix[:, 1].min()), float(uv_pix[:, 1].max())
    uv_pix_n = np.stack([
        (uv_pix[:, 0] - umin) / max(umax - umin, 1e-6),
        (uv_pix[:, 1] - vmin) / max(vmax - vmin, 1e-6),
    ], axis=1)

    # 3D arm orientation on mesh (length + radial direction).
    c3 = arm_verts.mean(axis=0)
    _, _, vt3 = np.linalg.svd(arm_verts - c3[None, :], full_matrices=False)
    mesh_u_axis = vt3[0]
    if mesh_u_axis[1] < 0.0:
        mesh_u_axis = -mesh_u_axis
    mesh_v_axis = vt3[1]
    uv_mesh = np.stack([
        (arm_verts - c3[None, :]) @ mesh_u_axis,
        (arm_verts - c3[None, :]) @ mesh_v_axis,
    ], axis=1)
    mu_min, mu_max = float(uv_mesh[:, 0].min()), float(uv_mesh[:, 0].max())
    mv_min, mv_max = float(uv_mesh[:, 1].min()), float(uv_mesh[:, 1].max())
    uv_mesh_n = np.stack([
        (uv_mesh[:, 0] - mu_min) / max(mu_max - mu_min, 1e-6),
        (uv_mesh[:, 1] - mv_min) / max(mv_max - mv_min, 1e-6),
    ], axis=1).astype(np.float32)

    # Nearest lookup in normalized arm UV domain -> stretched mapping.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(uv_pix_n.astype(np.float32))
        _dist, idx = tree.query(uv_mesh_n, k=1)
        idx = np.asarray(idx, dtype=np.int32)
    except Exception:
        idx = []
        for q in uv_mesh_n:
            d2 = np.sum((uv_pix_n - q[None, :]) ** 2, axis=1)
            idx.append(int(np.argmin(d2)))
        idx = np.asarray(idx, dtype=np.int32)

    src = pix_yx[idx]  # (y,x)
    colors = img_rgb[src[:, 0], src[:, 1]].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lum = gray[src[:, 0], src[:, 1]]
    shaded = _shade(colors, arm_normals, lum, strength=0.34, lum_boost=0.96)
    valid = np.ones((n,), dtype=bool)
    src_xy = src[:, [1, 0]].astype(np.int32)
    return shaded, src_xy, valid


def _chest_stretch_sample(
    chest_verts: np.ndarray,
    chest_normals: np.ndarray,
    img_rgb: np.ndarray,
    chest_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stretch-map detected chest pixels onto chest vertices.
    
    Key difference from arms: chest orientation is primarily vertical (Y-axis: top-to-bottom),
    not horizontal. We enforce Y as the primary axis.
    
    Returns:
      shaded_colors: (N,3) in [0,1]
      src_xy: (N,2) sampled source pixel coordinates (x,y)
      valid: (N,) bool
    """
    n = len(chest_verts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    pix_yx = np.argwhere(chest_mask > 0)
    if len(pix_yx) == 0:
        return np.zeros((n, 3), dtype=np.float32), np.zeros((n, 2), dtype=np.int32), np.zeros((n,), dtype=bool)

    # Remove near-white background leakage from detector masks.
    pix_yx_orig = pix_yx.copy()
    pix_rgb = img_rgb[pix_yx[:, 0], pix_yx[:, 1]].astype(np.float32)
    keep = np.mean(pix_rgb, axis=1) < 245.0
    if np.sum(keep) > 128:
        pix_yx = pix_yx[keep]
    if len(pix_yx) == 0:
        pix_yx = pix_yx_orig

    # 2D chest orientation in image: Y-axis is vertical (top-to-bottom), X-axis is horizontal (left-to-right)
    xy = pix_yx[:, [1, 0]].astype(np.float32)  # Convert to (x, y)
    xy_c = xy.mean(axis=0)
    
    # Primary axis: vertical (Y in image = top-to-bottom)
    # Secondary axis: horizontal (X in image = left-to-right)
    img_v_axis = np.array([0.0, 1.0], dtype=np.float32)  # Vertical in image space (Y increases downward)
    img_u_axis = np.array([1.0, 0.0], dtype=np.float32)  # Horizontal in image space (X increases rightward)
    
    uv_pix = np.stack([
        (xy - xy_c[None, :]) @ img_u_axis,  # Horizontal component
        (xy - xy_c[None, :]) @ img_v_axis,  # Vertical component
    ], axis=1)
    umin, umax = float(uv_pix[:, 0].min()), float(uv_pix[:, 0].max())
    vmin, vmax = float(uv_pix[:, 1].min()), float(uv_pix[:, 1].max())
    uv_pix_n = np.stack([
        (uv_pix[:, 0] - umin) / max(umax - umin, 1e-6),
        (uv_pix[:, 1] - vmin) / max(vmax - vmin, 1e-6),
    ], axis=1)

    # 3D chest orientation on mesh: Y-axis (top-to-bottom), X-axis (left-to-right)
    c3 = chest_verts.mean(axis=0)
    mesh_v_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Vertical in 3D space (Y increases upward in SMPL)
    mesh_u_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Horizontal in 3D space (X increases rightward)
    
    uv_mesh = np.stack([
        (chest_verts - c3[None, :]) @ mesh_u_axis,  # Horizontal component (X)
        (chest_verts - c3[None, :]) @ mesh_v_axis,  # Vertical component (Y)
    ], axis=1)
    mu_min, mu_max = float(uv_mesh[:, 0].min()), float(uv_mesh[:, 0].max())
    mv_min, mv_max = float(uv_mesh[:, 1].min()), float(uv_mesh[:, 1].max())
    uv_mesh_n = np.stack([
        (uv_mesh[:, 0] - mu_min) / max(mu_max - mu_min, 1e-6),
        (uv_mesh[:, 1] - mv_min) / max(mv_max - mv_min, 1e-6),
    ], axis=1).astype(np.float32)

    # Nearest lookup in normalized chest UV domain -> stretched mapping.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(uv_pix_n.astype(np.float32))
        _dist, idx = tree.query(uv_mesh_n, k=1)
        idx = np.asarray(idx, dtype=np.int32)
    except Exception:
        idx = []
        for q in uv_mesh_n:
            d2 = np.sum((uv_pix_n - q[None, :]) ** 2, axis=1)
            idx.append(int(np.argmin(d2)))
        idx = np.asarray(idx, dtype=np.int32)

    src = pix_yx[idx]  # (y,x)
    colors = img_rgb[src[:, 0], src[:, 1]].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lum = gray[src[:, 0], src[:, 1]]
    shaded = _shade(colors, chest_normals, lum, strength=0.34, lum_boost=0.96)
    valid = np.ones((n,), dtype=bool)
    src_xy = src[:, [1, 0]].astype(np.int32)
    return shaded, src_xy, valid


def _draw_arm_axis(canvas: np.ndarray, px: np.ndarray, py: np.ndarray, verts: np.ndarray, idx: np.ndarray, color: tuple[int, int, int]) -> None:
    if len(idx) < 8:
        return
    av = verts[idx]
    c3 = av.mean(axis=0)
    _, _, vt = np.linalg.svd(av - c3[None, :], full_matrices=False)
    axis = vt[0]
    t = (av - c3[None, :]) @ axis
    i0 = idx[int(np.argmin(t))]
    i1 = idx[int(np.argmax(t))]
    p0 = (int(px[i0]), int(py[i0]))
    p1 = (int(px[i1]), int(py[i1]))
    if 0 <= p0[0] < canvas.shape[1] and 0 <= p0[1] < canvas.shape[0] and 0 <= p1[0] < canvas.shape[1] and 0 <= p1[1] < canvas.shape[0]:
        cv2.arrowedLine(canvas, p0, p1, color, 3, cv2.LINE_AA, tipLength=0.06)


def _arm_detection_overlay(ref_bgr: np.ndarray, left_arm_mask: np.ndarray, right_arm_mask: np.ndarray, title: str = "") -> np.ndarray:
    """Visualize detected left and right arm regions on the reference image."""
    img = ref_bgr.copy()
    overlay = img.copy()
    overlay[left_arm_mask > 0] = (255, 200, 0)   # Blue for left arm
    overlay[right_arm_mask > 0] = (0, 140, 255)  # Orange for right arm
    blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0.0)
    if title:
        cv2.putText(blended, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return blended


def _mesh_arm_annotations(verts: np.ndarray, px: np.ndarray, py: np.ndarray, larm_idx: np.ndarray, rarm_idx: np.ndarray, title: str) -> np.ndarray:
    """Visualize SMPL arm vertices projected onto a canvas."""
    canvas = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    
    # Scale projections to fill the 1024x1024 canvas
    px_min, px_max = float(px.min()), float(px.max())
    py_min, py_max = float(py.min()), float(py.max())
    px_range = max(px_max - px_min, 1.0)
    py_range = max(py_max - py_min, 1.0)
    
    px_scaled = ((px.astype(np.float32) - px_min) / px_range * 900 + 62).astype(np.int32)
    py_scaled = ((py.astype(np.float32) - py_min) / py_range * 900 + 62).astype(np.int32)
    
    for idx, col in [(larm_idx, (255, 200, 0)), (rarm_idx, (0, 140, 255))]:
        step = max(1, len(idx) // 3000)
        for i in idx[::step]:
            if 0 <= px_scaled[i] < 1024 and 0 <= py_scaled[i] < 1024:
                canvas[py_scaled[i], px_scaled[i]] = col
    cv2.putText(canvas, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def _chest_with_shoulders_mask(
    torso_mask: np.ndarray,
    left_arm_mask: np.ndarray,
    right_arm_mask: np.ndarray,
    person_mask: np.ndarray,
    face_bottom_y: int | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
    face_kps5: np.ndarray | None = None,
) -> np.ndarray:
    """Detect chest from torso by finding shoulder contact and torso separation line.

    No geometric dorito wedges; only shoulder-contact + torso separation.
    """
    chest = cv2.bitwise_and(torso_mask, person_mask)
    ys, xs = np.where(chest > 0)
    if len(xs) == 0:
        return chest

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = max(1, y1 - y0 + 1)
    w = max(1, x1 - x0 + 1)

    # Strict upper torso band so legs/pelvis are excluded.
    chest_band = np.zeros_like(chest)
    band_top = int(y0 + 0.06 * h)
    band_bot = int(y0 + 0.60 * h)
    chest_band[max(0, band_top):min(chest.shape[0], band_bot + 1), max(0, x0):min(chest.shape[1], x1 + 1)] = 255
    chest = cv2.bitwise_and(chest, chest_band)

    # Keep chest below face for front view.
    if face_bottom_y is not None:
        cut = int(min(chest.shape[0] - 1, face_bottom_y + 4))
        chest[:cut, :] = 0

    # Explicitly carve out face/nose area if face detection is available.
    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
        fw = max(1, fx2 - fx1 + 1)
        fh = max(1, fy2 - fy1 + 1)
        excl = np.zeros_like(chest)
        ex1 = max(0, fx1 - int(0.16 * fw))
        ex2 = min(chest.shape[1] - 1, fx2 + int(0.16 * fw))
        ey1 = max(0, fy1 - int(0.08 * fh))
        # extend a bit below chin so nose/mouth/upper neck cannot leak into chest
        ey2 = min(chest.shape[0] - 1, fy2 + int(0.32 * fh))
        cv2.rectangle(excl, (ex1, ey1), (ex2, ey2), 255, -1, cv2.LINE_AA)

        # Extra nose-centered oval suppression.
        if face_kps5 is not None and len(face_kps5) >= 3:
            nose = np.asarray(face_kps5[2]).astype(np.int32)
            nx, ny = int(nose[0]), int(nose[1])
            axes = (max(12, int(0.20 * fw)), max(10, int(0.18 * fh)))
            cv2.ellipse(excl, (nx, ny + int(0.06 * fh)), axes, 0, 0, 360, 255, -1, cv2.LINE_AA)

        chest = cv2.bitwise_and(chest, cv2.bitwise_not(excl))

    # Shoulder detection from arm↔torso contact in upper body.
    torso_d = cv2.dilate(chest, np.ones((17, 17), np.uint8), iterations=1)
    l_arm = cv2.bitwise_and(left_arm_mask, person_mask)
    r_arm = cv2.bitwise_and(right_arm_mask, person_mask)
    l_contact = cv2.bitwise_and(cv2.dilate(l_arm, np.ones((13, 13), np.uint8), iterations=1), torso_d)
    r_contact = cv2.bitwise_and(cv2.dilate(r_arm, np.ones((13, 13), np.uint8), iterations=1), torso_d)
    shoulder_contact = cv2.bitwise_or(l_contact, r_contact)

    ys_sh, xs_sh = np.where(shoulder_contact > 0)
    if len(xs_sh) > 10:
        sep_y = int(np.percentile(ys_sh, 60))
    else:
        sep_y = int(y0 + 0.24 * h)

    # Torso separation: keep below shoulder seam but still upper torso only.
    chest[:max(0, sep_y), :] = 0

    # Include shoulder caps directly from contact, no geometric wedges.
    shoulder_caps = cv2.dilate(shoulder_contact, np.ones((15, 15), np.uint8), iterations=1)
    shoulder_caps = cv2.bitwise_and(shoulder_caps, chest_band)
    chest = cv2.bitwise_or(chest, shoulder_caps)

    # Keep largest connected component only to reject stray fragments.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((chest > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_lab = 1 + int(np.argmax(areas))
        chest = np.where(labels == keep_lab, 255, 0).astype(np.uint8)

    chest = cv2.morphologyEx(chest, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    chest = cv2.morphologyEx(chest, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    chest = cv2.bitwise_and(chest, person_mask)
    return chest


def _segment_chest_geometric(
    person_mask: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Geometric chest segmentor from person silhouette (upper-body trapezoid)."""
    out = np.zeros_like(person_mask)
    ys, xs = np.where(person_mask > 0)
    if len(xs) == 0:
        return out
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0 + 1)
    w = max(1, x1 - x0 + 1)

    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = face_bbox
        top_y = int(min(y1, fy2 + 0.06 * h))
    else:
        top_y = int(y0 + 0.18 * h)
    bot_y = int(y0 + 0.62 * h)

    cx = int(0.5 * (x0 + x1))
    top_w = int(0.36 * w)
    bot_w = int(0.56 * w)
    poly = np.array([
        [cx - top_w // 2, top_y],
        [cx + top_w // 2, top_y],
        [cx + bot_w // 2, bot_y],
        [cx - bot_w // 2, bot_y],
    ], dtype=np.int32)
    cv2.fillConvexPoly(out, poly, 255)
    out = cv2.bitwise_and(out, person_mask)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    return out


def _segment_chest_grabcut(
    img_rgb: np.ndarray,
    person_mask: np.ndarray,
    seed_mask: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """GrabCut chest segmentor seeded by torso/chest prior."""
    h, w = person_mask.shape[:2]
    out = np.zeros_like(person_mask)
    ys, xs = np.where(person_mask > 0)
    if len(xs) < 50:
        return out

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    hh = max(1, y1 - y0 + 1)

    if face_bbox is not None:
        _, _, _, fy2 = face_bbox
        ry0 = max(0, int(fy2 + 0.04 * hh))
    else:
        ry0 = max(0, int(y0 + 0.14 * hh))
    ry1 = min(h - 1, int(y0 + 0.68 * hh))
    rx0, rx1 = x0, x1
    if ry1 <= ry0 or rx1 <= rx0:
        return out

    rect = (rx0, ry0, max(1, rx1 - rx0), max(1, ry1 - ry0))
    gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc[person_mask > 0] = cv2.GC_PR_BGD
    gc[seed_mask > 0] = cv2.GC_PR_FGD

    # Strong foreground in central seed core.
    core = cv2.erode((seed_mask > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8), iterations=1)
    gc[core > 0] = cv2.GC_FGD

    # Face region forced background.
    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
        gc[max(0, fy1):min(h, fy2 + 1), max(0, fx1):min(w, fx2 + 1)] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), gc, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
        out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        out = cv2.bitwise_and(out, person_mask)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    except Exception:
        return np.zeros_like(person_mask)
    return out


def _mediapipe_chest_mask(
    img_rgb: np.ndarray,
    person_mask: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """MediaPipe-pose chest mask from shoulder/hip landmarks."""
    out = np.zeros_like(person_mask)
    PoseCls = None
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            PoseCls = mp.solutions.pose.Pose
    except Exception:
        pass

    if PoseCls is None:
        try:
            from mediapipe.python.solutions.pose import Pose as PoseCls  # type: ignore
        except Exception:
            return out

    h, w = person_mask.shape[:2]
    with PoseCls(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.35,
    ) as pose:
        res = pose.process(img_rgb)

    if res.pose_landmarks is None:
        return out

    lms = res.pose_landmarks.landmark

    def lm_xy(i: int) -> tuple[int, int] | None:
        if i >= len(lms):
            return None
        x = float(lms[i].x) * w
        y = float(lms[i].y) * h
        v = float(getattr(lms[i], "visibility", 1.0))
        if not np.isfinite(x) or not np.isfinite(y) or v < 0.25:
            return None
        return int(round(x)), int(round(y))

    lsh = lm_xy(11)
    rsh = lm_xy(12)
    lhp = lm_xy(23)
    rhp = lm_xy(24)
    if lsh is None or rsh is None or lhp is None or rhp is None:
        return out

    sh_w = max(8.0, float(np.hypot(lsh[0] - rsh[0], lsh[1] - rsh[1])))

    def lerp(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
        return (int(round((1.0 - t) * a[0] + t * b[0])), int(round((1.0 - t) * a[1] + t * b[1])))

    top_l = (int(round(lsh[0] - 0.08 * sh_w)), int(round(lsh[1] - 0.05 * sh_w)))
    top_r = (int(round(rsh[0] + 0.08 * sh_w)), int(round(rsh[1] - 0.05 * sh_w)))
    bot_l = lerp(lsh, lhp, 0.58)
    bot_r = lerp(rsh, rhp, 0.58)
    poly = np.array([top_l, top_r, bot_r, bot_l], dtype=np.int32)
    cv2.fillConvexPoly(out, poly, 255)

    # Remove detected face area.
    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
        cv2.rectangle(out, (max(0, fx1 - 6), max(0, fy1 - 6)), (min(w - 1, fx2 + 6), min(h - 1, fy2 + 10)), 0, -1)

    out = cv2.bitwise_and(out, person_mask)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return out


def _chest_from_inverse(
    person_mask: np.ndarray,
    face_mask: np.ndarray,
    head_mask: np.ndarray,
    left_arm_mask: np.ndarray,
    right_arm_mask: np.ndarray,
    left_leg_mask: np.ndarray,
    right_leg_mask: np.ndarray,
) -> np.ndarray:
    """
    Detect chest as the inverse: full body minus head, arms, and legs.
    Then constrain to upper-torso band to avoid belly/pelvis/between-legs.
    Vertical orientation: top-to-bottom of chest region.
    """
    # Start with full person mask
    chest = person_mask.copy().astype(np.uint8)
    
    # Subtract head (face + top-of-head region)
    chest[face_mask > 0] = 0
    chest[head_mask > 0] = 0
    
    # Subtract arms (both left and right) - more aggressive to avoid shoulder leakage
    chest[left_arm_mask > 0] = 0
    chest[right_arm_mask > 0] = 0
    
    # Subtract legs (both left and right) - very aggressive to prevent belly/pelvis inclusion
    # Dilate legs to catch more of the lower torso/pelvic area
    dilated_left_leg = cv2.dilate(left_leg_mask, np.ones((21, 21), np.uint8), iterations=2)
    dilated_right_leg = cv2.dilate(right_leg_mask, np.ones((21, 21), np.uint8), iterations=2)
    chest[dilated_left_leg > 0] = 0
    chest[dilated_right_leg > 0] = 0
    
    # Constrain to upper-torso band: find the bounding box of what remains
    ys, xs = np.where(chest > 0)
    if len(xs) == 0:
        return np.zeros_like(chest)
    
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    height = y_max - y_min + 1
    
    # Upper chest band: roughly top 50% of remaining torso (avoiding belly)
    upper_limit = int(y_min + 0.50 * height)
    band_mask = np.zeros_like(chest)
    band_mask[y_min:upper_limit, x_min:x_max] = 255
    chest = cv2.bitwise_and(chest, band_mask)
    
    # Clean up: morphological operations to fill small holes and smooth edges
    chest = cv2.morphologyEx(chest, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    chest = cv2.morphologyEx(chest, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # Keep largest connected component (main chest region)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats((chest > 0).astype(np.uint8), connectivity=8)
    if nlab > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        chest = np.where(labels == keep, 255, 0).astype(np.uint8)
    else:
        chest = (chest > 0).astype(np.uint8) * 255
    
    return chest


def _external_red_overlay_to_chest_mask(
    overlay_path: Path,
    person_mask: np.ndarray,
) -> np.ndarray:
    """Build a chest mask from an external screenshot where chest is marked in red.

    The screenshot is treated as a shape prior: red region is extracted, normalized,
    then fit into the detected upper-body band of the current person mask.
    """
    out = np.zeros_like(person_mask)
    if overlay_path is None or (not overlay_path.exists()):
        return out

    ov = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if ov is None:
        return out

    hsv = cv2.cvtColor(ov, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    r2 = cv2.inRange(hsv, (168, 70, 70), (180, 255, 255))
    red = cv2.bitwise_or(r1, r2)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats((red > 0).astype(np.uint8), connectivity=8)
    if nlab > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        red = np.where(labels == keep, 255, 0).astype(np.uint8)

    ys_r, xs_r = np.where(red > 0)
    ys_p, xs_p = np.where(person_mask > 0)
    if len(xs_r) == 0 or len(xs_p) == 0:
        return out

    rx0, rx1 = int(xs_r.min()), int(xs_r.max())
    ry0, ry1 = int(ys_r.min()), int(ys_r.max())
    red_crop = red[ry0:ry1 + 1, rx0:rx1 + 1]

    px0, px1 = int(xs_p.min()), int(xs_p.max())
    py0, py1 = int(ys_p.min()), int(ys_p.max())
    pw = max(1, px1 - px0 + 1)
    ph = max(1, py1 - py0 + 1)

    # Fit red chest prior to upper torso area of current frame.
    tw = max(32, int(0.58 * pw))
    th = max(32, int(0.42 * ph))
    tx0 = int((px0 + px1) * 0.5 - tw * 0.5)
    ty0 = int(py0 + 0.14 * ph)
    tx1 = tx0 + tw - 1
    ty1 = ty0 + th - 1

    tx0_c = max(0, tx0)
    ty0_c = max(0, ty0)
    tx1_c = min(person_mask.shape[1] - 1, tx1)
    ty1_c = min(person_mask.shape[0] - 1, ty1)
    if tx1_c <= tx0_c or ty1_c <= ty0_c:
        return out

    fit_w = tx1_c - tx0_c + 1
    fit_h = ty1_c - ty0_c + 1
    red_fit = cv2.resize(red_crop, (fit_w, fit_h), interpolation=cv2.INTER_NEAREST)

    out[ty0_c:ty1_c + 1, tx0_c:tx1_c + 1] = np.where(red_fit > 0, 255, 0).astype(np.uint8)
    out = cv2.bitwise_and(out, person_mask)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    nlab2, labels2, stats2, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    if nlab2 > 1:
        keep2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        out = np.where(labels2 == keep2, 255, 0).astype(np.uint8)

    return out



def _multi_segment_chest_mask(
    img_rgb: np.ndarray,
    torso_mask: np.ndarray,
    person_mask: np.ndarray,
    torso_joint_mask: np.ndarray | None = None,
    left_arm_joint_mask: np.ndarray | None = None,
    right_arm_joint_mask: np.ndarray | None = None,
    face_bottom_y: int | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
    face_kps5: np.ndarray | None = None,
    detector_mode: str = "yolo",
) -> np.ndarray:
    """Chest detector using one method only: YOLO or MediaPipe."""
    torso_src = torso_joint_mask if torso_joint_mask is not None else torso_mask
    left_src = left_arm_joint_mask if left_arm_joint_mask is not None else np.zeros_like(torso_src)
    right_src = right_arm_joint_mask if right_arm_joint_mask is not None else np.zeros_like(torso_src)

    mode = str(detector_mode).strip().lower()
    if mode == "mediapipe":
        chest = _mediapipe_chest_mask(img_rgb, person_mask, face_bbox=face_bbox)
    else:
        chest = _chest_with_shoulders_mask(
            torso_src,
            left_src,
            right_src,
            person_mask,
            face_bottom_y=face_bottom_y,
            face_bbox=face_bbox,
            face_kps5=face_kps5,
        )

    # Strict upper-body clamp to avoid lower-body leakage.
    ys, xs = np.where(person_mask > 0)
    if len(xs) > 0:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h = max(1, y1 - y0 + 1)
        band = np.zeros_like(chest)
        top = int(y0 + 0.08 * h)
        bot = int(y0 + 0.58 * h)
        band[max(0, top):min(chest.shape[0], bot + 1), max(0, x0):min(chest.shape[1], x1 + 1)] = 255
        chest = cv2.bitwise_and(chest, band)

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats((chest > 0).astype(np.uint8), connectivity=8)
    if nlab > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        chest = np.where(labels == keep, 255, 0).astype(np.uint8)

    chest = cv2.bitwise_and(chest, person_mask)
    chest = cv2.morphologyEx(chest, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    chest = cv2.morphologyEx(chest, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return chest


def _face_detection_overlay(ref_bgr: np.ndarray, det: dict, face_mask: np.ndarray, hair_mask: np.ndarray, chest_mask: np.ndarray, left_arm: np.ndarray, right_arm: np.ndarray) -> np.ndarray:
    """Visualize detected facial features and body regions on the reference image."""
    img = ref_bgr.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    labels = ["L-eye", "R-eye", "Nose", "L-mouth", "R-mouth"]
    kps = det["kps5"]
    for lab, pt, col in zip(labels, kps, [(255, 0, 0), (0, 180, 255), (0, 255, 0), (180, 0, 255), (255, 0, 180)]):
        p = tuple(np.round(pt).astype(int).tolist())
        cv2.circle(img, p, 4, col, -1, cv2.LINE_AA)
        cv2.putText(img, lab, (p[0] + 5, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    overlay = img.copy()
    for name, mask in [("face", face_mask), ("hair", hair_mask), ("chest", chest_mask), ("left_arm", left_arm), ("right_arm", right_arm)]:
        overlay[mask > 0] = REGION_BGR[name]
    return cv2.addWeighted(overlay, 0.42, img, 0.58, 0.0)


def _body_overlay(ref_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    img = ref_bgr.copy()
    overlay = img.copy()
    for name in ["face", "hair", "chest", "left_arm", "right_arm"]:
        overlay[masks[name] > 0] = REGION_BGR[name]
    img = cv2.addWeighted(overlay, 0.42, img, 0.58, 0.0)
    for name in ["face", "hair", "chest", "left_arm", "right_arm"]:
        cnts, _ = cv2.findContours((masks[name] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, REGION_BGR[name], 2, cv2.LINE_AA)
    return img


def _mesh_annotations(verts: np.ndarray, px: np.ndarray, py: np.ndarray, part_ids: np.ndarray, face_side: np.ndarray, back_side: np.ndarray, title: str) -> np.ndarray:
    canvas = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    for name, mask in [("face", (part_ids == PART_FACE) & face_side), ("hair", (part_ids == PART_FACE) & back_side), ("chest", (part_ids == PART_TORSO)), ("left_arm", part_ids == PART_LARM), ("right_arm", part_ids == PART_RARM)]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        step = max(1, len(idx) // 6000)
        for i in idx[::step]:
            if 0 <= px[i] < 1024 and 0 <= py[i] < 1024:
                canvas[py[i], px[i]] = REGION_BGR[name]
    cv2.putText(canvas, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def _face_mesh_features(verts: np.ndarray, px: np.ndarray, py: np.ndarray, face_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Locate the best mesh vertex (by highest forward Z) for each facial feature zone."""
    fx = px[face_idx].astype(np.float32)
    fy = py[face_idx].astype(np.float32)
    fz = verts[face_idx, 2].astype(np.float32)
    x0, x1 = float(fx.min()), float(fx.max())
    y0, y1 = float(fy.min()), float(fy.max())
    cx = 0.5 * (x0 + x1)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)

    def pick_fwd(xmask: np.ndarray, ymask: np.ndarray, fallback_xmask: np.ndarray | None = None) -> int:
        """Pick the most-forward vertex in (xmask & ymask); fall back to xmask only, then all."""
        zone = xmask & ymask
        if not np.any(zone):
            zone = xmask if fallback_xmask is None else fallback_xmask
        if not np.any(zone):
            zone = np.ones(len(face_idx), dtype=bool)
        best = int(np.argmax(fz * zone.astype(np.float32) + (-1e9 * (~zone).astype(np.float32))))
        return int(face_idx[best])

    # Eye rows:  upper 25-48 % of face height
    # Nose row:  middle 35-65 %
    # Mouth row: lower 62-88 %
    eye_y   = (fy >= y0 + 0.25 * h) & (fy <= y0 + 0.48 * h)
    nose_y  = (fy >= y0 + 0.35 * h) & (fy <= y0 + 0.65 * h)
    mouth_y = (fy >= y0 + 0.62 * h) & (fy <= y0 + 0.88 * h)

    # Horizontal bands
    left_half  = fx < cx
    right_half = fx >= cx
    center_x   = np.abs(fx - cx) <= 0.18 * w

    left_eye   = pick_fwd(left_half,  eye_y,   left_half)
    right_eye  = pick_fwd(right_half, eye_y,   right_half)
    nose       = pick_fwd(center_x,  nose_y,  np.ones(len(face_idx), dtype=bool))
    mouth_left = pick_fwd(left_half,  mouth_y, left_half)
    mouth_right= pick_fwd(right_half, mouth_y, right_half)

    # Forehead: top 18 % (above eyes)
    forehead_y = fy <= y0 + 0.18 * h
    forehead   = pick_fwd(np.ones(len(face_idx), dtype=bool), forehead_y)

    top        = int(face_idx[np.argmin(fy)])
    chin       = int(face_idx[np.argmax(fy)])
    left_face  = int(face_idx[np.argmin(fx)])
    right_face = int(face_idx[np.argmax(fx)])

    def pt(i: int) -> np.ndarray:
        return np.array([px[i], py[i]], dtype=np.float32)

    return {
        "left_eye":   pt(left_eye),
        "right_eye":  pt(right_eye),
        "nose":       pt(nose),
        "mouth_left": pt(mouth_left),
        "mouth_right":pt(mouth_right),
        "forehead":   pt(forehead),
        "top":        pt(top),
        "chin":       pt(chin),
        "left_face":  pt(left_face),
        "right_face": pt(right_face),
    }


def _face_warp(img_rgb: np.ndarray, det: dict, mesh_feats: dict[str, np.ndarray], qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin-plate spline warp: each detected facial keypoint maps exactly to the
    corresponding mesh feature vertex, so eye→eye, nose→nose, mouth→mouth."""
    from scipy.interpolate import RBFInterpolator

    bbox = det["bbox"]  # x1,y1,x2,y2
    kps5 = np.asarray(det["kps5"], dtype=np.float64)  # [L-eye, R-eye, Nose, L-mouth, R-mouth]
    bx_mid = 0.5 * (bbox[0] + bbox[2])
    by_mid = 0.5 * (bbox[1] + bbox[3])

    # Image-space control points (src)  <->  mesh-projection control points (dst)
    # 5 precise keypoints + 5 boundary anchors = 10 total
    src = np.vstack([
        kps5,                                                     # L-eye, R-eye, Nose, L-mouth, R-mouth
        [[bx_mid,  bbox[1]]],                                     # top-center of face bbox
        [[bx_mid,  bbox[3]]],                                     # chin-center
        [[bbox[0], by_mid ]],                                     # left edge
        [[bbox[2], by_mid ]],                                     # right edge
        [[bx_mid,  bbox[1] + 0.18*(bbox[3]-bbox[1])]],           # forehead
    ], dtype=np.float64)

    dst = np.vstack([
        mesh_feats["left_eye"],
        mesh_feats["right_eye"],
        mesh_feats["nose"],
        mesh_feats["mouth_left"],
        mesh_feats["mouth_right"],
        mesh_feats["top"],
        mesh_feats["chin"],
        mesh_feats["left_face"],
        mesh_feats["right_face"],
        mesh_feats["forehead"],
    ], dtype=np.float64)

    # Fit TPS:  mesh-space → image-space  (inverse map, used for pixel sampling)
    # smoothing=0 → exact interpolation at every control point
    try:
        tps = RBFInterpolator(dst, src, kernel="thin_plate_spline", smoothing=0.0)
    except Exception:
        # Fallback: slight regularisation if points are degenerate
        tps = RBFInterpolator(dst, src, kernel="thin_plate_spline", smoothing=1.0)

    src_q = tps(qxy.astype(np.float64))   # shape (N, 2): image-space sample positions
    sx, sy = src_q[:, 0], src_q[:, 1]

    h, w = img_rgb.shape[:2]
    valid = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
    colors = np.zeros((len(qxy), 3), dtype=np.float32)
    lum    = np.zeros((len(qxy),),  dtype=np.float32)
    if np.any(valid):
        mx = sx[valid].reshape(-1, 1).astype(np.float32)
        my = sy[valid].reshape(-1, 1).astype(np.float32)
        img_f = img_rgb.astype(np.float32) / 255.0
        samp  = cv2.remap(img_f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        colors[valid] = samp.reshape(-1, 3)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lum[valid] = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
    return colors, valid, lum


def _project_mesh(mesh: o3d.geometry.TriangleMesh, front_rgb: np.ndarray, back_rgb: np.ndarray, det_front: dict, masks_front: dict[str, np.ndarray], masks_back: dict[str, np.ndarray]) -> tuple[o3d.geometry.TriangleMesh, dict, dict]:
    out = o3d.geometry.TriangleMesh(mesh)
    out.compute_vertex_normals()
    verts = np.asarray(out.vertices)
    normals = np.asarray(out.vertex_normals)
    part_ids = _parts_for_mesh(verts)
    y = verts[:, 1]
    z = verts[:, 2]
    face_idx = np.where(part_ids == PART_FACE)[0]
    torso_idx = np.where(part_ids == PART_TORSO)[0]
    face_z = np.median(z[face_idx]) if len(face_idx) else np.median(z)
    torso_z = np.median(z[torso_idx]) if len(torso_idx) else np.median(z)

    # Front face should live on the front half of the face/head region.
    face_front = (part_ids == PART_FACE) & (z >= face_z)
    face_back = (part_ids == PART_FACE) & (z < face_z)
    torso_front = (part_ids == PART_TORSO) & (z >= torso_z)
    torso_back = (part_ids == PART_TORSO) & (z < torso_z)

    # Chest-only vertex masks: upper torso only (avoid abdomen/pelvis/leg-adjacent areas).
    if len(torso_idx) > 0:
        ty = y[torso_idx]
        chest_y_min = float(np.quantile(ty, 0.40))
        chest_y_max = float(np.quantile(ty, 0.93))
    else:
        chest_y_min = float(np.quantile(y, 0.40))
        chest_y_max = float(np.quantile(y, 0.93))
    chest_only = (part_ids == PART_TORSO) & (y >= chest_y_min) & (y <= chest_y_max)
    chest_front_verts = chest_only & (z >= torso_z)
    chest_back_verts = chest_only & (z < torso_z)

    # Division line on torso: body texture above, leg texture below, smooth blend in-between.
    if len(torso_idx) > 0:
        torso_div_y = float(np.quantile(ty, 0.30))
        lower_torso_y_min = float(np.quantile(ty, 0.02))
    else:
        torso_div_y = float(np.quantile(y, 0.30))
        lower_torso_y_min = float(np.quantile(y, 0.02))

    lower_torso_only = (part_ids == PART_TORSO) & (y >= lower_torso_y_min) & (y < chest_y_min)
    lower_torso_front_verts = lower_torso_only & (z >= torso_z)
    lower_torso_back_verts = lower_torso_only & (z < torso_z)
    arm_front = ((part_ids == PART_LARM) | (part_ids == PART_RARM)) & (z >= torso_z)
    arm_back = ((part_ids == PART_LARM) | (part_ids == PART_RARM)) & (z < torso_z)

    # Projection coordinates for front and back images.
    px_f, py_f, inside_f = _project(verts, front_rgb.shape[1], front_rgb.shape[0], mirrored=False)
    px_b, py_b, inside_b = _project(verts, back_rgb.shape[1], back_rgb.shape[0], mirrored=True)
    qf = np.stack([np.clip(px_f, 0, front_rgb.shape[1] - 1), np.clip(py_f, 0, front_rgb.shape[0] - 1)], axis=1)
    qb = np.stack([np.clip(px_b, 0, back_rgb.shape[1] - 1), np.clip(py_b, 0, back_rgb.shape[0] - 1)], axis=1)
    px_f_viz, py_f_viz, _ = _project(verts, 1024, 1024, mirrored=False)
    px_b_viz, py_b_viz, _ = _project(verts, 1024, 1024, mirrored=True)

    base = np.asarray(out.vertex_colors).copy()
    colors = base.copy()
    arm_splat_front = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    arm_splat_back = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    chest_splat_front = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    chest_splat_back = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    cv2.putText(arm_splat_front, "SMPL arm splat overlay (view: front, uses frame_000 + frame_165)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(arm_splat_back, "SMPL arm splat overlay (view: back, uses frame_000 + frame_165)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(chest_splat_front, "SMPL chest splat overlay (front: frame_000)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(chest_splat_back, "SMPL chest/back splat overlay (back: frame_165)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)

    if APPLY_FACE_HAIR_TEXTURE:
        # Face: direct face detection + aligned warp to front face half.
        mesh_feats = _face_mesh_features(verts, px_f, py_f, face_idx if len(face_idx) else np.arange(len(verts)))
        face_colors, face_valid, face_lum = _face_warp(front_rgb, det_front, mesh_feats, qf[face_front])
        face_shaded = _shade(face_colors, normals[face_front], face_lum, strength=0.14, lum_boost=0.32)
        colors[np.where(face_front)[0][face_valid]] = colors[np.where(face_front)[0][face_valid]] * 0.05 + face_shaded[face_valid] * 0.95

        # Hair: use top-of-head region from both views, darker and less glossy.
        hair_front = face_front & (verts[:, 1] >= np.quantile(verts[face_idx, 1], 0.72) if len(face_idx) else True)
        hair_back = face_back & (verts[:, 1] >= np.quantile(verts[face_idx, 1], 0.72) if len(face_idx) else True)
        for mask_name, vert_mask, rgb, qxy, masks in [
            ("hair_front", hair_front, front_rgb, qf, masks_front),
            ("hair_back", hair_back, back_rgb, qb, masks_back),
        ]:
            sampled, valid, lum = _direct_sample(rgb, masks["hair"], qxy[vert_mask])
            if not np.any(valid):
                sampled, valid, lum = _nearest_sample(rgb, masks["hair"], qxy[vert_mask])
            shaded = _shade(sampled, normals[vert_mask], lum, strength=0.08, lum_boost=0.18)
            shaded = np.clip(shaded * np.array([0.78, 0.74, 0.70], dtype=np.float32), 0.0, 1.0)
            ii = np.where(vert_mask)[0]
            colors[ii[valid]] = colors[ii[valid]] * 0.08 + shaded[valid] * 0.92

    # ------------------------------------------------------------------ #
    #  Helper: cylindrical shading for an arm segment
    # ------------------------------------------------------------------ #
    # ================================================================= #
    #  CHEST/BACK SHADING (EXTEND/SPREAD):
    #  no hard bbox re-apply; sample nearest chest seeds and blend in.
    # ================================================================= #
    for name, vert_mask, rgb, masks in [
        ("chest_front", chest_front_verts, front_rgb, masks_front),
        ("chest_back",  chest_back_verts,  back_rgb,  masks_back),
    ]:
        ii = np.where(vert_mask)[0]
        if len(ii) == 0:
            continue

        chest_mask = masks["chest"]
        pix_yx = np.argwhere(chest_mask > 0)
        if len(pix_yx) == 0:
            continue

        # Query points from torso-guided normalized XY (guidance only).
        qmask = masks.get("body", chest_mask)
        qyx = np.argwhere(qmask > 0)
        qy0, qx0 = int(qyx[:, 0].min()), int(qyx[:, 1].min())
        qy1, qx1 = int(qyx[:, 0].max()), int(qyx[:, 1].max())
        vx = verts[ii, 0].astype(np.float32)
        vy = verts[ii, 1].astype(np.float32)
        nx = (vx - vx.min()) / max(float(vx.max() - vx.min()), 1e-6)
        ny = 1.0 - (vy - vy.min()) / max(float(vy.max() - vy.min()), 1e-6)
        qx = np.clip((nx * (qx1 - qx0) + qx0).astype(np.float32), 0, rgb.shape[1] - 1)
        qy = np.clip((ny * (qy1 - qy0) + qy0).astype(np.float32), 0, rgb.shape[0] - 1)
        qxy = np.stack([qx, qy], axis=1)

        # Extend/spread from nearest chest seed pixels.
        seed_xy = pix_yx[:, [1, 0]].astype(np.float32)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(seed_xy)
            _d, nn = tree.query(qxy.astype(np.float32), k=1)
            nn = np.asarray(nn, dtype=np.int32)
        except Exception:
            nn = []
            for q in qxy.astype(np.float32):
                d2 = np.sum((seed_xy - q[None, :]) ** 2, axis=1)
                nn.append(int(np.argmin(d2)))
            nn = np.asarray(nn, dtype=np.int32)

        src = pix_yx[nn]
        raw = rgb[src[:, 0], src[:, 1]].astype(np.float32) / 255.0
        gray_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lum = gray_img[src[:, 0], src[:, 1]]
        shaded = _shade(raw, normals[ii], lum, strength=0.32, lum_boost=0.86)

        # Blend-in (extension), do not hard overwrite.
        alpha = 0.78
        colors[ii] = np.clip(colors[ii] * (1.0 - alpha) + shaded * alpha, 0.0, 1.0)

        # Debug overlay.
        canvas = chest_splat_front if "front" in name else chest_splat_back
        px_v = px_f_viz if "front" in name else px_b_viz
        py_v = py_f_viz if "front" in name else py_b_viz
        for local_i, vi in enumerate(ii):
            cx_, cy_ = int(px_v[vi]), int(py_v[vi])
            if 0 <= cx_ < 1024 and 0 <= cy_ < 1024:
                canvas[cy_, cx_] = (np.clip(shaded[local_i], 0.0, 1.0)[::-1] * 255.0).astype(np.uint8)
        _draw_arm_axis(canvas, px_v, py_v, verts, ii, (0, 210, 0))

    # ================================================================= #
    #  LOWER TORSO EXTENSION BLEND (NO SHRINK):
    #  - Map vertices to a torso-guided query position
    #  - Extend body/leg textures by nearest-seed lookup from their masks
    #  - Blend both so they meet smoothly at the division line
    # ================================================================= #
    for name, vert_mask, rgb, masks in [
        ("lower_front", lower_torso_front_verts, front_rgb, masks_front),
        ("lower_back",  lower_torso_back_verts,  back_rgb,  masks_back),
    ]:
        ii = np.where(vert_mask)[0]
        if len(ii) == 0:
            continue

        body_mask = masks.get("body", np.zeros(rgb.shape[:2], dtype=np.uint8))
        leg_mask = masks.get("pants", np.zeros(rgb.shape[:2], dtype=np.uint8))
        body_yx = np.argwhere(body_mask > 0)
        leg_yx = np.argwhere(leg_mask > 0)
        if len(body_yx) == 0 and len(leg_yx) == 0:
            continue

        # Query positions from torso bbox (guidance only; colors come from nearest masked seeds).
        guide_mask = body_mask if len(body_yx) > 0 else leg_mask
        gyx = np.argwhere(guide_mask > 0)
        gy0, gx0 = int(gyx[:, 0].min()), int(gyx[:, 1].min())
        gy1, gx1 = int(gyx[:, 0].max()), int(gyx[:, 1].max())

        vx = verts[ii, 0].astype(np.float32)
        vy = verts[ii, 1].astype(np.float32)
        nx = (vx - vx.min()) / max(float(vx.max() - vx.min()), 1e-6)
        ny = 1.0 - (vy - vy.min()) / max(float(vy.max() - vy.min()), 1e-6)
        qx = np.clip((nx * (gx1 - gx0) + gx0).astype(np.float32), 0, rgb.shape[1] - 1)
        qy = np.clip((ny * (gy1 - gy0) + gy0).astype(np.float32), 0, rgb.shape[0] - 1)
        qxy = np.stack([qx, qy], axis=1)

        def _nearest_from_seed(seed_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if len(seed_yx) == 0:
                return np.zeros((len(ii), 3), dtype=np.float32), np.zeros((len(ii),), dtype=np.float32)
            seed_xy = seed_yx[:, [1, 0]].astype(np.float32)
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(seed_xy)
                _d, nn = tree.query(qxy.astype(np.float32), k=1)
                nn = np.asarray(nn, dtype=np.int32)
            except Exception:
                nn = []
                for q in qxy.astype(np.float32):
                    d2 = np.sum((seed_xy - q[None, :]) ** 2, axis=1)
                    nn.append(int(np.argmin(d2)))
                nn = np.asarray(nn, dtype=np.int32)
            src = seed_yx[nn]
            c = rgb[src[:, 0], src[:, 1]].astype(np.float32) / 255.0
            gray_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            l = gray_img[src[:, 0], src[:, 1]]
            return c, l

        raw_b, lum_b = _nearest_from_seed(body_yx)
        raw_l, lum_l = _nearest_from_seed(leg_yx)

        # Smooth meet around division line: body above, legs below.
        yv = verts[ii, 1].astype(np.float32)
        band = max(1e-4, float(0.07 * (chest_y_min - lower_torso_y_min + 1e-6)))
        w_body = 1.0 / (1.0 + np.exp(-(yv - float(torso_div_y)) / band))
        w_leg = 1.0 - w_body

        if len(body_yx) == 0:
            w_body[:] = 0.0
            w_leg[:] = 1.0
        if len(leg_yx) == 0:
            w_body[:] = 1.0
            w_leg[:] = 0.0

        blended_src = raw_b * w_body[:, None] + raw_l * w_leg[:, None]
        lum = lum_b * w_body + lum_l * w_leg
        shaded = _shade(blended_src, normals[ii], lum, strength=0.28, lum_boost=0.72)

        # Extend and fill reliably (no white spots): high blend coverage in lower torso.
        alpha = 0.84 + 0.12 * np.clip(np.abs(w_body - w_leg), 0.0, 1.0)
        colors[ii] = np.clip(colors[ii] * (1.0 - alpha[:, None]) + shaded * alpha[:, None], 0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  TORSO GAP FILL (K-MEANS): spread textured torso colors to gaps.
    # ------------------------------------------------------------------ #
    torso_all = (part_ids == PART_TORSO)
    colors = _kmeans_spread_fill(colors, base, verts, torso_all, changed_thresh=0.045, k=7)

    # ================================================================= #
    if APPLY_ARM_TEXTURE:
        #  ARM SHADING: AABB bbox-stretch from detected arm region, depth-blend front+back
        # ================================================================= #
        for region, part_id, axis_col in [
            ("left_arm", PART_LARM, (255, 200, 0)),
            ("right_arm", PART_RARM, (0, 140, 255)),
        ]:
            ii = np.where(part_ids == part_id)[0]
            if len(ii) == 0:
                continue

            def _aabb_arm_sample(rgb, arm_mask, vv, nn):
                pix_yx = np.argwhere(arm_mask > 0)
                if len(pix_yx) == 0:
                    return np.zeros((len(vv), 3), dtype=np.float32), np.zeros(len(vv), dtype=bool)
                iy0, ix0 = int(pix_yx[:, 0].min()), int(pix_yx[:, 1].min())
                iy1, ix1 = int(pix_yx[:, 0].max()), int(pix_yx[:, 1].max())
                # For arms: primary axis is Y (top-to-bottom), secondary X
                vy_ = vv[:, 1]
                vx_ = vv[:, 0]
                nx_ = (vx_ - vx_.min()) / max(vx_.max() - vx_.min(), 1e-6)
                ny_ = 1.0 - (vy_ - vy_.min()) / max(vy_.max() - vy_.min(), 1e-6)
                sx = np.clip((nx_ * (ix1 - ix0) + ix0).astype(np.int32), 0, rgb.shape[1] - 1)
                sy = np.clip((ny_ * (iy1 - iy0) + iy0).astype(np.int32), 0, rgb.shape[0] - 1)
                raw = rgb[sy, sx].astype(np.float32) / 255.0
                gray_ = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                lum_ = gray_[sy, sx]
                return _shade(raw, nn, lum_, strength=0.34, lum_boost=0.96), np.ones(len(vv), dtype=bool)

            shaded_f, valid_f = _aabb_arm_sample(front_rgb, masks_front[region], verts[ii], normals[ii])
            shaded_b, valid_b = _aabb_arm_sample(back_rgb,  masks_back[region],  verts[ii], normals[ii])

            if not np.any(valid_f) and not np.any(valid_b):
                continue
            if np.any(valid_f) and not np.any(valid_b):
                final = shaded_f
            elif np.any(valid_b) and not np.any(valid_f):
                final = shaded_b
            else:
                # Depth-blend: front vertices use frame_000, back vertices use frame_165.
                arm_z = z[ii].astype(np.float32)
                sigma = max(float(np.std(arm_z)), 1e-4)
                w_front = 1.0 / (1.0 + np.exp(-(arm_z - torso_z) / (0.55 * sigma + 1e-6)))
                final = shaded_f * w_front[:, None] + shaded_b * (1.0 - w_front[:, None])

            colors[ii] = np.clip(final, 0.0, 1.0)

            # Overlay how splat lands on SMPL in front/back projections.
            for canvas, px_v, py_v in [
                (arm_splat_front, px_f_viz, py_f_viz),
                (arm_splat_back, px_b_viz, py_b_viz),
            ]:
                for local_i, vi in enumerate(ii):
                    x, y = int(px_v[vi]), int(py_v[vi])
                    if 0 <= x < 1024 and 0 <= y < 1024:
                        bgr = (np.clip(final[local_i], 0.0, 1.0)[::-1] * 255.0).astype(np.uint8)
                        canvas[y, x] = bgr
                _draw_arm_axis(canvas, px_v, py_v, verts, ii, axis_col)

    # Arm gap fill with k-NN extension to remove remaining white gaps.
    if APPLY_ARM_TEXTURE:
        arm_all = (part_ids == PART_LARM) | (part_ids == PART_RARM)
        colors = _kmeans_spread_fill(colors, base, verts, arm_all, changed_thresh=0.040, k=6)

    # Global iterative white-gap fill across ALL mesh vertices.
    colors = _iterative_knn_fill_white(
        verts,
        colors,
        passes=10,
        k=10,
        white_mean_thr=0.90,
        white_chroma_thr=0.09,
    )

    # Blend a faint global shading so the body looks more natural.
    global_light = np.array([0.15, -0.25, 0.95], dtype=np.float32)
    global_light /= np.linalg.norm(global_light)
    diffuse = np.clip(normals @ global_light, 0.0, 1.0)
    colors = np.clip(colors * (0.84 + 0.16 * diffuse[:, None]), 0.0, 1.0)

    # Final cleanup after shading (captures any remaining bright white triangles).
    colors = _iterative_knn_fill_white(
        verts,
        colors,
        passes=4,
        k=8,
        white_mean_thr=0.86,
        white_chroma_thr=0.10,
    )
    out.vertex_colors = o3d.utility.Vector3dVector(colors)

    ann_front = _mesh_annotations(verts, px_f, py_f, part_ids, face_front, face_back, "front face / front body")
    ann_back = _mesh_annotations(verts, px_b, py_b, part_ids, face_front, face_back, "back body / hair")
    body_masks_info = {
        "face_front": face_front,
        "face_back": face_back,
        "torso_front": torso_front,
        "torso_back": torso_back,
        "arm_front": arm_front,
        "arm_back": arm_back,
    }
    return out, body_masks_info, {
        "front": ann_front,
        "back": ann_back,
        "arm_splat_front": arm_splat_front,
        "arm_splat_back": arm_splat_back,
        "chest_splat_front": chest_splat_front,
        "chest_splat_back": chest_splat_back,
    }


def _save_preview(mesh: o3d.geometry.TriangleMesh, out_path: Path) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=900)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True
    ctr = vis.get_view_control()
    bbox = mesh.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center())
    ctr.set_front([0.0, -0.05, -1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.72)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()


def main() -> None:
    mesh = _load_mesh()
    front_rgb, front_mask = _read_rgb_and_mask(FRONT_REF_IMG)
    back_rgb, back_mask = _read_rgb_and_mask(BACK_REF_IMG if BACK_REF_IMG.exists() else FRONT_REF_IMG)

    yolo = YoloPosePartDetector(model_name=str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else 'yolov8x-pose.pt'))
    hybrid = HybridPartDetector(yolo_model=str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else 'yolov8x-pose.pt'))
    face_det = FaceDetectorYuNet(FACE_CACHE_DIR)

    front_face = face_det.best_detection(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
    if front_face is None:
        raise RuntimeError("Face detection failed on the front reference image")

    front_parts = hybrid.part_masks(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR), front_mask)
    back_parts = hybrid.part_masks(cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR), back_mask)
    # Joint-detector branch (YOLO pose) for shoulder-aware chest extraction.
    front_parts_joint = yolo.part_masks(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR), front_mask)
    back_parts_joint = yolo.part_masks(cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR), back_mask)

    front_face_mask = np.zeros(front_mask.shape, dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in front_face["bbox"]]
    cv2.rectangle(front_face_mask, (x1, y1), (x2, y2), 255, -1, cv2.LINE_AA)
    front_face_mask = cv2.bitwise_and(front_face_mask, front_mask)

    # Hair around top of head, clipped by person mask.
    face_ys, face_xs = np.where(front_face_mask > 0)
    if len(face_xs) > 0:
        fx0, fx1 = int(face_xs.min()), int(face_xs.max())
        fy0, fy1 = int(face_ys.min()), int(face_ys.max())
        hw = fx1 - fx0 + 1
        hh = fy1 - fy0 + 1
        hair_front = np.zeros_like(front_mask)
        hair_front[max(0, fy0 - int(0.58 * hh)):min(front_mask.shape[0], fy0 + int(0.18 * hh) + 1),
                    max(0, fx0 - int(0.18 * hw)):min(front_mask.shape[1], fx1 + int(0.18 * hw) + 1)] = 255
        hair_front = cv2.bitwise_and(hair_front, front_mask)
        hair_front = cv2.bitwise_and(hair_front, cv2.bitwise_not(front_face_mask))
    else:
        hair_front = np.zeros_like(front_mask)

    # Back hair/top-of-head mask from the back image.
    back_face_like = np.zeros_like(back_mask)
    b_ys, b_xs = np.where(back_parts[PART_TORSO] > 0)
    if len(b_xs) > 0:
        bx0, bx1 = int(b_xs.min()), int(b_xs.max())
        by0, by1 = int(b_ys.min()), int(b_ys.max())
        bw = bx1 - bx0 + 1
        bh = by1 - by0 + 1
        back_hair = np.zeros_like(back_mask)
        back_hair[max(0, by0 - int(0.52 * bh)):min(back_mask.shape[0], by0 + int(0.20 * bh) + 1),
                  max(0, bx0 - int(0.18 * bw)):min(back_mask.shape[1], bx1 + int(0.18 * bw) + 1)] = 255
        back_hair = cv2.bitwise_and(back_hair, back_mask)
    else:
        back_hair = np.zeros_like(back_mask)

    # Chest detection as inverse: full body minus all other parts (head, arms, legs).
    # This avoids direct chest detection artifacts and contamination.
    front_head_mask = np.zeros_like(front_mask)
    front_head_mask[max(0, int(y1) - 20):min(front_mask.shape[0], int(y2) + 10), 
                     max(0, int(x1)):min(front_mask.shape[1], int(x2))] = 255
    front_head_mask = cv2.bitwise_and(front_head_mask, front_mask)
    
    front_chest = _chest_from_inverse(
        person_mask=front_mask,
        face_mask=front_face_mask,
        head_mask=front_head_mask,
        left_arm_mask=front_parts[PART_LARM],
        right_arm_mask=front_parts[PART_RARM],
        left_leg_mask=front_parts[PART_LLEG],
        right_leg_mask=front_parts[PART_RLEG],
    )

    back_head_mask = np.zeros_like(back_mask)
    b_face_ys, b_face_xs = np.where(back_parts[PART_TORSO] > 0)
    if len(b_face_xs) > 0:
        bx0_face, bx1_face = int(b_face_xs.min()), int(b_face_xs.max())
        by0_face, by1_face = int(b_face_ys.min()), int(b_face_ys.max())
        bw_face = bx1_face - bx0_face + 1
        bh_face = by1_face - by0_face + 1
        back_head_mask[max(0, by0_face - int(0.3 * bh_face)):min(back_mask.shape[0], by0_face + int(0.08 * bh_face)),
                       max(0, bx0_face):min(back_mask.shape[1], bx1_face)] = 255
    back_head_mask = cv2.bitwise_and(back_head_mask, back_mask)
    
    back_chest = _chest_from_inverse(
        person_mask=back_mask,
        face_mask=np.zeros_like(back_mask),  # Face not visible from back
        head_mask=back_head_mask,
        left_arm_mask=back_parts[PART_LARM],
        right_arm_mask=back_parts[PART_RARM],
        left_leg_mask=back_parts[PART_LLEG],
        right_leg_mask=back_parts[PART_RLEG],
    )

    # External red overlay belongs to back view: use as strict back chest prior.
    ext_back_chest = _external_red_overlay_to_chest_mask(EXTERNAL_BACK_CHEST_RED_OVERLAY, back_mask)
    if np.any(ext_back_chest > 0):
        # Keep only torso pixels, and remove other body parts so it won't spill.
        ext_back_chest = cv2.bitwise_and(ext_back_chest, back_parts[PART_TORSO])
        ext_back_chest = cv2.bitwise_and(ext_back_chest, cv2.bitwise_not(back_head_mask))
        ext_back_chest = cv2.bitwise_and(ext_back_chest, cv2.bitwise_not(back_parts[PART_LARM]))
        ext_back_chest = cv2.bitwise_and(ext_back_chest, cv2.bitwise_not(back_parts[PART_RARM]))
        ext_back_chest = cv2.bitwise_and(ext_back_chest, cv2.bitwise_not(back_parts[PART_LLEG]))
        ext_back_chest = cv2.bitwise_and(ext_back_chest, cv2.bitwise_not(back_parts[PART_RLEG]))
        ext_back_chest = cv2.morphologyEx(ext_back_chest, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        ext_back_chest = cv2.morphologyEx(ext_back_chest, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        back_chest = ext_back_chest

    # Leg-patch masks (used to texture pelvis/butt/genital torso area only below division line).
    def _pants_from_parts(person_mask: np.ndarray, torso_mask: np.ndarray, lleg: np.ndarray, rleg: np.ndarray) -> np.ndarray:
        p = cv2.bitwise_or(lleg, rleg)
        p = cv2.bitwise_and(p, person_mask)
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return p

    front_pants = _pants_from_parts(front_mask, front_parts[PART_TORSO], front_parts[PART_LLEG], front_parts[PART_RLEG])
    back_pants = _pants_from_parts(back_mask, back_parts[PART_TORSO], back_parts[PART_LLEG], back_parts[PART_RLEG])


    zero_front = np.zeros_like(front_mask)
    zero_back = np.zeros_like(back_mask)

    front_body_overlay = {
        "face": zero_front,
        "hair": zero_front,
        "chest": front_chest,
        "left_arm": zero_front,
        "right_arm": zero_front,
    }
    back_body_overlay = {
        "face": zero_back,
        "hair": zero_back,
        "chest": back_chest,
        "left_arm": zero_back,
        "right_arm": zero_back,
    }

    front_det_img = _face_detection_overlay(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR), front_face, front_face_mask, hair_front, front_chest, front_parts[PART_LARM], front_parts[PART_RARM])
    front_body_img = _body_overlay(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR), front_body_overlay)
    back_body_img = _body_overlay(cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR), back_body_overlay)
    
    # Arm detection overlays
    front_arm_det = _arm_detection_overlay(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR), front_parts[PART_LARM], front_parts[PART_RARM], "Arms (front frame_000)")
    back_arm_det = _arm_detection_overlay(cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR), back_parts[PART_LARM], back_parts[PART_RARM], "Arms (back frame_165)")
    
    cv2.imwrite(str(OUT_FRONT_FACE_DET), front_det_img)
    cv2.imwrite(str(OUT_FRONT_BODY), front_body_img)
    cv2.imwrite(str(OUT_BACK_BODY), back_body_img)
    cv2.imwrite(str(OUT_FRONT_ARM_DET), front_arm_det)
    cv2.imwrite(str(OUT_BACK_ARM_DET), back_arm_det)
    cv2.imwrite(str(OUT_BACK_CHEST_MASK_USED), back_chest)

    verts = np.asarray(mesh.vertices)
    part_ids = _parts_for_mesh(verts)
    px_f, py_f, _ = _project(verts, front_rgb.shape[1], front_rgb.shape[0], mirrored=False)
    px_b, py_b, _ = _project(verts, back_rgb.shape[1], back_rgb.shape[0], mirrored=True)
    front_face_idx = np.where((part_ids == PART_FACE) & (verts[:, 2] >= np.median(verts[part_ids == PART_FACE, 2]) if np.any(part_ids == PART_FACE) else True))[0]
    back_face_idx = np.where((part_ids == PART_FACE) & (verts[:, 2] < np.median(verts[part_ids == PART_FACE, 2]) if np.any(part_ids == PART_FACE) else False))[0]
    front_larm_idx = np.where((part_ids == PART_LARM))[0]
    front_rarm_idx = np.where((part_ids == PART_RARM))[0]

    ann_front = _mesh_annotations(verts, px_f, py_f, part_ids, (part_ids == PART_FACE) & (verts[:, 2] >= np.median(verts[part_ids == PART_FACE, 2]) if np.any(part_ids == PART_FACE) else True), (part_ids == PART_FACE) & (verts[:, 2] < np.median(verts[part_ids == PART_FACE, 2]) if np.any(part_ids == PART_FACE) else False), "front face / front body")
    ann_back = _mesh_annotations(verts, px_b, py_b, part_ids, (part_ids == PART_FACE) & (verts[:, 2] >= np.median(verts[part_ids == PART_FACE, 2]) if np.any(part_ids == PART_FACE) else True), (part_ids == PART_FACE) & (verts[:, 2] < np.median(verts[part_ids == PART_FACE, 2]) if np.any(part_ids == PART_FACE) else False), "back body / hair")
    
    # Arm mesh annotations
    arm_ann_front = _mesh_arm_annotations(verts, px_f, py_f, front_larm_idx, front_rarm_idx, "SMPL arms (front)")
    arm_ann_back = _mesh_arm_annotations(verts, px_b, py_b, front_larm_idx, front_rarm_idx, "SMPL arms (back)")
    
    cv2.imwrite(str(OUT_FRONT_ANN), ann_front)
    cv2.imwrite(str(OUT_BACK_ANN), ann_back)
    cv2.imwrite(str(OUT_FRONT_ARM_ANN), arm_ann_front)
    cv2.imwrite(str(OUT_BACK_ARM_ANN), arm_ann_back)

    textured, masks_info, anns = _project_mesh(mesh, front_rgb, back_rgb, front_face, {
        "face": front_face_mask,
        "hair": hair_front,
        "body": front_parts[PART_TORSO],
        "chest": front_chest,
        "pants": front_pants,
        "left_arm": front_parts[PART_LARM],
        "right_arm": front_parts[PART_RARM],
    }, {
        "face": back_face_like,
        "hair": back_hair,
        "body": back_parts[PART_TORSO],
        "chest": back_chest,
        "pants": back_pants,
        "left_arm": back_parts[PART_LARM],
        "right_arm": back_parts[PART_RARM],
    })

    if "arm_splat_front" in anns:
        cv2.imwrite(str(OUT_FRONT_ARM_SPLAT), anns["arm_splat_front"])
    if "arm_splat_back" in anns:
        cv2.imwrite(str(OUT_BACK_ARM_SPLAT), anns["arm_splat_back"])
    if "chest_splat_front" in anns:
        cv2.imwrite(str(OUT_FRONT_CHEST_SPLAT), anns["chest_splat_front"])
    if "chest_splat_back" in anns:
        cv2.imwrite(str(OUT_BACK_CHEST_SPLAT), anns["chest_splat_back"])

    o3d.io.write_triangle_mesh(str(OUT_PLY), textured, write_vertex_colors=True)
    _save_preview(textured, OUT_PREVIEW)

    print(f"Saved: {OUT_FRONT_FACE_DET}")
    print(f"Saved: {OUT_FRONT_BODY}")
    print(f"Saved: {OUT_BACK_BODY}")
    print(f"Saved: {OUT_FRONT_ARM_DET}")
    print(f"Saved: {OUT_BACK_ARM_DET}")
    print(f"Saved: {OUT_FRONT_ANN}")
    print(f"Saved: {OUT_BACK_ANN}")
    print(f"Saved: {OUT_FRONT_ARM_ANN}")
    print(f"Saved: {OUT_BACK_ARM_ANN}")
    print(f"Saved: {OUT_FRONT_ARM_SPLAT}")
    print(f"Saved: {OUT_BACK_ARM_SPLAT}")
    print(f"Saved: {OUT_FRONT_CHEST_SPLAT}")
    print(f"Saved: {OUT_BACK_CHEST_SPLAT}")
    print(f"Saved: {OUT_BACK_CHEST_MASK_USED}")
    print(f"Saved: {OUT_PLY}")
    print(f"Saved: {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
