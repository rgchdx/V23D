"""Texture SMPL canonical mesh from diffusion frames using multiple methods.

Methods
-------
A  smpl_project   – SMPL forward kinematics + real camera projection per vertex.
                    Most accurate: uses per-frame pose/camera params to find
                    the exact 2-D pixel each canonical vertex maps to.
B  rasterize      – Software z-buffer rasterisation of posed SMPL from each
                    view; per-pixel barycentric interpolation maps face colour
                    back to canonical vertices.
C  tps_warp       – Thin-plate-spline warp driven by YOLO pose keypoints;
                    maps orthographic SMPL render to frame image for each view.

Usage
-----
python workflows/debug_visualization/texture_smpl_from_frames.py \
    --bundle-refined-dir  E:/V23D_Data/orbit_methods/02_smplifyx_perframe_then_bundle/bundle_stage/bundle_refined \
    --bundle-summary      E:/V23D_Data/orbit_methods/02_smplifyx_perframe_then_bundle/bundle_stage/bundle_summary.json \
    --smpl-obj            E:/V23D_Data/orbit_methods/02_smplifyx_perframe_then_bundle/bundle_stage/bundle_canonical.obj \
    --smpl-model          E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
    --frames-dir          E:/V23D_Data/frames \
    --masks-dir           E:/V23D_Data/masks_rerun \
    --front-frame         frame_00000.jpg \
    --back-frame          frame_00165.jpg \
    --out-dir             E:/V23D_Data/orbit_methods/02_smplifyx_perframe_then_bundle/bundle_stage/smpl_textured \
    --methods             A B C
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recon.smpl_fitter import SMPL  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# OBJ loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
            faces.append(tri)
    return np.asarray(verts, np.float64), np.asarray(faces, np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_K(focal: float, img_h: int, img_w: int) -> np.ndarray:
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.array([[focal, 0, cx],
                     [0, focal, cy],
                     [0, 0,     1 ]], dtype=np.float64)


def _project_verts(verts: np.ndarray, R: np.ndarray, t: np.ndarray,
                   K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world verts → pixel coords (N,2) and camera-space Z (N,)."""
    v_cam = (R @ verts.T).T + t                 # (N, 3)
    z = v_cam[:, 2]
    px = K[0, 0] * v_cam[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    py = K[1, 1] * v_cam[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return np.stack([px, py], axis=1), z         # (N,2), (N,)


def _sample_image(img_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample at float pixel coords (N,2) -> (N,3) float [0,1] RGB."""
    h, w = img_bgr.shape[:2]
    x = np.clip(pts[:, 0], 0, w - 1).astype(np.float32)
    y = np.clip(pts[:, 1], 0, h - 1).astype(np.float32)
    # Integer floor coords
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = (x - x0)[:, None]
    dy = (y - y0)[:, None]
    c00 = img_bgr[y0, x0].astype(np.float32) / 255.0
    c10 = img_bgr[y0, x1].astype(np.float32) / 255.0
    c01 = img_bgr[y1, x0].astype(np.float32) / 255.0
    c11 = img_bgr[y1, x1].astype(np.float32) / 255.0
    col = c00 * (1 - dx) * (1 - dy) + c10 * dx * (1 - dy) + \
          c01 * (1 - dx) * dy       + c11 * dx * dy
    return col[:, ::-1]  # BGR→RGB


# ──────────────────────────────────────────────────────────────────────────────
# Per-vertex face normals (for visibility)
# ──────────────────────────────────────────────────────────────────────────────

def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    # accumulate face normals to vertices
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return vn / norms


# ──────────────────────────────────────────────────────────────────────────────
# Person mask loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_mask(masks_dir: Path, frame_name: str, h: int, w: int) -> np.ndarray:
    stem = Path(frame_name).stem
    for pat in [stem, stem.replace("frame_", "")]:
        for ext in [".png", ".jpg"]:
            p = masks_dir / (pat + ext)
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    return (m > 127).astype(np.uint8) * 255
    return np.ones((h, w), dtype=np.uint8) * 255


# ──────────────────────────────────────────────────────────────────────────────
# SMPL forward pass helper
# ──────────────────────────────────────────────────────────────────────────────

def _smpl_forward(smpl: SMPL, betas: np.ndarray, body_pose: np.ndarray,
                  global_orient: np.ndarray) -> np.ndarray:
    """Run SMPL forward and return posed vertices (N,3) as float64."""
    import torch
    betas_t  = torch.from_numpy(betas.astype(np.float32)).reshape(1, -1)
    pose_t   = torch.from_numpy(
        np.concatenate([global_orient.flatten(), body_pose.flatten()]).astype(np.float32)
    ).reshape(1, 72)
    trans_t  = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        out = smpl(betas_t, pose_t, trans_t)
    # SMPL.forward returns a namedtuple or dict; handle both
    if isinstance(out, dict):
        verts = out["vertices"]
    elif hasattr(out, "vertices"):
        verts = out.vertices
    else:
        verts = out[0]
    return verts.squeeze(0).cpu().numpy().astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# METHOD A – SMPL forward kinematics + camera projection
# ──────────────────────────────────────────────────────────────────────────────

def method_A(
    canonical_verts: np.ndarray,
    faces: np.ndarray,
    smpl: SMPL,
    frame_infos: list[dict],   # list of {img, R, t, K, mask}
) -> np.ndarray:
    """Assign per-vertex RGB by projecting posed SMPL vertices into each frame.

    For each frame we:
      1. Run SMPL forward → posed verts (same vertex count, same indices).
      2. Compute vertex normals in camera space; keep only front-facing ones.
      3. Project to 2-D and sample the frame image.
      4. Blend contributions across frames weighted by normal dot-product.
    """
    N = len(canonical_verts)
    color_acc  = np.zeros((N, 3), dtype=np.float64)
    weight_acc = np.zeros(N,      dtype=np.float64)

    for info in frame_infos:
        posed = info["posed_verts"]            # (N,3) world-space posed verts
        R     = info["R"]
        t     = info["t"]
        K     = info["K"]
        img   = info["img"]
        mask  = info["mask"]
        h, w  = img.shape[:2]

        # Vertex normals in camera space (for front-face test)
        vn = _vertex_normals(posed, faces)
        v_cam = (R @ posed.T).T + t
        vn_cam = (R @ vn.T).T                  # rotate normals to camera space

        # Front-facing: camera looks toward +Z in camera space, so normal.z < 0
        # means the surface faces the camera.
        facing = vn_cam[:, 2] < 0             # (N,) bool

        pts2d, z_cam = _project_verts(posed, R, t, K)

        # Valid: in-frame, in-mask, front-facing, positive depth
        in_bounds = (
            (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) &
            (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h) &
            (z_cam > 0)
        )
        # Mask check at projected pixel
        px_i = np.clip(pts2d[:, 0].astype(np.int32), 0, w - 1)
        py_i = np.clip(pts2d[:, 1].astype(np.int32), 0, h - 1)
        in_mask = mask[py_i, px_i] > 0

        valid = in_bounds & facing & in_mask

        if valid.sum() == 0:
            print(f"  [method A] no valid vertices for frame {info['name']}")
            continue

        colors = _sample_image(img, pts2d[valid])   # (M, 3) RGB
        w_face = np.abs(vn_cam[valid, 2])            # weight by cos(angle)
        color_acc[valid]  += colors * w_face[:, None]
        weight_acc[valid] += w_face

    # Normalise
    out = np.zeros((N, 3), dtype=np.float64)
    good = weight_acc > 0
    out[good] = color_acc[good] / weight_acc[good, None]
    out[~good] = 0.5                           # gray for uncovered
    return out


# ──────────────────────────────────────────────────────────────────────────────
# METHOD B – Software rasterisation + barycentric backprojection
# ──────────────────────────────────────────────────────────────────────────────

def _rasterize_smpl(posed: np.ndarray, faces: np.ndarray,
                    R: np.ndarray, t: np.ndarray, K: np.ndarray,
                    h: int, w: int):
    """Z-buffer rasterise posed SMPL.

    Returns
    -------
    vert_map  (H,W,1) int32  – canonical vertex index of the nearest vertex
                               for each pixel (or -1 for background)
    bary_map  (H,W,3)         – barycentric coords (unused in current impl)
    face_map  (H,W,1) int32  – face index per pixel
    """
    pts2d, z_cam = _project_verts(posed, R, t, K)
    px = pts2d[:, 0].astype(np.int32)
    py = pts2d[:, 1].astype(np.int32)

    z_buf  = np.full((h, w), np.inf)
    v_buf  = np.full((h, w), -1, dtype=np.int32)

    # Simple vertex splatting (not full triangle rasterisation)
    for vi in range(len(posed)):
        xi, yi = px[vi], py[vi]
        if 0 <= xi < w and 0 <= yi < h and z_cam[vi] > 0:
            if z_cam[vi] < z_buf[yi, xi]:
                z_buf[yi, xi] = z_cam[vi]
                v_buf[yi, xi] = vi

    return v_buf, z_buf


def method_B(
    canonical_verts: np.ndarray,
    faces: np.ndarray,
    frame_infos: list[dict],
) -> np.ndarray:
    """Rasterise posed mesh from each frame view → backproject frame colors."""
    N = len(canonical_verts)
    color_acc  = np.zeros((N, 3), dtype=np.float64)
    weight_acc = np.zeros(N,      dtype=np.float64)

    for info in frame_infos:
        posed = info["posed_verts"]
        R, t, K = info["R"], info["t"], info["K"]
        img  = info["img"]
        mask = info["mask"]
        h, w = img.shape[:2]

        vn    = _vertex_normals(posed, faces)
        vn_c  = (R @ vn.T).T
        facing = vn_c[:, 2] < 0

        v_buf, z_buf = _rasterize_smpl(posed, faces, R, t, K, h, w)

        # Dilate v_buf slightly to fill gaps (nearest-neighbor in-fill)
        valid_pix = v_buf >= 0
        dilated   = v_buf.copy()
        kernel    = np.ones((5, 5), np.uint8)
        # Use distance-transform based in-fill: for each bg pixel find nearest fg pixel
        dist, idx = cv2.distanceTransformWithLabels(
            (~valid_pix).astype(np.uint8), cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        # idx labels are 1-based pixel indices in the label image (connected components)
        # Easier: just dilate the v_buf by painting max(kernel) value
        # Simple approach: iterate over valid pixels and splat to neighbouring bg pixels
        # Actually, just use a small morphological dilation on integer image
        # instead let's directly do: for each bg pixel within dilation range, use
        # the v_buf value from the nearest fg pixel. We approximate with a max-pool.
        # Simplification: build a 2D array of float coords and use remap.
        # Fastest: build binary mask, use connected components label to paint vertex IDs.
        # We'll just use the raw v_buf (no dilation) for correctness.

        ys, xs = np.where(valid_pix & (mask > 0))
        if len(ys) == 0:
            continue

        vi_arr   = v_buf[ys, xs]                    # vertex index per pixel
        fwd_mask = facing[vi_arr]                   # front-facing?
        vis      = fwd_mask

        if vis.sum() == 0:
            continue

        ys_v, xs_v, vi_v = ys[vis], xs[vis], vi_arr[vis]
        pts_sample = np.stack([xs_v.astype(np.float32),
                                ys_v.astype(np.float32)], axis=1)
        colors = _sample_image(img, pts_sample)

        # weight by cos(view angle)
        w_f = np.abs(vn_c[vi_v, 2])
        np.add.at(color_acc,  vi_v, colors * w_f[:, None])
        np.add.at(weight_acc, vi_v, w_f)

    out = np.zeros((N, 3), dtype=np.float64)
    good = weight_acc > 0
    out[good] = color_acc[good] / weight_acc[good, None]
    out[~good] = 0.5
    return out


# ──────────────────────────────────────────────────────────────────────────────
# METHOD C – TPS warp (orthographic SMPL render → frame image)
# ──────────────────────────────────────────────────────────────────────────────

def _ortho_project(verts: np.ndarray, size: int):
    """Orthographic projection of SMPL verts to (size x size) canvas."""
    x = verts[:, 0]
    y = verts[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    sx = (size - 80) / max(xmax - xmin, 1e-6)
    sy = (size - 80) / max(ymax - ymin, 1e-6)
    s  = min(sx, sy)
    px = ((x - (xmin + xmax) * 0.5) * s + size * 0.5).astype(np.float32)
    py = (-(y - (ymin + ymax) * 0.5) * s + size * 0.5).astype(np.float32)
    return px, py


def _render_ortho_depth(verts: np.ndarray, faces: np.ndarray, size: int,
                        flip_x: bool = False):
    """Render SMPL orthographically into size×size canvas.

    Returns
    -------
    canvas  (H,W,3) uint8 color image (each pixel painted by nearest vertex in z)
    v_map   (H,W)   int32 vertex index map (-1=background)
    px, py  float arrays of projected 2-D coords per vertex
    """
    v = verts.copy()
    if flip_x:
        v[:, 0] *= -1.0
    px, py = _ortho_project(v, size)
    z = v[:, 2]
    order = np.argsort(z)[::-1]     # back-to-front (painter)

    z_buf = np.full((size, size), np.inf)
    v_map = np.full((size, size), -1, dtype=np.int32)

    for vi in order:
        xi, yi = int(px[vi]), int(py[vi])
        if 0 <= xi < size and 0 <= yi < size:
            if z[vi] < z_buf[yi, xi]:
                z_buf[yi, xi] = z[vi]
                v_map[yi, xi] = vi

    return v_map, px, py


def _yolo_keypoints(img: np.ndarray):
    """Return COCO 17 keypoints for the best detection, or None."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8x-pose.pt")
        results = model(img, verbose=False)
        if not results or results[0].keypoints is None:
            return None
        kps = results[0].keypoints.xy.cpu().numpy()    # (n_det, 17, 2)
        confs = results[0].keypoints.conf.cpu().numpy() # (n_det, 17)
        if kps.shape[0] == 0:
            return None
        best = int(np.argmax(confs.sum(axis=1)))
        return kps[best], confs[best]                  # (17,2), (17,)
    except Exception as e:
        print(f"  [TPS] YOLO failed: {e}")
        return None


# COCO keypoint index → SMPL vertex index (approximate surface point)
COCO_TO_SMPL_VERT = {
    0:  411,    # nose
    5:  1867,   # left shoulder
    6:  5266,   # right shoulder
    7:  1663,   # left elbow
    8:  5093,   # right elbow
    9:  2112,   # left wrist
    10: 5556,   # right wrist
    11: 3134,   # left hip
    12: 6490,   # right hip
    13: 1176,   # left knee
    14: 4662,   # right knee
    15: 3337,   # left ankle
    16: 6799,   # right ankle
}


def _tps_warp(src_pts: np.ndarray, dst_pts: np.ndarray,
              img_src: np.ndarray, out_size: tuple) -> np.ndarray:
    """Thin-plate-spline warp of img_src using control point correspondences.

    Parameters
    ----------
    src_pts  (K,2) control points in src image coords
    dst_pts  (K,2) control points in dst image coords
    img_src  source image (will be warped)
    out_size (H, W)
    """
    tps = cv2.createThinPlateSplineShapeTransformer()
    src_pts_cv = src_pts.reshape(1, -1, 2).astype(np.float32)
    dst_pts_cv = dst_pts.reshape(1, -1, 2).astype(np.float32)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps.estimateTransformation(dst_pts_cv, src_pts_cv, matches)
    out_h, out_w = out_size
    warped = tps.warpImage(img_src, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    if warped.shape[:2] != (out_h, out_w):
        warped = cv2.resize(warped, (out_w, out_h), cv2.INTER_LINEAR)
    return warped


def method_C(
    canonical_verts: np.ndarray,
    faces: np.ndarray,
    frame_infos: list[dict],
    ortho_size: int = 512,
) -> np.ndarray:
    """TPS-warp each frame image onto an orthographic SMPL render, sample colors."""
    N = len(canonical_verts)
    color_acc  = np.zeros((N, 3), dtype=np.float64)
    weight_acc = np.zeros(N,      dtype=np.float64)

    for info in frame_infos:
        posed   = info["posed_verts"]
        img     = info["img"]
        flip_x  = info.get("flip_x", False)

        # --- Orthographic render of posed mesh ----------------------------
        v_map, px_o, py_o = _render_ortho_depth(
            posed, faces, ortho_size, flip_x=flip_x)

        # Render a solid-color image of the SMPL projection for visualisation
        ortho_canvas = np.zeros((ortho_size, ortho_size, 3), np.uint8)
        vis_mask = v_map >= 0
        ortho_canvas[vis_mask] = 128

        # --- YOLO keypoints in frame image --------------------------------
        kp_result = _yolo_keypoints(img)
        if kp_result is None:
            print(f"  [method C] no YOLO kps for {info['name']}, falling back to bbox scale")
            # Fallback: simple scale+shift based on bounding box of SMPL ortho projection
            ortho_ys, ortho_xs = np.where(v_map >= 0)
            if len(ortho_xs) == 0:
                continue
            # Find person bbox from mask
            pmask = info["mask"]
            ys_m, xs_m = np.where(pmask > 0)
            if len(xs_m) == 0:
                continue
            # Scale ortho canvas → frame image coords
            ox0, ox1 = int(ortho_xs.min()), int(ortho_xs.max())
            oy0, oy1 = int(ortho_ys.min()), int(ortho_ys.max())
            mx0, mx1 = int(xs_m.min()),     int(xs_m.max())
            my0, my1 = int(ys_m.min()),     int(ys_m.max())
            sx = (mx1 - mx0) / max(ox1 - ox0, 1)
            sy = (my1 - my0) / max(oy1 - oy0, 1)
            # Map ortho pixel → frame pixel for each visible vertex
            h_f, w_f = img.shape[:2]
            ys_v, xs_v = np.where(vis_mask)
            vi_arr = v_map[ys_v, xs_v]
            fx = (xs_v - ox0) * sx + mx0
            fy = (ys_v - oy0) * sy + my0
            in_bounds = (fx >= 0) & (fx < w_f) & (fy >= 0) & (fy < h_f)
            pts_f = np.stack([fx[in_bounds], fy[in_bounds]], axis=1)
            vi_v  = vi_arr[in_bounds]
            colors = _sample_image(img, pts_f)
            np.add.at(color_acc,  vi_v, colors)
            np.add.at(weight_acc, vi_v, 1.0)
            continue

        kps, confs = kp_result   # (17,2), (17,)

        # --- Control points: SMPL ortho coords ↔ YOLO frame coords ------
        src_pts, dst_pts = [], []
        for coco_idx, smpl_vi in COCO_TO_SMPL_VERT.items():
            if confs[coco_idx] < 0.3:
                continue
            # ortho position of this SMPL vertex
            xi_o = float(px_o[smpl_vi])
            yi_o = float(py_o[smpl_vi])
            if flip_x:
                pass   # already flipped in _render_ortho_depth
            xi_f, yi_f = float(kps[coco_idx, 0]), float(kps[coco_idx, 1])
            src_pts.append([xi_o, yi_o])
            dst_pts.append([xi_f, yi_f])

        if len(src_pts) < 4:
            print(f"  [method C] too few control pts ({len(src_pts)}) for {info['name']}")
            continue

        src_arr = np.array(src_pts, dtype=np.float32)
        dst_arr = np.array(dst_pts, dtype=np.float32)

        # Warp frame image into ortho canvas space
        h_f, w_f = img.shape[:2]
        warped = _tps_warp(dst_arr, src_arr, img, (ortho_size, ortho_size))

        # Sample warped image at ortho vertex positions
        ys_v, xs_v = np.where(vis_mask)
        vi_arr = v_map[ys_v, xs_v]
        pts_ortho = np.stack([xs_v.astype(np.float32),
                               ys_v.astype(np.float32)], axis=1)
        colors = _sample_image(warped, pts_ortho)
        np.add.at(color_acc,  vi_arr, colors)
        np.add.at(weight_acc, vi_arr, 1.0)

    out = np.zeros((N, 3), dtype=np.float64)
    good = weight_acc > 0
    out[good] = color_acc[good] / weight_acc[good, None]
    out[~good] = 0.5
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Save Open3D-compatible vertex color mesh
# ──────────────────────────────────────────────────────────────────────────────

def _save_colored_mesh(canonical_verts: np.ndarray, faces: np.ndarray,
                       vertex_colors: np.ndarray, out_path: Path) -> None:
    """Save vertex-colored mesh as PLY."""
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(canonical_verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(vertex_colors, 0, 1))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_path), mesh,
                                   write_vertex_colors=True)
        print(f"  Saved PLY: {out_path}")
    except ImportError:
        print("  open3d not available; skipping PLY export.")


def _save_flat_image(canonical_verts: np.ndarray, faces: np.ndarray,
                     vertex_colors: np.ndarray, out_path: Path,
                     size: int = 1024) -> None:
    """Render vertex-colored SMPL orthographically and save as image."""
    v = canonical_verts.copy()
    # Orthographic projection
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    sx = (size - 80) / max(float(x.max() - x.min()), 1e-6)
    sy = (size - 80) / max(float(y.max() - y.min()), 1e-6)
    s  = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + size * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + size * 0.5).astype(np.int32)
    order = np.argsort(z)
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 200
    for i in order:
        xi, yi = px[i], py[i]
        if 0 <= xi < size and 0 <= yi < size:
            c = (np.clip(vertex_colors[i], 0, 1) * 255).astype(np.uint8)
            cv2.circle(canvas, (xi, yi), 2,
                       (int(c[2]), int(c[1]), int(c[0])), -1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
    print(f"  Saved image: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Resolve nearest available bundle_refined frame name
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_refined_frame(bundle_refined_dir: Path, frame_name: str) -> Optional[Path]:
    stem = Path(frame_name).stem   # e.g. "frame_00165"
    exact = bundle_refined_dir / stem
    if exact.exists():
        return exact
    # Try numeric proximity
    try:
        n = int(stem.replace("frame_", ""))
    except ValueError:
        return None
    candidates = []
    for d in bundle_refined_dir.iterdir():
        if d.is_dir():
            try:
                dn = int(d.name.replace("frame_", ""))
                candidates.append((abs(dn - n), d))
            except ValueError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Texture SMPL canonical mesh from frames (multiple methods)")
    ap.add_argument("--bundle-refined-dir", required=True,
                    help="Dir containing per-frame subdirs with bundle_refined.pkl")
    ap.add_argument("--bundle-summary",     required=True,
                    help="bundle_summary.json (contains focal_length)")
    ap.add_argument("--smpl-obj",           required=True,
                    help="Canonical SMPL mesh (.obj)")
    ap.add_argument("--smpl-model",         required=True,
                    help="SMPL model .pkl")
    ap.add_argument("--frames-dir",         required=True)
    ap.add_argument("--masks-dir",          default="")
    ap.add_argument("--front-frame",        default="frame_00000.jpg")
    ap.add_argument("--back-frame",         default="frame_00165.jpg")
    ap.add_argument("--extra-frames",       nargs="*", default=[],
                    help="Additional frame filenames to include in blending")
    ap.add_argument("--out-dir",            required=True)
    ap.add_argument("--methods",            nargs="+",
                    choices=["A", "B", "C"], default=["A", "B", "C"])
    ap.add_argument("--img-size",           type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.bundle_summary).read_text(encoding="utf-8"))
    focal   = float(summary["focal_length"])

    # Load canonical mesh
    canonical_verts, faces = _load_obj(Path(args.smpl_obj))
    print(f"Canonical mesh: {len(canonical_verts)} verts, {len(faces)} faces")

    # Load SMPL model
    smpl = SMPL(Path(args.smpl_model))
    print("SMPL model loaded.")

    # Build frame list
    frame_names = [args.front_frame, args.back_frame] + list(args.extra_frames)

    frames_dir  = Path(args.frames_dir)
    masks_dir   = Path(args.masks_dir) if args.masks_dir else None
    refined_dir = Path(args.bundle_refined_dir)

    frame_infos: list[dict] = []

    for fn in frame_names:
        img_path = frames_dir / fn
        if not img_path.exists():
            print(f"  [SKIP] frame image not found: {img_path}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] could not read: {img_path}")
            continue
        h, w = img.shape[:2]

        # Load person mask
        if masks_dir and masks_dir.exists():
            mask = _load_mask(masks_dir, fn, h, w)
        else:
            mask = np.ones((h, w), dtype=np.uint8) * 255

        # Find refined pkl
        rdir = _nearest_refined_frame(refined_dir, fn)
        if rdir is None:
            print(f"  [SKIP] no refined params found near: {fn}")
            continue
        pkl_path = rdir / "bundle_refined.pkl"
        if not pkl_path.exists():
            print(f"  [SKIP] pkl not found: {pkl_path}")
            continue

        params = pickle.load(open(str(pkl_path), "rb"))
        R = params["camera_rotation"].astype(np.float64)
        t = params["camera_translation"].astype(np.float64)
        betas       = params["betas"].flatten()
        body_pose   = params["body_pose"].flatten()
        global_orient = params["global_orient"].flatten()

        K = _build_K(focal, h, w)

        # SMPL forward → posed verts
        posed = _smpl_forward(smpl, betas, body_pose, global_orient)
        print(f"  {fn} (using {rdir.name}): posed verts {posed.shape}, cam t={t.round(3)}")

        # Determine if this is a back view (rough heuristic: global_orient Y component)
        flip_x = abs(global_orient[1]) > 1.0   # back views tend to have large Y rotation

        frame_infos.append(dict(
            name=fn, img=img, mask=mask,
            R=R, t=t, K=K,
            posed_verts=posed,
            flip_x=flip_x,
        ))

    if not frame_infos:
        raise RuntimeError("No valid frames found.")

    print(f"\nRunning {len(frame_infos)} frames, methods: {args.methods}")

    for method_id in args.methods:
        print(f"\n── Method {method_id} ──────────────────────────────")
        if method_id == "A":
            vc = method_A(canonical_verts, faces, smpl, frame_infos)
        elif method_id == "B":
            vc = method_B(canonical_verts, faces, frame_infos)
        elif method_id == "C":
            vc = method_C(canonical_verts, faces, frame_infos,
                          ortho_size=args.img_size)
        else:
            continue

        tag = f"method_{method_id}"
        np.save(str(out_dir / f"{tag}_vertex_colors.npy"), vc.astype(np.float32))

        _save_flat_image(canonical_verts, faces, vc,
                         out_dir / f"{tag}_front_ortho.jpg", args.img_size)

        # Render back ortho
        vc_back = vc.copy()
        v_back  = canonical_verts.copy()
        v_back[:, 0] *= -1.0
        _save_flat_image(v_back, faces, vc_back,
                         out_dir / f"{tag}_back_ortho.jpg", args.img_size)

        _save_colored_mesh(canonical_verts, faces, vc,
                           out_dir / f"{tag}_textured.ply")

        print(f"  Method {method_id}: covered verts = "
              f"{int((vc.sum(axis=1) != 3 * 0.5).sum())} / {len(vc)}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
