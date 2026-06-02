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
    YoloPosePartDetector,
    _smpl_vertex_parts,
)

OBJ_PATH = Path(r"E:/smpl_textured_from_splat.obj")
PLY_FALLBACK_PATH = Path(r"E:/V23D_Data/per_part_splat_refetch1/smpl_textured_from_splat.ply")
REF_IMG_PATH = Path(r"E:/zero123_dataset/humans_train/person_017/frame_000/reference.png")
SMPL_PKL_PATH = Path(r"E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
YOLO_MODEL_PATH = Path(r"C:/V23D/V23D/yolov8x-pose.pt")
FACE_CACHE_DIR = Path(r"C:/V23D/output/face_cache")

OUT_FACE_ONLY_PLY = Path(r"E:/smpl_textured_face_feature_guided.ply")
OUT_BODY_FIXED_PLY = Path(r"E:/smpl_textured_face_body_fixed.ply")
OUT_FACE_DET_IMG = Path(r"E:/reference_face_detection.png")
OUT_BODY_MASKS_IMG = Path(r"E:/reference_body_regions.png")
OUT_MESH_ANNOT_IMG = Path(r"E:/smpl_feature_annotations.png")
OUT_FACE_PREVIEW = Path(r"E:/smpl_textured_face_feature_guided.png")
OUT_BODY_PREVIEW = Path(r"E:/smpl_textured_face_body_fixed.png")

REGION_COLORS_BGR = {
    "face": (0, 220, 255),
    "chest": (0, 210, 0),
    "left_arm": (255, 200, 0),
    "right_arm": (0, 140, 255),
    "hair": (180, 0, 180),
}


def _load_mesh_with_base_colors() -> o3d.geometry.TriangleMesh:
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


def _load_reference() -> tuple[np.ndarray, np.ndarray]:
    ref = cv2.imread(str(REF_IMG_PATH), cv2.IMREAD_UNCHANGED)
    if ref is None:
        raise FileNotFoundError(f"Could not read reference image: {REF_IMG_PATH}")

    alpha = None
    if ref.ndim == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGRA)
    if ref.shape[2] == 4:
        alpha = ref[:, :, 3]
        ref = ref[:, :, :3]

    if alpha is not None:
        pmask = (alpha > 10).astype(np.uint8) * 255
    else:
        corners = np.concatenate(
            [
                ref[:20, :20].reshape(-1, 3),
                ref[:20, -20:].reshape(-1, 3),
                ref[-20:, :20].reshape(-1, 3),
                ref[-20:, -20:].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(corners, axis=0)
        diff = np.abs(ref.astype(np.float32) - bg[None, None, :]).sum(axis=2)
        pmask = (diff > 28.0).astype(np.uint8) * 255
        pmask = cv2.morphologyEx(pmask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        pmask = cv2.morphologyEx(pmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    return rgb, pmask


def _vertex_parts_for_mesh(verts: np.ndarray) -> np.ndarray:
    base_parts = _smpl_vertex_parts(SMPL_PKL_PATH)
    if len(base_parts) == len(verts):
        return base_parts

    base_mesh = o3d.io.read_triangle_mesh(str(PLY_FALLBACK_PATH), enable_post_processing=True)
    base_verts = np.asarray(base_mesh.vertices)
    if len(base_verts) != len(base_parts):
        raise RuntimeError("Fallback SMPL mesh/part mapping mismatch")

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


def _project_front(verts: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    sx = (width - 40) / max(float(x.max() - x.min()), 1e-6)
    sy = (height - 40) / max(float(y.max() - y.min()), 1e-6)
    s = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + width * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + height * 0.5).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height) & np.isfinite(z)
    return px, py, inside


def _standardize_face_kps(kps5: np.ndarray) -> np.ndarray:
    eyes = kps5[:2].copy()
    mouth = kps5[3:5].copy()
    eyes = eyes[np.argsort(eyes[:, 0])]
    mouth = mouth[np.argsort(mouth[:, 0])]
    nose = kps5[2:3]
    return np.vstack([eyes[0], eyes[1], nose[0], mouth[0], mouth[1]]).astype(np.float32)


def _estimate_mesh_face_features(verts: np.ndarray, px: np.ndarray, py: np.ndarray, part_ids: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.where(part_ids == PART_FACE)[0]
    if len(idx) == 0:
        raise RuntimeError("No face vertices found on mesh")

    fx = px[idx].astype(np.float32)
    fy = py[idx].astype(np.float32)
    fz = verts[idx, 2].astype(np.float32)
    x0, x1 = float(fx.min()), float(fx.max())
    y0, y1 = float(fy.min()), float(fy.max())
    cx = 0.5 * (x0 + x1)
    h = max(1.0, y1 - y0)

    def choose(mask: np.ndarray, fallback_idx: np.ndarray) -> np.ndarray:
        loc = idx[mask]
        if len(loc) == 0:
            loc = idx[fallback_idx]
        z_sel = verts[loc, 2]
        return loc[int(np.argmax(z_sel))]

    left_eye_mask = (fx < cx) & (fy >= y0 + 0.20 * h) & (fy <= y0 + 0.52 * h)
    right_eye_mask = (fx >= cx) & (fy >= y0 + 0.20 * h) & (fy <= y0 + 0.52 * h)
    nose_mask = (np.abs(fx - cx) <= 0.10 * (x1 - x0 + 1.0)) & (fy >= y0 + 0.28 * h) & (fy <= y0 + 0.68 * h)
    mouth_left_mask = (fx < cx) & (fy >= y0 + 0.58 * h) & (fy <= y0 + 0.86 * h)
    mouth_right_mask = (fx >= cx) & (fy >= y0 + 0.58 * h) & (fy <= y0 + 0.86 * h)

    left_eye_idx = choose(left_eye_mask, np.where(fx < cx)[0])
    right_eye_idx = choose(right_eye_mask, np.where(fx >= cx)[0])
    nose_idx = choose(nose_mask, np.arange(len(idx)))
    mouth_left_idx = choose(mouth_left_mask, np.where(fx < cx)[0])
    mouth_right_idx = choose(mouth_right_mask, np.where(fx >= cx)[0])

    top_idx = idx[int(np.argmin(fy))]
    chin_idx = idx[int(np.argmax(fy))]
    left_face_idx = idx[int(np.argmin(fx))]
    right_face_idx = idx[int(np.argmax(fx))]

    return {
        "left_eye": np.array([px[left_eye_idx], py[left_eye_idx]], dtype=np.float32),
        "right_eye": np.array([px[right_eye_idx], py[right_eye_idx]], dtype=np.float32),
        "nose": np.array([px[nose_idx], py[nose_idx]], dtype=np.float32),
        "mouth_left": np.array([px[mouth_left_idx], py[mouth_left_idx]], dtype=np.float32),
        "mouth_right": np.array([px[mouth_right_idx], py[mouth_right_idx]], dtype=np.float32),
        "top": np.array([px[top_idx], py[top_idx]], dtype=np.float32),
        "chin": np.array([px[chin_idx], py[chin_idx]], dtype=np.float32),
        "left_face": np.array([px[left_face_idx], py[left_face_idx]], dtype=np.float32),
        "right_face": np.array([px[right_face_idx], py[right_face_idx]], dtype=np.float32),
        "indices": {
            "left_eye": int(left_eye_idx),
            "right_eye": int(right_eye_idx),
            "nose": int(nose_idx),
            "mouth_left": int(mouth_left_idx),
            "mouth_right": int(mouth_right_idx),
            "top": int(top_idx),
            "chin": int(chin_idx),
            "left_face": int(left_face_idx),
            "right_face": int(right_face_idx),
        },
    }


def _face_mask_from_detection(det: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = det["bbox"]
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    cx = int(round(0.5 * (x1 + x2)))
    cy = int(round(y1 + 0.55 * (y2 - y1)))
    ex = max(12, int(round(0.62 * (x2 - x1))))
    ey = max(14, int(round(0.72 * (y2 - y1))))
    cv2.ellipse(mask, (cx, cy), (ex, ey), 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _make_chest_mask(torso_mask: np.ndarray, face_mask: np.ndarray, left_arm: np.ndarray, right_arm: np.ndarray) -> np.ndarray:
    ys, xs = np.where(torso_mask > 0)
    if len(xs) == 0:
        return torso_mask.copy()
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    fy, _fx = np.where(face_mask > 0)
    face_bottom = int(fy.max()) if len(fy) else y0
    chest = np.zeros_like(torso_mask)
    top = max(y0, face_bottom + 2)
    bottom = min(y1, int(round(y0 + 0.58 * (y1 - y0 + 1))))
    cx = 0.5 * (x0 + x1)
    half_w = 0.25 * (x1 - x0 + 1)
    left = max(0, int(round(cx - half_w)))
    right = min(torso_mask.shape[1] - 1, int(round(cx + half_w)))
    chest[top:bottom + 1, left:right + 1] = 255
    chest = cv2.bitwise_and(chest, torso_mask)
    chest = cv2.bitwise_and(chest, cv2.bitwise_not(left_arm))
    chest = cv2.bitwise_and(chest, cv2.bitwise_not(right_arm))
    chest = cv2.GaussianBlur(chest, (0, 0), 3)
    return (chest > 28).astype(np.uint8) * 255


def _make_hair_mask(person_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(face_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(person_mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    hair = np.zeros_like(person_mask)
    hair[max(0, y0 - int(0.55 * h)):min(person_mask.shape[0], y0 + int(0.15 * h) + 1),
         max(0, x0 - int(0.15 * w)):min(person_mask.shape[1], x1 + int(0.15 * w) + 1)] = 255
    hair = cv2.bitwise_and(hair, person_mask)
    hair = cv2.bitwise_and(hair, cv2.bitwise_not(face_mask))
    return hair


def _build_body_masks(ref_bgr: np.ndarray, person_mask: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    detector_name = "yolo"
    detector = YoloPosePartDetector(model_name=str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else 'yolov8x-pose.pt'))
    try:
        detector = HybridPartDetector(yolo_model=str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else 'yolov8x-pose.pt'))
        detector_name = "hybrid"
    except Exception:
        pass

    parts = detector.part_masks(ref_bgr, person_mask)
    return {
        "left_arm": parts[PART_LARM],
        "right_arm": parts[PART_RARM],
        "torso": parts[PART_TORSO],
    }, detector_name


def _annotate_reference(ref_bgr: np.ndarray, det: dict, masks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    face_img = ref_bgr.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
    cv2.rectangle(face_img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    labels = ["L-eye", "R-eye", "Nose", "L-mouth", "R-mouth"]
    kps = _standardize_face_kps(det["kps5"])
    for lab, pt, col in zip(labels, kps, [(255, 0, 0), (0, 180, 255), (0, 255, 0), (180, 0, 255), (255, 0, 180)]):
        p = tuple(np.round(pt).astype(int).tolist())
        cv2.circle(face_img, p, 4, col, -1, cv2.LINE_AA)
        cv2.putText(face_img, lab, (p[0] + 6, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    top = (int(round(0.5 * (x1 + x2))), int(round(y1)))
    chin = (int(round(0.5 * (x1 + x2))), int(round(y2)))
    left_face = (int(round(x1)), int(round(0.5 * (y1 + y2))))
    right_face = (int(round(x2)), int(round(0.5 * (y1 + y2))))
    for lab, p, col in [
        ("top", top, (80, 80, 255)),
        ("chin", chin, (80, 255, 80)),
        ("left", left_face, (255, 80, 80)),
        ("right", right_face, (255, 80, 220)),
    ]:
        cv2.circle(face_img, p, 4, col, -1, cv2.LINE_AA)
        cv2.putText(face_img, lab, (p[0] + 6, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    body_img = ref_bgr.copy()
    overlay = body_img.copy()
    for name in ["chest", "left_arm", "right_arm", "hair", "face"]:
        m = masks[name] > 0
        overlay[m] = REGION_COLORS_BGR[name]
    body_img = cv2.addWeighted(overlay, 0.42, body_img, 0.58, 0.0)
    for name in ["face", "chest", "left_arm", "right_arm", "hair"]:
        cnts, _ = cv2.findContours((masks[name] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(body_img, cnts, -1, REGION_COLORS_BGR[name], 2, cv2.LINE_AA)
    return face_img, body_img


def _annotate_mesh_projection(verts: np.ndarray, px: np.ndarray, py: np.ndarray, inside: np.ndarray, part_ids: np.ndarray, feature_pts: dict[str, np.ndarray]) -> np.ndarray:
    h = int(py[inside].max() + 20) if np.any(inside) else 1024
    w = int(px[inside].max() + 20) if np.any(inside) else 1024
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    valid_idx = np.where(inside)[0]
    canvas[np.clip(py[valid_idx], 0, h - 1), np.clip(px[valid_idx], 0, w - 1)] = (220, 220, 220)

    region_map = {
        "face": part_ids == PART_FACE,
        "left_arm": part_ids == PART_LARM,
        "right_arm": part_ids == PART_RARM,
        "chest": part_ids == PART_TORSO,
    }
    for name, mask in region_map.items():
        idx = np.where(mask & inside)[0]
        for ii in idx[::max(1, len(idx) // 5000 + 1)]:
            cv2.circle(canvas, (int(px[ii]), int(py[ii])), 1, REGION_COLORS_BGR.get(name, (100, 100, 100)), -1, cv2.LINE_AA)

    feat_cols = {
        "left_eye": (255, 0, 0),
        "right_eye": (0, 180, 255),
        "nose": (0, 255, 0),
        "mouth_left": (180, 0, 255),
        "mouth_right": (255, 0, 180),
        "top": (80, 80, 255),
        "chin": (80, 255, 80),
        "left_face": (255, 80, 80),
        "right_face": (255, 80, 220),
    }
    for name, pt in feature_pts.items():
        if name == "indices":
            continue
        p = tuple(np.round(pt).astype(int).tolist())
        cv2.circle(canvas, p, 5, feat_cols[name], -1, cv2.LINE_AA)
        cv2.putText(canvas, name, (p[0] + 6, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, feat_cols[name], 1, cv2.LINE_AA)
    return canvas


def _sample_affine_face(img_rgb: np.ndarray, inv_aff: np.ndarray, qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homog = np.concatenate([qxy.astype(np.float32), np.ones((len(qxy), 1), dtype=np.float32)], axis=1)
    src = homog @ inv_aff.T
    sx = src[:, 0]
    sy = src[:, 1]
    h, w = img_rgb.shape[:2]
    valid = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
    colors = np.zeros((len(qxy), 3), dtype=np.float32)
    if np.any(valid):
        map_x = sx[valid].reshape(-1, 1).astype(np.float32)
        map_y = sy[valid].reshape(-1, 1).astype(np.float32)
        sampled = cv2.remap(img_rgb.astype(np.float32) / 255.0, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        colors[valid] = sampled.reshape(-1, 3)
    return colors, valid


def _sample_region_color(img_rgb: np.ndarray, mask: np.ndarray, qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    qx = np.clip(qxy[:, 0].astype(np.int32), 0, w - 1)
    qy = np.clip(qxy[:, 1].astype(np.int32), 0, h - 1)
    direct_valid = mask[qy, qx] > 0
    colors = np.zeros((len(qxy), 3), dtype=np.float32)
    if np.any(direct_valid):
        colors[direct_valid] = img_rgb[qy[direct_valid], qx[direct_valid]].astype(np.float32) / 255.0

    nearest_colors, near_valid = _nearest_masked_color(img_rgb, mask, qxy)
    fill = (~direct_valid) & near_valid
    colors[fill] = nearest_colors[fill]
    valid = direct_valid | fill

    lum_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lum = np.zeros((len(qxy),), dtype=np.float32)
    lum[valid] = lum_img[qy[valid], qx[valid]]
    return colors, valid, lum


def _nearest_masked_color(img_rgb: np.ndarray, mask: np.ndarray, qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.zeros((len(qxy), 3), dtype=np.float32), np.zeros((len(qxy),), dtype=bool)
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
    valid = np.asarray(dist) <= 42.0
    return colors, valid


def _shade(colors: np.ndarray, normals: np.ndarray, strength: float = 0.28) -> np.ndarray:
    light_dir = np.array([0.12, -0.30, 0.95], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    ndotl = np.clip(normals @ light_dir, 0.0, 1.0)
    return np.clip(colors * ((1.0 - strength) + strength * ndotl[:, None] + 0.25), 0.0, 1.0)


def _apply_region_shading(
    sampled: np.ndarray,
    luminance: np.ndarray,
    normals: np.ndarray,
    strength: float = 0.28,
    lum_boost: float = 0.40,
) -> np.ndarray:
    shaded = _shade(sampled, normals, strength=strength)
    valid_l = luminance > 0
    if np.any(valid_l):
        mu = float(np.mean(luminance[valid_l]))
        sigma = float(np.std(luminance[valid_l]))
        sigma = max(sigma, 0.06)
        rel = np.clip((luminance - mu) / (2.2 * sigma), -1.0, 1.0)
        factor = (1.0 + lum_boost * rel)[:, None]
        shaded = np.clip(shaded * factor, 0.0, 1.0)
    return shaded


def _apply_face_only(mesh: o3d.geometry.TriangleMesh, img_rgb: np.ndarray, det: dict, px: np.ndarray, py: np.ndarray, inside: np.ndarray, part_ids: np.ndarray, mesh_feats: dict[str, np.ndarray]) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(mesh)
    out.compute_vertex_normals()
    colors = np.asarray(out.vertex_colors).copy()
    normals = np.asarray(out.vertex_normals)
    face_mask = part_ids == PART_FACE
    face_idx = np.where(face_mask & inside)[0]
    qxy = np.stack([px[face_idx], py[face_idx]], axis=1).astype(np.float32)

    img_pts = _standardize_face_kps(det["kps5"])
    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
    img_pts = np.vstack([
        img_pts,
        np.array([
            [0.5 * (x1 + x2), y1],
            [0.5 * (x1 + x2), y2],
            [x1, 0.5 * (y1 + y2)],
            [x2, 0.5 * (y1 + y2)],
        ], dtype=np.float32),
    ])
    mesh_pts = np.vstack([
        mesh_feats["left_eye"],
        mesh_feats["right_eye"],
        mesh_feats["nose"],
        mesh_feats["mouth_left"],
        mesh_feats["mouth_right"],
        mesh_feats["top"],
        mesh_feats["chin"],
        mesh_feats["left_face"],
        mesh_feats["right_face"],
    ]).astype(np.float32)
    aff, _inliers = cv2.estimateAffine2D(img_pts, mesh_pts, method=cv2.LMEDS)
    if aff is None:
        aff, _inliers = cv2.estimateAffinePartial2D(img_pts, mesh_pts, method=cv2.LMEDS)
    if aff is None:
        raise RuntimeError("Could not estimate face affine transform")

    inv_aff = cv2.invertAffineTransform(aff)
    sampled, valid = _sample_affine_face(img_rgb, inv_aff, qxy)
    qx = np.clip(qxy[:, 0].astype(np.int32), 0, img_rgb.shape[1] - 1)
    qy = np.clip(qxy[:, 1].astype(np.int32), 0, img_rgb.shape[0] - 1)
    face_lum = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lum = np.zeros((len(qxy),), dtype=np.float32)
    lum[valid] = face_lum[qy[valid], qx[valid]]
    shaded = _apply_region_shading(sampled, lum, normals[face_idx], strength=0.14, lum_boost=0.32)
    alpha = 0.95
    sel = face_idx[valid]
    colors[sel] = colors[sel] * (1.0 - alpha) + shaded[valid] * alpha
    out.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return out


def _apply_body_fix(mesh: o3d.geometry.TriangleMesh, img_rgb: np.ndarray, masks: dict[str, np.ndarray], px: np.ndarray, py: np.ndarray, inside: np.ndarray, part_ids: np.ndarray) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(mesh)
    out.compute_vertex_normals()
    verts = np.asarray(out.vertices)
    normals = np.asarray(out.vertex_normals)
    colors = np.asarray(out.vertex_colors).copy()
    qxy = np.stack([np.clip(px, 0, img_rgb.shape[1] - 1), np.clip(py, 0, img_rgb.shape[0] - 1)], axis=1)

    torso = np.where(part_ids == PART_TORSO)[0]
    chest_mask_mesh = np.zeros(len(verts), dtype=bool)
    if len(torso) > 0:
        y = verts[torso, 1]
        ax = np.abs(verts[torso, 0])
        chest_mask_mesh[torso] = (y >= np.quantile(y, 0.50)) & (y <= np.quantile(y, 0.84)) & (ax <= np.quantile(ax, 0.70))

    region_mesh = {
        "chest": chest_mask_mesh,
        "left_arm": part_ids == PART_LARM,
        "right_arm": part_ids == PART_RARM,
        "hair": (part_ids == PART_FACE) & (verts[:, 1] >= np.quantile(verts[part_ids == PART_FACE, 1], 0.72) if np.any(part_ids == PART_FACE) else False),
    }
    alphas = {"chest": 0.90, "left_arm": 0.88, "right_arm": 0.88, "hair": 0.92}

    region_params = {
        "chest": {"alpha": 0.93, "strength": 0.18, "lum_boost": 0.58},
        "left_arm": {"alpha": 0.91, "strength": 0.20, "lum_boost": 0.52},
        "right_arm": {"alpha": 0.91, "strength": 0.20, "lum_boost": 0.52},
        "hair": {"alpha": 0.95, "strength": 0.10, "lum_boost": 0.35},
    }

    for name in ["chest", "left_arm", "right_arm", "hair"]:
        sampled, valid_img, lum = _sample_region_color(img_rgb, masks[name], qxy)
        valid = region_mesh[name] & inside & valid_img
        if not np.any(valid):
            continue
        prm = region_params[name]
        shaded = _apply_region_shading(sampled, lum, normals, strength=prm["strength"], lum_boost=prm["lum_boost"])
        a = prm["alpha"]
        if name == "hair":
            shaded[valid] = np.clip(shaded[valid] * np.array([0.72, 0.70, 0.68], dtype=np.float32), 0.0, 1.0)
        colors[valid] = colors[valid] * (1.0 - a) + shaded[valid] * a
    out.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return out


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
    mesh = _load_mesh_with_base_colors()
    verts = np.asarray(mesh.vertices)
    ref_rgb, person_mask = _load_reference()
    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
    part_ids = _vertex_parts_for_mesh(verts)
    px, py, inside = _project_front(verts, ref_rgb.shape[1], ref_rgb.shape[0])

    face_det = FaceDetectorYuNet(FACE_CACHE_DIR)
    det = face_det.best_detection(ref_bgr)
    if det is None:
        raise RuntimeError("Face detection failed on reference image")

    body_parts, detector_name = _build_body_masks(ref_bgr, person_mask)
    face_mask = _face_mask_from_detection(det, person_mask.shape)
    masks = {
        "face": cv2.bitwise_and(face_mask, person_mask),
        "left_arm": body_parts["left_arm"],
        "right_arm": body_parts["right_arm"],
    }
    masks["chest"] = _make_chest_mask(body_parts["torso"], masks["face"], masks["left_arm"], masks["right_arm"])
    masks["hair"] = _make_hair_mask(person_mask, masks["face"])

    face_ann, body_ann = _annotate_reference(ref_bgr, det, masks)
    cv2.imwrite(str(OUT_FACE_DET_IMG), face_ann)
    cv2.imwrite(str(OUT_BODY_MASKS_IMG), body_ann)

    mesh_feats = _estimate_mesh_face_features(verts, px, py, part_ids)
    mesh_ann = _annotate_mesh_projection(verts, px, py, inside, part_ids, mesh_feats)
    cv2.imwrite(str(OUT_MESH_ANNOT_IMG), mesh_ann)

    face_mesh = _apply_face_only(mesh, ref_rgb, det, px, py, inside, part_ids, mesh_feats)
    o3d.io.write_triangle_mesh(str(OUT_FACE_ONLY_PLY), face_mesh, write_vertex_colors=True)
    _save_preview(face_mesh, OUT_FACE_PREVIEW)

    body_mesh = _apply_body_fix(face_mesh, ref_rgb, masks, px, py, inside, part_ids)
    o3d.io.write_triangle_mesh(str(OUT_BODY_FIXED_PLY), body_mesh, write_vertex_colors=True)
    _save_preview(body_mesh, OUT_BODY_PREVIEW)

    print(f"Face detection saved: {OUT_FACE_DET_IMG}")
    print(f"Body regions ({detector_name}) saved: {OUT_BODY_MASKS_IMG}")
    print(f"Mesh annotations saved: {OUT_MESH_ANNOT_IMG}")
    print(f"Face-only mesh saved: {OUT_FACE_ONLY_PLY}")
    print(f"Face+body fixed mesh saved: {OUT_BODY_FIXED_PLY}")


if __name__ == "__main__":
    main()
