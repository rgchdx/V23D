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
    HybridPartDetector,
    PART_FACE,
    PART_LARM,
    PART_NAMES,
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

OUT_YOLO_PLY = Path(r"E:/smpl_textured_from_splat_detected_yolo.ply")
OUT_HYBRID_PLY = Path(r"E:/smpl_textured_from_splat_detected_hybrid.ply")
PREVIEW_YOLO = Path(r"E:/smpl_textured_from_splat_detected_yolo.png")
PREVIEW_HYBRID = Path(r"E:/smpl_textured_from_splat_detected_hybrid.png")


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

    h, w = ref.shape[:2]
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
    if pmask.shape[:2] != (h, w):
        pmask = cv2.resize(pmask, (w, h), interpolation=cv2.INTER_NEAREST)
    return rgb, pmask


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


def _nearest_masked_color(img_rgb: np.ndarray, mask: np.ndarray, qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.zeros((len(qxy), 3), dtype=np.float32), np.zeros((len(qxy),), dtype=bool)

    xy = coords[:, [1, 0]].astype(np.float32)
    qxy = qxy.astype(np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
        dist, idx = tree.query(qxy, k=1)
    except Exception:
        idx = []
        dist = []
        for q in qxy:
            d2 = np.sum((xy - q[None, :]) ** 2, axis=1)
            i = int(np.argmin(d2))
            idx.append(i)
            dist.append(float(np.sqrt(d2[i])))
        idx = np.asarray(idx, dtype=np.int32)
        dist = np.asarray(dist, dtype=np.float32)

    nearest = coords[np.asarray(idx).reshape(-1)]
    colors = img_rgb[nearest[:, 0], nearest[:, 1]].astype(np.float32) / 255.0
    valid = np.asarray(dist) <= 42.0
    return colors, valid


def _make_chest_mask(torso_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(torso_mask > 0)
    if len(xs) == 0:
        return torso_mask.copy()
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    chest = np.zeros_like(torso_mask)

    fy, fx = np.where(face_mask > 0)
    face_bottom = int(fy.max()) if len(fy) else y0
    top = max(y0, face_bottom)
    bottom = int(round(y0 + 0.58 * (y1 - y0 + 1)))
    cx = 0.5 * (x0 + x1)
    half_w = 0.28 * (x1 - x0 + 1)
    left = max(0, int(round(cx - half_w)))
    right = min(torso_mask.shape[1] - 1, int(round(cx + half_w)))
    chest[top:bottom + 1, left:right + 1] = 255
    chest = cv2.bitwise_and(chest, torso_mask)
    chest = cv2.GaussianBlur(chest, (0, 0), 3)
    return (chest > 32).astype(np.uint8) * 255


def _make_hair_mask(person_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(face_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(person_mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    hair = np.zeros_like(person_mask)
    hx0 = max(0, x0 - int(0.18 * w))
    hx1 = min(person_mask.shape[1] - 1, x1 + int(0.18 * w))
    hy0 = max(0, y0 - int(0.55 * h))
    hy1 = min(person_mask.shape[0] - 1, y0 + int(0.20 * h))
    hair[hy0:hy1 + 1, hx0:hx1 + 1] = 255
    hair = cv2.bitwise_and(hair, person_mask)
    hair = cv2.bitwise_and(hair, cv2.bitwise_not(face_mask))
    return hair


def _build_reference_part_masks(detector, ref_bgr: np.ndarray, person_mask: np.ndarray) -> dict[str, np.ndarray]:
    parts = detector.part_masks(ref_bgr, person_mask)
    face = parts[PART_FACE]
    left_arm = parts[PART_LARM]
    right_arm = parts[PART_RARM]
    chest = _make_chest_mask(parts[PART_TORSO], face)
    hair = _make_hair_mask(person_mask, face)
    return {
        "face": face,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "chest": chest,
        "hair": hair,
    }


def _shade(colors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    light_dir = np.array([0.10, -0.32, 0.94], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    ndotl = np.clip(normals @ light_dir, 0.0, 1.0)
    return np.clip(colors * (0.70 + 0.30 * ndotl[:, None]), 0.0, 1.0)


def _mesh_region_masks(verts: np.ndarray, part_ids: np.ndarray, sampled_mean: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["left_arm"] = part_ids == PART_LARM
    out["right_arm"] = part_ids == PART_RARM

    torso = np.where(part_ids == PART_TORSO)[0]
    chest = np.zeros(len(verts), dtype=bool)
    if len(torso) > 0:
        y = verts[torso, 1]
        ax = np.abs(verts[torso, 0])
        chest[torso] = (y >= np.quantile(y, 0.48)) & (y <= np.quantile(y, 0.86)) & (ax <= np.quantile(ax, 0.72))
    out["chest"] = chest

    face_idx = np.where(part_ids == PART_FACE)[0]
    face = np.zeros(len(verts), dtype=bool)
    hair = np.zeros(len(verts), dtype=bool)
    if len(face_idx) > 0:
        y = verts[face_idx, 1]
        face[face_idx] = y <= np.quantile(y, 0.82)
        hair[face_idx] = (y > np.quantile(y, 0.72)) & (sampled_mean[face_idx] < np.quantile(sampled_mean[face_idx], 0.6))
    out["face"] = face
    out["hair"] = hair
    return out


def _apply_detected_projection(mesh: o3d.geometry.TriangleMesh, img_rgb: np.ndarray, masks_img: dict[str, np.ndarray]) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(mesh)
    out.compute_vertex_normals()
    verts = np.asarray(out.vertices)
    normals = np.asarray(out.vertex_normals)
    base_colors = np.asarray(out.vertex_colors).copy()
    part_ids = _vertex_parts_for_mesh(verts)

    h, w = img_rgb.shape[:2]
    px, py, inside = _project_front(verts, w, h)
    qxy = np.stack([np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)], axis=1)
    sampled_mean = img_rgb[qxy[:, 1], qxy[:, 0]].astype(np.float32).mean(axis=1) / 255.0
    mesh_masks = _mesh_region_masks(verts, part_ids, sampled_mean)

    final = base_colors.copy()
    alpha = np.zeros((len(verts), 1), dtype=np.float32)

    region_order = ["chest", "left_arm", "right_arm", "face", "hair"]
    region_alpha = {
        "chest": 0.88,
        "left_arm": 0.90,
        "right_arm": 0.90,
        "face": 0.92,
        "hair": 0.95,
    }

    for name in region_order:
        colors, valid_img = _nearest_masked_color(img_rgb, masks_img[name], qxy)
        valid = mesh_masks[name] & inside & valid_img
        if not np.any(valid):
            continue
        shaded = _shade(colors, normals)
        a = region_alpha[name]
        final[valid] = final[valid] * (1.0 - a) + shaded[valid] * a
        alpha[valid] = np.maximum(alpha[valid], a)

    out.vertex_colors = o3d.utility.Vector3dVector(np.clip(final, 0.0, 1.0))
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


def _mask_stats(name: str, masks: dict[str, np.ndarray]) -> str:
    bits = [f"{k}={(masks[k] > 0).sum()}" for k in ["face", "chest", "left_arm", "right_arm", "hair"]]
    return f"{name}: " + ", ".join(bits)


def main() -> None:
    mesh = _load_mesh_with_base_colors()
    ref_rgb, person_mask = _load_reference()
    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

    yolo_detector = YoloPosePartDetector(model_name=str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else 'yolov8x-pose.pt'))
    yolo_masks = _build_reference_part_masks(yolo_detector, ref_bgr, person_mask)
    yolo_mesh = _apply_detected_projection(mesh, ref_rgb, yolo_masks)
    o3d.io.write_triangle_mesh(str(OUT_YOLO_PLY), yolo_mesh, write_vertex_colors=True)
    _save_preview(yolo_mesh, PREVIEW_YOLO)
    print(_mask_stats("YOLO", yolo_masks))
    print(f"Saved: {OUT_YOLO_PLY}")

    try:
        hybrid_detector = HybridPartDetector(
            yolo_model=str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else 'yolov8x-pose.pt')
        )
        hybrid_masks = _build_reference_part_masks(hybrid_detector, ref_bgr, person_mask)
        hybrid_mesh = _apply_detected_projection(mesh, ref_rgb, hybrid_masks)
        o3d.io.write_triangle_mesh(str(OUT_HYBRID_PLY), hybrid_mesh, write_vertex_colors=True)
        _save_preview(hybrid_mesh, PREVIEW_HYBRID)
        print(_mask_stats("Hybrid", hybrid_masks))
        print(f"Saved: {OUT_HYBRID_PLY}")
    except Exception as exc:
        print(f"Hybrid detector failed: {exc}")


if __name__ == "__main__":
    main()
