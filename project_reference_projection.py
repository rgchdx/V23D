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
    PART_FACE,
    PART_LARM,
    PART_RARM,
    PART_TORSO,
    _smpl_vertex_parts,
)


OBJ_PATH = Path(r"E:/smpl_textured_from_splat.obj")
PLY_FALLBACK_PATH = Path(r"E:/V23D_Data/per_part_splat_refetch1/smpl_textured_from_splat.ply")
REF_IMG_PATH = Path(r"E:/zero123_dataset/humans_train/person_017/frame_000/reference.png")
SMPL_PKL_PATH = Path(r"E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
OUT_PARTS_PLY = Path(r"E:/smpl_textured_from_splat_projected_parts.ply")
OUT_HEURISTIC_PLY = Path(r"E:/smpl_textured_from_splat_projected_heuristic.ply")
PREVIEW_PARTS = Path(r"E:/smpl_textured_from_splat_projected_parts.png")
PREVIEW_HEURISTIC = Path(r"E:/smpl_textured_from_splat_projected_heuristic.png")


def _load_mesh_with_base_colors() -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(OBJ_PATH), enable_post_processing=True)
    if not mesh.has_triangles():
        raise RuntimeError(f"Could not load mesh: {OBJ_PATH}")

    if not mesh.has_vertex_colors() and PLY_FALLBACK_PATH.exists():
        mesh_ply = o3d.io.read_triangle_mesh(str(PLY_FALLBACK_PATH), enable_post_processing=True)
        if mesh_ply.has_vertex_colors() and len(mesh_ply.vertices) == len(mesh.vertices):
            mesh.vertex_colors = mesh_ply.vertex_colors

    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color((0.62, 0.62, 0.62))

    mesh.compute_vertex_normals()
    return mesh


def _load_reference() -> tuple[np.ndarray, np.ndarray]:
    ref = cv2.imread(str(REF_IMG_PATH), cv2.IMREAD_UNCHANGED)
    if ref is None:
        raise FileNotFoundError(f"Could not read reference image: {REF_IMG_PATH}")

    if ref.ndim == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    alpha = None
    if ref.shape[2] == 4:
        alpha = ref[:, :, 3]
        ref = ref[:, :, :3]

    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    h, w = ref.shape[:2]

    if alpha is not None:
        fg = alpha > 10
    else:
        corners = np.stack([
            ref[:20, :20].reshape(-1, 3),
            ref[:20, -20:].reshape(-1, 3),
            ref[-20:, :20].reshape(-1, 3),
            ref[-20:, -20:].reshape(-1, 3),
        ], axis=0).reshape(-1, 3)
        bg = np.median(corners, axis=0)
        diff = np.abs(ref.astype(np.float32) - bg[None, None, :]).sum(axis=2)
        fg = diff > 28.0
        fg = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0

    return ref, fg.astype(bool)


def _project_front(verts: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    sx = (width - 40) / max(float(x.max() - x.min()), 1e-6)
    sy = (height - 40) / max(float(y.max() - y.min()), 1e-6)
    s = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + width * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + height * 0.5).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return px, py, inside & np.isfinite(z)


def _shade(colors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    light_dir = np.array([0.15, -0.35, 0.92], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    ndotl = np.clip(normals @ light_dir, 0.0, 1.0)
    shade = 0.68 + 0.32 * ndotl[:, None]
    out = np.clip(colors * shade, 0.0, 1.0)
    return out


def _hair_mask_from_parts(verts: np.ndarray, part_ids: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    face_idx = np.where(part_ids == PART_FACE)[0]
    if len(face_idx) == 0:
        return np.zeros(len(verts), dtype=bool)
    y_face = verts[face_idx, 1]
    y_thr = np.quantile(y_face, 0.72)
    bright = sampled[face_idx].mean(axis=1)
    dark_thr = min(0.38, float(np.quantile(bright, 0.55)))
    hair = np.zeros(len(verts), dtype=bool)
    hair[face_idx] = (verts[face_idx, 1] >= y_thr) & (bright <= dark_thr)
    return hair


def _vertex_parts_for_mesh(verts: np.ndarray) -> np.ndarray:
    base_parts = _smpl_vertex_parts(SMPL_PKL_PATH)
    if len(base_parts) == len(verts):
        return base_parts

    if not PLY_FALLBACK_PATH.exists():
        raise RuntimeError(f"Part label count mismatch: {len(base_parts)} vs {len(verts)} and fallback PLY missing")

    base_mesh = o3d.io.read_triangle_mesh(str(PLY_FALLBACK_PATH), enable_post_processing=True)
    base_verts = np.asarray(base_mesh.vertices)
    if len(base_verts) != len(base_parts):
        raise RuntimeError(
            f"Fallback PLY vertex count mismatch: base mesh {len(base_verts)} vs part ids {len(base_parts)}"
        )

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


def _target_mask_parts(verts: np.ndarray, part_ids: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    torso = np.where(part_ids == PART_TORSO)[0]
    chest = np.zeros(len(verts), dtype=bool)
    if len(torso) > 0:
        y_t = verts[torso, 1]
        x_t = np.abs(verts[torso, 0])
        chest[torso] = (y_t >= np.quantile(y_t, 0.45)) & (y_t <= np.quantile(y_t, 0.86)) & (x_t <= np.quantile(x_t, 0.75))

    arms = (part_ids == PART_LARM) | (part_ids == PART_RARM)
    hair = _hair_mask_from_parts(verts, part_ids, sampled)
    return chest | arms | hair


def _target_mask_heuristic(verts: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    vn = (verts - mins) / span
    x = vn[:, 0] - 0.5
    y = vn[:, 1]
    z = vn[:, 2]
    chest = (np.abs(x) < 0.18) & (y > 0.50) & (y < 0.74) & (z > 0.42)
    arms = (np.abs(x) >= 0.18) & (np.abs(x) < 0.42) & (y > 0.42) & (y < 0.80)
    hair = (np.abs(x) < 0.14) & (y > 0.80) & (sampled.mean(axis=1) < 0.42)
    return chest | arms | hair


def _apply_projection(mesh: o3d.geometry.TriangleMesh, target_mask: np.ndarray, sampled: np.ndarray, valid: np.ndarray) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(mesh)
    out.compute_vertex_normals()
    verts = np.asarray(out.vertices)
    normals = np.asarray(out.vertex_normals)
    base = np.asarray(out.vertex_colors).copy()

    shaded = _shade(sampled, normals)
    alpha = np.full((len(verts), 1), 0.0, dtype=np.float32)
    alpha[target_mask & valid] = 0.88

    # Slightly stronger blend for hair so it picks up darker projected appearance.
    hair_like = target_mask & (verts[:, 1] >= np.quantile(verts[:, 1], 0.82))
    alpha[hair_like & valid] = 0.94

    merged = base * (1.0 - alpha) + shaded * alpha
    out.vertex_colors = o3d.utility.Vector3dVector(np.clip(merged, 0.0, 1.0))
    return out


def _save_preview(mesh: o3d.geometry.TriangleMesh, out_path: Path, title: str) -> None:
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
    ctr.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    print(f"Saved preview: {out_path} ({title})")


def main() -> None:
    mesh = _load_mesh_with_base_colors()
    verts = np.asarray(mesh.vertices)
    ref_img, fg_mask = _load_reference()
    h, w = ref_img.shape[:2]
    px, py, inside = _project_front(verts, w, h)

    sampled = ref_img[np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)].astype(np.float32) / 255.0
    valid = inside & fg_mask[np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)]

    part_ids = _vertex_parts_for_mesh(verts)

    target_parts = _target_mask_parts(verts, part_ids, sampled)
    mesh_parts = _apply_projection(mesh, target_parts, sampled, valid)
    o3d.io.write_triangle_mesh(str(OUT_PARTS_PLY), mesh_parts, write_vertex_colors=True)
    _save_preview(mesh_parts, PREVIEW_PARTS, "parts")

    target_heur = _target_mask_heuristic(verts, sampled)
    mesh_heur = _apply_projection(mesh, target_heur, sampled, valid)
    o3d.io.write_triangle_mesh(str(OUT_HEURISTIC_PLY), mesh_heur, write_vertex_colors=True)
    _save_preview(mesh_heur, PREVIEW_HEURISTIC, "heuristic")

    print(f"Saved: {OUT_PARTS_PLY}")
    print(f"Saved: {OUT_HEURISTIC_PLY}")
    print(f"Projected vertices (parts): {int((target_parts & valid).sum())}")
    print(f"Projected vertices (heuristic): {int((target_heur & valid).sum())}")


if __name__ == "__main__":
    main()
