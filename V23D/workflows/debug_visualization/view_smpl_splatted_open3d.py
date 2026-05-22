from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
            faces.append(tri)
    return np.asarray(verts, np.float64), np.asarray(faces, np.int32)


def _project_xy(verts: np.ndarray, image_wh: int) -> tuple[np.ndarray, np.ndarray]:
    x = verts[:, 0]
    y = verts[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    sx = (image_wh - 40) / max(xmax - xmin, 1e-6)
    sy = (image_wh - 40) / max(ymax - ymin, 1e-6)
    s = min(sx, sy)

    px = ((x - (xmin + xmax) * 0.5) * s + image_wh * 0.5).astype(np.int32)
    py = ((-(y - (ymin + ymax) * 0.5)) * s + image_wh * 0.5).astype(np.int32)
    return px, py


def _sample(img_bgr: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    col = img_bgr[py, px, :].astype(np.float32) / 255.0
    return col[:, ::-1]  # RGB


def _build_vertex_colors(verts: np.ndarray, front_bgr: np.ndarray, back_bgr: np.ndarray) -> np.ndarray:
    im_wh = front_bgr.shape[0]

    # front projection
    pxf, pyf = _project_xy(verts, im_wh)
    cf = _sample(front_bgr, pxf, pyf)

    # back projection (x mirrored, same as exporter)
    vb = verts.copy()
    vb[:, 0] *= -1.0
    pxb, pyb = _project_xy(vb, im_wh)
    cb = _sample(back_bgr, pxb, pyb)

    # non-black masks
    mf = (cf.sum(axis=1) > 0.03)
    mb = (cb.sum(axis=1) > 0.03)

    # Use both sides when available; otherwise fallback to whichever exists.
    out = np.zeros_like(cf)
    both = mf & mb
    out[both] = 0.5 * cf[both] + 0.5 * cb[both]
    only_f = mf & (~mb)
    only_b = mb & (~mf)
    out[only_f] = cf[only_f]
    out[only_b] = cb[only_b]

    # Fill missing colors with neutral gray.
    miss = ~(mf | mb)
    out[miss] = np.array([0.55, 0.55, 0.55], dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="View SMPL mesh with splatted colors in Open3D")
    ap.add_argument("--smpl-obj",         default="")
    ap.add_argument("--front-splatted",   default="")
    ap.add_argument("--back-splatted",    default="")
    # New: load vertex colors directly from a .npy or colored .ply
    ap.add_argument("--vertex-colors-npy", default="",
                    help="(N,3) float32 RGB numpy file from texture_smpl_from_frames.py")
    ap.add_argument("--ply",              default="",
                    help="Colored PLY from texture_smpl_from_frames.py (open3d read)")
    args = ap.parse_args()

    # ── Mode 1: load colored PLY directly ──────────────────────────────────
    if args.ply:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(args.ply)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"SMPL Textured – {Path(args.ply).name}",
            width=1280, height=900,
            mesh_show_back_face=True,
        )
        return

    # ── Mode 2: smpl-obj + vertex-colors.npy ───────────────────────────────
    if args.vertex_colors_npy and args.smpl_obj:
        v, f = _load_obj(Path(args.smpl_obj))
        vc = np.load(args.vertex_colors_npy).astype(np.float64)
        vc = np.clip(vc, 0, 1)
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(f)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"SMPL Textured – {Path(args.vertex_colors_npy).name}",
            width=1280, height=900,
            mesh_show_back_face=True,
        )
        return

    # ── Mode 3 (legacy): smpl-obj + front/back splatted images ─────────────
    if not args.smpl_obj or not args.front_splatted or not args.back_splatted:
        raise RuntimeError("Provide --ply, or --smpl-obj+--vertex-colors-npy, "
                           "or --smpl-obj+--front-splatted+--back-splatted")

    v, f = _load_obj(Path(args.smpl_obj))
    front = cv2.imread(str(Path(args.front_splatted)), cv2.IMREAD_COLOR)
    back = cv2.imread(str(Path(args.back_splatted)), cv2.IMREAD_COLOR)
    if front is None or back is None:
        raise RuntimeError("Could not read splatted images")
    if front.shape[0] != back.shape[0] or front.shape[1] != back.shape[1]:
        raise RuntimeError("Front/back splatted image size mismatch")

    vc = _build_vertex_colors(v, front, back)

    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="SMPL Splatted (Open3D)",
        width=1280,
        height=900,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
