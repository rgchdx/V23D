from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(tri)
    if not verts or not faces:
        raise RuntimeError(f"Invalid mesh: {path}")
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64)


def save_ply_vertex_colors(path: Path, verts: np.ndarray, faces: np.ndarray, colors_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(len(verts)):
            v = verts[i]
            c = colors_u8[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    return n.astype(np.float32)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    vn = np.zeros_like(verts, dtype=np.float32)
    for i in range(3):
        np.add.at(vn, faces[:, i], face_normals)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
    return vn


def project_points(pts: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = (r @ pts.T).T + t
    z = cam[:, 2]
    fx, cx, cy = float(k[0, 0]), float(k[0, 2]), float(k[1, 2])
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + cy
    return np.stack([u, v], axis=1).astype(np.float32), z.astype(np.float32)


def rasterize_depth(verts: np.ndarray, faces: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray, w: int, h: int) -> np.ndarray:
    cam = (r @ verts.T).T + t
    z = cam[:, 2]
    fx, cx, cy = float(k[0, 0]), float(k[0, 2]), float(k[1, 2])
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + cy

    depth = np.full((h, w), np.inf, dtype=np.float32)
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    z0, z1, z2 = z[i0], z[i1], z[i2]
    x0, x1, x2 = u[i0], u[i1], u[i2]
    y0, y1, y2 = v[i0], v[i1], v[i2]
    front = (z0 > 1e-3) & (z1 > 1e-3) & (z2 > 1e-3)

    for fi in np.where(front)[0]:
        _x0, _y0, _z0 = float(x0[fi]), float(y0[fi]), float(z0[fi])
        _x1, _y1, _z1 = float(x1[fi]), float(y1[fi]), float(z1[fi])
        _x2, _y2, _z2 = float(x2[fi]), float(y2[fi]), float(z2[fi])

        xmin = max(0, int(min(_x0, _x1, _x2)))
        xmax = min(w - 1, int(max(_x0, _x1, _x2)) + 1)
        ymin = max(0, int(min(_y0, _y1, _y2)))
        ymax = min(h - 1, int(max(_y0, _y1, _y2)) + 1)
        if xmin > xmax or ymin > ymax:
            continue

        area = (_x1 - _x0) * (_y2 - _y0) - (_x2 - _x0) * (_y1 - _y0)
        if abs(area) < 1e-8:
            continue
        inv = 1.0 / area

        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1, dtype=np.float32), np.arange(ymin, ymax + 1, dtype=np.float32))
        w0 = ((gx - _x1) * (_y2 - _y1) - (gy - _y1) * (_x2 - _x1)) * inv
        w1 = ((gx - _x2) * (_y0 - _y2) - (gy - _y2) * (_x0 - _x2)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z_interp = (w0 * _z0 + w1 * _z1 + w2 * _z2).astype(np.float32)
        rows = gy.astype(np.int32)[inside]
        cols = gx.astype(np.int32)[inside]
        vals = z_interp[inside]

        flat = rows * w + cols
        order = np.argsort(flat)
        flat = flat[order]
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]

        uniq, first = np.unique(flat, return_index=True)
        for idx in first:
            rr, cc, zz = int(rows[idx]), int(cols[idx]), float(vals[idx])
            if zz < depth[rr, cc]:
                depth[rr, cc] = zz

    return depth


def load_mask(masks_dir: Path | None, frame_name: str, w: int, h: int) -> np.ndarray | None:
    if masks_dir is None:
        return None
    stem = Path(frame_name).stem
    for cand in [stem, stem.replace("frame_", "")]:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = masks_dir / f"{cand}{ext}"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 127)
    return None


def load_bundle_cameras(bundle_refined_dir: Path, frames_dir: Path, bundle_summary: Path, frame_names: list[str] | None):
    summary = json.loads(bundle_summary.read_text(encoding="utf-8"))
    focal = float(summary["focal_length"])

    if frame_names:
        frame_dirs = [bundle_refined_dir / Path(n).stem for n in frame_names if (bundle_refined_dir / Path(n).stem).exists()]
    else:
        frame_dirs = sorted([d for d in bundle_refined_dir.iterdir() if d.is_dir()])

    cams = []
    for d in frame_dirs:
        name = f"{d.name}.jpg"
        img_path = frames_dir / name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        with (d / "bundle_refined.pkl").open("rb") as f:
            p = pickle.load(f)
        r = np.asarray(p["camera_rotation"], dtype=np.float32)
        t = np.asarray(p["camera_translation"], dtype=np.float32)
        k = np.array([[focal, 0, w * 0.5], [0, focal, h * 0.5], [0, 0, 1]], dtype=np.float32)
        c = -(r.T @ t)
        cams.append({"name": name, "img_path": img_path, "W": w, "H": h, "R": r, "t": t, "K": k, "C": c})
    if not cams:
        raise RuntimeError("No cameras loaded")
    return cams


def bilinear_sample_rgb(img_rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    u0 = np.clip(u.astype(np.int32), 0, w - 2)
    v0 = np.clip(v.astype(np.int32), 0, h - 2)
    du = (u - u0).clip(0, 1)
    dv = (v - v0).clip(0, 1)
    c00 = img_rgb[v0, u0]
    c10 = img_rgb[v0 + 1, u0]
    c01 = img_rgb[v0, u0 + 1]
    c11 = img_rgb[v0 + 1, u0 + 1]
    return (
        (1 - dv[:, None]) * (1 - du[:, None]) * c00 +
        dv[:, None] * (1 - du[:, None]) * c10 +
        (1 - dv[:, None]) * du[:, None] * c01 +
        dv[:, None] * du[:, None] * c11
    ).astype(np.float32)


def fuse_vertex_colors(
    verts: np.ndarray,
    faces: np.ndarray,
    cameras: list[dict],
    masks_dir: Path | None,
    depth_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    face_n = compute_face_normals(verts, faces)
    vert_n = compute_vertex_normals(verts, faces, face_n)

    n = len(verts)
    color_sum = np.zeros((n, 3), dtype=np.float32)
    weight_sum = np.zeros((n,), dtype=np.float32)

    for i, cam in enumerate(cameras, start=1):
        bgr = cv2.imread(str(cam["img_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        w, h = cam["W"], cam["H"]
        depth = rasterize_depth(verts, faces, cam["K"], cam["R"], cam["t"], w, h)
        uv, z = project_points(verts, cam["K"], cam["R"], cam["t"])
        u, v = uv[:, 0], uv[:, 1]

        inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (z > 1e-3)
        if not inside.any():
            continue

        ui = np.clip(u[inside].astype(np.int32), 0, w - 1)
        vi = np.clip(v[inside].astype(np.int32), 0, h - 1)
        z_ok = np.abs(z[inside] - depth[vi, ui]) < depth_tol

        view_dir = cam["C"][None, :] - verts[inside]
        view_dir /= np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-8)
        ndotv = (vert_n[inside] * view_dir).sum(axis=1).clip(0)

        wgt = ndotv * z_ok.astype(np.float32)

        mask = load_mask(masks_dir, cam["name"], w, h)
        if mask is not None:
            wgt *= mask[vi, ui].astype(np.float32)

        good = wgt > 0
        if not np.any(good):
            continue

        idx = np.where(inside)[0][good]
        c = bilinear_sample_rgb(img, u[inside][good], v[inside][good])
        color_sum[idx] += c * wgt[good, None]
        weight_sum[idx] += wgt[good]

        if (i % 5) == 0 or i == len(cameras):
            print(f"camera {i}/{len(cameras)} contributes={int(np.sum(good))}")

    covered = weight_sum > 0
    out = np.zeros((n, 3), dtype=np.float32)
    out[covered] = color_sum[covered] / np.maximum(weight_sum[covered, None], 1e-8)

    if np.any(covered):
        known_idx = np.where(covered)[0]
        known_xyz = verts[known_idx]
        unknown_idx = np.where(~covered)[0]
        if len(unknown_idx) > 0:
            for j in unknown_idx:
                d2 = np.sum((known_xyz - verts[j]) ** 2, axis=1)
                nn = known_idx[int(np.argmin(d2))]
                out[j] = out[nn]

    return out, covered


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach frame colors to fitted SMPL mesh using multi-view projection (no UV baking)")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--bundle-refined-dir", required=True)
    ap.add_argument("--bundle-summary", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frame-names", nargs="*", default=[])
    ap.add_argument("--depth-tol", type=float, default=0.03)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts, faces = load_obj(Path(args.mesh))
    cams = load_bundle_cameras(
        bundle_refined_dir=Path(args.bundle_refined_dir),
        frames_dir=Path(args.frames_dir),
        bundle_summary=Path(args.bundle_summary),
        frame_names=args.frame_names or None,
    )

    colors, covered = fuse_vertex_colors(
        verts=verts,
        faces=faces,
        cameras=cams,
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        depth_tol=args.depth_tol,
    )

    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    out_ply = out_dir / "smpl_vertex_colored_from_frames.ply"
    save_ply_vertex_colors(out_ply, verts, faces, colors_u8)

    stats = {
        "num_vertices": int(len(verts)),
        "num_faces": int(len(faces)),
        "num_cameras": int(len(cams)),
        "observed_vertex_ratio": float(np.mean(covered)),
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(stats)
    print(f"Saved colored mesh: {out_ply}")


if __name__ == "__main__":
    main()
