import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def read_cameras_txt(path):
    cams = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t = line.split()
            cid = int(t[0])
            model = t[1]
            w, h = int(t[2]), int(t[3])
            params = list(map(float, t[4:]))
            cams[cid] = {"model": model, "w": w, "h": h, "params": params}
    return cams


def read_images_txt(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if (not s) or s.startswith("#"):
            i += 1
            continue
        t = s.split()
        if len(t) < 10:
            i += 1
            continue
        out.append(
            {
                "id": int(t[0]),
                "qvec": np.array(list(map(float, t[1:5])), dtype=np.float64),
                "tvec": np.array(list(map(float, t[5:8])), dtype=np.float64),
                "camera_id": int(t[8]),
                "name": t[9],
            }
        )
        i += 2
    return out


def qvec_to_R(qvec):
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def build_camera(im, cp):
    R = qvec_to_R(im["qvec"])
    t = im["tvec"]
    C = -R.T @ t

    model = cp["model"]
    p = cp["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        fx = fy = p[0]
        cx, cy = p[1], p[2]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return {"R": R, "t": t, "C": C, "K": K, "w": cp["w"], "h": cp["h"], "name": im["name"]}


def project(pts, cam):
    Xc = (cam["R"] @ pts.T).T + cam["t"][None, :]
    z = Xc[:, 2]
    u = cam["K"][0, 0] * (Xc[:, 0] / np.maximum(z, 1e-9)) + cam["K"][0, 2]
    v = cam["K"][1, 1] * (Xc[:, 1] / np.maximum(z, 1e-9)) + cam["K"][1, 2]
    return np.stack([u, v], axis=1), z


def bilinear_sample_rgb(img_rgb, uv):
    h, w = img_rgb.shape[:2]
    x = np.clip(uv[:, 0], 0, w - 1)
    y = np.clip(uv[:, 1], 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]

    c00 = img_rgb[y0, x0].astype(np.float32)
    c10 = img_rgb[y0, x1].astype(np.float32)
    c01 = img_rgb[y1, x0].astype(np.float32)
    c11 = img_rgb[y1, x1].astype(np.float32)

    return (
        c00 * (1 - fx) * (1 - fy)
        + c10 * fx * (1 - fy)
        + c01 * (1 - fx) * fy
        + c11 * fx * fy
    )


def main():
    ap = argparse.ArgumentParser(description="Colorize mesh vertices from multi-view images")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-views", type=int, default=297)
    args = ap.parse_args()

    print("[1/5] Loading mesh")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    nrms = np.asarray(mesh.vertex_normals, dtype=np.float64)
    nrms /= np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-9)
    print(f"    verts={len(verts):,} tris={len(mesh.triangles):,}")

    print("[2/5] Loading COLMAP cameras")
    cams = read_cameras_txt(args.cameras)
    ims = read_images_txt(args.images)
    if args.max_views > 0 and len(ims) > args.max_views:
        step = max(1, len(ims) // args.max_views)
        ims = ims[::step][: args.max_views]
    print(f"    views={len(ims)}")

    frames_dir = Path(args.frames)
    masks_dir = Path(args.masks) if args.masks else None

    color_sum = np.zeros((len(verts), 3), dtype=np.float64)
    weight_sum = np.zeros((len(verts),), dtype=np.float64)

    print("[3/5] Multi-view color accumulation")
    used = 0
    for i, im in enumerate(ims, start=1):
        cp = cams.get(im["camera_id"])
        if cp is None:
            continue
        cam = build_camera(im, cp)

        img_path = frames_dir / im["name"]
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        mask = None
        if masks_dir is not None:
            mp = masks_dir / f"{Path(im['name']).stem}.png"
            if mp.exists():
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        uv, z = project(verts, cam)
        inside = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

        if mask is not None:
            xi = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, w - 1)
            yi = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, h - 1)
            inside &= mask[yi, xi] > 127

        if not np.any(inside):
            continue

        to_cam = cam["C"][None, :] - verts
        to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
        facing = np.einsum("ij,ij->i", nrms, to_cam)
        inside &= facing > 0.05

        idx = np.where(inside)[0]
        if len(idx) == 0:
            continue

        cols = bilinear_sample_rgb(img_rgb, uv[idx]) / 255.0
        wgt = np.clip(facing[idx], 0.0, 1.0)

        color_sum[idx] += cols * wgt[:, None]
        weight_sum[idx] += wgt
        used += 1

        if i % 20 == 0 or i == len(ims):
            print(f"    view {i}/{len(ims)} used={used} contributing_vertices={len(idx):,}")

    print("[4/5] Finalizing vertex colors")
    vcols = np.zeros((len(verts), 3), dtype=np.float64)
    good = weight_sum > 0
    vcols[good] = color_sum[good] / weight_sum[good, None]

    if np.any(~good):
        if mesh.has_vertex_colors():
            old = np.asarray(mesh.vertex_colors)
            if len(old) == len(verts):
                vcols[~good] = old[~good]
            else:
                vcols[~good] = 0.6
        else:
            vcols[~good] = 0.6

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcols, 0.0, 1.0))
    mesh.compute_vertex_normals()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(out), mesh, write_vertex_colors=True)
    print(f"[5/5] Saved: {out} (ok={ok})")
    print(f"    colored_vertices={int(np.sum(good)):,}/{len(verts):,}")


if __name__ == "__main__":
    main()
