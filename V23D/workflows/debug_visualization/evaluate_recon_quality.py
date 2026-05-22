import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def parse_images_with_obs(images_txt: Path):
    lines = images_txt.read_text(encoding="utf-8").splitlines()
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
        name = t[9]
        pts = []
        if i + 1 < len(lines):
            p2d = lines[i + 1].strip().split()
            for j in range(0, len(p2d), 3):
                if j + 2 >= len(p2d):
                    break
                x = float(p2d[j])
                y = float(p2d[j + 1])
                pid = int(p2d[j + 2])
                pts.append((x, y, pid))
        out.append((name, pts))
        i += 2
    return out


def sfm_mask_consistency(images_txt: Path, masks_dir: Path, thr: int = 127):
    data = parse_images_with_obs(images_txt)
    total_obs = 0
    in_mask = 0
    per_img = []

    for name, obs in data:
        mp = masks_dir / f"{Path(name).stem}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        h, w = m.shape[:2]

        n = 0
        k = 0
        for x, y, pid in obs:
            if pid < 0:
                continue
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0 or yi < 0 or xi >= w or yi >= h:
                continue
            n += 1
            if m[yi, xi] > thr:
                k += 1
        if n > 0:
            total_obs += n
            in_mask += k
            per_img.append(k / n)

    if total_obs == 0:
        return {}
    ratios = np.array(per_img, dtype=np.float64)
    return {
        "images_with_obs": int(len(per_img)),
        "triangulated_observations": int(total_obs),
        "overall_in_mask_ratio": float(in_mask / total_obs),
        "median_per_image_ratio": float(np.median(ratios)),
        "p10_per_image_ratio": float(np.quantile(ratios, 0.10)),
    }


def sfm_reprojection(points3d_txt: Path):
    errs = []
    tracks = []
    for ln in points3d_txt.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if (not s) or s.startswith("#"):
            continue
        t = s.split()
        if len(t) < 8:
            continue
        errs.append(float(t[7]))
        tracks.append((len(t) - 8) // 2)

    if len(errs) == 0:
        return {}

    errs = np.array(errs, dtype=np.float64)
    tracks = np.array(tracks, dtype=np.float64)
    return {
        "points3D": int(len(errs)),
        "reproj_error_mean_px": float(np.mean(errs)),
        "reproj_error_median_px": float(np.median(errs)),
        "reproj_error_p90_px": float(np.quantile(errs, 0.90)),
        "track_length_mean": float(np.mean(tracks)),
        "track_length_median": float(np.median(tracks)),
        "track_length_p90": float(np.quantile(tracks, 0.90)),
    }


def image_brightness(images_dir: Path, n: int = 30):
    files = sorted([p for p in images_dir.glob("*.jpg")])[:n]
    if len(files) == 0:
        return {}
    vals = []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        vals.append(float(im.mean() / 255.0))
    if len(vals) == 0:
        return {}
    arr = np.array(vals, dtype=np.float64)
    return {
        "sample_count": int(len(vals)),
        "mean_luminance": float(np.mean(arr)),
        "median_luminance": float(np.median(arr)),
    }


def mesh_stats(mesh_path: Path):
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    out = {
        "vertices": int(len(m.vertices)),
        "triangles": int(len(m.triangles)),
        "has_vertex_colors": bool(m.has_vertex_colors()),
    }
    if m.has_vertex_colors():
        vc = np.asarray(m.vertex_colors)
        out["vertex_color_mean"] = [float(x) for x in vc.mean(axis=0)]
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate SfM/3DGS/mesh quality")
    ap.add_argument("--model", required=True, help="COLMAP model dir with images.txt and points3D.txt")
    ap.add_argument("--masks", required=True, help="Mask directory")
    ap.add_argument("--images", required=True, help="Image directory used in reconstruction")
    ap.add_argument("--mesh", required=True, help="Final mesh path")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    model = Path(args.model)

    report = {
        "sfm_mask_consistency": sfm_mask_consistency(model / "images.txt", Path(args.masks), 127),
        "sfm_reprojection": sfm_reprojection(model / "points3D.txt"),
        "image_brightness": image_brightness(Path(args.images), 30),
        "mesh": mesh_stats(Path(args.mesh)),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
