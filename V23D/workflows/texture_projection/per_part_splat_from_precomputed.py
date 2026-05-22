"""Build a colored SMPL mesh directly from exported SMPL splat images.

This version samples the full splat images in SMPL-view space and uses the
nearest non-black splat pixel for each projected vertex so the mesh is actually
colored from the splats, not from sparse exact-pixel hits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflows.debug_visualization.export_front_back_part_images import (
    PART_FACE,
    PART_TORSO,
    PART_LARM,
    PART_RARM,
    PART_LLEG,
    PART_RLEG,
    PART_NAMES,
    _load_obj,
    _smpl_vertex_parts,
)


def _save_colored_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colors_u8: np.ndarray) -> None:
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


def _nearest_fill(verts: np.ndarray, colors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = colors.copy()
    if np.all(valid):
        return out
    known = np.where(valid)[0]
    unknown = np.where(~valid)[0]
    if len(known) == 0:
        out[:] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return out
    tree = _kd_tree(verts[known])
    dist, idx = tree.query(verts[unknown], k=1)
    out[unknown] = colors[known[idx.flatten()]]
    return out


def _kd_tree(pts: np.ndarray):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(pts)
    except ImportError:
        from scipy.spatial import KDTree
        return KDTree(pts)


def _nearest_nonblack_colors(img_bgr: np.ndarray, points_xy: np.ndarray, black_threshold: int = 10, allow_gray: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest valid colors and distances for sampled 2D points.

    points_xy: (N,2) int/float array in image coordinates.
    Returns:
      colors_rgb: (N,3) float32 in [0,1]
      distances:  (N,) float32 pixel distances to nearest valid pixel
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((len(points_xy), 3), dtype=np.float32), np.full((len(points_xy),), np.inf, dtype=np.float32)

    if allow_gray:
        # Arms intentionally use a gray background so the silhouette is visible.
        # Keep gray, black, and skin-colored pixels alike.
        mask = img_bgr.sum(axis=2) > black_threshold
    else:
        # Ignore black and neutral-gray backgrounds; keep only pixels with actual chroma.
        chan_max = img_bgr.max(axis=2).astype(np.int16)
        chan_min = img_bgr.min(axis=2).astype(np.int16)
        chroma = chan_max - chan_min
        mask = (chroma > 18) & (img_bgr.sum(axis=2) > black_threshold)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((len(points_xy), 3), dtype=np.float32), np.full((len(points_xy),), np.inf, dtype=np.float32)

    # KDTree over colored pixels, query the point coordinates.
    tree = _kd_tree(coords[:, [1, 0]].astype(np.float32))
    q = np.asarray(points_xy, dtype=np.float32)
    dist, idx = tree.query(q, k=1)
    idx = np.asarray(idx).reshape(-1)
    nearest_xy = coords[idx]
    colors = img_bgr[nearest_xy[:, 0], nearest_xy[:, 1]].astype(np.float32)[:, ::-1] / 255.0
    return colors, np.asarray(dist, dtype=np.float32).reshape(-1)


def _smpl_view_map(verts: np.ndarray, view: str, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = verts.copy()
    if view == "back":
        v[:, 0] *= -1.0
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    sx = (size - 40) / max(float(x.max() - x.min()), 1e-6)
    sy = (size - 40) / max(float(y.max() - y.min()), 1e-6)
    s = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + size * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + size * 0.5).astype(np.int32)
    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    order = np.argsort(z) if view == "front" else np.argsort(-z)
    return px, py, z, order[inside[order]]


def _visible_vertex_ids(verts: np.ndarray, view: str, size: int) -> np.ndarray:
    px, py, _z, order = _smpl_view_map(verts, view=view, size=size)
    idx_map = np.full((size, size), -1, dtype=np.int32)
    for i in order:
        idx_map[py[i], px[i]] = int(i)
    vis = idx_map[idx_map >= 0]
    return np.unique(vis)


def _load_part_image(base_dir: Path, view: str, pid: int) -> np.ndarray:
    p = base_dir / f"smpl_{view}_splatted_{PART_NAMES[pid]}.jpg"
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Missing part splat image: {p}")
    return img


def _part_image_candidates(base_dir: Path, view: str, pid: int) -> list[np.ndarray]:
    """Return candidate images for a body part, ordered from most specific to fallback."""
    imgs = []
    specific = base_dir / f"smpl_{view}_splatted_{PART_NAMES[pid]}.jpg"
    if specific.exists():
        img = cv2.imread(str(specific), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)

    # fallback to the full view splat if the part crop is empty or missing
    full = base_dir / f"smpl_{view}_splatted.jpg"
    if full.exists():
        img = cv2.imread(str(full), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)

    return imgs


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts, faces = _load_obj(Path(args.mesh))
    print(f"Loaded mesh: {len(verts)} vertices, {len(faces)} faces")

    part_v = _smpl_vertex_parts(Path(args.smpl_model))
    if len(part_v) != len(verts):
        raise RuntimeError(f"SMPL vertex count mismatch: mesh={len(verts)} parts={len(part_v)}")

    size = int(args.size)
    colors = np.zeros((len(verts), 3), dtype=np.float32)
    weights = np.zeros(len(verts), dtype=np.float32)

    base_dir = Path(args.requested_images_dir)
    front_splat = cv2.imread(str(base_dir / "smpl_front_splatted.jpg"), cv2.IMREAD_COLOR)
    back_splat = cv2.imread(str(base_dir / "smpl_back_splatted.jpg"), cv2.IMREAD_COLOR)
    if front_splat is None or back_splat is None:
        raise RuntimeError("Could not load smpl_front_splatted.jpg / smpl_back_splatted.jpg from requested-images-dir")

    print(f"Front splat: {front_splat.shape}")
    print(f"Back splat: {back_splat.shape}")

    part_order = [PART_FACE, PART_TORSO, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]

    for view in ["front", "back"]:
        px, py, z, _order = _smpl_view_map(verts, view=view, size=size)
        zmin, zmax = float(z.min()), float(z.max())
        sample_pts = np.stack([px, py], axis=1)

        # Prefer part-specific splats, which are far less likely to be black or
        # contaminated by other body parts.
        for pid in part_order:
            pidx = np.where(part_v == pid)[0]
            if len(pidx) == 0:
                continue

            imgs = _part_image_candidates(base_dir, view, pid)
            if not imgs:
                continue

            best_rgb = None
            best_dist = None
            allow_gray = pid in (PART_LARM, PART_RARM)
            for img in imgs:
                sampled_rgb, dist = _nearest_nonblack_colors(
                    img,
                    sample_pts[pidx],
                    black_threshold=args.black_threshold,
                    allow_gray=allow_gray,
                )
                if best_rgb is None:
                    best_rgb, best_dist = sampled_rgb, dist
                else:
                    # Choose the candidate with the shorter nearest-valid-pixel distance.
                    better = dist < best_dist
                    best_rgb[better] = sampled_rgb[better]
                    best_dist[better] = dist[better]

            if best_rgb is None:
                continue

            if view == "front":
                depth_w = 0.5 + 0.5 * ((z[pidx] - zmin) / max(1e-6, zmax - zmin))
            else:
                depth_w = 0.5 + 0.5 * ((zmax - z[pidx]) / max(1e-6, zmax - zmin))

            good = np.isfinite(best_dist) & (best_dist <= args.max_nearest_dist)
            chosen = good & (depth_w > weights[pidx])
            colors[pidx[chosen]] = best_rgb[chosen]
            weights[pidx[chosen]] = depth_w[chosen]
            print(f"  [{view}] {PART_NAMES[pid]} sampled: {int(good.sum())}, chosen: {int(chosen.sum())}")

    gray = np.array([0.502, 0.502, 0.502], dtype=np.float32)   # 128/255
    face_idx  = np.where(part_v == PART_FACE)[0]
    torso_idx = np.where(part_v == PART_TORSO)[0]

    # ── Torso override: paint chest / torso solid gray ───────────────────────────
    torso_idx_all = np.where(part_v == PART_TORSO)[0]
    colors[torso_idx_all]  = gray
    weights[torso_idx_all] = 1.0   # mark as fully observed
    print(f"  [torso override] painted {len(torso_idx_all)} vertices gray")

    # ── Arm fallback: copy skin/clothing color from face & torso ─────────────────
    # Arms frequently miss splat coverage.  Rather than leaving them black or gray,
    # we flood them with the median color of colored face vertices (skin) for the
    # lower-arm / wrist region and median torso color for the upper-arm region.
    face_idx  = np.where(part_v == PART_FACE)[0]
    torso_idx = np.where(part_v == PART_TORSO)[0]

    face_colored  = face_idx[weights[face_idx]   > 0]
    torso_colored = torso_idx[weights[torso_idx] > 0]

    # Median colors – exclude dark hair pixels (brightness < 0.35) from face
    if len(face_colored) > 0:
        bright_mask = colors[face_colored].mean(axis=1) > 0.35   # skip dark hair
        face_bright = face_colored[bright_mask]
        skin_color = np.median(colors[face_bright if len(face_bright) > 0 else face_colored], axis=0)
    else:
        skin_color = np.array([0.8, 0.65, 0.55], dtype=np.float32)   # rough skin

    if len(torso_colored) > 0:
        chest_color = np.median(colors[torso_colored], axis=0)
    else:
        chest_color = skin_color.copy()

    for arm_pid in (PART_LARM, PART_RARM):
        arm_idx = np.where(part_v == arm_pid)[0]
        if len(arm_idx) == 0:
            continue

        # Identify the "wrist/hand" end: SMPL T-pose arms are horizontal;
        # the hand end has the highest absolute X value.
        arm_verts = verts[arm_idx]
        abs_x = np.abs(arm_verts[:, 0])
        x_min, x_max = abs_x.min(), abs_x.max()
        # normalised 0 (shoulder) → 1 (wrist/hand)
        t = (abs_x - x_min) / max(1e-6, x_max - x_min)

        # Only the very tip (hand) gets skin color; the rest stays gray
        blended = np.where(
            t[:, None] >= 0.85,
            skin_color[None, :],
            gray[None, :],
        ).astype(np.float32)

        # Only overwrite vertices that got no real splat hit (weight == 0)
        # OR that still carry the gray placeholder (all channels ~0.502)
        uncolored = weights[arm_idx] == 0
        gray_placeholder = (
            (np.abs(colors[arm_idx, 0] - 128/255.) < 0.015) &
            (np.abs(colors[arm_idx, 1] - 128/255.) < 0.015) &
            (np.abs(colors[arm_idx, 2] - 128/255.) < 0.015)
        )
        needs_fill = uncolored | gray_placeholder
        colors[arm_idx[needs_fill]]  = blended[needs_fill]
        weights[arm_idx[needs_fill]] = 0.1   # mark as filled so _nearest_fill keeps it
        print(f"  [arm fallback] {PART_NAMES[arm_pid]}: filled {int(needs_fill.sum())} vertices "
              f"(skin={skin_color.round(2)}, chest={chest_color.round(2)})")

    # Fill any remaining unobserved vertices (shoulders, armpits, etc.) with gray
    # to avoid black patches from nearest-fill pulling distant wrong colors.
    unobserved = weights == 0
    colors[unobserved] = gray
    print(f"  [gap fill] painted {int(unobserved.sum())} unobserved vertices gray (shoulders/armpits/etc)")

    # Fill unobserved
    observed = weights > 0
    colors = _nearest_fill(verts, colors, observed)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    out_ply = out_dir / "smpl_textured_from_splat.ply"
    _save_colored_ply(out_ply, verts, faces, colors_u8)
    print(f"Saved: {out_ply}")

    observed_ratio = float(np.mean(observed))
    print(f"Observed vertex ratio: {observed_ratio:.1%}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply exported per-part SMPL splat images directly to the SMPL mesh")
    ap.add_argument("--mesh", required=True, help="SMPL mesh OBJ")
    ap.add_argument("--requested-images-dir", required=True, help="Directory containing smpl_front_splatted_<part>.jpg and smpl_back_splatted_<part>.jpg")
    ap.add_argument("--smpl-model", default=r"E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--black-threshold", type=int, default=10)
    ap.add_argument("--max-nearest-dist", type=float, default=26.0, help="Maximum pixel distance to the nearest colored splat pixel")
    ap.add_argument("--output", required=True)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
