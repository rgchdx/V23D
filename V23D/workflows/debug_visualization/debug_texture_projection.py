"""
debug_texture_projection.py
============================
Visual debugger for texture baking. Shows:

  1. SMPL mesh projected onto N sample frames with:
       - orange wireframe overlay
       - green dots = texel sample points (from atlas UV regions)
       - magenta dots = where the texture is actually sampled from in the image

  2. Side-by-side: source image region | atlas patch

  3. Contact sheet saved as debug_texture_projection.jpg

Usage
------
python debug_texture_projection.py \
    --smpl-obj    E:/V23D_Data/smpl_out/smpl_canonical.obj \
    --colmap-dir  E:/V23D_Data/colmap_rerun/sparse/1 \
    --frames-dir  E:/V23D_Data/frames \
    --n-frames    12 \
    --out-dir     E:/V23D_Data/debug_tex_proj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.recon.smpl_fitter import (
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
    _build_K,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
        [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
        [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])


def load_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(tri)
    return np.array(verts, np.float32), np.array(faces, np.int64)


def project_verts(verts, K, R, t):
    """World → pixel.  Returns (N,2) float32 and z (N,)."""
    cam = (R.astype(np.float64) @ verts.T).T + t.astype(np.float64)
    z   = cam[:, 2]
    f, cx, cy = K[0,0], K[0,2], K[1,2]
    u = f * cam[:,0] / np.maximum(z, 1e-6) + cx
    v = f * cam[:,1] / np.maximum(z, 1e-6) + cy
    return np.stack([u, v], 1).astype(np.float32), z.astype(np.float32)


def face_normals_world(verts, faces):
    v0 = verts[faces[:,0]]; v1 = verts[faces[:,1]]; v2 = verts[faces[:,2]]
    n  = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(mag, 1e-8)


def sample_body_regions(verts, faces, n=200, seed=7):
    """Sample n face centroids roughly spread over the body."""
    rng = np.random.default_rng(seed)
    fi  = rng.choice(len(faces), n, replace=False)
    v0 = verts[faces[fi,0]]; v1 = verts[faces[fi,1]]; v2 = verts[faces[fi,2]]
    centroids = (v0 + v1 + v2) / 3.0
    normals   = face_normals_world(verts, faces)[fi]
    return centroids, normals, fi


def draw_wireframe(img, verts_px, z, faces, K, alpha=0.45):
    """Draw orange wireframe of visible edges."""
    H, W = img.shape[:2]
    overlay = img.copy()
    visible = z > 0.01
    drawn = set()
    for f in faces:
        for a, b in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]:
            key = (min(a,b), max(a,b))
            if key in drawn:
                continue
            drawn.add(key)
            if not (visible[a] and visible[b]):
                continue
            p1 = (int(verts_px[a,0]), int(verts_px[a,1]))
            p2 = (int(verts_px[b,0]), int(verts_px[b,1]))
            if (0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H):
                cv2.line(overlay, p1, p2, (0, 140, 255), 1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_sample_points(img, pts_px, z, color, radius=4):
    """Draw dots for projected sample points."""
    H, W = img.shape[:2]
    out = img.copy()
    for i in range(len(pts_px)):
        if z[i] < 0.01:
            continue
        x, y = int(pts_px[i,0]), int(pts_px[i,1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(out, (x, y), radius, color, -1)
            cv2.circle(out, (x, y), radius+1, (0,0,0), 1)
    return out


def compute_visibility_weight(centroids, normals, K, R, t, near=0.02):
    """Return weight [0,1] for each sample point in this camera."""
    cam_pos = -(R.T @ t)
    view    = cam_pos - centroids
    mag     = np.linalg.norm(view, axis=1, keepdims=True)
    view    = view / np.maximum(mag, 1e-8)
    dot     = (normals * view).sum(axis=1).clip(0)
    pts_cam = (R @ centroids.T).T + t
    z = pts_cam[:, 2]
    return dot * (z > near)


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame debug image
# ══════════════════════════════════════════════════════════════════════════════

def make_frame_debug(
    img_path: Path,
    verts:    np.ndarray,
    faces:    np.ndarray,
    K:        np.ndarray,
    R:        np.ndarray,
    t:        np.ndarray,
    centroids:np.ndarray,
    normals:  np.ndarray,
    mask_path: Path | None,
) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        H = int(K[1,2]*2); W = int(K[0,2]*2)
        img = np.zeros((H, W, 3), np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W = img.shape[:2]

    # Overlay mask lightly
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (W, H))
            tint = img.copy()
            tint[mask < 128] = (tint[mask < 128] * 0.4).astype(np.uint8)
            img = tint

    # Project mesh
    verts_px, z = project_verts(verts, K, R, t)

    # Draw wireframe
    img = draw_wireframe(img, verts_px, z, faces, K)

    # Project sample centroids
    cent_px, cent_z = project_verts(centroids, K, R, t)

    # Visibility weights
    w = compute_visibility_weight(centroids, normals, K, R, t)

    # Color dots by weight: green = high weight (used), red = low (ignored)
    for i in range(len(cent_px)):
        if cent_z[i] < 0.01:
            continue
        x, y = int(cent_px[i,0]), int(cent_px[i,1])
        if not (0 <= x < W and 0 <= y < H):
            continue
        wi = float(w[i])
        # green→red gradient based on weight
        g = int(wi * 255)
        r = int((1 - wi) * 255)
        color = (r, g, 0)
        cv2.circle(img, (x, y), 5, color, -1)
        cv2.circle(img, (x, y), 6, (0, 0, 0), 1)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Show what region of each image is actually feeding the atlas
# ══════════════════════════════════════════════════════════════════════════════

def make_sampling_heatmap(
    img_path:  Path,
    verts:     np.ndarray,
    faces:     np.ndarray,
    K:         np.ndarray,
    R:         np.ndarray,
    t:         np.ndarray,
    n_samples: int = 3000,
) -> np.ndarray:
    """
    Draw a heatmap on the source image showing which pixels are being
    sampled for the texture atlas (hot = many texels sample from here).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        H = int(K[1,2]*2); W = int(K[0,2]*2)
        return np.zeros((H, W, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # Dense surface sample
    rng = np.random.default_rng(42)
    fi  = rng.choice(len(faces), n_samples, replace=True)
    r1  = rng.random(n_samples); r2 = rng.random(n_samples)
    sqr = np.sqrt(r1)
    u_ = 1 - sqr; v_ = sqr * (1 - r2); w_ = sqr * r2
    pts = (u_[:,None] * verts[faces[fi,0]] +
           v_[:,None] * verts[faces[fi,1]] +
           w_[:,None] * verts[faces[fi,2]])

    pts_px, zv = project_verts(pts, K, R, t)

    # Normal-based visibility
    fn = face_normals_world(verts, faces)[fi]
    cam_pos = -(R.T @ t)
    view = cam_pos - pts
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
    dot = (fn * view).sum(axis=1).clip(0)

    # Build heatmap
    heat = np.zeros((H, W), np.float32)
    for i in range(n_samples):
        if zv[i] < 0.01 or dot[i] < 0.05:
            continue
        px, py = int(pts_px[i,0]), int(pts_px[i,1])
        if 0 <= px < W and 0 <= py < H:
            heat[py, px] += dot[i]

    # Blur and normalise
    heat = cv2.GaussianBlur(heat, (31, 31), 0)
    if heat.max() > 0:
        heat = heat / heat.max()

    # Colormap overlay
    heat_u8  = (heat * 255).astype(np.uint8)
    heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)

    # Blend with original
    mask_h = (heat > 0.05).astype(np.float32)[:, :, None]
    out = (img * (1 - 0.6 * mask_h) + heat_col * 0.6 * mask_h).clip(0, 255).astype(np.uint8)

    # Bounding box of sampled region
    ys, xs = np.where(heat > 0.1)
    if len(xs):
        cv2.rectangle(out,
                      (int(xs.min()), int(ys.min())),
                      (int(xs.max()), int(ys.max())),
                      (255, 255, 0), 2)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl-obj",    required=True,
                    help="SMPL canonical OBJ (e.g. E:/V23D_Data/smpl_out/smpl_canonical.obj)")
    ap.add_argument("--colmap-dir",  required=True)
    ap.add_argument("--frames-dir",  required=True)
    ap.add_argument("--masks-dir",   default=None)
    ap.add_argument("--out-dir",     required=True)
    ap.add_argument("--n-frames",    type=int, default=12,
                    help="Number of sample frames to visualise")
    ap.add_argument("--n-samples",   type=int, default=200,
                    help="Number of body surface sample points to show per frame")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    verts, faces = load_obj(Path(args.smpl_obj))
    print(f"Mesh: {len(verts)} verts, {len(faces)} faces")

    # Load COLMAP
    cams_def  = _read_colmap_cameras_txt(Path(args.colmap_dir) / "cameras.txt")
    imgs_def  = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")
    cam_K     = _build_K(cams_def[list(cams_def.keys())[0]]).astype(np.float32)
    print(f"K: f={cam_K[0,0]:.1f}  cx={cam_K[0,2]:.1f}  cy={cam_K[1,2]:.1f}")
    print(f"Cameras: {len(imgs_def)}")

    # Sample body points
    centroids, normals, _ = sample_body_regions(verts, faces, n=args.n_samples)

    # Pick evenly-spaced frames
    all_names = sorted(imgs_def.keys())
    step   = max(1, len(all_names) // args.n_frames)
    names  = all_names[::step][:args.n_frames]

    frames_dir = Path(args.frames_dir)
    masks_dir  = Path(args.masks_dir) if args.masks_dir else None

    panels = []
    for name in names:
        info = imgs_def[name]
        R    = info["R"].astype(np.float32)
        t    = info["t"].astype(np.float32)

        # Find frame image
        stem = Path(name).stem
        img_path = frames_dir / name
        if not img_path.exists():
            for ext in (".jpg", ".jpeg", ".png"):
                p = frames_dir / (stem + ext)
                if p.exists():
                    img_path = p; break

        mask_path = None
        if masks_dir is not None:
            for ext in (".png", ".jpg"):
                mp = masks_dir / (stem + ext)
                if mp.exists():
                    mask_path = mp; break

        print(f"  Processing {stem}...")

        # Left panel: mesh overlay + color-coded sample dots
        left = make_frame_debug(
            img_path, verts, faces, cam_K, R, t,
            centroids, normals, mask_path)

        # Right panel: sampling heatmap
        right = make_sampling_heatmap(
            img_path, verts, faces, cam_K, R, t, n_samples=3000)

        # Stack left | right
        H = left.shape[0]
        W = left.shape[1]
        if right.shape[0] != H or right.shape[1] != W:
            right = cv2.resize(right, (W, H))

        # Label
        label_h = 30
        combined = np.concatenate([left, right], axis=1)
        canvas = np.zeros((H + label_h, W * 2, 3), np.uint8)
        canvas[:H] = combined
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        draw.text((4, H + 4), f"{stem}  |  left: mesh overlay  right: sampling heatmap",
                  fill=(220, 220, 100))
        panels.append(np.array(pil))

        # Also save individual frame
        frame_out = out_dir / f"debug_{stem}.jpg"
        Image.fromarray(np.array(pil)).save(str(frame_out), quality=90)

    # Contact sheet
    if panels:
        cols = 2
        rows = (len(panels) + cols - 1) // cols
        ph, pw = panels[0].shape[:2]
        sheet  = np.zeros((rows * ph, cols * pw, 3), np.uint8)
        for i, p in enumerate(panels):
            r = i // cols; c = i % cols
            h_, w_ = p.shape[:2]
            if h_ != ph or w_ != pw:
                p = cv2.resize(p, (pw, ph))
            sheet[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = p
        # Downscale if huge
        MAX_W = 3200
        if sheet.shape[1] > MAX_W:
            sc = MAX_W / sheet.shape[1]
            sheet = cv2.resize(sheet, (MAX_W, int(sheet.shape[0]*sc)))
        out_path = out_dir / "contact_sheet.jpg"
        Image.fromarray(sheet).save(str(out_path), quality=88)
        print(f"\nContact sheet: {out_path}")

    # ── Legend ────────────────────────────────────────────────────────
    print("\nLEGEND")
    print("  Left panel:")
    print("    Orange wireframe = projected SMPL mesh")
    print("    Green dots  = sample points facing THIS camera (high weight → used for texture)")
    print("    Red dots    = sample points facing away (weight ~0, NOT sampled here)")
    print("    Dark region = outside person mask")
    print("  Right panel:")
    print("    Heatmap (blue→red) = pixel sampling density")
    print("    Hot = many atlas texels sample from here")
    print("    Yellow box = bounding region actually used")
    print(f"\nAll debug frames saved to {out_dir}")


if __name__ == "__main__":
    main()
