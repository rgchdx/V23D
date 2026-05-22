"""
UV Texture Baking Pipeline
===========================
1. Load a clean mesh (.ply with vertex colors)
2. UV-unwrap via xatlas (chart-based atlas, minimal distortion)
3. Load COLMAP cameras (cameras.txt + images.txt)
4. Rasterize each UV triangle → 3D position → project into N cameras
5. Pick the best-facing camera(s) per texel and blend color
6. Dilate the atlas to fill seam gaps (pull)
7. Save OBJ + MTL + PNG texture

Usage:
    python bake_texture.py \
        --mesh   E:/V23D_Data/mesh/human_mesh_s3_dense_best.ply \
        --cameras E:/V23D_Data/colmap_s3/sparse/2/cameras.txt \
        --images  E:/V23D_Data/colmap_s3/sparse/2/images.txt \
        --frames  E:/V23D_Data/frames_s3 \
        --output  E:/V23D_Data/mesh/textured \
        --tex-size 4096 \
        --blend-n  5
"""

import argparse, math, os, sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import xatlas
from PIL import Image


# ─────────────────────── COLMAP reader ────────────────────────────────────────

def read_cameras_txt(path):
    """Returns dict: camera_id -> dict(model, w, h, params)"""
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            cid = int(tok[0])
            model = tok[1]
            w, h = int(tok[2]), int(tok[3])
            params = list(map(float, tok[4:]))
            cams[cid] = dict(model=model, w=w, h=h, params=params)
    return cams


def read_images_txt(path):
    """Returns list of dicts with qvec, tvec, camera_id, name."""
    images = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        tok = line.split()
        if len(tok) < 10:
            i += 1
            continue

        image_id = int(tok[0])
        qw, qx, qy, qz = map(float, tok[1:5])
        tx, ty, tz = map(float, tok[5:8])
        cam_id = int(tok[8])
        name = tok[9]
        images.append(
            dict(
                id=image_id,
                qvec=np.array([qw, qx, qy, qz]),
                tvec=np.array([tx, ty, tz]),
                camera_id=cam_id,
                name=name,
            )
        )
        i += 2   # next line is POINTS2D (often blank)
    return images


def qvec_to_R(qvec):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R


def build_cameras(cam_txt, img_txt, frames_dir):
    """Build list of camera dicts ready for projection."""
    cam_params = read_cameras_txt(cam_txt)
    img_list   = read_images_txt(img_txt)
    frames_dir = Path(frames_dir)

    cameras = []
    for im in img_list:
        cp = cam_params[im['camera_id']]
        R  = qvec_to_R(im['qvec'])
        t  = im['tvec']
        # camera center in world
        C  = -R.T @ t

        # intrinsics – handle PINHOLE / SIMPLE_RADIAL / RADIAL
        model  = cp['model']
        p      = cp['params']
        w, h   = cp['w'], cp['h']
        if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'):
            fx = fy = p[0]; cx, cy = p[1], p[2]
            dist = np.array(p[3:]) if len(p) > 3 else np.zeros(4)
        elif model in ('PINHOLE',):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist = np.zeros(4)
        elif model in ('RADIAL',):
            fx = fy = p[0]; cx, cy = p[1], p[2]
            dist = np.array([p[3], p[4], 0, 0]) if len(p) >= 5 else np.zeros(4)
        elif model in ('OPENCV', 'FULL_OPENCV'):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist = np.array(p[4:8]) if len(p) >= 8 else np.zeros(4)
        else:
            fx = fy = p[0]; cx, cy = p[1], p[2]
            dist = np.zeros(4)

        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

        img_path = frames_dir / im['name']
        cameras.append(dict(
            R=R, t=t, C=C, K=K, dist=dist,
            w=w, h=h, path=img_path, name=im['name']
        ))
    return cameras


# ─────────────────────── Projection helpers ───────────────────────────────────

def project_points(pts_world, cam):
    """pts_world: (N,3) → pixel coords (N,2) and depth (N,)."""
    R, t, K, dist = cam['R'], cam['t'], cam['K'], cam['dist']
    pts_cam = (R @ pts_world.T).T + t           # (N,3)
    depth   = pts_cam[:, 2]

    # OpenCV projection with distortion
    rvec = cv2.Rodrigues(R)[0]
    tvec = t.reshape(3, 1)
    # Pad dist to 4 coefficients minimum
    d = np.zeros(4)
    d[:min(len(dist), 4)] = dist[:4]
    pts_img, _ = cv2.projectPoints(
        pts_world.astype(np.float64), rvec, tvec,
        K, d)
    pts_img = pts_img.reshape(-1, 2)
    return pts_img, depth


def sample_image(img_bgr, uv_px):
    """Bilinear sample img_bgr at float pixel coords (N,2). Returns (N,3) uint8."""
    h, w = img_bgr.shape[:2]
    x = np.clip(uv_px[:, 0], 0, w - 1)
    y = np.clip(uv_px[:, 1], 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0).astype(np.float32)[:, None]
    fy = (y - y0).astype(np.float32)[:, None]
    c00 = img_bgr[y0, x0].astype(np.float32)
    c10 = img_bgr[y0, x1].astype(np.float32)
    c01 = img_bgr[y1, x0].astype(np.float32)
    c11 = img_bgr[y1, x1].astype(np.float32)
    colors = (c00*(1-fx)*(1-fy) + c10*fx*(1-fy) +
              c01*(1-fx)*fy     + c11*fx*fy)
    return colors.astype(np.uint8)   # BGR uint8


# ─────────────────────── UV rasterizer ────────────────────────────────────────

def rasterize_uv_triangles(uv_verts, faces, tex_size):
    """
    For every texel in a (tex_size x tex_size) atlas, find which triangle
    covers it and compute barycentric coordinates.

    Returns:
        tri_map  : (H, W) int32  — triangle index, -1 = empty
        bary_map : (H, W, 3) float32 — barycentric coords
    """
    H = W = tex_size
    tri_map  = np.full((H, W), -1, dtype=np.int32)
    bary_map = np.zeros((H, W, 3), dtype=np.float32)

    uv_px = uv_verts * np.array([W, H], dtype=np.float32)  # scale to pixels

    for fi, face in enumerate(faces):
        a, b, c = uv_px[face[0]], uv_px[face[1]], uv_px[face[2]]
        # bounding box
        x0 = max(0, int(min(a[0], b[0], c[0])))
        x1 = min(W-1, int(max(a[0], b[0], c[0])) + 1)
        y0 = max(0, int(min(a[1], b[1], c[1])))
        y1 = min(H-1, int(max(a[1], b[1], c[1])) + 1)
        if x1 < x0 or y1 < y0:
            continue

        # vectorised barycentric test over the bbox
        xs = np.arange(x0, x1+1) + 0.5
        ys = np.arange(y0, y1+1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        p = np.stack([gx, gy], axis=-1)   # (h_box, w_box, 2)

        ab = b - a; ac = c - a
        ap = p - a

        denom = ab[0]*ac[1] - ab[1]*ac[0]
        if abs(denom) < 1e-8:
            continue

        v = (ap[...,0]*ac[1] - ap[...,1]*ac[0]) / denom
        w = (ab[0]*ap[...,1] - ab[1]*ap[...,0]) / denom
        u = 1.0 - v - w

        inside = (u >= 0) & (v >= 0) & (w >= 0)
        iy, ix = np.where(inside)
        if iy.size == 0:
            continue

        global_y = iy + y0
        global_x = ix + x0

        # only overwrite empty texels
        mask = tri_map[global_y, global_x] == -1
        gy_w = global_y[mask]; gx_w = global_x[mask]
        tri_map [gy_w, gx_w] = fi
        bary_map[gy_w, gx_w, 0] = u[iy[mask], ix[mask]]
        bary_map[gy_w, gx_w, 1] = v[iy[mask], ix[mask]]
        bary_map[gy_w, gx_w, 2] = w[iy[mask], ix[mask]]

    return tri_map, bary_map


# ─────────────────────── Main baking ──────────────────────────────────────────

def bake(args):
    tex_size = args.tex_size
    blend_n  = args.blend_n
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load mesh ──────────────────────────────────────────────────────────
    print("[1/6] Loading mesh …")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    verts   = np.asarray(mesh.vertices,  dtype=np.float64)   # (V, 3)
    faces   = np.asarray(mesh.triangles, dtype=np.int32)     # (F, 3)
    vnormals= np.asarray(mesh.vertex_normals, dtype=np.float64)
    print(f"   {len(verts):,} verts, {len(faces):,} faces")

    # ── 2. UV unwrap with xatlas ──────────────────────────────────────────────
    print("[2/6] UV unwrapping (xatlas) …")
    vmapping, indices, uvs = xatlas.parametrize(
        verts.astype(np.float32), faces.astype(np.uint32))
    # vmapping : (V_new,) maps new vert index → old vert index
    # indices  : (F, 3)  triangle indices into new vert array
    # uvs      : (V_new, 2) UV coords in [0,1]
    print(f"   UV atlas: {len(uvs):,} verts, {len(indices):,} faces")

    # Remap normals to new vertex array
    verts_new   = verts[vmapping]          # (V_new, 3)
    vnormals_new= vnormals[vmapping]       # (V_new, 3)

    # ── 3. Load cameras ───────────────────────────────────────────────────────
    print("[3/6] Loading cameras …")
    cameras = build_cameras(args.cameras, args.images, args.frames)
    print(f"   {len(cameras)} cameras loaded")

    masks_dir = Path(args.masks) if args.masks else None
    if masks_dir is not None:
        print(f"   using masks from: {masks_dir}")

    # ── 4. Rasterize UV atlas ─────────────────────────────────────────────────
    print(f"[4/6] Rasterizing UV atlas ({tex_size}×{tex_size}) …")
    tri_map, bary_map = rasterize_uv_triangles(uvs, indices, tex_size)
    filled = np.sum(tri_map >= 0)
    total  = tex_size * tex_size
    print(f"   Coverage: {filled:,}/{total:,} texels ({100*filled/total:.1f}%)")

    # ── 5. Bake colour per texel ──────────────────────────────────────────────
    print(f"[5/6] Baking colours (blend_n={blend_n}) …")

    # Pre-compute 3D position & normal for every texel
    valid_yx = np.argwhere(tri_map >= 0)       # (N_valid, 2)
    if valid_yx.size == 0:
        sys.exit("ERROR: No valid texels — check UV rasterisation.")

    fi_arr   = tri_map[valid_yx[:,0], valid_yx[:,1]]   # face indices
    bary_arr = bary_map[valid_yx[:,0], valid_yx[:,1]]  # (N, 3)

    # 3D positions by barycentric interpolation
    f_verts  = indices[fi_arr]   # (N, 3) new vert indices
    pos3d    = (bary_arr[:,0:1]*verts_new[f_verts[:,0]] +
                bary_arr[:,1:2]*verts_new[f_verts[:,1]] +
                bary_arr[:,2:3]*verts_new[f_verts[:,2]])  # (N, 3)
    nrm3d    = (bary_arr[:,0:1]*vnormals_new[f_verts[:,0]] +
                bary_arr[:,1:2]*vnormals_new[f_verts[:,1]] +
                bary_arr[:,2:3]*vnormals_new[f_verts[:,2]])  # (N, 3)
    nrm3d   /= (np.linalg.norm(nrm3d, axis=1, keepdims=True) + 1e-8)

    N = len(pos3d)
    # Accumulate weighted colours  (float32 to avoid overflow)
    colour_acc    = np.zeros((N, 3), dtype=np.float32)
    weight_acc    = np.zeros(N,      dtype=np.float32)

    for ci, cam in enumerate(cameras):
        img_path = cam['path']
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        mask = None
        if masks_dir is not None:
            mp = masks_dir / f"{Path(cam['name']).stem}.png"
            if mp.exists():
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.shape[:2] != img_bgr.shape[:2]:
                    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Direction from texel to camera
        dir_to_cam = cam['C'] - pos3d       # (N, 3)
        dist_cam   = np.linalg.norm(dir_to_cam, axis=1, keepdims=True) + 1e-8
        dir_to_cam_n = dir_to_cam / dist_cam

        # Visibility: dot(normal, dir_to_cam) > 0
        dot = np.einsum('ij,ij->i', nrm3d, dir_to_cam_n)
        vis = dot > 0.05

        if vis.sum() == 0:
            continue

        # Project
        pix, depth = project_points(pos3d[vis], cam)
        in_frame = (
            (depth > 0) &
            (pix[:,0] >= 0) & (pix[:,0] < cam['w']) &
            (pix[:,1] >= 0) & (pix[:,1] < cam['h'])
        )

        if mask is not None:
            uu = np.clip(np.round(pix[:, 0]).astype(np.int32), 0, cam['w'] - 1)
            vv = np.clip(np.round(pix[:, 1]).astype(np.int32), 0, cam['h'] - 1)
            in_frame &= (mask[vv, uu] > args.mask_thresh)

        if in_frame.sum() == 0:
            continue

        vis_idx = np.where(vis)[0][in_frame]
        pix_ok  = pix[in_frame]
        dot_ok  = dot[vis][in_frame]

        # Sample colours (BGR → RGB)
        cols_bgr = sample_image(img_bgr, pix_ok)
        cols_rgb = cols_bgr[:, ::-1].astype(np.float32)

        colour_acc[vis_idx] += dot_ok[:,None] * cols_rgb
        weight_acc[vis_idx] += dot_ok

        if (ci+1) % 20 == 0:
            print(f"   cam {ci+1}/{len(cameras)}  vis_texels={vis_idx.size:,}")

    # Normalise
    w = weight_acc[:, None]
    covered = w[:,0] > 0
    colour_final = np.zeros((N, 3), dtype=np.uint8)
    colour_final[covered] = np.clip(colour_acc[covered] / w[covered], 0, 255).astype(np.uint8)

    # Fill uncovered texels from vertex colors (fallback)
    if np.asarray(mesh.vertex_colors).shape[0] == len(verts):
        vc = (np.asarray(mesh.vertex_colors)[vmapping] * 255).astype(np.uint8)
        vc_rgb = vc[:, ::-1] if False else vc   # already RGB in open3d
        colour_final[~covered] = (
            bary_arr[~covered,0:1]*vc_rgb[f_verts[~covered,0]] +
            bary_arr[~covered,1:2]*vc_rgb[f_verts[~covered,1]] +
            bary_arr[~covered,2:3]*vc_rgb[f_verts[~covered,2]]
        ).astype(np.uint8)

    # Write into texture image (origin at bottom-left → flip Y for PNG top-left)
    tex = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    tex[valid_yx[:,0], valid_yx[:,1]] = colour_final
    tex = np.flipud(tex)   # UV origin is bottom-left; image origin is top-left

    # ── 6. Dilate to fill seam gaps ───────────────────────────────────────────
    print("[6/6] Dilating atlas seams …")
    alpha = np.flipud((tri_map >= 0).astype(np.uint8) * 255)
    kernel = np.ones((3,3), np.uint8)
    for _ in range(8):
        dilated = cv2.dilate(tex, kernel)
        mask    = (alpha == 0)
        tex[mask] = dilated[mask]
        alpha = cv2.dilate(alpha, kernel)

    # ── Save outputs ──────────────────────────────────────────────────────────
    stem = Path(args.mesh).stem
    tex_png = out_dir / f"{stem}_texture.png"
    obj_path= out_dir / f"{stem}.obj"
    mtl_path= out_dir / f"{stem}.mtl"

    print(f"   Saving texture → {tex_png}")
    Image.fromarray(tex).save(str(tex_png))

    print(f"   Saving OBJ/MTL → {obj_path}")
    mtl_name = mtl_path.name

    with open(str(mtl_path), 'w') as f:
        f.write(f"newmtl material0\n")
        f.write(f"Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
        f.write(f"map_Kd {tex_png.name}\n")

    with open(str(obj_path), 'w') as f:
        f.write(f"mtllib {mtl_name}\n")
        # vertices
        for v in verts_new:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # uvs  (flip V: OBJ uses bottom-left origin)
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {1.0-uv[1]:.6f}\n")
        # normals
        for n in vnormals_new:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("usemtl material0\n")
        # faces (1-indexed)
        for tri in indices:
            i0,i1,i2 = int(tri[0])+1, int(tri[1])+1, int(tri[2])+1
            f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

    print(f"\n✓ Done! Output in: {out_dir}")
    print(f"  OBJ:     {obj_path}")
    print(f"  Texture: {tex_png}")


# ─────────────────────── CLI ──────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Bake photo-realistic UV texture from COLMAP cameras onto a mesh.')
    ap.add_argument('--mesh',    required=True, help='Input .ply mesh')
    ap.add_argument('--cameras', required=True, help='COLMAP cameras.txt')
    ap.add_argument('--images',  required=True, help='COLMAP images.txt')
    ap.add_argument('--frames',  required=True, help='Directory of source images')
    ap.add_argument('--masks',   default=None, help='Optional mask directory (<frame_stem>.png)')
    ap.add_argument('--mask-thresh', type=int, default=127, help='Mask foreground threshold')
    ap.add_argument('--output',  required=True, help='Output directory')
    ap.add_argument('--tex-size', type=int, default=4096, help='Texture atlas size (default 4096)')
    ap.add_argument('--blend-n',  type=int, default=5,    help='Max cameras to blend per texel (unused, kept for API)')
    args = ap.parse_args()
    bake(args)
