"""
bake_smpl_texture.py
====================
Bakes video-frame texture onto the fitted SMPL mesh and exports a
ROMP-compatible package.

Outputs:
  smpl_textured.obj       -- mesh with UV coords
  smpl_textured.mtl       -- material referencing the texture
  smpl_texture.png        -- baked texture (default 2048x2048)
  smpl_romp_package.npz   -- ROMP animation package:
      betas       (10,)
      v_template  (6890,3)  shaped T-pose vertices
      faces       (13776,3)
      weights     (6890,24) LBS skinning weights
      J_regressor (24,6890)
      parents     (23,)
      uv_verts    (M,2)
      uv_faces    (13776,3)

Usage:
  python bake_smpl_texture.py ^
      --colmap-dir "E:/V23D_Data/colmap_rerun/sparse/1" ^
      --frames-dir "E:/V23D_Data/frames" ^
      --masks-dir  "E:/V23D_Data/masks_rerun" ^
      --output     "E:/V23D_Data/smpl_textured"

Apply ROMP theta afterwards:
  pkg   = np.load("smpl_romp_package.npz")
  model = SMPL("neutral.pkl")
  verts, joints = model(beta_tensor, romp_theta_tensor, trans_tensor)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import xatlas
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.recon.smpl_fitter import (
    _load_smpl_pkl,
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
    _build_K,
)


# ======================================================================
# SMPL shaped T-pose builder
# ======================================================================

def build_tpose_mesh(smpl_model_path: Path, betas: np.ndarray):
    import torch
    dd = _load_smpl_pkl(smpl_model_path)
    n  = len(betas)
    v_t = torch.from_numpy(dd["v_template"]).float()
    sd  = torch.from_numpy(dd["shapedirs"][:, :, :n]).float()
    b   = torch.from_numpy(betas).float().unsqueeze(0)
    v_shaped = (v_t + torch.einsum("vci,bi->bvc", sd, b))[0]
    return dict(
        verts       = v_shaped.numpy(),
        faces       = np.array(dd["f"], dtype=np.int32),
        weights     = dd["weights"],
        J_regressor = dd["J_regressor"],
        parents     = np.array(dd["kintree_table"][0, 1:], dtype=np.int32),
        v_template  = dd["v_template"],
        shapedirs   = dd["shapedirs"],
        posedirs    = dd["posedirs"],
    )


# ======================================================================
# UV atlas
# ======================================================================

def build_uv_atlas(verts, faces, tex_size):
    print("UV unwrapping with xatlas ...")
    vmapping, indices, uvs = xatlas.parametrize(verts, faces)
    print(f"  {len(uvs)} UV verts, {len(indices)} UV faces")
    return uvs.astype(np.float32), indices.astype(np.int32), vmapping.astype(np.int32)


# ======================================================================
# Rasterizer
# ======================================================================

def rasterize_uv(uv_verts, uv_faces, tex_size):
    H = W = tex_size
    tri_map  = np.full((H, W), -1, dtype=np.int32)
    bary_map = np.zeros((H, W, 3), dtype=np.float32)
    uv_px = uv_verts * np.array([W-1, H-1], dtype=np.float32)

    for fi, tri in enumerate(uv_faces):
        if fi % 2000 == 0:
            print(f"  rasterising {fi}/{len(uv_faces)} ...", end="\r")
        p0, p1, p2 = uv_px[tri[0]], uv_px[tri[1]], uv_px[tri[2]]
        xmin = max(0,   int(np.floor(min(p0[0], p1[0], p2[0]))))
        xmax = min(W-1, int(np.ceil( max(p0[0], p1[0], p2[0]))))
        ymin = max(0,   int(np.floor(min(p0[1], p1[1], p2[1]))))
        ymax = min(H-1, int(np.ceil( max(p0[1], p1[1], p2[1]))))
        v0 = p2-p0; v1 = p1-p0
        d00=v0@v0; d01=v0@v1; d11=v1@v1
        denom = d00*d11 - d01*d01
        if abs(denom) < 1e-8:
            continue
        inv = 1.0/denom
        xs = np.arange(xmin, xmax+1, dtype=np.float32)
        ys = np.arange(ymin, ymax+1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        v2 = pts-p0
        d20=(v2*v0).sum(1); d21=(v2*v1).sum(1)
        u = (d11*d20-d01*d21)*inv
        v = (d00*d21-d01*d20)*inv
        w = 1.0-u-v
        ins = (u >= 0)&(v >= 0)&(w >= 0)
        px_y = gy.ravel()[ins].astype(np.int32)
        px_x = gx.ravel()[ins].astype(np.int32)
        bary = np.stack([w[ins],v[ins],u[ins]], axis=1)
        empty = tri_map[px_y, px_x] == -1
        py2 = px_y[empty]; px2 = px_x[empty]
        tri_map[py2, px2]  = fi
        bary_map[py2, px2] = bary[empty]

    print()
    mask = tri_map >= 0
    print(f"  covered {mask.sum()} / {H*W} texels ({100*mask.mean():.1f}%)")
    return tri_map, bary_map, mask


# ======================================================================
# Texel world positions
# ======================================================================

def texel_world_positions(tri_map, bary_map, uv_faces, vert_map, verts3d):
    H, W = tri_map.shape
    world = np.zeros((H, W, 3), dtype=np.float32)
    ys, xs = np.where(tri_map >= 0)
    fi   = tri_map[ys, xs]
    bary = bary_map[ys, xs]
    ovi  = vert_map[uv_faces[fi]]
    world[ys, xs] = (bary[:,0:1]*verts3d[ovi[:,0]]
                   + bary[:,1:2]*verts3d[ovi[:,1]]
                   + bary[:,2:3]*verts3d[ovi[:,2]])
    return world


# ======================================================================
# Vertex normals
# ======================================================================

def compute_vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    v0=verts[faces[:,0]]; v1=verts[faces[:,1]]; v2=verts[faces[:,2]]
    fn = np.cross(v1-v0, v2-v0)
    np.add.at(n, faces[:,0], fn)
    np.add.at(n, faces[:,1], fn)
    np.add.at(n, faces[:,2], fn)
    return n / np.linalg.norm(n, axis=1, keepdims=True).clip(min=1e-8)


# ======================================================================
# Texture baking
# ======================================================================

def bake_texture(world_pos, texel_mask, world_normals,
                 colmap_images, cam_params, frames_dir, masks_dir, tex_size):
    H = W = tex_size
    ys, xs = np.where(texel_mask)
    pts  = world_pos[ys, xs]
    nrms = world_normals[ys, xs]
    nrms /= np.linalg.norm(nrms, axis=1, keepdims=True).clip(min=1e-8)

    accum_color = np.zeros((len(ys), 3), dtype=np.float64)
    accum_w     = np.zeros(len(ys),     dtype=np.float64)

    cam_list = sorted(colmap_images.keys())
    print(f"  Baking from {len(cam_list)} cameras ...")

    for ci, name in enumerate(cam_list):
        if ci % 30 == 0:
            print(f"  camera {ci}/{len(cam_list)}", end="\r")
        img_path = frames_dir / name
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        mask_arr = None
        if masks_dir:
            for ext in (".png", ".jpg"):
                mp = masks_dir / (img_path.stem + ext)
                if mp.exists():
                    mask_arr = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    break

        info = colmap_images[name]
        R    = info["R"]; t = info["t"]
        K    = _build_K(cam_params[info["cam_id"]])
        h_img, w_img = img_bgr.shape[:2]

        C     = -R.T @ t
        dirs  = C - pts
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-8)
        dot   = (nrms * dirs).sum(1)
        vis   = dot > 0.05
        if vis.sum() == 0:
            continue

        pts_cam = (R @ pts[vis].T).T + t
        valid_d = pts_cam[:, 2] > 0
        rvec, _ = cv2.Rodrigues(R)
        pts2d, _ = cv2.projectPoints(
            pts[vis].astype(np.float64), rvec, t.reshape(3,1), K, np.zeros(4))
        pts2d = pts2d.reshape(-1, 2)
        px = pts2d[:,0]; py2 = pts2d[:,1]
        inside = (px>=0)&(px<w_img-1)&(py2>=0)&(py2<h_img-1)&valid_d
        w_cam = np.where(inside, dot[vis]**2, 0.0)

        x0 = np.clip(np.floor(px).astype(int), 0, w_img-2)
        y0 = np.clip(np.floor(py2).astype(int), 0, h_img-2)
        fx = (px - x0).astype(np.float32)
        fy = (py2 - y0).astype(np.float32)
        bgr = (img_bgr[y0,   x0  ].astype(np.float32)*(1-fx[:,None])*(1-fy[:,None])
             + img_bgr[y0,   x0+1].astype(np.float32)*fx[:,None]*(1-fy[:,None])
             + img_bgr[y0+1, x0  ].astype(np.float32)*(1-fx[:,None])*fy[:,None]
             + img_bgr[y0+1, x0+1].astype(np.float32)*fx[:,None]*fy[:,None])
        rgb = bgr[:, ::-1]

        if mask_arr is not None:
            w_cam *= mask_arr[y0, x0].astype(np.float32) / 255.0

        vis_idx = np.where(vis)[0]
        accum_color[vis_idx] += rgb * w_cam[:,None]
        accum_w[vis_idx]     += w_cam

    print()
    valid = accum_w > 1e-6
    result = np.zeros_like(accum_color)
    result[valid] = accum_color[valid] / accum_w[valid, None]

    tex = np.zeros((H, W, 3), dtype=np.float32)
    tex[ys, xs] = result
    out = tex.clip(0, 255).astype(np.uint8)

    print("  Dilating ...")
    mk = (texel_mask.astype(np.uint8) * 255)
    for _ in range(6):
        d  = cv2.dilate(out, np.ones((3,3), np.uint8))
        dm = cv2.dilate(mk,  np.ones((3,3), np.uint8))
        fill = (mk == 0) & (dm > 0)
        out[fill] = d[fill]; mk[fill] = 255

    return out


# ======================================================================
# OBJ + MTL export
# ======================================================================

def export_textured_obj(verts, faces, uv_verts, uv_faces, vert_map,
                        tex_name, out_obj, out_mtl):
    out_mtl.write_text(
        f"newmtl smpl_material\nmap_Kd {tex_name}\n"
        "Ka 0.2 0.2 0.2\nKd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\n")
    lines = [f"mtllib {out_mtl.name}\n", "usemtl smpl_material\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for uv in uv_verts:
        lines.append(f"vt {uv[0]:.6f} {1.0-uv[1]:.6f}\n")
    for gf, uf in zip(faces, uv_faces):
        lines.append(f"f {gf[0]+1}/{uf[0]+1} {gf[1]+1}/{uf[1]+1} {gf[2]+1}/{uf[2]+1}\n")
    out_obj.write_text("".join(lines))
    print(f"Saved OBJ -> {out_obj}")


# ======================================================================
# ROMP NPZ
# ======================================================================

def export_romp_package(mesh_data, betas, scale, uv_verts, uv_faces, out_npz):
    np.savez_compressed(
        str(out_npz),
        betas        = betas.astype(np.float32),
        scale        = np.array([scale], dtype=np.float32),
        v_template   = mesh_data["verts"].astype(np.float32),
        faces        = mesh_data["faces"].astype(np.int32),
        weights      = mesh_data["weights"].astype(np.float32),
        J_regressor  = mesh_data["J_regressor"].astype(np.float32),
        parents      = mesh_data["parents"].astype(np.int32),
        uv_verts     = uv_verts.astype(np.float32),
        uv_faces     = uv_faces.astype(np.int32),
    )
    print(f"Saved ROMP package -> {out_npz}")


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    _SMPL_DEF = (r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models"
                 r"\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--smpl-model",  default=_SMPL_DEF)
    ap.add_argument("--smpl-out",    default="E:/V23D_Data/smpl_out")
    ap.add_argument("--colmap-dir",  required=True)
    ap.add_argument("--frames-dir",  required=True)
    ap.add_argument("--masks-dir",   default=None)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--tex-size",    type=int, default=2048)
    a = ap.parse_args()

    smpl_path  = Path(a.smpl_model)
    smpl_out   = Path(a.smpl_out)
    colmap_dir = Path(a.colmap_dir)
    frames_dir = Path(a.frames_dir)
    masks_dir  = Path(a.masks_dir) if a.masks_dir else None
    out_dir    = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = np.load(str(smpl_out / "betas.npy"))
    scale_path = smpl_out / "scale.npy"
    scale = float(np.load(str(scale_path)).reshape(-1)[0]) if scale_path.exists() else 1.0
    print(f"betas: {betas.round(3)}")
    print(f"scale: {scale:.4f}")

    print("Building T-pose mesh ...")
    mesh = build_tpose_mesh(smpl_path, betas)
    verts = mesh["verts"]; faces = mesh["faces"]

    uv_verts, uv_faces, vert_map = build_uv_atlas(verts, faces, a.tex_size)

    print(f"Rasterising UV ({a.tex_size}x{a.tex_size}) ...")
    tri_map, bary_map, texel_mask = rasterize_uv(uv_verts, uv_faces, a.tex_size)

    # Mean trans: SMPL local -> COLMAP world
    trans_p = smpl_out / "trans_per_frame.npy"
    if trans_p.exists():
        try:
            td = np.load(str(trans_p), allow_pickle=True).item()
            mean_t = np.mean([np.array(v) for v in td.values()], axis=0)
        except Exception:
            mean_t = np.zeros(3)
    else:
        mean_t = np.zeros(3)
    print(f"Mean translation: {mean_t.round(3)}")
    verts_w = verts * scale + mean_t[None, :]

    print("Texel positions ...")
    world_pos = texel_world_positions(tri_map, bary_map, uv_faces, vert_map, verts_w)

    vn = compute_vertex_normals(verts_w, faces)
    H = W = a.tex_size
    ys, xs = np.where(texel_mask)
    fi   = tri_map[ys, xs]
    bary = bary_map[ys, xs]
    ovi  = vert_map[uv_faces[fi]]
    wn = np.zeros((H, W, 3), dtype=np.float32)
    wn[ys, xs] = (bary[:,0:1]*vn[ovi[:,0]]
                + bary[:,1:2]*vn[ovi[:,1]]
                + bary[:,2:3]*vn[ovi[:,2]])

    print("Loading COLMAP ...")
    cam_params  = _read_colmap_cameras_txt(colmap_dir / "cameras.txt")
    colmap_imgs = _read_colmap_images_txt(colmap_dir / "images.txt")
    print(f"  {len(colmap_imgs)} cameras")

    print("Baking ...")
    tex = bake_texture(world_pos, texel_mask, wn,
                       colmap_imgs, cam_params, frames_dir, masks_dir, a.tex_size)

    tex_path = out_dir / "smpl_texture.png"
    Image.fromarray(tex).save(str(tex_path))
    print(f"Texture -> {tex_path}")

    export_textured_obj(verts, faces, uv_verts, uv_faces, vert_map,
                        tex_path.name,
                        out_dir / "smpl_textured.obj",
                        out_dir / "smpl_textured.mtl")

    export_romp_package(mesh, betas, scale, uv_verts, uv_faces,
                        out_dir / "smpl_romp_package.npz")

    print(f"\nDone -> {out_dir}")
    print("  smpl_textured.obj      -- open in Blender/MeshLab")
    print("  smpl_texture.png       -- texture atlas")
    print("  smpl_romp_package.npz  -- skinning weights for ROMP animation")


if __name__ == "__main__":
    main()
