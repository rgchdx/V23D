"""
bake_smpl_texture_visible_rigid.py
=================================
Bake a texture atlas for SMPL using the per-frame visible-joint rigid fit.

Key idea
--------
Use the per-frame `pose` + `trans` recovered by `fit_smpl_visible_rigid.py`
as frame-specific registration targets during baking:

- UVs live on the canonical SMPL mesh
- For each atlas texel, keep barycentric coordinates on the canonical face
- For each frame, deform the same SMPL topology with that frame's pose
- Project the deformed texel point into the frame
- Visibility-test with a rendered depth buffer
- Sample image color and accumulate top-N views

This is more accurate than baking from one canonical mesh when the person pose
varies across frames, while still exporting a canonical textured mesh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from bake_smpl_texture_raycast import (
    _compute_face_normals,
    _compute_vertex_normals,
    _dilate_atlas,
    _project_points,
    _rasterise_depth,
    export_obj_with_uv,
    unwrap_uv,
)
from src.recon.smpl_fitter import SMPL, _build_K, _read_colmap_cameras_txt, _read_colmap_images_txt


def _load_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(tri)
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64)


def _build_texel_bary_data(
    uv_verts: np.ndarray,
    uv_faces: np.ndarray,
    vmapping: np.ndarray,
    tex_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize UV atlas and return flattened texel-barycentric data.

    Returns:
      valid_yx  (N,2) int32
      vidx      (N,3) int32  original vertex indices
      bary      (N,3) float32
      valid_mask (H,W) bool
    """
    H = W = tex_size
    valid_mask = np.zeros((H, W), dtype=bool)
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vidx_all: list[np.ndarray] = []
    bary_all: list[np.ndarray] = []

    print(f"Rasterising UV barycentrics into {H}x{W} atlas...")
    for fi in range(len(uv_faces)):
        i0, i1, i2 = uv_faces[fi]
        u0, v0 = uv_verts[i0] * (tex_size - 1)
        u1, v1 = uv_verts[i1] * (tex_size - 1)
        u2, v2 = uv_verts[i2] * (tex_size - 1)
        c0, r0 = float(u0), float(v0)
        c1, r1 = float(u1), float(v1)
        c2, r2 = float(u2), float(v2)

        rmin = max(0, int(min(r0, r1, r2)))
        rmax = min(H - 1, int(max(r0, r1, r2)) + 1)
        cmin = max(0, int(min(c0, c1, c2)))
        cmax = min(W - 1, int(max(c0, c1, c2)) + 1)
        if rmin > rmax or cmin > cmax:
            continue

        area = (c1 - c0) * (r2 - r0) - (c2 - c0) * (r1 - r0)
        if abs(area) < 1e-6:
            continue
        inv_a = 1.0 / area

        rr = np.arange(rmin, rmax + 1)
        cc = np.arange(cmin, cmax + 1)
        gc, gr = np.meshgrid(cc, rr)
        gcf = gc.astype(np.float32)
        grf = gr.astype(np.float32)

        w0 = ((gcf - c1) * (r2 - r1) - (grf - r1) * (c2 - c1)) * inv_a
        w1 = ((gcf - c2) * (r0 - r2) - (grf - r2) * (c0 - c2)) * inv_a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not inside.any():
            continue

        rows = gr[inside].astype(np.int32)
        cols = gc[inside].astype(np.int32)
        valid_mask[rows, cols] = True

        ov = np.array([
            int(vmapping[i0]),
            int(vmapping[i1]),
            int(vmapping[i2]),
        ], dtype=np.int32)
        vidx = np.repeat(ov[None, :], len(rows), axis=0)
        bary = np.stack([w0[inside], w1[inside], w2[inside]], axis=1).astype(np.float32)

        rows_all.append(rows)
        cols_all.append(cols)
        vidx_all.append(vidx)
        bary_all.append(bary)

        if fi % 2000 == 0:
            print(f"  face {fi}/{len(uv_faces)}", end="\r", flush=True)

    print()
    rows = np.concatenate(rows_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    vidx = np.concatenate(vidx_all, axis=0)
    bary = np.concatenate(bary_all, axis=0)
    valid_yx = np.stack([rows, cols], axis=1)
    print(f"Valid texels: {len(valid_yx)}")
    return valid_yx, vidx, bary, valid_mask


def _interp_points(verts: np.ndarray, vidx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    v0 = verts[vidx[:, 0]]
    v1 = verts[vidx[:, 1]]
    v2 = verts[vidx[:, 2]]
    return (bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2).astype(np.float32)


def _load_mask(masks_dir: Path | None, stem: str, size: tuple[int, int]) -> np.ndarray | None:
    if masks_dir is None:
        return None
    W, H = size
    for ext in (".png", ".jpg", ".jpeg"):
        mp = masks_dir / (stem + ext)
        if mp.exists():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127
    return None


def _build_cameras_named(cameras: dict, images: dict, frames_dir: str | Path) -> list[dict]:
    frames_dir = Path(frames_dir)
    cams = []
    for name, info in images.items():
        cam_id = info["cam_id"]
        cam_def = cameras[cam_id]
        K = _build_K(cam_def).astype(np.float32)
        R = info["R"].astype(np.float32)
        t = info["t"].astype(np.float32)
        W, H = cam_def["w"], cam_def["h"]
        img_path = frames_dir / name
        if not img_path.exists():
            stem = Path(name).stem
            for ext in (".jpg", ".jpeg", ".png"):
                p = frames_dir / (stem + ext)
                if p.exists():
                    img_path = p
                    break
        cams.append(dict(K=K, R=R, t=t, W=W, H=H, img_path=img_path, name=name))
    return cams


def _load_camera_weights(path: str | Path | None) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "weights" in data and isinstance(data["weights"], dict):
        data = data["weights"]
    return {str(k): float(v) for k, v in data.items()}


def _fill_uncovered_texels_knn(
    tex: np.ndarray,
    cov: np.ndarray,
    valid_mask: np.ndarray,
    k_neighbors: int,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    if k_neighbors <= 0:
        return tex, np.zeros_like(valid_mask, dtype=bool)

    known_mask = valid_mask & (cov > 0)
    fill_mask = valid_mask & ~known_mask
    if not known_mask.any() or not fill_mask.any():
        return tex, np.zeros_like(valid_mask, dtype=bool)

    known_yx = np.argwhere(known_mask).astype(np.float32)
    fill_yx = np.argwhere(fill_mask).astype(np.int32)
    known_colors = tex[known_mask].astype(np.float32)
    out = tex.copy()
    hallucinated_mask = np.zeros_like(valid_mask, dtype=bool)

    kk = min(int(k_neighbors), len(known_yx))
    known_yx_f = known_yx[None, :, :]
    print(f"Filling {len(fill_yx)} uncovered texels using {kk}-NN color propagation...")

    for start in range(0, len(fill_yx), chunk_size):
        end = min(start + chunk_size, len(fill_yx))
        query_yx = fill_yx[start:end].astype(np.float32)
        d2 = ((query_yx[:, None, :] - known_yx_f) ** 2).sum(axis=2)
        nn_idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        nn_d2 = np.take_along_axis(d2, nn_idx, axis=1)
        order = np.argsort(nn_d2, axis=1)
        nn_idx = np.take_along_axis(nn_idx, order, axis=1)
        nn_d2 = np.take_along_axis(nn_d2, order, axis=1)

        weights = 1.0 / np.maximum(nn_d2, 1.0)
        nn_colors = known_colors[nn_idx]
        color = (weights[:, :, None] * nn_colors).sum(axis=1) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)

        rows = fill_yx[start:end, 0]
        cols = fill_yx[start:end, 1]
        out[rows, cols] = color
        hallucinated_mask[rows, cols] = True

    return out, hallucinated_mask


def bake_texture_visible_rigid(
    smpl: SMPL,
    betas: np.ndarray,
    canonical_verts: np.ndarray,
    faces: np.ndarray,
    cameras: list[dict],
    poses_dict: dict,
    trans_dict: dict,
    scale_fixed: float,
    tex_size: int,
    blend_n: int,
    depth_tol: float,
    masks_dir: Path | None,
    n_cams: int,
    knn_fill_k: int,
    camera_weights: dict[str, float] | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv_verts, uv_faces, vmapping = unwrap_uv(canonical_verts, faces)
    valid_yx, vidx, bary, valid_mask = _build_texel_bary_data(uv_verts, uv_faces, vmapping, tex_size)

    N = len(valid_yx)
    top_colors = np.zeros((N, blend_n, 3), dtype=np.float32)
    top_weights = np.full((N, blend_n), -np.inf, dtype=np.float32)

    cam_list = [c for c in cameras if c["name"] in poses_dict and c["name"] in trans_dict]
    if camera_weights:
        for c in cam_list:
            c["priority_weight"] = float(camera_weights.get(c["name"], 1.0))
        cam_list.sort(key=lambda c: (c.get("priority_weight", 1.0), c["name"]), reverse=True)
    else:
        for c in cam_list:
            c["priority_weight"] = 1.0
    if n_cams > 0 and len(cam_list) > n_cams:
        step = max(1, len(cam_list) // n_cams)
        cam_list = cam_list[::step][:n_cams]
    print(f"Using {len(cam_list)} cameras with visible-rigid fit")

    betas_t = torch.from_numpy(betas.astype(np.float32)).to(device).unsqueeze(0)
    zero_t = torch.zeros(1, 3, device=device)

    chunk = 200_000
    for ci, cam in enumerate(cam_list):
        name = cam["name"]
        cam_priority = float(cam.get("priority_weight", 1.0))
        img_path = cam["img_path"]
        if not img_path.exists():
            continue

        pose_np = np.asarray(poses_dict[name], dtype=np.float32)
        trans_np = np.asarray(trans_dict[name], dtype=np.float32)

        with torch.no_grad():
            pose_t = torch.from_numpy(pose_np).to(device).unsqueeze(0)
            vb, _ = smpl(betas_t, pose_t, zero_t)
            verts = (vb[0].cpu().numpy() * scale_fixed + trans_np).astype(np.float32)

        face_normals = _compute_face_normals(verts, faces)
        vert_normals = _compute_vertex_normals(verts, faces, face_normals)

        K, R, t, W, H = cam["K"], cam["R"], cam["t"], cam["W"], cam["H"]
        depth_buf = _rasterise_depth(verts, faces, K, R, t, W, H)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        mask = _load_mask(masks_dir, Path(name).stem, (W, H))

        cam_pos_world = -(R.T @ t)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            pts = _interp_points(verts, vidx[start:end], bary[start:end])
            nrms = _interp_points(vert_normals, vidx[start:end], bary[start:end])
            nmag = np.linalg.norm(nrms, axis=1, keepdims=True)
            nrms = nrms / np.maximum(nmag, 1e-8)

            proj_uv, proj_z = _project_points(pts, K, R, t)
            pu = proj_uv[:, 0]
            pv = proj_uv[:, 1]
            in_bounds = ((pu >= 0) & (pu < W - 1) & (pv >= 0) & (pv < H - 1) & (proj_z > 0.01))
            if not in_bounds.any():
                continue

            pi_u = pu[in_bounds].astype(np.int32).clip(0, W - 1)
            pi_v = pv[in_bounds].astype(np.int32).clip(0, H - 1)
            buf_z = depth_buf[pi_v, pi_u]
            vis = np.abs(proj_z[in_bounds] - buf_z) < depth_tol

            view_dir = cam_pos_world - pts[in_bounds]
            view_dir = view_dir / np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-8)
            dot = (nrms[in_bounds] * view_dir).sum(axis=1).clip(0)
            weight = dot * vis * cam_priority
            if mask is not None:
                weight = weight * mask[pi_v, pi_u]
            good = weight > 0
            if not good.any():
                continue

            pu_f = pu[in_bounds][good]
            pv_f = pv[in_bounds][good]
            pu0 = pu_f.astype(np.int32).clip(0, W - 2)
            pv0 = pv_f.astype(np.int32).clip(0, H - 2)
            du = (pu_f - pu0).clip(0, 1)
            dv = (pv_f - pv0).clip(0, 1)
            c00 = img[pv0,     pu0]
            c10 = img[pv0 + 1, pu0]
            c01 = img[pv0,     pu0 + 1]
            c11 = img[pv0 + 1, pu0 + 1]
            color = ((1 - dv[:, None]) * (1 - du[:, None]) * c00 +
                     (    dv[:, None]) * (1 - du[:, None]) * c10 +
                     (1 - dv[:, None]) * (    du[:, None]) * c01 +
                     (    dv[:, None]) * (    du[:, None]) * c11)

            valid_chunk_idx = np.where(in_bounds)[0][good] + start
            min_slots = np.argmin(top_weights[valid_chunk_idx], axis=1)
            cur_min = top_weights[valid_chunk_idx, min_slots]
            update = weight[good] > cur_min
            if update.any():
                upd_idx = valid_chunk_idx[update]
                upd_slots = min_slots[update]
                top_weights[upd_idx, upd_slots] = weight[good][update]
                top_colors[upd_idx, upd_slots] = color[update]

        if (ci + 1) % 10 == 0 or ci == len(cam_list) - 1:
            print(f"  camera {ci+1}/{len(cam_list)}: {name}", flush=True)

    wsum = np.maximum(top_weights, 0).sum(axis=1, keepdims=True)
    has_w = (wsum[:, 0] > 0)
    blended = np.zeros((N, 3), dtype=np.float32)
    blended[has_w] = (
        (np.maximum(top_weights[has_w], 0)[:, :, None] * top_colors[has_w]).sum(axis=1)
        / wsum[has_w]
    )

    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    cov = np.zeros((tex_size, tex_size), dtype=np.float32)
    tex[valid_yx[:, 0], valid_yx[:, 1]] = blended
    cov[valid_yx[:, 0], valid_yx[:, 1]] = has_w.astype(np.float32)
    tex, hallucinated_mask = _fill_uncovered_texels_knn(tex, cov, valid_mask, knn_fill_k)
    tex_u8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    tex_u8 = _dilate_atlas(tex_u8, valid_mask, n_iters=8)
    return tex_u8, cov, hallucinated_mask.astype(np.uint8), uv_verts, uv_faces, vmapping


def main():
    ap = argparse.ArgumentParser(description="Bake SMPL texture using visible-rigid per-frame registrations")
    ap.add_argument("--smpl-out", required=True,
                    help="SMPL output dir with betas.npy and smpl_canonical.obj")
    ap.add_argument("--rigid-out", required=True,
                    help="Visible-rigid output dir with poses_per_frame.json and trans_per_frame.npy")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--camera-weight-json", default=None,
                    help="Optional JSON mapping frame names to priority weights for front/back guided baking")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tex-size", type=int, default=1024)
    ap.add_argument("--blend-n", type=int, default=5)
    ap.add_argument("--depth-tol", type=float, default=0.025)
    ap.add_argument("--n-cams", type=int, default=80)
    ap.add_argument("--knn-fill-k", type=int, default=4,
                    help="Fill uncovered valid atlas texels from the K nearest colored texels")
    _DEFAULT_SMPL = (
        r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models"
        r"\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    )
    ap.add_argument("--smpl-model", default=_DEFAULT_SMPL)
    args = ap.parse_args()

    smpl_out = Path(args.smpl_out)
    rigid_out = Path(args.rigid_out)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = np.load(str(smpl_out / "betas.npy")).astype(np.float32)
    canonical_obj = smpl_out / "smpl_canonical.obj"
    if canonical_obj.exists():
        canonical_verts, faces = _load_obj(canonical_obj)
    else:
        raise FileNotFoundError(f"Missing canonical mesh: {canonical_obj}")

    poses_dict = json.loads((rigid_out / "poses_per_frame.json").read_text())
    trans_dict = np.load(str(rigid_out / "trans_per_frame.npy"), allow_pickle=True).item()

    scale_fixed = float(np.load(str(rigid_out / "scale.npy"))[0])
    print(f"Using fixed bake scale: {scale_fixed:.4f}")

    cams_txt = Path(args.colmap_dir) / "cameras.txt"
    images_txt = Path(args.colmap_dir) / "images.txt"
    cams_def = _read_colmap_cameras_txt(cams_txt)
    imgs_def = _read_colmap_images_txt(images_txt)
    cam_list = _build_cameras_named(cams_def, imgs_def, args.frames_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPL(args.smpl_model, n_betas=len(betas)).to(device)
    camera_weights = _load_camera_weights(args.camera_weight_json)

    tex_u8, cov, hallucinated_mask, uv_verts, uv_faces, vmapping = bake_texture_visible_rigid(
        smpl=smpl,
        betas=betas,
        canonical_verts=canonical_verts,
        faces=faces,
        cameras=cam_list,
        poses_dict=poses_dict,
        trans_dict=trans_dict,
        scale_fixed=scale_fixed,
        tex_size=args.tex_size,
        blend_n=args.blend_n,
        depth_tol=args.depth_tol,
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        n_cams=args.n_cams,
        knn_fill_k=args.knn_fill_k,
        camera_weights=camera_weights,
        device=device,
    )

    tex_name = "smpl_texture_visible_rigid.png"
    Image.fromarray(tex_u8).save(str(out_dir / tex_name))
    cov_u8 = (np.clip(cov, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(cov_u8).save(str(out_dir / "coverage_visible_rigid.png"))
    Image.fromarray((hallucinated_mask * 255).astype(np.uint8)).save(str(out_dir / "hallucinated_fill_mask.png"))
    export_obj_with_uv(canonical_verts, faces, uv_verts, uv_faces, vmapping, tex_name,
                       out_dir / "smpl_textured_visible_rigid.obj")
    print(f"Saved textured mesh to {out_dir}")


if __name__ == "__main__":
    main()
