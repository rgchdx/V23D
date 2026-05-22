from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from bake_smpl_texture_raycast import (
    _build_cameras,
    _compute_face_normals,
    _compute_vertex_normals,
    _dilate_atlas,
    _project_points,
    _rasterise_depth,
    export_obj_with_uv,
    unwrap_uv,
)
from src.pose.extract_mediapipe import extract_landmarks_single, load_landmarks_json, _make_landmarker
from src.recon.smpl_fitter import _build_K, _read_colmap_cameras_txt, _read_colmap_images_txt


MP_TO_SMPLX = [
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (23, "left_hip"),
    (24, "right_hip"),
    (25, "left_knee"),
    (26, "right_knee"),
    (27, "left_ankle"),
    (28, "right_ankle"),
]


def _camera_yaw_deg(R: np.ndarray, t: np.ndarray, center: np.ndarray) -> float:
    cam_pos = -(R.T @ t)
    v = cam_pos - center
    return float(np.degrees(np.arctan2(v[0], v[2])))


def _view_bin(yaw_deg: float) -> str:
    a = ((yaw_deg + 180.0) % 360.0) - 180.0
    if -45 <= a <= 45:
        return "front"
    if 45 < a <= 135:
        return "left"
    if -135 <= a < -45:
        return "right"
    return "back"


def _build_texel_data(uv_verts: np.ndarray, uv_faces: np.ndarray, vmapping: np.ndarray, tex_size: int):
    H = W = tex_size
    valid_mask = np.zeros((H, W), dtype=bool)
    rows_all, cols_all, vidx_all, bary_all = [], [], [], []

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

        ov = np.array([int(vmapping[i0]), int(vmapping[i1]), int(vmapping[i2])], dtype=np.int32)
        vidx = np.repeat(ov[None, :], len(rows), axis=0)
        bary = np.stack([w0[inside], w1[inside], w2[inside]], axis=1).astype(np.float32)

        rows_all.append(rows)
        cols_all.append(cols)
        vidx_all.append(vidx)
        bary_all.append(bary)

    rows = np.concatenate(rows_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    vidx = np.concatenate(vidx_all, axis=0)
    bary = np.concatenate(bary_all, axis=0)
    valid_yx = np.stack([rows, cols], axis=1)
    return valid_yx, vidx, bary, valid_mask


def _interp_points(arr: np.ndarray, vidx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    a0 = arr[vidx[:, 0]]
    a1 = arr[vidx[:, 1]]
    a2 = arr[vidx[:, 2]]
    return (bary[:, 0:1] * a0 + bary[:, 1:2] * a1 + bary[:, 2:3] * a2).astype(np.float32)


def fit_smplx_first_frame(model_path: Path, frame_path: Path, K: np.ndarray, R: np.ndarray, t: np.ndarray, device: torch.device):
    import smplx
    from smplx.joint_names import JOINT_NAMES

    if not model_path.exists():
        raise FileNotFoundError(f"SMPL-X model missing: {model_path}")

    model = smplx.create(
        model_path=str(model_path.parent),
        model_type="smplx",
        gender="neutral",
        ext=model_path.suffix.lstrip('.'),
        use_pca=False,
        num_pca_comps=12,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_transl=True,
    ).to(device)

    jidx = {n: JOINT_NAMES.index(n) for _, n in MP_TO_SMPLX}

    img_bgr = cv2.imread(str(frame_path))
    if img_bgr is None:
        raise RuntimeError(f"Cannot load frame: {frame_path}")

    with _make_landmarker(Path(r"E:/V23D_Data/pose_landmarker_lite.task")) as landmarker:
        lms = extract_landmarks_single(img_bgr, landmarker)
    if lms is None:
        raise RuntimeError("MediaPipe failed on first frame")

    obs_pts, obs_w, joint_ids = [], [], []
    for mp_i, name in MP_TO_SMPLX:
        x, y, v = float(lms[mp_i, 0]), float(lms[mp_i, 1]), float(lms[mp_i, 2])
        if np.isnan(x) or np.isnan(y) or v < 0.2:
            continue
        obs_pts.append([x, y])
        obs_w.append(np.clip(v, 0.2, 1.0))
        joint_ids.append(jidx[name])

    if len(obs_pts) < 6:
        raise RuntimeError("Not enough visible body joints on first frame")

    obs = torch.from_numpy(np.asarray(obs_pts, np.float32)).to(device)
    wts = torch.from_numpy(np.asarray(obs_w, np.float32)).to(device)
    K_t = torch.from_numpy(K.astype(np.float32)).to(device)
    R_t = torch.from_numpy(R.astype(np.float32)).to(device)
    t_t = torch.from_numpy(t.astype(np.float32)).to(device)

    global_orient = nn.Parameter(torch.zeros(1, 3, device=device))
    body_pose = nn.Parameter(torch.zeros(1, 63, device=device))
    betas = nn.Parameter(torch.zeros(1, 10, device=device))
    transl = nn.Parameter(torch.zeros(1, 3, device=device))
    log_scale = nn.Parameter(torch.zeros(1, device=device))

    opt = torch.optim.Adam([global_orient, body_pose, betas, transl, log_scale], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600, eta_min=0.002)

    for _ in range(600):
        opt.zero_grad()
        out = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=torch.zeros(1, 45, device=device),
            right_hand_pose=torch.zeros(1, 45, device=device),
            transl=torch.zeros(1, 3, device=device),
            return_verts=True,
        )
        s = torch.exp(log_scale)
        joints_w = out.joints[:, :144, :] * s + transl.unsqueeze(1)
        j = joints_w[0, joint_ids]
        j4 = torch.cat([j, torch.ones(j.shape[0], 1, device=device)], dim=1)
        P = K_t @ torch.cat([R_t, t_t.unsqueeze(1)], dim=1)
        pr = (P @ j4.T).T
        xy = pr[:, :2] / pr[:, 2:3].clamp(min=0.01)

        diff = xy - obs
        reproj = (torch.sqrt((diff ** 2).sum(dim=1) + 4.0) * wts).mean()
        reg = 0.02 * (global_orient ** 2).mean() + 0.2 * (body_pose ** 2).mean() + 0.1 * (betas ** 2).mean() + 0.01 * (log_scale ** 2).mean()
        loss = reproj + reg
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        out = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=torch.zeros(1, 45, device=device),
            right_hand_pose=torch.zeros(1, 45, device=device),
            transl=torch.zeros(1, 3, device=device),
            return_verts=True,
        )
        s = float(torch.exp(log_scale).item())
        verts = out.vertices[0].cpu().numpy() * s + transl[0].cpu().numpy()
        faces = model.faces.astype(np.int64)
        reproj_px = float(reproj.item())

    params = {
        "betas": betas[0].detach().cpu().numpy(),
        "global_orient": global_orient[0].detach().cpu().numpy(),
        "body_pose": body_pose[0].detach().cpu().numpy(),
        "transl": transl[0].detach().cpu().numpy(),
        "scale": s,
        "reproj_px": reproj_px,
    }
    return verts.astype(np.float32), faces, params


def bake_by_camera_angle(verts: np.ndarray, faces: np.ndarray, cam_list: list[dict], masks_dir: Path | None, tex_size: int, blend_n: int, depth_tol: float):
    uv_verts, uv_faces, vmapping = unwrap_uv(verts, faces)
    valid_yx, vidx, bary, valid_mask = _build_texel_data(uv_verts, uv_faces, vmapping, tex_size)

    N = len(valid_yx)
    top_colors = np.zeros((N, blend_n, 3), dtype=np.float32)
    top_weights = np.full((N, blend_n), -np.inf, dtype=np.float32)

    face_normals = _compute_face_normals(verts, faces)
    vert_normals = _compute_vertex_normals(verts, faces, face_normals)
    tex_pts = _interp_points(verts, vidx, bary)
    tex_nrm = _interp_points(vert_normals, vidx, bary)
    tex_nrm /= np.maximum(np.linalg.norm(tex_nrm, axis=1, keepdims=True), 1e-8)

    vtx_rgb_acc = np.zeros((len(verts), 3), dtype=np.float64)
    vtx_w_acc = np.zeros((len(verts),), dtype=np.float64)

    center = verts.mean(0)
    bins = {"front": 0, "left": 0, "right": 0, "back": 0}

    for ci, cam in enumerate(cam_list):
        K, R, t, W, H = cam["K"], cam["R"], cam["t"], cam["W"], cam["H"]
        img_path = cam["img_path"]
        if not img_path.exists():
            continue

        yaw = _camera_yaw_deg(R, t, center)
        b = _view_bin(yaw)
        bins[b] += 1

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))

        mask = None
        if masks_dir is not None:
            stem = Path(cam["name"]).stem
            for ext in (".png", ".jpg", ".jpeg"):
                mp = masks_dir / (stem + ext)
                if mp.exists():
                    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127
                    break

        depth_buf = _rasterise_depth(verts, faces, K, R, t, W, H)
        proj_uv, proj_z = _project_points(tex_pts, K, R, t)
        pu, pv = proj_uv[:, 0], proj_uv[:, 1]
        in_bounds = (pu >= 0) & (pu < W - 1) & (pv >= 0) & (pv < H - 1) & (proj_z > 0.01)
        if not in_bounds.any():
            continue

        pi_u = pu[in_bounds].astype(np.int32)
        pi_v = pv[in_bounds].astype(np.int32)
        buf_z = depth_buf[pi_v, pi_u]
        vis = np.abs(proj_z[in_bounds] - buf_z) < depth_tol

        cam_pos = -(R.T @ t)
        view_dir = cam_pos - tex_pts[in_bounds]
        view_dir /= np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-8)
        dot = (tex_nrm[in_bounds] * view_dir).sum(axis=1).clip(0)
        weight = dot * vis
        if mask is not None:
            weight *= mask[pi_v, pi_u]

        pu_f = pu[in_bounds]
        pv_f = pv[in_bounds]
        pu0 = pu_f.astype(np.int32).clip(0, W - 2)
        pv0 = pv_f.astype(np.int32).clip(0, H - 2)
        du = (pu_f - pu0).clip(0, 1)
        dv = (pv_f - pv0).clip(0, 1)
        c00 = img[pv0, pu0]
        c10 = img[pv0 + 1, pu0]
        c01 = img[pv0, pu0 + 1]
        c11 = img[pv0 + 1, pu0 + 1]
        color = ((1 - dv[:, None]) * (1 - du[:, None]) * c00 +
                 (dv[:, None]) * (1 - du[:, None]) * c10 +
                 (1 - dv[:, None]) * du[:, None] * c01 +
                 dv[:, None] * du[:, None] * c11)

        valid_ids = np.where(in_bounds)[0]
        good = weight > 0
        if good.any():
            vii = valid_ids[good]
            wi = weight[good]
            ci_ = color[good]
            mslot = np.argmin(top_weights[vii], axis=1)
            cur = top_weights[vii, mslot]
            up = wi > cur
            if up.any():
                ui = vii[up]
                us = mslot[up]
                top_weights[ui, us] = wi[up]
                top_colors[ui, us] = ci_[up]

        # Vertex color update (mesh itself)
        v2d, vz = _project_points(verts, K, R, t)
        inb_v = (v2d[:, 0] >= 0) & (v2d[:, 0] < W - 1) & (v2d[:, 1] >= 0) & (v2d[:, 1] < H - 1) & (vz > 0.01)
        vi = np.where(inb_v)[0]
        if len(vi) > 0:
            uu = v2d[vi, 0].astype(np.int32)
            vv = v2d[vi, 1].astype(np.int32)
            zbuf_v = depth_buf[vv, uu]
            vis_v = np.abs(vz[vi] - zbuf_v) < depth_tol
            if mask is not None:
                vis_v = vis_v & mask[vv, uu]
            if np.any(vis_v):
                vgood = vi[vis_v]
                uu = v2d[vgood, 0].astype(np.int32)
                vv = v2d[vgood, 1].astype(np.int32)
                c = img[vv, uu]
                vtx_rgb_acc[vgood] += c
                vtx_w_acc[vgood] += 1.0

        if (ci + 1) % 20 == 0 or ci == len(cam_list) - 1:
            print(f"camera {ci+1}/{len(cam_list)}  bin={b} yaw={yaw:.1f}")

    wsum = np.maximum(top_weights, 0).sum(axis=1, keepdims=True)
    has = wsum[:, 0] > 0
    blended = np.zeros((N, 3), dtype=np.float32)
    blended[has] = ((np.maximum(top_weights[has], 0)[:, :, None] * top_colors[has]).sum(axis=1) / wsum[has])

    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    cov = np.zeros((tex_size, tex_size), dtype=np.float32)
    tex[valid_yx[:, 0], valid_yx[:, 1]] = blended
    cov[valid_yx[:, 0], valid_yx[:, 1]] = has.astype(np.float32)
    tex_u8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    tex_u8 = _dilate_atlas(tex_u8, valid_mask, n_iters=8)

    vcol = np.zeros_like(verts, dtype=np.float32)
    ok = vtx_w_acc > 0
    vcol[ok] = (vtx_rgb_acc[ok] / vtx_w_acc[ok, None]).astype(np.float32)

    return tex_u8, cov, uv_verts, uv_faces, vmapping, vcol, bins


def save_obj_vertex_colors(path: Path, verts: np.ndarray, faces: np.ndarray, vcol: np.ndarray):
    lines = ["# vertex-colored mesh\n"]
    for v, c in zip(verts, vcol):
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Fit SMPL-X on first frame, then bake texture from all views by camera angle")
    ap.add_argument("--smplx-model", required=True, help="Path to SMPL-X model file, e.g. SMPLX_NEUTRAL.npz")
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tex-size", type=int, default=1024)
    ap.add_argument("--blend-n", type=int, default=5)
    ap.add_argument("--depth-tol", type=float, default=0.025)
    ap.add_argument("--n-cams", type=int, default=0, help="0=all")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = _read_colmap_cameras_txt(Path(args.colmap_dir) / "cameras.txt")
    imgs = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")
    K = _build_K(cams[list(cams.keys())[0]]).astype(np.float32)

    cam_list = _build_cameras(cams, imgs, args.frames_dir)
    if args.n_cams > 0 and len(cam_list) > args.n_cams:
        step = max(1, len(cam_list) // args.n_cams)
        cam_list = cam_list[::step][:args.n_cams]

    # First frame = earliest registered by name that exists in frames dir
    first_cam = None
    for c in sorted(cam_list, key=lambda x: x["name"]):
        if c["img_path"].exists():
            first_cam = c
            break
    if first_cam is None:
        raise RuntimeError("No valid first frame found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts, faces, params = fit_smplx_first_frame(
        model_path=Path(args.smplx_model),
        frame_path=first_cam["img_path"],
        K=first_cam["K"],
        R=first_cam["R"],
        t=first_cam["t"],
        device=device,
    )

    np.save(str(out_dir / "smplx_betas.npy"), params["betas"])
    np.save(str(out_dir / "smplx_global_orient.npy"), params["global_orient"])
    np.save(str(out_dir / "smplx_body_pose.npy"), params["body_pose"])
    np.save(str(out_dir / "smplx_transl.npy"), params["transl"])
    np.save(str(out_dir / "smplx_scale.npy"), np.array([params["scale"]], dtype=np.float32))
    (out_dir / "fit_info.json").write_text(json.dumps({
        "first_frame": first_cam["name"],
        "reproj_px": params["reproj_px"],
        "scale": params["scale"],
    }, indent=2))

    # Canonical posed mesh from first frame
    export_obj_with_uv(verts, faces, *unwrap_uv(verts, faces), "smplx_texture.png", out_dir / "smplx_first_frame_mesh.obj")

    tex_u8, cov, uv_verts, uv_faces, vmapping, vcol, bins = bake_by_camera_angle(
        verts=verts,
        faces=faces,
        cam_list=cam_list,
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        tex_size=args.tex_size,
        blend_n=args.blend_n,
        depth_tol=args.depth_tol,
    )

    Image.fromarray(tex_u8).save(out_dir / "smplx_texture.png")
    Image.fromarray((np.clip(cov, 0, 1) * 255).astype(np.uint8)).save(out_dir / "coverage.png")
    export_obj_with_uv(verts, faces, uv_verts, uv_faces, vmapping, "smplx_texture.png", out_dir / "smplx_textured.obj")
    save_obj_vertex_colors(out_dir / "smplx_vertex_colored.obj", verts, faces, vcol)

    (out_dir / "camera_view_bins.json").write_text(json.dumps(bins, indent=2))
    print(f"Done. First-frame fit reproj={params['reproj_px']:.2f}px")
    print(f"View bins: {bins}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
