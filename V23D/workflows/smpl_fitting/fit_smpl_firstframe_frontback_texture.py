from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_TEX = _ROOT / "workflows" / "texture_baking"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TEX))

from bake_smpl_texture_raycast import (
    _compute_face_normals,
    _compute_vertex_normals,
    _dilate_atlas,
    _project_points,
    _rasterise_depth,
    export_obj_with_uv,
    unwrap_uv,
)
from workflows.texture_baking.bake_smpl_texture_visible_rigid import _fill_uncovered_texels_knn
from src.recon.smpl_fitter import SMPL, _build_K, _read_colmap_cameras_txt, _read_colmap_images_txt


COCO_TO_SMPL_PAIRS: list[tuple[int, int]] = [
    (5, 16),
    (6, 17),
    (7, 18),
    (8, 19),
    (9, 20),
    (10, 21),
    (11, 1),
    (12, 2),
    (13, 4),
    (14, 5),
    (15, 7),
    (16, 8),
]


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


def _detect_keypoints_torchvision(frame_path: Path, device: torch.device) -> tuple[np.ndarray, float]:
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise FileNotFoundError(frame_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0

    with torch.no_grad():
        out = model([tensor])[0]

    if len(out.get("scores", [])) == 0:
        raise RuntimeError(f"No person detected in {frame_path}")

    det_scores = out["scores"].detach().cpu().numpy().astype(np.float32)
    best = int(np.argmax(det_scores))
    det_score = float(det_scores[best])

    kpts = out["keypoints"][best].detach().cpu().numpy().astype(np.float32)
    if "keypoints_scores" in out:
        kp_scores = out["keypoints_scores"][best].detach().cpu().numpy().astype(np.float32)
    else:
        kp_scores = np.full((kpts.shape[0],), det_score, dtype=np.float32)

    arr = np.full((17, 3), np.nan, dtype=np.float32)
    arr[:, :2] = kpts[:, :2]
    arr[:, 2] = kp_scores
    return arr, det_score


def _project_world(pts: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pts_cam = (R @ pts.T).T + t.unsqueeze(0)
    z = pts_cam[:, 2]
    uvw = (K @ pts_cam.T).T
    xy = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-4)
    return xy, z


def _bbox_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _bbox_from_keypoints(kpts: np.ndarray, conf_thr: float) -> np.ndarray:
    good = np.isfinite(kpts[:, 0]) & np.isfinite(kpts[:, 1]) & (kpts[:, 2] >= conf_thr)
    if not np.any(good):
        raise RuntimeError("No valid keypoints for bbox")
    xy = kpts[good, :2]
    return np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()], dtype=np.float32)


def _initial_translation_from_bbox(kpts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, conf_thr: float) -> np.ndarray:
    bbox = _bbox_from_keypoints(kpts, conf_thr)
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    bbox_h = max(float(bbox[3] - bbox[1]), 32.0)
    focal = float(0.5 * (K[0, 0] + K[1, 1]))
    depth = max(1.5, (focal * 1.65) / bbox_h)

    ray_cam = np.array([(cx - K[0, 2]) / K[0, 0], (cy - K[1, 2]) / K[1, 1], 1.0], dtype=np.float32)
    ray_cam = ray_cam / max(np.linalg.norm(ray_cam), 1e-8)
    cam_pos = -(R.T @ t)
    ray_world = R.T @ ray_cam
    ray_world = ray_world / max(np.linalg.norm(ray_world), 1e-8)
    return (cam_pos + ray_world * depth).astype(np.float32)


def fit_first_frame_smpl(
    smpl_model: Path,
    first_cam: dict,
    masks_dir: Path | None,
    device: torch.device,
    kp_conf_thr: float,
    n_iters: int,
):
    smpl = SMPL(smpl_model, n_betas=10).to(device)
    kpts, det_score = _detect_keypoints_torchvision(first_cam["img_path"], device)

    obs_pts = []
    obs_w = []
    joint_ids = []
    for coco_idx, smpl_idx in COCO_TO_SMPL_PAIRS:
        x, y, c = [float(v) for v in kpts[coco_idx]]
        if not np.isfinite(x) or not np.isfinite(y) or c < kp_conf_thr:
            continue
        obs_pts.append([x, y])
        obs_w.append(max(kp_conf_thr, c))
        joint_ids.append(smpl_idx)

    if len(obs_pts) < 6:
        raise RuntimeError("Not enough confident non-MediaPipe body joints on the first frame")

    first_mask = _load_mask(masks_dir, Path(first_cam["name"]).stem, (first_cam["W"], first_cam["H"]))
    bbox_np = _bbox_from_mask(first_mask)
    if bbox_np is None:
        bbox_np = _bbox_from_keypoints(kpts, kp_conf_thr)
    center_np = np.array([(bbox_np[0] + bbox_np[2]) * 0.5, (bbox_np[1] + bbox_np[3]) * 0.5], dtype=np.float32)

    init_trans = _initial_translation_from_bbox(kpts, first_cam["K"], first_cam["R"], first_cam["t"], kp_conf_thr)
    init_scale = 1.0

    obs = torch.from_numpy(np.asarray(obs_pts, np.float32)).to(device)
    wts = torch.from_numpy(np.asarray(obs_w, np.float32)).to(device)
    bbox_t = torch.from_numpy(bbox_np).to(device)
    center_t = torch.from_numpy(center_np).to(device)
    K_t = torch.from_numpy(first_cam["K"].astype(np.float32)).to(device)
    R_t = torch.from_numpy(first_cam["R"].astype(np.float32)).to(device)
    t_t = torch.from_numpy(first_cam["t"].astype(np.float32)).to(device)

    betas = nn.Parameter(torch.zeros(1, 10, device=device))
    global_orient = nn.Parameter(torch.zeros(1, 3, device=device))
    body_pose = nn.Parameter(torch.zeros(1, 69, device=device))
    trans = nn.Parameter(torch.from_numpy(init_trans).to(device).view(1, 3))
    log_scale = nn.Parameter(torch.tensor([math.log(init_scale)], dtype=torch.float32, device=device))

    optimizers = [
        (torch.optim.Adam([trans, log_scale], lr=0.03), max(80, n_iters // 6)),
        (torch.optim.Adam([global_orient, trans, log_scale], lr=0.02), max(120, n_iters // 4)),
        (torch.optim.Adam([global_orient, body_pose, betas, trans, log_scale], lr=0.01), n_iters),
    ]

    reproj_val = float("nan")
    for opt, stage_iters in optimizers:
        for _ in range(stage_iters):
            opt.zero_grad()
            scale = torch.exp(log_scale)
            pose = torch.cat([global_orient, body_pose], dim=1)
            verts, joints = smpl(betas, pose, trans, scale=scale)
            xy, z = _project_world(joints[0, joint_ids], K_t, R_t, t_t)
            diff = xy - obs
            reproj = (torch.sqrt((diff ** 2).sum(dim=1) + 4.0) * wts).mean()

            vxy, vz = _project_world(verts[0], K_t, R_t, t_t)
            good_v = vz > 0.01
            if torch.any(good_v):
                vv = vxy[good_v]
                bbox_pred = torch.stack([vv[:, 0].min(), vv[:, 1].min(), vv[:, 0].max(), vv[:, 1].max()])
            else:
                bbox_pred = bbox_t
            center_pred = 0.5 * (bbox_pred[:2] + bbox_pred[2:])

            bbox_loss = torch.abs(bbox_pred - bbox_t).mean()
            center_loss = torch.abs(center_pred - center_t).mean()
            reg = 0.05 * (global_orient ** 2).mean() + 0.15 * (body_pose ** 2).mean() + 0.08 * (betas ** 2).mean() + 0.02 * (log_scale ** 2).mean()
            loss = reproj + 0.01 * bbox_loss + 0.02 * center_loss + reg
            loss.backward()
            opt.step()
            reproj_val = float(reproj.detach().item())

    with torch.no_grad():
        scale = torch.exp(log_scale)
        pose = torch.cat([global_orient, body_pose], dim=1)
        verts, joints = smpl(betas, pose, trans, scale=scale)
        verts_np = verts[0].cpu().numpy().astype(np.float32)
        joints_np = joints[0].cpu().numpy().astype(np.float32)
        pose_np = pose[0].cpu().numpy().astype(np.float32)
        betas_np = betas[0].cpu().numpy().astype(np.float32)
        trans_np = trans[0].cpu().numpy().astype(np.float32)
        scale_np = float(scale.item())

    return {
        "verts": verts_np,
        "faces": smpl.faces.astype(np.int64),
        "joints": joints_np,
        "pose": pose_np,
        "betas": betas_np,
        "trans": trans_np,
        "scale": scale_np,
        "reproj_px": reproj_val,
        "det_score": det_score,
        "keypoints": kpts,
    }


def _build_texel_bary_data(uv_verts: np.ndarray, uv_faces: np.ndarray, vmapping: np.ndarray, tex_size: int):
    H = W = tex_size
    valid_mask = np.zeros((H, W), dtype=bool)
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vidx_all: list[np.ndarray] = []
    bary_all: list[np.ndarray] = []

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


def _camera_direction(cam: dict, center: np.ndarray) -> np.ndarray:
    cam_pos = -(cam["R"].T @ cam["t"])
    d = cam_pos - center
    return d / max(np.linalg.norm(d), 1e-8)


def select_front_back_views(cam_list: list[dict], first_cam: dict, center: np.ndarray) -> tuple[dict, dict]:
    front_cam = first_cam
    front_dir = _camera_direction(front_cam, center)
    best_back = None
    best_score = float("inf")
    for cam in cam_list:
        if cam["name"] == front_cam["name"] or not cam["img_path"].exists():
            continue
        d = _camera_direction(cam, center)
        score = float(np.dot(front_dir, d))
        if score < best_score:
            best_score = score
            best_back = cam
    if best_back is None:
        raise RuntimeError("Could not find a back-view frame")
    return front_cam, best_back


def bake_two_views(
    verts: np.ndarray,
    faces: np.ndarray,
    cams: list[dict],
    masks_dir: Path | None,
    tex_size: int,
    depth_tol: float,
    knn_fill_k: int,
):
    uv_verts, uv_faces, vmapping = unwrap_uv(verts, faces)
    valid_yx, vidx, bary, valid_mask = _build_texel_bary_data(uv_verts, uv_faces, vmapping, tex_size)

    N = len(valid_yx)
    top_colors = np.zeros((N, max(2, len(cams)), 3), dtype=np.float32)
    top_weights = np.full((N, max(2, len(cams))), -np.inf, dtype=np.float32)

    face_normals = _compute_face_normals(verts, faces)
    vert_normals = _compute_vertex_normals(verts, faces, face_normals)
    tex_pts = _interp_points(verts, vidx, bary)
    tex_nrm = _interp_points(vert_normals, vidx, bary)
    tex_nrm /= np.maximum(np.linalg.norm(tex_nrm, axis=1, keepdims=True), 1e-8)

    for ci, cam in enumerate(cams):
        K, R, t, W, H = cam["K"], cam["R"], cam["t"], cam["W"], cam["H"]
        img = cv2.imread(str(cam["img_path"]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))

        mask = _load_mask(masks_dir, Path(cam["name"]).stem, (W, H))
        depth_buf = _rasterise_depth(verts, faces, K, R, t, W, H)
        proj_uv, proj_z = _project_points(tex_pts, K, R, t)
        pu = proj_uv[:, 0]
        pv = proj_uv[:, 1]
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

        good = weight > 0
        if not good.any():
            continue

        pu_f = pu[in_bounds][good]
        pv_f = pv[in_bounds][good]
        pu0 = pu_f.astype(np.int32).clip(0, W - 2)
        pv0 = pv_f.astype(np.int32).clip(0, H - 2)
        du = (pu_f - pu0).clip(0, 1)
        dv = (pv_f - pv0).clip(0, 1)
        c00 = img[pv0, pu0]
        c10 = img[pv0 + 1, pu0]
        c01 = img[pv0, pu0 + 1]
        c11 = img[pv0 + 1, pu0 + 1]
        color = ((1 - dv[:, None]) * (1 - du[:, None]) * c00 +
                 dv[:, None] * (1 - du[:, None]) * c10 +
                 (1 - dv[:, None]) * du[:, None] * c01 +
                 dv[:, None] * du[:, None] * c11)

        valid_ids = np.where(in_bounds)[0][good]
        mslot = np.argmin(top_weights[valid_ids], axis=1)
        cur = top_weights[valid_ids, mslot]
        update = weight[good] > cur
        if update.any():
            ui = valid_ids[update]
            us = mslot[update]
            top_weights[ui, us] = weight[good][update]
            top_colors[ui, us] = color[update]

    wsum = np.maximum(top_weights, 0).sum(axis=1, keepdims=True)
    has = wsum[:, 0] > 0
    blended = np.zeros((N, 3), dtype=np.float32)
    blended[has] = ((np.maximum(top_weights[has], 0)[:, :, None] * top_colors[has]).sum(axis=1) / wsum[has])

    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    cov = np.zeros((tex_size, tex_size), dtype=np.float32)
    tex[valid_yx[:, 0], valid_yx[:, 1]] = blended
    cov[valid_yx[:, 0], valid_yx[:, 1]] = has.astype(np.float32)
    tex, hallucinated = _fill_uncovered_texels_knn(tex, cov, valid_mask, knn_fill_k)
    tex_u8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    tex_u8 = _dilate_atlas(tex_u8, valid_mask, n_iters=8)
    return tex_u8, cov, hallucinated.astype(np.uint8), uv_verts, uv_faces, vmapping


def main():
    ap = argparse.ArgumentParser(description="Fit SMPL from the first frame using a non-MediaPipe detector, then texture with front/back views")
    ap.add_argument("--smpl-model", required=True)
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tex-size", type=int, default=2048)
    ap.add_argument("--depth-tol", type=float, default=0.025)
    ap.add_argument("--knn-fill-k", type=int, default=4)
    ap.add_argument("--kp-conf-thr", type=float, default=0.2)
    ap.add_argument("--fit-iters", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = _read_colmap_cameras_txt(Path(args.colmap_dir) / "cameras.txt")
    imgs = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")
    cam_list = _build_cameras_named(cams, imgs, args.frames_dir)
    valid_cams = [c for c in cam_list if c["img_path"].exists()]
    if not valid_cams:
        raise RuntimeError("No registered frame images found")

    first_cam = sorted(valid_cams, key=lambda c: c["name"])[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = fit_first_frame_smpl(
        smpl_model=Path(args.smpl_model),
        first_cam=first_cam,
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        device=device,
        kp_conf_thr=args.kp_conf_thr,
        n_iters=args.fit_iters,
    )

    verts = fit["verts"]
    faces = fit["faces"]
    center = fit["joints"][0]
    front_cam, back_cam = select_front_back_views(valid_cams, first_cam, center)

    tex_u8, cov, hallucinated, uv_verts, uv_faces, vmapping = bake_two_views(
        verts=verts,
        faces=faces,
        cams=[front_cam, back_cam],
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        tex_size=args.tex_size,
        depth_tol=args.depth_tol,
        knn_fill_k=args.knn_fill_k,
    )

    np.save(str(out_dir / "betas.npy"), fit["betas"])
    np.save(str(out_dir / "pose_first_frame.npy"), fit["pose"])
    np.save(str(out_dir / "trans_first_frame.npy"), fit["trans"])
    np.save(str(out_dir / "scale_first_frame.npy"), np.array([fit["scale"]], dtype=np.float32))
    np.save(str(out_dir / "first_frame_keypoints_coco17.npy"), fit["keypoints"])

    tex_name = "smpl_texture_front_back.png"
    Image.fromarray(tex_u8).save(str(out_dir / tex_name))
    Image.fromarray((np.clip(cov, 0, 1) * 255).astype(np.uint8)).save(str(out_dir / "coverage_front_back.png"))
    Image.fromarray((hallucinated * 255).astype(np.uint8)).save(str(out_dir / "hallucinated_fill_mask.png"))
    export_obj_with_uv(verts, faces, uv_verts, uv_faces, vmapping, tex_name, out_dir / "smpl_textured_front_back.obj")
    export_obj_with_uv(verts, faces, *unwrap_uv(verts, faces), tex_name, out_dir / "smpl_first_frame_mesh.obj")

    fit_info = {
        "detector": "torchvision_keypointrcnn_coco17",
        "first_frame": first_cam["name"],
        "front_view": front_cam["name"],
        "back_view": back_cam["name"],
        "first_frame_detection_score": fit["det_score"],
        "first_frame_reproj_px": fit["reproj_px"],
        "scale": fit["scale"],
        "tex_size": args.tex_size,
        "knn_fill_k": args.knn_fill_k,
    }
    (out_dir / "fit_info.json").write_text(json.dumps(fit_info, indent=2), encoding="utf-8")
    print(json.dumps(fit_info, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
