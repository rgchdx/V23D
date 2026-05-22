from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

from src.pose.extract_mediapipe import extract_joint_observations, load_landmarks_json
from src.recon.smpl_fitter import (
    MP_TO_SMPL_PAIRS,
    SMPL,
    _build_K,
    _precompute_silhouette_dt,
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
    save_mesh_obj,
)


def _sample_names(names: list[str], n: int) -> list[str]:
    if n <= 0 or len(names) <= n:
        return names
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    return [names[i] for i in idx]


def _precompute_mask_stats(frame_names: list[str], masks_dir: Path, img_hw: tuple[int, int]):
    stats = {}
    H, W = img_hw
    for name in frame_names:
        stem = Path(name).stem
        mask_arr = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = masks_dir / f"{stem}{ext}"
            if p.exists():
                mask_arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                break
        if mask_arr is None:
            stats[name] = None
            continue
        if mask_arr.shape[:2] != (H, W):
            mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)
        fg = mask_arr > 127
        ys, xs = np.where(fg)
        if len(xs) == 0:
            stats[name] = None
            continue
        stats[name] = {
            "bbox": np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
            "centroid": np.array([xs.mean(), ys.mean()], dtype=np.float32),
            "area": float(len(xs)),
        }
    return stats


def _silhouette_dt_loss(verts_world: torch.Tensor, proj_mat: torch.Tensor, dt_tensor: torch.Tensor | None, vert_idx: np.ndarray) -> torch.Tensor:
    if dt_tensor is None or len(vert_idx) == 0:
        return verts_world.new_tensor(0.0)
    H_dt, W_dt = dt_tensor.shape
    v3 = verts_world[vert_idx]
    v4 = torch.cat([v3, torch.ones(v3.shape[0], 1, device=v3.device, dtype=v3.dtype)], dim=1)
    proj = (proj_mat @ v4.T).T
    z = proj[:, 2]
    valid = z > 0.01
    if not torch.any(valid):
        return verts_world.new_tensor(0.0)
    xy = proj[valid, :2] / z[valid, None].clamp(min=0.01)
    xn = (xy[:, 0] / max(W_dt - 1, 1)) * 2.0 - 1.0
    yn = (xy[:, 1] / max(H_dt - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([xn, yn], dim=1).view(1, 1, -1, 2)
    vals = F.grid_sample(dt_tensor.view(1, 1, H_dt, W_dt), grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)
    return vals.mean()


def _silhouette_bbox_loss(verts_world: torch.Tensor, proj_mat: torch.Tensor, mask_stats: dict[str, np.ndarray | float] | None, vert_idx: np.ndarray, image_hw: tuple[int, int]) -> torch.Tensor:
    if mask_stats is None or len(vert_idx) == 0:
        return verts_world.new_tensor(0.0)
    H, W = image_hw
    v3 = verts_world[vert_idx]
    v4 = torch.cat([v3, torch.ones(v3.shape[0], 1, device=v3.device, dtype=v3.dtype)], dim=1)
    proj = (proj_mat @ v4.T).T
    z = proj[:, 2]
    valid = z > 0.01
    if not torch.any(valid):
        return verts_world.new_tensor(0.0)
    xy = proj[valid, :2] / z[valid, None].clamp(min=0.01)
    pred_min = xy.min(dim=0).values
    pred_max = xy.max(dim=0).values
    pred_ctr = xy.mean(dim=0)
    pred_area = torch.clamp(pred_max[0] - pred_min[0], min=1.0) * torch.clamp(pred_max[1] - pred_min[1], min=1.0)

    tgt_bbox = torch.from_numpy(np.asarray(mask_stats["bbox"], dtype=np.float32)).to(verts_world.device)
    tgt_ctr = torch.from_numpy(np.asarray(mask_stats["centroid"], dtype=np.float32)).to(verts_world.device)
    tgt_area = torch.tensor(float(mask_stats["area"]), dtype=verts_world.dtype, device=verts_world.device)
    norm_bbox = torch.tensor([W, H, W, H], dtype=verts_world.dtype, device=verts_world.device)
    norm_ctr = torch.tensor([W, H], dtype=verts_world.dtype, device=verts_world.device)
    bbox_loss = torch.abs(torch.cat([pred_min, pred_max]) - tgt_bbox).div(norm_bbox).mean()
    ctr_loss = torch.abs(pred_ctr - tgt_ctr).div(norm_ctr).mean()
    area_loss = torch.abs(torch.log(pred_area.clamp(min=1.0)) - torch.log(tgt_area.clamp(min=1.0)))
    return bbox_loss + ctr_loss + 0.25 * area_loss


def main():
    ap = argparse.ArgumentParser(description="Optimize a single shared SMPL body shape from all fitted frames")
    ap.add_argument("--smpl-model", default=r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--smpl-out", required=True, help="Folder with initial betas.npy and smpl_canonical.obj")
    ap.add_argument("--rigid-out", required=True, help="Folder with poses_per_frame.json, trans_per_frame.npy, scale_per_frame.json")
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--landmarks-json", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--n-iters", type=int, default=250)
    ap.add_argument("--min-visibility", type=float, default=0.4)
    ap.add_argument("--lambda-joint", type=float, default=1.0)
    ap.add_argument("--lambda-beta", type=float, default=4.0)
    ap.add_argument("--lambda-sil", type=float, default=0.003)
    ap.add_argument("--lambda-sil-bbox", type=float, default=0.05)
    ap.add_argument("--n-sil-verts", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    smpl_out = Path(args.smpl_out)
    rigid_out = Path(args.rigid_out)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    betas_init = np.load(str(smpl_out / "betas.npy")).astype(np.float32)
    poses = json.loads((rigid_out / "poses_per_frame.json").read_text(encoding="utf-8"))
    trans = np.load(str(rigid_out / "trans_per_frame.npy"), allow_pickle=True).item()
    scale_json = rigid_out / "scale_per_frame.json"
    scales = json.loads(scale_json.read_text(encoding="utf-8")) if scale_json.exists() else {}
    landmarks = load_landmarks_json(args.landmarks_json)
    cams = _read_colmap_cameras_txt(Path(args.colmap_dir) / "cameras.txt")
    images = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")

    names = sorted([n for n in poses.keys() if n in trans and n in images and landmarks.get(n) is not None])
    names = _sample_names(names, args.max_frames)
    if not names:
        raise RuntimeError("No valid frames found")

    model = SMPL(args.smpl_model, n_betas=len(betas_init)).to(dev)
    model.eval()
    betas = nn.Parameter(torch.from_numpy(betas_init).to(dev).unsqueeze(0))
    poses_t = torch.stack([torch.from_numpy(np.asarray(poses[n], dtype=np.float32)) for n in names]).to(dev)
    trans_t = torch.stack([torch.from_numpy(np.asarray(trans[n], dtype=np.float32)) for n in names]).to(dev)
    scales_t = torch.tensor([float(scales.get(n, 1.0)) for n in names], dtype=torch.float32, device=dev)

    obs = []
    proj_mats = []
    dt_maps = {}
    mask_stats = {}
    first_info = images[names[0]]
    cam0 = cams[first_info["cam_id"]]
    image_hw = (cam0["h"], cam0["w"])
    if args.masks_dir and Path(args.masks_dir).exists():
        dt_np = _precompute_silhouette_dt(names, Path(args.masks_dir), image_hw)
        dt_maps = {k: torch.from_numpy(v.astype(np.float32)).to(dev) if v is not None else None for k, v in dt_np.items()}
        mask_stats = _precompute_mask_stats(names, Path(args.masks_dir), image_hw)
    rng = np.random.default_rng(0)
    sil_vert_idx = rng.choice(model.n_verts, size=min(args.n_sil_verts, model.n_verts), replace=False)

    for n in names:
        o = extract_joint_observations(landmarks[n], MP_TO_SMPL_PAIRS, min_visibility=args.min_visibility)
        obs.append(o)
        info = images[n]
        K = torch.from_numpy(_build_K(cams[info["cam_id"]]).astype(np.float32)).to(dev)
        R = torch.from_numpy(info["R"].astype(np.float32)).to(dev)
        t = torch.from_numpy(info["t"].astype(np.float32)).to(dev)
        proj_mats.append(K @ torch.cat([R, t.unsqueeze(1)], dim=1))

    opt = torch.optim.Adam([betas], lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_iters, eta_min=1e-4)

    for it in range(args.n_iters):
        opt.zero_grad()
        verts, joints = model(betas.expand(len(names), -1), poses_t, torch.zeros(len(names), 3, device=dev))
        verts_w = verts * scales_t[:, None, None] + trans_t[:, None, :]
        joints_w = joints * scales_t[:, None, None] + trans_t[:, None, :]
        total = betas.new_tensor(0.0)
        for i, n in enumerate(names):
            o = obs[i]
            if len(o["smpl_idx"]) > 0:
                smpl_ids = o["smpl_idx"].tolist()
                j_sel = joints_w[i, smpl_ids]
                j4 = torch.cat([j_sel, torch.ones(j_sel.shape[0], 1, device=dev)], dim=1)
                pr = (proj_mats[i] @ j4.T).T
                xy = pr[:, :2] / pr[:, 2:3].clamp(min=0.01)
                tgt_xy = torch.from_numpy(o["xy"]).to(dev)
                tgt_w = torch.from_numpy(o["conf"]).to(dev)
                joint_loss = (torch.sqrt(((xy - tgt_xy) ** 2).sum(dim=1) + 4.0) * tgt_w).mean()
            else:
                joint_loss = total.new_tensor(0.0)
            sil_loss = _silhouette_dt_loss(verts_w[i], proj_mats[i], dt_maps.get(n), sil_vert_idx)
            bbox_loss = _silhouette_bbox_loss(verts_w[i], proj_mats[i], mask_stats.get(n), sil_vert_idx, image_hw)
            total = total + args.lambda_joint * joint_loss + args.lambda_sil * sil_loss + args.lambda_sil_bbox * bbox_loss
        total = total / max(len(names), 1) + args.lambda_beta * (betas ** 2).mean()
        total.backward()
        opt.step()
        sched.step()
        if it % 25 == 0 or it == args.n_iters - 1:
            print({"iter": it, "total": float(total.detach().cpu()), "beta_norm": float((betas ** 2).mean().sqrt().detach().cpu())})

    with torch.no_grad():
        zero_pose = torch.zeros(1, 72, device=dev)
        zero_trans = torch.zeros(1, 3, device=dev)
        verts_can, _ = model(betas, zero_pose, zero_trans)

    np.save(str(out_dir / "betas.npy"), betas[0].detach().cpu().numpy())
    save_mesh_obj(verts_can[0].detach().cpu().numpy(), model.faces, out_dir / "smpl_canonical.obj")
    if (smpl_out / "landmarks_mediapipe.json").exists():
        (out_dir / "landmarks_mediapipe.json").write_text((smpl_out / "landmarks_mediapipe.json").read_text(encoding="utf-8"), encoding="utf-8")
    if (rigid_out / "scale.npy").exists():
        np.save(str(out_dir / "scale.npy"), np.load(str(rigid_out / "scale.npy")))
    print(f"Saved single-shape SMPL output -> {out_dir}")


if __name__ == "__main__":
    main()
