from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _numpy_compat_for_chumpy() -> None:
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "complex"):
        np.complex = complex
    if not hasattr(np, "object"):
        np.object = object
    if not hasattr(np, "str"):
        np.str = str
    if not hasattr(np, "unicode"):
        np.unicode = str


def _load_mask(masks_dir: Path, stem: str, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    for ext in (".png", ".jpg", ".jpeg"):
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 127).astype(np.uint8)
    raise FileNotFoundError(f"Mask not found for {stem} in {masks_dir}")


def _signed_dt(mask_u8: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    return (dist_out - dist_in).astype(np.float32)


def _sample_map(map_2d: torch.Tensor, uv: torch.Tensor, w: int, h: int) -> torch.Tensor:
    # map_2d: (1,1,H,W), uv: (N,2) in pixels
    x = (uv[:, 0] / max(w - 1, 1)) * 2.0 - 1.0
    y = (uv[:, 1] / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=1).view(1, -1, 1, 2)
    vals = torch.nn.functional.grid_sample(map_2d, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return vals.view(-1)


def _project(pts: torch.Tensor, k: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pts_cam = (r @ pts.T).T + t.view(1, 3)
    z = pts_cam[:, 2]
    uvw = (k @ pts_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-6)
    return uv, z


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = []
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {int(f[0])+1} {int(f[1])+1} {int(f[2])+1}\n")
    path.write_text("".join(lines), encoding="utf-8")


BODY25_TO_SMPL = [
    (5, 16),
    (2, 17),
    (6, 18),
    (3, 19),
    (7, 20),
    (4, 21),
    (12, 1),
    (9, 2),
    (13, 4),
    (10, 5),
    (14, 7),
    (11, 8),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine SMPLify-X (SMPL mode) results with mask-guided losses and strong priors")
    ap.add_argument("--run-dir", required=True, help="SMPLify-X run dir containing images/, keypoints/, model_folder/, smplifyx_output/")
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iters", type=int, default=220)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _numpy_compat_for_chumpy()
    import smplx

    model_folder = run_dir / "model_folder"
    images_dir = run_dir / "images"
    keypoints_dir = run_dir / "keypoints"
    smplifyx_out = run_dir / "smplifyx_output"

    info_path = run_dir / "run_info.json"
    focal = 5000.0
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        focal = float(info.get("focal_length", 5000.0))

    pkl_paths = sorted((smplifyx_out / "results").glob("*/*.pkl"))
    if not pkl_paths:
        raise RuntimeError(f"No pkl files under {smplifyx_out / 'results'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = smplx.create(str(model_folder), model_type="smpl", gender="neutral", use_pca=False, create_transl=False).to(device)
    body_model.eval()
    faces = np.asarray(body_model.faces, dtype=np.int64)

    summary = {}

    for pkl_path in pkl_paths:
        stem = pkl_path.parent.name
        img_candidates = list(images_dir.glob(stem + ".*"))
        if not img_candidates:
            continue
        img_path = img_candidates[0]
        kp_path = keypoints_dir / f"{stem}_keypoints.json"
        if not kp_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        mask = _load_mask(masks_dir, stem, (h, w))
        dt = _signed_dt(mask)

        mask_t = torch.from_numpy(mask.astype(np.float32)).to(device).view(1, 1, h, w)
        dt_t = torch.from_numpy(dt.astype(np.float32)).to(device).view(1, 1, h, w)

        kp = json.loads(kp_path.read_text(encoding="utf-8"))
        people = kp.get("people", [])
        if not people:
            continue
        body25 = np.asarray(people[0]["pose_keypoints_2d"], dtype=np.float32).reshape(25, 3)

        p = pickle.load(open(pkl_path, "rb"), encoding="latin1")
        betas0 = torch.tensor(p["betas"], dtype=torch.float32, device=device)
        body_pose0 = torch.tensor(p["body_pose"], dtype=torch.float32, device=device)
        global_orient0 = torch.tensor(p["global_orient"], dtype=torch.float32, device=device)
        cam_r = torch.tensor(np.asarray(p["camera_rotation"], dtype=np.float32).reshape(3, 3), dtype=torch.float32, device=device)
        cam_t0 = torch.tensor(np.asarray(p["camera_translation"], dtype=np.float32).reshape(3), dtype=torch.float32, device=device)

        betas = nn.Parameter(betas0.clone())
        body_pose = nn.Parameter(body_pose0.clone())
        global_orient = nn.Parameter(global_orient0.clone())
        cam_t = nn.Parameter(cam_t0.clone())

        opt = torch.optim.Adam([betas, body_pose, global_orient, cam_t], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=args.lr * 0.1)

        k = torch.tensor([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        obs_uv = []
        obs_w = []
        obs_sj = []
        for bi, sj in BODY25_TO_SMPL:
            x, y, c = [float(v) for v in body25[bi]]
            if not np.isfinite(x) or not np.isfinite(y) or c <= 0.05:
                continue
            obs_uv.append([x, y])
            obs_w.append(max(0.05, min(1.0, c)))
            obs_sj.append(sj)

        if len(obs_uv) < 5:
            continue

        obs_uv_t = torch.tensor(np.asarray(obs_uv, dtype=np.float32), device=device)
        obs_w_t = torch.tensor(np.asarray(obs_w, dtype=np.float32), device=device)
        obs_sj_t = torch.tensor(np.asarray(obs_sj, dtype=np.int64), device=device)

        mask_bbox = np.where(mask > 0)
        if len(mask_bbox[0]) > 0:
            mx0, my0 = float(mask_bbox[1].min()), float(mask_bbox[0].min())
            mx1, my1 = float(mask_bbox[1].max()), float(mask_bbox[0].max())
            mask_bbox_t = torch.tensor([mx0, my0, mx1, my1], dtype=torch.float32, device=device)
        else:
            mask_bbox_t = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32, device=device)

        for _ in range(args.iters):
            opt.zero_grad()

            out = body_model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)
            verts = out.vertices[0]
            joints = out.joints[0]

            j_obs = joints[obs_sj_t]
            uv_j, z_j = _project(j_obs, k, cam_r, cam_t)
            reproj = torch.sqrt(((uv_j - obs_uv_t) ** 2).sum(dim=1) + 4.0)
            loss_reproj = (reproj * obs_w_t).mean()

            mask_at_j = _sample_map(mask_t, uv_j, w, h)
            loss_joint_inside = ((1.0 - mask_at_j) * obs_w_t).mean()

            uv_v, z_v = _project(verts, k, cam_r, cam_t)
            valid_v = z_v > 1e-4
            uv_vv = uv_v[valid_v]
            if len(uv_vv) == 0:
                loss_occ = torch.tensor(0.0, device=device)
                loss_out = torch.tensor(0.0, device=device)
                loss_bbox = torch.tensor(0.0, device=device)
            else:
                mask_at_v = _sample_map(mask_t, uv_vv, w, h)
                sdt_at_v = _sample_map(dt_t, uv_vv, w, h)
                loss_occ = (1.0 - mask_at_v).mean()
                loss_out = torch.relu(sdt_at_v).mean() / max(h, w)

                bx0 = uv_vv[:, 0].min()
                by0 = uv_vv[:, 1].min()
                bx1 = uv_vv[:, 0].max()
                by1 = uv_vv[:, 1].max()
                pred_bbox = torch.stack([bx0, by0, bx1, by1])
                loss_bbox = torch.abs(pred_bbox - mask_bbox_t).mean() / max(h, w)

            loss_prior = (
                0.60 * (body_pose ** 2).mean()
                + 0.80 * (betas ** 2).mean()
                + 0.06 * ((cam_t - cam_t0) ** 2).mean()
                + 0.04 * (global_orient ** 2).mean()
            )

            loss = (
                1.00 * loss_reproj
                + 2.50 * loss_joint_inside
                + 1.60 * loss_occ
                + 2.00 * loss_out
                + 0.80 * loss_bbox
                + loss_prior
            )
            loss.backward()
            opt.step()
            sched.step()

        with torch.no_grad():
            out = body_model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)
            verts = out.vertices[0].cpu().numpy().astype(np.float32)
            joints = out.joints[0]
            uv_j, _ = _project(joints[obs_sj_t], k, cam_r, cam_t)
            mask_at_j = _sample_map(mask_t, uv_j, w, h)
            inside = float((mask_at_j > 0.5).float().mean().item())

        frame_out = out_dir / stem
        frame_out.mkdir(parents=True, exist_ok=True)

        refined_p = {
            "betas": betas.detach().cpu().numpy(),
            "body_pose": body_pose.detach().cpu().numpy(),
            "global_orient": global_orient.detach().cpu().numpy(),
            "camera_rotation": cam_r.detach().cpu().numpy(),
            "camera_translation": cam_t.detach().cpu().numpy(),
        }
        with open(frame_out / "refined.pkl", "wb") as f:
            pickle.dump(refined_p, f)

        _write_obj(frame_out / "refined.obj", verts, faces)

        vis = img.copy()
        uvj_np = uv_j.detach().cpu().numpy()
        for q in uvj_np:
            x, y = int(round(float(q[0]))), int(round(float(q[1])))
            if 0 <= x < w and 0 <= y < h:
                color = (0, 255, 0) if mask[y, x] > 0 else (0, 0, 255)
                cv2.circle(vis, (x, y), 4, color, -1, cv2.LINE_AA)

        cv2.putText(vis, f"{stem}  joints-inside={inside:.3f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(frame_out / "joints_overlay_refined.jpg"), vis)

        summary[stem] = {
            "joints_inside_ratio": inside,
            "refined_obj": str(frame_out / "refined.obj"),
            "overlay": str(frame_out / "joints_overlay_refined.jpg"),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved refinement outputs -> {out_dir}")


if __name__ == "__main__":
    main()
