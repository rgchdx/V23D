from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


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
    return np.zeros((h, w), dtype=np.uint8)


def _signed_dt(mask_u8: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    return (dist_out - dist_in).astype(np.float32)


def _sample_map(map_2d: torch.Tensor, uv: torch.Tensor, w: int, h: int) -> torch.Tensor:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="SMPLify-X per-frame init -> temporal/shared-shape bundle refinement")
    ap.add_argument("--run-dir", required=True, help="Folder with images/, keypoints/, smplifyx_output/, model_folder/")
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iters", type=int, default=180)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--sample-verts", type=int, default=1400)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _numpy_compat_for_chumpy()
    import smplx

    images_dir = run_dir / "images"
    keypoints_dir = run_dir / "keypoints"
    model_folder = run_dir / "model_folder"
    results_dir = run_dir / "smplifyx_output" / "results"

    info = {}
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    focal = float(info.get("focal_length", 5000.0))

    stems = sorted([p.name for p in results_dir.glob("*") if p.is_dir()])
    if not stems:
        raise RuntimeError(f"No frame result folders found in {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = smplx.create(str(model_folder), model_type="smpl", gender="neutral", use_pca=False, create_transl=False).to(device)
    body_model.eval()
    faces = np.asarray(body_model.faces, dtype=np.int64)

    betas_init = []
    body_pose_init = []
    global_orient_init = []
    cam_t_init = []
    cam_r_fixed = []

    obs_uv = []
    obs_w = []
    obs_sj = []
    masks_t = []
    dts_t = []
    image_hw = []
    mask_box = []

    valid_stems = []
    for stem in stems:
        pkl_path = results_dir / stem / "000.pkl"
        kp_path = keypoints_dir / f"{stem}_keypoints.json"
        img_candidates = list(images_dir.glob(stem + ".*"))
        if (not pkl_path.exists()) or (not kp_path.exists()) or (not img_candidates):
            continue

        img = cv2.imread(str(img_candidates[0]))
        if img is None:
            continue
        h, w = img.shape[:2]

        p = pickle.load(open(pkl_path, "rb"), encoding="latin1")
        kp = json.loads(kp_path.read_text(encoding="utf-8"))
        people = kp.get("people", [])
        if not people:
            continue
        body25 = np.asarray(people[0]["pose_keypoints_2d"], dtype=np.float32).reshape(25, 3)

        uv_i = []
        w_i = []
        sj_i = []
        for bi, sj in BODY25_TO_SMPL:
            x, y, c = [float(v) for v in body25[bi]]
            if not np.isfinite(x) or not np.isfinite(y) or c <= 0.05:
                continue
            uv_i.append([x, y])
            w_i.append(max(0.05, min(1.0, c)))
            sj_i.append(sj)
        if len(uv_i) < 5:
            continue

        mask = _load_mask(masks_dir, stem, (h, w))
        dt = _signed_dt(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            mw = float(x1 - x0 + 1.0)
            mh = float(y1 - y0 + 1.0)
            ma = float(len(xs))
        else:
            x0, y0, x1, y1 = 0.0, 0.0, float(w - 1), float(h - 1)
            mw, mh, ma = float(w), float(h), float(w * h)

        betas_init.append(np.asarray(p["betas"], dtype=np.float32).reshape(-1))
        body_pose_init.append(np.asarray(p["body_pose"], dtype=np.float32).reshape(-1))
        global_orient_init.append(np.asarray(p["global_orient"], dtype=np.float32).reshape(-1))
        cam_t_init.append(np.asarray(p["camera_translation"], dtype=np.float32).reshape(-1))
        cam_r_fixed.append(np.asarray(p["camera_rotation"], dtype=np.float32).reshape(3, 3))

        obs_uv.append(torch.tensor(np.asarray(uv_i, dtype=np.float32), device=device))
        obs_w.append(torch.tensor(np.asarray(w_i, dtype=np.float32), device=device))
        obs_sj.append(torch.tensor(np.asarray(sj_i, dtype=np.int64), device=device))
        masks_t.append(torch.from_numpy(mask.astype(np.float32)).to(device).view(1, 1, h, w))
        dts_t.append(torch.from_numpy(dt.astype(np.float32)).to(device).view(1, 1, h, w))
        image_hw.append((h, w))
        mask_box.append((x0, y0, x1, y1, mw, mh, ma))
        valid_stems.append(stem)

    B = len(valid_stems)
    if B == 0:
        raise RuntimeError("No valid frames available for bundle refinement")

    betas0 = torch.from_numpy(np.mean(np.stack(betas_init, axis=0), axis=0)).to(device).unsqueeze(0)
    betas0_clip = torch.clamp(betas0, min=-2.0, max=2.0)
    body_pose0 = torch.from_numpy(np.stack(body_pose_init, axis=0)).to(device)
    global_orient0 = torch.from_numpy(np.stack(global_orient_init, axis=0)).to(device)

    betas_raw = nn.Parameter(torch.atanh(torch.clamp(betas0_clip / 2.0, min=-0.95, max=0.95)))
    body_pose = nn.Parameter(body_pose0.clone())
    global_orient = nn.Parameter(global_orient0.clone())
    cam_t = nn.Parameter(torch.from_numpy(np.stack(cam_t_init, axis=0)).to(device))

    cam_t0 = torch.from_numpy(np.stack(cam_t_init, axis=0)).to(device)
    cam_r = torch.from_numpy(np.stack(cam_r_fixed, axis=0)).to(device)

    rng = np.random.default_rng(0)
    sv = min(args.sample_verts, int(body_model.faces.max() + 1))
    sv_idx_np = rng.choice(int(body_model.faces.max() + 1), sv, replace=False)
    sv_idx = torch.tensor(sv_idx_np, dtype=torch.long, device=device)

    opt = torch.optim.Adam([betas_raw, body_pose, global_orient, cam_t], lr=args.lr * 0.6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=args.lr * 0.1)

    logs = []
    for it in range(args.iters):
        opt.zero_grad()

        out = body_model(
            betas=(2.0 * torch.tanh(betas_raw)).expand(B, -1),
            body_pose=body_pose,
            global_orient=global_orient,
            return_verts=True,
        )
        verts = out.vertices
        joints = out.joints

        loss_reproj = torch.tensor(0.0, device=device)
        loss_joint_in = torch.tensor(0.0, device=device)
        loss_occ = torch.tensor(0.0, device=device)
        loss_out = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)

        for i in range(B):
            h, w = image_hw[i]
            k = torch.tensor(
                [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )

            j_obs = joints[i, obs_sj[i]]
            uv_j, _ = _project(j_obs, k, cam_r[i], cam_t[i])
            reproj = torch.sqrt(((uv_j - obs_uv[i]) ** 2).sum(dim=1) + 4.0)
            loss_reproj = loss_reproj + (reproj * obs_w[i]).mean()

            mask_at_j = _sample_map(masks_t[i], uv_j, w, h)
            loss_joint_in = loss_joint_in + ((1.0 - mask_at_j) * obs_w[i]).mean()

            uv_v, z_v = _project(verts[i, sv_idx], k, cam_r[i], cam_t[i])
            valid_v = z_v > 1e-4
            if valid_v.any():
                uv_vv = uv_v[valid_v]
                mask_at_v = _sample_map(masks_t[i], uv_vv, w, h)
                sdt_at_v = _sample_map(dts_t[i], uv_vv, w, h)
                loss_occ = loss_occ + (1.0 - mask_at_v).mean()
                loss_out = loss_out + (torch.relu(sdt_at_v).mean() / max(h, w))

                bx0 = uv_vv[:, 0].min()
                by0 = uv_vv[:, 1].min()
                bx1 = uv_vv[:, 0].max()
                by1 = uv_vv[:, 1].max()
                bwh = torch.stack([(bx1 - bx0).clamp(min=1.0), (by1 - by0).clamp(min=1.0)])
                barea = bwh[0] * bwh[1]
                x0, y0, x1, y1, mw, mh, ma = mask_box[i]
                tgt_min = torch.tensor([x0, y0], device=device)
                tgt_max = torch.tensor([x1, y1], device=device)
                tgt_wh = torch.tensor([mw, mh], device=device)
                tgt_area = torch.tensor(ma, device=device)
                norm = torch.tensor([w, h], device=device, dtype=torch.float32)
                loss_box = loss_box + (torch.abs(torch.stack([bx0, by0]) - tgt_min) / norm).mean()
                loss_box = loss_box + (torch.abs(torch.stack([bx1, by1]) - tgt_max) / norm).mean()
                loss_box = loss_box + 0.75 * (torch.abs(bwh - tgt_wh) / norm).mean()
                loss_box = loss_box + 0.25 * torch.abs(torch.log(barea.clamp(min=1.0)) - torch.log(tgt_area.clamp(min=1.0)))

        loss_reproj = loss_reproj / B
        loss_joint_in = loss_joint_in / B
        loss_occ = loss_occ / B
        loss_out = loss_out / B
        loss_box = loss_box / B

        betas_eff = 2.0 * torch.tanh(betas_raw)
        loss_shape = ((betas_eff - betas0_clip) ** 2).mean() + 0.5 * (betas_eff ** 2).mean()
        loss_pose = ((body_pose - body_pose0) ** 2).mean() + 0.2 * ((global_orient - global_orient0) ** 2).mean()
        loss_cam = ((cam_t - cam_t0) ** 2).mean()
        loss_bounds = 0.25 * torch.relu(torch.abs(body_pose) - 2.2).mean()

        if B > 1:
            loss_temp = (
                torch.abs(body_pose[1:] - body_pose[:-1]).mean()
                + 0.6 * torch.abs(global_orient[1:] - global_orient[:-1]).mean()
                + 0.7 * torch.abs(cam_t[1:] - cam_t[:-1]).mean()
            )
        else:
            loss_temp = torch.tensor(0.0, device=device)

        loss = (
            1.00 * loss_reproj
            + 2.30 * loss_joint_in
            + 1.40 * loss_occ
            + 2.00 * loss_out
            + 1.10 * loss_box
            + 1.40 * loss_shape
            + 0.80 * loss_pose
            + 0.05 * loss_cam
            + 0.90 * loss_temp
            + 4.00 * loss_bounds
        )

        loss.backward()
        opt.step()
        sched.step()

        if (it % 20 == 0) or (it == args.iters - 1):
            row = {
                "iter": int(it),
                "total": float(loss.detach().cpu()),
                "reproj": float(loss_reproj.detach().cpu()),
                "joint_inside": float(loss_joint_in.detach().cpu()),
                "occ": float(loss_occ.detach().cpu()),
                "outside": float(loss_out.detach().cpu()),
                "box": float(loss_box.detach().cpu()),
                "shape": float(loss_shape.detach().cpu()),
                "temp": float(loss_temp.detach().cpu()),
            }
            logs.append(row)
            print(row)

    bundle_dir = out_dir / "bundle_refined"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        out = body_model(
            betas=(2.0 * torch.tanh(betas_raw)).expand(B, -1),
            body_pose=body_pose,
            global_orient=global_orient,
            return_verts=True,
        )
        verts = out.vertices
        joints = out.joints

    inside_ratios = {}
    for i, stem in enumerate(valid_stems):
        h, w = image_hw[i]
        k = torch.tensor(
            [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        uv_j, _ = _project(joints[i, obs_sj[i]], k, cam_r[i], cam_t[i])
        mask_at_j = _sample_map(masks_t[i], uv_j, w, h)
        inside = float((mask_at_j > 0.5).float().mean().item())
        inside_ratios[stem] = inside

        frame_dir = bundle_dir / stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        verts_np = verts[i].detach().cpu().numpy().astype(np.float32)
        _write_obj(frame_dir / "bundle_refined.obj", verts_np, faces)

        with open(frame_dir / "bundle_refined.pkl", "wb") as f:
            pickle.dump(
                {
                    "betas": (2.0 * torch.tanh(betas_raw)).detach().cpu().numpy(),
                    "body_pose": body_pose[i].detach().cpu().numpy(),
                    "global_orient": global_orient[i].detach().cpu().numpy(),
                    "camera_rotation": cam_r[i].detach().cpu().numpy(),
                    "camera_translation": cam_t[i].detach().cpu().numpy(),
                },
                f,
            )

        img_candidates = list(images_dir.glob(stem + ".*"))
        if img_candidates:
            img = cv2.imread(str(img_candidates[0]))
            if img is not None:
                uvj_np = uv_j.detach().cpu().numpy()
                for q in uvj_np:
                    x, y = int(round(float(q[0]))), int(round(float(q[1])))
                    if 0 <= x < w and 0 <= y < h:
                        color = (0, 255, 0) if masks_t[i][0, 0, y, x] > 0.5 else (0, 0, 255)
                        cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
                cv2.putText(img, f"{stem} inside={inside:.3f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(frame_dir / "bundle_overlay.jpg"), img)

    with torch.no_grad():
        zero_pose = torch.zeros_like(body_pose[:1])
        zero_orient = torch.zeros_like(global_orient[:1])
        can_out = body_model(betas=(2.0 * torch.tanh(betas_raw)), body_pose=zero_pose, global_orient=zero_orient, return_verts=True)
    _write_obj(out_dir / "bundle_canonical.obj", can_out.vertices[0].detach().cpu().numpy().astype(np.float32), faces)

    summary = {
        "run_dir": str(run_dir),
        "frames": valid_stems,
        "num_frames": B,
        "focal_length": focal,
        "betas_refined": (2.0 * torch.tanh(betas_raw)).detach().cpu().numpy().reshape(-1).tolist(),
        "inside_ratio": inside_ratios,
        "inside_mean": float(np.mean(list(inside_ratios.values()))) if inside_ratios else 0.0,
        "loss_log": logs,
    }
    (out_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved bundle refinement -> {out_dir}")


if __name__ == "__main__":
    main()
