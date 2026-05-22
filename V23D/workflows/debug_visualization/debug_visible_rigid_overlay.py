from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from src.pose.extract_mediapipe import extract_joint_observations, load_landmarks_json
from src.recon.smpl_fitter import MP_TO_SMPL_PAIRS, SMPL, _build_K, _read_colmap_cameras_txt, _read_colmap_images_txt


def _sample_names(names: list[str], n: int) -> list[str]:
    if len(names) <= n:
        return names
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    return [names[i] for i in idx]


def _project_points(pts3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    pts_cam = (R @ pts3d.T).T + t.reshape(1, 3)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    pts2d = np.full((len(pts3d), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        uvw = (K @ pts_cam[valid].T).T
        pts2d[valid] = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
    return pts2d, valid


def _draw_points(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int, valid: np.ndarray | None = None):
    h, w = img.shape[:2]
    if valid is None:
        valid = np.ones(len(pts), dtype=bool)
    for i, (x, y) in enumerate(pts):
        if not valid[i] or np.isnan(x) or np.isnan(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(img, (xi, yi), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (xi, yi), radius + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def _draw_wire(img: np.ndarray, verts2d: np.ndarray, valid: np.ndarray, faces: np.ndarray, step: int = 12):
    h, w = img.shape[:2]
    for tri in faces[::step]:
        if not (valid[tri[0]] and valid[tri[1]] and valid[tri[2]]):
            continue
        pts = verts2d[tri]
        if np.any(np.isnan(pts)):
            continue
        if np.any(pts[:, 0] < -5) or np.any(pts[:, 0] > w + 5) or np.any(pts[:, 1] < -5) or np.any(pts[:, 1] > h + 5):
            continue
        cv2.polylines(img, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True, (0, 140, 255), 1, lineType=cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Debug visible-rigid SMPL overlays")
    ap.add_argument("--smpl-model", default=r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--betas-npy", default=r"E:\V23D_Data\smpl_out\betas.npy")
    ap.add_argument("--poses-json", default=r"E:\V23D_Data\smpl_visible_rigid\poses_per_frame.json")
    ap.add_argument("--trans-npy", default=r"E:\V23D_Data\smpl_visible_rigid\trans_per_frame.npy")
    ap.add_argument("--landmarks-json", default=r"E:\V23D_Data\smpl_out\landmarks_mediapipe.json")
    ap.add_argument("--colmap-dir", default=r"E:\V23D_Data\colmap_rerun\sparse\1")
    ap.add_argument("--frames-dir", default=r"E:\V23D_Data\frames")
    ap.add_argument("--output", default=r"E:\V23D_Data\smpl_visible_rigid_debug")
    ap.add_argument("--scale-json", default=None, help="Optional scale_per_frame.json from visible-rigid fitting")
    ap.add_argument("--scale-npy", default=None, help="Optional global scale.npy fallback")
    ap.add_argument("--min-visibility", type=float, default=0.4,
                    help="Only draw / compare MediaPipe joints with visibility >= this threshold")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--wire-step", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = np.load(str(args.betas_npy)).astype(np.float32)
    poses = json.loads(Path(args.poses_json).read_text())
    trans = np.load(str(args.trans_npy), allow_pickle=True).item()
    scales = {}
    scale_fixed = None
    if args.scale_json and Path(args.scale_json).exists():
        scales = {k: float(v) for k, v in json.loads(Path(args.scale_json).read_text()).items()}
    elif Path(args.poses_json).with_name("scale_per_frame.json").exists():
        scales = {k: float(v) for k, v in json.loads(Path(args.poses_json).with_name("scale_per_frame.json").read_text()).items()}
    if args.scale_npy and Path(args.scale_npy).exists():
        scale_fixed = float(np.load(str(args.scale_npy)).reshape(-1)[0])
    elif Path(args.trans_npy).with_name("scale.npy").exists():
        scale_fixed = float(np.load(str(Path(args.trans_npy).with_name("scale.npy"))).reshape(-1)[0])
    landmarks = load_landmarks_json(args.landmarks_json)

    cameras = _read_colmap_cameras_txt(Path(args.colmap_dir) / "cameras.txt")
    images = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")

    model = SMPL(args.smpl_model, n_betas=len(betas))
    model.eval()
    beta_t = torch.from_numpy(betas).float().unsqueeze(0)

    names = [n for n in sorted(poses.keys()) if n in images and n in trans]
    names = _sample_names(names, args.n_samples)

    saved = []
    with torch.no_grad():
        for name in names:
            pose_np = np.asarray(poses[name], dtype=np.float32)
            trans_np = np.asarray(trans[name], dtype=np.float32)
            scale = float(scales.get(name, scale_fixed if scale_fixed is not None else 1.0))
            scale_t = torch.tensor([scale], dtype=torch.float32)
            verts_t, joints_t = model(
                beta_t,
                torch.from_numpy(pose_np).float().unsqueeze(0),
                torch.from_numpy(trans_np).float().unsqueeze(0),
                scale=scale_t,
            )
            verts = verts_t[0].cpu().numpy()
            joints = joints_t[0].cpu().numpy()

            info = images[name]
            K = _build_K(cameras[info["cam_id"]])
            R = info["R"]
            t = info["t"]
            verts2d, verts_valid = _project_points(verts, K, R, t)
            joints2d, joints_valid = _project_points(joints, K, R, t)

            img = cv2.imread(str(Path(args.frames_dir) / name))
            if img is None:
                continue
            overlay = img.copy()
            _draw_wire(overlay, verts2d, verts_valid, model.faces, step=max(1, args.wire_step))

            lm = landmarks.get(name)
            gt_pts = []
            pred_pts = []
            if lm is not None:
                obs = extract_joint_observations(lm, MP_TO_SMPL_PAIRS, min_visibility=args.min_visibility)
                mp_draw = obs["xy"]
                smpl_ids = obs["smpl_idx"]
                _draw_points(overlay, mp_draw, (0, 255, 0), 4)
                for gt_xy, smpl_idx in zip(mp_draw, smpl_ids.tolist()):
                    pr_xy = joints2d[smpl_idx]
                    if np.any(np.isnan(gt_xy)) or np.any(np.isnan(pr_xy)):
                        continue
                    gt_pts.append(gt_xy)
                    pred_pts.append(pr_xy)
                smpl_draw = np.asarray([joints2d[smpl_idx] for smpl_idx in smpl_ids.tolist()], dtype=np.float32)
                smpl_valid = np.asarray([joints_valid[smpl_idx] for smpl_idx in smpl_ids.tolist()], dtype=bool)
                _draw_points(overlay, smpl_draw, (0, 0, 255), 4, smpl_valid)

            mean_err = float(np.linalg.norm(np.asarray(gt_pts) - np.asarray(pred_pts), axis=1).mean()) if gt_pts else float("nan")
            cv2.putText(overlay, name, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"mean joint err: {mean_err:.2f}px", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"scale: {scale:.3f}", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"detected joints: {len(gt_pts)}", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, "green=MediaPipe detected  red=matched SMPL joints  orange=wireframe", (18, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            out_path = out_dir / f"overlay_{Path(name).stem}.jpg"
            cv2.imwrite(str(out_path), overlay)
            saved.append(out_path)
            print(f"saved {out_path}")

    if saved:
        imgs = [Image.open(p).convert("RGB") for p in saved]
        cols = 2
        cell_w, cell_h = 660, 520
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 24))
        for i, (p, im) in enumerate(zip(saved, imgs)):
            im.thumbnail((cell_w - 20, cell_h - 40))
            canvas = Image.new("RGB", (cell_w, cell_h), (12, 12, 12))
            canvas.paste(im, ((cell_w - im.width) // 2, 10))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, cell_h - 24), p.stem.replace("overlay_", ""), fill=(255, 255, 255))
            sheet.paste(canvas, ((i % cols) * cell_w, (i // cols) * cell_h))
        sheet_path = out_dir / "contact_sheet.jpg"
        sheet.save(sheet_path, quality=90)
        print(f"saved {sheet_path}")


if __name__ == "__main__":
    main()
