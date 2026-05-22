from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.pose.extract_mediapipe import load_landmarks_json
from src.recon.smpl_fitter import (
    MP_TO_SMPL_PAIRS,
    SMPL,
    _build_K,
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
)


def _sample_frame_names(all_names: list[str], n: int) -> list[str]:
    if n <= 0 or len(all_names) <= n:
        return all_names
    idx = np.linspace(0, len(all_names) - 1, n).round().astype(int)
    return [all_names[i] for i in idx]


def _project_points(pts3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_cam = (R @ pts3d.T).T + t.reshape(1, 3)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    pts2d = np.full((len(pts3d), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        uvw = (K @ pts_cam[valid].T).T
        pts2d[valid] = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
    return pts2d, valid


def _draw_points(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int, valid: np.ndarray | None = None):
    if valid is None:
        valid = np.ones(len(pts), dtype=bool)
    h, w = img.shape[:2]
    for i, (x, y) in enumerate(pts):
        if not valid[i] or np.isnan(x) or np.isnan(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(img, (xi, yi), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_mesh_wireframe(img: np.ndarray, verts2d: np.ndarray, valid: np.ndarray, faces: np.ndarray,
                         color: tuple[int, int, int] = (255, 140, 0), step: int = 12):
    h, w = img.shape[:2]
    for tri in faces[::step]:
        if not (valid[tri[0]] and valid[tri[1]] and valid[tri[2]]):
            continue
        pts = verts2d[tri]
        if np.any(np.isnan(pts)):
            continue
        if np.any(pts[:, 0] < -5) or np.any(pts[:, 0] > w + 5) or np.any(pts[:, 1] < -5) or np.any(pts[:, 1] > h + 5):
            continue
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, color, 1, lineType=cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Project fitted SMPL onto sample frames for debugging.")
    ap.add_argument("--smpl-model", default=r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--smpl-out", default=r"E:\V23D_Data\smpl_out")
    ap.add_argument("--colmap-dir", default=r"E:\V23D_Data\colmap_rerun\sparse\1")
    ap.add_argument("--frames-dir", default=r"E:\V23D_Data\frames")
    ap.add_argument("--output", default=r"E:\V23D_Data\smpl_debug_overlays")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--wire-step", type=int, default=12, help="Draw every Nth triangle for wireframe.")
    args = ap.parse_args()

    smpl_out = Path(args.smpl_out)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = np.load(str(smpl_out / "betas.npy")).astype(np.float32)
    scale_path = smpl_out / "scale.npy"
    scale = float(np.load(str(scale_path)).reshape(-1)[0]) if scale_path.exists() else 1.0
    poses = json.loads((smpl_out / "poses_per_frame.json").read_text())
    trans = np.load(str(smpl_out / "trans_per_frame.npy"), allow_pickle=True).item()
    landmarks = load_landmarks_json(str(smpl_out / "landmarks_mediapipe.json"))

    cameras = _read_colmap_cameras_txt(Path(args.colmap_dir) / "cameras.txt")
    images = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")

    model = SMPL(args.smpl_model, n_betas=len(betas))
    model.eval()

    frame_names = [n for n in sorted(poses.keys()) if n in images]
    frame_names = _sample_frame_names(frame_names, args.n_samples)

    beta_t = torch.from_numpy(betas).float().unsqueeze(0)

    summary = []
    with torch.no_grad():
        for name in frame_names:
            pose_np = np.asarray(poses[name], dtype=np.float32)
            trans_np = np.asarray(trans[name], dtype=np.float32)

            verts_t, joints_t = model(
                beta_t,
                torch.from_numpy(pose_np).float().unsqueeze(0),
                torch.from_numpy(trans_np).float().unsqueeze(0),
            )
            verts = verts_t[0].cpu().numpy()
            joints = joints_t[0].cpu().numpy()

            info = images[name]
            K = _build_K(cameras[info["cam_id"]])
            R = info["R"]
            t = info["t"]

            verts2d, verts_valid = _project_points(verts, K, R, t)
            joints2d, joints_valid = _project_points(joints, K, R, t)

            img_path = Path(args.frames_dir) / name
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            overlay = img.copy()
            _draw_mesh_wireframe(overlay, verts2d, verts_valid, model.faces, step=max(1, args.wire_step))

            gt_pts = []
            pred_pts = []
            lm = landmarks.get(name)
            for mp_idx, smpl_idx in MP_TO_SMPL_PAIRS:
                if lm is None or lm.shape[0] <= mp_idx:
                    continue
                gt_xy = lm[mp_idx, :2]
                pr_xy = joints2d[smpl_idx]
                if np.any(np.isnan(gt_xy)) or np.any(np.isnan(pr_xy)):
                    continue
                gt_pts.append(gt_xy)
                pred_pts.append(pr_xy)

            if gt_pts:
                gt_pts = np.asarray(gt_pts, dtype=np.float32)
                pred_pts = np.asarray(pred_pts, dtype=np.float32)
                err = np.linalg.norm(gt_pts - pred_pts, axis=1)
                mean_err = float(err.mean())
            else:
                mean_err = float("nan")

            if lm is not None:
                mp_draw = np.asarray([lm[mp, :2] for mp, _ in MP_TO_SMPL_PAIRS], dtype=np.float32)
                _draw_points(overlay, mp_draw, (0, 255, 0), 4)
            smpl_draw = np.asarray([joints2d[smpl] for _, smpl in MP_TO_SMPL_PAIRS], dtype=np.float32)
            smpl_valid = np.asarray([joints_valid[smpl] for _, smpl in MP_TO_SMPL_PAIRS], dtype=bool)
            _draw_points(overlay, smpl_draw, (0, 0, 255), 4, smpl_valid)

            cv2.putText(overlay, f"{name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"mean joint err: {mean_err:.1f}px", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"scale: {scale:.3f}", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, "green=MediaPipe  red=SMPL joints  orange=SMPL wireframe", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            out_path = out_dir / f"overlay_{Path(name).stem}.jpg"
            cv2.imwrite(str(out_path), overlay)
            summary.append((name, mean_err, out_path.name))
            print(f"saved {out_path}   err={mean_err:.2f}px")

    summary_txt = out_dir / "summary.txt"
    lines = [f"{name}\t{err:.3f}\t{fname}" for name, err, fname in summary]
    summary_txt.write_text("\n".join(lines))
    print(f"wrote {summary_txt}")


if __name__ == "__main__":
    main()
