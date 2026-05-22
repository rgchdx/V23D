from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from workstreams.next_step_multiframe_bundle.iterative_refining.cameras import perspective_project
from workstreams.next_step_multiframe_bundle.iterative_refining.losses import (
    mesh_anchor_loss,
    mesh_edge_length_loss,
    mesh_laplacian_loss,
    perspective_mask_distance_loss,
    projected_bbox_loss,
)


@dataclass
class FrameObservation:
    name: str
    image_path: Path
    mask_path: Path | None
    image_bgr: np.ndarray
    mask_u8: np.ndarray
    distance_transform: torch.Tensor
    mask_small: torch.Tensor
    bbox_xyxy: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    focal: float
    width: int
    height: int


@dataclass
class MeshRefineConfig:
    n_iters: int = 250
    lr_global: float = 5e-3
    lr_offsets: float = 2e-3
    stage1_iters: int = 60
    w_mask_dt: float = 0.80
    w_soft_mask: float = 3.00
    w_bbox: float = 1.25
    w_anchor: float = 0.20
    w_edge: float = 1.50
    w_laplacian: float = 0.60
    offset_clip: float = 0.06


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(tri)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# refined mesh\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _build_edges(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            if i == j:
                continue
            edges.add((min(int(i), int(j)), max(int(i), int(j))))
    return np.asarray(sorted(edges), dtype=np.int64)


def _mask_path_for_frame(masks_dir: Path | None, frame_name: str) -> Path | None:
    if masks_dir is None or not masks_dir.exists():
        return None
    stem = Path(frame_name).stem
    candidates = [stem, stem.replace("frame_", "")]
    for cand in candidates:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = masks_dir / f"{cand}{ext}"
            if p.exists():
                return p
    return None


def _load_mask(mask_path: Path | None, h: int, w: int) -> np.ndarray:
    if mask_path is None:
        return np.ones((h, w), dtype=np.uint8) * 255
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.ones((h, w), dtype=np.uint8) * 255
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((m > 127).astype(np.uint8) * 255)


def _mask_bbox(mask_u8: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return np.array([0, 0, mask_u8.shape[1] - 1, mask_u8.shape[0] - 1], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _mask_distance_transform(mask_u8: np.ndarray) -> np.ndarray:
    outside = (mask_u8 == 0).astype(np.uint8)
    dt = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    return dt.astype(np.float32)


def _small_mask(mask_u8: np.ndarray, size: int = 96) -> np.ndarray:
    small = cv2.resize(mask_u8, (size, size), interpolation=cv2.INTER_AREA)
    return (small > 63).astype(np.float32)


def _load_camera_pkl(frame_dir: Path) -> dict:
    pkl_path = frame_dir / "bundle_refined.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}")
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def _render_silhouette_cpu(
    verts: np.ndarray,
    faces: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    pts = (verts @ R.T) + t[None, :]
    z = pts[:, 2]
    px = focal * pts[:, 0] / np.maximum(z, 1e-6) + width * 0.5
    py = focal * pts[:, 1] / np.maximum(z, 1e-6) + height * 0.5
    proj = np.stack([px, py], axis=1)
    face_depth = z[faces].mean(axis=1)
    order = np.argsort(face_depth)[::-1]
    canvas = np.zeros((height, width), dtype=np.uint8)
    for fi in order:
        tri = faces[fi]
        pts2 = proj[tri].astype(np.int32)
        if np.any(~np.isfinite(pts2)):
            continue
        if (pts2[:, 0].max() < 0) or (pts2[:, 0].min() >= width) or (pts2[:, 1].max() < 0) or (pts2[:, 1].min() >= height):
            continue
        cv2.fillConvexPoly(canvas, pts2, 255, lineType=cv2.LINE_AA)
    return canvas


def _overlay_mask(frame_bgr: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, gt_contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawContours(out, pred_contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
    overlap = cv2.bitwise_and(pred_mask, gt_mask)
    out[overlap > 0] = (0.6 * out[overlap > 0] + 0.4 * np.array([255, 255, 0], dtype=np.float32)).astype(np.uint8)
    return out


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def _collect_frame_dirs(bundle_refined_dir: Path, frame_names: list[str] | None = None) -> list[Path]:
    if frame_names:
        out = []
        for name in frame_names:
            stem = Path(name).stem
            d = bundle_refined_dir / stem
            if d.exists():
                out.append(d)
        return out
    return sorted([d for d in bundle_refined_dir.iterdir() if d.is_dir()])


def _frame_observations(
    bundle_refined_dir: Path,
    frames_dir: Path,
    masks_dir: Path | None,
    focal: float,
    device: torch.device,
    frame_names: list[str] | None,
    max_frames: int,
) -> list[FrameObservation]:
    frame_dirs = _collect_frame_dirs(bundle_refined_dir, frame_names)
    if max_frames > 0 and len(frame_dirs) > max_frames:
        idx = np.linspace(0, len(frame_dirs) - 1, max_frames).round().astype(int)
        frame_dirs = [frame_dirs[i] for i in idx]
    obs = []
    for frame_dir in frame_dirs:
        name = f"{frame_dir.name}.jpg"
        image_path = frames_dir / name
        if not image_path.exists():
            continue
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        mask_path = _mask_path_for_frame(masks_dir, name)
        mask_u8 = _load_mask(mask_path, h, w)
        dt = _mask_distance_transform(mask_u8)
        bbox = _mask_bbox(mask_u8)
        params = _load_camera_pkl(frame_dir)
        obs.append(
            FrameObservation(
                name=name,
                image_path=image_path,
                mask_path=mask_path,
                image_bgr=img,
                mask_u8=mask_u8,
                distance_transform=torch.from_numpy(dt).to(device),
                mask_small=torch.from_numpy(_small_mask(mask_u8)).to(device),
                bbox_xyxy=torch.from_numpy(bbox).to(device),
                R=torch.from_numpy(np.asarray(params["camera_rotation"], dtype=np.float32)).to(device),
                t=torch.from_numpy(np.asarray(params["camera_translation"], dtype=np.float32)).to(device),
                focal=float(focal),
                width=w,
                height=h,
            )
        )
    return obs


def _save_mesh(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    _write_obj(path.with_suffix(".obj"), verts, faces)
    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path.with_suffix(".ply")), mesh, write_ascii=False)
    except Exception:
        pass


def _soft_projected_mask_loss(
    projected_xy: torch.Tensor,
    valid: torch.Tensor,
    width: int,
    height: int,
    target_mask_small: torch.Tensor,
) -> torch.Tensor:
    """Differentiable coarse silhouette from bilinearly splatted vertices."""
    if projected_xy.numel() == 0 or not torch.any(valid):
        return target_mask_small.new_tensor(0.0)
    pts = projected_xy[valid]
    h_small, w_small = target_mask_small.shape[-2:]
    x = pts[:, 0] * ((w_small - 1) / max(width - 1, 1))
    y = pts[:, 1] * ((h_small - 1) / max(height - 1, 1))
    x = x.clamp(0, w_small - 1)
    y = y.clamp(0, h_small - 1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=w_small - 1)
    y1 = (y0 + 1).clamp(max=h_small - 1)
    wx = x - x0.float()
    wy = y - y0.float()

    flat = target_mask_small.new_zeros(h_small * w_small)

    def _acc(ix: torch.Tensor, iy: torch.Tensor, wgt: torch.Tensor) -> None:
        idx = iy * w_small + ix
        flat.scatter_add_(0, idx, wgt)

    _acc(x0, y0, (1.0 - wx) * (1.0 - wy))
    _acc(x1, y0, wx * (1.0 - wy))
    _acc(x0, y1, (1.0 - wx) * wy)
    _acc(x1, y1, wx * wy)

    img = flat.view(1, 1, h_small, w_small)
    kernel = torch.ones((1, 1, 5, 5), device=img.device, dtype=img.dtype) / 25.0
    img = torch.nn.functional.conv2d(img, kernel, padding=2)
    img = torch.nn.functional.conv2d(img, kernel, padding=2)
    img = 1.0 - torch.exp(-2.5 * img)
    pred = img[0, 0].clamp(0.0, 1.0)
    return torch.abs(pred - target_mask_small).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description="Iteratively refine a canonical mesh using real-camera silhouette projections")
    ap.add_argument("--mesh-obj", required=True)
    ap.add_argument("--bundle-refined-dir", required=True)
    ap.add_argument("--bundle-summary", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--masks-dir", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-names", nargs="*", default=[])
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--n-iters", type=int, default=250)
    ap.add_argument("--stage1-iters", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_before_dir = out_dir / "overlays_before"
    overlay_after_dir = out_dir / "overlays_after"
    overlay_before_dir.mkdir(parents=True, exist_ok=True)
    overlay_after_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mesh_verts_np, faces_np = _load_obj(Path(args.mesh_obj))
    summary = json.loads(Path(args.bundle_summary).read_text(encoding="utf-8"))
    focal = float(summary["focal_length"])

    observations = _frame_observations(
        bundle_refined_dir=Path(args.bundle_refined_dir),
        frames_dir=Path(args.frames_dir),
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        focal=focal,
        device=device,
        frame_names=args.frame_names or None,
        max_frames=args.max_frames,
    )
    if not observations:
        raise RuntimeError("No valid observations found")

    verts_ref = torch.from_numpy(mesh_verts_np).to(device)
    faces_t = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    edges_np = _build_edges(faces_np)
    edges_t = torch.from_numpy(edges_np).to(device)
    ref_edge_lengths = torch.linalg.norm(verts_ref[edges_t[:, 0]] - verts_ref[edges_t[:, 1]], dim=-1)

    offsets = nn.Parameter(torch.zeros_like(verts_ref))
    global_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
    global_trans = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))

    cfg = MeshRefineConfig(n_iters=args.n_iters, stage1_iters=args.stage1_iters)
    opt = torch.optim.Adam([
        {"params": [global_scale, global_trans], "lr": cfg.lr_global},
        {"params": [offsets], "lr": cfg.lr_offsets},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.n_iters, eta_min=1e-4)

    logs: list[dict[str, float]] = []

    with torch.no_grad():
        v_init = verts_ref.clone()
        for ob in observations:
            proj_init = perspective_project(v_init.unsqueeze(0), ob.R, ob.t, ob.focal, cx=ob.width * 0.5, cy=ob.height * 0.5)[0]
            pred_mask = _render_silhouette_cpu(
                v_init.detach().cpu().numpy(),
                faces_np,
                ob.R.detach().cpu().numpy(),
                ob.t.detach().cpu().numpy(),
                ob.focal,
                ob.width,
                ob.height,
            )
            overlay = _overlay_mask(ob.image_bgr, pred_mask, ob.mask_u8)
            cv2.putText(overlay, f"before IoU={_iou(pred_mask, ob.mask_u8):.3f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(overlay_before_dir / ob.name), overlay)

    for it in range(cfg.n_iters):
        opt.zero_grad()
        if it < cfg.stage1_iters:
            current = verts_ref * global_scale.clamp(min=0.92, max=1.08) + global_trans.unsqueeze(0)
            eff_offsets = torch.zeros_like(offsets)
        else:
            eff_offsets = offsets.clamp(min=-cfg.offset_clip, max=cfg.offset_clip)
            current = (verts_ref + eff_offsets) * global_scale.clamp(min=0.92, max=1.08) + global_trans.unsqueeze(0)

        total = current.new_tensor(0.0)
        stats = {
            "mask_dt": 0.0,
            "soft_mask": 0.0,
            "bbox": 0.0,
            "anchor": 0.0,
            "edge": 0.0,
            "lap": 0.0,
        }

        for ob in observations:
            proj = perspective_project(current.unsqueeze(0), ob.R, ob.t, ob.focal, cx=ob.width * 0.5, cy=ob.height * 0.5)[0]
            pts_cam = current @ ob.R.T + ob.t.unsqueeze(0)
            valid = pts_cam[:, 2] > 1e-4
            l_mask = cfg.w_mask_dt * perspective_mask_distance_loss(proj, ob.distance_transform, valid=valid)
            l_soft = cfg.w_soft_mask * _soft_projected_mask_loss(proj, valid, ob.width, ob.height, ob.mask_small)
            l_bbox = cfg.w_bbox * projected_bbox_loss(proj, valid, ob.bbox_xyxy, (ob.width, ob.height))
            total = total + l_mask + l_soft + l_bbox
            stats["mask_dt"] += float(l_mask.detach().cpu())
            stats["soft_mask"] += float(l_soft.detach().cpu())
            stats["bbox"] += float(l_bbox.detach().cpu())

        l_anchor = cfg.w_anchor * mesh_anchor_loss(current, verts_ref)
        l_edge = cfg.w_edge * mesh_edge_length_loss(current, edges_t, ref_edge_lengths)
        l_lap = cfg.w_laplacian * mesh_laplacian_loss(eff_offsets, edges_t)
        total = total + l_anchor + l_edge + l_lap
        stats["anchor"] = float(l_anchor.detach().cpu())
        stats["edge"] = float(l_edge.detach().cpu())
        stats["lap"] = float(l_lap.detach().cpu())

        total.backward()
        opt.step()
        sched.step()

        if (it % 25 == 0) or (it == cfg.n_iters - 1):
            row = {
                "iter": it,
                "total": float(total.detach().cpu()),
                "mask_dt": stats["mask_dt"] / len(observations),
                "soft_mask": stats["soft_mask"] / len(observations),
                "bbox": stats["bbox"] / len(observations),
                "anchor": stats["anchor"],
                "edge": stats["edge"],
                "lap": stats["lap"],
                "scale": float(global_scale.detach().cpu()),
                "trans_norm": float(torch.linalg.norm(global_trans.detach()).cpu()),
            }
            logs.append(row)
            print(row)

    with torch.no_grad():
        eff_offsets = offsets.clamp(min=-cfg.offset_clip, max=cfg.offset_clip)
        verts_final = ((verts_ref + eff_offsets) * global_scale.clamp(min=0.8, max=1.2) + global_trans.unsqueeze(0)).detach().cpu().numpy()

    _save_mesh(out_dir / "mesh_refined_perspective", verts_final, faces_np)
    (out_dir / "loss_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    np.save(str(out_dir / "vertex_offsets.npy"), eff_offsets.detach().cpu().numpy())
    (out_dir / "summary.json").write_text(json.dumps({
        "mesh_obj": str(Path(args.mesh_obj)),
        "bundle_refined_dir": str(Path(args.bundle_refined_dir)),
        "frames_dir": str(Path(args.frames_dir)),
        "masks_dir": str(Path(args.masks_dir)) if args.masks_dir else "",
        "num_frames": len(observations),
        "focal": focal,
        "final_scale": float(global_scale.detach().cpu()),
        "final_translation": global_trans.detach().cpu().numpy().tolist(),
    }, indent=2), encoding="utf-8")

    ious_before, ious_after = [], []
    for ob in observations:
        pred_before = _render_silhouette_cpu(
            mesh_verts_np,
            faces_np,
            ob.R.detach().cpu().numpy(),
            ob.t.detach().cpu().numpy(),
            ob.focal,
            ob.width,
            ob.height,
        )
        pred_after = _render_silhouette_cpu(
            verts_final,
            faces_np,
            ob.R.detach().cpu().numpy(),
            ob.t.detach().cpu().numpy(),
            ob.focal,
            ob.width,
            ob.height,
        )
        ious_before.append(_iou(pred_before, ob.mask_u8))
        ious_after.append(_iou(pred_after, ob.mask_u8))
        overlay = _overlay_mask(ob.image_bgr, pred_after, ob.mask_u8)
        cv2.putText(overlay, f"after IoU={ious_after[-1]:.3f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlay_after_dir / ob.name), overlay)
        cv2.imwrite(str(out_dir / f"render_{Path(ob.name).stem}.png"), pred_after)

    stats = {
        "mean_iou_before": float(np.mean(ious_before)) if ious_before else 0.0,
        "mean_iou_after": float(np.mean(ious_after)) if ious_after else 0.0,
        "min_iou_before": float(np.min(ious_before)) if ious_before else 0.0,
        "min_iou_after": float(np.min(ious_after)) if ious_after else 0.0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(stats)
    print(f"Saved mesh refinement outputs -> {out_dir}")


if __name__ == "__main__":
    main()
