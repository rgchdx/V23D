from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import open3d as o3d
except Exception:
    o3d = None

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.pose.extract_mediapipe import load_landmarks_json
from src.recon.smpl_fitter import MP_TO_SMPL_PAIRS, SMPL
from workstreams.next_step_multiframe_bundle.iterative_refining.cameras import WeakPerspectiveCamera, project_points_weakpersp
from workstreams.next_step_multiframe_bundle.iterative_refining.refiner import IterativeRefinementConfig, IterativeRefinementLossBuilder


def _load_trans_dict(path: Path) -> dict[str, np.ndarray]:
    obj = np.load(str(path), allow_pickle=True)
    if hasattr(obj, "item"):
        return {k: np.asarray(v, dtype=np.float32) for k, v in obj.item().items()}
    raise RuntimeError(f"Unsupported trans file: {path}")


def _sample_names(names: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(names) <= max_frames:
        return names
    idx = np.linspace(0, len(names) - 1, max_frames).round().astype(int)
    return [names[i] for i in idx]


def _init_weak_camera(model_xy: np.ndarray, target_xy: np.ndarray) -> tuple[float, float, float]:
    """Fit weak-perspective `s, tx, ty` by least squares from 2D correspondences."""
    mx = model_xy[:, 0]
    my = model_xy[:, 1]
    tx = float(target_xy[:, 0].mean() - mx.mean())
    ty = float(target_xy[:, 1].mean() - my.mean())
    num = ((target_xy[:, 0] - tx) * mx + (target_xy[:, 1] - ty) * my).sum()
    den = (mx * mx + my * my).sum() + 1e-6
    s = float(num / den)
    if not np.isfinite(s):
        s = 1.0
    return max(s, 1e-4), tx, ty


def _build_obs(names: list[str], landmarks: dict[str, np.ndarray | None], joint_pairs: list[tuple[int, int]]):
    obs_xy = []
    obs_conf = []
    pair_ids = []
    for name in names:
        lm = landmarks.get(name)
        pts = []
        conf = []
        smpl_ids = []
        for mp_idx, smpl_idx in joint_pairs:
            if lm is None:
                continue
            x, y, c = float(lm[mp_idx, 0]), float(lm[mp_idx, 1]), float(lm[mp_idx, 2])
            if np.isnan(x) or np.isnan(y):
                continue
            pts.append([x, y])
            conf.append(np.clip(c, 0.1, 1.0))
            smpl_ids.append(smpl_idx)
        obs_xy.append(np.asarray(pts, dtype=np.float32))
        obs_conf.append(np.asarray(conf, dtype=np.float32))
        pair_ids.append(np.asarray(smpl_ids, dtype=np.int64))
    return obs_xy, obs_conf, pair_ids


def _write_mesh_outputs(out_dir: Path, stem: str, verts: np.ndarray, faces: np.ndarray):
    obj_path = out_dir / f"{stem}.obj"
    lines = [f"# {stem}\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    obj_path.write_text("".join(lines), encoding="utf-8")

    ply_path = out_dir / f"{stem}.ply"
    if o3d is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(ply_path), mesh, write_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="Iteratively refine SMPL using weak-perspective cameras and multi-term losses")
    ap.add_argument("--smpl-model", default=r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--betas-npy", required=True)
    ap.add_argument("--poses-json", required=True)
    ap.add_argument("--trans-npy", required=True)
    ap.add_argument("--landmarks-json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shape-pseudogt-npy", default=None, help="Optional betas from SMPLify/SMPLify-X all-frame baseline")
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--n-iters", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    betas_np = np.load(str(args.betas_npy)).astype(np.float32)
    init_poses = json.loads(Path(args.poses_json).read_text(encoding="utf-8"))
    init_trans = _load_trans_dict(Path(args.trans_npy))
    landmarks = load_landmarks_json(args.landmarks_json)

    names = sorted([n for n in init_poses.keys() if n in init_trans and landmarks.get(n) is not None])
    names = _sample_names(names, args.max_frames)
    B = len(names)
    if B == 0:
        raise RuntimeError("No valid frames for refinement")

    obs_xy_np, obs_conf_np, pair_ids_np = _build_obs(names, landmarks, MP_TO_SMPL_PAIRS)

    model = SMPL(args.smpl_model, n_betas=len(betas_np)).to(device)
    model.eval()

    betas = nn.Parameter(torch.from_numpy(betas_np).to(device).unsqueeze(0))
    poses = nn.Parameter(torch.stack([torch.from_numpy(np.asarray(init_poses[n], dtype=np.float32)) for n in names]).to(device))
    trans = nn.Parameter(torch.stack([torch.from_numpy(init_trans[n]) for n in names]).to(device))

    with torch.no_grad():
        _, joints_init = model(betas.expand(B, -1), poses, trans)
    joints_init_np = joints_init.detach().cpu().numpy()
    template_joints = joints_init.detach().clone()

    s0, tx0, ty0 = [], [], []
    for i in range(B):
        ids = pair_ids_np[i]
        jxy = joints_init_np[i, ids, :2]
        s_i, tx_i, ty_i = _init_weak_camera(jxy, obs_xy_np[i])
        s0.append(s_i); tx0.append(tx_i); ty0.append(ty_i)
    cam = WeakPerspectiveCamera(
        s=nn.Parameter(torch.tensor(s0, dtype=torch.float32, device=device)),
        tx=nn.Parameter(torch.tensor(tx0, dtype=torch.float32, device=device)),
        ty=nn.Parameter(torch.tensor(ty0, dtype=torch.float32, device=device)),
    )

    shape_pseudogt = None
    if args.shape_pseudogt_npy and Path(args.shape_pseudogt_npy).exists():
        shape_pseudogt = torch.from_numpy(np.load(str(args.shape_pseudogt_npy)).astype(np.float32)).to(device).unsqueeze(0)

    builder = IterativeRefinementLossBuilder(IterativeRefinementConfig())
    opt = torch.optim.Adam([betas, poses, trans, cam.s, cam.tx, cam.ty], lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_iters, eta_min=1e-4)

    logs = []
    for it in range(args.n_iters):
        opt.zero_grad()
        _, joints = model(betas.expand(B, -1), poses, trans)

        loss_total = joints.new_tensor(0.0)
        accum = {k: 0.0 for k in ["projection", "joint_2d", "proc_2d", "joint_3d", "proc_3d", "body_prop", "pose", "shape", "camera", "sjc", "shape_gt", "temporal"]}
        for i in range(B):
            ids = torch.from_numpy(pair_ids_np[i]).to(device)
            j3 = joints[i:i+1, ids]
            target_2d = torch.from_numpy(obs_xy_np[i]).to(device).unsqueeze(0)
            conf_2d = torch.from_numpy(obs_conf_np[i]).to(device).unsqueeze(0)
            weak_cam_i = WeakPerspectiveCamera(cam.s[i:i+1], cam.tx[i:i+1], cam.ty[i:i+1])
            losses = builder.build(
                joints_3d=j3,
                joints_2d_target=target_2d,
                pose=poses[i:i+1],
                betas=betas,
                weak_camera=weak_cam_i,
                template_joints_3d=template_joints[i:i+1, ids],
                target_joints_3d=joints_init[i:i+1, ids],
                conf_2d=conf_2d,
            )
            if shape_pseudogt is not None:
                shape_gt = 0.25 * ((betas - shape_pseudogt) ** 2).mean()
            else:
                shape_gt = betas.new_tensor(0.0)
            losses["shape_gt"] = shape_gt
            loss_frame = losses["total"] + shape_gt
            loss_total = loss_total + loss_frame
            for k in accum:
                if k in losses:
                    accum[k] += float(losses[k].detach().cpu())

        if B > 1:
            temporal = 0.05 * (torch.abs(poses[1:] - poses[:-1]).mean() + torch.abs(trans[1:] - trans[:-1]).mean())
            loss_total = loss_total + temporal
            accum["temporal"] = float(temporal.detach().cpu())

        loss_total.backward()
        opt.step()
        sched.step()

        if (it % 25 == 0) or (it == args.n_iters - 1):
            row = {"iter": it, "total": float(loss_total.detach().cpu())}
            row.update({k: v / max(B, 1) for k, v in accum.items()})
            logs.append(row)
            print(row)

    with torch.no_grad():
        verts, joints = model(betas.expand(B, -1), poses, trans)
    refined = {
        n: {
            "pose": poses[i].detach().cpu().numpy().tolist(),
            "trans": trans[i].detach().cpu().numpy().tolist(),
            "camera": {
                "s": float(cam.s[i].detach().cpu()),
                "tx": float(cam.tx[i].detach().cpu()),
                "ty": float(cam.ty[i].detach().cpu()),
            },
        }
        for i, n in enumerate(names)
    }
    (out_dir / "refined_fit.json").write_text(json.dumps(refined, indent=2), encoding="utf-8")
    (out_dir / "loss_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    np.save(str(out_dir / "betas_refined.npy"), betas[0].detach().cpu().numpy())
    np.save(str(out_dir / "trans_refined.npy"), {n: refined[n]["trans"] for n in names})
    (out_dir / "poses_refined.json").write_text(json.dumps({n: refined[n]["pose"] for n in names}, indent=2), encoding="utf-8")

    with torch.no_grad():
        zero_pose = torch.zeros_like(poses[:1])
        zero_trans = torch.zeros_like(trans[:1])
        verts_can, _ = model(betas, zero_pose, zero_trans)
        mid_idx = B // 2
        verts_mid, _ = model(betas, poses[mid_idx:mid_idx+1], trans[mid_idx:mid_idx+1])
    faces = model.faces
    _write_mesh_outputs(out_dir, "smpl_refined_canonical", verts_can[0].detach().cpu().numpy(), faces)
    _write_mesh_outputs(out_dir, f"smpl_refined_frame_{mid_idx:03d}", verts_mid[0].detach().cpu().numpy(), faces)

    print(f"Saved refinement outputs -> {out_dir}")


if __name__ == "__main__":
    main()
