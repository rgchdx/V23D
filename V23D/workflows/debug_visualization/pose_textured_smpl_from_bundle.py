from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recon.smpl_fitter import SMPL


def _nearest_refined_frame(bundle_refined_dir: Path, frame_name: str) -> Path | None:
    stem = Path(frame_name).stem
    exact = bundle_refined_dir / stem
    if exact.exists():
        return exact
    try:
        n = int(stem.replace("frame_", ""))
    except ValueError:
        return None
    candidates = []
    for d in bundle_refined_dir.iterdir():
        if d.is_dir():
            try:
                dn = int(d.name.replace("frame_", ""))
                candidates.append((abs(dn - n), d))
            except ValueError:
                pass
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[0][1]


def _load_textured_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(m.vertices) == 0 or len(m.triangles) == 0:
        raise RuntimeError(f"Could not read mesh: {path}")
    if not m.has_vertex_colors():
        raise RuntimeError(f"Mesh has no vertex colors: {path}")
    v = np.asarray(m.vertices, dtype=np.float32)
    f = np.asarray(m.triangles, dtype=np.int32)
    c = np.asarray(m.vertex_colors, dtype=np.float32)
    return v, f, c


def _smpl_forward(smpl: SMPL, betas: np.ndarray, body_pose: np.ndarray, global_orient: np.ndarray) -> np.ndarray:
    betas_t = torch.from_numpy(betas.astype(np.float32)).reshape(1, -1)
    pose_t = torch.from_numpy(np.concatenate([global_orient.flatten(), body_pose.flatten()]).astype(np.float32)).reshape(1, 72)
    trans_t = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        verts, _joints = smpl(betas_t, pose_t, trans_t)
    return verts.squeeze(0).cpu().numpy().astype(np.float32)


def _save_colored_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces)
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), m, write_vertex_colors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply bundle joint rotations (SMPL pose) to a textured SMPL mesh")
    ap.add_argument("--textured-mesh", required=True, help="Vertex-colored canonical SMPL mesh (.ply), e.g. method_C_textured.ply")
    ap.add_argument("--smpl-model", required=True)
    ap.add_argument("--bundle-refined-dir", required=True)
    ap.add_argument("--frames", nargs="+", default=["frame_00000.jpg", "frame_00165.jpg"], help="Frame names whose SMPL poses should be applied")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--open3d", action="store_true", help="Open posed meshes in Open3D viewer")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v_can, faces, colors = _load_textured_mesh(Path(args.textured_mesh))
    smpl = SMPL(Path(args.smpl_model), n_betas=10)

    geoms = []
    for fn in args.frames:
        d = _nearest_refined_frame(Path(args.bundle_refined_dir), fn)
        if d is None:
            print(f"[skip] no refined frame for {fn}")
            continue
        pkl = d / "bundle_refined.pkl"
        if not pkl.exists():
            print(f"[skip] missing {pkl}")
            continue
        params = pickle.load(open(pkl, "rb"))
        betas = np.asarray(params["betas"], dtype=np.float32).reshape(-1)
        body_pose = np.asarray(params["body_pose"], dtype=np.float32).reshape(-1)
        global_orient = np.asarray(params["global_orient"], dtype=np.float32).reshape(-1)

        v_pose = _smpl_forward(smpl, betas, body_pose, global_orient)
        if len(v_pose) != len(v_can):
            raise RuntimeError(f"Vertex count mismatch: posed={len(v_pose)} canonical={len(v_can)}")

        out_ply = out_dir / f"posed_{d.name}.ply"
        _save_colored_ply(out_ply, v_pose, faces, colors)
        print(f"saved {out_ply}")

        if args.open3d:
            m = o3d.io.read_triangle_mesh(str(out_ply))
            m.compute_vertex_normals()
            geoms.append(m)

    if args.open3d and geoms:
        o3d.visualization.draw_geometries(geoms, window_name="Posed textured SMPL", width=1400, height=900, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
