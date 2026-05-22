"""
fit_smpl_to_mesh.py
====================
Top-level script: run the full body reconstruction pipeline from
an orbiting diffusion video.

Pipeline
--------
1.  Load COLMAP camera poses from sparse model.
2.  Extract 2D MediaPipe body landmarks per frame (or load cached JSON).
3a. If SMPL model file is provided:
      → Fit shared body shape beta + per-frame pose theta + translation t
        by minimising reprojection loss against the 2D landmarks.
      → Export canonical (mean-pose) SMPL mesh as .obj
3b. If no SMPL model:
      → Triangulate 3D joint positions via DLT from all views.
      → Export sparse skeleton as .obj

Usage
-----
Basic (no SMPL model, triangulation only):
  python fit_smpl_to_mesh.py \
      --colmap-dir  E:/V23D_Data/colmap_rerun/sparse/1 \
      --frames-dir  E:/V23D_Data/frames \
      --out-dir     E:/V23D_Data/smpl_out

With SMPL model (neutral already at E:\SMPL_extracted\...):
  python fit_smpl_to_mesh.py \
      --colmap-dir  E:/V23D_Data/colmap_rerun/sparse/1 \
      --frames-dir  E:/V23D_Data/frames \
      --out-dir     E:/V23D_Data/smpl_out
      (--smpl-model defaults to the neutral v1.1 pkl)

SMPL model files found at:
  E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models\
    basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
    basicmodel_m_lbs_10_207_0_v1.1.0.pkl
    basicmodel_f_lbs_10_207_0_v1.1.0.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---- project imports ----
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.pose.extract_mediapipe import (
    extract_landmarks_dir,
    load_landmarks_json,
    TRACKED_MP_INDICES,
)
from src.recon.smpl_fitter import (
    _read_colmap_cameras_txt,
    _read_colmap_images_txt,
    _build_K,
    triangulate_joints,
    save_skeleton_obj,
    save_mesh_obj,
)


# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser(
        description="Human body reconstruction from orbiting video frames."
    )
    p.add_argument("--colmap-dir",  required=True,
                   help="Path to COLMAP sparse model dir (contains cameras.txt, images.txt)")
    p.add_argument("--frames-dir",  required=True,
                   help="Directory of input frames (jpg/png).")
    p.add_argument("--masks-dir",   default=None,
                   help="Optional: mask directory (unused at this stage).")
    _DEFAULT_SMPL = (
        r"E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models"
        r"\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    )
    p.add_argument("--smpl-model",  default=_DEFAULT_SMPL,
                   help="Path to SMPL v1.1 .pkl model file.  "
                        "Defaults to the neutral model already on disk.")
    p.add_argument("--out-dir",     required=True,
                   help="Output directory.")
    p.add_argument("--landmarks-json", default=None,
                   help="Path to pre-computed MediaPipe landmarks JSON.  "
                        "If not provided, landmarks are extracted and saved here.")
    p.add_argument("--min-views",    type=int, default=4,
                   help="Minimum views required to triangulate a joint.")
    p.add_argument("--n-betas",      type=int, default=10,
                   help="Number of SMPL shape parameters (max 10 recommended).")
    p.add_argument("--n-frames",     type=int, default=40,
                   help="Number of evenly-spaced frames to use for fitting.")
    p.add_argument("--min-visible",  type=int, default=8,
                   help="Min visible landmarks required per frame.")
    p.add_argument("--n-iters",      type=int, default=1000,
                   help="Total optimisation iterations (divided 20/20/60 across stages).")
    p.add_argument("--lr",          type=float, default=5e-3,
                   help="Learning rate for SMPL optimiser.")
    p.add_argument("--device",      default="cuda",
                   help="torch device for SMPL fitting (cuda / cpu).")
    p.add_argument("--max-frames",  type=int, default=0,
                   help="Limit frames to this count (0 = all). "
                        "Useful for quick tests.")
    args = p.parse_args()

    colmap_dir  = Path(args.colmap_dir)
    frames_dir  = Path(args.frames_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load COLMAP cameras & images
    # ----------------------------------------------------------------
    print("=== Loading COLMAP poses ===")
    cams_txt   = colmap_dir / "cameras.txt"
    images_txt = colmap_dir / "images.txt"

    if not cams_txt.exists():
        # try binary conversion via colmap (requires colmap on PATH)
        print("cameras.txt not found — attempting model_converter …")
        import subprocess
        subprocess.run([
            "colmap", "model_converter",
            "--input_path",  str(colmap_dir),
            "--output_path", str(colmap_dir),
            "--output_type", "TXT",
        ], check=True)

    cameras = _read_colmap_cameras_txt(cams_txt)
    images  = _read_colmap_images_txt(images_txt)
    print(f"  {len(cameras)} camera model(s), {len(images)} registered images")

    # Assume single shared intrinsics (model 1)
    cam_id  = list(cameras.keys())[0]
    cam_K   = _build_K(cameras[cam_id])
    print(f"  K = f={cam_K[0,0]:.1f}, cx={cam_K[0,2]:.1f}, cy={cam_K[1,2]:.1f}")

    # ----------------------------------------------------------------
    # 2. Extract / load MediaPipe 2D landmarks
    # ----------------------------------------------------------------
    if args.landmarks_json and Path(args.landmarks_json).exists():
        print(f"=== Loading cached landmarks from {args.landmarks_json} ===")
        lm_dict = load_landmarks_json(args.landmarks_json)
    else:
        lm_json_path = (
            Path(args.landmarks_json)
            if args.landmarks_json
            else out_dir / "landmarks_mediapipe.json"
        )
        print(f"=== Extracting MediaPipe landmarks -> {lm_json_path} ===")
        lm_dict = extract_landmarks_dir(frames_dir, lm_json_path)

    # Optionally limit frames
    if args.max_frames > 0:
        keys = sorted(lm_dict.keys())[:args.max_frames]
        lm_dict = {k: lm_dict[k] for k in keys}

    n_detected = sum(1 for v in lm_dict.values() if v is not None)
    print(f"  Landmarks detected in {n_detected}/{len(lm_dict)} frames")

    # ----------------------------------------------------------------
    # 3a. DLT triangulation (always run as fast sanity check)
    # ----------------------------------------------------------------
    print("=== Stage 1: DLT multi-view triangulation ===")
    joints3d = triangulate_joints(
        lm_dict    = lm_dict,
        images     = images,
        cam_K      = cam_K,
        mp_indices = TRACKED_MP_INDICES,
        min_views  = args.min_views,
    )
    valid = np.sum(~np.any(np.isnan(joints3d), axis=1))
    print(f"  Triangulated {valid}/{len(TRACKED_MP_INDICES)} joints")

    skeleton_obj = out_dir / "skeleton_dlt.obj"
    save_skeleton_obj(joints3d, TRACKED_MP_INDICES, skeleton_obj)

    # Save joints as JSON too
    joints_json = out_dir / "joints3d_dlt.json"
    joints_data = {
        str(mp_idx): (
            joints3d[ji].tolist()
            if not np.any(np.isnan(joints3d[ji])) else None
        )
        for ji, mp_idx in enumerate(TRACKED_MP_INDICES)
    }
    joints_json.write_text(json.dumps(joints_data, indent=2))
    print(f"  Saved 3D joints -> {joints_json}")

    # ----------------------------------------------------------------
    # 3b. SMPL fitting (only if model file provided)
    # ----------------------------------------------------------------
    if args.smpl_model:
        smpl_path = Path(args.smpl_model)
        if not smpl_path.exists():
            print(f"[WARNING] SMPL model not found: {smpl_path}")
            print("  Skipping SMPL stage.  Download from https://smpl.is.tue.mpg.de/")
        else:
            print(f"=== Stage 2: SMPL fitting (model: {smpl_path.name}) ===")
            from src.recon.smpl_fitter import fit_smpl_multiview

            n_s0 = max(1, args.n_iters // 5)    # 20 %
            n_s1 = max(1, args.n_iters // 5)    # 20 %
            n_s2 = args.n_iters - n_s0 - n_s1   # 60 %

            result = fit_smpl_multiview(
                smpl_model_path = smpl_path,
                lm_dict         = lm_dict,
                images          = images,
                cam_K           = cam_K,
                masks_dir       = Path(args.masks_dir) if args.masks_dir else None,
                n_frames        = args.n_frames,
                min_visible     = args.min_visible,
                n_betas         = args.n_betas,
                n_iters_s0      = n_s0,
                n_iters_s1      = n_s1,
                n_iters_s2      = n_s2,
                lr              = args.lr,
                device          = args.device,
            )

            mesh_obj = out_dir / "smpl_canonical.obj"
            save_mesh_obj(result["verts"], result["faces"], mesh_obj)

            # Save betas
            np.save(str(out_dir / "betas.npy"), result["betas"])
            np.save(str(out_dir / "scale.npy"), np.array([1.0], dtype=np.float32))

            # Save per-frame poses dict
            poses_out = {k: v.tolist() for k, v in result["poses"].items()}
            np.save(str(out_dir / "trans_per_frame.npy"),
                    {k: v.tolist() for k, v in result["trans_per_frame"].items()})
            (out_dir / "poses_per_frame.json").write_text(
                json.dumps(poses_out, indent=2))

            print(f"  beta (shape): {result['betas']}")
            print(f"  scale: {result['scale']:.4f}")
            print(f"  Saved mesh -> {mesh_obj}")
    else:
        print("\n[INFO] --smpl-model path not found or not provided.")
        print("  DLT-only output: skeleton_dlt.obj + joints3d_dlt.json")

    print(f"\nDone.  Output directory: {out_dir}")


if __name__ == "__main__":
    main()