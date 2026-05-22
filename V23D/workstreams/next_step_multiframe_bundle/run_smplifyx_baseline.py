from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pose.extract_mediapipe import load_landmarks_json

# MediaPipe -> OpenPose BODY_25 indices used by SMPLify-X loader
# BODY_25: 0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist,
#          5 l_shoulder, 6 l_elbow, 7 l_wrist,
#          8 mid_hip, 9 r_hip, 10 r_knee, 11 r_ankle,
#          12 l_hip, 13 l_knee, 14 l_ankle, ...
MP_TO_BODY25 = {
    0: 0,
    12: 2,
    14: 3,
    16: 4,
    11: 5,
    13: 6,
    15: 7,
    24: 9,
    26: 10,
    28: 11,
    23: 12,
    25: 13,
    27: 14,
}


def _make_body25_from_mediapipe(lm: np.ndarray) -> list[float]:
    arr = np.zeros((25, 3), dtype=np.float32)
    for mp_i, b_i in MP_TO_BODY25.items():
        x, y, v = float(lm[mp_i, 0]), float(lm[mp_i, 1]), float(lm[mp_i, 2])
        if np.isnan(x) or np.isnan(y):
            continue
        arr[b_i] = [x, y, max(0.0, min(1.0, v))]

    # neck (1) from shoulders
    ls, rs = arr[5], arr[2]
    if ls[2] > 0 and rs[2] > 0:
        arr[1, :2] = 0.5 * (ls[:2] + rs[:2])
        arr[1, 2] = min(ls[2], rs[2])

    # mid-hip (8) from hips
    lh, rh = arr[12], arr[9]
    if lh[2] > 0 and rh[2] > 0:
        arr[8, :2] = 0.5 * (lh[:2] + rh[:2])
        arr[8, 2] = min(lh[2], rh[2])

    return arr.reshape(-1).tolist()


def main():
    ap = argparse.ArgumentParser(description="Prepare and run SMPLify-X baseline on first frame")
    ap.add_argument("--smplifyx-root", default=str(ROOT / "third_party" / "smplify-x"))
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--landmarks-json", required=True)
    ap.add_argument("--smplx-model-folder", required=True,
                    help="Folder containing SMPLX_* model files expected by SMPLify-X")
    ap.add_argument("--vposer-folder", default="", help="Optional VPoser checkpoint folder")
    ap.add_argument("--output", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    smplifyx_root = Path(args.smplifyx_root)
    frames_dir = Path(args.frames_dir)
    lm_json = Path(args.landmarks_json)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not smplifyx_root.exists():
        raise FileNotFoundError(f"SMPLify-X repo not found: {smplifyx_root}")

    lm_dict = load_landmarks_json(lm_json)
    first_name = sorted(lm_dict.keys())[0]
    first_img = frames_dir / first_name
    if not first_img.exists():
        stem = Path(first_name).stem
        cand = list(frames_dir.glob(stem + ".*"))
        if not cand:
            raise FileNotFoundError(f"Could not find first frame image for {first_name}")
        first_img = cand[0]

    lm = lm_dict[first_name]
    if lm is None:
        raise RuntimeError("First frame has no MediaPipe landmarks")

    # Prepare SMPLify-X expected folder layout
    img_dir = out_dir / "images"
    kp_dir = out_dir / "keypoints"
    img_dir.mkdir(parents=True, exist_ok=True)
    kp_dir.mkdir(parents=True, exist_ok=True)

    target_img = img_dir / first_img.name
    shutil.copy2(first_img, target_img)

    body25 = _make_body25_from_mediapipe(lm)
    kp = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": body25,
                "face_keypoints_2d": [0.0] * (70 * 3),
                "hand_left_keypoints_2d": [0.0] * (21 * 3),
                "hand_right_keypoints_2d": [0.0] * (21 * 3),
            }
        ],
    }
    kp_path = kp_dir / f"{target_img.stem}_keypoints.json"
    kp_path.write_text(json.dumps(kp), encoding="utf-8")

    # Build command (single image baseline)
    cmd = [
        args.python,
        str(smplifyx_root / "smplifyx" / "main.py"),
        "--config", str(smplifyx_root / "cfg_files" / "fit_smplx.yaml"),
        "--data_folder", str(out_dir),
        "--output_folder", str(out_dir / "smplifyx_output"),
        "--model_folder", str(Path(args.smplx_model_folder)),
        "--model_type", "smplx",
        "--gender", "neutral",
        "--use_vposer", "False" if not args.vposer_folder else "True",
    ]
    if args.vposer_folder:
        cmd += ["--vposer_ckpt", str(Path(args.vposer_folder))]

    print("Prepared first-frame baseline input:")
    print(f" image   : {target_img}")
    print(f" keypoint: {kp_path}")
    print("\nRunning SMPLify-X:")
    print(" ".join([f'\"{c}\"' if " " in c else c for c in cmd]))

    subprocess.run(cmd, check=True)
    print(f"\nDone. Output folder: {out_dir / 'smplifyx_output'}")


if __name__ == "__main__":
    main()
