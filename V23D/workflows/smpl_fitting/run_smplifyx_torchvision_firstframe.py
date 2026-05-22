from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


COCO_TO_BODY25 = {
    0: 0,   # nose
    6: 2,   # r_shoulder
    8: 3,   # r_elbow
    10: 4,  # r_wrist
    5: 5,   # l_shoulder
    7: 6,   # l_elbow
    9: 7,   # l_wrist
    12: 9,  # r_hip
    14: 10, # r_knee
    16: 11, # r_ankle
    11: 12, # l_hip
    13: 13, # l_knee
    15: 14, # l_ankle
}


def _detect_coco17(frame_path: Path, device: torch.device) -> tuple[np.ndarray, float]:
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )

    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()

    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise FileNotFoundError(frame_path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
    with torch.no_grad():
        out = model([x])[0]

    if len(out.get("scores", [])) == 0:
        raise RuntimeError(f"No person detected in {frame_path}")

    det_scores = out["scores"].detach().cpu().numpy().astype(np.float32)
    best = int(np.argmax(det_scores))
    det_score = float(det_scores[best])

    kpts = out["keypoints"][best].detach().cpu().numpy().astype(np.float32)
    if "keypoints_scores" in out:
        kconf = out["keypoints_scores"][best].detach().cpu().numpy().astype(np.float32)
    else:
        kconf = np.full((kpts.shape[0],), det_score, dtype=np.float32)

    coco = np.zeros((17, 3), dtype=np.float32)
    coco[:, :2] = kpts[:, :2]
    coco[:, 2] = kconf
    return coco, det_score


def _coco17_to_body25(coco: np.ndarray) -> np.ndarray:
    body25 = np.zeros((25, 3), dtype=np.float32)

    for ci, bi in COCO_TO_BODY25.items():
        body25[bi] = coco[ci]

    # neck
    if body25[2, 2] > 0 and body25[5, 2] > 0:
        body25[1, :2] = 0.5 * (body25[2, :2] + body25[5, :2])
        body25[1, 2] = min(body25[2, 2], body25[5, 2])

    # mid hip
    if body25[9, 2] > 0 and body25[12, 2] > 0:
        body25[8, :2] = 0.5 * (body25[9, :2] + body25[12, :2])
        body25[8, 2] = min(body25[9, 2], body25[12, 2])

    return body25


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SMPLify-X first-frame fitting using torchvision keypoints (no MediaPipe)")
    ap.add_argument("--smplifyx-root", default=r"C:/smplify-x")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--smpl-neutral-pkl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--python", default=None)
    args = ap.parse_args()

    smplifyx_root = Path(args.smplifyx_root)
    frames_dir = Path(args.frames_dir)
    smpl_pkl = Path(args.smpl_neutral_pkl)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not smplifyx_root.exists():
        raise FileNotFoundError(f"SMPLify-X root not found: {smplifyx_root}")
    if not smpl_pkl.exists():
        raise FileNotFoundError(f"SMPL neutral pkl not found: {smpl_pkl}")

    image_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    first = image_paths[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco17, det_score = _detect_coco17(first, device)
    body25 = _coco17_to_body25(coco17)

    img_dir = out_dir / "images"
    kp_dir = out_dir / "keypoints"
    img_dir.mkdir(parents=True, exist_ok=True)
    kp_dir.mkdir(parents=True, exist_ok=True)

    target_img = img_dir / first.name
    shutil.copy2(first, target_img)

    keypoint_payload = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": body25.reshape(-1).astype(float).tolist(),
                "face_keypoints_2d": [0.0] * (70 * 3),
                "hand_left_keypoints_2d": [0.0] * (21 * 3),
                "hand_right_keypoints_2d": [0.0] * (21 * 3),
            }
        ],
    }
    kp_path = kp_dir / f"{target_img.stem}_keypoints.json"
    kp_path.write_text(json.dumps(keypoint_payload), encoding="utf-8")

    model_folder = out_dir / "model_folder"
    smpl_dir = model_folder / "smpl"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smpl_pkl, smpl_dir / "SMPL_NEUTRAL.pkl")
    shutil.copy2(smpl_pkl, smpl_dir / "SMPL_MALE.pkl")
    shutil.copy2(smpl_pkl, smpl_dir / "SMPL_FEMALE.pkl")

    py_exec = args.python or str(Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe")
    if not Path(py_exec).exists():
        py_exec = "python"

    cmd = [
        py_exec,
        str(smplifyx_root / "smplifyx" / "main.py"),
        "--config", str(smplifyx_root / "cfg_files" / "fit_smpl.yaml"),
        "--data_folder", str(out_dir),
        "--output_folder", str(out_dir / "smplifyx_output"),
        "--model_folder", str(model_folder),
        "--model_type", "smpl",
        "--gender", "neutral",
        "--use_vposer", "False",
        "--interpenetration", "False",
    ]

    compat_dir = out_dir / "pycompat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(
        "import numpy as _np\n"
        "if not hasattr(_np, 'bool'): _np.bool = bool\n"
        "if not hasattr(_np, 'int'): _np.int = int\n"
        "if not hasattr(_np, 'float'): _np.float = float\n"
        "if not hasattr(_np, 'complex'): _np.complex = complex\n"
        "if not hasattr(_np, 'object'): _np.object = object\n"
        "if not hasattr(_np, 'str'): _np.str = str\n"
        "if not hasattr(_np, 'unicode'): _np.unicode = str\n"
        "try:\n"
        "    import os as _os, sys as _sys\n"
        "    _sr = _os.environ.get('SMPLIFYX_ROOT', '')\n"
        "    if _sr:\n"
        "        _sp = _os.path.join(_sr, 'smplifyx')\n"
        "        if _sp not in _sys.path:\n"
        "            _sys.path.insert(0, _sp)\n"
        "    import torch as _torch\n"
        "    import prior as _prior\n"
        "    if hasattr(_prior, 'L2Prior') and not hasattr(_prior.L2Prior, 'get_mean'):\n"
        "        _prior.L2Prior.get_mean = lambda self: _torch.zeros((1, 69), dtype=_torch.float32)\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )

    env = dict(**__import__("os").environ)
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(compat_dir) + ((";" + old_pp) if old_pp else "")
    env["SMPLIFYX_ROOT"] = str(smplifyx_root)

    print("Running:", " ".join([f'\"{c}\"' if " " in c else c for c in cmd]))
    subprocess.run(cmd, check=True, env=env)

    info = {
        "first_frame": first.name,
        "detector": "torchvision_keypointrcnn_coco17",
        "detection_score": det_score,
        "smplifyx_root": str(smplifyx_root),
        "output": str(out_dir / "smplifyx_output"),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
