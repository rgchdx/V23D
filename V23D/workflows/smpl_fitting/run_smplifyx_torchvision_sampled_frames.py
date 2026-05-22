from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

from src.recon.smpl_fitter import _build_K, _read_colmap_cameras_txt, _read_colmap_images_txt


COCO_TO_BODY25 = {
    0: 0,
    6: 2,
    8: 3,
    10: 4,
    5: 5,
    7: 6,
    9: 7,
    12: 9,
    14: 10,
    16: 11,
    11: 12,
    13: 13,
    15: 14,
}


def _detect_model(device: torch.device):
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()
    return model


def _detect_coco17(model, frame_path: Path, device: torch.device) -> tuple[np.ndarray, float]:
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

    if body25[2, 2] > 0 and body25[5, 2] > 0:
        body25[1, :2] = 0.5 * (body25[2, :2] + body25[5, :2])
        body25[1, 2] = min(body25[2, 2], body25[5, 2])

    if body25[9, 2] > 0 and body25[12, 2] > 0:
        body25[8, :2] = 0.5 * (body25[9, :2] + body25[12, :2])
        body25[8, 2] = min(body25[9, 2], body25[12, 2])

    return body25


def _infer_focal_from_colmap(colmap_dir: Path) -> float:
    model_dir = _resolve_colmap_model_dir(colmap_dir)
    cams = _read_colmap_cameras_txt(model_dir / "cameras.txt")
    imgs = _read_colmap_images_txt(model_dir / "images.txt")
    if not imgs:
        raise RuntimeError(f"No COLMAP images in {model_dir}")
    first_name = sorted(imgs.keys())[0]
    cam = cams[imgs[first_name]["cam_id"]]
    k = _build_K(cam)
    return float(0.5 * (k[0, 0] + k[1, 1]))


def _resolve_colmap_model_dir(colmap_dir: Path) -> Path:
    run_meta = colmap_dir / "colmap_run.json"
    if run_meta.exists():
        try:
            meta = json.loads(run_meta.read_text(encoding="utf-8"))
            selected = meta.get("selected_model_dir")
            if selected:
                p = Path(selected)
                if p.exists():
                    return p
        except Exception:
            pass

    if (colmap_dir / "cameras.txt").exists() and (colmap_dir / "images.txt").exists():
        return colmap_dir

    sparse0 = colmap_dir / "sparse" / "0"
    if (sparse0 / "cameras.txt").exists() and (sparse0 / "images.txt").exists():
        return sparse0

    raise FileNotFoundError(f"Could not resolve COLMAP TXT model dir under {colmap_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SMPLify-X on a sampled subset of frames (SMPLify-X only)")
    ap.add_argument("--smplifyx-root", default=r"C:/smplify-x")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--smpl-neutral-pkl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--python", default=None)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--focal-length", type=float, default=0.0,
                    help="If > 0, use this focal length instead of inferring from COLMAP")
    args = ap.parse_args()

    smplifyx_root = Path(args.smplifyx_root)
    frames_dir = Path(args.frames_dir)
    colmap_dir = Path(args.colmap_dir)
    smpl_pkl = Path(args.smpl_neutral_pkl)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not smplifyx_root.exists():
        raise FileNotFoundError(f"SMPLify-X root not found: {smplifyx_root}")
    if not smpl_pkl.exists():
        raise FileNotFoundError(f"SMPL neutral pkl not found: {smpl_pkl}")

    images = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise RuntimeError(f"No frames found in {frames_dir}")

    n = max(1, min(args.n_samples, len(images)))
    idx = np.linspace(0, len(images) - 1, n, dtype=int)
    sampled = [images[i] for i in idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _detect_model(device)

    img_dir = out_dir / "images"
    kp_dir = out_dir / "keypoints"
    img_dir.mkdir(parents=True, exist_ok=True)
    kp_dir.mkdir(parents=True, exist_ok=True)

    frame_scores: dict[str, float] = {}
    for p in sampled:
        coco17, det_score = _detect_coco17(model, p, device)
        body25 = _coco17_to_body25(coco17)

        dst_img = img_dir / p.name
        shutil.copy2(p, dst_img)

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
        kp_path = kp_dir / f"{dst_img.stem}_keypoints.json"
        kp_path.write_text(json.dumps(keypoint_payload), encoding="utf-8")
        frame_scores[p.name] = det_score

    model_folder = out_dir / "model_folder"
    smpl_dir = model_folder / "smpl"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smpl_pkl, smpl_dir / "SMPL_NEUTRAL.pkl")
    shutil.copy2(smpl_pkl, smpl_dir / "SMPL_MALE.pkl")
    shutil.copy2(smpl_pkl, smpl_dir / "SMPL_FEMALE.pkl")

    focal = float(args.focal_length) if args.focal_length > 0 else _infer_focal_from_colmap(colmap_dir)

    py_exec = args.python or "python"
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
        "--focal_length", str(focal),
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
        "detector": "torchvision_keypointrcnn_coco17",
        "smplifyx_root": str(smplifyx_root),
        "output": str(out_dir / "smplifyx_output"),
        "sampled_frames": [p.name for p in sampled],
        "focal_length": focal,
        "detection_scores": frame_scores,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
