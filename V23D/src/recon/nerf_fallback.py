from pathlib import Path
import json
import logging
import math
import shutil
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _parse_cameras_txt(cameras_txt_path: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    for raw in cameras_txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cam_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]

        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = params[0]
            cx = params[1]
            cy = params[2]
        elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            fx = params[0]
            fy = params[1]
            cx = params[2]
            cy = params[3]
        else:
            raise ValueError(f"Unsupported COLMAP camera model in cameras.txt: {model}")

        cameras[cam_id] = {
            "camera_id": cam_id,
            "model": model,
            "width": width,
            "height": height,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
        }

    if not cameras:
        raise RuntimeError(f"No camera definitions parsed from {cameras_txt_path}")

    return cameras


def _opencv_to_opengl_c2w(c2w: np.ndarray) -> np.ndarray:
    # Convert camera coordinate convention from OpenCV to OpenGL used by many NeRF pipelines.
    convert = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ convert


def _copy_registered_images(frames_dir: Path, image_names: list[str], images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in image_names:
        src = frames_dir / name
        if not src.exists() or src.suffix.lower() not in IMAGE_EXTS:
            continue
        shutil.copy2(src, images_dir / name)
        copied += 1
    return copied


def prepare_nerf_dataset_from_sfm(
    frames_dir: Path,
    colmap_dir: Path,
    dataset_dir: Path,
    convert_to_opengl: bool = True,
) -> Path:
    """Prepare NeRF dataset (images + transforms.json) from SfM outputs.

    Requires:
    - colmap_dir/poses.json
    - colmap_dir/sparse/0/cameras.txt
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not colmap_dir.exists():
        raise FileNotFoundError(f"SfM/COLMAP directory not found: {colmap_dir}")

    poses_path = colmap_dir / "poses.json"
    cameras_txt_path = colmap_dir / "sparse" / "0" / "cameras.txt"

    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")
    if not cameras_txt_path.exists():
        raise FileNotFoundError(
            f"Missing cameras.txt: {cameras_txt_path}. Ensure COLMAP model_converter generated TXT files."
        )

    poses = json.loads(poses_path.read_text(encoding="utf-8"))
    cameras_by_id = _parse_cameras_txt(cameras_txt_path)

    if not poses:
        raise ValueError("poses.json is empty")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"

    image_names = [p["image_name"] for p in poses]
    copied_count = _copy_registered_images(frames_dir, image_names, images_dir)
    if copied_count == 0:
        raise RuntimeError("No registered images were copied for NeRF dataset")

    first_cam = cameras_by_id.get(poses[0]["camera_id"], next(iter(cameras_by_id.values())))

    frames = []
    for pose in poses:
        cam = cameras_by_id.get(pose["camera_id"], first_cam)
        c2w = np.array(pose["camera_to_world"], dtype=float)
        if convert_to_opengl:
            c2w = _opencv_to_opengl_c2w(c2w)

        frames.append(
            {
                "file_path": f"./images/{pose['image_name']}",
                "transform_matrix": c2w.tolist(),
                "camera_id": pose["camera_id"],
                "fl_x": cam["fl_x"],
                "fl_y": cam["fl_y"],
                "cx": cam["cx"],
                "cy": cam["cy"],
                "w": cam["width"],
                "h": cam["height"],
            }
        )

    transforms = {
        "fl_x": first_cam["fl_x"],
        "fl_y": first_cam["fl_y"],
        "cx": first_cam["cx"],
        "cy": first_cam["cy"],
        "w": first_cam["width"],
        "h": first_cam["height"],
        "camera_model": first_cam["model"],
        "camera_angle_x": 2.0 * math.atan((first_cam["width"] * 0.5) / first_cam["fl_x"]),
        "camera_angle_y": 2.0 * math.atan((first_cam["height"] * 0.5) / first_cam["fl_y"]),
        "frames": frames,
    }
    (dataset_dir / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    prep_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frames_dir": str(frames_dir),
        "colmap_dir": str(colmap_dir),
        "dataset_dir": str(dataset_dir),
        "num_registered_frames": len(poses),
        "num_copied_images": copied_count,
        "convert_to_opengl": convert_to_opengl,
    }
    (dataset_dir / "nerf_dataset_info.json").write_text(json.dumps(prep_info, indent=2), encoding="utf-8")
    logging.info("Prepared NeRF dataset at %s", dataset_dir)

    return dataset_dir


# This is the main entry point for the NeRF fallback step, which prepares the dataset and launches training.
def train_nerf_from_dataset(
    nerf_repo_dir: Path,
    dataset_dir: Path,
    model_dir: Path,
    framework: str = "custom",
    iterations: int = 50000,
    python_exe: str = "python",
    train_script: str = "train.py",
) -> None:
    """Train a NeRF model from prepared dataset.

    framework='custom' uses:
      python <nerf_repo_dir>/<train_script> --data <dataset_dir> --output <model_dir> --iters <iterations>

    framework='nerfstudio' uses:
      ns-train nerfacto --data <dataset_dir> --output-dir <model_dir> --max-num-iterations <iterations>
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not (dataset_dir / "transforms.json").exists():
        raise FileNotFoundError(f"Missing transforms.json in dataset: {dataset_dir}")

    model_dir.mkdir(parents=True, exist_ok=True)

    fw = framework.strip().lower()
    if fw == "custom":
        train_path = nerf_repo_dir / train_script
        if not train_path.exists():
            raise FileNotFoundError(f"Could not find train script: {train_path}")
        cmd = [
            python_exe,
            str(train_path),
            "--data",
            str(dataset_dir),
            "--output",
            str(model_dir),
            "--iters",
            str(iterations),
        ]
        cwd = nerf_repo_dir
    elif fw == "nerfstudio":
        cmd = [
            "ns-train",
            "nerfacto",
            "--data",
            str(dataset_dir),
            "--output-dir",
            str(model_dir),
            "--max-num-iterations",
            str(iterations),
        ]
        cwd = None
    else:
        raise ValueError("framework must be one of: custom, nerfstudio")

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "framework": fw,
        "nerf_repo_dir": str(nerf_repo_dir),
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "iterations": iterations,
        "command": cmd,
        "status": "started",
    }
    (model_dir / "train_nerf_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    _run(cmd, cwd=cwd)

    run_info["status"] = "completed"
    (model_dir / "train_nerf_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    logging.info("NeRF training complete. Output: %s", model_dir)


def run_nerf_fallback(
    frames_dir: Path,
    colmap_dir: Path,
    nerf_repo_dir: Path,
    dataset_dir: Path,
    model_dir: Path,
    framework: str = "custom",
    iterations: int = 50000,
    python_exe: str = "python",
    train_script: str = "train.py",
    convert_to_opengl: bool = True,
) -> None:
    """Prepare NeRF dataset from SfM and launch training."""
    prepare_nerf_dataset_from_sfm(
        frames_dir=frames_dir,
        colmap_dir=colmap_dir,
        dataset_dir=dataset_dir,
        convert_to_opengl=convert_to_opengl,
    )
    train_nerf_from_dataset(
        nerf_repo_dir=nerf_repo_dir,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        framework=framework,
        iterations=iterations,
        python_exe=python_exe,
        train_script=train_script,
    )
