from pathlib import Path
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _copy_images(frames_dir: Path, images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(frames_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            dst = images_dir / p.name
            shutil.copy2(p, dst)
            count += 1
    return count


def _copy_sparse_model(colmap_sparse_0_dir: Path, out_sparse_0_dir: Path) -> None:
    out_sparse_0_dir.mkdir(parents=True, exist_ok=True)
    required = ["cameras.bin", "images.bin", "points3D.bin"]

    for name in required:
        src = colmap_sparse_0_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing COLMAP file: {src}")
        shutil.copy2(src, out_sparse_0_dir / name)


def prepare_3dgs_scene_from_sfm(
    frames_dir: Path,
    colmap_dir: Path,
    scene_dir: Path,
) -> Path:
    """Prepare a 3DGS scene directory from SfM/COLMAP outputs.

    Expected COLMAP structure: colmap_dir/sparse/0/{cameras.bin,images.bin,points3D.bin}
    Output scene structure: scene_dir/images + scene_dir/sparse/0
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not colmap_dir.exists():
        raise FileNotFoundError(f"SfM/COLMAP directory not found: {colmap_dir}")

    sparse_0 = colmap_dir / "sparse" / "0"
    if not sparse_0.exists():
        raise FileNotFoundError(f"COLMAP sparse model folder not found: {sparse_0}")

    scene_dir.mkdir(parents=True, exist_ok=True)
    images_dir = scene_dir / "images"
    out_sparse_0 = scene_dir / "sparse" / "0"

    image_count = _copy_images(frames_dir, images_dir)
    if image_count == 0:
        raise RuntimeError(f"No images copied from {frames_dir}")

    _copy_sparse_model(sparse_0, out_sparse_0)

    prep_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frames_dir": str(frames_dir),
        "colmap_dir": str(colmap_dir),
        "scene_dir": str(scene_dir),
        "num_images": image_count,
        "copied_sparse_files": ["cameras.bin", "images.bin", "points3D.bin"],
    }
    (scene_dir / "scene_prep_info.json").write_text(json.dumps(prep_info, indent=2), encoding="utf-8")
    logging.info("Prepared 3DGS scene at %s", scene_dir)
    return scene_dir


# This is the main entry point for the 3DGS follow-up step, which prepares the scene and launches training.
def train_3dgs_from_scene(
    gs_repo_dir: Path,
    scene_dir: Path,
    model_dir: Path,
    iterations: int = 30000,
    python_exe: str = "python",
) -> None:
    """Run Graphdeco 3DGS training using prepared scene directory."""
    train_py = gs_repo_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Could not find train.py in 3DGS repo: {train_py}")

    if not (scene_dir / "images").exists() or not (scene_dir / "sparse" / "0").exists():
        raise FileNotFoundError("Scene directory is incomplete. Expected images/ and sparse/0/.")

    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(train_py),
        "-s",
        str(scene_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(iterations),
    ]

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gs_repo_dir": str(gs_repo_dir),
        "scene_dir": str(scene_dir),
        "model_dir": str(model_dir),
        "iterations": iterations,
        "command": cmd,
        "status": "started",
    }
    (model_dir / "train_3dgs_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    _run(cmd, cwd=gs_repo_dir)

    run_info["status"] = "completed"
    (model_dir / "train_3dgs_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    logging.info("3DGS training complete. Model output: %s", model_dir)


# Command to run COLMAP and prepare the scene for 3DGS follow-up.
def run_3dgs_followup(
    frames_dir: Path,
    colmap_dir: Path,
    gs_repo_dir: Path,
    scene_dir: Path,
    model_dir: Path,
    iterations: int = 30000,
    python_exe: str = "python",
) -> None:
    """Prepare scene from SfM and launch 3DGS training."""
    prepare_3dgs_scene_from_sfm(frames_dir=frames_dir, colmap_dir=colmap_dir, scene_dir=scene_dir)
    train_3dgs_from_scene(
        gs_repo_dir=gs_repo_dir,
        scene_dir=scene_dir,
        model_dir=model_dir,
        iterations=iterations,
        python_exe=python_exe,
    )
