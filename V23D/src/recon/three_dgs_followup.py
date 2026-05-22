from pathlib import Path
import json
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np


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

    sparse_0 = colmap_dir / "sparse" / "0"
    if sparse_0.exists():
        return sparse_0

    raise FileNotFoundError(f"Could not resolve COLMAP sparse model under: {colmap_dir}")


def _find_mask_for_frame(masks_dir: Path, frame_name: str) -> Path | None:
    stem = Path(frame_name).stem
    p = masks_dir / f"{stem}.png"
    return p if p.exists() else None


def _write_colmap_named_masks(frames_dir: Path, masks_dir: Path, out_dir: Path) -> int:
    """Create mask images with the same filenames as COLMAP frame images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in sorted(frames_dir.iterdir()):
        if not frame.is_file() or frame.suffix.lower() not in IMAGE_EXTS:
            continue
        src_mask = _find_mask_for_frame(masks_dir, frame.name)
        if src_mask is None:
            continue

        m = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        m_rgb = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        dst = out_dir / frame.name
        cv2.imwrite(str(dst), m_rgb)
        count += 1
    return count


def _apply_undistorted_masks_to_scene_images(scene_dir: Path, undist_mask_images_dir: Path, threshold: int = 127) -> int:
    """Apply undistorted masks to scene/images in-place. Background becomes black."""
    scene_images = scene_dir / "images"
    if not scene_images.exists():
        return 0

    changed = 0
    for img_path in sorted(scene_images.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        mask_path = undist_mask_images_dir / img_path.name
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        fg = (mask > threshold).astype(np.uint8)
        out = img.copy()
        out[fg == 0] = 0
        cv2.imwrite(str(img_path), out)
        changed += 1

    return changed


def _ensure_sparse_zero_layout(scene_dir: Path) -> None:
    sparse_dir = scene_dir / "sparse"
    sparse_0 = sparse_dir / "0"
    sparse_0.mkdir(parents=True, exist_ok=True)

    # Some COLMAP versions write sparse model files directly under sparse/
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    root_has_all = all((sparse_dir / name).exists() for name in required)
    if root_has_all:
        for name in required:
            src = sparse_dir / name
            dst = sparse_0 / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        return

    # Other layouts already contain a numeric folder; copy from first that has full model.
    numeric_dirs = sorted([p for p in sparse_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    for d in numeric_dirs:
        if all((d / name).exists() for name in required):
            for name in required:
                src = d / name
                dst = sparse_0 / name
                if src.resolve() == dst.resolve():
                    continue
                shutil.copy2(src, dst)
            return

    if not all((sparse_0 / name).exists() for name in required):
        raise FileNotFoundError(f"Undistorted sparse model files not found under: {sparse_dir}")


def _prepare_undistorted_scene_from_colmap(
    frames_dir: Path,
    masks_dir: Path | None,
    colmap_dir: Path,
    scene_dir: Path,
    strict_mask_training: bool = True,
) -> int:
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP executable not found in PATH.")

    model_dir = _resolve_colmap_model_dir(colmap_dir)

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    undist_cmd = [
        "colmap",
        "image_undistorter",
        "--image_path",
        str(frames_dir),
        "--input_path",
        str(model_dir),
        "--output_path",
        str(scene_dir),
        "--output_type",
        "COLMAP",
    ]
    _run(undist_cmd)

    _ensure_sparse_zero_layout(scene_dir)

    images_dir = scene_dir / "images"
    image_count = len([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if image_count == 0:
        raise RuntimeError(f"No undistorted images found in {images_dir}")

    if strict_mask_training:
        if masks_dir is None or not masks_dir.exists():
            raise FileNotFoundError("strict_mask_training=True but masks_dir is missing.")

        with tempfile.TemporaryDirectory(prefix="v23d_mask_ud_") as tdir:
            troot = Path(tdir)
            named_masks_dir = troot / "named_masks"
            undist_masks_dir = troot / "undist_masks"

            n = _write_colmap_named_masks(frames_dir=frames_dir, masks_dir=masks_dir, out_dir=named_masks_dir)
            if n == 0:
                raise RuntimeError(f"No masks could be matched to frame names in {masks_dir}")

            undist_mask_cmd = [
                "colmap",
                "image_undistorter",
                "--image_path",
                str(named_masks_dir),
                "--input_path",
                str(model_dir),
                "--output_path",
                str(undist_masks_dir),
                "--output_type",
                "COLMAP",
            ]
            _run(undist_mask_cmd)

            applied = _apply_undistorted_masks_to_scene_images(
                scene_dir=scene_dir,
                undist_mask_images_dir=undist_masks_dir / "images",
                threshold=127,
            )
            logging.info("Applied strict foreground masks to %s undistorted images", applied)

    return image_count


def prepare_3dgs_scene_from_sfm(
    frames_dir: Path,
    masks_dir: Path | None,
    colmap_dir: Path,
    scene_dir: Path,
    strict_mask_training: bool = True,
) -> Path:
    """Prepare a 3DGS scene directory from SfM/COLMAP outputs.

    Expected COLMAP structure: colmap_dir/sparse/0/{cameras.bin,images.bin,points3D.bin}
    Output scene structure: scene_dir/images + scene_dir/sparse/0
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not colmap_dir.exists():
        raise FileNotFoundError(f"SfM/COLMAP directory not found: {colmap_dir}")

    image_count = _prepare_undistorted_scene_from_colmap(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        colmap_dir=colmap_dir,
        scene_dir=scene_dir,
        strict_mask_training=strict_mask_training,
    )

    prep_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frames_dir": str(frames_dir),
        "colmap_dir": str(colmap_dir),
        "masks_dir": str(masks_dir) if masks_dir else None,
        "scene_dir": str(scene_dir),
        "num_images": image_count,
        "scene_prep_mode": "colmap_image_undistorter",
        "strict_mask_training": strict_mask_training,
        "sparse_files": ["cameras.bin", "images.bin", "points3D.bin"],
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
    masks_dir: Path | None,
    colmap_dir: Path,
    gs_repo_dir: Path,
    scene_dir: Path,
    model_dir: Path,
    iterations: int = 30000,
    python_exe: str = "python",
    strict_mask_training: bool = True,
) -> None:
    """Prepare scene from SfM and launch 3DGS training."""
    prepare_3dgs_scene_from_sfm(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        colmap_dir=colmap_dir,
        scene_dir=scene_dir,
        strict_mask_training=strict_mask_training,
    )
    train_3dgs_from_scene(
        gs_repo_dir=gs_repo_dir,
        scene_dir=scene_dir,
        model_dir=model_dir,
        iterations=iterations,
        python_exe=python_exe,
    )
