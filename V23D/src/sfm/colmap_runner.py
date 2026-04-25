from pathlib import Path
import json
import logging
import shutil
import subprocess
from typing import Iterable

from src.sfm.pose_utils import load_colmap_poses_text

# Set of supported image extensions for COLMAP input.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# run_command is a helper to run a subprocess command with logging and error checking.
def _run_command(cmd: list[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# First collect the images to be preprocessed, ensuring they are sorted and valid
def _collect_images(frames_dir: Path) -> list[Path]:
    return sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


# COLMAP can use masks to ignore background features. We prepare a COLMAP-compatible mask directory if masks are provided.
def _write_image_list(images: Iterable[Path], out_path: Path) -> None:
    out_path.write_text("\n".join(p.name for p in images) + "\n", encoding="utf-8")


# If masks are provided, we copy them to a COLMAP-compatible directory structure. We return the path to the COLMAP mask directory
# or None if no masks were found/copied. COLMAP expects masks to be named <image_name>.png and placed in a single directory.
def _prepare_colmap_masks(images: list[Path], masks_dir: Path, output_dir: Path) -> Path | None:
    if not masks_dir.exists():
        return None

    # COLMAP expects masks at: <mask_path>/<image_name>.png
    colmap_mask_dir = output_dir / "colmap_masks"
    colmap_mask_dir.mkdir(parents=True, exist_ok=True)

    found_any = False
    for image_path in images:
        # We generate masks as <stem>.png in preprocessing.
        src_mask = masks_dir / f"{image_path.stem}.png"
        if not src_mask.exists():
            continue
        dst_mask = colmap_mask_dir / f"{image_path.name}.png"
        shutil.copy2(src_mask, dst_mask)
        found_any = True

    return colmap_mask_dir if found_any else None


# Main function to run COLMAP SfM with specified settings. 
""" 
HOW COLMAP WORKS:
1. Feature Extraction: COLMAP detects key points and computes descriptors for each image. We configure it to use SIFT features with 
    settings that are more robust to style/lighting changes, which are common in diffusion-generated images. We also enable GPU 
    acceleration if available. If masks are provided, we tell COLMAP to ignore background features.
2. Feature Matching: Depending on the matcher choice, COLMAP will either match features sequentially (only between adjacent images) or
    exhaustively (between all pairs). We use guided matching with a higher max ratio and error threshold to allow for more matches since 
    the outputs are diffusion-generated and may have less consistent features.
3. Mapping: COLMAP's mapper takes the matched features and estimates camera poses and a sparse point cloud. We set stricter thresholds
    for inliers and reprojection error to filter out bad matches, and we enable bundle adjustment refinement of focal lengths and extra
    parameters to improve accuracy
4. Model Conversion: After mapping, we convert the COLMAP model to TXT format for easier parsing of camera poses.
5. Pose Extraction: We parse the images.txt output from COLMAP to extract camera poses and save them as a JSON file for downstream use.
    (like for 3DGS training).
6. Metadata Saving: We also save a JSON file summarizing the COLMAP run, including settings and statistics about the number of images
    and registered poses.
"""
def run_colmap(
    frames_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    matcher: str = "sequential",
    use_gpu: bool = False,
) -> None:
    """Run COLMAP SfM with settings tuned for hard adjacent-view matching.

    Args:
        frames_dir: input frame directory.
        masks_dir: foreground masks directory (stem-matched png masks).
        output_dir: output COLMAP workspace.
        matcher: one of {"sequential", "exhaustive"}.
        use_gpu: enable COLMAP GPU paths when available.
    """
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP executable not found in PATH.")

    images = _collect_images(frames_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {frames_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    sparse_path = output_dir / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    image_list_path = output_dir / "image_list.txt"
    _write_image_list(images, image_list_path)

    colmap_mask_dir = _prepare_colmap_masks(images, masks_dir, output_dir)

    feature_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(frames_dir),
        "--image_list_path",
        str(image_list_path),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
        # More robust for style/lighting drift (common with diffusion outputs).
        "--SiftExtraction.estimate_affine_shape",
        "1",
        "--SiftExtraction.domain_size_pooling",
        "1",
        "--SiftExtraction.max_num_features",
        "8192",
        "--SiftExtraction.contrast_threshold",
        "0.01",
    ]
    if colmap_mask_dir is not None:
        feature_cmd.extend(["--ImageReader.mask_path", str(colmap_mask_dir)])

    if matcher not in {"sequential", "exhaustive"}:
        raise ValueError("matcher must be 'sequential' or 'exhaustive'")

    if matcher == "sequential":
        match_cmd = [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            "1" if use_gpu else "0",
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.max_ratio",
            "0.9",
            "--SiftMatching.max_error",
            "8",
            "--SiftMatching.min_num_inliers",
            "12",
            "--SequentialMatching.overlap",
            "20",
            "--SequentialMatching.quadratic_overlap",
            "1",
            "--SequentialMatching.loop_detection",
            "0",
        ]
    else:
        match_cmd = [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            "1" if use_gpu else "0",
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.max_ratio",
            "0.9",
            "--SiftMatching.max_error",
            "8",
            "--SiftMatching.min_num_inliers",
            "12",
        ]

    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(frames_dir),
        "--output_path",
        str(sparse_path),
        "--Mapper.init_min_num_inliers",
        "30",
        "--Mapper.abs_pose_min_num_inliers",
        "10",
        "--Mapper.abs_pose_min_inlier_ratio",
        "0.08",
        "--Mapper.filter_max_reproj_error",
        "8",
        "--Mapper.ba_refine_focal_length",
        "1",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "1",
    ]

    logging.info("Running COLMAP on %s images from %s", len(images), frames_dir)
    _run_command(feature_cmd)
    _run_command(match_cmd)
    _run_command(mapper_cmd)

    model_0 = sparse_path / "0"
    if not model_0.exists():
        raise RuntimeError("COLMAP mapper did not produce sparse/0 model.")

    convert_cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(model_0),
        "--output_path",
        str(model_0),
        "--output_type",
        "TXT",
    ]
    _run_command(convert_cmd)

    images_txt = model_0 / "images.txt"
    if not images_txt.exists():
        raise RuntimeError("COLMAP output missing images.txt after model conversion.")

    poses = load_colmap_poses_text(images_txt)
    poses_path = output_dir / "poses.json"
    poses_path.write_text(json.dumps(poses, indent=2), encoding="utf-8")

    run_meta = {
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "output_dir": str(output_dir),
        "matcher": matcher,
        "use_gpu": use_gpu,
        "num_input_images": len(images),
        "num_registered_poses": len(poses),
    }
    (output_dir / "colmap_run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    logging.info("COLMAP complete. Registered %s/%s images.", len(poses), len(images))

    
