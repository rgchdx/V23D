from pathlib import Path
import json
import logging
import shutil
import subprocess
from typing import Iterable

from src.sfm.pose_utils import load_colmap_poses_text

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_COLMAP_HELP_CACHE: dict[str, str] = {}


def _run_command(cmd: list[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _get_subcommand_help(subcommand: str) -> str:
    if subcommand in _COLMAP_HELP_CACHE:
        return _COLMAP_HELP_CACHE[subcommand]

    help_text = ""
    for help_flag in ["-h", "--help"]:
        try:
            proc = subprocess.run(
                ["colmap", subcommand, help_flag],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.stdout:
                help_text = proc.stdout
                break
        except Exception:
            continue

    _COLMAP_HELP_CACHE[subcommand] = help_text
    return help_text


def _supports_option(subcommand: str, option_name: str) -> bool:
    return option_name in _get_subcommand_help(subcommand)


def _append_if_supported(cmd: list[str], subcommand: str, option_name: str, option_value: str) -> None:
    if _supports_option(subcommand, option_name):
        cmd.extend([option_name, option_value])


def _append_one_of(cmd: list[str], subcommand: str, options: list[tuple[str, str]]) -> None:
    for opt, val in options:
        if _supports_option(subcommand, opt):
            cmd.extend([opt, val])
            return


def _collect_images(frames_dir: Path) -> list[Path]:
    return sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _write_image_list(images: Iterable[Path], out_path: Path) -> None:
    out_path.write_text("\n".join(p.name for p in images) + "\n", encoding="utf-8")


def _prepare_colmap_masks(images: list[Path], masks_dir: Path, output_dir: Path) -> Path | None:
    if not masks_dir.exists():
        return None

    colmap_mask_dir = output_dir / "colmap_masks"
    colmap_mask_dir.mkdir(parents=True, exist_ok=True)

    found_any = False
    for image_path in images:
        src_mask = masks_dir / f"{image_path.stem}.png"
        if not src_mask.exists():
            continue
        dst_mask = colmap_mask_dir / f"{image_path.name}.png"
        shutil.copy2(src_mask, dst_mask)
        found_any = True

    return colmap_mask_dir if found_any else None


def _list_sparse_models(sparse_path: Path) -> list[Path]:
    return sorted([p for p in sparse_path.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))


def _count_registered_images_from_images_txt(images_txt: Path) -> int:
    if not images_txt.exists():
        return 0
    rows = [line.strip() for line in images_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [line for line in rows if not line.startswith("#")]
    return len(rows) // 2


def _select_best_model_dir(sparse_path: Path) -> Path:
    model_dirs = _list_sparse_models(sparse_path)
    if not model_dirs:
        raise RuntimeError(f"COLMAP mapper did not produce sparse model under {sparse_path}")

    best_dir = None
    best_count = -1
    for model_dir in model_dirs:
        _run_command(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(model_dir),
                "--output_type",
                "TXT",
            ]
        )
        reg_count = _count_registered_images_from_images_txt(model_dir / "images.txt")
        logging.info("Model %s registered images: %s", model_dir.name, reg_count)
        if reg_count > best_count:
            best_count = reg_count
            best_dir = model_dir

    if best_dir is None:
        raise RuntimeError("No valid sparse model found")
    return best_dir


def run_colmap(
    frames_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    matcher: str = "sequential",
    use_gpu: bool = False,
) -> None:
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
    ]
    _append_one_of(
        feature_cmd,
        "feature_extractor",
        [
            ("--FeatureExtraction.use_gpu", "1" if use_gpu else "0"),
            ("--SiftExtraction.use_gpu", "1" if use_gpu else "0"),
        ],
    )
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.estimate_affine_shape", "1")
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.domain_size_pooling", "1")
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.max_num_features", "24000")
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.peak_threshold", "0.004")

    if colmap_mask_dir is not None:
        _append_if_supported(feature_cmd, "feature_extractor", "--ImageReader.mask_path", str(colmap_mask_dir))

    if matcher not in {"sequential", "exhaustive"}:
        raise ValueError("matcher must be 'sequential' or 'exhaustive'")

    if matcher == "sequential":
        match_cmd = ["colmap", "sequential_matcher", "--database_path", str(database_path)]
        _append_one_of(
            match_cmd,
            "sequential_matcher",
            [
                ("--FeatureMatching.use_gpu", "1" if use_gpu else "0"),
                ("--SiftMatching.use_gpu", "1" if use_gpu else "0"),
            ],
        )
        _append_one_of(
            match_cmd,
            "sequential_matcher",
            [
                ("--FeatureMatching.guided_matching", "1"),
                ("--SiftMatching.guided_matching", "1"),
            ],
        )
        _append_if_supported(match_cmd, "sequential_matcher", "--SiftMatching.max_ratio", "0.96")
        _append_if_supported(match_cmd, "sequential_matcher", "--SiftMatching.max_error", "12")
        _append_if_supported(match_cmd, "sequential_matcher", "--SiftMatching.min_num_inliers", "6")
        _append_if_supported(match_cmd, "sequential_matcher", "--TwoViewGeometry.max_error", "12")
        _append_if_supported(match_cmd, "sequential_matcher", "--TwoViewGeometry.min_num_inliers", "6")
        _append_if_supported(match_cmd, "sequential_matcher", "--SequentialMatching.overlap", "80")
        _append_if_supported(match_cmd, "sequential_matcher", "--SequentialMatching.quadratic_overlap", "1")
        _append_if_supported(match_cmd, "sequential_matcher", "--SequentialMatching.loop_detection", "1")
    else:
        match_cmd = ["colmap", "exhaustive_matcher", "--database_path", str(database_path)]
        _append_one_of(
            match_cmd,
            "exhaustive_matcher",
            [
                ("--FeatureMatching.use_gpu", "1" if use_gpu else "0"),
                ("--SiftMatching.use_gpu", "1" if use_gpu else "0"),
            ],
        )
        _append_one_of(
            match_cmd,
            "exhaustive_matcher",
            [
                ("--FeatureMatching.guided_matching", "1"),
                ("--SiftMatching.guided_matching", "1"),
            ],
        )
        _append_if_supported(match_cmd, "exhaustive_matcher", "--SiftMatching.max_ratio", "0.96")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--SiftMatching.max_error", "12")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--SiftMatching.min_num_inliers", "6")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--TwoViewGeometry.max_error", "12")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--TwoViewGeometry.min_num_inliers", "6")

    transitive_cmd = ["colmap", "transitive_matcher", "--database_path", str(database_path)]
    _append_one_of(
        transitive_cmd,
        "transitive_matcher",
        [
            ("--FeatureMatching.use_gpu", "1" if use_gpu else "0"),
            ("--SiftMatching.use_gpu", "1" if use_gpu else "0"),
        ],
    )
    _append_one_of(
        transitive_cmd,
        "transitive_matcher",
        [
            ("--FeatureMatching.guided_matching", "1"),
            ("--SiftMatching.guided_matching", "1"),
        ],
    )
    _append_if_supported(transitive_cmd, "transitive_matcher", "--SiftMatching.max_ratio", "0.97")
    _append_if_supported(transitive_cmd, "transitive_matcher", "--TwoViewGeometry.max_error", "14")
    _append_if_supported(transitive_cmd, "transitive_matcher", "--TwoViewGeometry.min_num_inliers", "5")

    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(frames_dir),
        "--output_path",
        str(sparse_path),
    ]
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.multiple_models", "0")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.init_min_num_inliers", "6")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.abs_pose_min_num_inliers", "4")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.abs_pose_min_inlier_ratio", "0.005")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.filter_max_reproj_error", "20")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.filter_min_tri_angle", "0.1")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.tri_ignore_two_view_tracks", "0")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.max_reg_trials", "5")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.ba_refine_focal_length", "1")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.ba_refine_principal_point", "0")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.ba_refine_extra_params", "1")

    logging.info("Running COLMAP on %s images from %s", len(images), frames_dir)
    _run_command(feature_cmd)
    _run_command(match_cmd)
    _run_command(transitive_cmd)
    _run_command(mapper_cmd)

    best_model_dir = _select_best_model_dir(sparse_path)
    images_txt = best_model_dir / "images.txt"
    if not images_txt.exists():
        raise RuntimeError("COLMAP output missing images.txt after model conversion")

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
        "selected_model_dir": str(best_model_dir),
    }
    (output_dir / "colmap_run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    logging.info("COLMAP complete. Registered %s/%s images", len(poses), len(images))
from pathlib import Path
import json
import logging
import shutil
import subprocess
import tempfile
from typing import Iterable

import cv2

from src.sfm.pose_utils import load_colmap_poses_text

# Set of supported image extensions for COLMAP input.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_COLMAP_HELP_CACHE: dict[str, str] = {}

# run_command is a helper to run a subprocess command with logging and error checking.
def _run_command(cmd: list[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _get_subcommand_help(subcommand: str) -> str:
    if subcommand in _COLMAP_HELP_CACHE:
        return _COLMAP_HELP_CACHE[subcommand]

    help_text = ""
    for help_flag in ["-h", "--help"]:
        try:
            proc = subprocess.run(
                ["colmap", subcommand, help_flag],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.stdout:
                help_text = proc.stdout
                break
        except Exception:
            continue

    _COLMAP_HELP_CACHE[subcommand] = help_text
    return help_text


def _supports_option(subcommand: str, option_name: str) -> bool:
    return option_name in _get_subcommand_help(subcommand)


def _append_if_supported(cmd: list[str], subcommand: str, option_name: str, option_value: str) -> None:
    if _supports_option(subcommand, option_name):
        cmd.extend([option_name, option_value])
    else:
        logging.debug("Skipping unsupported %s option: %s", subcommand, option_name)


def _append_one_of(cmd: list[str], subcommand: str, options: list[tuple[str, str]]) -> None:
    for opt, val in options:
        if _supports_option(subcommand, opt):
            cmd.extend([opt, val])
            return


def _list_sparse_models(sparse_path: Path) -> list[Path]:
    return sorted(
        [p for p in sparse_path.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def _count_registered_images_in_txt(images_txt: Path) -> int:
    if not images_txt.exists():
        return 0

    count = 0
    for raw in images_txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            count += 1
    return count


def _select_best_model_dir(sparse_path: Path) -> Path:
    model_dirs = _list_sparse_models(sparse_path)
    if not model_dirs:
        raise RuntimeError(f"COLMAP mapper did not produce sparse model under {sparse_path}")

    best_model = None
    best_count = -1

    for model_dir in model_dirs:
        convert_cmd = [
            "colmap",
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ]
        _run_command(convert_cmd)

        reg_count = _count_registered_images_in_txt(model_dir / "images.txt")
        logging.info("Model %s registered images: %s", model_dir.name, reg_count)
        if reg_count > best_count:
            best_count = reg_count
            best_model = model_dir

    if best_model is None:
        raise RuntimeError("Could not select a valid COLMAP sparse model")

    logging.info("Selected model %s with %s registered images", best_model.name, best_count)
    return best_model


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


def _write_hard_masked_frames(
    images: list[Path],
    frames_dir: Path,
    masks_dir: Path,
    out_dir: Path,
    threshold: int = 127,
) -> int:
    """Write RGB frames with background zeroed by masks. Returns count written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for image_path in images:
        src_img = frames_dir / image_path.name
        src_mask = masks_dir / f"{image_path.stem}.png"
        if not src_img.exists() or not src_mask.exists():
            continue

        img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue

        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        fg = (msk > threshold)
        out = img.copy()
        out[~fg] = 0
        cv2.imwrite(str(out_dir / image_path.name), out)
        written += 1

    return written


def _compute_in_mask_observation_ratio(images_txt: Path, masks_dir: Path, threshold: int = 127) -> float:
    if not images_txt.exists() or not masks_dir.exists():
        return 0.0

    lines = images_txt.read_text(encoding="utf-8").splitlines()
    i = 0
    total = 0
    inside = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_name = parts[9]
        obs_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        i += 2

        mask_path = masks_dir / f"{Path(image_name).stem}.png"
        if not mask_path.exists() or not obs_line:
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        vals = obs_line.split()
        h, w = mask.shape[:2]
        for j in range(0, len(vals), 3):
            if j + 2 >= len(vals):
                break
            x = int(round(float(vals[j])))
            y = int(round(float(vals[j + 1])))
            pid = int(vals[j + 2])
            if pid < 0:
                continue
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            total += 1
            if mask[y, x] > threshold:
                inside += 1

    return (inside / total) if total > 0 else 0.0


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
    strict_human_only: bool = False,
    min_in_mask_ratio: float = 0.98,
    hard_mask_images: bool = False,
    mask_threshold: int = 127,
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

    tmp_hardmask_dir = None
    image_path_for_colmap = frames_dir
    if hard_mask_images:
        if not masks_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {masks_dir}")
        tmp_hardmask_dir = Path(tempfile.mkdtemp(prefix="v23d_colmap_masked_"))
        written = _write_hard_masked_frames(
            images=images,
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            out_dir=tmp_hardmask_dir,
            threshold=mask_threshold,
        )
        if written == 0:
            raise RuntimeError("hard_mask_images=True but no masked frames were written.")
        image_path_for_colmap = tmp_hardmask_dir
        logging.info("Using hard-masked frames for COLMAP image_path: %s (%s images)", image_path_for_colmap, written)

    feature_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path_for_colmap),
        "--image_list_path",
        str(image_list_path),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
    ]
    _append_one_of(
        feature_cmd,
        "feature_extractor",
        [
            ("--FeatureExtraction.use_gpu", "1" if use_gpu else "0"),
            ("--SiftExtraction.use_gpu", "1" if use_gpu else "0"),
        ],
    )
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.estimate_affine_shape", "1")
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.domain_size_pooling", "1")
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.max_num_features", "24000")
    _append_if_supported(feature_cmd, "feature_extractor", "--SiftExtraction.peak_threshold", "0.004")

    if colmap_mask_dir is not None:
        _append_if_supported(feature_cmd, "feature_extractor", "--ImageReader.mask_path", str(colmap_mask_dir))
    elif strict_human_only:
        raise RuntimeError("strict_human_only=True but no valid masks were found.")

    if matcher not in {"sequential", "exhaustive"}:
        raise ValueError("matcher must be 'sequential' or 'exhaustive'")

    if matcher == "sequential":
        match_cmd = [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database_path),
        ]
        _append_one_of(
            match_cmd,
            "sequential_matcher",
            [
                ("--FeatureMatching.use_gpu", "1" if use_gpu else "0"),
                ("--SiftMatching.use_gpu", "1" if use_gpu else "0"),
            ],
        )
        _append_one_of(
            match_cmd,
            "sequential_matcher",
            [("--FeatureMatching.guided_matching", "1"), ("--SiftMatching.guided_matching", "1")],
        )
        _append_if_supported(match_cmd, "sequential_matcher", "--SiftMatching.max_ratio", "0.96")
        _append_if_supported(match_cmd, "sequential_matcher", "--SiftMatching.max_error", "12")
        _append_if_supported(match_cmd, "sequential_matcher", "--SiftMatching.min_num_inliers", "6")
        _append_if_supported(match_cmd, "sequential_matcher", "--TwoViewGeometry.max_error", "12")
        _append_if_supported(match_cmd, "sequential_matcher", "--TwoViewGeometry.min_num_inliers", "6")
        _append_if_supported(match_cmd, "sequential_matcher", "--SequentialMatching.overlap", "80")
        _append_if_supported(match_cmd, "sequential_matcher", "--SequentialMatching.quadratic_overlap", "1")
        _append_if_supported(match_cmd, "sequential_matcher", "--SequentialMatching.loop_detection", "1")
    else:
        match_cmd = [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ]
        _append_one_of(
            match_cmd,
            "exhaustive_matcher",
            [
                ("--FeatureMatching.use_gpu", "1" if use_gpu else "0"),
                ("--SiftMatching.use_gpu", "1" if use_gpu else "0"),
            ],
        )
        _append_one_of(
            match_cmd,
            "exhaustive_matcher",
            [("--FeatureMatching.guided_matching", "1"), ("--SiftMatching.guided_matching", "1")],
        )
        _append_if_supported(match_cmd, "exhaustive_matcher", "--SiftMatching.max_ratio", "0.96")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--SiftMatching.max_error", "12")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--SiftMatching.min_num_inliers", "6")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--TwoViewGeometry.max_error", "12")
        _append_if_supported(match_cmd, "exhaustive_matcher", "--TwoViewGeometry.min_num_inliers", "6")

    transitive_cmd = [
        "colmap",
        "transitive_matcher",
        "--database_path",
        str(database_path),
    ]
    _append_one_of(
        transitive_cmd,
        "transitive_matcher",
        [
            ("--FeatureMatching.use_gpu", "1" if use_gpu else "0"),
            ("--SiftMatching.use_gpu", "1" if use_gpu else "0"),
        ],
    )
    _append_one_of(
        transitive_cmd,
        "transitive_matcher",
        [("--FeatureMatching.guided_matching", "1"), ("--SiftMatching.guided_matching", "1")],
    )
    _append_if_supported(transitive_cmd, "transitive_matcher", "--SiftMatching.max_ratio", "0.97")
    _append_if_supported(transitive_cmd, "transitive_matcher", "--TwoViewGeometry.max_error", "14")
    _append_if_supported(transitive_cmd, "transitive_matcher", "--TwoViewGeometry.min_num_inliers", "5")

    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path_for_colmap),
        "--output_path",
        str(sparse_path),
    ]
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.multiple_models", "0")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.init_min_num_inliers", "6")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.abs_pose_min_num_inliers", "4")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.abs_pose_min_inlier_ratio", "0.005")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.filter_max_reproj_error", "20")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.filter_min_tri_angle", "0.1")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.tri_ignore_two_view_tracks", "0")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.max_reg_trials", "5")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.ba_refine_focal_length", "1")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.ba_refine_principal_point", "0")
    _append_if_supported(mapper_cmd, "mapper", "--Mapper.ba_refine_extra_params", "1")

    try:
        logging.info("Running COLMAP on %s images from %s", len(images), image_path_for_colmap)
        _run_command(feature_cmd)
        _run_command(match_cmd)
        _run_command(transitive_cmd)
        _run_command(mapper_cmd)

        best_model_dir = _select_best_model_dir(sparse_path)

        images_txt = best_model_dir / "images.txt"
        if not images_txt.exists():
            raise RuntimeError("COLMAP output missing images.txt after model conversion.")

        poses = load_colmap_poses_text(images_txt)
        poses_path = output_dir / "poses.json"
        poses_path.write_text(json.dumps(poses, indent=2), encoding="utf-8")

        in_mask_ratio = None
        if colmap_mask_dir is not None:
            in_mask_ratio = _compute_in_mask_observation_ratio(images_txt, masks_dir=masks_dir)
            logging.info("COLMAP in-mask observation ratio: %.4f", in_mask_ratio)
            if strict_human_only and in_mask_ratio < min_in_mask_ratio:
                raise RuntimeError(
                    f"In-mask ratio {in_mask_ratio:.4f} is below required {min_in_mask_ratio:.4f}. "
                    "Tighten masks or remove problematic frames."
                )

        run_meta = {
            "frames_dir": str(frames_dir),
            "masks_dir": str(masks_dir),
            "output_dir": str(output_dir),
            "matcher": matcher,
            "use_gpu": use_gpu,
            "num_input_images": len(images),
            "num_registered_poses": len(poses),
            "selected_model_dir": str(best_model_dir),
            "strict_human_only": strict_human_only,
            "min_in_mask_ratio": min_in_mask_ratio,
            "hard_mask_images": hard_mask_images,
            "mask_threshold": mask_threshold,
            "in_mask_observation_ratio": in_mask_ratio,
        }
        (output_dir / "colmap_run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        logging.info("COLMAP complete. Registered %s/%s images.", len(poses), len(images))
    finally:
        if tmp_hardmask_dir is not None:
            shutil.rmtree(tmp_hardmask_dir, ignore_errors=True)

    
