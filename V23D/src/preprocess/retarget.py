from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MaskStats:
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int
    bbox_w: int
    bbox_h: int
    centroid_x: float
    centroid_y: float
    area: int


def _iter_images(frames_dir: Path) -> list[Path]:
    return sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _read_mask(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask


def _largest_component(binary: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return binary
    fg_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == fg_idx] = 255
    return out


def _mask_stats(mask: np.ndarray) -> MaskStats | None:
    binary = (mask > 127).astype(np.uint8) * 255
    if binary.max() == 0:
        return None
    binary = _largest_component(binary)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area = int(len(xs))
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    return MaskStats(
        bbox_x0=x0,
        bbox_y0=y0,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_w=int(x1 - x0 + 1),
        bbox_h=int(y1 - y0 + 1),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        area=area,
    )


def _anchor_from_stats(stats: MaskStats, anchor_mode: str) -> tuple[float, float]:
    cx_box = 0.5 * (stats.bbox_x0 + stats.bbox_x1)
    if anchor_mode == "mask_centroid":
        return stats.centroid_x, stats.centroid_y
    if anchor_mode == "bbox_center":
        return cx_box, 0.5 * (stats.bbox_y0 + stats.bbox_y1)
    if anchor_mode == "head":
        return cx_box, stats.bbox_y0 + 0.18 * stats.bbox_h
    if anchor_mode == "torso":
        return cx_box, stats.bbox_y0 + 0.38 * stats.bbox_h
    if anchor_mode == "pelvis":
        return cx_box, stats.bbox_y0 + 0.58 * stats.bbox_h
    raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")


def _smooth_sequence(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def retarget_frames_by_masks(
    frames_dir: Path,
    masks_dir: Path,
    out_frames_dir: Path,
    out_masks_dir: Path | None = None,
    anchor_mode: str = "torso",
    output_size: tuple[int, int] | None = None,
    target_anchor_xy: tuple[float, float] = (0.5, 0.42),
    target_subject_height_ratio: float = 0.82,
    smooth_alpha: float = 0.25,
    hard_mask_background: bool = False,
    border_mode: str = "replicate",
) -> Path:
    """Normalize subject location/scale across frames using the masks.

    This is useful when SfM/3DGS suffers because the human drifts vertically,
    changes apparent size, or is not consistently centered. The function estimates
    a stable body anchor (e.g. torso) and writes affine-warped frames/masks.
    """
    if anchor_mode not in {"mask_centroid", "bbox_center", "head", "torso", "pelvis"}:
        raise ValueError("anchor_mode must be one of: mask_centroid, bbox_center, head, torso, pelvis")

    frame_paths = _iter_images(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No images found in {frames_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    if out_masks_dir is not None:
        out_masks_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")
    src_h, src_w = first.shape[:2]
    out_w, out_h = output_size if output_size is not None else (src_w, src_h)

    records: list[dict] = []
    raw_anchor_x: list[float] = []
    raw_anchor_y: list[float] = []
    raw_scales: list[float] = []
    valid_flags: list[bool] = []
    image_names: list[str] = []

    desired_anchor_x = float(target_anchor_xy[0]) * out_w
    desired_anchor_y = float(target_anchor_xy[1]) * out_h
    desired_subject_h = float(target_subject_height_ratio) * out_h

    last_anchor = (0.5 * src_w, 0.5 * src_h)
    last_scale = 1.0

    for frame_path in frame_paths:
        mask_path = masks_dir / f"{frame_path.stem}.png"
        mask = _read_mask(mask_path)
        stats = _mask_stats(mask) if mask is not None else None

        if stats is None or stats.bbox_h <= 1:
            anchor_x, anchor_y = last_anchor
            scale = last_scale
            valid = False
        else:
            anchor_x, anchor_y = _anchor_from_stats(stats, anchor_mode=anchor_mode)
            scale = desired_subject_h / max(float(stats.bbox_h), 1.0)
            scale = float(np.clip(scale, 0.5, 2.5))
            last_anchor = (anchor_x, anchor_y)
            last_scale = scale
            valid = True

        raw_anchor_x.append(anchor_x)
        raw_anchor_y.append(anchor_y)
        raw_scales.append(scale)
        valid_flags.append(valid)
        image_names.append(frame_path.name)
        records.append(
            {
                "file_name": frame_path.name,
                "mask_found": mask is not None,
                "mask_valid": valid,
                "raw_anchor_x": float(anchor_x),
                "raw_anchor_y": float(anchor_y),
                "raw_scale": float(scale),
                "bbox_height": None if stats is None else int(stats.bbox_h),
                "bbox_width": None if stats is None else int(stats.bbox_w),
            }
        )

    smooth_anchor_x = _smooth_sequence(raw_anchor_x, alpha=smooth_alpha)
    smooth_anchor_y = _smooth_sequence(raw_anchor_y, alpha=smooth_alpha)
    smooth_scales = _smooth_sequence(raw_scales, alpha=smooth_alpha)

    border_flag = cv2.BORDER_REPLICATE if border_mode == "replicate" else cv2.BORDER_CONSTANT

    for idx, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")

        scale = smooth_scales[idx]
        tx = desired_anchor_x - scale * smooth_anchor_x[idx]
        ty = desired_anchor_y - scale * smooth_anchor_y[idx]
        affine = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)

        warped = cv2.warpAffine(
            img,
            affine,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=border_flag,
            borderValue=(0, 0, 0),
        )

        src_mask_path = masks_dir / f"{frame_path.stem}.png"
        mask = _read_mask(src_mask_path)
        warped_mask = None
        if mask is not None:
            warped_mask = cv2.warpAffine(
                mask,
                affine,
                (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_mask = ((warped_mask > 127).astype(np.uint8) * 255)

        if hard_mask_background and warped_mask is not None:
            warped = warped.copy()
            warped[warped_mask <= 127] = 0

        out_frame_path = out_frames_dir / frame_path.name
        cv2.imwrite(str(out_frame_path), warped)

        if out_masks_dir is not None and warped_mask is not None:
            cv2.imwrite(str(out_masks_dir / f"{frame_path.stem}.png"), warped_mask)

        records[idx]["smoothed_anchor_x"] = float(smooth_anchor_x[idx])
        records[idx]["smoothed_anchor_y"] = float(smooth_anchor_y[idx])
        records[idx]["smoothed_scale"] = float(smooth_scales[idx])
        records[idx]["affine"] = [[float(v) for v in row] for row in affine.tolist()]

    summary = {
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "out_frames_dir": str(out_frames_dir),
        "out_masks_dir": str(out_masks_dir) if out_masks_dir is not None else None,
        "anchor_mode": anchor_mode,
        "output_size": [int(out_w), int(out_h)],
        "target_anchor_xy": [float(target_anchor_xy[0]), float(target_anchor_xy[1])],
        "target_subject_height_ratio": float(target_subject_height_ratio),
        "smooth_alpha": float(smooth_alpha),
        "hard_mask_background": bool(hard_mask_background),
        "num_frames": len(frame_paths),
        "num_valid_masks": int(sum(valid_flags)),
        "mean_scale": float(np.mean(smooth_scales)),
        "min_scale": float(np.min(smooth_scales)),
        "max_scale": float(np.max(smooth_scales)),
        "frames": records,
    }
    out_meta = out_frames_dir / "retarget_metadata.json"
    out_meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Retargeted %s frames into %s", len(frame_paths), out_frames_dir)
    return out_meta