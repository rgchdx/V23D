from pathlib import Path
import logging
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return binary

    # 0 is background; choose largest foreground area.
    fg_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == fg_idx] = 255
    return out


def _postprocess_mask(
    alpha: np.ndarray,
    alpha_threshold: int,
    open_kernel: int,
    close_kernel: int,
    erode_px: int,
    keep_largest: bool,
) -> np.ndarray:
    binary = (alpha >= alpha_threshold).astype(np.uint8) * 255

    if open_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    if keep_largest:
        binary = _largest_component(binary)

    if erode_px > 0:
        ksz = 2 * erode_px + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        binary = cv2.erode(binary, k, iterations=1)

    return binary


def generate_masks(
    frames_dir: Path,
    masks_dir: Path,
    model: str = "rembg",
    alpha_threshold: int = 180,
    open_kernel: int = 3,
    close_kernel: int = 7,
    erode_px: int = 2,
    keep_largest: bool = True,
    suppress_white_bg: bool = False,
    white_threshold: int = 245,
    white_soft_margin: int = 0,
) -> None:
    """
    Generate person masks for frames.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)

    if model not in {"rembg", "whitebg"}:
        raise ValueError(f"Unsupported masking model: {model}. Use rembg or whitebg")

    frame_paths: Iterable[Path] = sorted(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )

    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")

    session = None
    if model == "rembg":
        session = new_session(model_name="u2net", providers=["CPUExecutionProvider"])

    for frame_path in tqdm(frame_paths, desc="Generating masks", unit="frame"):
        logging.info("Masking %s", frame_path.name)

        with Image.open(frame_path).convert("RGB") as rgb:
            rgb_np = np.array(rgb)

        if model == "rembg":
            rgba = remove(Image.fromarray(rgb_np), session=session)
            alpha = np.array(rgba)[..., 3]
            binary = _postprocess_mask(
                alpha=alpha,
                alpha_threshold=alpha_threshold,
                open_kernel=open_kernel,
                close_kernel=close_kernel,
                erode_px=erode_px,
                keep_largest=keep_largest,
            )
        else:
            # Chroma-key style: keep non-white pixels as foreground.
            white = (
                (rgb_np[..., 0] >= white_threshold)
                & (rgb_np[..., 1] >= white_threshold)
                & (rgb_np[..., 2] >= white_threshold)
            )
            binary = (~white).astype(np.uint8) * 255
            if open_kernel > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
            if close_kernel > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            if keep_largest:
                binary = _largest_component(binary)
            if erode_px > 0:
                ksz = 2 * erode_px + 1
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                binary = cv2.erode(binary, k, iterations=1)

        if suppress_white_bg:
            t = int(white_threshold)
            white = (
                (rgb_np[..., 0] >= t)
                & (rgb_np[..., 1] >= t)
                & (rgb_np[..., 2] >= t)
            )
            if white_soft_margin > 0:
                ksz = 2 * int(white_soft_margin) + 1
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                white = cv2.dilate(white.astype(np.uint8), k, iterations=1).astype(bool)
            binary[white] = 0

            # Re-enforce largest component after white suppression.
            if keep_largest:
                binary = _largest_component(binary)

        mask_path = masks_dir / f"{frame_path.stem}.png"
        Image.fromarray(binary, mode="L").save(mask_path)

    print(f"Saved {len(frame_paths)} masks to {masks_dir}")
    logging.info("Saved masks to %s", masks_dir)

