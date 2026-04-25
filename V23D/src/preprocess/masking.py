from pathlib import Path
import logging
from typing import Iterable

import numpy as np
from PIL import Image
from rembg import remove


def generate_masks(frames_dir: Path, masks_dir: Path, model: str = "rembg") -> None:
    """
    Generate person masks for frames.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)

    if model != "rembg":
        raise ValueError(f"Unsupported masking model: {model}")

    frame_paths: Iterable[Path] = sorted(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )

    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")
    
    for frame_path in frame_paths:
        logging.info("Masking %s", frame_path.name)

        with Image.open(frame_path).convert("RGB") as rgb:
            rgba = remove(rgb)
        
        alpha = np.array(rgba)[..., 3]
        binary = (alpha > 0).astype(np.uint8) * 255

        mask_path = masks_dir / f"{frame_path.stem}.png"
        Image.fromarray(binary, mode="L").save(mask_path)
    logging.info("Saved masks to %s", masks_dir)

