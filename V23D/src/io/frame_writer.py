from pathlib import Path
from typing import Dict, List

import cv2
import json


def save_frame(frame_bgr, out_path: Path) -> None:
    """Save a single frame image to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame_bgr)


def save_metadata(records: List[Dict], metadata_path: Path) -> None:
    """Save extracted-frame metadata as JSON."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
