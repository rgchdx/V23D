from pathlib import Path
from typing import Iterator, Tuple

import cv2


def open_video(video_path: Path) -> cv2.VideoCapture:
    """Open a video and return the capture handle."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def iter_frames(cap: cv2.VideoCapture) -> Iterator[Tuple[int, float, any]]:
    """Yield (frame_index, timestamp_sec, frame_bgr) from an opened capture."""
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_sec = frame_index / fps if fps > 0 else 0.0
        yield frame_index, timestamp_sec, frame
        frame_index += 1
