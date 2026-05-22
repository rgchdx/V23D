"""
Extract 2D body landmarks per frame using MediaPipe Pose (Tasks API ≥0.10).
Returns 33 landmarks in pixel coords for each frame.

MediaPipe landmark indices (subset used for SMPL joint mapping):
  11 = left shoulder
  12 = right shoulder
  13 = left elbow
  14 = right elbow
  15 = left wrist
  16 = right wrist
  23 = left hip
  24 = right hip
  25 = left knee
  26 = right knee
  27 = left ankle
  28 = right ankle

Model file download (auto if not present):
  pose_landmarker_lite.task  — fast, ~6 MB
  pose_landmarker_full.task  — better accuracy, ~9 MB
  pose_landmarker_heavy.task — best accuracy, ~28 MB
  All available at:
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/
"""

from __future__ import annotations
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------------------------------------------
# Default model download location
# ------------------------------------------------------------------
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_DEFAULT_MODEL_PATH = Path(os.environ.get(
    "MEDIAPIPE_POSE_MODEL",
    "E:/V23D_Data/pose_landmarker_lite.task"
))


def _ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    print(f"Downloading MediaPipe pose model → {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_DEFAULT_MODEL_URL, str(model_path))
    print(f"  Done ({model_path.stat().st_size // 1024} KB)")
    return model_path


# ------------------------------------------------------------------
# MediaPipe landmark index → SMPL joint name (24-joint SMPL topology)
# ------------------------------------------------------------------
MP_TO_SMPL_NAME = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

MP_IDX_TO_SMPL_IDX = {
    11: 16,
    12: 17,
    13: 18,
    14: 19,
    15: 20,
    16: 21,
    23: 1,
    24: 2,
    25: 4,
    26: 5,
    27: 7,
    28: 8,
}

TRACKED_MP_INDICES = sorted(MP_IDX_TO_SMPL_IDX.keys())

# ------------------------------------------------------------------
# Single-image inference
# ------------------------------------------------------------------

def _make_landmarker(model_path: Path):
    """Create a PoseLandmarker in IMAGE mode."""
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    return mp.tasks.vision.PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
        )
    )


def extract_landmarks_single(
    image_bgr: np.ndarray,
    landmarker,
) -> Optional[np.ndarray]:
    """
    Run MediaPipe Tasks PoseLandmarker on one BGR image.

    Returns
    -------
    landmarks : np.ndarray shape (33, 3)  [x_px, y_px, visibility]
                or None if no person detected.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]   # first person
    pts = np.array(
        [[lm.x * w, lm.y * h, lm.visibility]
         for lm in lms],
        dtype=np.float32,
    )
    return pts   # (33, 3)


def extract_landmarks_dir(
    frames_dir:     str | Path,
    output_json:    str | Path,
    min_visibility: float = 0.3,
    model_path:     Optional[str | Path] = None,
) -> dict[str, Optional[np.ndarray]]:
    """
    Extract MediaPipe 2D body landmarks for every JPEG/PNG in frames_dir.

    Saves results to output_json:
      { "frame_00001.jpg": [[x, y, vis], ...33...], ... }

    Returns dict of {filename: ndarray(33,3) or None}.
    """
    frames_dir  = Path(frames_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    mp_model = _ensure_model(
        Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    )

    image_paths = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {frames_dir}")

    results: dict[str, Optional[list]] = {}
    failed = 0

    with _make_landmarker(mp_model) as landmarker:
        for idx, img_path in enumerate(image_paths):
            if idx % 50 == 0:
                print(f"  [{idx}/{len(image_paths)}] {img_path.name}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                failed += 1
                results[img_path.name] = None
                continue
            lms = extract_landmarks_single(bgr, landmarker)
            if lms is None:
                failed += 1
                results[img_path.name] = None
                continue
            # NaN-mask low-confidence points
            lms[lms[:, 2] < min_visibility, :2] = np.nan
            results[img_path.name] = lms.tolist()

    # Serialise — convert nan → null for JSON
    json_safe: dict = {}
    for name, val in results.items():
        if val is None:
            json_safe[name] = None
        else:
            cleaned = []
            for row in val:
                cleaned.append([
                    None if (isinstance(row[0], float) and np.isnan(row[0])) else float(row[0]),
                    None if (isinstance(row[1], float) and np.isnan(row[1])) else float(row[1]),
                    float(row[2]),
                ])
            json_safe[name] = cleaned

    output_json.write_text(json.dumps(json_safe, indent=2))
    print(f"Saved landmarks for {len(results)} frames "
          f"({failed} failed) → {output_json}")

    out_np: dict[str, Optional[np.ndarray]] = {}
    for name, val in results.items():
        if val is None:
            out_np[name] = None
        else:
            out_np[name] = np.array(
                [[r[0] if r[0] is not None else np.nan,
                  r[1] if r[1] is not None else np.nan,
                  r[2]]
                 for r in val],
                dtype=np.float32,
            )
    return out_np


def load_landmarks_json(json_path: str | Path) -> dict[str, Optional[np.ndarray]]:
    """Load previously saved landmarks JSON → dict of ndarray(33,3)."""
    data = json.loads(Path(json_path).read_text())
    out: dict[str, Optional[np.ndarray]] = {}
    for name, val in data.items():
        if val is None:
            out[name] = None
        else:
            arr = np.array(
                [[r[0] if r[0] is not None else np.nan,
                  r[1] if r[1] is not None else np.nan,
                  r[2]]
                 for r in val],
                dtype=np.float32,
            )
            out[name] = arr
    return out


def extract_joint_observations(
    lms: Optional[np.ndarray],
    joint_pairs: list[tuple[int, int]],
    min_visibility: float = 0.4,
) -> dict[str, np.ndarray]:
    """Return only actually detected joint correspondences for a frame.

    Output fields:
      - `xy`:      (K, 2) detected MediaPipe joint coordinates
      - `conf`:    (K,) visibility confidences
      - `mp_idx`:  (K,) MediaPipe indices used
      - `smpl_idx`:(K,) corresponding SMPL joint indices
    """
    if lms is None:
        return {
            "xy": np.zeros((0, 2), dtype=np.float32),
            "conf": np.zeros((0,), dtype=np.float32),
            "mp_idx": np.zeros((0,), dtype=np.int64),
            "smpl_idx": np.zeros((0,), dtype=np.int64),
        }

    pts: list[list[float]] = []
    conf: list[float] = []
    mp_ids: list[int] = []
    smpl_ids: list[int] = []
    for mp_idx, smpl_idx in joint_pairs:
        if mp_idx >= lms.shape[0]:
            continue
        x, y, v = float(lms[mp_idx, 0]), float(lms[mp_idx, 1]), float(lms[mp_idx, 2])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(v)):
            continue
        if v < min_visibility:
            continue
        pts.append([x, y])
        conf.append(v)
        mp_ids.append(mp_idx)
        smpl_ids.append(smpl_idx)

    return {
        "xy": np.asarray(pts, dtype=np.float32),
        "conf": np.asarray(conf, dtype=np.float32),
        "mp_idx": np.asarray(mp_ids, dtype=np.int64),
        "smpl_idx": np.asarray(smpl_ids, dtype=np.int64),
    }


def draw_landmarks(image_bgr: np.ndarray, lms: np.ndarray) -> np.ndarray:
    """Overlay 2D landmarks on an image for debug visualisation."""
    img = image_bgr.copy()
    h, w = img.shape[:2]
    for i, (x, y, v) in enumerate(lms):
        if np.isnan(x) or np.isnan(y):
            continue
        color = (0, 255, 0) if v > 0.5 else (0, 128, 255)
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
        cv2.putText(img, str(i), (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return img
