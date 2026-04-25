from pathlib import Path
from typing import Dict, List

import numpy as np


def _qvec_to_rotmat(qvec) -> np.ndarray:
    # Changes quaternion from (qw, qx, qy, qz) to rotation matrix with conversion formula
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=float,
    )



def colmap_pose_to_matrix(qvec, tvec) -> np.ndarray:
    # Convert COLMAP (world-to-camera) quaternion/translation to 4x4 matrix.
    r = _qvec_to_rotmat(qvec)
    t = np.array(tvec, dtype=float).reshape(3)
    m = np.eye(4, dtype=float)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def load_colmap_poses_text(images_txt_path: Path) -> List[Dict]:
    # Load COLMAP poses from images.txt and convert to world-to-camera and camera-to-world matrices.
    lines = images_txt_path.read_text(encoding="utf-8").splitlines()
    records: List[Dict] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        image_id = int(parts[0])
        qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
        camera_id = int(parts[8])
        image_name = parts[9]

        w2c = colmap_pose_to_matrix(qvec, tvec)
        c2w = np.linalg.inv(w2c)

        records.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "image_name": image_name,
                "qvec": qvec,
                "tvec": tvec,
                "world_to_camera": w2c.tolist(),
                "camera_to_world": c2w.tolist(),
            }
        )

        # Next line in images.txt is 2D points; skip it if present.
        if i < len(lines):
            i += 1

    return records
