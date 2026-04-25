from pathlib import Path
from typing import Any
import json
import logging

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# Load a grayscale image from disk, ensuring it was read successfully
def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


# Heuristic camera intrinsics for unknown input. Assumes a simple pinhole model with focal length proportional to max dims.
def _camera_matrix(width: int, height: int) -> np.ndarray:
    f = 1.2 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotmat_to_qvec(r: np.ndarray) -> list[float]:
    # trace is the sum of the diagonal elements of the rotation matrix.
    trace = np.trace(r)
    # If trace is positive, we can compute the quaternion directly. Otherwise, we need to find the largest diagonal element to avoid 
    # numerical instability.
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q.tolist()


# Read a binary mask image, ensuring it matches the expected shape. If the mask is missing or invalid, return None.
# The mask is returned as a uint8 array where 255 is foreground and 0 is background, resized to match the input image if necessary.
def _read_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8) * 255


def run_self_sfm(
    frames_dir: Path,
    output_dir: Path,
    masks_dir: Path | None = None,
    step: int = 1,
    min_matches: int = 80,
    min_inliers: int = 20,
) -> None:
    """Lightweight sequential SfM using ORB + essential matrix.

    Intended as a fallback when COLMAP registration is too strict or fails on synthetic/diffusion frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 frames for SfM.")

    image_paths = image_paths[:: max(1, step)]

    # Initially, load the first image to get dimensions and set up the camera intrinsics.
    first = _load_gray(image_paths[0])
    # h, w are the height and width of the first image, which we use to compute a heuristic camera matrix K.
    h, w = first.shape
    # k = camera intinsics matrix. Assumed as ismple pinhole camera
    k = _camera_matrix(w, h)

    # ORB is a fast feature detector and descriptor extractor.
    # ORB detects keypoints and computes binary descriptors. We configure it to detect up to 6000 features per image,
    # which is a reasonable number for SfM on synthetic images. We use a brute-force matcher with Hamming distance for ORB descriptors.
    orb = cv2.ORB_create(nfeatures=6000)
    # BFMathcer is the brute-force matcher for binary descriptors. We set crossCheck=False to allow for more matches.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


    # poses_w2c will store the estimated world-to-camera poses for each image. Init with identity for the first image so as reference.
    poses_w2c: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
    # Stats will store information about each pairwise registration attempt, including the number of keypoints, matches, inliers, 
    # and whether the pair was accepted for pose estimation. 
    stats: list[dict[str, Any]] = []


    # set prev image as first and mask as None initially.
    prev_img = first
    prev_mask = None
    if masks_dir is not None:
        # attempt to read the mask for the first image. If it doesn't exist, prev_mask will be None and ORB will use the whole image.
        prev_mask = _read_mask(masks_dir / f"{image_paths[0].stem}.png", prev_img.shape)
    prev_kp, prev_des = orb.detectAndCompute(prev_img, prev_mask)

    # Loop over the remaining images and attempt to register each one to the previous image using ORB features and essential matrix
    # estimation. For each image, we detect ORB keypoints and descriptors, match them to the previous image's features, filter matches
    # using Lowe's ratio test, and then estimate the essential matrix using RANSAC to find inliers. If we have enough inliers, we recover
    # the relative pose and compose it with the previous pose to get the current image's world-to-camera pose. We also record stats about 
    # the registration process for analysis.
    for i in range(1, len(image_paths)):
        cur_img = _load_gray(image_paths[i])
        cur_mask = None
        if masks_dir is not None:
            cur_mask = _read_mask(masks_dir / f"{image_paths[i].stem}.png", cur_img.shape)

        cur_kp, cur_des = orb.detectAndCompute(cur_img, cur_mask)

        # here inliers are the matches that are consistent with a single essential matrix, which implies a valid relative pose between the 
        # two images. We require at least min_inliers to accept the pose estimation; otherwise, we keep the previous pose as a fallback.
        stat = {
            "pair": [image_paths[i - 1].name, image_paths[i].name],
            "kp_prev": 0 if prev_kp is None else len(prev_kp),
            "kp_cur": 0 if cur_kp is None else len(cur_kp),
            "matches": 0,
            "inliers": 0,
            "accepted": False,
        }

        # If we failed to detect keypoints/descriptors in either image, or if we have too few matches, we skip pose estimation and repeat
        # the previous pose. This can happen if the image is too blurry, has too little texture, or if the mask removed too much of the 
        # image.
        if prev_des is None or cur_des is None or len(prev_des) < 10 or len(cur_des) < 10:
            # append the previous pose again as a fallback, and record the stats for this pair. We then move on to the next image.
            poses_w2c.append(poses_w2c[-1].copy())
            stats.append(stat)
            prev_img, prev_kp, prev_des = cur_img, cur_kp, cur_des
            continue

        # We use knnMatch to find the two nearest neighbors for each descriptor in the previous image. We then apply Lowe's ratio test
        # to filter out ambiguous matches. The ratio threshold of 0.82 is a common choice that allows for more matches while still
        # filtering out many false matches. The resulting "good" matches are those that have a clear best match in the current image.
        knn = bf.knnMatch(prev_des, cur_des, k=2)
        good = []
        # for each pair with knn, we check the distance of the best match (m) and the second-best match (n).
        # If the best match is significantly better than the second-best, good match.
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.82 * n.distance:
                good.append(m)

        stat["matches"] = len(good)
        if len(good) < min_matches:
            # if not enough good matches, skip pose estimation and repeat the prev pose.
            poses_w2c.append(poses_w2c[-1].copy())
            stats.append(stat)
            prev_img, prev_kp, prev_des = cur_img, cur_kp, cur_des
            continue

        # If enough good matches, proceed to estimate the essential matrix.
        # Extract the matched keypoint coordinates from the previous and current images. These will be used to estimate the essential
        # matrix, which encodes the relative rotation and translation between the two camera poses. 
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([cur_kp[m.trainIdx].pt for m in good])

        # Use RANSAC which does a robust estimation of the essential matrix by iteratively selecting random subsets of matches and 
        # computing the essential matrix. The method returns the best essential matrix found and a mask indicating which matches are
        # inliers to that model. Set a threshold of 2.0 pixels for inlier classification, which is reasonable for typical image
        # resolutions and expected noise levels. We also set a high confidence level of 0.999 to allow for more iterations if needed.
        e, inlier_mask = cv2.findEssentialMat(pts1, pts2, k, method=cv2.RANSAC, prob=0.999, threshold=2.0)
        if e is None or inlier_mask is None:
            poses_w2c.append(poses_w2c[-1].copy())
            stats.append(stat)
            prev_img, prev_kp, prev_des = cur_img, cur_kp, cur_des
            continue

        inlier_count = int(inlier_mask.ravel().sum())
        stat["inliers"] = inlier_count
        # same again if not enough inliers, skip pose estimation and repeat prev pose.
        if inlier_count < min_inliers:
            poses_w2c.append(poses_w2c[-1].copy())
            stats.append(stat)
            prev_img, prev_kp, prev_des = cur_img, cur_kp, cur_des
            continue

        _, r, t, _ = cv2.recoverPose(e, pts1, pts2, k, mask=inlier_mask)
        t = t / (np.linalg.norm(t) + 1e-12)

        t_rel = np.eye(4, dtype=np.float64)
        t_rel[:3, :3] = r
        t_rel[:3, 3] = t.reshape(3)

        current_pose = t_rel @ poses_w2c[-1]
        poses_w2c.append(current_pose)
        stat["accepted"] = True
        stats.append(stat)

        prev_img, prev_kp, prev_des = cur_img, cur_kp, cur_des

    # After processing all images, we have a list of world-to-camera poses for each image, as well as stats about the registration process.
    # Convert the poses to a format similar to COLMAP's output, including the quaternion and translation vector for each image.
    # We write the poses, stats, and camera intrincsics to JSON files in the output directory for later use in reconstruction.
    pose_records: list[dict[str, Any]] = []
    for idx, path in enumerate(image_paths):
        w2c = poses_w2c[idx]
        c2w = np.linalg.inv(w2c)
        qvec = _rotmat_to_qvec(w2c[:3, :3])
        tvec = w2c[:3, 3].tolist()
        pose_records.append(
            {
                "image_id": idx + 1,
                "camera_id": 1,
                "image_name": path.name,
                "qvec": qvec,
                "tvec": tvec,
                "world_to_camera": w2c.tolist(),
                "camera_to_world": c2w.tolist(),
            }
        )

    (output_dir / "poses.json").write_text(json.dumps(pose_records, indent=2), encoding="utf-8")
    (output_dir / "self_sfm_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (output_dir / "camera_intrinsics.json").write_text(
        json.dumps({"K": k.tolist(), "width": w, "height": h}, indent=2),
        encoding="utf-8",
    )

    ok_pairs = sum(1 for s in stats if s["accepted"])
    logging.info("Self-SfM complete. Accepted %s/%s frame pairs.", ok_pairs, len(stats))
