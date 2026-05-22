from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np


# 68-point landmark groups (OpenCV Facemark-LBF / iBUG convention)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH = list(range(48, 68))


def _pick_image(base: Path, preferred: str, fallback: str) -> Path:
    p = base / preferred
    if p.exists():
        return p
    f = base / fallback
    if f.exists():
        return f
    raise FileNotFoundError(f"Neither {preferred} nor {fallback} exists in {base}")


def _pts_from_idxs(landmarks, idxs, w: int, h: int):
    pts = []
    for i in idxs:
        if i >= len(landmarks):
            continue
        lm = landmarks[i]
        # Facemark-LBF: lm is (x,y) pixel coordinate; MediaPipe-style has .x/.y normalized.
        if hasattr(lm, "x") and hasattr(lm, "y"):
            x = int(round(float(lm.x) * (w - 1)))
            y = int(round(float(lm.y) * (h - 1)))
        else:
            x = int(round(float(lm[0])))
            y = int(round(float(lm[1])))
        pts.append((x, y))
    return pts


def _ensure_lbf_model(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "lbfmodel.yaml"
    if model_path.exists():
        return model_path
    url = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
    urllib.request.urlretrieve(url, str(model_path))
    return model_path


def _create_face_detector(cache_dir: Path):
    """Return (detector_type, detector_obj). Prefer YuNet then Haar fallback."""
    # YuNet detector (if available)
    try:
        if hasattr(cv2, "FaceDetectorYN"):
            yn_path = cache_dir / "yunet.onnx"
            if not yn_path.exists():
                yn_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(yn_url, str(yn_path))
            det = cv2.FaceDetectorYN.create(str(yn_path), "", (320, 320), 0.65, 0.3, 5000)
            return "yunet", det
    except Exception:
        pass

    # Haar fallback
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("No face detector available (YuNet and Haar both unavailable)")
    return "haar", cascade


def _detect_face_bbox(detector_type: str, detector, bgr: np.ndarray):
    h, w = bgr.shape[:2]
    if detector_type == "yunet":
        detector.setInputSize((w, h))
        _, faces = detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        # faces: [x,y,w,h,score,...]
        f = max(faces, key=lambda x: float(x[4]))
        x, y, fw, fh = map(float, f[:4])
        return (int(x), int(y), int(fw), int(fh))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    return (int(x), int(y), int(fw), int(fh))


def _draw_points(img: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], r: int = 2):
    for x, y in pts:
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)


def _draw_polyline(img: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], closed=True, thickness=2):
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], closed, color, thickness, cv2.LINE_AA)


def _process_one(detector_type, detector, facemark, img_path: Path, out_dir: Path) -> dict:
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    h, w = bgr.shape[:2]

    bbox = _detect_face_bbox(detector_type, detector, bgr)

    stem = img_path.stem
    overlay = bgr.copy()
    eyes_only = np.zeros_like(bgr)
    nose_only = np.zeros_like(bgr)
    mouth_only = np.zeros_like(bgr)

    out = {
        "image": img_path.name,
        "detected": False,
        "num_faces": 0,
        "features": {}
    }

    if bbox is not None:
        x, y, fw, fh = bbox
        face_rect = np.array([[x, y, fw, fh]], dtype=np.int32)
        ok, landmarks = facemark.fit(bgr, face_rect)
        if ok and landmarks is not None and len(landmarks) > 0:
            lm = landmarks[0][0]  # (68,2)
            out["detected"] = True
            out["num_faces"] = 1

            left_eye_pts = _pts_from_idxs(lm, LEFT_EYE, w, h)
            right_eye_pts = _pts_from_idxs(lm, RIGHT_EYE, w, h)
            nose_pts = _pts_from_idxs(lm, NOSE, w, h)
            mouth_pts = _pts_from_idxs(lm, MOUTH, w, h)

            _draw_points(overlay, left_eye_pts, (0, 255, 255), r=2)
            _draw_points(overlay, right_eye_pts, (0, 255, 255), r=2)
            _draw_points(overlay, nose_pts, (255, 255, 0), r=2)
            _draw_points(overlay, mouth_pts, (255, 0, 255), r=2)

            _draw_polyline(overlay, left_eye_pts, (0, 255, 255), closed=True)
            _draw_polyline(overlay, right_eye_pts, (0, 255, 255), closed=True)
            _draw_polyline(overlay, mouth_pts, (255, 0, 255), closed=True)

            _draw_points(eyes_only, left_eye_pts, (0, 255, 255), r=2)
            _draw_points(eyes_only, right_eye_pts, (0, 255, 255), r=2)
            _draw_polyline(eyes_only, left_eye_pts, (0, 255, 255), closed=True)
            _draw_polyline(eyes_only, right_eye_pts, (0, 255, 255), closed=True)

            _draw_points(nose_only, nose_pts, (255, 255, 0), r=2)

            _draw_points(mouth_only, mouth_pts, (255, 0, 255), r=2)
            _draw_polyline(mouth_only, mouth_pts, (255, 0, 255), closed=True)

            out["features"] = {
                "left_eye_count": len(left_eye_pts),
                "right_eye_count": len(right_eye_pts),
                "nose_count": len(nose_pts),
                "mouth_count": len(mouth_pts),
            }

            cv2.rectangle(overlay, (x, y), (x + fw, y + fh), (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "Face detected, landmark fit failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 140, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(overlay, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / f"{stem}_face_features_overlay.jpg"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_eyes.jpg"), eyes_only)
    cv2.imwrite(str(out_dir / f"{stem}_nose.jpg"), nose_only)
    cv2.imwrite(str(out_dir / f"{stem}_mouth.jpg"), mouth_only)

    return out


def main():
    ap = argparse.ArgumentParser(description="Extract face features (eyes/nose/mouth) for frame front/back and SMPL front/back")
    ap.add_argument("--requested-images-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    base = Path(args.requested_images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_front = base / "frame_front.jpg"
    frame_back = base / "frame_back.jpg"
    smpl_front = _pick_image(base, "smpl_front_splatted.jpg", "smpl_front.jpg")
    smpl_back = _pick_image(base, "smpl_back_splatted.jpg", "smpl_back.jpg")

    cache_dir = base / "_face_detector_cache"
    lbf_model = _ensure_lbf_model(cache_dir)
    detector_type, detector = _create_face_detector(cache_dir)

    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(lbf_model))

    results = []
    for p in [frame_front, frame_back, smpl_front, smpl_back]:
        results.append(_process_one(detector_type, detector, facemark, p, out_dir))
        print(f"processed: {p.name}")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved outputs -> {out_dir}")


if __name__ == "__main__":
    main()
