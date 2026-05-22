"""Export front/back frame images with body-part overlays and SMPL part views.

Part detection strategy (in priority order):
  1. YOLOv8-pose  – keypoint-driven filled polygon regions per body part.
  2. SegFormer human parsing  – pixel-level semantic segmentation fallback.

Face detection: OpenCV YuNet (ONNX) with Haar-cascade fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is importable when run from any cwd.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recon.smpl_fitter import SMPL  # noqa: E402

# ---------------------------------------------------------------------------
# Part IDs
# ---------------------------------------------------------------------------
PART_FACE  = 0
PART_LARM  = 1
PART_RARM  = 2
PART_TORSO = 3
PART_LLEG  = 4
PART_RLEG  = 5
PART_OTHER = 6

PARTS = [PART_FACE, PART_TORSO, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]

PART_NAMES = {
    PART_FACE:  "face",
    PART_TORSO: "torso",
    PART_LARM:  "left_arm",
    PART_RARM:  "right_arm",
    PART_LLEG:  "left_leg",
    PART_RLEG:  "right_leg",
}

PART_COLORS = {
    PART_FACE:  (255, 220,   0),
    PART_TORSO: (  0, 220,   0),
    PART_LARM:  (  0, 200, 255),
    PART_RARM:  (255, 120,   0),
    PART_LLEG:  (180,   0, 255),
    PART_RLEG:  (255,   0, 180),
}

# ---------------------------------------------------------------------------
# Shared mask helpers
# ---------------------------------------------------------------------------

def _blank_parts(h: int, w: int) -> dict:
    return {pid: np.zeros((h, w), dtype=np.uint8) for pid in PARTS}


def _apply_person_mask(masks: dict, pmask: np.ndarray) -> dict:
    for pid in masks:
        masks[pid] = cv2.bitwise_and(masks[pid], pmask)
    return masks


def _mask_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def _fill_unassigned_torso(masks: dict, pmask: np.ndarray) -> None:
    """Any person pixel not covered by a non-torso part goes to torso."""
    if pmask.ndim == 3:
        pmask = pmask[:, :, 0]
    assigned = np.zeros(pmask.shape, dtype=np.uint8)
    for pid in [PART_FACE, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]:
        assigned = cv2.bitwise_or(assigned, masks[pid])
    masks[PART_TORSO][(pmask > 0) & (assigned == 0)] = 255


def _finalize_parts(masks: dict, pmask: np.ndarray, fill_torso_rest: bool = True) -> dict:
    """Enforce non-overlap constraints with limb/face priority over torso."""
    if pmask.ndim == 3:
        pmask = pmask[:, :, 0]

    # Always clip to person mask first.
    masks = _apply_person_mask(masks, pmask)

    # Priority: face + limbs should not be overridden by torso.
    occ = np.zeros_like(pmask)
    for pid in [PART_FACE, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]:
        occ = cv2.bitwise_or(occ, masks[pid])
    masks[PART_TORSO][occ > 0] = 0

    # Optionally fill remaining person pixels as torso.
    if fill_torso_rest:
        _fill_unassigned_torso(masks, pmask)
    return _apply_person_mask(masks, pmask)


def _poly_fill(mask: np.ndarray, pts: list, expand: int = 0) -> None:
    valid = [p for p in pts if p is not None]
    if len(valid) < 2:
        return
    arr = np.array(valid, dtype=np.int32)
    if expand > 0:
        cx = float(arr[:, 0].mean())
        cy = float(arr[:, 1].mean())
        expanded = []
        for px_i, py_i in valid:
            dx, dy = float(px_i) - cx, float(py_i) - cy
            n = max(1.0, (dx * dx + dy * dy) ** 0.5)
            expanded.append((int(px_i + dx / n * expand), int(py_i + dy / n * expand)))
        arr = np.array(expanded, dtype=np.int32)
    hull = cv2.convexHull(arr)
    cv2.fillConvexPoly(mask, hull, 255)


def _thick_limb(mask: np.ndarray, a, b, thick: int) -> None:
    if a is None or b is None:
        return
    cv2.line(mask, a, b, 255, thickness=max(4, thick), lineType=cv2.LINE_AA)
    cv2.circle(mask, a, max(4, thick // 2), 255, -1, cv2.LINE_AA)
    cv2.circle(mask, b, max(4, thick // 2), 255, -1, cv2.LINE_AA)


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = max(3, int(px) * 2 + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, ker, iterations=1)


def _erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = max(3, int(px) * 2 + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, ker, iterations=1)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.max() == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == keep] = 255
    return out


def _clip_mask_to_points(mask: np.ndarray, pts: list[tuple[int, int] | None], pad_x: int, pad_y: int) -> np.ndarray:
    valid = [p for p in pts if p is not None]
    if len(valid) < 2:
        return mask
    xs = [int(p[0]) for p in valid]
    ys = [int(p[1]) for p in valid]
    x0 = max(0, min(xs) - pad_x)
    x1 = min(mask.shape[1] - 1, max(xs) + pad_x)
    y0 = max(0, min(ys) - pad_y)
    y1 = min(mask.shape[0] - 1, max(ys) + pad_y)
    clip = np.zeros_like(mask)
    clip[y0:y1 + 1, x0:x1 + 1] = 255
    return cv2.bitwise_and(mask, clip)


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return (int(round((a[0] + b[0]) * 0.5)), int(round((a[1] + b[1]) * 0.5)))


def _lerp_pt(a, b, t: float):
    if a is None or b is None:
        return None
    return (int(round((1.0 - t) * a[0] + t * b[0])), int(round((1.0 - t) * a[1] + t * b[1])))


def _extend_pt(a, b, t: float):
    """Extrapolate from a→b beyond b by factor t (t>0)."""
    if a is None or b is None:
        return None
    return (int(round(b[0] + t * (b[0] - a[0]))), int(round(b[1] + t * (b[1] - a[1]))))


def _kp(xy: np.ndarray, cf, idx: int, thr: float = 0.20):
    """Return (x, y) int tuple for YOLO keypoint, or None if low-conf / invalid."""
    if idx >= len(xy):
        return None
    x, y = float(xy[idx, 0]), float(xy[idx, 1])
    if not (np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0):
        return None
    if cf is not None and idx < len(cf) and float(cf[idx]) < thr:
        return None
    return int(round(x)), int(round(y))


# ---------------------------------------------------------------------------
# YOLO-pose part detector
# ---------------------------------------------------------------------------
# COCO 17-keypoint layout:
#  0=nose  1=leye  2=reye  3=lear  4=rear
#  5=lsh   6=rsh   7=lel   8=rel   9=lwr  10=rwr
# 11=lhp  12=rhp  13=lkn  14=rkn  15=lfk  16=rfk


class YoloPosePartDetector:
    """Body-part masks built from YOLOv8-pose keypoints using filled polygon regions."""

    def __init__(self, model_name: str = "yolov8x-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_name)

    def part_masks(self, img_bgr: np.ndarray, pmask: np.ndarray) -> dict:
        h, w = img_bgr.shape[:2]
        out = _blank_parts(h, w)

        res = self.model.predict(source=img_bgr, verbose=False, conf=0.2, iou=0.45)
        if not res:
            out[PART_TORSO] = pmask.copy()
            return _apply_person_mask(out, pmask)

        r = res[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            out[PART_TORSO] = pmask.copy()
            return _apply_person_mask(out, pmask)

        kxy = r.keypoints.xy.cpu().numpy()    # (N, 17, 2)
        kcf_t = r.keypoints.conf
        kcf = kcf_t.cpu().numpy() if kcf_t is not None else None  # (N, 17)

        # Pick the largest / most central person.
        cy_img, cx_img = h / 2.0, w / 2.0
        best_i, best_s = 0, -1.0
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
        for i in range(len(kxy)):
            vis = float(np.sum(kcf[i] > 0.2)) if kcf is not None else float(np.sum(kxy[i].sum(1) > 0))
            score = vis
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = ((bx - cx_img) ** 2 + (by - cy_img) ** 2) ** 0.5
                score += (x2 - x1) * (y2 - y1) / max(h * w, 1) * 5000 - dist / max(h, w)
            if score > best_s:
                best_s, best_i = score, i

        xy = kxy[best_i]
        cf = kcf[best_i] if kcf is not None else None

        nose = _kp(xy, cf, 0)
        leye = _kp(xy, cf, 1)
        reye = _kp(xy, cf, 2)
        lear = _kp(xy, cf, 3)
        rear = _kp(xy, cf, 4)
        lsh  = _kp(xy, cf, 5)
        rsh  = _kp(xy, cf, 6)
        lel  = _kp(xy, cf, 7)
        rel  = _kp(xy, cf, 8)
        lwr  = _kp(xy, cf, 9)
        rwr  = _kp(xy, cf, 10)
        lhp  = _kp(xy, cf, 11)
        rhp  = _kp(xy, cf, 12)
        lkn  = _kp(xy, cf, 13)
        rkn  = _kp(xy, cf, 14)
        lfk  = _kp(xy, cf, 15)
        rfk  = _kp(xy, cf, 16)

        # Reference thickness from torso height.
        torso_h = 120
        if lsh and lhp:
            torso_h = int(np.hypot(lsh[0] - lhp[0], lsh[1] - lhp[1]))
        elif rsh and rhp:
            torso_h = int(np.hypot(rsh[0] - rhp[0], rsh[1] - rhp[1]))
        # Separate arm/leg thickness so arms stop stealing torso while legs remain stable.
        arm_upper = max(10, int(0.21 * max(30, torso_h)))
        arm_lower = max(8, int(0.17 * max(30, torso_h)))
        leg_upper = max(12, int(0.24 * max(30, torso_h)))
        leg_lower = max(10, int(0.20 * max(30, torso_h)))
        thick_upper = leg_upper
        thick_lower = leg_lower

        # Face: build a full-face region (not only eyes/nose points).
        head_pts = [p for p in [nose, leye, reye, lear, rear] if p is not None]
        face_tmp = np.zeros((h, w), dtype=np.uint8)

        if len(head_pts) >= 2:
            _poly_fill(face_tmp, head_pts, expand=max(12, thick_upper // 2))

        # Scale full-face ellipse by eyes/shoulders/torso proportions.
        shoulder_w = 0.0
        if lsh is not None and rsh is not None:
            shoulder_w = float(np.hypot(lsh[0] - rsh[0], lsh[1] - rsh[1]))
        eye_w = 0.0
        if leye is not None and reye is not None:
            eye_w = float(np.hypot(leye[0] - reye[0], leye[1] - reye[1]))

        face_w = max(28.0, eye_w * 2.6, shoulder_w * 0.40, float(thick_upper) * 2.2)
        face_h = face_w * 1.28

        if nose is not None:
            cx = int(nose[0])
            cy = int(nose[1] + 0.08 * face_h)
        elif len(head_pts) > 0:
            cx = int(np.mean([p[0] for p in head_pts]))
            cy = int(np.mean([p[1] for p in head_pts]))
        else:
            cx, cy = int(w * 0.5), int(h * 0.20)

        cv2.ellipse(
            face_tmp,
            (cx, cy),
            (max(12, int(face_w * 0.52)), max(14, int(face_h * 0.50))),
            0,
            0,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )

        # Optional helper from selected person bbox: keep top body region as potential face area.
        if boxes is not None and best_i < len(boxes):
            x1, y1, x2, y2 = boxes[best_i]
            x1i, x2i = int(max(0, x1)), int(min(w - 1, x2))
            y1i = int(max(0, y1))
            top_cut = int(y1 + 0.42 * (y2 - y1))
            if lsh is not None and rsh is not None:
                shoulder_y = int(0.5 * (lsh[1] + rsh[1]))
                top_cut = min(top_cut, shoulder_y - max(2, thick_upper // 3))
            y2i = int(min(h - 1, max(y1i + 1, top_cut)))
            if x2i > x1i and y2i > y1i:
                cv2.rectangle(face_tmp, (x1i, y1i), (x2i, y2i), 255, -1, cv2.LINE_AA)

        # Final face mask clipped by person mask.
        out[PART_FACE] = cv2.bitwise_and(face_tmp, pmask)

        # Left arm (tapered thickness).
        _thick_limb(out[PART_LARM], lsh, lel, arm_upper)
        _thick_limb(out[PART_LARM], lel, lwr, arm_lower)
        lhand = _extend_pt(lel, lwr, 0.45)
        _thick_limb(out[PART_LARM], lwr, lhand, max(4, arm_lower // 2))

        # Right arm (tapered thickness).
        _thick_limb(out[PART_RARM], rsh, rel, arm_upper)
        _thick_limb(out[PART_RARM], rel, rwr, arm_lower)
        rhand = _extend_pt(rel, rwr, 0.45)
        _thick_limb(out[PART_RARM], rwr, rhand, max(4, arm_lower // 2))

        # Keep arms tighter so they do not consume torso/back.
        out[PART_LARM] = _dilate_mask(out[PART_LARM], max(2, arm_lower // 4))
        out[PART_RARM] = _dilate_mask(out[PART_RARM], max(2, arm_lower // 4))
        out[PART_LARM] = _clip_mask_to_points(out[PART_LARM], [lsh, lel, lwr], pad_x=max(10, arm_upper), pad_y=max(10, arm_upper))
        out[PART_RARM] = _clip_mask_to_points(out[PART_RARM], [rsh, rel, rwr], pad_x=max(10, arm_upper), pad_y=max(10, arm_upper))
        out[PART_LARM] = _keep_largest_component(out[PART_LARM])
        out[PART_RARM] = _keep_largest_component(out[PART_RARM])

        # Torso replacement: chest/front or back-band, depending on visible face cues.
        chest_l = _lerp_pt(lsh, lhp, 0.58)
        chest_r = _lerp_pt(rsh, rhp, 0.58)
        back_l  = _lerp_pt(lsh, lhp, 0.40)
        back_r  = _lerp_pt(rsh, rhp, 0.40)
        has_front_face = (nose is not None) and (leye is not None or reye is not None)

        # Robust torso from shoulder/hip quad.
        torso_quad = [lsh, rsh, rhp, lhp]
        if sum(1 for p in torso_quad if p is not None) >= 3:
            _poly_fill(out[PART_TORSO], torso_quad, expand=max(4, thick_upper // 3))

        if has_front_face:
            chest_poly = [lsh, rsh, chest_r, chest_l]
            if sum(1 for p in chest_poly if p is not None) >= 3:
                _poly_fill(out[PART_TORSO], chest_poly, expand=max(3, thick_upper // 3))
        else:
            back_poly = [back_l, back_r, rhp, lhp]
            if sum(1 for p in back_poly if p is not None) >= 3:
                _poly_fill(out[PART_TORSO], back_poly, expand=0)

            # Keep the back torso tightly banded. Do not inflate into the shoulders.

        # Small bridge to avoid disconnected torso for hard poses.
        if has_front_face and lsh is not None and rsh is not None and chest_l is not None and chest_r is not None:
            bridge = np.array([lsh, rsh, chest_r, chest_l], dtype=np.int32)
            cv2.fillConvexPoly(out[PART_TORSO], bridge, 255)

        # Left leg.
        _thick_limb(out[PART_LLEG], lhp, lkn, leg_upper)
        _thick_limb(out[PART_LLEG], lkn, lfk, leg_lower)

        # Right leg.
        _thick_limb(out[PART_RLEG], rhp, rkn, leg_upper)
        _thick_limb(out[PART_RLEG], rkn, rfk, leg_lower)

        out[PART_LLEG] = _dilate_mask(out[PART_LLEG], max(2, leg_lower // 4))
        out[PART_RLEG] = _dilate_mask(out[PART_RLEG], max(2, leg_lower // 4))
        out[PART_LLEG] = _clip_mask_to_points(out[PART_LLEG], [lhp, lkn, lfk], pad_x=max(10, leg_upper), pad_y=max(12, leg_upper))
        out[PART_RLEG] = _clip_mask_to_points(out[PART_RLEG], [rhp, rkn, rfk], pad_x=max(10, leg_upper), pad_y=max(12, leg_upper))
        out[PART_LLEG] = _keep_largest_component(out[PART_LLEG])
        out[PART_RLEG] = _keep_largest_component(out[PART_RLEG])
        out[PART_LLEG] = _erode_mask(out[PART_LLEG], 1)
        out[PART_RLEG] = _erode_mask(out[PART_RLEG], 1)

        # Torso should not eat arms/legs: only keep explicit torso geometry.
        # Reduce torso dilation to prevent arm/leg overlap.
        out[PART_TORSO] = _dilate_mask(out[PART_TORSO], max(0, thick_upper // 16))

        return _finalize_parts(out, pmask, fill_torso_rest=False)


# ---------------------------------------------------------------------------
# SegFormer human-parsing fallback
# ---------------------------------------------------------------------------
_CLOTHES_TO_TORSO = {
    "upper_clothes", "dress", "coat", "scarf", "jumpsuits",
    "pants", "skirt", "socks", "left_shoe", "right_shoe", "belt", "bag",
}
_SEG_PART_MAP = {
    "face": PART_FACE, "left_arm": PART_LARM, "right_arm": PART_RARM,
    "left_leg": PART_LLEG, "right_leg": PART_RLEG,
}


class HumanParsingPartDetector:
    """SegFormer-b2 clothes/body parsing."""

    def __init__(self, model_id: str = "mattmdjaga/segformer_b2_clothes"):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            self.processor = SegformerImageProcessor(
                do_resize=True, size={"height": 512, "width": 512}, do_normalize=True,
            )
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.eval()
        raw = getattr(self.model.config, "id2label", {})
        self.label_map = {int(k): str(v).lower().replace("-", "_").replace(" ", "_")
                          for k, v in raw.items()}

    def part_masks(self, img_bgr: np.ndarray, pmask: np.ndarray) -> dict:
        import torch
        h, w = img_bgr.shape[:2]
        out = _blank_parts(h, w)

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            up = torch.nn.functional.interpolate(logits, size=(h, w),
                                                 mode="bilinear", align_corners=False)
            seg = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

        for lid, name in self.label_map.items():
            m = seg == lid
            if name in _SEG_PART_MAP:
                out[_SEG_PART_MAP[name]][m] = 255
            elif name in _CLOTHES_TO_TORSO:
                out[PART_TORSO][m] = 255

        return _finalize_parts(out, pmask)


class HybridPartDetector:
    """Blend YOLO-pose geometry with SegFormer parsing for more stable body parts."""

    def __init__(self, yolo_model: str = "yolov8x-pose.pt", parsing_model: str = "mattmdjaga/segformer_b2_clothes"):
        self.yolo = YoloPosePartDetector(model_name=yolo_model)
        self.parse = HumanParsingPartDetector(model_id=parsing_model)

    def part_masks(self, img_bgr: np.ndarray, pmask: np.ndarray) -> dict:
        y = self.yolo.part_masks(img_bgr, pmask)
        p = self.parse.part_masks(img_bgr, pmask)

        out = _blank_parts(img_bgr.shape[0], img_bgr.shape[1])

        # Arms/legs: prefer YOLO skeleton geometry; fall back to parsing if too small.
        for pid in [PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]:
            y_a = _mask_area(y[pid])
            p_a = _mask_area(p[pid])
            if y_a >= max(220, int(0.18 * p_a)):
                out[pid] = y[pid]
            else:
                out[pid] = p[pid]

        # Face: union of both, parsing helps full region.
        out[PART_FACE] = cv2.bitwise_or(y[PART_FACE], p[PART_FACE])

        # Torso: prefer parsing torso then subtract selected parts.
        out[PART_TORSO] = p[PART_TORSO].copy()

        return _finalize_parts(out, pmask)


# ---------------------------------------------------------------------------
# Face detector – YuNet ONNX, Haar fallback
# ---------------------------------------------------------------------------

class FaceDetectorYuNet:
    def __init__(self, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._yunet = None
        self._cascade = None

        onnx = cache_dir / "face_detection_yunet_2023mar.onnx"
        if not onnx.exists():
            try:
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
                    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    str(onnx),
                )
            except Exception:
                pass

        if onnx.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                self._yunet = cv2.FaceDetectorYN_create(str(onnx), "", (320, 320), 0.6, 0.3, 5000)
            except Exception:
                pass

        try:
            c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._cascade = c if not c.empty() else None
        except Exception:
            pass

    def score(self, img_bgr: np.ndarray) -> float:
        h, w = img_bgr.shape[:2]
        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            _, det = self._yunet.detect(img_bgr)
            if det is not None and len(det) > 0:
                best = 0.0
                for row in det:
                    bw, bh_d = float(row[2]), float(row[3])
                    conf = float(row[-1])
                    area_n = max(0.0, bw * bh_d) / max(1.0, float(h * w))
                    s = conf * min(1.0, area_n / 0.10)
                    if s > best:
                        best = s
                return best

        if self._cascade is not None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.08, 4, minSize=(24, 24))
            if len(faces) > 0:
                area = max(fw * fh for _, _, fw, fh in faces)
                return float(min(1.0, area / max(1.0, 0.10 * h * w)))

        return 0.0

    def best_bbox(self, img_bgr: np.ndarray):
        """Return best face bbox as (x1,y1,x2,y2,score) or None."""
        h, w = img_bgr.shape[:2]
        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            _, det = self._yunet.detect(img_bgr)
            if det is not None and len(det) > 0:
                best = None
                best_s = -1.0
                for row in det:
                    x, y, bw, bh_d = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    conf = float(row[-1])
                    area_n = max(0.0, bw * bh_d) / max(1.0, float(h * w))
                    s = conf * min(1.0, area_n / 0.10)
                    if s > best_s:
                        best_s = s
                        best = (x, y, x + bw, y + bh_d, s)
                if best is not None:
                    return best

        if self._cascade is not None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.08, 4, minSize=(24, 24))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda t: float(t[2] * t[3]))
                s = float(min(1.0, (fw * fh) / max(1.0, 0.10 * h * w)))
                return (float(x), float(y), float(x + fw), float(y + fh), s)
        return None

    def best_detection(self, img_bgr: np.ndarray):
        """Return dict with bbox/score and 5 keypoints when available, else None."""
        h, w = img_bgr.shape[:2]
        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            _, det = self._yunet.detect(img_bgr)
            if det is not None and len(det) > 0:
                best_row = None
                best_s = -1.0
                for row in det:
                    x, y, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    conf = float(row[-1])
                    area_n = max(0.0, bw * bh) / max(1.0, float(h * w))
                    s = conf * min(1.0, area_n / 0.10)
                    if s > best_s:
                        best_s = s
                        best_row = row
                if best_row is not None:
                    x, y, bw, bh = map(float, best_row[:4])
                    kps = np.array(best_row[4:14], dtype=np.float32).reshape(5, 2)
                    return {
                        "bbox": (x, y, x + bw, y + bh),
                        "score": float(best_s),
                        "kps5": kps,
                    }

        bb = self.best_bbox(img_bgr)
        if bb is None:
            return None
        x1, y1, x2, y2, s = bb
        # Approximate fallback keypoints from bbox geometry.
        wbb = max(1.0, x2 - x1)
        hbb = max(1.0, y2 - y1)
        kps = np.array([
            [x1 + 0.35 * wbb, y1 + 0.40 * hbb],  # left eye
            [x1 + 0.65 * wbb, y1 + 0.40 * hbb],  # right eye
            [x1 + 0.50 * wbb, y1 + 0.56 * hbb],  # nose tip
            [x1 + 0.40 * wbb, y1 + 0.75 * hbb],  # left mouth
            [x1 + 0.60 * wbb, y1 + 0.75 * hbb],  # right mouth
        ], dtype=np.float32)
        return {"bbox": (x1, y1, x2, y2), "score": float(s), "kps5": kps}


def _refine_face_with_bbox(parts: dict, pmask: np.ndarray, face_bbox, grow: float = 1.35) -> None:
    if face_bbox is None:
        return
    if pmask.ndim == 3:
        pmask = pmask[:, :, 0]
    h, w = pmask.shape[:2]
    x1, y1, x2, y2, _s = face_bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * grow
    bh = (y2 - y1) * grow * 1.15
    ex1 = int(max(0, cx - bw * 0.5))
    ex2 = int(min(w - 1, cx + bw * 0.5))
    ey1 = int(max(0, cy - bh * 0.55))
    ey2 = int(min(h - 1, cy + bh * 0.45))
    if ex2 <= ex1 or ey2 <= ey1:
        return

    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        m,
        (int(cx), int(cy)),
        (max(8, int((ex2 - ex1) * 0.5)), max(10, int((ey2 - ey1) * 0.5))),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    m = cv2.bitwise_and(m, pmask)
    parts[PART_FACE] = cv2.bitwise_or(parts[PART_FACE], m)


# ---------------------------------------------------------------------------
# SMPL utilities
# ---------------------------------------------------------------------------

def _smpl_vertex_parts(smpl_pkl: Path) -> np.ndarray:
    smpl = SMPL(smpl_pkl, n_betas=10)
    w = smpl.weights.detach().cpu().numpy()  # (6890, 24)
    j = np.argmax(w, axis=1)
    part = np.full(w.shape[0], PART_OTHER, dtype=np.int32)
    for idx, ji in enumerate(j):
        if   ji in {12, 15}:              part[idx] = PART_FACE
        elif ji in {13, 16, 18, 20, 22}: part[idx] = PART_LARM
        elif ji in {14, 17, 19, 21, 23}: part[idx] = PART_RARM
        elif ji in {0, 3, 6, 9}:         part[idx] = PART_TORSO
        elif ji in {1, 4, 7, 10}:        part[idx] = PART_LLEG
        elif ji in {2, 5, 8, 11}:        part[idx] = PART_RLEG
    return part


def _load_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
            faces.append(tri)
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


def _person_mask_from_dir(masks_dir: Path, frame_name: str, h: int, w: int) -> np.ndarray:
    stem = Path(frame_name).stem
    for ext in (".png", ".jpg", ".jpeg"):
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.ndim == 3:
                    m = m[:, :, 0]
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 127).astype(np.uint8) * 255
    return np.ones((h, w), dtype=np.uint8) * 255


def _crop_part(img: np.ndarray, mask: np.ndarray, pad: int = 10) -> np.ndarray:
    out = np.zeros_like(img)
    keep = mask > 0
    if not keep.any():
        return out
    out[keep] = img[keep]
    ys, xs = np.where(keep)
    y0 = max(0,               int(ys.min()) - pad)
    y1 = min(img.shape[0] - 1, int(ys.max()) + pad)
    x0 = max(0,               int(xs.min()) - pad)
    x1 = min(img.shape[1] - 1, int(xs.max()) + pad)
    return out[y0:y1 + 1, x0:x1 + 1]


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _build_dense_smpl_mask(part_img: np.ndarray, pid: int | None = None) -> np.ndarray:
    m = (part_img.sum(axis=2) > 0).astype(np.uint8) * 255
    if m.max() == 0:
        return m

    ys, xs = np.where(m > 0)
    if len(xs) >= 3:
        hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 255)

    if pid == PART_TORSO:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        m = cv2.dilate(m, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        m = _erode_mask(m, 3)
    elif pid in (PART_LLEG, PART_RLEG):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        m = cv2.dilate(m, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    elif pid in (PART_LARM, PART_RARM):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        m = cv2.dilate(m, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        m = cv2.dilate(m, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


def _warp_face_to_target(frame_img: np.ndarray, src_kps5: np.ndarray, tgt_bbox, out_shape_hw: tuple[int, int]) -> np.ndarray:
    """Warp full frame face by 5-point similarity transform into target face bbox."""
    h, w = out_shape_hw
    x1, y1, x2, y2 = tgt_bbox
    tw = max(2, x2 - x1 + 1)
    th = max(2, y2 - y1 + 1)

    # Expand the target so the full head/hair region is preserved rather than
    # cropping the face down to a small inner region.
    x_pad = max(4, int(0.05 * tw))
    y_pad_top = max(4, int(0.08 * th))
    y_pad_bot = max(2, int(0.03 * th))
    x1 = max(0, x1 - x_pad)
    x2 = min(w - 1, x2 + x_pad)
    y1 = max(0, y1 - y_pad_top)
    y2 = min(h - 1, y2 + y_pad_bot)
    tw = max(2, x2 - x1 + 1)
    th = max(2, y2 - y1 + 1)
    # Canonical target points within target face bbox.
    tgt = np.array([
        [x1 + 0.35 * tw, y1 + 0.40 * th],
        [x1 + 0.65 * tw, y1 + 0.40 * th],
        [x1 + 0.50 * tw, y1 + 0.56 * th],
        [x1 + 0.40 * tw, y1 + 0.75 * th],
        [x1 + 0.60 * tw, y1 + 0.75 * th],
    ], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_kps5.astype(np.float32), tgt, method=cv2.LMEDS)
    if M is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.warpAffine(frame_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _warp_quad_by_keypoints(
    frame_img: np.ndarray,
    src_quad: list[tuple[float, float] | None],
    tgt_quad: list[tuple[float, float] | None],
    out_shape_hw: tuple[int, int],
    frame_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    src_pts = [p for p in src_quad if p is not None]
    tgt_pts = [p for p in tgt_quad if p is not None]
    if len(src_pts) < 4 or len(tgt_pts) < 4:
        return None
    src_arr = np.asarray(src_pts[:4], dtype=np.float32)
    tgt_arr = np.asarray(tgt_pts[:4], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_arr, tgt_arr)
    h, w = out_shape_hw
    warped_img = cv2.warpPerspective(frame_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if frame_mask is None:
        frame_mask = np.ones(frame_img.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(frame_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_img, warped_mask


def _extract_coco17_keypoints(part_det, img_bgr: np.ndarray) -> dict[int, tuple[float, float]]:
    """Return best-person COCO17 keypoints as {idx:(x,y)} using YOLO if available."""
    # Resolve YOLO model object from detector wrappers.
    yolo = None
    if hasattr(part_det, "model"):
        yolo = getattr(part_det, "model")
    elif hasattr(part_det, "yolo") and hasattr(part_det.yolo, "model"):
        yolo = part_det.yolo.model
    if yolo is None:
        return {}

    try:
        res = yolo.predict(source=img_bgr, verbose=False, conf=0.2, iou=0.45)
    except Exception:
        return {}
    if not res:
        return {}

    r = res[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return {}

    kxy = r.keypoints.xy.cpu().numpy()  # (N,17,2)
    kcf_t = r.keypoints.conf
    kcf = kcf_t.cpu().numpy() if kcf_t is not None else None
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None

    h, w = img_bgr.shape[:2]
    cy_img, cx_img = h * 0.5, w * 0.5
    best_i, best_s = 0, -1e9
    for i in range(len(kxy)):
        vis = float(np.sum(kcf[i] > 0.2)) if kcf is not None else float(np.sum(kxy[i].sum(1) > 0))
        score = vis
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
            bx, by = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            dist = float(np.hypot(bx - cx_img, by - cy_img))
            score += (x2 - x1) * (y2 - y1) / max(h * w, 1) * 5000.0 - dist / max(h, w)
        if score > best_s:
            best_s, best_i = score, i

    out = {}
    for j in range(min(17, kxy.shape[1])):
        x, y = float(kxy[best_i, j, 0]), float(kxy[best_i, j, 1])
        c = float(kcf[best_i, j]) if kcf is not None else 1.0
        if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0 and c >= 0.20:
            out[j] = (x, y)
    return out


def _part_anchor_indices(pid: int) -> list[int]:
    if pid == PART_LARM:
        return [5, 7, 9]
    if pid == PART_RARM:
        return [6, 8, 10]
    if pid == PART_LLEG:
        return [11, 13, 15]
    if pid == PART_RLEG:
        return [12, 14, 16]
    if pid == PART_TORSO:
        return [5, 6, 11, 12]
    return []


def _smpl_coco17_keypoints(smpl_pkl: Path, view: str, size: int = 1024) -> dict[int, tuple[float, float]]:
    """Project zero-pose SMPL joints into the front/back SMPL-view canvas.

    This gives stable target anchors for arms / legs / torso, instead of guessing
    them from the rendered preview image.
    """
    import torch

    smpl = SMPL(smpl_pkl, n_betas=10)
    betas = torch.zeros(1, 10, dtype=torch.float32)
    pose = torch.zeros(1, 72, dtype=torch.float32)
    trans = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        _verts, joints = smpl(betas, pose, trans)

    j = joints.squeeze(0).cpu().numpy().astype(np.float32)
    if view == "back":
        j = j.copy()
        j[:, 0] *= -1.0

    x, y, z = j[:, 0], j[:, 1], j[:, 2]
    sx = (size - 40) / max(float(x.max() - x.min()), 1e-6)
    sy = (size - 40) / max(float(y.max() - y.min()), 1e-6)
    s = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + size * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + size * 0.5).astype(np.int32)

    # COCO-17 target anchors from SMPL joints.
    mapping = {
        5: 16,  # left shoulder
        6: 17,  # right shoulder
        7: 18,  # left elbow
        8: 19,  # right elbow
        9: 20,  # left wrist
        10: 21, # right wrist
        11: 1,  # left hip
        12: 2,  # right hip
        13: 4,  # left knee
        14: 5,  # right knee
        15: 7,  # left ankle
        16: 8,  # right ankle
        0: 0,   # pelvis / fallback
    }

    out: dict[int, tuple[float, float]] = {}
    for coco_idx, smpl_idx in mapping.items():
        if 0 <= smpl_idx < len(px):
            out[coco_idx] = (float(px[smpl_idx]), float(py[smpl_idx]))
    return out


def _warp_part_by_keypoints(frame_img: np.ndarray, src_kps: dict[int, tuple[float, float]], tgt_kps: dict[int, tuple[float, float]], pid: int, out_shape_hw: tuple[int, int], frame_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    idxs = _part_anchor_indices(pid)
    if not idxs:
        return None
    src_pts, tgt_pts = [], []
    for i in idxs:
        if i in src_kps and i in tgt_kps:
            src_pts.append(src_kps[i])
            tgt_pts.append(tgt_kps[i])
    if len(src_pts) < 2:
        return None

    # Arm-specific quad warp: rotate and expand to preserve full arm potential.
    if pid in (PART_LARM, PART_RARM):
        sh, el, wr = None, None, None
        if pid == PART_LARM:
            sh, el, wr = src_kps.get(5), src_kps.get(7), src_kps.get(9)
            tsh, tel, twr = tgt_kps.get(5), tgt_kps.get(7), tgt_kps.get(9)
        else:
            sh, el, wr = src_kps.get(6), src_kps.get(8), src_kps.get(10)
            tsh, tel, twr = tgt_kps.get(6), tgt_kps.get(8), tgt_kps.get(10)
        if sh is not None and el is not None and wr is not None and tsh is not None and tel is not None and twr is not None:
            def _arm_quad(a, b, c):
                v = np.array([c[0] - a[0], c[1] - a[1]], dtype=np.float32)
                ln = float(np.linalg.norm(v))
                if ln < 1e-3:
                    return None
                u = v / ln
                p = np.array([-u[1], u[0]], dtype=np.float32)
                # Thickness from elbow-to-line distance, with extra room for full arm width.
                mid = np.array([(a[0] + c[0]) * 0.5, (a[1] + c[1]) * 0.5], dtype=np.float32)
                dx = float(b[0] - a[0])
                dy = float(b[1] - a[1])
                d = abs(float(v[0]) * dy - float(v[1]) * dx) / max(ln, 1e-6)
                half = max(14.0, 4.6 * d)
                a = np.array(a, dtype=np.float32)
                c = np.array(c, dtype=np.float32)
                return [tuple(a + p * half), tuple(a - p * half), tuple(c - p * half), tuple(c + p * half)]

            src_quad = _arm_quad(sh, el, wr)
            tgt_quad = _arm_quad(tsh, tel, twr)
            if src_quad is not None and tgt_quad is not None:
                warped = _warp_quad_by_keypoints(frame_img, src_quad, tgt_quad, out_shape_hw, frame_mask=frame_mask)
                if warped is not None:
                    return warped

    src_arr = np.asarray(src_pts, dtype=np.float32)
    tgt_arr = np.asarray(tgt_pts, dtype=np.float32)
    if len(src_pts) >= 3:
        M, _ = cv2.estimateAffinePartial2D(src_arr, tgt_arr, method=cv2.LMEDS)
    else:
        M, _ = cv2.estimateAffine2D(src_arr, tgt_arr, method=cv2.LMEDS)
    if M is None:
        return None
    h, w = out_shape_hw
    warped_img = cv2.warpAffine(frame_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if frame_mask is None:
        frame_mask = np.ones(frame_img.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpAffine(frame_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_img, warped_mask


def _splat_parts_to_smpl(
    frame_img: np.ndarray,
    frame_parts: dict,
    smpl_part_imgs: dict,
    face_det: dict | None = None,
    smpl_face_det: dict | None = None,
    frame_kps17: dict[int, tuple[float, float]] | None = None,
    smpl_kps17: dict[int, tuple[float, float]] | None = None,
    view_tag: str | None = None,
) -> np.ndarray:
    h_s, w_s = next(iter(smpl_part_imgs.values())).shape[:2]
    canvas = np.zeros((h_s, w_s, 3), dtype=np.uint8)

    # Priority order: torso/legs first, arms next, face last so the face is never
    # covered by the torso, and arms can still overwrite torso where needed.
    ordered_parts = [PART_TORSO, PART_LLEG, PART_RLEG, PART_LARM, PART_RARM, PART_FACE]

    for pid in ordered_parts:
        fmask = frame_parts[pid]
        smask = _build_dense_smpl_mask(smpl_part_imgs[pid], pid=pid)
        if pid in (PART_LARM, PART_RARM):
            smask = _dilate_mask(smask, 22)
        elif pid == PART_FACE:
            smask = _dilate_mask(smask, 4)
        fb = _bbox_from_mask(fmask)
        sb = _bbox_from_mask(smask)
        if fb is None or sb is None:
            continue

        fx1, fy1, fx2, fy2 = fb
        sx1, sy1, sx2, sy2 = sb
        if fx2 <= fx1 or fy2 <= fy1 or sx2 <= sx1 or sy2 <= sy1:
            continue

        src_patch = frame_img[fy1:fy2 + 1, fx1:fx2 + 1]
        src_mask = fmask[fy1:fy2 + 1, fx1:fx2 + 1]
        if src_patch.size == 0 or int((src_mask > 0).sum()) < 12:
            continue

        src_only = np.zeros_like(src_patch)
        src_only[src_mask > 0] = src_patch[src_mask > 0]
        full_masked = np.zeros_like(frame_img)
        full_masked[fmask > 0] = frame_img[fmask > 0]

        tw = sx2 - sx1 + 1
        th = sy2 - sy1 + 1
        tgt_mask = smask[sy1:sy2 + 1, sx1:sx2 + 1]

        if pid == PART_FACE and face_det is not None and "kps5" in face_det:
            # Face attachment by eyes/nose/mouth. Always use the frame landmarks
            # and the expanded SMPL face bbox, so hair and full head coverage survive.
            warped = _warp_face_to_target(
                frame_img,
                src_kps5=face_det["kps5"],
                tgt_bbox=(sx1, sy1, sx2, sy2),
                out_shape_hw=(h_s, w_s),
            )
            roi = canvas[sy1:sy2 + 1, sx1:sx2 + 1]
            wr = warped[sy1:sy2 + 1, sx1:sx2 + 1]
            use = tgt_mask > 0
            roi[use] = wr[use]
            canvas[sy1:sy2 + 1, sx1:sx2 + 1] = roi
            continue

        # Non-face parts: prefer keypoint-to-keypoint warp between frame and SMPL-view.
        if frame_kps17 and smpl_kps17:
            warped_part = _warp_part_by_keypoints(
                frame_img=full_masked,
                src_kps=frame_kps17,
                tgt_kps=smpl_kps17,
                pid=pid,
                out_shape_hw=(h_s, w_s),
                frame_mask=fmask,
            )
            if warped_part is not None:
                warped_img, warped_mask = warped_part
                roi = canvas[sy1:sy2 + 1, sx1:sx2 + 1]
                wr = warped_img[sy1:sy2 + 1, sx1:sx2 + 1]
                wm = warped_mask[sy1:sy2 + 1, sx1:sx2 + 1]
                use = (tgt_mask > 0) & (wm > 0) & (wr.sum(axis=2) > 0)
                roi[use] = wr[use]
                canvas[sy1:sy2 + 1, sx1:sx2 + 1] = roi
                continue

        # Torso and legs should fill the SMPL target region rather than preserving source
        # bbox aspect ratio, which collapses the back torso into a narrow strip.
        if pid in (PART_TORSO, PART_LLEG, PART_RLEG):
            patch_r = cv2.resize(src_only, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask_r = cv2.resize(src_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            roi = canvas[sy1:sy2 + 1, sx1:sx2 + 1]
            use = tgt_mask > 0
            if pid == PART_TORSO and view_tag == "back":
                roi[use] = patch_r[use]
            else:
                masked_use = use & (mask_r > 0)
                if not np.any(masked_use):
                    masked_use = use
                roi[masked_use] = patch_r[masked_use]
            canvas[sy1:sy2 + 1, sx1:sx2 + 1] = roi
            continue

        # Non-face parts: preserve aspect ratio to reduce arm/leg stretching artifacts.
        sh, sw = src_patch.shape[:2]
        if sh <= 0 or sw <= 0:
            continue
        scale = min(float(tw) / max(sw, 1), float(th) / max(sh, 1))
        nw = max(1, int(round(sw * scale)))
        nh = max(1, int(round(sh * scale)))
        patch_r = cv2.resize(src_only, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(src_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        xoff = max(0, (tw - nw) // 2)
        yoff = max(0, (th - nh) // 2)

        roi = canvas[sy1:sy2 + 1, sx1:sx2 + 1]
        roi_patch = roi[yoff:yoff + nh, xoff:xoff + nw]
        roi_tgt_m = tgt_mask[yoff:yoff + nh, xoff:xoff + nw]
        use = (mask_r > 0) & (roi_tgt_m > 0)
        if not np.any(use):
            use = roi_tgt_m > 0
        roi_patch[use] = patch_r[use]
        roi[yoff:yoff + nh, xoff:xoff + nw] = roi_patch
        canvas[sy1:sy2 + 1, sx1:sx2 + 1] = roi

    return canvas


def _smpl_part_view(verts: np.ndarray, part_v: np.ndarray, view: str, size: int = 1024):
    v = verts.copy()
    if view == "back":
        v[:, 0] *= -1.0
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    sx = (size - 40) / max(float(x.max() - x.min()), 1e-6)
    sy = (size - 40) / max(float(y.max() - y.min()), 1e-6)
    s = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + size * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + size * 0.5).astype(np.int32)
    order = np.argsort(z) if view == "front" else np.argsort(-z)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    per_part = {pid: np.zeros((size, size, 3), dtype=np.uint8) for pid in PARTS}
    for i in order:
        xi, yi = int(px[i]), int(py[i])
        if not (0 <= xi < size and 0 <= yi < size):
            continue
        pid = int(part_v[i])
        if pid in PART_COLORS:
            c = PART_COLORS[pid]
            cv2.circle(canvas,         (xi, yi), 2, c, -1, cv2.LINE_AA)
            cv2.circle(per_part[pid],  (xi, yi), 2, c, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"SMPL {view}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas, per_part


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-json",       required=True)
    ap.add_argument("--frames-dir",        required=True)
    ap.add_argument("--masks-dir",         required=True)
    ap.add_argument("--smpl-obj",          required=True,  help="Canonical SMPL .obj mesh")
    ap.add_argument("--smpl-model",        default=r"E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--out-dir",           required=True)
    ap.add_argument("--force-front-frame", default="frame_00000.jpg")
    ap.add_argument("--force-back-frame",  default="frame_00165.jpg")
    ap.add_argument("--part-detector",     choices=["hybrid", "yolo_pose", "human_parsing"], default="hybrid")
    ap.add_argument("--yolo-model",        default="yolov8x-pose.pt")
    ap.add_argument("--parsing-model",     default="mattmdjaga/segformer_b2_clothes")
    # Kept for CLI compatibility; not used.
    ap.add_argument("--landmarks-json",    default="")
    ap.add_argument("--smpl-canonical-obj", default="")
    ap.add_argument("--output",            default="")
    args = ap.parse_args()

    labels     = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
    frames_dir = Path(args.frames_dir)
    masks_dir  = Path(args.masks_dir)
    out_dir    = Path(args.out_dir or args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    smpl_obj = Path(args.smpl_obj or args.smpl_canonical_obj)

    # Resolve front / back frames.
    items = sorted(labels.items())

    front_name = args.force_front_frame if (frames_dir / args.force_front_frame).exists() else items[0][0]
    back_name  = args.force_back_frame  if (frames_dir / args.force_back_frame).exists()  else items[-1][0]
    front_meta = labels.get(front_name, {"frontness": 1.0})
    back_meta  = labels.get(back_name,  {"frontness": 0.0})

    print(f"Front frame : {front_name}")
    print(f"Back  frame : {back_name}")

    # Build detectors.
    print(f"Loading part detector : {args.part_detector}")
    if args.part_detector == "hybrid":
        try:
            part_det = HybridPartDetector(yolo_model=args.yolo_model, parsing_model=args.parsing_model)
            print("  Hybrid detector loaded (YOLO + SegFormer).")
        except Exception as exc:
            print(f"  Hybrid unavailable ({exc}), using YOLO-only fallback.")
            try:
                part_det = YoloPosePartDetector(model_name=args.yolo_model)
                print("  YOLOv8-pose loaded.")
            except Exception as exc2:
                print(f"  YOLO unavailable ({exc2}), using SegFormer fallback.")
                part_det = HumanParsingPartDetector(model_id=args.parsing_model)
    elif args.part_detector == "yolo_pose":
        try:
            part_det = YoloPosePartDetector(model_name=args.yolo_model)
            print("  YOLOv8-pose loaded.")
        except Exception as exc:
            print(f"  YOLO unavailable ({exc}), using SegFormer fallback.")
            part_det = HumanParsingPartDetector(model_id=args.parsing_model)
    else:
        part_det = HumanParsingPartDetector(model_id=args.parsing_model)
        print("  SegFormer loaded.")

    face_det = FaceDetectorYuNet(out_dir / "_face_detector_cache")

    # Load frame images.
    front_img = cv2.imread(str(frames_dir / front_name), cv2.IMREAD_COLOR)
    back_img  = cv2.imread(str(frames_dir / back_name),  cv2.IMREAD_COLOR)
    if front_img is None or back_img is None:
        raise RuntimeError("Could not load one or both frame images.")

    cv2.imwrite(str(out_dir / "frame_front.jpg"), front_img)
    cv2.imwrite(str(out_dir / "frame_back.jpg"),  back_img)

    frame_parts_by_tag = {}
    frame_face_det_by_tag = {}

    for name, img, tag in [(front_name, front_img, "front"), (back_name, back_img, "back")]:
        h, w = img.shape[:2]
        pmask  = _person_mask_from_dir(masks_dir, name, h, w)
        parts  = part_det.part_masks(img, pmask)
        face_det_info = face_det.best_detection(img)
        face_box = None
        if face_det_info is not None:
            x1, y1, x2, y2 = face_det_info["bbox"]
            face_box = (x1, y1, x2, y2, float(face_det_info.get("score", 0.0)))
        _refine_face_with_bbox(parts, pmask, face_box)
        parts = _finalize_parts(parts, pmask, fill_torso_rest=False)
        frame_parts_by_tag[tag] = parts
        frame_face_det_by_tag[tag] = face_det_info
        fscore = float(face_box[4]) if face_box is not None else face_det.score(img)

        overlay = img.copy()
        for pid in PARTS:
            c    = PART_COLORS[pid]
            keep = parts[pid] > 0
            if keep.any():
                overlay[keep] = (
                    0.50 * overlay[keep] + 0.50 * np.array(c, dtype=np.float32)
                ).astype(np.uint8)
            crop = _crop_part(img, parts[pid])
            cv2.imwrite(str(out_dir / f"frame_{tag}_{PART_NAMES[pid]}.jpg"), crop)

        nz = {PART_NAMES[pid]: int((parts[pid] > 0).sum()) for pid in PARTS}
        print(f"  [{tag}] face_score={fscore:.3f}  pixels={nz}")

        cv2.putText(overlay, f"{tag}: {name}  face={fscore:.2f}",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frame_{tag}_parts_overlay.jpg"), overlay)

    # SMPL canonical part views.
    verts, _ = _load_obj(smpl_obj)
    part_v   = _smpl_vertex_parts(Path(args.smpl_model))
    if len(part_v) != len(verts):
        raise RuntimeError(f"SMPL vertex count mismatch: mesh={len(verts)}, weights={len(part_v)}")

    for view in ("front", "back"):
        canvas, per_part = _smpl_part_view(verts, part_v, view=view)
        cv2.imwrite(str(out_dir / f"smpl_{view}.jpg"), canvas)
        for pid in PARTS:
            part_img = per_part[pid]
            if pid in (PART_LARM, PART_RARM):
                # Gray background for arms so the hand/skin pixels are visible.
                gray_bg = np.full_like(part_img, 128)
                arm_mask = (part_img.sum(axis=2) > 0)
                gray_bg[arm_mask] = part_img[arm_mask]
                part_img = gray_bg
            cv2.imwrite(str(out_dir / f"smpl_{view}_{PART_NAMES[pid]}.jpg"), part_img)

        # Splat frame part textures onto this SMPL view.
        src_frame = front_img if view == "front" else back_img
        src_parts = frame_parts_by_tag.get(view)
        if src_parts is not None:
            frame_kps17 = _extract_coco17_keypoints(part_det, src_frame)
            smpl_kps17 = _smpl_coco17_keypoints(Path(args.smpl_model), view=view, size=canvas.shape[0])
            smpl_face_det = face_det.best_detection(canvas)
            splat = _splat_parts_to_smpl(
                src_frame,
                src_parts,
                per_part,
                face_det=frame_face_det_by_tag.get(view),
                smpl_face_det=smpl_face_det,
                frame_kps17=frame_kps17,
                smpl_kps17=smpl_kps17,
                view_tag=view,
            )
            cv2.imwrite(str(out_dir / f"smpl_{view}_splatted.jpg"), splat)
            # Also export per-part splatted-on-smpl images.
            for pid in PARTS:
                part_canvas = np.zeros_like(splat)
                m = _build_dense_smpl_mask(per_part[pid], pid=pid) > 0
                part_canvas[m] = splat[m]
                cv2.imwrite(str(out_dir / f"smpl_{view}_splatted_{PART_NAMES[pid]}.jpg"), part_canvas)

    summary = {
        "front_frame": {"name": front_name, "frontness": float(front_meta.get("frontness", 1.0))},
        "back_frame":  {"name": back_name,  "frontness": float(back_meta.get("frontness", 0.0))},
        "output_dir":  str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved requested images -> {out_dir}")


if __name__ == "__main__":
    main()
