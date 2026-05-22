from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import xatlas
from PIL import Image
import mediapipe as mp

from src.pose.extract_mediapipe import extract_landmarks_dir, load_landmarks_json
from src.recon.smpl_fitter import SMPL
from workflows.texture_baking.bake_texture import build_cameras
from workflows.texture_baking.bake_smpl_texture_raycast import _rasterise_depth, _project_points


PART_FACE = 0
PART_LARM = 1
PART_RARM = 2
PART_TORSO = 3
PART_LLEG = 4
PART_RLEG = 5
PART_OTHER = 6


LIP_PART_TO_PID = {
    "face": PART_FACE,
    "left_arm": PART_LARM,
    "right_arm": PART_RARM,
    "left_leg": PART_LLEG,
    "right_leg": PART_RLEG,
}


class HumanParsingPartDetector:
    def __init__(self, model_id: str = "matei-dorian/segformer-b0-finetuned-human-parsing"):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            # Some community checkpoints ship only config+weights.
            self.processor = SegformerImageProcessor(
                do_resize=True,
                size={"height": 512, "width": 512},
                do_normalize=True,
            )
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", {})
        self.label_map = {int(k): str(v).lower().replace("-", "_").replace(" ", "_") for k, v in id2label.items()}

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits
            up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
            seg = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        return seg

    def part_masks(self, bgr: np.ndarray, person_mask: np.ndarray) -> dict[int, np.ndarray]:
        seg = self.infer(bgr)
        out = {
            PART_FACE: np.zeros(seg.shape, dtype=np.uint8),
            PART_LARM: np.zeros(seg.shape, dtype=np.uint8),
            PART_RARM: np.zeros(seg.shape, dtype=np.uint8),
            PART_TORSO: np.zeros(seg.shape, dtype=np.uint8),
            PART_LLEG: np.zeros(seg.shape, dtype=np.uint8),
            PART_RLEG: np.zeros(seg.shape, dtype=np.uint8),
        }

        for lid, name in self.label_map.items():
            m = (seg == lid)
            if name in LIP_PART_TO_PID:
                pid = LIP_PART_TO_PID[name]
                out[pid][m] = 255
            elif name in {"upper_clothes", "dress", "coat", "scarf", "jumpsuits", "jumpsuit", "pants", "skirt", "socks", "left_shoe", "right_shoe"}:
                out[PART_TORSO][m] = 255
            elif name in {"hair", "hat", "sunglasses", "glove"}:
                # keep as non-body accessory unless face/arm etc already set
                pass

        # Ensure torso includes remaining person pixels not assigned elsewhere.
        assigned = np.zeros(seg.shape, dtype=np.uint8)
        for pid in [PART_FACE, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]:
            assigned = cv2.bitwise_or(assigned, out[pid])
        rem = (person_mask > 0) & (assigned == 0)
        out[PART_TORSO][rem] = 255

        for pid in out:
            out[pid] = cv2.bitwise_and(out[pid], person_mask)
        return out


_MP_FACE_DET = None
_CV_FACE_CASCADE = None


def _get_face_detector():
    global _MP_FACE_DET
    if _MP_FACE_DET is None:
        try:
            _MP_FACE_DET = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        except Exception:
            _MP_FACE_DET = None
    return _MP_FACE_DET


def _get_cv_face_cascade():
    global _CV_FACE_CASCADE
    if _CV_FACE_CASCADE is None:
        try:
            p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            c = cv2.CascadeClassifier(p)
            _CV_FACE_CASCADE = c if not c.empty() else None
        except Exception:
            _CV_FACE_CASCADE = None
    return _CV_FACE_CASCADE


def _face_frontness(image_bgr: np.ndarray, lms: np.ndarray | None, force_front: bool = False) -> tuple[float, bool]:
    if force_front:
        return 1.0, True

    det = _get_face_detector()
    if det is not None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = det.process(rgb)
        if res.detections:
            d = max(res.detections, key=lambda x: float(x.score[0]) if x.score else 0.0)
            score = float(d.score[0]) if d.score else 0.0
            kps = d.location_data.relative_keypoints
            # right_eye(0), left_eye(1), nose_tip(2)
            if len(kps) >= 3:
                re = np.array([kps[0].x, kps[0].y], dtype=np.float32)
                le = np.array([kps[1].x, kps[1].y], dtype=np.float32)
                no = np.array([kps[2].x, kps[2].y], dtype=np.float32)
                eye_mid_x = float((re[0] + le[0]) * 0.5)
                eye_span = float(abs(le[0] - re[0]) + 1e-6)
                symmetry = 1.0 - min(1.0, abs(no[0] - eye_mid_x) / eye_span)
            else:
                symmetry = 0.5
            fr = float(np.clip(0.7 * score + 0.3 * symmetry, 0.0, 1.0))
            return fr, True

    # OpenCV fallback face detector
    cascade = _get_cv_face_cascade()
    if cascade is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
        if len(faces) > 0:
            area = max((w * h for (_, _, w, h) in faces))
            im_area = float(image_bgr.shape[0] * image_bgr.shape[1])
            score = min(1.0, area / max(1.0, 0.12 * im_area))
            fr = float(np.clip(0.65 * score + 0.35 * _frontness(lms), 0.0, 1.0))
            return fr, True

    # Fallback to landmark-only weak estimate.
    return (_frontness(lms) * 0.5, False)


def _load_obj_verts_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    v = []
    f = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            v.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            f.append(tri)
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32)


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
    return vn


def _smpl_vertex_parts(smpl_model: Path) -> np.ndarray:
    smpl = SMPL(smpl_model, n_betas=10)
    w = smpl.weights.detach().cpu().numpy()  # (6890,24)
    j = np.argmax(w, axis=1)
    part = np.full((w.shape[0],), PART_OTHER, dtype=np.int32)

    face = {12, 15}
    larm = {13, 16, 18, 20, 22}
    rarm = {14, 17, 19, 21, 23}
    torso = {0, 3, 6, 9}
    lleg = {1, 4, 7, 10}
    rleg = {2, 5, 8, 11}

    for idx, ji in enumerate(j):
        if ji in face:
            part[idx] = PART_FACE
        elif ji in larm:
            part[idx] = PART_LARM
        elif ji in rarm:
            part[idx] = PART_RARM
        elif ji in torso:
            part[idx] = PART_TORSO
        elif ji in lleg:
            part[idx] = PART_LLEG
        elif ji in rleg:
            part[idx] = PART_RLEG
    return part


def _valid_xy(lms: np.ndarray | None, idx: int) -> tuple[float, float] | None:
    if lms is None or idx >= lms.shape[0]:
        return None
    x, y = float(lms[idx, 0]), float(lms[idx, 1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def _draw_limb(mask: np.ndarray, pts: list[tuple[float, float] | None], thick: int) -> None:
    for a, b in zip(pts[:-1], pts[1:]):
        if a is None or b is None:
            continue
        p0 = (int(round(a[0])), int(round(a[1])))
        p1 = (int(round(b[0])), int(round(b[1])))
        cv2.line(mask, p0, p1, 255, thickness=thick, lineType=cv2.LINE_AA)
        cv2.circle(mask, p0, thick // 2, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, p1, thick // 2, 255, -1, cv2.LINE_AA)


def _build_part_masks(lms: np.ndarray | None, person_mask: np.ndarray, h: int, w: int) -> dict[int, np.ndarray]:
    out = {
        PART_FACE: np.zeros((h, w), dtype=np.uint8),
        PART_LARM: np.zeros((h, w), dtype=np.uint8),
        PART_RARM: np.zeros((h, w), dtype=np.uint8),
        PART_TORSO: np.zeros((h, w), dtype=np.uint8),
        PART_LLEG: np.zeros((h, w), dtype=np.uint8),
        PART_RLEG: np.zeros((h, w), dtype=np.uint8),
    }

    if lms is None:
        for k in out:
            out[k] = person_mask.copy()
        return out

    sh_l = _valid_xy(lms, 11)
    sh_r = _valid_xy(lms, 12)
    hip_l = _valid_xy(lms, 23)
    hip_r = _valid_xy(lms, 24)

    if sh_l and sh_r and hip_r and hip_l:
        torso_poly = np.array([
            [sh_l[0], sh_l[1]],
            [sh_r[0], sh_r[1]],
            [hip_r[0], hip_r[1]],
            [hip_l[0], hip_l[1]],
        ], dtype=np.int32)
        cv2.fillConvexPoly(out[PART_TORSO], torso_poly, 255)

    if sh_l and hip_l:
        torso_len = float(np.hypot(sh_l[0] - hip_l[0], sh_l[1] - hip_l[1]))
    elif sh_r and hip_r:
        torso_len = float(np.hypot(sh_r[0] - hip_r[0], sh_r[1] - hip_r[1]))
    else:
        torso_len = float(max(h, w) * 0.2)
    thick = max(6, int(0.18 * torso_len))

    _draw_limb(out[PART_LARM], [_valid_xy(lms, 11), _valid_xy(lms, 13), _valid_xy(lms, 15)], thick)
    _draw_limb(out[PART_RARM], [_valid_xy(lms, 12), _valid_xy(lms, 14), _valid_xy(lms, 16)], thick)
    _draw_limb(out[PART_LLEG], [_valid_xy(lms, 23), _valid_xy(lms, 25), _valid_xy(lms, 27)], thick)
    _draw_limb(out[PART_RLEG], [_valid_xy(lms, 24), _valid_xy(lms, 26), _valid_xy(lms, 28)], thick)

    face_pts = [
        _valid_xy(lms, 0), _valid_xy(lms, 1), _valid_xy(lms, 2), _valid_xy(lms, 3), _valid_xy(lms, 4),
        _valid_xy(lms, 7), _valid_xy(lms, 8), _valid_xy(lms, 9), _valid_xy(lms, 10),
    ]
    face_pts = [p for p in face_pts if p is not None]
    if len(face_pts) >= 3:
        hull = cv2.convexHull(np.array(face_pts, dtype=np.float32)).astype(np.int32)
        cv2.fillConvexPoly(out[PART_FACE], hull, 255)
    elif len(face_pts) > 0:
        cx = int(round(np.mean([p[0] for p in face_pts])))
        cy = int(round(np.mean([p[1] for p in face_pts])))
        cv2.circle(out[PART_FACE], (cx, cy), max(8, thick), 255, -1, cv2.LINE_AA)

    for k in out:
        out[k] = cv2.bitwise_and(out[k], person_mask)
    return out


def _frontness(lms: np.ndarray | None) -> float:
    if lms is None:
        return 0.5
    face_idx = [0, 1, 2, 3, 4, 7, 8, 9, 10]
    vis = []
    for i in face_idx:
        if i < lms.shape[0] and np.isfinite(lms[i, 2]):
            vis.append(float(lms[i, 2]))
    face_vis = float(np.mean(vis)) if vis else 0.0

    nose = _valid_xy(lms, 0)
    ls = _valid_xy(lms, 11)
    rs = _valid_xy(lms, 12)
    if nose and ls and rs:
        mid = 0.5 * (ls[0] + rs[0])
        span = abs(rs[0] - ls[0]) + 1e-3
        center = 1.0 - min(1.0, abs((nose[0] - mid) / span))
    else:
        center = 0.5
    return float(np.clip(0.65 * face_vis + 0.35 * center, 0.0, 1.0))


def _person_mask_from_file(masks_dir: Path | None, frame_name: str, h: int, w: int) -> np.ndarray:
    if masks_dir is None:
        return np.ones((h, w), dtype=np.uint8) * 255
    stem = Path(frame_name).stem
    for ext in (".png", ".jpg", ".jpeg"):
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 127).astype(np.uint8) * 255
    return np.ones((h, w), dtype=np.uint8) * 255


def _save_obj_mtl(out_dir: Path, stem: str, verts_new: np.ndarray, indices: np.ndarray, uvs: np.ndarray) -> None:
    tex_name = f"{stem}_texture.png"
    mtl_path = out_dir / f"{stem}.mtl"
    obj_path = out_dir / f"{stem}.obj"

    mtl_path.write_text(
        "newmtl material0\n"
        "Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\n"
        f"map_Kd {tex_name}\n",
        encoding="utf-8",
    )

    with obj_path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        for v in verts_new:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
        f.write("usemtl material0\n")
        for tri in indices:
            i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")


def _build_bundle_cameras(frames_dir: Path, bundle_camera_dir: Path, focal_length: float) -> list[dict]:
    cams: list[dict] = []
    for sd in sorted(bundle_camera_dir.glob("*")):
        if not sd.is_dir():
            continue
        pkl = sd / "bundle_refined.pkl"
        if not pkl.exists():
            continue
        data = pickle.load(open(pkl, "rb"))
        R = np.asarray(data.get("camera_rotation", np.eye(3)), dtype=np.float32).reshape(3, 3)
        t = np.asarray(data.get("camera_translation", np.zeros(3)), dtype=np.float32).reshape(3)
        stem = sd.name

        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = frames_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        tmp = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if tmp is None:
            continue
        h, w = tmp.shape[:2]
        K = np.array([[focal_length, 0, w / 2.0], [0, focal_length, h / 2.0], [0, 0, 1]], dtype=np.float32)
        C = -R.T @ t
        cams.append(dict(R=R, t=t, C=C, K=K, dist=np.zeros(4), w=w, h=h, path=img_path, name=img_path.name))
    return cams


def main() -> None:
    ap = argparse.ArgumentParser(description="Part-aware UV raycast texture baking with front/back camera labeling")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--smpl-model", default=r"E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--landmarks-json", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tex-size", type=int, default=2048)
    ap.add_argument("--depth-tol", type=float, default=0.06)
    ap.add_argument("--deformed-dir", default=None,
                    help="Optional per-frame deformed mesh root: <dir>/<frame_stem>/bundle_refined.obj")
    ap.add_argument("--bundle-camera-dir", default=None,
                    help="Optional per-frame camera PKL root: <dir>/<frame_stem>/bundle_refined.pkl")
    ap.add_argument("--focal-length", type=float, default=0.0,
                    help="Focal length for bundle-camera mode. If <=0 and run_info exists, inferred from run_info.json")
    ap.add_argument("--part-detector", choices=["human_parsing", "mediapipe"], default="human_parsing")
    ap.add_argument("--human-parsing-model", default="matei-dorian/segformer-b0-finetuned-human-parsing")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vnormals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    vmapping, indices, uvs = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    vmapping = vmapping.astype(np.int64)
    indices = indices.astype(np.int32)
    uvs = uvs.astype(np.float32)

    verts_new = verts[vmapping]
    nrm_new = vnormals[vmapping]
    nrm_new /= np.maximum(np.linalg.norm(nrm_new, axis=1, keepdims=True), 1e-8)

    H = W = int(args.tex_size)
    tri_map = np.full((H, W), -1, dtype=np.int32)
    bary_map = np.zeros((H, W, 3), dtype=np.float32)
    uv_px = uvs * np.array([W, H], dtype=np.float32)
    for fi, face in enumerate(indices):
        a, b, c = uv_px[face[0]], uv_px[face[1]], uv_px[face[2]]
        x0 = max(0, int(min(a[0], b[0], c[0])))
        x1 = min(W - 1, int(max(a[0], b[0], c[0])) + 1)
        y0 = max(0, int(min(a[1], b[1], c[1])))
        y1 = min(H - 1, int(max(a[1], b[1], c[1])) + 1)
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1) + 0.5
        ys = np.arange(y0, y1 + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        p = np.stack([gx, gy], axis=-1)
        ab = b - a
        ac = c - a
        ap = p - a
        denom = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(denom) < 1e-8:
            continue
        v = (ap[..., 0] * ac[1] - ap[..., 1] * ac[0]) / denom
        w = (ab[0] * ap[..., 1] - ab[1] * ap[..., 0]) / denom
        u = 1.0 - v - w
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        iy, ix = np.where(inside)
        if iy.size == 0:
            continue
        gy = iy + y0
        gx = ix + x0
        empty = tri_map[gy, gx] == -1
        gy = gy[empty]
        gx = gx[empty]
        tri_map[gy, gx] = fi
        bary_map[gy, gx, 0] = u[iy[empty], ix[empty]]
        bary_map[gy, gx, 1] = v[iy[empty], ix[empty]]
        bary_map[gy, gx, 2] = w[iy[empty], ix[empty]]

    valid_yx = np.argwhere(tri_map >= 0)
    fi_arr = tri_map[valid_yx[:, 0], valid_yx[:, 1]]
    bary_arr = bary_map[valid_yx[:, 0], valid_yx[:, 1]]
    f_verts = indices[fi_arr]

    pos3d = (
        bary_arr[:, 0:1] * verts_new[f_verts[:, 0]]
        + bary_arr[:, 1:2] * verts_new[f_verts[:, 1]]
        + bary_arr[:, 2:3] * verts_new[f_verts[:, 2]]
    )
    nrm3d = (
        bary_arr[:, 0:1] * nrm_new[f_verts[:, 0]]
        + bary_arr[:, 1:2] * nrm_new[f_verts[:, 1]]
        + bary_arr[:, 2:3] * nrm_new[f_verts[:, 2]]
    )
    nrm3d /= np.maximum(np.linalg.norm(nrm3d, axis=1, keepdims=True), 1e-8)

    part_v = _smpl_vertex_parts(Path(args.smpl_model))
    if len(part_v) != len(verts):
        raise RuntimeError(f"Mesh vertex count {len(verts)} does not match SMPL topology {len(part_v)}")
    part_new = part_v[vmapping]
    peak = np.argmax(bary_arr, axis=1)
    texel_part = part_new[f_verts[np.arange(len(f_verts)), peak]]

    lm_json = Path(args.landmarks_json) if args.landmarks_json else (out_dir / "landmarks_mediapipe.json")
    if lm_json.exists():
        lm_dict = load_landmarks_json(lm_json)
    else:
        lm_dict = extract_landmarks_dir(args.frames, lm_json, min_visibility=0.2)

    if args.bundle_camera_dir:
        bdir = Path(args.bundle_camera_dir)
        focal = float(args.focal_length)
        if focal <= 0:
            for cand in [bdir.parents[1] / "run_info.json", bdir.parents[0] / "run_info.json", Path(args.output).parent / "run_info.json"]:
                if cand.exists():
                    try:
                        info = json.loads(cand.read_text(encoding="utf-8"))
                        focal = float(info.get("focal_length", 0.0))
                        if focal > 0:
                            break
                    except Exception:
                        pass
        if focal <= 0:
            raise RuntimeError("bundle-camera mode requires --focal-length or run_info.json with focal_length")
        cams = _build_bundle_cameras(Path(args.frames), bdir, focal)
        print(f"Using bundle cameras: {len(cams)} frames, focal={focal:.3f}")
    else:
        cams = build_cameras(args.cameras, args.images, args.frames)
    cams = sorted(cams, key=lambda c: c["name"])
    masks_dir = Path(args.masks) if args.masks else None

    hp_detector: HumanParsingPartDetector | None = None
    if args.part_detector == "human_parsing":
        try:
            hp_detector = HumanParsingPartDetector(args.human_parsing_model)
            print(f"Loaded human parsing model: {args.human_parsing_model}")
        except Exception as e:
            print(f"[WARN] Human parsing unavailable ({e}). Falling back to mediapipe part masks.")
            hp_detector = None

    colour_acc = np.zeros((len(pos3d), 3), dtype=np.float32)
    weight_acc = np.zeros((len(pos3d),), dtype=np.float32)
    label_info: dict[str, dict[str, float | str]] = {}

    deformed_dir = Path(args.deformed_dir) if args.deformed_dir else None
    first_frame_name = cams[0]["name"] if cams else ""

    for ci, cam in enumerate(cams, start=1):
        if not cam["path"].exists():
            continue
        img_bgr = cv2.imread(str(cam["path"]), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        frame_verts_new = verts_new
        frame_nrm_new = nrm_new
        if deformed_dir is not None:
            stem = Path(cam["name"]).stem
            fp = deformed_dir / stem / "bundle_refined.obj"
            if fp.exists():
                fv, ff = _load_obj_verts_faces(fp)
                if len(fv) == len(verts) and len(ff) == len(faces):
                    fn = _compute_vertex_normals(fv, faces)
                    frame_verts_new = fv[vmapping]
                    frame_nrm_new = fn[vmapping]

        lms = lm_dict.get(cam["name"])
        person_mask = _person_mask_from_file(masks_dir, cam["name"], h, w)
        if hp_detector is not None:
            part_masks = hp_detector.part_masks(img_bgr, person_mask)
        else:
            part_masks = _build_part_masks(lms, person_mask, h, w)

        fr, has_face = _face_frontness(img_bgr, lms, force_front=(cam["name"] == first_frame_name))
        if cam["name"] == first_frame_name:
            lbl = "front"
        elif fr > 0.62 and has_face:
            lbl = "front"
        elif fr < 0.35 or not has_face:
            lbl = "back"
        else:
            lbl = "side"
        label_info[cam["name"]] = {"frontness": fr, "label": lbl, "has_face": bool(has_face)}

        pos3d_cam = (
            bary_arr[:, 0:1] * frame_verts_new[f_verts[:, 0]]
            + bary_arr[:, 1:2] * frame_verts_new[f_verts[:, 1]]
            + bary_arr[:, 2:3] * frame_verts_new[f_verts[:, 2]]
        )
        nrm3d_cam = (
            bary_arr[:, 0:1] * frame_nrm_new[f_verts[:, 0]]
            + bary_arr[:, 1:2] * frame_nrm_new[f_verts[:, 1]]
            + bary_arr[:, 2:3] * frame_nrm_new[f_verts[:, 2]]
        )
        nrm3d_cam /= np.maximum(np.linalg.norm(nrm3d_cam, axis=1, keepdims=True), 1e-8)

        depth_buf = _rasterise_depth(
            frame_verts_new.astype(np.float32),
            indices.astype(np.int64),
            cam["K"].astype(np.float32),
            cam["R"].astype(np.float32),
            cam["t"].astype(np.float32),
            int(cam["w"]),
            int(cam["h"]),
        )

        dir_to_cam = cam["C"] - pos3d_cam
        dir_to_cam_n = dir_to_cam / (np.linalg.norm(dir_to_cam, axis=1, keepdims=True) + 1e-8)
        dot = np.einsum("ij,ij->i", nrm3d_cam, dir_to_cam_n)
        facing = dot > 0.03

        pix, depth = _project_points(pos3d_cam.astype(np.float32), cam["K"].astype(np.float32), cam["R"].astype(np.float32), cam["t"].astype(np.float32))
        finite = np.isfinite(pix).all(axis=1) & np.isfinite(depth)
        in_frame = finite & (depth > 0.01) & (pix[:, 0] >= 0) & (pix[:, 0] < w - 1) & (pix[:, 1] >= 0) & (pix[:, 1] < h - 1)
        if not np.any(in_frame):
            continue

        uu = np.clip(np.round(pix[:, 0]).astype(np.int32), 0, w - 1)
        vv = np.clip(np.round(pix[:, 1]).astype(np.int32), 0, h - 1)
        zbuf = depth_buf[vv, uu]
        visible = np.abs(depth - zbuf) < args.depth_tol

        person_ok = person_mask[vv, uu] > 0
        part_w = np.full((len(pos3d),), 0.55, dtype=np.float32)
        for pid in (PART_FACE, PART_LARM, PART_RARM, PART_TORSO, PART_LLEG, PART_RLEG):
            sel = texel_part == pid
            if not np.any(sel):
                continue
            pm = part_masks[pid]
            in_part = pm[vv[sel], uu[sel]] > 0
            if pid == PART_FACE:
                # Face is strict to avoid bleeding from wrong side views.
                part_w[sel] = np.where(in_part, 1.0, 0.0).astype(np.float32)
            else:
                # Limbs/torso are soft-gated for better coverage.
                part_w[sel] = np.where(in_part, 1.0, 0.55).astype(np.float32)
        other = texel_part == PART_OTHER
        if np.any(other):
            part_w[other] = 1.0

        good = in_frame & facing & visible & person_ok & (part_w > 0)
        if not np.any(good):
            continue

        pu = pix[good, 0]
        pv = pix[good, 1]
        x0 = np.clip(np.floor(pu).astype(np.int32), 0, w - 2)
        y0 = np.clip(np.floor(pv).astype(np.int32), 0, h - 2)
        du = (pu - x0).astype(np.float32)
        dv = (pv - y0).astype(np.float32)

        c00 = img_bgr[y0, x0].astype(np.float32)
        c10 = img_bgr[y0 + 1, x0].astype(np.float32)
        c01 = img_bgr[y0, x0 + 1].astype(np.float32)
        c11 = img_bgr[y0 + 1, x0 + 1].astype(np.float32)
        cols = (c00 * (1 - du[:, None]) * (1 - dv[:, None]) + c10 * (1 - du[:, None]) * dv[:, None] + c01 * du[:, None] * (1 - dv[:, None]) + c11 * du[:, None] * dv[:, None])
        cols = cols[:, ::-1]  # RGB

        idx = np.where(good)[0]
        wgt = np.clip(dot[good], 0.0, 1.0) * part_w[good]

        face_sel = texel_part[idx] == PART_FACE
        if np.any(face_sel):
            face_boost = 1.0 if lbl == "front" else (0.5 if lbl == "side" else 0.1)
            wgt[face_sel] *= face_boost * (0.3 + 0.7 * fr)

        colour_acc[idx] += cols * wgt[:, None]
        weight_acc[idx] += wgt

        if ci % 20 == 0:
            print(f"camera {ci}/{len(cams)} good_texels={len(idx)} label={lbl} frontness={fr:.2f}")

    covered = weight_acc > 1e-8
    tex_rgb = np.zeros((len(pos3d), 3), dtype=np.uint8)
    tex_rgb[covered] = np.clip(colour_acc[covered] / weight_acc[covered, None], 0, 255).astype(np.uint8)

    if np.any(~covered):
        vc = np.asarray(mesh.vertex_colors)
        if len(vc) == len(verts):
            vc_rgb = (vc * 255.0).astype(np.uint8)
            tex_rgb[~covered] = (
                bary_arr[~covered, 0:1] * vc_rgb[vmapping[f_verts[~covered, 0]]]
                + bary_arr[~covered, 1:2] * vc_rgb[vmapping[f_verts[~covered, 1]]]
                + bary_arr[~covered, 2:3] * vc_rgb[vmapping[f_verts[~covered, 2]]]
            ).astype(np.uint8)
        else:
            tex_rgb[~covered] = 127

    tex = np.zeros((H, W, 3), dtype=np.uint8)
    tex[valid_yx[:, 0], valid_yx[:, 1]] = tex_rgb
    tex = np.flipud(tex)

    alpha = np.flipud((tri_map >= 0).astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(8):
        dil = cv2.dilate(tex, kernel)
        m = alpha == 0
        tex[m] = dil[m]
        alpha = cv2.dilate(alpha, kernel)

    stem = Path(args.mesh).stem
    Image.fromarray(tex).save(str(out_dir / f"{stem}_texture.png"))
    _save_obj_mtl(out_dir, stem, verts_new, indices, uvs)

    (out_dir / "camera_front_back_labels.json").write_text(json.dumps(label_info, indent=2), encoding="utf-8")
    cov = float(np.mean(covered))
    print(f"Saved part-aware raycast UV texture to {out_dir} (covered={cov:.3f})")


if __name__ == "__main__":
    main()
