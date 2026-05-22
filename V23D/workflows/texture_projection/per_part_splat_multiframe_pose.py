from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recon.smpl_fitter import SMPL
from workflows.debug_visualization.export_front_back_part_images import (
    PART_FACE,
    PART_LARM,
    PART_RARM,
    PART_TORSO,
    PART_LLEG,
    PART_RLEG,
    PART_NAMES,
    YoloPosePartDetector,
)

PART_IDS = [PART_FACE, PART_TORSO, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]


class OptionalMediaPipeFeatures:
    def __init__(self) -> None:
        self.enabled = False
        try:
            import mediapipe as mp  # noqa: F401
            self.mp = mp
            self.face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
            )
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.35,
            )
            self.enabled = True
        except Exception:
            self.enabled = False

    def detect_masks(self, img_bgr: np.ndarray) -> dict[str, np.ndarray]:
        h, w = img_bgr.shape[:2]
        out = {
            "eyes_lips": np.zeros((h, w), dtype=np.uint8),
            "hands": np.zeros((h, w), dtype=np.uint8),
        }
        if not self.enabled:
            return out

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Face mesh (eyes + lips regions)
        face_res = self.face.process(rgb)
        if face_res.multi_face_landmarks:
            lms = face_res.multi_face_landmarks[0].landmark

            def _pt(i: int):
                x = int(np.clip(lms[i].x * w, 0, w - 1))
                y = int(np.clip(lms[i].y * h, 0, h - 1))
                return [x, y]

            # robust landmark rings
            left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 246]
            right_eye_idx = [362, 263, 387, 386, 385, 384, 398, 466]
            lips_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

            for idxs in [left_eye_idx, right_eye_idx, lips_idx]:
                pts = np.array([_pt(i) for i in idxs], dtype=np.int32)
                if len(pts) >= 3:
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(out["eyes_lips"], hull, 255)

        # Hands (all 21 landmarks per hand)
        hand_res = self.hands.process(rgb)
        if hand_res.multi_hand_landmarks:
            for hland in hand_res.multi_hand_landmarks:
                pts = []
                for lm in hland.landmark:
                    x = int(np.clip(lm.x * w, 0, w - 1))
                    y = int(np.clip(lm.y * h, 0, h - 1))
                    pts.append([x, y])
                pts = np.array(pts, dtype=np.int32)
                if len(pts) >= 3:
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(out["hands"], hull, 255)

        return out


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(tri)
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64)


def _save_colored_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colors_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(len(verts)):
            v = verts[i]
            c = colors_u8[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    return n.astype(np.float32)


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    vn = np.zeros_like(verts, dtype=np.float32)
    for k in range(3):
        np.add.at(vn, faces[:, k], face_normals)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
    return vn


def _project_points(pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = (R @ pts.T).T + t
    z = cam[:, 2]
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    u = f * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = f * cam[:, 1] / np.maximum(z, 1e-6) + cy
    return np.stack([u, v], axis=1).astype(np.float32), z.astype(np.float32)


def _rasterise_depth(verts: np.ndarray, faces: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, W: int, H: int) -> np.ndarray:
    cam = (R @ verts.T).T + t
    z = cam[:, 2]
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    u = f * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = f * cam[:, 1] / np.maximum(z, 1e-6) + cy

    depth = np.full((H, W), np.inf, dtype=np.float32)
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    z0, z1, z2 = z[i0], z[i1], z[i2]
    x0, x1, x2 = u[i0], u[i1], u[i2]
    y0, y1, y2 = v[i0], v[i1], v[i2]
    front = (z0 > 1e-3) & (z1 > 1e-3) & (z2 > 1e-3)

    for fi in np.where(front)[0]:
        _x0, _y0, _z0 = float(x0[fi]), float(y0[fi]), float(z0[fi])
        _x1, _y1, _z1 = float(x1[fi]), float(y1[fi]), float(z1[fi])
        _x2, _y2, _z2 = float(x2[fi]), float(y2[fi]), float(z2[fi])

        xmin = max(0, int(min(_x0, _x1, _x2)))
        xmax = min(W - 1, int(max(_x0, _x1, _x2)) + 1)
        ymin = max(0, int(min(_y0, _y1, _y2)))
        ymax = min(H - 1, int(max(_y0, _y1, _y2)) + 1)
        if xmin > xmax or ymin > ymax:
            continue

        area = (_x1 - _x0) * (_y2 - _y0) - (_x2 - _x0) * (_y1 - _y0)
        if abs(area) < 1e-8:
            continue
        inv = 1.0 / area

        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1, dtype=np.float32), np.arange(ymin, ymax + 1, dtype=np.float32))
        w0 = ((gx - _x1) * (_y2 - _y1) - (gy - _y1) * (_x2 - _x1)) * inv
        w1 = ((gx - _x2) * (_y0 - _y2) - (gy - _y2) * (_x0 - _x2)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z_interp = (w0 * _z0 + w1 * _z1 + w2 * _z2).astype(np.float32)
        rows = gy.astype(np.int32)[inside]
        cols = gx.astype(np.int32)[inside]
        vals = z_interp[inside]

        flat = rows * W + cols
        order = np.argsort(flat)
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]
        flat = flat[order]

        _, first = np.unique(flat, return_index=True)
        for j in first:
            rr, cc, zz = int(rows[j]), int(cols[j]), float(vals[j])
            if zz < depth[rr, cc]:
                depth[rr, cc] = zz

    return depth


def _load_mask(masks_dir: Path | None, frame_name: str, w: int, h: int) -> np.ndarray | None:
    if masks_dir is None:
        return None
    stem = Path(frame_name).stem
    for cand in [stem, stem.replace("frame_", "")]:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = masks_dir / f"{cand}{ext}"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    return (m > 127).astype(np.uint8) * 255
    return None


def _bundle_cameras(bundle_refined_dir: Path, frames_dir: Path, focal: float, frame_names: list[str] | None):
    frame_dirs = []
    if frame_names:
        for n in frame_names:
            d = bundle_refined_dir / Path(n).stem
            if d.exists():
                frame_dirs.append(d)
    else:
        frame_dirs = sorted([d for d in bundle_refined_dir.iterdir() if d.is_dir()])

    cams = []
    for d in frame_dirs:
        name = f"{d.name}.jpg"
        img_path = frames_dir / name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        with (d / "bundle_refined.pkl").open("rb") as f:
            p = pickle.load(f)
        r = np.asarray(p["camera_rotation"], dtype=np.float32)
        t = np.asarray(p["camera_translation"], dtype=np.float32)
        k = np.array([[focal, 0, w * 0.5], [0, focal, h * 0.5], [0, 0, 1]], dtype=np.float32)
        c = -(r.T @ t)
        cams.append({
            "name": name,
            "dir": d,
            "img_path": img_path,
            "K": k,
            "R": r,
            "t": t,
            "C": c,
            "W": w,
            "H": h,
        })
    return cams


def _bilinear_rgb(img_rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    u0 = np.clip(u.astype(np.int32), 0, w - 2)
    v0 = np.clip(v.astype(np.int32), 0, h - 2)
    du = (u - u0).clip(0, 1)
    dv = (v - v0).clip(0, 1)
    c00 = img_rgb[v0, u0]
    c10 = img_rgb[v0 + 1, u0]
    c01 = img_rgb[v0, u0 + 1]
    c11 = img_rgb[v0 + 1, u0 + 1]
    return (
        (1 - dv[:, None]) * (1 - du[:, None]) * c00
        + dv[:, None] * (1 - du[:, None]) * c10
        + (1 - dv[:, None]) * du[:, None] * c01
        + dv[:, None] * du[:, None] * c11
    ).astype(np.float32)


def _nearest_fill(verts: np.ndarray, colors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = colors.copy()
    if np.all(valid):
        return out
    known = np.where(valid)[0]
    unknown = np.where(~valid)[0]
    if len(known) == 0:
        out[:] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return out
    kxyz = verts[known]
    for ui in unknown:
        d2 = np.sum((kxyz - verts[ui]) ** 2, axis=1)
        out[ui] = out[known[int(np.argmin(d2))]]
    return out


def _smpl_forward(smpl: SMPL, betas: np.ndarray, body_pose: np.ndarray, global_orient: np.ndarray) -> np.ndarray:
    betas_t = torch.from_numpy(betas.astype(np.float32)).reshape(1, -1)
    pose_t = torch.from_numpy(np.concatenate([global_orient.flatten(), body_pose.flatten()]).astype(np.float32)).reshape(1, 72)
    trans_t = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        verts, _ = smpl(betas_t, pose_t, trans_t)
    return verts.squeeze(0).cpu().numpy().astype(np.float32)


def _frame_to_refined_dir(bundle_refined_dir: Path, frame_name: str) -> Path | None:
    stem = Path(frame_name).stem
    exact = bundle_refined_dir / stem
    if exact.exists():
        return exact
    try:
        n = int(stem.replace("frame_", ""))
    except ValueError:
        return None
    cands: list[tuple[int, Path]] = []
    for d in bundle_refined_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            dn = int(d.name.replace("frame_", ""))
            cands.append((abs(dn - n), d))
        except ValueError:
            continue
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts, faces = _load_obj(Path(args.mesh))
    if len(verts) != 6890:
        print(f"[warn] mesh has {len(verts)} verts (expected 6890 for SMPL posing). Posed outputs may be skipped.")

    summary = json.loads(Path(args.bundle_summary).read_text(encoding="utf-8"))
    focal = float(summary["focal_length"])

    frame_names = args.frame_names if args.frame_names else None
    cams = _bundle_cameras(Path(args.bundle_refined_dir), Path(args.frames_dir), focal, frame_names)
    if not cams:
        raise RuntimeError("No cameras loaded")

    det = YoloPosePartDetector(model_name=args.yolo_model)
    mp_feat = OptionalMediaPipeFeatures()
    print(f"mediapipe_features={mp_feat.enabled}")

    face_n = _compute_face_normals(verts, faces)
    vert_n = _compute_vertex_normals(verts, faces, face_n)

    n = len(verts)
    part_idx = {pid: i for i, pid in enumerate(PART_IDS)}
    pcount = len(PART_IDS)
    csum = np.zeros((pcount, n, 3), dtype=np.float32)
    wsum = np.zeros((pcount, n), dtype=np.float32)

    for ci, cam in enumerate(cams, start=1):
        bgr = cv2.imread(str(cam["img_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        w, h = cam["W"], cam["H"]
        person_mask = _load_mask(Path(args.masks_dir) if args.masks_dir else None, cam["name"], w, h)
        if person_mask is None:
            person_mask = np.ones((h, w), dtype=np.uint8) * 255

        part_masks = det.part_masks(bgr, person_mask)
        feat_masks = mp_feat.detect_masks(bgr)

        depth = _rasterise_depth(verts, faces, cam["K"], cam["R"], cam["t"], w, h)
        uv, z = _project_points(verts, cam["K"], cam["R"], cam["t"])
        u, v = uv[:, 0], uv[:, 1]

        inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (z > 1e-3)
        if not inside.any():
            continue

        ui = np.clip(u[inside].astype(np.int32), 0, w - 1)
        vi = np.clip(v[inside].astype(np.int32), 0, h - 1)
        vis = np.abs(z[inside] - depth[vi, ui]) < args.depth_tol

        view = cam["C"][None, :] - verts[inside]
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
        ndotv = (vert_n[inside] * view).sum(axis=1).clip(0)

        base_w = ndotv * vis.astype(np.float32)
        base_w *= (person_mask[vi, ui] > 127).astype(np.float32)

        if not np.any(base_w > 0):
            continue

        idx_global = np.where(inside)[0]
        colors = _bilinear_rgb(rgb, u[inside], v[inside])

        for pid in PART_IDS:
            pm = part_masks[pid]
            in_part = (pm[vi, ui] > 127)
            if not np.any(in_part):
                continue

            pw = base_w.copy()
            pw[~in_part] = 0.0

            # Feature-aware boosts
            if pid == PART_FACE:
                eye_lip = feat_masks["eyes_lips"][vi, ui] > 127
                pw[eye_lip] *= args.face_feature_boost
            if pid in (PART_LARM, PART_RARM):
                hand = feat_masks["hands"][vi, ui] > 127
                pw[hand] *= args.hand_feature_boost
            if pid in (PART_LLEG, PART_RLEG):
                # boost lower body contact (knee/shoe region): bottom image prior
                lower_prior = np.clip((vi.astype(np.float32) / max(h - 1, 1) - 0.45) / 0.55, 0, 1)
                pw *= (1.0 + args.leg_lower_boost * lower_prior)

            good = pw > 0
            if not np.any(good):
                continue

            pi = part_idx[pid]
            gidx = idx_global[good]
            csum[pi, gidx] += colors[good] * pw[good, None]
            wsum[pi, gidx] += pw[good]

        if (ci % 2) == 0 or ci == len(cams):
            print(f"camera {ci}/{len(cams)} processed")

    # Combine per-part colors with per-vertex winner-take-most
    final = np.zeros((n, 3), dtype=np.float32)
    total_w = wsum.sum(axis=0)
    observed = total_w > 0
    winner = np.argmax(wsum, axis=0)

    for vi in np.where(observed)[0]:
        wi = winner[vi]
        if wsum[wi, vi] > 1e-8:
            final[vi] = csum[wi, vi] / wsum[wi, vi]
        else:
            # fallback weighted blend over parts
            ww = wsum[:, vi]
            cc = np.zeros((3,), dtype=np.float32)
            if ww.sum() > 0:
                for pi in range(pcount):
                    if ww[pi] > 0:
                        cc += (csum[pi, vi] / max(wsum[pi, vi], 1e-8)) * ww[pi]
                final[vi] = cc / ww.sum()

    final = _nearest_fill(verts, final, observed)
    final_u8 = np.clip(final * 255.0, 0, 255).astype(np.uint8)

    out_can = out_dir / "smpl_per_part_splatted_canonical.ply"
    _save_colored_ply(out_can, verts, faces, final_u8)

    # Also save per-part confidence maps as vertex colors for debugging.
    part_debug = {}
    for pid in PART_IDS:
        pi = part_idx[pid]
        conf = np.zeros((n, 3), dtype=np.float32)
        mx = float(np.max(wsum[pi])) if np.max(wsum[pi]) > 0 else 1.0
        c = (wsum[pi] / mx).clip(0, 1)
        conf[:, 0] = c
        conf[:, 1] = c
        conf[:, 2] = c
        dbg_path = out_dir / f"debug_conf_{PART_NAMES[pid]}.ply"
        _save_colored_ply(dbg_path, verts, faces, (conf * 255).astype(np.uint8))
        part_debug[PART_NAMES[pid]] = float(np.mean(wsum[pi] > 0))

    # Posed exports (rotating SMPL with bundle pose params)
    posed_saved = []
    if args.smpl_model and len(verts) == 6890:
        smpl = SMPL(Path(args.smpl_model), n_betas=10)
        pose_frames = args.pose_frames if args.pose_frames else [c["name"] for c in cams]
        for fn in pose_frames:
            d = _frame_to_refined_dir(Path(args.bundle_refined_dir), fn)
            if d is None:
                continue
            pkl = d / "bundle_refined.pkl"
            if not pkl.exists():
                continue
            params = pickle.load(open(pkl, "rb"))
            betas = np.asarray(params["betas"], dtype=np.float32).reshape(-1)
            body_pose = np.asarray(params["body_pose"], dtype=np.float32).reshape(-1)
            global_orient = np.asarray(params["global_orient"], dtype=np.float32).reshape(-1)
            v_pose = _smpl_forward(smpl, betas, body_pose, global_orient)
            if len(v_pose) != len(verts):
                continue
            p_out = out_dir / f"posed_per_part_splatted_{Path(fn).stem}.ply"
            _save_colored_ply(p_out, v_pose, faces, final_u8)
            posed_saved.append(str(p_out))

    stats = {
        "num_vertices": int(n),
        "num_faces": int(len(faces)),
        "num_cameras": int(len(cams)),
        "observed_vertex_ratio": float(np.mean(observed)),
        "part_observed_ratio": part_debug,
        "mediapipe_enabled": bool(mp_feat.enabled),
        "posed_meshes": posed_saved,
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Saved canonical: {out_can}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Per-body-part multi-frame splatting to SMPL with feature-aware weighting and optional posed exports")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--bundle-refined-dir", required=True)
    ap.add_argument("--bundle-summary", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frame-names", nargs="*", default=[])
    ap.add_argument("--yolo-model", default="yolov8x-pose.pt")
    ap.add_argument("--depth-tol", type=float, default=0.03)
    ap.add_argument("--face-feature-boost", type=float, default=2.5)
    ap.add_argument("--hand-feature-boost", type=float, default=2.0)
    ap.add_argument("--leg-lower-boost", type=float, default=0.7)
    ap.add_argument("--smpl-model", default="", help="SMPL_NEUTRAL.pkl for posed mesh export")
    ap.add_argument("--pose-frames", nargs="*", default=[])
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
