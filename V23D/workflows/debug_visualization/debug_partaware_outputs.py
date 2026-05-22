from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from workflows.texture_baking.bake_texture_part_aware_raycast import (
    PART_FACE,
    PART_LARM,
    PART_RARM,
    PART_TORSO,
    PART_LLEG,
    PART_RLEG,
    _build_part_masks,
    _person_mask_from_file,
    _smpl_vertex_parts,
)
from workflows.texture_baking.bake_smpl_texture_raycast import _project_points


PART_NAMES = {
    PART_FACE: "face",
    PART_LARM: "left_arm",
    PART_RARM: "right_arm",
    PART_TORSO: "torso",
    PART_LLEG: "left_leg",
    PART_RLEG: "right_leg",
}

PART_COLORS = {
    PART_FACE: (255, 220, 0),
    PART_LARM: (0, 200, 255),
    PART_RARM: (255, 120, 0),
    PART_TORSO: (0, 255, 0),
    PART_LLEG: (180, 0, 255),
    PART_RLEG: (255, 0, 180),
}

LABEL_COLORS = {
    "front": (0, 220, 0),
    "side": (0, 180, 255),
    "back": (255, 80, 80),
}


def _load_obj_verts(path: Path) -> np.ndarray:
    verts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(list(map(float, line.split()[1:4])))
    return np.asarray(verts, dtype=np.float32)


def _sample_names(all_names: list[str], n: int) -> list[str]:
    if n <= 0 or len(all_names) <= n:
        return all_names
    idx = np.linspace(0, len(all_names) - 1, n).round().astype(int)
    return [all_names[i] for i in idx]


def _draw_part_legend(img: np.ndarray) -> None:
    y = 24
    for pid in [PART_FACE, PART_TORSO, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]:
        c = PART_COLORS[pid]
        cv2.rectangle(img, (16, y - 12), (30, y + 2), c, -1)
        cv2.putText(img, PART_NAMES[pid], (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def main() -> None:
    ap = argparse.ArgumentParser(description="Export debug outputs for front/back labels, per-part image masks, and projected SMPL part overlays")
    ap.add_argument("--run-dir", required=True, help="Approach run dir (contains run_info.json and images/keypoints) or parent of stage dir")
    ap.add_argument("--stage-dir", required=True, help="Refinement stage dir (bundle_stage or refine_stage)")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--landmarks-json", required=True)
    ap.add_argument("--smpl-model", default=r"E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    stage_dir = Path(args.stage_dir)
    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_json = stage_dir / "uv_part_aware_raycast" / "camera_front_back_labels.json"
    if not label_json.exists():
        raise FileNotFoundError(f"Missing labels json: {label_json}")
    labels = json.loads(label_json.read_text(encoding="utf-8"))

    lms = json.loads(Path(args.landmarks_json).read_text(encoding="utf-8"))

    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        run_info_path = run_dir.parent / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    focal = float(run_info.get("focal_length", 247.3))

    frame_names = sorted([k for k in labels.keys() if (frames_dir / k).exists()])
    frame_names = _sample_names(frame_names, args.n_samples)

    summary = {"frames": []}

    part_masks_dir = out_dir / "per_frame_part_masks"
    label_vis_dir = out_dir / "front_back_labels"
    mesh_part_dir = out_dir / "smpl_part_projection"
    part_masks_dir.mkdir(parents=True, exist_ok=True)
    label_vis_dir.mkdir(parents=True, exist_ok=True)
    mesh_part_dir.mkdir(parents=True, exist_ok=True)

    part_v = _smpl_vertex_parts(Path(args.smpl_model))

    for name in frame_names:
        stem = Path(name).stem
        img = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        arr = lms.get(name)
        lm_np = None
        if arr is not None:
            lm_np = np.array(
                [[(np.nan if r[0] is None else float(r[0])), (np.nan if r[1] is None else float(r[1])), float(r[2])] for r in arr],
                dtype=np.float32,
            )

        person_mask = _person_mask_from_file(masks_dir, name, h, w)
        pm = _build_part_masks(lm_np, person_mask, h, w)

        frame_mask_dir = part_masks_dir / stem
        frame_mask_dir.mkdir(parents=True, exist_ok=True)

        # save individual part masks + composite
        comp = np.zeros((h, w, 3), dtype=np.uint8)
        for pid in [PART_FACE, PART_TORSO, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]:
            mk = pm[pid]
            cv2.imwrite(str(frame_mask_dir / f"part_{PART_NAMES[pid]}.png"), mk)
            color = PART_COLORS[pid]
            for ch in range(3):
                comp[..., ch] = np.where(mk > 0, color[ch], comp[..., ch])

        blend = cv2.addWeighted(img, 0.55, comp, 0.45, 0.0)
        _draw_part_legend(blend)
        cv2.imwrite(str(frame_mask_dir / "parts_overlay.jpg"), blend)

        # label visual
        info = labels.get(name, {"label": "side", "frontness": 0.5})
        lbl = str(info.get("label", "side"))
        fr = float(info.get("frontness", 0.5))
        col = LABEL_COLORS.get(lbl, (255, 255, 255))
        vis = img.copy()
        cv2.putText(vis, f"{name}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"label={lbl} frontness={fr:.3f}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
        cv2.rectangle(vis, (4, 4), (w - 5, h - 5), col, 3)
        cv2.imwrite(str(label_vis_dir / f"label_{stem}.jpg"), vis)

        # projected SMPL part attachment view
        pkl_path = stage_dir / "bundle_refined" / stem / "bundle_refined.pkl"
        obj_path = stage_dir / "bundle_refined" / stem / "bundle_refined.obj"
        if pkl_path.exists() and obj_path.exists():
            d = pickle.load(open(pkl_path, "rb"))
            R = np.asarray(d["camera_rotation"], dtype=np.float32).reshape(3, 3)
            t = np.asarray(d["camera_translation"], dtype=np.float32).reshape(3)
            K = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

            v = _load_obj_verts(obj_path)
            if len(v) == len(part_v):
                uv, z = _project_points(v, K, R, t)
                in_frame = (
                    np.isfinite(uv).all(axis=1)
                    & np.isfinite(z)
                    & (z > 1e-3)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < w)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < h)
                )

                mesh_vis = img.copy()
                idx = np.where(in_frame)[0]
                if len(idx) > 0:
                    idx = idx[::5]  # thinner draw
                for pid in [PART_FACE, PART_TORSO, PART_LARM, PART_RARM, PART_LLEG, PART_RLEG]:
                    sel = idx[part_v[idx] == pid]
                    if len(sel) == 0:
                        continue
                    pts = np.round(uv[sel]).astype(np.int32)
                    c = PART_COLORS[pid]
                    for x, y in pts:
                        cv2.circle(mesh_vis, (int(x), int(y)), 1, c, -1, cv2.LINE_AA)

                _draw_part_legend(mesh_vis)
                cv2.putText(mesh_vis, "SMPL projected parts (what gets attached)", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(mesh_part_dir / f"mesh_parts_{stem}.jpg"), mesh_vis)

        summary["frames"].append({
            "name": name,
            "label": lbl,
            "frontness": fr,
            "parts_dir": str(frame_mask_dir),
            "label_vis": str(label_vis_dir / f"label_{stem}.jpg"),
            "mesh_part_vis": str(mesh_part_dir / f"mesh_parts_{stem}.jpg"),
        })

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved debug outputs -> {out_dir}")


if __name__ == "__main__":
    main()
