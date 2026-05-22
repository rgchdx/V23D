from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _numpy_compat_for_chumpy() -> None:
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "complex"):
        np.complex = complex
    if not hasattr(np, "object"):
        np.object = object
    if not hasattr(np, "str"):
        np.str = str
    if not hasattr(np, "unicode"):
        np.unicode = str


def _collect_results(smplifyx_out: Path) -> list[Path]:
    return sorted((smplifyx_out / "results").glob("*/*.pkl"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Render overlays for sampled SMPLify-X fits")
    ap.add_argument("--run-dir", required=True, help="Directory containing images/, model_folder/, smplifyx_output/, run_info.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    smplifyx_out = run_dir / "smplifyx_output"
    model_folder = run_dir / "model_folder"
    images_dir = run_dir / "images"
    info_path = run_dir / "run_info.json"

    if not smplifyx_out.exists():
        raise FileNotFoundError(smplifyx_out)
    if not model_folder.exists():
        raise FileNotFoundError(model_folder)

    focal = 5000.0
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        focal = float(info.get("focal_length", 5000.0))

    _numpy_compat_for_chumpy()
    import smplx

    model = smplx.create(str(model_folder), model_type="smpl", gender="neutral", use_pca=False, create_transl=False)
    model.eval()

    pkl_paths = _collect_results(smplifyx_out)
    if not pkl_paths:
        raise RuntimeError(f"No results under {smplifyx_out / 'results'}")

    saved: list[Path] = []
    for pkl_path in pkl_paths:
        frame_name = pkl_path.parent.name + ".jpg"
        img_path = images_dir / frame_name
        if not img_path.exists():
            cands = list(images_dir.glob(pkl_path.parent.name + ".*"))
            if not cands:
                continue
            img_path = cands[0]

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        res = pickle.load(open(pkl_path, "rb"), encoding="latin1")
        betas = torch.tensor(res["betas"], dtype=torch.float32)
        body_pose = torch.tensor(res["body_pose"], dtype=torch.float32)
        global_orient = torch.tensor(res["global_orient"], dtype=torch.float32)

        with torch.no_grad():
            out = model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)

        joints = out.joints[0].cpu().numpy()
        r = np.asarray(res["camera_rotation"], dtype=np.float32).reshape(3, 3)
        t = np.asarray(res["camera_translation"], dtype=np.float32).reshape(3)
        k = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        j_cam = (r @ joints.T).T + t.reshape(1, 3)
        valid = j_cam[:, 2] > 1e-4
        uv = np.full((len(joints), 2), np.nan, dtype=np.float32)
        uvw = (k @ j_cam[valid].T).T
        uv[valid] = uvw[:, :2] / uvw[:, 2:3]

        vis = img.copy()
        drawn = 0
        for x, y in uv:
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(vis, (xi, yi), 3, (0, 255, 0), -1, cv2.LINE_AA)
                drawn += 1

        cv2.putText(vis, f"{img_path.name}  focal={focal:.1f}  joints={drawn}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = out_dir / f"overlay_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        saved.append(out_path)

    if not saved:
        raise RuntimeError("No overlays generated")

    ims = [Image.open(str(p)).convert("RGB") for p in saved]
    w = max(im.width for im in ims)
    h = max(im.height for im in ims)
    cols = min(3, len(ims))
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(ims):
        r = i // cols
        c = i % cols
        sheet.paste(im.resize((w, h)), (c * w, r * h))

    cs = out_dir / "smplifyx_sampled_overlay_contact_sheet.jpg"
    sheet.save(str(cs), quality=95)
    print(f"Saved overlays -> {out_dir}")
    print(f"Saved contact sheet -> {cs}")


if __name__ == "__main__":
    main()
