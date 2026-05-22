from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


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


def _find_first_result_pkl(smplifyx_out: Path) -> Path:
    cands = sorted((smplifyx_out / "results").glob("*/*.pkl"))
    if not cands:
        raise FileNotFoundError(f"No SMPLify-X result PKL files found under {smplifyx_out / 'results'}")
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Render projected SMPLify-X first-frame joints overlay")
    ap.add_argument("--smplifyx-out", required=True, help="Path to SMPLify-X output folder containing results/ and meshes/")
    ap.add_argument("--model-folder", required=True, help="Path to model folder used by SMPLify-X run")
    ap.add_argument("--image", required=True, help="Original image used for first-frame fitting")
    ap.add_argument("--output", required=True, help="Output overlay image path")
    args = ap.parse_args()

    smplifyx_out = Path(args.smplifyx_out)
    model_folder = Path(args.model_folder)
    image_path = Path(args.image)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    pkl_path = _find_first_result_pkl(smplifyx_out)

    _numpy_compat_for_chumpy()
    import smplx

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    h, w = img.shape[:2]

    res = pickle.load(open(pkl_path, "rb"), encoding="latin1")
    model = smplx.create(str(model_folder), model_type="smpl", gender="neutral", use_pca=False, create_transl=False)
    model.eval()

    betas = torch.tensor(res["betas"], dtype=torch.float32)
    body_pose = torch.tensor(res["body_pose"], dtype=torch.float32)
    global_orient = torch.tensor(res["global_orient"], dtype=torch.float32)

    with torch.no_grad():
        out = model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)

    joints = out.joints[0].cpu().numpy()
    r = np.asarray(res["camera_rotation"], dtype=np.float32).reshape(3, 3)
    t = np.asarray(res["camera_translation"], dtype=np.float32).reshape(3)
    k = np.array([[5000.0, 0.0, w / 2.0], [0.0, 5000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

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

    cv2.putText(vis, "SMPLify-X first-frame projected joints", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"drawn_joints={drawn}  pkl={pkl_path.name}", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis)
    print(f"Saved overlay -> {out_path}")


if __name__ == "__main__":
    main()
