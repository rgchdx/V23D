from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

from src.recon.smpl_fitter import _read_colmap_images_txt


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -(R.T @ t)


def _sample_nearest(names: list[str], centers: dict[str, np.ndarray], anchor_name: str, k: int) -> list[str]:
    anchor = centers[anchor_name]
    ordered = sorted(names, key=lambda n: float(np.linalg.norm(centers[n] - anchor)))
    return ordered[: max(1, k)]


def _load_preview(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    img = cv2.imread(str(path))
    if img is None:
        canvas = Image.new("RGB", size, (32, 32, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), label, fill=(255, 255, 255))
        return canvas
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, size[1] - 22), label, fill=(255, 255, 255))
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Select front/back support frames for SMPL texture baking")
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--rigid-out", required=True, help="Visible-rigid output with trans_per_frame.npy")
    ap.add_argument("--output", required=True)
    ap.add_argument("--front-frame", default=None, help="Optional explicit front frame name")
    ap.add_argument("--k-front", type=int, default=8)
    ap.add_argument("--k-back", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _read_colmap_images_txt(Path(args.colmap_dir) / "images.txt")
    trans = np.load(str(Path(args.rigid_out) / "trans_per_frame.npy"), allow_pickle=True).item()
    names = sorted([n for n in images.keys() if n in trans])
    if not names:
        raise RuntimeError("No overlapping frame names between COLMAP and rigid fit")

    centers = {n: _camera_center(images[n]["R"], images[n]["t"]) for n in names}
    front_name = args.front_frame if args.front_frame in centers else names[0]
    back_name = max(names, key=lambda n: float(np.linalg.norm(centers[n] - centers[front_name])))

    front_group = _sample_nearest(names, centers, front_name, args.k_front)
    back_group = _sample_nearest(names, centers, back_name, args.k_back)

    weights = {n: 1.0 for n in names}
    for n in front_group:
        weights[n] = max(weights[n], 3.0)
    for n in back_group:
        weights[n] = max(weights[n], 3.0)
    weights[front_name] = 4.0
    weights[back_name] = 4.0

    payload = {
        "front_anchor": front_name,
        "back_anchor": back_name,
        "front_group": front_group,
        "back_group": back_group,
        "weights": weights,
    }
    (out_dir / "support_views.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    frames_dir = Path(args.frames_dir)
    tiles = []
    for title, seq in [("front", front_group), ("back", back_group)]:
        for n in seq:
            img_path = frames_dir / n
            if not img_path.exists():
                stem = Path(n).stem
                matches = list(frames_dir.glob(stem + ".*"))
                if matches:
                    img_path = matches[0]
            tiles.append(_load_preview(img_path, f"{title}: {Path(n).stem}", (420, 320)))

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 420, rows * 320), (24, 24, 24))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 420, (i // cols) * 320))
    sheet.save(out_dir / "support_views_contact_sheet.jpg", quality=90)
    print(f"Saved support-view selection -> {out_dir / 'support_views.json'}")
    print(f"front_anchor={front_name}  back_anchor={back_name}")


if __name__ == "__main__":
    main()
