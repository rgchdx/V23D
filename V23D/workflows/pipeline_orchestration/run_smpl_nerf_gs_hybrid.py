from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.recon.nerf_fallback import prepare_nerf_dataset_from_sfm
from src.recon.three_dgs_followup import prepare_3dgs_scene_from_sfm


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _resolve_sparse_model_dir(colmap_dir: Path) -> Path:
    run_meta = colmap_dir / "colmap_run.json"
    if run_meta.exists():
        try:
            meta = json.loads(run_meta.read_text(encoding="utf-8"))
            p = Path(meta.get("selected_model_dir", ""))
            if p.exists():
                return p
        except Exception:
            pass

    sparse_root = colmap_dir / "sparse"
    if (sparse_root / "0").exists():
        return sparse_root / "0"
    if sparse_root.exists():
        candidates = sorted([d for d in sparse_root.iterdir() if d.is_dir()])
        for d in candidates:
            if all((d / n).exists() for n in ("cameras.bin", "images.bin", "points3D.bin")):
                return d
    raise FileNotFoundError(f"Could not resolve sparse model dir under {colmap_dir}")


def _apply_mask_to_image(img_path: Path, mask_path: Path, threshold: int = 127) -> bool:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return False
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    fg = (mask > threshold).astype(np.uint8)
    out = img.copy()
    out[fg == 0] = 0
    cv2.imwrite(str(img_path), out)
    return True


def _prepare_3dgs_scene_fallback_no_colmap(frames_dir: Path, masks_dir: Path, colmap_dir: Path, scene_dir: Path) -> None:
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    sparse_0 = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_0.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in sorted(frames_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            dst = images_dir / p.name
            shutil.copy2(p, dst)
            stem = p.stem
            m = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = masks_dir / f"{stem}{ext}"
                if cand.exists():
                    m = cand
                    break
            if m is not None:
                _apply_mask_to_image(dst, m)
            copied += 1

    src_sparse = _resolve_sparse_model_dir(colmap_dir)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(src_sparse / name, sparse_0 / name)

    info = {
        "scene_prep_mode": "fallback_copy_no_undistort",
        "num_images": copied,
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "source_sparse": str(src_sparse),
    }
    (scene_dir / "scene_prep_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Approach 4: SMPL + NeRF/3DGS hybrid preparation")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--colmap-dir", required=True, help="COLMAP run root (contains sparse/*)")
    ap.add_argument("--smpl-canonical", required=True, help="Canonical SMPL mesh from prior stage")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-strict-mask-training", action="store_true")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    colmap_dir = Path(args.colmap_dir)
    smpl_canonical = Path(args.smpl_canonical)
    out_dir = Path(args.output)

    out_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir = out_dir / "smpl_prior"
    gs_scene_dir = out_dir / "gs_scene"
    nerf_dataset_dir = out_dir / "nerf_dataset"
    smpl_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "colmap_dir": str(colmap_dir),
        "smpl_canonical_input": str(smpl_canonical),
        "output_dir": str(out_dir),
        "steps": {},
    }

    copied = _copy_if_exists(smpl_canonical, smpl_dir / "smpl_canonical.obj")
    summary["steps"]["copy_smpl_prior"] = {
        "ok": bool(copied),
        "output": str(smpl_dir / "smpl_canonical.obj"),
    }

    try:
        prepare_3dgs_scene_from_sfm(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            colmap_dir=colmap_dir,
            scene_dir=gs_scene_dir,
            strict_mask_training=not args.no_strict_mask_training,
        )
        summary["steps"]["prepare_3dgs_scene"] = {
            "ok": True,
            "output": str(gs_scene_dir),
            "info": str(gs_scene_dir / "scene_prep_info.json"),
            "mode": "colmap_undistort",
        }
    except Exception as e:
        err = str(e)
        if "COLMAP executable not found in PATH" in err:
            try:
                _prepare_3dgs_scene_fallback_no_colmap(
                    frames_dir=frames_dir,
                    masks_dir=masks_dir,
                    colmap_dir=colmap_dir,
                    scene_dir=gs_scene_dir,
                )
                summary["steps"]["prepare_3dgs_scene"] = {
                    "ok": True,
                    "output": str(gs_scene_dir),
                    "info": str(gs_scene_dir / "scene_prep_info.json"),
                    "mode": "fallback_copy_no_undistort",
                    "warning": err,
                }
            except Exception as e2:
                summary["steps"]["prepare_3dgs_scene"] = {
                    "ok": False,
                    "error": f"{err}; fallback_failed={e2}",
                }
        else:
            summary["steps"]["prepare_3dgs_scene"] = {
                "ok": False,
                "error": err,
            }

    try:
        prepare_nerf_dataset_from_sfm(
            frames_dir=frames_dir,
            colmap_dir=colmap_dir,
            dataset_dir=nerf_dataset_dir,
            convert_to_opengl=True,
        )
        summary["steps"]["prepare_nerf_dataset"] = {
            "ok": True,
            "output": str(nerf_dataset_dir),
            "info": str(nerf_dataset_dir / "nerf_dataset_info.json"),
        }
    except Exception as e:
        summary["steps"]["prepare_nerf_dataset"] = {
            "ok": False,
            "error": str(e),
        }

    summary_path = out_dir / "hybrid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved hybrid prep summary -> {summary_path}")


if __name__ == "__main__":
    main()
