from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.retarget import retarget_frames_by_masks
from src.recon.three_dgs_followup import run_3dgs_followup
from src.sfm.colmap_runner import run_colmap


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Torso-centered frames -> SMPL init -> 3DGS pipeline")
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--output-root", required=True)

    p.add_argument("--gs-repo", required=True, help="Path to gaussian-splatting repo (contains train.py)")
    p.add_argument("--smplifyx-root", default="C:/smplify-x")
    p.add_argument("--smpl-neutral-pkl", required=True)

    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--n-smpl-samples", type=int, default=12)
    p.add_argument("--min-registered-poses", type=int, default=80)
    p.add_argument("--existing-colmap-search-root", default="E:/V23D_Data")

    p.add_argument("--target-anchor-x", type=float, default=0.5)
    p.add_argument("--target-anchor-y", type=float, default=0.42)
    p.add_argument("--target-height-ratio", type=float, default=0.82)
    p.add_argument("--smooth-alpha", type=float, default=0.25)

    p.add_argument("--skip-smpl", action="store_true")
    p.add_argument("--skip-3dgs", action="store_true")
    return p.parse_args()


def _read_colmap_count(colmap_dir: Path) -> int:
    meta_path = colmap_dir / "colmap_run.json"
    if not meta_path.exists():
        return 0
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta.get("num_registered_poses", 0))
    except Exception:
        return 0


def _run_colmap_attempt(
    *,
    frames_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    matcher: str,
    use_gpu: bool,
    strict_human_only: bool,
    min_in_mask_ratio: float,
    hard_mask_images: bool,
) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    run_colmap(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        matcher=matcher,
        use_gpu=use_gpu,
        strict_human_only=strict_human_only,
        min_in_mask_ratio=min_in_mask_ratio,
        hard_mask_images=hard_mask_images,
        mask_threshold=127,
    )
    return _read_colmap_count(output_dir)


def _find_existing_colmap_with_min_poses(search_root: Path, min_poses: int) -> Path | None:
    if not search_root.exists():
        return None
    best_dir = None
    best_count = -1
    for meta_path in search_root.rglob("colmap_run.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        count = int(meta.get("num_registered_poses", 0))
        cand_dir = meta_path.parent
        if count >= min_poses and count > best_count and (cand_dir / "sparse").exists():
            best_count = count
            best_dir = cand_dir
    return best_dir


def main() -> None:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    out_root = Path(args.output_root)
    gs_repo = Path(args.gs_repo)
    smplifyx_root = Path(args.smplifyx_root)
    smpl_pkl = Path(args.smpl_neutral_pkl)

    centered_frames = out_root / "centered" / "frames"
    centered_masks = out_root / "centered" / "masks"
    colmap_dir = out_root / "colmap"
    smpl_init_dir = out_root / "smpl_init"
    gs_scene_dir = out_root / "gs_scene"
    gs_model_dir = out_root / "gs_model"

    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "timestamp_utc": _ts(),
        "inputs": {
            "frames_dir": str(frames_dir),
            "masks_dir": str(masks_dir),
            "gs_repo": str(gs_repo),
            "smplifyx_root": str(smplifyx_root),
            "smpl_neutral_pkl": str(smpl_pkl),
        },
        "outputs": {
            "root": str(out_root),
            "centered_frames": str(centered_frames),
            "centered_masks": str(centered_masks),
            "colmap": str(colmap_dir),
            "smpl_init": str(smpl_init_dir),
            "gs_scene": str(gs_scene_dir),
            "gs_model": str(gs_model_dir),
        },
        "steps": {},
    }

    # 1) Torso centering for each frame
    try:
        retarget_summary = retarget_frames_by_masks(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            out_frames_dir=centered_frames,
            out_masks_dir=centered_masks,
            anchor_mode="torso",
            target_anchor_xy=(float(args.target_anchor_x), float(args.target_anchor_y)),
            target_subject_height_ratio=float(args.target_height_ratio),
            smooth_alpha=float(args.smooth_alpha),
            hard_mask_background=False,
        )
        summary["steps"]["retarget_torso_center"] = {
            "ok": True,
            "summary_json": str(retarget_summary),
        }
        print(f"[OK] torso centering -> {retarget_summary}")
    except Exception as e:
        summary["steps"]["retarget_torso_center"] = {"ok": False, "error": str(e)}
        (out_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise

    # 2) COLMAP on centered frames, relaxing until enough poses are registered
    try:
        attempts = [
            {
                "name": "strict_centered",
                "frames_dir": centered_frames,
                "masks_dir": centered_masks,
                "output_dir": colmap_dir / "strict_centered",
                "matcher": args.matcher,
                "use_gpu": bool(args.gpu),
                "strict_human_only": True,
                "min_in_mask_ratio": 0.99,
                "hard_mask_images": True,
            },
            {
                "name": "relaxed_centered",
                "frames_dir": centered_frames,
                "masks_dir": centered_masks,
                "output_dir": colmap_dir / "relaxed_centered",
                "matcher": "exhaustive",
                "use_gpu": bool(args.gpu),
                "strict_human_only": False,
                "min_in_mask_ratio": 0.0,
                "hard_mask_images": False,
            },
            {
                "name": "relaxed_original",
                "frames_dir": frames_dir,
                "masks_dir": masks_dir,
                "output_dir": colmap_dir / "relaxed_original",
                "matcher": "exhaustive",
                "use_gpu": bool(args.gpu),
                "strict_human_only": False,
                "min_in_mask_ratio": 0.0,
                "hard_mask_images": False,
            },
        ]

        selected_colmap_dir: Path | None = None
        selected_count = -1
        attempt_results: list[dict[str, object]] = []
        for attempt in attempts:
            count = _run_colmap_attempt(
                frames_dir=attempt["frames_dir"],
                masks_dir=attempt["masks_dir"],
                output_dir=attempt["output_dir"],
                matcher=attempt["matcher"],
                use_gpu=attempt["use_gpu"],
                strict_human_only=attempt["strict_human_only"],
                min_in_mask_ratio=attempt["min_in_mask_ratio"],
                hard_mask_images=attempt["hard_mask_images"],
            )
            attempt_results.append({
                "name": attempt["name"],
                "output_dir": str(attempt["output_dir"]),
                "num_registered_poses": count,
            })
            if count > selected_count:
                selected_count = count
                selected_colmap_dir = attempt["output_dir"]
            print(f"[COLMAP] {attempt['name']}: {count} poses")
            if count >= int(args.min_registered_poses):
                selected_colmap_dir = attempt["output_dir"]
                break

        if selected_colmap_dir is None:
            raise RuntimeError("COLMAP produced no usable sparse model")

        # If local attempts are still too weak, adopt an existing strong COLMAP run.
        if selected_count < int(args.min_registered_poses):
            fallback_src = _find_existing_colmap_with_min_poses(
                search_root=Path(args.existing_colmap_search_root),
                min_poses=int(args.min_registered_poses),
            )
            if fallback_src is not None:
                fallback_dst = out_root / "colmap" / "existing_fallback"
                if fallback_dst.exists():
                    shutil.rmtree(fallback_dst, ignore_errors=True)
                shutil.copytree(fallback_src, fallback_dst)
                selected_colmap_dir = fallback_dst
                selected_count = _read_colmap_count(fallback_dst)
                attempt_results.append(
                    {
                        "name": "existing_fallback",
                        "output_dir": str(fallback_dst),
                        "source_dir": str(fallback_src),
                        "num_registered_poses": selected_count,
                    }
                )

        colmap_dir = selected_colmap_dir
        if selected_count < int(args.min_registered_poses):
            raise RuntimeError(
                f"Could not reach required pose count: got {selected_count}, required {int(args.min_registered_poses)}"
            )
        summary["steps"]["colmap_centered"] = {
            "ok": True,
            "colmap_run_json": str(colmap_dir / "colmap_run.json"),
            "attempts": attempt_results,
            "selected_colmap_dir": str(colmap_dir),
            "selected_registered_poses": int(selected_count),
        }
        print(f"[OK] colmap -> {colmap_dir}")
    except Exception as e:
        summary["steps"]["colmap_centered"] = {"ok": False, "error": str(e)}
        (out_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise

    # 3) SMPL init from centered frames + COLMAP focal
    if args.skip_smpl:
        summary["steps"]["smpl_init"] = {"ok": True, "skipped": True}
    else:
        try:
            smpl_cmd = [
                args.python_exe,
                str(Path(__file__).resolve().parents[1] / "smpl_fitting" / "run_smplifyx_torchvision_sampled_frames.py"),
                "--smplifyx-root", str(smplifyx_root),
                "--frames-dir", str(centered_frames),
                "--colmap-dir", str(colmap_dir),
                "--smpl-neutral-pkl", str(smpl_pkl),
                "--output", str(smpl_init_dir),
                "--python", str(args.python_exe),
                "--n-samples", str(args.n_smpl_samples),
            ]
            subprocess.run(smpl_cmd, check=True)
            summary["steps"]["smpl_init"] = {
                "ok": True,
                "run_info": str(smpl_init_dir / "run_info.json"),
                "smplifyx_output": str(smpl_init_dir / "smplifyx_output"),
            }
            print(f"[OK] smpl init -> {smpl_init_dir}")
        except Exception as e:
            summary["steps"]["smpl_init"] = {"ok": False, "error": str(e)}

    # 4) 3DGS training on centered/masked scene
    if args.skip_3dgs:
        summary["steps"]["train_3dgs"] = {"ok": True, "skipped": True}
    else:
        try:
            run_3dgs_followup(
                frames_dir=centered_frames,
                masks_dir=centered_masks,
                colmap_dir=colmap_dir,
                gs_repo_dir=gs_repo,
                scene_dir=gs_scene_dir,
                model_dir=gs_model_dir,
                iterations=int(args.iterations),
                python_exe=args.python_exe,
                strict_mask_training=True,
            )
            summary["steps"]["train_3dgs"] = {
                "ok": True,
                "scene_info": str(gs_scene_dir / "scene_prep_info.json"),
                "train_info": str(gs_model_dir / "train_3dgs_info.json"),
            }
            print(f"[OK] 3dgs -> {gs_model_dir}")
        except Exception as e:
            summary["steps"]["train_3dgs"] = {"ok": False, "error": str(e)}

    summary_path = out_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
