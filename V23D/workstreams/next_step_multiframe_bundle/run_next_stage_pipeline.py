from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # C:/V23D/V23D


def _run(cmd: list[str], title: str, strict: bool = False) -> bool:
    print(f"\n=== {title} ===")
    print(" ".join([f'\"{c}\"' if " " in c else c for c in cmd]))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Stage failed with exit code {e.returncode}: {title}")
        if strict:
            raise
        return False


def _check_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def main():
    ap = argparse.ArgumentParser(description="Next-stage SMPLify-X-only first-frame pipeline (no MediaPipe rigid execution path)")
    ap.add_argument("--smplifyx-root", default=r"C:/smplify-x")
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--smpl-neutral-pkl", required=True,
                    help="Path to SMPL_NEUTRAL-compatible .pkl (e.g. basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl)")
    ap.add_argument("--output-root", required=True, help="Root output for this next-stage run")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--strict", action="store_true", help="Stop pipeline immediately on first stage failure")
    ap.add_argument("--skip-fit", action="store_true", help="Skip SMPLify-X first-frame fit stage")
    ap.add_argument("--skip-overlay", action="store_true", help="Skip SMPLify-X overlay rendering stage")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_dir)
    frames_dir = Path(args.frames_dir)
    smpl_neutral_pkl = Path(args.smpl_neutral_pkl)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    smplifyx_out = out_root / "smplifyx_firstframe"
    dbg_out = out_root / "smplifyx_debug"
    smplifyx_out.mkdir(parents=True, exist_ok=True)
    dbg_out.mkdir(parents=True, exist_ok=True)

    _check_exists(colmap_dir / "cameras.txt", "COLMAP cameras.txt")
    _check_exists(colmap_dir / "images.txt", "COLMAP images.txt")
    _check_exists(smpl_neutral_pkl, "SMPL neutral pkl")

    fit_script = ROOT / "workflows" / "smpl_fitting" / "run_smplifyx_torchvision_firstframe.py"
    dbg_script = ROOT / "workflows" / "debug_visualization" / "debug_smplifyx_firstframe_overlay.py"

    # Stage 1: SMPLify-X first-frame fit only
    if not args.skip_fit:
        fit_cmd = [
            args.python,
            str(fit_script),
            "--frames-dir", str(frames_dir),
            "--smplifyx-root", str(Path(args.smplifyx_root)),
            "--smpl-neutral-pkl", str(smpl_neutral_pkl),
            "--output", str(smplifyx_out),
            "--python", str(args.python),
        ]
        _run(fit_cmd, "Stage 1/2: SMPLify-X first-frame fit", strict=args.strict)

    # Stage 2: SMPLify-X first-frame overlay
    if not args.skip_overlay:
        first_img = smplifyx_out / "images" / "frame_00000.jpg"
        if not first_img.exists():
            imgs = sorted((smplifyx_out / "images").glob("*.*"))
            if not imgs:
                raise FileNotFoundError(f"No copied first-frame image in {smplifyx_out / 'images'}")
            first_img = imgs[0]
        dbg_cmd = [
            args.python,
            str(dbg_script),
            "--smplifyx-out", str(smplifyx_out / "smplifyx_output"),
            "--model-folder", str(smplifyx_out / "model_folder"),
            "--image", str(first_img),
            "--output", str(dbg_out / "smplifyx_firstframe_overlay.jpg"),
        ]
        _run(dbg_cmd, "Stage 2/2: SMPLify-X overlay", strict=args.strict)

    print("\nDone.")
    print(f"smplifyx fit : {smplifyx_out}")
    print(f"overlay      : {dbg_out / 'smplifyx_firstframe_overlay.jpg'}")
    print(f"mesh         : {smplifyx_out / 'smplifyx_output' / 'meshes'}")


if __name__ == "__main__":
    main()
