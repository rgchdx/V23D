from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.preprocess.masking import generate_masks
from src.recon.three_dgs_followup import run_3dgs_followup
from src.sfm.colmap_runner import run_colmap


def build_argparser() -> ArgumentParser:
    p = ArgumentParser(description="Strict human-only pipeline: masks -> SfM -> 3DGS")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--colmap", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gs-repo", type=Path, required=True)

    p.add_argument("--gpu", action="store_true")
    p.add_argument("--matcher", type=str, default="sequential", choices=["sequential", "exhaustive"])
    p.add_argument("--iterations", type=int, default=30000)

    # strict masking defaults
    p.add_argument("--alpha-threshold", type=int, default=190)
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--close-kernel", type=int, default=7)
    p.add_argument("--erode-px", type=int, default=3)
    p.add_argument("--min-in-mask-ratio", type=float, default=0.99)

    p.add_argument("--skip-3dgs", action="store_true", help="Run only masks + SfM")
    p.add_argument("--python-exe", type=str, default="python")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    print("=" * 72)
    print("Step 1/3: Strict masks")
    print("=" * 72)
    generate_masks(
        frames_dir=args.frames,
        masks_dir=args.masks,
        model="rembg",
        alpha_threshold=args.alpha_threshold,
        open_kernel=args.open_kernel,
        close_kernel=args.close_kernel,
        erode_px=args.erode_px,
        keep_largest=True,
    )

    print("\n" + "=" * 72)
    print("Step 2/3: Strict human-only SfM (COLMAP)")
    print("=" * 72)
    run_colmap(
        frames_dir=args.frames,
        masks_dir=args.masks,
        output_dir=args.colmap,
        matcher=args.matcher,
        use_gpu=args.gpu,
        strict_human_only=True,
        min_in_mask_ratio=args.min_in_mask_ratio,
        hard_mask_images=True,
        mask_threshold=127,
    )

    if args.skip_3dgs:
        print("\nSkipping 3DGS stage (--skip-3dgs).")
        return

    print("\n" + "=" * 72)
    print("Step 3/3: 3DGS follow-up (strict masked undistorted training images)")
    print("=" * 72)
    run_3dgs_followup(
        frames_dir=args.frames,
        masks_dir=args.masks,
        colmap_dir=args.colmap,
        gs_repo_dir=args.gs_repo,
        scene_dir=args.scene,
        model_dir=args.model,
        iterations=args.iterations,
        python_exe=args.python_exe,
        strict_mask_training=True,
    )


if __name__ == "__main__":
    main()
