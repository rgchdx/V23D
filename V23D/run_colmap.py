from argparse import ArgumentParser
from pathlib import Path

from src.sfm.colmap_runner import run_colmap


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Run COLMAP SfM on extracted frames.")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"), help="Input frames directory")
    parser.add_argument("--masks", type=Path, default=Path("data/masks"), help="Input masks directory")
    parser.add_argument("--out", type=Path, default=Path("data/colmap"), help="COLMAP output directory")
    parser.add_argument(
        "--matcher",
        type=str,
        default="sequential",
        choices=["sequential", "exhaustive"],
        help="COLMAP matcher type",
    )
    parser.add_argument("--gpu", action="store_true", help="Enable COLMAP GPU options when supported")
    parser.add_argument(
        "--strict-human-only",
        action="store_true",
        default=False,
        help="Require masks and enforce in-mask match quality checks",
    )
    parser.add_argument(
        "--no-strict-human-only",
        action="store_false",
        dest="strict_human_only",
        help="Allow SfM even when masks are missing/looser",
    )
    parser.add_argument(
        "--min-in-mask-ratio",
        type=float,
        default=0.98,
        help="Fail SfM if triangulated observations inside masks are below this ratio",
    )
    parser.add_argument(
        "--hard-mask-images",
        action="store_true",
        default=False,
        help="Zero out background pixels before COLMAP so they can never contribute features",
    )
    parser.add_argument(
        "--no-hard-mask-images",
        action="store_false",
        dest="hard_mask_images",
        help="Use raw frames for COLMAP image_path",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Binary threshold for hard image masking",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_colmap(
        frames_dir=args.frames,
        masks_dir=args.masks,
        output_dir=args.out,
        matcher=args.matcher,
        use_gpu=args.gpu,
        strict_human_only=args.strict_human_only,
        min_in_mask_ratio=args.min_in_mask_ratio,
        hard_mask_images=args.hard_mask_images,
        mask_threshold=args.mask_threshold,
    )


if __name__ == "__main__":
    main()
