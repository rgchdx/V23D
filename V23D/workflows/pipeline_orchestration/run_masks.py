from argparse import ArgumentParser
from pathlib import Path

from src.preprocess.masking import generate_masks


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate person masks for extracted frames.")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"), help="Input frames directory")
    parser.add_argument("--masks", type=Path, default=Path("data/masks"), help="Output masks directory")
    parser.add_argument("--model", type=str, default="rembg", help="Masking backend: rembg or whitebg")
    parser.add_argument("--alpha-threshold", type=int, default=180, help="Alpha threshold (0-255)")
    parser.add_argument("--open-kernel", type=int, default=3, help="Morphological open kernel size (odd)")
    parser.add_argument("--close-kernel", type=int, default=7, help="Morphological close kernel size (odd)")
    parser.add_argument("--erode-px", type=int, default=2, help="Foreground erosion in pixels")
    parser.add_argument("--no-keep-largest", action="store_true", help="Do not keep only largest mask component")
    parser.add_argument("--suppress-white-bg", action="store_true", help="Force near-white pixels to background")
    parser.add_argument("--white-threshold", type=int, default=245, help="RGB threshold for white background suppression")
    parser.add_argument("--white-soft-margin", type=int, default=1, help="Dilate detected white area by N px")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    generate_masks(
        frames_dir=args.frames,
        masks_dir=args.masks,
        model=args.model,
        alpha_threshold=args.alpha_threshold,
        open_kernel=args.open_kernel,
        close_kernel=args.close_kernel,
        erode_px=args.erode_px,
        keep_largest=not args.no_keep_largest,
        suppress_white_bg=args.suppress_white_bg,
        white_threshold=args.white_threshold,
        white_soft_margin=args.white_soft_margin,
    )


if __name__ == "__main__":
    main()
