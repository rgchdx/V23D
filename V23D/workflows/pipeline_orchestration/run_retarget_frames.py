from argparse import ArgumentParser
from pathlib import Path

from src.preprocess.retarget import retarget_frames_by_masks


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Retarget human frames to a stable body anchor using masks.")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"), help="Input frames directory")
    parser.add_argument("--masks", type=Path, default=Path("data/masks"), help="Input masks directory")
    parser.add_argument("--out-frames", type=Path, required=True, help="Output normalized frames directory")
    parser.add_argument("--out-masks", type=Path, default=None, help="Optional output normalized masks directory")
    parser.add_argument(
        "--anchor-mode",
        type=str,
        default="torso",
        choices=["mask_centroid", "bbox_center", "head", "torso", "pelvis"],
        help="Body anchor used to stabilize the sequence",
    )
    parser.add_argument("--width", type=int, default=None, help="Output width (defaults to source width)")
    parser.add_argument("--height", type=int, default=None, help="Output height (defaults to source height)")
    parser.add_argument("--target-anchor-x", type=float, default=0.5, help="Normalized output x for the anchor")
    parser.add_argument("--target-anchor-y", type=float, default=0.42, help="Normalized output y for the anchor")
    parser.add_argument(
        "--target-subject-height-ratio",
        type=float,
        default=0.82,
        help="Desired subject bbox height as a fraction of output height",
    )
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA smoothing for anchor and scale")
    parser.add_argument("--hard-mask-background", action="store_true", help="Zero out background after warping")
    parser.add_argument(
        "--border-mode",
        type=str,
        default="replicate",
        choices=["replicate", "constant"],
        help="How to fill warped image borders",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_size = None
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            raise ValueError("Provide both --width and --height, or neither.")
        output_size = (args.width, args.height)

    meta_path = retarget_frames_by_masks(
        frames_dir=args.frames,
        masks_dir=args.masks,
        out_frames_dir=args.out_frames,
        out_masks_dir=args.out_masks,
        anchor_mode=args.anchor_mode,
        output_size=output_size,
        target_anchor_xy=(args.target_anchor_x, args.target_anchor_y),
        target_subject_height_ratio=args.target_subject_height_ratio,
        smooth_alpha=args.smooth_alpha,
        hard_mask_background=args.hard_mask_background,
        border_mode=args.border_mode,
    )
    print(f"Saved retargeted frames to {args.out_frames}")
    if args.out_masks is not None:
        print(f"Saved retargeted masks to {args.out_masks}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()