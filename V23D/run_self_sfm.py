from pathlib import Path

from src.sfm.self_sfm import run_self_sfm


def main() -> None:
    run_self_sfm(
        frames_dir=Path("data/frames"),
        masks_dir=Path("data/masks"),
        output_dir=Path("data/colmap_self"),
        step=1,
        min_matches=80,
        min_inliers=20,
    )


if __name__ == "__main__":
    main()
