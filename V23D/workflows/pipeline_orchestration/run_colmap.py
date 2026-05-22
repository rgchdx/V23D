from pathlib import Path

from src.sfm.colmap_runner import run_colmap


def main() -> None:
    run_colmap(
        frames_dir=Path("data/frames"),
        masks_dir=Path("data/masks"),
        output_dir=Path("data/colmap"),
    )


if __name__ == "__main__":
    main()
