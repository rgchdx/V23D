from pathlib import Path

from src.preprocess.masking import generate_masks


def main() -> None:
    frames_dir = Path("data/frames")
    masks_dir = Path("data/masks")
    generate_masks(frames_dir=frames_dir, masks_dir=masks_dir, model="rembg")


if __name__ == "__main__":
    main()
