from pathlib import Path

from src.recon.train_launcher import train_reconstruction


def main() -> None:
    train_reconstruction(
        frames_dir=Path("data/frames"),
        poses_dir=Path("data/colmap"),
        output_dir=Path("data/recon"),
        backend="gaussian-splatting",
    )


if __name__ == "__main__":
    main()
