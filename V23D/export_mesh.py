from pathlib import Path

from src.recon.mesh_extractor import export_mesh


def main() -> None:
    export_mesh(recon_dir=Path("data/recon"), mesh_dir=Path("data/mesh"))


if __name__ == "__main__":
    main()
