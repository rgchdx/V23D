from argparse import ArgumentParser
from pathlib import Path

from src.recon.three_dgs_followup import run_3dgs_followup
from src.utils.logging_utils import configure_logging


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Run 3DGS followup from SfM/COLMAP outputs.")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"), help="Input frames directory")
    parser.add_argument("--colmap", type=Path, default=Path("data/colmap"), help="COLMAP/SfM output directory")
    parser.add_argument(
        "--gs-repo",
        type=Path,
        required=True,
        help="Path to local gaussian-splatting repository (contains train.py)",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/3dgs_scene"),
        help="Prepared 3DGS scene directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/3dgs_model"),
        help="3DGS output model directory",
    )
    parser.add_argument("--iterations", type=int, default=30000, help="3DGS train iterations")
    parser.add_argument("--python-exe", type=str, default="python", help="Python executable for 3DGS train.py")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging(args.log_level)

    run_3dgs_followup(
        frames_dir=args.frames,
        colmap_dir=args.colmap,
        gs_repo_dir=args.gs_repo,
        scene_dir=args.scene_dir,
        model_dir=args.model_dir,
        iterations=args.iterations,
        python_exe=args.python_exe,
    )


if __name__ == "__main__":
    main()
