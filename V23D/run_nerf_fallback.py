from argparse import ArgumentParser
from pathlib import Path

from src.recon.nerf_fallback import run_nerf_fallback
from src.utils.logging_utils import configure_logging


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Run NeRF fallback from SfM/COLMAP outputs.")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"), help="Input frames directory")
    parser.add_argument("--colmap", type=Path, default=Path("data/colmap"), help="COLMAP/SfM output directory")
    parser.add_argument(
        "--nerf-repo",
        type=Path,
        required=True,
        help="Path to local NeRF repository",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/nerf_dataset"),
        help="Prepared NeRF dataset directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/nerf_model"),
        help="NeRF model output directory",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="custom",
        choices=["custom", "nerfstudio"],
        help="Training framework",
    )
    parser.add_argument("--iterations", type=int, default=50000, help="Training iterations")
    parser.add_argument("--python-exe", type=str, default="python", help="Python executable")
    parser.add_argument("--train-script", type=str, default="train.py", help="Train script name for custom framework")
    parser.add_argument(
        "--no-opengl-convert",
        action="store_true",
        help="Disable OpenCV->OpenGL camera conversion in transforms.json",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging(args.log_level)

    run_nerf_fallback(
        frames_dir=args.frames,
        colmap_dir=args.colmap,
        nerf_repo_dir=args.nerf_repo,
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        framework=args.framework,
        iterations=args.iterations,
        python_exe=args.python_exe,
        train_script=args.train_script,
        convert_to_opengl=not args.no_opengl_convert,
    )


if __name__ == "__main__":
    main()
