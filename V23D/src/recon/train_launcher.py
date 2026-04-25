from pathlib import Path
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    logging.info("Running reconstruction command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _build_backend_command(backend: str, frames_dir: Path, poses_dir: Path, output_dir: Path) -> list[str] | None:
    name = backend.strip().lower()

    # NOTE: These are placeholders for common CLI entry points.
    if name in {"gaussian-splatting", "3dgs", "gaussians"}:
        if shutil.which("python") is None:
            return None
        return ["python", "train.py", "-s", str(frames_dir), "--poses", str(poses_dir), "-m", str(output_dir)]

    if name in {"nerf"}:
        if shutil.which("python") is None:
            return None
        return ["python", "train.py", "--data", str(frames_dir), "--poses", str(poses_dir), "--out", str(output_dir)]

    return None


def train_reconstruction(frames_dir: Path, poses_dir: Path, output_dir: Path, backend: str) -> None:
    """Launch the selected reconstruction backend (3DGS/NeuS/etc.)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not poses_dir.exists():
        raise FileNotFoundError(f"Poses directory not found: {poses_dir}")

    cmd = _build_backend_command(backend=backend, frames_dir=frames_dir, poses_dir=poses_dir, output_dir=output_dir)

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "frames_dir": str(frames_dir),
        "poses_dir": str(poses_dir),
        "output_dir": str(output_dir),
    }

    if cmd is None:
        run_info["status"] = "skipped"
        run_info["reason"] = "No known command template for backend."
        (output_dir / "recon_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        logging.warning("No launch command configured for backend '%s'. Wrote recon_run_info.json", backend)
        return

    run_info["status"] = "started"
    run_info["command"] = cmd
    (output_dir / "recon_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    _run(cmd)

    run_info["status"] = "completed"
    (output_dir / "recon_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
