from pathlib import Path
import json
import logging
import shutil


MESH_EXTS = (".ply", ".obj", ".glb", ".gltf", ".stl")


def _find_mesh_candidates(recon_dir: Path) -> list[Path]:
    return sorted(
        [p for p in recon_dir.rglob("*") if p.is_file() and p.suffix.lower() in MESH_EXTS],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )


def export_mesh(recon_dir: Path, mesh_dir: Path) -> None:
    """Extract/export mesh from trained reconstruction outputs."""
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if not recon_dir.exists():
        raise FileNotFoundError(f"Reconstruction directory not found: {recon_dir}")

    candidates = _find_mesh_candidates(recon_dir)
    manifest = {
        "recon_dir": str(recon_dir),
        "mesh_dir": str(mesh_dir),
        "candidate_count": len(candidates),
        "candidates": [str(p) for p in candidates],
    }

    manifest_path = mesh_dir / "mesh_manifest.json"

    if not candidates:
        manifest["status"] = "no_mesh_found"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logging.warning("No mesh files found under %s", recon_dir)
        return

    best = candidates[0]
    out_name = f"model{best.suffix.lower()}"
    out_path = mesh_dir / out_name
    shutil.copy2(best, out_path)

    manifest["status"] = "copied"
    manifest["selected_source"] = str(best)
    manifest["exported_mesh"] = str(out_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Exported mesh to %s", out_path)
