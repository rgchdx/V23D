from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _obj_to_ply(py: str, src_obj: Path, dst_ply: Path) -> None:
    code = (
        "import open3d as o3d; "
        f"m=o3d.io.read_triangle_mesh(r'{str(src_obj)}'); "
        f"ok=o3d.io.write_triangle_mesh(r'{str(dst_ply)}', m); "
        "print('ok=',ok,'verts=',len(m.vertices),'tris=',len(m.triangles))"
    )
    _run([py, "-c", code])


def _bake(py: str, root: Path, subdir: str, cameras: Path, images: Path, frames: Path, masks: Path, tex_size: int) -> None:
    base = root / subdir
    obj = base / "bundle_canonical.obj"
    ply = base / "bundle_canonical.ply"
    out = base / "uv_raycast_texture"

    if not obj.exists():
        raise FileNotFoundError(f"Missing mesh: {obj}")

    _obj_to_ply(py, obj, ply)
    _run([
        py,
        str(Path(__file__).resolve().parent / "bake_texture.py"),
        "--mesh", str(ply),
        "--cameras", str(cameras),
        "--images", str(images),
        "--frames", str(frames),
        "--masks", str(masks),
        "--output", str(out),
        "--tex-size", str(tex_size),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="UV raycast texture baking for orbit methods 2 and 3")
    ap.add_argument("--python", default="python")
    ap.add_argument("--orbit-root", default=r"E:/V23D_Data/orbit_methods")
    ap.add_argument("--colmap-dir", default=r"E:/V23D_Data/colmap_rerun/sparse/1")
    ap.add_argument("--frames-dir", default=r"E:/V23D_Data/frames")
    ap.add_argument("--masks-dir", default=r"E:/V23D_Data/masks_rerun")
    ap.add_argument("--tex-size", type=int, default=2048)
    args = ap.parse_args()

    root = Path(args.orbit_root)
    cameras = Path(args.colmap_dir) / "cameras.txt"
    images = Path(args.colmap_dir) / "images.txt"
    frames = Path(args.frames_dir)
    masks = Path(args.masks_dir)

    _bake(args.python, root, "02_smplifyx_perframe_then_bundle/bundle_stage", cameras, images, frames, masks, args.tex_size)
    _bake(args.python, root, "03_regression_plus_refinement/refine_stage", cameras, images, frames, masks, args.tex_size)

    print("Done: UV raycast textures generated for approaches 2 and 3")


if __name__ == "__main__":
    main()
