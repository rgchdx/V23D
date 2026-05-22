"""
extract_mesh.py  –  Build a coloured, scale-normalised surface mesh.

Two modes
---------
    --input  : use an existing 3DGS point_cloud.ply   (fast, no COLMAP needed)
    --scene  : run COLMAP patch_match_stereo + stereo_fusion first

Scale normalisation
-------------------
After meshing the result is centred at the origin and the largest bounding-box
extent is scaled to --target-height (default 0 = disabled).  This puts the mesh in
metric space suitable for SMPL / SMPLify-X fitting.

Usage
-----
# from 3DGS point cloud (recommended — already ran successfully):
python extract_mesh.py \
        --input  "E:/V23D_Data/3dgs_model/point_cloud/iteration_30000/point_cloud.ply" \
        --output "E:/V23D_Data/mesh/human_mesh.ply"

# from COLMAP dense stereo:
python extract_mesh.py \
        --scene  "E:/V23D_Data/3dgs_scene" \
        --output "E:/V23D_Data/mesh/human_mesh.ply"
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None  # checked lazily


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a coloured, scale-normalised mesh")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", type=pathlib.Path,
                     help="3DGS point_cloud.ply (fast path)")
    src.add_argument("--scene", type=pathlib.Path,
                     help="Undistorted COLMAP scene dir for dense stereo")
    p.add_argument("--output", type=pathlib.Path,
                   default=pathlib.Path("E:/V23D_Data/mesh/human_mesh.ply"))
    p.add_argument("--fused",  type=pathlib.Path, default=None,
                   help="Save COLMAP fused cloud here (default: temp)")
    p.add_argument("--depth",  type=int,   default=10,
                   help="Poisson octree depth (8-12)")
    p.add_argument("--trim",   type=float, default=0.05,
                   help="Low-density trim fraction (0-0.2)")
    p.add_argument("--no-geom-consistency", action="store_true", default=False)
    p.add_argument("--target-height", type=float, default=0.0,
                   help="Normalise mesh so tallest extent = this value in metres (0 = skip)")
    return p


# This runs a subprocess and prints the command for tarnsparentcy. If the subprocess fails, it prints an error and exits.
def run(cmd: list, label: str) -> None:
    print(f"\n[colmap] {label}")
    print("  " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] colmap exited with code {result.returncode}")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# 3DGS point cloud -> Poisson mesh
# ---------------------------------------------------------------------------


# This section is for the "fast path" where we already have a 3DGS point cloud with SH DC coefficients. 
# We decode the colours from the SH DC coefficients, then run Poisson surface reconstruction to get a mesh.
def decode_3dgs_ply(ply_path: pathlib.Path) -> "o3d.geometry.PointCloud":
    """Load a 3DGS PLY, decode SH DC colour coefficients, return Open3D cloud."""
    from plyfile import PlyData
    pd = PlyData.read(str(ply_path))
    v  = pd["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)

    # DC SH coefficient -> linear RGB: C0 * f_dc + 0.5
    C0 = 0.28209479177387814
    has_sh = all(f"f_dc_{c}" in v.data.dtype.names for c in range(3))
    if has_sh:
        rgb = np.stack([np.asarray(v[f"f_dc_{c}"]) for c in range(3)], axis=1)
        rgb = np.clip(C0 * rgb + 0.5, 0.0, 1.0)
        print("      Decoded colours from SH DC coefficients")
    else:
        rgb = np.full((len(xyz), 3), 0.7)
        print("      [warn] No f_dc_* fields; using grey")

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz)
    pcd.colors  = o3d.utility.Vector3dVector(rgb)
    return pcd


# This section is for the "COLMAP dense path" where we start from an undistorted COLMAP scene, run patch_match_stereo + stereo_fusion to get a dense point cloud, then run Poisson surface reconstruction to get a mesh.
def poisson_from_pcd(pcd: "o3d.geometry.PointCloud",
                     out_ply: pathlib.Path, depth: int, trim: float) -> "o3d.geometry.TriangleMesh":
    print(f"\n[1/5] Point cloud: {len(pcd.points):,} points")
    print("      Removing outliers ...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"      After outlier removal: {len(pcd.points):,} points")

    print("[2/5] Estimating normals ...")
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

    print(f"[3/5] Poisson surface reconstruction (depth={depth}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    print(f"      Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    if trim > 0:
        print(f"[4/5] Trimming low-density floaters (bottom {trim*100:.0f}%) ...")
        d = np.asarray(densities)
        mesh.remove_vertices_by_mask(d < np.quantile(d, trim))
        print(f"      After trim: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    else:
        print("[4/5] Skipping trim.")

    if pcd.has_colors():
        print("[4b]  Transferring colours from point cloud to mesh vertices ...")
        tree = o3d.geometry.KDTreeFlann(pcd)
        cols = np.asarray(pcd.colors)
        vc = np.zeros((len(mesh.vertices), 3))
        for i, v in enumerate(np.asarray(mesh.vertices)):
            _, idx, _ = tree.search_knn_vector_3d(v, 1)
            vc[i] = cols[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

    return mesh


# ---------------------------------------------------------------------------
# COLMAP dense path
# ---------------------------------------------------------------------------

def colmap_dense(scene_dir: pathlib.Path, fused_ply: pathlib.Path, geom: bool) -> None:
    if shutil.which("colmap") is None:
        print("[ERROR] COLMAP not found in PATH.")
        print("  Install with:  conda install -c conda-forge colmap")
        sys.exit(1)

    run([
        "colmap", "patch_match_stereo",
        "--workspace_path",   str(scene_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true" if geom else "false",
        "--PatchMatchStereo.gpu_index", "0",
    ], "patch_match_stereo  (depth estimation — 5-20 min)")

    run([
        "colmap", "stereo_fusion",
        "--workspace_path",   str(scene_dir),
        "--workspace_format", "COLMAP",
        "--input_type",       "geometric" if geom else "photometric",
        "--output_path",      str(fused_ply),
    ], "stereo_fusion  (fuse depth maps -> dense point cloud)")


def poisson_from_fused(fused_ply: pathlib.Path, out_ply: pathlib.Path,
                       depth: int, trim: float) -> "o3d.geometry.TriangleMesh":
    print(f"\n[mesh] Loading fused point cloud: {fused_ply} ...")
    pcd = o3d.io.read_point_cloud(str(fused_ply))
    print(f"       {len(pcd.points):,} points | colors={pcd.has_colors()} | normals={pcd.has_normals()}")
    return poisson_from_pcd(pcd, out_ply, depth, trim)


# ---------------------------------------------------------------------------
# Scale normalisation
# ---------------------------------------------------------------------------

def normalise_mesh(mesh: "o3d.geometry.TriangleMesh",
                   target_height: float) -> "o3d.geometry.TriangleMesh":
    """Centre mesh at origin; scale so largest extent == target_height."""
    bb = mesh.get_axis_aligned_bounding_box()
    centre = (bb.max_bound + bb.min_bound) / 2.0
    extent = bb.get_extent()
    max_ext = float(np.max(extent))
    scale   = target_height / max_ext if max_ext > 0 else 1.0
    verts = np.asarray(mesh.vertices)
    verts = (verts - centre) * scale
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    print(f"\n[norm] Centred + scaled  (×{scale:.6g})  ->  extent now ≈ {np.max(extent)*scale:.3f} m")
    bb2 = mesh.get_axis_aligned_bounding_box()
    print(f"       New bounding box: {np.array(bb2.min_bound).round(3)} -> {np.array(bb2.max_bound).round(3)}")
    return mesh


def save_mesh(mesh: "o3d.geometry.TriangleMesh", out_ply: pathlib.Path) -> None:
    mesh.compute_vertex_normals()
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(out_ply), mesh, write_vertex_colors=True)
    if ok:
        print(f"\n✓  Mesh saved: {out_ply}")
        print(f"   Vertices : {len(mesh.vertices):,}")
        print(f"   Triangles: {len(mesh.triangles):,}")
    else:
        print("[ERROR] Failed to write mesh.")
        sys.exit(1)


def main() -> None:
    args = build_argparser().parse_args()

    if o3d is None:
        print("[ERROR] open3d not installed. Run:  pip install open3d")
        sys.exit(1)

    if args.input:
        # ── fast path: 3DGS PLY ──────────────────────────────────────────
        if not args.input.exists():
            print(f"[ERROR] File not found: {args.input}")
            sys.exit(1)
        print(f"[1/5] Loading point cloud from {args.input} ...")
        pcd  = decode_3dgs_ply(args.input)
        mesh = poisson_from_pcd(pcd, args.output, args.depth, args.trim)

    elif args.scene:
        # ── COLMAP dense path ─────────────────────────────────────────────
        if not args.scene.exists():
            print(f"[ERROR] Scene not found: {args.scene}")
            sys.exit(1)
        geom    = not args.no_geom_consistency
        tmp_dir = None
        if args.fused:
            fused_ply = args.fused
            fused_ply.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp_dir   = tempfile.mkdtemp(prefix="v23d_fused_")
            fused_ply = pathlib.Path(tmp_dir) / "fused.ply"
        try:
            colmap_dense(args.scene, fused_ply, geom)
            mesh = poisson_from_fused(fused_ply, args.output, args.depth, args.trim)
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print("[ERROR] Provide either --input <point_cloud.ply> or --scene <scene_dir>")
        sys.exit(1)

    if args.target_height > 0:
        mesh = normalise_mesh(mesh, args.target_height)

    print(f"\n[5/5] Saving mesh to {args.output} ...")
    save_mesh(mesh, args.output)


if __name__ == "__main__":
    main()
