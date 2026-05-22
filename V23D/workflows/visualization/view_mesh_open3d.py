"""
view_mesh_open3d.py
-------------------
Visualise the refined SMPL mesh (OBJ or PLY) with Open3D.

Usage
-----
# Interactive window (default mesh path):
    python workflows/visualization/view_mesh_open3d.py

# Custom mesh:
    python workflows/visualization/view_mesh_open3d.py --mesh E:/V23D_Data/mesh_refine_perspective_4view/mesh_refined_perspective.ply

# Also overlay the canonical (original) mesh for comparison:
    python workflows/visualization/view_mesh_open3d.py --compare-canonical

# Export a screenshot instead of opening the window:
    python workflows/visualization/view_mesh_open3d.py --screenshot E:/V23D_Data/mesh_open3d.png

Notes
-----
- The script DOES NOT modify any reconstruction files.
- If you pass a .ply that already has vertex colours they will be shown;
  otherwise the mesh is rendered with a neutral grey material.
- Press [ ] keys to toggle wireframe, Q to quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is not installed.  Run: pip install open3d")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_MESH = Path(
    r"E:\V23D_Data\mesh_refine_perspective_4view\mesh_refined_perspective.ply"
)
_DEFAULT_CANONICAL = Path(
    r"E:\V23D_Data\orbit_methods\02_smplifyx_perframe_then_bundle"
    r"\bundle_stage\bundle_canonical.obj"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load OBJ or PLY into an Open3D TriangleMesh."""
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if not mesh.has_triangles():
        sys.exit(f"Could not load triangles from {path}")
    mesh.compute_vertex_normals()
    return mesh


def _colour_mesh(mesh: o3d.geometry.TriangleMesh, colour: tuple[float, float, float]) -> None:
    """Paint all vertices a uniform colour (R, G, B each in [0, 1])."""
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color(colour)


def _print_info(mesh: o3d.geometry.TriangleMesh, label: str = "Mesh") -> None:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(f"  {label}: {len(verts)} vertices, {len(faces)} faces")
    print(f"    AABB min: {verts.min(axis=0).round(4)}")
    print(f"    AABB max: {verts.max(axis=0).round(4)}")
    print(f"    Centre:   {verts.mean(axis=0).round(4)}")


def _make_wireframe(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


def _coordinate_frame(size: float = 0.15) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open3D SMPL mesh viewer")
    p.add_argument(
        "--mesh",
        type=Path,
        default=_DEFAULT_MESH,
        help="Path to the refined mesh (.obj or .ply)",
    )
    p.add_argument(
        "--compare-canonical",
        action="store_true",
        help="Load and show the canonical (pre-refinement) mesh side-by-side",
    )
    p.add_argument(
        "--canonical-mesh",
        type=Path,
        default=_DEFAULT_CANONICAL,
        help="Path to the canonical mesh (used with --compare-canonical)",
    )
    p.add_argument(
        "--wireframe",
        action="store_true",
        help="Show wireframe edges on top of the mesh",
    )
    p.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="If given, render to this PNG path instead of opening an interactive window",
    )
    p.add_argument(
        "--point-cloud",
        action="store_true",
        help="Also show a vertex point cloud coloured by normal direction",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load refined mesh
    # -----------------------------------------------------------------------
    print(f"\nLoading refined mesh: {args.mesh}")
    if not args.mesh.exists():
        sys.exit(f"File not found: {args.mesh}")

    refined = _load_mesh(args.mesh)
    # Blue-grey for refined mesh
    _colour_mesh(refined, (0.55, 0.70, 0.85))
    _print_info(refined, "Refined mesh")

    geometries: list = [refined, _coordinate_frame()]

    if args.wireframe:
        wf = _make_wireframe(refined)
        wf.paint_uniform_color((0.1, 0.1, 0.1))
        geometries.append(wf)

    # -----------------------------------------------------------------------
    # Optional: canonical mesh for comparison
    # -----------------------------------------------------------------------
    if args.compare_canonical:
        print(f"\nLoading canonical mesh: {args.canonical_mesh}")
        if not args.canonical_mesh.exists():
            print(f"  WARNING: canonical mesh not found at {args.canonical_mesh}")
        else:
            canonical = _load_mesh(args.canonical_mesh)
            # Shift it slightly to the side so both are visible
            shift = np.asarray(refined.vertices).max(axis=0)[0] - np.asarray(canonical.vertices).min(axis=0)[0] + 0.1
            canonical.translate([shift, 0.0, 0.0])
            _colour_mesh(canonical, (0.85, 0.65, 0.50))   # warm orange = original
            _print_info(canonical, "Canonical mesh")
            geometries.append(canonical)
            if args.wireframe:
                wf2 = _make_wireframe(canonical)
                wf2.paint_uniform_color((0.3, 0.1, 0.0))
                geometries.append(wf2)

    # -----------------------------------------------------------------------
    # Optional: vertex point cloud
    # -----------------------------------------------------------------------
    if args.point_cloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = refined.vertices
        # Colour by normal direction for visual interest
        normals = np.asarray(refined.vertex_normals)
        colours = (normals * 0.5 + 0.5).clip(0.0, 1.0)   # map [-1,1] → [0,1]
        pcd.colors = o3d.utility.Vector3dVector(colours)
        geometries.append(pcd)

    # -----------------------------------------------------------------------
    # Render / display
    # -----------------------------------------------------------------------
    if args.screenshot:
        print(f"\nRendering screenshot → {args.screenshot}")
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=960)
        for g in geometries:
            vis.add_geometry(g)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().light_on = True
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(args.screenshot), do_render=True)
        vis.destroy_window()
        print(f"  Saved: {args.screenshot}")
    else:
        print("\nOpening Open3D viewer  (press Q to quit, [ ] to toggle wireframe)")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="SMPL Refined Mesh",
            width=1280,
            height=960,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    main()
