from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main():
    ap = argparse.ArgumentParser(description="View an OBJ/PLY mesh in Open3D.")
    ap.add_argument(
        "mesh",
        nargs="?",
        default=r"E:\V23D_Data\smpl_textured\smpl_textured.obj",
        help="Mesh path. Defaults to the baked SMPL OBJ.",
    )
    args = ap.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), True)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Failed to load mesh: {mesh_path}")

    mesh.compute_vertex_normals()
    print(f"mesh: {mesh_path}")
    print(f"vertices: {len(mesh.vertices)}")
    print(f"triangles: {len(mesh.triangles)}")
    print(f"has_triangle_uvs: {mesh.has_triangle_uvs()}")
    print(f"textures: {len(mesh.textures)}")

    try:
        o3d.visualization.draw(
            [{"name": mesh_path.name, "geometry": mesh}],
            show_ui=True,
        )
    except Exception:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=mesh_path.name,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    main()
