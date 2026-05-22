from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def _mesh_stats(mesh: o3d.geometry.TriangleMesh, title: str) -> None:
    verts = len(mesh.vertices)
    tris = len(mesh.triangles)
    bb = mesh.get_axis_aligned_bounding_box()
    ext = bb.get_extent()
    print(f"[{title}] verts={verts:,} tris={tris:,} extent=({ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f})")


def _remove_small_components(mesh: o3d.geometry.TriangleMesh, min_tris: int) -> o3d.geometry.TriangleMesh:
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if counts.size == 0:
        return mesh

    keep_mask = np.zeros(len(labels), dtype=bool)
    for cid, c in enumerate(counts):
        if int(c) >= min_tris:
            keep_mask[labels == cid] = True

    mesh.remove_triangles_by_mask(~keep_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def _keep_n_largest_components(mesh: o3d.geometry.TriangleMesh, n: int) -> o3d.geometry.TriangleMesh:
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if counts.size == 0:
        return mesh

    order = np.argsort(-counts)
    keep_ids = set(order[: max(1, n)].tolist())
    keep_mask = np.array([l in keep_ids for l in labels], dtype=bool)

    mesh.remove_triangles_by_mask(~keep_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def _remove_bottom_by_quantile(mesh: o3d.geometry.TriangleMesh, q: float, axis: int) -> o3d.geometry.TriangleMesh:
    if q <= 0:
        return mesh
    v = np.asarray(mesh.vertices)
    cut = np.quantile(v[:, axis], q)
    mesh.remove_vertices_by_mask(v[:, axis] < cut)
    mesh.remove_unreferenced_vertices()
    return mesh


def _remove_downward_low_triangles(
    mesh: o3d.geometry.TriangleMesh,
    low_frac: float,
    normal_axis: int,
    normal_thresh: float,
    axis: int,
) -> o3d.geometry.TriangleMesh:
    if low_frac <= 0:
        return mesh

    mesh.compute_triangle_normals()
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    n = np.asarray(mesh.triangle_normals)

    axis_vals = v[:, axis]
    mn, mx = float(np.min(axis_vals)), float(np.max(axis_vals))
    h = max(mx - mn, 1e-9)

    c = v[t].mean(axis=1)
    low = c[:, axis] < (mn + low_frac * h)
    down = n[:, normal_axis] < normal_thresh
    skirt = low & down

    mesh.remove_triangles_by_mask(skirt)
    mesh.remove_unreferenced_vertices()
    return mesh


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refine mesh: components, skirt removal, smoothing, decimation")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p.add_argument("--axis", type=int, default=1, choices=[0, 1, 2], help="Up axis index used for bottom clipping")

    p.add_argument("--keep-largest", type=int, default=1, help="Keep N largest connected components")
    p.add_argument("--min-component-tris", type=int, default=0, help="Drop connected components below this triangle count")

    p.add_argument("--bottom-quantile", type=float, default=0.0, help="Remove vertices below axis quantile [0..0.5]")
    p.add_argument("--remove-skirt-frac", type=float, default=0.0, help="Remove downward triangles in lowest fraction of height")
    p.add_argument("--skirt-normal-thresh", type=float, default=0.35, help="Triangle normal threshold along +axis")

    p.add_argument("--smooth-iters", type=int, default=0)
    p.add_argument("--target-tris", type=int, default=0, help="Quadric decimation target triangles (0=skip)")

    p.add_argument("--print-components", action="store_true", help="Print connected component sizes")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input mesh not found: {args.input}")

    mesh = o3d.io.read_triangle_mesh(str(args.input))
    if mesh.is_empty():
        raise RuntimeError("Loaded mesh is empty")

    _mesh_stats(mesh, "input")

    if args.print_components:
        labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        top = np.sort(counts)[::-1][:20]
        print(f"[components] total={len(counts)} top20={top.tolist()}")

    if args.bottom_quantile > 0:
        mesh = _remove_bottom_by_quantile(mesh, args.bottom_quantile, axis=args.axis)
        _mesh_stats(mesh, "after_bottom_clip")

    if args.remove_skirt_frac > 0:
        mesh = _remove_downward_low_triangles(
            mesh,
            low_frac=args.remove_skirt_frac,
            normal_axis=args.axis,
            normal_thresh=args.skirt_normal_thresh,
            axis=args.axis,
        )
        _mesh_stats(mesh, "after_skirt_removal")

    if args.min_component_tris > 0:
        mesh = _remove_small_components(mesh, args.min_component_tris)
        _mesh_stats(mesh, "after_min_component")

    if args.keep_largest > 0:
        mesh = _keep_n_largest_components(mesh, args.keep_largest)
        _mesh_stats(mesh, "after_keep_largest")

    if args.smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iters)
        _mesh_stats(mesh, "after_smooth")

    if args.target_tris > 0 and len(mesh.triangles) > args.target_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_tris)
        _mesh_stats(mesh, "after_decimate")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(args.output), mesh, write_vertex_colors=True)
    if not ok:
        raise RuntimeError("Failed to write output mesh")

    _mesh_stats(mesh, "output")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
