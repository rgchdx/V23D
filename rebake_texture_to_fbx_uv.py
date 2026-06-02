from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def _uv_to_px(uv: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    x = uv[..., 0] * (w - 1)
    y = (1.0 - uv[..., 1]) * (h - 1)
    return x, y


def _sample_bilinear_rgb(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = _uv_to_px(uv, w, h)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]

    c00 = img[y0, x0].astype(np.float32)
    c10 = img[y0, x1].astype(np.float32)
    c01 = img[y1, x0].astype(np.float32)
    c11 = img[y1, x1].astype(np.float32)

    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    c = c0 * (1.0 - wy) + c1 * wy
    return np.clip(c, 0, 255).astype(np.uint8)


def _barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # p: (N,2), a/b/c: (2,)
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(den) < 1e-12:
        n = p.shape[0]
        z = np.zeros((n,), dtype=np.float32)
        return z, z, z

    w0 = ((b[1] - c[1]) * (p[:, 0] - c[0]) + (c[0] - b[0]) * (p[:, 1] - c[1])) / den
    w1 = ((c[1] - a[1]) * (p[:, 0] - c[0]) + (a[0] - c[0]) * (p[:, 1] - c[1])) / den
    w2 = 1.0 - w0 - w1
    return w0.astype(np.float32), w1.astype(np.float32), w2.astype(np.float32)


def rebake_texture(
    source_obj: Path,
    source_texture: Path,
    target_fbx: Path,
    output_texture: Path,
    tex_size: int,
) -> None:
    src_mesh = o3d.io.read_triangle_mesh(str(source_obj), enable_post_processing=True)
    tgt_mesh = o3d.io.read_triangle_mesh(str(target_fbx), enable_post_processing=True)

    if src_mesh.is_empty() or tgt_mesh.is_empty():
        raise RuntimeError("Failed to load source OBJ or target FBX mesh.")

    src_tri_uv = np.asarray(src_mesh.triangle_uvs, dtype=np.float32).reshape(-1, 3, 2)
    tgt_tri_uv = np.asarray(tgt_mesh.triangle_uvs, dtype=np.float32).reshape(-1, 3, 2)

    if src_tri_uv.shape[0] == 0 or tgt_tri_uv.shape[0] == 0:
        raise RuntimeError("Source OBJ and target FBX must both have UVs.")

    tri_count = min(src_tri_uv.shape[0], tgt_tri_uv.shape[0])
    if src_tri_uv.shape[0] != tgt_tri_uv.shape[0]:
        print(f"[warn] triangle count mismatch in UV buffers: src={src_tri_uv.shape[0]} tgt={tgt_tri_uv.shape[0]}. Using min={tri_count}.")

    src_tex = cv2.imread(str(source_texture), cv2.IMREAD_COLOR)
    if src_tex is None:
        raise RuntimeError(f"Failed to read source texture: {source_texture}")

    out = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    covered = np.zeros((tex_size, tex_size), dtype=np.uint8)

    for i in range(tri_count):
        tu = tgt_tri_uv[i]
        su = src_tri_uv[i]

        tx, ty = _uv_to_px(tu, tex_size, tex_size)
        tri_px = np.stack([tx, ty], axis=1)

        minx = max(int(np.floor(np.min(tri_px[:, 0]))), 0)
        maxx = min(int(np.ceil(np.max(tri_px[:, 0]))), tex_size - 1)
        miny = max(int(np.floor(np.min(tri_px[:, 1]))), 0)
        maxy = min(int(np.ceil(np.max(tri_px[:, 1]))), tex_size - 1)

        if maxx < minx or maxy < miny:
            continue

        xs = np.arange(minx, maxx + 1, dtype=np.float32)
        ys = np.arange(miny, maxy + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel() + 0.5, gy.ravel() + 0.5], axis=1)

        w0, w1, w2 = _barycentric(pts, tri_px[0], tri_px[1], tri_px[2])
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue

        pts_in = pts[inside]
        px = np.clip(np.floor(pts_in[:, 0]).astype(np.int32), 0, tex_size - 1)
        py = np.clip(np.floor(pts_in[:, 1]).astype(np.int32), 0, tex_size - 1)

        wi0 = w0[inside][:, None]
        wi1 = w1[inside][:, None]
        wi2 = w2[inside][:, None]
        src_uv = wi0 * su[0][None, :] + wi1 * su[1][None, :] + wi2 * su[2][None, :]

        col = _sample_bilinear_rgb(src_tex, src_uv)
        out[py, px] = col
        covered[py, px] = 255

    # fill tiny UV raster gaps
    inv = (covered == 0).astype(np.uint8) * 255
    if np.any(inv):
        out = cv2.inpaint(out, inv, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    output_texture.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_texture), out)

    coverage = float(np.mean(covered > 0))
    print(f"[ok] Wrote rebaked texture: {output_texture}")
    print(f"[ok] UV coverage before inpaint: {coverage * 100:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebake existing SMPL texture from source OBJ UV layout to target FBX UV layout.")
    ap.add_argument("--source_obj", type=Path, default=Path(r"E:/unity_vertex_match_output/our_pipeline_vertex_match.obj"))
    ap.add_argument("--source_texture", type=Path, default=Path(r"E:/unity_vertex_match_output/our_pipeline_vertex_match_albedo.png"))
    ap.add_argument("--target_fbx", type=Path, default=Path(r"\\students\student-n-r\rgdix\Downloads\SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"))
    ap.add_argument("--output_texture", type=Path, default=Path(r"E:/unity_vertex_match_output/SMPL_m_unityDoubleBlends_rebaked_albedo.png"))
    ap.add_argument("--tex_size", type=int, default=4096)
    args = ap.parse_args()

    rebake_texture(
        source_obj=args.source_obj,
        source_texture=args.source_texture,
        target_fbx=args.target_fbx,
        output_texture=args.output_texture,
        tex_size=args.tex_size,
    )
