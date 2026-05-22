"""
Side-by-side Open3D viewer for two 3DGS point clouds.
  Window 1 -> 297-view model (colmap_rerun)
  Window 2 -> 87-view model  (colmap_rerun_v3)
Press Q or Escape to close each window.
"""
import numpy as np
import open3d as o3d
from plyfile import PlyData

PLY_297 = r"E:\V23D_Data\3dgs_model_rerun\point_cloud\iteration_30000\point_cloud.ply"
PLY_87  = r"E:\V23D_Data\3dgs_model_rerun_v3\point_cloud\iteration_30000\point_cloud.ply"

def load(path, label):
    """Load a 3DGS PLY and recover colors from SH DC coefficients."""
    print(f"Loading {label} …", flush=True)
    ply = PlyData.read(path)
    v = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)

    # SH DC -> linear RGB:  C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    r = np.clip(v['f_dc_0'] / C0 * 0.5 + 0.5, 0, 1)
    g = np.clip(v['f_dc_1'] / C0 * 0.5 + 0.5, 0, 1)
    b = np.clip(v['f_dc_2'] / C0 * 0.5 + 0.5, 0, 1)
    colors = np.stack([r, g, b], axis=1)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colors)
    print(f"  {label}: {len(pc.points):,} points", flush=True)
    return pc

pc297 = load(PLY_297, "297-view")
pc87  = load(PLY_87,  "87-view")

# --- 297-view window ---
vis1 = o3d.visualization.Visualizer()
vis1.create_window(window_name="3DGS – 297 registered views (colmap_rerun)", width=1280, height=800, left=0, top=0)
vis1.add_geometry(pc297)
opt1 = vis1.get_render_option()
opt1.point_size = 1.5
opt1.background_color = [0.1, 0.1, 0.1]
vis1.run()
vis1.destroy_window()

# --- 87-view window ---
vis2 = o3d.visualization.Visualizer()
vis2.create_window(window_name="3DGS – 87 registered views (colmap_rerun_v3)", width=1280, height=800, left=0, top=0)
vis2.add_geometry(pc87)
opt2 = vis2.get_render_option()
opt2.point_size = 1.5
opt2.background_color = [0.1, 0.1, 0.1]
vis2.run()
vis2.destroy_window()

print("Done.")
