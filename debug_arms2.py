import sys
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

ROOT = Path(r'C:/V23D/V23D')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflows.debug_visualization.export_front_back_part_images import (
    HybridPartDetector, PART_LARM, PART_RARM, PART_TORSO, _smpl_vertex_parts
)

FRONT_REF_IMG = Path(r'E:/zero123_dataset/humans_train/person_017/frame_000/reference.png')
YOLO_MODEL_PATH = Path(r'C:/V23D/V23D/yolov8x-pose.pt')
OBJ_PATH = Path(r'E:/smpl_textured_from_splat.obj')
SMPL_PKL_PATH = Path(r'E:/SMPL_extracted/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')
PLY_FALLBACK_PATH = Path(r'E:/V23D_Data/per_part_splat_refetch1/smpl_textured_from_splat.ply')

# Read reference image
front_img = cv2.imread(str(FRONT_REF_IMG), cv2.IMREAD_UNCHANGED)
width = front_img.shape[1]
height = front_img.shape[0]

# Get mask
if front_img.shape[2] == 4:
    front_mask = (front_img[:, :, 3] > 10).astype(np.uint8) * 255
else:
    front_mask = np.ones((height, width), dtype=np.uint8) * 255

# Load SMPL parts
base_parts = _smpl_vertex_parts(SMPL_PKL_PATH)
base_mesh = o3d.io.read_triangle_mesh(str(PLY_FALLBACK_PATH), enable_post_processing=True)
base_verts = np.asarray(base_mesh.vertices)

mesh = o3d.io.read_triangle_mesh(str(OBJ_PATH), enable_post_processing=True)
verts = np.asarray(mesh.vertices)

from scipy.spatial import cKDTree
tree = cKDTree(base_verts)
dist, nn = tree.query(verts, k=1)
part_ids = base_parts[np.asarray(nn, dtype=np.int32)]

# Get z-median for torso
z = verts[:, 2]
torso_idx = np.where(part_ids == PART_TORSO)[0]
torso_z = np.median(z[torso_idx]) if len(torso_idx) else np.median(z)

# Project mesh
def _project(verts, width, height, mirrored=False):
    v = verts.copy()
    if mirrored:
        v[:, 0] *= -1.0
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    sx = (width - 40) / max(float(x.max() - x.min()), 1e-6)
    sy = (height - 40) / max(float(y.max() - y.min()), 1e-6)
    s = min(sx, sy)
    px = ((x - (x.min() + x.max()) * 0.5) * s + width * 0.5).astype(np.int32)
    py = ((-(y - (y.min() + y.max()) * 0.5)) * s + height * 0.5).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height) & np.isfinite(z)
    return px, py, inside

px, py, inside = _project(verts, width, height, mirrored=False)

# Get arm indices
larm_idx = np.where((part_ids == PART_LARM) & (z >= torso_z))[0]
rarm_idx = np.where((part_ids == PART_RARM) & (z >= torso_z))[0]

print(f'Left arm verts: {len(larm_idx)}, Right arm verts: {len(rarm_idx)}')

if len(larm_idx) > 0:
    print(f'Left arm projected px range: [{px[larm_idx].min()}, {px[larm_idx].max()}]')
    print(f'Left arm projected py range: [{py[larm_idx].min()}, {py[larm_idx].max()}]')

if len(rarm_idx) > 0:
    print(f'Right arm projected px range: [{px[rarm_idx].min()}, {px[rarm_idx].max()}]')
    print(f'Right arm projected py range: [{py[rarm_idx].min()}, {py[rarm_idx].max()}]')

# Detect arm regions in reference
hybrid = HybridPartDetector(yolo_model=str(YOLO_MODEL_PATH))
front_parts = hybrid.part_masks(front_img[:, :, :3], front_mask)
larm_mask = front_parts[PART_LARM]
rarm_mask = front_parts[PART_RARM]

arm_pixels = np.where((larm_mask > 0) | (rarm_mask > 0))
print(f'\nDetected arm pixels in reference:')
print(f'Arm pixel Y range: [{arm_pixels[0].min()}, {arm_pixels[0].max()}]')
print(f'Arm pixel X range: [{arm_pixels[1].min()}, {arm_pixels[1].max()}]')

print(f'\nMESH ARMS PROJECT TO: Y~{py[larm_idx].mean() if len(larm_idx) > 0 else "?"}, X~{px[larm_idx].mean() if len(larm_idx) > 0 else "?"}')
print(f'REFERENCE ARMS ARE AT: Y~{arm_pixels[0].mean()}, X~{arm_pixels[1].mean()}')
print('\nCONCLUSION: Arms do NOT overlap! Need to use detected arm pixels as source directly.')
