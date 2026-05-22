import numpy as np
from pathlib import Path
from PIL import Image
import os

def read_images_txt(path):
    cameras = []
    lines = Path(path).read_text().splitlines()
    lines = [l for l in lines if not l.startswith('#') and l.strip()]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw,qx,qy,qz = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
        tx,ty,tz = float(parts[5]),float(parts[6]),float(parts[7])
        cameras.append((parts[9], qw,qx,qy,qz, tx,ty,tz))
    return cameras

def quat_to_R(qw,qx,qy,qz):
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

cams = read_images_txt(r'E:\V23D_Data\colmap_rerun\sparse\1\images.txt')
centers = []
forward_dirs = []
for name,qw,qx,qy,qz,tx,ty,tz in cams:
    R = quat_to_R(qw,qx,qy,qz)
    C = -R.T @ np.array([tx,ty,tz])
    centers.append(C)
    # Camera forward direction (Z column of R^T, i.e. third row of R)
    forward_dirs.append(R[2])  # world-space forward = R^T[:,2] = R[2] row
centers = np.array(centers)
forward_dirs = np.array(forward_dirs)
centroid = centers.mean(0)

print(f"Camera centroid: {centroid}")
print(f"Scene extent: X={centers[:,0].min():.3f}..{centers[:,0].max():.3f}  Y={centers[:,1].min():.3f}..{centers[:,1].max():.3f}  Z={centers[:,2].min():.3f}..{centers[:,2].max():.3f}")

# Angular coverage check
rel = centers - centroid
angles_xz = np.arctan2(rel[:,2], rel[:,0]) * 180/np.pi
print(f"\nAngular coverage (top-down XZ): {angles_xz.min():.1f} to {angles_xz.max():.1f} deg")
gaps = np.diff(np.sort(angles_xz))
print(f"Largest gap in orbit: {gaps.max():.1f} deg at angle {np.sort(angles_xz)[np.argmax(gaps)]:.1f}")

# Are cameras pointing at centroid?
to_centroid = centroid - centers
to_centroid_n = to_centroid / np.linalg.norm(to_centroid, axis=1, keepdims=True)
dots = (forward_dirs * to_centroid_n).sum(1)
print(f"\nCamera pointing toward centroid (dot product, 1=perfect):")
print(f"  mean={dots.mean():.3f} std={dots.std():.3f} min={dots.min():.3f}")

# Print camera distances from centroid 
dists = np.linalg.norm(rel, axis=1)
print(f"\nCamera distances from centroid: mean={dists.mean():.3f} min={dists.min():.3f} max={dists.max():.3f}")

# Check cameras.txt for focal length
cam_txt = Path(r'E:\V23D_Data\colmap_rerun\sparse\1\cameras.txt').read_text()
print(f"\ncameras.txt:\n{cam_txt[:500]}")

# SVG orbit plot (top-down)
lines_svg = ['<svg width="600" height="620" xmlns="http://www.w3.org/2000/svg">']
lines_svg.append('<rect width="600" height="620" fill="#111"/>')
xz = centers[:,[0,2]]
mn,mx = xz.min(), xz.max()
def sc(v): return int((v-mn)/(mx-mn)*560+20)
for i,(x,z) in enumerate(xz):
    color = f'hsl({int(i/len(xz)*360)},100%,60%)'
    lines_svg.append(f'<circle cx="{sc(x)}" cy="{sc(z)}" r="3" fill="{color}"/>')
cx_s = sc(centroid[0]); cz_s = sc(centroid[2])
lines_svg.append(f'<circle cx="{cx_s}" cy="{cz_s}" r="8" fill="white" opacity="0.8"/>')
lines_svg.append(f'<text x="{cx_s+10}" y="{cz_s}" fill="white" font-size="11">centroid</text>')
lines_svg.append('<text x="10" y="595" fill="white" font-size="12">Top-down XZ  |  color = frame order (red=early, cyan=mid, purple=late)</text>')
lines_svg.append('<text x="10" y="612" fill="white" font-size="12">White dot = scene centroid</text>')
lines_svg.append('</svg>')
Path(r'E:\V23D_Data\debug_cameras_xz.svg').write_text('\n'.join(lines_svg))
print("\nSaved E:\\V23D_Data\\debug_cameras_xz.svg")

# Sample frame grid
frames_dir = r'E:\V23D_Data\frames'
frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
sample_idx = [0, 37, 74, 111, 148, 185, 222, 259, 296]
imgs = [Image.open(os.path.join(frames_dir, frames[i])).resize((160,160)) for i in sample_idx]
grid = Image.new('RGB', (len(imgs)*160, 180), (30,30,30))
from PIL import ImageDraw
draw = ImageDraw.Draw(grid)
for i,im in enumerate(imgs):
    grid.paste(im, (i*160, 0))
    draw.text((i*160+2, 162), frames[sample_idx[i]], fill=(200,200,200))
grid.save(r'E:\V23D_Data\debug_frames_sample.jpg')
print("Saved E:\\V23D_Data\\debug_frames_sample.jpg")
