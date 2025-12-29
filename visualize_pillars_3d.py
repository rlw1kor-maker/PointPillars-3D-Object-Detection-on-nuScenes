import numpy as np
import open3d as o3d
import torch

from datasets.nuscenes_dataset import NuScenesDataset
from utils.voxelization import create_pillars
from models.bev import scatter_pillars_to_bev

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = NuScenesDataset("nuscenes")
voxel_cfg = {
    "x_min": -50, "x_max": 50,
    "y_min": -50, "y_max": 50,
    "z_min": -5,  "z_max": 3,
    "voxel_x": 0.5, "voxel_y": 0.5,
    "max_points_per_pillar": 100,
    "max_pillars": 12000
}

# ----------------------------
# Helpers
# ----------------------------
def draw_pillar_embeddings(pillar_embeddings, pillar_coords, points_per_pillar, voxel_cfg, window_name="Pillars 3D"):
    """
    Visualize pillar embeddings as vertical 3D columns.
    - Height = embedding norm
    - Color = number of points
    """
    VX, VY = voxel_cfg["voxel_x"], voxel_cfg["voxel_y"]
    num_pillars = pillar_embeddings.shape[0]

    # Compute pillar center coordinates
    x = pillar_coords[:,0].cpu().numpy() * VX + voxel_cfg["x_min"] + VX/2
    y = pillar_coords[:,1].cpu().numpy() * VY + voxel_cfg["y_min"] + VY/2

    # Height = norm of embedding vector
    h = pillar_embeddings.norm(dim=1).cpu().numpy()

    # Color = normalized number of points per pillar
    num_pts = points_per_pillar.cpu().numpy()
    colors = np.stack([num_pts/num_pts.max(), np.zeros_like(num_pts), 1 - num_pts/num_pts.max()], axis=1)

    # Create vertical line segments for each pillar
    lines = []
    points_line = []
    for i in range(num_pillars):
        points_line.append([x[i], y[i], 0])
        points_line.append([x[i], y[i], h[i]])
        lines.append([2*i, 2*i+1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_line)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Color per line: pick color from pillar
    line_colors = np.repeat(colors, 2, axis=0)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    o3d.visualization.draw_geometries([line_set], window_name=window_name)

# ----------------------------
# Load one sample
# ----------------------------
idx = 15
points = dataset.get_lidar(idx)

# 1️⃣ Pillarization
pillars_tensor, pillar_mask, pillar_coords = create_pillars(points, voxel_cfg, device=DEVICE)

# Count points per pillar
points_per_pillar = pillar_mask.sum(dim=1)  # (num_pillars,)

# 2️⃣ Dummy embeddings (replace with PillarEncoder outputs if available)
C = 64
pillar_embeddings = torch.randn((pillar_coords.shape[0], C), device=DEVICE)

# 3️⃣ Visualize
draw_pillar_embeddings(pillar_embeddings, pillar_coords, points_per_pillar, voxel_cfg, window_name="Pillar Embeddings 3D")
