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
H = int((voxel_cfg["y_max"] - voxel_cfg["y_min"]) / voxel_cfg["voxel_y"])
W = int((voxel_cfg["x_max"] - voxel_cfg["x_min"]) / voxel_cfg["voxel_x"])

# ----------------------------
# Helpers
# ----------------------------
def draw_points(points, color=[0.5,0.5,0.5], window_name="Point Cloud"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def draw_pillars(pillar_coords, voxel_cfg, window_name="Pillars"):
    """Visualize pillar centers"""
    VX, VY = voxel_cfg["voxel_x"], voxel_cfg["voxel_y"]
    x = pillar_coords[:,0].cpu().numpy() * VX + voxel_cfg["x_min"] + VX/2
    y = pillar_coords[:,1].cpu().numpy() * VY + voxel_cfg["y_min"] + VY/2
    z = np.zeros_like(x)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([x, y, z], axis=1))
    pcd.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def draw_bev(bev, window_name="BEV"):
    """Visualize BEV feature map as points colored by intensity"""
    bev = bev.cpu().numpy()
    C, H, W = bev.shape
    # Sum over channels for simple intensity
    intensity = bev.sum(axis=0)
    ys, xs = np.nonzero(intensity > 0)
    zs = intensity[ys, xs] / intensity.max()

    points = np.stack([xs, ys, zs*2], axis=1)
    colors = np.stack([zs, np.zeros_like(zs), 1-zs], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

# ----------------------------
# Select a sample
# ----------------------------
idx = 15
points = dataset.get_lidar(idx)  # (N,4)

# 1️⃣ Raw LiDAR
draw_points(points, window_name="Raw LiDAR")

# 2️⃣ Filtered points (ROI)
mask = (
    (points[:,0] > voxel_cfg["x_min"]) & (points[:,0] < voxel_cfg["x_max"]) &
    (points[:,1] > voxel_cfg["y_min"]) & (points[:,1] < voxel_cfg["y_max"]) &
    (points[:,2] > voxel_cfg["z_min"]) & (points[:,2] < voxel_cfg["z_max"])
)
points_filtered = points[mask]
draw_points(points_filtered, color=[0,1,0], window_name="Filtered Points (ROI)")

# 3️⃣ Pillarization
pillars_tensor, pillar_mask, pillar_coords = create_pillars(points, voxel_cfg, device=DEVICE)
draw_pillars(pillar_coords, voxel_cfg, window_name="Pillar Centers")

# 4️⃣ Optional: BEV grid visualization
# Use zeros and sum over channels to visualize density
C, H, W = 64, int((voxel_cfg["y_max"]-voxel_cfg["y_min"])/voxel_cfg["voxel_y"]), int((voxel_cfg["x_max"]-voxel_cfg["x_min"])/voxel_cfg["voxel_x"])
pillar_embeddings = torch.randn((pillar_coords.shape[0], C), device=DEVICE)  # dummy embeddings
bev = scatter_pillars_to_bev(pillar_embeddings, pillar_coords, (H,W))
draw_bev(bev, window_name="BEV Grid")
