import torch
import yaml
import numpy as np
import open3d as o3d

from datasets.nuscenes_dataset import NuScenesDataset
from models.pointpillars import PointPillars
from utils.voxelization import create_pillars
from models.bev import scatter_pillars_to_bev
from utils.anchors import generate_anchors

# ----------------------------
# Config
# ----------------------------
CFG_PATH = "configs/pointpillars.yaml"
CKPT_PATH = "checkpoints/pointpillars_epoch_3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load config
# ----------------------------
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

voxel_cfg = cfg["voxel"]
num_classes = cfg["model"]["num_classes"]

# BEV size
H = int((voxel_cfg["y_max"] - voxel_cfg["y_min"]) / voxel_cfg["voxel_y"])
W = int((voxel_cfg["x_max"] - voxel_cfg["x_min"]) / voxel_cfg["voxel_x"])

# ----------------------------
# Dataset
# ----------------------------
dataset = NuScenesDataset("nuscenes")

# ----------------------------
# Model
# ----------------------------
model = PointPillars(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

anchors = generate_anchors(voxel_cfg)  # (N,7)
anchors = torch.from_numpy(anchors).float().to(DEVICE)

# ----------------------------
# Decode boxes
# ----------------------------
def decode_boxes(anchors, deltas):
    """
    anchors: (N,7)
    deltas:  (N,7)
    """
    xa, ya, za, wa, la, ha, ra = anchors.T
    dx, dy, dz, dw, dl, dh, dr = deltas.T

    x = dx * wa + xa
    y = dy * la + ya
    z = dz * ha + za

    w = torch.exp(dw) * wa
    l = torch.exp(dl) * la
    h = torch.exp(dh) * ha
    r = ra + dr

    return torch.stack([x, y, z, w, l, h, r], dim=1)

# ----------------------------
# Open3D box helper
# ----------------------------
def create_o3d_box(box, color=[1, 0, 0]):
    x, y, z, w, l, h, yaw = box
    box3d = o3d.geometry.OrientedBoundingBox()
    box3d.center = [x, y, z]
    box3d.extent = [w, l, h]
    R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw])
    box3d.R = R
    box3d.color = color
    return box3d

# ----------------------------
# Inference on one sample
# ----------------------------
idx = 11
points = dataset.get_lidar(idx)  # (N,4)

pillars, mask, coords = create_pillars(points, voxel_cfg, device=DEVICE)

with torch.no_grad():
    preds = model(
        pillars,     # (Np, P, 7)
        mask,        # (Np, P)
        coords,      # (Np, 2)
        (H, W)
    )

cls_preds = preds["cls"].squeeze(0)  # (A, H, W)
reg_preds = preds["reg"].squeeze(0)  # (A*7, H, W)

A, H, W = cls_preds.shape

# Flatten cls
cls_preds = cls_preds.permute(1, 2, 0).contiguous().view(-1)
cls_scores = cls_preds.sigmoid()

# Flatten reg
reg_preds = reg_preds.permute(1, 2, 0).contiguous().view(-1, 7)

score_thresh = 0.5
keep = cls_preds > score_thresh


print("anchors:", anchors.shape)
print("cls_preds:", cls_preds.shape)
print("reg_preds:", reg_preds.shape)
print("keep:", keep.shape)


boxes = decode_boxes(
    anchors[keep],
    reg_preds[keep]
)
scores = cls_preds[keep]

print(f"Detected boxes: {boxes.shape[0]}")

# ----------------------------
# Visualization
# ----------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.paint_uniform_color([0.5, 0.5, 0.5])

boxes_o3d = []
for box in boxes.cpu().numpy():
    boxes_o3d.append(create_o3d_box(box))

o3d.visualization.draw_geometries([pcd, *boxes_o3d])
