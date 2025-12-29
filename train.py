import os
import yaml
import torch
import torch.optim as optim
import numpy as np

from datasets.nuscenes_dataset import NuScenesDataset
from utils.voxelization import create_pillars
from utils.anchors import generate_anchors
from utils.target_assigner import assign_targets
from models.pointpillars import PointPillars
from models.losses import pointpillars_loss
from models.pillar_encoder import BatchedPillarEncoder

# ----------------------------
# Config
# ----------------------------
CFG_PATH = "configs/pointpillars.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LR = 1e-3
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Load config
# ----------------------------
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

voxel_cfg = cfg["voxel"]
num_classes = cfg["model"]["num_classes"]

# ----------------------------
# Dataset
# ----------------------------
dataset = NuScenesDataset("nuscenes")

# ----------------------------
# Model
# ----------------------------
model = PointPillars(num_classes=num_classes).to(DEVICE)
optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# ----------------------------
# Anchors (fixed for dataset)
# ----------------------------
anchors = generate_anchors(voxel_cfg)  # (N, 7)

# BEV size
H = int((voxel_cfg["y_max"] - voxel_cfg["y_min"]) / voxel_cfg["voxel_y"])
W = int((voxel_cfg["x_max"] - voxel_cfg["x_min"]) / voxel_cfg["voxel_x"])

print(f"Training on {len(dataset)} samples")
print(f"BEV size: {H} x {W}")
print(f"Total anchors: {anchors.shape[0]}")

# ----------------------------
# Dummy GT loader (demo)
# ----------------------------
def load_gt_boxes_from_nuscenes(nusc, sample):
    """
    Returns GT boxes in format:
    [x, y, z, w, l, h, yaw]
    Car class only (demo)
    """
    gt_boxes = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if ann["category_name"].startswith("vehicle"):
            x, y, z = ann["translation"]
            w, l, h = ann["size"]
            yaw = ann["rotation"][-1]  # simplified
            gt_boxes.append([x, y, z, w, l, h, yaw])

    if len(gt_boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    return np.array(gt_boxes, dtype=np.float32)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for idx in range(len(dataset)):
        # ----------------------------
        # Load LiDAR
        # ----------------------------
        points = dataset.get_lidar(idx)

        # ----------------------------
        # Pillarization
        # ----------------------------
        pillars_tensor, pillar_mask, pillar_coords = create_pillars(points, voxel_cfg)
        
        # ----------------------------
        # Forward (ONLY ONCE)
        # ----------------------------
        preds = model(
            pillars_tensor,
            pillar_mask,
            pillar_coords,
            (H, W)
        )

        # ----------------------------
        # Targets → Torch
        # ----------------------------
        # ---------------------------- # Load GT boxes # ---------------------------- 
        sample = dataset.samples[idx]
        gt_boxes = load_gt_boxes_from_nuscenes(dataset.nusc, sample)
        # ---------------------------- # Target assignment # ---------------------------- 
        labels, reg_targets, dir_targets = assign_targets( anchors, gt_boxes )

        labels = torch.from_numpy(labels).to(DEVICE)
        reg_targets = torch.from_numpy(reg_targets).to(DEVICE)
        dir_targets = torch.from_numpy(dir_targets).to(DEVICE)

        loss = pointpillars_loss(preds, (labels, reg_targets, dir_targets))

        # ----------------------------
        # Backprop
        # ----------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Step [{idx}/{len(dataset)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataset)
    print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}")

    # ----------------------------
    # Save checkpoint
    # ----------------------------
    ckpt_path = os.path.join(SAVE_DIR, f"pointpillars_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

print("Training complete.")
