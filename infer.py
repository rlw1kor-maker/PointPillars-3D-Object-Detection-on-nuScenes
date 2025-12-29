import yaml
import torch
import numpy as np
from datasets.nuscenes_dataset import NuScenesDataset
from utils.voxelization import create_pillars
from models.pointpillars import PointPillars
from models.pillar_encoder import PillarEncoder
from utils.box_decoder import decode_boxes
from utils.visualization import visualize

cfg = yaml.safe_load(open('configs/pointpillars.yaml'))

dataset = NuScenesDataset('nuscenes')
points = dataset.get_lidar(1)

pillar_feats, pillar_coords = create_pillars(points, cfg['voxel'])

H = int((cfg['voxel']['y_max'] - cfg['voxel']['y_min']) / cfg['voxel']['voxel_y'])
W = int((cfg['voxel']['x_max'] - cfg['voxel']['x_min']) / cfg['voxel']['voxel_x'])

# ---- pillar encoder ----
pillar_encoder = PillarEncoder(in_channels=7, out_channels=64)
pillar_encoder.eval()

pillar_feats_torch = torch.tensor(pillar_feats, dtype=torch.float32)

with torch.no_grad():
    encoded_feats = pillar_encoder(pillar_feats_torch)  # (P, 64)

encoded_feats = encoded_feats.numpy()

# ---- scatter to BEV ----
bev = np.zeros((64, H, W), dtype=np.float32)

for feat, (x, y) in zip(encoded_feats, pillar_coords):
    if x < W and y < H:
        bev[:, y, x] = feat

bev = torch.tensor(bev).unsqueeze(0)

model = PointPillars(num_classes=1)
model.eval()

with torch.no_grad():
    preds = model(bev)

# ---- decode boxes ----
boxes = decode_boxes(preds, cfg['voxel'], score_thresh=0.6)

print(f"Detected {len(boxes)} boxes")

# ---- visualize ----
visualize(points, boxes)
