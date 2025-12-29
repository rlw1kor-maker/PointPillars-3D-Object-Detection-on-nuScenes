
import yaml
import numpy as np
from datasets.nuscenes_dataset import NuScenesDataset
from utils.visualization_boxes import visualize_points_and_boxes

# Load dataset
dataset = NuScenesDataset('nuscenes')
points = dataset.get_lidar(0)

# Dummy predicted boxes (for visualization demo)
boxes = [
    {'center': [10, 0, 0], 'size': [4.0, 1.8, 1.6], 'yaw': 0.3},
    {'center': [20, -5, 0], 'size': [4.2, 2.0, 1.7], 'yaw': -0.5}
]

visualize_points_and_boxes(points, boxes)
