# PointPillars-3D-Object-Detection-on-nuScenes
This repository contains a from-scratch PyTorch implementation of the PointPillars 3D object detection pipeline using LiDAR point clouds from the nuScenes mini dataset.

Project Highlights

End-to-end PointPillars architecture implemented in PyTorch

LiDAR pillarization (voxelization in BEV) and feature encoding

Dense BEV convolutional backbone with anchor-based detection heads

GPU-optimized training and inference pipeline

Rich Open3D visualizations for debugging and interpretability

Built and tested on nuScenes v1.0 mini

# Repository Structure
.
├── configs/
│   └── pointpillars.yaml
│
├── datasets/
│   └── nuscenes_dataset.py
│
├── models/
│   ├── pointpillars.py
│   ├── pillar_encoder.py
│   └── losses.py
│
├── utils/
│   ├── voxelization.py
│   ├── bev.py
│   ├── anchors.py
│   └── target_assigner.py
│
├── visualization/
│   ├── visualize_pipeline.py
│   └── visualize_pillars_3d.py
│
├── train.py
├── inference.py
└── README.md

# Installation

git clone https://github.com/your-username/pointpillars-nuscenes.git
cd pointpillars-nuscenes


conda create -n pointpillars python=3.9
conda activate pointpillars
pip install torch numpy open3d nuscenes-devkit pyyaml

# Dataset Setup (nuScenes Mini)
Download nuScenes v1.0 mini from the official site

Set the dataset directory:
export NUSCENES_ROOT=/path/to/nuscenes

# Training
python train.py


# Inference & Visualization
python inference.py

Loads trained .pth checkpoint

Decodes anchors into 3D bounding boxes

Visualizes predictions using Open3D

# Example Visualizations

Raw LiDAR point clouds

Pillar centers and density

BEV feature grids

Predicted 3D bounding boxes

# Limitations

Uses simplified anchor configuration

Not tuned for benchmark-level mAP

# Future Work

3D Non-Max Suppression (NMS)

Camera–LiDAR mid-level fusion

CenterPoint-style anchor-free head

Multi-frame temporal fusion

Quantitative evaluation (mAP)

# Acknowledgements

Original PointPillars paper (Lang et al., CVPR 2019)

nuScenes dataset by Aptiv / Motional

Open3D for visualization