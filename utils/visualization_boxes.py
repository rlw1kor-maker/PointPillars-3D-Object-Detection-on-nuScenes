
import open3d as o3d
import numpy as np
import math

def create_3d_box(center, size, yaw, color=[1,0,0]):
    cx, cy, cz = center
    l, w, h = size

    box = o3d.geometry.OrientedBoundingBox()
    box.center = [cx, cy, cz]
    box.extent = [l, w, h]

    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box.R = R
    box.color = color
    return box

def visualize_points_and_boxes(points, boxes):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    geometries = [pcd]
    for b in boxes:
        box = create_3d_box(
            center=b['center'],
            size=b['size'],
            yaw=b['yaw']
        )
        geometries.append(box)

    o3d.visualization.draw_geometries(geometries)
