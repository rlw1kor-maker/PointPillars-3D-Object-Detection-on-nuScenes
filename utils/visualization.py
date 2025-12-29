import open3d as o3d
import numpy as np

def create_o3d_boxes(boxes):
    o3d_boxes = []

    for box in boxes:
        center = box["center"]
        size = box["size"]
        yaw = box["yaw"]

        R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw])
        obb = o3d.geometry.OrientedBoundingBox(center, R, size)
        obb.color = (1, 0, 0)  # red boxes

        o3d_boxes.append(obb)

    return o3d_boxes


def visualize(points, boxes):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    o3d_boxes = create_o3d_boxes(boxes)

    o3d.visualization.draw_geometries([pcd, *o3d_boxes])
