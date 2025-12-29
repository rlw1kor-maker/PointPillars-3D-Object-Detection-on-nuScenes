import numpy as np

def generate_anchors(cfg):
    x_min, x_max = cfg['x_min'], cfg['x_max']
    y_min, y_max = cfg['y_min'], cfg['y_max']
    vx, vy = cfg['voxel_x'], cfg['voxel_y']

    W = int((x_max - x_min) / vx)
    H = int((y_max - y_min) / vy)

    anchor_sizes = [
        [1.6, 3.9, 1.5],  # car (w,l,h)
    ]
    anchor_yaws = [0, np.pi / 2]

    anchors = []

    for y in range(H):
        for x in range(W):
            cx = x_min + (x + 0.5) * vx
            cy = y_min + (y + 0.5) * vy

            for size in anchor_sizes:
                for yaw in anchor_yaws:
                    anchors.append([
                        cx, cy, 0.0,
                        size[0], size[1], size[2],
                        yaw
                    ])

    return np.array(anchors)  # (N,7)
