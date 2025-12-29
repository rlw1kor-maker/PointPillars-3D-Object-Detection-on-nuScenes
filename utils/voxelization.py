import numpy as np
import torch

def create_pillars(points, cfg, device="cuda"):
    """
    Vectorized Pillarization for PointPillars
    Args:
        points: (N_pts, 4) numpy array, x,y,z,intensity
        cfg: voxelization config dict
        device: "cuda" or "cpu"
    Returns:
        pillars_tensor: (num_pillars, max_points, 7) torch tensor
        pillar_mask: (num_pillars, max_points) boolean tensor
        pillar_coords: (num_pillars, 2) BEV indices
    """
    X_MIN, X_MAX = cfg['x_min'], cfg['x_max']
    Y_MIN, Y_MAX = cfg['y_min'], cfg['y_max']
    Z_MIN, Z_MAX = cfg['z_min'], cfg['z_max']
    VX, VY = cfg['voxel_x'], cfg['voxel_y']
    MAX_PTS = cfg['max_points_per_pillar']
    MAX_PILLARS = cfg['max_pillars']

    # 1. Filter points in ROI
    mask = (
        (points[:,0] > X_MIN) & (points[:,0] < X_MAX) &
        (points[:,1] > Y_MIN) & (points[:,1] < Y_MAX) &
        (points[:,2] > Z_MIN) & (points[:,2] < Z_MAX)
    )
    points = points[mask]

    # 2. Compute voxel indices
    x_idx = ((points[:,0] - X_MIN) / VX).astype(np.int32)
    y_idx = ((points[:,1] - Y_MIN) / VY).astype(np.int32)

    coords = np.stack([x_idx, y_idx], axis=1)  # (N_pts,2)
    unique_coords, inv_idx = np.unique(coords, axis=0, return_inverse=True)

    num_pillars = min(len(unique_coords), MAX_PILLARS)
    pillars_tensor = torch.zeros((num_pillars, MAX_PTS, 7), dtype=torch.float32, device=device)
    pillar_mask = torch.zeros((num_pillars, MAX_PTS), dtype=torch.bool, device=device)
    pillar_coords = torch.zeros((num_pillars, 2), dtype=torch.int32, device=device)

    # 3. Assign points to pillars
    for i in range(num_pillars):
        idxs = np.where(inv_idx == i)[0]
        pts = points[idxs]
        if len(pts) > MAX_PTS:
            pts = pts[:MAX_PTS]
        P = pts.shape[0]

        # per-point features: x,y,z,i + x-mean,y-mean,z-mean
        mean_xyz = pts[:, :3].mean(axis=0)
        f = np.concatenate([pts[:, :4], pts[:, :3] - mean_xyz], axis=1)  # (P,7)

        pillars_tensor[i, :P] = torch.from_numpy(f).to(device)
        pillar_mask[i, :P] = True
        pillar_coords[i] = torch.from_numpy(unique_coords[i]).to(device)

    return pillars_tensor, pillar_mask, pillar_coords
