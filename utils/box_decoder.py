import numpy as np
import torch

def decode_boxes(preds, cfg, score_thresh=0.5, topk=50):
    """
    preds: (1, C, H, W)
    returns list of 3D boxes in world frame
    """
    preds = preds.squeeze(0)
    scores = torch.sigmoid(preds[-1])  # last channel = confidence

    H, W = scores.shape
    scores_flat = scores.view(-1)

    topk_scores, topk_idx = torch.topk(scores_flat, topk)
    boxes = []

    for score, idx in zip(topk_scores, topk_idx):
        if score < score_thresh:
            continue

        y = idx // W
        x = idx % W

        # Decode regression outputs
        dx, dy, w, l, h, yaw = preds[:6, y, x]

        # Convert BEV grid → meters
        x_world = cfg['x_min'] + (x + dx) * cfg['voxel_x']
        y_world = cfg['y_min'] + (y + dy) * cfg['voxel_y']
        z_world = -1.0  # fixed ground height (demo)

        boxes.append({
            "center": np.array([x_world, y_world, z_world]),
            "size": np.array([l.item(), w.item(), h.item()]),
            "yaw": yaw.item(),
            "score": score.item()
        })

    return boxes
