import numpy as np

def box_iou_bev(boxes_a, boxes_b):
    # axis-aligned IoU (yaw ignored, simplified)
    ious = np.zeros((len(boxes_a), len(boxes_b)))

    for i,a in enumerate(boxes_a):
        for j,b in enumerate(boxes_b):
            xa1, ya1 = a[0]-a[3]/2, a[1]-a[4]/2
            xa2, ya2 = a[0]+a[3]/2, a[1]+a[4]/2
            xb1, yb1 = b[0]-b[3]/2, b[1]-b[4]/2
            xb2, yb2 = b[0]+b[3]/2, b[1]+b[4]/2

            inter = max(0, min(xa2, xb2)-max(xa1, xb1)) * \
                    max(0, min(ya2, yb2)-max(ya1, yb1))
            union = a[3]*a[4] + b[3]*b[4] - inter
            ious[i,j] = inter / (union + 1e-6)

    return ious


def assign_targets(anchors, gt_boxes, pos_iou=0.6, neg_iou=0.45):
    labels = np.full(len(anchors), -1)
    reg_targets = np.zeros((len(anchors), 7))
    dir_targets = np.zeros(len(anchors), dtype=np.int64)

    ious = box_iou_bev(anchors, gt_boxes)

    max_iou = ious.max(axis=1)
    gt_idx = ious.argmax(axis=1)

    labels[max_iou < neg_iou] = 0
    labels[max_iou >= pos_iou] = 1

    pos_mask = labels == 1
    assigned_gt = gt_boxes[gt_idx[pos_mask]]

    a = anchors[pos_mask]

    # regression targets
    reg_targets[pos_mask, 0] = (assigned_gt[:,0] - a[:,0]) / a[:,3]
    reg_targets[pos_mask, 1] = (assigned_gt[:,1] - a[:,1]) / a[:,4]
    reg_targets[pos_mask, 2] = (assigned_gt[:,2] - a[:,2]) / a[:,5]
    reg_targets[pos_mask, 3] = np.log(assigned_gt[:,3] / a[:,3])
    reg_targets[pos_mask, 4] = np.log(assigned_gt[:,4] / a[:,4])
    reg_targets[pos_mask, 5] = np.log(assigned_gt[:,5] / a[:,5])
    reg_targets[pos_mask, 6] = assigned_gt[:,6] - a[:,6]

    dir_targets[pos_mask] = (assigned_gt[:,6] > 0).astype(np.int64)

    return labels, reg_targets, dir_targets
