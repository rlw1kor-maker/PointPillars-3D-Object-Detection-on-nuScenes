import torch
import torch.nn.functional as F

def pointpillars_loss(preds, targets):
    cls_preds = preds["cls"]
    reg_preds = preds["reg"]
    dir_preds = preds["dir"]

    cls_labels, reg_targets, dir_targets = targets

    cls_labels = torch.tensor(cls_labels).long().to(cls_preds.device)
    reg_targets = torch.tensor(reg_targets).float().to(reg_preds.device)
    dir_targets = torch.tensor(dir_targets).long().to(dir_preds.device)

    cls_loss = F.binary_cross_entropy_with_logits(
        cls_preds.view(-1),
        cls_labels.float(),
        reduction="mean"
    )

    pos_mask = cls_labels == 1
    reg_loss = F.smooth_l1_loss(
        reg_preds.view(-1,7)[pos_mask],
        reg_targets[pos_mask],
        reduction="mean"
    )

    dir_loss = F.cross_entropy(
        dir_preds.view(-1,2)[pos_mask],
        dir_targets[pos_mask]
    )

    return cls_loss + 2.0 * reg_loss + 0.2 * dir_loss
