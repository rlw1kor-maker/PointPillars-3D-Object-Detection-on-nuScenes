import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.head = nn.Conv2d(128, 7 + num_classes, 1)

    def forward(self, x):
        return self.head(x)

class AnchorHead(nn.Module):
    def __init__(self, num_classes, num_anchors=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls_head = nn.Conv2d(128, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(128, num_anchors * 7, 1)
        self.dir_head = nn.Conv2d(128, num_anchors * 2, 1)  # 2-bin direction

    def forward(self, x):
        return {
            "cls": self.cls_head(x),
            "reg": self.reg_head(x),
            "dir": self.dir_head(x)
        }