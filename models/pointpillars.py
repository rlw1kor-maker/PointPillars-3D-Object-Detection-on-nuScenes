import torch.nn as nn
from .pillar_encoder import BatchedPillarEncoder
from .bev_backbone import BEVBackbone
from .detection_head import AnchorHead

class PointPillars(nn.Module):
    def __init__(self, num_classes, num_anchors=2):
        super().__init__()

        self.pillar_encoder = BatchedPillarEncoder()
        self.backbone = BEVBackbone()
        self.head = AnchorHead(num_classes, num_anchors)

    def forward(self, pillar_features, mask, coords, bev_shape):
        """
        pillar_features: (N, P, 7)
        mask: (N, P)
        coords: (N, 2) pillar (x, y)
        bev_shape: (H, W)
        """

        # Encode pillars
        pillar_embeddings = self.pillar_encoder(pillar_features, mask)  # (N, 64)

        # Scatter to BEV
        H, W = bev_shape
        bev = pillar_embeddings.new_zeros((64, H, W))

        for emb, (x, y) in zip(pillar_embeddings, coords):
            if x < W and y < H:
                bev[:, y, x] = emb

        bev = bev.unsqueeze(0)  # (1, 64, H, W)

        # Backbone + head
        x = self.backbone(bev)
        return self.head(x)
