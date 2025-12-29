import torch
import torch.nn as nn

class BatchedPillarEncoder(nn.Module):
    def __init__(self, in_channels=7, out_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, pillar_features, mask):
        """
        pillar_features: (N, P, 7)
        mask: (N, P) boolean tensor
        """
        x = self.mlp(pillar_features)                     # (N, P, 64)
        x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        x = x.max(dim=1)[0]                               # (N, 64)
        return x
